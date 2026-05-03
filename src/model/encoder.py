"""
encoder.py — R-GCN Encoder para o grafo heterogêneo Decagon.

Arquitetura (fiel ao paper Zitnik et al., 2018):
  1. Projeção linear das features de cada tipo de nó para um
     espaço compartilhado de dimensão hidden_dim.
     - drug: one-hot (n_drugs) ou features pré-treinadas
     - protein: ESM-2 (320-dim) ou one-hot fallback
  2. Duas camadas R-GCN com decomposição de base (num_bases) para
     reduzir o número de parâmetros com muitos tipos de relação.
  3. Saída: embeddings z_drug e z_protein de dimensão embed_dim.

Organização dos nós no grafo homogêneo:
  - Índices [0, n_drugs)           → drogas
  - Índices [n_drugs, n_drugs+n_proteins) → proteínas

Tipos de relação (53 total):
  - 0:  (protein, interacts, protein)
  - 1:  (drug, targets, protein)
  - 2:  (protein, targeted_by, drug)
  - 3–52: (drug, side_effect_<CUI>, drug) × 50
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class DecagonEncoder(nn.Module):
    """R-GCN encoder de 2 camadas com decomposição de base.

    Projeta features heterogêneas para embeddings densos num
    espaço compartilhado, depois aplica convolução relacional.

    Args:
        n_drugs:          Número de nós do tipo droga.
        n_proteins:       Número de nós do tipo proteína.
        hidden_dim:       Dimensão da camada intermediária.
        embed_dim:        Dimensão final dos embeddings.
        n_relations:      Total de tipos de relação no grafo homogêneo.
        n_bases:          Número de bases para decomposição (reduz parâmetros).
        protein_feat_dim: Dimensão das features de proteína (320 para ESM-2).
        dropout:          Taxa de dropout entre camadas.
    """

    def __init__(
        self,
        n_drugs: int,
        n_proteins: int,
        hidden_dim: int = 64,
        embed_dim: int = 64,
        n_relations: int = 53,
        n_bases: int = 10,
        dropout: float = 0.1,
        protein_feat_dim: int = 320,
    ):
        super().__init__()

        self.n_drugs = n_drugs
        self.n_proteins = n_proteins

        # ── Projeções de entrada (dims diferentes por tipo de nó) ─────
        self.drug_proj = nn.Linear(n_drugs, hidden_dim)
        self.protein_proj = nn.Linear(protein_feat_dim, hidden_dim)

        # ── Camadas R-GCN com decomposição de base ───────────────────
        self.conv1 = RGCNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            num_relations=n_relations,
            num_bases=n_bases,
        )
        self.conv2 = RGCNConv(
            in_channels=hidden_dim,
            out_channels=embed_dim,
            num_relations=n_relations,
            num_bases=n_bases,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_drug: torch.Tensor,
        x_protein: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass do encoder.

        Args:
            x_drug:     (n_drugs, n_drugs) features das drogas (one-hot).
            x_protein:  (n_proteins, protein_feat_dim) features das proteínas (ESM-2).
            edge_index: (2, E) índices de arestas no grafo homogêneo.
            edge_type:  (E,) tipo de relação de cada aresta.

        Returns:
            z_drug:    (n_drugs, embed_dim) embeddings das drogas.
            z_protein: (n_proteins, embed_dim) embeddings das proteínas.
        """
        # Projetar para espaço compartilhado
        h_drug = self.drug_proj(x_drug)          # (n_drugs, hidden_dim)
        h_protein = self.protein_proj(x_protein)  # (n_proteins, hidden_dim)

        # Concatenar: [drogas | proteínas]
        x = torch.cat([h_drug, h_protein], dim=0)  # (N, hidden_dim)

        # Camada 1
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.dropout(x)

        # Camada 2
        x = self.conv2(x, edge_index, edge_type)

        # Separar embeddings por tipo de nó
        z_drug = x[: self.n_drugs]
        z_protein = x[self.n_drugs :]

        return z_drug, z_protein
