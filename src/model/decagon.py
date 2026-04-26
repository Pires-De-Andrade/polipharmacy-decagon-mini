"""
decagon.py — Modelo Decagon completo (Encoder R-GCN + Decoder DEDICOM).

Combina:
  - DecagonEncoder: R-GCN de 2 camadas sobre grafo heterogêneo
  - DEDICOMDecoder: fatoração bilinear para predição drug↔drug

Este módulo também contém a função utilitária para converter o
HeteroData em representação homogênea (edge_index + edge_type).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import DecagonEncoder
from .decoder import DEDICOMDecoder


class DecagonModel(nn.Module):
    """Modelo Decagon completo para predição de efeitos adversos
    de polifarmácia.

    Args:
        n_drugs:        Número de drogas.
        n_proteins:     Número de proteínas.
        n_drug_drug_rel: Número de tipos de efeito adverso (relações drug-drug).
        hidden_dim:     Dimensão da camada intermediária do encoder.
        embed_dim:      Dimensão final dos embeddings.
        n_bases:        Número de bases para decomposição R-GCN.
        dropout:        Taxa de dropout.
    """

    # Índices fixos para relações estruturais (não-preditas)
    REL_PPI = 0
    REL_DRUG_PROTEIN = 1
    REL_PROTEIN_DRUG = 2
    REL_SIDE_EFFECT_OFFSET = 3  # side effects começam no índice 3

    def __init__(
        self,
        n_drugs: int,
        n_proteins: int,
        n_drug_drug_rel: int,
        hidden_dim: int = 64,
        embed_dim: int = 64,
        n_bases: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_drugs = n_drugs
        self.n_proteins = n_proteins
        self.n_drug_drug_rel = n_drug_drug_rel

        # Total de relações: 3 estruturais + n_drug_drug_rel
        n_relations = 3 + n_drug_drug_rel

        self.encoder = DecagonEncoder(
            n_drugs=n_drugs,
            n_proteins=n_proteins,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            n_relations=n_relations,
            n_bases=n_bases,
            dropout=dropout,
        )

        self.decoder = DEDICOMDecoder(
            embed_dim=embed_dim,
            n_relations=n_drug_drug_rel,
        )

    def encode(
        self,
        x_drug: torch.Tensor,
        x_protein: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Executa o encoder R-GCN.

        Returns:
            z_drug:    (n_drugs, embed_dim)
            z_protein: (n_proteins, embed_dim)
        """
        return self.encoder(x_drug, x_protein, edge_index, edge_type)

    def decode(
        self,
        z_drug: torch.Tensor,
        edge_index: torch.Tensor,
        relation_idx: int,
    ) -> torch.Tensor:
        """Executa o decoder DEDICOM para uma relação.

        Args:
            z_drug:       (n_drugs, embed_dim) embeddings.
            edge_index:   (2, B) pares a pontuar.
            relation_idx: Índice do efeito adverso (0-indexed nos 50 tipos).

        Returns:
            scores: (B,) logits pré-sigmoid.
        """
        return self.decoder(z_drug, edge_index, relation_idx)

    def decode_all(
        self, z_drug: torch.Tensor, relation_idx: int
    ) -> torch.Tensor:
        """Matriz completa de scores para uma relação."""
        return self.decoder.forward_all(z_drug, relation_idx)


def build_homogeneous_graph(
    data,
    n_drugs: int,
    side_effect_order: list[str],
    train_edges: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Converte o HeteroData em representação homogênea para o R-GCN.

    Mapeia todos os nós para um espaço contínuo:
      - Drogas:    [0, n_drugs)
      - Proteínas: [n_drugs, n_drugs + n_proteins)

    E atribui índices inteiros aos tipos de relação:
      - 0: PPI (protein↔protein)
      - 1: drug→protein
      - 2: protein→drug
      - 3+: side effects (na ordem de side_effect_order)

    Args:
        data:               HeteroData original.
        n_drugs:            Número de drogas.
        side_effect_order:  Lista ordenada de códigos CUI dos efeitos.
        train_edges:        Se fornecido, usa estas arestas drug-drug
                            em vez das do data (para split treino).
                            Dict {se_code: edge_index (2, E) com índices
                            locais de drogas}.

    Returns:
        edge_index: (2, E_total) todas as arestas no espaço homogêneo.
        edge_type:  (E_total,)   tipo de cada aresta.
    """
    all_edges = []
    all_types = []

    # ── PPI (type 0) ─────────────────────────────────────────────────
    ppi_key = ("protein", "interacts", "protein")
    if ppi_key in data.edge_types:
        ei = data[ppi_key].edge_index
        # Offset: proteínas começam em n_drugs
        all_edges.append(ei + n_drugs)
        all_types.append(torch.full((ei.shape[1],), 0, dtype=torch.long))

    # ── Drug → Protein (type 1) ──────────────────────────────────────
    dp_key = ("drug", "targets", "protein")
    if dp_key in data.edge_types:
        ei = data[dp_key].edge_index
        src = ei[0]               # drug indices (sem offset)
        dst = ei[1] + n_drugs     # protein indices (com offset)
        all_edges.append(torch.stack([src, dst]))
        all_types.append(torch.full((ei.shape[1],), 1, dtype=torch.long))

    # ── Protein → Drug (type 2) ──────────────────────────────────────
    pd_key = ("protein", "targeted_by", "drug")
    if pd_key in data.edge_types:
        ei = data[pd_key].edge_index
        src = ei[0] + n_drugs     # protein indices (com offset)
        dst = ei[1]               # drug indices (sem offset)
        all_edges.append(torch.stack([src, dst]))
        all_types.append(torch.full((ei.shape[1],), 2, dtype=torch.long))

    # ── Drug ↔ Drug por efeito adverso (types 3+) ────────────────────
    for se_idx, se_code in enumerate(side_effect_order):
        rel_type_id = DecagonModel.REL_SIDE_EFFECT_OFFSET + se_idx

        if train_edges is not None and se_code in train_edges:
            # Usar apenas arestas de treino
            unique_ei = train_edges[se_code]
            # Tornar bidirecional
            ei = torch.cat([unique_ei, unique_ei.flip(0)], dim=1)
        else:
            # Usar todas as arestas do data
            dd_key = ("drug", f"side_effect_{se_code}", "drug")
            if dd_key not in data.edge_types:
                continue
            ei = data[dd_key].edge_index

        # Drug indices não precisam de offset
        all_edges.append(ei)
        all_types.append(torch.full((ei.shape[1],), rel_type_id, dtype=torch.long))

    edge_index = torch.cat(all_edges, dim=1)
    edge_type = torch.cat(all_types)

    return edge_index, edge_type
