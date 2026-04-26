"""
decoder.py — Decoder bilinear DEDICOM para predição de efeitos adversos.

Formulação DEDICOM (fiel ao paper):
  score(i, j, r) = z_i^T · D_r · R · D_r · z_j

Onde:
  - z_i, z_j ∈ R^d         — embeddings das drogas (saída do encoder)
  - R ∈ R^{d×d}             — matriz global compartilhada entre relações
  - D_r = diag(d_r) ∈ R^{d} — vetor diagonal por tipo de efeito adverso

Computação vetorizada eficiente:
  score = sum_k [ (z_i ⊙ d_r) · R · (z_j ⊙ d_r) ]
        = (z_i * d_r) @ R @ (z_j * d_r)^T  (diagonal do resultado)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DEDICOMDecoder(nn.Module):
    """Decoder DEDICOM para predição de arestas drug↔drug.

    Para cada tipo de efeito adverso r, prediz a probabilidade de um
    par de drogas (i, j) causar o efeito r usando a fatoração DEDICOM.

    Args:
        embed_dim:    Dimensão dos embeddings de entrada (saída do encoder).
        n_relations:  Número de tipos de efeito adverso (drug-drug relations).
    """

    def __init__(self, embed_dim: int, n_relations: int):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_relations = n_relations

        # Matriz global R — compartilhada entre todas as relações
        self.R = nn.Parameter(torch.empty(embed_dim, embed_dim))
        nn.init.xavier_uniform_(self.R)

        # Vetores diagonais D_r — um por tipo de relação
        self.D = nn.Parameter(torch.empty(n_relations, embed_dim))
        nn.init.normal_(self.D, std=0.01)

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        relation_idx: int,
    ) -> torch.Tensor:
        """Computa scores para um batch de arestas de uma relação.

        Args:
            z:             (n_drugs, embed_dim) embeddings das drogas.
            edge_index:    (2, B) pares de drogas a pontuar.
            relation_idx:  Índice do tipo de efeito adverso.

        Returns:
            scores: (B,) logits (pré-sigmoid) para cada par.
        """
        z_i = z[edge_index[0]]  # (B, d)
        z_j = z[edge_index[1]]  # (B, d)

        d_r = self.D[relation_idx]  # (d,)

        # Aplicar diagonal: z ⊙ d_r
        z_i_d = z_i * d_r  # (B, d)
        z_j_d = z_j * d_r  # (B, d)

        # score = (z_i ⊙ d_r)^T · R · (z_j ⊙ d_r)
        # Vetorizado: sum over d of (z_i_d @ R) * z_j_d
        scores = (z_i_d @ self.R * z_j_d).sum(dim=-1)  # (B,)

        return scores

    def forward_all(
        self,
        z: torch.Tensor,
        relation_idx: int,
    ) -> torch.Tensor:
        """Computa matriz completa de scores para uma relação.

        Útil para avaliação (ranking de todos os pares possíveis).

        Args:
            z:             (n_drugs, embed_dim) embeddings das drogas.
            relation_idx:  Índice do tipo de efeito adverso.

        Returns:
            score_matrix: (n_drugs, n_drugs) scores para todos os pares.
        """
        d_r = self.D[relation_idx]  # (d,)
        z_d = z * d_r               # (N, d)

        # (N, d) @ (d, d) @ (d, N) = (N, N)
        return z_d @ self.R @ z_d.T
