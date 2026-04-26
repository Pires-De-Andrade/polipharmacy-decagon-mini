"""
negative_sampling.py — Amostragem de arestas negativas para treinamento.

Estratégia otimizada para grafos pequenos (100 drogas):
  Em vez de amostrar negativos aleatoriamente a cada época (lento e
  ruidoso), PRÉ-COMPUTAMOS todos os pares negativos possíveis para
  cada relação. Com 100 drogas, há no máximo 4,950 pares únicos (i<j)
  por relação, e tipicamente ~2,800 são positivos — restam ~2,150
  negativos. Isso é pequeno o suficiente para armazenar em memória.

  Para treino: amostramos um subconjunto dos negativos pré-computados
  a cada época (com seed variável para diversidade).
  Para avaliação: usamos TODOS os negativos (determinístico).
"""

from __future__ import annotations

import torch


def precompute_negatives(
    pos_edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Pré-computa TODOS os pares negativos para uma relação.

    Enumera todos os pares possíveis (i, j) com i < j, remove os
    positivos, e retorna os negativos.

    Args:
        pos_edge_index: (2, N_pos) arestas positivas (únicas, i < j).
        num_nodes:      Número de nós (e.g., n_drugs = 100).

    Returns:
        neg_edge_index: (2, N_neg) arestas negativas (únicas, i < j).
    """
    # Codificar arestas positivas como inteiros únicos para lookup rápido
    pos_codes = pos_edge_index[0] * num_nodes + pos_edge_index[1]
    pos_set = set(pos_codes.tolist())

    # Enumerar todos os pares possíveis (i < j)
    neg_src = []
    neg_dst = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            code = i * num_nodes + j
            if code not in pos_set:
                neg_src.append(i)
                neg_dst.append(j)

    if not neg_src:
        return torch.zeros(2, 0, dtype=torch.long)

    return torch.tensor([neg_src, neg_dst], dtype=torch.long)


def sample_from_precomputed(
    neg_edge_index: torch.Tensor,
    n_samples: int,
    seed: int | None = None,
) -> torch.Tensor:
    """Amostra um subconjunto dos negativos pré-computados.

    Args:
        neg_edge_index: (2, N_neg) negativos pré-computados.
        n_samples:      Número de amostras desejado.
        seed:           Seed para reprodutibilidade. Se None, aleatório.

    Returns:
        sampled: (2, min(n_samples, N_neg)) arestas negativas.
    """
    n_available = neg_edge_index.shape[1]

    if n_samples >= n_available:
        return neg_edge_index

    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_available, generator=gen)
    else:
        perm = torch.randperm(n_available)

    return neg_edge_index[:, perm[:n_samples]]


def precompute_all_negatives(
    splits: dict[str, dict[str, torch.Tensor]],
    num_nodes: int,
    se_order: list[str],
) -> dict[str, torch.Tensor]:
    """Pré-computa negativos para todas as relações.

    Usa a UNIÃO de arestas (train + val + test) como positivas para
    garantir que nenhum negativo coincida com um positivo em qualquer
    split.

    Args:
        splits:    dict[se_code] → {'train', 'val', 'test'} edge_index.
        num_nodes: Número de drogas.
        se_order:  Lista ordenada dos códigos CUI.

    Returns:
        all_negatives: dict[se_code] → (2, N_neg) negativos pré-computados.
    """
    all_negatives: dict[str, torch.Tensor] = {}

    for se_code in se_order:
        # Unir todas as arestas positivas (train + val + test)
        all_pos = torch.cat(
            [splits[se_code]["train"],
             splits[se_code]["val"],
             splits[se_code]["test"]],
            dim=1,
        )
        all_negatives[se_code] = precompute_negatives(all_pos, num_nodes)

    return all_negatives
