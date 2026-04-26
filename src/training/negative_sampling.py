"""
negative_sampling.py — Amostragem de arestas negativas para treinamento.

Para cada aresta positiva (drug_i, drug_j) sob um efeito adverso r,
amostramos K arestas negativas (drug_i, drug_j') onde drug_j' é
escolhido aleatoriamente e (drug_i, drug_j') NÃO existe sob r.

Com apenas 100 drogas e esparsidade ~0.43, a chance de amostrar
acidentalmente uma aresta positiva é significativa, então filtramos
colisões explicitamente.
"""

from __future__ import annotations

import torch


def sample_negatives(
    pos_edge_index: torch.Tensor,
    num_nodes: int,
    existing_edges: set[tuple[int, int]] | None = None,
    num_neg_per_pos: int = 1,
    max_retries: int = 10,
) -> torch.Tensor:
    """Amostra arestas negativas para um tipo de relação.

    Args:
        pos_edge_index:  (2, N_pos) arestas positivas.
        num_nodes:       Número total de nós do tipo (e.g., n_drugs).
        existing_edges:  Conjunto de arestas existentes {(i, j)} para
                         evitar colisões. Se None, não filtra.
        num_neg_per_pos: Número de negativos por positivo (default: 1).
        max_retries:     Tentativas máximas para resolver colisões.

    Returns:
        neg_edge_index: (2, N_pos * num_neg_per_pos) arestas negativas.
    """
    n_pos = pos_edge_index.shape[1]
    n_neg = n_pos * num_neg_per_pos

    # Repetir source nodes
    src = pos_edge_index[0].repeat_interleave(num_neg_per_pos)  # (n_neg,)

    # Amostrar destinos aleatórios
    dst = torch.randint(0, num_nodes, (n_neg,))

    if existing_edges is not None:
        # Filtrar colisões (amostras que coincidem com arestas reais)
        for _ in range(max_retries):
            collision_mask = torch.tensor(
                [(s.item(), d.item()) in existing_edges or s.item() == d.item()
                 for s, d in zip(src, dst)],
                dtype=torch.bool,
            )
            n_collisions = collision_mask.sum().item()
            if n_collisions == 0:
                break
            # Re-amostrar apenas as colisões
            dst[collision_mask] = torch.randint(
                0, num_nodes, (n_collisions,)
            )
    else:
        # Ao menos evitar self-loops
        self_loop_mask = src == dst
        while self_loop_mask.any():
            dst[self_loop_mask] = torch.randint(
                0, num_nodes, (self_loop_mask.sum().item(),)
            )
            self_loop_mask = src == dst

    return torch.stack([src, dst])


def build_existing_edges_set(
    edge_index: torch.Tensor,
) -> set[tuple[int, int]]:
    """Constrói o conjunto de arestas existentes a partir de um edge_index.

    Inclui ambas as direções para arestas não-direcionadas.

    Args:
        edge_index: (2, E) tensor de arestas.

    Returns:
        Set de tuplas (src, dst).
    """
    edges = set()
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for s, d in zip(src, dst):
        edges.add((s, d))
        edges.add((d, s))  # bidirecional
    return edges
