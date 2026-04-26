"""
split.py — Divisão train/val/test das arestas drug↔drug.

Estratégia:
  - Apenas arestas drug↔drug são divididas (são o alvo de predição).
  - Arestas PPI e drug→protein são estruturais e ficam intactas.
  - Para cada tipo de efeito adverso, extraímos arestas únicas (i < j)
    e dividimos aleatoriamente em 80/10/10.
  - O encoder usa APENAS arestas de treino durante o treinamento
    (setup de link prediction transductivo).

Uso:
    python -m src.training.split
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def train_val_test_split(
    data: HeteroData,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, dict[str, torch.Tensor]]:
    """Divide arestas drug↔drug por tipo de efeito em train/val/test.

    Para cada relação side_effect_*, extrai arestas únicas (i < j),
    embaralha com seed fixa, e divide nos ratios especificados.

    Args:
        data:        HeteroData com arestas drug-drug bidirecionais.
        train_ratio: Fração para treino (default: 0.8).
        val_ratio:   Fração para validação (default: 0.1).
        seed:        Seed para reprodutibilidade.

    Returns:
        splits: dict[se_code] → {
            'train': (2, N_train) edge_index,
            'val':   (2, N_val) edge_index,
            'test':  (2, N_test) edge_index,
        }
        Onde edge_index contém arestas únicas (i < j) com índices
        locais de drogas.
    """
    rng = torch.Generator().manual_seed(seed)
    splits: dict[str, dict[str, torch.Tensor]] = {}

    # Identificar relações drug-drug (side effects) em ordem
    se_edge_types = sorted(
        [et for et in data.edge_types if et[1].startswith("side_effect_")],
        key=lambda et: et[1],
    )

    total_train = total_val = total_test = 0

    for edge_type in se_edge_types:
        _, rel, _ = edge_type
        se_code = rel.replace("side_effect_", "")

        edge_index = data[edge_type].edge_index

        # Extrair arestas únicas (i < j) — remove duplicatas bidirecionais
        mask = edge_index[0] < edge_index[1]
        unique_ei = edge_index[:, mask]
        n = unique_ei.shape[1]

        # Embaralhar e dividir
        perm = torch.randperm(n, generator=rng)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # Resto vai para test
        n_test = n - n_train - n_val

        train_ei = unique_ei[:, perm[:n_train]]
        val_ei = unique_ei[:, perm[n_train : n_train + n_val]]
        test_ei = unique_ei[:, perm[n_train + n_val :]]

        splits[se_code] = {
            "train": train_ei,
            "val": val_ei,
            "test": test_ei,
        }

        total_train += n_train
        total_val += n_val
        total_test += n_test

    log.info(
        "Split concluido: %d relacoes | train=%d, val=%d, test=%d arestas unicas",
        len(splits),
        total_train,
        total_val,
        total_test,
    )

    return splits


def save_splits(
    splits: dict[str, dict[str, torch.Tensor]],
    path: Path | str,
) -> None:
    """Salva splits em arquivo .pt para reprodutibilidade."""
    path = Path(path)
    torch.save(splits, path)
    log.info("Splits salvos em %s", path)


def load_splits(path: Path | str) -> dict[str, dict[str, torch.Tensor]]:
    """Carrega splits de arquivo .pt."""
    return torch.load(Path(path), weights_only=False)
