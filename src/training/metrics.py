"""
metrics.py — Métricas de avaliação para predição de efeitos adversos.

Implementa AUROC e AUPRC per-relation e agregadas (macro/micro).
Usa scikit-learn internamente.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


@dataclass
class RelationMetrics:
    """Métricas para um tipo de efeito adverso."""

    se_code: str
    auroc: float
    auprc: float
    n_pos: int
    n_neg: int


@dataclass
class AggregatedMetrics:
    """Métricas agregadas sobre todos os tipos de efeito."""

    macro_auroc: float = 0.0
    macro_auprc: float = 0.0
    micro_auroc: float = 0.0
    micro_auprc: float = 0.0
    per_relation: list[RelationMetrics] = field(default_factory=list)

    def summary_str(self) -> str:
        """Retorna string formatada com resumo das métricas."""
        lines = [
            "=" * 50,
            f"  AUROC  macro={self.macro_auroc:.4f}  micro={self.micro_auroc:.4f}",
            f"  AUPRC  macro={self.macro_auprc:.4f}  micro={self.micro_auprc:.4f}",
            "=" * 50,
        ]
        return "\n".join(lines)


def compute_relation_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    se_code: str,
) -> RelationMetrics | None:
    """Computa AUROC e AUPRC para uma relação.

    Args:
        y_true:  (N,) rótulos binários (0 ou 1).
        y_score: (N,) scores preditos (probabilidades ou logits).
        se_code: Código CUI do efeito adverso.

    Returns:
        RelationMetrics ou None se não há exemplos suficientes.
    """
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos

    # Precisamos de ao menos 1 positivo e 1 negativo
    if n_pos == 0 or n_neg == 0:
        return None

    try:
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
    except ValueError:
        return None

    return RelationMetrics(
        se_code=se_code,
        auroc=auroc,
        auprc=auprc,
        n_pos=n_pos,
        n_neg=n_neg,
    )


def compute_aggregated_metrics(
    all_y_true: dict[str, np.ndarray],
    all_y_score: dict[str, np.ndarray],
) -> AggregatedMetrics:
    """Computa métricas per-relation e agregadas (macro/micro).

    Args:
        all_y_true:  dict[se_code] → (N_r,) rótulos binários.
        all_y_score: dict[se_code] → (N_r,) scores preditos.

    Returns:
        AggregatedMetrics com per-relation, macro e micro.
    """
    per_relation: list[RelationMetrics] = []

    # Para micro-average: pool de todas as predições
    all_true_pooled = []
    all_score_pooled = []

    for se_code in sorted(all_y_true.keys()):
        y_true = all_y_true[se_code]
        y_score = all_y_score[se_code]

        rm = compute_relation_metrics(y_true, y_score, se_code)
        if rm is not None:
            per_relation.append(rm)
            all_true_pooled.append(y_true)
            all_score_pooled.append(y_score)

    # Macro-average: média das métricas per-relation
    if per_relation:
        macro_auroc = np.mean([r.auroc for r in per_relation])
        macro_auprc = np.mean([r.auprc for r in per_relation])
    else:
        macro_auroc = macro_auprc = 0.0

    # Micro-average: pool de todas as predições
    if all_true_pooled:
        pooled_true = np.concatenate(all_true_pooled)
        pooled_score = np.concatenate(all_score_pooled)
        micro_auroc = roc_auc_score(pooled_true, pooled_score)
        micro_auprc = average_precision_score(pooled_true, pooled_score)
    else:
        micro_auroc = micro_auprc = 0.0

    return AggregatedMetrics(
        macro_auroc=float(macro_auroc),
        macro_auprc=float(macro_auprc),
        micro_auroc=float(micro_auroc),
        micro_auprc=float(micro_auprc),
        per_relation=per_relation,
    )
