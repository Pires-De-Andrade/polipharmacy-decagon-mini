"""
trainer.py — Loop de treinamento completo do modelo Decagon.

Responsabilidades:
  1. Construir o grafo homogêneo com apenas arestas de treino.
  2. Para cada época:
     a. Forward pass (encoder R-GCN → embeddings)
     b. Para cada relação: decoder DEDICOM + negative sampling → BCE loss
     c. Backward pass + optimizer step
  3. Avaliar no val set a cada época.
  4. Early stopping baseado em AUROC macro no val.
  5. Salvar melhor modelo e log de métricas.

O treinamento é full-batch (513 nós, ~300K arestas cabem em RAM).
"""

from __future__ import annotations

import csv
import logging
import sys
import io
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.model.decagon import DecagonModel, build_homogeneous_graph
from src.training.negative_sampling import (
    build_existing_edges_set,
    sample_negatives,
)
from src.training.metrics import compute_aggregated_metrics, AggregatedMetrics

# Força UTF-8 no stdout (Windows)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class DecagonTrainer:
    """Orquestra o treinamento e avaliação do modelo Decagon.

    Args:
        model:       Instância de DecagonModel.
        data:        HeteroData original (com todas as arestas).
        splits:      dict[se_code] → {'train', 'val', 'test'} edge_index.
        se_order:    Lista ordenada dos códigos CUI dos efeitos.
        lr:          Learning rate.
        weight_decay: Regularização L2.
        grad_clip:   Valor máximo do gradient norm.
        patience:    Épocas sem melhoria para early stopping.
        save_dir:    Diretório para salvar modelo e logs.
        results_dir: Diretório para salvar métricas finais.
    """

    def __init__(
        self,
        model: DecagonModel,
        data,
        splits: dict[str, dict[str, torch.Tensor]],
        se_order: list[str],
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
        patience: int = 10,
        save_dir: Path | str = "saved_models",
        results_dir: Path | str = "results",
    ):
        self.model = model
        self.data = data
        self.splits = splits
        self.se_order = se_order
        self.grad_clip = grad_clip
        self.patience = patience
        self.save_dir = Path(save_dir)
        self.results_dir = Path(results_dir)

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Features dos nós (fixas durante treino)
        self.x_drug = data["drug"].x
        self.x_protein = data["protein"].x
        self.n_drugs = data["drug"].num_nodes

        # ── Construir grafo do encoder (estrutural + train drug-drug) ─
        log.info("Construindo grafo homogeneo para o encoder (train edges) ...")
        train_edges_dict = {se: sp["train"] for se, sp in splits.items()}
        self.train_edge_index, self.train_edge_type = build_homogeneous_graph(
            data, self.n_drugs, se_order, train_edges=train_edges_dict
        )
        log.info(
            "  Grafo do encoder: %d arestas, %d tipos de relacao",
            self.train_edge_index.shape[1],
            self.train_edge_type.max().item() + 1,
        )

        # ── Conjuntos de arestas existentes por relação (para neg. sampling) ─
        log.info("Construindo conjuntos de arestas existentes ...")
        self.existing_edges: dict[str, set[tuple[int, int]]] = {}
        for se_code in se_order:
            # Usar TODAS as arestas (train+val+test) para evitar
            # amostrar negativos que são positivos em qualquer split
            all_ei = torch.cat(
                [splits[se_code]["train"],
                 splits[se_code]["val"],
                 splits[se_code]["test"]],
                dim=1,
            )
            self.existing_edges[se_code] = build_existing_edges_set(all_ei)

        # ── Tracking ─────────────────────────────────────────────────
        self.best_auroc = 0.0
        self.best_epoch = 0
        self.history: list[dict] = []

    # ── Treino de uma época ───────────────────────────────────────────

    def train_epoch(self) -> float:
        """Executa uma época de treinamento.

        Returns:
            loss_total: Soma das losses de todas as relações.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward: encoder
        z_drug, z_protein = self.model.encode(
            self.x_drug, self.x_protein,
            self.train_edge_index, self.train_edge_type,
        )

        total_loss = torch.tensor(0.0)

        for se_idx, se_code in enumerate(self.se_order):
            pos_ei = self.splits[se_code]["train"]
            if pos_ei.shape[1] == 0:
                continue

            # Tornar bidirecional para treino
            pos_ei_bidir = torch.cat([pos_ei, pos_ei.flip(0)], dim=1)

            # Negative sampling
            neg_ei = sample_negatives(
                pos_ei_bidir,
                self.n_drugs,
                existing_edges=self.existing_edges[se_code],
            )

            # Decoder scores
            pos_scores = self.model.decode(z_drug, pos_ei_bidir, se_idx)
            neg_scores = self.model.decode(z_drug, neg_ei, se_idx)

            # BCE loss
            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([
                torch.ones(pos_scores.shape[0]),
                torch.zeros(neg_scores.shape[0]),
            ])

            loss = F.binary_cross_entropy_with_logits(scores, labels)
            total_loss = total_loss + loss

        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        return total_loss.item()

    # ── Avaliação ─────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, split: str = "val") -> AggregatedMetrics:
        """Avalia o modelo num split (val ou test).

        Args:
            split: 'val' ou 'test'.

        Returns:
            AggregatedMetrics com AUROC/AUPRC per-relation e agregados.
        """
        self.model.eval()

        # Forward: encoder (usa grafo de treino para embeddings)
        z_drug, _ = self.model.encode(
            self.x_drug, self.x_protein,
            self.train_edge_index, self.train_edge_type,
        )

        all_y_true: dict[str, np.ndarray] = {}
        all_y_score: dict[str, np.ndarray] = {}

        for se_idx, se_code in enumerate(self.se_order):
            pos_ei = self.splits[se_code][split]
            if pos_ei.shape[1] == 0:
                continue

            # Negative sampling para avaliação (mesma proporção 1:1)
            neg_ei = sample_negatives(
                pos_ei,
                self.n_drugs,
                existing_edges=self.existing_edges[se_code],
            )

            # Scores
            pos_scores = self.model.decode(z_drug, pos_ei, se_idx)
            neg_scores = self.model.decode(z_drug, neg_ei, se_idx)

            # Converter para numpy
            scores = torch.cat([pos_scores, neg_scores]).sigmoid().cpu().numpy()
            labels = np.concatenate([
                np.ones(pos_scores.shape[0]),
                np.zeros(neg_scores.shape[0]),
            ])

            all_y_true[se_code] = labels
            all_y_score[se_code] = scores

        return compute_aggregated_metrics(all_y_true, all_y_score)

    # ── Loop de treinamento completo ──────────────────────────────────

    def fit(self, n_epochs: int = 100) -> AggregatedMetrics:
        """Executa o loop de treinamento completo com early stopping.

        Args:
            n_epochs: Número máximo de épocas.

        Returns:
            Métricas finais no test set com o melhor modelo.
        """
        log.info("=" * 60)
        log.info("INICIO DO TREINAMENTO")
        log.info("  Epocas max: %d | Patience: %d", n_epochs, self.patience)
        log.info("  Parametros: %s", f"{sum(p.numel() for p in self.model.parameters()):,}")
        log.info("=" * 60)

        epochs_no_improve = 0
        t_start = time.perf_counter()

        for epoch in range(1, n_epochs + 1):
            t0 = time.perf_counter()

            # Treino
            loss = self.train_epoch()

            # Avaliação no val
            val_metrics = self.evaluate("val")

            dt = time.perf_counter() - t0

            # Logging
            log.info(
                "Epoch %3d/%d | loss=%.4f | val_AUROC=%.4f | val_AUPRC=%.4f | %.1fs",
                epoch, n_epochs, loss,
                val_metrics.macro_auroc, val_metrics.macro_auprc, dt,
            )

            # Histórico
            self.history.append({
                "epoch": epoch,
                "loss": loss,
                "val_auroc": val_metrics.macro_auroc,
                "val_auprc": val_metrics.macro_auprc,
                "time_s": dt,
            })

            # Early stopping
            if val_metrics.macro_auroc > self.best_auroc:
                self.best_auroc = val_metrics.macro_auroc
                self.best_epoch = epoch
                epochs_no_improve = 0
                # Salvar melhor modelo
                torch.save(
                    self.model.state_dict(),
                    self.save_dir / "best_model.pt",
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    log.info(
                        "Early stopping na epoca %d (melhor: epoca %d, AUROC=%.4f)",
                        epoch, self.best_epoch, self.best_auroc,
                    )
                    break

        total_time = time.perf_counter() - t_start
        log.info("Treinamento concluido em %.1fs (%d epocas)", total_time, epoch)

        # ── Avaliação final no test set com melhor modelo ─────────────
        log.info("Carregando melhor modelo (epoca %d) ...", self.best_epoch)
        self.model.load_state_dict(
            torch.load(self.save_dir / "best_model.pt", weights_only=True)
        )

        test_metrics = self.evaluate("test")

        log.info("=" * 60)
        log.info("RESULTADOS FINAIS (test set, melhor modelo)")
        log.info(test_metrics.summary_str())

        # Salvar resultados
        self._save_results(test_metrics)
        self._save_training_log()

        return test_metrics

    # ── Persistência ──────────────────────────────────────────────────

    def _save_results(self, test_metrics: AggregatedMetrics) -> None:
        """Salva métricas finais em CSV."""
        # Métricas agregadas
        agg_path = self.results_dir / "test_metrics_aggregated.csv"
        with open(agg_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerow(["macro_auroc", f"{test_metrics.macro_auroc:.6f}"])
            w.writerow(["macro_auprc", f"{test_metrics.macro_auprc:.6f}"])
            w.writerow(["micro_auroc", f"{test_metrics.micro_auroc:.6f}"])
            w.writerow(["micro_auprc", f"{test_metrics.micro_auprc:.6f}"])
            w.writerow(["best_epoch", str(self.best_epoch)])

        # Métricas per-relation
        rel_path = self.results_dir / "test_metrics_per_relation.csv"
        with open(rel_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["se_code", "auroc", "auprc", "n_pos", "n_neg"])
            for rm in sorted(test_metrics.per_relation, key=lambda r: r.auroc, reverse=True):
                w.writerow([rm.se_code, f"{rm.auroc:.6f}", f"{rm.auprc:.6f}", rm.n_pos, rm.n_neg])

        log.info("Resultados salvos em %s", self.results_dir)

    def _save_training_log(self) -> None:
        """Salva histórico de treinamento em CSV."""
        log_path = self.results_dir / "training_log.csv"
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "loss", "val_auroc", "val_auprc", "time_s"])
            w.writeheader()
            for row in self.history:
                w.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in row.items()})

        log.info("Log de treinamento salvo em %s", log_path)
