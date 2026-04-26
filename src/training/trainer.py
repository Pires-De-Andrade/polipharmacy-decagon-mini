"""
trainer.py — Loop de treinamento completo do modelo Decagon.

Responsabilidades:
  1. Construir o grafo homogeneo com apenas arestas de treino.
  2. Pre-computar negativos para treino e avaliacao.
  3. Para cada epoca:
     a. Forward pass (encoder R-GCN -> embeddings)
     b. Para cada relacao: decoder DEDICOM + neg pre-computados -> BCE loss
     c. Backward pass + optimizer step (loss MEDIA sobre relacoes)
  4. Avaliar no val set a cada epoca (deterministico).
  5. Early stopping baseado em AUROC macro no val.
  6. LR scheduler (ReduceLROnPlateau) para refinamento.
  7. Salvar melhor modelo e log de metricas.

O treinamento eh full-batch (513 nos, ~300K arestas cabem em RAM).
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
from tqdm import tqdm

from src.model.decagon import DecagonModel, build_homogeneous_graph
from src.training.negative_sampling import (
    precompute_all_negatives,
    sample_from_precomputed,
)
from src.training.metrics import compute_aggregated_metrics, AggregatedMetrics

# Forca UTF-8 no stdout (Windows)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class DecagonTrainer:
    """Orquestra o treinamento e avaliacao do modelo Decagon.

    Args:
        model:       Instancia de DecagonModel.
        data:        HeteroData original (com todas as arestas).
        splits:      dict[se_code] -> {'train', 'val', 'test'} edge_index.
        se_order:    Lista ordenada dos codigos CUI dos efeitos.
        lr:          Learning rate.
        weight_decay: Regularizacao L2.
        grad_clip:   Valor maximo do gradient norm.
        patience:    Epocas sem melhoria para early stopping.
        lr_patience: Epocas sem melhoria para reduzir LR.
        lr_factor:   Fator de reducao do LR.
        save_dir:    Diretorio para salvar modelo e logs.
        results_dir: Diretorio para salvar metricas finais.
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
        patience: int = 25,
        lr_patience: int = 10,
        lr_factor: float = 0.5,
        save_dir: Path | str = "saved_models",
        results_dir: Path | str = "results",
    ):
        self.model = model
        self.data = data
        self.splits = splits
        self.se_order = se_order
        self.n_relations = len(se_order)
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

        # LR Scheduler — reduz LR quando val_AUROC estagna
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",           # maximizar AUROC
            factor=lr_factor,
            patience=lr_patience,
        )

        # Features dos nos (fixas durante treino)
        self.x_drug = data["drug"].x
        self.x_protein = data["protein"].x
        self.n_drugs = data["drug"].num_nodes

        # -- Construir grafo do encoder (estrutural + train drug-drug) --
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

        # -- Pre-computar negativos (uma unica vez) --
        log.info("Pre-computando arestas negativas para todas as relacoes ...")
        self.all_negatives = precompute_all_negatives(
            splits, self.n_drugs, se_order
        )
        n_neg_total = sum(v.shape[1] for v in self.all_negatives.values())
        log.info("  Total de negativos pre-computados: %d", n_neg_total)

        # -- Pre-computar negativos fixos para avaliacao (deterministico) --
        log.info("Pre-computando negativos fixos para avaliacao ...")
        self.val_negatives: dict[str, torch.Tensor] = {}
        self.test_negatives: dict[str, torch.Tensor] = {}
        for se_code in se_order:
            n_val_pos = splits[se_code]["val"].shape[1]
            n_test_pos = splits[se_code]["test"].shape[1]
            neg_pool = self.all_negatives[se_code]
            # Para avaliacao: ratio 1:1 com seed fixa
            self.val_negatives[se_code] = sample_from_precomputed(
                neg_pool, n_val_pos, seed=12345
            )
            self.test_negatives[se_code] = sample_from_precomputed(
                neg_pool, n_test_pos, seed=67890
            )

        # -- Tracking --
        self.best_auroc = 0.0
        self.best_epoch = 0
        self.history: list[dict] = []

    # -- Treino de uma epoca --

    def train_epoch(self, epoch: int) -> float:
        """Executa uma epoca de treinamento.

        Args:
            epoch: Numero da epoca (usado como seed para neg. sampling).

        Returns:
            loss_avg: Loss MEDIA sobre todas as relacoes.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward: encoder
        z_drug, z_protein = self.model.encode(
            self.x_drug, self.x_protein,
            self.train_edge_index, self.train_edge_type,
        )

        total_loss = torch.tensor(0.0)
        n_active_relations = 0

        for se_idx, se_code in enumerate(self.se_order):
            pos_ei = self.splits[se_code]["train"]
            if pos_ei.shape[1] == 0:
                continue

            # Tornar bidirecional para treino
            pos_ei_bidir = torch.cat([pos_ei, pos_ei.flip(0)], dim=1)

            # Amostrar negativos dos pre-computados (seed varia por epoca)
            neg_pool = self.all_negatives[se_code]
            neg_ei_unique = sample_from_precomputed(
                neg_pool, pos_ei.shape[1], seed=epoch * 1000 + se_idx
            )
            # Tornar bidirecional
            neg_ei_bidir = torch.cat([neg_ei_unique, neg_ei_unique.flip(0)], dim=1)

            # Decoder scores
            pos_scores = self.model.decode(z_drug, pos_ei_bidir, se_idx)
            neg_scores = self.model.decode(z_drug, neg_ei_bidir, se_idx)

            # BCE loss
            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([
                torch.ones(pos_scores.shape[0]),
                torch.zeros(neg_scores.shape[0]),
            ])

            loss = F.binary_cross_entropy_with_logits(scores, labels)
            total_loss = total_loss + loss
            n_active_relations += 1

        # MEDIA sobre relacoes (nao soma!)
        if n_active_relations > 0:
            total_loss = total_loss / n_active_relations

        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        return total_loss.item()

    # -- Avaliacao --

    @torch.no_grad()
    def evaluate(self, split: str = "val") -> AggregatedMetrics:
        """Avalia o modelo num split (val ou test).

        Usa negativos pre-computados com seed fixa para avaliacao
        DETERMINISTICA — elimina ruido no early stopping.

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

        # Selecionar negativos pre-computados para este split
        neg_dict = self.val_negatives if split == "val" else self.test_negatives

        all_y_true: dict[str, np.ndarray] = {}
        all_y_score: dict[str, np.ndarray] = {}

        for se_idx, se_code in enumerate(self.se_order):
            pos_ei = self.splits[se_code][split]
            if pos_ei.shape[1] == 0:
                continue

            neg_ei = neg_dict[se_code]
            if neg_ei.shape[1] == 0:
                continue

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

    # -- Loop de treinamento completo --

    def fit(self, n_epochs: int = 200) -> AggregatedMetrics:
        """Executa o loop de treinamento completo com early stopping.

        Args:
            n_epochs: Numero maximo de epocas.

        Returns:
            Metricas finais no test set com o melhor modelo.
        """
        n_params = sum(p.numel() for p in self.model.parameters())
        print()
        print("=" * 60)
        print("  INICIO DO TREINAMENTO")
        print(f"  Epocas max: {n_epochs} | Patience: {self.patience}")
        print(f"  Parametros: {n_params:,}")
        print(f"  LR inicial: {self.optimizer.param_groups[0]['lr']}")
        print("=" * 60)
        print()

        epochs_no_improve = 0
        t_start = time.perf_counter()

        pbar = tqdm(
            range(1, n_epochs + 1),
            desc="Treinamento",
            unit="epoch",
            bar_format="{l_bar}{bar:30}{r_bar}",
            ncols=110,
        )

        for epoch in pbar:
            t0 = time.perf_counter()

            # Treino
            loss = self.train_epoch(epoch)

            # Avaliacao no val (deterministico)
            val_metrics = self.evaluate("val")

            # LR scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_metrics.macro_auroc)
            new_lr = self.optimizer.param_groups[0]["lr"]
            if new_lr < current_lr:
                tqdm.write(f"  >> LR reduzido: {current_lr:.6f} -> {new_lr:.6f}")

            dt = time.perf_counter() - t0

            # Atualizar barra de progresso
            pbar.set_postfix(
                loss=f"{loss:.4f}",
                AUROC=f"{val_metrics.macro_auroc:.4f}",
                best=f"{self.best_auroc:.4f}",
                lr=f"{new_lr:.1e}",
            )

            # Historico
            self.history.append({
                "epoch": epoch,
                "loss": loss,
                "val_auroc": val_metrics.macro_auroc,
                "val_auprc": val_metrics.macro_auprc,
                "lr": new_lr,
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
                    pbar.set_description(
                        f"Early stop (best epoch {self.best_epoch})"
                    )
                    break

        pbar.close()
        total_time = time.perf_counter() - t_start
        print(f"\nTreinamento concluido em {total_time:.1f}s ({epoch} epocas)")

        # -- Avaliacao final no test set com melhor modelo --
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

    # -- Persistencia --

    def _save_results(self, test_metrics: AggregatedMetrics) -> None:
        """Salva metricas finais em CSV."""
        # Metricas agregadas
        agg_path = self.results_dir / "test_metrics_aggregated.csv"
        with open(agg_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerow(["macro_auroc", f"{test_metrics.macro_auroc:.6f}"])
            w.writerow(["macro_auprc", f"{test_metrics.macro_auprc:.6f}"])
            w.writerow(["micro_auroc", f"{test_metrics.micro_auroc:.6f}"])
            w.writerow(["micro_auprc", f"{test_metrics.micro_auprc:.6f}"])
            w.writerow(["best_epoch", str(self.best_epoch)])

        # Metricas per-relation
        rel_path = self.results_dir / "test_metrics_per_relation.csv"
        with open(rel_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["se_code", "auroc", "auprc", "n_pos", "n_neg"])
            for rm in sorted(test_metrics.per_relation, key=lambda r: r.auroc, reverse=True):
                w.writerow([rm.se_code, f"{rm.auroc:.6f}", f"{rm.auprc:.6f}", rm.n_pos, rm.n_neg])

        log.info("Resultados salvos em %s", self.results_dir)

    def _save_training_log(self) -> None:
        """Salva historico de treinamento em CSV."""
        log_path = self.results_dir / "training_log.csv"
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["epoch", "loss", "val_auroc", "val_auprc", "lr", "time_s"]
            )
            w.writeheader()
            for row in self.history:
                w.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in row.items()})

        log.info("Log de treinamento salvo em %s", log_path)
