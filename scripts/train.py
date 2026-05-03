"""
train.py — Script principal de treinamento do modelo Decagon.

Orquestra todo o pipeline:
  1. Carrega o grafo HeteroData processado
  2. Divide arestas drug-drug em train/val/test (80/10/10)
  3. Instancia o modelo (R-GCN encoder + DEDICOM decoder)
  4. Treina com early stopping (monitorando AUROC no val)
  5. Avalia no test set e salva resultados

Uso:
    python scripts/train.py

Saídas:
    saved_models/best_model.pt            — pesos do melhor modelo
    results/test_metrics_aggregated.csv   — AUROC/AUPRC macro e micro
    results/test_metrics_per_relation.csv — métricas por efeito adverso
    results/training_log.csv              — loss e métricas por época
"""

from __future__ import annotations

import sys
import io
from pathlib import Path

# Força UTF-8 no stdout (Windows)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Adiciona raiz do projeto ao path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.graph_builder import DecagonGraphBuilder
from src.model.decagon import DecagonModel
from src.training.split import train_val_test_split, save_splits
from src.training.trainer import DecagonTrainer

# ─────────────────────────────────────────────────────────────────────
# Hiperparametros (ajustaveis)
# ─────────────────────────────────────────────────────────────────────
HIDDEN_DIM = 64          # Dimensao da camada intermediaria
EMBED_DIM = 64           # Dimensao final dos embeddings
N_BASES = 10             # Bases para decomposicao R-GCN
DROPOUT = 0.1            # Dropout entre camadas
LR = 0.001               # Learning rate
WEIGHT_DECAY = 1e-5      # Regularizacao L2
GRAD_CLIP = 1.0          # Gradient clipping
N_EPOCHS = 500           # Epocas maximas
PATIENCE = 25            # Early stopping patience
LR_PATIENCE = 10         # Epocas para reduzir LR
LR_FACTOR = 0.5          # Fator de reducao do LR
TRAIN_RATIO = 0.8        # Fracao de treino
VAL_RATIO = 0.1          # Fracao de validacao
SEED = 42                # Seed para reprodutibilidade

# Caminhos
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
SAVE_DIR = PROJECT_ROOT / "saved_models"
RESULTS_DIR = PROJECT_ROOT / "results"


def main() -> None:
    torch.manual_seed(SEED)

    # ── 1. Carregar grafo ─────────────────────────────────────────────
    print("=" * 60)
    print("  DECAGON MINI — TREINAMENTO")
    print("=" * 60)
    print()

    print("[1/4] Carregando grafo HeteroData ...")
    data = DecagonGraphBuilder.load_graph(PROCESSED_DIR)

    n_drugs = data["drug"].num_nodes
    n_proteins = data["protein"].num_nodes

    # Extrair ordem dos efeitos adversos (consistente com graph_builder)
    se_order = sorted(data.side_effect_to_idx.keys())
    n_se = len(se_order)

    print(f"      {n_drugs} drogas, {n_proteins} proteinas, {n_se} efeitos adversos")

    # ── 2. Split das arestas ──────────────────────────────────────────
    print(f"[2/4] Dividindo arestas drug-drug ({TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{1-TRAIN_RATIO-VAL_RATIO:.0%}) ...")
    splits = train_val_test_split(data, TRAIN_RATIO, VAL_RATIO, SEED)

    # Salvar splits para reprodutibilidade
    splits_path = PROCESSED_DIR / "splits.pt"
    save_splits(splits, splits_path)

    # Contar arestas por split
    n_train = sum(sp["train"].shape[1] for sp in splits.values())
    n_val = sum(sp["val"].shape[1] for sp in splits.values())
    n_test = sum(sp["test"].shape[1] for sp in splits.values())
    print(f"      train={n_train:,} | val={n_val:,} | test={n_test:,} arestas unicas")

    # ── 3. Instanciar modelo ──────────────────────────────────────────
    print("[3/4] Instanciando modelo Decagon ...")

    protein_feat_dim = data["protein"].x.shape[1]

    model = DecagonModel(
        n_drugs=n_drugs,
        n_proteins=n_proteins,
        n_drug_drug_rel=n_se,
        hidden_dim=HIDDEN_DIM,
        embed_dim=EMBED_DIM,
        n_bases=N_BASES,
        dropout=DROPOUT,
        protein_feat_dim=protein_feat_dim,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Parametros totais: {n_params:,}")
    print(f"      hidden_dim={HIDDEN_DIM}, embed_dim={EMBED_DIM}, n_bases={N_BASES}")
    print(f"      protein_feat_dim={protein_feat_dim}")
    print()

    # ── 4. Treinar ────────────────────────────────────────────────────
    print("[4/4] Iniciando treinamento ...")
    print()

    # Parameter groups: protein_proj usa LR menor para preservar
    # a estrutura dos embeddings ESM-2 pre-treinados
    protein_proj_params = set(model.encoder.protein_proj.parameters())
    other_params = [p for p in model.parameters()
                    if p not in protein_proj_params]

    LR_PROJ = LR * 0.1  # 10x menor para a camada de projecao
    optimizer = torch.optim.Adam([
        {"params": list(protein_proj_params), "lr": LR_PROJ},
        {"params": other_params, "lr": LR},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR, patience=LR_PATIENCE,
    )

    print(f"      Optimizer: Adam (protein_proj lr={LR_PROJ}, rest lr={LR})")
    print()

    trainer = DecagonTrainer(
        model=model,
        data=data,
        splits=splits,
        se_order=se_order,
        grad_clip=GRAD_CLIP,
        patience=PATIENCE,
        save_dir=SAVE_DIR,
        results_dir=RESULTS_DIR,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    test_metrics = trainer.fit(n_epochs=N_EPOCHS)

    # ── Resumo final ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  RESUMO FINAL")
    print("=" * 60)
    print(f"  Melhor epoca:   {trainer.best_epoch}")
    print(f"  AUROC (macro):  {test_metrics.macro_auroc:.4f}")
    print(f"  AUPRC (macro):  {test_metrics.macro_auprc:.4f}")
    print(f"  AUROC (micro):  {test_metrics.micro_auroc:.4f}")
    print(f"  AUPRC (micro):  {test_metrics.micro_auprc:.4f}")
    print()
    print("  Top 5 efeitos (por AUROC):")
    for rm in sorted(test_metrics.per_relation, key=lambda r: r.auroc, reverse=True)[:5]:
        print(f"    {rm.se_code}  AUROC={rm.auroc:.4f}  AUPRC={rm.auprc:.4f}")
    print()
    print("  Pior 5 efeitos (por AUROC):")
    for rm in sorted(test_metrics.per_relation, key=lambda r: r.auroc)[:5]:
        print(f"    {rm.se_code}  AUROC={rm.auroc:.4f}  AUPRC={rm.auprc:.4f}")
    print()
    print(f"  Modelo salvo em:     {SAVE_DIR / 'best_model.pt'}")
    print(f"  Metricas salvas em:  {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
