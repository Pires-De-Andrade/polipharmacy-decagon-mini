"""
graph_builder.py — Constrói o grafo heterogêneo PyG (HeteroData) a partir
dos dados filtrados produzidos por loader.py.

Estrutura do grafo (fiel ao artigo Decagon, Zitnik et al. 2018):
  - Nós: protein, drug
  - Arestas:
      • ('protein', 'interacts', 'protein')  — rede PPI
      • ('drug', 'targets', 'protein')       — alvos moleculares
      • ('protein', 'targeted_by', 'drug')    — reversa (para message-passing)
      • ('drug', 'side_effect_<CUI>', 'drug') — um tipo por efeito adverso

Features dos nós:
  - protein: vetor one-hot (identidade) de dimensão n_proteins
  - drug:    vetor one-hot (identidade) de dimensão n_drugs

  Nota: no artigo original, as features são aprendidas. Aqui usamos
  identidade como ponto de partida — o encoder R-GCN vai projetar
  esses embeddings.

Uso:
    python -m src.data.graph_builder
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch_geometric.data import HeteroData

# ─────────────────────────────────────────────────────────────────────
# Caminhos
# ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class DecagonGraphBuilder:
    """Converte os CSVs processados num grafo heterogêneo PyG (HeteroData)."""

    def __init__(self, processed_dir: Path | str = PROCESSED_DIR):
        self.processed_dir = Path(processed_dir)
        self.data: HeteroData | None = None

        # Mapeamentos id → índice contínuo
        self.drug_to_idx: dict[str, int] = {}
        self.protein_to_idx: dict[int, int] = {}
        self.side_effect_to_idx: dict[str, int] = {}

        # Metadados do processamento
        self.metadata: dict[str, Any] = {}

    # ── Leitura dos dados processados ─────────────────────────────────

    def _load_processed(self) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
    ]:
        """Lê CSVs processados e metadados."""
        log.info("Lendo dados processados de %s ...", self.processed_dir)

        combo = pd.read_csv(
            self.processed_dir / "combo_filtered.csv",
            dtype={"STITCH 1": str, "STITCH 2": str,
                   "Polypharmacy Side Effect": str, "Side Effect Name": str},
        )
        ppi = pd.read_csv(
            self.processed_dir / "ppi_filtered.csv",
            dtype={"Gene 1": int, "Gene 2": int},
        )
        targets = pd.read_csv(
            self.processed_dir / "targets_filtered.csv",
            dtype={"STITCH": str, "Gene": int},
        )
        mono = pd.read_csv(
            self.processed_dir / "mono_filtered.csv",
            dtype={"STITCH": str, "Individual Side Effect": str,
                   "Side Effect Name": str},
        )

        with open(self.processed_dir / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        log.info("  combo:   %s linhas", f"{len(combo):,}")
        log.info("  ppi:     %s linhas", f"{len(ppi):,}")
        log.info("  targets: %s linhas", f"{len(targets):,}")
        log.info("  mono:    %s linhas", f"{len(mono):,}")

        return combo, ppi, targets, mono

    # ── Construção dos mapeamentos ────────────────────────────────────

    def _build_mappings(
        self,
        combo: pd.DataFrame,
        ppi: pd.DataFrame,
        targets: pd.DataFrame,
    ) -> None:
        """Cria mapeamentos de IDs originais → índices contínuos [0, N)."""
        # Drogas: união de IDs no combo e targets
        all_drugs = sorted(
            set(combo["STITCH 1"]) | set(combo["STITCH 2"]) |
            set(targets["STITCH"])
        )
        self.drug_to_idx = {d: i for i, d in enumerate(all_drugs)}

        # Proteínas: união de IDs no PPI e targets
        all_proteins = sorted(
            set(ppi["Gene 1"]) | set(ppi["Gene 2"]) |
            set(targets["Gene"])
        )
        self.protein_to_idx = {p: i for i, p in enumerate(all_proteins)}

        # Efeitos adversos (tipos de aresta drug-drug)
        all_se = sorted(set(combo["Polypharmacy Side Effect"]))
        self.side_effect_to_idx = {se: i for i, se in enumerate(all_se)}

        log.info("Mapeamentos construídos:")
        log.info("  Drogas:   %d", len(self.drug_to_idx))
        log.info("  Proteínas: %d", len(self.protein_to_idx))
        log.info("  Efeitos:  %d tipos de aresta drug-drug",
                 len(self.side_effect_to_idx))

    # ── Construção do HeteroData ──────────────────────────────────────

    def _build_hetero_data(
        self,
        combo: pd.DataFrame,
        ppi: pd.DataFrame,
        targets: pd.DataFrame,
        mono: pd.DataFrame,
    ) -> HeteroData:
        """Monta o objeto HeteroData com todos os tipos de nó e aresta."""
        data = HeteroData()

        n_drugs = len(self.drug_to_idx)
        n_proteins = len(self.protein_to_idx)

        # ─── Node features ────────────────────────────────────────
        # Drug: one-hot (identidade) — será substituído por ChemBERTa futuramente
        data["drug"].x = torch.eye(n_drugs, dtype=torch.float32)
        data["drug"].num_nodes = n_drugs

        # Protein: embeddings ESM-2 pré-computados (320-dim)
        esm2_path = self.processed_dir / "protein_esm2.pt"
        if esm2_path.exists():
            esm2 = torch.load(esm2_path, weights_only=True)
            assert esm2.shape[0] == n_proteins, (
                f"ESM-2 tensor has {esm2.shape[0]} rows but graph has "
                f"{n_proteins} proteins"
            )
            data["protein"].x = esm2
            log.info("Protein features: ESM-2 embeddings %s", list(esm2.shape))
        else:
            data["protein"].x = torch.eye(n_proteins, dtype=torch.float32)
            log.warning("ESM-2 not found at %s — using one-hot fallback", esm2_path)
        data["protein"].num_nodes = n_proteins

        log.info("Nós criados: %d drugs, %d proteins", n_drugs, n_proteins)

        # ─── Arestas PPI (protein ↔ protein) ─────────────────────────
        src_ppi = ppi["Gene 1"].map(self.protein_to_idx).values
        dst_ppi = ppi["Gene 2"].map(self.protein_to_idx).values

        # Bidirecional (grafo não-direcionado)
        edge_ppi = torch.tensor(
            [list(src_ppi) + list(dst_ppi),
             list(dst_ppi) + list(src_ppi)],
            dtype=torch.long,
        )
        data["protein", "interacts", "protein"].edge_index = edge_ppi
        log.info("  PPI edges: %d (bidirecional)", edge_ppi.shape[1])

        # ─── Arestas drug → protein (targets) ────────────────────────
        src_targets = targets["STITCH"].map(self.drug_to_idx).values
        dst_targets = targets["Gene"].map(self.protein_to_idx).values

        edge_targets = torch.tensor(
            [list(src_targets), list(dst_targets)],
            dtype=torch.long,
        )
        data["drug", "targets", "protein"].edge_index = edge_targets
        log.info("  Drug→Protein edges: %d", edge_targets.shape[1])

        # Reversa: protein ← drug (necessária para message-passing bidirecional)
        edge_rev_targets = torch.tensor(
            [list(dst_targets), list(src_targets)],
            dtype=torch.long,
        )
        data["protein", "targeted_by", "drug"].edge_index = edge_rev_targets
        log.info("  Protein→Drug edges (rev): %d", edge_rev_targets.shape[1])

        # ─── Arestas drug ↔ drug (por tipo de efeito adverso) ────────
        n_se_edges_total = 0
        for se_code in sorted(self.side_effect_to_idx.keys()):
            se_rows = combo[combo["Polypharmacy Side Effect"] == se_code]

            src = se_rows["STITCH 1"].map(self.drug_to_idx).values
            dst = se_rows["STITCH 2"].map(self.drug_to_idx).values

            # Bidirecional (efeito é simétrico: droga A + droga B)
            edge = torch.tensor(
                [list(src) + list(dst),
                 list(dst) + list(src)],
                dtype=torch.long,
            )

            # Nome sanitizado para a relação
            rel_name = f"side_effect_{se_code}"
            data["drug", rel_name, "drug"].edge_index = edge
            n_se_edges_total += edge.shape[1]

        log.info("  Drug↔Drug edges: %d total (%d tipos de efeito)",
                 n_se_edges_total, len(self.side_effect_to_idx))

        # ─── Metadados extras armazenados no grafo ────────────────────
        # Guardar mapeamentos e info de efeitos mono como atributos
        data.drug_to_idx = self.drug_to_idx
        data.protein_to_idx = self.protein_to_idx
        data.side_effect_to_idx = self.side_effect_to_idx
        data.idx_to_drug = {v: k for k, v in self.drug_to_idx.items()}
        data.idx_to_protein = {v: k for k, v in self.protein_to_idx.items()}

        # Features mono (efeitos individuais por droga) — para o decoder
        drug_mono_features = self._build_mono_features(mono, n_drugs)
        data["drug"].mono_side_effects = drug_mono_features

        return data

    # ── Features de efeitos individuais (mono) ────────────────────────

    def _build_mono_features(
        self, mono: pd.DataFrame, n_drugs: int
    ) -> torch.Tensor:
        """Constrói uma matriz binária (n_drugs × n_mono_effects) indicando
        quais efeitos individuais cada droga apresenta.

        Estas features podem ser usadas como entrada adicional para o
        encoder ou como informação auxiliar no decoder.
        """
        # Mapear efeitos mono para índices
        all_mono_se = sorted(mono["Individual Side Effect"].unique())
        mono_se_to_idx = {se: i for i, se in enumerate(all_mono_se)}

        n_mono = len(all_mono_se)
        feat = torch.zeros(n_drugs, n_mono, dtype=torch.float32)

        for _, row in mono.iterrows():
            drug_id = row["STITCH"]
            se_id = row["Individual Side Effect"]
            if drug_id in self.drug_to_idx and se_id in mono_se_to_idx:
                feat[self.drug_to_idx[drug_id], mono_se_to_idx[se_id]] = 1.0

        n_nonzero = int(feat.sum().item())
        log.info("  Mono features: %d drugs × %d effects, %d nonzero (%.1f%%)",
                 n_drugs, n_mono, n_nonzero,
                 100 * n_nonzero / max(n_drugs * n_mono, 1))

        return feat

    # ── Salvamento ────────────────────────────────────────────────────

    def _save(self, data: HeteroData) -> Path:
        """Salva o HeteroData como .pt."""
        out_path = self.processed_dir / "decagon_hetero_graph.pt"
        torch.save(data, out_path)
        log.info("Grafo salvo em %s (%.1f MB)",
                 out_path, out_path.stat().st_size / 1024 / 1024)
        return out_path

    # ── Pipeline completo ─────────────────────────────────────────────

    def run(self) -> HeteroData:
        """Executa o pipeline: load → build mappings → build graph → save."""
        t0 = time.perf_counter()

        combo, ppi, targets, mono = self._load_processed()
        self._build_mappings(combo, ppi, targets)
        self.data = self._build_hetero_data(combo, ppi, targets, mono)
        self._save(self.data)

        elapsed = time.perf_counter() - t0
        log.info("Construção do grafo concluída em %.1fs ✓", elapsed)

        return self.data

    # ── Carregamento do grafo salvo ───────────────────────────────────

    @staticmethod
    def load_graph(processed_dir: Path | str = PROCESSED_DIR) -> HeteroData:
        """Carrega um HeteroData previamente salvo."""
        path = Path(processed_dir) / "decagon_hetero_graph.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"Grafo não encontrado em {path}. "
                "Execute 'python -m src.data.graph_builder' primeiro."
            )
        return torch.load(path, weights_only=False)


# ─────────────────────────────────────────────────────────────────────
# Execução direta
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    builder = DecagonGraphBuilder()
    graph = builder.run()

    print("\n" + "=" * 60)
    print("HETERODATA — RESUMO")
    print("=" * 60)
    print(graph)
    print("=" * 60)
