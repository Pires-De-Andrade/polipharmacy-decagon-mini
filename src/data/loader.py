"""
loader.py — Leitura e filtragem dos dados brutos do SNAP Stanford (Decagon).

Estratégia de filtragem (fiel ao artigo Zitnik et al., 2018):
  1. Seleciona as TOP N_DRUGS drogas com maior cobertura no TWOSIDES
     (bio-decagon-combo.csv), i.e. as que aparecem em mais pares de
     combinação.
  2. Filtra os pares de drogas (combo) mantendo apenas aqueles em que
     AMBAS as drogas estão no subconjunto selecionado.
  3. Seleciona os TOP N_SIDE_EFFECTS efeitos adversos mais frequentes
     entre esses pares filtrados.
  4. Filtra as arestas droga→proteína (targets) para as drogas selecionadas,
     extraindo o conjunto de proteínas conectadas.
  5. Filtra as interações proteína-proteína (PPI) mantendo apenas arestas
     em que AMBAS as proteínas pertencem ao subconjunto extraído.
  6. Filtra os efeitos individuais (mono) para as drogas selecionadas.
  7. Salva tudo em dataset/processed/ como CSVs + metadados JSON.

Uso:
    python -m src.data.loader
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────────────────────────────
# Constantes configuráveis — ajuste conforme necessidade
# ─────────────────────────────────────────────────────────────────────
N_DRUGS: int = 100          # Top drogas por cobertura no TWOSIDES
N_SIDE_EFFECTS: int = 50    # Top efeitos adversos mais frequentes
MIN_COMBO_PER_SE: int = 10  # Mín. de pares por efeito p/ ser incluído

# Caminhos
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"

# Nomes dos arquivos brutos
COMBO_FILE = "bio-decagon-combo.csv"
PPI_FILE = "bio-decagon-ppi.csv"
TARGETS_FILE = "bio-decagon-targets.csv"
MONO_FILE = "bio-decagon-mono.csv"
CATEGORIES_FILE = "bio-decagon-effectcategories.csv"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class DecagonDataLoader:
    """Carrega e filtra os dados brutos do Decagon, gerando um subconjunto
    reduzido adequado para treinamento em CPU."""

    def __init__(
        self,
        raw_dir: Path | str = RAW_DIR,
        processed_dir: Path | str = PROCESSED_DIR,
        n_drugs: int = N_DRUGS,
        n_side_effects: int = N_SIDE_EFFECTS,
        min_combo_per_se: int = MIN_COMBO_PER_SE,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.n_drugs = n_drugs
        self.n_side_effects = n_side_effects
        self.min_combo_per_se = min_combo_per_se

        # DataFrames (preenchidos após load/filter)
        self.combo_df: pd.DataFrame | None = None
        self.ppi_df: pd.DataFrame | None = None
        self.targets_df: pd.DataFrame | None = None
        self.mono_df: pd.DataFrame | None = None
        self.categories_df: pd.DataFrame | None = None

        # Conjuntos selecionados
        self.selected_drugs: set[str] = set()
        self.selected_side_effects: set[str] = set()
        self.selected_proteins: set[int] = set()

    # ── Leitura dos CSVs brutos ───────────────────────────────────────

    def _load_raw(self) -> None:
        """Lê todos os CSVs brutos para DataFrames pandas."""
        log.info("Lendo CSVs brutos de %s ...", self.raw_dir)

        t0 = time.perf_counter()

        self.combo_df = pd.read_csv(
            self.raw_dir / COMBO_FILE,
            dtype={"STITCH 1": str, "STITCH 2": str,
                   "Polypharmacy Side Effect": str, "Side Effect Name": str},
        )
        log.info("  combo:      %s linhas (%.1fs)",
                 f"{len(self.combo_df):>10,}", time.perf_counter() - t0)

        t1 = time.perf_counter()
        self.ppi_df = pd.read_csv(
            self.raw_dir / PPI_FILE,
            dtype={"Gene 1": int, "Gene 2": int},
        )
        log.info("  ppi:        %s linhas (%.1fs)",
                 f"{len(self.ppi_df):>10,}", time.perf_counter() - t1)

        t2 = time.perf_counter()
        self.targets_df = pd.read_csv(
            self.raw_dir / TARGETS_FILE,
            dtype={"STITCH": str, "Gene": int},
        )
        log.info("  targets:    %s linhas (%.1fs)",
                 f"{len(self.targets_df):>10,}", time.perf_counter() - t2)

        t3 = time.perf_counter()
        self.mono_df = pd.read_csv(
            self.raw_dir / MONO_FILE,
            dtype={"STITCH": str, "Individual Side Effect": str,
                   "Side Effect Name": str},
        )
        log.info("  mono:       %s linhas (%.1fs)",
                 f"{len(self.mono_df):>10,}", time.perf_counter() - t3)

        t4 = time.perf_counter()
        self.categories_df = pd.read_csv(
            self.raw_dir / CATEGORIES_FILE,
            dtype={"Side Effect": str, "Side Effect Name": str,
                   "Disease Class": str},
        )
        log.info("  categories: %s linhas (%.1fs)",
                 f"{len(self.categories_df):>10,}", time.perf_counter() - t4)

        log.info("Leitura concluída em %.1fs total.", time.perf_counter() - t0)

    # ── Seleção de drogas ─────────────────────────────────────────────

    def _select_top_drugs(self) -> None:
        """Seleciona as top-N drogas com maior cobertura no TWOSIDES.

        Cobertura = número total de pares em que a droga participa
        (contando tanto como droga 1 quanto droga 2).
        """
        log.info("Selecionando top %d drogas por cobertura ...", self.n_drugs)

        # Conta aparições de cada droga em qualquer posição do par
        drug_counts = pd.concat([
            self.combo_df["STITCH 1"],
            self.combo_df["STITCH 2"],
        ]).value_counts()

        top_drugs = set(drug_counts.head(self.n_drugs).index)
        self.selected_drugs = top_drugs

        log.info("  Drogas selecionadas: %d", len(self.selected_drugs))
        log.info("  Cobertura mín/máx: %d / %d pares",
                 drug_counts[drug_counts.index.isin(top_drugs)].min(),
                 drug_counts[drug_counts.index.isin(top_drugs)].max())

    # ── Filtragem do combo ────────────────────────────────────────────

    def _filter_combo(self) -> None:
        """Filtra combo para pares em que AMBAS as drogas estão no subconjunto,
        e depois seleciona os top-K efeitos adversos mais frequentes."""
        log.info("Filtrando pares de drogas (combo) ...")

        # Manter apenas pares onde ambas as drogas estão selecionadas
        mask = (
            self.combo_df["STITCH 1"].isin(self.selected_drugs) &
            self.combo_df["STITCH 2"].isin(self.selected_drugs)
        )
        filtered = self.combo_df[mask].copy()
        log.info("  Pares com ambas drogas no subconjunto: %s",
                 f"{len(filtered):,}")

        # Contar frequência de cada efeito adverso
        se_counts = filtered["Polypharmacy Side Effect"].value_counts()
        log.info("  Efeitos adversos distintos (pré-filtro): %d", len(se_counts))

        # Aplicar filtro mínimo de pares por efeito
        se_counts = se_counts[se_counts >= self.min_combo_per_se]

        # Selecionar top-K efeitos
        top_se = set(se_counts.head(self.n_side_effects).index)
        self.selected_side_effects = top_se

        log.info("  Efeitos adversos selecionados (top %d): %d",
                 self.n_side_effects, len(self.selected_side_effects))

        # Filtro final do combo: ambas drogas + efeito no subconjunto
        self.combo_df = filtered[
            filtered["Polypharmacy Side Effect"].isin(self.selected_side_effects)
        ].reset_index(drop=True)

        log.info("  Pares finais (combo filtrado): %s",
                 f"{len(self.combo_df):,}")

    # ── Filtragem de targets e proteínas ─────────────────────────────

    def _filter_targets(self) -> None:
        """Filtra drug→protein para drogas selecionadas e extrai proteínas."""
        log.info("Filtrando arestas drug → protein (targets) ...")

        self.targets_df = self.targets_df[
            self.targets_df["STITCH"].isin(self.selected_drugs)
        ].reset_index(drop=True)

        self.selected_proteins = set(self.targets_df["Gene"].unique())

        log.info("  Arestas drug→protein: %s", f"{len(self.targets_df):,}")
        log.info("  Proteínas conectadas: %d", len(self.selected_proteins))

    # ── Filtragem da PPI ──────────────────────────────────────────────

    def _filter_ppi(self) -> None:
        """Filtra PPI para manter apenas arestas entre proteínas conectadas."""
        log.info("Filtrando rede PPI ...")

        mask = (
            self.ppi_df["Gene 1"].isin(self.selected_proteins) &
            self.ppi_df["Gene 2"].isin(self.selected_proteins)
        )
        self.ppi_df = self.ppi_df[mask].reset_index(drop=True)

        log.info("  Arestas PPI filtradas: %s", f"{len(self.ppi_df):,}")

    # ── Filtragem do mono ─────────────────────────────────────────────

    def _filter_mono(self) -> None:
        """Filtra efeitos individuais para as drogas selecionadas."""
        log.info("Filtrando efeitos individuais (mono) ...")

        self.mono_df = self.mono_df[
            self.mono_df["STITCH"].isin(self.selected_drugs)
        ].reset_index(drop=True)

        log.info("  Efeitos mono filtrados: %s", f"{len(self.mono_df):,}")

    # ── Salvamento ────────────────────────────────────────────────────

    def _save_processed(self) -> None:
        """Salva DataFrames processados e metadados em dataset/processed/."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        log.info("Salvando dados processados em %s ...", self.processed_dir)

        self.combo_df.to_csv(
            self.processed_dir / "combo_filtered.csv", index=False
        )
        self.ppi_df.to_csv(
            self.processed_dir / "ppi_filtered.csv", index=False
        )
        self.targets_df.to_csv(
            self.processed_dir / "targets_filtered.csv", index=False
        )
        self.mono_df.to_csv(
            self.processed_dir / "mono_filtered.csv", index=False
        )
        self.categories_df.to_csv(
            self.processed_dir / "categories.csv", index=False
        )

        # Metadados do processamento
        metadata = {
            "n_drugs": len(self.selected_drugs),
            "n_side_effects": len(self.selected_side_effects),
            "n_proteins": len(self.selected_proteins),
            "n_combo_edges": len(self.combo_df),
            "n_ppi_edges": len(self.ppi_df),
            "n_target_edges": len(self.targets_df),
            "n_mono_records": len(self.mono_df),
            "config": {
                "N_DRUGS": self.n_drugs,
                "N_SIDE_EFFECTS": self.n_side_effects,
                "MIN_COMBO_PER_SE": self.min_combo_per_se,
            },
            "drug_ids": sorted(self.selected_drugs),
            "side_effect_ids": sorted(self.selected_side_effects),
            "protein_ids": sorted(int(p) for p in self.selected_proteins),
        }

        meta_path = self.processed_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        log.info("  Metadados salvos em %s", meta_path.name)
        log.info("Processamento concluído ✓")

    # ── Pipeline completo ─────────────────────────────────────────────

    def run(self) -> None:
        """Executa o pipeline completo: load → filter → save."""
        t0 = time.perf_counter()

        self._load_raw()
        self._select_top_drugs()
        self._filter_combo()
        self._filter_targets()
        self._filter_ppi()
        self._filter_mono()
        self._save_processed()

        elapsed = time.perf_counter() - t0
        log.info("Pipeline total: %.1fs", elapsed)

    # ── Resumo ────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Retorna um dicionário com estatísticas do subconjunto."""
        return {
            "drugs": len(self.selected_drugs),
            "proteins": len(self.selected_proteins),
            "side_effects": len(self.selected_side_effects),
            "combo_edges": len(self.combo_df) if self.combo_df is not None else 0,
            "ppi_edges": len(self.ppi_df) if self.ppi_df is not None else 0,
            "target_edges": len(self.targets_df) if self.targets_df is not None else 0,
        }


# ─────────────────────────────────────────────────────────────────────
# Execução direta
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loader = DecagonDataLoader()
    loader.run()

    print("\n" + "=" * 60)
    print("RESUMO DO SUBCONJUNTO")
    print("=" * 60)
    for k, v in loader.summary().items():
        print(f"  {k:<20s}: {v:>8,}")
    print("=" * 60)
