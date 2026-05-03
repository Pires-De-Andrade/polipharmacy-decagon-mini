"""
check_protein_mapping.py — Valida mapeamento Entrez Gene ID → UniProt AC
para uma amostra de 20 proteínas do grafo Decagon Mini.

Consulta a API REST do UniProt, filtra por Swiss-Prot (reviewed=true),
e escolhe a isoforma canônica de menor comprimento.

Uso:
    python scripts/check_protein_mapping.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

# ─────────────────────────────────────────────────────────────────────
# Configuração
# ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
SAMPLE_SIZE = 20
ESM_MAX_LEN = 1022  # Janela máxima do ESM-2 (sem tokens especiais)

UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb/search"


def get_uniprot_for_entrez(entrez_id: int) -> dict:
    """Consulta a API do UniProt para um Entrez Gene ID.

    Returns:
        dict com chaves: accession, length, status, truncated, n_hits
    """
    params = {
        "query": f"(xref:geneid-{entrez_id}) AND (reviewed:true)",
        "format": "json",
        "fields": "accession,reviewed,length,sequence",
        "size": "50",
    }

    try:
        resp = requests.get(UNIPROT_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return {
            "accession": "—",
            "length": 0,
            "status": f"error ({e})",
            "truncated": False,
            "n_hits": 0,
        }

    results = data.get("results", [])

    if len(results) == 0:
        return {
            "accession": "—",
            "length": 0,
            "status": "orphan",
            "truncated": False,
            "n_hits": 0,
        }

    # Ordenar por comprimento (menor primeiro) e pegar a canônica
    entries = []
    for r in results:
        acc = r.get("primaryAccession", "?")
        seq = r.get("sequence", {})
        length = seq.get("length", 0)
        entries.append({"accession": acc, "length": length})

    entries.sort(key=lambda x: x["length"])
    best = entries[0]

    if len(results) == 1:
        status = "ok"
    else:
        status = f"multiple ({len(results)})"

    return {
        "accession": best["accession"],
        "length": best["length"],
        "status": status,
        "truncated": best["length"] > ESM_MAX_LEN,
        "n_hits": len(results),
    }


def main():
    # ── Carregar proteínas do grafo ───────────────────────────────────
    ppi = pd.read_csv(
        PROCESSED_DIR / "ppi_filtered.csv",
        dtype={"Gene 1": int, "Gene 2": int},
    )
    targets = pd.read_csv(
        PROCESSED_DIR / "targets_filtered.csv",
        dtype={"STITCH": str, "Gene": int},
    )

    all_genes = sorted(
        set(ppi["Gene 1"]) | set(ppi["Gene 2"]) | set(targets["Gene"])
    )
    sample = all_genes[:SAMPLE_SIZE]

    print(f"Total de proteínas no grafo: {len(all_genes)}")
    print(f"Amostra para validação:      {SAMPLE_SIZE}")
    print()

    # ── Consultar UniProt ─────────────────────────────────────────────
    rows = []
    for i, entrez_id in enumerate(sample):
        info = get_uniprot_for_entrez(entrez_id)
        rows.append({
            "Entrez ID": entrez_id,
            "UniProt AC": info["accession"],
            "Length": info["length"],
            "Truncated?": "YES" if info["truncated"] else "no",
            "Status": info["status"],
        })
        print(f"  [{i+1:2d}/{SAMPLE_SIZE}] Entrez {entrez_id:>8d} -> "
              f"{info['accession']:>10s}  len={info['length']:>5d}  "
              f"{'TRUNC' if info['truncated'] else '  ok '}  "
              f"{info['status']}")

        # Rate limiting (respeita a API do UniProt)
        time.sleep(0.5)

    # ── Tabela resumo ─────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    print("\n" + "=" * 72)
    print("RESULTADO DA AMOSTRA")
    print("=" * 72)
    print(df.to_string(index=False))

    # ── Contagem por categoria ────────────────────────────────────────
    print("\n" + "-" * 40)
    counts = df["Status"].value_counts()
    for status, count in counts.items():
        print(f"  {status:>20s}: {count}")

    n_trunc = (df["Truncated?"] == "YES").sum()
    print(f"  {'truncated (>1022)':>20s}: {n_trunc}")
    print("-" * 40)


if __name__ == "__main__":
    main()
