"""
generate_esm2_embeddings.py — Gera embeddings ESM-2 para todas as
proteinas do grafo Decagon Mini.

Pipeline:
  1. Consulta a API REST do UniProt (Entrez -> sequencia FASTA)
  2. Roda ESM-2 (esm2_t6_8M_UR50D, 8M params, 320-dim) em CPU
  3. Mean pooling sobre residuos (exclui [CLS] e [EOS])
  4. Salva tensor [n_proteins, 320] e metadados JSON

Uso:
    python scripts/generate_esm2_embeddings.py
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests
import torch
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────
# Configuracao
# ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
ESM_MAX_LEN = 1022  # Janela maxima do ESM-2 (sem tokens especiais)
ESM_DIM = 320       # Dimensao de saida do esm2_t6_8M

UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb/search"
API_DELAY = 0.5     # Rate limit em segundos

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Etapa 1: Mapeamento Entrez -> UniProt + Sequencia
# ─────────────────────────────────────────────────────────────────────

def fetch_sequence(entrez_id: int) -> dict:
    """Consulta a API do UniProt para um Entrez Gene ID.

    Filtra por Swiss-Prot (reviewed=true) e escolhe a isoforma
    canonica de menor comprimento.

    Returns:
        dict: entrez_id, uniprot_ac, length, sequence, fallback
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
        log.warning("API error for Entrez %d: %s", entrez_id, e)
        return {
            "entrez_id": entrez_id,
            "uniprot_ac": "ORPHAN",
            "length": 0,
            "sequence": "",
            "fallback": True,
        }

    results = data.get("results", [])
    if len(results) == 0:
        log.warning("Orphan: Entrez %d has no Swiss-Prot entry", entrez_id)
        return {
            "entrez_id": entrez_id,
            "uniprot_ac": "ORPHAN",
            "length": 0,
            "sequence": "",
            "fallback": True,
        }

    # Escolher a isoforma de menor comprimento
    entries = []
    for r in results:
        acc = r.get("primaryAccession", "?")
        seq_info = r.get("sequence", {})
        length = seq_info.get("length", 0)
        seq_val = seq_info.get("value", "")
        entries.append({
            "accession": acc,
            "length": length,
            "sequence": seq_val,
        })

    entries.sort(key=lambda x: x["length"])
    best = entries[0]

    return {
        "entrez_id": entrez_id,
        "uniprot_ac": best["accession"],
        "length": best["length"],
        "sequence": best["sequence"],
        "fallback": False,
    }


def fetch_all_sequences(entrez_ids: list[int]) -> list[dict]:
    """Busca sequencias para todos os Entrez IDs com rate limiting."""
    results = []
    for eid in tqdm(entrez_ids, desc="UniProt API", unit="prot"):
        info = fetch_sequence(eid)
        results.append(info)
        time.sleep(API_DELAY)
    return results


# ─────────────────────────────────────────────────────────────────────
# Etapa 2: Inferencia ESM-2
# ─────────────────────────────────────────────────────────────────────

def generate_embeddings(seq_data: list[dict]) -> torch.Tensor:
    """Roda ESM-2 sobre as sequencias e retorna embeddings [N, 320].

    Mean pooling sobre os token embeddings da ultima camada,
    excluindo [CLS] (posicao 0) e [EOS] (ultima posicao).
    """
    from transformers import AutoTokenizer, AutoModel

    log.info("Carregando ESM-2 (esm2_t6_8M_UR50D) ...")
    model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    log.info("ESM-2 carregado com sucesso.")

    n_proteins = len(seq_data)
    embeddings = torch.zeros(n_proteins, ESM_DIM, dtype=torch.float32)
    n_truncated = 0

    with torch.no_grad():
        for i, entry in enumerate(
            tqdm(seq_data, desc="ESM-2 inference", unit="prot")
        ):
            if entry["fallback"]:
                # Manter zeros para orfaos
                continue

            seq = entry["sequence"]

            # Truncar se necessario
            truncated = False
            if len(seq) > ESM_MAX_LEN:
                seq = seq[:ESM_MAX_LEN]
                truncated = True
                n_truncated += 1

            # Tokenizar e rodar
            inputs = tokenizer(
                seq, return_tensors="pt", add_special_tokens=True
            )
            outputs = model(**inputs)

            # outputs.last_hidden_state: [1, seq_len+2, 320]
            # Excluir [CLS] (pos 0) e [EOS] (ultima pos)
            hidden = outputs.last_hidden_state[0]  # [seq_len+2, 320]
            residue_embeddings = hidden[1:-1]       # [seq_len, 320]

            # Mean pooling
            embeddings[i] = residue_embeddings.mean(dim=0)

    log.info(
        "Inferencia concluida: %d proteinas, %d truncadas",
        n_proteins, n_truncated,
    )
    return embeddings, n_truncated


# ─────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.perf_counter()

    # ── Carregar proteinas do grafo ───────────────────────────────────
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
    n_proteins = len(all_genes)
    log.info("Proteinas no grafo: %d", n_proteins)

    # ── Etapa 1: Buscar sequencias ────────────────────────────────────
    log.info("Etapa 1/2: Consultando API do UniProt ...")
    seq_data = fetch_all_sequences(all_genes)

    n_orphans = sum(1 for s in seq_data if s["fallback"])
    log.info("Sequencias obtidas: %d ok, %d orfaos", n_proteins - n_orphans, n_orphans)

    # ── Etapa 2: Gerar embeddings ESM-2 ──────────────────────────────
    log.info("Etapa 2/2: Rodando ESM-2 na CPU ...")
    embeddings, n_truncated = generate_embeddings(seq_data)

    # ── Validacao ─────────────────────────────────────────────────────
    assert embeddings.shape == (n_proteins, ESM_DIM), (
        f"Shape incorreto: {embeddings.shape} != ({n_proteins}, {ESM_DIM})"
    )
    assert not torch.isnan(embeddings).any(), "Tensor contem NaN!"
    assert not torch.isinf(embeddings).any(), "Tensor contem Inf!"

    norms = embeddings.norm(dim=1)
    mean_norm = norms.mean().item()
    log.info("Validacao OK: shape=%s, norma media=%.2f", embeddings.shape, mean_norm)

    # ── Salvar tensor ─────────────────────────────────────────────────
    out_tensor = PROCESSED_DIR / "protein_esm2.pt"
    torch.save(embeddings, out_tensor)
    log.info("Tensor salvo em %s", out_tensor)

    # ── Salvar metadados ──────────────────────────────────────────────
    metadata = []
    for entry in seq_data:
        metadata.append({
            "entrez_id": entry["entrez_id"],
            "uniprot_ac": entry["uniprot_ac"],
            "length": entry["length"],
            "fallback": entry["fallback"],
        })

    out_meta = PROCESSED_DIR / "protein_esm2_metadata.json"
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    log.info("Metadados salvos em %s", out_meta)

    # ── Resumo final ──────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    print()
    print("=" * 60)
    print("  ESM-2 EMBEDDING GENERATION - SUMMARY")
    print("=" * 60)
    print(f"  Proteins processed:  {n_proteins}")
    print(f"  Orphans (fallback):  {n_orphans}")
    print(f"  Truncated (>1022):   {n_truncated}")
    print(f"  Embedding dim:       {ESM_DIM}")
    print(f"  Mean norm:           {mean_norm:.2f}")
    print(f"  Tensor saved:        {out_tensor}")
    print(f"  Metadata saved:      {out_meta}")
    print(f"  Total time:          {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
