"""
check_data.py — Verificação e estatísticas do grafo heterogêneo Decagon.

Carrega o HeteroData salvo e imprime:
  - Nós por tipo (drug, protein)
  - Arestas por tipo (PPI, targets, side effects)
  - Esparsidade por tipo de aresta
  - Top efeitos adversos (por número de pares)
  - Estatísticas de grau dos nós

Uso:
    python scripts/check_data.py
"""

from __future__ import annotations

import sys
import io
from pathlib import Path

# Força UTF-8 no stdout (resolve problemas de encoding no Windows)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Adiciona raiz do projeto ao path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch


def fmt(n: int | float) -> str:
    """Formata números com separador de milhar."""
    if isinstance(n, float):
        return f"{n:,.2f}"
    return f"{n:,}"


def sep(char: str = "=", width: int = 64) -> str:
    return char * width


def main() -> None:
    processed_dir = PROJECT_ROOT / "dataset" / "processed"
    graph_path = processed_dir / "decagon_hetero_graph.pt"

    if not graph_path.exists():
        print(f"[X] Grafo nao encontrado: {graph_path}")
        print("  Execute primeiro:")
        print("    python -m src.data.loader")
        print("    python -m src.data.graph_builder")
        sys.exit(1)

    print("Carregando grafo ...")
    data = torch.load(graph_path, weights_only=False)

    # ── Header ────────────────────────────────────────────────────────
    print()
    print(sep("="))
    print("  DECAGON MINI — ESTATISTICAS DO GRAFO".center(64))
    print(sep("="))

    # ── Nós ───────────────────────────────────────────────────────────
    print("  NOS POR TIPO")
    print(sep("-"))

    total_nodes = 0
    for node_type in data.node_types:
        n = data[node_type].num_nodes
        total_nodes += n
        feat_shape = tuple(data[node_type].x.shape) if hasattr(data[node_type], "x") else "N/A"
        print(f"  {node_type:<20s}  {fmt(n):>10s} nos    features: {str(feat_shape):<16s}")

    print(f"  {'TOTAL':<20s}  {fmt(total_nodes):>10s} nos")

    # ── Arestas ───────────────────────────────────────────────────────
    print(sep("-"))
    print("  ARESTAS POR TIPO")
    print(sep("-"))

    total_edges = 0
    se_edge_counts: list[tuple[str, int]] = []

    for edge_type in data.edge_types:
        src_type, rel, dst_type = edge_type
        n_edges = data[edge_type].edge_index.shape[1]
        total_edges += n_edges

        if rel.startswith("side_effect_"):
            se_code = rel.replace("side_effect_", "")
            se_edge_counts.append((se_code, n_edges))
        else:
            label = f"({src_type}, {rel}, {dst_type})"
            print(f"  {label:<42s}  {fmt(n_edges):>10s} arestas")

    # Resumo dos side effects
    n_se_types = len(se_edge_counts)
    n_se_edges = sum(c for _, c in se_edge_counts)
    label = f"(drug, side_effect_*, drug) x {n_se_types}"
    print(f"  {label:<42s}  {fmt(n_se_edges):>10s} arestas")
    print(f"  {'TOTAL':<42s}  {fmt(total_edges):>10s} arestas")

    # ── Esparsidade ───────────────────────────────────────────────────
    print(sep("-"))
    print("  ESPARSIDADE")
    print(sep("-"))

    n_drugs = data["drug"].num_nodes
    n_proteins = data["protein"].num_nodes

    # PPI
    ppi_key = ("protein", "interacts", "protein")
    if ppi_key in data.edge_types:
        n_ppi = data[ppi_key].edge_index.shape[1]
        max_ppi = n_proteins * (n_proteins - 1)
        sparsity_ppi = 1 - (n_ppi / max_ppi) if max_ppi > 0 else 0
        print(f"  PPI:         {sparsity_ppi:.6f}  ({fmt(n_ppi)} / {fmt(max_ppi)} possiveis)")

    # Drug-Protein
    target_key = ("drug", "targets", "protein")
    if target_key in data.edge_types:
        n_tgt = data[target_key].edge_index.shape[1]
        max_tgt = n_drugs * n_proteins
        sparsity_tgt = 1 - (n_tgt / max_tgt) if max_tgt > 0 else 0
        print(f"  Drug->Prot:  {sparsity_tgt:.6f}  ({fmt(n_tgt)} / {fmt(max_tgt)} possiveis)")

    # Drug-Drug (média por tipo de efeito)
    max_dd = n_drugs * (n_drugs - 1)
    if se_edge_counts and max_dd > 0:
        avg_edges_per_se = n_se_edges / n_se_types
        avg_sparsity = 1 - (avg_edges_per_se / max_dd)
        print(f"  Drug<>Drug:  {avg_sparsity:.6f}  (media por tipo de efeito)")

    # ── Top efeitos adversos ──────────────────────────────────────────
    print(sep("-"))
    print("  TOP 10 EFEITOS ADVERSOS (por no. de pares)")
    print(sep("-"))

    se_edge_counts.sort(key=lambda x: x[1], reverse=True)
    for i, (se_code, count) in enumerate(se_edge_counts[:10]):
        print(f"  {i+1:>2d}. {se_code:<12s}  {fmt(count):>8s} arestas")

    # ── Estatísticas de grau ──────────────────────────────────────────
    print(sep("-"))
    print("  ESTATISTICAS DE GRAU")
    print(sep("-"))

    # Grau dos nós drug (soma de todas as arestas incidentes)
    drug_degree = torch.zeros(n_drugs, dtype=torch.long)
    for edge_type in data.edge_types:
        src_type, _, dst_type = edge_type
        ei = data[edge_type].edge_index
        if src_type == "drug":
            drug_degree.scatter_add_(0, ei[0], torch.ones(ei.shape[1], dtype=torch.long))
        if dst_type == "drug":
            drug_degree.scatter_add_(0, ei[1], torch.ones(ei.shape[1], dtype=torch.long))

    protein_degree = torch.zeros(n_proteins, dtype=torch.long)
    for edge_type in data.edge_types:
        src_type, _, dst_type = edge_type
        ei = data[edge_type].edge_index
        if src_type == "protein":
            protein_degree.scatter_add_(0, ei[0], torch.ones(ei.shape[1], dtype=torch.long))
        if dst_type == "protein":
            protein_degree.scatter_add_(0, ei[1], torch.ones(ei.shape[1], dtype=torch.long))

    for name, deg in [("Drug", drug_degree), ("Protein", protein_degree)]:
        mn = deg.min().item()
        mx = deg.max().item()
        avg = deg.float().mean().item()
        med = deg.float().median().item()
        print(f"  {name:<10s}  min={fmt(mn):>6s}  max={fmt(mx):>6s}  avg={avg:>8.1f}  med={med:>6.0f}")

    # ── Mono features ─────────────────────────────────────────────────
    if hasattr(data["drug"], "mono_side_effects"):
        mono = data["drug"].mono_side_effects
        n_mono_effects = mono.shape[1]
        n_nonzero = int(mono.sum().item())
        density = n_nonzero / (mono.shape[0] * mono.shape[1]) * 100
        print(sep("-"))
        print(f"  Mono features: {n_drugs} x {n_mono_effects}, {fmt(n_nonzero)} nonzero ({density:.1f}%)")

    print(sep("="))
    print("  [OK] Verificacao concluida com sucesso!")
    print(sep("="))


if __name__ == "__main__":
    main()
