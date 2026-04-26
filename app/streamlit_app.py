"""
Decagon Mini — Polypharmacy Adverse Effect Prediction System

Clinical-grade Streamlit interface for predicting potential adverse
effects from drug combinations, powered by the Decagon graph
convolutional network model (Zitnik et al., 2018).

Usage:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.model.decagon import DecagonModel, build_homogeneous_graph
from src.data.graph_builder import DecagonGraphBuilder

# ─────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Decagon Mini — Polypharmacy Side Effect Predictor",
    page_icon="Rx",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
MODEL_PATH = PROJECT_ROOT / "saved_models" / "best_model.pt"
RESULTS_DIR = PROJECT_ROOT / "results"

# ─────────────────────────────────────────────────────────────────────
# Custom CSS — Clinical-grade styling
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Libre Baskerville */
    @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Libre Baskerville', 'Georgia', serif;
    }

    /* Global beige background */
    .stApp {
        background-color: #faf6f0;
    }

    /* Header bar — warm dark tone */
    .clinical-header {
        background: linear-gradient(135deg, #3e2c1c 0%, #5a3e2b 50%, #6d4c35 100%);
        padding: 1.5rem 2rem;
        border-radius: 6px;
        margin-bottom: 1.5rem;
        color: #faf6f0;
        box-shadow: 0 2px 10px rgba(62, 44, 28, 0.25);
        border-bottom: 3px solid #8c6d52;
    }
    .clinical-header h1 {
        margin: 0; font-size: 1.5rem; font-weight: 700; letter-spacing: -0.01em;
    }
    .clinical-header p {
        margin: 0.3rem 0 0 0; font-size: 0.82rem; opacity: 0.8; font-weight: 400;
    }

    /* Metric cards */
    .metric-card {
        background: #fffcf7;
        border: 1px solid #ddd5c8;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-card .label {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8c7a6a;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #3e2c1c;
    }

    /* Risk level badges */
    .risk-high {
        background: #fce8e6; color: #a83227; border: 1px solid #e0b4ae;
        padding: 0.2rem 0.7rem; border-radius: 12px; font-weight: 700;
        font-size: 0.72rem; display: inline-block;
    }
    .risk-moderate {
        background: #fef5e7; color: #b7791f; border: 1px solid #e8d5a8;
        padding: 0.2rem 0.7rem; border-radius: 12px; font-weight: 700;
        font-size: 0.72rem; display: inline-block;
    }
    .risk-low {
        background: #e8f5e9; color: #1b6d2f; border: 1px solid #a5d6a7;
        padding: 0.2rem 0.7rem; border-radius: 12px; font-weight: 700;
        font-size: 0.72rem; display: inline-block;
    }

    /* Sidebar — darker beige tone */
    section[data-testid="stSidebar"] {
        background: #ede4d6 !important;
        border-right: 1px solid #d4c9b8;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        background: #ede4d6 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label {
        font-weight: 700; color: #3e2c1c; font-size: 0.85rem;
    }

    /* Table styling */
    .dataframe {
        font-size: 0.82rem !important;
    }
    .dataframe thead th {
        background: #f0ebe3 !important;
        color: #3e2c1c !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        font-size: 0.70rem !important;
        letter-spacing: 0.05em;
    }

    /* Section headers */
    .section-header {
        font-size: 1rem;
        font-weight: 700;
        color: #3e2c1c;
        border-bottom: 2px solid #8c6d52;
        padding-bottom: 0.4rem;
        margin: 1.2rem 0 0.8rem 0;
    }

    /* Info panel */
    .info-panel {
        background: #f5f0e8;
        border-left: 4px solid #8c6d52;
        padding: 0.8rem 1rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.82rem;
        color: #3e2c1c;
        margin: 0.8rem 0;
    }

    /* Warning panel */
    .warning-panel {
        background: #f5efe5;
        border-left: 4px solid #b7791f;
        padding: 0.8rem 1rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.82rem;
        color: #5d4037;
        margin: 0.8rem 0;
    }

    /* Footer */
    .clinical-footer {
        margin-top: 2rem;
        padding: 1rem;
        text-align: center;
        font-size: 0.72rem;
        color: #9e9585;
        border-top: 1px solid #ddd5c8;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Libre Baskerville', 'Georgia', serif;
        font-weight: 700;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Data & Model Loading (cached)
# ─────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_everything():
    """Load model, graph data, and metadata. Cached across sessions."""

    # Load HeteroData graph
    data = DecagonGraphBuilder.load_graph(PROCESSED_DIR)
    n_drugs = data["drug"].num_nodes
    n_proteins = data["protein"].num_nodes
    se_order = sorted(data.side_effect_to_idx.keys())

    # Load model
    model = DecagonModel(
        n_drugs=n_drugs,
        n_proteins=n_proteins,
        n_drug_drug_rel=len(se_order),
        hidden_dim=64,
        embed_dim=64,
        n_bases=10,
        dropout=0.0,  # No dropout at inference
    )
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    # Build full homogeneous graph (all edges for best embeddings)
    edge_index, edge_type = build_homogeneous_graph(
        data, n_drugs, se_order
    )

    # Pre-compute drug embeddings
    with torch.no_grad():
        z_drug, z_protein = model.encode(
            data["drug"].x, data["protein"].x, edge_index, edge_type
        )

    # Load side effect name mapping
    combo_df = pd.read_csv(PROCESSED_DIR / "combo_filtered.csv")
    se_name_map = dict(zip(
        combo_df["Polypharmacy Side Effect"],
        combo_df["Side Effect Name"],
    ))

    # Load categories
    cat_df = pd.read_csv(PROCESSED_DIR / "categories.csv")
    se_category_map = dict(zip(cat_df["Side Effect"], cat_df["Disease Class"]))

    # Load metadata
    with open(PROCESSED_DIR / "metadata.json") as f:
        metadata = json.load(f)

    # Drug ID list
    drug_ids = sorted(data.drug_to_idx.keys())

    # Load training metrics (if available)
    train_log = None
    test_agg = None
    test_rel = None
    log_path = RESULTS_DIR / "training_log.csv"
    agg_path = RESULTS_DIR / "test_metrics_aggregated.csv"
    rel_path = RESULTS_DIR / "test_metrics_per_relation.csv"

    if log_path.exists():
        train_log = pd.read_csv(log_path)
    if agg_path.exists():
        test_agg = pd.read_csv(agg_path)
    if rel_path.exists():
        test_rel = pd.read_csv(rel_path)

    return {
        "model": model,
        "data": data,
        "z_drug": z_drug,
        "se_order": se_order,
        "se_name_map": se_name_map,
        "se_category_map": se_category_map,
        "drug_ids": drug_ids,
        "metadata": metadata,
        "train_log": train_log,
        "test_agg": test_agg,
        "test_rel": test_rel,
        "n_drugs": n_drugs,
        "n_proteins": n_proteins,
    }


def predict_side_effects(
    ctx: dict, drug_a: str, drug_b: str
) -> pd.DataFrame:
    """Predict side effects for a drug pair."""
    model = ctx["model"]
    z_drug = ctx["z_drug"]
    drug_to_idx = ctx["data"].drug_to_idx

    idx_a = drug_to_idx[drug_a]
    idx_b = drug_to_idx[drug_b]

    edge = torch.tensor([[idx_a], [idx_b]], dtype=torch.long)

    results = []
    with torch.no_grad():
        for se_idx, se_code in enumerate(ctx["se_order"]):
            score = model.decode(z_drug, edge, se_idx).sigmoid().item()
            se_name = ctx["se_name_map"].get(se_code, se_code)
            se_cat = ctx["se_category_map"].get(se_code, "—")

            if score >= 0.65:
                risk = "High"
            elif score >= 0.45:
                risk = "Moderate"
            else:
                risk = "Low"

            results.append({
                "CUI": se_code,
                "Side Effect": se_name.title(),
                "Category": se_cat.title(),
                "Score": score,
                "Risk": risk,
            })

    df = pd.DataFrame(results)
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # 1-based
    return df


def risk_badge(risk: str) -> str:
    """Return HTML badge for risk level."""
    cls = {"High": "risk-high", "Moderate": "risk-moderate", "Low": "risk-low"}
    return f'<span class="{cls.get(risk, "risk-low")}">{risk}</span>'


def format_cid(cid: str) -> str:
    """Format STITCH CID for display."""
    # Remove leading zeros: CID000002173 -> CID-2173
    numeric = cid.replace("CID", "").lstrip("0") or "0"
    return f"CID-{numeric}"


# ─────────────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────────────

def main():
    # Load data
    ctx = load_everything()

    # Header
    st.markdown("""
    <div class="clinical-header">
        <h1>Decagon Mini &mdash; Polypharmacy Side Effect Predictor</h1>
        <p>Graph Convolutional Network model for predicting adverse effects
        from drug combinations &nbsp;|&nbsp; Based on Zitnik et al., Bioinformatics 2018</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar: Drug Selection ───────────────────────────────────────
    with st.sidebar:
        st.markdown("### Drug Selection")
        st.markdown('<div class="info-panel">Select two drugs to analyze '
                    'potential polypharmacy adverse effects.</div>',
                    unsafe_allow_html=True)

        drug_options = {format_cid(d): d for d in ctx["drug_ids"]}
        display_names = list(drug_options.keys())

        drug_a_display = st.selectbox(
            "Drug A",
            display_names,
            index=0,
            key="drug_a",
        )
        drug_b_display = st.selectbox(
            "Drug B",
            display_names,
            index=min(1, len(display_names) - 1),
            key="drug_b",
        )

        drug_a = drug_options[drug_a_display]
        drug_b = drug_options[drug_b_display]

        if drug_a == drug_b:
            st.warning("Please select two different drugs.")

        st.markdown("---")
        predict_btn = st.button(
            "Analyze Interaction",
            use_container_width=True,
            type="primary",
            disabled=(drug_a == drug_b),
        )

        st.markdown("---")
        st.markdown("### System Information")
        st.markdown(f"""
        - **Drugs in model:** {ctx['n_drugs']}
        - **Proteins:** {ctx['n_proteins']}
        - **Side effect types:** {len(ctx['se_order'])}
        - **Model params:** 131,556
        - **Embedding dim:** 64
        """)

        st.markdown(
            '<div class="warning-panel">'
            "<strong>Research Use Only.</strong> This tool is for academic "
            "study purposes. Not intended for clinical decision-making."
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Main Content: Tabs ────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "Interaction Analysis",
        "Model Performance",
        "Dataset Overview",
    ])

    # ── Tab 1: Interaction Analysis ───────────────────────────────────
    with tab1:
        if predict_btn and drug_a != drug_b:
            st.markdown(
                f'<div class="section-header">'
                f'Predicted Adverse Effects: {drug_a_display} + {drug_b_display}'
                f'</div>',
                unsafe_allow_html=True,
            )

            df = predict_side_effects(ctx, drug_a, drug_b)

            # Summary metrics
            n_high = (df["Risk"] == "High").sum()
            n_mod = (df["Risk"] == "Moderate").sum()
            n_low = (df["Risk"] == "Low").sum()
            avg_score = df["Score"].mean()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(
                    '<div class="metric-card">'
                    '<div class="label">Avg. Risk Score</div>'
                    f'<div class="value">{avg_score:.3f}</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    '<div class="metric-card">'
                    '<div class="label">High Risk</div>'
                    f'<div class="value" style="color:#c0392b">{n_high}</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            with col3:
                st.markdown(
                    '<div class="metric-card">'
                    '<div class="label">Moderate Risk</div>'
                    f'<div class="value" style="color:#d68910">{n_mod}</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            with col4:
                st.markdown(
                    '<div class="metric-card">'
                    '<div class="label">Low Risk</div>'
                    f'<div class="value" style="color:#1e8449">{n_low}</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("")

            # Risk filter
            risk_filter = st.multiselect(
                "Filter by risk level:",
                ["High", "Moderate", "Low"],
                default=["High", "Moderate", "Low"],
            )
            filtered = df[df["Risk"].isin(risk_filter)]

            # Display table with risk badges
            display_df = filtered.copy()
            display_df["Risk Level"] = display_df["Risk"].apply(risk_badge)
            display_df["Score"] = display_df["Score"].apply(lambda x: f"{x:.4f}")
            display_df = display_df[["CUI", "Side Effect", "Category", "Score", "Risk Level"]]

            st.markdown(
                display_df.to_html(escape=False, index=True),
                unsafe_allow_html=True,
            )

            # Detail: score distribution
            st.markdown('<div class="section-header">Score Distribution</div>',
                        unsafe_allow_html=True)
            chart_df = df[["Side Effect", "Score"]].set_index("Side Effect")
            st.bar_chart(chart_df, height=350, color="#8c6d52")

        else:
            # Landing state
            st.markdown("""
            <div style="text-align:center; padding:4rem 2rem; color:#8c7a6a;">
                <p style="font-size:1.8rem; margin-bottom:0.3rem; color:#6d4c35; font-weight:700;">Rx</p>
                <h3 style="color:#3e2c1c; font-weight:700;">Drug Interaction Analyzer</h3>
                <p style="max-width:500px; margin:0.5rem auto; font-size:0.9rem; line-height:1.6;">
                    Select two drugs from the sidebar and click
                    <strong>Analyze Interaction</strong> to predict potential
                    polypharmacy adverse effects using the Decagon GCN model.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ── Tab 2: Model Performance ──────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">Model Evaluation Metrics</div>',
                    unsafe_allow_html=True)

        if ctx["test_agg"] is not None:
            agg = ctx["test_agg"]
            metrics_dict = dict(zip(agg["metric"], agg["value"]))

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(
                    '<div class="metric-card">'
                    '<div class="label">Macro AUROC</div>'
                    f'<div class="value">{float(metrics_dict.get("macro_auroc", 0)):.4f}</div>'
                    '</div>', unsafe_allow_html=True)
            with col2:
                st.markdown(
                    '<div class="metric-card">'
                    '<div class="label">Macro AUPRC</div>'
                    f'<div class="value">{float(metrics_dict.get("macro_auprc", 0)):.4f}</div>'
                    '</div>', unsafe_allow_html=True)
            with col3:
                st.markdown(
                    '<div class="metric-card">'
                    '<div class="label">Micro AUROC</div>'
                    f'<div class="value">{float(metrics_dict.get("micro_auroc", 0)):.4f}</div>'
                    '</div>', unsafe_allow_html=True)
            with col4:
                st.markdown(
                    '<div class="metric-card">'
                    '<div class="label">Best Epoch</div>'
                    f'<div class="value">{metrics_dict.get("best_epoch", "—")}</div>'
                    '</div>', unsafe_allow_html=True)

        # Training curves
        if ctx["train_log"] is not None:
            st.markdown('<div class="section-header">Training History</div>',
                        unsafe_allow_html=True)

            log_df = ctx["train_log"]

            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("**Loss per Epoch**")
                st.line_chart(
                    log_df.set_index("epoch")["loss"],
                    height=280,
                    color="#e74c3c",
                )
            with col_right:
                st.markdown("**Validation AUROC per Epoch**")
                st.line_chart(
                    log_df.set_index("epoch")["val_auroc"],
                    height=280,
                    color="#8c6d52",
                )

        # Per-relation performance
        if ctx["test_rel"] is not None:
            st.markdown('<div class="section-header">Per Side-Effect Performance</div>',
                        unsafe_allow_html=True)

            rel_df = ctx["test_rel"].copy()
            rel_df["Side Effect"] = rel_df["se_code"].map(
                lambda x: ctx["se_name_map"].get(x, x)
            ).str.title()
            rel_df["auroc"] = rel_df["auroc"].apply(lambda x: f"{x:.4f}")
            rel_df["auprc"] = rel_df["auprc"].apply(lambda x: f"{x:.4f}")
            rel_df = rel_df.rename(columns={
                "se_code": "CUI",
                "auroc": "AUROC",
                "auprc": "AUPRC",
                "n_pos": "Positives",
                "n_neg": "Negatives",
            })
            rel_df = rel_df[["CUI", "Side Effect", "AUROC", "AUPRC", "Positives", "Negatives"]]
            rel_df.index = range(1, len(rel_df) + 1)

            st.dataframe(rel_df, width="stretch", height=500)

    # ── Tab 3: Dataset Overview ───────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">Heterogeneous Graph Statistics</div>',
                    unsafe_allow_html=True)

        meta = ctx["metadata"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                '<div class="metric-card">'
                '<div class="label">Drug Nodes</div>'
                f'<div class="value">{meta["n_drugs"]}</div>'
                '</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(
                '<div class="metric-card">'
                '<div class="label">Protein Nodes</div>'
                f'<div class="value">{meta["n_proteins"]}</div>'
                '</div>', unsafe_allow_html=True)
        with col3:
            st.markdown(
                '<div class="metric-card">'
                '<div class="label">Side Effect Types</div>'
                f'<div class="value">{meta["n_side_effects"]}</div>'
                '</div>', unsafe_allow_html=True)
        with col4:
            st.markdown(
                '<div class="metric-card">'
                '<div class="label">Total Edges</div>'
                f'<div class="value">{meta["n_combo_edges"] + meta["n_ppi_edges"] + meta["n_target_edges"]:,}</div>'
                '</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Edge Distribution</div>',
                    unsafe_allow_html=True)

        edge_data = pd.DataFrame([
            {"Edge Type": "Protein-Protein (PPI)", "Count": meta["n_ppi_edges"]},
            {"Edge Type": "Drug-Protein (Targets)", "Count": meta["n_target_edges"]},
            {"Edge Type": "Drug-Drug (Side Effects)", "Count": meta["n_combo_edges"]},
        ])
        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.dataframe(edge_data, width="stretch", hide_index=True)
        with col_right:
            st.bar_chart(
                edge_data.set_index("Edge Type"),
                height=250,
                color="#6d4c35",
            )

        st.markdown('<div class="section-header">Configuration</div>',
                    unsafe_allow_html=True)
        config = meta.get("config", {})
        config_df = pd.DataFrame([
            {"Parameter": "N_DRUGS (top drugs by TWOSIDES coverage)", "Value": config.get("N_DRUGS", "—")},
            {"Parameter": "N_SIDE_EFFECTS (top frequent effects)", "Value": config.get("N_SIDE_EFFECTS", "—")},
            {"Parameter": "MIN_COMBO_PER_SE (min pairs per effect)", "Value": config.get("MIN_COMBO_PER_SE", "—")},
        ])
        st.dataframe(config_df, width="stretch", hide_index=True)

    # ── Footer ────────────────────────────────────────────────────────
    st.markdown(
        '<div class="clinical-footer">'
        "Decagon Mini &mdash; Reproduction of Zitnik, Agrawal &amp; Leskovec, "
        "<em>Modeling polypharmacy side effects with graph convolutional networks</em>, "
        "Bioinformatics 2018 &nbsp;|&nbsp; For academic research only"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
