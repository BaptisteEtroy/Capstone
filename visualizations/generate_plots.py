#!/usr/bin/env python3
"""
Generate all thesis figures from medical_outputs JSON data.
Outputs PNGs to visualizations/figures/.

Usage:
    python visualizations/generate_plots.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO    = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAYERS = [4, 8, 12]


def layer_dir(layer: int) -> Path:
    return REPO / "medical_outputs" / f"layer_{layer}"


def load(layer: int, filename: str):
    with open(layer_dir(layer) / filename) as f:
        return json.load(f)


# ── Style (clean white) ────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f9f9f9",
    "axes.edgecolor":    "#cccccc",
    "axes.labelcolor":   "#333333",
    "axes.titlecolor":   "#111111",
    "axes.grid":         True,
    "grid.color":        "#e0e0e0",
    "grid.linewidth":    0.6,
    "xtick.color":       "#555555",
    "ytick.color":       "#555555",
    "text.color":        "#333333",
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "#cccccc",
    "font.family":       "sans-serif",
    "font.size":         9,
})

# Per-layer palette (works well on white)
L_COLORS = {4: "#3b82f6", 8: "#8b5cf6", 12: "#10b981"}

# Dataset palette
DS_COLORS = {"medmcqa": "#3b82f6", "pubmed_qa": "#8b5cf6", "pubmed_abs": "#10b981"}
DS_LABELS = {"medmcqa": "MedMCQA", "pubmed_qa": "PubMed QA", "pubmed_abs": "PubMed Abs"}

# Category order (consistent across figures)
CATEGORIES = [
    "Clinical/Diagnostic",
    "Pharmacology",
    "Biochemical/Molecular",
    "Anatomical",
    "Research Methodology",
    "Epidemiological",
    "Microbiology/Infectious",
    "Oncology",
    "Mental Health/Psych",
    "Structural/Linguistic",
    "General/Other",
]

CAT_COLORS = [
    "#ef4444", "#f97316", "#eab308", "#22c55e",
    "#3b82f6", "#8b5cf6", "#06b6d4", "#f43f5e",
    "#a855f7", "#ec4899", "#94a3b8",
]


def make_fig(figsize=(9, 4.5)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def make_fig_multi(nrows, ncols, figsize=(14, 4.5), **kwargs):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, axes


def save(fig, name: str):
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.relative_to(REPO)}")


# =============================================================================
# 1. Dataset Table
# =============================================================================
def fig_dataset_table():
    """Table: MedMCQA / PubMed QA / PubMed Abstracts — sample count, token count, avg seq len."""
    source_list = json.load(open(layer_dir(8) / "source_list.json"))
    src_counts  = Counter(s.split(":")[0] for s in source_list)
    total_tokens = 46_331_504   # from summary.json — identical across layers
    total_docs   = sum(src_counts.values())

    datasets   = ["medmcqa", "pubmed_qa", "pubmed_abs"]
    ds_display = ["MedMCQA", "PubMed QA", "PubMed Abstracts"]

    rows = []
    for ds in datasets:
        n_samples       = src_counts.get(ds, 0)
        n_tokens_approx = round(total_tokens * (n_samples / total_docs))
        avg_seq         = round(n_tokens_approx / n_samples) if n_samples else 0
        rows.append([f"{n_samples:,}", f"~{n_tokens_approx:,}", f"~{avg_seq}"])

    col_labels = ["Source", "Samples", "Tokens (approx.)", "Avg seq len (tokens)"]
    cell_data  = [[ds_display[i]] + rows[i] for i in range(3)]

    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.axis("off")

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.26, 0.18, 0.28, 0.28],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#e8eaf6")
            cell.set_text_props(color="#111111", fontweight="bold")
        else:
            ds_key = datasets[r - 1]
            cell.set_facecolor("#f9f9f9" if r % 2 == 0 else "white")
            if c == 0:
                cell.set_text_props(color=DS_COLORS[ds_key], fontweight="semibold")
            else:
                cell.set_text_props(color="#333333")

    ax.set_title("Dataset Breakdown by Source", fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    save(fig, "01_dataset_table")


# =============================================================================
# 2. Training Loss — Final Cross-Layer Run
# =============================================================================
def fig_training_loss_crosslayer():
    fig, ax = make_fig(figsize=(8, 4.5))
    epochs = list(range(1, 9))
    for layer in LAYERS:
        th = load(layer, "training_history.json")
        ax.plot(epochs, th["epoch_losses"], marker="o", markersize=5, linewidth=2,
                color=L_COLORS[layer], label=f"Layer {layer}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Loss — Cross-Layer Run (L4 / L8 / L12)", fontsize=12, fontweight="bold")
    ax.set_xticks(epochs)
    ax.legend()
    fig.tight_layout()
    save(fig, "02_training_loss_crosslayer")


# =============================================================================
# 3. Source Attribution Stacked Bar — fractions per category (Layer 8)
# =============================================================================
def fig_source_attribution_by_category():
    lf = load(8, "labeled_features.json")

    cat_ds: dict = {cat: defaultdict(float) for cat in CATEGORIES}
    cat_n: dict  = {cat: 0 for cat in CATEGORIES}
    for feat in lf:
        cat = feat.get("category", "General/Other")
        if cat not in cat_ds:
            cat = "General/Other"
        for ds, frac in feat.get("source_breakdown", {}).items():
            cat_ds[cat][ds] += frac
        cat_n[cat] += 1

    datasets_order = ["medmcqa", "pubmed_qa", "pubmed_abs"]
    cat_fracs = {}
    for cat in CATEGORIES:
        total = sum(cat_ds[cat].values()) or 1
        cat_fracs[cat] = {ds: cat_ds[cat].get(ds, 0) / total for ds in datasets_order}

    fig, ax = make_fig(figsize=(12, 5))
    x      = np.arange(len(CATEGORIES))
    bottom = np.zeros(len(CATEGORIES))
    for ds in datasets_order:
        vals = np.array([cat_fracs[cat][ds] for cat in CATEGORIES])
        ax.bar(x, vals, 0.55, bottom=bottom, color=DS_COLORS[ds],
               label=DS_LABELS[ds], alpha=0.85, edgecolor="white", linewidth=0.4)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("/", "/\n") for c in CATEGORIES], fontsize=8)
    ax.set_ylabel("Source Fraction")
    ax.set_ylim(0, 1.05)
    ax.set_title("Source Attribution per Feature Category  (Layer 8)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "03_source_attribution_by_category")


# =============================================================================
# 4. Cross-Layer Cosine Similarity Histograms (3 panels)
# =============================================================================
def fig_cosine_similarity_histograms():
    cross     = json.load(open(REPO / "medical_outputs" / "cross_layer_analysis.json"))
    pairs_cfg = [("4_8", "L4 ↔ L8"), ("4_12", "L4 ↔ L12"), ("8_12", "L8 ↔ L12")]
    colors    = ["#3b82f6", "#f59e0b", "#8b5cf6"]

    fig, axes = make_fig_multi(1, 3, figsize=(14, 4.5))

    for ax, (key, title), color in zip(axes, pairs_cfg, colors):
        pair   = cross["pairs"][key]
        hist   = pair["a_to_b"]["histogram"]
        bins   = hist["bins"]
        counts = hist["counts"]
        widths = [bins[i + 1] - bins[i] for i in range(len(counts))]

        ax.bar(bins[:-1], counts, width=widths, align="edge",
               color=color, alpha=0.75, edgecolor="white", linewidth=0.5)

        mean_sim   = pair["a_to_b"]["mean_max_similarity"]
        shared_pct = pair["a_to_b"]["shared_features_pct"]
        ax.axvline(mean_sim, color="#333333", linewidth=1.2, linestyle="--")
        ax.text(mean_sim + 0.02, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 300,
                f"mean={mean_sim:.2f}", color="#333333", fontsize=7.5, va="top")

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Max cosine similarity")
        ax.set_ylabel("Feature count" if ax is axes[0] else "")
        ax.set_xlim(0, 1)
        ax.text(0.98, 0.97, f"shared ≥0.9: {shared_pct:.1f}%",
                transform=ax.transAxes, ha="right", va="top",
                color="#555555", fontsize=7.5)

    fig.suptitle("Cross-Layer Decoder Cosine Similarity (max per feature)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "04_cosine_similarity_histograms")


# =============================================================================
# 5. L0 Sparsity & Dead Feature Rate — Training Progression (real log data)
# =============================================================================
def fig_sparsity_progression():
    """
    L0 and dead feature % across real training milestones extracted from logs.
    All values are taken directly from the attempt summaries in logs.pdf —
    no fabricated numbers.

    Key milestones:
      Phase 1 — L1-penalty SAE on GPT-2 small (attempts 1, 4-6, 16)
      Phase 2 — TopK SAE on GPT-2 medium (attempt 22)
      Phase 3 — TopK SAE on Llama (attempts 31, 41, 42, 53 L4/L8/L12)
    """
    # Each entry: (short label, dead%, L0, phase)
    # Sources: logs.pdf attempt summaries
    milestones = [
        # Phase 1 — L1 on GPT-2 small
        ("Att 1\nGPT-2s\nL1=5e-3",        0.0,   1696,  1),   # att 1:  0 dead / 6144, L0=1696
        ("Att 4–6\nGPT-2s\nL1=1.0",       22.4,  1285,  1),   # att 4-6: 1378/6144=22.4%, L0=1285
        ("Att 16\nGPT-2m\nL1=50",         39.5,   121,  1),   # att 16: 3241/8192=39.5%, L0=121
        # Phase 2 — TopK on GPT-2 medium
        ("Att 22\nGPT-2m\nTopK K=100",    44.5,   100,  2),   # att 22: 3645/8192=44.5%, L0=100
        # Phase 3 — TopK on Llama
        ("Att 31\nLlama-1B\nK=100",       76.6,   100,  3),   # att 31: 6275/8192=76.6%, L0=100
        ("Att 41\nLlama-Ins\n8× K=96",    83.4,    96,  3),   # att 41: 13669/16384=83.4%, L0=96
        ("Att 42\nLlama-Ins\n4× K=64",    50.0,    64,  3),   # att 42: 4099/8192=50.0%, L0=64
        ("Final\nL4",                      16.7,    64,  3),   # att 53: 1372/8192=16.7%
        ("Final\nL8",                      47.1,    64,  3),   # att 53: 3861/8192=47.1%
        ("Final\nL12",                     15.4,    64,  3),   # att 53: 1264/8192=15.4%
    ]

    labels   = [m[0] for m in milestones]
    dead_pct = [m[1] for m in milestones]
    l0_vals  = [m[2] for m in milestones]
    phases   = [m[3] for m in milestones]

    phase_colors = {1: "#94a3b8", 2: "#f59e0b", 3: "#3b82f6"}
    bar_colors   = [phase_colors[p] for p in phases]

    # Highlight the final three bars
    bar_colors[-3] = L_COLORS[4]
    bar_colors[-2] = L_COLORS[8]
    bar_colors[-1] = L_COLORS[12]

    fig, ax1 = make_fig(figsize=(13, 5.5))
    ax2 = ax1.twinx()
    ax2.set_facecolor("#f9f9f9")

    x = np.arange(len(milestones))
    bars = ax1.bar(x, dead_pct, 0.55, color=bar_colors, alpha=0.82,
                   edgecolor="white", linewidth=0.5, zorder=2)

    # Value labels above bars
    for bar, val in zip(bars, dead_pct):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                 f"{val:.0f}%", ha="center", va="bottom", fontsize=7.5, color="#333333")

    # L0 line on secondary axis
    ax2.plot(x, l0_vals, marker="D", color="#dc2626", linewidth=2,
             markersize=6, zorder=5, label="L0 (activations/token)")
    for xi, l0 in zip(x, l0_vals):
        ax2.text(xi, l0 + 35, str(l0), ha="center", va="bottom",
                 fontsize=7, color="#dc2626")

    ax1.set_ylabel("Dead features (%)", fontsize=10)
    ax1.set_ylim(0, 100)
    ax2.set_ylabel("Avg L0 (activations / token)", fontsize=10, color="#dc2626")
    ax2.tick_params(axis="y", colors="#dc2626")
    ax2.set_ylim(0, 2000)
    ax2.spines["right"].set_edgecolor("#dc262644")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7.5)
    ax1.set_title("L0 Sparsity & Dead Feature Rate — Training Progression",
                  fontsize=12, fontweight="bold")
    ax1.grid(True, axis="y", zorder=0)

    # Phase dividers + labels
    dividers = [2.5, 3.5]   # after index 2 (end of phase 1), after index 3 (end of phase 2)
    for xd in dividers:
        ax1.axvline(xd, color="#cccccc", linewidth=1, linestyle="--", zorder=1)

    ax1.text(1.0,  93, "Phase 1\nL1-penalty\nGPT-2",  ha="center", fontsize=7.5, color="#555555",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc"))
    ax1.text(3.0,  93, "Phase 2\nTopK\nGPT-2",        ha="center", fontsize=7.5, color="#555555",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc"))
    ax1.text(6.75, 93, "Phase 3 — TopK on Llama 3.2-Instruct",
             ha="center", fontsize=7.5, color="#555555",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc"))

    # Legend for final bars
    patches = [
        mpatches.Patch(color="#94a3b8", alpha=0.82, label="L1-penalty (GPT-2 small)"),
        mpatches.Patch(color="#f59e0b", alpha=0.82, label="TopK (GPT-2 medium)"),
        mpatches.Patch(color="#3b82f6", alpha=0.82, label="TopK (Llama, intermediate)"),
        mpatches.Patch(color=L_COLORS[4],  alpha=0.82, label="Final run — Layer 4"),
        mpatches.Patch(color=L_COLORS[8],  alpha=0.82, label="Final run — Layer 8"),
        mpatches.Patch(color=L_COLORS[12], alpha=0.82, label="Final run — Layer 12"),
    ]
    ax1.legend(handles=patches, fontsize=7.5, loc="upper left",
               ncol=2, framealpha=0.9, edgecolor="#cccccc")

    fig.tight_layout()
    save(fig, "05_sparsity_progression")


# =============================================================================
# 6. Category Distribution — Layer 8 vs Layer 12
# =============================================================================
def fig_category_distribution_l8_l12():
    data = {}
    for layer in [8, 12]:
        lf    = load(layer, "labeled_features.json")
        total = len(lf)
        cnts  = Counter(f["category"] for f in lf)
        data[layer] = {cat: cnts.get(cat, 0) / total * 100 for cat in CATEGORIES}

    fig, ax = make_fig(figsize=(12, 5))
    x     = np.arange(len(CATEGORIES))
    width = 0.38

    ax.bar(x - width / 2, [data[8][c]  for c in CATEGORIES], width,
           color=L_COLORS[8],  alpha=0.82, label="Layer 8",
           edgecolor="white", linewidth=0.4)
    ax.bar(x + width / 2, [data[12][c] for c in CATEGORIES], width,
           color=L_COLORS[12], alpha=0.82, label="Layer 12",
           edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("/", "/\n") for c in CATEGORIES], fontsize=8)
    ax.set_ylabel("% of labeled features")
    ax.set_title("Feature Category Distribution — Layer 8 vs Layer 12",
                 fontsize=12, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save(fig, "06_category_distribution_l8_l12")


# =============================================================================
# 7. Source Attribution per Category — L4, L8, L12 side by side
# =============================================================================
def fig_source_per_category_all_layers():
    datasets_order = ["medmcqa", "pubmed_qa", "pubmed_abs"]

    fig, axes = make_fig_multi(1, 3, figsize=(18, 5))

    for ax, layer in zip(axes, LAYERS):
        lf     = load(layer, "labeled_features.json")
        cat_ds = {cat: defaultdict(float) for cat in CATEGORIES}
        cat_n  = {cat: 0 for cat in CATEGORIES}

        for feat in lf:
            cat = feat.get("category", "General/Other")
            if cat not in cat_ds:
                cat = "General/Other"
            for ds, frac in feat.get("source_breakdown", {}).items():
                cat_ds[cat][ds] += frac
            cat_n[cat] += 1

        x      = np.arange(len(CATEGORIES))
        bottom = np.zeros(len(CATEGORIES))
        for ds in datasets_order:
            vals = np.array([
                (cat_ds[cat].get(ds, 0) / (sum(cat_ds[cat].values()) or 1))
                if cat_n[cat] > 0 else 0.0
                for cat in CATEGORIES
            ])
            ax.bar(x, vals, 0.6, bottom=bottom, color=DS_COLORS[ds],
                   label=DS_LABELS[ds], alpha=0.85, edgecolor="white", linewidth=0.3)
            bottom += vals

        short = [c.split("/")[0][:8] for c in CATEGORIES]
        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=7, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Layer {layer}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Source fraction" if ax is axes[0] else "")
        if layer == LAYERS[-1]:
            ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Source Attribution per Feature Category (L4 / L8 / L12)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "07_source_per_category_all_layers")


# =============================================================================
# 8. Training Loss Curves — L4, L8, L12
# =============================================================================
def fig_training_loss_all_layers():
    fig, ax = make_fig(figsize=(8, 4.5))
    epochs  = list(range(1, 9))
    for layer in LAYERS:
        th = load(layer, "training_history.json")
        ax.plot(epochs, th["epoch_losses"], marker="o", markersize=5, linewidth=2,
                color=L_COLORS[layer], label=f"Layer {layer}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Loss Curves — L4, L8, L12", fontsize=12, fontweight="bold")
    ax.set_xticks(epochs)
    ax.legend()
    fig.tight_layout()
    save(fig, "08_training_loss_all_layers")


# =============================================================================
# 9. Dead Features per Epoch — L4, L8, L12
# =============================================================================
def fig_dead_features_per_epoch():
    fig, ax = make_fig(figsize=(9, 4.5))
    epochs  = list(range(1, 9))

    for layer in LAYERS:
        th = load(layer, "training_history.json")
        ax.plot(epochs, th["dead_features_per_epoch"], marker="o", markersize=5,
                linewidth=2, color=L_COLORS[layer], label=f"Layer {layer}")

    # Annotate L8 spike at epoch 2
    th8        = load(8, "training_history.json")
    spike_val  = th8["dead_features_per_epoch"][1]   # epoch 2 → index 1
    ax.annotate("L8 spike\n(epoch 2)", xy=(2, spike_val),
                xytext=(2.6, spike_val + 100),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1),
                color="#555555", fontsize=8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dead features")
    ax.set_title("Dead Features per Epoch — L4, L8, L12", fontsize=12, fontweight="bold")
    ax.set_xticks(epochs)
    ax.legend()
    fig.tight_layout()
    save(fig, "09_dead_features_per_epoch")


# =============================================================================
# 10. Shannon Entropy Distributions — Violin Plot per Layer
# =============================================================================
def fig_entropy_violin():
    entropies = {}
    for layer in LAYERS:
        feats = load(layer, "features.json")
        entropies[layer] = [
            f["maxact_entropy"] for f in feats
            if f.get("maxact_entropy") is not None and f.get("frequency", 0) > 1e-6
        ]

    fig, ax = make_fig(figsize=(8, 5))
    positions = [1, 2, 3]
    data      = [entropies[layer] for layer in LAYERS]

    parts = ax.violinplot(data, positions=positions, showmedians=True,
                          showextrema=True, widths=0.6)
    for pc, layer in zip(parts["bodies"], LAYERS):
        pc.set_facecolor(L_COLORS[layer])
        pc.set_alpha(0.45)
        pc.set_edgecolor(L_COLORS[layer])

    for part_name in ("cbars", "cmins", "cmaxes", "cmedians"):
        if part_name in parts:
            parts[part_name].set_color("#444444")
            parts[part_name].set_linewidth(1.2)

    for pos, layer in zip(positions, LAYERS):
        ax.scatter(pos, np.mean(entropies[layer]), color=L_COLORS[layer],
                   zorder=5, s=45, marker="D", edgecolors="#333333", linewidths=0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"Layer {l}" for l in LAYERS], fontsize=10)
    ax.set_ylabel("Shannon Entropy (bits)")
    ax.set_title("MaxAct Token Entropy Distribution per Layer",
                 fontsize=12, fontweight="bold")

    patches = [mpatches.Patch(color=L_COLORS[l], alpha=0.6, label=f"Layer {l}") for l in LAYERS]
    ax.legend(handles=patches, fontsize=9)
    fig.tight_layout()
    save(fig, "10_entropy_violin")


# =============================================================================
# Main
# =============================================================================
FIGURES = [
    ("01 Dataset table",                        fig_dataset_table),
    ("02 Training loss — cross-layer run",      fig_training_loss_crosslayer),
    ("03 Source attribution by category",       fig_source_attribution_by_category),
    ("04 Cosine similarity histograms",         fig_cosine_similarity_histograms),
    ("05 Sparsity progression (real log data)",  fig_sparsity_progression),
    ("06 Category distribution L8 vs L12",      fig_category_distribution_l8_l12),
    ("07 Source per category — all layers",     fig_source_per_category_all_layers),
    ("08 Training loss — all layers",           fig_training_loss_all_layers),
    ("09 Dead features per epoch",              fig_dead_features_per_epoch),
    ("10 Entropy violin",                       fig_entropy_violin),
]


def main():
    print(f"Generating {len(FIGURES)} figures → {OUT_DIR}\n")
    for label, fn in FIGURES:
        print(f"[{label}]")
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
    print("\nDone.")


if __name__ == "__main__":
    main()
