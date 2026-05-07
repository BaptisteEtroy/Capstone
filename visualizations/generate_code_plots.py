#!/usr/bin/env python3
"""
Generate thesis figures from code_outputs JSON data.
Outputs PNGs to visualizations/code_figures/.

Usage:
    python visualizations/generate_code_plots.py
"""

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO    = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "code_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAYERS = [4, 8, 12]


def layer_dir(layer: int) -> Path:
    return REPO / "code_outputs" / f"layer_{layer}"


def load(layer: int, filename: str):
    with open(layer_dir(layer) / filename) as f:
        return json.load(f)


# ── Style ──────────────────────────────────────────────────────────────────────
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

L_COLORS = {4: "#3b82f6", 8: "#8b5cf6", 12: "#10b981"}

DS_COLORS = {
    "code_search_net": "#3b82f6",
    "code_alpaca":     "#f97316",
    "flytech_python":  "#10b981",
}
DS_LABELS = {
    "code_search_net": "CodeSearchNet",
    "code_alpaca":     "CodeAlpaca-20k",
    "flytech_python":  "flytech/python-25k",
}
DS_ORDER = ["code_search_net", "code_alpaca", "flytech_python"]

# Code domain categories — ordered from most to least frequent at L12
CATEGORIES = [
    "Structural/Linguistic",
    "General/Other",
    "Functions/Methods",
    "Data Structures",
    "Databases/Queries",
    "Network/Protocols",
    "Types/Annotations",
    "Control Flow",
    "Concurrency/Async",
    "IO/File Operations",
    "Numeric/Math",
    "OOP/Classes",
    "String/Text Processing",
    "Error Handling",
    "Version Control/DevOps",
    "Testing/QA",
    "ML/AI",
]

# Semantic-only (excluding structural catch-alls) for focused plots
SEM_CATS = [c for c in CATEGORIES if c not in ("Structural/Linguistic", "General/Other")]

CAT_COLORS = [
    "#94a3b8",  # Structural/Linguistic — grey
    "#cbd5e1",  # General/Other — light grey
    "#3b82f6",  # Functions/Methods
    "#6366f1",  # Data Structures
    "#8b5cf6",  # Databases/Queries
    "#a855f7",  # Network/Protocols
    "#ec4899",  # Types/Annotations
    "#f43f5e",  # Control Flow
    "#f97316",  # Concurrency/Async
    "#eab308",  # IO/File Operations
    "#22c55e",  # Numeric/Math
    "#10b981",  # OOP/Classes
    "#06b6d4",  # String/Text Processing
    "#0ea5e9",  # Error Handling
    "#64748b",  # Version Control/DevOps
    "#f59e0b",  # Testing/QA
    "#dc2626",  # ML/AI
]
CAT_COLOR_MAP = dict(zip(CATEGORIES, CAT_COLORS))


def make_fig(figsize=(9, 4.5)):
    return plt.subplots(figsize=figsize)


def make_fig_multi(nrows, ncols, figsize=(14, 4.5), **kwargs):
    return plt.subplots(nrows, ncols, figsize=figsize, **kwargs)


def save(fig, name: str):
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.relative_to(REPO)}")


# =============================================================================
# 1. Dataset Table
# =============================================================================
def fig_dataset_table():
    source_list = load(8, "source_list.json")
    src_counts  = Counter(s.split(":")[0] for s in source_list)
    total_tokens = load(8, "summary.json")["n_tokens"]
    total_docs   = sum(src_counts.values())

    ds_display = {
        "code_search_net": "CodeSearchNet (Python)",
        "code_alpaca":     "CodeAlpaca-20k",
        "flytech_python":  "flytech/python-codes-25k",
    }
    rows = []
    for ds in DS_ORDER:
        n_samples       = src_counts.get(ds, 0)
        n_tokens_approx = round(total_tokens * (n_samples / total_docs)) if total_docs else 0
        avg_seq         = round(n_tokens_approx / n_samples) if n_samples else 0
        rows.append([f"{n_samples:,}", f"~{n_tokens_approx:,}", f"~{avg_seq}"])

    col_labels = ["Source", "Samples", "Tokens (approx.)", "Avg seq len (tokens)"]
    cell_data  = [[ds_display[ds]] + rows[i] for i, ds in enumerate(DS_ORDER)]

    fig, ax = plt.subplots(figsize=(10, 2.4))
    ax.axis("off")
    table = ax.table(
        cellText=cell_data, colLabels=col_labels,
        cellLoc="center", loc="center",
        colWidths=[0.30, 0.18, 0.28, 0.24],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#e8eaf6")
            cell.set_text_props(color="#111111", fontweight="bold")
        else:
            ds_key = DS_ORDER[r - 1]
            cell.set_facecolor("#f9f9f9" if r % 2 == 0 else "white")
            if c == 0:
                cell.set_text_props(color=DS_COLORS[ds_key], fontweight="semibold")
    ax.set_title("Code Dataset Breakdown by Source", fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    save(fig, "01_dataset_table")


# =============================================================================
# 2. Training Loss Curves — L4, L8, L12
# =============================================================================
def fig_training_loss():
    fig, ax = make_fig(figsize=(8, 4.5))
    epochs = list(range(1, 9))
    for layer in LAYERS:
        th = load(layer, "training_history.json")
        ax.plot(epochs, th["epoch_losses"], marker="o", markersize=5, linewidth=2,
                color=L_COLORS[layer], label=f"Layer {layer}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Loss Curves — Code Domain (L4 / L8 / L12)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(epochs)
    ax.legend()
    fig.tight_layout()
    save(fig, "02_training_loss")


# =============================================================================
# 3. Dead Features per Epoch
# =============================================================================
def fig_dead_features_per_epoch():
    fig, ax = make_fig(figsize=(9, 4.5))
    epochs = list(range(1, 9))
    for layer in LAYERS:
        th = load(layer, "training_history.json")
        ax.plot(epochs, th["dead_features_per_epoch"], marker="o", markersize=5,
                linewidth=2, color=L_COLORS[layer], label=f"Layer {layer}")

    # annotate L8 non-monotonic spike at epoch 2
    th8       = load(8, "training_history.json")
    spike_val = th8["dead_features_per_epoch"][1]
    ax.annotate("L8 increases\nat epoch 2", xy=(2, spike_val),
                xytext=(2.7, spike_val + 120),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1),
                color="#555555", fontsize=8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dead features")
    ax.set_title("Dead Feature Recovery per Epoch — Code Domain",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(epochs)
    ax.legend()
    fig.tight_layout()
    save(fig, "03_dead_features_per_epoch")


# =============================================================================
# 4. Shannon Entropy Violin — L4, L8, L12
# =============================================================================
def fig_entropy_violin():
    entropies = {}
    for layer in LAYERS:
        feats = load(layer, "features.json")
        entropies[layer] = [
            f["maxact_entropy"] for f in feats
            if f.get("maxact_entropy") and f.get("frequency", 0) > 1e-6
        ]

    fig, ax = make_fig(figsize=(8, 5))
    positions = [1, 2, 3]
    data      = [entropies[l] for l in LAYERS]

    parts = ax.violinplot(data, positions=positions, showmedians=True,
                          showextrema=True, widths=0.6)
    for pc, layer in zip(parts["bodies"], LAYERS):
        pc.set_facecolor(L_COLORS[layer])
        pc.set_alpha(0.45)
        pc.set_edgecolor(L_COLORS[layer])
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_color("#444444")
            parts[key].set_linewidth(1.2)
    for pos, layer in zip(positions, LAYERS):
        ax.scatter(pos, np.mean(entropies[layer]), color=L_COLORS[layer],
                   zorder=5, s=45, marker="D", edgecolors="#333333", linewidths=0.6)

    # add monosemantic / polysemantic counts as text
    for pos, layer in zip(positions, LAYERS):
        mono = sum(1 for v in entropies[layer] if v < 1.0)
        poly = sum(1 for v in entropies[layer] if v > 3.0)
        ax.text(pos, -0.15, f"mono: {mono}\npoly: {poly}",
                ha="center", va="top", fontsize=7, color="#555555",
                transform=ax.get_xaxis_transform())

    ax.set_xticks(positions)
    ax.set_xticklabels([f"Layer {l}" for l in LAYERS], fontsize=10)
    ax.set_ylabel("Shannon Entropy (bits)")
    ax.set_title("MaxAct Token Entropy Distribution — Code Domain",
                 fontsize=12, fontweight="bold")
    patches = [mpatches.Patch(color=L_COLORS[l], alpha=0.6, label=f"Layer {l}") for l in LAYERS]
    ax.legend(handles=patches)
    fig.tight_layout()
    save(fig, "04_entropy_violin")


# =============================================================================
# 5. Category Distribution — All 3 Layers Grouped Bar
# =============================================================================
def fig_category_distribution_all_layers():
    data = {}
    for layer in LAYERS:
        lf    = load(layer, "labeled_features.json")
        total = len(lf)
        cnts  = Counter(f.get("category", "General/Other") for f in lf)
        data[layer] = {cat: 100 * cnts.get(cat, 0) / total for cat in CATEGORIES}

    fig, ax = make_fig(figsize=(14, 5.5))
    x     = np.arange(len(CATEGORIES))
    width = 0.26
    offsets = [-width, 0, width]

    for offset, layer in zip(offsets, LAYERS):
        vals = [data[layer][c] for c in CATEGORIES]
        ax.bar(x + offset, vals, width, color=L_COLORS[layer], alpha=0.82,
               label=f"Layer {layer}", edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("/", "/\n") for c in CATEGORIES], fontsize=7.5)
    ax.set_ylabel("% of labeled features")
    ax.set_title("Feature Category Distribution — Code Domain (L4 / L8 / L12)",
                 fontsize=12, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save(fig, "05_category_distribution_all_layers")


# =============================================================================
# 6. Category Trajectories — Line Plot L4 → L8 → L12 (semantic cats only)
# =============================================================================
def fig_category_trajectories():
    gpt_counts = {}
    cat_pcts   = {cat: [] for cat in SEM_CATS}

    for layer in LAYERS:
        lf = load(layer, "labeled_features.json")
        gpt = [f for f in lf if f.get("label_source") == "gpt4omini"]
        gpt_counts[layer] = len(gpt) or 1
        cnts = Counter(f.get("category", "General/Other") for f in gpt)
        for cat in SEM_CATS:
            cat_pcts[cat].append(100 * cnts.get(cat, 0) / gpt_counts[layer])

    # split into growing and shrinking for cleaner layout
    deltas  = {cat: cat_pcts[cat][2] - cat_pcts[cat][0] for cat in SEM_CATS}
    growing = sorted([c for c in SEM_CATS if deltas[c] >= 0], key=lambda c: -deltas[c])
    shrink  = sorted([c for c in SEM_CATS if deltas[c] < 0],  key=lambda c: deltas[c])
    ordered = growing + shrink

    fig, ax = make_fig(figsize=(12, 5.5))
    layer_x = [4, 8, 12]
    for cat in ordered:
        color = CAT_COLOR_MAP.get(cat, "#888888")
        vals  = cat_pcts[cat]
        delta = deltas[cat]
        style = "-" if abs(delta) > 0.3 else "--"
        lw    = 2.0 if abs(delta) > 0.5 else 1.2
        ax.plot(layer_x, vals, marker="o", markersize=5, linewidth=lw,
                linestyle=style, color=color, label=cat)
        # label at right end
        ax.text(12.15, vals[2], cat.split("/")[0][:18], fontsize=7,
                va="center", color=color)

    ax.set_xticks(layer_x)
    ax.set_xticklabels(["Layer 4", "Layer 8", "Layer 12"])
    ax.set_xlabel("Layer")
    ax.set_ylabel("% of GPT-labeled features")
    ax.set_title("Semantic Category Trajectories L4 → L8 → L12 (Code Domain)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(3.5, 14.5)
    ax.grid(True, axis="y")
    fig.tight_layout()
    save(fig, "06_category_trajectories")


# =============================================================================
# 7. Source Attribution by Category — L12 stacked bar
# =============================================================================
def fig_source_attribution_by_category():
    lf = load(12, "labeled_features.json")
    cat_ds = {cat: defaultdict(float) for cat in CATEGORIES}
    for feat in lf:
        cat = feat.get("category", "General/Other")
        if cat not in cat_ds:
            cat = "General/Other"
        for ds, frac in feat.get("source_breakdown", {}).items():
            cat_ds[cat][ds] += frac

    cat_fracs = {}
    for cat in CATEGORIES:
        total = sum(cat_ds[cat].values()) or 1
        cat_fracs[cat] = {ds: cat_ds[cat].get(ds, 0) / total for ds in DS_ORDER}

    fig, ax = make_fig(figsize=(14, 5))
    x      = np.arange(len(CATEGORIES))
    bottom = np.zeros(len(CATEGORIES))
    for ds in DS_ORDER:
        vals = np.array([cat_fracs[cat][ds] for cat in CATEGORIES])
        ax.bar(x, vals, 0.6, bottom=bottom, color=DS_COLORS[ds],
               label=DS_LABELS[ds], alpha=0.85, edgecolor="white", linewidth=0.4)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("/", "/\n") for c in CATEGORIES], fontsize=7.5)
    ax.set_ylabel("Source Fraction")
    ax.set_ylim(0, 1.05)
    ax.set_title("Source Attribution per Feature Category — Layer 12 (Code Domain)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "07_source_attribution_by_category")


# =============================================================================
# 8. Frequency Spectrum — Feature Distribution Across Firing Rate Buckets
# =============================================================================
def fig_frequency_spectrum():
    buckets = [
        (0,      1e-6,  "dead\n(0%)"),
        (1e-6,   0.001, "<0.1%"),
        (0.001,  0.01,  "0.1–1%"),
        (0.01,   0.05,  "1–5%"),
        (0.05,   0.20,  "5–20%"),
        (0.20,   0.80,  "20–80%"),
        (0.80,   1.01,  ">80%"),
    ]
    labels = [b[2] for b in buckets]

    counts_per_layer = {}
    for layer in LAYERS:
        feats = load(layer, "features.json")
        cnts  = []
        for lo, hi, _ in buckets:
            cnts.append(sum(1 for f in feats if lo <= f["frequency"] < hi))
        counts_per_layer[layer] = cnts

    fig, ax = make_fig(figsize=(10, 5))
    x     = np.arange(len(buckets))
    width = 0.26
    offsets = [-width, 0, width]

    for offset, layer in zip(offsets, LAYERS):
        ax.bar(x + offset, counts_per_layer[layer], width,
               color=L_COLORS[layer], alpha=0.82, label=f"Layer {layer}",
               edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Activation Frequency Bucket")
    ax.set_ylabel("Feature count")
    ax.set_title("Feature Frequency Spectrum — Code Domain (L4 / L8 / L12)",
                 fontsize=12, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save(fig, "08_frequency_spectrum")


# =============================================================================
# 9. Monosemanticity Progression — mono / normal / poly per layer
# =============================================================================
def fig_monosemanticity_progression():
    rows = {"monosemantic\n(ent < 1)": [], "normal\n(1–3)": [], "polysemantic\n(ent > 3)": []}

    for layer in LAYERS:
        feats  = load(layer, "features.json")
        active = [f for f in feats if f.get("frequency", 0) > 1e-6 and f.get("maxact_entropy")]
        mono   = sum(1 for f in active if f["maxact_entropy"] < 1.0)
        poly   = sum(1 for f in active if f["maxact_entropy"] > 3.0)
        norm   = len(active) - mono - poly
        rows["monosemantic\n(ent < 1)"].append(mono)
        rows["normal\n(1–3)"].append(norm)
        rows["polysemantic\n(ent > 3)"].append(poly)

    fig, ax = make_fig(figsize=(8, 5))
    x      = np.arange(len(LAYERS))
    bottom = np.zeros(len(LAYERS))
    seg_colors = ["#10b981", "#94a3b8", "#ef4444"]

    for (label, vals), color in zip(rows.items(), seg_colors):
        ax.bar(x, vals, 0.5, bottom=bottom, color=color, alpha=0.82,
               label=label, edgecolor="white", linewidth=0.5)
        for xi, v, b in zip(x, vals, bottom):
            if v > 100:
                ax.text(xi, b + v / 2, str(v), ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottom += np.array(vals, dtype=float)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {l}" for l in LAYERS], fontsize=10)
    ax.set_ylabel("Feature count (active features only)")
    ax.set_title("Feature Monosemanticity Progression — Code Domain",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "09_monosemanticity_progression")


# =============================================================================
# 10. Causal Impact (KL Divergence) by Category — Heatmap-style bar chart
# =============================================================================
def fig_kl_by_category():
    fig, axes = make_fig_multi(1, 3, figsize=(16, 5.5))

    for ax, layer in zip(axes, LAYERS):
        lf       = load(layer, "labeled_features.json")
        feats    = load(layer, "features.json")
        feat_map = {f["index"]: f for f in feats}

        cat_kls = defaultdict(list)
        for lfi in lf:
            fi  = feat_map.get(lfi["index"], {})
            kl  = fi.get("token_change_kl", 0)
            cat = lfi.get("category", "General/Other")
            if cat in SEM_CATS:
                cat_kls[cat].append(kl)

        means  = {cat: np.mean(v) for cat, v in cat_kls.items() if v}
        cats_s = sorted(means.keys(), key=lambda c: -means[c])
        vals   = [means[c] for c in cats_s]
        colors = [CAT_COLOR_MAP.get(c, "#888888") for c in cats_s]

        y = np.arange(len(cats_s))
        ax.barh(y, vals, 0.6, color=colors, alpha=0.82, edgecolor="white", linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels([c.replace("/", "/\n") for c in cats_s], fontsize=7.5)
        ax.set_xlabel("Mean Token-Change KL (nats)")
        ax.set_title(f"Layer {layer}", fontsize=11, fontweight="bold")
        ax.invert_yaxis()

    fig.suptitle("Causal Impact (Token-Change KL) by Category — Code Domain",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "10_kl_by_category")


# =============================================================================
# 11. Quality Score vs Entropy Scatter — L12 colored by category
# =============================================================================
def fig_quality_vs_entropy_l12():
    lf       = load(12, "labeled_features.json")
    feats    = load(12, "features.json")
    feat_map = {f["index"]: f for f in feats}

    gpt = [f for f in lf if f.get("label_source") == "gpt4omini"
           and f.get("confidence") in ("high", "medium")]

    fig, ax = make_fig(figsize=(10, 6))

    # plot General/Other and Structural first (background)
    for cat in ["General/Other", "Structural/Linguistic"]:
        subset = [f for f in gpt if f.get("category") == cat]
        if not subset:
            continue
        qs  = [f["quality_score"] for f in subset]
        ent = [feat_map.get(f["index"], {}).get("maxact_entropy", 0) for f in subset]
        ax.scatter(ent, qs, s=12, alpha=0.18, color=CAT_COLOR_MAP[cat], label=cat, zorder=1)

    # plot semantic categories on top
    for cat in SEM_CATS:
        subset = [f for f in gpt if f.get("category") == cat]
        if not subset:
            continue
        qs  = [f["quality_score"] for f in subset]
        ent = [feat_map.get(f["index"], {}).get("maxact_entropy", 0) for f in subset]
        ax.scatter(ent, qs, s=20, alpha=0.65, color=CAT_COLOR_MAP[cat], label=cat, zorder=2)

    ax.set_xlabel("MaxAct Shannon Entropy (bits)  ← monosemantic | polysemantic →")
    ax.set_ylabel("Quality Score")
    ax.set_title("Quality Score vs. Entropy — Layer 12 (Code Domain)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, markerscale=1.5, loc="lower right",
              ncol=2, framealpha=0.9)
    fig.tight_layout()
    save(fig, "11_quality_vs_entropy_l12")


# =============================================================================
# 12. Explained Variance & Dead Features Summary
# =============================================================================
def fig_ev_and_dead_summary():
    summaries = {l: load(l, "summary.json") for l in LAYERS}

    ev_vals   = [summaries[l]["explained_variance"] * 100 for l in LAYERS]
    dead_vals = [100 * summaries[l]["dead_features"] / 8192 for l in LAYERS]

    fig, ax1 = make_fig(figsize=(7, 4.5))
    ax2 = ax1.twinx()
    ax2.set_facecolor("#f9f9f9")

    x = np.arange(len(LAYERS))
    b1 = ax1.bar(x - 0.18, ev_vals, 0.35, color=[L_COLORS[l] for l in LAYERS],
                 alpha=0.82, edgecolor="white", linewidth=0.5, label="Explained Variance")
    b2 = ax2.bar(x + 0.18, dead_vals, 0.35, color=[L_COLORS[l] for l in LAYERS],
                 alpha=0.40, edgecolor="white", linewidth=0.5, hatch="///", label="Dead Features %")

    for bar, v in zip(b1, ev_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, v in zip(b2, dead_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=9, color="#666666")

    ax1.set_ylabel("Explained Variance (%)", fontsize=10)
    ax1.set_ylim(0, 65)
    ax2.set_ylabel("Dead Features (%)", fontsize=10, color="#666666")
    ax2.tick_params(axis="y", colors="#666666")
    ax2.set_ylim(0, 45)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Layer {l}" for l in LAYERS], fontsize=10)
    ax1.set_title("Explained Variance & Dead Feature Rate — Code Domain",
                  fontsize=12, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
    fig.tight_layout()
    save(fig, "12_ev_and_dead_summary")


# =============================================================================
# 13. Top Causally Active Features — L4 / L8 / L12 horizontal bars
# =============================================================================
def fig_top_kl_features():
    fig, axes = make_fig_multi(1, 3, figsize=(18, 6))

    for ax, layer in zip(axes, LAYERS):
        feats    = load(layer, "features.json")
        lf       = load(layer, "labeled_features.json")
        lf_map   = {f["index"]: f for f in lf}

        top = sorted(
            [f for f in feats if f.get("token_change_kl", 0) > 0],
            key=lambda x: -x["token_change_kl"]
        )[:12]

        labels_text = []
        kl_vals     = []
        colors      = []
        for f in reversed(top):
            lfi   = lf_map.get(f["index"], {})
            label = lfi.get("label", "unlabeled")
            cat   = lfi.get("category", "?")
            short = (label[:35] + "…") if len(label) > 35 else label
            labels_text.append(f"[{f['index']}] {short}")
            kl_vals.append(f["token_change_kl"])
            colors.append(CAT_COLOR_MAP.get(cat, "#94a3b8"))

        y = np.arange(len(top))
        ax.barh(y, kl_vals, 0.6, color=colors, alpha=0.82,
                edgecolor="white", linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels(labels_text, fontsize=7)
        ax.set_xlabel("Token-Change KL (nats)")
        ax.set_title(f"Layer {layer}", fontsize=11, fontweight="bold")

    fig.suptitle("Top Causally Active Features per Layer — Code Domain",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "13_top_kl_features")


# =============================================================================
# 14. Source Attribution — All 3 Layers Side by Side
# =============================================================================
def fig_source_per_category_all_layers():
    fig, axes = make_fig_multi(1, 3, figsize=(20, 5.5))

    for ax, layer in zip(axes, LAYERS):
        lf     = load(layer, "labeled_features.json")
        cat_ds = {cat: defaultdict(float) for cat in CATEGORIES}
        for feat in lf:
            cat = feat.get("category", "General/Other")
            if cat not in cat_ds:
                cat = "General/Other"
            for ds, frac in feat.get("source_breakdown", {}).items():
                cat_ds[cat][ds] += frac

        x      = np.arange(len(CATEGORIES))
        bottom = np.zeros(len(CATEGORIES))
        for ds in DS_ORDER:
            vals = np.array([
                cat_ds[cat].get(ds, 0) / (sum(cat_ds[cat].values()) or 1)
                for cat in CATEGORIES
            ])
            ax.bar(x, vals, 0.65, bottom=bottom, color=DS_COLORS[ds],
                   label=DS_LABELS[ds], alpha=0.85, edgecolor="white", linewidth=0.3)
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels([c.split("/")[0][:9] for c in CATEGORIES],
                           fontsize=6.5, rotation=35, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Layer {layer}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Source fraction" if ax is axes[0] else "")
        if layer == LAYERS[-1]:
            ax.legend(loc="upper right", fontsize=7.5)

    fig.suptitle("Source Attribution per Feature Category — Code Domain (L4 / L8 / L12)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "14_source_per_category_all_layers")


# =============================================================================
# Main
# =============================================================================
FIGURES = [
    ("01 Dataset table",                         fig_dataset_table),
    ("02 Training loss curves",                  fig_training_loss),
    ("03 Dead features per epoch",               fig_dead_features_per_epoch),
    ("04 Shannon entropy violin",                fig_entropy_violin),
    ("05 Category distribution — all layers",    fig_category_distribution_all_layers),
    ("06 Category trajectories L4→L8→L12",       fig_category_trajectories),
    ("07 Source attribution by category (L12)",  fig_source_attribution_by_category),
    ("08 Frequency spectrum",                    fig_frequency_spectrum),
    ("09 Monosemanticity progression",           fig_monosemanticity_progression),
    ("10 KL divergence by category",             fig_kl_by_category),
    ("11 Quality vs entropy scatter (L12)",      fig_quality_vs_entropy_l12),
    ("12 Explained variance & dead summary",     fig_ev_and_dead_summary),
    ("13 Top causally active features",          fig_top_kl_features),
    ("14 Source attribution — all layers",       fig_source_per_category_all_layers),
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
