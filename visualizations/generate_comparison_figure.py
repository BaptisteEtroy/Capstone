#!/usr/bin/env python3
"""
Generate side-by-side biomed vs code category distribution figure
showing the L8 → L12 shift in both domains.
Output: research paper/graphics/15_category_comparison_biomed_code.png
"""

import json
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO.parent / "research paper" / "graphics" / "15_category_comparison_biomed_code.png"

def load_labeled(domain: str, layer: int):
    path = REPO / f"{domain}_outputs" / f"layer_{layer}" / "labeled_features.json"
    with open(path) as f:
        return json.load(f)

# ── Biomed categories to highlight ──────────────────────────────────────────
BIOMED_CATS = [
    "Research Methodology",
    "Clinical/Diagnostic",
    "Structural/Linguistic",
    "Pharmacology/Treatment",
    "Biochemical/Molecular",
    "Anatomical",
    "General/Other",
]

# ── Code categories to highlight ────────────────────────────────────────────
CODE_CATS = [
    "Control Flow",
    "Functions/Methods",
    "Data Structures",
    "Network/Protocols",
    "Structural/Linguistic",
    "Types/Annotations",
    "General/Other",
]

def get_distribution(features, cats):
    total = sum(1 for f in features if f.get("label_source") != "heuristic_positional"
                and f.get("category") and f.get("category") != "")
    counts = Counter(f["category"] for f in features
                     if f.get("label_source") != "heuristic_positional"
                     and f.get("category"))
    result = {}
    for cat in cats:
        result[cat] = counts.get(cat, 0) / total * 100 if total > 0 else 0
    return result

# Load data
biomed_l8  = load_labeled("medical", 8)
biomed_l12 = load_labeled("medical", 12)
code_l8    = load_labeled("code", 8)
code_l12   = load_labeled("code", 12)

d_biomed_l8  = get_distribution(biomed_l8,  BIOMED_CATS)
d_biomed_l12 = get_distribution(biomed_l12, BIOMED_CATS)
d_code_l8    = get_distribution(code_l8,    CODE_CATS)
d_code_l12   = get_distribution(code_l12,   CODE_CATS)

# ── Plotting ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.facecolor": "#f9f9f9",
    "axes.edgecolor": "#cccccc",
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
})

fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))

def plot_panel(ax, cats, d_l8, d_l12, title, highlight_decline, highlight_grow):
    x = np.arange(len(cats))
    w = 0.35
    bars_l8  = ax.bar(x - w/2, [d_l8[c]  for c in cats], w, label="Layer 8",
                      color="#8b5cf6", alpha=0.85, zorder=3)
    bars_l12 = ax.bar(x + w/2, [d_l12[c] for c in cats], w, label="Layer 12",
                      color="#10b981", alpha=0.85, zorder=3)

    # Highlight key categories
    for i, cat in enumerate(cats):
        if cat == highlight_decline:
            bars_l8[i].set_edgecolor("#dc2626")
            bars_l8[i].set_linewidth(1.8)
        if cat == highlight_grow:
            bars_l12[i].set_edgecolor("#dc2626")
            bars_l12[i].set_linewidth(1.8)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("/", "/\n").replace(" ", "\n") for c in cats],
                       fontsize=6.5, ha="center")
    ax.set_ylabel("% of labelled features", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_ylim(0, max(max(d_l8[c], d_l12[c]) for c in cats) * 1.3 + 1)

plot_panel(axes[0], BIOMED_CATS, d_biomed_l8, d_biomed_l12,
           "Biomedical domain", "Research Methodology", "Clinical/Diagnostic")
plot_panel(axes[1], CODE_CATS, d_code_l8, d_code_l12,
           "Code domain", "Control Flow", "Functions/Methods")

# Shared annotation
fig.text(0.5, -0.02,
         "Red outlines: writing-convention category peaks at L8 (left bar) "
         "and semantic category grows at L12 (right bar).",
         ha="center", fontsize=7, color="#555555")

plt.tight_layout(pad=1.0)
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"Saved: {OUT}")
