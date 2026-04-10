#!/usr/bin/env python3
"""
Pipeline stage comparison: Raw NF L/D → IPOPT NF L/D → XFOIL-validated L/D.

Shows the value added at each stage of the two-stage optimization pipeline
for each method's best design.  Displayed as a slope/parallel-coordinates
chart: each method is a connected line across three stages.

Raw NF L/D values: best near-feasible reward from results.csv (pre-IPOPT).
IPOPT NF L/D + XF L/D: loaded from XFOIL validation summary.json files.

Output: environments/NeuralFoil/results/convergence_plots_LAM500/pipeline_stages.pdf/.png
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = "environments/NeuralFoil/results"
OUT_DIR = f"{BASE}/convergence_plots_LAM500"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style (matches other LAM500 plots) ───────────────────────────────────────
C = {"GA": "#e07b39", "LBFGSB": "#2176ae", "v3": "#57a773", "BO": "#8e4ec6"}
STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 200,
}

# ── Data ──────────────────────────────────────────────────────────────────────
# Raw NF L/D: best near-feasible value from results.csv (total_violation ≈ 0).
# Source: max(fitness_total) over rows where total_violation < 0.001 for the
# best-performing seed/attempt of each method.
RAW_NF = {
    "GA":     354.14,   # att13, gbest_reward plateau
    "LBFGSB": 324.60,   # s25, best near-feasible L/D (limited by conf≥0.90)
    "v3":     343.69,   # att15, best near-feasible L/D
    "BO":     263.82,   # s1, best near-feasible L/D (GP stagnated early)
}

# IPOPT NF L/D + XF L/D: from cm125/conf85 XFOIL validation summary.json files.
def load_entry(path, label):
    with open(path) as f:
        data = json.load(f)
    for e in data:
        if e["label"] == label:
            return e["nf_ld"], e["xfoil"]["L_D"]
    raise KeyError(f"{label} not found in {path}")

IPOPT_NF = {}
XF_LD    = {}

IPOPT_NF["GA"], XF_LD["GA"] = load_entry(
    f"{BASE}/xfoil_validation_GA_multi_cm125_conf85/summary.json", "GA_att13_raw")
IPOPT_NF["LBFGSB"], XF_LD["LBFGSB"] = load_entry(
    f"{BASE}/xfoil_validation_lbfgsb_cm125_conf85/summary.json", "lbfgsb_s25")
IPOPT_NF["v3"], XF_LD["v3"] = load_entry(
    f"{BASE}/xfoil_validation_cm125_conf85_att15_att6/summary.json", "att15")
IPOPT_NF["BO"], XF_LD["BO"] = load_entry(
    f"{BASE}/xfoil_validation_BO_torch_best_cm125_conf85/summary.json", "BO_torch_seed1")

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update(STYLE)
fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")

STAGES   = [0, 1, 2]
LABELS   = ["Stage 1\n(Raw optimizer\nbest NF L/D)",
            "Stage 2\n(IPOPT-refined\nNF L/D)",
            "Stage 3\n(XFOIL-validated\nL/D)"]
METHODS  = [
    ("LBFGSB", "L-BFGS-B s25"),
    ("BO",     "Bayesian Opt. (exact GP) s1"),
    ("GA",     r"PSO (120p$\times$500i) att13"),
    ("v3",     "ShapeEvolve att15"),
]

for key, label in METHODS:
    ys = [RAW_NF[key], IPOPT_NF[key], XF_LD[key]]
    ax.plot(STAGES, ys, "o-", color=C[key], lw=2.0, ms=7,
            markerfacecolor=C[key], markeredgecolor="white",
            markeredgewidth=0.8, label=label, zorder=3)
    # Annotate final XF value on the right
    ax.text(2.06, XF_LD[key], f"{XF_LD[key]:.1f}",
            color=C[key], fontsize=8, va="center")

# Stage dividers
for x in [0.5, 1.5]:
    ax.axvline(x, color="#cccccc", lw=0.8, ls="--", zorder=1)

ax.set_xticks(STAGES)
ax.set_xticklabels(LABELS, fontsize=9)
ax.set_ylabel("L/D", fontsize=11)
ax.set_xlim(-0.3, 2.5)
ax.set_ylim(220, 385)
ax.set_title(
    "ShapeEvolve — NeuralFoil LAM500\nTwo-stage pipeline: best design per method",
    fontweight="medium", pad=8,
)
ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
for ext in (".pdf", ".png"):
    out = f"{OUT_DIR}/pipeline_stages{ext}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close(fig)
