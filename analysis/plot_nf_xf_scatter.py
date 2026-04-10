#!/usr/bin/env python3
"""
NeuralFoil vs XFOIL L/D scatter plot for all IPOPT-validated candidates.

Each point is an IPOPT-optimised design evaluated by both NeuralFoil and XFOIL.
Colour = method; marker fill = CM threshold (solid = CM≥−0.125, open = CM≥−0.130).
Reference lines: identity (y = x) and empirical ratio line (y ≈ 0.910 x).

Output: environments/NeuralFoil/results/convergence_plots_LAM500/nf_xf_scatter.pdf/.png
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

# ── Style ─────────────────────────────────────────────────────────────────────
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

# ── Data loading ──────────────────────────────────────────────────────────────
def load_all(path):
    with open(path) as f:
        data = json.load(f)
    return [(e["label"], e["nf_ld"], e["xfoil"]["L_D"]) for e in data]

# cm125 (solid markers): CM≥−0.125 IPOPT threshold
lbfgsb_cm125 = load_all(f"{BASE}/xfoil_validation_lbfgsb_cm125_conf85/summary.json")
v3_cm125     = load_all(f"{BASE}/xfoil_validation_cm125_conf85_att15_att6/summary.json")
ga_cm125     = load_all(f"{BASE}/xfoil_validation_GA_multi_cm125_conf85/summary.json")
bo_cm125     = load_all(f"{BASE}/xfoil_validation_BO_torch_best_cm125_conf85/summary.json")

# cm130 (open markers): CM≥−0.130 IPOPT threshold
lbfgsb_cm130 = load_all(f"{BASE}/xfoil_validation_lbfgsb_ipopt_cm130/summary.json")
v3_cm130     = load_all(f"{BASE}/xfoil_validation_conf85_v3_flash2_5/summary.json")

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update(STYLE)
fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")

# Reference lines
x_ref = np.array([240, 390])
ax.plot(x_ref, x_ref,          color="#999999", lw=0.9, ls="-",  zorder=1, label="y = x")
ax.plot(x_ref, 0.910 * x_ref,  color="#999999", lw=0.9, ls="--", zorder=1,
        label=r"$y = 0.910\,x$  (empirical gap)")

# Helper: scatter one group
def scatter(pts, color, marker, filled, label, zorder=3):
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    mfc = color if filled else "none"
    ax.scatter(xs, ys, color=color, marker=marker,
               facecolors=mfc, edgecolors=color,
               s=55, lw=1.2, zorder=zorder, label=label)

# cm125 (solid)
scatter(lbfgsb_cm125, C["LBFGSB"], "o", True,  r"L-BFGS-B  CM≥$-$0.125")
scatter(bo_cm125,     C["BO"],     "D", True,  r"Bayesian Opt. (exact GP)  CM≥$-$0.125")
scatter(ga_cm125,     C["GA"],     "^", True,  r"PSO (120p$\times$500i)  CM≥$-$0.125")
scatter(v3_cm125,     C["v3"],     "s", True,  r"ShapeEvolve  CM≥$-$0.125")

# cm130 (open)
scatter(lbfgsb_cm130, C["LBFGSB"], "o", False, r"L-BFGS-B  CM≥$-$0.130")
scatter(v3_cm130,     C["v3"],     "s", False, r"ShapeEvolve  CM≥$-$0.130")

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlabel("NeuralFoil L/D  (IPOPT-refined)", fontsize=11)
ax.set_ylabel("XFOIL L/D  (validated)", fontsize=11)
ax.set_title(
    "ShapeEvolve — NeuralFoil LAM500\n"
    "NF vs XFOIL L/D: all IPOPT candidates  "
    "(solid = CM≥−0.125, open = CM≥−0.130)",
    fontweight="medium", pad=8,
)
ax.set_xlim(240, 390)
ax.set_ylim(240, 345)
ax.legend(fontsize=7.5, loc="upper left", framealpha=0.95, ncol=2)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
ax.grid(alpha=0.2)

plt.tight_layout()
for ext in (".pdf", ".png"):
    out = f"{OUT_DIR}/nf_xf_scatter{ext}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close(fig)
