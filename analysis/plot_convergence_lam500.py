#!/usr/bin/env python3
"""
Convergence plot for NeuralFoil LAM500 benchmark.

Metric: best penalized reward seen so far vs. cumulative NF evaluations per run.
  reward = NF L/D − 500 × normalized_violation
  At convergence, violation ≈ 0 so reward ≈ NF L/D for the best near-feasible design.

The raw optimizer plateaus (GA~354, v3~344, LBFGSB~325, BO~264) differ because each
method hits the conf≥0.90 constraint boundary differently. All methods are then
post-processed with IPOPT (conf≥0.85), which uniformly brings designs to NF L/D~356-362.
The dashed lines show the final XFOIL-validated L/D after that post-processing.

Output: convergence_plots_LAM500/convergence_best_reward.pdf/.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── Paths (relative to repo root /scratch/ShapeEvolve) ───────────────────────
BASE_LAM = (
    "environments/NeuralFoil/results/"
    "SAVED_DIRS_reward_ld_ratio_constrained_m02_re1e7_normalized_LAM500"
)
BASE_RES = "environments/NeuralFoil/results"
OUT_DIR = f"{BASE_RES}/convergence_plots_LAM500"
os.makedirs(OUT_DIR, exist_ok=True)

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 150,
}

# Colors match BlendedNet convergence plot
COLORS = {
    "L-BFGS-B":                "#e377c2",
    "Bayesian Opt. (exact GP)": "#ff7f0e",
    "PSO":                     "#1f77b4",
    "ShapeEvolve":             "#2ca02c",
}

# ── Helper ────────────────────────────────────────────────────────────────────
def load_curve(f: str) -> np.ndarray:
    """
    Return the precomputed running-max reward curve from results.csv.
    GA uses 'gbest_reward'; all others use 'best_reward'.
    """
    df = pd.read_csv(f)
    col = "gbest_reward" if "gbest_reward" in df.columns and "best_reward" not in df.columns else "best_reward"
    return df[col].values.astype(float)


def interp_to_grid(curve: np.ndarray, x_out: np.ndarray,
                   extend: bool = True) -> np.ndarray:
    """
    Forward-fill curve (indexed 1..len) onto x_out.
    extend=True: hold last value beyond end of run.
    extend=False: NaN beyond end (run ended/cancelled).
    """
    n = len(curve)
    idx = np.searchsorted(np.arange(1, n + 1), x_out, side="right") - 1
    idx = np.clip(idx, 0, n - 1)
    out = curve[idx].copy()
    if not extend:
        out[x_out > n] = np.nan
    out[x_out < 1] = np.nan
    return out


def compute_band(curves, x_grid, extend=True):
    """Median, min, max across runs, interpolated onto x_grid."""
    mat = np.vstack([interp_to_grid(c, x_grid, extend=extend) for c in curves])
    n_valid = np.sum(~np.isnan(mat), axis=0)
    mask = n_valid >= max(1, len(curves) // 2)
    med = np.where(mask, np.nanpercentile(mat, 50, axis=0), np.nan)
    lo  = np.where(mask, np.nanmin(mat, axis=0),             np.nan)
    hi  = np.where(mask, np.nanmax(mat, axis=0),             np.nan)
    return med, lo, hi


def plot_band(ax, curves, x_grid, color, label, extend=True, ls="-"):
    if not curves:
        return
    med, lo, hi = compute_band(curves, x_grid, extend=extend)
    ax.fill_between(x_grid, lo, hi, color=color, alpha=0.18)
    ax.plot(x_grid, med, color=color, lw=1.8, label=label, ls=ls)


# ── Load GA ───────────────────────────────────────────────────────────────────
ga_curves = []
for a in range(1, 26):
    f = (f"{BASE_LAM}/run_GA_ld_ratio_constrained_m02_re1e7_normalized_"
         f"120particles_500iterations_attempt_{a}_AWS/results.csv")
    if os.path.exists(f):
        ga_curves.append(load_curve(f))
print(f"GA: {len(ga_curves)} attempts loaded, max evals = {max(len(c) for c in ga_curves)}")

# ── Load L-BFGS-B ─────────────────────────────────────────────────────────────
lbfgsb_curves = []
for s in range(40):
    f = (f"{BASE_LAM}/run_lbfgsb_ld_ratio_constrained_m02_re1e7_normalized_"
         f"seed{s}_nr3/results.csv")
    if os.path.exists(f):
        lbfgsb_curves.append(load_curve(f))
print(f"L-BFGS-B: {len(lbfgsb_curves)} seeds, max evals = {max(len(c) for c in lbfgsb_curves)}")

# ── Load BO_torch ─────────────────────────────────────────────────────────────
bo_curves = []
BO_SAVED = f"{BASE_RES}/SAVED_DIRS_run_BO_torch_ld_ratio_constrained_m02_re1e7_normalized"
for s in [0, 1, 2, 3]:  # seeds 0-3 (n5000, cancelled early at ~2600-2850 evals)
    f = (f"{BO_SAVED}/run_BO_torch_ld_ratio_constrained_m02_re1e7_normalized_"
         f"seed{s}_n5000/results.csv")
    if os.path.exists(f):
        bo_curves.append(load_curve(f))
for s in [5, 6, 7, 8, 9]:  # seeds 5-9 (n6000)
    run_dir = (f"{BO_SAVED}/run_BO_torch_ld_ratio_constrained_m02_re1e7_normalized_"
               f"seed{s}_n6000")
    f_recovered = f"{run_dir}/results_recovered.csv"
    f = f_recovered if os.path.exists(f_recovered) else f"{run_dir}/results.csv"
    if os.path.exists(f):
        bo_curves.append(load_curve(f))
print(f"BO_torch: {len(bo_curves)} seeds, max evals = {max(len(c) for c in bo_curves)}")

# ── Load v3 LLM ──────────────────────────────────────────────────────────────
v3_curves = []
V3_SAVED = f"{BASE_RES}/SAVED_DIRS_run_v3_dynamic_optimizer_ld_ratio_constrained_m02_re1e7_normalized"
for a in range(1, 25):
    f = (f"{V3_SAVED}/run_v3_dynamic_optimizer_ld_ratio_constrained_m02_re1e7_normalized_"
         f"attempt_{a}_flash_2_5/results.csv")
    if os.path.exists(f):
        v3_curves.append(load_curve(f))
print(f"v3 LLM: {len(v3_curves)} attempts, max evals = {max(len(c) for c in v3_curves)}")

# ── Common x grid ─────────────────────────────────────────────────────────────
all_curves_flat = lbfgsb_curves + bo_curves + ga_curves + v3_curves
x_max = max(len(c) for c in all_curves_flat)
x_grid = np.unique(np.concatenate([
    np.geomspace(1, x_max, 1200).astype(int), [x_max]
]))


def xg(curves):
    # Each method's x-grid is clipped to its own max run length (extend=True within
    # that range). Complete runs that ended earlier extend flat to the method's x_max
    # (honest — they converged and the best-found value is still valid), but no flat
    # tail bleeds into other methods' x territory.
    return x_grid[x_grid <= max(len(c) for c in curves)]


# ── XFOIL-validated final L/D (dashed horizontals) ───────────────────────────
XFOIL_BEST = {"LBFGSB": 329.5, "BO": 324.9, "GA": 325.6, "v3": 329.0}

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update(STYLE)
fig, ax = plt.subplots(figsize=(8.5, 5), facecolor="white")

if lbfgsb_curves:
    plot_band(ax, lbfgsb_curves, xg(lbfgsb_curves), COLORS["L-BFGS-B"], "L-BFGS-B")

if bo_curves:
    plot_band(ax, bo_curves, xg(bo_curves), COLORS["Bayesian Opt. (exact GP)"],
              "Bayesian Opt. (exact GP)")

if ga_curves:
    plot_band(ax, ga_curves, xg(ga_curves), COLORS["PSO"],
              r"PSO (120p $\times$ 500i)")

if v3_curves:
    plot_band(ax, v3_curves, xg(v3_curves), COLORS["ShapeEvolve"], "ShapeEvolve")

# XFOIL-validated best: dashed horizontals
C_xf = {
    "LBFGSB": COLORS["L-BFGS-B"],
    "BO":     COLORS["Bayesian Opt. (exact GP)"],
    "GA":     COLORS["PSO"],
    "v3":     COLORS["ShapeEvolve"],
}
for key, xf in XFOIL_BEST.items():
    ax.axhline(xf, color=C_xf[key], lw=1.0, ls="--", alpha=0.6)

# Label XFOIL lines — stagger x to avoid overlap (values: BO≈324.9, GA≈325.6, v3≈329.0, LBFGSB≈329.5)
XF_LABEL_X = {"BO": 1.3, "GA": 8, "v3": 25, "LBFGSB": 100}
for key, xf in XFOIL_BEST.items():
    ax.text(XF_LABEL_X[key], xf + 0.3, f"XF {xf:.1f}", color=C_xf[key],
            fontsize=6.5, va="bottom", ha="left")

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xscale("log")
ax.set_xlabel("NeuralFoil evaluations (per run)")
ax.set_ylabel(r"$C_L/C_D - 500\!\sum_k v_k$")
ax.set_title(
    r"Constrained Laminar Airfoil $C_L/C_D$ Maximisation (Ma=0.2, Re=10$^7$)" + "\n"
    "Stage 1 convergence: best penalized reward per run   "
    r"($\lambda=500$, dashed = XFOIL-validated best after IPOPT post-processing)",
)
ax.set_xlim(1, x_max)
ax.set_ylim(0, 368)
ax.grid(True, which="both", alpha=0.25)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)

style_legend = [
    Patch(facecolor="grey", alpha=0.25, label="Min–max range"),
    Line2D([0], [0], color="grey", lw=1.8, label="Median best"),
]
leg1 = ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95, title="Method")
ax.add_artist(leg1)
ax.legend(handles=style_legend, loc="lower left", fontsize=8.5,
          framealpha=0.95, title="Style key")

plt.tight_layout()

out_pdf = f"{OUT_DIR}/NeuralFoil_LD_convergence_best_reward.pdf"
out_png = f"{OUT_DIR}/NeuralFoil_LD_convergence_best_reward.png"
plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved:\n  {out_pdf}\n  {out_png}")
