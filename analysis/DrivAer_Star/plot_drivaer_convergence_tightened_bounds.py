"""Convergence curves (best Cd vs function evaluations) — tightened-bounds ablation.

Shows BO (n=10 seeds, 1000 evals/run) and ShapeEvolve v3 (n=10 attempts, up to 10000
evals/run) under tightened parameter bounds (car_size in [0.9,1.1],
diffusor_angle in [-4,4]).  vtk_E (Estateback) only — tight_bounds runs were not
extended to vtk_F/N.

X-axis: log scale.  Baseline Cd and full-bounds best Cd shown as reference lines.

Usage:
    cd /scratch/ShapeEvolve
    source venv/bin/activate
    python analysis/DrivAer_Star/plot_drivaer_convergence_tightened_bounds.py

Outputs:
    environments/DrivAer_Star/results/analysis_plots_cd_only_tightened_bounds/
        DrivAer_Star_convergence_cd_vs_evals_tightened_bounds_vtk_E.png
        DrivAer_Star_convergence_cd_vs_evals_tightened_bounds_vtk_E.pdf
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO_DIR, "environments", "DrivAer_Star", "results")
OUT_DIR = os.path.join(RESULTS_DIR, "analysis_plots_cd_only_tightened_bounds")

BASELINE_CD = 0.22334          # vtk_E undeformed
FULL_BOUNDS_BEST_CD = 0.06509  # BO cd_only full bounds best (all 4 methods ~0.065)

COLORS = {
    "Bayesian Opt. (exact GP)": "#ff7f0e",
    "ShapeEvolve":              "#2ca02c",
}


def _load_curves(csv_paths, best_reward_col):
    curves = []
    for csv in sorted(csv_paths):
        if not os.path.exists(csv):
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if best_reward_col not in df.columns:
            continue
        arr = -df[best_reward_col].values.astype(float)
        arr = np.minimum.accumulate(arr)
        curves.append(arr)
    return curves


def load_bo_tight():
    pattern = os.path.join(RESULTS_DIR, "run_BO_torch_cd_only_tight_bounds_vtk_E_seed*_n1000", "results.csv")
    return _load_curves(glob.glob(pattern), "best_reward")


def load_v3_tight():
    pattern = os.path.join(
        RESULTS_DIR,
        "run_v3_dynamic_optimizer_cd_only_tight_bounds_drivaer_star_vtk_E_attempt_*_flash_2_5_n10000",
        "results.csv",
    )
    curves = _load_curves(glob.glob(pattern), "best_reward")
    if curves:
        max_len = max(len(c) for c in curves)
        curves = [c for c in curves if len(c) >= max_len * 0.5]
    return curves


def interp_to_grid(curve, x_out, extend=True):
    n = len(curve)
    idx = np.searchsorted(np.arange(1, n + 1), x_out, side="right") - 1
    idx = np.clip(idx, 0, n - 1)
    out = curve[idx].copy()
    if not extend:
        out[x_out > n] = np.nan
    out[x_out < 1] = np.nan
    return out


def compute_band(curves, x_grid, extend=True):
    mat = np.vstack([interp_to_grid(c, x_grid, extend=extend) for c in curves])
    n_valid = np.sum(~np.isnan(mat), axis=0)
    mask = n_valid >= max(1, len(curves) // 2)
    med = np.where(mask, np.nanpercentile(mat, 50, axis=0), np.nan)
    p25 = np.where(mask, np.nanpercentile(mat, 25, axis=0), np.nan)
    p75 = np.where(mask, np.nanpercentile(mat, 75, axis=0), np.nan)
    lo  = np.where(mask, np.nanmin(mat,             axis=0), np.nan)
    hi  = np.where(mask, np.nanmax(mat,             axis=0), np.nan)
    return med, p25, p75, lo, hi


def plot_band(ax, curves, x_grid, color, label, extend=True):
    if not curves:
        return
    med, p25, p75, lo, hi = compute_band(curves, x_grid, extend=extend)
    ax.fill_between(x_grid, lo, hi, color=color, alpha=0.12)
    ax.fill_between(x_grid, p25, p75, color=color, alpha=0.28)
    ax.plot(x_grid, med, color=color, lw=2.0, label=label)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    bo_curves = load_bo_tight()
    v3_curves = load_v3_tight()

    if bo_curves:
        print(f"BO tight_bounds:  {len(bo_curves)} runs,  max evals = {max(len(c) for c in bo_curves)}")
    if v3_curves:
        print(f"SE tight_bounds:  {len(v3_curves)} runs,  max evals = {max(len(c) for c in v3_curves)}")

    all_curves = bo_curves + v3_curves
    if not all_curves:
        print("No data found — exiting.")
        return

    x_max = max(len(c) for c in all_curves)
    x_grid = np.unique(
        np.concatenate([np.geomspace(1, x_max, 2000).astype(int), [x_max]])
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    # Reference lines
    ax.axhline(BASELINE_CD, color="#999999", lw=1.2, ls="--", zorder=1)
    ax.text(1.3, BASELINE_CD + 0.004,
            f"Baseline  $C_D$ = {BASELINE_CD:.5f}  (vtk_E undeformed)",
            color="#666666", fontsize=8, va="bottom")

    ax.axhline(FULL_BOUNDS_BEST_CD, color="#cccccc", lw=1.0, ls=":", zorder=1)
    ax.text(1.3, FULL_BOUNDS_BEST_CD - 0.007,
            f"Full-bounds best  $C_D$ = {FULL_BOUNDS_BEST_CD:.5f}  (surrogate exploit)",
            color="#aaaaaa", fontsize=8, va="top")

    # BO — complete, extend flat
    if bo_curves:
        x_max_bo = max(len(c) for c in bo_curves)
        x_grid_bo = x_grid[x_grid <= x_max_bo + 1]
        plot_band(ax, bo_curves, x_grid_bo,
                  COLORS["Bayesian Opt. (exact GP)"], "Bayesian Opt. (exact GP)", extend=True)

    # SE — may be in-progress, do NOT extend past actual run length
    if v3_curves:
        x_max_v3 = max(len(c) for c in v3_curves)
        x_grid_v3 = x_grid[x_grid <= x_max_v3 + 1]
        plot_band(ax, v3_curves, x_grid_v3,
                  COLORS["ShapeEvolve"], "ShapeEvolve", extend=False)

    ax.set_xscale("log")
    ax.set_xlabel("Function evaluations (per run)", fontsize=11)
    ax.set_ylabel(r"$C_D$", fontsize=11)
    ax.set_title(
        "DrivAer Star (Estate, vtk_E) — Tightened-Bounds Ablation: best $C_D$ vs evaluations\n"
        r"car\_size $\in [0.9, 1.1]$,  diffusor\_angle $\in [-4^\circ, 4^\circ]$"
        "  |  solid = median,  band = min–max",
        fontsize=10,
    )
    ax.set_xlim(1, x_max)
    ax.set_ylim(0.08, BASELINE_CD * 1.08)
    ax.grid(True, which="both", alpha=0.25)

    method_leg = ax.legend(fontsize=9, loc="upper right", framealpha=0.95, title="Method")
    ax.add_artist(method_leg)
    style_handles = [
        Line2D([0], [0], color="grey", lw=2.0, label="Median best $C_D$"),
        Patch(facecolor="grey", alpha=0.18, label="Min–max range"),
        Patch(facecolor="grey", alpha=0.40, label="25th–75th percentile"),
    ]
    ax.legend(handles=style_handles, loc="lower left", fontsize=9,
              framealpha=0.95, title="Style")

    stem = "DrivAer_Star_convergence_cd_vs_evals_tightened_bounds_vtk_E"
    fig.savefig(os.path.join(OUT_DIR, f"{stem}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUT_DIR}/{stem}.png")
    print(f"Saved: {OUT_DIR}/{stem}.pdf")

    print("\nMethod                     n_runs  best_Cd   median_final  mean_final")
    print("-" * 70)
    for name, curves in [("Bayesian Opt. (exact GP)", bo_curves), ("ShapeEvolve", v3_curves)]:
        if not curves:
            continue
        finals = [c[-1] for c in curves]
        print(f"{name:<26}  {len(curves):>5}  {min(finals):.5f}    "
              f"{np.median(finals):.5f}     {np.mean(finals):.5f}")


if __name__ == "__main__":
    main()
