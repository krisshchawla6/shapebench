"""Convergence curves (best Cd vs function evaluations) for all DrivAer benchmark methods.

For each method: median best-Cd across runs (solid line) with min/max envelope
(shaded band). v3 shows only n10000 runs (in-progress snapshot); other methods
show all completed runs.

X-axis: log scale. Baseline Cd = 0.22334 shown as a reference dashed line.

Usage:
    cd /scratch/ShapeEvolve
    source venv/bin/activate
    python analysis/DrivAer_Star/plot_drivaer_convergence.py

Outputs:
    environments/DrivAer_Star/results/analysis_plots_cd_only/convergence_cd_vs_evals.png
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
OUT_DIR = os.path.join(RESULTS_DIR, "analysis_plots_cd_only")

BASELINE_CD = 0.22334   # Transolver on undeformed mesh (car_size=1, all params=0)

COLORS = {
    "GA/PSO":       "#e07b39",
    "L-BFGS-B":     "#7b9e87",
    "BO_torch":     "#4a90d9",
    "v3 n10000":    "#9b59b6",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_curves(csv_paths, best_reward_col):
    """Return list of 1-D best-Cd arrays (one per run), 1-indexed by eval."""
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
        # Ensure monotone non-increasing (forward-fill running min)
        arr = np.minimum.accumulate(arr)
        curves.append(arr)
    return curves


def load_ga():
    base = os.path.join(RESULTS_DIR, "SAVED_DIRS_run_GA_cd_only_drivaer_star_120p_250i")
    return _load_curves(
        glob.glob(os.path.join(base, "run_GA_cd_only_*", "results.csv")),
        "gbest_reward",
    )


def load_lbfgsb():
    base = os.path.join(RESULTS_DIR, "SAVED_DIRS_run_lbfgsb_cd_only_drivaer_star_nr3")
    return _load_curves(
        glob.glob(os.path.join(base, "run_lbfgsb_cd_only_*", "results.csv")),
        "best_reward",
    )


def load_bo():
    return _load_curves(
        glob.glob(os.path.join(RESULTS_DIR, "run_BO_torch_cd_only_seed*_n1000", "results.csv")),
        "best_reward",
    )


def load_v3_n10000():
    """n10000 runs only — may be in-progress (do NOT extend past actual length)."""
    return _load_curves(
        glob.glob(os.path.join(RESULTS_DIR, "run_v3_*_n10000", "results.csv")),
        "best_reward",
    )


# ── Interpolation onto common log-spaced grid ─────────────────────────────────

def interp_to_grid(curve, x_out, extend=True):
    """Forward-fill curve (1-indexed) onto x_out grid.
    extend=True  → hold last value beyond end of run.
    extend=False → NaN beyond end (run still in progress / cancelled).
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
    """Median, min, max across runs on x_grid. Suppress where <50% runs have data."""
    mat = np.vstack([interp_to_grid(c, x_grid, extend=extend) for c in curves])
    n_valid = np.sum(~np.isnan(mat), axis=0)
    mask = n_valid >= max(1, len(curves) // 2)
    med = np.where(mask, np.nanpercentile(mat, 50, axis=0), np.nan)
    lo  = np.where(mask, np.nanmin(mat,             axis=0), np.nan)
    hi  = np.where(mask, np.nanmax(mat,             axis=0), np.nan)
    return med, lo, hi


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_band(ax, curves, x_grid, color, label, extend=True):
    if not curves:
        return
    med, lo, hi = compute_band(curves, x_grid, extend=extend)
    ax.fill_between(x_grid, lo, hi, color=color, alpha=0.18)
    ax.plot(x_grid, med, color=color, lw=2.0, label=label)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ga_curves  = load_ga()
    lb_curves  = load_lbfgsb()
    bo_curves  = load_bo()
    v3_curves  = load_v3_n10000()

    print(f"GA/PSO:    {len(ga_curves)} runs,  max evals = {max(len(c) for c in ga_curves)}")
    print(f"L-BFGS-B:  {len(lb_curves)} runs,  max evals = {max(len(c) for c in lb_curves)}")
    print(f"BO_torch:  {len(bo_curves)} runs,  max evals = {max(len(c) for c in bo_curves)}")
    print(f"v3 n10000: {len(v3_curves)} runs,  max evals = {max(len(c) for c in v3_curves)}")

    x_max = max(
        max(len(c) for c in ga_curves),
        max(len(c) for c in lb_curves),
    )
    x_grid = np.unique(
        np.concatenate([np.geomspace(1, x_max, 2000).astype(int), [x_max]])
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    # Baseline reference
    ax.axhline(BASELINE_CD, color="#999999", lw=1.2, ls="--", zorder=1)
    ax.text(1.3, BASELINE_CD + 0.004, f"Baseline  Cd = {BASELINE_CD:.3f}",
            color="#666666", fontsize=8.5, va="bottom")

    # v3 n10000 — in-progress, do NOT extend past actual run length
    x_max_v3 = max(len(c) for c in v3_curves)
    x_grid_v3 = x_grid[x_grid <= x_max_v3 + 1]
    plot_band(ax, v3_curves, x_grid_v3, COLORS["v3 n10000"],
              f"v3 flash-2.5 n10000  (n={len(v3_curves)}, in progress)", extend=False)

    # BO_torch — complete at n=1000; clip grid to avoid flat extrapolation clutter
    x_max_bo = max(len(c) for c in bo_curves)
    x_grid_bo = x_grid[x_grid <= x_max_bo + 1]
    plot_band(ax, bo_curves, x_grid_bo, COLORS["BO_torch"],
              f"BO_torch  (n={len(bo_curves)} seeds)", extend=True)

    # L-BFGS-B — complete, extend flat
    plot_band(ax, lb_curves, x_grid, COLORS["L-BFGS-B"],
              f"L-BFGS-B  (n={len(lb_curves)} seeds)", extend=True)

    # GA/PSO — complete, extend flat
    plot_band(ax, ga_curves, x_grid, COLORS["GA/PSO"],
              f"GA/PSO  (n={len(ga_curves)} runs)", extend=True)

    # Axes
    ax.set_xscale("log")
    ax.set_xlabel("Function evaluations (per run)", fontsize=11)
    ax.set_ylabel("Best Cd (drag coefficient)", fontsize=11)
    ax.set_title(
        "DrivAer Star — Convergence: best Cd vs function evaluations\n"
        "(solid = median across runs,  band = min–max range)",
        fontsize=11,
    )
    ax.set_xlim(1, x_max)
    ax.set_ylim(0.05, BASELINE_CD * 1.08)
    ax.grid(True, which="both", alpha=0.25)

    # Two-part legend: method + style key
    method_leg = ax.legend(fontsize=9, loc="upper right", framealpha=0.95, title="Method")
    ax.add_artist(method_leg)
    style_handles = [
        Line2D([0], [0], color="grey", lw=2.0, label="Median best Cd"),
        Patch(facecolor="grey", alpha=0.25, label="Min–max range"),
    ]
    ax.legend(handles=style_handles, loc="lower left", fontsize=9,
              framealpha=0.95, title="Style")

    out_png = os.path.join(OUT_DIR, "convergence_cd_vs_evals.png")
    out_pdf = os.path.join(OUT_DIR, "convergence_cd_vs_evals.pdf")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")

    # Summary table
    print("\nMethod            n_runs  best_Cd   median_final  mean_final")
    print("-" * 62)
    for name, curves in [
        ("GA/PSO",      ga_curves),
        ("L-BFGS-B",   lb_curves),
        ("BO_torch",   bo_curves),
        ("v3 n10000",  v3_curves),
    ]:
        finals = [c[-1] for c in curves]
        print(f"{name:<18} {len(curves):>5}  {min(finals):.5f}    "
              f"{np.median(finals):.5f}     {np.mean(finals):.5f}")


if __name__ == "__main__":
    main()
