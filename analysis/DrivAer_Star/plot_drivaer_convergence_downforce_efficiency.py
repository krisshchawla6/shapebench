"""Convergence curves (best reward vs. evaluations) for DrivAer Star downforce_efficiency.

Reward: reward = -Cl / max(Cd, 1e-4)   (maximized)
Convergence curves are monotone non-decreasing (running max).
Y-axis shows best_reward directly (higher = better).

Methods: Bayesian Opt. (exact GP, 10 seeds) and ShapeEvolve (v3, 10 attempts).
Stalled runs shorter than 50% of the longest are excluded.

Usage:
    cd /scratch/ShapeEvolve
    source venv/bin/activate
    python analysis/DrivAer_Star/plot_drivaer_convergence_downforce_efficiency.py [--body {E,F,N}]

Outputs:
    environments/DrivAer_Star/results/analysis_plots_downforce_efficiency/
        DrivAer_Star_convergence_reward_vs_evals_vtk_{body}_downforce_efficiency.png/.pdf
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO_DIR, "environments", "DrivAer_Star", "results")
OUT_DIR     = os.path.join(RESULTS_DIR, "analysis_plots_downforce_efficiency")

# Downforce efficiency of undeformed baseline (from iter_0_s0 results)
BASELINE_REWARD_BY_BODY = {
    "E": -0.332,
    "F": -0.393,
    "N": -0.028,
}

COLORS = {
    "Bayesian Opt. (exact GP)": "#ff7f0e",
    "ShapeEvolve":              "#2ca02c",
}

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


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_curves(csv_paths, min_len_fraction=0.5):
    """Return list of 1-D best-reward arrays (one per run).

    Y-value = best_reward (higher = better); monotone non-decreasing.
    Runs shorter than min_len_fraction * max_len are excluded (stalled/cancelled).
    """
    curves = []
    for csv in sorted(csv_paths):
        if not os.path.exists(csv):
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if "best_reward" not in df.columns:
            continue
        arr = df["best_reward"].values.astype(float)
        arr = np.maximum.accumulate(arr)
        curves.append(arr)

    if curves and min_len_fraction > 0:
        max_len = max(len(c) for c in curves)
        curves  = [c for c in curves if len(c) >= max_len * min_len_fraction]

    return curves


def load_v3(body):
    csvs = glob.glob(os.path.join(
        RESULTS_DIR,
        f"run_v3_dynamic_optimizer_downforce_efficiency_drivaer_star_vtk_{body}"
        f"_attempt_*_flash_2_5_n6000",
        "results.csv",
    ))
    return _load_curves(csvs)


def load_bo(body):
    csvs = glob.glob(os.path.join(
        RESULTS_DIR,
        f"run_BO_torch_downforce_efficiency_vtk_{body}_seed*_n1000",
        "results.csv",
    ))
    # BO runs are shorter (n=1000) — don't filter by length
    return _load_curves(csvs, min_len_fraction=0.0)


# ── Interpolation ─────────────────────────────────────────────────────────────

def interp_to_grid(curve, x_out, extend=True):
    n   = len(curve)
    idx = np.searchsorted(np.arange(1, n + 1), x_out, side="right") - 1
    idx = np.clip(idx, 0, n - 1)
    out = curve[idx].copy()
    if not extend:
        out[x_out > n] = np.nan
    out[x_out < 1] = np.nan
    return out


def compute_band(curves, x_grid, extend=True):
    mat     = np.vstack([interp_to_grid(c, x_grid, extend=extend) for c in curves])
    n_valid = np.sum(~np.isnan(mat), axis=0)
    mask    = n_valid >= max(1, len(curves) // 2)
    med = np.where(mask, np.nanpercentile(mat, 50, axis=0), np.nan)
    lo  = np.where(mask, np.nanmin(mat,             axis=0), np.nan)
    hi  = np.where(mask, np.nanmax(mat,             axis=0), np.nan)
    return med, lo, hi


def plot_band(ax, curves, x_grid, color, label, extend=True):
    if not curves:
        return
    med, lo, hi = compute_band(curves, x_grid, extend=extend)
    ax.fill_between(x_grid, lo, hi, color=color, alpha=0.18)
    ax.plot(x_grid, med, color=color, lw=2.0, label=label)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--body", choices=["E", "F", "N"], default="E")
    args = parser.parse_args()
    body = args.body

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.rcParams.update(STYLE)

    body_names   = {"E": "Estate", "F": "Fastback", "N": "Notchback"}
    body_label   = body_names[body]
    BASELINE_REW = BASELINE_REWARD_BY_BODY[body]

    bo_curves = load_bo(body)
    v3_curves = load_v3(body)

    for name, curves in [("Bayesian Opt. (exact GP)", bo_curves),
                          ("ShapeEvolve",              v3_curves)]:
        if curves:
            print(f"{name}: {len(curves)} runs, "
                  f"max evals = {max(len(c) for c in curves)}, "
                  f"best final = {max(c[-1] for c in curves):.5f}")

    all_curves = bo_curves + v3_curves
    if not all_curves:
        print("No data found — exiting.")
        return

    x_max  = max(len(c) for c in all_curves)
    x_grid = np.unique(
        np.concatenate([np.geomspace(1, x_max, 2000).astype(int), [x_max]])
    )

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")

    # Baseline reference
    ax.axhline(BASELINE_REW, color="#999999", lw=1.2, ls="--", zorder=1)
    ax.text(1.3, BASELINE_REW + (0.04 if BASELINE_REW > -1 else 0.2),
            f"Baseline  reward = {BASELINE_REW:.3f}",
            color="#666666", fontsize=8.5, va="bottom")

    # BO — extend flat to x_max_bo
    if bo_curves:
        x_max_bo  = max(len(c) for c in bo_curves)
        x_grid_bo = x_grid[x_grid <= x_max_bo + 1]
        plot_band(ax, bo_curves, x_grid_bo,
                  COLORS["Bayesian Opt. (exact GP)"],
                  "Bayesian Opt. (exact GP)", extend=True)

    # ShapeEvolve — do NOT extend past actual run length
    if v3_curves:
        x_max_v3  = max(len(c) for c in v3_curves)
        x_grid_v3 = x_grid[x_grid <= x_max_v3 + 1]
        plot_band(ax, v3_curves, x_grid_v3,
                  COLORS["ShapeEvolve"], "ShapeEvolve", extend=False)

    ax.set_xscale("log")
    ax.set_xlabel("Function evaluations (per run)", fontsize=11)
    ax.set_ylabel(r"Best reward  ($-C_l\,/\,C_D$, downforce efficiency)", fontsize=11)
    ax.set_title(
        f"DrivAer$^\\star$ ({body_label}, vtk_{body}) — "
        r"Downforce efficiency ($-C_l\,/\,C_D$) optimisation" + "\n"
        "Convergence: best reward per run  "
        "(solid = median,  band = min–max)",
        fontsize=11,
    )
    ax.set_xlim(1, x_max)

    # Y-axis: from slightly below baseline to slightly above best observed
    all_finals = [c[-1] for c in all_curves]
    y_lo = min(BASELINE_REW, min(all_finals)) * 1.1 if min(BASELINE_REW, min(all_finals)) < 0 else \
           min(BASELINE_REW, min(all_finals)) * 0.9
    y_hi = max(all_finals) * 1.08
    ax.set_ylim(y_lo, y_hi)

    ax.grid(True, which="both", alpha=0.25)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    method_leg = ax.legend(fontsize=9, loc="upper left",
                           framealpha=0.95, title="Method")
    ax.add_artist(method_leg)
    style_handles = [
        Line2D([0], [0], color="grey", lw=2.0, label="Median best"),
        Patch(facecolor="grey", alpha=0.25, label="Min–max range"),
    ]
    ax.legend(handles=style_handles, loc="lower right", fontsize=9,
              framealpha=0.95, title="Style")

    out_base = os.path.join(
        OUT_DIR, f"DrivAer_Star_convergence_reward_vs_evals_vtk_{body}_downforce_efficiency"
    )
    for ext in (".png", ".pdf"):
        fig.savefig(out_base + ext, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_base}{ext}")
    plt.close(fig)

    print(f"\n{'Method':<26}  {'n_runs':>6}  {'best':>8}  {'median_final':>12}  {'mean_final':>10}")
    print("-" * 68)
    for name, curves in [("Bayesian Opt. (exact GP)", bo_curves),
                          ("ShapeEvolve",              v3_curves)]:
        if not curves:
            continue
        finals = [c[-1] for c in curves]
        print(f"{name:<26}  {len(curves):>6}  {max(finals):>8.5f}  "
              f"{np.median(finals):>12.5f}  {np.mean(finals):>10.5f}")


if __name__ == "__main__":
    main()
