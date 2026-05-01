"""Convergence curves (best penalized objective vs. evaluations) for
DrivAer Star cd_cl_constrained benchmark.

Reward: reward = -Cd - max(0, Cl - (-0.10))
At convergence (constraint satisfied): reward ≈ -Cd, so -best_reward ≈ Cd.

Data source: results_recovered.csv (reconstructed from per-design results.json
after SLURM preemption wiped results.csv).

Methods plotted: ShapeEvolve (v3, 10 attempts) and Bayesian Opt. (BO, 10 seeds).
L-BFGS-B and PSO were not run for this reward variant.

Usage:
    cd /scratch/ShapeEvolve
    source venv/bin/activate
    python analysis/DrivAer_Star/plot_drivaer_convergence_cd_cl_constrained.py [--body {E,F,N}]

Outputs:
    environments/DrivAer_Star/results/analysis_plots_cd_cl_constrained/
        convergence_cd_vs_evals_vtk_{body}.pdf/.png
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
OUT_DIR     = os.path.join(RESULTS_DIR, "analysis_plots_cd_cl_constrained")

# Baseline Cd: undeformed geometry (same base VTK as cd_only benchmark)
BASELINE_CD_BY_BODY = {
    "E": 0.22334,
    "F": 0.18990,
    "N": 0.18132,
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

def _load_curves(csv_paths):
    """Return list of 1-D best-penalised-objective arrays (one per run).

    Y-value = -best_reward  (→ Cd at convergence when constraint satisfied).
    Curve is monotone non-increasing (forward-fill running min).
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
        arr = -df["best_reward"].values.astype(float)
        arr = np.minimum.accumulate(arr)
        curves.append(arr)
    return curves


def load_v3(body):
    csvs = glob.glob(os.path.join(
        RESULTS_DIR,
        f"run_v3_dynamic_optimizer_cd_cl_constrained_drivaer_star_vtk_{body}"
        f"_attempt_*_flash_2_5_n6000",
        "results_recovered.csv",
    ))
    curves = _load_curves(csvs)
    # Exclude runs shorter than 50% of the longest (cancelled/reset early)
    if curves:
        max_len = max(len(c) for c in curves)
        curves = [c for c in curves if len(c) >= max_len * 0.5]
    return curves


def load_bo(body):
    csvs = glob.glob(os.path.join(
        RESULTS_DIR,
        f"run_BO_torch_cd_cl_constrained_vtk_{body}_seed*_n1000",
        "results_recovered.csv",
    ))
    return _load_curves(csvs)


# ── Interpolation ─────────────────────────────────────────────────────────────

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
    args   = parser.parse_args()
    body   = args.body

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.rcParams.update(STYLE)

    body_names  = {"E": "Estate", "F": "Fastback", "N": "Notchback"}
    body_label  = body_names[body]
    BASELINE_CD = BASELINE_CD_BY_BODY[body]

    bo_curves = load_bo(body)
    v3_curves = load_v3(body)

    for name, curves in [("Bayesian Opt. (exact GP)", bo_curves),
                          ("ShapeEvolve",              v3_curves)]:
        if curves:
            print(f"{name}: {len(curves)} runs, "
                  f"max evals = {max(len(c) for c in curves)}, "
                  f"best final = {min(c[-1] for c in curves):.5f}")

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
    ax.axhline(BASELINE_CD, color="#999999", lw=1.2, ls="--", zorder=1)
    ax.text(1.3, BASELINE_CD + 0.003, f"Baseline  $C_D$ = {BASELINE_CD:.5f}",
            color="#666666", fontsize=8.5, va="bottom")

    # BO — near-complete (79–99%), extend flat to x_max_bo
    if bo_curves:
        x_max_bo   = max(len(c) for c in bo_curves)
        x_grid_bo  = x_grid[x_grid <= x_max_bo + 1]
        plot_band(ax, bo_curves, x_grid_bo,
                  COLORS["Bayesian Opt. (exact GP)"],
                  "Bayesian Opt. (exact GP)", extend=True)

    # ShapeEvolve — partial (54–75%), do NOT extend past actual run length
    if v3_curves:
        x_max_v3  = max(len(c) for c in v3_curves)
        x_grid_v3 = x_grid[x_grid <= x_max_v3 + 1]
        plot_band(ax, v3_curves, x_grid_v3,
                  COLORS["ShapeEvolve"], "ShapeEvolve", extend=False)

    ax.set_xscale("log")
    ax.set_xlabel("Function evaluations (per run)", fontsize=11)
    ax.set_ylabel(r"$-\mathrm{reward} \approx C_D$  (penalised objective)", fontsize=11)
    ax.set_title(
        f"DrivAer$^\\star$ ({body_label}, vtk_{body}) — "
        r"$C_D$-constrained ($C_l \leq -0.10$) optimisation" + "\n"
        "Convergence: best penalised objective per run  "
        "(solid = median,  band = min–max)",
        fontsize=11,
    )
    ax.set_xlim(1, x_max)
    ax.set_ylim(0.02, BASELINE_CD * 1.08)
    ax.grid(True, which="both", alpha=0.25)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    method_leg = ax.legend(fontsize=9, loc="upper right",
                           framealpha=0.95, title="Method")
    ax.add_artist(method_leg)
    style_handles = [
        Line2D([0], [0], color="grey", lw=2.0, label="Median best"),
        Patch(facecolor="grey", alpha=0.25, label="Min–max range"),
    ]
    ax.legend(handles=style_handles, loc="lower left", fontsize=9,
              framealpha=0.95, title="Style")

    suffix = f"_vtk_{body}"
    for ext in (".png", ".pdf"):
        out = os.path.join(OUT_DIR, f"DrivAer_Star_convergence_cd_vs_evals{suffix}{ext}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)

    print(f"\nMethod            n_runs  best    median_final  mean_final")
    print("-" * 60)
    for name, curves in [("Bayesian Opt. (exact GP)", bo_curves),
                          ("ShapeEvolve",              v3_curves)]:
        if not curves:
            continue
        finals = [c[-1] for c in curves]
        print(f"{name:<26} {len(curves):>4}  {min(finals):.5f}  "
              f"{np.median(finals):.5f}     {np.mean(finals):.5f}")


if __name__ == "__main__":
    main()
