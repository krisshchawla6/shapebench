"""Convergence curves (best mean-L/D vs function evaluations) for shapebench_5_max_LD.

For each method: median best-L/D across runs (solid line) with min/max envelope
(shaded band).  v3 shown dashed while runs are in-progress.  BO n=1000 shown
dashed (still running for most seeds at time of plot).

X-axis: log scale.

Usage:
    cd /scratch/ShapeEvolve
    source /home/jack/venv_torch210/bin/activate
    python analysis/BlendedNet/plot_blendednet_convergence_max_LD.py

Outputs:
    environments/BlendedNet/results/analysis_plots_shapebench_5_max_LD/convergence_LD_vs_evals.png/.pdf
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
RESULTS_DIR = os.path.join(REPO_DIR, "environments", "BlendedNet", "results")
OUT_DIR = os.path.join(RESULTS_DIR, "analysis_plots_shapebench_5_max_LD")

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

COLORS = {
    "Bayesian Opt. n=500":  "#ff7f0e",
    "Bayesian Opt. n=1000": "#ffb366",
    "PSO (20p × 200i)":     "#1f77b4",
    "CMA-ES":               "#d62728",
    "ShapeEvolve":          "#2ca02c",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_curves(csv_paths, best_reward_col):
    """Return list of 1-D best-L/D arrays (one per run), running max."""
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
        arr = df[best_reward_col].values.astype(float)
        arr = np.maximum.accumulate(arr)
        curves.append(arr)
    return curves


def load_bo_n500():
    return _load_curves(
        glob.glob(os.path.join(RESULTS_DIR, "run_BO_torch_shapebench_5_max_LD_seed*_n500", "results.csv")),
        "best_reward",
    )


def load_bo_n1000():
    return _load_curves(
        glob.glob(os.path.join(RESULTS_DIR, "run_BO_torch_shapebench_5_max_LD_seed*_n1000", "results.csv")),
        "best_reward",
    )


def load_ga_parallel():
    return _load_curves(
        glob.glob(os.path.join(RESULTS_DIR, "run_GA_parallel_shapebench_5_max_LD_seed*_20p_200i", "results.csv")),
        "gbest_reward",
    )


def load_cmaes():
    return _load_curves(
        glob.glob(os.path.join(RESULTS_DIR, "run_cmaes_shapebench_5_max_LD_seed*_n1000", "results.csv")),
        "best_reward",
    )


def load_v3():
    return _load_curves(
        glob.glob(os.path.join(RESULTS_DIR, "run_v3_flash2_5_shapebench_5_max_LD_attempt_*_n2000", "results.csv")),
        "best_reward",
    )


# ── Interpolation onto common log-spaced grid ─────────────────────────────────

def interp_to_grid(curve, x_out, extend=False):
    n = len(curve)
    idx = np.searchsorted(np.arange(1, n + 1), x_out, side="right") - 1
    idx = np.clip(idx, 0, n - 1)
    out = curve[idx].copy()
    if not extend:
        out[x_out > n] = np.nan
    out[x_out < 1] = np.nan
    return out


def compute_band(curves, x_grid, extend=False):
    mat = np.vstack([interp_to_grid(c, x_grid, extend=extend) for c in curves])
    n_valid = np.sum(~np.isnan(mat), axis=0)
    mask = n_valid >= max(1, len(curves) // 2)
    med = np.where(mask, np.nanpercentile(mat, 50, axis=0), np.nan)
    lo  = np.where(mask, np.nanmin(mat, axis=0), np.nan)
    hi  = np.where(mask, np.nanmax(mat, axis=0), np.nan)
    return med, lo, hi


def plot_band(ax, curves, x_grid, color, label, ls="-"):
    if not curves:
        return
    med, lo, hi = compute_band(curves, x_grid, extend=True)
    ax.fill_between(x_grid, lo, hi, color=color, alpha=0.18)
    ax.plot(x_grid, med, color=color, lw=1.8, label=label, ls=ls)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    plt.rcParams.update(STYLE)
    os.makedirs(OUT_DIR, exist_ok=True)

    bo1000_curves = load_bo_n1000()
    ga_curves     = load_ga_parallel()
    cmaes_curves  = load_cmaes()
    v3_curves     = load_v3()

    for name, curves in [
        ("Bayesian Opt. n=1000", bo1000_curves),
        ("PSO (20p × 200i)",     ga_curves),
        ("CMA-ES",               cmaes_curves),
        ("ShapeEvolve",          v3_curves),
    ]:
        if curves:
            print(f"{name:<28} {len(curves):>3} runs  max_evals={max(len(c) for c in curves)}")

    all_curves = [c for g in [bo1000_curves, ga_curves, cmaes_curves, v3_curves] for c in g]
    if not all_curves:
        print("No data found.")
        return

    x_max = max(len(c) for c in all_curves)
    x_grid = np.unique(
        np.concatenate([np.geomspace(1, x_max, 3000).astype(int), [x_max]])
    )

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")

    def xg(curves):
        return x_grid[x_grid <= max(len(c) for c in curves)]

    if bo1000_curves:
        plot_band(ax, bo1000_curves, xg(bo1000_curves),
                  COLORS["Bayesian Opt. n=1000"], "Bayesian Opt.")

    if ga_curves:
        plot_band(ax, ga_curves, xg(ga_curves),
                  COLORS["PSO (20p × 200i)"], "PSO (20p × 200i)")

    if cmaes_curves:
        plot_band(ax, cmaes_curves, xg(cmaes_curves),
                  COLORS["CMA-ES"], "CMA-ES")

    if v3_curves:
        plot_band(ax, v3_curves, xg(v3_curves),
                  COLORS["ShapeEvolve"], "ShapeEvolve")

    ax.set_xscale("log")
    ax.set_xlabel("Function evaluations (per run)")
    ax.set_ylabel("Mean L/D")
    ax.set_title(
        "BlendedNet (BWB) — Convergence: best mean-L/D vs function evaluations\n"
        r"Reward = mean($C_L/C_D$) at $C_L^\star \in \{0.185, 0.206, 0.227\}$, "
        r"$M_\infty=0.3$, $\mathrm{Re}=10^7$",
        fontweight="medium", pad=8,
    )
    ax.set_xlim(1, x_max)
    ax.grid(True, which="both", alpha=0.25)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    method_leg = ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95, title="Method")
    ax.add_artist(method_leg)
    style_handles = [
        Patch(facecolor="grey", alpha=0.25, label="Min–max range"),
        Line2D([0], [0], color="grey", lw=1.8, label="Median best"),
        Line2D([0], [0], color="grey", lw=1.8, ls="--", label="In progress"),
    ]
    ax.legend(handles=style_handles, loc="upper left", fontsize=8.5,
              framealpha=0.95, title="Style key")

    out_png = os.path.join(OUT_DIR, "convergence_LD_vs_evals.png")
    out_pdf = os.path.join(OUT_DIR, "convergence_LD_vs_evals.pdf")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")

    print("\nMethod                      n_runs  best_LD   median_final")
    print("-" * 60)
    for name, curves in [
        ("Bayesian Opt. n=1000", bo1000_curves),
        ("PSO (20p × 200i)",     ga_curves),
        ("CMA-ES",               cmaes_curves),
        ("ShapeEvolve",          v3_curves),
    ]:
        if not curves:
            continue
        finals = [c[-1] for c in curves]
        print(f"{name:<28} {len(curves):>5}  {max(finals):.6f}    {np.median(finals):.6f}")


if __name__ == "__main__":
    main()
