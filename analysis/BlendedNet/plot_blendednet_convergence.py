"""Convergence curves (best mean-CD vs function evaluations) for all BlendedNet methods.

For each method: median best-CD across runs (solid line) with min/max envelope
(shaded band). L-BFGS-B nr10 and v3 shown as dashed while runs are in-progress.

X-axis: log scale.

Usage:
    cd /scratch/ShapeEvolve
    source venv/bin/activate
    python analysis/BlendedNet/plot_blendednet_convergence.py

Outputs:
    environments/BlendedNet/results/analysis_plots/convergence_cd_vs_evals.png/.pdf
"""

import os
import re
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
OUT_DIR = os.path.join(RESULTS_DIR, "analysis_plots_shapebench_mean_cd")

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
    "L-BFGS-B":                "#e377c2",
    "Bayesian Opt. (exact GP)": "#ff7f0e",
    "PSO (20p × 200i)":        "#1f77b4",
    "ShapeEvolve":             "#2ca02c",
    "CMA-ES":                  "#d62728",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_curves(csv_paths, best_reward_col):
    """Return list of 1-D best-CD arrays (one per run), forward-fill running min."""
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


def load_lbfgsb():
    return _load_curves(
        glob.glob(os.path.join(RESULTS_DIR, "run_lbfgsb_shapebench_5_seed*_nr3_normfix", "results.csv")),
        "best_reward",
    )


def load_bo():
    """Concatenate n500 + n1000 extension per seed for full 1000-eval BO min-CD curve."""
    curves = []
    for csv500 in sorted(glob.glob(os.path.join(
            RESULTS_DIR, "run_BO_torch_shapebench_5_seed*_n500", "results.csv"))):
        m = re.search(r"_seed(\d+)_n500", os.path.basename(os.path.dirname(csv500)))
        if m is None:
            continue
        try:
            df500 = pd.read_csv(csv500)
        except Exception:
            continue
        if "best_reward" not in df500.columns:
            continue
        arr = -df500["best_reward"].values.astype(float)
        csv1000 = os.path.join(
            RESULTS_DIR, f"run_BO_torch_shapebench_5_seed{m.group(1)}_n1000", "results.csv")
        if os.path.exists(csv1000):
            try:
                df1000 = pd.read_csv(csv1000)
                if "best_reward" in df1000.columns:
                    arr = np.concatenate([arr, -df1000["best_reward"].values.astype(float)])
            except Exception:
                pass
        curves.append(np.minimum.accumulate(arr))
    return curves


def load_ga():
    csvs = []
    for pat in [
        os.path.join(RESULTS_DIR, "run_GA_shapebench_5_seed*_20p_200i", "results.csv"),
        os.path.join(RESULTS_DIR, "run_GA_shapebench_5_attempt*_20p_200i", "results.csv"),
    ]:
        csvs.extend(glob.glob(pat))
    return _load_curves(csvs, "gbest_reward")


def load_v3():
    return _load_curves(
        glob.glob(os.path.join(RESULTS_DIR, "run_v3_flash2_5_shapebench_5_attempt_*_n2000", "results.csv")),
        "best_reward",
    )


def load_cmaes():
    return _load_curves(
        glob.glob(os.path.join(RESULTS_DIR, "run_cmaes_shapebench_5_seed*_n500to1000", "results.csv")),
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


def plot_band(ax, curves, x_grid, color, label, extend=True, ls="-"):
    if not curves:
        return
    med, lo, hi = compute_band(curves, x_grid, extend=extend)
    ax.fill_between(x_grid, lo, hi, color=color, alpha=0.18)
    ax.plot(x_grid, med, color=color, lw=1.8, label=label, ls=ls)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    plt.rcParams.update(STYLE)
    os.makedirs(OUT_DIR, exist_ok=True)

    lbfgsb_curves = load_lbfgsb()
    bo_curves     = load_bo()
    ga_curves     = load_ga()
    v3_curves     = load_v3()
    cmaes_curves  = load_cmaes()

    for name, curves in [
        ("L-BFGS-B",                lbfgsb_curves),
        ("Bayesian Opt. (exact GP)", bo_curves),
        ("PSO (20p × 200i)",        ga_curves),
        ("ShapeEvolve",             v3_curves),
        ("CMA-ES",                  cmaes_curves),
    ]:
        if curves:
            print(f"{name:<28} {len(curves):>3} runs  max_evals={max(len(c) for c in curves)}")

    all_curves = [c for g in [lbfgsb_curves, bo_curves, ga_curves, v3_curves, cmaes_curves] for c in g]
    if not all_curves:
        print("No data found.")
        return

    x_max = max(len(c) for c in all_curves)
    x_grid = np.unique(
        np.concatenate([np.geomspace(1, x_max, 3000).astype(int), [x_max]])
    )

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")

    def xg(curves):
        # Each method's x-grid is clipped to its own max run length (extend=True within
        # that range). This matches the DrivAer convergence plot convention: complete runs
        # that ended earlier within a method extend flat to the method's x_max (honest —
        # they converged and the best-found value is still valid), but no flat tail bleeds
        # into other methods' x territory. Using extend=False instead introduces selection
        # bias near the tail: the median is computed over only the longest-running subset,
        # which can artificially lower the apparent final CD if those runs happened to find
        # better designs (observed for ShapeEvolve: att8 exits at 1737, leaving only the
        # 5 longest runs — which include att1/att5, the two best — in the final median).
        return x_grid[x_grid <= max(len(c) for c in curves)]

    if lbfgsb_curves:
        plot_band(ax, lbfgsb_curves, xg(lbfgsb_curves), COLORS["L-BFGS-B"], "L-BFGS-B")

    if bo_curves:
        plot_band(ax, bo_curves, xg(bo_curves), COLORS["Bayesian Opt. (exact GP)"],
                  "Bayesian Opt. (exact GP)")

    if ga_curves:
        plot_band(ax, ga_curves, xg(ga_curves), COLORS["PSO (20p × 200i)"],
                  "PSO (20p × 200i)")

    if v3_curves:
        plot_band(ax, v3_curves, xg(v3_curves), COLORS["ShapeEvolve"], "ShapeEvolve")

    if cmaes_curves:
        plot_band(ax, cmaes_curves, xg(cmaes_curves), COLORS["CMA-ES"], "CMA-ES")

    ax.set_xscale("log")
    ax.set_xlabel("Function evaluations (per run)")
    ax.set_ylabel(r"$\overline{C_D}$")
    ax.set_title(
        r"BlendedNet (BWB) — Convergence: best $\overline{C_D}$ vs function evaluations",
        fontweight="medium", pad=8,
    )
    ax.set_xlim(1, x_max)
    ax.grid(True, which="both", alpha=0.25)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    method_leg = ax.legend(fontsize=8.5, loc="upper right", framealpha=0.95, title="Method")
    ax.add_artist(method_leg)
    style_handles = [
        Patch(facecolor="grey", alpha=0.25, label="Min–max range"),
        Line2D([0], [0], color="grey", lw=1.8, label="Median best"),
    ]
    ax.legend(handles=style_handles, loc="lower left", fontsize=8.5,
              framealpha=0.95, title="Style key",
              bbox_to_anchor=(0.0, 0.08))

    out_png = os.path.join(OUT_DIR, "BlendedNet_convergence_cd_vs_evals.png")
    out_pdf = os.path.join(OUT_DIR, "BlendedNet_convergence_cd_vs_evals.pdf")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")

    print("\nMethod                      n_runs  best_CD   median_final")
    print("-" * 60)
    for name, curves in [
        ("L-BFGS-B",                lbfgsb_curves),
        ("Bayesian Opt. (exact GP)", bo_curves),
        ("PSO (20p × 200i)",        ga_curves),
        ("ShapeEvolve",             v3_curves),
        ("CMA-ES",                  cmaes_curves),
    ]:
        if not curves:
            continue
        finals = [c[-1] for c in curves]
        print(f"{name:<18} {len(curves):>5}  {min(finals):.6f}    {np.median(finals):.6f}")


if __name__ == "__main__":
    main()
