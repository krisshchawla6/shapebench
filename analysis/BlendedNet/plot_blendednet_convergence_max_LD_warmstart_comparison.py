"""Two-panel convergence comparison for shapebench_5_max_LD.

Left panel : random-start convergence — median best-L/D ± min/max band for
             each method (BO n=1000, PSO, CMA-ES, ShapeEvolve).
Right panel: warm-start convergence — same methods from Corner A warm-start
             (PSO, CMA-ES, ShapeEvolve; BO has no warm-start run).

X-axis: log scale, shared between panels.
Y-axis: shared range across both panels.

Usage:
    cd /scratch/ShapeEvolve
    source /home/jack/venv_torch210/bin/activate
    python analysis/BlendedNet/plot_blendednet_convergence_max_LD_warmstart_comparison.py

Outputs:
    environments/BlendedNet/results/analysis_plots_shapebench_5_max_LD/convergence_LD_vs_evals_warmstart_comparison.png/.pdf
"""

import os
import re
import glob
import json
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
    "Bayesian Opt. n=1000": "#ff7f0e",
    "PSO (20p × 200i)":     "#1f77b4",
    "CMA-ES":               "#d62728",
    "ShapeEvolve":          "#2ca02c",
}


# ── Data loading ──────────────────────────────────────────────────────────────

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
        arr = df[best_reward_col].values.astype(float)
        arr = np.maximum.accumulate(arr)
        curves.append(arr)
    return curves


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


def load_ga_warmstart_curves():
    return _load_curves(
        glob.glob(os.path.join(RESULTS_DIR, "run_GA_parallel_shapebench_5_max_LD_warmstart_cornerA_seed*_20p_100i", "results.csv")),
        "gbest_reward",
    )


def load_cmaes_warmstart_curves():
    return _load_curves(
        glob.glob(os.path.join(RESULTS_DIR, "run_cmaes_shapebench_5_max_LD_warmstart_cornerA_seed*_n500", "results.csv")),
        "best_reward",
    )


def load_v3_warmstart_curves():
    """Build convergence curves from new evaluations only (skip iter=-1 initial design)."""
    curves = []
    for db_path in sorted(glob.glob(os.path.join(
            RESULTS_DIR,
            "run_v3_flash2_5_shapebench_5_max_LD_warmstart_cornerA_attempt_*_n2000",
            "database.json"))):
        try:
            db = json.load(open(db_path))
        except Exception:
            continue
        if not db:
            continue
        pairs = []
        for e in db:
            m = re.match(r"iter_(\d+)", os.path.basename(e["path"]))
            if m is None:
                continue  # skip iter=-1 initial design (pre-loaded, not a new eval)
            pairs.append((int(m.group(1)), float(e["reward"])))
        if not pairs:
            continue
        pairs.sort()
        arr = np.maximum.accumulate(np.array([r for _, r in pairs]))
        curves.append(arr)
    return curves


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


def plot_band(ax, curves, x_grid, color, label, ls="-", extend=True):
    if not curves:
        return
    med, lo, hi = compute_band(curves, x_grid, extend=extend)
    ax.fill_between(x_grid, lo, hi, color=color, alpha=0.18)
    ax.plot(x_grid, med, color=color, lw=1.8, label=label, ls=ls)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    plt.rcParams.update(STYLE)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Random-start
    bo1000_curves = load_bo_n1000()
    ga_curves     = load_ga_parallel()
    cmaes_curves  = load_cmaes()
    v3_curves     = load_v3()

    # Warm-start
    ga_ws_curves    = load_ga_warmstart_curves()
    cmaes_ws_curves = load_cmaes_warmstart_curves()
    v3_ws_curves    = load_v3_warmstart_curves()

    for label, curves in [
        ("BO n=1000",           bo1000_curves),
        ("PSO random",          ga_curves),
        ("CMA-ES random",       cmaes_curves),
        ("ShapeEvolve random",  v3_curves),
        ("PSO warm-start",      ga_ws_curves),
        ("CMA-ES warm-start",   cmaes_ws_curves),
        ("SE warm-start",       v3_ws_curves),
    ]:
        if curves:
            print(f"{label:<24} {len(curves):>3} runs  max_evals={max(len(c) for c in curves)}")

    rand_curves = [c for g in [bo1000_curves, ga_curves, cmaes_curves, v3_curves] for c in g]
    ws_curves_all = [c for g in [ga_ws_curves, cmaes_ws_curves, v3_ws_curves] for c in g]
    if not rand_curves:
        print("No random-start data found.")
        return

    x_max_rand = max(len(c) for c in rand_curves)
    x_max_ws   = max(len(c) for c in ws_curves_all) if ws_curves_all else x_max_rand

    x_grid_rand = np.unique(np.concatenate([np.geomspace(1, x_max_rand, 3000).astype(int), [x_max_rand]]))
    x_grid_ws   = np.unique(np.concatenate([np.geomspace(1, x_max_ws,   3000).astype(int), [x_max_ws]]))

    def xg(curves, x_grid):
        return x_grid[x_grid <= max(len(c) for c in curves)]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6), facecolor="white", sharey=True)
    fig.subplots_adjust(wspace=0.08)

    # ── Left panel: random-start ──────────────────────────────────────────────
    if bo1000_curves:
        plot_band(ax_l, bo1000_curves, xg(bo1000_curves, x_grid_rand),
                  COLORS["Bayesian Opt. n=1000"], "Bayesian Opt.")
    if ga_curves:
        plot_band(ax_l, ga_curves, xg(ga_curves, x_grid_rand),
                  COLORS["PSO (20p × 200i)"], "PSO (20p × 200i)")
    if cmaes_curves:
        plot_band(ax_l, cmaes_curves, xg(cmaes_curves, x_grid_rand),
                  COLORS["CMA-ES"], "CMA-ES")
    if v3_curves:
        plot_band(ax_l, v3_curves, xg(v3_curves, x_grid_rand),
                  COLORS["ShapeEvolve"], "ShapeEvolve")

    ax_l.set_xscale("log")
    ax_l.set_xlim(1, x_max_rand)
    ax_l.set_xlabel("Function evaluations (per run)")
    ax_l.set_ylabel(r"$\overline{L/D}$")
    ax_l.set_title("Random start", fontweight="medium")
    ax_l.grid(True, which="both", alpha=0.25)
    for sp in ["top", "right"]:
        ax_l.spines[sp].set_visible(False)
    ax_l.legend(fontsize=8.5, loc="lower right", framealpha=0.95, title="Method")

    # ── Right panel: warm-start ───────────────────────────────────────────────
    if ga_ws_curves:
        plot_band(ax_r, ga_ws_curves, xg(ga_ws_curves, x_grid_ws),
                  COLORS["PSO (20p × 200i)"], "PSO (20p × 200i)")
    if cmaes_ws_curves:
        plot_band(ax_r, cmaes_ws_curves, xg(cmaes_ws_curves, x_grid_ws),
                  COLORS["CMA-ES"], "CMA-ES")
    if v3_ws_curves:
        plot_band(ax_r, v3_ws_curves, xg(v3_ws_curves, x_grid_ws),
                  COLORS["ShapeEvolve"], "ShapeEvolve")

    ax_r.set_xscale("log")
    ax_r.set_xlim(1, x_max_ws)
    ax_r.set_xlabel("Function evaluations (per run)")
    ax_r.set_title("Warm-start from Corner A", fontweight="medium")
    ax_r.grid(True, which="both", alpha=0.25)
    for sp in ["top", "right"]:
        ax_r.spines[sp].set_visible(False)
    ax_r.legend(fontsize=8.5, loc="lower right", framealpha=0.95, title="Method")

    # Shared style key
    style_handles = [
        Patch(facecolor="grey", alpha=0.25, label="Min–max range"),
        Line2D([0], [0], color="grey", lw=1.8, label="Median best"),
    ]
    fig.legend(handles=style_handles, fontsize=8.5, loc="upper center",
               ncol=2, framealpha=0.95, title="Style key",
               bbox_to_anchor=(0.5, 0.98))

    fig.suptitle(
        r"BlendedNet (BWB) — Convergence: best $\overline{L/D}$ vs function evaluations",
        fontsize=10, fontweight="medium", y=1.04,
    )

    out_png = os.path.join(OUT_DIR, "BlendedNet_convergence_LD_vs_evals_warmstart_comparison.png")
    out_pdf = os.path.join(OUT_DIR, "BlendedNet_convergence_LD_vs_evals_warmstart_comparison.pdf")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")

    print("\nMethod                      n_runs  best_LD   median_final")
    print("-" * 60)
    for name, curves in [
        ("BO random",               bo1000_curves),
        ("PSO random",              ga_curves),
        ("CMA-ES random",           cmaes_curves),
        ("ShapeEvolve random",      v3_curves),
        ("PSO warm-start",          ga_ws_curves),
        ("CMA-ES warm-start",       cmaes_ws_curves),
        ("ShapeEvolve warm-start",  v3_ws_curves),
    ]:
        if not curves:
            continue
        finals = [c[-1] for c in curves]
        print(f"{name:<28} {len(curves):>5}  {max(finals):.6f}    {np.median(finals):.6f}")


if __name__ == "__main__":
    main()
