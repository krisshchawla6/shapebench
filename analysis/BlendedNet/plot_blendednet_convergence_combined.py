"""Two-panel convergence: min-CD (left) and max-L/D (right) for all BlendedNet methods.

Left panel: best mean-CD vs evaluations (L-BFGS-B, BO, PSO, CMA-ES, ShapeEvolve).
Right panel: best mean-L/D vs evaluations (BO, PSO, CMA-ES, ShapeEvolve) with
  warm-start reference lines (dotted, same color) for PSO, CMA-ES, ShapeEvolve.

Consistent method colors across both panels.  Separate x-grids (different budgets).

Usage:
    cd /scratch/ShapeEvolve
    source venv/bin/activate
    python analysis/BlendedNet/plot_blendednet_convergence_combined.py

Outputs:
    environments/BlendedNet/results/analysis_plots_reward_comparison/convergence_combined.png/.pdf
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
OUT_DIR = os.path.join(RESULTS_DIR, "analysis_plots_reward_comparison")

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
    "L-BFGS-B":         "#e377c2",
    "Bayesian Opt.":    "#ff7f0e",
    "PSO (20p × 200i)": "#1f77b4",
    "CMA-ES":           "#d62728",
    "ShapeEvolve":      "#2ca02c",
}


# ── Min-CD data loading ────────────────────────────────────────────────────────

def _load_cd_curves(csv_paths, col):
    """Running-min CD curves (sign-flipped from negative reward)."""
    curves = []
    for csv in sorted(csv_paths):
        if not os.path.exists(csv):
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if col not in df.columns:
            continue
        arr = -df[col].values.astype(float)
        curves.append(np.minimum.accumulate(arr))
    return curves


def load_cd_lbfgsb():
    return _load_cd_curves(
        glob.glob(os.path.join(RESULTS_DIR, "run_lbfgsb_shapebench_5_seed*_nr3_normfix", "results.csv")),
        "best_reward",
    )


def load_cd_bo():
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


def load_cd_ga():
    csvs = (glob.glob(os.path.join(RESULTS_DIR, "run_GA_shapebench_5_seed*_20p_200i", "results.csv")) +
            glob.glob(os.path.join(RESULTS_DIR, "run_GA_shapebench_5_attempt*_20p_200i", "results.csv")))
    return _load_cd_curves(csvs, "gbest_reward")


def load_cd_cmaes():
    return _load_cd_curves(
        glob.glob(os.path.join(RESULTS_DIR, "run_cmaes_shapebench_5_seed*_n500to1000", "results.csv")),
        "best_reward",
    )


def load_cd_v3():
    return _load_cd_curves(
        glob.glob(os.path.join(RESULTS_DIR, "run_v3_flash2_5_shapebench_5_attempt_*_n2000", "results.csv")),
        "best_reward",
    )


# ── Max-L/D data loading ───────────────────────────────────────────────────────

def _load_ld_curves(csv_paths, col):
    """Running-max L/D curves."""
    curves = []
    for csv in sorted(csv_paths):
        if not os.path.exists(csv):
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if col not in df.columns:
            continue
        arr = df[col].values.astype(float)
        curves.append(np.maximum.accumulate(arr))
    return curves


def load_ld_bo():
    return _load_ld_curves(
        glob.glob(os.path.join(RESULTS_DIR, "run_BO_torch_shapebench_5_max_LD_seed*_n1000", "results.csv")),
        "best_reward",
    )


def load_ld_ga():
    return _load_ld_curves(
        glob.glob(os.path.join(RESULTS_DIR, "run_GA_parallel_shapebench_5_max_LD_seed*_20p_200i", "results.csv")),
        "gbest_reward",
    )


def load_ld_cmaes():
    return _load_ld_curves(
        glob.glob(os.path.join(RESULTS_DIR, "run_cmaes_shapebench_5_max_LD_seed*_n1000", "results.csv")),
        "best_reward",
    )


def load_ld_v3():
    return _load_ld_curves(
        glob.glob(os.path.join(RESULTS_DIR, "run_v3_flash2_5_shapebench_5_max_LD_attempt_*_n2000", "results.csv")),
        "best_reward",
    )


# ── Warm-start loaders (max-L/D panel reference lines) ────────────────────────

def load_ld_ga_ws():
    return _load_ld_curves(
        glob.glob(os.path.join(RESULTS_DIR, "run_GA_parallel_shapebench_5_max_LD_warmstart_cornerA_seed*_20p_100i", "results.csv")),
        "gbest_reward",
    )


def load_ld_cmaes_ws():
    return _load_ld_curves(
        glob.glob(os.path.join(RESULTS_DIR, "run_cmaes_shapebench_5_max_LD_warmstart_cornerA_seed*_n500", "results.csv")),
        "best_reward",
    )


def load_ld_v3_ws():
    """Convergence from new evaluations only (skip iter=-1 pre-loaded Corner A design)."""
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
                continue
            pairs.append((int(m.group(1)), float(e["reward"])))
        if not pairs:
            continue
        pairs.sort()
        arr = np.maximum.accumulate(np.array([r for _, r in pairs]))
        curves.append(arr)
    return curves


# ── Band infrastructure ────────────────────────────────────────────────────────

def interp_to_grid(curve, x_out, extend=False):
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
    lo  = np.where(mask, np.nanmin(mat, axis=0), np.nan)
    hi  = np.where(mask, np.nanmax(mat, axis=0), np.nan)
    return med, lo, hi


def plot_band(ax, curves, x_grid, color, label, ls="-"):
    if not curves:
        return
    med, lo, hi = compute_band(curves, x_grid)
    ax.fill_between(x_grid, lo, hi, color=color, alpha=0.18)
    ax.plot(x_grid, med, color=color, lw=1.8, label=label, ls=ls)


def make_grid(all_curves):
    xmax = max(len(c) for c in all_curves)
    grid = np.unique(np.concatenate([np.geomspace(1, xmax, 3000).astype(int), [xmax]]))
    return grid, xmax, lambda curves: grid[grid <= max(len(c) for c in curves)]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    plt.rcParams.update(STYLE)
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Load min-CD data ---
    lbfgsb   = load_cd_lbfgsb()
    cd_bo    = load_cd_bo()
    cd_ga    = load_cd_ga()
    cd_cmaes = load_cd_cmaes()
    cd_v3    = load_cd_v3()

    # --- Load max-L/D data ---
    ld_bo    = load_ld_bo()
    ld_ga    = load_ld_ga()
    ld_cmaes = load_ld_cmaes()
    ld_v3    = load_ld_v3()

    # --- Warm-start reference curves ---
    ld_ga_ws    = load_ld_ga_ws()
    ld_cmaes_ws = load_ld_cmaes_ws()
    ld_v3_ws    = load_ld_v3_ws()

    # Print diagnostics
    for tag, curves in [
        ("L-BFGS-B (CD)", lbfgsb), ("BO (CD)", cd_bo), ("PSO (CD)", cd_ga),
        ("CMA-ES (CD)", cd_cmaes), ("SE (CD)", cd_v3),
        ("BO (LD)", ld_bo), ("PSO (LD)", ld_ga),
        ("CMA-ES (LD)", ld_cmaes), ("SE (LD)", ld_v3),
        ("PSO WS", ld_ga_ws), ("CMA-ES WS", ld_cmaes_ws), ("SE WS", ld_v3_ws),
    ]:
        if curves:
            print(f"  {tag:<18} {len(curves):>3} runs  max_evals={max(len(c) for c in curves)}")

    cd_all = [c for g in [lbfgsb, cd_bo, cd_ga, cd_cmaes, cd_v3] for c in g]
    ld_all = [c for g in [ld_bo, ld_ga, ld_cmaes, ld_v3] for c in g]
    if not cd_all or not ld_all:
        print("Missing data for one or both panels.")
        return

    cd_grid, cd_xmax, xg_cd = make_grid(cd_all)
    ld_grid, ld_xmax, xg_ld = make_grid(ld_all)

    fig, (ax_cd, ax_ld) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="white",
                                        gridspec_kw={"wspace": 0.28})

    # ── Left panel: min-CD ────────────────────────────────────────────────────

    if lbfgsb:
        plot_band(ax_cd, lbfgsb,   xg_cd(lbfgsb),   COLORS["L-BFGS-B"],         "L-BFGS-B")
    if cd_bo:
        plot_band(ax_cd, cd_bo,    xg_cd(cd_bo),    COLORS["Bayesian Opt."],     "Bayesian Opt.")
    if cd_ga:
        plot_band(ax_cd, cd_ga,    xg_cd(cd_ga),    COLORS["PSO (20p × 200i)"],  "PSO (20p × 200i)")
    if cd_cmaes:
        plot_band(ax_cd, cd_cmaes, xg_cd(cd_cmaes), COLORS["CMA-ES"],            "CMA-ES")
    if cd_v3:
        plot_band(ax_cd, cd_v3,    xg_cd(cd_v3),    COLORS["ShapeEvolve"],       "ShapeEvolve")

    ax_cd.set_xscale("log")
    ax_cd.set_xlabel("Function evaluations (per run)")
    ax_cd.set_ylabel(r"Mean $C_D$")
    ax_cd.set_title(r"(a)  min-$C_D$ reward", fontweight="medium", pad=6)
    ax_cd.set_xlim(1, cd_xmax)
    ax_cd.grid(True, which="both", alpha=0.25)
    for sp in ["top", "right"]:
        ax_cd.spines[sp].set_visible(False)
    ax_cd.legend(fontsize=8.5, loc="upper right", framealpha=0.95, title="Method")

    # ── Right panel: max-L/D ─────────────────────────────────────────────────

    if ld_bo:
        plot_band(ax_ld, ld_bo,    xg_ld(ld_bo),    COLORS["Bayesian Opt."],     "Bayesian Opt.")
    if ld_ga:
        plot_band(ax_ld, ld_ga,    xg_ld(ld_ga),    COLORS["PSO (20p × 200i)"],  "PSO (20p × 200i)")
    if ld_cmaes:
        plot_band(ax_ld, ld_cmaes, xg_ld(ld_cmaes), COLORS["CMA-ES"],            "CMA-ES")
    if ld_v3:
        plot_band(ax_ld, ld_v3,    xg_ld(ld_v3),    COLORS["ShapeEvolve"],       "ShapeEvolve")

    # Warm-start reference lines: dotted horizontal at median final, shaded range
    for ws_curves, color_key in [
        (ld_ga_ws,    "PSO (20p × 200i)"),
        (ld_cmaes_ws, "CMA-ES"),
        (ld_v3_ws,    "ShapeEvolve"),
    ]:
        if not ws_curves:
            continue
        color = COLORS[color_key]
        finals = [c[-1] for c in ws_curves]
        ax_ld.axhspan(float(np.min(finals)), float(np.max(finals)),
                      color=color, alpha=0.10, zorder=3)
        ax_ld.axhline(float(np.median(finals)), color=color, lw=1.5, ls=":", zorder=5)

    ax_ld.set_xscale("log")
    ax_ld.set_xlabel("Function evaluations (per run)")
    ax_ld.set_ylabel(r"Mean $L/D$")
    ax_ld.set_title(r"(b)  max-$L/D$ reward", fontweight="medium", pad=6)
    ax_ld.set_xlim(1, ld_xmax)
    ax_ld.grid(True, which="both", alpha=0.25)
    for sp in ["top", "right"]:
        ax_ld.spines[sp].set_visible(False)

    # Right panel: method legend (lower right) + style key (upper left)
    method_handles = [
        Line2D([0], [0], color=COLORS[n], lw=2, label=n)
        for n in ["Bayesian Opt.", "PSO (20p × 200i)", "CMA-ES", "ShapeEvolve"]
    ]
    style_handles = [
        Patch(facecolor="grey", alpha=0.25, label="Min–max range"),
        Line2D([0], [0], color="grey", lw=1.8,  label="Median best"),
        Line2D([0], [0], color="grey", lw=1.5, ls=":", label="Warm-start ceiling"),
    ]
    leg_method = ax_ld.legend(handles=method_handles, loc="lower right",
                              fontsize=8.5, framealpha=0.95, title="Method")
    ax_ld.add_artist(leg_method)
    ax_ld.legend(handles=style_handles, loc="upper left",
                 fontsize=8.5, framealpha=0.95, title="Style key")

    fig.suptitle(
        "BlendedNet (BWB) — Convergence: best value vs. function evaluations",
        fontsize=12, fontweight="medium",
    )

    out_png = os.path.join(OUT_DIR, "convergence_combined.png")
    out_pdf = os.path.join(OUT_DIR, "convergence_combined.pdf")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
