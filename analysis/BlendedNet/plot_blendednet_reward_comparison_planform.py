"""Planform overlay comparing best designs between shapebench_5 (min-CD) and
shapebench_5_max_LD (max-L/D) for each method.

One subplot per method (2×2 grid).  Each subplot shows the two outlines
overlaid at the same scale so shape differences are immediately visible.
All subplots share the same axis limits for cross-method comparison.

Usage:
    cd /scratch/ShapeEvolve
    source /home/jack/venv_torch210/bin/activate
    python analysis/BlendedNet/plot_blendednet_reward_comparison_planform.py

Outputs:
    environments/BlendedNet/results/analysis_plots_reward_comparison/planform_reward_comparison.png/.pdf
"""

import os
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
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 150,
}

COLOR_CD = "#1f77b4"   # blue  — min-CD reward
COLOR_LD = "#d62728"   # red   — max-L/D reward


# ── Geometry ──────────────────────────────────────────────────────────────────

C1 = 1000.0

def full_span_polygon(params):
    B1 = params["B1"]; B2 = params["B2"]; B3 = params["B3"]
    C2 = params["C2"]; C3 = params["C3"]; C4 = params["C4"]
    S1 = np.radians(params["S1"])
    S2 = np.radians(params["S2"])
    S3 = np.radians(params["S3"])

    y = np.array([0.0, B1, B1 + B2, B1 + B2 + B3])
    le = np.array([
        0.0,
        B1 * np.tan(S1),
        B1 * np.tan(S1) + B2 * np.tan(S2),
        B1 * np.tan(S1) + B2 * np.tan(S2) + B3 * np.tan(S3),
    ])
    chord = np.array([C1, C2, C3, C4])
    te = le + chord

    xs_le = y;          xs_te = y[::-1]
    ys_le = le;         ys_te = te[::-1]
    xp_le = -y[::-1];  xp_te = -y
    yp_le = le[::-1];  yp_te = te

    x = np.concatenate([xp_te, xp_le, xs_le, xs_te, [xp_te[0]]])
    y_out = np.concatenate([yp_te, yp_le, ys_le, ys_te, [yp_te[0]]])
    return x, y_out


# ── Data loading ──────────────────────────────────────────────────────────────

def _design_dir(run_dir, row, method_key):
    if method_key == "ga":
        return os.path.join(run_dir, f"iter_{int(row['iteration']):04d}_p{int(row['particle']):03d}")
    else:
        return os.path.join(run_dir, str(row["design"]))


def load_best(dirs, reward_col, method_key):
    """Returns (params, reward_val, results_dict) for the single best design."""
    best_r, best_dir, best_row = -np.inf, None, None
    for d in dirs:
        csv = os.path.join(d, "results.csv")
        if not os.path.exists(csv):
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if reward_col not in df.columns:
            continue
        idx = int(df[reward_col].idxmax())
        row = df.iloc[idx]
        r = float(row[reward_col])
        if r > best_r:
            best_r, best_dir, best_row = r, d, row
    if best_dir is None:
        return None, None, None
    ddir = _design_dir(best_dir, best_row, method_key)
    rj = os.path.join(ddir, "save", "results.json")
    if not os.path.exists(rj):
        return None, None, None
    with open(rj) as f:
        res = json.load(f)
    return res["design"], best_r, res


def _cross_metrics(res):
    """Return (mean_CD, mean_LD) computed from results dict, regardless of reward type."""
    ops = res.get("operating_points", [])
    if "mean_CD" in res:
        mean_cd = res["mean_CD"]
    elif ops:
        mean_cd = float(np.mean([op["CD_approx"] for op in ops]))
    else:
        mean_cd = None
    if "mean_LD" in res:
        mean_ld = res["mean_LD"]
    elif ops:
        cds = [op["CD_approx"] for op in ops]
        cls = [op["CL_approx"] for op in ops]
        mean_ld = float(np.mean([cl / cd for cl, cd in zip(cls, cds)]))
    else:
        mean_ld = None
    return mean_cd, mean_ld


METHODS = {
    "Bayesian Opt.": {
        "cd_dirs": lambda: sorted(glob.glob(os.path.join(RESULTS_DIR, "run_BO_torch_shapebench_5_seed*_n500"))),
        "ld_dirs": lambda: sorted(glob.glob(os.path.join(RESULTS_DIR, "run_BO_torch_shapebench_5_max_LD_seed*_n1000"))),
        "mkey": "bo",
    },
    "PSO (20p × 200i)": {
        "cd_dirs": lambda: (
            sorted(glob.glob(os.path.join(RESULTS_DIR, "run_GA_shapebench_5_seed*_20p_200i"))) +
            sorted(glob.glob(os.path.join(RESULTS_DIR, "run_GA_shapebench_5_attempt*_20p_200i")))
        ),
        "ld_dirs": lambda: sorted(glob.glob(os.path.join(RESULTS_DIR, "run_GA_parallel_shapebench_5_max_LD_seed*_20p_200i"))),
        "mkey": "ga",
    },
    "CMA-ES": {
        "cd_dirs": lambda: sorted(glob.glob(os.path.join(RESULTS_DIR, "run_cmaes_shapebench_5_seed*_n500to1000"))),
        "ld_dirs": lambda: sorted(glob.glob(os.path.join(RESULTS_DIR, "run_cmaes_shapebench_5_max_LD_seed*_n1000"))),
        "mkey": "bo",
    },
    "ShapeEvolve": {
        "cd_dirs": lambda: sorted(glob.glob(os.path.join(RESULTS_DIR, "run_v3_flash2_5_shapebench_5_attempt_*_n2000"))),
        "ld_dirs": lambda: sorted(glob.glob(os.path.join(RESULTS_DIR, "run_v3_flash2_5_shapebench_5_max_LD_attempt_*_n2000"))),
        "mkey": "v3",
    },
}

METHOD_ORDER = ["Bayesian Opt.", "PSO (20p × 200i)", "CMA-ES", "ShapeEvolve"]


# ── Annotation helper ─────────────────────────────────────────────────────────

def _ann(params, res):
    hs = params["B1"] + params["B2"] + params["B3"]
    mean_cd, mean_ld = _cross_metrics(res)
    cd_str = f"CD  = {mean_cd:.5f}" if mean_cd is not None else "CD  = n/a"
    ld_str = f"L/D = {mean_ld:.2f}"  if mean_ld is not None else "L/D = n/a"
    return (
        f"half-span={hs:.0f} mm\n"
        f"S1={params['S1']:.0f}°  C2={params['C2']:.0f}  C4={params['C4']:.0f}\n"
        f"B2={params['B2']:.0f}\n"
        f"{cd_str}\n"
        f"{ld_str}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    plt.rcParams.update(STYLE)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load all designs
    designs = {}
    for name in METHOD_ORDER:
        cfg = METHODS[name]
        p_cd, r_cd, res_cd = load_best(cfg["cd_dirs"](), "reward", cfg["mkey"])
        p_ld, r_ld, res_ld = load_best(cfg["ld_dirs"](), "reward", cfg["mkey"])
        designs[name] = (p_cd, r_cd, res_cd, p_ld, r_ld, res_ld)
        if p_cd and p_ld:
            cd_cd, ld_cd = _cross_metrics(res_cd)
            cd_ld, ld_ld = _cross_metrics(res_ld)
            hs_cd = p_cd["B1"]+p_cd["B2"]+p_cd["B3"]
            hs_ld = p_ld["B1"]+p_ld["B2"]+p_ld["B3"]
            print(f"{name}:")
            print(f"  min-CD best → CD={cd_cd:.5f}  L/D={ld_cd:.2f}  hs={hs_cd:.0f}mm")
            print(f"  max-LD best → CD={cd_ld:.5f}  L/D={ld_ld:.2f}  hs={hs_ld:.0f}mm")

    # Compute shared axis limits from all designs
    all_x, all_y = [], []
    for p_cd, _, _res_cd, p_ld, _, _res_ld in designs.values():
        for p in [p_cd, p_ld]:
            if p is None:
                continue
            x, y = full_span_polygon(p)
            all_x.extend(x); all_y.extend(y)
    pad_x = 80; pad_y = 60
    xlim = (min(all_x) - pad_x, max(all_x) + pad_x)
    ylim = (max(all_y) + pad_y, min(all_y) - pad_y)  # inverted (LE at top)

    fig, axes = plt.subplots(2, 2, figsize=(14, 16), facecolor="white")
    axes = axes.flatten()

    for ax, name in zip(axes, METHOD_ORDER):
        p_cd, r_cd, res_cd, p_ld, r_ld, res_ld = designs[name]

        if p_cd is not None:
            x, y = full_span_polygon(p_cd)
            ax.fill(x, y, color=COLOR_CD, alpha=0.15)
            ax.plot(x, y, color=COLOR_CD, lw=2.0, label="min-CD  (shapebench_5)")

        if p_ld is not None:
            x, y = full_span_polygon(p_ld)
            ax.fill(x, y, color=COLOR_LD, alpha=0.15)
            ax.plot(x, y, color=COLOR_LD, lw=2.0, ls="--", label="max-L/D (shapebench_5_max_LD)")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.axvline(0, color="grey", lw=0.5, ls="--", alpha=0.4)
        ax.grid(True, alpha=0.18)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.set_xlabel("Span (mm)")
        ax.set_ylabel("Chord (mm)")
        ax.set_title(name, fontweight="medium", pad=6)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

        # Annotation blocks — place CD text on left side, LD on right
        if p_cd is not None:
            ax.text(0.02, 0.97, _ann(p_cd, res_cd), transform=ax.transAxes,
                    fontsize=7.5, va="top", ha="left", color=COLOR_CD,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=COLOR_CD, alpha=0.85, linewidth=0.8))
        if p_ld is not None:
            ax.text(0.98, 0.97, _ann(p_ld, res_ld), transform=ax.transAxes,
                    fontsize=7.5, va="top", ha="right", color=COLOR_LD,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=COLOR_LD, alpha=0.85, linewidth=0.8))

    # Shared legend in figure
    legend_handles = [
        Line2D([0], [0], color=COLOR_CD, lw=2.0,      label="min-CD  (shapebench_5)"),
        Patch(facecolor=COLOR_CD, alpha=0.25,           label="min-CD fill"),
        Line2D([0], [0], color=COLOR_LD, lw=2.0, ls="--", label="max-L/D (shapebench_5_max_LD)"),
        Patch(facecolor=COLOR_LD, alpha=0.25,           label="max-L/D fill"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=9, framealpha=0.95, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        "BlendedNet (BWB) — Best design planform: min-CD vs max-L/D per method\n"
        "Full span shown (symmetric).  LE at top, TE at bottom.  Same scale across all panels.",
        fontsize=12, fontweight="medium", y=0.995,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])

    out_png = os.path.join(OUT_DIR, "planform_reward_comparison.png")
    out_pdf = os.path.join(OUT_DIR, "planform_reward_comparison.pdf")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
