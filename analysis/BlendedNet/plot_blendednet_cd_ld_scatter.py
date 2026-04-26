"""CD vs L/D cross-metric scatter for BlendedNet best designs.

PSO, CMA-ES, ShapeEvolve: circle markers (filled = min-CD reward, open = max-L/D reward).
  Per-method arrows from open → filled show the cross-metric anomaly: min-CD reward
  finds Corner A, which is simultaneously lower CD AND higher L/D than the max-L/D
  reward result (Corner B local trap).

Bayesian Opt.: diamond markers, separate annotation. BO max-L/D partially reaches
  Corner A (2/10 seeds), so both BO markers sit in the Corner A neighbourhood rather
  than spanning Corner A → Corner B like the other three methods.

Usage:
    cd /scratch/ShapeEvolve
    source /home/jack/venv_torch210/bin/activate
    python analysis/BlendedNet/plot_blendednet_cd_ld_scatter.py

Outputs:
    environments/BlendedNet/results/analysis_plots_reward_comparison/cd_ld_scatter.png/.pdf
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
from matplotlib.patches import FancyArrowPatch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO_DIR, "environments", "BlendedNet", "results")
OUT_DIR = os.path.join(RESULTS_DIR, "analysis_plots_reward_comparison")

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 12,
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

COLORS = {
    "Bayesian Opt.":    "#ff7f0e",
    "PSO (20p × 200i)": "#1f77b4",
    "CMA-ES":           "#d62728",
    "ShapeEvolve":      "#2ca02c",
}

# Methods that cleanly show the cross-metric anomaly (Corner A via min-CD,
# Corner B via max-L/D, matched budgets)
ANOMALY_METHODS = ["PSO (20p × 200i)", "CMA-ES", "ShapeEvolve"]
BO_METHOD       = "Bayesian Opt."
METHOD_ORDER    = ANOMALY_METHODS + [BO_METHOD]

MARKER_SIZE = 130
EDGE_WIDTH  = 1.8


# ── Data loading ──────────────────────────────────────────────────────────────

def _design_dir(run_dir, row, method_key):
    if method_key == "ga":
        return os.path.join(run_dir, f"iter_{int(row['iteration']):04d}_p{int(row['particle']):03d}")
    return os.path.join(run_dir, str(row["design"]))


def load_best(dirs, reward_col, method_key):
    """(params, reward, results_dict) for the single best design across dirs."""
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
    """(mean_CD, mean_LD) from results dict, regardless of reward type."""
    ops = res.get("operating_points", [])
    if "mean_CD" in res:
        mean_cd = float(res["mean_CD"])
    elif ops:
        mean_cd = float(np.mean([op["CD_approx"] for op in ops]))
    else:
        mean_cd = None
    if "mean_LD" in res:
        mean_ld = float(res["mean_LD"])
    elif ops:
        cds = [op["CD_approx"] for op in ops]
        cls = [op["CL_approx"] for op in ops]
        mean_ld = float(np.mean([cl / cd for cl, cd in zip(cls, cds)]))
    else:
        mean_ld = None
    return mean_cd, mean_ld


METHODS = {
    "Bayesian Opt.": {
        # Include both n=500 and n=1000 dirs so the best across all seeds is used
        "cd_dirs": lambda: (
            sorted(glob.glob(os.path.join(RESULTS_DIR, "run_BO_torch_shapebench_5_seed*_n500"))) +
            sorted(glob.glob(os.path.join(RESULTS_DIR, "run_BO_torch_shapebench_5_seed*_n1000")))
        ),
        "ld_dirs": lambda: sorted(glob.glob(
            os.path.join(RESULTS_DIR, "run_BO_torch_shapebench_5_max_LD_seed*_n1000"))),
        "mkey": "bo",
    },
    "PSO (20p × 200i)": {
        "cd_dirs": lambda: (
            sorted(glob.glob(os.path.join(RESULTS_DIR, "run_GA_shapebench_5_seed*_20p_200i"))) +
            sorted(glob.glob(os.path.join(RESULTS_DIR, "run_GA_shapebench_5_attempt*_20p_200i")))
        ),
        "ld_dirs": lambda: sorted(glob.glob(
            os.path.join(RESULTS_DIR, "run_GA_parallel_shapebench_5_max_LD_seed*_20p_200i"))),
        "mkey": "ga",
    },
    "CMA-ES": {
        "cd_dirs": lambda: sorted(glob.glob(
            os.path.join(RESULTS_DIR, "run_cmaes_shapebench_5_seed*_n500to1000"))),
        "ld_dirs": lambda: sorted(glob.glob(
            os.path.join(RESULTS_DIR, "run_cmaes_shapebench_5_max_LD_seed*_n1000"))),
        "mkey": "bo",
    },
    "ShapeEvolve": {
        "cd_dirs": lambda: sorted(glob.glob(
            os.path.join(RESULTS_DIR, "run_v3_flash2_5_shapebench_5_attempt_*_n2000"))),
        "ld_dirs": lambda: sorted(glob.glob(
            os.path.join(RESULTS_DIR, "run_v3_flash2_5_shapebench_5_max_LD_attempt_*_n2000"))),
        "mkey": "v3",
    },
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    plt.rcParams.update(STYLE)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Pre-load all best designs
    data = {}
    for name in METHOD_ORDER:
        cfg = METHODS[name]
        _, _, res_cd = load_best(cfg["cd_dirs"](), "reward", cfg["mkey"])
        _, _, res_ld = load_best(cfg["ld_dirs"](), "reward", cfg["mkey"])
        data[name] = (res_cd, res_ld)

    fig, ax = plt.subplots(figsize=(7.5, 6), facecolor="white")

    # Store plotted coordinates for drawing annotations afterwards
    coords = {}  # name -> {"cd": (x, y), "ld": (x, y)}

    # ── Anomaly methods: circles ───────────────────────────────────────────────
    for name in ANOMALY_METHODS:
        color = COLORS[name]
        res_cd, res_ld = data[name]
        cd_pt = ld_pt = None

        if res_cd is not None:
            cd, ld = _cross_metrics(res_cd)
            if cd is not None and ld is not None:
                ax.scatter(cd, ld, facecolors=color, edgecolors=color,
                           s=MARKER_SIZE, marker="o", linewidths=EDGE_WIDTH, zorder=5)
                cd_pt = (cd, ld)

        if res_ld is not None:
            cd, ld = _cross_metrics(res_ld)
            if cd is not None and ld is not None:
                ax.scatter(cd, ld, facecolors="none", edgecolors=color,
                           s=MARKER_SIZE, marker="o", linewidths=EDGE_WIDTH, zorder=5)
                ld_pt = (cd, ld)

        coords[name] = {"cd": cd_pt, "ld": ld_pt}

    # ── Bayesian Opt.: diamonds ────────────────────────────────────────────────
    res_cd, res_ld = data[BO_METHOD]
    bo_color = COLORS[BO_METHOD]
    bo_cd_pt = bo_ld_pt = None

    if res_cd is not None:
        cd, ld = _cross_metrics(res_cd)
        if cd is not None and ld is not None:
            ax.scatter(cd, ld, facecolors=bo_color, edgecolors=bo_color,
                       s=MARKER_SIZE, marker="D", linewidths=EDGE_WIDTH, zorder=5)
            bo_cd_pt = (cd, ld)

    if res_ld is not None:
        cd, ld = _cross_metrics(res_ld)
        if cd is not None and ld is not None:
            ax.scatter(cd, ld, facecolors="none", edgecolors=bo_color,
                       s=MARKER_SIZE, marker="D", linewidths=EDGE_WIDTH, zorder=5)
            bo_ld_pt = (cd, ld)

    coords[BO_METHOD] = {"cd": bo_cd_pt, "ld": bo_ld_pt}

    # ── Anomaly arrows: open circle → filled circle for each anomaly method ───
    for name in ANOMALY_METHODS:
        src = coords[name]["ld"]   # max-L/D (open) — Corner B
        dst = coords[name]["cd"]   # min-CD  (filled) — Corner A
        if src is not None and dst is not None:
            ax.annotate(
                "", xy=dst, xytext=src,
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=COLORS[name],
                    lw=1.2,
                    mutation_scale=10,
                    shrinkA=7, shrinkB=7,
                ),
                zorder=4,
            )

    # ── Region labels: Corner A and Corner B ──────────────────────────────────
    # Compute centroid of filled-circle (Corner A) cluster
    ca_pts = [coords[n]["cd"] for n in ANOMALY_METHODS if coords[n]["cd"] is not None]
    cb_pts = [coords[n]["ld"] for n in ANOMALY_METHODS if coords[n]["ld"] is not None]

    if ca_pts:
        cx = np.mean([p[0] for p in ca_pts])
        cy = np.mean([p[1] for p in ca_pts])
        # Place below the cluster so the label stays within the plot interior
        ax.text(cx, cy - 0.5, "Corner A", ha="center", va="top",
                fontsize=9, fontstyle="italic", color="#444444",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#aaaaaa", lw=0.7, alpha=0.85))

    if cb_pts:
        cx = np.mean([p[0] for p in cb_pts])
        cy = np.mean([p[1] for p in cb_pts])
        # Place above the cluster so the label stays within the plot interior
        ax.text(cx, cy + 0.5, "Corner B", ha="center", va="bottom",
                fontsize=9, fontstyle="italic", color="#444444",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#aaaaaa", lw=0.7, alpha=0.85))

    # ── Anomaly text annotation — upper left of centre, clear of Method legend ─
    ax.text(
        0.36, 0.97,
        r"$\longrightarrow$ min-$\overline{C_D}$ reward finds Corner A" + "\n"
        r"(lower $\overline{C_D}$ and higher $\overline{L/D}$ than max-$\overline{L/D}$ reward)",
        transform=ax.transAxes,
        va="top", ha="center", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="#f0f4ff", ec="#7090c0", lw=0.8, alpha=0.92),
    )

    # ── Bayesian Opt. callout — text in axes fraction to keep arrow short ─────
    if bo_ld_pt is not None:
        ax.annotate(
            "Bayesian Opt.: both designs in\nCorner A neighbourhood —\nCorner B trap escaped on both rewards",
            xy=bo_ld_pt,
            xytext=(0.97, 0.42),
            textcoords="axes fraction",
            fontsize=8, color=bo_color,
            ha="right", va="center",
            arrowprops=dict(arrowstyle="-", color=bo_color, lw=0.9,
                            connectionstyle="arc3,rad=0.15"),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=bo_color, lw=0.7, alpha=0.9),
            zorder=6,
        )

    # ── Axes and labels ───────────────────────────────────────────────────────
    ax.set_xlabel(r"$\overline{C_D}$")
    ax.set_ylabel(r"$\overline{L/D}$")
    ax.set_title(
        r"BlendedNet (BWB) — Best design: $\overline{C_D}$ vs $\overline{L/D}$",
        fontweight="medium", pad=8,
    )
    ax.grid(True, alpha=0.25)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    # ── Legend ────────────────────────────────────────────────────────────────
    color_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COLORS[n], markeredgecolor=COLORS[n],
               markersize=9, label=n)
        for n in ANOMALY_METHODS
    ] + [
        Line2D([0], [0], marker="D", color="w",
               markerfacecolor=bo_color, markeredgecolor=bo_color,
               markersize=9, label=BO_METHOD),
    ]
    style_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="grey", markeredgecolor="grey",
               markersize=9, label=r"min-$\overline{C_D}$ reward (filled)"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="none", markeredgecolor="grey",
               markersize=9, label=r"max-$\overline{L/D}$ reward (open)"),
        Line2D([0], [0], marker="D", color="w",
               markerfacecolor="grey", markeredgecolor="grey",
               markersize=9, label=r"Bayesian Opt. $\diamondsuit$"),
    ]
    leg1 = ax.legend(handles=color_handles, loc="upper right", fontsize=9,
                     framealpha=0.95, title="Method")
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc="lower left", fontsize=9,
              framealpha=0.95, title="Marker key")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_png = os.path.join(OUT_DIR, "BlendedNet_cd_ld_scatter.png")
    out_pdf = os.path.join(OUT_DIR, "BlendedNet_cd_ld_scatter.pdf")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")

    print("\n{:<20} {:>12} {:>10}   {:>12} {:>10}".format(
        "Method", "CD(minCD)", "LD(minCD)", "CD(maxLD)", "LD(maxLD)"))
    print("-" * 72)
    for name in METHOD_ORDER:
        res_cd, res_ld = data[name]
        cd1, ld1 = _cross_metrics(res_cd) if res_cd else (None, None)
        cd2, ld2 = _cross_metrics(res_ld) if res_ld else (None, None)
        print("{:<20} {:>12} {:>10}   {:>12} {:>10}".format(
            name,
            f"{cd1:.5f}" if cd1 is not None else "n/a",
            f"{ld1:.2f}"  if ld1 is not None else "n/a",
            f"{cd2:.5f}" if cd2 is not None else "n/a",
            f"{ld2:.2f}"  if ld2 is not None else "n/a",
        ))


if __name__ == "__main__":
    main()
