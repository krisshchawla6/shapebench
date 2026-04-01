#!/usr/bin/env python3
"""
Combined method comparison plot.

Reads pre-computed trajectory CSVs (from plot_method_summary.py) for
multiple methods and overlays them on a single figure, with one legend
entry per method.  Visual style:
  - Solid line  = median best
  - Shaded band = min–max range
  - Dashed line = region where n_active has dropped (if n_active column present)
  - × marker    = each point where a run stopped (if n_active column present)
  - Horizontal dashed line = adjoint reference

Usage:
    python analysis/plot_combined_methods.py \\
        --labels "L-BFGS-B" "Bayesian Opt." "PSO (120p×500i)" "v3 LLM" \\
        --csvs  path/lbfgsb.csv  path/bo.csv  path/pso.csv  path/v3.csv \\
        --colors "#e377c2" "#ff7f0e" "#1f77b4" "#2ca02c" \\
        --adjoint environments/NeuralFoil/results/adjoint_run/ \\
        --adjoint-label "Adjoint / IPOPT" \\
        --output environments/NeuralFoil/results/combined_comparison.png \\
        --max-evals 6000
"""

import argparse
import csv as csv_mod
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


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
    "figure.dpi": 200,
}


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_trajectory(csv_path, max_evals=None):
    """Load a trajectory CSV produced by plot_method_summary.py.

    Returns a dict of arrays:
        eval, median_best, min_best, max_best
    and optionally n_active (None if column absent).
    """
    evals, median, min_b, max_b, n_active = [], [], [], [], []
    has_n_active = False

    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            ev = float(row["eval"])
            if max_evals is not None and ev > max_evals:
                break
            med = row.get("median_best", "")
            mn  = row.get("min_best", "")
            mx  = row.get("max_best", "")
            if med == "" or mn == "" or mx == "":
                continue  # NaN rows (past last active run) — skip
            evals.append(ev)
            median.append(float(med))
            min_b.append(float(mn))
            max_b.append(float(mx))
            na = row.get("n_active")
            if na is not None:
                has_n_active = True
                n_active.append(int(na))

    result = dict(
        eval=np.array(evals),
        median=np.maximum(np.array(median), 1e-9),
        min_b=np.maximum(np.array(min_b),  1e-9),
        max_b=np.maximum(np.array(max_b),  1e-9),
        n_active=np.array(n_active) if has_n_active else None,
    )
    return result


# ---------------------------------------------------------------------------
# Adjoint reference
# ---------------------------------------------------------------------------

def load_adjoint_reference(adjoint_dir):
    path = os.path.join(adjoint_dir, "save", "results.json")
    if not os.path.exists(path):
        print(f"[adjoint] results.json not found at {path}")
        return None, None
    with open(path) as f:
        d = json.load(f)
    weighted_cd = d.get("weighted_cd") or d.get("weighted_CD_mean")
    feasible = d.get("feasible", None)
    if weighted_cd is None:
        print(f"[adjoint] weighted_cd not found in {path}")
        return None, None
    return float(weighted_cd), feasible


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _draw_methods(ax, trajs, adjoint_ref=None, lw=1.8):
    """Draw all method trajectories (and optional adjoint line) onto ax."""
    for color, ev, med, mn, mx, solid_mask, dash_mask in trajs:
        ax.fill_between(ev[solid_mask], mn[solid_mask], mx[solid_mask],
                        color=color, alpha=0.18)
        ax.plot(ev[solid_mask], med[solid_mask], color=color, lw=lw)
        if dash_mask.any():
            ax.fill_between(ev[dash_mask], mn[dash_mask], mx[dash_mask],
                            color=color, alpha=0.07)
            ax.plot(ev[dash_mask], med[dash_mask],
                    color=color, lw=lw, linestyle="--")
    if adjoint_ref is not None:
        ax.axhline(adjoint_ref, color="#333333", lw=1.2, ls=":", zorder=4)


def plot_combined(labels, csv_paths, colors, adjoint_dir=None,
                  adjoint_label="Adjoint (IPOPT)", max_evals=None,
                  x_max=25000, y_min=None, y_max=None,
                  output_path="combined.png", title=None):

    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")

    all_vals = []
    method_legend = []
    trajs = []  # (color, ev, med, mn, mx, solid_mask, dash_mask)

    for label, csv_path, color in zip(labels, csv_paths, colors):
        traj = load_trajectory(csv_path, max_evals)
        ev   = traj["eval"]
        med  = traj["median"]
        mn   = traj["min_b"]
        mx   = traj["max_b"]
        na   = traj["n_active"]

        if len(ev) == 0:
            print(f"[warn] No data in {csv_path}, skipping.")
            continue

        # Shift all evals by +1 so the first point lands at x=1 on log scale
        # (log(0) is undefined and would be silently dropped by matplotlib).
        ev = ev + 1

        all_vals.extend(med.tolist())
        all_vals.extend(mn.tolist())
        all_vals.extend(mx.tolist())

        # Determine split point where n_active first drops (if available)
        split_ev = None
        if na is not None:
            max_na = na[0]
            drop = np.where(na < max_na)[0]
            if len(drop):
                split_ev = ev[drop[0]]

        solid_mask = ev <= split_ev if split_ev is not None else np.ones(len(ev), dtype=bool)
        dash_mask  = ev >= split_ev if split_ev is not None else np.zeros(len(ev), dtype=bool)

        trajs.append((color, ev, med, mn, mx, solid_mask, dash_mask))
        method_legend.append(Line2D([0], [0], color=color, lw=2.0, label=label))

    # Adjoint reference
    adjoint_ref = None
    if adjoint_dir is not None:
        weighted_cd, adj_feasible = load_adjoint_reference(adjoint_dir)
        if weighted_cd is not None:
            adjoint_ref = weighted_cd
            feas_str = " (feasible)" if adj_feasible else " (infeasible)"
            print(f"[adjoint] reference={adjoint_ref:.6f}{feas_str}")
            method_legend.append(
                Line2D([0], [0], color="#333333", lw=1.2, linestyle=":",
                       label=f"{adjoint_label}  (reward = −{adjoint_ref:.4f})")
            )

    # Draw main axes
    _draw_methods(ax, trajs, adjoint_ref=adjoint_ref)

    # Y limits (dynamic from data, overridden by --y-min/--y-max if provided)
    if all_vals:
        y_lo = max(min(all_vals) * 0.8, 1e-9)
        y_hi = max(all_vals) * 1.2
        if adjoint_ref is not None:
            y_lo = min(y_lo, adjoint_ref * 0.90)
    else:
        y_lo, y_hi = 1e-9, 1.0
    if y_min is not None:
        y_lo = y_min
    if y_max is not None:
        y_hi = y_max

    # Main axes: log x and log y
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, x_max * 1.02)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Model (Airfoil Design) Evaluations")
    ax.set_ylabel("Penalized Objective  (−reward, lower is better)")
    ax.set_title(title or "NeuralFoil Optimisation — Method Comparison",
                 fontweight="medium", pad=8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    # Legends
    style_legend = [
        Patch(facecolor="grey", alpha=0.25, label="Min–max range"),
        Line2D([0], [0], color="grey", lw=1.8, label="Median best"),
        Line2D([0], [0], color="grey", lw=1.8, linestyle="--",
               label="Failed/partial runs begin"),
    ]
    leg1 = ax.legend(handles=method_legend, loc="upper right",
                     framealpha=0.95, title="Method")
    ax.add_artist(leg1)
    ax.legend(handles=style_legend, loc="lower left",
              framealpha=0.95, title="Style key",
              bbox_to_anchor=(0.0, 0.1))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[combined] Plot -> {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Method labels (one per CSV)")
    parser.add_argument("--csvs", nargs="+", required=True,
                        help="Trajectory CSV paths (one per method, same order as --labels)")
    parser.add_argument("--colors", nargs="+", default=None,
                        help="Hex colours (one per method). Defaults to matplotlib tab10.")
    parser.add_argument("--adjoint", default=None, metavar="DIR",
                        help="Adjoint result directory (for reference line)")
    parser.add_argument("--adjoint-label", default="Adjoint (IPOPT)")
    parser.add_argument("--max-evals", type=int, default=None,
                        help="Truncate all trajectories at this eval count")
    parser.add_argument("--x-max", type=int, default=25000,
                        help="X-axis upper limit in evals (default: 25000)")
    parser.add_argument("--y-min", type=float, default=None,
                        help="Y-axis lower limit override (default: dynamic from data)")
    parser.add_argument("--y-max", type=float, default=None,
                        help="Y-axis upper limit override (default: dynamic from data)")
    parser.add_argument("--output", default="combined.png",
                        help="Output PNG path")
    parser.add_argument("--title", default=None,
                        help="Plot title override")
    args = parser.parse_args()

    if len(args.labels) != len(args.csvs):
        parser.error("--labels and --csvs must have the same number of entries")

    tab10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
             "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    colors = args.colors or tab10[:len(args.labels)]
    if len(colors) < len(args.labels):
        parser.error("--colors must have at least as many entries as --labels")

    plot_combined(
        labels=args.labels,
        csv_paths=args.csvs,
        colors=colors,
        adjoint_dir=args.adjoint,
        adjoint_label=args.adjoint_label,
        max_evals=args.max_evals,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        output_path=args.output,
        title=args.title,
    )
