#!/usr/bin/env python3
"""
Overlay best airfoil designs from multiple methods on a single figure.

Required:
  --v3      PATH   Path to a v3 design's save/results.json
  --adjoint DIR    Adjoint result directory (loads save/results.json automatically)

Optional per-method overrides (each takes a path to design.json or save/results.json):
  --ga      PATH
  --pso     PATH
  --bo      PATH
  --lbfgsb  PATH

Additional arbitrary methods:
  --extra-labels  LABEL [LABEL ...]
  --extra-designs PATH  [PATH  ...]
  --extra-colors  COLOR [COLOR ...]

Usage (minimal):
    python analysis/plot_airfoil_comparison.py \\
        --v3      environments/NeuralFoil/results/run_v3_dynamic_optimizer_reward_exact_notebook_COMPLETE_FOR_PLOT/iter_0_s0/save/results.json \\
        --adjoint environments/NeuralFoil/results/adjoint_run/ \\
        --output  environments/NeuralFoil/results/airfoil_comparison.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import aerosandbox as asb


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

# Default colors per named method
DEFAULT_COLORS = {
    "v3":     "#2ca02c",
    "adjoint": "#333333",
    "ga":     "#1f77b4",
    "pso":    "#ff7f0e",
    "bo":     "#9467bd",
    "lbfgsb": "#e377c2",
}

DEFAULT_LABELS = {
    "v3":     "ShapeEvolve",
    "adjoint": "Adjoint (IPOPT)",
    "ga":     "GA",
    "pso":    "PSO",
    "bo":     "Bayesian Opt.",
    "lbfgsb": "L-BFGS-B",
}


def load_kulfan(path):
    """Load Kulfan parameters from a design.json or save/results.json.

    Handles two layouts:
      - Flat (adjoint results.json): keys upper_weights, lower_weights,
        leading_edge_weight at top level.
      - Nested (v3/GA/PSO results.json): same keys under a "design" sub-dict.
    """
    with open(path) as f:
        d = json.load(f)
    params = d.get("design", d)  # unwrap "design" wrapper if present
    return {
        "upper_weights":     np.array(params["upper_weights"], dtype=float),
        "lower_weights":     np.array(params["lower_weights"], dtype=float),
        "leading_edge_weight": float(params["leading_edge_weight"]),
    }


def build_airfoil(kulfan):
    return asb.KulfanAirfoil(
        upper_weights=kulfan["upper_weights"],
        lower_weights=kulfan["lower_weights"],
        leading_edge_weight=kulfan["leading_edge_weight"],
        TE_thickness=0.0,
    )


def plot_airfoils(entries, output_path, title=None):
    """entries: list of (label, color, path)"""
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="white")

    for label, color, path in entries:
        kulfan = load_kulfan(path)
        airfoil = build_airfoil(kulfan)
        coords = np.array(airfoil.coordinates)
        coords = np.vstack([coords, [[1.0, 0.0]]])  # force sharp TE closure
        ax.plot(coords[:, 0], coords[:, 1], color=color, lw=1.5, label=label)

    ax.set_aspect("equal")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title(title or "Best Airfoil Design — Method Comparison", fontweight="medium", pad=8)
    ax.legend(loc="upper right", framealpha=0.95)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    # Expand top y-limit only if the legend overlaps any curve
    for _ in range(10):
        fig.canvas.draw()
        leg_disp = ax.get_legend().get_window_extent()
        inv = ax.transData.inverted()
        corners = inv.transform([[leg_disp.x0, leg_disp.y0], [leg_disp.x1, leg_disp.y1]])
        lx0, ly0, lx1, ly1 = corners[0, 0], corners[0, 1], corners[1, 0], corners[1, 1]
        overlap = any(
            np.any((line.get_xdata() >= lx0) & (line.get_xdata() <= lx1) &
                   (line.get_ydata() >= ly0) & (line.get_ydata() <= ly1))
            for line in ax.get_lines()
        )
        if not overlap:
            break
        y_lo, y_hi = ax.get_ylim()
        ax.set_ylim(y_lo, y_hi + (y_hi - y_lo) * 0.2)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    base, _ = os.path.splitext(output_path)
    for ext in (".png", ".pdf"):
        path = base + ext
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"[airfoil_comparison] Plot -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Named methods
    parser.add_argument("--v3",      default=None, metavar="PATH",
                        help="v3 LLM best design (save/results.json)")
    parser.add_argument("--adjoint", default=None, metavar="DIR",
                        help="Adjoint result directory (loads save/results.json)")
    parser.add_argument("--ga",      default=None, metavar="PATH",
                        help="GA best design (design.json or save/results.json)")
    parser.add_argument("--pso",     default=None, metavar="PATH",
                        help="PSO best design (design.json or save/results.json)")
    parser.add_argument("--bo",      default=None, metavar="PATH",
                        help="Bayesian Opt. best design")
    parser.add_argument("--lbfgsb",  default=None, metavar="PATH",
                        help="L-BFGS-B best design")

    # Color/label overrides for named methods
    parser.add_argument("--v3-label",      default=None)
    parser.add_argument("--adjoint-label", default=None)
    parser.add_argument("--ga-label",      default=None)
    parser.add_argument("--pso-label",     default=None)
    parser.add_argument("--bo-label",      default=None)
    parser.add_argument("--lbfgsb-label",  default=None)

    parser.add_argument("--v3-color",      default=None)
    parser.add_argument("--adjoint-color", default=None)
    parser.add_argument("--ga-color",      default=None)
    parser.add_argument("--pso-color",     default=None)
    parser.add_argument("--bo-color",      default=None)
    parser.add_argument("--lbfgsb-color",  default=None)

    # Extra arbitrary methods
    parser.add_argument("--extra-labels",  nargs="+", default=[], metavar="LABEL")
    parser.add_argument("--extra-designs", nargs="+", default=[], metavar="PATH")
    parser.add_argument("--extra-colors",  nargs="+", default=[], metavar="COLOR")

    parser.add_argument("--output", default="airfoil_comparison.png",
                        help="Output PNG path")
    parser.add_argument("--title", default=None,
                        help="Plot title override")

    args = parser.parse_args()

    entries = []

    # Named methods in a fixed display order (adjoint last so it's on top)
    named = [
        ("lbfgsb", args.lbfgsb),
        ("bo",     args.bo),
        ("pso",    args.pso),
        ("v3",     args.v3),
        ("ga",     args.ga),
    ]
    for key, path in named:
        if path is None:
            continue
        label = getattr(args, f"{key}_label") or DEFAULT_LABELS[key]
        color = getattr(args, f"{key}_color") or DEFAULT_COLORS[key]
        entries.append((label, color, path))

    # Adjoint: resolve path from directory
    if args.adjoint is not None:
        adj_path = os.path.join(args.adjoint, "save", "results.json")
        if not os.path.exists(adj_path):
            parser.error(f"Adjoint results.json not found at {adj_path}")
        label = args.adjoint_label or DEFAULT_LABELS["adjoint"]
        color = args.adjoint_color or DEFAULT_COLORS["adjoint"]
        entries.append((label, color, adj_path))

    # Extra methods
    if len(args.extra_labels) != len(args.extra_designs):
        parser.error("--extra-labels and --extra-designs must have the same number of entries")
    tab10_extra = ["#8c564b", "#17becf", "#bcbd22", "#7f7f7f"]
    for i, (lbl, pth) in enumerate(zip(args.extra_labels, args.extra_designs)):
        color = args.extra_colors[i] if i < len(args.extra_colors) else tab10_extra[i % len(tab10_extra)]
        entries.append((lbl, color, pth))

    if not entries:
        parser.error("No methods specified. Provide at least --v3 or --adjoint.")

    plot_airfoils(entries, args.output, title=args.title)
