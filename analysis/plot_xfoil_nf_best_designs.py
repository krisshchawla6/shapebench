#!/usr/bin/env python3
"""
Bar chart comparing NeuralFoil vs XFOIL weighted_CD for the best design from
each method on the reward_exact_notebook multipoint task.

Primary axis  : grouped bars — NF wCD (filled) and XFOIL wCD (hatched)
Secondary axis: % error = 100 * (XFOIL - NF) / NF  (dot + dashed line)

Methods are sorted by XFOIL wCD (best on the left).
Adjoint has no NF bar (NF failed for that design) and no % error point.

Usage:
    python analysis/plot_xfoil_nf_best_designs.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


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

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(
    REPO_DIR, "environments", "NeuralFoil", "results",
    "xfoil_evaluation_reward_exact_notebook"
)
JSON_PATH = os.path.join(RESULTS_DIR, "xfoil_multipoint_best_results.json")
OUT_PATH  = os.path.join(RESULTS_DIR, "NeuralFoil_multipoint_xfoil_vs_nf_fidelity.png")

# Color per method (consistent with plot_combined_methods.py palette)
METHOD_COLORS = {
    "L-BFGS-B":              "#e377c2",
    "Bayesian Opt. (exact GP)": "#ff7f0e",
    "PSO (120p × 500i)":     "#1f77b4",
    "ShapeEvolve":           "#2ca02c",
    "Adjoint (IPOPT)":       "#7f7f7f",
}

# Shorter x-tick labels
SHORT_LABELS = {
    "L-BFGS-B":              "L-BFGS-B",
    "Bayesian Opt. (exact GP)": "Bayes. Opt.\n(exact GP)",
    "PSO (120p × 500i)":     "PSO\n(120p×500i)",
    "ShapeEvolve":           "ShapeEvolve",
    "Adjoint (IPOPT)":       "Adjoint\n(IPOPT)",
}


def main():
    with open(JSON_PATH) as f:
        results = json.load(f)

    # Sort by XFOIL wCD ascending (best first); None → push to end
    results.sort(key=lambda r: r["xfoil"]["weighted_CD"] or float("inf"))

    names   = [r["name"] for r in results]
    xf_wcd  = [r["xfoil"]["weighted_CD"] for r in results]
    nf_wcd  = [r["neuralfoil"]["weighted_CD"] for r in results]   # None for Adjoint

    # % error only where both values exist
    pct_err = []
    for nf, xf in zip(nf_wcd, xf_wcd):
        if nf is not None and xf is not None:
            pct_err.append(100.0 * (xf - nf) / nf)
        else:
            pct_err.append(None)

    n = len(names)
    x = np.arange(n)
    bar_w = 0.32

    with plt.rc_context(STYLE):
        fig, ax1 = plt.subplots(figsize=(7.0, 4.2))
        ax2 = ax1.twinx()

        # ── bars ──────────────────────────────────────────────────────────────
        for i, name in enumerate(names):
            color = METHOD_COLORS.get(name, "#333333")
            short = SHORT_LABELS.get(name, name)

            # XFOIL bar (hatched, slightly right)
            ax1.bar(x[i] + bar_w / 2, xf_wcd[i], width=bar_w,
                    color=color, alpha=0.55, hatch="///", edgecolor=color,
                    linewidth=0.7, zorder=3)

            # NF bar (solid, slightly left) — skip if NF failed
            if nf_wcd[i] is not None:
                ax1.bar(x[i] - bar_w / 2, nf_wcd[i], width=bar_w,
                        color=color, alpha=0.90, edgecolor=color,
                        linewidth=0.7, zorder=3)

        # ── % error on secondary axis ─────────────────────────────────────────
        err_x  = [x[i] for i, e in enumerate(pct_err) if e is not None]
        err_y  = [e    for e in pct_err if e is not None]
        ax2.plot(err_x, err_y, color="#d62728", marker="o", markersize=5,
                 linewidth=1.2, linestyle="--", zorder=5, label="% error (XFOIL−NF)/NF")
        ax2.axhline(0, color="#d62728", linewidth=0.5, linestyle=":", alpha=0.6)

        # annotate % error values
        for xi, yi in zip(err_x, err_y):
            ax2.annotate(f"{yi:+.2f}%",
                         xy=(xi, yi),
                         xytext=(4, 5), textcoords="offset points",
                         fontsize=7.5, color="#d62728")

        # ── axes formatting ───────────────────────────────────────────────────
        ax1.set_xticks(x)
        ax1.set_xticklabels([SHORT_LABELS.get(n, n) for n in names],
                            fontsize=9, linespacing=1.3)
        ax1.set_ylabel(r"Weighted $\overline{C_D}$  (lower is better)")
        ax1.set_xlim(-0.55, n - 0.45)
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))

        # y-range: a bit below the best XFOIL bar, above the worst
        ymin = min(v for v in xf_wcd + nf_wcd if v is not None) * 0.985
        ymax = max(v for v in xf_wcd + nf_wcd if v is not None) * 1.015
        ax1.set_ylim(ymin, ymax)

        ax2.set_ylabel("XFOIL vs NeuralFoil error  (%)", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%+.1f%%"))

        # keep secondary axis symmetric and readable
        max_abs_err = max(abs(e) for e in err_y) if err_y else 3.0
        ax2.set_ylim(-max_abs_err * 1.8, max_abs_err * 2.2)

        # ── legend ────────────────────────────────────────────────────────────
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor="#888888", alpha=0.90, edgecolor="#555", linewidth=0.7,
                  label="NeuralFoil wCD"),
            Patch(facecolor="#888888", alpha=0.55, hatch="///", edgecolor="#555",
                  linewidth=0.7, label="XFOIL wCD"),
            plt.Line2D([0], [0], color="#d62728", marker="o", markersize=5,
                       linewidth=1.2, linestyle="--",
                       label=r"% error $\frac{\mathrm{XFOIL}-\mathrm{NF}}{\mathrm{NF}}$"),
        ]
        ax1.legend(handles=legend_handles, loc="upper left", framealpha=0.85,
                   edgecolor="#cccccc", fontsize=8)

        ax1.set_title(
            "Best-design XFOIL vs NeuralFoil evaluation — multipoint $C_L$ targets\n"
            r"($C_L \in \{0.8,\,1.0,\,1.2,\,1.4,\,1.5,\,1.6\}$, "
            r"weighted $\overline{C_D} = \mathrm{mean}(C_{D,i}\,w_i)$)",
            fontsize=9.5, pad=6,
        )

        # annotate any method where NF data is missing
        for i, nf in enumerate(nf_wcd):
            if nf is None:
                ax1.annotate("NF\nfailed", xy=(x[i] - bar_w / 2, ymin + 0.0002),
                             ha="center", va="bottom", fontsize=7, color="#666666",
                             style="italic")

        ax1.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0)
        fig.tight_layout()
        fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
        print(f"Saved to: {OUT_PATH}")
        pdf_path = OUT_PATH.replace(".png", ".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved to: {pdf_path}")


if __name__ == "__main__":
    main()
