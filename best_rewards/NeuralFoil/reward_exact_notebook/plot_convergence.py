"""
Convergence plots for NeuralFoil multipoint trajectory data.
Generates both dark-mode and academic-style versions.
"""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Method definitions ─────────────────────────────────────────────────────────
METHODS = [
    {
        "key":   "v3_dynamic_optimizer",
        "label": "SE v3 (flash-2.5)",
        "csv":   os.path.join(BASE, "v3_dynamic_optimizer/trajectory_fitness_total.csv"),
        "color_dark":   "#FF6B35",
        "color_light":  "#E05C1A",
        "lw": 2.2,
        "zorder": 5,
    },
    {
        "key":   "lbfgsb",
        "label": "L-BFGS-B",
        "csv":   os.path.join(BASE, "lbfgsb/trajectory_fitness_total.csv"),
        "color_dark":   "#00D4FF",
        "color_light":  "#0077BB",
        "lw": 1.8,
        "zorder": 4,
    },
    {
        "key":   "BO_torch",
        "label": "Bayesian Opt.",
        "csv":   os.path.join(BASE, "BO_torch/trajectory_fitness_total.csv"),
        "color_dark":   "#88DD44",
        "color_light":  "#228B22",
        "lw": 1.8,
        "zorder": 3,
    },
    {
        "key":   "PSO",
        "label": "PSO (120 particles)",
        "csv":   os.path.join(BASE, "PSO/trajectory_fitness_total.csv"),
        "color_dark":   "#FF69B4",
        "color_light":  "#CC1177",
        "lw": 1.8,
        "zorder": 2,
    },
]


def load_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    data = {k: np.array([float(r[k]) for r in rows])
            for k in rows[0].keys()}
    return data


def load_all():
    out = {}
    for m in METHODS:
        out[m["key"]] = load_csv(m["csv"])
    return out


# ── Dark-mode plot ─────────────────────────────────────────────────────────────
def plot_dark(all_data, out_path):
    BG   = "#0D1117"
    GRID = "#1E2A38"
    FG   = "#C9D1D9"

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG)
    ax.set_facecolor(BG)

    for m in METHODS:
        d = all_data[m["key"]]
        evs  = d["eval"]
        col  = m["color_dark"]
        lw   = m["lw"]
        zo   = m["zorder"]

        # IQR shading
        ax.fill_between(evs, d["p25_best"], d["p75_best"],
                        alpha=0.13, color=col, zorder=zo - 1)
        # median
        ax.plot(evs, d["median_best"], color=col, lw=lw * 0.7,
                linestyle="--", alpha=0.7, zorder=zo)
        # best (min)
        ax.plot(evs, d["min_best"], color=col, lw=lw,
                label=m["label"], zorder=zo)

        # endpoint dot + value annotation
        best_val = float(d["min_best"][-1])
        ax.scatter(evs[-1], best_val, color=col, s=40, zorder=zo + 2,
                   edgecolors="white", linewidths=0.5)

    # Best overall reference line
    global_best = min(float(all_data[m["key"]]["min_best"].min()) for m in METHODS)
    ax.axhline(global_best, color="#AAAAAA", lw=0.8, linestyle=":",
               zorder=1, alpha=0.6)
    ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] > 0 else 1,
            global_best * 0.93, f"  best = {global_best:.4f}",
            color="#AAAAAA", fontsize=7.5, va="top", fontfamily="monospace")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=max(1, min(float(all_data[m["key"]]["eval"][1]) for m in METHODS)))

    ax.set_xlabel("Function Evaluations (simulations)", color=FG, fontsize=11,
                  fontfamily="monospace")
    ax.set_ylabel("|Reward|  (lower = better)", color=FG, fontsize=11,
                  fontfamily="monospace")
    ax.set_title(
        "ShapeEvolve · Convergence Comparison\nNeuralFoil Multipoint — best / median / IQR across seeds",
        color=FG, fontsize=13, fontfamily="monospace", pad=14,
    )

    ax.tick_params(colors=FG, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(True, which="both", color=GRID, linewidth=0.5, linestyle="-")

    # Legend: solid=best, dashed=median, shaded=IQR
    method_handles = [
        Line2D([0], [0], color=m["color_dark"], lw=m["lw"], label=m["label"])
        for m in METHODS
    ]
    style_handles = [
        Line2D([0], [0], color="white", lw=1.5,            label="best (min seed)"),
        Line2D([0], [0], color="white", lw=1.0, ls="--",   label="median seed"),
        matplotlib.patches.Patch(facecolor="white", alpha=0.2, label="IQR (p25–p75)"),
    ]
    leg = ax.legend(
        handles=method_handles + style_handles,
        loc="upper right", fontsize=8.5, framealpha=0.25,
        facecolor=GRID, edgecolor="#334455", labelcolor=FG,
        prop={"family": "monospace"},
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Academic (light) plot ──────────────────────────────────────────────────────
def plot_academic(all_data, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    for m in METHODS:
        d = all_data[m["key"]]
        evs = d["eval"]
        col = m["color_light"]
        lw  = m["lw"]
        zo  = m["zorder"]

        # IQR shading
        ax.fill_between(evs, d["p25_best"], d["p75_best"],
                        alpha=0.12, color=col, zorder=zo - 1)
        # median
        ax.plot(evs, d["median_best"], color=col, lw=lw * 0.7,
                linestyle="--", alpha=0.75, zorder=zo)
        # best
        ax.plot(evs, d["min_best"], color=col, lw=lw,
                label=m["label"], zorder=zo)

        # endpoint marker
        ax.scatter(evs[-1], float(d["min_best"][-1]),
                   color=col, s=35, zorder=zo + 2, edgecolors="black", lw=0.5)

    # Best overall reference line
    global_best = min(float(all_data[m["key"]]["min_best"].min()) for m in METHODS)
    ax.axhline(global_best, color="black", lw=0.8, linestyle=":",
               alpha=0.5, zorder=1)
    ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 60000,
            global_best * 0.94, f"  best observed = {global_best:.4f}",
            color="black", fontsize=7.5, ha="right", va="top")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Function Evaluations", fontsize=12)
    ax.set_ylabel("Best-so-far |reward|", fontsize=12)
    ax.set_title("NeuralFoil Multipoint — Method Convergence Comparison", fontsize=13)

    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    method_handles = [
        Line2D([0], [0], color=m["color_light"], lw=m["lw"], label=m["label"])
        for m in METHODS
    ]
    style_handles = [
        Line2D([0], [0], color="gray", lw=1.5,            label="best (min seed)"),
        Line2D([0], [0], color="gray", lw=1.0, ls="--",   label="median seed"),
        matplotlib.patches.Patch(facecolor="gray", alpha=0.2, label="IQR (p25–p75)"),
    ]
    ax.legend(handles=method_handles + style_handles,
              loc="upper right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Per-method zoom plots (eval budget ≤ 5000) ─────────────────────────────────
def plot_early_dark(all_data, out_path, max_eval=5000):
    """Dark-mode zoom on the first `max_eval` evals — useful for comparing methods at equal budget."""
    BG   = "#0D1117"
    GRID = "#1E2A38"
    FG   = "#C9D1D9"

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG)
    ax.set_facecolor(BG)

    for m in METHODS:
        d   = all_data[m["key"]]
        mask = d["eval"] <= max_eval
        if mask.sum() < 2:
            continue
        evs = d["eval"][mask]
        col = m["color_dark"]
        lw  = m["lw"]
        zo  = m["zorder"]

        ax.fill_between(evs, d["p25_best"][mask], d["p75_best"][mask],
                        alpha=0.13, color=col, zorder=zo - 1)
        ax.plot(evs, d["median_best"][mask], color=col, lw=lw * 0.7,
                linestyle="--", alpha=0.7, zorder=zo)
        ax.plot(evs, d["min_best"][mask], color=col, lw=lw,
                label=m["label"], zorder=zo)

        # mark value at exactly max_eval (last point ≤ max_eval)
        ax.scatter(evs[-1], float(d["min_best"][mask][-1]),
                   color=col, s=40, zorder=zo + 2,
                   edgecolors="white", linewidths=0.5)

    global_best = min(float(all_data[m["key"]]["min_best"].min()) for m in METHODS)
    ax.axhline(global_best, color="#AAAAAA", lw=0.8, linestyle=":", zorder=1, alpha=0.6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=1)

    ax.set_xlabel(f"Function Evaluations  (first {max_eval:,})", color=FG,
                  fontsize=11, fontfamily="monospace")
    ax.set_ylabel("|Reward|  (lower = better)", color=FG, fontsize=11,
                  fontfamily="monospace")
    ax.set_title(
        f"ShapeEvolve · Convergence  (≤ {max_eval:,} evals)\nNeuralFoil Multipoint — equal-budget comparison",
        color=FG, fontsize=13, fontfamily="monospace", pad=14,
    )

    ax.tick_params(colors=FG, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(True, which="both", color=GRID, linewidth=0.5)

    handles = [
        Line2D([0], [0], color=m["color_dark"], lw=m["lw"], label=m["label"])
        for m in METHODS
    ]
    handles += [
        Line2D([0], [0], color="white", lw=1.5,           label="best (min seed)"),
        Line2D([0], [0], color="white", lw=1.0, ls="--",  label="median seed"),
        matplotlib.patches.Patch(facecolor="white", alpha=0.2, label="IQR (p25–p75)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8.5, framealpha=0.25,
              facecolor=GRID, edgecolor="#334455", labelcolor=FG,
              prop={"family": "monospace"})

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    import matplotlib.patches
    all_data = load_all()

    plot_dark(all_data,
              os.path.join(BASE, "convergence_dark.png"))
    plot_academic(all_data,
                  os.path.join(BASE, "convergence_academic.png"))
    plot_early_dark(all_data,
                    os.path.join(BASE, "convergence_dark_5k.png"),
                    max_eval=5000)

    print("\nSummary (at final eval):")
    print(f"{'Method':<25} {'n_seeds':>8} {'best':>10} {'median':>10}")
    print("-" * 57)
    for m in METHODS:
        d = all_data[m["key"]]
        print(f"{m['label']:<25} {int(max(d['n_active'])):>8} "
              f"{float(d['min_best'].min()):>10.5f} "
              f"{float(d['median_best'][d['n_active']==max(d['n_active'])][-1]):>10.5f}")
