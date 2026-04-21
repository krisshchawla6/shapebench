"""Parameter scatter: each of the 9 planform parameters vs mean CD, coloured by method.

Samples up to MAX_PER_RUN evaluations per run directory to keep loading fast.
The median-best design per method is highlighted with a star marker.
Method legend order: L-BFGS-B → BO → GA/PSO → ShapeEvolve (v3).

Usage:
    cd /scratch/ShapeEvolve
    source venv/bin/activate
    python analysis/BlendedNet/plot_blendednet_param_scatter.py

Outputs:
    environments/BlendedNet/results/analysis_plots/param_scatter.png/.pdf
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

METHOD_ORDER = ["L-BFGS-B", "Bayesian Opt. (exact GP)", "PSO (20p × 200i)", "ShapeEvolve", "CMA-ES"]

GEOM_KEYS = ["B1", "B2", "B3", "C2", "C3", "C4", "S1", "S2", "S3"]

BOUNDS = {
    "B1": (100, 200),  "B2": (50,  200),  "B3": (200, 700),
    "C2": (550, 850),  "C3": (180, 280),  "C4": (60,  90),
    "S1": (40,  60),   "S2": (40,  60),   "S3": (24,  40),
}

MAX_PER_RUN = 200


# ── Data loading ──────────────────────────────────────────────────────────────

def _design_dir(run_dir, row, method_key):
    if method_key == "lbfgsb":
        return os.path.join(run_dir, f"call_{int(row['call']):05d}_r{int(row['restart'])}")
    elif method_key == "ga":
        return os.path.join(run_dir, f"iter_{int(row['iteration']):04d}_p{int(row['particle']):03d}")
    else:
        return os.path.join(run_dir, str(row["design"]))


def _load_params(design_dir, method_key):
    dj = os.path.join(design_dir, "design.json")
    if os.path.exists(dj):
        with open(dj) as f:
            return json.load(f)
    name = os.path.basename(design_dir)
    nj = os.path.join(design_dir, f"{name}.json")
    if os.path.exists(nj):
        with open(nj) as f:
            d = json.load(f)
        if "B1" in d:
            return d
    rj = os.path.join(design_dir, "save", "results.json")
    if os.path.exists(rj):
        with open(rj) as f:
            d = json.load(f)
        return d.get("design")
    return None


def _load_run(run_dir, reward_col, method_key, max_n):
    csv = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv):
        return []
    try:
        df = pd.read_csv(csv)
    except Exception:
        return []
    if reward_col not in df.columns:
        return []
    n = len(df)
    if n == 0:
        return []
    step = max(1, n // max_n)
    records = []
    for _, row in df.iloc[::step].iterrows():
        ddir = _design_dir(run_dir, row, method_key)
        params = _load_params(ddir, method_key)
        if params is None or not all(k in params for k in GEOM_KEYS):
            continue
        records.append({
            "mean_CD": -float(row[reward_col]),
            **{k: float(params[k]) for k in GEOM_KEYS},
        })
    return records


def load_all_designs():
    specs = [
        ("L-BFGS-B",
         sorted(glob.glob(os.path.join(RESULTS_DIR, "run_lbfgsb_shapebench_5_seed*_nr3_normfix"))),
         "reward", "lbfgsb"),
        ("Bayesian Opt. (exact GP)",
         sorted(glob.glob(os.path.join(RESULTS_DIR, "run_BO_torch_shapebench_5_seed*_n500"))),
         "reward", "bo"),
        ("PSO (20p × 200i)", [
            *sorted(glob.glob(os.path.join(RESULTS_DIR, "run_GA_shapebench_5_seed*_20p_200i"))),
            *sorted(glob.glob(os.path.join(RESULTS_DIR, "run_GA_shapebench_5_attempt*_20p_200i"))),
        ], "reward", "ga"),
        ("ShapeEvolve",
         sorted(glob.glob(os.path.join(RESULTS_DIR, "run_v3_flash2_5_shapebench_5_attempt_*_n2000"))),
         "reward", "v3"),
        ("CMA-ES",
         sorted(glob.glob(os.path.join(RESULTS_DIR, "run_cmaes_shapebench_5_seed*_n500"))),
         "reward", "bo"),
    ]
    dataset = {}
    for name, dirs, reward_col, method_key in specs:
        records = []
        for run_dir in dirs:
            records.extend(_load_run(run_dir, reward_col, method_key, MAX_PER_RUN))
        dataset[name] = records
        print(f"  {name:<18}  {len(records):>5} designs")
    return dataset


def median_best_per_method(dataset):
    """For each method, find the design whose reward equals the run-median."""
    result = {}
    for name, records in dataset.items():
        if not records:
            continue
        # Median CD across designs (proxy for median-best run)
        cds = [r["mean_CD"] for r in records]
        target = float(np.median(cds))
        closest = min(records, key=lambda r: abs(r["mean_CD"] - target))
        result[name] = closest
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    plt.rcParams.update(STYLE)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading designs...")
    dataset = load_all_designs()
    median_designs = median_best_per_method(dataset)

    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5),
                              facecolor="white",
                              gridspec_kw={"hspace": 0.45, "wspace": 0.35})

    for idx, key in enumerate(GEOM_KEYS):
        ax = axes[idx // ncols][idx % ncols]
        lo, hi = BOUNDS[key]

        for name in METHOD_ORDER:
            records = dataset.get(name, [])
            if not records:
                continue
            xs = [r[key] for r in records]
            ys = [r["mean_CD"] for r in records]
            ax.scatter(xs, ys, c=COLORS[name], s=6, alpha=0.35, linewidths=0)

        for name in METHOD_ORDER:
            rec = median_designs.get(name)
            if rec is not None:
                ax.scatter(rec[key], rec["mean_CD"], c=COLORS[name],
                           s=120, marker="*", edgecolors="k", linewidths=0.5, zorder=5)

        ax.set_xlabel(key)
        ax.set_ylabel("mean CD" if idx % ncols == 0 else "")
        ax.set_xlim(lo - (hi - lo) * 0.03, hi + (hi - lo) * 0.03)
        ax.grid(True, alpha=0.25)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    legend_handles = [
        Patch(facecolor=COLORS[n], label=n) for n in METHOD_ORDER if dataset.get(n)
    ]
    legend_handles.append(
        plt.scatter([], [], c="grey", s=120, marker="*", edgecolors="k",
                    linewidths=0.5, label="Median-best design")
    )
    axes[0][0].legend(handles=legend_handles, fontsize=7.5, loc="upper right",
                      framealpha=0.9, ncol=1, title="Method")

    fig.suptitle(
        "BlendedNet (BWB) — Planform Parameter Scatter vs mean CD\n"
        "Stars = median-best design per method  |  dots = sampled evaluations",
        fontsize=11, fontweight="medium", y=1.01,
    )

    out_png = os.path.join(OUT_DIR, "param_scatter.png")
    out_pdf = os.path.join(OUT_DIR, "param_scatter.pdf")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
