"""PCA of the 9D planform parameter space explored by all methods.

Two side-by-side scatter plots of PC1 vs PC2:
  Left  — coloured by method (legend order: L-BFGS-B → BO → GA/PSO → ShapeEvolve)
  Right — coloured by mean CD (viridis_r; lower = better)
Star markers indicate the median-best design from each method.

Usage:
    cd /scratch/ShapeEvolve
    source venv/bin/activate
    python analysis/BlendedNet/plot_blendednet_pca.py

Outputs:
    environments/BlendedNet/results/analysis_plots/pca_designs.png/.pdf
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
from sklearn.decomposition import PCA

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
         sorted(glob.glob(os.path.join(RESULTS_DIR, "run_BO_torch_shapebench_5_seed*_n500"))) +
         sorted(glob.glob(os.path.join(RESULTS_DIR, "run_BO_torch_shapebench_5_seed*_n1000"))),
         "reward", "bo"),
        ("PSO (20p × 200i)", [
            *sorted(glob.glob(os.path.join(RESULTS_DIR, "run_GA_shapebench_5_seed*_20p_200i"))),
            *sorted(glob.glob(os.path.join(RESULTS_DIR, "run_GA_shapebench_5_attempt*_20p_200i"))),
        ], "reward", "ga"),
        ("ShapeEvolve",
         sorted(glob.glob(os.path.join(RESULTS_DIR, "run_v3_flash2_5_shapebench_5_attempt_*_n2000"))),
         "reward", "v3"),
        ("CMA-ES",
         sorted(glob.glob(os.path.join(RESULTS_DIR, "run_cmaes_shapebench_5_seed*_n500to1000"))),
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    plt.rcParams.update(STYLE)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading designs...")
    dataset = load_all_designs()

    all_records = []
    method_labels = []
    for name in METHOD_ORDER:
        records = dataset.get(name, [])
        all_records.extend(records)
        method_labels.extend([name] * len(records))

    if not all_records:
        print("No data found.")
        return

    X = np.array([[r[k] for k in GEOM_KEYS] for r in all_records])
    cds = np.array([r["mean_CD"] for r in all_records])

    lo = np.array([BOUNDS[k][0] for k in GEOM_KEYS])
    hi = np.array([BOUNDS[k][1] for k in GEOM_KEYS])
    X_norm = (X - lo) / (hi - lo)

    pca = PCA(n_components=2)
    Z = pca.fit_transform(X_norm)
    var_exp = pca.explained_variance_ratio_
    print(f"\nPCA variance explained: PC1={var_exp[0]:.1%}  PC2={var_exp[1]:.1%}")

    # Median-best design per method (design closest to median CD for that method)
    median_z = {}
    for name in METHOD_ORDER:
        idx_m = [i for i, m in enumerate(method_labels) if m == name]
        if not idx_m:
            continue
        cds_m = cds[idx_m]
        target = float(np.median(cds_m))
        best_local = idx_m[int(np.argmin(np.abs(cds_m - target)))]
        median_z[name] = Z[best_local]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white",
                              gridspec_kw={"wspace": 0.3})

    # ── Left: coloured by method ──────────────────────────────────────
    ax = axes[0]
    for name in METHOD_ORDER:
        mask = np.array([m == name for m in method_labels])
        if mask.any():
            ax.scatter(Z[mask, 0], Z[mask, 1], c=COLORS[name], s=8, alpha=0.4,
                       linewidths=0)
    for name, z in median_z.items():
        ax.scatter(z[0], z[1], c=COLORS[name], s=160, marker="*",
                   edgecolors="k", linewidths=0.6, zorder=6)

    legend_handles = [
        Patch(facecolor=COLORS[n], label=n)
        for n in METHOD_ORDER if dataset.get(n)
    ]
    legend_handles.append(
        plt.scatter([], [], c="grey", s=160, marker="*", edgecolors="k",
                    linewidths=0.6, label="Median-best design")
    )
    ax.legend(handles=legend_handles, fontsize=8.5, loc="best", framealpha=0.9,
              title="Method")
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1%} variance)")
    ax.set_title("Coloured by method", fontweight="medium")
    ax.grid(True, alpha=0.25)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    # ── Right: coloured by mean CD ────────────────────────────────────
    ax = axes[1]
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=cds, cmap="viridis_r",
                    s=8, alpha=0.5, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("mean CD (lower is better)", fontsize=9)
    for name, z in median_z.items():
        ax.scatter(z[0], z[1], c=COLORS[name], s=160, marker="*",
                   edgecolors="k", linewidths=0.6, zorder=6)

    ax.set_xlabel(f"PC1 ({var_exp[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1%} variance)")
    ax.set_title("Coloured by mean CD", fontweight="medium")
    ax.grid(True, alpha=0.25)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    loading_txt = "  ".join(
        f"{k}: [{pca.components_[0, i]:+.2f}, {pca.components_[1, i]:+.2f}]"
        for i, k in enumerate(GEOM_KEYS)
    )
    fig.text(0.5, -0.02, f"PCA loadings [PC1, PC2]:  {loading_txt}",
             ha="center", fontsize=7, color="#555555")

    fig.suptitle(
        "BlendedNet (BWB) — PCA of 9D Planform Parameter Space\n"
        f"n={len(all_records)} designs  |  stars = median-best design per method  |  "
        f"normalised to [0,1] before PCA",
        fontsize=11, fontweight="medium", y=1.01,
    )

    out_png = os.path.join(OUT_DIR, "BlendedNet_pca_designs.png")
    out_pdf = os.path.join(OUT_DIR, "BlendedNet_pca_designs.pdf")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
