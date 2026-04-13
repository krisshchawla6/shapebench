"""PCA scatter of all evaluated designs, coloured by Cd.

Collects all design files across all methods and runs, normalizes the 20 FFD
parameters to [-1, 1], projects to 2D with PCA, and produces:

  1. A scatter plot coloured by Cd (all methods together).
  2. A scatter coloured by method (to show how each method covers design space).

Designs are sampled uniformly per run to keep the plot readable when there are
thousands of evaluations (GA has 30k+).

Usage:
    cd /scratch/ShapeEvolve
    source venv/bin/activate
    python analysis/DrivAer_Star/plot_drivaer_pca_scatter.py

Outputs:
    environments/DrivAer_Star/results/analysis_plots_cd_only/pca_scatter_cd.png
    environments/DrivAer_Star/results/analysis_plots_cd_only/pca_scatter_method.png
"""

import os
import glob
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO_DIR, "environments", "DrivAer_Star", "results")
OUT_DIR = os.path.join(RESULTS_DIR, "analysis_plots_cd_only")

sys.path.insert(0, os.path.join(REPO_DIR, "environments", "DrivAer_Star"))
from mesh_generator import PARAM_KEYS, BOUNDS

# Max designs sampled per run (to keep GA from dominating)
MAX_PER_RUN = 300

COLORS = {
    "GA/PSO":         "#e07b39",
    "L-BFGS-B":       "#7b9e87",
    "BO_torch":       "#4a90d9",
    "v3 flash-2.5":   "#9b59b6",
}


def normalize_params(params):
    """Return normalized parameter array, or None if any value is non-finite."""
    result = []
    for k in PARAM_KEYS:
        lo, hi = BOUNDS[k]
        center = (lo + hi) / 2.0
        half_range = (hi - lo) / 2.0
        v = float(params.get(k, 0.0))
        result.append((v - center) / half_range)
    arr = np.array(result)
    if not np.isfinite(arr).all():
        return None
    return arr


def _sample_indices(n, max_n):
    if n <= max_n:
        return np.arange(n)
    return np.sort(np.random.choice(n, max_n, replace=False))


def load_ga_designs():
    base = os.path.join(RESULTS_DIR, "SAVED_DIRS_run_GA_cd_only_drivaer_star_120p_250i")
    X, cd_vals = [], []
    for run_dir in sorted(glob.glob(os.path.join(base, "run_GA_cd_only_*"))):
        csv = os.path.join(run_dir, "results.csv")
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv)
        indices = _sample_indices(len(df), MAX_PER_RUN)
        for idx in indices:
            row = df.iloc[idx]
            iteration = int(row["iteration"])
            particle = int(row["particle"])
            p = os.path.join(run_dir, f"iter_{iteration:04d}_p{particle:03d}", "design.json")
            if os.path.exists(p):
                with open(p) as f:
                    params = json.load(f)
                x = normalize_params(params)
                if x is not None:
                    X.append(x)
                    cd_vals.append(float(row["Cd"]))
    return np.array(X), np.array(cd_vals)


def load_lbfgsb_designs():
    base = os.path.join(RESULTS_DIR, "SAVED_DIRS_run_lbfgsb_cd_only_drivaer_star_nr3")
    X, cd_vals = [], []
    for run_dir in sorted(glob.glob(os.path.join(base, "run_lbfgsb_cd_only_*"))):
        csv = os.path.join(run_dir, "results.csv")
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv)
        indices = _sample_indices(len(df), MAX_PER_RUN)
        for idx in indices:
            row = df.iloc[idx]
            call = int(row["call"])
            restart = int(row["restart"])
            p = os.path.join(run_dir, f"call_{call:05d}_r{restart}", "design.json")
            if os.path.exists(p):
                with open(p) as f:
                    params = json.load(f)
                x = normalize_params(params)
                if x is not None:
                    X.append(x)
                    cd_vals.append(float(row["Cd"]))
    return np.array(X), np.array(cd_vals)


def load_bo_designs():
    X, cd_vals = [], []
    for run_dir in sorted(glob.glob(os.path.join(RESULTS_DIR, "run_BO_torch_cd_only_seed*_n1000"))):
        csv = os.path.join(run_dir, "results.csv")
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv)
        indices = _sample_indices(len(df), MAX_PER_RUN)
        for idx in indices:
            row = df.iloc[idx]
            design_name = str(row["design"])
            p = os.path.join(run_dir, design_name, f"{design_name}.json")
            if os.path.exists(p):
                with open(p) as f:
                    params = json.load(f)
                x = normalize_params(params)
                if x is not None:
                    X.append(x)
                    cd_vals.append(float(row["Cd"]))
    return np.array(X), np.array(cd_vals)


def load_v3_designs():
    X, cd_vals = [], []
    for pattern in [
        os.path.join(RESULTS_DIR, "SAVED_DIRS_run_v3_*", "run_v3_*"),
        os.path.join(RESULTS_DIR, "run_v3_*_n10000"),
    ]:
        for run_dir in sorted(glob.glob(pattern)):
            csv = os.path.join(run_dir, "results.csv")
            if not os.path.exists(csv):
                continue
            df = pd.read_csv(csv)
            indices = _sample_indices(len(df), MAX_PER_RUN)
            for idx in indices:
                row = df.iloc[idx]
                design_name = str(row["design"])
                p = os.path.join(run_dir, f"{design_name}.json")
                if os.path.exists(p):
                    with open(p) as f:
                        params = json.load(f)
                    x = normalize_params(params)
                    if x is not None:
                        X.append(x)
                        cd_vals.append(float(row["Cd"]))
    return np.array(X), np.array(cd_vals)


def pca_2d(X_all):
    """Fit PCA on X_all (n x 20).

    Uses the 20x20 covariance matrix (much smaller than n x 20) via
    scipy.linalg.eigh, which is robust even for large n.
    """
    from scipy.linalg import eigh
    mean = X_all.mean(axis=0)
    X_centered = X_all - mean
    cov = (X_centered.T @ X_centered) / (len(X_all) - 1)   # 20 x 20
    # eigh returns eigenvalues in ascending order
    eigvals, eigvecs = eigh(cov)
    # Sort descending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    components = eigvecs[:, :2]   # 20 x 2
    total_var = eigvals[eigvals > 0].sum()
    explained = eigvals[:2] / total_var * 100

    def transform(X):
        return (X - mean) @ components

    return transform, explained, components


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(42)

    print("Loading designs...")
    ga_X,   ga_cd   = load_ga_designs()
    lb_X,   lb_cd   = load_lbfgsb_designs()
    bo_X,   bo_cd   = load_bo_designs()
    v3_X,   v3_cd   = load_v3_designs()

    datasets = [
        ("GA/PSO",       ga_X,  ga_cd),
        ("L-BFGS-B",     lb_X,  lb_cd),
        ("BO_torch",     bo_X,  bo_cd),
        ("v3 flash-2.5", v3_X,  v3_cd),
    ]
    datasets = [(n, X, cd) for n, X, cd in datasets if len(X) > 0]

    X_all = np.vstack([d[1] for d in datasets])
    cd_all = np.concatenate([d[2] for d in datasets])

    print(f"Total designs: {len(X_all)}")
    transform, explained, _ = pca_2d(X_all)
    print(f"PCA explained variance: PC1={explained[0]:.1f}%, PC2={explained[1]:.1f}%")

    pc_all = transform(X_all)

    # Clip extreme Cd for colour scale
    cd_lo = np.percentile(cd_all, 2)
    cd_hi = np.percentile(cd_all, 98)

    # ── Plot 1: coloured by Cd ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        pc_all[:, 0], pc_all[:, 1],
        c=np.clip(cd_all, cd_lo, cd_hi),
        cmap="plasma_r", s=8, alpha=0.6, linewidths=0,
        vmin=cd_lo, vmax=cd_hi,
    )
    plt.colorbar(sc, ax=ax, label="Cd (drag coefficient)")

    # Mark best design per method
    offset = 0
    for name, X, cd in datasets:
        n = len(X)
        cd_chunk = cd_all[offset:offset + n]
        pc_chunk = pc_all[offset:offset + n]
        best_idx = int(np.argmin(cd_chunk))
        ax.scatter(*pc_chunk[best_idx], marker="*", s=200,
                   color=COLORS.get(name, "black"), zorder=5,
                   edgecolors="white", linewidths=0.5, label=f"Best: {name} (Cd={cd_chunk[best_idx]:.5f})")
        offset += n

    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% var)", fontsize=11)
    ax.set_title("DrivAer Star — Design Space PCA (coloured by Cd)", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)

    out1_png = os.path.join(OUT_DIR, "pca_scatter_cd.png")
    out1_pdf = os.path.join(OUT_DIR, "pca_scatter_cd.pdf")
    fig.savefig(out1_png, dpi=150, bbox_inches="tight")
    fig.savefig(out1_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1_png}")
    print(f"Saved: {out1_pdf}")

    # ── Plot 2: coloured by method ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    offset = 0
    for name, X, cd in datasets:
        n = len(X)
        pc_chunk = pc_all[offset:offset + n]
        color = COLORS.get(name, "gray")
        ax.scatter(pc_chunk[:, 0], pc_chunk[:, 1],
                   color=color, s=8, alpha=0.5, linewidths=0, label=f"{name} (n={n})")
        offset += n

    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% var)", fontsize=11)
    ax.set_title("DrivAer Star — Design Space PCA (coloured by method)", fontsize=12)
    ax.legend(fontsize=9, markerscale=2)
    ax.grid(True, alpha=0.2)

    out2_png = os.path.join(OUT_DIR, "pca_scatter_method.png")
    out2_pdf = os.path.join(OUT_DIR, "pca_scatter_method.pdf")
    fig.savefig(out2_png, dpi=150, bbox_inches="tight")
    fig.savefig(out2_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2_png}")
    print(f"Saved: {out2_pdf}")


if __name__ == "__main__":
    main()
