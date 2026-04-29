"""Radar (spider) chart of normalized FFD parameters for best design per method.

Finds the overall best design (lowest Cd) across all runs for each method,
normalises each parameter to [-1, 1] using BOUNDS from mesh_generator.py,
and plots as an overlaid spider chart.

Usage:
    cd /scratch/ShapeEvolve
    source venv/bin/activate
    python analysis/DrivAer_Star/plot_drivaer_radar.py

Outputs:
    environments/DrivAer_Star/results/analysis_plots_cd_only/radar_best_designs.png
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO_DIR, "environments", "DrivAer_Star", "results")
OUT_DIR = os.path.join(RESULTS_DIR, "analysis_plots_cd_only")

# Import PARAM_KEYS and BOUNDS from the environment
import sys
sys.path.insert(0, os.path.join(REPO_DIR, "environments", "DrivAer_Star"))
from mesh_generator import PARAM_KEYS, BOUNDS

COLORS = {
    "L-BFGS-B":                 "#e377c2",
    "Bayesian Opt. (exact GP)": "#ff7f0e",
    "PSO (120p × 500i)":        "#1f77b4",
    "ShapeEvolve":              "#2ca02c",
}


def normalize_params(params):
    """Normalize param dict to [-1, 1] using BOUNDS. Returns array aligned to PARAM_KEYS."""
    result = []
    for k in PARAM_KEYS:
        lo, hi = BOUNDS[k]
        v = float(params.get(k, 0.0))
        center = (lo + hi) / 2.0
        half_range = (hi - lo) / 2.0
        result.append((v - center) / half_range)
    return np.array(result)


def _find_best_row(csv_path, reward_col):
    """Return the row index with the highest reward (= lowest Cd)."""
    df = pd.read_csv(csv_path)
    if reward_col not in df.columns:
        return None, None
    best_idx = int(df[reward_col].idxmax())
    return df, best_idx


def load_best_ga():
    """Best design across all GA runs."""
    base = os.path.join(RESULTS_DIR, "SAVED_DIRS_run_GA_cd_only_drivaer_star_120p_250i")
    best_cd = np.inf
    best_params = None
    for run_dir in sorted(glob.glob(os.path.join(base, "run_GA_cd_only_*"))):
        csv = os.path.join(run_dir, "results.csv")
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv)
        idx = int(df["reward"].idxmax())
        row = df.iloc[idx]
        cd = float(row["Cd"])
        if cd < best_cd:
            iteration = int(row["iteration"])
            particle = int(row["particle"])
            design_path = os.path.join(
                run_dir, f"iter_{iteration:04d}_p{particle:03d}", "design.json"
            )
            if os.path.exists(design_path):
                with open(design_path) as f:
                    best_params = json.load(f)
                best_cd = cd
    print(f"PSO best Cd: {best_cd:.5f}")
    return best_params, best_cd


def load_best_lbfgsb():
    """Best design across all L-BFGS-B runs."""
    base = os.path.join(RESULTS_DIR, "SAVED_DIRS_run_lbfgsb_cd_only_drivaer_star_nr3")
    best_cd = np.inf
    best_params = None
    for run_dir in sorted(glob.glob(os.path.join(base, "run_lbfgsb_cd_only_*"))):
        csv = os.path.join(run_dir, "results.csv")
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv)
        idx = int(df["reward"].idxmax())
        row = df.iloc[idx]
        cd = float(row["Cd"])
        if cd < best_cd:
            call = int(row["call"])
            restart = int(row["restart"])
            design_path = os.path.join(
                run_dir, f"call_{call:05d}_r{restart}", "design.json"
            )
            if os.path.exists(design_path):
                with open(design_path) as f:
                    best_params = json.load(f)
                best_cd = cd
    print(f"L-BFGS-B best Cd: {best_cd:.5f}")
    return best_params, best_cd


def load_best_bo():
    """Best design across all BO_torch runs."""
    best_cd = np.inf
    best_params = None
    for run_dir in sorted(glob.glob(os.path.join(RESULTS_DIR, "run_BO_torch_cd_only_seed*_n1000"))):
        csv = os.path.join(run_dir, "results.csv")
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv)
        idx = int(df["reward"].idxmax())
        row = df.iloc[idx]
        cd = float(row["Cd"])
        if cd < best_cd:
            design_name = str(row["design"])
            design_path = os.path.join(run_dir, design_name, f"{design_name}.json")
            if os.path.exists(design_path):
                with open(design_path) as f:
                    best_params = json.load(f)
                best_cd = cd
    print(f"Bayesian Opt. best Cd: {best_cd:.5f}")
    return best_params, best_cd


def load_best_v3():
    """Best design across all completed v3 runs (n800 + n2400) and live n10000."""
    best_cd = np.inf
    best_params = None
    # Completed SAVED_DIRS
    for pattern in [
        os.path.join(RESULTS_DIR, "SAVED_DIRS_run_v3_*", "run_v3_*"),
        os.path.join(RESULTS_DIR, "run_v3_*_n10000"),
    ]:
        for run_dir in sorted(glob.glob(pattern)):
            csv = os.path.join(run_dir, "results.csv")
            if not os.path.exists(csv):
                continue
            df = pd.read_csv(csv)
            idx = int(df["reward"].idxmax())
            row = df.iloc[idx]
            cd = float(row["Cd"])
            if cd < best_cd:
                design_name = str(row["design"])
                design_path = os.path.join(run_dir, f"{design_name}.json")
                if os.path.exists(design_path):
                    with open(design_path) as f:
                        best_params = json.load(f)
                    best_cd = cd
    print(f"ShapeEvolve best Cd: {best_cd:.5f}")
    return best_params, best_cd


def radar_plot(ax, values_list, labels, colors, method_names, best_cds):
    n = len(PARAM_KEYS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(PARAM_KEYS, size=7)
    ax.set_ylim(-1.15, 1.15)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels(["-1", "-0.5", "0", "0.5", "1"], size=6)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

    for vals, color, name, cd in zip(values_list, colors, method_names, best_cds):
        v = list(vals) + [vals[0]]
        ax.plot(angles, v, color=color, linewidth=1.8, label=f"{name}  (Cd={cd:.5f})")
        ax.fill(angles, v, color=color, alpha=0.08)

    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.15), fontsize=9)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ga_params,   ga_cd   = load_best_ga()
    lb_params,   lb_cd   = load_best_lbfgsb()
    bo_params,   bo_cd   = load_best_bo()
    v3_params,   v3_cd   = load_best_v3()

    entries = []
    for params, cd, name, color in [
        (lb_params, lb_cd,   "L-BFGS-B",                 COLORS["L-BFGS-B"]),
        (bo_params, bo_cd,   "Bayesian Opt. (exact GP)", COLORS["Bayesian Opt. (exact GP)"]),
        (ga_params, ga_cd,   "PSO (120p × 500i)",        COLORS["PSO (120p × 500i)"]),
        (v3_params, v3_cd,   "ShapeEvolve",              COLORS["ShapeEvolve"]),
    ]:
        if params is not None:
            entries.append((normalize_params(params), cd, name, color))

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})
    ax.set_title(
        "DrivAer Star — Best Design FFD Parameters (normalized to bounds)",
        size=12, pad=20,
    )

    radar_plot(
        ax,
        [e[0] for e in entries],
        PARAM_KEYS,
        [e[3] for e in entries],
        [e[2] for e in entries],
        [e[1] for e in entries],
    )

    out_png = os.path.join(OUT_DIR, "DrivAer_Star_radar_best_designs.png")
    out_pdf = os.path.join(OUT_DIR, "DrivAer_Star_radar_best_designs.pdf")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
