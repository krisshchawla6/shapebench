"""Parameter evolution over function evaluations for the best run of each method.

For each method, loads the single best run (lowest final Cd), reads every design
file in evaluation order, and plots how each of the 20 FFD parameters evolves
over the course of the optimisation.

Layout: 20-panel grid (4 cols x 5 rows), one panel per parameter.
Each panel has one coloured curve per method.

Usage:
    cd /scratch/ShapeEvolve
    source venv/bin/activate
    python analysis/DrivAer_Star/plot_drivaer_param_evolution.py

Outputs:
    environments/DrivAer_Star/results/analysis_plots_cd_only/param_evolution_best_run.png
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

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO_DIR, "environments", "DrivAer_Star", "results")
OUT_DIR = os.path.join(RESULTS_DIR, "analysis_plots_cd_only")

sys.path.insert(0, os.path.join(REPO_DIR, "environments", "DrivAer_Star"))
from mesh_generator import PARAM_KEYS, BOUNDS

COLORS = {
    "L-BFGS-B":                 "#e377c2",
    "Bayesian Opt. (exact GP)": "#ff7f0e",
    "PSO (120p × 500i)":        "#1f77b4",
    "ShapeEvolve":              "#2ca02c",
}


def normalize(value, key):
    lo, hi = BOUNDS[key]
    center = (lo + hi) / 2.0
    half_range = (hi - lo) / 2.0
    return (value - center) / half_range


def load_ga_best_run():
    """Return (evals, param_matrix [n_evals x 20]) for the best GA run.

    Tracks the gbest particle over iterations: at each iteration, the particle
    that holds the current gbest is the one with reward == gbest_reward for that
    iteration. We read design files for each cumulative best improvement.
    """
    base = os.path.join(RESULTS_DIR, "SAVED_DIRS_run_GA_cd_only_drivaer_star_120p_250i")
    best_final_cd = np.inf
    best_run_data = None

    for run_dir in sorted(glob.glob(os.path.join(base, "run_GA_cd_only_*"))):
        csv = os.path.join(run_dir, "results.csv")
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv)
        final_cd = -df["gbest_reward"].iloc[-1]
        if final_cd < best_final_cd:
            best_final_cd = final_cd
            best_run_data = (run_dir, df)

    if best_run_data is None:
        return None, None

    run_dir, df = best_run_data
    # Track gbest design over iterations: sample one design per iteration
    # (the particle that achieves the running gbest)
    evals_list = []
    params_list = []
    current_best_reward = -np.inf
    current_params = {k: 0.0 for k in PARAM_KEYS}

    for idx, row in df.iterrows():
        eval_num = idx  # row index = cumulative eval
        r = float(row["reward"])
        if r > current_best_reward:
            current_best_reward = r
            iteration = int(row["iteration"])
            particle = int(row["particle"])
            design_path = os.path.join(
                run_dir, f"iter_{iteration:04d}_p{particle:03d}", "design.json"
            )
            if os.path.exists(design_path):
                with open(design_path) as f:
                    current_params = json.load(f)
        evals_list.append(eval_num)
        params_list.append([normalize(current_params.get(k, 0.0), k) for k in PARAM_KEYS])

    return np.array(evals_list), np.array(params_list)


def load_lbfgsb_best_run():
    base = os.path.join(RESULTS_DIR, "SAVED_DIRS_run_lbfgsb_cd_only_drivaer_star_nr3")
    best_final_cd = np.inf
    best_run_data = None

    for run_dir in sorted(glob.glob(os.path.join(base, "run_lbfgsb_cd_only_*"))):
        csv = os.path.join(run_dir, "results.csv")
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv)
        final_cd = -df["best_reward"].iloc[-1]
        if final_cd < best_final_cd:
            best_final_cd = final_cd
            best_run_data = (run_dir, df)

    if best_run_data is None:
        return None, None

    run_dir, df = best_run_data
    evals_list = []
    params_list = []
    current_best_reward = -np.inf
    current_params = {k: 0.0 for k in PARAM_KEYS}

    for idx, row in df.iterrows():
        r = float(row["reward"])
        if r > current_best_reward:
            current_best_reward = r
            call = int(row["call"])
            restart = int(row["restart"])
            design_path = os.path.join(run_dir, f"call_{call:05d}_r{restart}", "design.json")
            if os.path.exists(design_path):
                with open(design_path) as f:
                    current_params = json.load(f)
        evals_list.append(idx)
        params_list.append([normalize(current_params.get(k, 0.0), k) for k in PARAM_KEYS])

    return np.array(evals_list), np.array(params_list)


def load_bo_best_run():
    best_final_cd = np.inf
    best_run_data = None

    for run_dir in sorted(glob.glob(os.path.join(RESULTS_DIR, "run_BO_torch_cd_only_seed*_n1000"))):
        csv = os.path.join(run_dir, "results.csv")
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv)
        final_cd = -df["best_reward"].iloc[-1]
        if final_cd < best_final_cd:
            best_final_cd = final_cd
            best_run_data = (run_dir, df)

    if best_run_data is None:
        return None, None

    run_dir, df = best_run_data
    evals_list = []
    params_list = []
    current_best_reward = -np.inf
    current_params = {k: 0.0 for k in PARAM_KEYS}

    for idx, row in df.iterrows():
        r = float(row["reward"])
        if r > current_best_reward:
            current_best_reward = r
            design_name = str(row["design"])
            design_path = os.path.join(run_dir, design_name, f"{design_name}.json")
            if os.path.exists(design_path):
                with open(design_path) as f:
                    current_params = json.load(f)
        evals_list.append(idx)
        params_list.append([normalize(current_params.get(k, 0.0), k) for k in PARAM_KEYS])

    return np.array(evals_list), np.array(params_list)


def load_v3_best_run():
    best_final_cd = np.inf
    best_run_data = None

    for pattern in [
        os.path.join(RESULTS_DIR, "SAVED_DIRS_run_v3_*", "run_v3_*"),
        os.path.join(RESULTS_DIR, "run_v3_*_n10000"),
    ]:
        for run_dir in sorted(glob.glob(pattern)):
            csv = os.path.join(run_dir, "results.csv")
            if not os.path.exists(csv):
                continue
            df = pd.read_csv(csv)
            final_cd = -df["best_reward"].iloc[-1]
            if final_cd < best_final_cd:
                best_final_cd = final_cd
                best_run_data = (run_dir, df)

    if best_run_data is None:
        return None, None

    run_dir, df = best_run_data
    evals_list = []
    params_list = []
    current_best_reward = -np.inf
    current_params = {k: 0.0 for k in PARAM_KEYS}

    for idx, row in df.iterrows():
        r = float(row["reward"])
        if r > current_best_reward:
            current_best_reward = r
            design_name = str(row["design"])
            design_path = os.path.join(run_dir, f"{design_name}.json")
            if os.path.exists(design_path):
                with open(design_path) as f:
                    current_params = json.load(f)
        evals_list.append(idx)
        params_list.append([normalize(current_params.get(k, 0.0), k) for k in PARAM_KEYS])

    return np.array(evals_list), np.array(params_list)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    datasets = []
    loaders = [
        ("L-BFGS-B",                 COLORS["L-BFGS-B"],                 load_lbfgsb_best_run),
        ("Bayesian Opt. (exact GP)", COLORS["Bayesian Opt. (exact GP)"], load_bo_best_run),
        ("PSO (120p × 500i)",        COLORS["PSO (120p × 500i)"],        load_ga_best_run),
        ("ShapeEvolve",              COLORS["ShapeEvolve"],               load_v3_best_run),
    ]
    for name, color, loader in loaders:
        print(f"Loading {name}...")
        evals, param_matrix = loader()
        if evals is not None:
            datasets.append((name, color, evals, param_matrix))

    ncols = 4
    nrows = 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 2.5), sharex=False)
    axes = axes.flatten()

    for pi, param_name in enumerate(PARAM_KEYS):
        ax = axes[pi]
        lo, hi = BOUNDS[param_name]
        for name, color, evals, param_matrix in datasets:
            # Thin out for large runs (GA has 30k+ rows)
            step = max(1, len(evals) // 500)
            ax.plot(evals[::step], param_matrix[::step, pi],
                    color=color, linewidth=1.2, alpha=0.85, label=name)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.set_ylim(-1.15, 1.15)
        ax.set_title(param_name, fontsize=8)
        ax.set_xlabel("Eval #", fontsize=7)
        ax.set_ylabel("Norm.", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    # Hide unused panels
    for j in range(len(PARAM_KEYS), len(axes)):
        axes[j].axis("off")

    # Global legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right",
               bbox_to_anchor=(0.99, 0.01), fontsize=9, ncol=2)

    fig.suptitle(
        "DrivAer Star — Parameter Evolution (best run per method, normalized to bounds)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()

    out_png = os.path.join(OUT_DIR, "DrivAer_Star_param_evolution_best_run.png")
    out_pdf = os.path.join(OUT_DIR, "DrivAer_Star_param_evolution_best_run.pdf")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
