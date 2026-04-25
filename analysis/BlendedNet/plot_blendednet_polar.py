"""Operating-point polar: CD vs CL for median-best design per method.

Loads save/results.json operating_points for the run whose peak reward equals
the median across seeds, and plots CD (y-axis) vs CL (x-axis) per method.
The three unique CL targets are {0.185, 0.206, 0.227}.

Usage:
    cd /scratch/ShapeEvolve
    source venv/bin/activate
    python analysis/BlendedNet/plot_blendednet_polar.py

Outputs:
    environments/BlendedNet/results/analysis_plots/polar_best_designs.png/.pdf
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

MARKERS = {
    "L-BFGS-B":                "s",
    "Bayesian Opt. (exact GP)": "D",
    "PSO (20p × 200i)":        "o",
    "ShapeEvolve":             "P",
    "CMA-ES":                  "^",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def _design_dir(run_dir, row, method_key):
    if method_key == "lbfgsb":
        return os.path.join(run_dir, f"call_{int(row['call']):05d}_r{int(row['restart'])}")
    elif method_key == "ga":
        return os.path.join(run_dir, f"iter_{int(row['iteration']):04d}_p{int(row['particle']):03d}")
    else:
        return os.path.join(run_dir, str(row["design"]))


def _load_median_results_json(run_dirs, reward_col, method_key):
    """Load results.json from the run whose peak reward equals the median across seeds."""
    run_data = []
    for run_dir in run_dirs:
        csv = os.path.join(run_dir, "results.csv")
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
        run_data.append((float(row[reward_col]), run_dir, row))

    if not run_data:
        return None, None

    finals = np.array([x[0] for x in run_data])
    median_val = float(np.median(finals))
    _, run_dir, row = min(run_data, key=lambda x: abs(x[0] - median_val))

    ddir = _design_dir(run_dir, row, method_key)
    rj = os.path.join(ddir, "save", "results.json")
    if not os.path.exists(rj):
        return None, None
    with open(rj) as f:
        d = json.load(f)
    return d, -median_val


def load_median(name):
    if name == "L-BFGS-B":
        dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "run_lbfgsb_shapebench_5_seed*_nr3_normfix")))
        return _load_median_results_json(dirs, "reward", "lbfgsb")
    elif name == "Bayesian Opt. (exact GP)":
        dirs = (sorted(glob.glob(os.path.join(RESULTS_DIR, "run_BO_torch_shapebench_5_seed*_n500"))) +
                sorted(glob.glob(os.path.join(RESULTS_DIR, "run_BO_torch_shapebench_5_seed*_n1000"))))
        return _load_median_results_json(dirs, "reward", "bo")
    elif name == "PSO (20p × 200i)":
        dirs = []
        for pat in [
            os.path.join(RESULTS_DIR, "run_GA_shapebench_5_seed*_20p_200i"),
            os.path.join(RESULTS_DIR, "run_GA_shapebench_5_attempt*_20p_200i"),
        ]:
            dirs.extend(sorted(glob.glob(pat)))
        return _load_median_results_json(dirs, "reward", "ga")
    elif name == "ShapeEvolve":
        dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "run_v3_flash2_5_shapebench_5_attempt_*_n2000")))
        return _load_median_results_json(dirs, "reward", "v3")
    elif name == "CMA-ES":
        dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "run_cmaes_shapebench_5_seed*_n500to1000")))
        return _load_median_results_json(dirs, "reward", "bo")
    return None, None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    plt.rcParams.update(STYLE)
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor="white")

    method_order = ["L-BFGS-B", "Bayesian Opt. (exact GP)", "PSO (20p × 200i)", "ShapeEvolve", "CMA-ES"]

    print(f"{'Method':<18}  {'median_CD':>10}  {'CL targets':>30}")
    print("-" * 65)

    for name in method_order:
        d, median_cd = load_median(name)
        if d is None:
            print(f"{name:<18}  no data")
            continue

        ops = d["operating_points"]
        seen = {}
        for op in ops:
            cl = op["cl_target"]
            if cl not in seen:
                seen[cl] = op
        pts = sorted(seen.values(), key=lambda x: x["cl_target"])

        cls = [p["CL_approx"] for p in pts]
        cds = [p["CD_approx"] for p in pts]

        color  = COLORS[name]
        marker = MARKERS[name]
        ax.plot(cls, cds, color=color, lw=1.8, marker=marker, markersize=8,
                label=f"{name}  (median CD={median_cd:.5f})")

        print(f"{name:<18}  {median_cd:>10.6f}  "
              f"CL={[f'{c:.3f}' for c in cls]}  CD={[f'{c:.5f}' for c in cds]}")

    ax.set_xlabel("Lift coefficient CL (approx)")
    ax.set_ylabel("Drag coefficient CD")
    ax.set_title(
        "BlendedNet (BWB) — Operating-Point Polar: median-best design per method\n"
        "Shapebench-5 CL targets: {0.185, 0.206, 0.227}",
        fontweight="medium", pad=8,
    )
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1),
              framealpha=0.95, title="Method", borderaxespad=0)
    ax.grid(True, alpha=0.25)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    y_top = ax.get_ylim()[1]
    for cl_t in [0.185, 0.206, 0.227]:
        ax.axvline(cl_t, color="grey", lw=0.8, ls=":", alpha=0.6)
        ax.text(cl_t, y_top, f"CL={cl_t}", fontsize=7, ha="center", color="grey",
                va="top", rotation=90)

    out_png = os.path.join(OUT_DIR, "polar_best_designs.png")
    out_pdf = os.path.join(OUT_DIR, "polar_best_designs.pdf")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
