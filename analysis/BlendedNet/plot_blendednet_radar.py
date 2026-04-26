"""Radar/spider chart of normalized planform parameters for median-best design per method.

Nine axes: B1, B2, B3 (span sections), C2, C3, C4 (chord sections), S1, S2, S3 (sweeps).
Values are normalized to [0, 1] using parameter bounds.
Each method's median-best design is plotted as a filled polygon.

Usage:
    cd /scratch/ShapeEvolve
    source venv/bin/activate
    python analysis/BlendedNet/plot_blendednet_radar.py

Outputs:
    environments/BlendedNet/results/analysis_plots/radar_best_designs.png/.pdf
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

GEOM_KEYS = ["B1", "B2", "B3", "C2", "C3", "C4", "S1", "S2", "S3"]

BOUNDS = {
    "B1": (100, 200),
    "B2": (50,  200),
    "B3": (200, 700),
    "C2": (550, 850),
    "C3": (180, 280),
    "C4": (60,  90),
    "S1": (40,  60),
    "S2": (40,  60),
    "S3": (24,  40),
}

AXIS_LABELS = {
    "B1": "B1\n(inner span)",
    "B2": "B2\n(mid span)",
    "B3": "B3\n(outer span)",
    "C2": "C2\n(inner chord)",
    "C3": "C3\n(mid chord)",
    "C4": "C4\n(tip chord)",
    "S1": "S1\n(inner sweep)",
    "S2": "S2\n(mid sweep)",
    "S3": "S3\n(outer sweep)",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def _design_dir(run_dir, row, method_key):
    if method_key == "lbfgsb":
        return os.path.join(run_dir, f"call_{int(row['call']):05d}_r{int(row['restart'])}")
    elif method_key == "ga":
        return os.path.join(run_dir, f"iter_{int(row['iteration']):04d}_p{int(row['particle']):03d}")
    else:
        return os.path.join(run_dir, str(row["design"]))


def _load_median_params(run_dirs, reward_col, method_key):
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
    return d["design"], -median_val


def load_median(name):
    if name == "L-BFGS-B":
        dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "run_lbfgsb_shapebench_5_seed*_nr3_normfix")))
        return _load_median_params(dirs, "reward", "lbfgsb")
    elif name == "Bayesian Opt. (exact GP)":
        dirs = (sorted(glob.glob(os.path.join(RESULTS_DIR, "run_BO_torch_shapebench_5_seed*_n500"))) +
                sorted(glob.glob(os.path.join(RESULTS_DIR, "run_BO_torch_shapebench_5_seed*_n1000"))))
        return _load_median_params(dirs, "reward", "bo")
    elif name == "PSO (20p × 200i)":
        dirs = []
        for pat in [
            os.path.join(RESULTS_DIR, "run_GA_shapebench_5_seed*_20p_200i"),
            os.path.join(RESULTS_DIR, "run_GA_shapebench_5_attempt*_20p_200i"),
        ]:
            dirs.extend(sorted(glob.glob(pat)))
        return _load_median_params(dirs, "reward", "ga")
    elif name == "ShapeEvolve":
        dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "run_v3_flash2_5_shapebench_5_attempt_*_n2000")))
        return _load_median_params(dirs, "reward", "v3")
    elif name == "CMA-ES":
        dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "run_cmaes_shapebench_5_seed*_n500to1000")))
        return _load_median_params(dirs, "reward", "bo")
    return None, None


def _normalize(params):
    return np.array([
        (params[k] - BOUNDS[k][0]) / (BOUNDS[k][1] - BOUNDS[k][0])
        for k in GEOM_KEYS
    ])


# ── Radar helpers ─────────────────────────────────────────────────────────────

def _radar_polygon(ax, values, color, label, alpha_fill=0.12, lw=1.8):
    n = len(values)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    vals = np.concatenate([values, [values[0]]])
    angs = np.concatenate([angles, [angles[0]]])
    xs = vals * np.cos(angs)
    ys = vals * np.sin(angs)
    ax.fill(xs, ys, color=color, alpha=alpha_fill)
    ax.plot(xs, ys, color=color, lw=lw, label=label)


def _draw_radar_axes(ax):
    n = len(GEOM_KEYS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    for r in [0.25, 0.5, 0.75, 1.0]:
        circle = plt.Circle((0, 0), r, color="grey", fill=False,
                              lw=0.4, ls="--", alpha=0.5, zorder=0)
        ax.add_patch(circle)
    for k, angle in zip(GEOM_KEYS, angles):
        x, y = np.cos(angle), np.sin(angle)
        ax.plot([0, x], [0, y], color="grey", lw=0.8, zorder=0)
        lbl_r = 1.18
        ax.text(lbl_r * x, lbl_r * y, AXIS_LABELS[k],
                ha="center", va="center", fontsize=8, color="#333333")
    for r, txt in [(0.25, "25%"), (0.5, "50%"), (0.75, "75%"), (1.0, "100%")]:
        ax.text(0, r + 0.02, txt, ha="center", va="bottom", fontsize=6.5, color="grey")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    plt.rcParams.update(STYLE)
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)

    _draw_radar_axes(ax)

    # Midpoint reference polygon (dashed grey)
    mid_vals = np.full(len(GEOM_KEYS), 0.5)
    _radar_polygon(ax, mid_vals, color="grey", label="Midpoint (bounds centre)",
                   alpha_fill=0.0, lw=1.0)

    print(f"{'Method':<18}  {'median_CD':>10}  {'B1–S3 (normalised)':>40}")
    print("-" * 75)

    method_order = ["L-BFGS-B", "Bayesian Opt. (exact GP)", "PSO (20p × 200i)", "ShapeEvolve", "CMA-ES"]
    for name in method_order:
        params, median_cd = load_median(name)
        if params is None:
            print(f"{name:<18}  no data")
            continue
        norm = _normalize(params)
        _radar_polygon(ax, norm, color=COLORS[name],
                       label=f"{name}  (median CD={median_cd:.5f})")
        print(f"{name:<18}  {median_cd:>10.6f}  {'  '.join(f'{v:.2f}' for v in norm)}")

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), fontsize=8.5,
              framealpha=0.95, ncol=2, title="Method")
    ax.set_title(
        "BlendedNet (BWB) — Planform Parameters of Median-Best Design per Method\n"
        "Normalised to [0 = lower bound, 1 = upper bound]",
        fontsize=11, fontweight="medium", pad=12,
    )

    out_png = os.path.join(OUT_DIR, "BlendedNet_radar_best_designs.png")
    out_pdf = os.path.join(OUT_DIR, "BlendedNet_radar_best_designs.pdf")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
