"""Best design 3D panel: shallow isometric BWB render per method.

One column per method in legend order (L-BFGS-B nr3, L-BFGS-B nr10, BO, GA, v3),
single row.  Shallow isometric camera (elevation ≈20°, azimuth ≈45°) shows
wing planform and leading-edge sweep.

Requires: openvsp310 conda env reachable via ~/miniconda3/envs/openvsp310/bin/python
          (called automatically via subprocess inside generate_mesh).

Usage:
    cd /scratch/ShapeEvolve
    source venv/bin/activate
    python analysis/BlendedNet/plot_blendednet_3d_panel.py

Outputs:
    environments/BlendedNet/results/analysis_plots/3d_panel_best_designs.png/.pdf
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO_DIR, "environments", "BlendedNet", "results")
OUT_DIR = os.path.join(RESULTS_DIR, "analysis_plots_shapebench_mean_cd")
CACHE_DIR = os.path.join(REPO_DIR, "environments", "BlendedNet", "mesh_cache")

sys.path.insert(0, os.path.join(REPO_DIR, "environments", "BlendedNet"))
from mesh_generator import generate_mesh

import pyvista as pv
pv.global_theme.background = "#e8e8e8"

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

CELL_W, CELL_H = 720, 420


# ── Design loading ─────────────────────────────────────────────────────────────

def _design_dir(run_dir, row, method_key):
    if method_key == "lbfgsb":
        return os.path.join(run_dir, f"call_{int(row['call']):05d}_r{int(row['restart'])}")
    elif method_key == "ga":
        return os.path.join(run_dir, f"iter_{int(row['iteration']):04d}_p{int(row['particle']):03d}")
    else:
        return os.path.join(run_dir, str(row["design"]))


def _load_best(run_dirs, reward_col, method_key):
    """Best design = run whose peak reward is the median across all run peaks."""
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
    return d["design"], -median_val  # median_CD (positive)


def load_best(name):
    if name == "L-BFGS-B":
        dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "run_lbfgsb_shapebench_5_seed*_nr3_normfix")))
        return _load_best(dirs, "reward", "lbfgsb")
    elif name == "Bayesian Opt. (exact GP)":
        dirs = (sorted(glob.glob(os.path.join(RESULTS_DIR, "run_BO_torch_shapebench_5_seed*_n500"))) +
                sorted(glob.glob(os.path.join(RESULTS_DIR, "run_BO_torch_shapebench_5_seed*_n1000"))))
        return _load_best(dirs, "reward", "bo")
    elif name == "PSO (20p × 200i)":
        dirs = []
        for pat in [
            os.path.join(RESULTS_DIR, "run_GA_shapebench_5_seed*_20p_200i"),
            os.path.join(RESULTS_DIR, "run_GA_shapebench_5_attempt*_20p_200i"),
        ]:
            dirs.extend(sorted(glob.glob(pat)))
        return _load_best(dirs, "reward", "ga")
    elif name == "ShapeEvolve":
        dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "run_v3_flash2_5_shapebench_5_attempt_*_n2000")))
        return _load_best(dirs, "reward", "v3")
    elif name == "CMA-ES":
        dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "run_cmaes_shapebench_5_seed*_n500to1000")))
        return _load_best(dirs, "reward", "bo")
    return None, None


# ── Rendering ─────────────────────────────────────────────────────────────────

def _render(params):
    """Render BWB with shallow isometric camera (elev≈20°, az≈45°). Returns RGB array."""
    mesh = generate_mesh(params, cache_dir=CACHE_DIR)

    plotter = pv.Plotter(off_screen=True, window_size=(CELL_W, CELL_H))
    plotter.add_mesh(
        mesh,
        color="#5a7fa0",
        show_edges=False,
        smooth_shading=True,
        specular=0.15,
        specular_power=8,
        ambient=0.08,
        diffuse=0.9,
    )
    plotter.add_light(pv.Light(position=(-3, -3, 6), intensity=0.85, light_type="scene light"))
    plotter.add_light(pv.Light(position=(3,  3, 1),  intensity=0.35, light_type="scene light"))
    plotter.enable_ssao(radius=0.05, bias=0.005, kernel_size=32)

    bounds = mesh.bounds
    cx = (bounds[0] + bounds[1]) / 2
    cy = (bounds[2] + bounds[3]) / 2
    cz = (bounds[4] + bounds[5]) / 2
    span = max(bounds[3] - bounds[2], bounds[1] - bounds[0])
    dist = span * 1.6

    el = np.radians(20)
    az = np.radians(45)
    px = cx + dist * np.cos(el) * np.cos(az)
    py = cy + dist * np.cos(el) * np.sin(az)
    pz = cz + dist * np.sin(el)

    plotter.camera.position    = (px, py, pz)
    plotter.camera.focal_point = (cx, cy, cz)
    plotter.camera.up          = (0, 0, 1)
    plotter.camera.zoom(0.85)

    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    plt.rcParams.update(STYLE)
    os.makedirs(OUT_DIR, exist_ok=True)

    method_order = ["L-BFGS-B", "Bayesian Opt. (exact GP)", "PSO (20p × 200i)", "ShapeEvolve", "CMA-ES"]

    entries = []
    for name in method_order:
        params, median_cd = load_best(name)
        if params is not None:
            entries.append((name, params, median_cd))
            print(f"{name:<18}  median_CD={median_cd:.6f}")
        else:
            print(f"{name:<18}  no data")

    if not entries:
        print("No data found.")
        return

    ncols = len(entries)
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 4.2, 3.8),
                              gridspec_kw={"wspace": 0.04})
    if ncols == 1:
        axes = [axes]

    for col_idx, (name, params, median_cd) in enumerate(entries):
        ax = axes[col_idx]
        print(f"Rendering {name}...")
        try:
            img = _render(params)
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"render failed\n{e}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, wrap=True)
        ax.axis("off")
        color = COLORS.get(name, "#333333")
        ax.set_title(
            f"{name}\nmedian CD = {median_cd:.5f}",
            fontsize=9, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.85),
        )

    fig.suptitle(
        "BlendedNet (BWB) — Best Design Shapes per Method  (shallow isometric view)\n"
        "Shapebench-5: reward = −mean_CD  at  CL ∈ {0.185, 0.206, 0.227}  "
        "— CD shown = median best across seeds",
        fontsize=10, y=1.02,
    )

    out_png = os.path.join(OUT_DIR, "3d_panel_best_designs.png")
    out_pdf = os.path.join(OUT_DIR, "3d_panel_best_designs.pdf")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
