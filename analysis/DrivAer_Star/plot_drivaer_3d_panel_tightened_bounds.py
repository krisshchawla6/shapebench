"""3D surface rendering panel — tightened-bounds ablation best designs.

Renders Baseline, BO best, and ShapeEvolve best under tightened parameter bounds
(car_size in [0.9,1.1], diffusor_angle in [-4,4]).  vtk_E (Estateback) only.

Two rows per design: side profile and rear quarter view.

Usage:
    cd /scratch/ShapeEvolve
    source venv/bin/activate
    python analysis/DrivAer_Star/plot_drivaer_3d_panel_tightened_bounds.py

Outputs:
    environments/DrivAer_Star/results/analysis_plots_cd_only_tightened_bounds/
        DrivAer_Star_3d_panel_best_designs_tightened_bounds_vtk_E.png
        DrivAer_Star_3d_panel_best_designs_tightened_bounds_vtk_E.pdf
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
OUT_DIR = os.path.join(RESULTS_DIR, "analysis_plots_cd_only_tightened_bounds")

BASELINE_CD = 0.22334

sys.path.insert(0, os.path.join(REPO_DIR, "environments", "DrivAer_Star"))
from mesh_generator import PARAM_KEYS, apply_ffd

import pyvista as pv
pv.global_theme.background = "#e8e8e8"

CELL_W, CELL_H = 700, 350

METHOD_COLORS = {
    "Baseline":                  "#888888",
    "Bayesian Opt. (exact GP)":  "#ff7f0e",
    "ShapeEvolve":               "#2ca02c",
}

_BASE_VTK = None


def _render(params, view):
    mesh = pv.read(_BASE_VTK)
    pts = np.array(mesh.points, dtype=np.float64)
    mesh.points = apply_ffd(pts, params)

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
    plotter.add_light(pv.Light(position=(-2, -4, 5), intensity=0.85, light_type="scene light"))
    plotter.add_light(pv.Light(position=(4,  3, 1),  intensity=0.35, light_type="scene light"))
    plotter.enable_ssao(radius=0.3, bias=0.025, kernel_size=32)

    plotter.reset_camera()
    bounds = mesh.bounds
    cx = (bounds[0] + bounds[1]) / 2
    cy = (bounds[2] + bounds[3]) / 2
    cz = (bounds[4] + bounds[5]) / 2
    xlen = bounds[1] - bounds[0]
    zlen = bounds[5] - bounds[4]
    dist = max(xlen, zlen) * 1.5

    if view == "side":
        plotter.camera.position    = (cx, cy - dist, cz)
        plotter.camera.focal_point = (cx, cy,         cz)
        plotter.camera.up          = (0, 0, 1)
        plotter.camera.zoom(0.95)
    elif view == "rear_quarter":
        plotter.camera.position    = (bounds[1] + xlen * 0.6,
                                      cy - dist * 0.7,
                                      cz + zlen * 1.1)
        plotter.camera.focal_point = (cx, cy, cz)
        plotter.camera.up          = (0, 0, 1)
        plotter.camera.zoom(0.9)

    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img


def load_best_bo_tight():
    """Best BO design across all 10 tight_bounds seeds.
    CSV columns: iteration,particle,sample,design,reward,best_reward,drag,Cd,lift
    Design JSON: {run_dir}/{design_name}/{design_name}.json
    """
    best_cd, best_params = np.inf, None
    pattern = os.path.join(RESULTS_DIR, "run_BO_torch_cd_only_tight_bounds_vtk_E_seed*_n1000")
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
            p = os.path.join(run_dir, design_name, f"{design_name}.json")
            if os.path.exists(p):
                with open(p) as f:
                    best_params = json.load(f)
                best_cd = cd
    return best_params, best_cd


def load_best_v3_tight():
    """Best ShapeEvolve design across all 10 tight_bounds attempts.
    CSV columns: iteration,sample,design,reward,best_reward,sample_type,drag,Cd,lift,island
    Design JSON: {run_dir}/{design_name}.json  (flat, no subdir)
    """
    best_cd, best_params = np.inf, None
    pattern = os.path.join(
        RESULTS_DIR,
        "run_v3_dynamic_optimizer_cd_only_tight_bounds_drivaer_star_vtk_E_attempt_*_flash_2_5_n10000",
    )
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
            p = os.path.join(run_dir, f"{design_name}.json")
            if os.path.exists(p):
                with open(p) as f:
                    best_params = json.load(f)
                best_cd = cd
    return best_params, best_cd


def main():
    global _BASE_VTK

    os.makedirs(OUT_DIR, exist_ok=True)

    _BASE_VTK = os.path.join(REPO_DIR, "environments", "DrivAer_Star", "data", "vtk_E", "00000.vtk")

    baseline_params = {k: 0.0 for k in PARAM_KEYS}
    baseline_params["car_size"] = 1.0

    entries = [("Baseline", baseline_params, BASELINE_CD)]

    for name, loader in [
        ("Bayesian Opt. (exact GP)", load_best_bo_tight),
        ("ShapeEvolve",             load_best_v3_tight),
    ]:
        params, cd = loader()
        if params is not None:
            print(f"{name}: best Cd = {cd:.5f}")
            entries.append((name, params, cd))
        else:
            print(f"{name}: no data found")

    views = [("side", "Side profile"), ("rear_quarter", "Rear quarter")]
    ncols = len(entries)
    nrows = len(views)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.5, nrows * 2.8),
        gridspec_kw={"hspace": 0.08, "wspace": 0.04},
    )
    if nrows == 1:
        axes = [axes]
    if ncols == 1:
        axes = [[ax] for ax in axes]

    for row_idx, (view_key, view_label) in enumerate(views):
        for col_idx, (name, params, cd) in enumerate(entries):
            ax = axes[row_idx][col_idx]
            print(f"Rendering {name} / {view_key}...")
            img = _render(params, view_key)
            ax.imshow(img)
            ax.axis("off")

            if row_idx == 0:
                cd_str = f"$C_D$ = {cd:.5f}"
                color = METHOD_COLORS.get(name, "#333333")
                ax.set_title(
                    f"{name}\n{cd_str}",
                    fontsize=14,
                    fontweight="bold",
                    color="white",
                    pad=2,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.85),
                )

        axes[row_idx][0].set_ylabel(view_label, fontsize=14, rotation=90,
                                     labelpad=4, va="center")

    fig.suptitle(
        r"DrivAer Star (Estate, vtk\_E) — Tightened-Bounds Ablation: Best Design Shapes"
        "\n"
        r"car\_size $\in [0.9, 1.1]$,  diffusor\_angle $\in [-4^\circ, 4^\circ]$"
        "  |  side profile and rear quarter views",
        fontsize=15, y=1.01,
    )

    stem = "DrivAer_Star_3d_panel_best_designs_tightened_bounds_vtk_E"
    fig.savefig(os.path.join(OUT_DIR, f"{stem}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_DIR}/{stem}.png")
    print(f"Saved: {OUT_DIR}/{stem}.pdf")


if __name__ == "__main__":
    main()
