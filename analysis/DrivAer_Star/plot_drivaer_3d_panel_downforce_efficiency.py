"""3D surface rendering panel: baseline vs best design per method (downforce_efficiency).

Reward: reward = -Cl / max(Cd, 1e-4)   (maximized)

Produces two rows of renders for each design (baseline + BO + ShapeEvolve):
  Row 1 — side profile view (x-z plane)
  Row 2 — rear three-quarter view

Usage:
    cd /scratch/ShapeEvolve
    source venv/bin/activate
    python analysis/DrivAer_Star/plot_drivaer_3d_panel_downforce_efficiency.py [--body {E,F,N}]

Outputs:
    environments/DrivAer_Star/results/analysis_plots_downforce_efficiency/
        DrivAer_Star_3d_panel_best_designs_vtk_{body}_downforce_efficiency.png/.pdf
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO_DIR, "environments", "DrivAer_Star", "results")
OUT_DIR     = os.path.join(RESULTS_DIR, "analysis_plots_downforce_efficiency")

# Baseline downforce efficiency for undeformed geometry (iter_0_s0)
BASELINE_REWARD_BY_BODY = {
    "E": -0.332,
    "F": -0.393,
    "N": -0.028,
}

sys.path.insert(0, os.path.join(REPO_DIR, "environments", "DrivAer_Star"))
from mesh_generator import PARAM_KEYS, BOUNDS, apply_ffd

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
    pts  = np.array(mesh.points, dtype=np.float64)
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


def load_best_bo(body):
    pattern = os.path.join(
        RESULTS_DIR,
        f"run_BO_torch_downforce_efficiency_vtk_{body}_seed*_n1000",
    )
    best_reward, best_params = -np.inf, None
    for run_dir in sorted(glob.glob(pattern)):
        csv = os.path.join(run_dir, "results.csv")
        if not os.path.exists(csv):
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if "reward" not in df.columns:
            continue
        idx = int(df["reward"].idxmax())
        row = df.iloc[idx]
        r   = float(row["reward"])
        if r > best_reward:
            design_name = str(row["design"])
            p = os.path.join(run_dir, design_name, f"{design_name}.json")
            if os.path.exists(p):
                with open(p) as f:
                    best_params = json.load(f)
                best_reward = r
    return best_params, best_reward


def load_best_v3(body):
    pattern = os.path.join(
        RESULTS_DIR,
        f"run_v3_dynamic_optimizer_downforce_efficiency_drivaer_star_vtk_{body}"
        f"_attempt_*_flash_2_5_n6000",
    )
    best_reward, best_params = -np.inf, None
    for run_dir in sorted(glob.glob(pattern)):
        csv = os.path.join(run_dir, "results.csv")
        if not os.path.exists(csv):
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if "reward" not in df.columns:
            continue
        idx = int(df["reward"].idxmax())
        row = df.iloc[idx]
        r   = float(row["reward"])
        if r > best_reward:
            design_name = str(row["design"])
            p = os.path.join(run_dir, f"{design_name}.json")
            if os.path.exists(p):
                with open(p) as f:
                    best_params = json.load(f)
                best_reward = r
    return best_params, best_reward


def main():
    global _BASE_VTK

    parser = argparse.ArgumentParser(description="DrivAer 3D panel — downforce efficiency")
    parser.add_argument("--body", choices=["E", "F", "N"], default="E",
                        help="Body style: E=Estate, F=Fastback, N=Notchback (default: E)")
    args = parser.parse_args()
    body = args.body

    os.makedirs(OUT_DIR, exist_ok=True)

    body_names   = {"E": "Estate", "F": "Fastback", "N": "Notchback"}
    body_label   = body_names[body]
    BASELINE_REW = BASELINE_REWARD_BY_BODY[body]

    base_vtk_file = "00001.vtk" if body == "N" else "00000.vtk"
    _BASE_VTK = os.path.join(
        REPO_DIR, "environments", "DrivAer_Star", "data", f"vtk_{body}", base_vtk_file
    )

    baseline_params = {k: 0.0 for k in PARAM_KEYS}
    baseline_params["car_size"] = 1.0

    entries = [("Baseline", baseline_params, BASELINE_REW)]

    for name, loader in [
        ("Bayesian Opt. (exact GP)", load_best_bo),
        ("ShapeEvolve",             load_best_v3),
    ]:
        params, reward = loader(body)
        if params is not None:
            entries.append((name, params, reward))

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
        for col_idx, (name, params, reward) in enumerate(entries):
            ax = axes[row_idx][col_idx]
            print(f"Rendering {name} / {view_key}...")
            img = _render(params, view_key)
            ax.imshow(img)
            ax.axis("off")

            if row_idx == 0:
                rew_str = f"reward = {reward:.4f}"
                color   = METHOD_COLORS.get(name, "#333333")
                ax.set_title(
                    f"{name}\n{rew_str}",
                    fontsize=9,
                    fontweight="bold",
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.85),
                )

        axes[row_idx][0].set_ylabel(view_label, fontsize=9, rotation=90,
                                     labelpad=4, va="center")

    fig.suptitle(
        f"DrivAer$^\\star$ ({body_label}, vtk_{body}) — Downforce Efficiency ($-C_l/C_d$) — Best Design Shapes\n"
        "(side profile shows ramp/hood/trunklid/diffusor angles; "
        "rear quarter shows trunklid, diffusor, greenhouse)",
        fontsize=10, y=1.01,
    )

    out_base = os.path.join(
        OUT_DIR, f"DrivAer_Star_3d_panel_best_designs_vtk_{body}_downforce_efficiency"
    )
    fig.savefig(out_base + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_base}.png")
    print(f"Saved: {out_base}.pdf")


if __name__ == "__main__":
    main()
