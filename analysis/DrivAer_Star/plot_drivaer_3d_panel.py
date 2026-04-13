"""3D surface rendering panel: baseline vs best design per method.

Produces two rows of renders for each of the 5 designs (baseline + 4 methods):
  Row 1 — side profile view (x-z plane): reveals ramp_angle, hood angle,
           windscreen slope, trunklid, diffusor angle.
  Row 2 — rear three-quarter view: reveals trunklid, diffusor, rear window,
           greenhouse angle.

Camera auto-resets to data bounds per design so different car_size values are
all rendered at comparable apparent scale.

Usage:
    cd /scratch/ShapeEvolve
    source venv/bin/activate
    python analysis/DrivAer_Star/plot_drivaer_3d_panel.py

Outputs:
    environments/DrivAer_Star/results/analysis_plots_cd_only/3d_panel_best_designs.png
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
BASE_VTK = os.path.join(REPO_DIR, "environments", "DrivAer_Star", "data", "vtk_E", "00000.vtk")

sys.path.insert(0, os.path.join(REPO_DIR, "environments", "DrivAer_Star"))
from mesh_generator import PARAM_KEYS, BOUNDS, apply_ffd

import pyvista as pv
pv.global_theme.background = "#e8e8e8"  # light gray — avoids white wash-out

# Render resolution per cell
CELL_W, CELL_H = 700, 350

# Method colours for title bars
METHOD_COLORS = {
    "Baseline":     "#888888",
    "GA/PSO":       "#e07b39",
    "L-BFGS-B":     "#7b9e87",
    "BO_torch":     "#4a90d9",
    "v3 flash-2.5": "#9b59b6",
}


def _render(params, view):
    """Render design to RGB array for a given view ('side' or 'rear_quarter')."""
    mesh = pv.read(BASE_VTK)
    pts = np.array(mesh.points, dtype=np.float64)
    mesh.points = apply_ffd(pts, params)

    plotter = pv.Plotter(off_screen=True, window_size=(CELL_W, CELL_H))
    plotter.add_mesh(
        mesh,
        color="#5a7fa0",       # muted steel blue — dark enough to show shading
        show_edges=False,
        smooth_shading=True,
        specular=0.15,
        specular_power=8,
        ambient=0.08,          # low ambient so shadows read clearly
        diffuse=0.9,
    )
    # Single strong key light from upper front-left; fill from rear-right at half power
    plotter.add_light(pv.Light(position=(-2, -4, 5), intensity=0.85, light_type="scene light"))
    plotter.add_light(pv.Light(position=(4,  3, 1),  intensity=0.35, light_type="scene light"))
    # Ambient occlusion for concavity depth cues
    plotter.enable_ssao(radius=0.3, bias=0.025, kernel_size=32)

    # Auto-fit camera to data, then override direction
    plotter.reset_camera()
    bounds = mesh.bounds  # (xmin,xmax, ymin,ymax, zmin,zmax)
    cx = (bounds[0] + bounds[1]) / 2
    cy = (bounds[2] + bounds[3]) / 2
    cz = (bounds[4] + bounds[5]) / 2
    xlen = bounds[1] - bounds[0]
    zlen = bounds[5] - bounds[4]
    dist = max(xlen, zlen) * 1.5

    if view == "side":
        # Pure side view: camera at -y, looking in +y direction
        plotter.camera.position    = (cx, cy - dist, cz)
        plotter.camera.focal_point = (cx, cy,         cz)
        plotter.camera.up          = (0, 0, 1)
        plotter.camera.zoom(0.95)

    elif view == "rear_quarter":
        # Rear-right elevated view
        plotter.camera.position    = (bounds[1] + xlen * 0.6,
                                      cy - dist * 0.7,
                                      cz + zlen * 1.1)
        plotter.camera.focal_point = (cx, cy, cz)
        plotter.camera.up          = (0, 0, 1)
        plotter.camera.zoom(0.9)

    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img


def load_best_ga():
    base = os.path.join(RESULTS_DIR, "SAVED_DIRS_run_GA_cd_only_drivaer_star_120p_250i")
    best_cd, best_params = np.inf, None
    for run_dir in sorted(glob.glob(os.path.join(base, "run_GA_cd_only_*"))):
        csv = os.path.join(run_dir, "results.csv")
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv)
        idx = int(df["reward"].idxmax())
        row = df.iloc[idx]
        cd = float(row["Cd"])
        if cd < best_cd:
            p = os.path.join(
                run_dir,
                f"iter_{int(row['iteration']):04d}_p{int(row['particle']):03d}",
                "design.json",
            )
            if os.path.exists(p):
                with open(p) as f:
                    best_params = json.load(f)
                best_cd = cd
    return best_params, best_cd


def load_best_lbfgsb():
    base = os.path.join(RESULTS_DIR, "SAVED_DIRS_run_lbfgsb_cd_only_drivaer_star_nr3")
    best_cd, best_params = np.inf, None
    for run_dir in sorted(glob.glob(os.path.join(base, "run_lbfgsb_cd_only_*"))):
        csv = os.path.join(run_dir, "results.csv")
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv)
        idx = int(df["reward"].idxmax())
        row = df.iloc[idx]
        cd = float(row["Cd"])
        if cd < best_cd:
            p = os.path.join(
                run_dir,
                f"call_{int(row['call']):05d}_r{int(row['restart'])}",
                "design.json",
            )
            if os.path.exists(p):
                with open(p) as f:
                    best_params = json.load(f)
                best_cd = cd
    return best_params, best_cd


def load_best_bo():
    best_cd, best_params = np.inf, None
    for run_dir in sorted(
        glob.glob(os.path.join(RESULTS_DIR, "run_BO_torch_cd_only_seed*_n1000"))
    ):
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


def load_best_v3():
    best_cd, best_params = np.inf, None
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
                p = os.path.join(run_dir, f"{design_name}.json")
                if os.path.exists(p):
                    with open(p) as f:
                        best_params = json.load(f)
                    best_cd = cd
    return best_params, best_cd


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # car_size=1.0 is the no-deformation nominal; all other params at 0 = no change.
    # Baseline Cd=0.22334 evaluated by running Transolver on the undeformed mesh.
    baseline_params = {k: 0.0 for k in PARAM_KEYS}
    baseline_params["car_size"] = 1.0
    BASELINE_CD = 0.22334

    entries = [("Baseline", baseline_params, BASELINE_CD)]
    for name, loader in [
        ("GA/PSO",       load_best_ga),
        ("L-BFGS-B",     load_best_lbfgsb),
        ("BO_torch",     load_best_bo),
        ("v3 flash-2.5", load_best_v3),
    ]:
        params, cd = loader()
        if params is not None:
            entries.append((name, params, cd))

    views = [("side", "Side profile"), ("rear_quarter", "Rear quarter")]
    ncols = len(entries)   # 5
    nrows = len(views)     # 2

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.5, nrows * 2.8),
        gridspec_kw={"hspace": 0.08, "wspace": 0.04},
    )

    for row_idx, (view_key, view_label) in enumerate(views):
        for col_idx, (name, params, cd) in enumerate(entries):
            ax = axes[row_idx][col_idx]
            label = f"  {view_label}  " if col_idx == 0 else ""
            print(f"Rendering {name} / {view_key}...")
            img = _render(params, view_key)
            ax.imshow(img)
            ax.axis("off")

            # Column title on top row only
            if row_idx == 0:
                cd_str = f"Cd = {cd:.5f}"
                color = METHOD_COLORS.get(name, "#333333")
                ax.set_title(
                    f"{name}\n{cd_str}",
                    fontsize=9,
                    fontweight="bold",
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.85),
                )

        # Row label on left
        axes[row_idx][0].set_ylabel(view_label, fontsize=9, rotation=90,
                                     labelpad=4, va="center")

    fig.suptitle(
        "DrivAer Star — Best Design Shapes\n"
        "(side profile shows ramp/hood/trunklid/diffusor angles; "
        "rear quarter shows trunklid, diffusor, greenhouse)",
        fontsize=10, y=1.01,
    )

    out_png = os.path.join(OUT_DIR, "3d_panel_best_designs.png")
    out_pdf = os.path.join(OUT_DIR, "3d_panel_best_designs.pdf")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
