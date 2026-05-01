"""3D surface rendering panel: baseline vs best design per method.

Produces two rows of renders for each design (baseline + methods available):
  Row 1 — side profile view (x-z plane): reveals ramp_angle, hood angle,
           windscreen slope, trunklid, diffusor angle.
  Row 2 — rear three-quarter view: reveals trunklid, diffusor, rear window,
           greenhouse angle.

Camera auto-resets to data bounds per design so different car_size values are
all rendered at comparable apparent scale.

Usage:
    cd /scratch/ShapeEvolve
    source venv/bin/activate
    python analysis/DrivAer_Star/plot_drivaer_3d_panel.py [--body {E,F,N}] [--variant {cd_only,cd_cl_constrained}]

Outputs (cd_only):
    environments/DrivAer_Star/results/analysis_plots_cd_only/DrivAer_Star_3d_panel_best_designs_vtk_{body}.png
Outputs (cd_cl_constrained):
    environments/DrivAer_Star/results/analysis_plots_cd_cl_constrained/DrivAer_Star_3d_panel_best_designs_vtk_{body}_cd_cl_constrained.png
"""

import argparse
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

BASELINE_CD_BY_BODY = {
    "E": 0.22334,
    "F": 0.18990,
    "N": 0.18132,
}

sys.path.insert(0, os.path.join(REPO_DIR, "environments", "DrivAer_Star"))
from mesh_generator import PARAM_KEYS, BOUNDS, apply_ffd

import pyvista as pv
pv.global_theme.background = "#e8e8e8"  # light gray — avoids white wash-out

# Render resolution per cell
CELL_W, CELL_H = 700, 350

# Method colours for title bars
METHOD_COLORS = {
    "Baseline":                  "#888888",
    "L-BFGS-B":                  "#e377c2",
    "Bayesian Opt. (exact GP)":  "#ff7f0e",
    "PSO (120p × 500i)":         "#1f77b4",
    "ShapeEvolve":               "#2ca02c",
}

# Set by main() before _render() is called
_BASE_VTK = None


def _render(params, view):
    """Render design to RGB array for a given view ('side' or 'rear_quarter')."""
    mesh = pv.read(_BASE_VTK)
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


def load_best_bo(body="E"):
    if body == "E":
        pattern = os.path.join(RESULTS_DIR, "run_BO_torch_cd_only_seed*_n1000")
    else:
        pattern = os.path.join(RESULTS_DIR, f"run_BO_torch_cd_only_vtk_{body}_seed*_n1000")
    best_cd, best_params = np.inf, None
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


def load_best_v3(body="E"):
    if body == "E":
        patterns = [
            os.path.join(RESULTS_DIR, "SAVED_DIRS_run_v3_*", "run_v3_*"),
            os.path.join(RESULTS_DIR, "run_v3_*_n10000"),
        ]
    else:
        patterns = [
            os.path.join(
                RESULTS_DIR,
                f"run_v3_dynamic_optimizer_cd_only_drivaer_star_vtk_{body}_attempt_*_flash_2_5_n6000",
            )
        ]
    best_cd, best_params = np.inf, None
    for pattern in patterns:
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


def _load_best_recovered(run_dirs):
    """Load best design from results_recovered.csv (preempted runs).

    Params are stored in {design}/save/results.json under the 'design' key.
    """
    best_cd, best_params = np.inf, None
    for run_dir in run_dirs:
        csv = os.path.join(run_dir, "results_recovered.csv")
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv)
        if df.empty:
            continue
        idx = int(df["reward"].idxmax())
        row = df.iloc[idx]
        cd = float(row["Cd"])
        if cd < best_cd:
            design_name = str(row["design"])
            p = os.path.join(run_dir, design_name, "save", "results.json")
            if os.path.exists(p):
                with open(p) as f:
                    best_params = json.load(f)["design"]
                best_cd = cd
    return best_params, best_cd


def load_best_bo_constrained(body):
    run_dirs = [
        os.path.join(RESULTS_DIR, f"run_BO_torch_cd_cl_constrained_vtk_{body}_seed{s}_n1000")
        for s in range(10)
    ]
    return _load_best_recovered(run_dirs)


def load_best_v3_constrained(body):
    run_dirs = [
        os.path.join(
            RESULTS_DIR,
            f"run_v3_dynamic_optimizer_cd_cl_constrained_"
            f"drivaer_star_vtk_{body}_attempt_{a}_flash_2_5_n6000",
        )
        for a in range(1, 11)
    ]
    return _load_best_recovered(run_dirs)


def main():
    global _BASE_VTK

    parser = argparse.ArgumentParser(description="DrivAer 3D panel plot")
    parser.add_argument("--body", choices=["E", "F", "N"], default="E",
                        help="Body style: E=Estate, F=Fastback, N=Notchback (default: E)")
    parser.add_argument("--variant", choices=["cd_only", "cd_cl_constrained"], default="cd_only",
                        help="Reward variant (default: cd_only)")
    args = parser.parse_args()
    body    = args.body
    variant = args.variant

    out_dir = os.path.join(RESULTS_DIR, f"analysis_plots_{variant}")
    os.makedirs(out_dir, exist_ok=True)

    body_names = {"E": "Estate", "F": "Fastback", "N": "Notchback"}
    body_label = body_names[body]
    BASELINE_CD = BASELINE_CD_BY_BODY[body]

    # Set module-level VTK path used by _render()
    # vtk_N dataset was sampled without design index 0; 00001.vtk is its reference mesh
    base_vtk_file = "00001.vtk" if body == "N" else "00000.vtk"
    _BASE_VTK = os.path.join(
        REPO_DIR, "environments", "DrivAer_Star", "data", f"vtk_{body}", base_vtk_file
    )

    # car_size=1.0 is the no-deformation nominal; all other params at 0 = no change.
    baseline_params = {k: 0.0 for k in PARAM_KEYS}
    baseline_params["car_size"] = 1.0

    entries = [("Baseline", baseline_params, BASELINE_CD)]

    if variant == "cd_only":
        # GA/PSO and L-BFGS-B only available for Estate (E)
        if body == "E":
            for name, loader in [
                ("PSO (120p × 500i)", load_best_ga),
                ("L-BFGS-B", load_best_lbfgsb),
            ]:
                params, cd = loader()
                if params is not None:
                    entries.append((name, params, cd))

        for name, loader, kwargs in [
            ("Bayesian Opt. (exact GP)", load_best_bo, {"body": body}),
            ("ShapeEvolve",             load_best_v3, {"body": body}),
        ]:
            params, cd = loader(**kwargs)
            if params is not None:
                entries.append((name, params, cd))

    else:  # cd_cl_constrained
        for name, loader in [
            ("Bayesian Opt. (exact GP)", load_best_bo_constrained),
            ("ShapeEvolve",             load_best_v3_constrained),
        ]:
            params, cd = loader(body)
            if params is not None:
                entries.append((name, params, cd))

    views = [("side", "Side profile"), ("rear_quarter", "Rear quarter")]
    ncols = len(entries)
    nrows = len(views)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.5, nrows * 2.8),
        gridspec_kw={"hspace": 0.08, "wspace": 0.04},
    )
    # Ensure axes is always 2D
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

            # Column title on top row only
            if row_idx == 0:
                cd_str = f"$C_D$ = {cd:.5f}"
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

    variant_label = "" if variant == "cd_only" else " ($C_D$-constrained)"
    fig.suptitle(
        f"DrivAer Star ({body_label}, vtk_{body}){variant_label} — Best Design Shapes\n"
        "(side profile shows ramp/hood/trunklid/diffusor angles; "
        "rear quarter shows trunklid, diffusor, greenhouse)",
        fontsize=10, y=1.01,
    )

    suffix = f"_vtk_{body}" + ("_cd_cl_constrained" if variant == "cd_cl_constrained" else "")
    out_png = os.path.join(out_dir, f"DrivAer_Star_3d_panel_best_designs{suffix}.png")
    out_pdf = os.path.join(out_dir, f"DrivAer_Star_3d_panel_best_designs{suffix}.pdf")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
