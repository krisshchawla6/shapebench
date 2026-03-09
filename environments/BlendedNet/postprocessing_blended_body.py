"""
Post-processing for BlendedNet surrogate predictions.

Uses PyVista for 3D surface rendering with CFD-standard conventions:
  - Cp: jet colormap, percentile-clipped
  - |Cf|: log10 scale, sequential colormap
  - Cfx: diverging colormap showing flow direction

Produces 6 images + quantities.json.

Usage:
    python postprocessing_blended_body.py <case_dir>
"""

import os
import sys
import json
import argparse
import numpy as np

import pyvista as pv
pv.OFF_SCREEN = True


def load_case(case_dir):
    npz_path = os.path.join(case_dir, "fields.npz")
    results_path = os.path.join(case_dir, "results.json")

    fields = np.load(npz_path)
    meta = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            meta = json.load(f)

    vtk_path = meta.get("design", {}).get("vtk_path")
    return fields, meta, vtk_path


def build_mesh(vtk_path, fields):
    """Load VTK mesh and map surrogate predictions onto full surface."""
    mesh = pv.read(vtk_path)

    from scipy.spatial import cKDTree
    tree = cKDTree(fields["pos"])
    _, idx = tree.query(np.array(mesh.points, dtype=np.float32), k=1)

    Cp = fields["Cp"][idx]
    Cfx = fields["Cfx"][idx]
    Cfz = fields["Cfz"][idx]
    Cf_mag = np.sqrt(Cfx**2 + Cfz**2)

    mesh.point_data["Cp"] = Cp
    mesh.point_data["Cfx"] = Cfx
    mesh.point_data["Cfz"] = Cfz
    mesh.point_data["Cf_mag"] = Cf_mag
    mesh.point_data["log10_Cf"] = np.log10(np.clip(Cf_mag, 1e-6, None))

    return mesh


def render(mesh, scalar, out_path, title, cmap, clim, view, fmt="%.3f"):
    pl = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    pl.set_background("white")
    pl.add_mesh(mesh, scalars=scalar, cmap=cmap, clim=clim,
                smooth_shading=True, show_edges=False,
                scalar_bar_args={"title": title, "shadow": True,
                                 "fmt": fmt, "n_labels": 9,
                                 "title_font_size": 18,
                                 "label_font_size": 14,
                                 "position_x": 0.80, "width": 0.14,
                                 "color": "black"})
    if view == "top":
        pl.view_xy()
        pl.camera.zoom(1.3)
    else:
        pl.camera_position = [(0.5, -1.6, 0.5), (0.5, 0.0, 0.0), (0, 0, 1)]
    pl.add_axes(color="black")
    pl.screenshot(out_path, transparent_background=False)
    pl.close()


def compute_quantities(mesh, meta):
    Cp = np.array(mesh.point_data["Cp"])
    Cfx = np.array(mesh.point_data["Cfx"])
    Cfz = np.array(mesh.point_data["Cfz"])
    Cf_mag = np.array(mesh.point_data["Cf_mag"])
    pts = np.array(mesh.points)

    peak_idx = int(np.argmin(Cp))
    CL = float(-Cp.mean())
    CD = float(Cfx.mean())

    return {
        "n_points": int(mesh.n_points),
        "n_cells": int(mesh.n_cells),
        "Cp_mean": float(Cp.mean()),
        "Cp_min": float(Cp.min()),
        "Cp_max": float(Cp.max()),
        "suction_peak_Cp": float(Cp[peak_idx]),
        "suction_peak_pos": pts[peak_idx].tolist(),
        "Cfx_mean": float(Cfx.mean()),
        "Cfz_mean": float(Cfz.mean()),
        "Cf_mag_mean": float(Cf_mag.mean()),
        "Cf_mag_max": float(Cf_mag.max()),
        "separation_fraction": float((Cfx < 0).mean()),
        "CL_approx": CL,
        "CD_approx": CD,
        "L_D_approx": CL / CD if abs(CD) > 1e-12 else 0.0,
        **({"input_meta": {k: meta[k] for k in ["Re", "Mach", "alpha", "reward", "L_D"] if k in meta}} if meta else {}),
    }


def run(case_dir, out_dir=None):
    out_dir = out_dir or os.path.join(case_dir, "postprocessing")
    os.makedirs(out_dir, exist_ok=True)

    fields, meta, vtk_path = load_case(case_dir)
    if not vtk_path or not os.path.exists(vtk_path):
        print(f"ERROR: VTK not found: {vtk_path}")
        sys.exit(1)

    mesh = build_mesh(vtk_path, fields)
    print(f"Mesh: {mesh.n_points} pts, {mesh.n_cells} cells")

    Cp = np.array(mesh.point_data["Cp"])
    Cfx = np.array(mesh.point_data["Cfx"])

    cp_lo, cp_hi = float(np.percentile(Cp, 2)), float(np.percentile(Cp, 98))
    cfx_lim = float(np.percentile(np.abs(Cfx), 98))

    specs = [
        ("Cp",  "Cp",  "coolwarm", (cp_lo, cp_hi),      "%.3f"),
        ("Cfx", "Cfx", "seismic",  (-cfx_lim, cfx_lim), "%.4f"),
    ]

    for scalar, title, cmap, clim, fmt in specs:
        print(f"  {title}: clim=[{clim[0]:.4f}, {clim[1]:.4f}]")
        for view in ["iso", "top"]:
            path = os.path.join(out_dir, f"{scalar}_{view}.png")
            render(mesh, scalar, path, f"{title} ({view})", cmap, clim, view, fmt)

    q = compute_quantities(mesh, meta)
    with open(os.path.join(out_dir, "quantities.json"), "w") as f:
        json.dump(q, f, indent=2)

    print(f"\n{out_dir}/")
    for fn in sorted(os.listdir(out_dir)):
        print(f"  {fn}")
    print(f"\n  L/D: {q['L_D_approx']:.3f}  |  Cp min: {q['Cp_min']:.3f}  |  Sep: {q['separation_fraction']*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run(args.case_dir, out_dir=args.out)
