"""2D top-view planform overlay + 3 vertically-stacked isometric 3D renders.

Same layout as plot_blendednet_planform.py but selects the single BEST design
per method (lowest CD across all runs/seeds) rather than the median-best.

Usage:
    cd /scratch/ShapeEvolve
    source venv/bin/activate
    python analysis/BlendedNet/plot_blendednet_planform_best_cd.py

Outputs:
    environments/BlendedNet/results/analysis_plots_shapebench_mean_cd/planform_best_designs_best_cd.png/.pdf
"""

import os
import sys
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
CACHE_DIR = os.path.join(REPO_DIR, "environments", "BlendedNet", "mesh_cache")

sys.path.insert(0, os.path.join(REPO_DIR, "environments", "BlendedNet"))
from mesh_generator import generate_mesh

import pyvista as pv
pv.global_theme.background = "#e8e8e8"

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "cm",
    "font.size": 22,
    "axes.labelsize": 26,
    "axes.titlesize": 26,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 21,
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

METHOD_ORDER = ["L-BFGS-B", "Bayesian Opt. (exact GP)", "PSO (20p × 200i)", "ShapeEvolve", "CMA-ES"]

# Three representative designs for the 3D panels:
#   cluster  — BO median-best, representative of BO/PSO/CMA-ES (all converged to same corner)
#   distinct — ShapeEvolve median-best (different sweep/fuselage geometry)
#   outlier  — L-BFGS-B median-best (trapped local optimum, clearly different shape)
RENDER_PANELS = [
    ("Bayesian Opt. (exact GP)", "BO / PSO / CMA-ES\n(converged cluster)"),
    ("ShapeEvolve",              "ShapeEvolve"),
    ("L-BFGS-B",                 "L-BFGS-B"),
]

C1 = 1000.0  # fixed root chord (mm)


# ── Geometry ──────────────────────────────────────────────────────────────────

def planform_polygon(params):
    B1 = params["B1"]; B2 = params["B2"]; B3 = params["B3"]
    C2 = params["C2"]; C3 = params["C3"]; C4 = params["C4"]
    S1 = np.radians(params["S1"])
    S2 = np.radians(params["S2"])
    S3 = np.radians(params["S3"])

    y = np.array([0.0, B1, B1 + B2, B1 + B2 + B3])
    le = np.array([
        0.0,
        B1 * np.tan(S1),
        B1 * np.tan(S1) + B2 * np.tan(S2),
        B1 * np.tan(S1) + B2 * np.tan(S2) + B3 * np.tan(S3),
    ])
    chord = np.array([C1, C2, C3, C4])
    te = le + chord
    return y, le, te


def full_span_polygon(params):
    y_half, le, te = planform_polygon(params)

    xs_le = y_half;       xs_te = y_half[::-1]
    ys_le = le;           ys_te = te[::-1]
    xp_le = -y_half[::-1]; xp_te = -y_half
    yp_le = le[::-1];     yp_te = te

    x = np.concatenate([xp_te, xp_le, xs_le, xs_te, [xp_te[0]]])
    y = np.concatenate([yp_te, yp_le, ys_le, ys_te, [yp_te[0]]])
    return x, y


# ── 3D render ─────────────────────────────────────────────────────────────────

def _render(params, ref_span=None):
    """Shallow isometric BWB render (elev≈20°, az≈45°) with 500 mm scale bar.

    ref_span: full-span in normalized units (mm/1000 * 2) shared across all
    panels in a plot so every panel uses the same camera distance → consistent
    zoom level.  Falls back to this design's own span if None.
    """
    half_span_mm = params["B1"] + params["B2"] + params["B3"]
    bar_mm = 500.0

    mesh = generate_mesh(params, cache_dir=CACHE_DIR)

    plotter = pv.Plotter(off_screen=True, window_size=(700, 500))
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
    plotter.add_light(pv.Light(position=(3,   3, 1), intensity=0.35, light_type="scene light"))
    plotter.enable_ssao(radius=0.05, bias=0.005, kernel_size=32)

    bounds = mesh.bounds
    cx = (bounds[0] + bounds[1]) / 2
    cy = (bounds[2] + bounds[3]) / 2
    cz = (bounds[4] + bounds[5]) / 2
    mesh_span = max(bounds[3] - bounds[2], bounds[1] - bounds[0])
    dist = (ref_span if ref_span is not None else mesh_span) * 1.0

    el = np.radians(20)
    az = np.radians(45)
    plotter.camera.position    = (cx + dist * np.cos(el) * np.cos(az),
                                   cy + dist * np.cos(el) * np.sin(az),
                                   cz + dist * np.sin(el))
    plotter.camera.focal_point = (cx, cy, cz)
    plotter.camera.up          = (0, 0, 1)
    plotter.camera.zoom(1.1)

    # Scale bar — normalized mesh units; bar placed just below (z) and in front
    # of LE (x < 0) so it is unobstructed in the isometric view.
    half_y_norm  = bounds[3]                              # = half_span_mm / 1000
    bar_len_norm = bar_mm / half_span_mm * half_y_norm
    bx  = bounds[0] - 0.12
    bz  = bounds[4] - 0.03
    by0 = cy - bar_len_norm / 2
    by1 = cy + bar_len_norm / 2

    bar = pv.Line((bx, by0, bz), (bx, by1, bz))
    plotter.add_mesh(bar, color="black", line_width=3)

    cap_r = 0.05
    for y_cap in [by0, by1]:
        cap = pv.Line((bx - cap_r, y_cap, bz), (bx + cap_r, y_cap, bz))
        plotter.add_mesh(cap, color="black", line_width=4)

    plotter.add_point_labels(
        [np.array([bx - 0.12, (by0 + by1) / 2, bz])],
        [f"{bar_mm:.0f} mm"],
        font_size=22, text_color="black",
        point_color="black", point_size=0,
        always_visible=True, shape=None,
    )

    plotter.reset_camera_clipping_range()
    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img


# ── Data loading ──────────────────────────────────────────────────────────────

def _design_dir(run_dir, row, method_key):
    if method_key == "lbfgsb":
        return os.path.join(run_dir, f"call_{int(row['call']):05d}_r{int(row['restart'])}")
    elif method_key == "ga":
        return os.path.join(run_dir, f"iter_{int(row['iteration']):04d}_p{int(row['particle']):03d}")
    else:
        return os.path.join(run_dir, str(row["design"]))


def _load_best_params(run_dirs, reward_col, method_key):
    """Return params and best CD for the single best design across all runs."""
    best_reward, best_run_dir, best_row = -np.inf, None, None
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
        r = float(row[reward_col])
        if r > best_reward:
            best_reward, best_run_dir, best_row = r, run_dir, row

    if best_run_dir is None:
        return None, None

    ddir = _design_dir(best_run_dir, best_row, method_key)
    rj = os.path.join(ddir, "save", "results.json")
    if not os.path.exists(rj):
        return None, None
    with open(rj) as f:
        d = json.load(f)
    return d["design"], -best_reward


def load_best(name):
    if name == "L-BFGS-B":
        dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "run_lbfgsb_shapebench_5_seed*_nr3_normfix")))
        return _load_best_params(dirs, "reward", "lbfgsb")
    elif name == "Bayesian Opt. (exact GP)":
        dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "run_BO_torch_shapebench_5_seed*_n500")))
        return _load_best_params(dirs, "reward", "bo")
    elif name == "PSO (20p × 200i)":
        dirs = []
        for pat in [
            os.path.join(RESULTS_DIR, "run_GA_shapebench_5_seed*_20p_200i"),
            os.path.join(RESULTS_DIR, "run_GA_shapebench_5_attempt*_20p_200i"),
        ]:
            dirs.extend(sorted(glob.glob(pat)))
        return _load_best_params(dirs, "reward", "ga")
    elif name == "ShapeEvolve":
        dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "run_v3_flash2_5_shapebench_5_attempt_*_n2000")))
        return _load_best_params(dirs, "reward", "v3")
    elif name == "CMA-ES":
        dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "run_cmaes_shapebench_5_seed*_n500to1000")))
        return _load_best_params(dirs, "reward", "bo")
    return None, None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    plt.rcParams.update(STYLE)
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Layout: 2D planform (left column, full height) + 3 × 3D renders (right
    #    column, stacked vertically).  width_ratios give the 2D panel ~47% of
    #    the total width; each 3D render is compact and square-ish.
    fig = plt.figure(figsize=(20, 18), facecolor="white")
    outer_gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 2], wspace=0.1,
                                left=0.07, right=0.97, top=0.93, bottom=0.05)
    left_gs  = outer_gs[0, 0].subgridspec(2, 1, height_ratios=[8, 2], hspace=0.05)
    right_gs = outer_gs[0, 1].subgridspec(3, 1, hspace=0.02)
    ax        = fig.add_subplot(left_gs[0])   # 2D planform overlay
    legend_ax = fig.add_subplot(left_gs[1])   # dedicated legend panel (left col only)
    legend_ax.axis("off")
    axes_3d   = [fig.add_subplot(right_gs[i]) for i in range(3)]  # 3D render panels

    # ── Left: 2D planform overlay ──────────────────────────────────────────────
    print(f"{'Method':<28}  {'best_CD':>10}  {'half_span':>10}  {'root_chord':>10}")
    print("-" * 65)

    for name in METHOD_ORDER:
        params, best_cd = load_best(name)
        if params is None:
            print(f"{name:<28}  no data")
            continue

        x, y = full_span_polygon(params)
        half_span = params["B1"] + params["B2"] + params["B3"]

        ax.fill(x, y, color=COLORS[name], alpha=0.18)
        ax.plot(x, y, color=COLORS[name], lw=1.6,
                label=f"{name}  (best CD={best_cd:.5f})")

        print(f"{name:<28}  {best_cd:>10.6f}  {half_span:>10.1f}  {C1:>10.1f}")

    ax.set_xlabel("Span (mm)")
    ax.set_ylabel("Chord (mm, LE → TE)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axvline(0, color="grey", lw=0.6, ls="--", alpha=0.5)
    ax.grid(True, alpha=0.2)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    legend_ax.legend(handles, labels, fontsize=21, loc="center",
                     framealpha=0.95, title="Method", borderaxespad=0, ncol=1)
    ax.set_title(
        "Top-view planform — best design per method\n"
        "Full span shown (symmetric).  LE at top, TE at bottom.",
        fontweight="medium", pad=8,
    )

    # ── Right: 3 isometric 3D renders (one per distinct design cluster) ────────
    # Pre-collect to compute consistent ref_span (same camera distance for all panels)
    _render_data = [(m, lbl, *load_best(m)) for m, lbl in RENDER_PANELS]
    ref_span = max(
        (2 * (p["B1"] + p["B2"] + p["B3"]) / 1000.0)
        for _, _, p, _ in _render_data if p is not None
    ) if any(p is not None for _, _, p, _ in _render_data) else None

    for ax_3d, (method, panel_label, params, cd) in zip(axes_3d, _render_data):
        ax_3d.axis("off")
        if params is not None:
            print(f"Rendering 3D: {method}  (best CD={cd:.5f})...")
            try:
                img = _render(params, ref_span=ref_span)
                ax_3d.imshow(img)
            except Exception as e:
                ax_3d.text(0.5, 0.5, f"render failed\n{e}", ha="center", va="center",
                           transform=ax_3d.transAxes, fontsize=7, wrap=True)
        else:
            ax_3d.text(0.5, 0.5, "no data", ha="center", va="center",
                       transform=ax_3d.transAxes, fontsize=9)

        cd_str = f"best CD = {cd:.5f}" if params is not None else ""
        ax_3d.text(0.5, 0.97, f"{panel_label}\n{cd_str}",
                   transform=ax_3d.transAxes, fontsize=22, fontweight="medium",
                   ha="center", va="top",
                   bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                             alpha=0.75, edgecolor="none"))

    fig.suptitle(
        "BlendedNet (BWB) — Planform comparison: best design per method",
        fontsize=26, fontweight="medium", y=1.005,
    )

    out_png = os.path.join(OUT_DIR, "planform_best_designs_best_cd.png")
    out_pdf = os.path.join(OUT_DIR, "planform_best_designs_best_cd.pdf")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
