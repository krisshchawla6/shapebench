#!/usr/bin/env python3
"""
generate_design_corrected.py — Delta wing VLM + VortexNet GNN correction.

Runs VLM for the low-fidelity baseline, then applies the pretrained VortexNet
GNN to predict per-panel pressure corrections, and re-runs VLM with the
corrected DCP to obtain high-fidelity aerodynamic coefficients.

Usage:
    python generate_design_corrected.py design.json --aoa 10 --mach 0.3
    python generate_design_corrected.py design.json --aoa 10 --mach 0.3 --re 3e6
"""

import argparse
import json
import os
import sys
import math
import importlib.util
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import torch
from scipy.spatial import distance_matrix

# ── paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / 'scripts'
VNET_DIR    = ROOT / 'MF-VortexNet'

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(VNET_DIR))

_spec = importlib.util.spec_from_file_location(
    "generateDeltawing1", SCRIPTS_DIR / "generateDeltawing 1.py",
)
gendw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gendw)

vehicle_setup     = gendw.vehicle_setup
point_analysis    = gendw.point_analysis
NACA4DigitAirfoil = gendw.NACA4DigitAirfoil

from VortexNet import GNN4
from VortexNet.VortexNetUtils import VortexNetUtils
from SUAVE.Core import Units

# ── constants ────────────────────────────────────────────────────────────────
PRETRAINED_WEIGHTS = VNET_DIR / 'pretrained_model' / 'model_weights_20241116_045918.pth'
PRETRAINED_HP      = VNET_DIR / 'pretrained_model' / 'tuning_results_20241116_045918.json'

# dataset-fixed constants (must match VortexNet training setup)
_ROOT_CHORD_IN = 25.734
_TWIST_ROOT    = 0.0
_TWIST_TIP     = 0.0
_DIHEDRAL      = 0.0
_CHORD_LENGTH  = 1.0


# ── helpers ──────────────────────────────────────────────────────────────────

def _geom_kwargs():
    return dict(
        root_chord_in=_ROOT_CHORD_IN,
        twist_root=_TWIST_ROOT,
        twist_tip=_TWIST_TIP,
        dihedral=_DIHEDRAL,
    )


# ── model loading ────────────────────────────────────────────────────────────

def load_vortexnet():
    hp = json.load(open(PRETRAINED_HP))['hyperparameters']
    model = GNN4(
        node_in_channels=11,
        edge_in_channels=1,
        hidden_channels=hp['hidden_channels'],
        out_channels=1,
        num_coarse=900,
        num_fine=900,
        dropout_rate=hp['dropout_rate'],
        HEADS=hp['HEADS'],
        ALPHA=hp['ALPHA'],
        HOP=hp['HOP'],
    )
    state = torch.load(str(PRETRAINED_WEIGHTS), map_location='cpu', weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


# ── geometry helpers ─────────────────────────────────────────────────────────

def compute_lattice_slopes_and_curvature(VD, naca_params, le_sweep):
    """Compute slopes, gaussian curvature, thickness per panel from airfoil + panel geometry.

    Stock SUAVE 2.5.2 VD doesn't carry ZCU/ZCL, so we reconstruct upper/lower
    surface Z at each control point from the NACA airfoil definition.
    """
    airfoil = NACA4DigitAirfoil(
        m=naca_params['m'], p=naca_params['p'], t=naca_params['t'],
    )
    total_panels = len(VD.XC)
    ZCU = np.zeros(total_panels)
    ZCL = np.zeros(total_panels)

    for wing_idx in range(len(VD.n_cw)):
        n_cw = VD.n_cw[wing_idx]
        n_sw = VD.n_sw[wing_idx]
        start_strip = VD.spanwise_breaks[wing_idx]
        end_strip = VD.spanwise_breaks[wing_idx + 1] if wing_idx + 1 < len(VD.spanwise_breaks) else len(VD.chordwise_breaks)
        start_panel = VD.chordwise_breaks[start_strip]
        end_panel = VD.chordwise_breaks[end_strip] if end_strip < len(VD.chordwise_breaks) else total_panels

        XC_strip = VD.XC[start_panel:end_panel].reshape((n_sw, n_cw))
        for j in range(n_sw):
            x_row = XC_strip[j, :]
            local_le = x_row.min()
            local_chord = x_row.max() - local_le
            if local_chord < 1e-12:
                continue
            x_norm = np.clip((x_row - local_le) / local_chord, 0, 1)
            yt = airfoil.thickness_distribution(x_norm) * local_chord
            yc = airfoil.camber_line(x_norm) * local_chord
            idx_start = start_panel + j * n_cw
            idx_end   = idx_start + n_cw
            ZCU[idx_start:idx_end] = yc + yt
            ZCL[idx_start:idx_end] = yc - yt

    thickness_arr = np.zeros(total_panels)
    slope_u_arr   = np.zeros(total_panels)
    curv_u_arr    = np.zeros(total_panels)
    slope_l_arr   = np.zeros(total_panels)
    curv_l_arr    = np.zeros(total_panels)

    panel_start = 0
    for wing_idx in range(len(VD.n_cw)):
        n_cw = VD.n_cw[wing_idx]
        n_sw = VD.n_sw[wing_idx]
        num_panels = n_cw * n_sw
        start_strip = VD.spanwise_breaks[wing_idx]
        end_strip = VD.spanwise_breaks[wing_idx + 1] if wing_idx + 1 < len(VD.spanwise_breaks) else len(VD.chordwise_breaks)
        sp = VD.chordwise_breaks[start_strip]
        ep = VD.chordwise_breaks[end_strip] if end_strip < len(VD.chordwise_breaks) else total_panels

        XC_g = VD.XC[sp:ep].reshape((n_sw, n_cw))
        YC_g = VD.YC[sp:ep].reshape((n_sw, n_cw))
        Z_u  = ZCU[sp:ep].reshape((n_sw, n_cw))
        Z_l  = ZCL[sp:ep].reshape((n_sw, n_cw))

        thick_g = Z_u - Z_l
        dXc = np.gradient(XC_g, axis=1)
        dYs = np.gradient(YC_g, axis=0)
        dXc[dXc == 0] = 1e-12
        dYs[dYs == 0] = 1e-12

        Zx_u = np.gradient(Z_u, axis=1) / dXc
        Zy_u = np.gradient(Z_u, axis=0) / dYs
        Zx_l = np.gradient(Z_l, axis=1) / dXc
        Zy_l = np.gradient(Z_l, axis=0) / dYs

        Zxx_u = np.gradient(Zx_u, axis=1) / dXc
        Zyy_u = np.gradient(Zy_u, axis=0) / dYs
        Zxy_u = np.gradient(Zx_u, axis=0) / dYs
        denom_u = (1 + Zx_u**2) * (1 + Zy_u**2) - (Zx_u * Zy_u)**2
        denom_u[denom_u == 0] = 1e-12
        K_u = (Zxx_u * Zyy_u - Zxy_u**2) / denom_u

        Zxx_l = np.gradient(Zx_l, axis=1) / dXc
        Zyy_l = np.gradient(Zy_l, axis=0) / dYs
        Zxy_l = np.gradient(Zx_l, axis=0) / dYs
        denom_l = (1 + Zx_l**2) * (1 + Zy_l**2) - (Zx_l * Zy_l)**2
        denom_l[denom_l == 0] = 1e-12
        K_l = (Zxx_l * Zyy_l - Zxy_l**2) / denom_l

        pe = panel_start + num_panels
        thickness_arr[panel_start:pe] = thick_g.flatten()
        slope_u_arr[panel_start:pe]   = Zx_u.flatten()
        curv_u_arr[panel_start:pe]    = K_u.flatten()
        slope_l_arr[panel_start:pe]   = Zx_l.flatten()
        curv_l_arr[panel_start:pe]    = K_l.flatten()
        panel_start = pe

    return thickness_arr, slope_u_arr, curv_u_arr, slope_l_arr, curv_l_arr


# ── graph construction for inference ─────────────────────────────────────────

def build_graph_input(vlm_results, aoa_deg, mach, re, naca_params, le_sweep):
    """Build a PyG Data object from VLM results for VortexNet inference."""
    VD = vlm_results.VD
    cp = np.squeeze(vlm_results.cp)
    n_panels = len(VD.XC)

    coords = np.column_stack((VD.XC, VD.YC))
    thickness, slope_u, curv_u, slope_l, curv_l = compute_lattice_slopes_and_curvature(VD, naca_params, le_sweep)

    tanh = VortexNetUtils.tanh_standardization
    aoa_rad = aoa_deg * math.pi / 180.0
    ff = np.tile([tanh(aoa_rad), tanh(mach), re / 1e7], (n_panels, 1))

    node_features = np.hstack((
        cp.reshape(-1, 1),
        ff,
        tanh(thickness).reshape(-1, 1),
        tanh(curv_u).reshape(-1, 1),
        tanh(curv_l).reshape(-1, 1),
        tanh(slope_u).reshape(-1, 1),
        tanh(slope_l).reshape(-1, 1),
        coords,
    ))

    dist_mat = distance_matrix(coords, coords)
    edges = []
    for i in range(n_panels):
        for nb in np.argsort(dist_mat[i])[1:5]:
            edges.append([i, nb])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(
        [[np.sqrt((coords[e[0],0]-coords[e[1],0])**2 + (coords[e[0],1]-coords[e[1],1])**2)] for e in edges],
        dtype=torch.float,
    )

    from torch_geometric.data import Data as PyGData
    data = PyGData(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    data.aic_matrix = torch.tensor(np.squeeze(vlm_results.A), dtype=torch.float)
    data.rhs_matrix = torch.tensor(np.squeeze(vlm_results.RHS), dtype=torch.float)
    data.dcpsid     = torch.tensor(np.squeeze(vlm_results.DCPSID), dtype=torch.float)
    data.factor     = torch.tensor(np.squeeze(vlm_results.FACTOR), dtype=torch.float)
    data.chord      = torch.tensor(np.squeeze(vlm_results.CHORD), dtype=torch.float)
    data.rnmax      = torch.tensor(np.squeeze(vlm_results.RNMAX), dtype=torch.float)

    return data


# ── main pipeline ────────────────────────────────────────────────────────────

def run_full_pipeline(params, aoa, mach, re, output_dir):
    le_sweep     = params['le_sweep']
    naca         = params['naca']
    NACA_4DIGITS = {'m': naca['m'], 'p': naca['p'], 't': naca['t'],
                    'chord_length': _CHORD_LENGTH}
    gkw = _geom_kwargs()

    orig_dir = os.getcwd()
    os.chdir(output_dir)
    try:
        vehicle = vehicle_setup(le_sweep, NACA_4DIGITS, **gkw)

        vlm_results = point_analysis(vehicle, aoa, mach, le_sweep, NACA_4DIGITS, **gkw)

        model = load_vortexnet()
        graph = build_graph_input(vlm_results, aoa, mach, re, NACA_4DIGITS, le_sweep)

        with torch.no_grad():
            predicted_dcp = model(graph).cpu().numpy().flatten()

        corrected_results = point_analysis(
            vehicle, aoa, mach, le_sweep, NACA_4DIGITS,
            DCP_overwrite=predicted_dcp.reshape(1, -1), **gkw,
        )
    finally:
        os.chdir(orig_dir)

    return vlm_results, corrected_results, predicted_dcp, vehicle


# ── output ───────────────────────────────────────────────────────────────────

def save_results_json(params, aoa, mach, re, vlm_res, corr_res, output_dir):
    def s(v): return float(np.squeeze(v))

    vlm_cl, vlm_cdi, vlm_cm = s(vlm_res.CL), s(vlm_res.CDi), s(vlm_res.CM)
    cor_cl, cor_cdi, cor_cm = s(corr_res.CL), s(corr_res.CDi), s(corr_res.CM)

    payload = {
        'design_name': params.get('design_name', 'unnamed'),
        'design_params': {
            'le_sweep_deg':   params['le_sweep'],
            'root_chord_in':  _ROOT_CHORD_IN,
            'twist_root_deg': _TWIST_ROOT,
            'twist_tip_deg':  _TWIST_TIP,
            'dihedral_deg':   _DIHEDRAL,
            'naca':           params['naca'],
        },
        'sim_conditions': {
            'aoa_deg':  aoa,
            'mach':     mach,
            'reynolds': re,
        },
        'vlm_results': {
            'CL': vlm_cl, 'CDi': vlm_cdi, 'CM': vlm_cm,
            'L_D': vlm_cl / vlm_cdi if vlm_cdi else None,
        },
        'corrected_results': {
            'CL': cor_cl, 'CDi': cor_cdi, 'CM': cor_cm,
            'L_D': cor_cl / cor_cdi if cor_cdi else None,
        },
    }
    path = os.path.join(output_dir, 'results.json')
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    return path


def save_geometry_png(params, aoa, mach, vlm_res, corr_res, predicted_dcp, geometry, output_dir):
    VD        = vlm_res.VD
    vlm_cp    = np.squeeze(vlm_res.cp)
    corr_cp   = np.squeeze(corr_res.cp)
    naca      = params['naca']
    le        = params['le_sweep']

    def s(v): return float(np.squeeze(v))
    vlm_cl, vlm_cdi, vlm_cm = s(vlm_res.CL), s(vlm_res.CDi), s(vlm_res.CM)
    cor_cl, cor_cdi, cor_cm = s(corr_res.CL), s(corr_res.CDi), s(corr_res.CM)

    BG, PANEL, TC = '#0f0f1a', '#16213e', 'white'
    fig = plt.figure(figsize=(20, 6), facecolor=BG)
    fig.suptitle(
        f"Delta Wing  |  LE Sweep {le} deg  |  "
        f"NACA {naca['m']}{naca['p']}{naca['t']:02d}  |  "
        f"AoA {aoa} deg  |  M {mach}",
        color=TC, fontsize=12, fontweight='bold', y=0.99,
    )

    def draw_cp_panel(ax, VD, cp_arr, title):
        patches, vals = [], []
        for i in range(len(VD.XA1)):
            quad = np.array([
                [VD.XA1[i], VD.YA1[i]], [VD.XB1[i], VD.YB1[i]],
                [VD.XB2[i], VD.YB2[i]], [VD.XA2[i], VD.YA2[i]],
            ])
            patches.append(Polygon(quad, closed=True))
            vals.append(cp_arr[i])
        vals = np.array(vals)
        vmin, vmax = np.percentile(vals, 5), np.percentile(vals, 95)
        pc = PatchCollection(patches, cmap='coolwarm', alpha=0.92)
        pc.set_array(vals)
        pc.set_clim(vmin, vmax)
        ax.add_collection(pc)
        cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('Cp', color=TC)
        cb.ax.yaxis.set_tick_params(color=TC)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=TC)
        ax.autoscale_view()
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)', color=TC)
        ax.set_ylabel('y (m)', color=TC)
        ax.set_title(title, color=TC, fontsize=10)
        ax.tick_params(colors=TC)
        for sp in ax.spines.values():
            sp.set_edgecolor('#444')

    ax1 = fig.add_subplot(1, 3, 1, facecolor=PANEL)
    draw_cp_panel(ax1, VD, vlm_cp, 'VLM  (low-fidelity)')

    ax2 = fig.add_subplot(1, 3, 2, facecolor=PANEL)
    draw_cp_panel(ax2, corr_res.VD, corr_cp, 'VortexNet Corrected  (high-fidelity)')

    ax3 = fig.add_subplot(1, 3, 3, facecolor=PANEL)
    ax3.axis('off')
    ax3.set_title('Results Comparison', color=TC, fontsize=10)

    vlm_ld = vlm_cl / vlm_cdi if vlm_cdi else 0
    cor_ld = cor_cl / cor_cdi if cor_cdi else 0

    rows = [
        ('Design', params.get('design_name', '-'), '', '#7ec8e3', True),
        ('Airfoil', f"NACA {naca['m']}{naca['p']}{naca['t']:02d}", '', TC, True),
        ('LE Sweep', f"{le} deg", '', TC, True),
        ('Root chord', f"{_ROOT_CHORD_IN} in", '', TC, True),
        ('Twist', f"root {_TWIST_ROOT}, tip {_TWIST_TIP} deg", '', TC, True),
        ('Dihedral', f"{_DIHEDRAL} deg", '', TC, True),
        None,
        ('', 'VLM', 'Corrected', '#aaa', True),
        ('CL',  f"{vlm_cl:.4f}",  f"{cor_cl:.4f}",  '#7ff97f', False),
        ('CDi', f"{vlm_cdi:.5f}", f"{cor_cdi:.5f}", '#ff9f7f', False),
        ('CM',  f"{vlm_cm:.4f}",  f"{cor_cm:.4f}",  TC,        False),
        ('L/D', f"{vlm_ld:.2f}",  f"{cor_ld:.2f}",  '#ffd97f', False),
    ]
    y = 0.92
    for row in rows:
        if row is None:
            y -= 0.03
            continue
        label, v1, v2, color, is_header = row
        fw = 'bold' if is_header else 'normal'
        ax3.text(0.04, y, label, color='#aaa', fontsize=9, fontweight=fw, transform=ax3.transAxes)
        ax3.text(0.40, y, v1, color=color, fontsize=9, fontweight='bold', transform=ax3.transAxes)
        if v2:
            ax3.text(0.68, y, v2, color=color, fontsize=9, fontweight='bold', transform=ax3.transAxes)
        y -= 0.06

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    png_path = os.path.join(output_dir, 'geometry.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return png_path


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate delta wing with VLM + VortexNet GNN correction.'
    )
    parser.add_argument('params', help='Path to design parameters JSON file')
    parser.add_argument('--aoa',  type=float, required=True, help='Angle of attack (deg)')
    parser.add_argument('--mach', type=float, required=True, help='Freestream Mach number')
    parser.add_argument('--re',   type=float, default=3.0e6, help='Reynolds number (default: 3e6)')
    parser.add_argument(
        '--output-dir', default='/scratch/3D/designs_3d/test',
        help='Root output directory (subfolder per design_name)',
    )
    args = parser.parse_args()

    with open(args.params) as f:
        params = json.load(f)

    design_name = params.get('design_name', 'unnamed')
    output_dir  = os.path.join(args.output_dir, design_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[pipeline] design : {design_name}")
    print(f"[pipeline] output : {output_dir}")

    vlm_res, corr_res, predicted_dcp, geometry = run_full_pipeline(
        params, args.aoa, args.mach, args.re, output_dir,
    )

    def s(v): return float(np.squeeze(v))
    print(f"[pipeline] VLM        : CL={s(vlm_res.CL):.4f}  CDi={s(vlm_res.CDi):.5f}  CM={s(vlm_res.CM):.4f}")
    print(f"[pipeline] Corrected  : CL={s(corr_res.CL):.4f}  CDi={s(corr_res.CDi):.5f}  CM={s(corr_res.CM):.4f}")

    json_path = save_results_json(params, args.aoa, args.mach, args.re, vlm_res, corr_res, output_dir)
    png_path  = save_geometry_png(params, args.aoa, args.mach, vlm_res, corr_res, predicted_dcp, geometry, output_dir)

    print(f"[pipeline] saved  : {json_path}")
    print(f"[pipeline] saved  : {png_path}")


if __name__ == '__main__':
    main()
