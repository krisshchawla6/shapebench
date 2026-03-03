#!/usr/bin/env python3
"""
generate_design.py — Delta wing geometry + VLM wrapper.

Usage:
    python generate_design.py design.json --aoa 10 --mach 0.3
    python generate_design.py design.json --aoa 10 --mach 0.3 --output-dir /scratch/3D/designs_3d/test

Output (per design, inside --output-dir/<design_name>/):
    results.json   — input params + VLM aerodynamic coefficients
    geometry.png   — planform Cp map, airfoil cross-section, design summary
    NACA****.dat   — generated airfoil coordinate file
"""

import argparse
import json
import os
import sys
import importlib.util
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# ── bootstrap: make the existing scripts importable ──────────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

_spec = importlib.util.spec_from_file_location(
    "generateDeltawing1",
    SCRIPTS_DIR / "generateDeltawing 1.py",
)
gendw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gendw)

vehicle_setup     = gendw.vehicle_setup
point_analysis    = gendw.point_analysis
NACA4DigitAirfoil = gendw.NACA4DigitAirfoil


# ── dataset-fixed constants (must match VortexNet training setup) ─────────────
_ROOT_CHORD_IN = 25.734
_TWIST_ROOT    = 0.0
_TWIST_TIP     = 0.0
_DIHEDRAL      = 0.0
_CHORD_LENGTH  = 1.0

def _geom_kwargs():
    return dict(
        root_chord_in=_ROOT_CHORD_IN,
        twist_root=_TWIST_ROOT,
        twist_tip=_TWIST_TIP,
        dihedral=_DIHEDRAL,
    )


# ── run ──────────────────────────────────────────────────────────────────────

def run_design(params, aoa, mach, output_dir):
    le_sweep     = params['le_sweep']
    naca         = params['naca']
    NACA_4DIGITS = {'m': naca['m'], 'p': naca['p'], 't': naca['t'],
                    'chord_length': _CHORD_LENGTH}
    gkw = _geom_kwargs()

    orig_dir = os.getcwd()
    os.chdir(output_dir)
    try:
        vehicle = vehicle_setup(le_sweep, NACA_4DIGITS, **gkw)
        results = point_analysis(vehicle, aoa, mach, le_sweep, NACA_4DIGITS, **gkw)
    finally:
        os.chdir(orig_dir)

    return results, vehicle


# ── outputs ──────────────────────────────────────────────────────────────────

def save_results_json(params, aoa, mach, results, geometry, output_dir):
    wing = geometry.wings['main_wing']
    cl   = float(np.squeeze(results.CL))
    cdi  = float(np.squeeze(results.CDi))
    cm   = float(np.squeeze(results.CM))

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
            'aoa_deg': aoa,
            'mach':    mach,
        },
        'geometry': {
            'root_chord_m':  float(wing.chords.root),
            'semi_span_m':   float(wing.spans.projected / 2),
            'ref_area_m2':   float(wing.areas.reference),
            'aspect_ratio':  float(wing.aspect_ratio),
            't_over_c':      float(wing.thickness_to_chord),
        },
        'vlm_results': {
            'CL':  cl,
            'CDi': cdi,
            'CM':  cm,
            'L_D': cl / cdi if cdi != 0 else None,
        },
    }
    path = os.path.join(output_dir, 'results.json')
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    return path


def save_geometry_png(params, aoa, mach, results, geometry, output_dir):
    VD   = results.VD
    cp   = np.squeeze(results.cp)
    naca = params['naca']
    le   = params['le_sweep']
    wing = geometry.wings['main_wing']
    cl   = float(np.squeeze(results.CL))
    cdi  = float(np.squeeze(results.CDi))
    cm   = float(np.squeeze(results.CM))

    BG, PANEL, TC = '#0f0f1a', '#16213e', 'white'
    fig = plt.figure(figsize=(17, 6), facecolor=BG)
    fig.suptitle(
        f"Delta Wing  |  LE Sweep {le} deg  |  "
        f"NACA {naca['m']}{naca['p']}{naca['t']:02d}  |  "
        f"AoA {aoa} deg  |  M {mach}",
        color=TC, fontsize=12, fontweight='bold', y=0.99,
    )

    # planform Cp
    ax1 = fig.add_subplot(1, 3, 1, facecolor=PANEL)
    patches, cp_vals = [], []
    for i in range(len(VD.XA1)):
        quad = np.array([
            [VD.XA1[i], VD.YA1[i]],
            [VD.XB1[i], VD.YB1[i]],
            [VD.XB2[i], VD.YB2[i]],
            [VD.XA2[i], VD.YA2[i]],
        ])
        patches.append(Polygon(quad, closed=True))
        cp_vals.append(cp[i])

    cp_arr     = np.array(cp_vals)
    vmin, vmax = np.percentile(cp_arr, 5), np.percentile(cp_arr, 95)
    pc         = PatchCollection(patches, cmap='coolwarm', alpha=0.92)
    pc.set_array(cp_arr)
    pc.set_clim(vmin, vmax)
    ax1.add_collection(pc)
    cb = fig.colorbar(pc, ax=ax1, fraction=0.046, pad=0.04)
    cb.set_label('Cp', color=TC)
    cb.ax.yaxis.set_tick_params(color=TC)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TC)
    ax1.autoscale_view()
    ax1.set_aspect('equal')
    ax1.set_xlabel('x (m)', color=TC)
    ax1.set_ylabel('y (m)', color=TC)
    ax1.set_title('Planform  -  Cp', color=TC, fontsize=10)
    ax1.tick_params(colors=TC)
    for sp in ax1.spines.values():
        sp.set_edgecolor('#444')

    # airfoil cross-section
    ax2 = fig.add_subplot(1, 3, 2, facecolor=PANEL)
    af         = NACA4DigitAirfoil(m=naca['m'], p=naca['p'], t=naca['t'])
    x_af, y_af = af.generate_coordinates(300)
    ax2.fill(x_af, y_af, color='#3a7bd5', alpha=0.75)
    ax2.plot(x_af, y_af, color='#7ec8e3', linewidth=1.5)
    ax2.axhline(0, color='#555', linewidth=0.8, linestyle='--')
    ax2.set_aspect('equal')
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_xlabel('x/c', color=TC)
    ax2.set_ylabel('z/c', color=TC)
    ax2.set_title(f"Airfoil: NACA {naca['m']}{naca['p']}{naca['t']:02d}", color=TC, fontsize=10)
    ax2.tick_params(colors=TC)
    for sp in ax2.spines.values():
        sp.set_edgecolor('#444')
    t_half = naca['t'] / 200.0
    ax2.annotate(
        f"t/c = {naca['t']}%", xy=(0.3, t_half),
        xytext=(0.45, t_half + 0.025), color='#aaa', fontsize=8,
        arrowprops=dict(arrowstyle='->', color='#aaa', lw=0.8),
    )

    # design summary
    ax3 = fig.add_subplot(1, 3, 3, facecolor=PANEL)
    ax3.axis('off')
    ax3.set_title('Design Summary', color=TC, fontsize=10)
    LD = cl / cdi if cdi != 0 else float('inf')

    rows = [
        ('Design',      params.get('design_name', '-'), '#7ec8e3', True),
        None,
        ('Geometry',    '',                              '#aaa',   True),
        ('  LE Sweep',  f"{le} deg",                     TC,       False),
        ('  Airfoil',   f"NACA {naca['m']}{naca['p']}{naca['t']:02d}", TC, False),
        ('  Root chord',f"{_ROOT_CHORD_IN} in",   TC, False),
        ('  Twist root',f"{_TWIST_ROOT} deg",    TC, False),
        ('  Twist tip', f"{_TWIST_TIP} deg",     TC, False),
        ('  Dihedral',  f"{_DIHEDRAL} deg",       TC, False),
        ('  Semi-span', f"{wing.spans.projected/2:.4f} m",       TC, False),
        ('  Ref. area', f"{wing.areas.reference:.4f} m2",        TC, False),
        ('  AR',        f"{wing.aspect_ratio:.3f}",              TC, False),
        None,
        ('Freestream',  '',                              '#aaa',   True),
        ('  AoA',       f"{aoa} deg",                    TC,       False),
        ('  Mach',      f"{mach}",                       TC,       False),
        None,
        ('VLM Results', '',                              '#aaa',   True),
        ('  CL',        f"{cl:.4f}",                     '#7ff97f', False),
        ('  CDi',       f"{cdi:.5f}",                    '#ff9f7f', False),
        ('  CM',        f"{cm:.4f}",                     TC,       False),
        ('  L/D',       f"{LD:.2f}",                     '#ffd97f', False),
    ]
    y = 0.97
    for row in rows:
        if row is None:
            y -= 0.028
            continue
        label, value, color, is_header = row
        fw = 'bold' if is_header else 'normal'
        if is_header and value == '':
            ax3.text(0.04, y, label, color=color, fontsize=9,
                     fontweight=fw, transform=ax3.transAxes)
        elif is_header:
            ax3.text(0.04, y, label, color=color, fontsize=9,
                     fontweight=fw, transform=ax3.transAxes)
        else:
            ax3.text(0.04, y, label, color='#aaa', fontsize=8.5,
                     transform=ax3.transAxes)
            ax3.text(0.62, y, value, color=color, fontsize=8.5,
                     fontweight='bold', transform=ax3.transAxes)
        y -= 0.043

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    png_path = os.path.join(output_dir, 'geometry.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return png_path


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate delta wing design + VLM analysis from a JSON params file.'
    )
    parser.add_argument('params', help='Path to design parameters JSON file')
    parser.add_argument('--aoa',  type=float, required=True, help='Angle of attack (deg)')
    parser.add_argument('--mach', type=float, required=True, help='Freestream Mach number')
    parser.add_argument(
        '--output-dir', default='/scratch/3D/designs_3d/test',
        help='Root output directory (subfolder per design_name)',
    )
    args = parser.parse_args()

    with open(args.params) as f:
        params = json.load(f)

    design_name = params.get('design_name', 'design')
    output_dir  = os.path.join(args.output_dir, design_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[generate_design] design : {design_name}")
    print(f"[generate_design] output : {output_dir}")

    results, geometry = run_design(params, args.aoa, args.mach, output_dir)

    cl  = float(np.squeeze(results.CL))
    cdi = float(np.squeeze(results.CDi))
    cm  = float(np.squeeze(results.CM))
    print(f"[generate_design] VLM    : CL={cl:.4f}  CDi={cdi:.5f}  "
          f"CM={cm:.4f}  L/D={cl/cdi:.2f}")

    json_path = save_results_json(params, args.aoa, args.mach, results, geometry, output_dir)
    png_path  = save_geometry_png(params, args.aoa, args.mach, results, geometry, output_dir)

    print(f"[generate_design] saved  : {json_path}")
    print(f"[generate_design] saved  : {png_path}")


if __name__ == '__main__':
    main()
