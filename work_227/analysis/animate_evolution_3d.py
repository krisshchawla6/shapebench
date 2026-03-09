#!/usr/bin/env python3
"""
animate_evolution_3d.py — Academic-grade animated GIF of 3D delta-wing
LLM-driven evolutionary optimisation.

Left  : reward progression (builds up iteration by iteration).
Right : VortexNet-corrected (high-fidelity) Cp distribution for the
        current iteration, cropped from geometry.png.

Usage:
    python animate_evolution_3d.py <results_dir> [step]
"""

import os, sys, re, json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.collections import PatchCollection
from PIL import Image

# ── Academic style ───────────────────────────────────────────────────
STYLE = {
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset':  'cm',
    'font.size':         9,
    'axes.labelsize':    11,
    'axes.titlesize':    11,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   8.5,
    'axes.linewidth':    0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'figure.dpi':        200,
}

C_POS      = '#1a7a2e'   # positive-reward scatter
C_NEG      = '#b02020'   # negative-reward scatter
C_BEST     = '#1f4e8c'   # best-so-far line
C_CURRENT  = '#d4600e'   # ring around current design
C_LINEAGE  = '#b0b0b0'   # parent→child lines
C_BASELINE = '#888888'   # zero-line


# ── Data helpers ─────────────────────────────────────────────────────
def load_results_csv(results_dir):
    import csv as csv_mod
    csv_path = os.path.join(results_dir, 'results.csv')
    data = {'iteration': [], 'design': [], 'reward': [], 'best_reward': [],
            'CL': [], 'CDi': [], 'L_D': [], 'island': []}
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            data['iteration'].append(int(row['iteration']))
            data['design'].append(row['design'])
            data['reward'].append(float(row['reward']))
            data['best_reward'].append(float(row['best_reward']))
            data['CL'].append(float(row.get('CL', 0)))
            data['CDi'].append(float(row.get('CDi', 0)))
            data['L_D'].append(float(row.get('L_D', 0)))
            data['island'].append(int(row['island']) if 'island' in row else 0)
    for k in data:
        if k != 'design':
            data[k] = np.array(data[k])
    return data


def reconstruct_lineage(results_dir, n_designs):
    parent_of = {}
    for i in range(n_designs):
        ctx_file = os.path.join(results_dir, f'design_{i}', 'context', 'context.txt')
        if not os.path.exists(ctx_file):
            parent_of[i] = None
            continue
        with open(ctx_file) as f:
            text = f.read()
        if 'No previous designs available' in text:
            parent_of[i] = None
            continue
        m = re.search(r'Design 1:\s*\n\s*- Parameters:.*?le_sweep=([\d.]+)', text)
        if not m:
            parent_of[i] = None
            continue
        parent_sweep = float(m.group(1))
        best_match, best_dist = None, float('inf')
        for j in range(n_designs):
            if j == i:
                continue
            jpath = os.path.join(results_dir, f'design_{j}.json')
            if not os.path.exists(jpath):
                continue
            with open(jpath) as jf:
                d = json.load(jf)
            dist = abs(d.get('le_sweep', 0) - parent_sweep)
            if dist < best_dist:
                best_dist = dist
                best_match = j
        parent_of[i] = best_match if best_dist < 0.001 else None
    return parent_of


def crop_hifi_panel(geometry_png_path):
    """Return PIL Image of only the VortexNet-corrected (middle) Cp panel."""
    img = Image.open(geometry_png_path)
    w, h = img.size
    # geometry.png layout: [VLM | VortexNet Corrected | Results Table]
    # Crop the centre panel, excluding the original dark-themed title and
    # any bleed from the adjacent panels / results table.
    left   = int(w * 0.335)
    right  = int(w * 0.645)
    top    = int(h * 0.14)
    bottom = int(h * 0.96)
    return img.crop((left, top, right, bottom))


# ── Main renderer ────────────────────────────────────────────────────
def make_animation(results_dir, step=1):
    data = load_results_csv(results_dir)
    n = len(data['iteration'])
    if n == 0:
        print("No data.")
        return

    parent_of = reconstruct_lineage(results_dir, n)
    rewards = data['reward'].copy()
    islands = data['island']
    n_islands = int(islands.max()) + 1
    island_cmap = plt.cm.Set2(np.linspace(0, 1, max(n_islands, 2)))

    p5, p95 = np.percentile(rewards, 5), np.percentile(rewards, 95)
    margin = max((p95 - p5) * 0.25, 2.0)
    y_lo, y_hi = p5 - margin, p95 + margin

    plt.rcParams.update(STYLE)

    frames = []
    frame_indices = list(range(0, n, step))
    if frame_indices[-1] != n - 1:
        frame_indices.append(n - 1)

    print(f"Rendering {len(frame_indices)} frames …")

    for fi, up_to in enumerate(frame_indices):
        k = up_to + 1

        fig = plt.figure(figsize=(13, 5), facecolor='white')
        gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1], wspace=0.02)

        # ── Left: reward chart ───────────────────────────────────────
        ax = fig.add_subplot(gs[0])

        # Lineage arcs
        for child in range(k):
            p = parent_of.get(child)
            if p is not None and p < k:
                ax.plot([p, child],
                        [np.clip(rewards[p], y_lo, y_hi),
                         np.clip(rewards[child], y_lo, y_hi)],
                        color=C_LINEAGE, linewidth=0.35, alpha=0.35, zorder=1)

        # Scatter colored by island
        cols = [island_cmap[int(isl)] for isl in islands[:k]]
        ax.scatter(data['iteration'][:k],
                   np.clip(rewards[:k], y_lo, y_hi),
                   c=cols, s=16, alpha=0.75,
                   edgecolors='k', linewidths=0.2, zorder=2)

        # Current iteration highlight
        ax.scatter([data['iteration'][up_to]],
                   [np.clip(rewards[up_to], y_lo, y_hi)],
                   s=90, facecolors='none', edgecolors=C_CURRENT,
                   linewidths=2, zorder=5)

        # Best-so-far
        ax.plot(data['iteration'][:k],
                np.clip(data['best_reward'][:k], y_lo, y_hi),
                color=C_BEST, linewidth=1.4, zorder=3,
                label='Best reward')

        # Baseline
        ax.axhline(0, color=C_BASELINE, linewidth=0.5, linestyle='--', alpha=0.6)

        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(r'Iteration')
        ax.set_ylabel(r'Reward  ($C_L / C_{D_i} - 5.45$)')
        handles = [plt.Line2D([0], [0], color=C_BEST, lw=1.4, label='Best reward')]
        for isl in range(n_islands):
            handles.append(plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=island_cmap[isl], markersize=5,
                           label=f'Island {isl}'))
        ax.legend(handles=handles, loc='upper left', frameon=True, fancybox=False,
                  edgecolor='#666', framealpha=1, borderpad=0.4, ncol=2)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        # Title with metrics
        r_now  = rewards[up_to]
        r_best = data['best_reward'][up_to]
        ld_now = data['L_D'][up_to]
        isl_now = int(islands[up_to])
        ax.set_title(
            f'Iteration {up_to}  (Island {isl_now})    '
            r'$r={' + f'{r_now:.1f}' + r'}$'
            f'    Best $={r_best:.1f}$'
            f'    $L/D={ld_now:.1f}$',
            fontweight='medium', pad=8)

        # ── Right: VortexNet Cp image ────────────────────────────────
        ax2 = fig.add_subplot(gs[1])
        ax2.axis('off')

        geom_png = os.path.join(results_dir, f'design_{up_to}', 'geometry.png')
        if os.path.exists(geom_png):
            hifi = crop_hifi_panel(geom_png)
            ax2.imshow(hifi)
            ax2.set_title(
                r'VortexNet Corrected $C_p$' + f'   (design {up_to})',
                fontsize=9, fontweight='medium', pad=6)
        else:
            ax2.text(0.5, 0.5, f'design_{up_to}\n(no image)',
                     ha='center', va='center', fontsize=10,
                     transform=ax2.transAxes, color='#888')

        fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.14, wspace=0.04)

        # Render to PIL
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(Image.fromarray(buf))
        plt.close(fig)

        if (fi + 1) % 10 == 0:
            print(f"  {fi + 1}/{len(frame_indices)} frames")

    # Hold final frame
    for _ in range(10):
        frames.append(frames[-1])

    out_path = os.path.join(results_dir, 'evolution.gif')
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=150, loop=0, optimize=True,
    )
    print(f"Saved → {out_path}  ({len(frames)} frames, "
          f"{os.path.getsize(out_path)/1e6:.1f} MB)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python animate_evolution_3d.py <results_dir> [step]")
        sys.exit(1)
    make_animation(sys.argv[1], step=int(sys.argv[2]) if len(sys.argv) > 2 else 1)
