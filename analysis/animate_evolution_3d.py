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

import os, sys, json
import csv as csv_mod
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

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

C_BEST     = '#1f4e8c'
C_CURRENT  = '#d4600e'
C_LINEAGE  = '#b0b0b0'
C_BASELINE = '#888888'


def load_results_csv(results_dir):
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
    lineage_path = os.path.join(results_dir, 'lineage.json')
    if os.path.exists(lineage_path):
        with open(lineage_path) as f:
            lineage = json.load(f)
        for entry in lineage:
            cid = entry.get('id')
            pid = entry.get('parent_id')
            if isinstance(cid, int):
                parent_of[cid] = pid
    return parent_of


def crop_hifi_panel(geometry_png_path):
    img = Image.open(geometry_png_path)
    w, h = img.size
    return img.crop((int(w * 0.335), int(h * 0.14), int(w * 0.645), int(h * 0.96)))


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

    print(f"Rendering {len(frame_indices)} frames ...")

    for fi, up_to in enumerate(frame_indices):
        k = up_to + 1

        fig = plt.figure(figsize=(13, 5), facecolor='white')
        gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1], wspace=0.02)

        ax = fig.add_subplot(gs[0])

        for child in range(k):
            p = parent_of.get(child)
            if p is not None and isinstance(p, int) and p < k:
                ax.plot([p, child],
                        [np.clip(rewards[p], y_lo, y_hi),
                         np.clip(rewards[child], y_lo, y_hi)],
                        color=C_LINEAGE, linewidth=0.35, alpha=0.35, zorder=1)

        cols = [island_cmap[int(isl)] for isl in islands[:k]]
        ax.scatter(data['iteration'][:k],
                   np.clip(rewards[:k], y_lo, y_hi),
                   c=cols, s=16, alpha=0.75,
                   edgecolors='k', linewidths=0.2, zorder=2)

        ax.scatter([data['iteration'][up_to]],
                   [np.clip(rewards[up_to], y_lo, y_hi)],
                   s=90, facecolors='none', edgecolors=C_CURRENT,
                   linewidths=2, zorder=5)

        ax.plot(data['iteration'][:k],
                np.clip(data['best_reward'][:k], y_lo, y_hi),
                color=C_BEST, linewidth=1.4, zorder=3, label='Best reward')

        ax.axhline(0, color=C_BASELINE, linewidth=0.5, linestyle='--', alpha=0.6)

        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(r'Iteration')
        ax.set_ylabel(r'Reward  ($C_L / C_{D_i} - 5.45$)')
        handles = [plt.Line2D([0], [0], color=C_BEST, lw=1.4, label='Best reward')]
        if n_islands > 1:
            for isl in range(n_islands):
                handles.append(plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=island_cmap[isl], markersize=5,
                               label=f'Island {isl}'))
        ax.legend(handles=handles, loc='upper left', frameon=True, fancybox=False,
                  edgecolor='#666', framealpha=1, borderpad=0.4,
                  ncol=2 if n_islands > 1 else 1)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        r_now  = rewards[up_to]
        r_best = data['best_reward'][up_to]
        ld_now = data['L_D'][up_to]
        isl_now = int(islands[up_to])
        isl_str = f'  (Island {isl_now})' if n_islands > 1 else ''
        ax.set_title(
            f'Iteration {up_to}{isl_str}    '
            r'$r={' + f'{r_now:.1f}' + r'}$'
            f'    Best $={r_best:.1f}$'
            f'    $L/D={ld_now:.1f}$',
            fontweight='medium', pad=8)

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

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(Image.fromarray(buf))
        plt.close(fig)

        if (fi + 1) % 20 == 0:
            print(f"  {fi + 1}/{len(frame_indices)} frames")

    for _ in range(8):
        frames.append(frames[-1])

    out_path = os.path.join(results_dir, 'evolution.gif')
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=150, loop=0, optimize=True,
    )
    print(f"Saved -> {out_path}  ({len(frames)} frames, "
          f"{os.path.getsize(out_path)/1e6:.1f} MB)")


def make_summary_plot(results_dir):
    data = load_results_csv(results_dir)
    n = len(data['iteration'])
    if n == 0:
        return

    iters = data['iteration']
    rewards = data['reward']
    best = data['best_reward']
    islands = data['island']
    ld = data['L_D']
    n_islands = int(islands.max()) + 1
    cmap = plt.cm.Set2(np.linspace(0, 1, max(n_islands, 2)))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    title = os.path.basename(results_dir)
    fig.suptitle(f'{title} -- {n} iterations, {n_islands} island(s)',
                 fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    for isl in range(n_islands):
        m = islands == isl
        ax.scatter(iters[m], rewards[m], color=cmap[isl], alpha=0.6, s=20,
                   label=f'Island {isl}' if n_islands > 1 else 'Reward')
    ax.plot(iters, best, 'k-', lw=2, alpha=0.7, label='Best so far')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Reward')
    ax.set_title('Reward per Iteration')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(iters, best, 'r-', lw=2)
    ax.fill_between(iters, best, alpha=0.15, color='red')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Best Reward')
    ax.set_title('Best Reward Progression'); ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95,
            f'Final Best: {best[-1]:.2f}\nBest Iter: {iters[np.argmax(rewards)]}\nTotal: {n}',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    ax = axes[1, 0]
    if n_islands > 1:
        data_box = [ld[islands == isl] for isl in range(n_islands)]
        bp = ax.boxplot(data_box,
                        tick_labels=[f'Island {i}' for i in range(n_islands)],
                        patch_artist=True)
        for patch, color in zip(bp['boxes'], cmap):
            patch.set_facecolor(color)
        ax.set_title('L/D Distribution per Island')
    else:
        ax.hist(ld, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
        ax.set_title('L/D Distribution')
    ax.set_ylabel('L/D'); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.hist(rewards, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(best[-1], color='red', ls='--', lw=2, label=f'Best: {best[-1]:.2f}')
    ax.axvline(np.mean(rewards), color='orange', ls='--', lw=2, label=f'Mean: {np.mean(rewards):.2f}')
    ax.set_xlabel('Reward'); ax.set_ylabel('Count')
    ax.set_title('Reward Distribution'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(results_dir, 'summary_plot.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved -> {out}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python animate_evolution_3d.py <results_dir> [step]")
        sys.exit(1)
    rdir = sys.argv[1]
    st = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    make_summary_plot(rdir)
    make_animation(rdir, step=st)
