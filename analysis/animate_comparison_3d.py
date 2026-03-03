#!/usr/bin/env python3
"""
animate_comparison_3d.py — Side-by-side comparison GIF of two evolutionary runs.

Left  : unified reward plot with both runs overlaid (scatter + best-so-far).
Right : stacked Cp panels (top = run A, bottom = run B).

Usage:
    python animate_comparison_3d.py <dir_a> <dir_b> [step] [--label-a X] [--label-b Y]
"""

import os, sys, json, argparse
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
    'figure.dpi':        180,
}

C_A       = '#c0392b'
C_A_BEST  = '#922b21'
C_B       = '#2471a3'
C_B_BEST  = '#1a5276'
C_RING    = '#d4600e'
C_BASE    = '#888888'


def load_csv(results_dir):
    path = os.path.join(results_dir, 'results.csv')
    data = {'iteration': [], 'reward': [], 'best_reward': [], 'L_D': []}
    with open(path) as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            data['iteration'].append(int(row['iteration']))
            data['reward'].append(float(row['reward']))
            data['best_reward'].append(float(row['best_reward']))
            data['L_D'].append(float(row.get('L_D', 0)))
    for k in data:
        data[k] = np.array(data[k])
    return data


def crop_hifi_panel(geometry_png_path):
    img = Image.open(geometry_png_path)
    w, h = img.size
    return img.crop((int(w * 0.335), int(h * 0.14), int(w * 0.645), int(h * 0.96)))


def make_comparison(dir_a, dir_b, step=1, label_a='No Islands', label_b='5 Islands'):
    da = load_csv(dir_a)
    db = load_csv(dir_b)
    na, nb = len(da['iteration']), len(db['iteration'])
    n_max = max(na, nb)

    all_rewards = np.concatenate([da['reward'], db['reward']])
    p5, p95 = np.percentile(all_rewards, 5), np.percentile(all_rewards, 95)
    margin = max((p95 - p5) * 0.25, 2.0)
    y_lo, y_hi = p5 - margin, p95 + margin

    plt.rcParams.update(STYLE)

    frames = []
    frame_indices = list(range(0, n_max, step))
    if frame_indices[-1] != n_max - 1:
        frame_indices.append(n_max - 1)

    print(f"Rendering {len(frame_indices)} frames ...")

    for fi, up_to in enumerate(frame_indices):
        ka = min(up_to + 1, na)
        kb = min(up_to + 1, nb)

        fig = plt.figure(figsize=(14, 6.5), facecolor='white')
        gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1],
                              wspace=0.03, hspace=0.08)

        ax = fig.add_subplot(gs[:, 0])

        ax.scatter(da['iteration'][:ka], np.clip(da['reward'][:ka], y_lo, y_hi),
                   c=C_A, s=10, alpha=0.45, edgecolors='none', zorder=2)
        ax.scatter(db['iteration'][:kb], np.clip(db['reward'][:kb], y_lo, y_hi),
                   c=C_B, s=10, alpha=0.45, edgecolors='none', zorder=2)

        ax.plot(da['iteration'][:ka], np.clip(da['best_reward'][:ka], y_lo, y_hi),
                color=C_A_BEST, linewidth=1.8, zorder=3, label=f'{label_a} best')
        ax.plot(db['iteration'][:kb], np.clip(db['best_reward'][:kb], y_lo, y_hi),
                color=C_B_BEST, linewidth=1.8, zorder=3, label=f'{label_b} best')

        idx_a = up_to if up_to < na else na - 1
        idx_b = up_to if up_to < nb else nb - 1
        ax.scatter([da['iteration'][idx_a]], [np.clip(da['reward'][idx_a], y_lo, y_hi)],
                   s=80, facecolors='none', edgecolors=C_A_BEST, linewidths=2, zorder=5)
        ax.scatter([db['iteration'][idx_b]], [np.clip(db['reward'][idx_b], y_lo, y_hi)],
                   s=80, facecolors='none', edgecolors=C_B_BEST, linewidths=2, zorder=5)

        ax.axhline(0, color=C_BASE, linewidth=0.5, linestyle='--', alpha=0.6)
        ax.set_xlim(-0.5, n_max - 0.5)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(r'Iteration')
        ax.set_ylabel(r'Reward  ($C_L / C_{D_i} - 5.45$)')
        ax.legend(loc='upper left', frameon=True, fancybox=False,
                  edgecolor='#666', framealpha=1, borderpad=0.4)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        best_a = da['best_reward'][idx_a]
        best_b = db['best_reward'][idx_b]
        ax.set_title(
            f'Iteration {up_to}    '
            f'{label_a}: Best$={best_a:.1f}$    '
            f'{label_b}: Best$={best_b:.1f}$',
            fontweight='medium', pad=8)

        ax_top = fig.add_subplot(gs[0, 1])
        ax_top.axis('off')
        design_idx_a = min(up_to, na - 1)
        geom_a = os.path.join(dir_a, f'design_{design_idx_a}', 'geometry.png')
        if os.path.exists(geom_a):
            img_a = crop_hifi_panel(geom_a)
            ax_top.imshow(img_a)
        else:
            ax_top.text(0.5, 0.5, f'design_{design_idx_a}\n(no image)',
                        ha='center', va='center', fontsize=9,
                        transform=ax_top.transAxes, color='#888')
        r_a = da['reward'][design_idx_a]
        ld_a = da['L_D'][design_idx_a]
        ax_top.set_title(f'{label_a}  d{design_idx_a}  r={r_a:.1f}  L/D={ld_a:.1f}',
                         fontsize=8.5, color=C_A_BEST, fontweight='bold', pad=4)

        ax_bot = fig.add_subplot(gs[1, 1])
        ax_bot.axis('off')
        design_idx_b = min(up_to, nb - 1)
        geom_b = os.path.join(dir_b, f'design_{design_idx_b}', 'geometry.png')
        if os.path.exists(geom_b):
            img_b = crop_hifi_panel(geom_b)
            ax_bot.imshow(img_b)
        else:
            ax_bot.text(0.5, 0.5, f'design_{design_idx_b}\n(no image)',
                        ha='center', va='center', fontsize=9,
                        transform=ax_bot.transAxes, color='#888')
        r_b = db['reward'][design_idx_b]
        ld_b = db['L_D'][design_idx_b]
        ax_bot.set_title(f'{label_b}  d{design_idx_b}  r={r_b:.1f}  L/D={ld_b:.1f}',
                         fontsize=8.5, color=C_B_BEST, fontweight='bold', pad=4)

        fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.10)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(Image.fromarray(buf))
        plt.close(fig)

        if (fi + 1) % 20 == 0:
            print(f"  {fi + 1}/{len(frame_indices)} frames")

    for _ in range(10):
        frames.append(frames[-1])

    out_dir = os.path.dirname(dir_a)
    out_path = os.path.join(out_dir, 'comparison_evolution.gif')
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=150, loop=0, optimize=True)
    print(f"Saved -> {out_path}  ({len(frames)} frames, "
          f"{os.path.getsize(out_path)/1e6:.1f} MB)")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('dir_a')
    p.add_argument('dir_b')
    p.add_argument('step', type=int, nargs='?', default=5)
    p.add_argument('--label-a', default='No Islands')
    p.add_argument('--label-b', default='5 Islands')
    args = p.parse_args()
    make_comparison(args.dir_a, args.dir_b, args.step, args.label_a, args.label_b)
