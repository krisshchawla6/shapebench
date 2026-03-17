#!/usr/bin/env python3
"""
animate_ttt_run.py — Single-run evolution GIF for TTT results.

Handles both BlendedNet (Cp_top.png) and Delta Wing (sup/geometry.png) designs.
Marks epoch boundaries (= Tinker gradient updates) as vertical lines.
Output GIF is saved inside the results directory.

Usage:
    python analysis/animate_ttt_run.py <results_dir> [--max-frames N]
    python analysis/animate_ttt_run.py environments/BlendedNet/results/ttt_1000
    python analysis/animate_ttt_run.py environments/vlm_3d_2pt/results/ttt_1000
"""

import os, sys, csv as csv_mod, json, argparse
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
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
    'axes.linewidth':    0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'figure.dpi':        180,
}

C_SCATTER = '#2471a3'
C_BEST    = '#c0392b'
C_EPOCH   = '#888888'


def load_results(results_dir):
    """Deduplicated load: last row per iteration = most recent run."""
    csv_path = os.path.join(results_dir, 'results.csv')
    rows_by_iter = {}
    with open(csv_path) as f:
        for row in csv_mod.DictReader(f):
            try:
                rows_by_iter[int(row['iteration'])] = {
                    'reward':  float(row['reward']),
                    'epoch':   int(row.get('epoch', 0)),
                    'elapsed': float(row.get('elapsed_s', 0)),
                }
            except (ValueError, KeyError):
                continue
    iters = sorted(rows_by_iter)
    rewards  = np.array([rows_by_iter[i]['reward']  for i in iters])
    epochs   = np.array([rows_by_iter[i]['epoch']   for i in iters])
    iters    = np.array(iters)
    best     = np.maximum.accumulate(rewards)
    return iters, rewards, best, epochs


def epoch_boundaries(epochs):
    """Return iteration indices where epoch number changes."""
    boundaries = []
    for i in range(1, len(epochs)):
        if epochs[i] != epochs[i - 1]:
            boundaries.append(i)
    return boundaries


def find_design_image(designs_dir, design_idx):
    """Try known image paths for BlendedNet and Delta Wing."""
    candidates = [
        f'design_{design_idx}/save/sol/Cp_top.png',
        f'design_{design_idx}/save/sol/Cp_iso.png',
        f'design_{design_idx}/sup/geometry.png',
        f'design_{design_idx}/sub/geometry.png',
    ]
    for c in candidates:
        p = os.path.join(designs_dir, c)
        if os.path.exists(p):
            return p
    return None


def make_gif(results_dir, max_frames=150):
    iters, rewards, best, epochs = load_results(results_dir)
    n = len(iters)
    if n == 0:
        print("No data found.")
        return

    designs_dir = os.path.join(results_dir, 'designs')
    ep_bounds = epoch_boundaries(epochs)
    n_epochs  = int(epochs.max()) + 1

    valid = rewards[rewards > -9]
    if len(valid) > 2:
        p3, p97 = np.percentile(valid, 3), np.percentile(valid, 97)
        margin = max((p97 - p3) * 0.15, 1.0)
        y_lo, y_hi = p3 - margin, p97 + margin
    else:
        y_lo, y_hi = rewards.min() - 1, rewards.max() + 1

    step = max(1, n // max_frames)
    frames_at = list(range(0, n, step))
    if frames_at[-1] != n - 1:
        frames_at.append(n - 1)

    plt.rcParams.update(STYLE)
    frames = []
    print(f"Rendering {len(frames_at)} frames ({n} iters, {n_epochs} epochs) ...")

    for fi, up_to in enumerate(frames_at):
        fig, (ax_rw, ax_img) = plt.subplots(
            1, 2, figsize=(14, 5.5),
            gridspec_kw={'width_ratios': [2, 1]},
            facecolor='white',
        )

        k = up_to + 1
        x = np.arange(1, k + 1)

        # Epoch boundary lines (gradient update points)
        for bi in ep_bounds:
            if bi < k:
                ax_rw.axvline(bi + 1, color=C_EPOCH, lw=0.7, alpha=0.45, zorder=1)

        ax_rw.scatter(x, np.clip(rewards[:k], y_lo, y_hi),
                      color=C_SCATTER, s=9, alpha=0.4, zorder=2)
        ax_rw.plot(x, np.clip(best[:k], y_lo, y_hi),
                   color=C_BEST, lw=2.0, zorder=3,
                   label=f'Best: {best[up_to]:.2f}')
        ax_rw.scatter([k], [np.clip(rewards[up_to], y_lo, y_hi)],
                      s=55, facecolors='none', edgecolors=C_SCATTER,
                      linewidths=1.8, zorder=5)

        ax_rw.set_xlim(0, n + 1)
        ax_rw.set_ylim(y_lo, y_hi)
        ax_rw.set_xlabel('Iteration')
        ax_rw.set_ylabel('Reward')
        ep_now = int(epochs[up_to])
        ax_rw.set_title(
            f'TTT Run — Iteration {k} / {n}   |   Epoch {ep_now} / {n_epochs - 1}',
            fontsize=11, fontweight='bold', pad=7,
        )
        ax_rw.legend(loc='lower right', frameon=True, fancybox=False,
                     edgecolor='#aaa', framealpha=0.95)
        # Epoch boundary legend
        ax_rw.plot([], [], color=C_EPOCH, lw=0.9, alpha=0.6,
                   label=f'Gradient update ({len([b for b in ep_bounds if b < k])} so far)')
        ax_rw.legend(loc='lower right', frameon=True, fancybox=False,
                     edgecolor='#aaa', framealpha=0.95, fontsize=7.5)
        for sp in ['top', 'right']:
            ax_rw.spines[sp].set_visible(False)
        ax_rw.grid(True, alpha=0.12)

        # Best design image
        ax_img.axis('off')
        bi_idx = int(np.argmax(rewards[:k]))
        img_path = find_design_image(designs_dir, iters[bi_idx])
        if img_path:
            img = Image.open(img_path)
            ax_img.imshow(img)
            ax_img.set_title(
                f'Best design #{iters[bi_idx]}  (reward={best[up_to]:.2f})',
                fontsize=8.5, color=C_BEST, fontweight='medium', pad=4,
            )
        else:
            ax_img.text(0.5, 0.5, 'No image', ha='center', va='center',
                        transform=ax_img.transAxes, fontsize=9, color='#aaa')

        fig.tight_layout(pad=1.2)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(Image.fromarray(buf))
        plt.close(fig)

        if (fi + 1) % 30 == 0:
            print(f"  {fi + 1}/{len(frames_at)} frames")

    # Hold last frame
    for _ in range(8):
        frames.append(frames[-1])

    out = os.path.join(results_dir, 'evolution.gif')
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=120, loop=0, optimize=True)
    sz = os.path.getsize(out) / 1e6
    print(f"  -> {out}  ({len(frames)} frames, {sz:.1f} MB)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', help='Path to TTT results directory')
    parser.add_argument('--max-frames', type=int, default=150)
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.isabs(results_dir):
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(repo, results_dir)

    make_gif(results_dir, max_frames=args.max_frames)
