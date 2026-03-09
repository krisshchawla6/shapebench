#!/usr/bin/env python3
"""
animate_blendednet_comparison.py — 3-way animated GIF comparing
Baseline vs TTT-25 vs TTT-50 for BlendedNet BWB optimisation.

Left column  : reward scatter + best-so-far line.
Right column : Cp_top bird's-eye view of the current best design.

Usage:
    python analysis/animate_blendednet_comparison.py
    python analysis/animate_blendednet_comparison.py --max-iter 200
"""

import os, sys, csv as csv_mod, argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

ROW_COLORS = ['#27ae60', '#c0392b', '#2471a3']
C_BEST = '#1f4e8c'
C_CURRENT = '#d4600e'


def load_run(results_dir, designs_subdir=None):
    """Load results.csv and return arrays.  Handles both baseline and TTT formats."""
    csv_path = os.path.join(results_dir, 'results.csv')
    if not os.path.exists(csv_path):
        return None
    iterations, rewards = [], []
    with open(csv_path) as f:
        for row in csv_mod.DictReader(f):
            try:
                iterations.append(int(row['iteration']))
                rewards.append(float(row['reward']))
            except (ValueError, KeyError):
                continue
    if not iterations:
        return None
    iterations = np.array(iterations)
    rewards = np.array(rewards)
    order = np.argsort(iterations)
    iterations, rewards = iterations[order], rewards[order]
    best = np.maximum.accumulate(rewards)
    img_base = os.path.join(results_dir, designs_subdir) if designs_subdir else results_dir
    return dict(iterations=iterations, rewards=rewards, best=best,
                n=len(iterations), dir=results_dir, img_base=img_base)


def get_top_image(img_base, design_idx):
    for subpath in [
        f'design_{design_idx}/save/sol/Cp_top.png',
        f'design_{design_idx}/save/sol/Cp_iso.png',
    ]:
        p = os.path.join(img_base, subpath)
        if os.path.exists(p):
            return Image.open(p)
    return None


def best_so_far_idx(rewards, up_to):
    return int(np.argmax(rewards[:up_to + 1]))


def make_comparison_gif(runs, output_path, max_frames=150):
    """
    All runs overlayed on a single reward plot (left).
    Right: Cp_top images of each run's current best, stacked vertically.
    """
    runs = [(l, d, c) for l, d, c in runs if d is not None]
    if not runs:
        print("No data to animate.")
        return
    n_runs = len(runs)
    n_total = max(d['n'] for _, d, _ in runs)
    if n_total == 0:
        return

    all_rewards = np.concatenate([d['rewards'] for _, d, _ in runs])
    valid = all_rewards[all_rewards > -9]
    if len(valid) < 2:
        p5, p95 = -12, 30
    else:
        p5, p95 = np.percentile(valid, 3), np.percentile(valid, 97)
    margin = max((p95 - p5) * 0.2, 2.0)
    y_lo, y_hi = p5 - margin, p95 + margin

    step = max(1, n_total // max_frames)
    frame_indices = list(range(0, n_total, step))
    if frame_indices[-1] != n_total - 1:
        frame_indices.append(n_total - 1)

    plt.rcParams.update(STYLE)
    frames = []
    print(f"Rendering {len(frame_indices)} frames for {n_runs} runs (overlayed) ...")

    for fi, up_to in enumerate(frame_indices):
        fig = plt.figure(figsize=(16, 6), facecolor='white')
        gs = fig.add_gridspec(1, 1 + n_runs, width_ratios=[2.0] + [1] * n_runs,
                              wspace=0.08)

        ax = fig.add_subplot(gs[0, 0])
        legend_entries = []

        for label, data, color in runs:
            k = min(up_to + 1, data['n'])
            if k == 0:
                continue
            x = np.arange(1, k + 1)
            ax.scatter(x, np.clip(data['rewards'][:k], y_lo, y_hi),
                       color=color, s=10, alpha=0.35, zorder=2)
            r_best = data['best'][k - 1]
            ax.plot(x, np.clip(data['best'][:k], y_lo, y_hi),
                    color=color, lw=2.0, zorder=3)
            ax.scatter([k], [np.clip(data['rewards'][k - 1], y_lo, y_hi)],
                       s=60, facecolors='none', edgecolors=color,
                       linewidths=2, zorder=5)
            legend_entries.append(
                plt.Line2D([0], [0], color=color, lw=2,
                           label=f'{label}  (best={r_best:.1f}, n={k})'))

        ax.set_xlim(0, n_total + 1)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Reward  (L/D)', fontsize=12)
        ax.set_title(f'BlendedNet — Iteration {min(up_to + 1, n_total)}',
                     fontsize=13, fontweight='bold', pad=8)
        ax.legend(handles=legend_entries, loc='lower right', frameon=True,
                  fancybox=False, edgecolor='#999', framealpha=0.95, fontsize=9)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        ax.grid(True, alpha=0.15)

        for col, (label, data, color) in enumerate(runs):
            ax_img = fig.add_subplot(gs[0, 1 + col])
            ax_img.axis('off')
            k = min(up_to + 1, data['n'])
            if k > 0:
                bi = best_so_far_idx(data['rewards'], k - 1)
                img = get_top_image(data['img_base'], data['iterations'][bi])
                if img is not None:
                    ax_img.imshow(img)
                short = label.replace('Baseline (Gemini)', 'Baseline')
                ax_img.set_title(f'{short}\nbest #{data["iterations"][bi]}',
                                 fontsize=8, fontweight='medium', color=color, pad=4)

        fig.subplots_adjust(left=0.05, right=0.99, top=0.90, bottom=0.10)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(Image.fromarray(buf))
        plt.close(fig)

        if (fi + 1) % 25 == 0:
            print(f"  {fi + 1}/{len(frame_indices)} frames")

    for _ in range(10):
        frames.append(frames[-1])

    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=140, loop=0, optimize=True)
    sz = os.path.getsize(output_path) / 1e6
    print(f"  -> {output_path}  ({len(frames)} frames, {sz:.1f} MB)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-iter', type=int, default=200,
                        help='Cap iteration count for the animation')
    parser.add_argument('--max-frames', type=int, default=150)
    args = parser.parse_args()

    env_base = os.path.join(REPO, 'environments', 'BlendedNet', 'results')
    out_dir = os.path.join(REPO, 'analysis', 'plots')
    os.makedirs(out_dir, exist_ok=True)

    baseline = load_run(os.path.join(env_base, 'run_blended_net_3d'))
    ttt25 = load_run(os.path.join(env_base, 'test_time_discovery_750'), designs_subdir='designs')
    ttt50 = load_run(os.path.join(env_base, 'test_time_discovery_50steps'), designs_subdir='designs')

    if args.max_iter:
        for d in [baseline, ttt25, ttt50]:
            if d is not None:
                mask = np.arange(d['n']) < args.max_iter
                for key in ('iterations', 'rewards', 'best'):
                    d[key] = d[key][mask]
                d['n'] = int(mask.sum())

    runs = [
        ('Baseline (Gemini)', baseline, ROW_COLORS[0]),
        ('TTT 25-epoch',      ttt25,    ROW_COLORS[1]),
        ('TTT 50-epoch',      ttt50,    ROW_COLORS[2]),
    ]

    make_comparison_gif(runs, os.path.join(out_dir, 'BlendedNet_comparison.gif'),
                        max_frames=args.max_frames)
