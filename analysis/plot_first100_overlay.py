#!/usr/bin/env python3
"""Overlay first-100-iteration reward curves for TTT 25-epoch, TTT 50-epoch, and baseline."""

import os, csv
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STYLE = {
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset':  'cm',
    'font.size':         11,
    'axes.labelsize':    13,
    'axes.titlesize':    14,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'legend.fontsize':   10,
    'axes.linewidth':    0.7,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'figure.dpi':        220,
}

C_TTT25    = '#c0392b'
C_TTT50    = '#2471a3'
C_BASELINE = '#27ae60'


def load_rewards(csv_path, max_iter=100):
    if not os.path.exists(csv_path):
        return np.array([]), np.array([]), np.array([])
    # Deduplicate: last row per iteration = most recent run
    rows_by_iter = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                rows_by_iter[int(row['iteration'])] = float(row['reward'])
            except (ValueError, KeyError):
                continue
    iters = np.array(sorted(rows_by_iter.keys()))
    rewards = np.array([rows_by_iter[i] for i in iters])
    mask = np.arange(len(iters)) < max_iter
    iters, rewards = iters[mask], rewards[mask]
    return iters, rewards, np.maximum.accumulate(rewards)


def rolling(arr, w=15):
    if len(arr) < w:
        return np.arange(len(arr)), arr
    k = np.ones(w) / w
    rm = np.convolve(arr, k, mode='valid')
    return np.arange(w // 2, w // 2 + len(rm)), rm


def make_overlay(env_name, ttt_csv, baseline_csv, ylabel, out_path):
    plt.rcParams.update(STYLE)
    runs = []
    for path, label, color in [
        (ttt_csv,      'TTT 1000',          C_TTT25),
        (baseline_csv, 'Baseline (Gemini)', C_BASELINE),
    ]:
        if os.path.exists(path):
            iters, rewards, best = load_rewards(path, max_iter=100)
            if len(iters) > 0:
                runs.append((label, iters, rewards, best, color))
    if not runs:
        print(f"  No data for {env_name}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    for label, iters, rewards, best, color in runs:
        x = np.arange(1, len(rewards) + 1)
        ax1.scatter(x, rewards, color=color, s=12, alpha=0.35, zorder=2)
        ax1.plot(x, best, color=color, linewidth=2.2,
                 label=f'{label} (best: {best[-1]:.2f})', zorder=3)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel(ylabel)
    ax1.set_title(f'{env_name} — Reward (first 100 iterations)')
    ax1.set_xlim(0, 101)
    ax1.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='#999', framealpha=0.95)
    ax1.grid(True, alpha=0.15)
    for sp in ['top', 'right']:
        ax1.spines[sp].set_visible(False)

    for label, iters, rewards, best, color in runs:
        rx, rm = rolling(rewards, w=15)
        ax2.plot(rx + 1, rm, color=color, linewidth=2, label=label, alpha=0.9)
        ax2.fill_between(rx + 1, rm, alpha=0.08, color=color)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel(f'Rolling Mean {ylabel} (window=15)')
    ax2.set_title(f'{env_name} — Smoothed Reward Trend')
    ax2.set_xlim(0, 101)
    ax2.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='#999', framealpha=0.95)
    ax2.grid(True, alpha=0.15)
    for sp in ['top', 'right']:
        ax2.spines[sp].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out_path}")


if __name__ == '__main__':
    out_dir = os.path.join(REPO, 'analysis', 'plots')
    os.makedirs(out_dir, exist_ok=True)
    make_overlay(
        'BlendedNet',
        os.path.join(REPO, 'environments/BlendedNet/results/ttt_1000/results.csv'),
        os.path.join(REPO, 'environments/BlendedNet/results/run_blended_net_3d/results.csv'),
        ylabel='Reward (L/D)',
        out_path=os.path.join(out_dir, 'BlendedNet_first100_overlay.png'),
    )
    make_overlay(
        'Delta Wing 2-Point',
        os.path.join(REPO, 'environments/vlm_3d_2pt/results/ttt_1000/results.csv'),
        os.path.join(REPO, 'environments/vlm_3d_2pt/results/run_delta_wing_3d_750/results.csv'),
        ylabel='Reward',
        out_path=os.path.join(out_dir, 'vlm_3d_2pt_first100_overlay.png'),
    )
