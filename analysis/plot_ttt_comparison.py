#!/usr/bin/env python3
"""
plot_ttt_comparison.py — Comparison plots for TTT discovery runs.

Handles both TTT CSV format (iteration,epoch,group,rollout,reward,best_reward,elapsed_s)
and baseline CSV format (iteration,design,reward,best_reward,...).

Usage:
    python analysis/plot_ttt_comparison.py
    python analysis/plot_ttt_comparison.py --env BlendedNet
"""

import os, sys, argparse, csv
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STYLE = {
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset':  'cm',
    'font.size':         10,
    'axes.labelsize':    12,
    'axes.titlesize':    12,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
    'axes.linewidth':    0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'figure.dpi':        200,
}

COLORS = {
    'ttt_25':   '#c0392b',
    'ttt_50':   '#2471a3',
    'baseline': '#27ae60',
    'baseline2':'#8e44ad',
}


def load_results(results_dir):
    csv_path = os.path.join(results_dir, 'results.csv')
    if not os.path.exists(csv_path):
        return None

    iterations, rewards = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                it = int(row['iteration'])
                rw = float(row['reward'])
                iterations.append(it)
                rewards.append(rw)
            except (ValueError, KeyError):
                continue

    if not iterations:
        return None

    iterations = np.array(iterations)
    rewards = np.array(rewards)
    order = np.argsort(iterations)
    iterations = iterations[order]
    rewards = rewards[order]
    best_so_far = np.maximum.accumulate(rewards)

    return {
        'iterations': iterations,
        'rewards': rewards,
        'best': best_so_far,
        'n': len(iterations),
        'dir': results_dir,
    }


def rolling_mean(arr, window=20):
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='valid')


def plot_env_comparison(runs, env_name, output_dir):
    plt.rcParams.update(STYLE)
    n_runs = len(runs)
    if n_runs == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, data, color in runs:
        ax.plot(np.arange(1, data['n'] + 1), data['best'],
                color=color, linewidth=2, label=f'{label} (best: {data["best"][-1]:.2f})',
                alpha=0.9)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best Reward')
    ax.set_title(f'{env_name} — Best Reward Progression')
    ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='#999', framealpha=0.95)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = os.path.join(output_dir, f'{env_name}_best_reward.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out}")

    fig, axes = plt.subplots(1, n_runs, figsize=(5.5 * n_runs, 5), squeeze=False)
    for idx, (label, data, color) in enumerate(runs):
        ax = axes[0, idx]
        x = np.arange(1, data['n'] + 1)
        ax.scatter(x, data['rewards'], color=color, s=8, alpha=0.4, zorder=2)
        ax.plot(x, data['best'], color='black', linewidth=1.8, zorder=3,
                label=f'Best: {data["best"][-1]:.2f}')
        if len(data['rewards']) >= 20:
            rm = rolling_mean(data['rewards'], 20)
            ax.plot(np.arange(10, 10 + len(rm)), rm,
                    color=color, linewidth=1.5, linestyle='--',
                    alpha=0.8, zorder=4, label='Rolling mean (20)')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Reward')
        ax.set_title(f'{label}  ({data["n"]} iters)', fontweight='medium')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.2)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
    fig.suptitle(f'{env_name} — Reward per Iteration', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = os.path.join(output_dir, f'{env_name}_reward_scatter.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out}")

    fig, ax = plt.subplots(figsize=(10, 5))
    valid_runs = [(l, d, c) for l, d, c in runs if d['n'] > 5]
    if valid_runs:
        all_rewards = [d['rewards'][d['rewards'] > -9] for _, d, _ in valid_runs]
        labels_box = [f'{l}\n(n={d["n"]})' for l, d, _ in valid_runs]
        colors_box = [c for _, _, c in valid_runs]
        non_empty = [(r, l, c) for r, l, c in zip(all_rewards, labels_box, colors_box) if len(r) > 0]
        if non_empty:
            rewards_list, labels_list, colors_list = zip(*non_empty)
            bp = ax.boxplot(rewards_list, tick_labels=labels_list, patch_artist=True,
                            widths=0.5, showfliers=True,
                            flierprops=dict(marker='.', markersize=3, alpha=0.4))
            for patch, color in zip(bp['boxes'], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.3)
            for i, (rewards_arr, color) in enumerate(zip(rewards_list, colors_list)):
                jitter = np.random.normal(0, 0.04, len(rewards_arr))
                ax.scatter(np.full_like(rewards_arr, i + 1) + jitter, rewards_arr,
                           color=color, s=6, alpha=0.3, zorder=5)
    ax.set_ylabel('Reward')
    ax.set_title(f'{env_name} — Reward Distribution (failures excluded)')
    ax.grid(True, alpha=0.2, axis='y')
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = os.path.join(output_dir, f'{env_name}_reward_distribution.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out}")

    fig, ax = plt.subplots(figsize=(10, 2 + 0.4 * n_runs))
    ax.axis('off')
    headers = ['Run', 'Iters', 'Best', 'Mean', 'Median', 'Failures', 'Success %']
    rows = []
    for label, data, color in runs:
        valid = data['rewards'][data['rewards'] > -9]
        failures = int(np.sum(data['rewards'] <= -9))
        rows.append([
            label, str(data['n']),
            f'{data["best"][-1]:.2f}',
            f'{np.mean(valid):.2f}' if len(valid) > 0 else 'N/A',
            f'{np.median(valid):.2f}' if len(valid) > 0 else 'N/A',
            str(failures),
            f'{100 * len(valid) / data["n"]:.0f}%' if data['n'] > 0 else 'N/A',
        ])
    table = ax.table(cellText=rows, colLabels=headers, loc='center',
                     cellLoc='center', colColours=['#f0f0f0'] * len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for i, (_, _, color) in enumerate(runs):
        table[i + 1, 0].set_facecolor(color)
        table[i + 1, 0].set_text_props(color='white', fontweight='bold')
    ax.set_title(f'{env_name} — Summary Statistics', fontsize=13, fontweight='bold', pad=20)
    fig.tight_layout()
    out = os.path.join(output_dir, f'{env_name}_summary.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out}")


def discover_runs(env_name):
    results_base = os.path.join(REPO_ROOT, 'environments', env_name, 'results')
    runs = []
    ttt_dirs = {
        'test_time_discovery_750':     ('TTT 25-epoch', COLORS['ttt_25']),
        'test_time_discovery_50steps': ('TTT 50-epoch', COLORS['ttt_50']),
    }
    for dirname, (label, color) in ttt_dirs.items():
        data = load_results(os.path.join(results_base, dirname))
        if data:
            runs.append((label, data, color))
    if env_name == 'BlendedNet':
        baselines = [('run_blended_net_3d', 'Baseline (Gemini)', COLORS['baseline'])]
    elif env_name == 'vlm_3d_2pt':
        baselines = [('run_delta_wing_3d_750', 'Baseline (Gemini)', COLORS['baseline'])]
    else:
        baselines = []
    for dirname, label, color in baselines:
        data = load_results(os.path.join(results_base, dirname))
        if data:
            runs.append((label, data, color))
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, nargs='*')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()
    envs = args.env or ['BlendedNet', 'vlm_3d_2pt']
    output_dir = args.output_dir or os.path.join(REPO_ROOT, 'analysis', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    for env_name in envs:
        print(f"\n{'='*60}\n  {env_name}\n{'='*60}")
        runs = discover_runs(env_name)
        if not runs:
            print(f"  No results found for {env_name}")
            continue
        for label, data, _ in runs:
            print(f"  {label}: {data['n']} iters, best={data['best'][-1]:.2f}")
        plot_env_comparison(runs, env_name, output_dir)
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
