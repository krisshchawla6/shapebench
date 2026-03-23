#!/usr/bin/env python3
"""
plot_ttt_comparison.py — Comparison plots for TTT discovery runs.

Handles both TTT CSV format (iteration,epoch,group,rollout,reward,best_reward,elapsed_s)
and baseline CSV format (iteration,design,reward,best_reward,...).

Usage:
    python analysis/plot_ttt_comparison.py
    python analysis/plot_ttt_comparison.py --env BlendedNet
"""

import os, sys, argparse, csv, json
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
    'pso':      '#e67e22',
}


def load_results(results_dir):
    csv_path = os.path.join(results_dir, 'results.csv')
    if not os.path.exists(csv_path):
        return None

    # For multi-sample runs, keep max reward per iteration.
    rows_by_iter = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                it = int(row['iteration'])
                rw = float(row['reward'])
                if it not in rows_by_iter or rw > rows_by_iter[it]:
                    rows_by_iter[it] = rw
            except (ValueError, KeyError):
                continue

    if not rows_by_iter:
        return None

    iterations = np.array(sorted(rows_by_iter.keys()))
    rewards = np.array([rows_by_iter[i] for i in iterations])
    best_so_far = np.maximum.accumulate(rewards)

    return {
        'iterations': iterations,
        'rewards': rewards,
        'best': best_so_far,
        'n': len(iterations),
        'dir': results_dir,
    }


def load_results_from_design_jsons(results_dir):
    """Load rewards by scanning iter_X_sY/save/results.json files."""
    import re, glob as _glob
    pattern = os.path.join(results_dir, 'iter_*_s*/save/results.json')
    files = _glob.glob(pattern)
    if not files:
        return None

    rows_by_iter = {}
    for fpath in files:
        m = re.search(r'iter_(\d+)_s\d+', fpath)
        if not m:
            continue
        it = int(m.group(1))
        try:
            with open(fpath) as f:
                data = json.load(f)
            rw = float(data['reward'])
            if it not in rows_by_iter or rw > rows_by_iter[it]:
                rows_by_iter[it] = rw
        except Exception:
            continue

    if not rows_by_iter:
        return None

    iterations = np.array(sorted(rows_by_iter.keys()))
    rewards = np.array([rows_by_iter[i] for i in iterations])
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


def plot_env_comparison(runs, env_name, output_dir, tag=None, abs_reward=False):
    plt.rcParams.update(STYLE)
    n_runs = len(runs)
    if n_runs == 0:
        return
    tag = tag or env_name

    def _transform(arr):
        return np.abs(arr) if abs_reward else arr

    ylabel_reward = '|Reward| (log scale)' if abs_reward else 'Reward'
    ylabel_best   = 'Best |Reward| (log scale)' if abs_reward else 'Best Reward'
    # For abs: best = minimum |reward| (least drag). Legend shows smallest value.
    def _best_label(data):
        v = float(np.abs(data['best'][-1])) if abs_reward else float(data['best'][-1])
        return f'{v:.4f}' if abs_reward else f'{v:.2f}'

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, data, color in runs:
        y = _transform(data['best'])
        ax.plot(np.arange(1, data['n'] + 1), y,
                color=color, linewidth=2, label=f'{label} (best: {_best_label(data)})',
                alpha=0.9)
    ax.set_xlabel('Iteration (log scale)')
    ax.set_ylabel(ylabel_best)
    ax.set_title(f'{env_name} — Best Reward Progression')
    ax.set_xscale('log')
    if abs_reward:
        ax.set_yscale('log')
        ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='#999', framealpha=0.95)
    else:
        ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='#999', framealpha=0.95)
    ax.grid(True, alpha=0.2, which='both')
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = os.path.join(output_dir, f'{tag}_best_reward.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out}")

    fig, axes = plt.subplots(1, n_runs, figsize=(5.5 * n_runs, 5), squeeze=False)
    for idx, (label, data, color) in enumerate(runs):
        ax = axes[0, idx]
        x = np.arange(1, data['n'] + 1)
        r = _transform(data['rewards'])
        b = _transform(data['best'])
        ax.scatter(x, r, color=color, s=8, alpha=0.4, zorder=2)
        ax.plot(x, b, color='black', linewidth=1.8, zorder=3,
                label=f'Best: {_best_label(data)}')
        if len(r) >= 20:
            rm = rolling_mean(r, 20)
            ax.plot(np.arange(10, 10 + len(rm)), rm,
                    color=color, linewidth=1.5, linestyle='--',
                    alpha=0.8, zorder=4, label='Rolling mean (20)')
        ax.set_xlabel('Iteration (log scale)')
        ax.set_ylabel(ylabel_reward)
        ax.set_title(f'{label}  ({data["n"]} iters)', fontweight='medium')
        ax.set_xscale('log')
        if abs_reward:
            ax.set_yscale('log')
        ax.legend(loc='upper right' if abs_reward else 'lower right', fontsize=8)
        ax.grid(True, alpha=0.2, which='both')
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
    fig.suptitle(f'{env_name} — Reward per Iteration', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = os.path.join(output_dir, f'{tag}_reward_scatter.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out}")

    fig, ax = plt.subplots(figsize=(10, 5))
    valid_runs = [(l, d, c) for l, d, c in runs if d['n'] > 5]
    if valid_runs:
        all_rewards = [_transform(d['rewards'][d['rewards'] > -9]) for _, d, _ in valid_runs]
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
    ax.set_ylabel(ylabel_reward)
    ax.set_title(f'{env_name} — Reward Distribution (failures excluded)')
    if abs_reward:
        ax.set_yscale('log')
    ax.grid(True, alpha=0.2, axis='y')
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = os.path.join(output_dir, f'{tag}_reward_distribution.png')
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
    out = os.path.join(output_dir, f'{tag}_summary.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out}")


DRIVAER_RUNS = [
    ('run_drivaer_star_3d_islands',     'Islands (n=1000)',       COLORS['ttt_25']),
    ('run_drivaer_star_3d_islands_800', 'Islands (n=800 start)',  COLORS['ttt_50']),
    ('v3',                              'v3 Dynamic Optimizer',   COLORS['baseline']),
    ('geo_b30_f001',                    'Batch-30 Geom',          COLORS['baseline2']),
    ('GA/particle_swarm',               'PSO (GA)',               COLORS['pso']),
]

# BlendedNet main runs (L/D ratio reward, same scale)
BLENDEDNET_RUNS = [
    ('ttt_1000',           'TTT 1000',             COLORS['ttt_25']),
    ('v3',                 'v3 Dynamic Optimizer', COLORS['baseline']),
    ('GA/particle_swarm',  'PSO (GA)',              COLORS['pso']),
]

# ShapeBench runs — separate base path, different reward scale per benchmark
SHAPEBENCH_GROUPS = {
    'BlendedNet shapebecnh1': {
        'base': os.path.join(REPO_ROOT, 'environments', 'BlendedNet', 'results', 'shapebecnh1'),
        'runs': [
            ('pso', 'PSO',                COLORS['ttt_25']),
            ('v3',  'v3 Dynamic Optimizer', COLORS['baseline']),
        ],
    },
    'BlendedNet shapebench5': {
        'base': os.path.join(REPO_ROOT, 'environments', 'BlendedNet', 'results', 'shapebench5'),
        'runs': [
            ('pso', 'PSO',                COLORS['ttt_25']),
            ('v3',  'v3 Dynamic Optimizer', COLORS['baseline']),
        ],
    },
}


def discover_runs(env_name):
    runs = []

    if env_name == 'DrivAer_Star':
        results_base = os.path.join(REPO_ROOT, 'environments', 'DrivAer_Star', 'results')
        for dirname, label, color in DRIVAER_RUNS:
            d = os.path.join(results_base, dirname)
            data = load_results(d) or load_results_from_design_jsons(d)
            if data:
                runs.append((label, data, color))
        return runs

    if env_name == 'BlendedNet':
        results_base = os.path.join(REPO_ROOT, 'environments', 'BlendedNet', 'results')
        for dirname, label, color in BLENDEDNET_RUNS:
            data = load_results(os.path.join(results_base, dirname))
            if data:
                runs.append((label, data, color))
        return runs

    if env_name in SHAPEBENCH_GROUPS:
        grp = SHAPEBENCH_GROUPS[env_name]
        for dirname, label, color in grp['runs']:
            data = load_results(os.path.join(grp['base'], dirname))
            if data:
                runs.append((label, data, color))
        return runs

    if env_name == 'vlm_3d_2pt':
        results_base = os.path.join(REPO_ROOT, 'environments', 'vlm_3d_2pt', 'results')
        for dirname, label, color in [
            ('ttt_1000',              'TTT 1000',          COLORS['ttt_25']),
            ('run_delta_wing_3d_750', 'Baseline (Gemini)', COLORS['baseline']),
        ]:
            data = load_results(os.path.join(results_base, dirname))
            if data:
                runs.append((label, data, color))
        return runs

    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, nargs='*')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    default_envs = [
        'BlendedNet',
        'BlendedNet shapebecnh1',
        'BlendedNet shapebench5',
        'vlm_3d_2pt',
        'DrivAer_Star',
    ]
    envs = args.env or default_envs
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
        tag = env_name.replace(' ', '_')
        abs_r = env_name in ('BlendedNet shapebecnh1', 'BlendedNet shapebench5', 'DrivAer_Star')
        plot_env_comparison(runs, env_name, output_dir, tag=tag, abs_reward=abs_r)
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
