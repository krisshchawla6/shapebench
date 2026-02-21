#!/usr/bin/env python3
"""
plot_rewards_3d.py — Plot reward progression and lineage tree for 3D benchmark.

Usage:
    python plot_rewards_3d.py /scratch/3D/3d/benchmark_results_3d_gaussain
    python plot_rewards_3d.py /scratch/3D/3d/local_run_3d_200
"""

import os
import re
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path


def load_results_csv(results_dir):
    csv_path = os.path.join(results_dir, 'results.csv')
    data = {'iteration': [], 'design': [], 'reward': [], 'best_reward': [],
            'CL': [], 'CDi': [], 'CM': [], 'L_D': []}
    with open(csv_path) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
            data['iteration'].append(int(parts[0]))
            data['design'].append(parts[1])
            data['reward'].append(float(parts[2]))
            data['best_reward'].append(float(parts[3]))
            data['CL'].append(float(parts[4]))
            data['CDi'].append(float(parts[5]))
            data['CM'].append(float(parts[6]))
            data['L_D'].append(float(parts[7]))
    for k in data:
        if k != 'design':
            data[k] = np.array(data[k])
    return data


def reconstruct_lineage(results_dir, n_designs):
    """Parse context.txt files to find parent for each design."""
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
        # "Design 1" in context is the parent — extract its params to match
        m = re.search(r'Design 1:\s*\n\s*- Parameters:.*?le_sweep=([\d.]+)', text)
        if not m:
            parent_of[i] = None
            continue
        parent_sweep = float(m.group(1))
        # Match against all design JSONs
        best_match = None
        best_dist = float('inf')
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


def plot(results_dir):
    data = load_results_csv(results_dir)
    n = len(data['iteration'])
    if n == 0:
        print("No data found.")
        return

    parent_of = reconstruct_lineage(results_dir, n)

    rewards = data['reward'].copy()
    p5, p95 = np.percentile(rewards, 5), np.percentile(rewards, 95)
    margin = max((p95 - p5) * 0.5, 1.0)
    y_lo, y_hi = p5 - margin, p95 + margin

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 13,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300,
    })

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), facecolor='white',
                             gridspec_kw={'height_ratios': [3, 2]})

    # ── Panel 1: Reward over iterations ──────────────────────────────────
    ax = axes[0]
    ax.set_facecolor('white')

    for child, parent in parent_of.items():
        if parent is not None and parent < n and child < n:
            ax.plot(
                [parent, child],
                [np.clip(rewards[parent], y_lo, y_hi),
                 np.clip(rewards[child], y_lo, y_hi)],
                color='#bbb', linewidth=0.5, alpha=0.4, zorder=1,
            )

    colors = np.where(rewards >= 0, '#2ca02c', '#d62728')
    ax.scatter(data['iteration'], np.clip(rewards, y_lo, y_hi),
               c=colors, s=20, alpha=0.8, edgecolors='k', linewidths=0.3, zorder=2)

    ax.plot(data['iteration'], np.clip(data['best_reward'], y_lo, y_hi),
            color='#1f77b4', linewidth=1.8, label='Best so far', zorder=3)

    ax.axhline(0, color='grey', linewidth=0.7, linestyle='--', alpha=0.6)

    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'Reward  ($C_L / C_{D_i} - 5.45$)')
    ax.set_title('(a) Reward progression', fontweight='bold')
    ax.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Panel 2: Lineage tree ────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor('white')

    depth = {}
    for i in range(n):
        if parent_of.get(i) is None:
            depth[i] = 0
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if i in depth:
                continue
            p = parent_of.get(i)
            if p is not None and p in depth:
                depth[i] = depth[p] + 1
                changed = True
    for i in range(n):
        if i not in depth:
            depth[i] = 0

    node_x = {i: i for i in range(n)}
    node_y = {i: depth[i] for i in range(n)}
    max_depth = max(depth.values()) if depth else 0

    for child, parent in parent_of.items():
        if parent is not None and parent in node_x and child in node_x:
            ax2.plot([node_x[parent], node_x[child]],
                     [node_y[parent], node_y[child]],
                     color='#999', linewidth=0.5, alpha=0.5, zorder=1)

    xs = [node_x[i] for i in range(n)]
    ys = [node_y[i] for i in range(n)]
    rs_clipped = np.clip(rewards, p5, p95)

    sc = ax2.scatter(xs, ys, c=rs_clipped, cmap='RdYlGn', s=22, alpha=0.9,
                     edgecolors='k', linewidths=0.3, zorder=2)
    cb = fig.colorbar(sc, ax=ax2, fraction=0.025, pad=0.02)
    cb.set_label('Reward')
    cb.outline.set_linewidth(0.5)

    ax2.set_xlabel('Design ID')
    ax2.set_ylabel('Generation depth')
    ax2.set_title(r'(b) Lineage tree (parent $\rightarrow$ child)', fontweight='bold')
    ax2.set_ylim(-0.5, max_depth + 0.5)
    ax2.invert_yaxis()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(results_dir, 'reward_plot.png')
    plt.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_rewards_3d.py <results_dir>")
        sys.exit(1)
    plot(sys.argv[1])
