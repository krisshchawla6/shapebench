#!/usr/bin/env python
"""Analyze strategy performance: before/after rewards, improvements, etc."""

import os
import argparse
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


def parse_strategy(design_dir):
    """Parse strategy name and index from context/strategy.txt."""
    path = os.path.join(design_dir, 'context', 'strategy.txt')
    if not os.path.exists(path):
        return None, None
    with open(path, 'r') as f:
        line = f.read().strip()
    # Format: "Strategy: name (idx=N)"
    name = line.split('Strategy: ')[1].split(' (idx=')[0]
    idx = int(line.split('idx=')[1].rstrip(')'))
    return name, idx


def parse_reward(design_dir):
    """Parse this design's reward from last line of reward_penalization (column 2)."""
    path = os.path.join(design_dir, 'save', 'reward_penalization')
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        lines = f.readlines()
    if not lines:
        return None
    parts = lines[-1].split()
    return float(parts[1]) if len(parts) >= 2 else None


def collect_data(results_dir):
    """Collect per-design data: strategy, reward, and best-before reward."""
    designs = []
    i = 0
    while True:
        d = os.path.join(results_dir, f'design_{i}')
        if not os.path.isdir(d):
            break
        name, idx = parse_strategy(d)
        reward = parse_reward(d)
        if name is not None and reward is not None:
            designs.append({'id': i, 'strategy': name, 'strategy_idx': idx, 'reward': reward})
        i += 1

    # Compute best-before for each design
    best_so_far = float('-inf')
    for d in designs:
        d['best_before'] = best_so_far
        best_so_far = max(best_so_far, d['reward'])
        d['improvement'] = d['reward'] - d['best_before'] if d['best_before'] != float('-inf') else None
        d['beat_best'] = d['reward'] > d['best_before'] if d['best_before'] != float('-inf') else None

    return designs


def print_strategy_stats(designs):
    """Print per-strategy statistics."""
    by_strategy = defaultdict(list)
    for d in designs:
        by_strategy[d['strategy']].append(d)

    print("=" * 90)
    print(f"{'STRATEGY PERFORMANCE ANALYSIS':^90}")
    print("=" * 90)
    print(f"\nTotal designs: {len(designs)}")
    print(f"Strategies found: {', '.join(sorted(by_strategy.keys()))}\n")

    # Per-strategy stats
    summary_rows = []
    for strat in sorted(by_strategy.keys()):
        entries = by_strategy[strat]
        rewards = [e['reward'] for e in entries]
        improvements = [e['improvement'] for e in entries if e['improvement'] is not None]
        beats = [e['beat_best'] for e in entries if e['beat_best'] is not None]

        count = len(entries)
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        best_reward = max(rewards)
        worst_reward = min(rewards)

        if improvements:
            avg_imp = np.mean(improvements)
            max_imp = max(improvements)
            min_imp = min(improvements)
            positive_imps = [x for x in improvements if x > 0]
            avg_pos_imp = np.mean(positive_imps) if positive_imps else 0.0
        else:
            avg_imp = max_imp = min_imp = avg_pos_imp = float('nan')

        beat_count = sum(1 for b in beats if b)
        beat_rate = beat_count / len(beats) * 100 if beats else float('nan')

        summary_rows.append({
            'strategy': strat, 'count': count,
            'avg_reward': avg_reward, 'std_reward': std_reward,
            'best_reward': best_reward, 'worst_reward': worst_reward,
            'avg_improvement': avg_imp, 'max_improvement': max_imp,
            'min_improvement': min_imp, 'avg_positive_improvement': avg_pos_imp,
            'beat_rate': beat_rate, 'beat_count': beat_count,
        })

        print("-" * 90)
        print(f"  Strategy: {strat.upper()} (count={count})")
        print("-" * 90)
        print(f"    Reward:  avg={avg_reward:.4f}  std={std_reward:.4f}  "
              f"best={best_reward:.4f}  worst={worst_reward:.4f}")
        if improvements:
            print(f"    Improvement over prev best:")
            print(f"      avg={avg_imp:.4f}  max={max_imp:.4f}  min={min_imp:.4f}")
            print(f"      avg (when positive)={avg_pos_imp:.4f}")
            print(f"    Beat previous best: {beat_count}/{len(beats)} ({beat_rate:.1f}%)")

            # Which design achieved the greatest improvement?
            best_entry = max(entries, key=lambda e: e['improvement'] if e['improvement'] is not None else float('-inf'))
            print(f"    Greatest improvement: design_{best_entry['id']} "
                  f"(reward={best_entry['reward']:.4f}, before={best_entry['best_before']:.4f}, "
                  f"Δ={best_entry['improvement']:.4f})")
        print()

    return summary_rows, by_strategy


def plot_strategy_stats(designs, by_strategy, results_dir):
    """Generate plots for strategy comparison."""
    strategies = sorted(by_strategy.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(strategies)))
    strat_colors = {s: c for s, c in zip(strategies, colors)}

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1) Reward distribution per strategy (box plot)
    ax = axes[0, 0]
    data_box = [np.array([e['reward'] for e in by_strategy[s]]) for s in strategies]
    bp = ax.boxplot(data_box, tick_labels=strategies, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel('Reward')
    ax.set_title('Reward Distribution per Strategy')
    ax.grid(True, alpha=0.3)

    # 2) Improvement distribution per strategy (box plot)
    ax = axes[0, 1]
    data_imp = []
    labels_imp = []
    for s in strategies:
        imps = [e['improvement'] for e in by_strategy[s] if e['improvement'] is not None]
        if imps:
            data_imp.append(np.array(imps))
            labels_imp.append(s)
    if data_imp:
        bp2 = ax.boxplot(data_imp, tick_labels=labels_imp, patch_artist=True)
        for patch, s in zip(bp2['boxes'], labels_imp):
            patch.set_facecolor(strat_colors[s])
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Improvement over Previous Best')
    ax.set_title('Improvement Distribution per Strategy')
    ax.grid(True, alpha=0.3)

    # 3) Beat-best rate bar chart
    ax = axes[1, 0]
    beat_rates = []
    for s in strategies:
        beats = [e['beat_best'] for e in by_strategy[s] if e['beat_best'] is not None]
        rate = sum(1 for b in beats if b) / len(beats) * 100 if beats else 0
        beat_rates.append(rate)
    bars = ax.bar(strategies, beat_rates, color=[strat_colors[s] for s in strategies])
    ax.set_ylabel('Beat Previous Best (%)')
    ax.set_title('Success Rate: Beat Previous Best')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, beat_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    # 4) Reward over time colored by strategy
    ax = axes[1, 1]
    for d in designs:
        ax.scatter(d['id'], d['reward'], color=strat_colors[d['strategy']],
                   alpha=0.7, s=30, zorder=2)
    # Best-so-far line
    best_so_far = []
    cur = float('-inf')
    for d in designs:
        cur = max(cur, d['reward'])
        best_so_far.append(cur)
    ax.plot([d['id'] for d in designs], best_so_far, 'k-', linewidth=1.5, alpha=0.6, label='Best so far')
    # Legend
    for s in strategies:
        ax.scatter([], [], color=strat_colors[s], label=s, s=50)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.8)
    ax.set_xlabel('Design Iteration')
    ax.set_ylabel('Reward')
    ax.set_title('Reward over Time (colored by Strategy)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(results_dir, 'strategy_stats.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved plot to {out}")


def main():
    parser = argparse.ArgumentParser(description='Strategy performance statistics')
    parser.add_argument('results_dir', type=str, help='Path to benchmark results directory')
    args = parser.parse_args()

    designs = collect_data(args.results_dir)
    if not designs:
        print("No design data found!")
        return

    summary_rows, by_strategy = print_strategy_stats(designs)
    plot_strategy_stats(designs, by_strategy, args.results_dir)


if __name__ == "__main__":
    main()
