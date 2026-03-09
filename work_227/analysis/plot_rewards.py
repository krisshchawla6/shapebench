#!/usr/bin/env python
"""Plot reward vs iteration from benchmark results."""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

def load_results(results_dir):
    """Load results from results.json in the benchmark output directory."""
    results_json = os.path.join(results_dir, 'results.json')
    if os.path.exists(results_json):
        with open(results_json, 'r') as f:
            return json.load(f)
    return []

def plot_rewards(results_dir, output_path=None, log_x=False):
    """Plot reward vs iteration."""
    data = load_results(results_dir)
    
    if not data:
        print("No data found!")
        return
    
    # Extract rewards and iterations
    if 'iteration' in data[0]:
        iterations = [d['iteration'] for d in data]
        rewards = [d['reward'] for d in data]
    else:
        iterations = list(range(len(data)))
        rewards = [d['reward'] for d in data]
    
    # Calculate best reward at each iteration
    best_rewards = []
    current_best = float('-inf')
    for r in rewards:
        current_best = max(current_best, r)
        best_rewards.append(current_best)

    # Best iteration (first occurrence of max reward)
    best_reward_value = max(rewards)
    best_iteration = iterations[rewards.index(best_reward_value)]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot individual rewards
    ax.scatter(iterations, rewards, alpha=0.6, label='Reward per iteration', color='blue', s=50)
    
    # Plot best reward line
    ax.plot(iterations, best_rewards, 'r-', linewidth=2, label='Best reward so far')
    
    # Styling
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Reward vs Iteration', fontsize=14)
    ax.legend(loc='lower right', framealpha=0.8)
    ax.grid(True, alpha=0.3)

    if log_x:
        ax.set_xscale('log')
        ax.set_xlabel('Iteration (log scale)', fontsize=12)
    
    # Add stats
    stats_text = (
        f'Final Best: {best_rewards[-1]:.4f}\n'
        f'Total Iterations: {len(iterations)}\n'
        f'Best Iteration: {best_iteration}'
    )
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        suffix = "_logx" if log_x else ""
        output_path = os.path.join(results_dir, f'reward_plot{suffix}.png')
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    
    plt.close()
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Iterations: {len(iterations)}")
    print(f"  Best Reward: {best_reward_value:.4f}")
    print(f"  Best Iteration: {best_iteration}")
    print(f"  Worst Reward: {min(rewards):.4f}")
    print(f"  Mean Reward: {np.mean(rewards):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot reward vs iteration')
    parser.add_argument('results_dir', type=str, help='Path to benchmark results directory')
    parser.add_argument('--output', type=str, default=None, help='Output path for plot')
    parser.add_argument('--log-x', action='store_true', help='Use log scale for x-axis')
    args = parser.parse_args()
    
    plot_rewards(args.results_dir, args.output, log_x=args.log_x)
