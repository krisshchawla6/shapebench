#!/usr/bin/env python
"""Plot reward vs iteration from benchmark results."""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

def load_results(results_dir):
    """Load results from benchmark directory by parsing reward_penalization files."""
    rewards = []
    iterations = []
    
    # Check initial - use reward_penalization file, column 2 (reward) from last line
    initial_rp = os.path.join(results_dir, 'initial', 'save', 'reward_penalization')
    if os.path.exists(initial_rp):
        with open(initial_rp, 'r') as f:
            lines = f.readlines()
            if lines:
                # Last line, column 2 is the reward for this specific run
                parts = lines[-1].split()
                if len(parts) >= 2:
                    reward = float(parts[1])
                    rewards.append(reward)
                    iterations.append(0)
    
    # Check design folders
    i = 0
    while True:
        design_rp = os.path.join(results_dir, f'design_{i}', 'save', 'reward_penalization')
        if not os.path.exists(design_rp):
            break
        with open(design_rp, 'r') as f:
            lines = f.readlines()
            if lines:
                # Last line, column 2 is the reward
                parts = lines[-1].split()
                if len(parts) >= 2:
                    reward = float(parts[1])
                    rewards.append(reward)
                    iterations.append(i + 1)
        i += 1
    
    # Also try results.json as fallback
    if not rewards:
        results_json = os.path.join(results_dir, 'results.json')
        if os.path.exists(results_json):
            with open(results_json, 'r') as f:
                return json.load(f)
    
    return [{'reward': r, 'iteration': it} for r, it in zip(rewards, iterations)]

def plot_rewards(results_dir, output_path=None):
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
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add stats
    stats_text = f'Final Best: {best_rewards[-1]:.4f}\nTotal Iterations: {len(iterations)}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        output_path = os.path.join(results_dir, 'reward_plot.png')
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    
    plt.close()
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Iterations: {len(iterations)}")
    print(f"  Best Reward: {max(rewards):.4f}")
    print(f"  Worst Reward: {min(rewards):.4f}")
    print(f"  Mean Reward: {np.mean(rewards):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot reward vs iteration')
    parser.add_argument('results_dir', type=str, help='Path to benchmark results directory')
    parser.add_argument('--output', type=str, default=None, help='Output path for plot')
    args = parser.parse_args()
    
    plot_rewards(args.results_dir, args.output)
