#!/usr/bin/env python
"""Run benchmark for all 4 LLM actions sequentially."""

import os
import sys
import json
import time

# Add current directory first, then modified_env paths
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)
sys.path.insert(1, os.path.join(base_dir, 'modified_env'))
sys.path.insert(2, os.path.join(base_dir, 'modified_env/LLM_Actions'))

from run_benchmark_action import run_benchmark

ACTIONS = ['generate', 'generate_direct', 'modify', 'modify_direct']
N_ITERATIONS = 5
N_INSPIRATIONS = 2
OUTPUT_BASE = 'tmp_out/tmp_test'
DEBUG = True


def run_action_benchmark(action, output_dir, baseline_csv):
    """Run benchmark for a single action."""
    print(f"\n{'='*60}")
    print(f"Running benchmark: {action}")
    print(f"Iterations: {N_ITERATIONS}")
    print(f"Output: {output_dir}")
    print(f"Debug: {DEBUG}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    try:
        best_design, cached = run_benchmark(
            baseline_csv, 
            N_ITERATIONS, 
            N_INSPIRATIONS, 
            action, 
            output_dir, 
            debug=DEBUG
        )
        
        # Save results.json
        cache_data = []
        for entry in cached:
            csv_path, rank, reward, results = entry[0], entry[1], entry[2], entry[3]
            cache_data.append({
                'csv_path': str(csv_path),
                'rank': int(rank),
                'reward': float(reward),
                'drag_lift': results[0],
                'sol_images': results[1]
            })
        
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        returncode = 0
        print(f"\nBest design: {best_design}")
    except Exception as e:
        print(f"Error running {action}: {e}")
        import traceback
        traceback.print_exc()
        returncode = 1
    
    elapsed = time.time() - start_time
    print(f"\n{action} completed in {elapsed:.1f}s (exit code: {returncode})")
    return returncode, elapsed


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    modified_env_dir = os.path.join(base_dir, 'modified_env')
    output_base = os.path.join(base_dir, OUTPUT_BASE)
    os.makedirs(output_base, exist_ok=True)
    
    baseline_csv = os.path.join(modified_env_dir, 'baseline_action.csv')
    if not os.path.exists(baseline_csv):
        print(f"Error: {baseline_csv} not found")
        sys.exit(1)
    
    # Change to modified_env directory (required for reset folder)
    os.chdir(modified_env_dir)
    print(f"Working directory: {os.getcwd()}")
    
    print(f"Running all {len(ACTIONS)} action benchmarks")
    print(f"Iterations per action: {N_ITERATIONS}")
    print(f"Output base: {output_base}")
    
    results = {}
    total_start = time.time()
    
    for action in ACTIONS:
        output_dir = os.path.join(output_base, f'{action}_benchmark')
        returncode, elapsed = run_action_benchmark(action, output_dir, baseline_csv)
        results[action] = {'returncode': returncode, 'elapsed': elapsed}
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for action, info in results.items():
        status = "✓" if info['returncode'] == 0 else "✗"
        print(f"  {status} {action}: {info['elapsed']:.1f}s")
    print(f"\nTotal time: {total_elapsed:.1f}s")
    print(f"Results saved to: {output_base}/")


if __name__ == "__main__":
    main()
