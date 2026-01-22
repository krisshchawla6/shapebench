#!/usr/bin/env python
"""Run benchmark for all 4 LLM actions sequentially."""

import os
import sys
import subprocess
import time

ACTIONS = ['generate', 'generate_direct', 'modify', 'modify_direct']
N_ITERATIONS = 5
N_INSPIRATIONS = 2
OUTPUT_BASE = 'test_actions'

def run_action_benchmark(action, output_dir):
    """Run benchmark for a single action."""
    script_path = os.path.join(os.path.dirname(__file__), 'modified_env', 'run_benchmark_action.py')
    
    cmd = [
        sys.executable,
        script_path,
        '--action', action,
        '--iterations', str(N_ITERATIONS),
        '--inspirations', str(N_INSPIRATIONS)
    ]
    
    # Override output dir by modifying env or passing via temp modification
    # Since run_benchmark_action.py builds output_dir from action name,
    # we'll run from the right directory and symlink/move results after
    
    print(f"\n{'='*60}")
    print(f"Running benchmark: {action}")
    print(f"Iterations: {N_ITERATIONS}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Run from modified_env directory
    result = subprocess.run(
        cmd,
        cwd=os.path.join(os.path.dirname(__file__), 'modified_env'),
        capture_output=False
    )
    
    elapsed = time.time() - start_time
    
    # Move results to desired location
    src_dir = os.path.join(os.path.dirname(__file__), 'modified_env', f'benchmark_results_{action}')
    if os.path.exists(src_dir):
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
        os.rename(src_dir, output_dir)
    
    print(f"\n{action} completed in {elapsed:.1f}s (exit code: {result.returncode})")
    return result.returncode, elapsed

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_base = os.path.join(base_dir, OUTPUT_BASE)
    os.makedirs(output_base, exist_ok=True)
    
    print(f"Running all {len(ACTIONS)} action benchmarks")
    print(f"Iterations per action: {N_ITERATIONS}")
    print(f"Output base: {output_base}")
    
    results = {}
    total_start = time.time()
    
    for action in ACTIONS:
        output_dir = os.path.join(output_base, f'{action}_benchmark')
        returncode, elapsed = run_action_benchmark(action, output_dir)
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
