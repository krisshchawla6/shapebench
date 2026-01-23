#!/usr/bin/env python
"""
Benchmark script: Run CFD simulations on 5 test cases and time each execution.
"""
import os
import sys
import time
import numpy as np

# Setup paths
env_dir = '/scratch/LLM_Evolve/AirFoil_becnhmark/modified_env'
sys.path.insert(0, env_dir)
os.chdir(env_dir)

from parametered_env import resume_env

def run_single_action(env, action_values):
    """Execute a single action and return results."""
    env.reset()
    result = env.execute(action_values)
    return result

def main():
    # Initialize environment once
    print("Initializing FEniCS environment...")
    init_start = time.time()
    env = resume_env()
    print(f"Environment initialized in {time.time() - init_start:.2f}s")
    
    # 5 different test action sets (4 control points × 3 params = 12 values each)
    # Values in range [-1, 1], will be converted internally
    test_actions = [
        np.array([0.5, 0.0, 0.3, -0.2, 0.1, 0.5, 0.3, -0.1, 0.4, -0.4, 0.2, 0.6]),
        np.array([0.3, 0.2, 0.4, -0.1, -0.1, 0.3, 0.4, 0.1, 0.5, -0.3, 0.0, 0.4]),
        np.array([-0.2, 0.1, 0.2, 0.4, -0.2, 0.6, -0.3, 0.0, 0.3, 0.2, 0.1, 0.5]),
        np.array([0.1, -0.1, 0.5, 0.2, 0.2, 0.4, -0.1, -0.1, 0.6, 0.0, 0.3, 0.3]),
        np.array([-0.4, 0.0, 0.3, 0.3, 0.1, 0.5, 0.2, -0.2, 0.4, -0.2, -0.1, 0.6]),
    ]
    
    times = []
    
    print("\n" + "=" * 60)
    print("Benchmarking CFD Solver - 5 Test Cases")
    print("=" * 60)
    
    for i, action in enumerate(test_actions):
        print(f"\n[{i+1}/5] Running test case {i+1}")
        print("-" * 40)
        
        start_time = time.time()
        try:
            result = run_single_action(env, action)
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            # result = (next_state, terminal, reward)
            print(f"Drag: {env.drag[-1]:.4f}, Lift: {env.lift[-1]:.4f}")
            print(f"Reward: {result[2]:.4f}")
            print(f"Completed in {elapsed:.2f} seconds")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"ERROR after {elapsed:.2f} seconds: {e}")
            import traceback
            traceback.print_exc()
            times.append(elapsed)
    
    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    for i, t in enumerate(times):
        print(f"  Test case {i+1}: {t:.2f}s")
    
    print("-" * 40)
    print(f"  Total time: {sum(times):.2f}s")
    print(f"  Average time per case: {sum(times)/len(times):.2f}s")
    print(f"  Min: {min(times):.2f}s, Max: {max(times):.2f}s")
    print("=" * 60)

if __name__ == "__main__":
    main()
