"""Test script for run_case.py with 4-point airfoil configuration."""

import numpy as np
import os
import sys

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
env_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, env_dir)
os.chdir(env_dir)  # Required for reset folder

# Create test CSV with sample actions
test_csv = os.path.join(script_dir, 'test_actions.csv')

# Action matrix: each row is a control point (radius, angle, edge)
# Values in range [-1, 1], converted internally by env.execute()
test_action = np.array([
    # radius, angle, edge
    [ 0.5,  0.0,  0.3],  # Point 0
    [-0.2,  0.1,  0.5],  # Point 1
    [ 0.3, -0.1,  0.4],  # Point 2
    [-0.4,  0.2,  0.6],  # Point 3
])
test_actions = test_action.flatten().reshape(1, -1)  # Flatten to single row for CSV
np.savetxt(test_csv, test_actions, delimiter=',')
print(f"Created test CSV: {test_csv}")
print(f"Actions shape: {test_actions.shape}")

# Run test
from run_case import env, run_from_csv

print("\n" + "="*60)
print("Running 4-point airfoil test")
print("="*60)

results = run_from_csv(test_csv)

print("\n" + "="*60)
print("Test complete")
print(f"Final drag: {env.drag[-1]:.4f}")
print(f"Final lift: {env.lift[-1]:.4f}")
print(f"Final reward: {env.reward[-1]:.4f}")
print("="*60)

# Cleanup test CSV
os.remove(test_csv)
