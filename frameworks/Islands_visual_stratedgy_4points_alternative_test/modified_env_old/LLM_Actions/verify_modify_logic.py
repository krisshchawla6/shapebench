import os
import sys
import numpy as np

# Add parent directory for imports
sys.path.insert(0, '/home/Xevolve/AirFoil_becnhmark/modified_env/LLM_Actions')
from llm_design_actions import modify_direct

base_csv = '/home/Xevolve/AirFoil_becnhmark/modified_env/tests/geometry_actions/modify_existing/parent_0.csv'
out_dir = '/tmp/test_modify'
os.makedirs(out_dir, exist_ok=True)

# Try to modify point 0 with noticeably different values
pt_idx = [0]
new_values = [[2.0, 2.0, 0.9]] # x, y, edgy (base is 1.2, -0.05, 0.95)

print(f"Modifying {base_csv}...")
res_path = modify_direct(base_csv, pt_idx, new_values, out_dir=out_dir, name='manual_test')

print(f"Result saved to: {res_path}")

with open(base_csv, 'r') as f:
    print("\n--- BASE CSV ---")
    print(f.read())

with open(res_path, 'r') as f:
    print("\n--- MODIFIED CSV ---")
    print(f.read())
