#!/usr/bin/env python
"""Simple test of FEniCS solver with existing mesh."""
import os
import sys
import time

# Setup paths
env_dir = '/scratch/LLM_Evolve/AirFoil_becnhmark/modified_env'
sys.path.insert(0, env_dir)
os.chdir(env_dir)

print("Testing FEniCS import...")
start = time.time()
from fenics_solver import solve_flow
print(f"FEniCS imported in {time.time()-start:.2f}s")

# Use existing mesh from reset folder
mesh_file = 'reset/4/shape_0.xml'
print(f"\nTesting with mesh: {mesh_file}")

# Check mesh exists
if not os.path.exists(mesh_file):
    print(f"ERROR: Mesh file not found: {mesh_file}")
    sys.exit(1)

print("Mesh file found, running CFD solver...")
start = time.time()

try:
    drag, lift, solved = solve_flow(
        mesh_file=mesh_file,
        final_time=10.0,  # Short simulation
        reynolds=100.0,
        output=False,
        cfl=0.5,
        pts_x=[1.0, 0.0, -1.0, 0.0],
        pts_y=[0.0, 1.0, 0.0, -1.0],
        xmin=-15.0,
        xmax=30.0,
        ymin=-15.0,
        ymax=15.0
    )
    elapsed = time.time() - start
    
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Solved: {solved}")
    print(f"Drag: {drag:.4f}")
    print(f"Lift: {lift:.4f}")
    print(f"Time: {elapsed:.2f}s")
    print(f"{'='*50}")

except Exception as e:
    elapsed = time.time() - start
    print(f"\nERROR after {elapsed:.2f}s: {e}")
    import traceback
    traceback.print_exc()
