import os
import sys
import numpy as np

# Add LLM_Actions to path
sys.path.insert(0, '/home/Xevolve/AirFoil_becnhmark/modified_env/LLM_Actions')
from LLM_agent import run_llm_action

def run_batch_test(n_designs=10):
    output_dir = '/home/Xevolve/AirFoil_becnhmark/modified_env/tests/geometry_actions/generate_new'
    baseline_csv = '/home/Xevolve/AirFoil_becnhmark/modified_env/LLM_Actions/output/shape_0.csv'
    
    # Context based on the baseline
    context = [
        {
            "vector": [1.0, 0.0, 0.5, 0.5, 0.0, 1.0, 0.5, 0.5, -1.0, 0.0, 0.5, 0.5, 0.0, -1.0, 0.5, 0.5],
            "reward": 0.0,
            "ranking": 1,
            "images": []
        }
    ]
    
    print(f"Starting batch generation of {n_designs} designs...")
    
    for i in range(n_designs):
        design_name = f"design{i}"
        print(f"\n--- Generating Design {i+1}/{n_designs} ---")
        
        # We use 'generate_direct' action for pure direct generation (no sampling)
        try:
            csv_path = run_llm_action(
                action='generate_direct',
                context=context,
                output_dir=output_dir,
                name=design_name,
                temperature=1.0  # Temperature for creative diversity
            )
            if csv_path:
                print(f"Successfully generated: {csv_path}")
            else:
                print(f"Failed to generate design {i+1}")
        except Exception as e:
            print(f"Error in batch generation {i+1}: {e}")

if __name__ == "__main__":
    run_batch_test(10)
