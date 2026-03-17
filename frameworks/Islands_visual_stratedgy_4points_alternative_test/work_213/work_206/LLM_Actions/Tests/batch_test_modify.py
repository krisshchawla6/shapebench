import os
import sys
import numpy as np

# Add LLM_Actions to path
sys.path.insert(0, '/home/Xevolve/AirFoil_becnhmark/modified_env/LLM_Actions')
from LLM_agent import run_llm_action

def run_batch_modify_test(n_modifications=10):
    output_dir = '/home/Xevolve/AirFoil_becnhmark/modified_env/tests/geometry_actions/modify_existing'
    parent_csv = '/home/Xevolve/AirFoil_becnhmark/modified_env/tests/geometry_actions/modify_existing/parent_0.csv'
    
    # Context based on the parent design
    context = [
        {
            "vector": [1.2, -0.05, 0.15, 0.95, -0.2, 0.45, 0.8, 0.1, -1.0, 0.0, 0.6, 0.05, -0.1, -0.15, 0.7, 0.1],
            "reward": 0.0,
            "ranking": 1,
            "images": []
        }
    ]
    
    print(f"Starting batch modification of {n_modifications} variants from parent design...")
    
    for i in range(n_modifications):
        design_name = f"modified{i}"
        print(f"\n--- Generating Modification {i+1}/{n_modifications} ---")
        
        # Use 'modify_direct' action for pure direct modification (no sampling)
        try:
            csv_path = run_llm_action(
                action='modify_direct',
                context=context,
                output_dir=output_dir,
                base_csv=parent_csv,
                name=design_name,
                temperature=1.0
            )
            if csv_path:
                print(f"Successfully generated: {csv_path}")
            else:
                print(f"Failed to generate modification {i+1}")
        except Exception as e:
            print(f"Error in batch modification {i+1}: {e}")

if __name__ == "__main__":
    run_batch_modify_test(10)
