import os
import sys
import numpy as np
import shutil
import subprocess

# Add LLM_Actions to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../LLM_Actions')))
from LLM_agent import run_llm_action
from action_to_shape import convert_action_to_shape

def run_batch_generate_bounds_test(n_generations=10):
    """
    Batch test for LLM GENERATE action using BOUNDS (generate.json schema).
    The LLM provides [min, max] ranges for each parameter, and we sample from them.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'geometry_actions/generate_bounds')
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Context
    context = [
        {
            "vector": [1.0, 0.0, 0.5,  0.0, 1.0, 0.5,  -1.0, 0.0, 0.5,  0.0, -1.0, 0.5],
            "reward": 0.5,
            "ranking": 1,
            "images": []
        }
    ]
    
    print(f"Starting batch GENERATION (BOUNDS) of {n_generations} designs...")
    print(f"Output Directory: {output_dir}")
    
    generated_files = []

    # ---------------------------------------------------------
    # STEP 1: GENERATE ALL ACTIONS via LLM
    # ---------------------------------------------------------
    for i in range(n_generations):
        design_name = f"gen_bounds{i}"
        print(f"\n--- [LLM] Generating Design {i+1}/{n_generations} (Bounds) ---")
        
        try:
            # We use 'generate' (not direct) which supports bounds
            action_csv_path = run_llm_action(
                action='generate', 
                context=context,
                output_dir=output_dir,
                name=design_name,
                temperature=1.2,
                skip_vis=True
            )
            
            if action_csv_path:
                print(f"   Generated: {action_csv_path}")
                generated_files.append(action_csv_path)
            else:
                print(f"   Failed to generate {design_name}")
                
        except Exception as e:
            print(f"Error generating {design_name}: {e}")

    # ---------------------------------------------------------
    # STEP 2: PROCESS GENERATED FILES (Convert & Vis)
    # ---------------------------------------------------------
    print(f"\n--- Processing {len(generated_files)} generated files ---")
    
    vis_script = os.path.abspath(os.path.join(os.path.dirname(__file__), '../LLM_Actions/test_modification.py'))
    
    for csv_path in generated_files:
        print(f"\nProcessing: {os.path.basename(csv_path)}")
        
        # A. Convert to Shape CSV (optional)
        shape_csv_path = csv_path.replace('.csv', '_shape.csv')
        if convert_action_to_shape(csv_path, shape_csv_path):
            print(f"   Converted to Shape CSV: {os.path.basename(shape_csv_path)}")
        
        # B. Run Visualization Pipeline as Subprocess
        print("   Running visualization subprocess...")
        try:
            cmd = [sys.executable, vis_script, csv_path, '-o', output_dir, '--xmin', '-15', '--xmax', '30', '--ymin', '-15', '--ymax', '15']
            subprocess.run(cmd, check=True)
            print("   Visualization complete.")
        except subprocess.CalledProcessError as e:
            print(f"   Visualization failed: {e}")
        except Exception as e:
            print(f"   Error running subprocess: {e}")

if __name__ == "__main__":
    run_batch_generate_bounds_test(3)
