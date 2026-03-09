import os
import sys
import numpy as np
import shutil
import subprocess

# Add LLM_Actions to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../LLM_Actions')))
from LLM_agent import run_llm_action
from action_to_shape import convert_action_to_shape

def run_batch_modify_test_v2(n_modifications=10):
    """
    Batch test for LLM modifications using the new [x, y, edgy] action format.
    Generates actions FIRST, then converts/visualizes them separately.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'geometry_actions/modify_existing')
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Define a dummy parent action CSV for context (Action format: flat list)
    # 4 points: (1,0), (0,1), (-1,0), (0,-1) with edgy=0.5
    parent_action = [1.0, 0.0, 0.5,  0.0, 1.0, 0.5,  -1.0, 0.0, 0.5,  0.0, -1.0, 0.5]
    parent_csv = os.path.join(output_dir, 'parent_action.csv')
    np.savetxt(parent_csv, [parent_action], delimiter=',', fmt='%.4f')
    
    # Context based on the parent design
    context = [
        {
            "vector": parent_action,
            "reward": 0.0,
            "ranking": 1,
            "images": []
        }
    ]
    
    print(f"Starting batch modification of {n_modifications} variants from parent design...")
    print(f"Output Directory: {output_dir}")
    
    generated_files = []

    # ---------------------------------------------------------
    # STEP 1: GENERATE ALL ACTIONS via LLM
    # ---------------------------------------------------------
    for i in range(n_modifications):
        design_name = f"modified{i}"
        print(f"\n--- [LLM] Generating Modification {i+1}/{n_modifications} ---")
        
        try:
            # Note: run_llm_action calls 'run_action' which writes the CSV
            # BUT run_llm_action ALSO tries to run visualization internally.
            # We should probably modify run_llm_action to optionally SKIP visualization
            # OR just ignore the fact it runs it for now, but you wanted separate loops.
            # Assuming run_llm_action does everything currently.
            
            # To decouple, we can call the LLM generation part directly or modify run_llm_action.
            # Since run_llm_action is "Agent" level code, let's use it but perhaps we can 
            # prevent it from blocking/crashing on vis by just letting it run.
            
            # Wait, run_llm_action in LLM_agent.py *DOES* call subprocess for vis at the end.
            # "result = subprocess.run(cmd, ...)"
            # If we want to avoid that loop there, we should edit LLM_agent.py or mock it.
            
            # For this script, let's assume we proceed with generation and collect paths.
            
            action_csv_path = run_llm_action(
                action='modify_direct', 
                context=context,
                output_dir=output_dir,
                base_csv=parent_csv,
                name=design_name,
                temperature=1.0,
                skip_vis=True # Skip vis inside agent to avoid double running
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
        
        # A. Convert to Shape CSV (optional, but requested)
        shape_csv_path = csv_path.replace('.csv', '_shape.csv')
        if convert_action_to_shape(csv_path, shape_csv_path):
            print(f"   Converted to Shape CSV: {os.path.basename(shape_csv_path)}")
        
        # B. Run Visualization Pipeline as Subprocess
        # We run test_modification.py as a script
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
    run_batch_modify_test_v2(3)
