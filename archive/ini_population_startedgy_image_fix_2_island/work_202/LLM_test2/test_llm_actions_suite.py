import os
import sys
import shutil
import numpy as np

# Add parent directory to path to find LLM_Actions modules
# Current file is in AirFoil_becnhmark/modified_env/LLM_test2/
# We need to reach AirFoil_becnhmark/modified_env/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from LLM_Actions.LLM_agent import run_llm_action
from LLM_Actions.llm_design_actions import run_action

def test_generate_actions():
    """Test 'generate' and 'generate_direct' actions with new [x, y, edgy] format."""
    print("\n=== Testing GENERATE Actions ===")
    
    output_dir = os.path.join(os.path.dirname(__file__), 'geometry_actions/generate_new')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # 1. Test generate_direct (Mocking LLM output)
    print("1. Testing generate_direct...")
    # New format: [x, y, edgy] per point
    # Creating a simple square-ish shape for testing
    params_direct = {
        "n_cp": 4, 
        "n_sp": 10,
        "params": [
            [1.0, 0.0, 0.5],   # Right
            [0.0, 1.0, 0.5],   # Top
            [-1.0, 0.0, 0.5],  # Left
            [0.0, -1.0, 0.5]   # Bottom
        ],
        "name": "gen_direct_test",
        "out_dir": output_dir
    }
    
    try:
        csv_path = run_action('generate_direct', **params_direct)
        print(f"   SUCCESS: Generated {csv_path}")
        
        # Verify content
        data = np.loadtxt(csv_path, delimiter=',')
        if data.size == 12:
            print("   Content check PASS: Size 12 detected.")
        else:
            print(f"   Content check FAIL: Size {data.size} detected (expected 12).")
            
    except Exception as e:
        print(f"   FAIL: {e}")

    # 2. Test generate (with sampling bounds)
    print("\n2. Testing generate (linear sampling)...")
    params_sample = {
        "n_cp": 4, 
        "n_sp": 10,
        "params": [
            [[0.9, 1.1], [-0.1, 0.1], 0.5], # Right x in [0.9, 1.1]
            [0.0, 1.0, 0.5],
            [-1.0, 0.0, 0.5],
            [0.0, -1.0, 0.5]
        ],
        "name": "gen_sample_test",
        "out_dir": output_dir
    }
    
    try:
        csv_path = run_action('generate', **params_sample)
        print(f"   SUCCESS: Generated {csv_path}")
        data = np.loadtxt(csv_path, delimiter=',')
        print(f"   Sampled value 0 (should be ~1.0): {data[0]}")
    except Exception as e:
        print(f"   FAIL: {e}")


def test_modify_actions():
    """Test 'modify' and 'modify_direct' actions."""
    print("\n=== Testing MODIFY Actions ===")
    
    output_dir = os.path.join(os.path.dirname(__file__), 'geometry_actions/modify_existing')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Create a dummy base file first
    base_csv = os.path.join(output_dir, 'base_shape.csv')
    base_data = [1.0, 0.0, 0.5,  0.0, 1.0, 0.5,  -1.0, 0.0, 0.5,  0.0, -1.0, 0.5]
    np.savetxt(base_csv, [base_data], delimiter=',', fmt='%.4f')
    
    # 1. Test modify_direct
    print("1. Testing modify_direct...")
    params_mod_direct = {
        "base_csv": base_csv,
        "pt_idx": [0, 2], # Modify Right and Left points
        "values": [
            [2.0, 0.0, 0.9],  # Stretch Right to x=2, sharp
            [-2.0, 0.0, 0.9]  # Stretch Left to x=-2, sharp
        ],
        "name": "mod_direct_test",
        "out_dir": output_dir
    }
    
    try:
        csv_path = run_action('modify_direct', **params_mod_direct)
        print(f"   SUCCESS: Generated {csv_path}")
        
        data = np.loadtxt(csv_path, delimiter=',')
        # Check indices 0,1,2 (point 0) -> should be 2.0, 0.0, 0.9
        if np.allclose(data[0:3], [2.0, 0.0, 0.9]):
             print("   Content check PASS: Point 0 modified correctly.")
        else:
             print(f"   Content check FAIL: Point 0 is {data[0:3]}")
             
    except Exception as e:
        print(f"   FAIL: {e}")


if __name__ == "__main__":
    test_generate_actions()
    test_modify_actions()
