#Wrapper around env 
import sys
import numpy as np
sys.stdout.reconfigure(line_buffering=True)
from parametered_env import *
from LLM_Actions.LLM_agent import *

# Initialize environment
env = resume_env()

def run_from_csv(csv_path, reset_first=True):
    """
    Run actions from CSV file. Each row is one action (12 values for 4 points × 3 params).
    Returns list of (next_state, terminal, reward) tuples.
    """
    airfoil_shape = np.loadtxt(csv_path, delimiter=',')
    if airfoil_shape.ndim == 1:
        airfoil_shape = airfoil_shape.reshape(1, -1)
    
    # Take the first shape
    airfoil_shape = airfoil_shape[0]

    # Reset environment before execution to ensure deformations are applied to the baseline
    if reset_first:
        # Reset shape index to prevent file accumulation across runs
        from environment import env as env_singleton
        env_singleton.shape.index = 0
        env.reset()
        
    result = env.execute(airfoil_shape)
    print(f"Reward: {result[2]:.4f}, Drag: {env.drag[-1]:.4f}, Lift: {env.lift[-1]:.4f}", flush=True)
    
    return result

def run_LLM_action(LLM_action, context, reset_first=False):
    """Execute a single LLM action. Returns (next_state, terminal, reward)."""
    if reset_first:
        env.reset()
    
    csv, csv_path = run_llm_action(LLM_action, context)
    
    return [csv, csv_path]

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_from_csv(sys.argv[1])
    else:
        print("Usage: python run_case.py <actions.csv>")
