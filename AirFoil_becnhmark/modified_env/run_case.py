#Wrapper around env 
import sys
import numpy as np
sys.stdout.reconfigure(line_buffering=True)
from parametered_env import *

# Initialize environment
env = resume_env()

def run_from_csv(csv_path, reset_first=True):
    """
    Run actions from CSV file. Each row is one action (12 values for 4 points × 3 params).
    Returns list of (next_state, terminal, reward) tuples.
    """
    actions = np.loadtxt(csv_path, delimiter=',')
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)  # Single action case
    
    if reset_first:
        env.reset()
    
    results = []
    for action in actions:
        result = env.execute(action)
        results.append(result)
        print(f"Reward: {result[2]:.4f}, Drag: {env.drag[-1]:.4f}, Lift: {env.lift[-1]:.4f}", flush=True)
    return results

def run_action(action, reset_first=False):
    """Execute a single action. Returns (next_state, terminal, reward)."""
    if reset_first:
        env.reset()
    return env.execute(action)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_from_csv(sys.argv[1])
    else:
        print("Usage: python run_case.py <actions.csv>")
