import os
import sys
import json
import argparse
import shutil
import numpy as np
import scipy.stats as stats

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modified_env'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modified_env/LLM_Actions'))

from run_case import run_from_csv
from LLM_agent import run_llm_action
from Analysis_LLM import run_simulation_analysis

def run_simulation(csv_path, case_dir):
    result = run_from_csv(csv_path, reset_first=True)
    reward = result[2]
    
    case_save_dir = os.path.join(case_dir, 'save')
    if os.path.exists('./save'):
        if os.path.exists(case_save_dir):
            shutil.rmtree(case_save_dir)
        shutil.copytree('./save', case_save_dir)
    
    results = post_process_results(case_save_dir, reward)
    return reward, results

def post_process_results(save_dir, reward=None):
    """Extract simulation results from save directory.
    
    Returns:
        [[drag, lift], sol_images, analysis_text, shape_image]
    """
    drag, lift = 0.0, 0.0
    
    # Extract drag/lift from drag_lift file
    dl_path = os.path.join(save_dir, 'drag_lift')
    if os.path.exists(dl_path):
        with open(dl_path, 'r') as f:
            lines = f.readlines()
            if lines:
                last = lines[-1].split()
                if len(last) >= 3:
                    drag = float(last[1])
                    lift = float(last[2])
    
    # If reward not provided, try to extract from reward_penalization file
    if reward is None:
        reward_file = os.path.join(save_dir, 'reward_penalization')
        if os.path.exists(reward_file):
            with open(reward_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    # Last line, column 2 is the reward
                    parts = lines[-1].split()
                    if len(parts) >= 2:
                        reward = float(parts[1])
    
    sol_dir = os.path.join(save_dir, 'sol')
    sol_images = [
        os.path.join(sol_dir, '1_p.png'),
        os.path.join(sol_dir, '1_u.png'),
        os.path.join(sol_dir, '1_v.png')
    ]
    
    # Find shape image (geometry visualization)
    png_dir = os.path.join(save_dir, 'png')
    shape_image = None
    if os.path.exists(png_dir):
        shape_pngs = sorted([f for f in os.listdir(png_dir) if f.startswith('shape_') and f.endswith('.png')])
        if shape_pngs:
            shape_image = os.path.join(png_dir, shape_pngs[-1])
    
    # Run qualitative analysis
    metrics = {'drag': drag, 'lift': lift}
    if reward is not None:
        metrics['reward'] = reward
        
    try:
        analysis_text = run_simulation_analysis(sol_images, metrics)
    except Exception as e:
        print(f"Analysis failed: {e}")
        analysis_text = ""
    
    # DEBUG: Print what we're storing
    print(f"DEBUG post_process: save_dir={save_dir}")
    print(f"DEBUG post_process: shape_image={shape_image}")
    print(f"DEBUG post_process: sol_images[0]={sol_images[0] if sol_images else 'None'}")
    
    return [[drag, lift], sol_images, analysis_text, shape_image]

def update_database(database, x, reward, results):
    entry = np.array([[x, 0, reward, results]], dtype=object)
    if len(database) == 0:
        return entry
    database = np.append(database, entry, axis=0)
    indices = np.argsort(database[:, 2].astype(float))[::-1]
    database = database[indices]
    for i in range(len(database)):
        database[i, 1] = i
    return database

def generate_design(parent, inspirations, output_dir, iteration_nb, action, debug=False):
    llm_context = []
    
    # Build context with full information including feedback
    if parent is not None:
        vec = np.loadtxt(parent[0], delimiter=',')
        if vec.ndim == 2: vec = vec[0]
        results = parent[3] if len(parent) > 3 else []
        drag_lift = results[0] if len(results) > 0 else [0, 0]
        sol_images = results[1] if len(results) > 1 else []
        feedback = results[2] if len(results) > 2 else ""
        shape_image = results[3] if len(results) > 3 else None
        
        # DEBUG: Print image paths
        print(f"DEBUG: Parent CSV: {parent[0]}")
        print(f"DEBUG: Shape image path: {shape_image}")
        print(f"DEBUG: Sol images: {sol_images}")
        
        # For parent: include shape image + sol visualizations (4 images total)
        parent_images = []
        if shape_image and os.path.exists(shape_image):
            parent_images.append(shape_image)
            print(f"DEBUG: Added shape image: {shape_image}")
        # Add sol images (pressure, u-velocity, v-velocity)
        for sol_img in sol_images:
            if sol_img and os.path.exists(sol_img):
                parent_images.append(sol_img)
                print(f"DEBUG: Added sol image: {sol_img}")
        
        llm_context.append({
            'vector': vec.tolist(),
            'reward': parent[2],
            'ranking': parent[1],
            'drag': drag_lift[0],
            'lift': drag_lift[1],
            'feedback': feedback,
            'images': parent_images  # Shape image only for parent
        })
    
    if inspirations is not None:
        for insp in inspirations:
            vec = np.loadtxt(insp[0], delimiter=',')
            if vec.ndim == 2: vec = vec[0]
            results = insp[3] if len(insp) > 3 else []
            drag_lift = results[0] if len(results) > 0 else [0, 0]
            feedback = results[2] if len(results) > 2 else ""
            
            # Inspirations: no images, just feedback text
            llm_context.append({
                'vector': vec.tolist(),
                'reward': insp[2],
                'ranking': insp[1],
                'drag': drag_lift[0],
                'lift': drag_lift[1],
                'feedback': feedback,
                'images': []  # No images for inspirations
            })
    
    base_csv = parent[0] if parent is not None else None
    name = f"design_{iteration_nb}"
    os.makedirs(output_dir, exist_ok=True)
    
    debug_dir = os.path.join(output_dir, name, 'context') if debug else None
    x = run_llm_action(action, llm_context, output_dir, base_csv=base_csv, name=name, skip_vis=True, debug_dir=debug_dir)
    return x

def powerlaw_sample_parent_and_inspiration(database, n_inspiration, alpha=3.0):
    """
    Sample parent and inspirations using rank-based selection.
    Database is sorted by fitness (best first), so rank r_i = i + 1 (1-indexed).
    Selection probability: p_i = r_i^(-alpha) / sum(r_j^(-alpha))
    
    Higher alpha = more exploitation (favor top ranks)
    Lower alpha = more exploration (more uniform)
    """
    n_items = len(database)
    if n_items == 0:
        return None, []
    
    # Calculate rank-based selection probabilities
    # ranks are 1, 2, 3, ..., n_items (1-indexed, rank 1 = best)
    ranks = np.arange(1, n_items + 1)
    probabilities = ranks ** (-alpha)
    probabilities = probabilities / probabilities.sum()  # normalize
    
    # Sample without replacement
    n_needed = min(1 + n_inspiration, n_items)
    indices = np.random.choice(n_items, size=n_needed, replace=False, p=probabilities)
    
    parent = database[indices[0]]
    inspirations = [database[i] for i in indices[1:]]
    return parent, inspirations

def run_iteration(database, iteration_nb, output_dir, n_inspirations, action, debug=False):
    parent, inspirations = powerlaw_sample_parent_and_inspiration(database, n_inspirations)
    x = generate_design(parent, inspirations, output_dir, iteration_nb, action, debug=debug)
    
    if x:
        case_dir = os.path.join(output_dir, f'design_{iteration_nb}')
        print(f"DEBUG run_iteration: output_dir={output_dir}")
        print(f"DEBUG run_iteration: case_dir={case_dir}")
        print(f"DEBUG run_iteration: case_dir absolute={os.path.abspath(case_dir)}")
        os.makedirs(case_dir, exist_ok=True)
        reward, results = run_simulation(x, case_dir)
        database = update_database(database, x, reward, results)
    else:
        reward = -10.0
    
    best_reward = np.max(database[:, 2].astype(float))
    return database, reward, best_reward

def initialize_database(d0, output_dir):
    case_dir = os.path.join(output_dir, 'initial')
    os.makedirs(case_dir, exist_ok=True)
    reward, results = run_simulation(d0, case_dir)
    return np.array([[d0, 0, reward, results]], dtype=object)

def run_benchmark(d0, n_iterations, n_inspirations, action, output_dir, initialize_n_sample=0, debug=False):
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize database with baseline design if provided, otherwise start empty
    if d0 is not None:
        database = initialize_database(d0, output_dir)
        best_x = d0
        best_reward = database[0, 2]
        cached = [database[0].copy()]
    else:
        database = np.array([], dtype=object).reshape(0, 4)
        best_x = None
        best_reward = -np.inf
        cached = []
    
    for i in range(n_iterations):
        if i < initialize_n_sample:
            # Initial population phase: generate with no context for diversity
            print(f"\n--- Iteration {i+1}/{n_iterations} [INITIAL SAMPLE {i+1}/{initialize_n_sample}] (action: {action}, no context) ---")
            x = generate_design(None, None, output_dir, i, action, debug=debug)
            if x:
                case_dir = os.path.join(output_dir, f'design_{i}')
                os.makedirs(case_dir, exist_ok=True)
                reward, results = run_simulation(x, case_dir)
                database = update_database(database, x, reward, results)
            else:
                reward = -10.0
            current_best = np.max(database[:, 2].astype(float)) if len(database) > 0 else -np.inf
        else:
            # Context-driven phase: ramp inspirations from 0 based on iterations since init phase ended
            context_iter = i - initialize_n_sample
            current_inspirations = max(0, min(context_iter, n_inspirations, len(database) - 1))
            print(f"\n--- Iteration {i+1}/{n_iterations} (action: {action}, inspirations: {current_inspirations}) ---")
            database, reward, current_best = run_iteration(database, i, output_dir, current_inspirations, action, debug=debug)
        
        if len(database) > 0:
            best_idx = np.argmax(database[:, 2].astype(float))
            cached.append(database[best_idx].copy())
            if current_best > best_reward:
                best_reward = current_best
                best_x = database[best_idx, 0]

        print(f"Reward: {reward:.4f}, Best: {best_reward:.4f}")
    
    return best_x, cached

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run benchmark with specific LLM action')
    parser.add_argument('--action', type=str, required=True,
                        choices=['generate', 'generate_direct', 'modify', 'modify_direct', 'gaussain', 'gaussian'],
                        help='LLM action to use for all iterations')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations')
    parser.add_argument('--inspirations', type=int, default=2, help='Maximum number of inspirations (grows from 0 to this value)')
    parser.add_argument('--baseline', type=str, default=None, help='Optional baseline CSV to initialize database (default: None, starts from scratch)')
    parser.add_argument('--initialize_n_sample', type=int, default=0,
                        help='Number of initial designs to generate without context for population diversity (default: 0, single design_0 as before)')
    parser.add_argument('--debug', action='store_true', help='Save LLM context and prompts to context/ subfolder')
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load baseline if provided
    baseline_csv = None
    if args.baseline:
        baseline_csv = os.path.join(base_dir, args.baseline)
        if not os.path.exists(baseline_csv):
            print(f"Error: Baseline CSV {baseline_csv} not found")
            sys.exit(1)
        print(f"Using baseline: {baseline_csv}")
    else:
        print("Starting from scratch (no baseline)")
    
    output_dir = os.path.join(base_dir, f'benchmark_results_{args.action}')
    best_design, cached = run_benchmark(baseline_csv, args.iterations, args.inspirations, args.action, output_dir, initialize_n_sample=args.initialize_n_sample, debug=args.debug)
    print(f"\nBest design: {best_design}")
    
    cache_data = []
    for entry in cached:
        csv_path, rank, reward, results = entry[0], entry[1], entry[2], entry[3]
        cache_data.append({
            'csv_path': str(csv_path),
            'rank': int(rank),
            'reward': float(reward),
            'drag_lift': results[0],
            'sol_images': results[1],
            'analysis': results[2] if len(results) > 2 else "",
            'shape_image': results[3] if len(results) > 3 else None
        })
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(cache_data, f, indent=2)
    
    print(f"Results saved to {output_dir}/results.json")
