import os
import sys
import json
import random
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
    # Clear contents of ./save subdirs before running to prevent accumulation
    # but keep directory structure intact for the environment
    for subdir in ['csv', 'png', 'sol', 'xml', 'rejected']:
        subdir_path = os.path.join('./save', subdir)
        if os.path.exists(subdir_path):
            for f in os.listdir(subdir_path):
                fp = os.path.join(subdir_path, f)
                if os.path.isfile(fp):
                    os.remove(fp)
    # Also clear drag_lift file
    dl_path = './save/drag_lift'
    if os.path.exists(dl_path):
        os.remove(dl_path)
    
    result = run_from_csv(csv_path, reset_first=True)
    reward = result[2]
    
    # Copy results from ./save to case-specific directory
    case_save_dir = os.path.join(case_dir, 'save')
    if os.path.exists('./save'):
        if os.path.exists(case_save_dir):
            shutil.rmtree(case_save_dir)
        shutil.copytree('./save', case_save_dir)
    
    results = post_process_results(case_save_dir, reward)
    return reward, results

def post_process_results(save_dir, reward=None):
    drag, lift = 0.0, 0.0
    dl_path = os.path.join(save_dir, 'drag_lift')
    if os.path.exists(dl_path):
        with open(dl_path, 'r') as f:
            lines = f.readlines()
            if lines:
                last = lines[-1].split()
                if len(last) >= 3:
                    drag = float(last[1])
                    lift = float(last[2])
    
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
        # Get the most recent shape PNG
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

def generate_design(parent, inspirations, output_dir, iteration_nb):
    llm_actions = ['generate', 'generate_direct', 'modify', 'modify_direct']
    action = random.choice(llm_actions)
    
    llm_context = []
    if parent is not None:
        vec = np.loadtxt(parent[0], delimiter=',')
        if vec.ndim == 2: vec = vec[0]
        results = parent[3] if len(parent) > 3 else []
        drag_lift = results[0] if len(results) > 0 else [0, 0]
        feedback = results[2] if len(results) > 2 else ""
        shape_image = results[3] if len(results) > 3 else None
        
        # For parent: include shape image for visual context
        parent_images = []
        if shape_image and os.path.exists(shape_image):
            parent_images.append(shape_image)
        
        llm_context.append({
            'vector': vec.tolist(), 
            'reward': parent[2], 
            'ranking': parent[1], 
            'images': parent_images,  # Shape image only for parent
            'drag': drag_lift[0],
            'lift': drag_lift[1],
            'feedback': feedback
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
                'images': [],  # No images for inspirations
                'drag': drag_lift[0],
                'lift': drag_lift[1],
                'feedback': feedback
            })
    
    base_csv = parent[0] if parent is not None else None
    name = f"design_{iteration_nb}"
    os.makedirs(output_dir, exist_ok=True)
    
    x = run_llm_action(action, llm_context, output_dir, base_csv=base_csv, name=name, skip_vis=True)
    return x

def powerlaw_sample_parent_and_inspiration(database, n_inspiration, alpha=0.1):
    n_items = len(database)
    if n_items == 0:
        return None, []
    
    r = stats.powerlaw.rvs(alpha, size=1 + n_inspiration)
    indices = np.clip(np.floor(r * n_items).astype(int), 0, n_items - 1)
    
    parent = database[indices[0]]
    inspirations = [database[i] for i in indices[1:]]
    return parent, inspirations

def run_iteration(database, iteration_nb, output_dir, n_inspirations):
    parent, inspirations = powerlaw_sample_parent_and_inspiration(database, n_inspirations)
    x = generate_design(parent, inspirations, output_dir, iteration_nb)
    
    if x:
        case_dir = os.path.join(output_dir, f'design_{iteration_nb}')
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

def run_benchmark(d0, n_iterations, n_inspirations, output_dir='benchmark_results'):
    os.makedirs(output_dir, exist_ok=True)
    database = initialize_database(d0, output_dir)
    best_x = d0
    best_reward = database[0, 2]
    
    cached = [database[0].copy()]
    
    for i in range(n_iterations):
        print(f"\n--- Iteration {i+1}/{n_iterations} ---")
        database, reward, current_best = run_iteration(database, i, output_dir, n_inspirations)
        
        # Get current best from database (rank 0)
        best_idx = np.argmax(database[:, 2].astype(float))
        cached.append(database[best_idx].copy())  # store best at this iteration
        
        if current_best > best_reward:
            best_reward = current_best
            best_x = database[best_idx, 0]

        print(f"Reward: {reward:.4f}, Best: {best_reward:.4f}")
    
    return best_x, cached

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    baseline_csv = os.path.join(base_dir, 'modified_env/baseline_action.csv')
    
    if not os.path.exists(baseline_csv):
        print(f"Error: {baseline_csv} not found")
        sys.exit(1)
    
    output_dir = os.path.join(base_dir, 'benchmark_results')
    best_design, cached = run_benchmark(baseline_csv, n_iterations=10, n_inspirations=2, output_dir=output_dir)
    print(f"\nBest design: {best_design}")
    
    # Save cached results to JSON
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
    
