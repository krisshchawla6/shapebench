import os
import sys
import re
import json
import argparse
import shutil
import signal
import subprocess
import numpy as np
import scipy.stats as stats
from PIL import Image

import google.generativeai as genai
from dotenv import load_dotenv

ITERATION_TIMEOUT = 2700  # 45 minutes in seconds

class IterationTimeout(Exception):
    """Raised when a single iteration exceeds the time limit."""
    pass

def _timeout_handler(signum, frame):
    raise IterationTimeout("Iteration exceeded 45-minute time limit")

# Add paths for imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'modified_env'))
sys.path.insert(0, os.path.join(BASE_DIR, 'modified_env/LLM_Actions'))

from run_case import run_from_csv
from LLM_agent import run_llm_action
from Analysis_LLM import run_simulation_analysis
from prompts.reflection import (
    REFLECTION_SYSTEM, SCRATCHPAD_UPDATE_SYSTEM,
    build_reflection_prompt, build_scratchpad_update_prompt
)

try:
    from plot_lineage import plot_lineage_tree, save_lineage_json
    HAS_LINEAGE_PLOT = True
except ImportError:
    HAS_LINEAGE_PLOT = False
    print("Warning: plot_lineage not available (missing networkx/matplotlib), lineage plots disabled")

# API setup
load_dotenv(os.path.join(BASE_DIR, 'modified_env', 'LLM_Actions', '.env'))
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_KEY")
if api_key:
    genai.configure(api_key=api_key)


# =============================================================================
# REFLECTION HELPERS
# =============================================================================

def call_gemini(system_prompt, user_prompt, images=None, temperature=0.7):
    """Simple Gemini call returning text."""
    model = genai.GenerativeModel('gemini-3-flash-preview', system_instruction=system_prompt)
    content = [user_prompt] + (list(images) if images else [])
    config = genai.types.GenerationConfig(temperature=temperature)
    response = model.generate_content(content, generation_config=config)
    return response.text


def extract_scratchpad(text):
    """Extract scratchpad content from LLM response."""
    match = re.search(r'```\n?(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def visualize_design(csv_path, case_dir):
    """Generate geometry PNG from action CSV. Returns image path or None."""
    test_script = os.path.join(BASE_DIR, 'modified_env', 'LLM_Actions', 'test_modification.py')
    vis_dir = os.path.join(case_dir, 'geometry')
    os.makedirs(vis_dir, exist_ok=True)
    subprocess.run([sys.executable, test_script, csv_path, '-o', vis_dir],
                   capture_output=True, text=True)
    for f in os.listdir(vis_dir):
        if f.endswith('_geometry.png'):
            return os.path.join(vis_dir, f)
    return None


def run_reflection(intended_params, actual_action, designer_analysis, designer_reasoning,
                   geometry_image_path, debug_dir):
    """Run reflection: compare designer intent vs produced geometry."""
    user_prompt = build_reflection_prompt(
        intended_params, actual_action, designer_analysis, designer_reasoning
    )
    images = []
    if geometry_image_path and os.path.exists(geometry_image_path):
        images.append(Image.open(geometry_image_path))

    reflection = call_gemini(REFLECTION_SYSTEM, user_prompt, images)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, 'reflection.txt'), 'w') as f:
            f.write(reflection)
        with open(os.path.join(debug_dir, 'reflection_prompt.txt'), 'w') as f:
            f.write(user_prompt)
    return reflection


def run_scratchpad_update(current_scratchpad, reflection_text, intended_params, iteration, debug_dir):
    """Run scratchpad update and return new scratchpad text."""
    user_prompt = build_scratchpad_update_prompt(
        current_scratchpad, reflection_text, intended_params, iteration
    )
    response = call_gemini(SCRATCHPAD_UPDATE_SYSTEM, user_prompt)
    updated = extract_scratchpad(response)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, 'scratchpad_update_prompt.txt'), 'w') as f:
            f.write(user_prompt)
        with open(os.path.join(debug_dir, 'scratchpad_update_raw.txt'), 'w') as f:
            f.write(response)
    return updated


def do_reflection_cycle(csv_path, case_dir, iteration_nb, scratchpad, scratchpad_path):
    """Full reflection cycle: visualize -> reflect -> update scratchpad. Returns updated scratchpad."""
    debug_dir = os.path.join(case_dir, 'context')
    os.makedirs(debug_dir, exist_ok=True)

    # Load designer outputs saved by run_llm_action
    intended_params = {}
    params_path = os.path.join(debug_dir, 'llm_params.json')
    if os.path.exists(params_path):
        with open(params_path) as f:
            intended_params = json.load(f)

    actual_action = np.loadtxt(csv_path, delimiter=',').flatten()

    designer_analysis = ""
    analysis_path = os.path.join(debug_dir, 'llm_analysis.txt')
    if os.path.exists(analysis_path):
        with open(analysis_path) as f:
            designer_analysis = f.read().strip()

    designer_reasoning = ""
    reasoning_path = os.path.join(debug_dir, 'llm_rationale.txt')
    if os.path.exists(reasoning_path):
        with open(reasoning_path) as f:
            designer_reasoning = f.read().strip()

    # Visualize geometry
    geom_image = visualize_design(csv_path, case_dir)

    # Reflect
    reflection = run_reflection(
        intended_params, actual_action,
        designer_analysis, designer_reasoning,
        geom_image, debug_dir
    )
    print(f"  Reflection: {len(reflection)} chars")

    # Update scratchpad
    scratchpad = run_scratchpad_update(
        scratchpad, reflection, intended_params, iteration_nb + 1, debug_dir
    )
    with open(scratchpad_path, 'w') as f:
        f.write(scratchpad)
    print(f"  Scratchpad: {len(scratchpad)} chars")

    return scratchpad


# =============================================================================
# CORE BENCHMARK FUNCTIONS
# =============================================================================

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

    dl_path = os.path.join(save_dir, 'drag_lift')
    if os.path.exists(dl_path):
        with open(dl_path, 'r') as f:
            lines = f.readlines()
            if lines:
                last = lines[-1].split()
                if len(last) >= 3:
                    drag = float(last[1])
                    lift = float(last[2])

    if reward is None:
        reward_file = os.path.join(save_dir, 'reward_penalization')
        if os.path.exists(reward_file):
            with open(reward_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    parts = lines[-1].split()
                    if len(parts) >= 2:
                        reward = float(parts[1])

    sol_dir = os.path.join(save_dir, 'sol')
    sol_images = [
        os.path.join(sol_dir, '1_p.png'),
        os.path.join(sol_dir, '1_u.png'),
        os.path.join(sol_dir, '1_v.png')
    ]

    png_dir = os.path.join(save_dir, 'png')
    shape_image = None
    if os.path.exists(png_dir):
        shape_pngs = sorted([f for f in os.listdir(png_dir) if f.startswith('shape_') and f.endswith('.png')])
        if shape_pngs:
            shape_image = os.path.join(png_dir, shape_pngs[-1])

    metrics = {'drag': drag, 'lift': lift}
    if reward is not None:
        metrics['reward'] = reward

    try:
        analysis_text = run_simulation_analysis(sol_images, metrics)
    except Exception as e:
        print(f"Analysis failed: {e}")
        analysis_text = ""

    return [[drag, lift], sol_images, analysis_text, shape_image]


def update_database(database, x, reward, results, island_idx=0):
    """Database columns: [csv_path, rank, reward, results, island_idx]"""
    entry = np.array([[x, 0, reward, results, island_idx]], dtype=object)
    if len(database) == 0:
        return entry
    database = np.append(database, entry, axis=0)
    indices = np.argsort(database[:, 2].astype(float))[::-1]
    database = database[indices]
    for i in range(len(database)):
        database[i, 1] = i
    return database


def generate_design(parent, inspirations, output_dir, iteration_nb, action,
                    debug=False, scratchpad=""):
    llm_context = []

    if parent is not None:
        vec = np.loadtxt(parent[0], delimiter=',')
        if vec.ndim == 2: vec = vec[0]
        results = parent[3] if len(parent) > 3 else []
        drag_lift = results[0] if len(results) > 0 else [0, 0]
        sol_images = results[1] if len(results) > 1 else []
        feedback = results[2] if len(results) > 2 else ""
        shape_image = results[3] if len(results) > 3 else None

        parent_images = []
        if shape_image and os.path.exists(shape_image):
            parent_images.append(shape_image)
        for sol_img in sol_images:
            if sol_img and os.path.exists(sol_img):
                parent_images.append(sol_img)

        llm_context.append({
            'vector': vec.tolist(),
            'reward': parent[2],
            'ranking': parent[1],
            'drag': drag_lift[0],
            'lift': drag_lift[1],
            'feedback': feedback,
            'images': parent_images
        })

    if inspirations is not None:
        for insp in inspirations:
            vec = np.loadtxt(insp[0], delimiter=',')
            if vec.ndim == 2: vec = vec[0]
            results = insp[3] if len(insp) > 3 else []
            drag_lift = results[0] if len(results) > 0 else [0, 0]
            feedback = results[2] if len(results) > 2 else ""

            llm_context.append({
                'vector': vec.tolist(),
                'reward': insp[2],
                'ranking': insp[1],
                'drag': drag_lift[0],
                'lift': drag_lift[1],
                'feedback': feedback,
                'images': []
            })

    base_csv = parent[0] if parent is not None else None
    name = f"design_{iteration_nb}"
    os.makedirs(output_dir, exist_ok=True)

    debug_dir = os.path.join(output_dir, name, 'context') if debug else None
    x = run_llm_action(action, llm_context, output_dir, base_csv=base_csv, name=name,
                       skip_vis=True, debug_dir=debug_dir, scratchpad=scratchpad)
    return x

# =============================================================================
# SAMPLING
# =============================================================================

def powerlaw_sample_parent_and_inspiration(database, n_inspiration, alpha=3.0):
    """Rank-based powerlaw selection. Higher alpha = more exploitation."""
    n_items = len(database)
    if n_items == 0:
        return None, []

    ranks = np.arange(1, n_items + 1)
    probabilities = ranks ** (-alpha)
    probabilities = probabilities / probabilities.sum()

    n_needed = min(1 + n_inspiration, n_items)
    indices = np.random.choice(n_items, size=n_needed, replace=False, p=probabilities)

    parent = database[indices[0]]
    inspirations = [database[i] for i in indices[1:]]
    return parent, inspirations

def powerlaw_sample_parent_from_island(database, island_idx, alpha=3.0):
    """Sample a single parent from an island using powerlaw rank-based selection."""
    mask = np.array([int(entry[4]) == island_idx for entry in database])
    island_db = database[mask]
    if len(island_db) == 0:
        return None
    ranks = np.arange(1, len(island_db) + 1)
    probabilities = ranks ** (-alpha)
    probabilities = probabilities / probabilities.sum()
    idx = np.random.choice(len(island_db), p=probabilities)
    return island_db[idx]

def sample_inspirations_from_island(database, island_idx, parent_csv, n_inspiration, elite_ratio=0.3):
    """Sample inspirations from an island."""
    if n_inspiration <= 0:
        return []
    mask = np.array([int(entry[4]) == island_idx for entry in database])
    island_db = database[mask]
    if len(island_db) == 0:
        return []

    pool = [entry for entry in island_db if entry[0] != parent_csv]
    if not pool:
        return []

    inspirations = []
    used_csvs = set()

    inspirations.append(pool[0])
    used_csvs.add(pool[0][0])
    if len(inspirations) >= n_inspiration:
        return inspirations[:n_inspiration]

    num_elites = max(0, int(n_inspiration * elite_ratio))
    for entry in pool[1:]:
        if len(inspirations) >= n_inspiration or num_elites <= 0:
            break
        if entry[0] not in used_csvs:
            inspirations.append(entry)
            used_csvs.add(entry[0])
            num_elites -= 1
    if len(inspirations) >= n_inspiration:
        return inspirations[:n_inspiration]

    remaining_pool = [e for e in pool if e[0] not in used_csvs]
    if remaining_pool:
        needed = n_inspiration - len(inspirations)
        n_random = min(needed, len(remaining_pool))
        random_indices = np.random.choice(len(remaining_pool), size=n_random, replace=False)
        for idx in random_indices:
            inspirations.append(remaining_pool[idx])

    return inspirations[:n_inspiration]

# =============================================================================
# MIGRATION
# =============================================================================

def perform_migration(database, num_islands, migration_rate):
    """Move-based elitist migration."""
    if num_islands < 2 or migration_rate <= 0:
        return database
    total_migrated = 0
    for source_idx in range(num_islands):
        mask = np.array([int(entry[4]) == source_idx for entry in database])
        island_indices = np.where(mask)[0]
        island_size = len(island_indices)
        if island_size <= 1:
            continue
        best_idx = island_indices[0]
        eligible = [idx for idx in island_indices if idx != best_idx]
        if not eligible:
            continue
        num_migrants = max(1, int(island_size * migration_rate))
        num_migrants = min(num_migrants, len(eligible))
        migrants = np.random.choice(eligible, size=num_migrants, replace=False)
        dest_islands = [i for i in range(num_islands) if i != source_idx]
        for migrant_idx in migrants:
            new_island = np.random.choice(dest_islands)
            database[migrant_idx, 4] = new_island
            total_migrated += 1
    if total_migrated > 0:
        print(f"  Migration: moved {total_migrated} designs between islands")
    return database

# =============================================================================
# ITERATION
# =============================================================================

def run_iteration(database, iteration_nb, output_dir, n_inspirations, action,
                  alpha=3.0, num_islands=1, debug=False,
                  scratchpad="", scratchpad_path=""):
    if num_islands > 1:
        occupied = list(set(int(entry[4]) for entry in database))
        island_idx = np.random.choice(occupied)
        parent = powerlaw_sample_parent_from_island(database, island_idx, alpha=alpha)
        parent_csv = parent[0] if parent is not None else None
        inspirations = sample_inspirations_from_island(database, island_idx, parent_csv, n_inspirations) if parent is not None else []
    else:
        parent, inspirations = powerlaw_sample_parent_and_inspiration(database, n_inspirations, alpha=alpha)

    x = generate_design(parent, inspirations, output_dir, iteration_nb, action,
                        debug=debug, scratchpad=scratchpad)
    parent_island = int(parent[4]) if parent is not None else 0
    parent_csv = parent[0] if parent is not None else None

    if x:
        case_dir = os.path.join(output_dir, f'design_{iteration_nb}')
        os.makedirs(case_dir, exist_ok=True)
        reward, results = run_simulation(x, case_dir)
        database = update_database(database, x, reward, results, island_idx=parent_island)

        # Reflection cycle
        try:
            scratchpad = do_reflection_cycle(x, case_dir, iteration_nb, scratchpad, scratchpad_path)
        except Exception as e:
            print(f"  Reflection failed (non-fatal): {e}")
    else:
        reward = -10.0

    best_reward = np.max(database[:, 2].astype(float))
    return database, reward, best_reward, parent_csv, x, scratchpad

def initialize_database(d0, output_dir, island_idx=0):
    case_dir = os.path.join(output_dir, 'initial')
    os.makedirs(case_dir, exist_ok=True)
    reward, results = run_simulation(d0, case_dir)
    return np.array([[d0, 0, reward, results, island_idx]], dtype=object)

# =============================================================================
# MAIN BENCHMARK LOOP
# =============================================================================

def run_benchmark(d0, n_iterations, n_inspirations, action, output_dir,
                  initialize_n_sample=0, alpha=3.0, num_islands=1,
                  migration_interval=10, migration_rate=0.1, debug=False):
    os.makedirs(output_dir, exist_ok=True)

    # Lineage tracking
    csv_to_iter = {}
    lineage = []

    # Initialize scratchpad
    scratchpad = ""
    scratchpad_path = os.path.join(output_dir, 'action_space.txt')

    if d0 is not None:
        database = initialize_database(d0, output_dir, island_idx=0)
        best_x = d0
        best_reward = database[0, 2]
        cached = [database[0].copy()]
        csv_to_iter[d0] = 'baseline'
        lineage.append({'id': 'baseline', 'parent_id': None,
                        'reward': float(database[0, 2]), 'island': 0})
    else:
        database = np.array([], dtype=object).reshape(0, 5)
        best_x = None
        best_reward = -np.inf
        cached = []

    prev_alarm_handler = signal.signal(signal.SIGALRM, _timeout_handler)

    for i in range(n_iterations):
        try:
            signal.alarm(ITERATION_TIMEOUT)

            if i < initialize_n_sample:
                island_idx = i % num_islands
                print(f"\n--- Iteration {i+1}/{n_iterations} [INITIAL SAMPLE {i+1}/{initialize_n_sample}] (action: {action}, island: {island_idx}, no context) ---")
                print(f"  Scratchpad: {len(scratchpad)} chars")
                x = generate_design(None, None, output_dir, i, action, debug=debug,
                                    scratchpad=scratchpad)
                if x:
                    case_dir = os.path.join(output_dir, f'design_{i}')
                    os.makedirs(case_dir, exist_ok=True)
                    reward, results = run_simulation(x, case_dir)
                    database = update_database(database, x, reward, results, island_idx=island_idx)
                    csv_to_iter[x] = i
                    lineage.append({'id': i, 'parent_id': None,
                                    'reward': float(reward), 'island': island_idx})
                    # Reflection on initial samples too
                    try:
                        scratchpad = do_reflection_cycle(x, case_dir, i, scratchpad, scratchpad_path)
                    except Exception as e:
                        print(f"  Reflection failed (non-fatal): {e}")
                else:
                    reward = -10.0
                    lineage.append({'id': i, 'parent_id': None,
                                    'reward': -10.0, 'island': island_idx})
                current_best = np.max(database[:, 2].astype(float)) if len(database) > 0 else -np.inf
            else:
                current_inspirations = min(n_inspirations, len(database) - 1)
                print(f"\n--- Iteration {i+1}/{n_iterations} (action: {action}, inspirations: {current_inspirations}, islands: {num_islands}) ---")
                print(f"  Scratchpad: {len(scratchpad)} chars")
                database, reward, current_best, parent_csv, x, scratchpad = run_iteration(
                    database, i, output_dir, current_inspirations, action,
                    alpha=alpha, num_islands=num_islands, debug=debug,
                    scratchpad=scratchpad, scratchpad_path=scratchpad_path)

                parent_id = csv_to_iter.get(parent_csv) if parent_csv else None
                parent_island = int([e[4] for e in database if e[0] == parent_csv][0]) if parent_csv and any(e[0] == parent_csv for e in database) else 0
                if x:
                    csv_to_iter[x] = i
                lineage.append({'id': i, 'parent_id': parent_id,
                                'reward': float(reward),
                                'island': parent_island if x else 0})

                # Periodic migration
                context_iter = i - initialize_n_sample
                if num_islands > 1 and migration_interval > 0 and context_iter > 0 and context_iter % migration_interval == 0:
                    print(f"  Triggering migration at context iteration {context_iter}")
                    database = perform_migration(database, num_islands, migration_rate)

            signal.alarm(0)

            if len(database) > 0:
                best_idx = np.argmax(database[:, 2].astype(float))
                cached.append(database[best_idx].copy())
                if current_best > best_reward:
                    best_reward = current_best
                    best_x = database[best_idx, 0]

            if num_islands > 1 and len(database) > 0:
                pops = {isl: sum(1 for e in database if int(e[4]) == isl) for isl in range(num_islands)}
                print(f"Reward: {reward:.4f}, Best: {best_reward:.4f}, Islands: {pops}")
            else:
                print(f"Reward: {reward:.4f}, Best: {best_reward:.4f}")

        except IterationTimeout:
            signal.alarm(0)
            print(f"\n*** Iteration {i+1}/{n_iterations} TIMED OUT after 45 minutes — skipping to next iteration ***")
            lineage.append({'id': i, 'parent_id': None,
                            'reward': -10.0, 'island': 0})

        # Update lineage plot after each iteration
        if HAS_LINEAGE_PLOT:
            try:
                save_lineage_json(lineage, os.path.join(output_dir, 'lineage.json'))
                plot_lineage_tree(lineage, os.path.join(output_dir, 'lineage_tree.png'),
                    title=f"Design Lineage (iter {i+1}/{n_iterations})")
            except Exception as e:
                print(f"Lineage plot failed: {e}")

    signal.signal(signal.SIGALRM, prev_alarm_handler)
    return best_x, cached

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run benchmark with specific LLM action')
    parser.add_argument('--action', type=str, required=True,
                        choices=['generate', 'generate_direct', 'modify', 'modify_direct', 'gaussain', 'gaussian'],
                        help='LLM action to use for all iterations')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations')
    parser.add_argument('--inspirations', type=int, default=2, help='Maximum number of inspirations')
    parser.add_argument('--baseline', type=str, default=None, help='Optional baseline CSV (default: None)')
    parser.add_argument('--initialize_n_sample', type=int, default=0,
                        help='Number of initial designs without context (default: 0)')
    parser.add_argument('--alpha', type=float, default=3.0,
                        help='Power-law exponent for rank-based selection (default: 3.0)')
    parser.add_argument('--num_islands', type=int, default=1,
                        help='Number of islands for parallel evolution (default: 1)')
    parser.add_argument('--migration_interval', type=int, default=10,
                        help='Context-driven iterations between migrations (default: 10)')
    parser.add_argument('--migration_rate', type=float, default=0.1,
                        help='Fraction of island population to migrate (default: 0.1)')
    parser.add_argument('--debug', action='store_true', help='Save LLM context and prompts')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

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
    best_design, cached = run_benchmark(
        baseline_csv, args.iterations, args.inspirations, args.action, output_dir,
        initialize_n_sample=args.initialize_n_sample, alpha=args.alpha,
        num_islands=args.num_islands, migration_interval=args.migration_interval,
        migration_rate=args.migration_rate, debug=args.debug)
    print(f"\nBest design: {best_design}")

    cache_data = []
    for entry in cached:
        csv_path, rank, reward, results = entry[0], entry[1], entry[2], entry[3]
        island = int(entry[4]) if len(entry) > 4 else 0
        cache_data.append({
            'csv_path': str(csv_path),
            'rank': int(rank),
            'reward': float(reward),
            'island': island,
            'drag_lift': results[0],
            'sol_images': results[1],
            'analysis': results[2] if len(results) > 2 else "",
            'shape_image': results[3] if len(results) > 3 else None
        })

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(cache_data, f, indent=2)

    print(f"Results saved to {output_dir}/results.json")
