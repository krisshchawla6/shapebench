import os
import sys
import re
import json
import argparse
import shutil
import subprocess
import numpy as np
import scipy.stats as stats
from PIL import Image

import google.generativeai as genai
from dotenv import load_dotenv

# Add paths for imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'LLM_Actions'))

from run_case import run_from_csv
from LLM_agent import run_llm_action
from prompts.reflection import (
    REFLECTION_SYSTEM, SCRATCHPAD_UPDATE_SYSTEM,
    build_reflection_prompt, build_scratchpad_update_prompt
)

# API setup
load_dotenv(os.path.join(BASE_DIR, 'LLM_Actions', '.env'))
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_KEY")
if api_key:
    genai.configure(api_key=api_key)


# --- Reflection helpers ---

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
    test_script = os.path.join(BASE_DIR, 'LLM_Actions', 'test_modification.py')
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


# --- Core benchmark functions ---

def run_simulation(csv_path, case_dir):
    result = run_from_csv(csv_path, reset_first=True)
    reward = result[2]

    case_save_dir = os.path.join(case_dir, 'save')
    if os.path.exists('./save'):
        if os.path.exists(case_save_dir):
            shutil.rmtree(case_save_dir)
        shutil.copytree('./save', case_save_dir)

    results = post_process_results(case_save_dir)
    return reward, results


def post_process_results(save_dir):
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

    reward_file = os.path.join(save_dir, 'reward_penalization')
    if os.path.exists(reward_file):
        with open(reward_file, 'r') as f:
            lines = f.readlines()
            if lines:
                parts = lines[-1].split()
                if len(parts) >= 2:
                    reward = float(parts[1])

    sol_dir = os.path.join(save_dir, 'sol')
    image_paths = []
    for name in ['1_p.png', '1_u.png', '1_v.png']:
        p = os.path.join(sol_dir, name)
        if os.path.exists(p):
            image_paths.append(p)

    return [[drag, lift], image_paths]


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


def load_images(image_paths):
    imgs = []
    for p in image_paths:
        if os.path.exists(p):
            try:
                imgs.append(Image.open(p))
            except Exception as e:
                print(f"Warning: Could not load image {p}: {e}")
    return imgs


def generate_design(parent, inspirations, output_dir, iteration_nb, action, scratchpad=""):
    llm_context = []
    if parent is not None:
        vec = np.loadtxt(parent[0], delimiter=',')
        if vec.ndim == 2: vec = vec[0]
        image_paths = parent[3][1] if parent[3] is not None else []
        llm_context.append({'vector': vec.tolist(), 'reward': parent[2], 'ranking': parent[1], 'images': load_images(image_paths)})

    if inspirations is not None:
        for insp in inspirations:
            vec = np.loadtxt(insp[0], delimiter=',')
            if vec.ndim == 2: vec = vec[0]
            llm_context.append({'vector': vec.tolist(), 'reward': insp[2], 'ranking': insp[1], 'images': []})

    base_csv = parent[0] if parent is not None else None
    name = f"design_{iteration_nb}"
    case_dir = os.path.join(output_dir, name)
    debug_dir = os.path.join(case_dir, 'context')
    os.makedirs(output_dir, exist_ok=True)

    x = run_llm_action(action, llm_context, output_dir, base_csv=base_csv, name=name,
                       skip_vis=True, debug_dir=debug_dir, random_strategy=True,
                       scratchpad=scratchpad)
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


def run_iteration(database, iteration_nb, output_dir, n_inspirations, action,
                  scratchpad="", scratchpad_path=""):
    parent, inspirations = powerlaw_sample_parent_and_inspiration(database, n_inspirations)
    x = generate_design(parent, inspirations, output_dir, iteration_nb, action, scratchpad=scratchpad)

    if x:
        case_dir = os.path.join(output_dir, f'design_{iteration_nb}')
        os.makedirs(case_dir, exist_ok=True)
        reward, results = run_simulation(x, case_dir)
        database = update_database(database, x, reward, results)

        # Reflection cycle
        try:
            scratchpad = do_reflection_cycle(x, case_dir, iteration_nb, scratchpad, scratchpad_path)
        except Exception as e:
            print(f"  Reflection failed (non-fatal): {e}")
    else:
        reward = -10.0

    best_reward = np.max(database[:, 2].astype(float))
    return database, reward, best_reward, scratchpad


def initialize_database(d0, output_dir):
    case_dir = os.path.join(output_dir, 'initial')
    os.makedirs(case_dir, exist_ok=True)
    reward, results = run_simulation(d0, case_dir)
    return np.array([[d0, 0, reward, results]], dtype=object)


def run_benchmark(d0, n_iterations, n_inspirations, action, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    database = initialize_database(d0, output_dir)
    best_x = d0
    best_reward = database[0, 2]
    cached = [database[0].copy()]

    # Initialize scratchpad
    scratchpad = ""
    scratchpad_path = os.path.join(output_dir, 'action_space.txt')

    for i in range(n_iterations):
        print(f"\n--- Iteration {i+1}/{n_iterations} (action: {action}) ---")
        print(f"  Scratchpad: {len(scratchpad)} chars")

        database, reward, current_best, scratchpad = run_iteration(
            database, i, output_dir, n_inspirations, action,
            scratchpad=scratchpad, scratchpad_path=scratchpad_path
        )

        best_idx = np.argmax(database[:, 2].astype(float))
        cached.append(database[best_idx].copy())

        if current_best > best_reward:
            best_reward = current_best
            best_x = database[best_idx, 0]

        print(f"  Reward: {reward:.4f}, Best: {best_reward:.4f}")

    return best_x, cached


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run benchmark with specific LLM action')
    parser.add_argument('--action', type=str, required=True,
                        choices=['generate', 'generate_direct', 'modify', 'modify_direct', 'gaussain', 'gaussian'],
                        help='LLM action to use for all iterations')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations')
    parser.add_argument('--inspirations', type=int, default=2, help='Number of inspirations')
    parser.add_argument('--output', type=str, default=None, help='Custom output directory')
    args = parser.parse_args()

    baseline_csv = os.path.join(BASE_DIR, 'baseline_action.csv')

    if not os.path.exists(baseline_csv):
        print(f"Error: {baseline_csv} not found")
        sys.exit(1)

    output_dir = args.output if args.output else os.path.join(BASE_DIR, f'benchmark_results_{args.action}')
    best_design, cached = run_benchmark(baseline_csv, args.iterations, args.inspirations, args.action, output_dir)
    print(f"\nBest design: {best_design}")

    cache_data = []
    for entry in cached:
        csv_path, rank, reward, results = entry[0], entry[1], entry[2], entry[3]
        cache_data.append({
            'csv_path': str(csv_path),
            'rank': int(rank),
            'reward': float(reward),
            'drag_lift': results[0],
            'sol_images': results[1]
        })

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(cache_data, f, indent=2)

    print(f"Results saved to {output_dir}/results.json")
