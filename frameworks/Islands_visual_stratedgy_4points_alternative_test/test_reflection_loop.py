"""
Test reflection loop: 5 iterative designs with parameter-geometry learning.
Each iteration: generate -> visualize -> reflect -> update scratchpad -> next.
No simulation -- uses geometry images only.
"""
import os
import sys
import re
import json
import shutil
import subprocess
import numpy as np

import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REFLECTION_ENV = os.path.join(BASE_DIR, 'testing_reflection')
sys.path.insert(0, REFLECTION_ENV)
sys.path.insert(0, os.path.join(REFLECTION_ENV, 'LLM_Actions'))

from LLM_agent import run_llm_action
from prompts.reflection import (
    REFLECTION_SYSTEM, SCRATCHPAD_UPDATE_SYSTEM,
    build_reflection_prompt, build_scratchpad_update_prompt
)

# API setup
load_dotenv(os.path.join(REFLECTION_ENV, 'LLM_Actions', '.env'))
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_KEY")
if api_key:
    genai.configure(api_key=api_key)

# Config
EXAMPLE_PARENT = os.path.join(BASE_DIR, 'example_parent')
OUTPUT_DIR = os.path.join(BASE_DIR, 'test_reflection_output')
N_ITERATIONS = 10
ACTION = 'gaussain'


def call_gemini(system_prompt, user_prompt, images=None, temperature=0.7):
    """Simple Gemini call returning text."""
    model = genai.GenerativeModel('gemini-3-flash-preview', system_instruction=system_prompt)
    content = [user_prompt] + (list(images) if images else [])
    config = genai.types.GenerationConfig(temperature=temperature)
    response = model.generate_content(content, generation_config=config)
    return response.text


def extract_scratchpad(text):
    """Extract scratchpad content from LLM response (everything after the prompt preamble)."""
    # Try to find a markdown code block first
    match = re.search(r'```\n?(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Otherwise return the full text
    return text.strip()


def load_parent_from_example():
    """Load parent data from example_parent directory."""
    save_dir = os.path.join(EXAMPLE_PARENT, 'save')

    # Action vector from context
    parent_action = [1.0, 0.004328, -0.93674, 0.833076, 0.303893, -0.690358,
                     0.879421, 0.670615, -0.250714, 1.0, 0.937862, 0.125332,
                     0.932845, 0.870981, 1.0, -1.0, 0.935066, 0.175294,
                     -0.853371, 0.498291, -0.355829, -0.96408, 0.011296, -0.87506]

    # Drag/lift from last line
    drag, lift = 0.0, 0.0
    dl_path = os.path.join(save_dir, 'drag_lift')
    if os.path.exists(dl_path):
        with open(dl_path) as f:
            lines = f.readlines()
            if lines:
                parts = lines[-1].split()
                if len(parts) >= 3:
                    drag, lift = float(parts[1]), float(parts[2])

    # Images
    images = []
    shape_img = os.path.join(save_dir, 'png', 'shape_1.png')
    if os.path.exists(shape_img):
        images.append(shape_img)
    for name in ['1_p.png', '1_u.png', '1_v.png']:
        p = os.path.join(save_dir, 'sol', name)
        if os.path.exists(p):
            images.append(p)

    # Feedback
    feedback = ""
    analysis_path = os.path.join(EXAMPLE_PARENT, 'context', 'llm_analysis.txt')
    if os.path.exists(analysis_path):
        with open(analysis_path) as f:
            feedback = f.read().strip()

    return {
        'action': parent_action,
        'reward': 1.036,
        'rank': 0,
        'drag': drag,
        'lift': lift,
        'images': images,
        'feedback': feedback
    }


def visualize_design(action_csv, output_dir, name):
    """Run test_modification.py to generate geometry PNG (no control points)."""
    test_script = os.path.join(REFLECTION_ENV, 'LLM_Actions', 'test_modification.py')
    vis_dir = os.path.join(output_dir, name)
    os.makedirs(vis_dir, exist_ok=True)
    cmd = [sys.executable, test_script, action_csv, '-o', vis_dir]
    subprocess.run(cmd, capture_output=True, text=True)
    for f in os.listdir(vis_dir):
        if f.endswith('_geometry.png'):
            return os.path.join(vis_dir, f)
    return None


def run_reflection(intended_params, actual_action, designer_analysis, designer_reasoning,
                   geometry_image_path, debug_dir):
    """Run reflection LLM call: compare intent vs result."""
    user_prompt = build_reflection_prompt(
        intended_params, actual_action, designer_analysis, designer_reasoning
    )
    images = []
    if geometry_image_path and os.path.exists(geometry_image_path):
        images.append(Image.open(geometry_image_path))

    reflection = call_gemini(REFLECTION_SYSTEM, user_prompt, images)

    if debug_dir:
        with open(os.path.join(debug_dir, 'reflection.txt'), 'w') as f:
            f.write(reflection)
        with open(os.path.join(debug_dir, 'reflection_prompt.txt'), 'w') as f:
            f.write(user_prompt)
    return reflection


def run_scratchpad_update(current_scratchpad, reflection_text, intended_params, iteration, debug_dir):
    """Run scratchpad update LLM call."""
    user_prompt = build_scratchpad_update_prompt(
        current_scratchpad, reflection_text, intended_params, iteration
    )
    response = call_gemini(SCRATCHPAD_UPDATE_SYSTEM, user_prompt)
    updated = extract_scratchpad(response)

    if debug_dir:
        with open(os.path.join(debug_dir, 'scratchpad_update_prompt.txt'), 'w') as f:
            f.write(user_prompt)
        with open(os.path.join(debug_dir, 'scratchpad_update_raw.txt'), 'w') as f:
            f.write(response)
    return updated


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    parent = load_parent_from_example()

    # Save parent action CSV
    parent_csv = os.path.join(OUTPUT_DIR, 'parent_action.csv')
    np.savetxt(parent_csv, [parent['action']], delimiter=',', fmt='%.6f')

    # Initialize scratchpad
    scratchpad = ""
    scratchpad_path = os.path.join(OUTPUT_DIR, 'action_space.txt')

    # Current parent context for iteration
    current_parent = parent

    for i in range(N_ITERATIONS):
        print(f"\n{'='*60}")
        print(f"ITERATION {i+1}/{N_ITERATIONS}")
        print(f"{'='*60}")

        design_name = f"design_{i}"
        debug_dir = os.path.join(OUTPUT_DIR, design_name, 'context')
        os.makedirs(debug_dir, exist_ok=True)

        # Save current scratchpad to this design's context
        with open(os.path.join(debug_dir, 'action_space.txt'), 'w') as f:
            f.write(scratchpad)

        # --- Step 1: Generate design (gaussain) with scratchpad in context ---
        llm_context = [{
            'vector': current_parent['action'],
            'reward': current_parent['reward'],
            'ranking': current_parent['rank'],
            'drag': current_parent['drag'],
            'lift': current_parent['lift'],
            'feedback': current_parent['feedback'],
            'images': current_parent['images']
        }]

        print(f"  Generating design with scratchpad ({len(scratchpad)} chars)...")
        csv_path = run_llm_action(
            ACTION, llm_context, OUTPUT_DIR,
            base_csv=parent_csv, name=design_name,
            skip_vis=True, debug_dir=debug_dir,
            random_strategy=True, scratchpad=scratchpad
        )

        if not csv_path or not os.path.exists(csv_path):
            print(f"  FAILED to generate design, skipping iteration")
            continue

        # Load actual action and LLM params
        actual_action = np.loadtxt(csv_path, delimiter=',').flatten()
        intended_params = {}
        params_path = os.path.join(debug_dir, 'llm_params.json')
        if os.path.exists(params_path):
            with open(params_path) as f:
                intended_params = json.load(f)

        # Load designer's analysis and reasoning
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

        print(f"  Generated: {csv_path} ({len(actual_action)} values)")

        # --- Step 2: Visualize geometry ---
        print(f"  Visualizing geometry...")
        geom_image = visualize_design(csv_path, OUTPUT_DIR, design_name)
        print(f"  Geometry: {geom_image}")

        # --- Step 3: Reflection ---
        print(f"  Running reflection...")
        reflection = run_reflection(
            intended_params, actual_action,
            designer_analysis, designer_reasoning,
            geom_image, debug_dir
        )
        print(f"  Reflection ({len(reflection)} chars):")
        for line in reflection.strip().split('\n')[:5]:
            print(f"    {line.strip()}")

        # --- Step 4: Update scratchpad ---
        print(f"  Updating scratchpad...")
        scratchpad = run_scratchpad_update(
            scratchpad, reflection, intended_params, i + 1, debug_dir
        )
        with open(scratchpad_path, 'w') as f:
            f.write(scratchpad)
        print(f"  Scratchpad updated ({len(scratchpad)} chars)")

        # --- Step 5: New design becomes parent for next iteration ---
        # Copy geometry image to be used as parent image next iteration
        new_images = []
        if geom_image and os.path.exists(geom_image):
            new_images.append(geom_image)

        current_parent = {
            'action': actual_action.tolist(),
            'reward': 0.0,  # no sim, unknown reward
            'rank': 'N/A',
            'drag': 0.0,
            'lift': 0.0,
            'images': new_images,
            'feedback': reflection  # use reflection as feedback
        }
        # Update parent CSV for next iteration
        parent_csv = csv_path

    print(f"\n{'='*60}")
    print(f"DONE. Results in: {OUTPUT_DIR}")
    print(f"Final scratchpad: {scratchpad_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
