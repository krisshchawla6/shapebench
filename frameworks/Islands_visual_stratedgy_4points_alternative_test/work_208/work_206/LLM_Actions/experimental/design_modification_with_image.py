"""
Experimental: Multimodal design generation using image input.
Takes a parent action CSV + geometry/flow images, generates new designs
via the gaussain action pipeline (same as run_benchmark).
"""
import os
import sys
import json
import numpy as np
import subprocess
from PIL import Image
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from dotenv import load_dotenv

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, os.path.dirname(PARENT_DIR))

from llm_design_actions import run_action, gaussian_sampling
from prompts.base import format_context, format_response_instructions
from prompts.gaussain import (
    GENERATE_SYSTEM, GENERATE_USER, GENERATE_format,
    GENERATE_STRATEGIES, GENERATE_STRATEGY_NAMES, sample_strategy
)

# Load API key
dotenv_path = os.path.join(PARENT_DIR, '.env')
load_dotenv(dotenv_path)
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_KEY")
if api_key:
    genai.configure(api_key=api_key)


def load_images(image_paths: List[str]) -> List[Any]:
    """Load images from paths as PIL Image objects."""
    images = []
    for path in image_paths:
        if path and os.path.exists(path):
            try:
                images.append(Image.open(path))
            except Exception as e:
                print(f"Warning: Could not load image {path}: {e}")
    return images


def get_gemini_response(system_prompt, user_prompt, images=None, temperature=1.0):
    """Get response from Gemini with images. Returns (params, analysis, rationale)."""
    try:
        model = genai.GenerativeModel(
            'gemini-3-pro-preview',
            system_instruction=system_prompt
        )
        content = [user_prompt] + (list(images) if images else [])
        config = genai.types.GenerationConfig(temperature=temperature)
        response = model.generate_content(content, generation_config=config)
        text = response.text

        # Extract structured sections
        import re
        analysis = None
        rationale = None
        analysis_match = re.search(r'##\s*Analysis\s*\n(.*?)(?=##|\{|$)', text, re.DOTALL | re.IGNORECASE)
        if analysis_match:
            analysis = analysis_match.group(1).strip()
        rationale_match = re.search(r'##\s*(?:Design\s+)?Reasoning\s*\n(.*?)(?=##|\{|$)', text, re.DOTALL | re.IGNORECASE)
        if rationale_match:
            rationale = rationale_match.group(1).strip()

        # Extract JSON
        params_match = re.search(r'##\s*Design\s+Parameters\s*\n.*?(\{.*?\})', text, re.DOTALL | re.IGNORECASE)
        json_str = params_match.group(1).strip() if params_match else None
        if json_str is None:
            start, end = text.find('{'), text.rfind('}') + 1
            if start != -1 and end > start:
                json_str = text[start:end]
        if json_str is None:
            print(f"No JSON found in response: {text[:500]}")
            return {}, analysis, rationale

        params = json.loads(json_str)
        return params, analysis, rationale

    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
        return {}, None, None
    except Exception as e:
        print(f"Gemini Error: {e}")
        return {}, None, None


def generate_with_images(
    parent_action_csv: str,
    image_paths: List[str],
    output_dir: str,
    name: str = "design",
    parent_reward: float = 0.0,
    parent_rank: int = 0,
    parent_drag: float = 0.0,
    parent_lift: float = 0.0,
    feedback: str = "",
    strategy_idx: int = None,
    temperature: float = 1.0,
    debug: bool = True
) -> Optional[str]:
    """
    Generate a new design using the gaussain action pipeline with image context.
    Mirrors run_benchmark_action.py's generate_design() + run_llm_action() flow.

    Args:
        parent_action_csv: Path to parent's flat action CSV (24 values for 8 pts)
        image_paths: List of image paths [shape_png, 1_p.png, 1_u.png, 1_v.png]
        output_dir: Output directory
        name: Design name prefix
        parent_reward/rank/drag/lift: Parent metrics
        feedback: Analysis text for parent
        strategy_idx: Strategy index (None = random)
        temperature: LLM temperature
        debug: Save debug context

    Returns:
        Path to generated action CSV, or None on failure
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load parent action vector
    vec = np.loadtxt(parent_action_csv, delimiter=',')
    if vec.ndim == 2:
        vec = vec[0]

    # Build LLM context exactly like run_benchmark_action.py
    llm_context = [{
        'vector': vec.tolist(),
        'reward': parent_reward,
        'ranking': parent_rank,
        'drag': parent_drag,
        'lift': parent_lift,
        'feedback': feedback,
        'images': [p for p in image_paths if p and os.path.exists(p)]
    }]

    # Format context text
    ctx_text = format_context(llm_context)

    # Select strategy
    if strategy_idx is None:
        strategy_idx, strategy_name = sample_strategy()
    else:
        strategy_idx = strategy_idx % len(GENERATE_STRATEGY_NAMES)
        strategy_name = GENERATE_STRATEGY_NAMES[strategy_idx]

    # Build prompts (same as LLM_agent.py gaussain branch)
    system_prompt = GENERATE_SYSTEM
    if GENERATE_STRATEGIES and strategy_idx is not None:
        sidx = strategy_idx % len(GENERATE_STRATEGIES)
        strategy_text = GENERATE_STRATEGIES[sidx]
        strategy_block = f"\n**Strategy Focus**: {strategy_text}\n"
    else:
        strategy_block = ""

    response_fmt = format_response_instructions(GENERATE_format)
    user_prompt = GENERATE_USER.format(
        context=ctx_text,
        strategy_block=strategy_block,
        response_format=response_fmt
    )

    # Load images as PIL objects
    pil_images = load_images(image_paths)

    # Save debug info
    debug_dir = os.path.join(output_dir, name, 'context') if debug else None
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, 'context.txt'), 'w') as f:
            f.write(ctx_text)
        with open(os.path.join(debug_dir, 'system_prompt.txt'), 'w') as f:
            f.write(system_prompt)
        with open(os.path.join(debug_dir, 'user_prompt.txt'), 'w') as f:
            f.write(user_prompt)
        with open(os.path.join(debug_dir, 'strategy.txt'), 'w') as f:
            f.write(f"Strategy: {strategy_name} (idx={strategy_idx})")
        import shutil
        for idx, img_path in enumerate(image_paths):
            if img_path and os.path.exists(img_path):
                dest = os.path.join(debug_dir, f"image_{idx}_{os.path.basename(img_path)}")
                shutil.copy2(img_path, dest)

    # Call Gemini
    params, analysis, rationale = get_gemini_response(
        system_prompt, user_prompt, pil_images, temperature=temperature
    )
    print(f"Strategy: {strategy_name}")
    print(f"LLM params: {json.dumps(params, indent=2)}")

    if debug_dir:
        if analysis:
            with open(os.path.join(debug_dir, 'llm_analysis.txt'), 'w') as f:
                f.write(analysis)
        if rationale:
            with open(os.path.join(debug_dir, 'llm_rationale.txt'), 'w') as f:
                f.write(rationale)
        with open(os.path.join(debug_dir, 'llm_params.json'), 'w') as f:
            json.dump(params, f, indent=2)

    if not params or 'params' not in params:
        print(f"Error: Missing 'params' in LLM response. Got: {list(params.keys())}")
        return None

    # Clean schema fields
    for field in ['$schema', 'title', 'description', 'type', 'properties', 'required']:
        params.pop(field, None)
    if 'n_sp' not in params:
        params['n_sp'] = 20
    if 'n_cp' not in params:
        params['n_cp'] = 8

    # Execute gaussain action (LLM mean -> gaussian sample)
    csv_path = run_action('gaussain', n_cp=params['n_cp'], n_sp=params['n_sp'],
                          params=params['params'], out_dir=output_dir, name=name)

    # Generate visualization
    if csv_path:
        test_mod_script = os.path.join(PARENT_DIR, 'test_modification.py')
        vis_dir = os.path.join(output_dir, name)
        os.makedirs(vis_dir, exist_ok=True)
        cmd = [sys.executable, test_mod_script, csv_path, '-o', vis_dir]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Visualization warning: {result.stderr[:200]}")

    return csv_path


def batch_generate(
    parent_action_csv: str,
    image_paths: List[str],
    output_dir: str,
    n_designs: int = 5,
    parent_reward: float = 0.0,
    parent_rank: int = 0,
    parent_drag: float = 0.0,
    parent_lift: float = 0.0,
    feedback: str = "",
    temperature: float = 1.0
) -> List[str]:
    """Generate multiple designs from one parent."""
    results = []
    for i in range(n_designs):
        print(f"\n{'='*60}")
        print(f"Design {i+1}/{n_designs}")
        print(f"{'='*60}")
        csv_path = generate_with_images(
            parent_action_csv=parent_action_csv,
            image_paths=image_paths,
            output_dir=output_dir,
            name=f"design_{i}",
            parent_reward=parent_reward,
            parent_rank=parent_rank,
            parent_drag=parent_drag,
            parent_lift=parent_lift,
            feedback=feedback,
            temperature=temperature
        )
        if csv_path:
            results.append(csv_path)
            action = np.loadtxt(csv_path, delimiter=',').flatten()
            print(f"  -> {csv_path} ({len(action)} values)")
        else:
            print(f"  -> FAILED")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate airfoil designs using image + action context")
    parser.add_argument("action_csv", help="Path to parent action CSV (flat 24-value)")
    parser.add_argument("--images", nargs='+', required=True,
                        help="Image paths: shape.png 1_p.png 1_u.png 1_v.png")
    parser.add_argument("-o", "--output", default="./output", help="Output directory")
    parser.add_argument("--batch", type=int, default=1, help="Number of designs to generate")
    parser.add_argument("--reward", type=float, default=0.0, help="Parent reward")
    parser.add_argument("--rank", type=int, default=0, help="Parent rank")
    parser.add_argument("--drag", type=float, default=0.0, help="Parent drag")
    parser.add_argument("--lift", type=float, default=0.0, help="Parent lift")
    parser.add_argument("-t", "--temperature", type=float, default=1.0, help="LLM temperature")
    args = parser.parse_args()

    if args.batch > 1:
        batch_generate(
            args.action_csv, args.images, args.output, args.batch,
            parent_reward=args.reward, parent_rank=args.rank,
            parent_drag=args.drag, parent_lift=args.lift,
            temperature=args.temperature
        )
    else:
        generate_with_images(
            args.action_csv, args.images, args.output,
            parent_reward=args.reward, parent_rank=args.rank,
            parent_drag=args.drag, parent_lift=args.lift,
            temperature=args.temperature
        )
