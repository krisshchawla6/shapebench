#!/usr/bin/env python3
"""
run_benchmark_action_3d.py — LLM-driven evolutionary delta wing optimization
using VLM + VortexNet corrected aerodynamic analysis.

Same architecture as the 2D airfoil run_benchmark_action.py but swaps
FEniCS CFD for the VortexNet-corrected VLM pipeline.

Usage:
    python run_benchmark_action_3d.py --action gaussain --iterations 40 \
        --inspirations 5 --initialize_n_sample 6 --aoa 10 --mach 0.3 --debug
"""

import os
import sys
import json
import argparse
import shutil
import importlib.util
import numpy as np
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
ROOT_3D     = SCRIPT_DIR.parent  # /scratch/3D
ENV_DIR     = SCRIPT_DIR / 'modified_env3d'
SCRIPTS_DIR = ROOT_3D / 'scripts'

sys.path.insert(0, str(ENV_DIR / 'LLM_Actions'))
sys.path.insert(0, str(SCRIPTS_DIR))

from LLM_agent_3d import run_llm_action_3d

# Import generate_design_corrected pipeline
_spec = importlib.util.spec_from_file_location(
    "generate_design_corrected",
    SCRIPTS_DIR / "generate_design_corrected.py",
)
_gdc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gdc)

run_full_pipeline    = _gdc.run_full_pipeline
save_results_json    = _gdc.save_results_json
save_geometry_png    = _gdc.save_geometry_png

BASELINE_LD = 5.45
FAIL_REWARD = -5.0


# ── simulation ───────────────────────────────────────────────────────────────

def run_simulation_3d(json_path, case_dir, aoa, mach, re):
    """Run VLM + VortexNet on a design JSON. Returns (reward, results_dict)."""
    os.makedirs(case_dir, exist_ok=True)

    with open(json_path) as f:
        params = json.load(f)

    try:
        vlm_res, corr_res, predicted_dcp, geometry = run_full_pipeline(
            params, aoa, mach, re, case_dir,
        )

        cl  = float(np.squeeze(corr_res.CL))
        cdi = float(np.squeeze(corr_res.CDi))
        cm  = float(np.squeeze(corr_res.CM))
        ld  = cl / cdi if abs(cdi) > 1e-12 else 0.0

        reward = ld - BASELINE_LD

        # Save outputs using the existing pipeline functions
        save_results_json(params, aoa, mach, re, vlm_res, corr_res, case_dir)
        save_geometry_png(params, aoa, mach, vlm_res, corr_res, predicted_dcp, geometry, case_dir)

    except Exception as e:
        print(f"[sim] FAILED: {e}")
        cl, cdi, cm, ld = 0.0, 0.0, 0.0, 0.0
        reward = FAIL_REWARD

    results = post_process_results(case_dir, cl, cdi, cm, ld, reward)
    return reward, results


def post_process_results(case_dir, cl, cdi, cm, ld, reward):
    """Package results for the database.

    Returns: [aero_dict, [geometry_png_path], analysis_text]
    """
    aero = {'CL': cl, 'CDi': cdi, 'CM': cm, 'L_D': ld, 'reward': reward}

    geometry_png = os.path.join(case_dir, 'geometry.png')
    images = [geometry_png] if os.path.exists(geometry_png) else []

    # LLM analysis of the corrected Cp image — present but disabled for now.
    # Uncomment the block below to enable Gemini-based post-sim analysis:
    # try:
    #     from Analysis_LLM import run_simulation_analysis
    #     metrics = {'CL': cl, 'CDi': cdi, 'L_D': ld, 'reward': reward}
    #     analysis_text = run_simulation_analysis(images, metrics)
    # except Exception as e:
    #     print(f"Analysis failed: {e}")
    #     analysis_text = ""
    analysis_text = ""

    return [aero, images, analysis_text]


# ── database ─────────────────────────────────────────────────────────────────

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


def powerlaw_sample_parent_and_inspiration(database, n_inspiration, alpha=3.0):
    n_items = len(database)
    if n_items == 0:
        return None, []

    ranks = np.arange(1, n_items + 1)
    probabilities = ranks ** (-alpha)
    probabilities /= probabilities.sum()

    n_needed = min(1 + n_inspiration, n_items)
    indices = np.random.choice(n_items, size=n_needed, replace=False, p=probabilities)

    parent = database[indices[0]]
    inspirations = [database[i] for i in indices[1:]]
    return parent, inspirations


# ── design generation ────────────────────────────────────────────────────────

def _load_design_params(json_path):
    """Load a design JSON and return a flat params dict for context."""
    with open(json_path) as f:
        d = json.load(f)
    return {
        'le_sweep':      d.get('le_sweep'),
        'root_chord_in': d.get('root_chord_in', 25.734),
        'twist_root':    d.get('twist_root', 0.0),
        'twist_tip':     d.get('twist_tip', 0.0),
        'dihedral':      d.get('dihedral', 0.0),
        'naca_m':        d.get('naca', {}).get('m', 0),
        'naca_p':        d.get('naca', {}).get('p', 0),
        'naca_t':        d.get('naca', {}).get('t', 12),
    }


def generate_design(parent, inspirations, output_dir, iteration_nb, action, debug=False):
    llm_context = []

    if parent is not None:
        params = _load_design_params(parent[0])
        results = parent[3] if len(parent) > 3 else []
        images_list = results[1] if len(results) > 1 else []
        feedback = results[2] if len(results) > 2 else ""

        parent_images = [p for p in images_list if isinstance(p, str) and os.path.exists(p)]

        llm_context.append({
            'params':  params,
            'reward':  parent[2],
            'ranking': parent[1],
            'feedback': feedback,
            'images':  parent_images,
        })

    if inspirations is not None:
        for insp in inspirations:
            params = _load_design_params(insp[0])
            results = insp[3] if len(insp) > 3 else []
            feedback = results[2] if len(results) > 2 else ""
            llm_context.append({
                'params':  params,
                'reward':  insp[2],
                'ranking': insp[1],
                'feedback': feedback,
                'images':  [],
            })

    name = f"design_{iteration_nb}"
    os.makedirs(output_dir, exist_ok=True)
    debug_dir = os.path.join(output_dir, name, 'context') if debug else None

    json_path = run_llm_action_3d(
        action, llm_context, output_dir, name=name, debug_dir=debug_dir,
    )
    return json_path


# ── iteration / benchmark ────────────────────────────────────────────────────

def run_iteration(database, iteration_nb, output_dir, n_inspirations, action,
                  aoa, mach, re, debug=False):
    parent, inspirations = powerlaw_sample_parent_and_inspiration(database, n_inspirations)
    x = generate_design(parent, inspirations, output_dir, iteration_nb, action, debug=debug)

    if x:
        case_dir = os.path.join(output_dir, f'design_{iteration_nb}')
        reward, results = run_simulation_3d(x, case_dir, aoa, mach, re)
        database = update_database(database, x, reward, results)
    else:
        reward = -10.0

    best_reward = np.max(database[:, 2].astype(float))
    return database, reward, best_reward


def _init_results_csv(path):
    with open(path, 'w') as f:
        f.write('iteration,design,reward,best_reward,CL,CDi,CM,L_D\n')


def _append_results_csv(path, iteration, design_name, reward, best_reward, aero):
    with open(path, 'a') as f:
        f.write(f"{iteration},{design_name},{reward:.6f},{best_reward:.6f},"
                f"{aero.get('CL',0):.6f},{aero.get('CDi',0):.6f},"
                f"{aero.get('CM',0):.6f},{aero.get('L_D',0):.4f}\n")


def run_benchmark(n_iterations, n_inspirations, action, output_dir,
                  aoa, mach, re, initialize_n_sample=0, debug=False):
    os.makedirs(output_dir, exist_ok=True)
    database = np.array([], dtype=object).reshape(0, 4)
    best_x = None
    best_reward = -np.inf
    cached = []

    csv_path = os.path.join(output_dir, 'results.csv')
    _init_results_csv(csv_path)

    for i in range(n_iterations):
        if i < initialize_n_sample:
            print(f"\n--- Iteration {i+1}/{n_iterations} "
                  f"[INIT {i+1}/{initialize_n_sample}] (action: {action}, no context) ---")
            x = generate_design(None, None, output_dir, i, action, debug=debug)
            if x:
                case_dir = os.path.join(output_dir, f'design_{i}')
                os.makedirs(case_dir, exist_ok=True)
                reward, results = run_simulation_3d(x, case_dir, aoa, mach, re)
                database = update_database(database, x, reward, results)
            else:
                reward = -10.0
                results = [{}, [], ""]
        else:
            context_iter = i - initialize_n_sample
            current_inspirations = max(0, min(context_iter, n_inspirations, len(database) - 1))
            print(f"\n--- Iteration {i+1}/{n_iterations} "
                  f"(action: {action}, inspirations: {current_inspirations}) ---")
            database, reward, _ = run_iteration(
                database, i, output_dir, current_inspirations, action,
                aoa, mach, re, debug=debug,
            )

        if len(database) > 0:
            best_idx = np.argmax(database[:, 2].astype(float))
            cached.append(database[best_idx].copy())
            current_best = float(database[best_idx, 2])
            if current_best > best_reward:
                best_reward = current_best
                best_x = database[best_idx, 0]

        # Find this iteration's aero data from the database
        aero = {}
        for entry in database:
            if entry[0] and os.path.basename(entry[0]).startswith(f'design_{i}'):
                aero = entry[3][0] if len(entry[3]) > 0 else {}
                break

        _append_results_csv(csv_path, i, f'design_{i}', reward, best_reward, aero)
        print(f"Reward: {reward:.4f}, Best: {best_reward:.4f}")

    return best_x, cached


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D Delta Wing LLM Evolutionary Benchmark')
    parser.add_argument('--action', type=str, default='gaussain',
                        choices=['gaussain', 'gaussian'],
                        help='LLM action type')
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--inspirations', type=int, default=2,
                        help='Max inspirations (grows from 0)')
    parser.add_argument('--initialize_n_sample', type=int, default=0,
                        help='Initial context-free designs for population diversity')
    parser.add_argument('--aoa', type=float, default=10.0, help='Angle of attack (deg)')
    parser.add_argument('--mach', type=float, default=0.3, help='Mach number')
    parser.add_argument('--re', type=float, default=3.0e6, help='Reynolds number')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Custom output directory (default: benchmark_results_3d_<action>)')
    parser.add_argument('--debug', action='store_true',
                        help='Save LLM prompts/context per iteration')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f'benchmark_results_3d_{args.action}',
    )

    print("=" * 60)
    print(f"3D Delta Wing Benchmark")
    print(f"  action:      {args.action}")
    print(f"  iterations:  {args.iterations}")
    print(f"  inspirations:{args.inspirations}")
    print(f"  init_sample: {args.initialize_n_sample}")
    print(f"  aoa:         {args.aoa} deg")
    print(f"  mach:        {args.mach}")
    print(f"  re:          {args.re:.0f}")
    print(f"  output:      {output_dir}")
    print("=" * 60)

    best_design, cached = run_benchmark(
        args.iterations, args.inspirations, args.action, output_dir,
        args.aoa, args.mach, args.re,
        initialize_n_sample=args.initialize_n_sample, debug=args.debug,
    )
    print(f"\nBest design: {best_design}")

    cache_data = []
    for entry in cached:
        json_path, rank, reward, results = entry[0], entry[1], entry[2], entry[3]
        aero = results[0] if len(results) > 0 else {}
        images = results[1] if len(results) > 1 else []
        analysis = results[2] if len(results) > 2 else ""
        cache_data.append({
            'json_path': str(json_path),
            'rank': int(rank),
            'reward': float(reward),
            'aero': aero,
            'geometry_png': images[0] if images else None,
            'analysis': analysis,
        })

    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(cache_data, f, indent=2)
    print(f"Results saved to {results_path}")
