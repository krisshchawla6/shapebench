"""LLM-guided evolutionary framework for 3D delta wing optimization.

Adapted from 3d/run_benchmark_action_3d.py for the modular framework.
No island model, no migration, no timeouts — simple single-population loop."""

import os
import json
import numpy as np

from .database import update_database, empty_database
from .sampling import powerlaw_sample_parent_and_inspiration
from .agent import run_llm_action_3d, set_env_format_context


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_args(parser):
    """Add framework-specific CLI arguments."""
    parser.add_argument('--action', type=str, default='gaussain',
                        choices=['gaussain', 'gaussian'],
                        help='LLM action type')
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--inspirations', type=int, default=2,
                        help='Max inspirations (grows from 0)')
    parser.add_argument('--initialize_n_sample', type=int, default=0,
                        help='Initial context-free designs')
    parser.add_argument('--debug', action='store_true',
                        help='Save LLM context and prompts')


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _generate_design(parent_entry, inspirations, output_dir, iteration_nb, action,
                     environment, debug=False):
    """Build LLM context from DB entries and generate a new design."""
    llm_context = []

    if parent_entry is not None:
        llm_context.append(environment.build_context_entry(parent_entry))

    if inspirations is not None:
        for insp in inspirations:
            ctx = environment.build_context_entry(insp)
            ctx['images'] = []
            llm_context.append(ctx)

    name = f"design_{iteration_nb}"
    os.makedirs(output_dir, exist_ok=True)
    debug_dir = os.path.join(output_dir, name, 'context') if debug else None

    json_path = run_llm_action_3d(
        action, llm_context, output_dir, name=name, debug_dir=debug_dir,
    )
    return json_path


def _run_iteration(database, iteration_nb, output_dir, n_inspirations, action,
                   environment, alpha=3.0, debug=False):
    parent, inspirations = powerlaw_sample_parent_and_inspiration(
        database, n_inspirations, alpha=alpha)

    x = _generate_design(parent, inspirations, output_dir, iteration_nb, action,
                         environment, debug=debug)

    if x:
        case_dir = os.path.join(output_dir, f'design_{iteration_nb}')
        os.makedirs(case_dir, exist_ok=True)
        reward, results = environment.simulate(x, case_dir)
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(environment, args, output_dir):
    """Execute the evolutionary loop for 3D delta wings."""
    prompt_blocks = environment.get_prompt_blocks()
    set_env_format_context(prompt_blocks['format_context'])

    os.makedirs(output_dir, exist_ok=True)

    action = args.action
    n_iterations = args.iterations
    n_inspirations = args.inspirations
    initialize_n_sample = args.initialize_n_sample
    debug = args.debug

    database = empty_database()
    best_x = None
    best_reward = -np.inf
    cached = []

    csv_path = os.path.join(output_dir, 'results.csv')
    _init_results_csv(csv_path)

    for i in range(n_iterations):
        reward = -10.0
        results = {'metrics': {}, 'images': [], 'feedback': ''}

        if i < initialize_n_sample:
            print(f"\n--- Iteration {i+1}/{n_iterations} "
                  f"[INIT {i+1}/{initialize_n_sample}] (action: {action}, no context) ---")
            x = _generate_design(None, None, output_dir, i, action,
                                 environment, debug=debug)
            if x:
                case_dir = os.path.join(output_dir, f'design_{i}')
                os.makedirs(case_dir, exist_ok=True)
                reward, results = environment.simulate(x, case_dir)
                database = update_database(database, x, reward, results)
        else:
            context_iter = i - initialize_n_sample
            current_inspirations = max(0, min(context_iter, n_inspirations, len(database) - 1))
            print(f"\n--- Iteration {i+1}/{n_iterations} "
                  f"(action: {action}, inspirations: {current_inspirations}) ---")
            database, reward, _ = _run_iteration(
                database, i, output_dir, current_inspirations, action,
                environment, debug=debug,
            )

        if len(database) > 0:
            best_idx = np.argmax(database[:, 2].astype(float))
            cached.append(database[best_idx].copy())
            current_best = float(database[best_idx, 2])
            if current_best > best_reward:
                best_reward = current_best
                best_x = database[best_idx, 0]

        aero = {}
        for entry in database:
            if entry[0] and os.path.basename(entry[0]).startswith(f'design_{i}'):
                res = entry[3]
                aero = res.get('metrics', {}) if isinstance(res, dict) else {}
                break

        _append_results_csv(csv_path, i, f'design_{i}', reward, best_reward, aero)
        print(f"Reward: {reward:.4f}, Best: {best_reward:.4f}")

    cache_data = []
    for entry in cached:
        json_path, rank, reward_val, res = entry[0], entry[1], entry[2], entry[3]
        metrics = res.get('metrics', {}) if isinstance(res, dict) else {}
        images = res.get('images', []) if isinstance(res, dict) else []
        feedback = res.get('feedback', '') if isinstance(res, dict) else ''
        cache_data.append({
            'json_path': str(json_path),
            'rank': int(rank),
            'reward': float(reward_val),
            'aero': metrics,
            'geometry_png': images[0] if images else None,
            'analysis': feedback,
        })

    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(cache_data, f, indent=2)

    print(f"\nBest design: {best_x}")
    print(f"Results saved to {results_path}")
    return best_x, cached
