"""v2 island framework: island-model evolution + persistent meta-scratchpad + reflection cycle.

Extends the base islands loop with two additions:
1. A persistent scratchpad (scratchpad.txt) that accumulates per-iteration
   parameter-geometry knowledge via LLM, injected into every design prompt.
2. A post-design reflection cycle (after every design including init samples)
   that compares the LLM's own parameter predictions against the resulting
   geometry and updates the scratchpad.

Reflection is env-specific: add frameworks/v2/prompts/<env_name>/reflection.py
and override environment.get_reflection_inputs() to enable it for a new env.
"""

import os
import json
import signal
import numpy as np

from ..core.database import update_database, empty_database
from ..core.sampling import (
    powerlaw_sample_parent_and_inspiration,
    powerlaw_sample_parent_from_island,
    sample_inspirations_from_island,
)
from ..core.migration import perform_migration

try:
    from ..core.lineage import plot_lineage_tree, save_lineage_json
    HAS_LINEAGE_PLOT = True
except ImportError:
    HAS_LINEAGE_PLOT = False

from .reflection import ReflectionAgent

ITERATION_TIMEOUT = 2700  # 45 minutes


class IterationTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise IterationTimeout("Iteration exceeded 45-minute time limit")


def add_args(parser):
    parser.add_argument('--action', type=str, default='gaussain',
                        help='LLM action type (e.g. gaussain, gaussian)')
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--inspirations', type=int, default=2,
                        help='Max inspirations per iteration')
    parser.add_argument('--initialize_n_sample', type=int, default=0,
                        help='Context-free designs to generate at start')
    parser.add_argument('--pw_alpha', type=float, default=3.0,
                        help='Power-law exponent for rank-based selection')
    parser.add_argument('--num_islands', type=int, default=1,
                        help='Number of islands (1 = single population)')
    parser.add_argument('--migration_interval', type=int, default=10)
    parser.add_argument('--migration_rate', type=float, default=0.1)
    parser.add_argument('--baseline', type=str, default=None,
                        help='Optional baseline design path to seed the database')
    parser.add_argument('--debug', action='store_true',
                        help='Save LLM context, prompts, responses, and reflection files')


def _generate_design(parent_entry, inspirations, output_dir, iteration_nb, action,
                     environment, scratchpad="", debug=False):
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
    parent_path = parent_entry[0] if parent_entry is not None else None

    return environment.run_llm_action(
        action, llm_context, output_dir, name=name,
        debug_dir=debug_dir, parent_path=parent_path,
        scratchpad=scratchpad,
    )


def _run_iteration(database, iteration_nb, output_dir, n_inspirations, action,
                   environment, alpha=3.0, num_islands=1, scratchpad="", debug=False):
    if num_islands > 1:
        occupied = list(set(int(entry[4]) for entry in database))
        island_idx = np.random.choice(occupied) if occupied else np.random.randint(0, num_islands)
        parent = powerlaw_sample_parent_from_island(database, island_idx, alpha=alpha)
        parent_path = parent[0] if parent is not None else None
        inspirations = (
            sample_inspirations_from_island(database, island_idx, parent_path, n_inspirations)
            if parent is not None else []
        )
    else:
        parent, inspirations = powerlaw_sample_parent_and_inspiration(
            database, n_inspirations, alpha=alpha)
        island_idx = 0

    x = _generate_design(parent, inspirations, output_dir, iteration_nb, action,
                         environment, scratchpad=scratchpad, debug=debug)
    parent_island = int(parent[4]) if parent is not None else 0
    parent_path = parent[0] if parent is not None else None

    if x:
        case_dir = os.path.join(output_dir, f'design_{iteration_nb}')
        os.makedirs(case_dir, exist_ok=True)
        reward, results = environment.simulate(x, case_dir)
        database = update_database(database, x, reward, results, island_idx=parent_island)
    else:
        reward = -10.0

    best_reward = float(np.max(database[:, 2].astype(float))) if len(database) > 0 else reward
    return database, reward, best_reward, parent_path, x


def _init_results_csv(path, extra_cols):
    with open(path, 'w') as f:
        cols = ['iteration', 'design', 'reward', 'best_reward'] + extra_cols + ['island']
        f.write(','.join(cols) + '\n')


def _append_results_csv(path, iteration, design_name, reward, best_reward,
                        extra_values, island=0):
    with open(path, 'a') as f:
        base = [str(iteration), design_name, f"{reward:.6f}", f"{best_reward:.6f}"]
        f.write(','.join(base + extra_values + [str(island)]) + '\n')


def run(environment, args, output_dir):
    """Execute the v2 island loop with scratchpad and reflection."""
    os.makedirs(output_dir, exist_ok=True)

    action = args.action
    n_iterations = args.iterations
    n_inspirations = args.inspirations
    initialize_n_sample = args.initialize_n_sample
    alpha = args.pw_alpha
    num_islands = args.num_islands
    migration_interval = args.migration_interval
    migration_rate = args.migration_rate
    debug = args.debug
    baseline = getattr(args, 'baseline', None)

    extra_cols = environment.get_results_csv_columns()

    # Derive env name for loading reflection prompts
    env_module = type(environment).__module__  # e.g. 'environments.fenics_2d.environment'
    env_name = env_module.split('.')[-2] if '.' in env_module else env_module
    reflection_agent = ReflectionAgent(env_name)

    # Scratchpad: load existing (supports resuming a run) or start fresh
    scratchpad = ""
    scratchpad_path = os.path.join(output_dir, 'scratchpad.txt')
    if os.path.exists(scratchpad_path):
        with open(scratchpad_path) as f:
            scratchpad = f.read()
        print(f"Loaded existing scratchpad ({len(scratchpad)} chars)")

    path_to_iter = {}
    lineage = []

    if baseline:
        baseline_path = os.path.abspath(baseline)
        case_dir = os.path.join(output_dir, 'initial')
        os.makedirs(case_dir, exist_ok=True)
        reward, results = environment.simulate(baseline_path, case_dir)
        database = np.array([[baseline_path, 0, reward, results, 0]], dtype=object)
        best_x = baseline_path
        best_reward = reward
        cached = [database[0].copy()]
        path_to_iter[baseline_path] = 'baseline'
        lineage.append({'id': 'baseline', 'parent_id': None,
                        'reward': float(reward), 'island': 0})
    else:
        database = empty_database()
        best_x = None
        best_reward = -np.inf
        cached = []

    csv_path = os.path.join(output_dir, 'results.csv')
    _init_results_csv(csv_path, extra_cols)

    prev_alarm_handler = signal.signal(signal.SIGALRM, _timeout_handler)

    for i in range(n_iterations):
        reward = -10.0
        metrics = {}
        current_island = 0

        try:
            signal.alarm(ITERATION_TIMEOUT)
            print(f"  Scratchpad: {len(scratchpad)} chars")

            if i < initialize_n_sample:
                current_island = i % num_islands
                print(f"\n--- Iteration {i+1}/{n_iterations} "
                      f"[INIT {i+1}/{initialize_n_sample}] "
                      f"(action: {action}, island: {current_island}, no context) ---")
                x = _generate_design(None, None, output_dir, i, action,
                                     environment, scratchpad=scratchpad, debug=debug)
                if x:
                    case_dir = os.path.join(output_dir, f'design_{i}')
                    os.makedirs(case_dir, exist_ok=True)
                    reward, results = environment.simulate(x, case_dir)
                    database = update_database(database, x, reward, results,
                                               island_idx=current_island)
                    path_to_iter[x] = i
                    metrics = results.get('metrics', {}) if isinstance(results, dict) else {}
                    lineage.append({'id': i, 'parent_id': None,
                                    'reward': float(reward), 'island': current_island})
                    scratchpad = _run_reflection(
                        reflection_agent, environment, x, case_dir, i,
                        scratchpad, scratchpad_path, debug)
                else:
                    lineage.append({'id': i, 'parent_id': None,
                                    'reward': -10.0, 'island': current_island})
                current_best = (float(np.max(database[:, 2].astype(float)))
                                if len(database) > 0 else -np.inf)

            else:
                current_inspirations = min(n_inspirations, len(database) - 1)
                print(f"\n--- Iteration {i+1}/{n_iterations} "
                      f"(action: {action}, inspirations: {current_inspirations}, "
                      f"islands: {num_islands}) ---")
                database, reward, current_best, parent_path, x = _run_iteration(
                    database, i, output_dir, current_inspirations, action,
                    environment, alpha=alpha, num_islands=num_islands,
                    scratchpad=scratchpad, debug=debug)

                parent_id = path_to_iter.get(parent_path) if parent_path else None
                parent_island = 0
                if parent_path:
                    for e in database:
                        if e[0] == parent_path:
                            parent_island = int(e[4])
                            break
                current_island = parent_island
                if x:
                    path_to_iter[x] = i
                    for entry in database:
                        if entry[0] == x:
                            res = entry[3]
                            metrics = res.get('metrics', {}) if isinstance(res, dict) else {}
                            break
                    case_dir = os.path.join(output_dir, f'design_{i}')
                    scratchpad = _run_reflection(
                        reflection_agent, environment, x, case_dir, i,
                        scratchpad, scratchpad_path, debug)
                lineage.append({'id': i, 'parent_id': parent_id,
                                'reward': float(reward),
                                'island': parent_island if x else 0})

                context_iter = i - initialize_n_sample
                if (num_islands > 1 and migration_interval > 0
                        and context_iter > 0 and context_iter % migration_interval == 0):
                    print(f"  Triggering migration at context iteration {context_iter}")
                    database = perform_migration(database, num_islands, migration_rate)

            signal.alarm(0)

            if len(database) > 0:
                best_idx = np.argmax(database[:, 2].astype(float))
                cached.append(database[best_idx].copy())
                if current_best > best_reward:
                    best_reward = current_best
                    best_x = database[best_idx, 0]

            extra_values = environment.get_results_csv_row(metrics)
            _append_results_csv(csv_path, i, f'design_{i}', reward, best_reward,
                                 extra_values, current_island)

            if num_islands > 1 and len(database) > 0:
                pops = {isl: sum(1 for e in database if int(e[4]) == isl)
                        for isl in range(num_islands)}
                print(f"Reward: {reward:.4f}, Best: {best_reward:.4f}, Islands: {pops}")
            else:
                print(f"Reward: {reward:.4f}, Best: {best_reward:.4f}")

        except IterationTimeout:
            signal.alarm(0)
            print(f"\n*** Iteration {i+1}/{n_iterations} TIMED OUT "
                  f"after 45 minutes — skipping ***")
            lineage.append({'id': i, 'parent_id': None, 'reward': -10.0, 'island': 0})
            _append_results_csv(csv_path, i, f'design_{i}', -10.0, best_reward,
                                 ['0'] * len(extra_cols), 0)

        if HAS_LINEAGE_PLOT:
            try:
                save_lineage_json(lineage, os.path.join(output_dir, 'lineage.json'))
                plot_lineage_tree(lineage, os.path.join(output_dir, 'lineage_tree.png'),
                                  title=f"Design Lineage (iter {i+1}/{n_iterations})")
            except Exception as e:
                print(f"Lineage plot failed: {e}")

    signal.signal(signal.SIGALRM, prev_alarm_handler)

    cache_data = []
    for entry in cached:
        design_path, rank, reward_val, res = entry[0], entry[1], entry[2], entry[3]
        island = int(entry[4]) if len(entry) > 4 else 0
        metrics_out = res.get('metrics', {}) if isinstance(res, dict) else {}
        images = res.get('images', []) if isinstance(res, dict) else []
        feedback = res.get('feedback', '') if isinstance(res, dict) else ''
        cache_data.append({
            'design_path': str(design_path),
            'rank': int(rank),
            'reward': float(reward_val),
            'island': island,
            'metrics': metrics_out,
            'images': images,
            'feedback': feedback,
        })

    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(cache_data, f, indent=2)

    print(f"\nBest design: {best_x}")
    print(f"Results saved to {results_path}")
    return best_x, cached


def _run_reflection(reflection_agent, environment, design_path, case_dir,
                    iteration_nb, scratchpad, scratchpad_path, debug):
    """Run the reflection cycle and persist the updated scratchpad. Non-fatal."""
    reflection_inputs = environment.get_reflection_inputs(design_path, case_dir)
    if not reflection_inputs:
        return scratchpad
    try:
        debug_dir = os.path.join(case_dir, 'context') if debug else None
        updated = reflection_agent.run_cycle(
            scratchpad, reflection_inputs, iteration_nb, debug_dir)
        with open(scratchpad_path, 'w') as f:
            f.write(updated)
        return updated
    except Exception as e:
        print(f"  Reflection failed (non-fatal): {e}")
        return scratchpad
