"""Island-model LLM-guided evolutionary framework for BWB optimization.

Combines the island model (migration, lineage, timeouts) with the
BWB agent and design actions."""

import os
import json
import signal
import numpy as np

from .database import update_database, empty_database
from .sampling import (
    powerlaw_sample_parent_and_inspiration,
    powerlaw_sample_parent_from_island,
    sample_inspirations_from_island,
)
from .migration import perform_migration
from .agent import run_llm_action_bwb, set_env_format_context

try:
    from .lineage import plot_lineage_tree, save_lineage_json
    HAS_LINEAGE_PLOT = True
except ImportError:
    HAS_LINEAGE_PLOT = False

ITERATION_TIMEOUT = 2700  # 45 minutes


class IterationTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise IterationTimeout("Iteration exceeded 45-minute time limit")


def add_args(parser):
    parser.add_argument('--action', type=str, default='gaussain',
                        choices=['gaussain', 'gaussian'],
                        help='LLM action type')
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--inspirations', type=int, default=2,
                        help='Max inspirations (grows from 0)')
    parser.add_argument('--initialize_n_sample', type=int, default=0,
                        help='Initial context-free designs')
    parser.add_argument('--pw_alpha', type=float, default=3.0,
                        help='Power-law exponent for rank-based selection')
    parser.add_argument('--num_islands', type=int, default=1)
    parser.add_argument('--migration_interval', type=int, default=10)
    parser.add_argument('--migration_rate', type=float, default=0.1)
    parser.add_argument('--debug', action='store_true',
                        help='Save LLM context and prompts')


def _generate_design(parent_entry, inspirations, output_dir, iteration_nb, action,
                     environment, debug=False):
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

    json_path = run_llm_action_bwb(
        action, llm_context, output_dir, name=name, debug_dir=debug_dir,
    )
    return json_path


def _run_iteration(database, iteration_nb, output_dir, n_inspirations, action,
                   environment, alpha=3.0, num_islands=1, debug=False):
    if num_islands > 1:
        occupied = list(set(int(entry[4]) for entry in database))
        island_idx = np.random.choice(occupied)
        parent = powerlaw_sample_parent_from_island(database, island_idx, alpha=alpha)
        parent_path = parent[0] if parent is not None else None
        inspirations = (sample_inspirations_from_island(database, island_idx, parent_path, n_inspirations)
                        if parent is not None else [])
    else:
        parent, inspirations = powerlaw_sample_parent_and_inspiration(database, n_inspirations, alpha=alpha)
        island_idx = 0

    x = _generate_design(parent, inspirations, output_dir, iteration_nb, action,
                         environment, debug=debug)
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


def _init_results_csv(path):
    with open(path, 'w') as f:
        f.write('iteration,design,reward,best_reward,Cp_mean,Cfx_mean,L_D,island\n')


def _append_results_csv(path, iteration, design_name, reward, best_reward, aero, island=0):
    with open(path, 'a') as f:
        f.write(f"{iteration},{design_name},{reward:.6f},{best_reward:.6f},"
                f"{aero.get('Cp_mean',0):.6f},{aero.get('Cfx_mean',0):.6f},"
                f"{aero.get('L_D',0):.4f},{island}\n")


def run(environment, args, output_dir):
    """Execute the island-model evolutionary loop for BWB designs."""
    prompt_blocks = environment.get_prompt_blocks()
    set_env_format_context(prompt_blocks['format_context'])

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

    path_to_iter = {}
    lineage = []

    database = empty_database()
    best_x = None
    best_reward = -np.inf
    cached = []

    csv_path = os.path.join(output_dir, 'results.csv')
    _init_results_csv(csv_path)

    prev_alarm_handler = signal.signal(signal.SIGALRM, _timeout_handler)

    for i in range(n_iterations):
        reward = -10.0
        aero = {}
        current_island = 0

        try:
            signal.alarm(ITERATION_TIMEOUT)

            if i < initialize_n_sample:
                current_island = i % num_islands
                print(f"\n--- Iteration {i+1}/{n_iterations} "
                      f"[INIT {i+1}/{initialize_n_sample}] "
                      f"(action: {action}, island: {current_island}, no context) ---")
                x = _generate_design(None, None, output_dir, i, action,
                                     environment, debug=debug)
                if x:
                    case_dir = os.path.join(output_dir, f'design_{i}')
                    os.makedirs(case_dir, exist_ok=True)
                    reward, results = environment.simulate(x, case_dir)
                    database = update_database(database, x, reward, results,
                                               island_idx=current_island)
                    path_to_iter[x] = i
                    aero = results.get('metrics', {}) if isinstance(results, dict) else {}
                    lineage.append({'id': i, 'parent_id': None,
                                    'reward': float(reward), 'island': current_island})
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
                    environment, alpha=alpha, num_islands=num_islands, debug=debug)

                parent_id = path_to_iter.get(parent_path) if parent_path else None
                parent_island = int([e[4] for e in database if e[0] == parent_path][0]) \
                    if parent_path and any(e[0] == parent_path for e in database) else 0
                current_island = parent_island
                if x:
                    path_to_iter[x] = i
                    for entry in database:
                        if entry[0] == x:
                            res = entry[3]
                            aero = res.get('metrics', {}) if isinstance(res, dict) else {}
                            break
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

            _append_results_csv(csv_path, i, f'design_{i}', reward, best_reward, aero, current_island)

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
            lineage.append({'id': i, 'parent_id': None,
                            'reward': -10.0, 'island': 0})
            _append_results_csv(csv_path, i, f'design_{i}', -10.0, best_reward, {}, 0)

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
        json_path, rank, reward_val, res = entry[0], entry[1], entry[2], entry[3]
        island = int(entry[4]) if len(entry) > 4 else 0
        metrics = res.get('metrics', {}) if isinstance(res, dict) else {}
        images = res.get('images', []) if isinstance(res, dict) else []
        feedback = res.get('feedback', '') if isinstance(res, dict) else ''
        cache_data.append({
            'json_path': str(json_path),
            'rank': int(rank),
            'reward': float(reward_val),
            'island': island,
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
