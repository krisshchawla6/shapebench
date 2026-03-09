"""Island-model LLM-guided evolutionary framework with migration, lineage tracking,
and timeout handling. Extracted from ini_population_startedgy_image_fix_2_island/run_benchmark_action.py."""

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
from .agent import run_llm_action, set_env_format_context

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_args(parser):
    """Add framework-specific CLI arguments."""
    parser.add_argument('--action', type=str, required=True,
                        choices=['generate', 'generate_direct', 'modify', 'modify_direct',
                                 'gaussain', 'gaussian'],
                        help='LLM action to use for all iterations')
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--inspirations', type=int, default=2)
    parser.add_argument('--baseline', type=str, default=None,
                        help='Optional baseline design file')
    parser.add_argument('--initialize_n_sample', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=3.0,
                        help='Power-law exponent for rank-based selection')
    parser.add_argument('--num_islands', type=int, default=1)
    parser.add_argument('--migration_interval', type=int, default=10)
    parser.add_argument('--migration_rate', type=float, default=0.1)
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
        ctx = environment.build_context_entry(parent_entry)
        llm_context.append(ctx)

    if inspirations is not None:
        for insp in inspirations:
            ctx = environment.build_context_entry(insp)
            ctx['images'] = []  # only parent gets images
            llm_context.append(ctx)

    base_csv = parent_entry[0] if parent_entry is not None else None
    name = f"design_{iteration_nb}"
    os.makedirs(output_dir, exist_ok=True)

    debug_dir = os.path.join(output_dir, name, 'context') if debug else None
    x = run_llm_action(action, llm_context, output_dir,
                        base_csv=base_csv, name=name, skip_vis=True,
                        debug_dir=debug_dir)
    return x


def _run_iteration(database, iteration_nb, output_dir, n_inspirations, action,
                   environment, alpha=3.0, num_islands=1, debug=False):
    if num_islands > 1:
        occupied = list(set(int(entry[4]) for entry in database))
        island_idx = np.random.choice(occupied)
        parent = powerlaw_sample_parent_from_island(database, island_idx, alpha=alpha)
        parent_csv = parent[0] if parent is not None else None
        inspirations = (sample_inspirations_from_island(database, island_idx, parent_csv, n_inspirations)
                        if parent is not None else [])
    else:
        parent, inspirations = powerlaw_sample_parent_and_inspiration(database, n_inspirations, alpha=alpha)
        island_idx = 0

    x = _generate_design(parent, inspirations, output_dir, iteration_nb, action,
                         environment, debug=debug)
    parent_island = int(parent[4]) if parent is not None else 0
    parent_csv = parent[0] if parent is not None else None

    if x:
        case_dir = os.path.join(output_dir, f'design_{iteration_nb}')
        os.makedirs(case_dir, exist_ok=True)
        reward, results = environment.simulate(x, case_dir)
        database = update_database(database, x, reward, results, island_idx=parent_island)
    else:
        reward = -10.0

    best_reward = np.max(database[:, 2].astype(float))
    return database, reward, best_reward, parent_csv, x


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(environment, args, output_dir):
    """Execute the island-model evolutionary loop."""
    # Wire up env prompt blocks into the agent
    prompt_blocks = environment.get_prompt_blocks()
    set_env_format_context(prompt_blocks['format_context'])

    os.makedirs(output_dir, exist_ok=True)

    action = args.action
    n_iterations = args.iterations
    n_inspirations = args.inspirations
    initialize_n_sample = args.initialize_n_sample
    alpha = args.alpha
    num_islands = args.num_islands
    migration_interval = args.migration_interval
    migration_rate = args.migration_rate
    debug = args.debug

    # Lineage tracking
    csv_to_iter = {}
    lineage = []

    if args.baseline:
        baseline_path = args.baseline
        if not os.path.isabs(baseline_path):
            baseline_path = os.path.abspath(baseline_path)
        case_dir = os.path.join(output_dir, 'initial')
        os.makedirs(case_dir, exist_ok=True)
        reward, results = environment.simulate(baseline_path, case_dir)
        database = np.array([[baseline_path, 0, reward, results, 0]], dtype=object)
        best_x = baseline_path
        best_reward = reward
        cached = [database[0].copy()]
        csv_to_iter[baseline_path] = 'baseline'
        lineage.append({'id': 'baseline', 'parent_id': None,
                        'reward': float(reward), 'island': 0})
    else:
        database = empty_database()
        best_x = None
        best_reward = -np.inf
        cached = []

    prev_alarm_handler = signal.signal(signal.SIGALRM, _timeout_handler)

    for i in range(n_iterations):
        try:
            signal.alarm(ITERATION_TIMEOUT)

            if i < initialize_n_sample:
                island_idx = i % num_islands
                print(f"\n--- Iteration {i+1}/{n_iterations} "
                      f"[INITIAL SAMPLE {i+1}/{initialize_n_sample}] "
                      f"(action: {action}, island: {island_idx}, no context) ---")
                x = _generate_design(None, None, output_dir, i, action,
                                     environment, debug=debug)
                if x:
                    case_dir = os.path.join(output_dir, f'design_{i}')
                    os.makedirs(case_dir, exist_ok=True)
                    reward, results = environment.simulate(x, case_dir)
                    database = update_database(database, x, reward, results,
                                               island_idx=island_idx)
                    csv_to_iter[x] = i
                    lineage.append({'id': i, 'parent_id': None,
                                    'reward': float(reward), 'island': island_idx})
                else:
                    reward = -10.0
                    lineage.append({'id': i, 'parent_id': None,
                                    'reward': -10.0, 'island': island_idx})
                current_best = (np.max(database[:, 2].astype(float))
                                if len(database) > 0 else -np.inf)
            else:
                current_inspirations = min(n_inspirations, len(database) - 1)
                print(f"\n--- Iteration {i+1}/{n_iterations} "
                      f"(action: {action}, inspirations: {current_inspirations}, "
                      f"islands: {num_islands}) ---")
                database, reward, current_best, parent_csv, x = _run_iteration(
                    database, i, output_dir, current_inspirations, action,
                    environment, alpha=alpha, num_islands=num_islands, debug=debug)

                parent_id = csv_to_iter.get(parent_csv) if parent_csv else None
                parent_island = int([e[4] for e in database if e[0] == parent_csv][0]) \
                    if parent_csv and any(e[0] == parent_csv for e in database) else 0
                if x:
                    csv_to_iter[x] = i
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

        if HAS_LINEAGE_PLOT:
            try:
                save_lineage_json(lineage, os.path.join(output_dir, 'lineage.json'))
                plot_lineage_tree(lineage, os.path.join(output_dir, 'lineage_tree.png'),
                                  title=f"Design Lineage (iter {i+1}/{n_iterations})")
            except Exception as e:
                print(f"Lineage plot failed: {e}")

    signal.signal(signal.SIGALRM, prev_alarm_handler)

    # Save results
    cache_data = []
    for entry in cached:
        design_path, rank, reward, results = entry[0], entry[1], entry[2], entry[3]
        island = int(entry[4]) if len(entry) > 4 else 0
        cache_data.append({
            'design_path': str(design_path),
            'rank': int(rank),
            'reward': float(reward),
            'island': island,
        })

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(cache_data, f, indent=2)

    print(f"\nBest design: {best_x}")
    print(f"Results saved to {output_dir}/results.json")
    return best_x, cached
