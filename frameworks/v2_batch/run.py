"""v2_batch: v2 island framework with N Gaussian samples per LLM proposal.

Per iteration:
  1. Sample parent + inspirations from database (same as v2).
  2. One LLM call proposes a mean design (sample 0).
     llm_params.json is always written so reflection and re-sampling work.
  3. N-1 additional Gaussian samples are drawn around that LLM mean.
  4. All N designs are evaluated sequentially and added to the database.
  5. Reflection runs on sample 0 (the only one with LLM context).

CSV has one row per design (iteration × batch_size rows total), matching
the PSO results format for fair budget comparisons.
"""

import os
import json
import signal
import numpy as np

from ..core.database import update_database, empty_database, save_database
from ..core.sampling import (
    powerlaw_sample_parent_and_inspiration,
    powerlaw_sample_parent_from_island,
    sample_inspirations_from_island,
)
from ..core.migration import perform_migration
from ..v2.reflection import ReflectionAgent

try:
    from ..core.lineage import plot_lineage_tree, save_lineage_json
    HAS_LINEAGE_PLOT = True
except ImportError:
    HAS_LINEAGE_PLOT = False

ITERATION_TIMEOUT = 2700 * 5  # 5× v2 timeout to cover a full batch


class IterationTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise IterationTimeout("Iteration exceeded time limit")


def add_args(parser):
    parser.add_argument('--action', type=str, default='gaussain',
                        help='LLM action type (e.g. gaussain, gaussian)')
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=30,
                        help='Gaussian samples per LLM call per iteration (default: 30)')
    parser.add_argument('--mutation_scheduler', type=str, default=None,
                        choices=['fixed', 'geometric', 'adaptive'],
                        help='Mutation scheduler for Gaussian offspring')
    parser.add_argument('--gaussian_decay', action='store_true',
                        help='Geometrically decay Gaussian std across iterations')
    parser.add_argument('--gaussian_final_scale', type=float, default=0.1,
                        help='Final std scale at the last iteration when decay is enabled')
    parser.add_argument('--gaussian_initial_scale', type=float, default=1.0,
                        help='Initial std scale at the first iteration when geometric scheduler is enabled')
    parser.add_argument('--adaptive_target_success', type=float, default=0.2,
                        help='Target offspring success rate for adaptive scheduler')
    parser.add_argument('--adaptive_target_feasible', type=float, default=0.2,
                        help='Target feasible fraction for adaptive scheduler')
    parser.add_argument('--adaptive_eta_success', type=float, default=1.0,
                        help='Adaptation gain for success rate')
    parser.add_argument('--adaptive_eta_feasible', type=float, default=1.0,
                        help='Adaptation gain for feasibility shortfall')
    parser.add_argument('--adaptive_sigma_min', type=float, default=0.02,
                        help='Minimum std scale for adaptive scheduler')
    parser.add_argument('--adaptive_sigma_max', type=float, default=1.5,
                        help='Maximum std scale for adaptive scheduler')
    parser.add_argument('--adaptive_patience', type=int, default=3,
                        help='Stagnation iterations before reheating')
    parser.add_argument('--adaptive_reheat_factor', type=float, default=1.5,
                        help='Multiplier used when reheating adaptive scheduler')
    parser.add_argument('--inspirations', type=int, default=2,
                        help='Max inspirations per iteration')
    parser.add_argument('--initialize_n_sample', type=int, default=0,
                        help='Context-free designs to generate at start')
    parser.add_argument('--pw_alpha', type=float, default=3.0,
                        help='Power-law exponent for rank-based selection')
    parser.add_argument('--num_islands', type=int, default=1)
    parser.add_argument('--migration_interval', type=int, default=10)
    parser.add_argument('--migration_rate', type=float, default=0.1)
    parser.add_argument('--baseline', type=str, default=None)
    parser.add_argument('--debug', action='store_true',
                        help='Save LLM prompts, responses, and reflection files')


def _build_llm_context(parent_entry, inspirations, environment):
    ctx = []
    if parent_entry is not None:
        ctx.append(environment.build_context_entry(parent_entry))
    if inspirations:
        for insp in inspirations:
            entry = environment.build_context_entry(insp)
            entry['images'] = []
            ctx.append(entry)
    return ctx


def _resolve_mutation_scheduler(args):
    """Resolve new scheduler arg plus backward-compatible legacy flags."""
    if args.mutation_scheduler is not None:
        return args.mutation_scheduler
    if args.gaussian_decay:
        return 'geometric'
    return 'fixed'


def _geometric_std_scale(iteration_nb, total_iterations, final_scale, initial_scale=1.0):
    """Return geometric-decay std scale for this iteration."""
    if total_iterations <= 1:
        return float(initial_scale)
    init = max(float(initial_scale), 1e-12)
    final = max(float(final_scale), 1e-12)
    progress = iteration_nb / (total_iterations - 1)
    return float(init * ((final / init) ** progress))


def _init_scheduler_state(args):
    """Initialize scheduler state stored across iterations."""
    scheduler = _resolve_mutation_scheduler(args)
    return {
        'scheduler': scheduler,
        'std_scale': 1.0,
        'stall_count': 0,
        'last_success_rate': None,
        'last_feasible_rate': None,
        'last_reheated': False,
    }


def _current_std_scale(state, iteration_nb, total_iterations, args):
    """Return the std scale to use for the current iteration."""
    if state['scheduler'] == 'fixed':
        return 1.0
    if state['scheduler'] == 'geometric':
        return _geometric_std_scale(
            iteration_nb, total_iterations, args.gaussian_final_scale, args.gaussian_initial_scale)
    return float(state['std_scale'])


def _result_metrics(result):
    return result.get('metrics', {}) if isinstance(result, dict) else {}


def _batch_success_rate(batch_results):
    """Fraction of offspring s1..sN that beat the center sample s0."""
    if len(batch_results) <= 1:
        return 0.0
    center_reward = float(batch_results[0][2])
    offspring_rewards = [float(r) for _, _, r, _ in batch_results[1:]]
    if not offspring_rewards:
        return 0.0
    successes = sum(r > center_reward for r in offspring_rewards)
    return successes / len(offspring_rewards)


def _batch_feasible_rate(batch_results):
    """Fraction of batch members marked feasible; defaults to 1 if metric absent."""
    if not batch_results:
        return 1.0
    feasibility_flags = []
    for _, _, _, result in batch_results:
        metrics = _result_metrics(result)
        if 'feasible' in metrics:
            feasibility_flags.append(1.0 if bool(metrics['feasible']) else 0.0)
    if not feasibility_flags:
        return 1.0
    return float(np.mean(feasibility_flags))


def adaptive_mutation_control(state, batch_results, current_best, previous_best, args):
    """Update adaptive std scale from success, feasibility, and stagnation."""
    success_rate = _batch_success_rate(batch_results)
    feasible_rate = _batch_feasible_rate(batch_results)

    sigma = float(state['std_scale'])
    sigma *= np.exp(
        args.adaptive_eta_success * (success_rate - args.adaptive_target_success)
        - args.adaptive_eta_feasible * max(0.0, args.adaptive_target_feasible - feasible_rate)
    )
    sigma = float(np.clip(sigma, args.adaptive_sigma_min, args.adaptive_sigma_max))

    improved = current_best > previous_best
    state['stall_count'] = 0 if improved else state['stall_count'] + 1
    state['last_reheated'] = False
    if state['stall_count'] >= args.adaptive_patience:
        sigma = float(min(args.adaptive_sigma_max, sigma * args.adaptive_reheat_factor))
        state['stall_count'] = 0
        state['last_reheated'] = True

    state['std_scale'] = sigma
    state['last_success_rate'] = success_rate
    state['last_feasible_rate'] = feasible_rate
    return state


def _run_batch_iteration(database, iteration_nb, output_dir, n_inspirations, action,
                         environment, batch_size, std_scale, alpha=3.0, num_islands=1,
                         scratchpad="", debug=False):
    """One LLM call + batch_size Gaussian samples, all evaluated and added to database."""
    if num_islands > 1:
        occupied = list(set(int(e[4]) for e in database))
        island_idx = np.random.choice(occupied) if occupied else np.random.randint(0, num_islands)
        parent = powerlaw_sample_parent_from_island(database, island_idx, alpha=alpha)
        inspirations = (
            sample_inspirations_from_island(
                database, island_idx,
                parent[0] if parent is not None else None,
                n_inspirations)
            if parent is not None else []
        )
    else:
        parent, inspirations = powerlaw_sample_parent_and_inspiration(
            database, n_inspirations, alpha=alpha)

    parent_island = int(parent[4]) if parent is not None else 0
    parent_path   = parent[0]      if parent is not None else None
    llm_context   = _build_llm_context(parent, inspirations, environment)

    # --- Sample 0: LLM call ---
    # debug_dir always set so llm_params.json is written (needed for re-sampling + reflection)
    name_0       = f'iter_{iteration_nb}_s0'
    context_dir  = os.path.join(output_dir, name_0, 'context')
    x0 = environment.run_llm_action(
        action, llm_context, output_dir, name=name_0,
        debug_dir=context_dir, parent_path=parent_path, scratchpad=scratchpad)

    # Load LLM mean params for re-sampling
    llm_params_path = os.path.join(context_dir, 'llm_params.json')
    llm_mean = None
    if os.path.exists(llm_params_path):
        with open(llm_params_path) as f:
            llm_mean = json.load(f)

    # --- Evaluate all batch designs ---
    batch_results = []
    for j in range(batch_size):
        if j == 0:
            x = x0
        else:
            if llm_mean is None:
                batch_results.append((None, None, -10.0, {}))
                continue
            x = environment.sample_gaussian(
                llm_mean, output_dir, f'iter_{iteration_nb}_s{j}', std_scale=std_scale)

        if not x:
            batch_results.append((None, None, -10.0, {}))
            continue

        case_dir = os.path.join(output_dir, f'iter_{iteration_nb}_s{j}')
        os.makedirs(case_dir, exist_ok=True)
        reward, results = environment.simulate(x, case_dir)
        database = update_database(database, x, reward, results, island_idx=parent_island)
        batch_results.append((x, case_dir, reward, results))

    current_best = float(np.max(database[:, 2].astype(float))) if len(database) > 0 else -10.0
    return database, batch_results, current_best, parent_path, parent_island, x0


def _init_results_csv(path, extra_cols):
    with open(path, 'w') as f:
        cols = ['iteration', 'sample', 'design', 'reward', 'best_reward'] + extra_cols + ['island']
        f.write(','.join(cols) + '\n')


def _append_results_csv(path, iteration, sample, design_name, reward, best_reward,
                        extra_values, island=0):
    with open(path, 'a') as f:
        base = [str(iteration), str(sample), design_name,
                f'{reward:.6f}', f'{best_reward:.6f}']
        f.write(','.join(base + extra_values + [str(island)]) + '\n')


def run(environment, args, output_dir, _start_iter=0, _initial_database=None):
    os.makedirs(output_dir, exist_ok=True)

    action             = args.action
    n_iterations       = args.iterations
    batch_size         = args.batch_size
    scheduler_state    = _init_scheduler_state(args)
    n_inspirations     = args.inspirations
    initialize_n_sample = args.initialize_n_sample
    alpha              = args.pw_alpha
    num_islands        = args.num_islands
    migration_interval = args.migration_interval
    migration_rate     = args.migration_rate
    debug              = args.debug
    baseline           = getattr(args, 'baseline', None)

    extra_cols = environment.get_results_csv_columns()

    env_module = type(environment).__module__
    env_name   = env_module.split('.')[-2] if '.' in env_module else env_module
    reflection_agent = ReflectionAgent(env_name)

    scratchpad      = ""
    scratchpad_path = os.path.join(output_dir, 'scratchpad.txt')
    if os.path.exists(scratchpad_path):
        with open(scratchpad_path) as f:
            scratchpad = f.read()
        print(f"Loaded existing scratchpad ({len(scratchpad)} chars)")

    path_to_iter = {}
    lineage      = []

    if _initial_database is not None:
        database    = _initial_database
        best_reward = (float(np.max(database[:, 2].astype(float)))
                       if len(database) > 0 else -np.inf)
        best_x      = database[0, 0] if len(database) > 0 else None
        cached      = []
        print(f"Resuming from iteration {_start_iter} "
              f"with {len(database)} designs in database "
              f"(best reward: {best_reward:.4f})")
    elif baseline:
        baseline_path = os.path.abspath(baseline)
        case_dir = os.path.join(output_dir, 'initial')
        os.makedirs(case_dir, exist_ok=True)
        reward, results = environment.simulate(baseline_path, case_dir)
        database   = np.array([[baseline_path, 0, reward, results, 0]], dtype=object)
        best_x     = baseline_path
        best_reward = reward
        cached     = [database[0].copy()]
        path_to_iter[baseline_path] = 'baseline'
        lineage.append({'id': 'baseline', 'parent_id': None,
                        'reward': float(reward), 'island': 0})
    else:
        database    = empty_database()
        best_x      = None
        best_reward = -np.inf
        cached      = []

    csv_path       = os.path.join(output_dir, 'results.csv')
    db_path        = os.path.join(output_dir, 'database.json')
    checkpoint_path = os.path.join(output_dir, 'checkpoint.json')
    if _start_iter == 0:
        _init_results_csv(csv_path, extra_cols)

    prev_alarm_handler = signal.signal(signal.SIGALRM, _timeout_handler)

    for i in range(_start_iter, n_iterations):
        try:
            signal.alarm(ITERATION_TIMEOUT)
            std_scale = _current_std_scale(scheduler_state, i, n_iterations, args)
            print(f"\n--- Iteration {i+1}/{n_iterations}  "
                  f"scheduler={scheduler_state['scheduler']}  "
                  f"batch={batch_size}  "
                  f"std_scale={std_scale:.4f}  "
                  f"scratchpad={len(scratchpad)} chars ---")

            if i < initialize_n_sample:
                # Single design init (no LLM context available yet)
                current_island = i % num_islands
                name_init   = f'iter_{i}_s0'
                context_dir = os.path.join(output_dir, name_init, 'context')
                x = environment.run_llm_action(
                    action, [], output_dir, name=name_init,
                    debug_dir=context_dir, scratchpad=scratchpad)
                if x:
                    case_dir = os.path.join(output_dir, name_init)
                    os.makedirs(case_dir, exist_ok=True)
                    reward, results = environment.simulate(x, case_dir)
                    database = update_database(database, x, reward, results,
                                              island_idx=current_island)
                    path_to_iter[x] = i
                    metrics  = results.get('metrics', {}) if isinstance(results, dict) else {}
                    lineage.append({'id': f'{i}_s0', 'parent_id': None,
                                    'reward': float(reward), 'island': current_island})
                    scratchpad = _run_reflection(
                        reflection_agent, environment, x, case_dir, i,
                        scratchpad, scratchpad_path, debug)
                else:
                    reward, metrics, current_island = -10.0, {}, i % num_islands
                    lineage.append({'id': f'{i}_s0', 'parent_id': None,
                                    'reward': -10.0, 'island': current_island})

                current_best = (float(np.max(database[:, 2].astype(float)))
                                if len(database) > 0 else -np.inf)
                extra_values = environment.get_results_csv_row(metrics)
                _append_results_csv(csv_path, i, 0, name_init, reward, current_best,
                                    extra_values, current_island)

            else:
                current_inspirations = min(n_inspirations, len(database) - 1)
                previous_best = best_reward
                (database, batch_results, current_best,
                 parent_path, parent_island, x0) = _run_batch_iteration(
                    database, i, output_dir, current_inspirations, action,
                    environment, batch_size, std_scale, alpha=alpha, num_islands=num_islands,
                    scratchpad=scratchpad, debug=debug)

                # Log one CSV row per design
                for j, (x, case_dir, r, res) in enumerate(batch_results):
                    metrics      = res.get('metrics', {}) if isinstance(res, dict) else {}
                    extra_values = environment.get_results_csv_row(metrics)
                    _append_results_csv(csv_path, i, j, f'iter_{i}_s{j}',
                                        r, current_best, extra_values, parent_island)

                # Reflection on s0 (the only design with llm_params.json)
                if x0:
                    case_dir_0 = os.path.join(output_dir, f'iter_{i}_s0')
                    path_to_iter[x0] = i
                    scratchpad = _run_reflection(
                        reflection_agent, environment, x0, case_dir_0, i,
                        scratchpad, scratchpad_path, debug)

                lineage.append({'id': i, 'parent_id': path_to_iter.get(parent_path),
                                'reward': current_best, 'island': parent_island})

                if scheduler_state['scheduler'] == 'adaptive':
                    scheduler_state = adaptive_mutation_control(
                        scheduler_state, batch_results, current_best, previous_best, args)
                    print(f"  Adaptive update: success={scheduler_state['last_success_rate']:.3f}  "
                          f"feasible={scheduler_state['last_feasible_rate']:.3f}  "
                          f"next_std_scale={scheduler_state['std_scale']:.4f}"
                          f"{'  reheated' if scheduler_state['last_reheated'] else ''}")

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
                    best_x      = database[best_idx, 0]

            print(f"Batch best: {current_best:.4f}  Global best: {best_reward:.4f}")

        except IterationTimeout:
            signal.alarm(0)
            print(f"*** Iteration {i+1}/{n_iterations} TIMED OUT ***")
            for j in range(batch_size):
                _append_results_csv(csv_path, i, j, f'iter_{i}_s{j}',
                                    -10.0, best_reward, ['0'] * len(extra_cols), 0)

        if HAS_LINEAGE_PLOT:
            try:
                save_lineage_json(lineage, os.path.join(output_dir, 'lineage.json'))
                plot_lineage_tree(lineage, os.path.join(output_dir, 'lineage_tree.png'),
                                  title=f"Design Lineage (iter {i+1}/{n_iterations})")
            except Exception as e:
                print(f"Lineage plot failed: {e}")

        try:
            save_database(database, db_path)
            with open(checkpoint_path, 'w') as _f:
                json.dump({'last_completed_iter': i}, _f)
        except Exception as e:
            print(f"  Checkpoint save failed (non-fatal): {e}")

    signal.signal(signal.SIGALRM, prev_alarm_handler)

    cache_data = []
    for entry in cached:
        design_path, rank, reward_val, res = entry[0], entry[1], entry[2], entry[3]
        island      = int(entry[4]) if len(entry) > 4 else 0
        metrics_out = res.get('metrics', {}) if isinstance(res, dict) else {}
        cache_data.append({
            'design_path': str(design_path),
            'rank':        int(rank),
            'reward':      float(reward_val),
            'island':      island,
            'metrics':     metrics_out,
            'images':      res.get('images', []) if isinstance(res, dict) else [],
            'feedback':    res.get('feedback', '') if isinstance(res, dict) else '',
        })

    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(cache_data, f, indent=2)

    print(f"\nBest design: {best_x}")
    print(f"Results saved to {results_path}")
    return best_x, cached


def _run_reflection(reflection_agent, environment, design_path, case_dir,
                    iteration_nb, scratchpad, scratchpad_path, debug):
    """Run reflection cycle and persist scratchpad. Non-fatal."""
    reflection_inputs = environment.get_reflection_inputs(design_path, case_dir)
    if not reflection_inputs:
        return scratchpad
    try:
        debug_dir = os.path.join(case_dir, 'context') if debug else None
        updated   = reflection_agent.run_cycle(
            scratchpad, reflection_inputs, iteration_nb, debug_dir)
        with open(scratchpad_path, 'w') as f:
            f.write(updated)
        return updated
    except Exception as e:
        print(f"  Reflection failed (non-fatal): {e}")
        return scratchpad
