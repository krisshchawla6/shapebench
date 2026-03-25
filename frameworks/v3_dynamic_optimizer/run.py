"""v3_dynamic_optimizer: LLM designer + LLM mini-optimizer per iteration.

Per iteration:
  1. Sample parent + inspirations from database.
  2. One LLM call proposes a center design (s0); s0 is evaluated immediately.
  3. SamplerAgent generates an optimizer function that calls evaluate(params) -> float
     up to `budget` (= n_llm) times sequentially. The optimizer may compute
     finite-difference gradients, line searches, CMA-ES steps, etc.
  4. In --hybrid mode, n_gauss additional Gaussian offspring are also evaluated.
  5. SamplerAgent.update_performance() updates the code-block database.
  6. Reflection runs on s0.

No fallback: if all sampler retries fail, those offspring slots are skipped.
"""

import json
import os
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
from .sampler_agent import SamplerAgent

try:
    from ..core.lineage import plot_lineage_tree, save_lineage_json
    HAS_LINEAGE_PLOT = True
except ImportError:
    HAS_LINEAGE_PLOT = False

ITERATION_TIMEOUT = 2700 * 5   # generous upper bound; oracle budget limits actual work
_TOP_DESIGNS_K    = 8


class IterationTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise IterationTimeout("Iteration exceeded time limit")


def add_args(parser):
    parser.add_argument('--action', type=str, default='gaussain',
                        help='LLM action type (e.g. gaussain, gaussian)')
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=30,
                        help='Total designs per iteration (s0 + optimizer budget)')
    parser.add_argument('--gaussian_decay', action='store_true',
                        help='Geometrically decay Gaussian std in --hybrid mode')
    parser.add_argument('--gaussian_final_scale', type=float, default=0.1,
                        help='Final std scale at last iteration (hybrid only)')
    parser.add_argument('--sampler_model', type=str, default='gemini-3-flash-preview',
                        help='LLM model for the optimizer agent (SamplerAgent)')
    parser.add_argument('--designer_model', type=str, default=None,
                        help='LLM model for the s0 designer agent; defaults to --sampler_model if not set')
    parser.add_argument('--sampler_max_retries', type=int, default=3,
                        help='Max LLM retry attempts per iteration if code fails')
    parser.add_argument('--hybrid', action='store_true',
                        help='Split offspring: half LLM-optimizer budget, half Gaussian; '
                             'tracks beat_gaussian in code DB')
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
                        help='Save LLM prompts, responses, and optimizer code files')


def _build_llm_context(parent_entry, inspirations, environment):
    ctx = []
    if parent_entry is not None:
        ctx.append(environment.build_context_entry(parent_entry))
    for insp in inspirations:
        entry = environment.build_context_entry(insp)
        entry['images'] = []
        ctx.append(entry)
    return ctx


def _geometric_std_scale(iteration_nb, total_iterations, final_scale):
    if total_iterations <= 1:
        return 1.0
    progress = iteration_nb / (total_iterations - 1)
    return float(final_scale ** progress)


def _build_top_design_contexts(database, k):
    """Return top-k {params, reward} dicts from DB sorted by reward DESC."""
    if len(database) == 0:
        return []
    sorted_db = database[np.argsort(database[:, 2].astype(float))[::-1]]
    contexts = []
    for entry in sorted_db[:k]:
        path = entry[0]
        if not path or not os.path.exists(str(path)):
            continue
        try:
            with open(str(path)) as f:
                params = json.load(f)
            contexts.append({'params': params, 'reward': float(entry[2])})
        except Exception:
            continue
    return contexts


def _update_param_trajectory(window, database, max_len=8):
    """Append current best DB entry params to the rolling window."""
    if len(database) == 0:
        return window
    best_idx  = int(np.argmax(database[:, 2].astype(float)))
    best_path = database[best_idx, 0]
    if not best_path or not os.path.exists(str(best_path)):
        return window
    try:
        with open(str(best_path)) as f:
            params = json.load(f)
        window.append({'params': params, 'reward': float(database[best_idx, 2])})
        return window[-max_len:]
    except Exception:
        return window


def _run_batch_iteration(database, iteration_nb, output_dir, n_inspirations, action,
                         environment, batch_size, sampler_agent, hybrid, std_scale,
                         top_design_contexts, reward_trajectory, param_trajectory_window,
                         alpha=3.0, num_islands=1, scratchpad="", debug=False):
    """Designer LLM call (s0) + oracle-budget mini-optimizer + optional Gaussian.

    Returns:
        database, batch_results, current_best, parent_path, parent_island, x0, n_oracle_used
    """
    if num_islands > 1:
        occupied   = list(set(int(e[4]) for e in database))
        island_idx = np.random.choice(occupied)
        parent     = powerlaw_sample_parent_from_island(database, island_idx, alpha=alpha)
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

    # --- s0: designer agent LLM call ---
    name_0      = f'iter_{iteration_nb}_s0'
    context_dir = os.path.join(output_dir, name_0, 'context')
    x0 = environment.run_llm_action(
        action, llm_context, output_dir, name=name_0,
        debug_dir=context_dir, parent_path=parent_path, scratchpad=scratchpad)

    # Evaluate s0
    case_dir_0 = os.path.join(output_dir, name_0)
    os.makedirs(case_dir_0, exist_ok=True)
    if x0:
        reward_0, results_0 = environment.simulate(x0, case_dir_0)
        database = update_database(database, x0, reward_0, results_0, island_idx=parent_island)
    else:
        reward_0, results_0 = -10.0, {}

    # --- Load LLM mean params ---
    llm_params_path = os.path.join(context_dir, 'llm_params.json')
    llm_mean = None
    if os.path.exists(llm_params_path):
        with open(llm_params_path) as f:
            llm_mean = json.load(f)

    # --- Budget split ---
    n_offspring = batch_size - 1
    if hybrid:
        n_llm   = n_offspring // 2
        n_gauss = n_offspring - n_llm
    else:
        n_llm   = n_offspring
        n_gauss = 0

    # --- Oracle for LLM mini-optimizer ---
    debug_dir = os.path.join(output_dir, name_0, 'context') if debug else None
    try:
        named_bounds = environment.get_named_param_bounds()
    except NotImplementedError:
        named_bounds = {}

    oracle_results: list[tuple] = []   # (x, case_dir, reward, results)
    _db = [database]                   # mutable wrapper — updated by oracle

    def _oracle(params_dict: dict) -> float:
        slot     = len(oracle_results)
        name     = f'iter_{iteration_nb}_o{slot}'
        case_dir = os.path.join(output_dir, name)
        os.makedirs(case_dir, exist_ok=True)
        x = environment.sample_gaussian(params_dict, output_dir, name, std_scale=0.0)
        if not x:
            return -10.0
        reward, results = environment.simulate(x, case_dir)
        _db[0] = update_database(_db[0], x, reward, results, island_idx=parent_island)
        oracle_results.append((x, case_dir, reward, results))
        return reward

    if llm_mean and n_llm > 0:
        sampler_agent.run_optimizer(
            center_params        = llm_mean,
            evaluate_fn          = _oracle,
            budget               = n_llm,
            bounds               = named_bounds,
            top_design_contexts  = top_design_contexts,
            reward_trajectory    = reward_trajectory,
            param_trajectory     = param_trajectory_window,
            iteration            = iteration_nb,
            debug_dir            = debug_dir,
        )

    database      = _db[0]
    n_oracle_used = len(oracle_results)

    # --- Gaussian offspring (hybrid only) ---
    gauss_results: list[tuple] = []
    if hybrid and llm_mean and n_gauss > 0:
        for k in range(n_gauss):
            name_g     = f'iter_{iteration_nb}_g{k}'
            x_g        = environment.sample_gaussian(
                llm_mean, output_dir, name_g, std_scale=std_scale)
            case_dir_g = os.path.join(output_dir, name_g)
            os.makedirs(case_dir_g, exist_ok=True)
            if x_g:
                reward_g, results_g = environment.simulate(x_g, case_dir_g)
                database = update_database(database, x_g, reward_g, results_g,
                                           island_idx=parent_island)
            else:
                reward_g, results_g = -10.0, {}
            gauss_results.append((x_g, case_dir_g, reward_g, results_g))

    # --- Update sampler code database ---
    oracle_rewards = [r[2] for r in oracle_results]
    gauss_rewards  = [r[2] for r in gauss_results]
    gaussian_best  = float(max(gauss_rewards)) if gauss_rewards else None
    sampler_agent.update_performance(
        llm_offspring_rewards = oracle_rewards,
        gaussian_best_reward  = gaussian_best,
        iteration             = iteration_nb,
    )

    # batch_results: [s0] + oracle + gauss — used for CSV logging
    batch_results = [(x0, case_dir_0, reward_0, results_0)] + oracle_results + gauss_results

    current_best = float(np.max(database[:, 2].astype(float))) if len(database) > 0 else -10.0
    return database, batch_results, current_best, parent_path, parent_island, x0, n_oracle_used


def _init_results_csv(path, extra_cols):
    with open(path, 'w') as f:
        cols = ['iteration', 'sample', 'design', 'reward', 'best_reward',
                'sample_type'] + extra_cols + ['island']
        f.write(','.join(cols) + '\n')


def _append_results_csv(path, iteration, sample, design_name, reward, best_reward,
                        sample_type, extra_values, island=0):
    with open(path, 'a') as f:
        base = [str(iteration), str(sample), design_name,
                f'{reward:.6f}', f'{best_reward:.6f}', sample_type]
        f.write(','.join(base + extra_values + [str(island)]) + '\n')


def run(environment, args, output_dir, _start_iter=0, _initial_database=None):
    os.makedirs(output_dir, exist_ok=True)

    action               = args.action
    n_iterations         = args.iterations
    batch_size           = args.batch_size
    n_inspirations       = args.inspirations
    initialize_n_sample  = args.initialize_n_sample
    alpha                = args.pw_alpha
    num_islands          = args.num_islands
    migration_interval   = args.migration_interval
    migration_rate       = args.migration_rate
    debug                = args.debug
    baseline             = getattr(args, 'baseline', None)
    hybrid               = args.hybrid
    gaussian_decay       = args.gaussian_decay
    gaussian_final_scale = args.gaussian_final_scale

    extra_cols = environment.get_results_csv_columns()

    env_module  = type(environment).__module__
    env_name    = env_module.split('.')[-2] if '.' in env_module else env_module
    reflection_agent = ReflectionAgent(env_name)

    sampler_agent = SamplerAgent(
        output_dir  = output_dir,
        model       = args.sampler_model,
        max_retries = args.sampler_max_retries,
        hybrid      = hybrid,
    )

    if hasattr(environment, 'set_designer_model'):
        designer_model = args.designer_model or args.sampler_model
        environment.set_designer_model(designer_model)

    scratchpad      = ""
    scratchpad_path = os.path.join(output_dir, 'scratchpad.txt')
    if os.path.exists(scratchpad_path):
        with open(scratchpad_path) as f:
            scratchpad = f.read()
        print(f"Loaded existing scratchpad ({len(scratchpad)} chars)")

    reward_trajectory:       list[float] = []
    param_trajectory_window: list[dict]  = []

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
        database    = np.array([[baseline_path, 0, reward, results, 0]], dtype=object)
        best_x      = baseline_path
        best_reward = reward
        cached      = [database[0].copy()]
        path_to_iter[baseline_path] = 'baseline'
        lineage.append({'id': 'baseline', 'parent_id': None,
                        'reward': float(reward), 'island': 0})
    else:
        database    = empty_database()
        best_x      = None
        best_reward = -np.inf
        cached      = []

    csv_path        = os.path.join(output_dir, 'results.csv')
    db_path         = os.path.join(output_dir, 'database.json')
    checkpoint_path = os.path.join(output_dir, 'checkpoint.json')
    if _start_iter == 0:
        _init_results_csv(csv_path, extra_cols)

    prev_alarm_handler = signal.signal(signal.SIGALRM, _timeout_handler)

    for i in range(_start_iter, n_iterations):
        try:
            signal.alarm(ITERATION_TIMEOUT)

            std_scale = (_geometric_std_scale(i, n_iterations, gaussian_final_scale)
                         if gaussian_decay else 1.0)

            n_offspring = batch_size - 1
            n_llm_   = n_offspring // 2 if hybrid else n_offspring
            n_gauss_ = n_offspring - n_llm_ if hybrid else 0
            print(f"\n--- Iteration {i+1}/{n_iterations}  "
                  f"hybrid={hybrid}  batch={batch_size}  "
                  f"budget_llm={n_llm_}  n_gauss={n_gauss_}  "
                  f"{'std_scale=' + str(round(std_scale, 4)) + '  ' if hybrid else ''}"
                  f"scratchpad={len(scratchpad)} chars ---")

            if i < initialize_n_sample:
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
                                    'llm', extra_values, current_island)

            else:
                current_inspirations = min(n_inspirations, len(database) - 1)
                top_design_contexts  = _build_top_design_contexts(database, _TOP_DESIGNS_K)

                (database, batch_results, current_best,
                 parent_path, parent_island, x0,
                 n_oracle_used) = _run_batch_iteration(
                    database, i, output_dir, current_inspirations, action,
                    environment, batch_size, sampler_agent, hybrid, std_scale,
                    top_design_contexts, reward_trajectory, param_trajectory_window,
                    alpha=alpha, num_islands=num_islands,
                    scratchpad=scratchpad, debug=debug)

                # Update running best before CSV so best_reward is monotonically non-decreasing
                if current_best > best_reward:
                    best_reward = current_best

                # Deduplicate batch_results by design path (keeps last occurrence = best data)
                seen: dict[str, int] = {}
                for j, (x, case_dir, r, res) in enumerate(batch_results):
                    key = str(case_dir)
                    seen[key] = j
                deduped = [batch_results[j] for j in sorted(seen.values())]

                # CSV: s0 is index 0, oracle slots 1..n_oracle_used, Gaussian after
                for j, (x, case_dir, r, res) in enumerate(deduped):
                    if j == 0:
                        stype = 'llm_center'
                    elif j <= n_oracle_used:
                        stype = 'llm_optimizer'
                    else:
                        stype = 'gaussian'
                    design_name  = os.path.basename(case_dir) if case_dir else f'iter_{i}_unknown_{j}'
                    metrics      = res.get('metrics', {}) if isinstance(res, dict) else {}
                    extra_values = environment.get_results_csv_row(metrics)
                    _append_results_csv(csv_path, i, j, design_name,
                                        r, best_reward, stype, extra_values, parent_island)

                if x0:
                    case_dir_0 = os.path.join(output_dir, f'iter_{i}_s0')
                    path_to_iter[x0] = i
                    scratchpad = _run_reflection(
                        reflection_agent, environment, x0, case_dir_0, i,
                        scratchpad, scratchpad_path, debug)

                lineage.append({'id': i, 'parent_id': path_to_iter.get(parent_path),
                                'reward': current_best, 'island': parent_island})

                context_iter = i - initialize_n_sample
                if (num_islands > 1 and migration_interval > 0
                        and context_iter > 0 and context_iter % migration_interval == 0):
                    print(f"  Triggering migration at context iteration {context_iter}")
                    database = perform_migration(database, num_islands, migration_rate)

            signal.alarm(0)

            if len(database) > 0:
                best_idx = np.argmax(database[:, 2].astype(float))
                cached.append(database[best_idx].copy())
                # best_reward already updated before CSV writing; update best_x here
                if current_best >= best_reward:
                    best_x = database[best_idx, 0]

            reward_trajectory = (reward_trajectory + [float(current_best)])[-8:]
            param_trajectory_window = _update_param_trajectory(
                param_trajectory_window, database)

            print(f"Batch best: {current_best:.4f}  Global best: {best_reward:.4f}")

        except IterationTimeout:
            signal.alarm(0)
            print(f"*** Iteration {i+1}/{n_iterations} TIMED OUT ***")
            for j in range(batch_size):
                _append_results_csv(csv_path, i, j, f'iter_{i}_s{j}',
                                    -10.0, best_reward, 'timeout',
                                    ['0'] * len(extra_cols), 0)

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
