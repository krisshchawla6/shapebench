"""GA framework entry point — PSO optimizer."""

import json
import multiprocessing as _mp
import os
import csv
import importlib
import sys

import numpy as np

from frameworks.core.database import update_database, empty_database
from frameworks.GA.PSO.pso import Swarm


def add_args(parser):
    parser.add_argument('--n_particles', type=int, default=30,
                        help='Number of PSO particles (default: 30)')
    parser.add_argument('--n_iterations', type=int, default=300,
                        help='Number of PSO iterations (default: 300)')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Parallel workers for particle evaluation (default: 1 = sequential). '
                             'Set to n_particles to evaluate all in parallel. '
                             'Requires --cpus-per-task >= n_workers in SLURM.')


# ---------------------------------------------------------------------------
# Parallel worker (top-level so it is picklable by multiprocessing spawn)
# ---------------------------------------------------------------------------

def _simulate_worker(payload):
    """Reconstruct env from run_config and run one simulation.

    Each spawn worker imports the environment fresh — safe for SUAVE global state.
    """
    cfg_path, design_path, case_dir = payload
    with open(cfg_path) as f:
        cfg = json.load(f)
    workspace = cfg['workspace_root']
    if workspace not in sys.path:
        sys.path.insert(0, workspace)

    reward_mod = importlib.import_module(cfg['reward_module'])
    reward_cls = getattr(reward_mod, cfg['reward_class'])
    reward = reward_cls(**cfg.get('reward_kwargs', {}))

    env_mod = importlib.import_module(cfg['env_module'])
    env_cls = getattr(env_mod, cfg['env_class'])
    env = env_cls(reward=reward, **cfg.get('env_kwargs', {}))

    return env.simulate(design_path, case_dir)


def _write_run_config(environment, output_dir):
    """Write a minimal run_config.json so spawn workers can reconstruct the env."""
    env_cls    = type(environment)
    reward     = environment.reward
    reward_cls = type(reward)

    reward_kwargs = {}
    for attr in vars(reward):
        val = getattr(reward, attr)
        if isinstance(val, (int, float, str, bool)):
            reward_kwargs[attr] = val

    env_kwargs = {}
    if getattr(environment, 'skip_images', False):
        env_kwargs['skip_images'] = True

    workspace = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..'))

    cfg = {
        'workspace_root': workspace,
        'env_module':     env_cls.__module__,
        'env_class':      env_cls.__name__,
        'reward_module':  reward_cls.__module__,
        'reward_class':   reward_cls.__name__,
        'reward_kwargs':  reward_kwargs,
        'env_kwargs':     env_kwargs,
    }
    path = os.path.join(output_dir, 'run_config.json')
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(environment, args, output_dir):
    lb, ub = environment.get_param_bounds()
    swarm  = Swarm(n_particles=args.n_particles, lb=lb, ub=ub)
    T      = args.n_iterations
    n_wk   = min(getattr(args, 'n_workers', 1), args.n_particles)
    parallel = n_wk > 1

    database = empty_database()
    cfg_path  = _write_run_config(environment, output_dir)

    csv_path  = os.path.join(output_dir, 'results.csv')
    base_cols = ['iteration', 'particle', 'reward', 'gbest_reward']
    env_cols  = environment.get_results_csv_columns()
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(base_cols + env_cols)

    print(f'[PSO] n_particles={swarm.n}  n_iter={T}  n_workers={n_wk}  '
          f'parallel={"yes (spawn)" if parallel else "no"}')

    ctx = _mp.get_context('spawn')

    for t in range(T):
        rewards     = np.full(swarm.n, -np.inf)
        all_results = [None] * swarm.n

        # write all designs (sequential, fast)
        payloads = []
        for i in range(swarm.n):
            case_dir    = os.path.join(output_dir, f'iter_{t:04d}_p{i:03d}')
            design_path = environment.write_design(swarm.x[i], case_dir, 'design')
            payloads.append((cfg_path, design_path, case_dir))

        # evaluate particles (parallel or sequential)
        if parallel:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=n_wk, mp_context=ctx) as ex:
                sim_results = list(ex.map(_simulate_worker, payloads))
        else:
            sim_results = [_simulate_worker(p) for p in payloads]

        for i, (reward, results) in enumerate(sim_results):
            rewards[i]     = reward
            all_results[i] = results
            database = update_database(database, payloads[i][1], reward, results)

        selection_fitness = np.array([
            all_results[i].get('metrics', {}).get('fitness_total', rewards[i])
            if all_results[i] else rewards[i]
            for i in range(swarm.n)
        ])
        swarm.update_bests(selection_fitness)

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for i in range(swarm.n):
                metrics = all_results[i].get('metrics', {}) if all_results[i] else {}
                writer.writerow(
                    [t, i, f'{rewards[i]:.6f}', f'{swarm.gbest:.6f}']
                    + environment.get_results_csv_row(metrics)
                )

        finite = rewards[np.isfinite(rewards)]
        mean_r = float(np.mean(finite)) if len(finite) else float('nan')
        print(f'[PSO] iter {t + 1}/{T}  gbest={swarm.gbest:.4f}  mean={mean_r:.4f}')

        swarm.step(t, T)

    print(f'[PSO] Done. Global best reward: {swarm.gbest:.6f}')
