"""GA_parallel framework — PSO optimizer with per-iteration parallel particle evaluation.

Identical to GA/run.py in algorithm and output format. Particles within each
iteration are evaluated concurrently via ProcessPoolExecutor, giving a ~n_particles×
wall-clock speedup on environments with expensive simulations (e.g. BlendedNet ~41 s/eval).

For fast environments (NeuralFoil, ms-range evals) use GA instead — subprocess
IPC overhead exceeds the evaluation cost there.

Worker processes each load their own copy of the surrogate model. torch.set_num_threads(1)
is set per worker to prevent thread contention across concurrent subprocesses.
"""

import os
import csv
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from frameworks.core.database import update_database, empty_database
from frameworks.GA.PSO.pso import Swarm


# ── Worker helpers (module-level so they are picklable) ────────────────────────

_worker_env = None


def _worker_init(env):
    """Called once per worker process on startup."""
    import torch
    torch.set_num_threads(1)
    global _worker_env
    _worker_env = env


def _eval_particle(payload):
    """Evaluate one particle; runs in a worker subprocess."""
    i, x, case_dir = payload
    design_path = _worker_env.write_design(x, case_dir, 'design')
    reward, results = _worker_env.simulate(design_path, case_dir)
    return i, reward, results, design_path


# ── CLI args ───────────────────────────────────────────────────────────────────

def add_args(parser):
    parser.add_argument('--n_particles', type=int, default=30,
                        help='Number of PSO particles (default: 30)')
    parser.add_argument('--n_iterations', type=int, default=300,
                        help='Number of PSO iterations (default: 300)')
    parser.add_argument('--n_workers', type=int, default=0,
                        help='Worker processes for parallel evaluation; '
                             '0 = n_particles (default)')
    parser.add_argument('--random-state', type=int, default=None,
                        help='NumPy random seed for reproducibility (default: None)')
    parser.add_argument('--x0-design', default=None, metavar='JSON',
                        help='Path to a design JSON file to use as the warm-start '
                             'centre.  All particles are initialised with Gaussian '
                             'noise (sigma=0.1*span per dim) around this point.')


# ── Main ───────────────────────────────────────────────────────────────────────

def run(environment, args, output_dir):
    if getattr(args, 'random_state', None) is not None:
        np.random.seed(args.random_state)

    lb, ub = environment.get_param_bounds()
    swarm  = Swarm(n_particles=args.n_particles, lb=lb, ub=ub)
    T      = args.n_iterations
    n_workers = swarm.n if args.n_workers == 0 else args.n_workers

    x0_design = getattr(args, 'x0_design', None)
    if x0_design:
        x0 = np.asarray(environment.read_design(x0_design), dtype=float)
        span = ub - lb
        noise = np.random.randn(swarm.n, swarm.dim) * 0.1 * span
        swarm.x = np.clip(x0 + noise, lb, ub)
        swarm.p = swarm.x.copy()
        swarm.g = swarm.x[0].copy()
        print(f'[PSO] Warm-start: all particles initialised near {x0_design}')
        print(f'[PSO] x0 = {np.round(x0, 2).tolist()}')

    database = empty_database()

    csv_path  = os.path.join(output_dir, 'results.csv')
    base_cols = ['iteration', 'particle', 'reward', 'gbest_reward']
    env_cols  = environment.get_results_csv_columns()
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(base_cols + env_cols)

    print(f'[PSO] n_particles={swarm.n}  n_iterations={T}  n_workers={n_workers}')

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(environment,),
    ) as pool:
        for t in range(T):
            rewards     = np.full(swarm.n, -np.inf)
            all_results = [None] * swarm.n

            payloads = [
                (i, swarm.x[i].copy(), os.path.join(output_dir, f'iter_{t:04d}_p{i:03d}'))
                for i in range(swarm.n)
            ]

            for i, reward, results, design_path in pool.map(_eval_particle, payloads):
                rewards[i]     = reward
                all_results[i] = results
                database = update_database(database, design_path, reward, results)

            # Use fitness_total for PSO selection to avoid reward-cap deadlock:
            # when all designs are infeasible, capped rewards are identical (-10)
            # and pbest/gbest updates carry no information. fitness_total preserves
            # the penalty magnitude so PSO can rank infeasible designs and drift
            # toward the feasible boundary. Falls back to reward if unavailable.
            selection_fitness = np.array([
                all_results[i].get('metrics', {}).get('fitness_total', rewards[i])
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
