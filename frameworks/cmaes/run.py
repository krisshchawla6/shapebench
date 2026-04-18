"""CMA-ES framework — thin wrapper around the `cma` package.

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is a derivative-free
optimizer that adapts a full covariance matrix over the search space, learning
parameter correlations and curvature. Unlike GA/PSO (axis-aligned) it is
rotation-invariant and scale-invariant, making it the standard reference solver
for continuous black-box optimization (BBOB/COCO benchmark suite).

Internally operates in normalized [0, 1] space (sigma0 applies to this space).
"""

import csv
import os

import numpy as np

from frameworks.core.database import update_database, empty_database


def add_args(parser):
    parser.add_argument('--n_calls', type=int, default=500,
                        help='Total evaluations (default: 500)')
    parser.add_argument('--sigma0', type=float, default=0.3,
                        help='Initial step size in normalized [0,1] space (default: 0.3)')
    parser.add_argument('--random_state', type=int, default=0,
                        help='Random seed (default: 0)')
    parser.add_argument('--popsize', type=int, default=0,
                        help='CMA-ES population size; 0 = use cma default '
                             '(4 + floor(3*ln(d)), default: 0)')


def run(environment, args, output_dir):
    import cma

    lb, ub = environment.get_param_bounds()
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    scale = ub - lb
    dim = len(lb)

    def denorm(x_norm):
        return lb + np.asarray(x_norm) * scale

    csv_path = os.path.join(output_dir, 'results.csv')
    base_cols = ['iteration', 'particle', 'sample', 'design', 'reward', 'best_reward']
    env_cols = environment.get_results_csv_columns()
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(base_cols + env_cols)

    database = empty_database()
    best_reward = -np.inf

    cma_options = {
        'seed': args.random_state,
        'bounds': [0.0, 1.0],          # normalized space
        'maxfevals': args.n_calls,
        'verbose': -9,                  # suppress cma stdout
        'tolx': 1e-8,
        'tolfun': 1e-9,
    }
    if args.popsize > 0:
        cma_options['popsize'] = args.popsize

    x0_norm = np.random.default_rng(args.random_state).uniform(0, 1, dim)
    es = cma.CMAEvolutionStrategy(x0_norm, args.sigma0, cma_options)

    call = 0
    while not es.stop() and call < args.n_calls:
        solutions = es.ask()
        fitnesses = []

        for x_norm in solutions:
            if call >= args.n_calls:
                fitnesses.append(0.0)
                continue

            x_raw = denorm(np.clip(x_norm, 0.0, 1.0))
            design_name = f'iter_{call:05d}'
            case_dir = os.path.join(output_dir, design_name)
            design_path = environment.write_design(x_raw, case_dir, design_name)

            reward, results = environment.simulate(design_path, case_dir)
            database = update_database(database, design_path, reward, results)

            if reward > best_reward:
                best_reward = reward

            metrics = results.get('metrics', {}) if isinstance(results, dict) else {}
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow(
                    [call, 0, design_name, design_name,
                     f'{reward:.6f}', f'{best_reward:.6f}']
                    + environment.get_results_csv_row(metrics)
                )

            fitnesses.append(float(reward))
            call += 1

            phase = 'init' if call <= 2 * dim else 'CMA-ES'
            print(f'[CMA-ES] eval {call}/{args.n_calls}  '
                  f'reward={reward:.4f}  best={best_reward:.4f}  phase={phase}')

        # CMA-ES minimizes; negate rewards
        es.tell(solutions, [-f for f in fitnesses])

    print(f'[CMA-ES] Done. Best reward: {best_reward:.6f}  '
          f'Stop conditions: {es.stop()}')
