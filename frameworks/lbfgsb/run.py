"""L-BFGS-B framework entry point."""

import os
import csv

import numpy as np

from frameworks.core.database import update_database, empty_database
from frameworks.lbfgsb.lbfgsb import LBFGSOptimizer


def add_args(parser):
    parser.add_argument('--maxiter', type=int, default=200,
                        help='Max L-BFGS-B iterations per restart (default: 200)')
    parser.add_argument('--eps', type=float, default=1e-4,
                        help='Finite-difference step size (default: 1e-4)')
    parser.add_argument('--n_restarts', type=int, default=1,
                        help='Number of random restarts (default: 1)')


def run(environment, args, output_dir):
    lb, ub = environment.get_param_bounds()

    csv_path = os.path.join(output_dir, 'results.csv')
    base_cols = ['call', 'restart', 'reward', 'best_reward']
    env_cols = environment.get_results_csv_columns()
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(base_cols + env_cols)

    database = empty_database()
    global_best = -np.inf
    call_counter = [0]

    for restart in range(args.n_restarts):
        x0 = lb + np.random.rand(len(lb)) * (ub - lb)
        opt = LBFGSOptimizer(x0, lb, ub, maxiter=args.maxiter, eps=args.eps)

        def reward_fn(x, _restart=restart):
            t = call_counter[0]
            case_dir = os.path.join(output_dir, f'call_{t:05d}_r{_restart}')
            design_path = environment.write_design(x, case_dir, 'design')
            reward, results = environment.simulate(design_path, case_dir)
            database_ref = update_database(database, design_path, reward, results)

            nonlocal global_best
            if reward > global_best:
                global_best = reward

            metrics = results.get('metrics', {}) if results else {}
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow(
                    [t, _restart, f'{reward:.6f}', f'{global_best:.6f}']
                    + environment.get_results_csv_row(metrics)
                )

            call_counter[0] += 1
            return reward

        result = opt.run(reward_fn)
        print(f'[L-BFGS-B] restart {restart + 1}/{args.n_restarts}  '
              f'best={global_best:.4f}  '
              f'converged={result.success}  msg={result.message}')

    print(f'[L-BFGS-B] Done. Global best reward: {global_best:.6f}')
