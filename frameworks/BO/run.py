"""Bayesian Optimisation framework — Gaussian Process + Expected Improvement.

Uses scikit-optimize (skopt) as the GP surrogate. One design is evaluated per
iteration. The GP is updated after each evaluation and the acquisition function
(EI by default) suggests the next point.

CSV format matches v3_dynamic_optimizer so results are directly comparable in
animate_neuralfoil_comparison_eval_axis.py.
"""

import csv
import os

import numpy as np

from frameworks.core.database import update_database, empty_database


def add_args(parser):
    parser.add_argument('--n_initial', type=int, default=30,
                        help='Random initial samples before GP activates (default: 30)')
    parser.add_argument('--n_calls', type=int, default=500,
                        help='Total evaluations including initial random samples (default: 500)')
    parser.add_argument('--acq_func', type=str, default='EI',
                        choices=['EI', 'PI', 'LCB'],
                        help='Acquisition function (default: EI)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--gradient_infeasible', action='store_true', default=True,
                        help='Return actual penalty value for infeasible samples so the GP '
                             'surrogate has gradient signal toward the feasible region (default: True)')


def run(environment, args, output_dir):
    from skopt import Optimizer

    lb, ub = environment.get_param_bounds()
    dimensions = list(zip(lb.tolist(), ub.tolist()))

    opt = Optimizer(
        dimensions=dimensions,
        base_estimator='GP',
        acq_func=args.acq_func,
        n_initial_points=args.n_initial,
        random_state=args.random_state,
    )

    database = empty_database()

    csv_path  = os.path.join(output_dir, 'results.csv')
    # 'particle' column is included so the comparison plotting script correctly
    # detects this as a 0-LLM-calls method (same convention as PSO/GA).
    base_cols = ['iteration', 'particle', 'sample', 'design', 'reward', 'best_reward']
    env_cols  = environment.get_results_csv_columns()
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(base_cols + env_cols)

    best_reward = -np.inf

    for i in range(args.n_calls):
        x = opt.ask()

        case_dir    = os.path.join(output_dir, f'iter_{i:04d}')
        design_name = f'iter_{i:04d}'
        design_path = environment.write_design(np.array(x), case_dir, design_name)

        reward, results = environment.simulate(design_path, case_dir)
        database = update_database(database, design_path, reward, results)

        # skopt minimises — negate reward
        opt.tell(x, -float(reward))

        if reward > best_reward:
            best_reward = reward

        metrics      = results.get('metrics', {}) if isinstance(results, dict) else {}
        extra_values = environment.get_results_csv_row(metrics)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow(
                [i, 0, design_name, design_name, f'{reward:.6f}', f'{best_reward:.6f}']
                + extra_values
            )

        phase = 'random' if i < args.n_initial else 'BO'
        print(f'[BO] iter {i + 1}/{args.n_calls}  reward={reward:.4f}'
              f'  best={best_reward:.4f}  phase={phase}')

    print(f'[BO] Done. Best reward: {best_reward:.6f}')
