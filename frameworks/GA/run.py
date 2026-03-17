"""GA framework entry point — PSO optimizer."""

import os
import csv

import numpy as np

from frameworks.core.database import update_database, empty_database
from frameworks.GA.PSO.pso import Swarm


def add_args(parser):
    parser.add_argument('--n_particles', type=int, default=30,
                        help='Number of PSO particles (default: 30)')
    parser.add_argument('--n_iterations', type=int, default=300,
                        help='Number of PSO iterations (default: 300)')


def run(environment, args, output_dir):
    lb, ub = environment.get_param_bounds()
    swarm  = Swarm(n_particles=args.n_particles, lb=lb, ub=ub)
    T      = args.n_iterations

    database = empty_database()

    csv_path  = os.path.join(output_dir, 'results.csv')
    base_cols = ['iteration', 'particle', 'reward', 'gbest_reward']
    env_cols  = environment.get_results_csv_columns()
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(base_cols + env_cols)

    for t in range(T):
        rewards     = np.full(swarm.n, -np.inf)
        all_results = [None] * swarm.n

        for i in range(swarm.n):
            case_dir    = os.path.join(output_dir, f'iter_{t:04d}_p{i:03d}')
            design_path = environment.write_design(swarm.x[i], case_dir, 'design')
            reward, results = environment.simulate(design_path, case_dir)
            rewards[i]     = reward
            all_results[i] = results
            database = update_database(database, design_path, reward, results)

        swarm.update_bests(rewards)

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
