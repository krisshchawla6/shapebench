"""Bayesian Optimisation framework — BoTorch (GPyTorch + PyTorch backend).

Replaces scikit-optimize with BoTorch for better scaling at high eval counts.
Key differences vs frameworks/BO/run.py (skopt):
  - GP linear system solved via conjugate gradients (GPyTorch) rather than
    exact Cholesky — same O(n^3) asymptotic but much better constants.
  - Acquisition function optimised via autograd with multiple random restarts
    rather than random sampling, giving more reliable next-point suggestions
    in 18D.
  - Input normalisation to [0,1] and output standardisation are explicit
    (skopt did these internally).
  - Random phase written explicitly (skopt's n_initial_points handled it).

CSV format is identical to frameworks/BO/run.py so results are directly
comparable in animate_neuralfoil_comparison_eval_axis.py.
"""

import csv
import os

import numpy as np
import torch

from frameworks.core.database import update_database, empty_database


def add_args(parser):
    parser.add_argument('--n_initial', type=int, default=30,
                        help='Random initial samples before GP activates (default: 30)')
    parser.add_argument('--n_calls', type=int, default=500,
                        help='Total evaluations including initial random samples (default: 500)')
    parser.add_argument('--num_restarts', type=int, default=10,
                        help='Random restarts for acquisition function optimisation (default: 10)')
    parser.add_argument('--raw_samples', type=int, default=256,
                        help='Raw samples for initialising acquisition optimisation (default: 256)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--gradient_infeasible', action='store_true', default=True,
                        help='Return actual penalty value for infeasible samples so the GP '
                             'surrogate has gradient signal toward the feasible region (default: True)')


def run(environment, args, output_dir):
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition import LogExpectedImprovement
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood

    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    lb, ub = environment.get_param_bounds()
    lb_t = torch.tensor(lb, dtype=torch.float64)
    ub_t = torch.tensor(ub, dtype=torch.float64)
    dim = len(lb)

    # Acquisition optimisation bounds: [0,1]^dim (normalised space)
    acq_bounds = torch.stack([torch.zeros(dim, dtype=torch.float64),
                               torch.ones(dim, dtype=torch.float64)])

    database = empty_database()

    csv_path  = os.path.join(output_dir, 'results.csv')
    base_cols = ['iteration', 'particle', 'sample', 'design', 'reward', 'best_reward']
    env_cols  = environment.get_results_csv_columns()
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(base_cols + env_cols)

    best_reward = -np.inf

    # Accumulate observations as plain lists; convert to tensors each iteration
    obs_X = []   # normalised, shape (n, dim)
    obs_Y = []   # raw reward, shape (n,)

    for i in range(args.n_calls):
        if i < args.n_initial:
            # Random phase: uniform sample in [0,1]^dim, then denormalise
            x_norm = torch.rand(dim, dtype=torch.float64)
        else:
            # BO phase: fit GP on all observations, optimise EI
            train_X = torch.stack(obs_X)                          # (n, dim)
            train_Y = torch.tensor(obs_Y, dtype=torch.float64).unsqueeze(-1)  # (n, 1)

            # Standardise outputs for numerical stability
            Y_mean = train_Y.mean()
            Y_std  = train_Y.std().clamp(min=1e-6)
            train_Y_std = (train_Y - Y_mean) / Y_std

            model = SingleTaskGP(train_X, train_Y_std)
            mll   = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            model.eval()

            # Best observed value in standardised units
            best_f = train_Y_std.max()

            ei = LogExpectedImprovement(model=model, best_f=best_f)
            x_norm, _ = optimize_acqf(
                acq_function=ei,
                bounds=acq_bounds,
                q=1,
                num_restarts=args.num_restarts,
                raw_samples=args.raw_samples,
            )
            x_norm = x_norm.squeeze(0)   # (1, dim) -> (dim,)

        # Denormalise to original parameter space
        x_raw = (x_norm * (ub_t - lb_t) + lb_t).detach().numpy()

        case_dir    = os.path.join(output_dir, f'iter_{i:04d}')
        design_name = f'iter_{i:04d}'
        design_path = environment.write_design(x_raw, case_dir, design_name)

        reward, results = environment.simulate(design_path, case_dir)
        database = update_database(database, design_path, reward, results)

        obs_X.append(x_norm.detach())
        obs_Y.append(float(reward))

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
        print(f'[BO_torch] iter {i + 1}/{args.n_calls}  reward={reward:.4f}'
              f'  best={best_reward:.4f}  phase={phase}')

    print(f'[BO_torch] Done. Best reward: {best_reward:.6f}')
