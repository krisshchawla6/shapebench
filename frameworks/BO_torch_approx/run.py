"""Bayesian Optimisation framework — BoTorch with approximate GP (SVGP).

Replaces SingleTaskGP (exact GP, O(n^3)) with SingleTaskVariationalGP
(sparse variational GP, O(m^2 * n)) where m = num_inducing points.
Trained with Adam + VariationalELBO rather than L-BFGS-B on the exact MLL.

Scaling vs BO_torch (exact GP, CG solver):
  exact   n=5000 end:  ~70s/iter    total: ~35-45 hrs
  approx  any n:       ~5-10s/iter  (dominated by eval cost, not GP fit)

num_inducing=300 is sufficient for 18D — increase toward 500 if reward
quality is noticeably worse than exact GP on the same budget.

CSV format is identical to frameworks/BO_torch/run.py so results are
directly comparable in animate_neuralfoil_comparison_eval_axis.py.
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
    parser.add_argument('--num_inducing', type=int, default=300,
                        help='Inducing points for approximate GP (default: 300). '
                             'Capped at n_observations if fewer data available. '
                             'Start at 300; increase toward 500 if reward quality suffers.')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Adam training epochs per BO iteration (default: 100)')
    parser.add_argument('--warmstart_csv', type=str, default=None,
                        help='Path to CSV with pre-existing observations (columns: x_0,...,x_{d-1},reward). '
                             'Loaded into the GP before the run starts. (default: None)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Adam learning rate (default: 0.01)')


def _load_warmstart(csv_path, dim, lb_t, ub_t):
    import csv as _csv
    obs_X, obs_Y = [], []
    with open(csv_path, newline='') as f:
        for row in _csv.DictReader(f):
            try:
                x_raw = [float(row[f'x_{i}']) for i in range(dim)]
                reward = float(row['reward'])
            except (KeyError, ValueError):
                continue
            lb = lb_t.numpy(); ub = ub_t.numpy()
            x_norm = torch.tensor(
                [(v - lo) / (hi - lo + 1e-12) for v, lo, hi in zip(x_raw, lb, ub)],
                dtype=torch.float64
            ).clamp(0.0, 1.0)
            obs_X.append(x_norm)
            obs_Y.append(reward)
    best = max(obs_Y) if obs_Y else float('-inf')
    print(f'[BO_torch_approx] Warmstart: loaded {len(obs_X)} obs, best={best:.4f}')
    return obs_X, obs_Y, best


def run(environment, args, output_dir):
    from botorch.models import SingleTaskVariationalGP
    from botorch.acquisition import LogExpectedImprovement
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import VariationalELBO

    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    lb, ub = environment.get_param_bounds()
    lb_t = torch.tensor(lb, dtype=torch.float64)
    ub_t = torch.tensor(ub, dtype=torch.float64)
    dim  = len(lb)

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
    prev_state = None   # warm-start: carry model state across iterations
    # Pre-populate from warmstart CSV
    if getattr(args, 'warmstart_csv', None):
        obs_X, obs_Y, best_reward = _load_warmstart(
            args.warmstart_csv, dim, lb_t, ub_t)

    for i in range(args.n_calls):
        if len(obs_X) < args.n_initial:
            # Random phase: uniform sample in [0,1]^dim, then denormalise
            x_norm = torch.rand(dim, dtype=torch.float64)
        else:
            # BO phase: fit approx GP on all observations, optimise EI
            train_X = torch.stack(obs_X)                                         # (n, dim)
            train_Y = torch.tensor(obs_Y, dtype=torch.float64).unsqueeze(-1)    # (n, 1)

            # Standardise outputs for numerical stability
            Y_mean = train_Y.mean()
            Y_std  = train_Y.std().clamp(min=1e-6)
            train_Y_std = (train_Y - Y_mean) / Y_std

            # Cap inducing points at number of observations (early iterations)
            n_inducing = min(args.num_inducing, len(train_X))

            # K-means inducing point initialisation — spreads points across the
            # full observed distribution rather than a random subset, giving
            # coverage near the feasibility boundary rather than clustering in
            # the infeasible region. When n <= n_inducing all points are used.
            if n_inducing >= len(train_X):
                inducing_pts = train_X.clone()
            else:
                from sklearn.cluster import MiniBatchKMeans
                km = MiniBatchKMeans(n_clusters=n_inducing, n_init=3,
                                     random_state=args.random_state)
                km.fit(train_X.numpy())
                inducing_pts = torch.tensor(km.cluster_centers_, dtype=torch.float64)

            model = SingleTaskVariationalGP(
                train_X,
                train_Y=train_Y_std,
                inducing_points=inducing_pts,
            )
            if prev_state is not None:
                # Warm-start: copy learned parameters from previous iteration.
                # Bypasses state_dict/load_state_dict to avoid BoTorch's
                # internal buffer handling (train_inputs, train_targets).
                for name, param in model.named_parameters():
                    if name in prev_state and prev_state[name].shape == param.shape:
                        param.data.copy_(prev_state[name])

            mll = VariationalELBO(model.likelihood, model.model, num_data=len(train_X))

            # Scale epochs with dataset size. Fixed 100 epochs is insufficient
            # once n exceeds the inducing point cap and the ELBO landscape grows
            # more complex. With warm-starting, each iteration only needs
            # incremental updates, so the cost stays manageable.
            # args.n_epochs acts as a floor; scaling kicks in above it.
            n_epochs = min(500, max(args.n_epochs, len(train_X) // 3))

            # Train with Adam — SVGP uses ELBO, not exact MLL
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            for _ in range(n_epochs):
                optimizer.zero_grad()
                output = model(train_X)
                loss   = -mll(output, train_Y_std.squeeze())
                loss.backward()
                optimizer.step()

            model.eval()
            # Save named parameters only (not buffers like train_inputs/train_targets
            # which grow each iteration and would cause shape mismatches on load).
            prev_state = {name: param.detach().clone()
                          for name, param in model.named_parameters()}

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
        print(f'[BO_torch_approx] iter {i + 1}/{args.n_calls}  reward={reward:.4f}'
              f'  best={best_reward:.4f}  phase={phase}')

    print(f'[BO_torch_approx] Done. Best reward: {best_reward:.6f}')
