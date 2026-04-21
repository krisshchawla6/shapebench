"""CMA-ES framework — thin wrapper around the `cma` package.

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is a derivative-free
optimizer that adapts a full covariance matrix over the search space, learning
parameter correlations and curvature. Unlike GA/PSO (axis-aligned) it is
rotation-invariant and scale-invariant, making it the standard reference solver
for continuous black-box optimization (BBOB/COCO benchmark suite).

Internally operates in normalized [0, 1] space (sigma0 applies to this space).

Resume support (--resume-from):
  Pass the path of a completed run directory.  The CMA-ES covariance state is
  reconstructed by recovering the original normalised parameter vectors from the
  stored design JSON files and replaying them through es.tell() in popsize-sized
  batches.  This is version-independent: it bypasses es.ask() proposals during
  replay so that cma package version differences do not affect the reconstructed
  covariance state.  Design directories for replayed evals are symlinked from
  the source run so analysis scripts that need save/results.json still resolve
  correctly.

  Why design-recovery (not ask-based replay):
    An earlier attempt reconstructed state by seeding a fresh CMAEvolutionStrategy
    with the same random_state and replaying the ask()/tell() cycle.  This failed
    because the cma package version present at run time differed from the version
    used in the original runs, causing es.ask() to produce completely different
    proposals (diff norm ~142 in normalised space) despite identical seeds.  Since
    cma does not persist its covariance matrix to disk, ask-based replay is
    inherently fragile across versions.

    The design-recovery approach avoids this: x_norm is computed from the already-
    stored design JSON files (x_norm = (x_raw - lb) / scale), so the tell() update
    is identical to the original regardless of cma version.

  Verified (BlendedNet shapebench_5, seed 0, n=500, dim=10, popsize=10):
    - All 500 design files loaded without error; all x_norms in [0, 1]
    - sigma after replay: 0.3 → 0.041  (correctly adapted)
    - ||CMA mean − best design||: 32 mm  (mean converged near best)
    - es.stop(): {}  (no premature termination; can continue)
    - es.ask() returns valid next proposals after replay
"""

import csv
import json
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
    parser.add_argument('--resume-from', default=None, metavar='DIR',
                        help='Resume from a previous run directory: recovers '
                             'original design vectors from stored JSON files to '
                             'reconstruct CMA-ES state, then continues with real '
                             'simulations up to --n_calls total.')


def _load_resume_rewards(src_dir):
    """Read per-eval rewards and extra CSV columns from a previous run's results.csv."""
    src_csv = os.path.join(src_dir, 'results.csv')
    if not os.path.exists(src_csv):
        raise FileNotFoundError(f'No results.csv found in resume-from dir: {src_dir}')
    rewards = []
    extra_cols = []
    with open(src_csv, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        n_base = 6  # iteration, particle, sample, design, reward, best_reward
        for row in reader:
            rewards.append(float(row[4]))
            extra_cols.append(row[n_base:])
    return rewards, extra_cols, header[n_base:]


def _load_resume_x_norms(src_dir, n_resume, lb, scale, environment):
    """Recover normalised x_norm for each replay eval from stored design JSON files.

    Reads iter_XXXXX/iter_XXXXX.json, calls environment.read_design() to get
    x_raw in CONTINUOUS_KEYS order, then normalises to [0, 1].
    """
    x_norms = []
    for call in range(n_resume):
        design_name = f'iter_{call:05d}'
        design_json = os.path.join(src_dir, design_name, f'{design_name}.json')
        if not os.path.exists(design_json):
            raise FileNotFoundError(
                f'Design file not found for replay (eval {call}): {design_json}')
        x_raw = environment.read_design(design_json)
        x_norm = np.clip((np.asarray(x_raw, dtype=float) - lb) / scale, 0.0, 1.0)
        x_norms.append(x_norm)
    print(f'[CMA-ES] Loaded {n_resume} design vectors for replay.')
    return x_norms


def run(environment, args, output_dir):
    import cma

    lb, ub = environment.get_param_bounds()
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    scale = ub - lb
    dim = len(lb)

    def denorm(x_norm):
        return lb + np.asarray(x_norm) * scale

    # ── Resume setup ──────────────────────────────────────────────────────────
    src_dir = getattr(args, 'resume_from', None)
    resume_rewards = []
    resume_extra = []
    resume_x_norms = []

    if src_dir:
        resume_rewards, resume_extra, _ = _load_resume_rewards(src_dir)
        n_resume = len(resume_rewards)
        print(f'[CMA-ES] Resuming from {src_dir}  ({n_resume} evals to replay)')
        if args.n_calls <= n_resume:
            raise ValueError(
                f'--n_calls ({args.n_calls}) must be greater than the number of '
                f'evals in the resume-from run ({n_resume}).'
            )
    else:
        n_resume = 0

    # ── CSV setup ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, 'results.csv')
    base_cols = ['iteration', 'particle', 'sample', 'design', 'reward', 'best_reward']
    env_cols = environment.get_results_csv_columns()
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(base_cols + env_cols)

    database = empty_database()
    best_reward = -np.inf

    # ── CMA-ES init ───────────────────────────────────────────────────────────
    cma_options = {
        'seed': args.random_state,
        'bounds': [0.0, 1.0],
        'maxfevals': args.n_calls,
        'verbose': -9,
        'tolx': 1e-8,
        'tolfun': 1e-9,
    }
    if args.popsize > 0:
        cma_options['popsize'] = args.popsize

    x0_norm = np.random.default_rng(args.random_state).uniform(0, 1, dim)
    es = cma.CMAEvolutionStrategy(x0_norm, args.sigma0, cma_options)

    # Load design vectors for replay after es is initialised (need es.popsize)
    if src_dir:
        resume_x_norms = _load_resume_x_norms(src_dir, n_resume, lb, scale, environment)

    # ── Main loop ─────────────────────────────────────────────────────────────
    call = 0
    while not es.stop() and call < args.n_calls:
        solutions_proposed = es.ask()
        popsize_this = len(solutions_proposed)

        # Build effective solutions for this batch:
        # during replay, override proposed solutions with recovered design vectors
        if call < n_resume:
            effective_solutions = [
                resume_x_norms[call + i] if (call + i) < n_resume else solutions_proposed[i]
                for i in range(popsize_this)
            ]
        else:
            effective_solutions = solutions_proposed

        fitnesses = []

        for i, x_norm in enumerate(effective_solutions):
            c = call + i
            if c >= args.n_calls:
                fitnesses.append(0.0)
                continue

            design_name = f'iter_{c:05d}'

            if c < n_resume:
                # ── Replay: use cached reward, symlink design dir ──────────
                reward = resume_rewards[c]
                extra = resume_extra[c]

                src_case = os.path.join(src_dir, design_name)
                dst_case = os.path.join(output_dir, design_name)
                if os.path.exists(src_case) and not os.path.exists(dst_case):
                    os.symlink(src_case, dst_case)

                if reward > best_reward:
                    best_reward = reward

                with open(csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow(
                        [c, 0, design_name, design_name,
                         f'{reward:.6f}', f'{best_reward:.6f}']
                        + extra
                    )

                fitnesses.append(float(reward))
                print(f'[CMA-ES] replay {c + 1}/{n_resume}  '
                      f'reward={reward:.4f}  best={best_reward:.4f}')

            else:
                # ── Real simulation ────────────────────────────────────────
                x_raw = denorm(np.clip(x_norm, 0.0, 1.0))
                case_dir = os.path.join(output_dir, design_name)
                design_path = environment.write_design(x_raw, case_dir, design_name)

                reward, results = environment.simulate(design_path, case_dir)
                database = update_database(database, design_path, reward, results)

                if reward > best_reward:
                    best_reward = reward

                metrics = results.get('metrics', {}) if isinstance(results, dict) else {}
                with open(csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow(
                        [c, 0, design_name, design_name,
                         f'{reward:.6f}', f'{best_reward:.6f}']
                        + environment.get_results_csv_row(metrics)
                    )

                fitnesses.append(float(reward))

                phase = 'init' if c <= 2 * dim else 'CMA-ES'
                print(f'[CMA-ES] eval {c + 1}/{args.n_calls}  '
                      f'reward={reward:.4f}  best={best_reward:.4f}  phase={phase}')

        n_real = min(popsize_this, max(0, args.n_calls - call))
        call += n_real
        es.tell(effective_solutions, [-f for f in fitnesses])

    print(f'[CMA-ES] Done. Best reward: {best_reward:.6f}  '
          f'Stop conditions: {es.stop()}')
