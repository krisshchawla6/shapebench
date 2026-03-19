#!/usr/bin/env python3
"""Resume or fork a benchmark run, seeding state from an existing run directory.

Loads run_config.json to reconstruct all arguments, restores the in-memory
database, then dispatches to the selected framework's run().

Two modes depending on whether --output-dir differs from --run-dir:

  IN-PLACE RESUME (--output-dir omitted or same as --run-dir):
    Appends to the existing results.csv, starting from the iteration after the
    last completed one. Use when a run crashed and you want to continue it.

  FORK (--output-dir is a new/different directory):
    Writes all new results to a fresh output directory. The database from
    --run-dir seeds the new run so the LLM inherits the best designs found
    so far. Use to extend a completed run or branch off with new settings.

Database loading priority:
  1. database.json in --run-dir  (written after every iteration, full history)
  2. Legacy fallback: top-N designs by reward from results.csv + iter files
     (for runs predating the database checkpoint feature; default N=10)

Usage:
    python resume_benchmark_neuralfoil.py \\
        --run-dir PATH \\
        [--output-dir PATH] \\
        [--extra-iterations N] \\
        [--seed-top-n N]
"""

import argparse
import csv
import inspect
import json
import os
import shutil
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from run_benchmark import (
    _load_framework,
    _load_environment_class,
    _load_reward_class,
)
from frameworks.core.database import update_database, empty_database, load_database


def _normalize_framework(value):
    """Accept either framework name (v2_batch) or path (.../frameworks/v2_batch)."""
    if not value:
        return value
    return os.path.basename(os.path.normpath(value))


def _seed_database_from_top_n(run_dir, n):
    """Fallback: seed database with top-N designs by reward from results.csv."""
    csv_path = os.path.join(run_dir, 'results.csv')
    if not os.path.exists(csv_path):
        print("  No results.csv found — starting with empty database.")
        return empty_database(), -1

    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return empty_database(), -1

    last_complete_iter = max(int(r['iteration']) for r in rows)

    rows_with_files = []
    for row in rows:
        name      = row['design']
        json_path = os.path.join(run_dir, f'{name}.json')
        if os.path.exists(json_path):
            rows_with_files.append((float(row['reward']), row, json_path))

    rows_with_files.sort(key=lambda t: t[0], reverse=True)
    top = rows_with_files[:n]

    database = empty_database()
    for reward_val, row, json_path in top:
        save_path = os.path.join(run_dir, row['design'], 'save', 'results.json')
        results   = {}
        if os.path.exists(save_path):
            with open(save_path) as f:
                raw = json.load(f)
            results = {'metrics': raw, 'images': [], 'feedback': ''}
        island   = int(row['island']) if 'island' in row else 0
        database = update_database(database, json_path, reward_val, results, island_idx=island)

    print(f"  Seeded database with top-{len(top)} designs from results.csv "
          f"(best reward: {float(database[0, 2]):.4f}, "
          f"last complete iteration: {last_complete_iter}).")
    return database, last_complete_iter


def _truncate_csv_to_iter(output_dir, last_complete_iter):
    """Remove rows beyond last_complete_iter from results.csv (in-place)."""
    csv_path = os.path.join(output_dir, 'results.csv')
    if not os.path.exists(csv_path):
        return
    with open(csv_path, newline='') as f:
        reader     = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows       = [r for r in reader if int(r['iteration']) <= last_complete_iter]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _call_run(fw_module, env_obj, args, output_dir, start_iter, database):
    """Call fw_module.run(), passing resume params only if the framework supports them."""
    sig    = inspect.signature(fw_module.run)
    params = sig.parameters
    kwargs = {}
    if '_start_iter' in params:
        kwargs['_start_iter'] = start_iter
    else:
        print(f"  Warning: framework does not support _start_iter — "
              f"starting from iteration 0 (database seed still applied if supported).")
    if '_initial_database' in params:
        kwargs['_initial_database'] = database
    else:
        print(f"  Warning: framework does not support _initial_database — "
              f"database seed will not be injected.")
    fw_module.run(env_obj, args, output_dir, **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description='Resume or fork a benchmark run from an existing output directory.')
    parser.add_argument('--run-dir', required=True,
                        help='Source run directory to load state from (contains run_config.json)')
    parser.add_argument('--output-dir', default=None,
                        help='Directory for new results. Defaults to --run-dir (in-place resume). '
                             'Provide a different path to fork into a new directory.')
    parser.add_argument('--extra-iterations', type=int, default=0,
                        help='Additional iterations beyond the original plan')
    parser.add_argument('--iterations', type=int, default=None,
                        help='Set total iterations explicitly for this run.')
    parser.add_argument('--seed-top-n', type=int, default=10,
                        help='When database.json is absent, seed with top-N designs (default: 10)')
    parser.add_argument('--framework', default=None,
                        help='Optional framework override (name or path).')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size for this run.')
    parser.add_argument('--mutation-scheduler', type=str, default=None,
                        choices=['fixed', 'geometric', 'adaptive'],
                        help='Override mutation scheduler for this run.')
    parser.add_argument('--gaussian-initial-scale', type=float, default=None,
                        help='Override initial geometric std scale.')
    parser.add_argument('--gaussian-final-scale', type=float, default=None,
                        help='Override final geometric std scale.')
    cli_args = parser.parse_args()

    run_dir    = os.path.abspath(cli_args.run_dir)
    output_dir = os.path.abspath(cli_args.output_dir) if cli_args.output_dir else run_dir
    fork_mode  = (output_dir != run_dir)

    config_path = os.path.join(run_dir, 'run_config.json')
    if not os.path.exists(config_path):
        parser.error(f"run_config.json not found in {run_dir}")

    with open(config_path) as f:
        config = json.load(f)

    framework    = _normalize_framework(cli_args.framework) if cli_args.framework else config['framework']
    environment  = config['environment']
    reward_name  = config['reward']

    fw_module    = _load_framework(framework)
    env_class    = _load_environment_class(environment)
    reward_class = _load_reward_class(environment, reward_name)

    # Build args namespace from config (mirrors run_benchmark.py)
    full_parser = argparse.ArgumentParser(add_help=False)
    if hasattr(fw_module, 'add_args'):
        fw_module.add_args(full_parser)
    if hasattr(env_class, 'add_args'):
        env_class.add_args(full_parser)
    if hasattr(reward_class, 'add_args'):
        reward_class.add_args(full_parser)

    argv = []
    for key, value in config.items():
        if key in ('framework', 'environment', 'reward', 'output_dir'):
            continue
        arg_name = f'--{key.replace("_", "-")}'
        if isinstance(value, bool):
            if value:
                argv.append(arg_name)
        elif value is not None:
            argv.extend([arg_name, str(value)])
    args, _ = full_parser.parse_known_args(argv)

    if cli_args.iterations is not None:
        args.iterations = cli_args.iterations
    elif cli_args.extra_iterations > 0:
        args.iterations = config['iterations'] + cli_args.extra_iterations
    if cli_args.batch_size is not None:
        args.batch_size = cli_args.batch_size
    if cli_args.mutation_scheduler is not None and hasattr(args, 'mutation_scheduler'):
        args.mutation_scheduler = cli_args.mutation_scheduler
    if cli_args.gaussian_initial_scale is not None and hasattr(args, 'gaussian_initial_scale'):
        args.gaussian_initial_scale = cli_args.gaussian_initial_scale
    if cli_args.gaussian_final_scale is not None and hasattr(args, 'gaussian_final_scale'):
        args.gaussian_final_scale = cli_args.gaussian_final_scale

    # --- Restore database and determine start_iter ---
    db_path         = os.path.join(run_dir, 'database.json')
    checkpoint_path = os.path.join(run_dir, 'checkpoint.json')

    if os.path.exists(db_path) and os.path.exists(checkpoint_path):
        database = load_database(db_path)
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        last_complete_iter = int(checkpoint['last_completed_iter'])
        print(f"Loaded database.json: {len(database)} designs, "
              f"last completed iteration: {last_complete_iter}.")
    else:
        print("database.json not found — seeding from top designs in results.csv...")
        database, last_complete_iter = _seed_database_from_top_n(
            run_dir, cli_args.seed_top_n)

    # --- Determine start_iter and output setup ---
    if fork_mode:
        start_iter = 0
        os.makedirs(output_dir, exist_ok=True)
        # Carry over scratchpad and sampler code DB so the LLM has context
        for fname in ('scratchpad.txt', 'sampler_code_db.json'):
            src = os.path.join(run_dir, fname)
            dst = os.path.join(output_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
        # Save updated run_config to the new output dir
        new_config = dict(config)
        new_config['framework'] = framework
        new_config['output_dir'] = output_dir
        new_config['iterations'] = args.iterations
        if hasattr(args, 'batch_size'):
            new_config['batch_size'] = args.batch_size
        if hasattr(args, 'mutation_scheduler'):
            new_config['mutation_scheduler'] = args.mutation_scheduler
        if hasattr(args, 'gaussian_initial_scale'):
            new_config['gaussian_initial_scale'] = args.gaussian_initial_scale
        if hasattr(args, 'gaussian_final_scale'):
            new_config['gaussian_final_scale'] = args.gaussian_final_scale
        with open(os.path.join(output_dir, 'run_config.json'), 'w') as f:
            json.dump(new_config, f, indent=2)
    else:
        start_iter = last_complete_iter + 1
        if start_iter >= args.iterations:
            print(f"\nRun already complete ({last_complete_iter + 1}/{args.iterations} iterations). "
                  f"Use --extra-iterations to extend.")
            return
        _truncate_csv_to_iter(output_dir, last_complete_iter)

    shared_kwargs = {k: v for k, v in vars(args).items()}
    reward_obj    = reward_class(**shared_kwargs)
    env_obj       = env_class(reward=reward_obj, **shared_kwargs)

    print("=" * 60)
    print(f"  Mode:        {'fork → new dir' if fork_mode else 'in-place resume'}")
    print(f"  Source:      {run_dir}")
    print(f"  Output:      {output_dir}")
    print(f"  Framework:   {framework}")
    print(f"  Environment: {environment}")
    print(f"  Reward:      {reward_name}")
    print(f"  Start iter:  {start_iter}/{args.iterations}")
    print(f"  DB size:     {len(database)} designs seeded")
    print("=" * 60)

    _call_run(fw_module, env_obj, args, output_dir, start_iter, database)


if __name__ == '__main__':
    main()
