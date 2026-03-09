#!/usr/bin/env python3
"""Unified entry point for LLM-guided evolutionary design benchmarks.

Dynamically loads a framework and environment by folder name, then runs the
evolutionary loop.

Usage:
    python run_benchmark.py --framework island_2d_gaussian --environment fenics_2d \
        --action gaussain --iterations 50 --inspirations 10 --num_islands 2

    python run_benchmark.py --framework delta_wing_3d --environment vlm_3d \
        --action gaussain --iterations 40 --inspirations 5 --aoa 10 --mach 0.3
"""

import os
import sys
import argparse
import importlib

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)


def _discover(base_dir):
    """List available module folder names under base_dir."""
    if not os.path.isdir(base_dir):
        return []
    return sorted(
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
        and not d.startswith('_')
        and os.path.exists(os.path.join(base_dir, d, '__init__.py'))
    )


def _load_framework(name):
    """Import frameworks/<name>/run.py and return the module."""
    mod_path = f"frameworks.{name}.run"
    return importlib.import_module(mod_path)


def _load_environment_class(name):
    """Import environments/<name>/environment.py and return the Environment class."""
    mod_path = f"environments.{name}.environment"
    mod = importlib.import_module(mod_path)
    # Find the first class that is a subclass of BaseEnvironment
    from environments.base import BaseEnvironment
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if (isinstance(attr, type)
                and issubclass(attr, BaseEnvironment)
                and attr is not BaseEnvironment):
            return attr
    raise RuntimeError(f"No BaseEnvironment subclass found in {mod_path}")


def main():
    frameworks_dir = os.path.join(REPO_ROOT, 'frameworks')
    environments_dir = os.path.join(REPO_ROOT, 'environments')

    available_fw = _discover(frameworks_dir)
    available_env = _discover(environments_dir)

    # Phase 1: parse --framework and --environment so we can call add_args
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--framework', type=str, required=True,
                            choices=available_fw,
                            help=f'Framework to use: {available_fw}')
    pre_parser.add_argument('--environment', type=str, required=True,
                            choices=available_env,
                            help=f'Environment to use: {available_env}')
    pre_parser.add_argument('--output-dir', type=str, default=None,
                            help='Output directory (default: results/<framework>_<environment>)')
    pre_args, remaining = pre_parser.parse_known_args()

    # Load modules
    fw_module = _load_framework(pre_args.framework)
    env_class = _load_environment_class(pre_args.environment)

    # Phase 2: build the full argparser with framework + environment args
    parser = argparse.ArgumentParser(
        description='LLM-Guided Evolutionary Design Benchmark',
        parents=[pre_parser],
    )
    if hasattr(fw_module, 'add_args'):
        fw_module.add_args(parser)
    if hasattr(env_class, 'add_args'):
        env_class.add_args(parser)

    args = parser.parse_args()

    # Determine output directory: environments/<env>/results/run_<case>
    output_dir = args.output_dir or os.path.join(
        REPO_ROOT, 'environments', args.environment, 'results',
        f'run_{args.framework}')

    # Instantiate environment (pass all args as kwargs, env picks what it needs)
    env_kwargs = {k: v for k, v in vars(args).items()
                  if k not in ('framework', 'environment', 'output_dir')}
    environment = env_class(**env_kwargs)

    # Banner
    print("=" * 60)
    print(f"  Framework:   {args.framework}")
    print(f"  Environment: {args.environment}")
    print(f"  Output:      {output_dir}")
    print("=" * 60)

    # Run
    fw_module.run(environment, args, output_dir)


if __name__ == '__main__':
    main()
