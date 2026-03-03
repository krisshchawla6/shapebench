#!/usr/bin/env python3
"""Unified CLI for LLM-guided evolutionary design benchmarks.

Dynamically discovers frameworks and environments by folder name,
builds a combined argparser, then runs the selected combination.
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

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--framework', type=str, required=True,
                            choices=available_fw,
                            help=f'Framework to use: {available_fw}')
    pre_parser.add_argument('--environment', type=str, required=True,
                            choices=available_env,
                            help=f'Environment to use: {available_env}')
    pre_parser.add_argument('--output-dir', type=str, default=None,
                            help='Output directory (default: environments/<env>/results/run_<framework>)')
    pre_args, remaining = pre_parser.parse_known_args()

    fw_module = _load_framework(pre_args.framework)
    env_class = _load_environment_class(pre_args.environment)

    parser = argparse.ArgumentParser(
        description='LLM-Guided Evolutionary Design Benchmark',
        parents=[pre_parser],
    )
    if hasattr(fw_module, 'add_args'):
        fw_module.add_args(parser)
    if hasattr(env_class, 'add_args'):
        env_class.add_args(parser)

    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        REPO_ROOT, 'environments', args.environment, 'results',
        f'run_{args.framework}')

    env_kwargs = {k: v for k, v in vars(args).items()
                  if k not in ('framework', 'environment', 'output_dir')}
    environment = env_class(**env_kwargs)

    print("=" * 60)
    print(f"  Framework:   {args.framework}")
    print(f"  Environment: {args.environment}")
    print(f"  Output:      {output_dir}")
    print("=" * 60)

    fw_module.run(environment, args, output_dir)


if __name__ == '__main__':
    main()
