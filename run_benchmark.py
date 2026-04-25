#!/usr/bin/env python3
"""Unified CLI for LLM-guided evolutionary design benchmarks.

Dynamically discovers frameworks, environments, and reward evaluators by
folder name, builds a combined argparser, then runs the selected combination.

Supports --config <path.json> to load all arguments from a JSON file.
The resolved configuration is always saved to <output_dir>/run_config.json.
"""

import os
import sys
import json
import inspect
import argparse
import importlib

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)


def _discover(base_dir):
    """List available module folder names under base_dir.

    For frameworks: requires both __init__.py and run.py.
    For environments: requires __init__.py and environment.py.
    """
    if not os.path.isdir(base_dir):
        return []
    results = []
    for d in os.listdir(base_dir):
        path = os.path.join(base_dir, d)
        if not os.path.isdir(path) or d.startswith('_'):
            continue
        if not os.path.exists(os.path.join(path, '__init__.py')):
            continue
        has_run = os.path.exists(os.path.join(path, 'run.py'))
        has_env = os.path.exists(os.path.join(path, 'environment.py'))
        if has_run or has_env:
            results.append(d)
    return sorted(results)


def _discover_rewards(env_name):
    """List available reward module names for a given environment."""
    rewards_dir = os.path.join(REPO_ROOT, 'environments', env_name, 'rewards')
    if not os.path.isdir(rewards_dir):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(rewards_dir)
        if f.endswith('.py') and not f.startswith('_')
    )


def _load_framework(name):
    """Import frameworks/<name>/run.py and return the module."""
    return importlib.import_module(f"frameworks.{name}.run")


def _load_environment_class(name):
    """Import environments/<name>/environment.py and return the Environment class."""
    mod = importlib.import_module(f"environments.{name}.environment")
    from environments.base import BaseEnvironment
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if (isinstance(attr, type)
                and issubclass(attr, BaseEnvironment)
                and attr is not BaseEnvironment):
            return attr
    raise RuntimeError(f"No BaseEnvironment subclass found in environments.{name}.environment")


def _load_reward_class(env_name, reward_name):
    """Import environments/<env>/rewards/<reward>.py and return the BaseReward subclass."""
    mod = importlib.import_module(f"environments.{env_name}.rewards.{reward_name}")
    from environments.base_reward import BaseReward
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if (isinstance(attr, type)
                and issubclass(attr, BaseReward)
                and attr is not BaseReward):
            return attr
    raise RuntimeError(
        f"No BaseReward subclass found in environments.{env_name}.rewards.{reward_name}")


def _config_to_argv(config: dict) -> list:
    """Convert a config dict to a flat argv list for argparse."""
    argv = []
    for key, value in config.items():
        arg_name = f"--{key.replace('_', '-') if key in ('output_dir',) else key}"
        if isinstance(value, bool):
            if value:
                argv.append(arg_name)
        elif value is None:
            pass  # skip null config values; don't pass --baseline None etc.
        else:
            argv.append(arg_name)
            argv.append(str(value))
    return argv


def main():
    frameworks_dir = os.path.join(REPO_ROOT, 'frameworks')
    environments_dir = os.path.join(REPO_ROOT, 'environments')

    available_fw = _discover(frameworks_dir)
    available_env = _discover(environments_dir)

    # Check for --config before full parse
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, default=None,
                               help='Path to JSON config file')
    config_args, cli_remaining = config_parser.parse_known_args()

    extra_argv = []
    if config_args.config:
        with open(config_args.config) as f:
            config = json.load(f)
        extra_argv = _config_to_argv(config)

    effective_argv = extra_argv + cli_remaining

    # First pre-parse: get framework + environment to discover valid rewards
    # allow_abbrev=False prevents --re from being matched to --reward by prefix
    pre_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre_parser.add_argument('--framework', type=str, required=True,
                            choices=available_fw,
                            help=f'Framework to use: {available_fw}')
    pre_parser.add_argument('--environment', type=str, required=True,
                            choices=available_env,
                            help=f'Environment to use: {available_env}')
    pre_parser.add_argument('--reward', type=str, required=True,
                            help='Reward evaluator to use (discovered from environments/<env>/rewards/)')
    pre_parser.add_argument('--output-dir', type=str, default=None,
                            help='Output directory (default: environments/<env>/results/run_<framework>_<reward>)')
    pre_parser.add_argument('--resume', action='store_true', default=False,
                            help='Resume from checkpoint.json in the output directory')
    pre_args, remaining = pre_parser.parse_known_args(effective_argv)

    available_rewards = _discover_rewards(pre_args.environment)
    if pre_args.reward not in available_rewards:
        pre_parser.error(
            f"--reward {pre_args.reward!r} is not valid for environment "
            f"{pre_args.environment!r}. Available: {available_rewards}")

    fw_module = _load_framework(pre_args.framework)
    env_class = _load_environment_class(pre_args.environment)
    reward_class = _load_reward_class(pre_args.environment, pre_args.reward)

    parser = argparse.ArgumentParser(
        description='LLM-Guided Evolutionary Design Benchmark',
        parents=[pre_parser],
    )
    if hasattr(fw_module, 'add_args'):
        fw_module.add_args(parser)
    if hasattr(env_class, 'add_args'):
        env_class.add_args(parser)
    if hasattr(reward_class, 'add_args'):
        reward_class.add_args(parser)

    args = parser.parse_args(effective_argv)

    output_dir = args.output_dir or os.path.join(
        REPO_ROOT, 'environments', args.environment, 'results',
        f'run_{args.framework}_{args.reward}')

    os.makedirs(output_dir, exist_ok=True)

    run_config = {k: v for k, v in vars(args).items() if k != 'output_dir'}
    run_config['output_dir'] = output_dir
    config_path = os.path.join(output_dir, 'run_config.json')
    with open(config_path, 'w') as f:
        json.dump(run_config, f, indent=2)

    shared_kwargs = {k: v for k, v in vars(args).items()
                    if k not in ('framework', 'environment', 'reward', 'output_dir')}

    reward = reward_class(**shared_kwargs)
    environment = env_class(reward=reward, **shared_kwargs)

    # Resume support: read checkpoint and database if --resume is set
    from frameworks.core.database import load_database
    start_iter = 0
    initial_db = None
    if args.resume:
        ckpt_path = os.path.join(output_dir, 'checkpoint.json')
        db_path   = os.path.join(output_dir, 'database.json')
        if os.path.exists(ckpt_path) and os.path.exists(db_path):
            start_iter = json.load(open(ckpt_path))['last_completed_iter'] + 1
            initial_db = load_database(db_path)
            print(f"  Resuming from iteration {start_iter} "
                  f"({len(initial_db)} designs in database)")
        else:
            print("  --resume set but no checkpoint found; starting fresh")

    print("=" * 60)
    print(f"  Framework:   {args.framework}")
    print(f"  Environment: {args.environment}")
    print(f"  Reward:      {args.reward}")
    print(f"  Output:      {output_dir}")
    print(f"  Config:      {config_path}")
    print("=" * 60)

    run_sig = inspect.signature(fw_module.run)
    if '_start_iter' in run_sig.parameters:
        fw_module.run(environment, args, output_dir,
                      _start_iter=start_iter, _initial_database=initial_db)
    else:
        if args.resume:
            print(f"  Warning: --resume has no effect for framework '{args.framework}' "
                  f"(run() does not accept _start_iter)")
        fw_module.run(environment, args, output_dir)


if __name__ == '__main__':
    main()
