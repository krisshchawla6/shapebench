"""TTT-Discovery: ShapeEvolve's framework loop + Tinker RL updates.

ShapeEvolve's framework drives the entire loop (sampling, prompts, gaussian,
simulation). Only the LLM backend is swapped from Gemini → Tinker for RL
training. After each epoch (groups_per_batch * group_size rollouts), a
batch RL gradient update is applied to the Tinker-hosted model.
"""

import os
os.environ["PYTHONUNBUFFERED"] = "1"

import sys
import json
import time
import logging
import argparse
import importlib
import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent / "discover"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import tinker
from ttt_discover.rl.train import (
    compute_advantages,
    train_step,
    save_checkpoint_and_get_sampling_client,
)
from ttt_discover.rl.data_processing import (
    assemble_training_data,
    remove_constant_reward_groups,
)
from ttt_discover.rl.types import (
    Trajectory,
    TrajectoryGroup,
    Transition,
)
from ttt_discover.tinker_utils.completers import TokensWithLogprobs
from ttt_discover.tinker_utils.misc_utils import get_last_checkpoint

from Test_Time_Discovery.training import tinker_backend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_framework(name):
    return importlib.import_module(f"frameworks.{name}.run")

def _load_agent(name):
    return importlib.import_module(f"frameworks.{name}.agent")

def _load_environment_class(name):
    mod = importlib.import_module(f"environments.{name}.environment")
    from environments.base import BaseEnvironment
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if isinstance(attr, type) and issubclass(attr, BaseEnvironment) and attr is not BaseEnvironment:
            return attr
    raise RuntimeError(f"No BaseEnvironment subclass in environments.{name}.environment")


def _init_results_csv(path, framework_name):
    with open(path, 'w') as f:
        f.write('iteration,epoch,group,rollout,reward,best_reward,elapsed_s\n')


def _append_results_csv(path, iteration, epoch, group, rollout, reward, best_reward, elapsed):
    with open(path, 'a') as f:
        f.write(f"{iteration},{epoch},{group},{rollout},{reward:.6f},{best_reward:.6f},{elapsed:.1f}\n")


def _write_debug(debug_dir, iteration, epoch, group, rollout, reward, elapsed, had_traj, n_tokens):
    os.makedirs(debug_dir, exist_ok=True)
    entry = {
        "iteration": iteration,
        "epoch": epoch,
        "group": group,
        "rollout": rollout,
        "reward": reward,
        "elapsed_s": round(elapsed, 2),
        "had_trajectory": had_traj,
        "n_tokens": n_tokens,
    }
    with open(os.path.join(debug_dir, "tinker_progress.jsonl"), "a") as f:
        f.write(json.dumps(entry) + "\n")


def _do_single_rollout(
    database, iteration_nb, output_dir, n_inspirations, action,
    environment, fw_module, update_database_fn, alpha=3.0, debug=False,
):
    """Run a single rollout using the framework's native loop."""
    t0 = time.time()
    try:
        parent, inspirations = fw_module.powerlaw_sample_parent_and_inspiration(
            database, n_inspirations, alpha=alpha)

        x = fw_module._generate_design(
            parent, inspirations, os.path.join(output_dir, 'designs'),
            iteration_nb, action, environment, debug=debug)

        if x:
            case_dir = os.path.join(output_dir, 'designs', f'design_{iteration_nb}')
            os.makedirs(case_dir, exist_ok=True)
            reward, results = environment.simulate(x, case_dir)
            database = update_database_fn(database, x, reward, results)
        else:
            reward = -10.0
    except Exception as e:
        logger.error(f"Rollout {iteration_nb} failed: {e}")
        reward = -10.0

    dt = time.time() - t0
    return database, reward, dt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Two-pass arg parsing: first get config, then merge ---
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument('--config', type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    config_dict = {}
    if pre_args.config:
        with open(pre_args.config) as f:
            config_dict = json.load(f)

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--environment', type=str, required=True)
    parser.add_argument('--framework', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--model-name', type=str, default='openai/gpt-oss-120b')
    parser.add_argument('--local-model-path', type=str, default=None)
    parser.add_argument('--lora-rank', type=int, default=32)
    parser.add_argument('--renderer-name', type=str, default='gpt_oss_no_sysprompt')
    parser.add_argument('--learning-rate', type=float, default=4e-5)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--num-epochs', type=int, default=25)
    parser.add_argument('--groups-per-batch', type=int, default=5)
    parser.add_argument('--group-size', type=int, default=6)
    parser.add_argument('--num-substeps', type=int, default=1)
    parser.add_argument('--adv-estimator', type=str, default='entropic_adaptive_beta')
    parser.add_argument('--adv-estimator-beta', type=float, default=2.0)
    parser.add_argument('--kl-penalty-coef', type=float, default=0.1)
    parser.add_argument('--loss-fn', type=str, default='importance_sampling')
    parser.add_argument('--remove-constant-reward-groups', action='store_true', default=True)
    parser.add_argument('--save-every', type=int, default=2)
    parser.add_argument('--phase1-max-tokens', type=int, default=26000)
    parser.add_argument('--load-checkpoint-path', type=str, default=None)
    parser.add_argument('--gemini-model', type=str, default='gemini-2.5-flash')
    parser.add_argument('--no-image-analysis', action='store_true', default=False)
    parser.add_argument('--wandb-project', type=str, default=None)
    parser.add_argument('--wandb-name', type=str, default=None)
    parser.add_argument('--pw-alpha', type=float, default=3.0)
    parser.add_argument('--n-inspirations', type=int, default=2)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--reward-offset', type=float, default=0.0,
                        help='Constant added to raw reward before storing/training')

    # Build effective argv from config + CLI
    argv = []
    for key, value in config_dict.items():
        if key.startswith('_'):
            continue
        arg_name = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                argv.append(arg_name)
        elif value is not None:
            argv.append(arg_name)
            argv.append(str(value))
    argv.extend(sys.argv[1:])

    args, extra_argv = parser.parse_known_args(argv)

    env_name = args.environment
    fw_name = args.framework
    output_dir = args.output_dir or os.path.join(
        str(REPO_ROOT), 'environments', env_name, 'results', 'test_time_discovery')
    output_dir = os.path.join(str(REPO_ROOT), output_dir) if not os.path.isabs(output_dir) else output_dir
    os.makedirs(output_dir, exist_ok=True)

    total_iters = args.num_epochs * args.groups_per_batch * args.group_size
    print("=" * 60)
    print(f"  TTT-Discovery")
    print(f"  Framework:   {fw_name}")
    print(f"  Environment: {env_name}")
    print(f"  Output:      {output_dir}")
    print(f"  Model:       {args.model_name}")
    print(f"  Epochs:      {args.num_epochs}")
    print(f"  Groups/batch:{args.groups_per_batch}")
    print(f"  Group size:  {args.group_size}")
    print(f"  Total iters: {total_iters}")
    print(f"  Reward offset: {args.reward_offset}")
    print("=" * 60)

    # Save config
    run_config = vars(args).copy()
    run_config['output_dir'] = output_dir
    run_config['total_iterations'] = total_iters
    run_config['_breakdown'] = (
        f"{args.num_epochs} epochs x {args.groups_per_batch} groups x "
        f"{args.group_size} rollouts = {total_iters}"
    )
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)
    with open(os.path.join(output_dir, 'run_config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)

    # --- Load framework & environment ---
    fw_module = _load_framework(fw_name)
    fw_agent = _load_agent(fw_name)
    env_class = _load_environment_class(env_name)

    # Parse extra args (env-specific like --mach, --re, --alpha, etc.)
    env_parser = argparse.ArgumentParser(allow_abbrev=False)
    if hasattr(env_class, 'add_args'):
        env_class.add_args(env_parser)
    env_args, _ = env_parser.parse_known_args(extra_argv)
    env_kwargs = vars(env_args)

    # Also pass through any config-dict keys that landed in extra_argv
    i = 0
    while i < len(extra_argv):
        if extra_argv[i].startswith('--'):
            key = extra_argv[i].lstrip('-').replace('-', '_')
            if key not in env_kwargs and i + 1 < len(extra_argv):
                val = extra_argv[i + 1]
                try:
                    val = float(val)
                    if val == int(val):
                        val = int(val)
                except ValueError:
                    pass
                env_kwargs[key] = val
                i += 2
                continue
        i += 1

    environment = env_class(**env_kwargs)

    prompt_blocks = environment.get_prompt_blocks()
    fw_agent.set_env_format_context(prompt_blocks['format_context'])

    # --- Tinker setup ---
    async def _setup_tinker():
        service_client = tinker.ServiceClient(base_url=None)

        resume_info = get_last_checkpoint(output_dir)
        if resume_info:
            training_client = (
                await service_client.create_training_client_from_state_with_optimizer_async(
                    resume_info["state_path"]
                )
            )
            start_epoch = resume_info["batch"]
            print(f"Resumed from checkpoint epoch {start_epoch}")
        else:
            training_client = await service_client.create_lora_training_client_async(
                args.model_name, rank=args.lora_rank
            )
            start_epoch = 0

        if args.local_model_path:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.local_model_path, use_fast=True)
        else:
            tokenizer = training_client.get_tokenizer()

        sampling_client = await training_client.save_weights_and_get_sampling_client_async()

        return service_client, training_client, tokenizer, sampling_client, start_epoch

    loop = asyncio.new_event_loop()
    service_client, training_client, tokenizer, sampling_client, start_epoch = (
        loop.run_until_complete(_setup_tinker())
    )

    os.environ["TTT_RENDERER"] = args.renderer_name
    tinker_backend.configure(
        sampling_client=sampling_client,
        tokenizer=tokenizer,
        phase1_max_tokens=args.phase1_max_tokens,
        temperature=args.temperature,
    )
    fw_agent.set_llm_backend(tinker_backend.generate_design)

    # --- Results tracking ---
    csv_path = os.path.join(output_dir, 'results.csv')
    db_module = importlib.import_module(f"frameworks.{fw_name}.database")
    update_database_fn = db_module.update_database
    database = db_module.empty_database()
    iteration = 0
    best_reward = -np.inf

    # Resume from existing CSV
    if os.path.exists(csv_path) and start_epoch > 0:
        import csv
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                iteration = max(iteration, int(row['iteration']))
                best_reward = max(best_reward, float(row['best_reward']))
        iteration += 1
        print(f"Resuming from iteration {iteration}, best_reward={best_reward:.4f}")
    else:
        _init_results_csv(csv_path, fw_name)

    # --- Main training loop ---
    executor = ThreadPoolExecutor(max_workers=1)

    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*60}")

        epoch_trajectory_groups = []

        for group in range(args.groups_per_batch):
            group_transitions = []

            for rollout in range(args.group_size):
                print(f"\n--- Epoch {epoch} | Group {group} | Rollout {rollout} "
                      f"| Iter {iteration} ---")

                current_inspirations = max(0, min(
                    iteration, args.n_inspirations, len(database) - 1))

                future = executor.submit(
                    _do_single_rollout,
                    database, iteration,
                    output_dir, current_inspirations,
                    'gaussain', environment, fw_module,
                    update_database_fn,
                    alpha=args.pw_alpha, debug=args.debug,
                )
                database, reward, dt = future.result()

                reward += args.reward_offset

                ob, ac = tinker_backend.pop_last_trajectory_data()
                had_traj = ob is not None and ac is not None
                n_tokens = len(ac.tokens) if ac else 0

                best_reward = max(best_reward, reward)
                _append_results_csv(csv_path, iteration, epoch, group, rollout,
                                    reward, best_reward, dt)
                _write_debug(output_dir, iteration, epoch, group, rollout,
                             reward, dt, had_traj, n_tokens)

                print(f"  reward={reward:.4f}  best={best_reward:.4f}  "
                      f"dt={dt:.1f}s  tokens={n_tokens}  traj={'Y' if had_traj else 'N'}")

                if had_traj:
                    transition = Transition(
                        ob=ob,
                        ac=ac,
                        reward=reward,
                        episode_done=True,
                    )
                    group_transitions.append(transition)

                iteration += 1

            if group_transitions:
                trajectories = [
                    Trajectory(transitions=[t], final_ob=tinker.ModelInput.empty())
                    for t in group_transitions
                ]
                tg = TrajectoryGroup(
                    trajectories_G=trajectories,
                    final_rewards_G=[0.0] * len(trajectories),
                    metrics_G=[{}] * len(trajectories),
                )
                epoch_trajectory_groups.append(tg)

        # --- RL gradient update ---
        if not epoch_trajectory_groups:
            print(f"  [Epoch {epoch}] No trajectories, skipping train step")
            continue

        if args.remove_constant_reward_groups:
            epoch_trajectory_groups = remove_constant_reward_groups(epoch_trajectory_groups)

        print(f"\n  [Epoch {epoch}] Computing advantages & training "
              f"({len(epoch_trajectory_groups)} groups) ...")

        advantages_P = compute_advantages(
            epoch_trajectory_groups,
            args.adv_estimator,
            args.adv_estimator_beta,
        )
        data_D, _meta = assemble_training_data(epoch_trajectory_groups, advantages_P)

        async def _train_and_checkpoint():
            training_logprobs = await train_step(
                data_D, training_client,
                args.learning_rate, args.num_substeps, args.loss_fn,
            )
            sc, metrics = await save_checkpoint_and_get_sampling_client(
                training_client, epoch + 1, output_dir, args.save_every, start_epoch,
            )
            return sc, metrics, training_logprobs

        sampling_client, ckpt_metrics, train_logprobs = (
            loop.run_until_complete(_train_and_checkpoint())
        )

        tinker_backend.update_sampling_client(sampling_client)

        print(f"  [Epoch {epoch}] Train step done. "
              f"{len(data_D)} data items, {len(train_logprobs)} logprobs")

        # Log epoch summary
        with open(os.path.join(output_dir, 'train.log'), 'a') as f:
            f.write(f"epoch={epoch} data_items={len(data_D)} "
                    f"groups={len(epoch_trajectory_groups)} "
                    f"best_reward={best_reward:.4f}\n")

    print(f"\n{'='*60}")
    print(f"  Training complete. {iteration} total iterations.")
    print(f"  Best reward: {best_reward:.4f}")
    print(f"  Results: {output_dir}")
    print(f"{'='*60}")

    # Save final checkpoint
    async def _final_save():
        from ttt_discover.tinker_utils.misc_utils import save_checkpoint_async
        await save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=output_dir,
            kind="both",
            loop_state={"batch": args.num_epochs},
        )
    loop.run_until_complete(_final_save())
    loop.close()


if __name__ == '__main__':
    main()
