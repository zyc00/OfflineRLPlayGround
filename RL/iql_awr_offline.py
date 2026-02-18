"""IQL + AWR Offline Finetuning — replace MC re-rollout with IQL critic.

Instead of using an optimal policy for MC re-rollout to estimate advantages,
this script trains IQL Q(s,a) and V(s) networks on offline data collected from
multiple policy checkpoints, then uses A(s,a) = Q(s,a) - V(s) for AWR updates.

Phase 1: Collect offline data from multiple checkpoints (random → optimal)
Phase 2: Train IQL Q and V networks on the offline data
Phase 3: Rollout finetune data with initial policy, compute advantages via IQL
Phase 4: AWR update on fixed finetune data (same as mc_finetune_awr_offline.py)

No optimal policy or MC re-rollout needed — fully offline advantage estimation.

Usage:
  python -m RL.iql_awr_offline
  python -m RL.iql_awr_offline --iql_expectile_tau 0.9 --awr_beta 0.5
"""

import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from torch.utils.tensorboard import SummaryWriter

from data.data_collection.ppo import Agent
from methods.iql.iql import QNetwork, train_iql, compute_nstep_targets


@dataclass
class Args:
    # IQL data collection
    iql_data_checkpoints: tuple[str, ...] = (
        "runs/pickcube_ppo/ckpt_1.pt",    # ~random (~0% SR)
        "runs/pickcube_ppo/ckpt_51.pt",   # bad (~20-30% SR)
        "runs/pickcube_ppo/ckpt_101.pt",  # medium (62.7% SR)
        "runs/pickcube_ppo/ckpt_201.pt",  # good (~90% SR)
        "runs/pickcube_ppo/ckpt_301.pt",  # optimal (99% SR)
    )
    """checkpoints for collecting IQL training data (mixed quality, random to optimal)"""
    iql_episodes_per_ckpt: int = 200
    """number of episodes to collect per checkpoint"""

    # IQL training hyperparameters
    iql_expectile_tau: float = 0.7
    """IQL expectile tau (higher = closer to max Q)"""
    iql_epochs: int = 200
    """number of IQL training epochs"""
    iql_lr: float = 3e-4
    """IQL learning rate"""
    iql_batch_size: int = 256
    """IQL minibatch size"""
    iql_nstep: int = 1
    """n-step TD return for IQL (1=standard, >1=multi-step)"""
    iql_patience: int = 50
    """early stopping patience for IQL"""

    # Finetune checkpoint (policy to improve)
    checkpoint: str = "runs/pickcube_ppo/ckpt_101.pt"
    """pretrained checkpoint to finetune from"""

    # Environment
    env_id: str = "PickCube-v1"
    num_envs: int = 512
    num_eval_envs: int = 128
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"
    max_episode_steps: int = 50

    # AWR hyperparameters (actor only)
    gamma: float = 0.8
    learning_rate: float = 3e-4
    num_steps: int = 50
    """rollout length for finetune data collection"""
    num_minibatches: int = 32
    update_epochs: int = 4
    awr_beta: float = 1.0
    """AWR temperature. Lower = more greedy, higher = closer to uniform"""
    awr_max_weight: float = 20.0
    """AWR weight clamp upper bound to prevent exp explosion"""
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    reward_scale: float = 1.0

    # Training
    num_iterations: int = 100
    """number of AWR update iterations on the fixed offline dataset"""
    eval_freq: int = 5
    """evaluate every N iterations"""
    seed: int = 1
    cuda: bool = True

    # Logging
    exp_name: Optional[str] = None
    capture_video: bool = True
    save_model: bool = True

    # Computed at runtime
    batch_size: int = 0
    minibatch_size: int = 0


def collect_iql_data(checkpoints, episodes_per_ckpt, envs, num_envs, device,
                     clip_action_fn, reward_scale):
    """Collect mixed-quality offline data from multiple policy checkpoints.

    Returns a list of trajectory dicts, each with keys:
      states, actions, rewards, next_states, dones
    """
    all_trajectories = []

    for ckpt_path in checkpoints:
        print(f"  Collecting {episodes_per_ckpt} episodes from {ckpt_path}...")

        collector = Agent(envs).to(device)
        collector.load_state_dict(torch.load(ckpt_path, map_location=device))
        collector.eval()

        # Per-step storage for this checkpoint
        step_obs = []
        step_actions = []
        step_rewards = []
        step_next_obs = []
        step_dones = []

        episodes_collected = 0
        obs, _ = envs.reset()

        while episodes_collected < episodes_per_ckpt:
            with torch.no_grad():
                action = collector.get_action(obs, deterministic=False)
            next_obs, reward, term, trunc, infos = envs.step(
                clip_action_fn(action)
            )
            done = (term | trunc).float()

            step_obs.append(obs.cpu())
            step_actions.append(action.cpu())
            step_rewards.append((reward.view(-1) * reward_scale).cpu())
            step_next_obs.append(next_obs.cpu())
            step_dones.append(done.view(-1).cpu())

            if "final_info" in infos:
                episodes_collected += infos["_final_info"].sum().item()

            obs = next_obs

        del collector
        torch.cuda.empty_cache()

        # Stack this checkpoint's data: (T, num_envs, ...)
        ckpt_obs = torch.stack(step_obs)
        ckpt_actions = torch.stack(step_actions)
        ckpt_rewards = torch.stack(step_rewards)
        ckpt_next_obs = torch.stack(step_next_obs)
        ckpt_dones = torch.stack(step_dones)
        T = ckpt_obs.shape[0]

        # Segment into per-env trajectories (split at episode boundaries)
        for env_idx in range(num_envs):
            env_obs = ckpt_obs[:, env_idx]
            env_acts = ckpt_actions[:, env_idx]
            env_rews = ckpt_rewards[:, env_idx]
            env_nobs = ckpt_next_obs[:, env_idx]
            env_dns = ckpt_dones[:, env_idx]

            ep_start = 0
            for t in range(T):
                if env_dns[t] > 0.5 or t == T - 1:
                    all_trajectories.append({
                        "states": env_obs[ep_start : t + 1],
                        "actions": env_acts[ep_start : t + 1],
                        "next_states": env_nobs[ep_start : t + 1],
                        "rewards": env_rews[ep_start : t + 1],
                        "dones": env_dns[ep_start : t + 1],
                    })
                    ep_start = t + 1

        print(f"    {int(episodes_collected)} episodes, {T} steps, "
              f"{T * num_envs} transitions")

        del step_obs, step_actions, step_rewards, step_next_obs, step_dones
        del ckpt_obs, ckpt_actions, ckpt_rewards, ckpt_next_obs, ckpt_dones

    return all_trajectories


if __name__ == "__main__":
    args = tyro.cli(Args)

    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches

    if args.exp_name is None:
        args.exp_name = "iql_awr_offline"
    run_name = f"{args.exp_name}__seed{args.seed}__{int(time.time())}"

    print(f"=== IQL + AWR Offline Finetuning ===")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  IQL data: {len(args.iql_data_checkpoints)} checkpoints, "
          f"{args.iql_episodes_per_ckpt} episodes each")
    print(f"  IQL: tau={args.iql_expectile_tau}, epochs={args.iql_epochs}, "
          f"nstep={args.iql_nstep}")
    print(f"  AWR: beta={args.awr_beta}, max_weight={args.awr_max_weight}")
    print(f"  Reward: {args.reward_mode}, gamma: {args.gamma}")
    print(f"  Envs: {args.num_envs}, Steps: {args.num_steps}")
    print(f"  Batch: {args.batch_size}, Minibatch: {args.minibatch_size}")
    print(f"  Iterations: {args.num_iterations}")

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ── Environment setup (no mc_envs needed) ─────────────────────────
    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="physx_cuda" if device.type == "cuda" else "cpu",
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
    )

    envs = gym.make(args.env_id, num_envs=args.num_envs, **env_kwargs)
    eval_envs = gym.make(
        args.env_id, num_envs=args.num_eval_envs, **env_kwargs,
    )

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        print(f"  Saving eval videos to {eval_output_dir}")
        eval_envs = RecordEpisode(
            eval_envs, output_dir=eval_output_dir,
            save_trajectory=False,
            max_steps_per_video=args.max_episode_steps,
            video_fps=30,
        )

    envs = ManiSkillVectorEnv(
        envs, args.num_envs,
        ignore_terminations=False,
        record_metrics=True,
    )
    eval_envs = ManiSkillVectorEnv(
        eval_envs, args.num_eval_envs,
        ignore_terminations=False,
        record_metrics=True,
    )

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)

    # ── Agent setup (actor only — critic exists but is not trained) ────
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"  Loaded checkpoint: {args.checkpoint}")

    # Only optimize actor parameters (actor_mean + actor_logstd)
    actor_params = list(agent.actor_mean.parameters()) + [agent.actor_logstd]
    optimizer = optim.Adam(actor_params, lr=args.learning_rate, eps=1e-5)
    print(f"  Actor params: {sum(p.numel() for p in actor_params):,}")

    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(a):
        return torch.clamp(a.detach(), action_low, action_high)

    # ── Logger ─────────────────────────────────────────────────────────
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )

    # ══════════════════════════════════════════════════════════════════
    #  Phase 1: Collect offline data from multiple checkpoints
    # ══════════════════════════════════════════════════════════════════
    print("\nPhase 1: Collecting IQL training data...")
    iql_t0 = time.time()

    trajectories = collect_iql_data(
        args.iql_data_checkpoints, args.iql_episodes_per_ckpt,
        envs, args.num_envs, device, clip_action, args.reward_scale,
    )

    # Flatten trajectories into tensors for train_iql
    flat_states = torch.cat([t["states"] for t in trajectories])
    flat_actions = torch.cat([t["actions"] for t in trajectories])
    flat_rewards = torch.cat([t["rewards"] for t in trajectories])
    flat_next_states = torch.cat([t["next_states"] for t in trajectories])
    flat_dones = torch.cat([t["dones"] for t in trajectories])

    total_transitions = flat_states.shape[0]
    total_episodes = len(trajectories)
    traj_lens = [t["states"].shape[0] for t in trajectories]
    positive_rewards = (flat_rewards > 0).sum().item()

    collect_time = time.time() - iql_t0
    print(f"  Total: {total_transitions} transitions, {total_episodes} trajectories")
    print(f"  Trajectory lengths: min={min(traj_lens)}, max={max(traj_lens)}, "
          f"mean={sum(traj_lens)/len(traj_lens):.1f}")
    print(f"  Rewards: {positive_rewards} positive / {total_transitions}")
    print(f"  Collection time: {collect_time:.1f}s")

    # ══════════════════════════════════════════════════════════════════
    #  Phase 2: Train IQL (Q and V networks)
    # ══════════════════════════════════════════════════════════════════
    print(f"\nPhase 2: Training IQL (tau={args.iql_expectile_tau}, "
          f"nstep={args.iql_nstep})...")
    iql_train_t0 = time.time()

    # Adapter: map our args to what train_iql expects
    iql_args = SimpleNamespace(
        lr=args.iql_lr,
        weight_decay=1e-4,
        epochs=args.iql_epochs,
        batch_size=args.iql_batch_size,
        gamma=args.gamma,
        expectile_tau=args.iql_expectile_tau,
        tau_polyak=0.005,
        patience=args.iql_patience,
        grad_clip=0.5,
    )

    # Compute n-step targets if requested
    nstep_kw = {}
    if args.iql_nstep > 1:
        print(f"  Computing {args.iql_nstep}-step TD targets...")
        nret, boot_s, ndisc = compute_nstep_targets(
            trajectories, args.iql_nstep, args.gamma
        )
        nstep_kw = dict(
            nstep_returns=nret, bootstrap_states=boot_s, nstep_discounts=ndisc
        )
        print(f"  n-step returns: mean={nret.mean():.4f}, std={nret.std():.4f}")
        frac_boot = (ndisc > 0).float().mean()
        print(f"  Bootstrapped: {frac_boot:.1%} of transitions")

    q_net, v_net = train_iql(
        flat_states, flat_actions, flat_rewards, flat_next_states, flat_dones,
        device, iql_args, **nstep_kw,
    )

    iql_train_time = time.time() - iql_train_t0
    print(f"  IQL training time: {iql_train_time:.1f}s")

    # Free IQL training data
    del trajectories, flat_states, flat_actions, flat_rewards
    del flat_next_states, flat_dones, nstep_kw
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════
    #  Phase 3: Collect finetune data + compute IQL advantages
    # ══════════════════════════════════════════════════════════════════
    print("\nPhase 3: Collecting finetune data and computing IQL advantages...")

    # ── Storage ────────────────────────────────────────────────────────
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape,
        device=device,
    )
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape,
        device=device,
    )
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)

    agent.eval()
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    rollout_t0 = time.time()
    for step in range(args.num_steps):
        obs[step] = next_obs
        dones[step] = next_done

        with torch.no_grad():
            action = agent.get_action(next_obs, deterministic=False)
        actions[step] = action

        next_obs, reward, terminations, truncations, infos = envs.step(
            clip_action(action)
        )
        next_done = (terminations | truncations).float()
        rewards[step] = reward.view(-1) * args.reward_scale

    rollout_time = time.time() - rollout_t0

    # ── Flatten batch ──────────────────────────────────────────────────
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)

    # ── Compute IQL advantages: A(s,a) = Q(s,a) - V(s) ────────────────
    with torch.no_grad():
        iql_advantages = []
        for start in range(0, args.batch_size, 1024):
            end = min(start + 1024, args.batch_size)
            s = b_obs[start:end]
            a = b_actions[start:end]
            q = q_net(s, a).squeeze(-1)
            v = v_net(s).squeeze(-1)
            iql_advantages.append(q - v)
        b_advantages = torch.cat(iql_advantages)

    # Free IQL networks (no longer needed)
    del q_net, v_net
    torch.cuda.empty_cache()

    # ── Precompute AWR weights (fixed, not updated) ────────────────────
    if args.norm_adv:
        b_advantages_norm = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
    else:
        b_advantages_norm = b_advantages

    b_weights = torch.exp(b_advantages_norm / args.awr_beta)
    b_weights = torch.clamp(b_weights, max=args.awr_max_weight)

    # ── Logging ────────────────────────────────────────────────────────
    writer.add_scalar("charts/advantage_mean", b_advantages.mean().item(), 0)
    writer.add_scalar("charts/advantage_std", b_advantages.std().item(), 0)
    writer.add_scalar("time/iql_data_collection", collect_time, 0)
    writer.add_scalar("time/iql_training", iql_train_time, 0)
    writer.add_scalar("time/finetune_rollout", rollout_time, 0)

    print(f"\n  Dataset: {args.batch_size} samples")
    print(f"  IQL Advantage: mean={b_advantages.mean():.4f}, "
          f"std={b_advantages.std():.4f}, "
          f"pos%={(b_advantages > 0).float().mean().item():.1%}")
    print(f"  AWR weights: mean={b_weights.mean().item():.2f}, "
          f"max={b_weights.max().item():.2f}")
    print(f"  Finetune rollout: {rollout_time:.1f}s")
    print(f"  Starting {args.num_iterations} iterations of offline AWR...\n")

    # ══════════════════════════════════════════════════════════════════
    #  Phase 4: AWR update on fixed data (same as mc_finetune_awr_offline)
    # ══════════════════════════════════════════════════════════════════
    start_time = time.time()

    for iteration in range(1, args.num_iterations + 1):
        global_step = iteration
        agent.eval()

        # ── Evaluation ─────────────────────────────────────────────
        if iteration == 1 or iteration % args.eval_freq == 0 or iteration == args.num_iterations:
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.max_episode_steps):
                with torch.no_grad():
                    eval_obs, _, eval_term, eval_trunc, eval_infos = eval_envs.step(
                        agent.get_action(eval_obs, deterministic=True)
                    )
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v[mask])

            sr_vals = eval_metrics.get("success_once", [])
            if sr_vals:
                success_rate = torch.cat(sr_vals).float().mean().item()
            else:
                success_rate = 0.0

            print(
                f"Iter {iteration}/{args.num_iterations} | "
                f"SR={success_rate:.1%} | "
                f"episodes={num_episodes}"
            )
            writer.add_scalar("eval/success_rate", success_rate, global_step)
            for k, v in eval_metrics.items():
                vals = torch.cat(v) if len(v) > 1 else v[0]
                writer.add_scalar(f"eval/{k}", vals.float().mean().item(), global_step)

            if args.save_model:
                os.makedirs(f"runs/{run_name}", exist_ok=True)
                torch.save(
                    agent.state_dict(),
                    f"runs/{run_name}/ckpt_{iteration}.pt",
                )

        # ── AWR update on fixed data ───────────────────────────────
        agent.train()
        b_inds = np.arange(args.batch_size)
        update_t0 = time.time()

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, _ = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )

                # AWR: weighted negative log-likelihood
                mb_weights = b_weights[mb_inds]
                mb_weights = mb_weights / mb_weights.sum() * len(mb_weights)  # normalize per minibatch
                policy_loss = -(mb_weights.detach() * newlogprob).mean()

                entropy_loss = entropy.mean()
                loss = policy_loss - args.ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor_params, args.max_grad_norm)
                optimizer.step()

        update_time = time.time() - update_t0

        # ── Logging ────────────────────────────────────────────────
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/awr_weight_mean", b_weights.mean().item(), global_step)
        writer.add_scalar("losses/awr_weight_max", b_weights.max().item(), global_step)
        writer.add_scalar("time/update", update_time, global_step)

    # ── Final save ─────────────────────────────────────────────────────
    if args.save_model:
        torch.save(agent.state_dict(), f"runs/{run_name}/final_ckpt.pt")
        print(f"\nFinal model saved to runs/{run_name}/final_ckpt.pt")

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.1f}s")

    writer.close()
    envs.close()
    eval_envs.close()
