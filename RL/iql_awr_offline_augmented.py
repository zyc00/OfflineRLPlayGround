"""IQL + AWR Offline Finetuning with MC Re-rollout Data Augmentation.

Extends iql_awr_offline.py by augmenting IQL training data with trajectories
from MC re-rollout (optimal policy executing from suboptimal states).

Phase 1:   Collect offline data from multiple checkpoints (same as original)
Phase 1.5: Rollout with initial policy, MC re-rollout with optimal policy,
           save re-rollout trajectories as additional IQL training data
Phase 2:   Train IQL on merged (Phase 1 + Phase 1.5) data
Phase 3:   Rollout finetune data with initial policy, compute IQL advantages
Phase 4:   AWR update on fixed finetune data (same as original)

The MC re-rollout trajectories provide (suboptimal_state, optimal_action) pairs
that pure checkpoint rollouts cannot produce, giving IQL better coverage.

Usage:
  # With MC augmentation (default)
  python -m RL.iql_awr_offline_augmented --augment_with_mc

  # Without augmentation (equivalent to iql_awr_offline.py)
  python -m RL.iql_awr_offline_augmented --no-augment_with_mc
"""

import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal, Optional

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from tqdm import tqdm
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

    # MC data augmentation
    augment_with_mc: bool = True
    """whether to augment IQL training data with MC re-rollout trajectories"""
    optimal_checkpoint: str = "runs/pickcube_ppo/ckpt_301.pt"
    """optimal policy for MC re-rollout (only used if augment_with_mc=True)"""
    mc_samples: int = 16
    """MC samples per (s,a) for Q and V each"""
    mc_num_envs: int = 100
    """num envs for the initial policy rollout in augmentation phase"""
    mc_num_mc_envs: int = 0
    """MC re-rollout envs. 0 = auto (mc_num_envs * 2 * mc_samples)"""
    mc_num_steps: int = 50
    """rollout length for augmentation data collection"""

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
    advantage_mode: Literal["iql", "random", "reversed"] = "iql"
    """iql=normal IQL advantages, random=random noise, reversed=negate IQL advantages"""

    # Training
    num_iterations: int = 100
    """number of AWR update iterations on the fixed offline dataset"""
    eval_freq: int = 5
    """evaluate every N iterations"""
    seed: int = 1
    cuda: bool = True

    # Cache
    cache_path: Optional[str] = None
    """Path to cache Phase 1 + 1.5 data (offline + MC trajectories).
    If exists, skip data collection and load from cache."""

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


def collect_mc_augmentation(args, envs, device, clip_action_fn):
    """Phase 1.5: Collect MC re-rollout trajectories for IQL data augmentation.

    1. Create mc_envs for parallel re-rollout
    2. Load optimal_agent
    3. Rollout with initial policy (checkpoint), saving env states
    4. MC re-rollout from each saved state using optimal policy
    5. Save complete trajectories from re-rollout
    6. Cleanup mc_envs and optimal_agent

    Returns a list of trajectory dicts (same format as collect_iql_data).
    """
    num_mc_envs = args.mc_num_mc_envs
    mc_num_envs = args.mc_num_envs
    mc_samples = args.mc_samples
    mc_num_steps = args.mc_num_steps
    max_episode_steps = args.max_episode_steps

    samples_per_env = num_mc_envs // mc_num_envs

    print(f"  MC augmentation: {mc_num_envs} rollout envs, "
          f"{num_mc_envs} MC envs, {mc_samples} samples, "
          f"{mc_num_steps} steps")

    # ── Create mc_envs ────────────────────────────────────────────────
    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="physx_cuda" if device.type == "cuda" else "cpu",
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
    )

    # Rollout envs (separate small set for augmentation)
    aug_envs_raw = gym.make(args.env_id, num_envs=mc_num_envs, **env_kwargs)
    mc_envs_raw = gym.make(args.env_id, num_envs=num_mc_envs, **env_kwargs)

    if isinstance(aug_envs_raw.action_space, gym.spaces.Dict):
        aug_envs_raw = FlattenActionSpaceWrapper(aug_envs_raw)
        mc_envs_raw = FlattenActionSpaceWrapper(mc_envs_raw)

    aug_envs = ManiSkillVectorEnv(
        aug_envs_raw, mc_num_envs,
        ignore_terminations=False,
        record_metrics=True,
    )
    mc_envs = ManiSkillVectorEnv(
        mc_envs_raw, num_mc_envs,
        ignore_terminations=False,
        record_metrics=False,
    )

    # ── Load agents ───────────────────────────────────────────────────
    rollout_agent = Agent(aug_envs).to(device)
    rollout_agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    rollout_agent.eval()

    optimal_agent = Agent(aug_envs).to(device)
    optimal_agent.load_state_dict(torch.load(args.optimal_checkpoint, map_location=device))
    optimal_agent.eval()

    action_low = torch.from_numpy(aug_envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(aug_envs.single_action_space.high).to(device)

    def clip_action(a):
        return torch.clamp(a.detach(), action_low, action_high)

    # ── MC env helpers ────────────────────────────────────────────────
    _mc_zero_action = torch.zeros(
        num_mc_envs, *aug_envs.single_action_space.shape, device=device
    )

    def _clone_state(state_dict):
        if isinstance(state_dict, dict):
            return {k: _clone_state(v) for k, v in state_dict.items()}
        return state_dict.clone()

    def _expand_state(state_dict, repeats):
        if isinstance(state_dict, dict):
            return {k: _expand_state(v, repeats) for k, v in state_dict.items()}
        if isinstance(state_dict, torch.Tensor) and state_dict.dim() > 0:
            return state_dict.repeat_interleave(repeats, dim=0)
        return state_dict

    def _restore_mc_state(state_dict, seed=None):
        mc_envs.reset(seed=seed if seed is not None else args.seed)
        mc_envs.base_env.set_state_dict(state_dict)
        mc_envs.base_env.step(_mc_zero_action)
        mc_envs.base_env.set_state_dict(state_dict)
        mc_envs.base_env._elapsed_steps[:] = 0
        return mc_envs.base_env.get_obs()

    # Precompute V-replica indices
    v_indices = []
    for i in range(mc_num_envs):
        base = i * samples_per_env
        v_indices.extend(range(base + mc_samples, base + 2 * mc_samples))
    v_indices = torch.tensor(v_indices, device=device, dtype=torch.long)

    # ── Rollout with initial policy, saving states ────────────────────
    print(f"  Rolling out {mc_num_steps} steps with initial policy...")
    rollout_t0 = time.time()

    saved_states = []
    rollout_actions = torch.zeros(
        (mc_num_steps, mc_num_envs) + aug_envs.single_action_space.shape,
        device=device,
    )

    next_obs, _ = aug_envs.reset(seed=args.seed)
    with torch.no_grad():
        for step in range(mc_num_steps):
            saved_states.append(_clone_state(aug_envs.base_env.get_state_dict()))
            action = rollout_agent.get_action(next_obs, deterministic=False)
            rollout_actions[step] = action
            next_obs, _, _, _, _ = aug_envs.step(clip_action(action))

    rollout_time = time.time() - rollout_t0
    print(f"  Rollout time: {rollout_time:.1f}s")

    # ── MC re-rollout: collect trajectories ───────────────────────────
    print(f"  MC re-rollout ({mc_num_steps} timesteps)...")
    rerollout_t0 = time.time()
    mc_augment_trajectories = []

    with torch.no_grad():
        for t in tqdm(range(mc_num_steps), desc="  MC augmentation"):
            # 1. Expand states → mc_envs
            expanded_state = _expand_state(saved_states[t], samples_per_env)
            mc_obs = _restore_mc_state(expanded_state, seed=args.seed + t)

            # 2. Build first actions
            first_actions = torch.zeros(
                num_mc_envs, *aug_envs.single_action_space.shape, device=device
            )
            # Q replicas: use initial policy's action
            for i in range(mc_num_envs):
                base = i * samples_per_env
                first_actions[base : base + mc_samples] = rollout_actions[t][i]
            # V replicas: sample from optimal policy
            v_obs = mc_obs[v_indices]
            first_actions[v_indices] = optimal_agent.get_action(
                v_obs, deterministic=False
            )

            # 3. Step and collect trajectory data
            traj_obs = [mc_obs.cpu()]
            traj_acts = [first_actions.cpu()]

            mc_obs, rew, term, trunc, _ = mc_envs.step(clip_action(first_actions))
            traj_rews = [(rew.view(-1) * args.reward_scale).cpu()]
            traj_next = [mc_obs.cpu()]
            traj_dones = [(term | trunc).view(-1).float().cpu()]
            env_done = (term | trunc).view(-1).bool()

            # Follow optimal policy until all done
            for _ in range(max_episode_steps - 1):
                if env_done.all():
                    break
                prev_obs = mc_obs.clone()
                a = optimal_agent.get_action(mc_obs, deterministic=False)
                mc_obs, rew, term, trunc, _ = mc_envs.step(clip_action(a))

                traj_obs.append(prev_obs.cpu())
                traj_acts.append(a.cpu())
                traj_rews.append(
                    (rew.view(-1) * args.reward_scale * (~env_done).float()).cpu()
                )
                traj_next.append(mc_obs.cpu())
                traj_dones.append((term | trunc).view(-1).float().cpu())
                env_done = env_done | (term | trunc).view(-1).bool()

            # 4. Stack: (rollout_len, num_mc_envs, ...)
            t_obs = torch.stack(traj_obs)
            t_acts = torch.stack(traj_acts)
            t_rews = torch.stack(traj_rews)
            t_next = torch.stack(traj_next)
            t_dones = torch.stack(traj_dones)
            rollout_len = t_obs.shape[0]

            # 5. Extract per-env trajectories (only Q and V replicas, skip idle)
            for env_idx in range(num_mc_envs):
                local_idx = env_idx % samples_per_env
                if local_idx >= 2 * mc_samples:
                    continue  # idle replica, skip

                # Find episode length (first done)
                env_dones = t_dones[:, env_idx]
                done_indices = (env_dones > 0.5).nonzero(as_tuple=True)[0]
                ep_len = (done_indices[0].item() + 1) if len(done_indices) > 0 else rollout_len

                mc_augment_trajectories.append({
                    "states": t_obs[:ep_len, env_idx],
                    "actions": t_acts[:ep_len, env_idx],
                    "rewards": t_rews[:ep_len, env_idx],
                    "next_states": t_next[:ep_len, env_idx],
                    "dones": t_dones[:ep_len, env_idx],
                })

            # Free per-timestep data
            del t_obs, t_acts, t_rews, t_next, t_dones
            del traj_obs, traj_acts, traj_rews, traj_next, traj_dones

    rerollout_time = time.time() - rerollout_t0
    print(f"  MC re-rollout time: {rerollout_time:.1f}s")

    n_mc_transitions = sum(t["states"].shape[0] for t in mc_augment_trajectories)
    print(f"  MC trajectories: {len(mc_augment_trajectories)}, "
          f"{n_mc_transitions} transitions")

    # ── Cleanup ───────────────────────────────────────────────────────
    del rollout_agent, optimal_agent
    del saved_states, rollout_actions
    aug_envs.close()
    mc_envs.close()
    del aug_envs, mc_envs
    torch.cuda.empty_cache()

    return mc_augment_trajectories


if __name__ == "__main__":
    args = tyro.cli(Args)

    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches

    # Auto-compute mc_num_mc_envs
    if args.augment_with_mc and args.mc_num_mc_envs == 0:
        args.mc_num_mc_envs = args.mc_num_envs * 2 * args.mc_samples

    if args.augment_with_mc:
        samples_per_env = args.mc_num_mc_envs // args.mc_num_envs
        assert args.mc_num_mc_envs % args.mc_num_envs == 0, (
            f"mc_num_mc_envs ({args.mc_num_mc_envs}) must be divisible by "
            f"mc_num_envs ({args.mc_num_envs})"
        )
        assert samples_per_env >= 2 * args.mc_samples, (
            f"samples_per_env ({samples_per_env}) must be >= "
            f"2 * mc_samples ({2 * args.mc_samples})"
        )

    if args.exp_name is None:
        args.exp_name = "iql_awr_offline_aug" if args.augment_with_mc else "iql_awr_offline"
    run_name = f"{args.exp_name}__seed{args.seed}__{int(time.time())}"

    print(f"=== IQL + AWR Offline Finetuning {'(MC Augmented)' if args.augment_with_mc else ''} ===")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  IQL data: {len(args.iql_data_checkpoints)} checkpoints, "
          f"{args.iql_episodes_per_ckpt} episodes each")
    print(f"  IQL: tau={args.iql_expectile_tau}, epochs={args.iql_epochs}, "
          f"nstep={args.iql_nstep}")
    if args.augment_with_mc:
        print(f"  MC augmentation: mc_samples={args.mc_samples}, "
              f"optimal={args.optimal_checkpoint}")
        print(f"  MC envs: rollout={args.mc_num_envs}, "
              f"mc={args.mc_num_mc_envs}, steps={args.mc_num_steps}")
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

    # ── Environment setup ─────────────────────────────────────────────
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
    #  Phase 1 + 1.5: Collect data (or load from cache)
    # ══════════════════════════════════════════════════════════════════
    use_cache = args.cache_path and os.path.exists(args.cache_path)

    if use_cache:
        print(f"\n  Loading cached data from {args.cache_path}...")
        cache = torch.load(args.cache_path, map_location="cpu")
        trajectories = cache["trajectories"]
        cache_reward_scale = cache.get("reward_scale", 1.0)

        # Rescale if reward_scale differs from cache
        if abs(args.reward_scale - cache_reward_scale) > 1e-8:
            scale_factor = args.reward_scale / cache_reward_scale
            print(f"  Rescaling: cache={cache_reward_scale} → {args.reward_scale} (×{scale_factor})")
            for t in trajectories:
                t["rewards"] = t["rewards"] * scale_factor

        n_offline = cache.get("n_offline", 0)
        n_mc = cache.get("n_mc", 0)
        total = sum(t["states"].shape[0] for t in trajectories)
        print(f"  Loaded: {total} transitions, {len(trajectories)} trajectories")
        print(f"  (offline: {n_offline}, mc: {n_mc})")
        collect_time = 0.0
        del cache
    else:
        # ── Phase 1: Collect offline data ──────────────────────────────
        print("\nPhase 1: Collecting IQL training data...")
        iql_t0 = time.time()

        trajectories_offline = collect_iql_data(
            args.iql_data_checkpoints, args.iql_episodes_per_ckpt,
            envs, args.num_envs, device, clip_action, args.reward_scale,
        )

        collect_time = time.time() - iql_t0

        n_offline = sum(t["states"].shape[0] for t in trajectories_offline)
        print(f"  Offline data: {n_offline} transitions, "
              f"{len(trajectories_offline)} trajectories")
        print(f"  Collection time: {collect_time:.1f}s")

        # ── Phase 1.5: MC re-rollout augmentation (optional) ──────────
        n_mc = 0
        if args.augment_with_mc:
            print(f"\nPhase 1.5: MC re-rollout data augmentation...")
            aug_t0 = time.time()

            trajectories_mc = collect_mc_augmentation(args, envs, device, clip_action)

            aug_time = time.time() - aug_t0
            n_mc = sum(t["states"].shape[0] for t in trajectories_mc)
            print(f"  MC augment:   {n_mc} transitions, "
                  f"{len(trajectories_mc)} trajectories")
            print(f"  Augmentation time: {aug_time:.1f}s")

            # Merge
            trajectories = trajectories_offline + trajectories_mc
            print(f"  Total:        {n_offline + n_mc} transitions, "
                  f"{len(trajectories)} trajectories")

            writer.add_scalar("time/mc_augmentation", aug_time, 0)
            writer.add_scalar("data/mc_transitions", n_mc, 0)
            writer.add_scalar("data/mc_trajectories", len(trajectories_mc), 0)

            del trajectories_mc
        else:
            trajectories = trajectories_offline

        del trajectories_offline

        # ── Save cache ─────────────────────────────────────────────────
        if args.cache_path:
            os.makedirs(os.path.dirname(args.cache_path) or ".", exist_ok=True)
            torch.save({
                "trajectories": trajectories,
                "n_offline": n_offline,
                "n_mc": n_mc,
                "reward_scale": args.reward_scale,
            }, args.cache_path)
            print(f"  Cached data to {args.cache_path}")

    writer.add_scalar("data/offline_transitions", n_offline, 0)

    # ══════════════════════════════════════════════════════════════════
    #  Phase 2: Train IQL (Q and V networks)
    # ══════════════════════════════════════════════════════════════════
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

    print(f"\nPhase 2: Training IQL (tau={args.iql_expectile_tau}, "
          f"nstep={args.iql_nstep})...")
    print(f"  Total: {total_transitions} transitions, {total_episodes} trajectories")
    print(f"  Trajectory lengths: min={min(traj_lens)}, max={max(traj_lens)}, "
          f"mean={sum(traj_lens)/len(traj_lens):.1f}")
    print(f"  Rewards: {positive_rewards} positive / {total_transitions}")

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

    # ── Advantage ablation ─────────────────────────────────────────────
    if args.advantage_mode == "random":
        print("  [ABLATION] Replacing IQL advantages with random noise")
        b_advantages = torch.randn_like(b_advantages)
    elif args.advantage_mode == "reversed":
        print("  [ABLATION] Reversing IQL advantages (negating)")
        b_advantages = -b_advantages

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
