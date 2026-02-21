"""IQL Fitting Quality Analysis on MC16 Ground Truth Data.

Core question: If we give IQL the best data (MC16 re-rollout trajectories),
can it learn accurate Q and V? This distinguishes two bottlenecks:
  - Data problem: offline checkpoint data coverage insufficient
  - Method problem: IQL function approximation / expectile regression is inherently limited

Pipeline:
  1. Rollout with ckpt_101, save env states
  2. MC16 optimal re-rollout → mc_q, mc_v ground truth + trajectory data
  3. (Optional) Collect multi-checkpoint offline data
  4. Train IQL on chosen data mode (mc_only / mc+offline / offline_only)
  5. IQL predict Q(s,a), V(s) on rollout states
  6. Scatter plot: IQL predictions vs MC ground truth

Usage:
  python -m RL.iql_fit_analysis --iql_data_mode mc_only --iql_expectile_tau 0.7
  python -m RL.iql_fit_analysis --iql_data_mode mc+offline --iql_expectile_tau 0.7
  python -m RL.iql_fit_analysis --iql_data_mode offline_only --iql_expectile_tau 0.7
"""

import os
import random
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal, Optional

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from data.data_collection.ppo import Agent
from methods.iql.iql import QNetwork, train_iql, compute_nstep_targets


@dataclass
class Args:
    checkpoint: str = "runs/pickcube_ppo/ckpt_76_logstd-1.5.pt"
    """pretrained checkpoint for rollout (V3: det SR=43.8%)"""
    optimal_checkpoint: str = "runs/pickcube_ppo/ckpt_301.pt"
    """optimal policy for MC re-rollouts"""

    # MC re-rollout (V3: 128 envs, 800 steps)
    mc_samples: int = 16
    """number of MC samples per (s,a) for Q and V each"""
    num_envs: int = 128
    """number of parallel envs for rollout"""
    num_mc_envs: int = 0
    """MC re-rollout envs. 0 = auto (num_envs * 2 * mc_samples)"""
    num_steps: int = 800
    """rollout length (V3: 800 for 4x data scaling)"""
    max_episode_steps: int = 50
    gamma: float = 0.8
    reward_scale: float = 1.0

    # IQL data
    iql_data_mode: Literal["mc_only", "mc+offline", "offline_only"] = "mc_only"
    """mc_only: only MC re-rollout trajectories
    mc+offline: MC + multi-checkpoint offline data
    offline_only: only multi-checkpoint offline data (baseline)"""
    iql_data_checkpoints: tuple[str, ...] = (
        "runs/pickcube_ppo/ckpt_1.pt",
        "runs/pickcube_ppo/ckpt_51.pt",
        "runs/pickcube_ppo/ckpt_101.pt",
        "runs/pickcube_ppo/ckpt_201.pt",
        "runs/pickcube_ppo/ckpt_301.pt",
    )
    iql_episodes_per_ckpt: int = 200
    """episodes per checkpoint for offline data"""
    iql_offline_num_envs: int = 512
    """num envs for offline data collection (V3: 512)"""

    # IQL training
    iql_expectile_tau: float = 0.7
    """IQL expectile tau"""
    iql_epochs: int = 200
    iql_lr: float = 3e-4
    iql_batch_size: int = 256
    iql_nstep: int = 1
    iql_patience: int = 50
    iql_max_transitions: int = 0
    """max transitions for IQL training (0 = no limit). Useful for mc_only mode
    which can have millions of transitions. Subsamples trajectories to stay under limit."""

    # Environment
    env_id: str = "PickCube-v1"
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"
    seed: int = 1
    cuda: bool = True

    # Cache
    cache_path: Optional[str] = None
    """Path to cache MC re-rollout data (obs, actions, mc_q, mc_v, mc_trajectories).
    If exists, skip Steps 1-2 (rollout + MC re-rollout) and load from cache.
    Allows re-running with different IQL settings without re-collecting."""

    # Output
    output: str = "runs/iql_fit_analysis.png"
    save_data: bool = True
    """save raw data as .pt for later analysis"""


def collect_offline_data(checkpoints, episodes_per_ckpt, env_id, num_envs,
                         device, env_kwargs, reward_scale):
    """Collect mixed-quality offline data from multiple policy checkpoints.

    Creates its own envs internally to avoid interference with rollout envs.
    Returns a list of trajectory dicts with keys:
      states, actions, rewards, next_states, dones
    """
    envs = gym.make(env_id, num_envs=num_envs, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(
        envs, num_envs, ignore_terminations=False, record_metrics=True,
    )

    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(a):
        return torch.clamp(a.detach(), action_low, action_high)

    all_trajectories = []

    for ckpt_path in checkpoints:
        print(f"  Collecting {episodes_per_ckpt} episodes from {ckpt_path}...")

        collector = Agent(envs).to(device)
        collector.load_state_dict(torch.load(ckpt_path, map_location=device))
        collector.eval()

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
            next_obs, reward, term, trunc, infos = envs.step(clip_action(action))
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

        ckpt_obs = torch.stack(step_obs)
        ckpt_actions = torch.stack(step_actions)
        ckpt_rewards = torch.stack(step_rewards)
        ckpt_next_obs = torch.stack(step_next_obs)
        ckpt_dones = torch.stack(step_dones)
        T = ckpt_obs.shape[0]

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

    envs.close()
    return all_trajectories


if __name__ == "__main__":
    import tyro
    args = tyro.cli(Args)

    # Auto-compute num_mc_envs
    if args.num_mc_envs == 0:
        args.num_mc_envs = args.num_envs * 2 * args.mc_samples

    samples_per_env = args.num_mc_envs // args.num_envs
    assert args.num_mc_envs % args.num_envs == 0
    assert samples_per_env >= 2 * args.mc_samples

    print(f"=== IQL Fitting Quality Analysis ===")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Optimal: {args.optimal_checkpoint}")
    print(f"  MC samples: {args.mc_samples}, gamma: {args.gamma}")
    print(f"  Envs: {args.num_envs}, MC envs: {args.num_mc_envs}, Steps: {args.num_steps}")
    print(f"  IQL data mode: {args.iql_data_mode}")
    print(f"  IQL tau: {args.iql_expectile_tau}, epochs: {args.iql_epochs}")
    if args.cache_path:
        print(f"  Cache: {args.cache_path}")

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="physx_cuda" if device.type == "cuda" else "cpu",
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
    )

    use_cache = args.cache_path and os.path.exists(args.cache_path)

    if use_cache:
        # ══════════════════════════════════════════════════════════════
        #  Load cached MC re-rollout data (skip Steps 1-2)
        # ══════════════════════════════════════════════════════════════
        print(f"\n  Loading cached data from {args.cache_path}...")
        cache = torch.load(args.cache_path, map_location=device)
        obs = cache["obs"]
        actions = cache["actions"]
        mc_q = cache["mc_q"]
        mc_v = cache["mc_v"]
        mc_trajectories = cache["mc_trajectories"]
        cache_reward_scale = cache.get("reward_scale", 1.0)

        # Rescale if reward_scale differs from cache
        if abs(args.reward_scale - cache_reward_scale) > 1e-8:
            scale_factor = args.reward_scale / cache_reward_scale
            print(f"  Rescaling: cache={cache_reward_scale} → {args.reward_scale} (×{scale_factor})")
            mc_q *= scale_factor
            mc_v *= scale_factor
            for t in mc_trajectories:
                t["rewards"] = t["rewards"] * scale_factor

        mc_adv = mc_q - mc_v
        n_mc_transitions = sum(t["states"].shape[0] for t in mc_trajectories)
        print(f"  Loaded: obs={list(obs.shape)}, mc_q={list(mc_q.shape)}")
        print(f"  MC trajectories: {len(mc_trajectories)}, {n_mc_transitions} transitions")
        print(f"  MC Q: mean={mc_q.mean():.4f}, std={mc_q.std():.4f}")
        print(f"  MC V: mean={mc_v.mean():.4f}, std={mc_v.std():.4f}")
        print(f"  MC A: mean={mc_adv.mean():.4f}, std={mc_adv.std():.4f}")
        del cache

    else:
        # ── Environment setup ──────────────────────────────────────────
        envs = gym.make(args.env_id, num_envs=args.num_envs, **env_kwargs)
        mc_envs_raw = gym.make(args.env_id, num_envs=args.num_mc_envs, **env_kwargs)

        if isinstance(envs.action_space, gym.spaces.Dict):
            envs = FlattenActionSpaceWrapper(envs)
            mc_envs_raw = FlattenActionSpaceWrapper(mc_envs_raw)

        envs = ManiSkillVectorEnv(
            envs, args.num_envs, ignore_terminations=False, record_metrics=True,
        )
        mc_envs = ManiSkillVectorEnv(
            mc_envs_raw, args.num_mc_envs, ignore_terminations=False, record_metrics=False,
        )

        max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)

        # ── Load agents ────────────────────────────────────────────────
        agent = Agent(envs).to(device)
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
        agent.eval()
        print(f"  Loaded rollout policy: {args.checkpoint}")

        optimal_agent = Agent(envs).to(device)
        optimal_agent.load_state_dict(torch.load(args.optimal_checkpoint, map_location=device))
        optimal_agent.eval()
        print(f"  Loaded optimal policy: {args.optimal_checkpoint}")

        action_low = torch.from_numpy(envs.single_action_space.low).to(device)
        action_high = torch.from_numpy(envs.single_action_space.high).to(device)

        def clip_action(a):
            return torch.clamp(a.detach(), action_low, action_high)

        # ── Env helpers ────────────────────────────────────────────────
        _zero_action = torch.zeros(
            args.num_envs, *envs.single_action_space.shape, device=device
        )
        _mc_zero_action = torch.zeros(
            args.num_mc_envs, *envs.single_action_space.shape, device=device
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
        for i in range(args.num_envs):
            base = i * samples_per_env
            v_indices.extend(range(base + args.mc_samples, base + 2 * args.mc_samples))
        v_indices = torch.tensor(v_indices, device=device, dtype=torch.long)

        # ══════════════════════════════════════════════════════════════
        #  Step 1: Rollout with initial policy, saving states
        # ══════════════════════════════════════════════════════════════
        print("\nStep 1: Rolling out with initial policy...")
        rollout_t0 = time.time()

        obs = torch.zeros(
            (args.num_steps, args.num_envs) + envs.single_observation_space.shape,
            device=device,
        )
        actions = torch.zeros(
            (args.num_steps, args.num_envs) + envs.single_action_space.shape,
            device=device,
        )
        saved_states = []

        next_obs, _ = envs.reset(seed=args.seed)
        with torch.no_grad():
            for step in range(args.num_steps):
                saved_states.append(_clone_state(envs.base_env.get_state_dict()))
                obs[step] = next_obs
                action = agent.get_action(next_obs, deterministic=False)
                actions[step] = action
                next_obs, _, _, _, _ = envs.step(clip_action(action))

        rollout_time = time.time() - rollout_t0
        print(f"  Rollout time: {rollout_time:.1f}s")

        # ══════════════════════════════════════════════════════════════
        #  Step 2: MC re-rollout → mc_q, mc_v ground truth + trajectories
        # ══════════════════════════════════════════════════════════════
        print(f"\nStep 2: MC{args.mc_samples} re-rollout (collecting trajectories)...")
        rerollout_t0 = time.time()

        mc_q = torch.zeros((args.num_steps, args.num_envs), device=device)
        mc_v = torch.zeros((args.num_steps, args.num_envs), device=device)
        mc_trajectories = []

        with torch.no_grad():
            for t in tqdm(range(args.num_steps), desc="  MC re-rollout"):
                # 1. Expand num_envs states → num_mc_envs
                expanded_state = _expand_state(saved_states[t], samples_per_env)

                # 2. Restore to mc_envs
                mc_obs = _restore_mc_state(expanded_state, seed=args.seed + t)

                # 3. Build first actions
                first_actions = torch.zeros(
                    args.num_mc_envs, *envs.single_action_space.shape, device=device
                )
                # Q replicas: use initial policy's action
                for i in range(args.num_envs):
                    base = i * samples_per_env
                    first_actions[base : base + args.mc_samples] = actions[t][i]
                # V replicas: sample from optimal policy
                v_obs = mc_obs[v_indices]
                first_actions[v_indices] = optimal_agent.get_action(
                    v_obs, deterministic=False
                )

                # 4. Step and collect trajectory data
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

                # 5. Stack: (rollout_len, num_mc_envs, ...)
                t_obs = torch.stack(traj_obs)
                t_acts = torch.stack(traj_acts)
                t_rews = torch.stack(traj_rews)
                t_next = torch.stack(traj_next)
                t_dones = torch.stack(traj_dones)
                rollout_len = t_obs.shape[0]

                # 6. Extract per-env trajectories (only Q and V replicas)
                for env_idx in range(args.num_mc_envs):
                    local_idx = env_idx % samples_per_env
                    if local_idx >= 2 * args.mc_samples:
                        continue  # idle replica

                    env_dones = t_dones[:, env_idx]
                    done_indices = (env_dones > 0.5).nonzero(as_tuple=True)[0]
                    ep_len = (done_indices[0].item() + 1) if len(done_indices) > 0 else rollout_len

                    mc_trajectories.append({
                        "states": t_obs[:ep_len, env_idx],
                        "actions": t_acts[:ep_len, env_idx],
                        "rewards": t_rews[:ep_len, env_idx],
                        "next_states": t_next[:ep_len, env_idx],
                        "dones": t_dones[:ep_len, env_idx],
                    })

                # 7. Compute discounted returns for mc_q and mc_v
                ret = torch.zeros(args.num_mc_envs, device=device)
                for s in reversed(range(len(traj_rews))):
                    ret = traj_rews[s].to(device) + args.gamma * ret

                ret = ret.view(args.num_envs, samples_per_env)
                mc_q[t] = ret[:, :args.mc_samples].mean(dim=1)
                mc_v[t] = ret[:, args.mc_samples : 2 * args.mc_samples].mean(dim=1)

                # Free per-timestep data
                del t_obs, t_acts, t_rews, t_next, t_dones
                del traj_obs, traj_acts, traj_rews, traj_next, traj_dones

        rerollout_time = time.time() - rerollout_t0

        mc_adv = mc_q - mc_v
        n_mc_transitions = sum(t["states"].shape[0] for t in mc_trajectories)
        print(f"  MC re-rollout time: {rerollout_time:.1f}s")
        print(f"  MC trajectories: {len(mc_trajectories)}, {n_mc_transitions} transitions")
        print(f"  MC Q: mean={mc_q.mean():.4f}, std={mc_q.std():.4f}")
        print(f"  MC V: mean={mc_v.mean():.4f}, std={mc_v.std():.4f}")
        print(f"  MC A: mean={mc_adv.mean():.4f}, std={mc_adv.std():.4f}")

        # ── Save cache ─────────────────────────────────────────────────
        if args.cache_path:
            os.makedirs(os.path.dirname(args.cache_path) or ".", exist_ok=True)
            torch.save({
                "obs": obs.cpu(), "actions": actions.cpu(),
                "mc_q": mc_q.cpu(), "mc_v": mc_v.cpu(),
                "mc_trajectories": mc_trajectories,
                "reward_scale": args.reward_scale,
            }, args.cache_path)
            print(f"  Cached MC data to {args.cache_path}")

        # Free MC envs and optimal agent
        del optimal_agent, saved_states
        mc_envs.close()
        envs.close()
        del mc_envs, envs
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════
    #  Step 3: Prepare IQL training data
    # ══════════════════════════════════════════════════════════════════
    print(f"\nStep 3: Preparing IQL training data (mode={args.iql_data_mode})...")

    if args.iql_data_mode == "mc_only":
        trajectories = mc_trajectories
    elif args.iql_data_mode == "mc+offline":
        print("  Collecting offline data...")
        offline_t0 = time.time()
        offline_trajectories = collect_offline_data(
            args.iql_data_checkpoints, args.iql_episodes_per_ckpt,
            args.env_id, args.iql_offline_num_envs, device, env_kwargs,
            args.reward_scale,
        )
        offline_time = time.time() - offline_t0
        n_offline = sum(t["states"].shape[0] for t in offline_trajectories)
        print(f"  Offline: {len(offline_trajectories)} trajectories, "
              f"{n_offline} transitions ({offline_time:.1f}s)")
        trajectories = mc_trajectories + offline_trajectories
        del offline_trajectories
    elif args.iql_data_mode == "offline_only":
        print("  Collecting offline data...")
        offline_t0 = time.time()
        trajectories = collect_offline_data(
            args.iql_data_checkpoints, args.iql_episodes_per_ckpt,
            args.env_id, args.iql_offline_num_envs, device, env_kwargs,
            args.reward_scale,
        )
        offline_time = time.time() - offline_t0
        n_offline = sum(t["states"].shape[0] for t in trajectories)
        print(f"  Offline: {len(trajectories)} trajectories, "
              f"{n_offline} transitions ({offline_time:.1f}s)")
    else:
        raise ValueError(f"Unknown iql_data_mode: {args.iql_data_mode}")

    # Subsample trajectories if over limit
    if args.iql_max_transitions > 0:
        total = sum(t["states"].shape[0] for t in trajectories)
        if total > args.iql_max_transitions:
            import random
            random.shuffle(trajectories)
            kept, count = [], 0
            for t in trajectories:
                kept.append(t)
                count += t["states"].shape[0]
                if count >= args.iql_max_transitions:
                    break
            trajectories = kept
            print(f"  Subsampled: {total} -> {count} transitions "
                  f"({len(kept)} trajectories)")

    # Flatten trajectories
    flat_states = torch.cat([t["states"] for t in trajectories])
    flat_actions = torch.cat([t["actions"] for t in trajectories])
    flat_rewards = torch.cat([t["rewards"] for t in trajectories])
    flat_next_states = torch.cat([t["next_states"] for t in trajectories])
    flat_dones = torch.cat([t["dones"] for t in trajectories])

    total_transitions = flat_states.shape[0]
    traj_lens = [t["states"].shape[0] for t in trajectories]
    positive_rewards = (flat_rewards > 0).sum().item()

    print(f"  Total: {total_transitions} transitions, {len(trajectories)} trajectories")
    print(f"  Traj lengths: min={min(traj_lens)}, max={max(traj_lens)}, "
          f"mean={sum(traj_lens)/len(traj_lens):.1f}")
    print(f"  Rewards: {positive_rewards} positive / {total_transitions}")

    # ══════════════════════════════════════════════════════════════════
    #  Step 4: Train IQL
    # ══════════════════════════════════════════════════════════════════
    print(f"\nStep 4: Training IQL (tau={args.iql_expectile_tau}, "
          f"epochs={args.iql_epochs})...")
    iql_train_t0 = time.time()

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

    nstep_kw = {}
    if args.iql_nstep > 1:
        print(f"  Computing {args.iql_nstep}-step TD targets...")
        nret, boot_s, ndisc = compute_nstep_targets(
            trajectories, args.iql_nstep, args.gamma
        )
        nstep_kw = dict(
            nstep_returns=nret, bootstrap_states=boot_s, nstep_discounts=ndisc
        )

    q_net, v_net = train_iql(
        flat_states, flat_actions, flat_rewards, flat_next_states, flat_dones,
        device, iql_args, **nstep_kw,
    )

    iql_train_time = time.time() - iql_train_t0
    print(f"  IQL training time: {iql_train_time:.1f}s")

    # Free training data
    del trajectories, flat_states, flat_actions, flat_rewards
    del flat_next_states, flat_dones, nstep_kw, mc_trajectories
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════
    #  Step 5: IQL predictions on rollout data
    # ══════════════════════════════════════════════════════════════════
    print("\nStep 5: Computing IQL predictions...")

    with torch.no_grad():
        iql_q_list = []
        iql_v_list = []
        for t in range(args.num_steps):
            s = obs[t]  # (num_envs, obs_dim)
            a = actions[t]  # (num_envs, act_dim)
            iql_q_list.append(q_net(s, a).squeeze(-1))
            iql_v_list.append(v_net(s).squeeze(-1))
        iql_q = torch.stack(iql_q_list)  # (num_steps, num_envs)
        iql_v = torch.stack(iql_v_list)
        iql_adv = iql_q - iql_v

    print(f"  IQL Q: mean={iql_q.mean():.4f}, std={iql_q.std():.4f}")
    print(f"  IQL V: mean={iql_v.mean():.4f}, std={iql_v.std():.4f}")
    print(f"  IQL A: mean={iql_adv.mean():.4f}, std={iql_adv.std():.4f}")

    # ══════════════════════════════════════════════════════════════════
    #  Step 6: Save raw data
    # ══════════════════════════════════════════════════════════════════
    if args.save_data:
        data_path = args.output.replace(".png", ".pt")
        torch.save({
            "mc_q": mc_q.cpu(), "mc_v": mc_v.cpu(), "mc_adv": mc_adv.cpu(),
            "iql_q": iql_q.cpu(), "iql_v": iql_v.cpu(), "iql_adv": iql_adv.cpu(),
            "obs": obs.cpu(), "actions": actions.cpu(),
            "args": vars(args),
        }, data_path)
        print(f"\n  Saved raw data to {data_path}")

    # ══════════════════════════════════════════════════════════════════
    #  Step 7: Scatter plots
    # ══════════════════════════════════════════════════════════════════
    print("\nStep 7: Generating scatter plots...")

    mc_q_flat = mc_q.cpu().flatten().numpy()
    mc_v_flat = mc_v.cpu().flatten().numpy()
    mc_adv_flat = mc_adv.cpu().flatten().numpy()
    iql_q_flat = iql_q.cpu().flatten().numpy()
    iql_v_flat = iql_v.cpu().flatten().numpy()
    iql_adv_flat = iql_adv.cpu().flatten().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(21, 14))

    # ── Row 0: Raw value scatter plots ─────────────────────────────────

    # (0,0): Q scatter
    ax = axes[0, 0]
    ax.scatter(mc_q_flat, iql_q_flat, alpha=0.1, s=4, edgecolors="none")
    lo, hi = mc_q_flat.min(), mc_q_flat.max()
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    r_q = pearsonr(mc_q_flat, iql_q_flat)[0]
    rho_q = spearmanr(mc_q_flat, iql_q_flat)[0]
    ax.set_xlabel(r"MC $Q^{\pi^*}(s,a)$")
    ax.set_ylabel("IQL Q(s,a)")
    ax.set_title(f"Q(s,a): r={r_q:.4f}, rho={rho_q:.4f}")
    ax.grid(True, alpha=0.3)

    # (0,1): V scatter
    ax = axes[0, 1]
    ax.scatter(mc_v_flat, iql_v_flat, alpha=0.1, s=4, edgecolors="none")
    lo, hi = mc_v_flat.min(), mc_v_flat.max()
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    r_v = pearsonr(mc_v_flat, iql_v_flat)[0]
    rho_v = spearmanr(mc_v_flat, iql_v_flat)[0]
    ax.set_xlabel(r"MC $V^{\pi^*}(s)$")
    ax.set_ylabel("IQL V(s)")
    ax.set_title(f"V(s): r={r_v:.4f}, rho={rho_v:.4f}")
    ax.grid(True, alpha=0.3)

    # (0,2): Advantage scatter
    ax = axes[0, 2]
    ax.scatter(mc_adv_flat, iql_adv_flat, alpha=0.1, s=4, edgecolors="none")
    lo, hi = mc_adv_flat.min(), mc_adv_flat.max()
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    r_a = pearsonr(mc_adv_flat, iql_adv_flat)[0]
    rho_a = spearmanr(mc_adv_flat, iql_adv_flat)[0]
    ax.set_xlabel(r"MC $A^{\pi^*}(s,a)$")
    ax.set_ylabel("IQL A(s,a)")
    ax.set_title(f"A(s,a): r={r_a:.4f}, rho={rho_a:.4f}")
    ax.grid(True, alpha=0.3)

    # ── Row 1: Analysis ────────────────────────────────────────────────

    # (1,0): Per-timestep Pearson r
    ax = axes[1, 0]
    per_t_r_q, per_t_r_v, per_t_r_a = [], [], []
    per_t_rho_a = []
    for t in range(args.num_steps):
        mq = mc_q[t].cpu().numpy()
        mv = mc_v[t].cpu().numpy()
        ma = mc_adv[t].cpu().numpy()
        iq = iql_q[t].cpu().numpy()
        iv = iql_v[t].cpu().numpy()
        ia = iql_adv[t].cpu().numpy()
        per_t_r_q.append(pearsonr(mq, iq)[0])
        per_t_r_v.append(pearsonr(mv, iv)[0])
        per_t_r_a.append(pearsonr(ma, ia)[0])
        per_t_rho_a.append(spearmanr(ma, ia)[0])
    ax.plot(per_t_r_q, label="Q (Pearson)", marker=".", markersize=4)
    ax.plot(per_t_r_v, label="V (Pearson)", marker=".", markersize=4)
    ax.plot(per_t_r_a, label="A (Pearson)", marker=".", markersize=4)
    ax.plot(per_t_rho_a, label="A (Spearman)", marker="x", markersize=4, linestyle="--")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Correlation")
    ax.set_title("Per-timestep: IQL vs MC")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1): Advantage residual histogram
    ax = axes[1, 1]
    residuals = iql_adv_flat - mc_adv_flat
    ax.hist(residuals, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="r", ls="--", lw=1)
    ax.set_xlabel("IQL A - MC A")
    ax.set_ylabel("Count")
    ax.set_title(f"Advantage residual: mean={residuals.mean():.4f}, std={residuals.std():.4f}")
    ax.grid(True, alpha=0.3)

    # (1,2): Summary table
    ax = axes[1, 2]
    ax.axis("off")

    rmse_q = np.sqrt(((mc_q_flat - iql_q_flat) ** 2).mean())
    rmse_v = np.sqrt(((mc_v_flat - iql_v_flat) ** 2).mean())
    rmse_a = np.sqrt(((mc_adv_flat - iql_adv_flat) ** 2).mean())

    rows = [
        ["Metric", "Q(s,a)", "V(s)", "A(s,a)"],
        ["Pearson r", f"{r_q:.4f}", f"{r_v:.4f}", f"{r_a:.4f}"],
        ["Spearman rho", f"{rho_q:.4f}", f"{rho_v:.4f}", f"{rho_a:.4f}"],
        ["MC mean", f"{mc_q_flat.mean():.4f}", f"{mc_v_flat.mean():.4f}", f"{mc_adv_flat.mean():.4f}"],
        ["IQL mean", f"{iql_q_flat.mean():.4f}", f"{iql_v_flat.mean():.4f}", f"{iql_adv_flat.mean():.4f}"],
        ["MC std", f"{mc_q_flat.std():.4f}", f"{mc_v_flat.std():.4f}", f"{mc_adv_flat.std():.4f}"],
        ["IQL std", f"{iql_q_flat.std():.4f}", f"{iql_v_flat.std():.4f}", f"{iql_adv_flat.std():.4f}"],
        ["RMSE", f"{rmse_q:.4f}", f"{rmse_v:.4f}", f"{rmse_a:.4f}"],
        ["", "", "", ""],
        ["IQL data", f"{args.iql_data_mode}", "", ""],
        ["IQL tau", f"{args.iql_expectile_tau}", "", ""],
        ["MC samples", f"{args.mc_samples}", "", ""],
        ["gamma", f"{args.gamma}", "", ""],
    ]
    table = ax.table(cellText=rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    # Bold header row
    for j in range(4):
        table[0, j].set_text_props(fontweight="bold")
    ax.set_title("Summary", fontsize=11, pad=10)

    fig.suptitle(
        f"IQL Fitting Quality on MC{args.mc_samples} Data "
        f"(data={args.iql_data_mode}, tau={args.iql_expectile_tau})",
        fontsize=13,
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\n  Saved figure to {args.output}")
    plt.close(fig)

    # ── Print summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  IQL Fit Quality Summary (data={args.iql_data_mode}, tau={args.iql_expectile_tau})")
    print(f"{'='*60}")
    print(f"  Q(s,a):  Pearson r={r_q:.4f}, Spearman rho={rho_q:.4f}, RMSE={rmse_q:.4f}")
    print(f"  V(s):    Pearson r={r_v:.4f}, Spearman rho={rho_v:.4f}, RMSE={rmse_v:.4f}")
    print(f"  A(s,a):  Pearson r={r_a:.4f}, Spearman rho={rho_a:.4f}, RMSE={rmse_a:.4f}")
    print(f"{'='*60}")

    # Cleanup
    del q_net, v_net
