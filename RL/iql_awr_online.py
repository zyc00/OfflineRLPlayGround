"""Online Iterative IQL + AWR Finetuning.

Combines offline IQL training (Phase 0) with an online iterative loop:
each iteration rolls out new data → computes IQL advantages → AWR update →
appends transitions to buffer → finetunes IQL periodically.

No optimal policy or MC re-rollout needed — IQL advantage is just Q(s,a) - V(s).

Phase 0: collect_iql_data() + train_iql() (from iql_awr_offline.py)
Online loop (from mc_finetune_awr_parallel.py structure):
  for iteration 1..N:
    1. Eval (deterministic)
    2. Rollout (collect obs, actions, rewards, next_obs, dones)
    3. IQL advantages = Q(s,a) - V(s) forward pass
    4. AWR update (weighted NLL)
    5. Append rollout transitions to IQL buffer
    6. Finetune IQL on accumulated buffer (every iql_finetune_freq iters)

Usage:
  # Quick test
  python -m RL.iql_awr_online --num_envs 16 --num_steps 50 --total_timesteps 5000 \
      --iql_epochs 20 --update_epochs 10 --eval_freq 1

  # Full run (v2 settings)
  python -m RL.iql_awr_online --num_envs 128 --num_steps 200 \
      --num_minibatches 32 --update_epochs 4 --total_timesteps 50000
"""

import copy
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
from methods.iql.iql import QNetwork, train_iql, compute_nstep_targets, expectile_loss
from methods.gae.gae import Critic


@dataclass
class Args:
    # IQL data collection (Phase 0)
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

    # IQL training hyperparameters (Phase 0)
    iql_expectile_tau: float = 0.7
    """IQL expectile tau (higher = closer to max Q)"""
    iql_epochs: int = 200
    """number of IQL training epochs (Phase 0)"""
    iql_lr: float = 3e-4
    """IQL learning rate (Phase 0)"""
    iql_batch_size: int = 256
    """IQL minibatch size"""
    iql_nstep: int = 1
    """n-step TD return for IQL (1=standard, >1=multi-step)"""
    iql_patience: int = 50
    """early stopping patience for IQL (Phase 0 only)"""

    # IQL online finetuning hyperparameters
    iql_finetune_epochs: int = 10
    """epochs per IQL finetune step"""
    iql_finetune_freq: int = 1
    """finetune IQL every N iterations"""
    iql_finetune_lr: float = 1e-4
    """lower lr for online IQL finetuning"""
    iql_tau_polyak: float = 0.005
    """target Q Polyak averaging rate"""

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

    # Advantage estimation
    use_gae: bool = False
    """use GAE with IQL's V instead of Q-V advantages"""
    gae_lambda: float = 0.95
    """GAE lambda (only used when use_gae=True)"""
    v_update_epochs: int = 4
    """epochs to fit V on rollout returns each iteration (only when use_gae=True)"""
    v_lr: float = 3e-4
    """V network learning rate for online fitting (only when use_gae=True)"""

    # AWR hyperparameters (actor only)
    gamma: float = 0.8
    learning_rate: float = 3e-4
    num_steps: int = 50
    """rollout length per iteration"""
    num_minibatches: int = 32
    update_epochs: int = 200
    """AWR tolerates high UTD"""
    awr_beta: float = 0.5
    """AWR temperature. Lower = more greedy, higher = closer to uniform"""
    awr_max_weight: float = 20.0
    """AWR weight clamp upper bound to prevent exp explosion"""
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    reward_scale: float = 1.0

    # Training
    total_timesteps: int = 2_000_000
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
    num_iterations: int = 0


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


def finetune_iql(q_net, q_target, v_net, q_optimizer, v_optimizer,
                 buf_states, buf_actions, buf_rewards, buf_next_states, buf_dones,
                 device, gamma, expectile_tau, tau_polyak, grad_clip,
                 batch_size, epochs, patience=20):
    """Finetune IQL Q and V networks on accumulated buffer data.

    Same losses as train_iql: Q MSE with V bootstrap, V expectile regression,
    Polyak target update. Includes train/val split, cosine LR scheduler, and
    early stopping to prevent V from collapsing toward Q.

    Returns dict of average losses for logging.
    """
    N = buf_states.shape[0]

    # Train/val split (same as train_iql)
    perm = torch.randperm(N)
    val_size = max(1, int(N * 0.1))
    train_idx = perm[val_size:]
    val_idx = perm[:val_size]
    train_size = train_idx.shape[0]

    # Val data on device
    val_s = buf_states[val_idx].to(device)
    val_a = buf_actions[val_idx].to(device)
    val_r = buf_rewards[val_idx].to(device)
    val_ns = buf_next_states[val_idx].to(device)
    val_term = buf_dones[val_idx].to(device)

    # Cosine LR schedulers (same as train_iql)
    q_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        q_optimizer, T_max=epochs, eta_min=1e-5
    )
    v_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        v_optimizer, T_max=epochs, eta_min=1e-5
    )

    # Early stopping state
    best_val_loss = float("inf")
    best_q_state = None
    best_v_state = None
    epochs_no_improve = 0

    q_net.train()
    v_net.train()

    last_q_loss = 0.0
    last_v_loss = 0.0

    for epoch in range(epochs):
        indices = train_idx[torch.randperm(train_size)]
        epoch_q_loss = 0.0
        epoch_v_loss = 0.0
        num_batches = 0

        for start in range(0, train_size, batch_size):
            batch_idx = indices[start : start + batch_size]
            s = buf_states[batch_idx].to(device)
            a = buf_actions[batch_idx].to(device)
            r = buf_rewards[batch_idx].to(device)
            ns = buf_next_states[batch_idx].to(device)
            term = buf_dones[batch_idx].to(device)

            # Q loss: MSE with V bootstrap
            with torch.no_grad():
                v_next = v_net(ns).squeeze(-1)
                q_target_val = r + gamma * v_next * (1.0 - term)
            q_pred = q_net(s, a).squeeze(-1)
            q_loss = 0.5 * ((q_pred - q_target_val) ** 2).mean()

            q_optimizer.zero_grad()
            q_loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), grad_clip)
            q_optimizer.step()

            # V loss: expectile regression against target Q
            with torch.no_grad():
                q_val = q_target(s, a).squeeze(-1)
            v_pred = v_net(s).squeeze(-1)
            v_loss = expectile_loss(q_val - v_pred, expectile_tau)

            v_optimizer.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(v_net.parameters(), grad_clip)
            v_optimizer.step()

            # Polyak update target Q
            with torch.no_grad():
                for p, p_targ in zip(q_net.parameters(), q_target.parameters()):
                    p_targ.data.mul_(1.0 - tau_polyak).add_(
                        p.data, alpha=tau_polyak
                    )

            epoch_q_loss += q_loss.item()
            epoch_v_loss += v_loss.item()
            num_batches += 1

        last_q_loss = epoch_q_loss / num_batches
        last_v_loss = epoch_v_loss / num_batches
        q_scheduler.step()
        v_scheduler.step()

        # Validation
        q_net.eval()
        v_net.eval()
        with torch.no_grad():
            v_next_val = v_net(val_ns).squeeze(-1)
            q_tgt = val_r + gamma * v_next_val * (1.0 - val_term)
            q_pred_val = q_net(val_s, val_a).squeeze(-1)
            val_q_loss = 0.5 * ((q_pred_val - q_tgt) ** 2).mean().item()

            q_val_for_v = q_target(val_s, val_a).squeeze(-1)
            v_pred_val = v_net(val_s).squeeze(-1)
            diff = q_val_for_v - v_pred_val
            weight = torch.where(diff > 0, expectile_tau, 1.0 - expectile_tau)
            val_v_loss = (weight * (diff**2)).mean().item()

        val_total = val_q_loss + val_v_loss
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_q_state = {k: v.clone() for k, v in q_net.state_dict().items()}
            best_v_state = {k: v.clone() for k, v in v_net.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

        q_net.train()
        v_net.train()

    # Restore best checkpoint
    if best_q_state is not None:
        q_net.load_state_dict(best_q_state)
    if best_v_state is not None:
        v_net.load_state_dict(best_v_state)
    # Also sync q_target to best q_net
    with torch.no_grad():
        for p, p_targ in zip(q_net.parameters(), q_target.parameters()):
            p_targ.data.copy_(p.data)

    q_net.eval()
    v_net.eval()

    return {
        "q_loss": last_q_loss,
        "v_loss": last_v_loss,
    }


if __name__ == "__main__":
    args = tyro.cli(Args)

    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    if args.exp_name is None:
        args.exp_name = "iql_awr_online"
    run_name = f"{args.exp_name}__seed{args.seed}__{int(time.time())}"

    print(f"=== Online Iterative IQL + AWR Finetuning ===")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  IQL data: {len(args.iql_data_checkpoints)} checkpoints, "
          f"{args.iql_episodes_per_ckpt} episodes each")
    print(f"  IQL Phase 0: tau={args.iql_expectile_tau}, epochs={args.iql_epochs}, "
          f"nstep={args.iql_nstep}")
    print(f"  IQL finetune: lr={args.iql_finetune_lr}, epochs={args.iql_finetune_epochs}, "
          f"freq={args.iql_finetune_freq}")
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
    #  Phase 0a: Collect offline data from multiple checkpoints
    # ══════════════════════════════════════════════════════════════════
    print("\nPhase 0a: Collecting IQL training data...")
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
    #  Phase 0b: Train IQL (Q and V networks)
    # ══════════════════════════════════════════════════════════════════
    print(f"\nPhase 0b: Training IQL (tau={args.iql_expectile_tau}, "
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

    if args.use_gae:
        # GAE mode: only keep V, discard Q
        del q_net
        v_optimizer_online = optim.Adam(v_net.parameters(), lr=args.v_lr, eps=1e-5)
        print(f"  GAE mode: using IQL V for bootstrapping, discarding Q")
        print(f"  V params: {sum(p.numel() for p in v_net.parameters()):,}")

        # Free Phase 0 flat data (not needed for GAE)
        del flat_states, flat_actions, flat_rewards, flat_next_states, flat_dones
    else:
        # IQL Q-V mode: keep Q, V, buffer
        q_target = copy.deepcopy(q_net)

        # Initialize IQL buffer with Phase 0 data (on CPU)
        buf_states = flat_states
        buf_actions = flat_actions
        buf_rewards = flat_rewards
        buf_next_states = flat_next_states
        buf_dones = flat_dones

    # Free trajectory structures
    del trajectories, nstep_kw
    torch.cuda.empty_cache()

    writer.add_scalar("time/iql_data_collection", collect_time, 0)
    writer.add_scalar("time/iql_training", iql_train_time, 0)

    # ══════════════════════════════════════════════════════════════════
    #  Online Training Loop
    # ══════════════════════════════════════════════════════════════════
    print(f"\nStarting online loop: {args.num_iterations} iterations\n")

    # Rollout storage (on GPU)
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
    next_obs_buf = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape,
        device=device,
    )

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    for iteration in range(1, args.num_iterations + 1):
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

            buf_info = f" | buf={buf_states.shape[0]}" if not args.use_gae else ""
            print(
                f"Iter {iteration}/{args.num_iterations} | "
                f"step={global_step} | SR={success_rate:.1%} | "
                f"episodes={num_episodes}{buf_info}"
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

        # ── Rollout ────────────────────────────────────────────────
        rollout_t0 = time.time()
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action = agent.get_action(next_obs, deterministic=False)
            actions[step] = action

            next_obs, reward, terminations, truncations, infos = envs.step(
                clip_action(action)
            )
            done_flag = (terminations | truncations).float()
            rewards[step] = reward.view(-1) * args.reward_scale

            # Store true next observation
            next_obs_buf[step] = next_obs.clone()
            dones[step] = done_flag

            # For done envs, next_obs is already the reset obs.
            # Use final_observation as the true next state before reset.
            if "final_info" in infos:
                done_mask = infos["_final_info"]
                next_obs_buf[step, done_mask] = infos["final_observation"][done_mask]
                for k, v in infos["final_info"]["episode"].items():
                    writer.add_scalar(
                        f"train/{k}", v[done_mask].float().mean(), global_step
                    )

            next_done = done_flag

        rollout_time = time.time() - rollout_t0

        # ── Flatten batch ──────────────────────────────────────────
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)

        if args.use_gae:
            # ── GAE advantages using IQL's V ───────────────────────
            with torch.no_grad():
                # V(s) for all rollout states: (num_steps, num_envs)
                values = torch.zeros((args.num_steps, args.num_envs), device=device)
                for start in range(0, args.num_steps, 32):
                    end = min(start + 32, args.num_steps)
                    values[start:end] = v_net(obs[start:end].reshape(-1, obs.shape[-1])).squeeze(-1).reshape(end - start, args.num_envs)

                # V(s_{T}) for bootstrap (next_obs after last step)
                next_value = v_net(next_obs).squeeze(-1)

                # GAE computation (reverse scan)
                advantages = torch.zeros_like(rewards)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam

                returns = advantages + values

            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # ── Fit V on rollout returns ───────────────────────────
            v_net.train()
            v_fit_t0 = time.time()
            v_inds = np.arange(args.batch_size)
            for v_epoch in range(args.v_update_epochs):
                np.random.shuffle(v_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = v_inds[start:end]
                    v_pred = v_net(b_obs[mb_inds]).squeeze(-1)
                    v_loss = 0.5 * ((v_pred - b_returns[mb_inds]) ** 2).mean()
                    v_optimizer_online.zero_grad()
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(v_net.parameters(), args.max_grad_norm)
                    v_optimizer_online.step()
            v_net.eval()
            v_fit_time = time.time() - v_fit_t0

            writer.add_scalar("losses/v_loss", v_loss.item(), global_step)
            writer.add_scalar("time/v_fit", v_fit_time, global_step)
        else:
            # ── IQL Q-V mode: buffer + finetune + advantages ───────
            # Append rollout transitions to IQL buffer (CPU)
            roll_states = obs.reshape((-1,) + envs.single_observation_space.shape).cpu()
            roll_actions = actions.reshape((-1,) + envs.single_action_space.shape).cpu()
            roll_rewards = rewards.reshape(-1).cpu()
            roll_next_states = next_obs_buf.reshape((-1,) + envs.single_observation_space.shape).cpu()
            roll_dones = dones.reshape(-1).cpu()

            buf_states = torch.cat([buf_states, roll_states])
            buf_actions = torch.cat([buf_actions, roll_actions])
            buf_rewards = torch.cat([buf_rewards, roll_rewards])
            buf_next_states = torch.cat([buf_next_states, roll_next_states])
            buf_dones = torch.cat([buf_dones, roll_dones])

            # Finetune IQL on accumulated buffer
            iql_metrics = None
            if iteration % args.iql_finetune_freq == 0:
                iql_ft_t0 = time.time()
                q_optimizer = torch.optim.Adam(
                    q_net.parameters(), lr=args.iql_finetune_lr, eps=1e-5, weight_decay=1e-4
                )
                v_optimizer = torch.optim.Adam(
                    v_net.parameters(), lr=args.iql_finetune_lr, eps=1e-5, weight_decay=1e-4
                )
                iql_metrics = finetune_iql(
                    q_net, q_target, v_net, q_optimizer, v_optimizer,
                    buf_states, buf_actions, buf_rewards, buf_next_states, buf_dones,
                    device, args.gamma, args.iql_expectile_tau, args.iql_tau_polyak,
                    args.max_grad_norm, args.iql_batch_size, args.iql_finetune_epochs,
                )
                iql_ft_time = time.time() - iql_ft_t0
                writer.add_scalar("iql/q_loss", iql_metrics["q_loss"], global_step)
                writer.add_scalar("iql/v_loss", iql_metrics["v_loss"], global_step)
                writer.add_scalar("time/iql_finetune", iql_ft_time, global_step)

            writer.add_scalar("iql/buffer_size", buf_states.shape[0], global_step)

            # Compute IQL advantages: A(s,a) = Q(s,a) - V(s)
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

        adv_mean = b_advantages.mean().item()
        adv_std = b_advantages.std().item()
        adv_pos_frac = (b_advantages > 0).float().mean().item()

        writer.add_scalar("charts/advantage_mean", adv_mean, global_step)
        writer.add_scalar("charts/advantage_std", adv_std, global_step)
        writer.add_scalar("charts/advantage_pos_frac", adv_pos_frac, global_step)

        # ── AWR update (actor only) ────────────────────────────────
        agent.train()
        b_inds = np.arange(args.batch_size)
        update_t0 = time.time()

        # Precompute AWR weights
        if args.norm_adv:
            b_advantages_norm = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        else:
            b_advantages_norm = b_advantages

        b_weights = torch.exp(b_advantages_norm / args.awr_beta)
        b_weights = torch.clamp(b_weights, max=args.awr_max_weight)

        awr_weight_mean = b_weights.mean().item()
        awr_weight_max = b_weights.max().item()

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
        writer.add_scalar("losses/awr_weight_mean", awr_weight_mean, global_step)
        writer.add_scalar("losses/awr_weight_max", awr_weight_max, global_step)
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("time/rollout", rollout_time, global_step)
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
