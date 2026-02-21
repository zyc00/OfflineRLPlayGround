#!/usr/bin/env python
"""Test: Can IQL-SARSA (tau=0.5) on on-policy data learn V^π accurately?

IQL with tau=0.5 = SARSA:
  Q(s,a) → r + γ V(s')          (standard Bellman)
  V(s)   → E_a~data[Q(s,a)]     (tau=0.5 = MSE = expectation)

On on-policy data, E_a~data = E_a~π, so V → V^π, Q → Q^π.

Steps:
  1. Rollout π_k, save env states, collect (s, a, r, s', done)
  2. MC16 re-rollout → V^π_k, Q^π_k ground truth
  3. Train IQL (tau=0.5) on rollout transitions
  4. Compare IQL V, Q vs MC16 ground truth

Supports --load_data to skip Steps 1-2 and reuse cached data.
"""

import os
import random
import time
from dataclasses import dataclass
from types import SimpleNamespace

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import copy

from data.data_collection.ppo import Agent
from methods.iql.iql import train_iql
from methods.gae.gae import Critic


def train_td0_v(states, rewards, next_states, terminated, device, args):
    """Direct TD(0) for V only: V(s) → r + γ(1-d)V(s'). No Q network."""

    state_dim = states.shape[1]
    v_net = Critic("state", state_dim=state_dim).to(device)
    v_target = copy.deepcopy(v_net)

    optimizer = torch.optim.Adam(
        v_net.parameters(), lr=args.lr, eps=1e-5, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    N = states.shape[0]
    perm = torch.randperm(N)
    val_size = max(1, int(N * 0.1))
    train_idx = perm[val_size:]
    val_idx = perm[:val_size]
    train_size = train_idx.shape[0]

    val_s = states[val_idx].to(device)
    val_r = rewards[val_idx].to(device)
    val_ns = next_states[val_idx].to(device)
    val_term = terminated[val_idx].to(device)

    for epoch in range(args.epochs):
        v_net.train()
        indices = train_idx[torch.randperm(train_size)]
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, train_size, args.batch_size):
            batch_idx = indices[start : start + args.batch_size]
            s = states[batch_idx].to(device)
            r = rewards[batch_idx].to(device)
            ns = next_states[batch_idx].to(device)
            term = terminated[batch_idx].to(device)

            with torch.no_grad():
                v_next = v_target(ns).squeeze(-1)
                td_target = r + args.gamma * v_next * (1.0 - term)

            v_pred = v_net(s).squeeze(-1)
            loss = 0.5 * ((v_pred - td_target) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(v_net.parameters(), args.grad_clip)
            optimizer.step()

            # Polyak update target
            with torch.no_grad():
                for p, pt in zip(v_net.parameters(), v_target.parameters()):
                    pt.data.mul_(1.0 - args.tau_polyak).add_(
                        p.data, alpha=args.tau_polyak
                    )

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            v_net.eval()
            with torch.no_grad():
                v_next_val = v_target(val_ns).squeeze(-1)
                val_tgt = val_r + args.gamma * v_next_val * (1.0 - val_term)
                val_pred = v_net(val_s).squeeze(-1)
                val_loss = 0.5 * ((val_pred - val_tgt) ** 2).mean().item()
            print(
                f"  Epoch {epoch + 1}/{args.epochs}: "
                f"loss={epoch_loss / num_batches:.6f}, val_loss={val_loss:.6f}"
            )

    v_net.eval()
    with torch.no_grad():
        all_v = []
        for start in range(0, N, args.batch_size):
            end = min(start + args.batch_size, N)
            s = states[start:end].to(device)
            all_v.append(v_net(s).squeeze(-1).cpu())
        all_v = torch.cat(all_v)
        print(f"  V(s): mean={all_v.mean():.4f}, std={all_v.std():.4f}")

    return v_net


def precompute_nstep_targets(rewards_2d, dones_2d, nstep, gamma):
    """Precompute n-step reward sums and bootstrap info.

    Args:
        rewards_2d: (T, E) rewards
        dones_2d: (T, E) done flags
        nstep: number of steps
        gamma: discount factor

    Returns:
        nstep_rewards: (T, E) discounted reward sums
        bootstrap_t: (T, E) timestep to bootstrap from (-1 if no bootstrap)
        bootstrap_coeff: (T, E) discount coefficient for bootstrap
    """
    T, E = rewards_2d.shape
    nstep_rewards = torch.zeros(T, E)
    bootstrap_t = torch.full((T, E), -1, dtype=torch.long)
    bootstrap_coeff = torch.zeros(T, E)

    for t in range(T):
        G = torch.zeros(E)
        gamma_k = torch.ones(E)
        alive = torch.ones(E, dtype=torch.bool)

        for k in range(nstep):
            step = t + k
            if step >= T:
                alive[:] = False
                break
            G += gamma_k * rewards_2d[step] * alive.float()
            gamma_k *= gamma
            just_done = dones_2d[step].bool() & alive
            alive = alive & ~just_done

        nstep_rewards[t] = G
        bs = t + nstep
        if bs < T:
            bootstrap_t[t, alive] = bs
            bootstrap_coeff[t, alive] = gamma_k[alive]

    return nstep_rewards, bootstrap_t, bootstrap_coeff


def train_nstep_td_v(obs_buf_2d, rewards_2d, dones_2d, nstep, device, args):
    """N-step TD for V: V(s_t) -> R_n + gamma^n V(s_{t+n})."""
    T, E, obs_dim = obs_buf_2d.shape

    print(f"  Precomputing {nstep}-step targets...")
    nstep_rewards, bootstrap_t, bootstrap_coeff = precompute_nstep_targets(
        rewards_2d, dones_2d, nstep, args.gamma,
    )

    v_net = Critic("state", state_dim=obs_dim).to(device)
    v_target = copy.deepcopy(v_net)
    optimizer = torch.optim.Adam(
        v_net.parameters(), lr=args.lr, eps=1e-5, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5,
    )

    # Flatten for batched training
    N = T * E
    flat_obs = obs_buf_2d.reshape(N, obs_dim)
    flat_nstep_rewards = nstep_rewards.reshape(N)
    flat_bootstrap_t = bootstrap_t.reshape(N)
    flat_bootstrap_coeff = bootstrap_coeff.reshape(N)
    env_indices = torch.arange(E).unsqueeze(0).expand(T, E).reshape(N)
    flat_bootstrap_idx = flat_bootstrap_t * E + env_indices
    has_bootstrap = flat_bootstrap_t >= 0

    # Train/val split
    perm = torch.randperm(N)
    val_size = max(1, int(N * 0.1))
    train_idx = perm[val_size:]
    val_idx = perm[:val_size]
    train_size = train_idx.shape[0]

    for epoch in range(args.epochs):
        v_net.train()
        indices = train_idx[torch.randperm(train_size)]
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, train_size, args.batch_size):
            batch_idx = indices[start : start + args.batch_size]
            s = flat_obs[batch_idx].to(device)
            r_n = flat_nstep_rewards[batch_idx].to(device)
            b_coeff = flat_bootstrap_coeff[batch_idx].to(device)
            b_idx = flat_bootstrap_idx[batch_idx]
            b_mask = has_bootstrap[batch_idx]

            with torch.no_grad():
                bootstrap_v = torch.zeros(batch_idx.shape[0], device=device)
                if b_mask.any():
                    bs_obs = flat_obs[b_idx[b_mask]].to(device)
                    bootstrap_v[b_mask] = v_target(bs_obs).squeeze(-1)
                td_target = r_n + b_coeff * bootstrap_v

            v_pred = v_net(s).squeeze(-1)
            loss = 0.5 * ((v_pred - td_target) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(v_net.parameters(), args.grad_clip)
            optimizer.step()

            with torch.no_grad():
                for p, pt in zip(v_net.parameters(), v_target.parameters()):
                    pt.data.mul_(1.0 - args.tau_polyak).add_(
                        p.data, alpha=args.tau_polyak,
                    )

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1}/{args.epochs}: "
                f"loss={epoch_loss / num_batches:.6f}"
            )

    v_net.eval()
    with torch.no_grad():
        all_v = []
        for start in range(0, N, args.batch_size):
            end = min(start + args.batch_size, N)
            s = flat_obs[start:end].to(device)
            all_v.append(v_net(s).squeeze(-1).cpu())
        all_v = torch.cat(all_v)
        print(f"  V(s): mean={all_v.mean():.4f}, std={all_v.std():.4f}")

    return v_net


def train_mc_v(flat_states, mc_v_flat, device, args):
    """Supervised regression: V(s) -> MC V(s). No bootstrapping."""
    N = flat_states.shape[0]
    state_dim = flat_states.shape[1]

    v_net = Critic("state", state_dim=state_dim).to(device)
    optimizer = torch.optim.Adam(
        v_net.parameters(), lr=args.lr, eps=1e-5, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5,
    )

    perm = torch.randperm(N)
    val_size = max(1, int(N * 0.1))
    train_idx = perm[val_size:]
    val_idx = perm[:val_size]
    train_size = train_idx.shape[0]

    val_s = flat_states[val_idx].to(device)
    val_v = mc_v_flat[val_idx].to(device)

    for epoch in range(args.epochs):
        v_net.train()
        indices = train_idx[torch.randperm(train_size)]
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, train_size, args.batch_size):
            batch_idx = indices[start : start + args.batch_size]
            s = flat_states[batch_idx].to(device)
            target = mc_v_flat[batch_idx].to(device)

            pred = v_net(s).squeeze(-1)
            loss = 0.5 * ((pred - target) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(v_net.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            v_net.eval()
            with torch.no_grad():
                val_pred = v_net(val_s).squeeze(-1)
                val_loss = 0.5 * ((val_pred - val_v) ** 2).mean().item()
            print(
                f"  Epoch {epoch + 1}/{args.epochs}: "
                f"loss={epoch_loss / num_batches:.6f}, val_loss={val_loss:.6f}"
            )

    v_net.eval()
    with torch.no_grad():
        all_v = []
        for start in range(0, N, args.batch_size):
            end = min(start + args.batch_size, N)
            s = flat_states[start:end].to(device)
            all_v.append(v_net(s).squeeze(-1).cpu())
        all_v = torch.cat(all_v)
        print(f"  V(s): mean={all_v.mean():.4f}, std={all_v.std():.4f}")

    return v_net


def train_gae_v(obs_2d, rewards_2d, dones_2d, gamma, gae_lambda, update_epochs,
                device, args, n_iters=1, mc_v_gt=None):
    """Simulate PPO's critic training: init random V → [compute GAE returns → train V] × n_iters.

    n_iters=1: single-iteration (like PPO iter 1).
    n_iters>1: iterative refinement — recompute GAE returns with improved V each iteration.
    mc_v_gt: optional MC16 ground truth for per-iteration quality tracking.
    """
    T, E = rewards_2d.shape
    state_dim = obs_2d.shape[-1]

    v_net = Critic("state", state_dim=state_dim).to(device)

    obs_2d_d = obs_2d.to(device)
    rewards_2d_d = rewards_2d.to(device)
    dones_2d_d = dones_2d.to(device)
    flat_s = obs_2d_d.reshape(-1, state_dim)
    N = flat_s.shape[0]

    for gae_iter in range(1, n_iters + 1):
        # Compute V for all obs using current V
        with torch.no_grad():
            values = torch.zeros(T, E, device=device)
            for t in range(T):
                values[t] = v_net(obs_2d_d[t]).squeeze(-1)

        # GAE computation
        # dones_2d[t] = 1 means episode ended AFTER step t
        advantages = torch.zeros(T, E, device=device)
        lastgaelam = 0
        for t in reversed(range(T)):
            if t == T - 1:
                next_not_done = 1.0 - dones_2d_d[t]
                nextvalues = torch.zeros(E, device=device)
            else:
                next_not_done = 1.0 - dones_2d_d[t]
                nextvalues = values[t + 1]
            delta = rewards_2d_d[t] + gamma * next_not_done * nextvalues - values[t]
            advantages[t] = lastgaelam = (
                delta + gamma * gae_lambda * next_not_done * lastgaelam
            )
        returns = (advantages + values).detach()
        flat_returns = returns.reshape(-1)

        # Train V on GAE returns
        optimizer = torch.optim.Adam(
            v_net.parameters(), lr=args.lr, eps=1e-5, weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=update_epochs, eta_min=1e-5,
        )

        v_net.train()
        for epoch in range(update_epochs):
            perm = torch.randperm(N, device=device)
            for start in range(0, N, args.batch_size):
                idx = perm[start : start + args.batch_size]
                v_pred = v_net(flat_s[idx]).squeeze(-1)
                loss = 0.5 * ((v_pred - flat_returns[idx]) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(v_net.parameters(), args.grad_clip)
                optimizer.step()

            scheduler.step()

        # Per-iteration quality
        v_net.eval()
        if mc_v_gt is not None:
            with torch.no_grad():
                pred_v = []
                for t in range(T):
                    pred_v.append(v_net(obs_2d_d[t]).squeeze(-1))
                pred_v = torch.stack(pred_v)
            gt_np = mc_v_gt.cpu().flatten().numpy()
            pred_np = pred_v.cpu().flatten().numpy()
            r = pearsonr(gt_np, pred_np)[0]
            rho = spearmanr(gt_np, pred_np).statistic
            print(f"  GAE-V iter {gae_iter}/{n_iters}: V r={r:.4f}, ρ={rho:.4f}")
        else:
            print(f"  GAE-V iter {gae_iter}/{n_iters}: loss={loss.item():.6f}")

    v_net.eval()
    return v_net


@dataclass
class Args:
    checkpoint: str = "runs/pickcube_ppo/ckpt_76_logstd-1.5.pt"
    """Policy checkpoint to evaluate"""
    env_id: str = "PickCube-v1"
    num_envs: int = 128
    mc_samples: int = 16
    num_steps: int = 100
    """Rollout length for data collection"""
    max_episode_steps: int = 50
    gamma: float = 0.8
    reward_scale: float = 1.0
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"

    # IQL SARSA settings
    iql_tau: float = 0.5
    """Expectile tau. 0.5 = SARSA (symmetric MSE)"""
    iql_epochs: int = 500
    iql_lr: float = 3e-4
    iql_batch_size: int = 256
    iql_patience: int = 9999
    """Set very large to disable early stopping (TD needs full propagation)"""
    tau_polyak: float = 1.0
    """Polyak averaging rate. 1.0 = no target network (use current Q directly)"""
    direct_td: bool = False
    """Use direct TD(0) for V only — no Q network"""
    nstep: int = 1
    """N-step TD. 1=TD(0). >1 uses trajectory structure for n-step returns."""
    mc_target: bool = False
    """Train V with MC ground truth targets (supervised regression, no bootstrap)"""
    gae_v: bool = False
    """Simulate PPO's critic training: random V → GAE returns → train V"""
    gae_lambda: float = 0.9
    """GAE lambda for gae_v mode"""
    gae_update_epochs: int = 100
    """Number of training epochs for gae_v mode (matches PPO's update_epochs)"""
    gae_iters: int = 1
    """Number of GAE iterations (recompute GAE returns each iter). 1=single, 10=simulate PPO"""

    load_data: str = ""
    """Path to cached data (.pt) from a previous run — skips Steps 1-2"""

    seed: int = 1
    cuda: bool = True
    output: str = "runs/iql_sarsa_onpolicy"


def collect_and_mc(args, device):
    """Steps 1-2: Rollout + MC re-rollout. Returns cached data dict."""
    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="physx_cuda" if device.type == "cuda" else "cpu",
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
    )

    envs = gym.make(args.env_id, num_envs=args.num_envs, **env_kwargs)
    samples_per_env = 2 * args.mc_samples
    num_mc_envs = args.num_envs * samples_per_env
    mc_envs_raw = gym.make(args.env_id, num_envs=num_mc_envs, **env_kwargs)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        mc_envs_raw = FlattenActionSpaceWrapper(mc_envs_raw)

    envs = ManiSkillVectorEnv(
        envs, args.num_envs, ignore_terminations=False, record_metrics=True
    )
    mc_envs = ManiSkillVectorEnv(
        mc_envs_raw, num_mc_envs, ignore_terminations=False, record_metrics=False
    )
    print(f"  MC envs: {num_mc_envs} ({args.num_envs} × {samples_per_env})")

    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()

    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(a):
        return torch.clamp(a.detach(), action_low, action_high)

    _mc_zero = torch.zeros(
        num_mc_envs, *envs.single_action_space.shape, device=device
    )

    def _clone_state(sd):
        if isinstance(sd, dict):
            return {k: _clone_state(v) for k, v in sd.items()}
        return sd.clone()

    def _expand_state(sd, reps):
        if isinstance(sd, dict):
            return {k: _expand_state(v, reps) for k, v in sd.items()}
        if isinstance(sd, torch.Tensor) and sd.dim() > 0:
            return sd.repeat_interleave(reps, dim=0)
        return sd

    def _restore_mc(sd, seed=None):
        mc_envs.reset(seed=seed if seed is not None else args.seed)
        mc_envs.base_env.set_state_dict(sd)
        mc_envs.base_env.step(_mc_zero)
        mc_envs.base_env.set_state_dict(sd)
        mc_envs.base_env._elapsed_steps[:] = 0
        return mc_envs.base_env.get_obs()

    v_indices = []
    for i in range(args.num_envs):
        base = i * samples_per_env
        v_indices.extend(range(base + args.mc_samples, base + 2 * args.mc_samples))
    v_indices = torch.tensor(v_indices, device=device, dtype=torch.long)

    # ── Step 1: On-policy rollout ─────────────────────────────────────
    print("\nStep 1: On-policy rollout...")
    t0 = time.time()

    obs_buf = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape,
        device=device,
    )
    actions_buf = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape,
        device=device,
    )
    rewards_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    saved_states = []

    next_obs, _ = envs.reset(seed=args.seed)
    with torch.no_grad():
        for step in range(args.num_steps):
            saved_states.append(_clone_state(envs.base_env.get_state_dict()))
            obs_buf[step] = next_obs
            action = agent.get_action(next_obs, deterministic=False)
            actions_buf[step] = action
            next_obs, reward, term, trunc, _ = envs.step(clip_action(action))
            rewards_buf[step] = reward.view(-1) * args.reward_scale
            dones_buf[step] = (term | trunc).float().view(-1)

    rollout_time = time.time() - t0
    N_transitions = args.num_steps * args.num_envs
    print(f"  {rollout_time:.1f}s, {N_transitions} transitions")
    print(f"  Rewards > 0: {(rewards_buf > 0).sum().item()} / {N_transitions}")

    # Build flat IQL training data
    obs_dim = obs_buf.shape[-1]
    act_dim = actions_buf.shape[-1]

    flat_states = obs_buf.reshape(-1, obs_dim).cpu()
    flat_actions = actions_buf.reshape(-1, act_dim).cpu()
    flat_rewards = rewards_buf.reshape(-1).cpu()
    flat_terminated = dones_buf.reshape(-1).cpu()

    flat_next_states = torch.zeros_like(obs_buf)
    flat_next_states[:-1] = obs_buf[1:]
    flat_next_states[-1] = next_obs
    flat_next_states = flat_next_states.reshape(-1, obs_dim).cpu()

    # ── Step 2: MC16 re-rollout ───────────────────────────────────────
    print(f"\nStep 2: MC{args.mc_samples} on-policy re-rollout...")
    mc_t0 = time.time()

    mc_q = torch.zeros((args.num_steps, args.num_envs), device=device)
    mc_v = torch.zeros((args.num_steps, args.num_envs), device=device)

    with torch.no_grad():
        for t in tqdm(range(args.num_steps), desc="  MC re-rollout"):
            expanded = _expand_state(saved_states[t], samples_per_env)
            mc_obs = _restore_mc(expanded, seed=args.seed + t)

            first_actions = torch.zeros(
                num_mc_envs, *envs.single_action_space.shape, device=device
            )
            for i in range(args.num_envs):
                base = i * samples_per_env
                first_actions[base : base + args.mc_samples] = actions_buf[t, i]
            first_actions[v_indices] = agent.get_action(
                mc_obs[v_indices], deterministic=False
            )

            mc_obs, rew, term_, trunc_, _ = mc_envs.step(clip_action(first_actions))
            all_rews = [rew.view(-1) * args.reward_scale]
            env_done = (term_ | trunc_).view(-1).bool()

            for _ in range(args.max_episode_steps - 1):
                if env_done.all():
                    break
                a = agent.get_action(mc_obs, deterministic=False)
                mc_obs, rew, term_, trunc_, _ = mc_envs.step(clip_action(a))
                all_rews.append(
                    rew.view(-1) * args.reward_scale * (~env_done).float()
                )
                env_done = env_done | (term_ | trunc_).view(-1).bool()

            ret = torch.zeros(num_mc_envs, device=device)
            for s in reversed(range(len(all_rews))):
                ret = all_rews[s] + args.gamma * ret

            ret = ret.view(args.num_envs, samples_per_env)
            mc_q[t] = ret[:, : args.mc_samples].mean(dim=1)
            mc_v[t] = ret[:, args.mc_samples : 2 * args.mc_samples].mean(dim=1)

    mc_time = time.time() - mc_t0
    mc_adv = mc_q - mc_v
    print(f"  MC time: {mc_time:.1f}s")
    print(f"  MC V^π: mean={mc_v.mean():.4f}, std={mc_v.std():.4f}")
    print(f"  MC Q^π: mean={mc_q.mean():.4f}, std={mc_q.std():.4f}")
    print(f"  MC A^π: mean={mc_adv.mean():.4f}, std={mc_adv.std():.4f}")

    del saved_states
    mc_envs.close()
    envs.close()
    del mc_envs, envs
    torch.cuda.empty_cache()

    data = {
        "obs_buf": obs_buf.cpu(),
        "actions_buf": actions_buf.cpu(),
        "flat_states": flat_states,
        "flat_actions": flat_actions,
        "flat_rewards": flat_rewards,
        "flat_next_states": flat_next_states,
        "flat_terminated": flat_terminated,
        "mc_v": mc_v.cpu(),
        "mc_q": mc_q.cpu(),
        "mc_adv": mc_adv.cpu(),
        "gamma": args.gamma,
        "num_steps": args.num_steps,
        "num_envs": args.num_envs,
    }

    # Save cached data
    os.makedirs(args.output, exist_ok=True)
    cache_path = os.path.join(args.output, "cached_data.pt")
    torch.save(data, cache_path)
    print(f"  Cached data saved to {cache_path}")

    return data


def train_and_eval(data, args, device):
    """Steps 3-4: Train IQL and evaluate vs MC ground truth."""
    flat_states = data["flat_states"]
    flat_actions = data["flat_actions"]
    flat_rewards = data["flat_rewards"].clone()
    flat_next_states = data["flat_next_states"]
    flat_terminated = data["flat_terminated"]
    mc_v = data["mc_v"].clone().to(device)
    mc_q = data["mc_q"].clone().to(device)
    mc_adv = data["mc_adv"].clone().to(device)
    obs_buf = data["obs_buf"].to(device)
    actions_buf = data["actions_buf"].to(device)
    num_steps = data["num_steps"]
    num_envs = data["num_envs"]
    N_transitions = num_steps * num_envs

    # Apply reward_scale to loaded data (assumes cached data has scale=1.0)
    if args.reward_scale != 1.0 and args.load_data:
        s = args.reward_scale
        print(f"  Scaling loaded data by reward_scale={s}")
        flat_rewards = flat_rewards * s
        mc_v = mc_v * s
        mc_q = mc_q * s
        mc_adv = mc_adv * s

    # ── Step 3: Train ────────────────────────────────────────────────
    iql_t0 = time.time()
    td_args = SimpleNamespace(
        lr=args.iql_lr,
        weight_decay=1e-4,
        epochs=args.iql_epochs,
        batch_size=args.iql_batch_size,
        gamma=args.gamma,
        expectile_tau=args.iql_tau,
        tau_polyak=args.tau_polyak,
        patience=args.iql_patience,
        grad_clip=0.5,
    )

    rs_tag = f"_rs{args.reward_scale}" if args.reward_scale != 1.0 else ""

    q_net = None
    if args.gae_v:
        method_tag = f"gae_v_lam{args.gae_lambda}_ep{args.gae_update_epochs}_iter{args.gae_iters}{rs_tag}"
        print(f"\nStep 3: Training V with GAE returns (lambda={args.gae_lambda}, epochs={args.gae_update_epochs}, iters={args.gae_iters})...")
        rewards_2d = flat_rewards.reshape(num_steps, num_envs)
        dones_2d = flat_terminated.reshape(num_steps, num_envs)
        v_net = train_gae_v(
            data["obs_buf"], rewards_2d, dones_2d,
            args.gamma, args.gae_lambda, args.gae_update_epochs, device, td_args,
            n_iters=args.gae_iters, mc_v_gt=mc_v,
        )
    elif args.mc_target:
        method_tag = f"mc_target_v{rs_tag}"
        print(f"\nStep 3: Training V with MC targets (supervised regression)...")
        mc_v_flat = mc_v.cpu().reshape(-1)
        v_net = train_mc_v(flat_states, mc_v_flat, device, td_args)
    elif args.nstep > 1:
        method_tag = f"nstep{args.nstep}{rs_tag}"
        rewards_2d = flat_rewards.reshape(num_steps, num_envs)
        dones_2d = flat_terminated.reshape(num_steps, num_envs)
        print(f"\nStep 3: Training {args.nstep}-step TD V...")
        v_net = train_nstep_td_v(
            data["obs_buf"], rewards_2d, dones_2d, args.nstep, device, td_args,
        )
    elif args.direct_td:
        method_tag = f"td0_polyak{args.tau_polyak}{rs_tag}"
        print(f"\nStep 3: Training direct TD(0) V (polyak={args.tau_polyak})...")
        v_net = train_td0_v(
            flat_states, flat_rewards, flat_next_states,
            flat_terminated, device, td_args,
        )
    else:
        method_tag = f"polyak{args.tau_polyak}{rs_tag}"
        print(f"\nStep 3: Training IQL (tau={args.iql_tau}, polyak={args.tau_polyak})...")
        q_net, v_net = train_iql(
            flat_states, flat_actions, flat_rewards,
            flat_next_states, flat_terminated, device, td_args,
        )

    iql_time = time.time() - iql_t0
    print(f"  Training: {iql_time:.1f}s")

    # ── Step 4: Evaluate ──────────────────────────────────────────────
    print("\nStep 4: Comparing vs MC16...")

    with torch.no_grad():
        iql_v_list = []
        iql_q_list = []
        for t in range(num_steps):
            s = obs_buf[t]
            a = actions_buf[t]
            iql_v_list.append(v_net(s).squeeze(-1))
            if q_net is not None:
                iql_q_list.append(q_net(s, a).squeeze(-1))
        iql_v = torch.stack(iql_v_list)
        if q_net is not None:
            iql_q = torch.stack(iql_q_list)
            iql_adv = iql_q - iql_v
        else:
            iql_q = iql_v  # placeholder for V-only mode
            iql_adv = torch.zeros_like(iql_v)

    mc_v_np = mc_v.cpu().flatten().numpy()
    mc_q_np = mc_q.cpu().flatten().numpy()
    mc_adv_np = mc_adv.cpu().flatten().numpy()
    iql_v_np = iql_v.cpu().flatten().numpy()
    iql_q_np = iql_q.cpu().flatten().numpy()
    iql_adv_np = iql_adv.cpu().flatten().numpy()

    metrics = {}
    for name, gt, pred in [
        ("V", mc_v_np, iql_v_np),
        ("Q", mc_q_np, iql_q_np),
        ("A", mc_adv_np, iql_adv_np),
    ]:
        metrics[name] = {
            "r": pearsonr(gt, pred)[0],
            "rho": spearmanr(gt, pred).statistic,
            "bias": float((pred - gt).mean()),
            "rmse": float(np.sqrt(((pred - gt) ** 2).mean())),
            "gt_mean": float(gt.mean()),
            "gt_std": float(gt.std()),
            "pred_mean": float(pred.mean()),
            "pred_std": float(pred.std()),
        }

    if args.gae_v:
        method_name = f"GAE V (lambda={args.gae_lambda}, epochs={args.gae_update_epochs}, iters={args.gae_iters})"
    elif args.mc_target:
        method_name = "MC target V (supervised)"
    elif args.nstep > 1:
        method_name = f"{args.nstep}-step TD V"
    elif args.direct_td:
        method_name = f"TD(0) polyak={args.tau_polyak}"
    else:
        method_name = f"IQL SARSA (tau={args.iql_tau}, polyak={args.tau_polyak})"
    if args.reward_scale != 1.0:
        method_name += f" [rs={args.reward_scale}]"
    print(f"\n{'='*75}")
    print(f"  {method_name} vs MC{args.mc_samples} V^π  [gamma={args.gamma}]")
    print(f"{'='*75}")
    print(f"  {'':5s} {'r':>8s} {'ρ':>8s} {'bias':>10s} {'RMSE':>8s}  "
          f"{'MC μ':>8s} {'MC σ':>8s} {'IQL μ':>8s} {'IQL σ':>8s}")
    print(f"  {'-'*73}")
    for name in ["V", "Q", "A"]:
        m = metrics[name]
        print(
            f"  {name:5s} {m['r']:8.4f} {m['rho']:8.4f} {m['bias']:+10.4f} "
            f"{m['rmse']:8.4f}  {m['gt_mean']:8.4f} {m['gt_std']:8.4f} "
            f"{m['pred_mean']:8.4f} {m['pred_std']:8.4f}"
        )
    print(f"{'='*75}")

    # ── Plot ──────────────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for ax, name, gt, pred in [
        (axes[0, 0], "V(s)", mc_v_np, iql_v_np),
        (axes[0, 1], "Q(s,a)", mc_q_np, iql_q_np),
        (axes[0, 2], "A(s,a)", mc_adv_np, iql_adv_np),
    ]:
        ax.scatter(gt, pred, alpha=0.05, s=2, edgecolors="none")
        lo, hi = gt.min(), gt.max()
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        key = name.split("(")[0]
        m = metrics[key]
        ax.set_xlabel(f"MC {name}")
        ax.set_ylabel(f"IQL {name}")
        ax.set_title(f"{name}: r={m['r']:.4f}, ρ={m['rho']:.4f}, bias={m['bias']:+.4f}")
        ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    per_t = {"V_r": [], "Q_r": [], "A_r": [], "A_rho": []}
    for t in range(num_steps):
        mv = mc_v[t].cpu().numpy()
        mq = mc_q[t].cpu().numpy()
        ma = mc_adv[t].cpu().numpy()
        iv = iql_v[t].cpu().numpy()
        iq = iql_q[t].cpu().numpy()
        ia = iql_adv[t].cpu().numpy()
        per_t["V_r"].append(pearsonr(mv, iv)[0])
        per_t["Q_r"].append(pearsonr(mq, iq)[0])
        per_t["A_r"].append(pearsonr(ma, ia)[0])
        per_t["A_rho"].append(spearmanr(ma, ia).statistic)

    ax.plot(per_t["V_r"], label="V (Pearson r)", alpha=0.7)
    ax.plot(per_t["Q_r"], label="Q (Pearson r)", alpha=0.7)
    ax.plot(per_t["A_r"], label="A (Pearson r)", alpha=0.7)
    ax.plot(per_t["A_rho"], label="A (Spearman ρ)", ls="--", alpha=0.7)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Correlation")
    ax.set_title("Per-timestep IQL vs MC")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    v_bias_t = [
        (iql_v[t].cpu() - mc_v[t].cpu()).mean().item()
        for t in range(num_steps)
    ]
    ax.plot(v_bias_t, alpha=0.7)
    ax.axhline(0, color="r", ls="--", lw=1)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("V bias (IQL - MC)")
    ax.set_title(f"V bias per timestep: mean={np.mean(v_bias_t):.4f}")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.axis("off")
    rows = [
        ["Metric", "V(s)", "Q(s,a)", "A(s,a)"],
        ["Pearson r", f"{metrics['V']['r']:.4f}", f"{metrics['Q']['r']:.4f}",
         f"{metrics['A']['r']:.4f}"],
        ["Spearman ρ", f"{metrics['V']['rho']:.4f}", f"{metrics['Q']['rho']:.4f}",
         f"{metrics['A']['rho']:.4f}"],
        ["Bias", f"{metrics['V']['bias']:+.4f}", f"{metrics['Q']['bias']:+.4f}",
         f"{metrics['A']['bias']:+.4f}"],
        ["RMSE", f"{metrics['V']['rmse']:.4f}", f"{metrics['Q']['rmse']:.4f}",
         f"{metrics['A']['rmse']:.4f}"],
        ["", "", "", ""],
        ["gamma", f"{args.gamma}", "polyak", f"{args.tau_polyak}"],
        ["transitions", f"{N_transitions}", "epochs", f"{args.iql_epochs}"],
    ]
    table = ax.table(cellText=rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    for j in range(4):
        table[0, j].set_text_props(fontweight="bold")
    ax.set_title("Summary")

    fig.suptitle(
        f"{method_name} vs MC{args.mc_samples} V^π\n"
        f"(γ={args.gamma}, {N_transitions} transitions, {args.iql_epochs} epochs)",
        fontsize=12,
    )
    plt.tight_layout()
    plot_path = os.path.join(args.output, f"result_{method_tag}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {plot_path}")
    plt.close()

    save_path = os.path.join(args.output, f"results_{method_tag}.pt")
    torch.save(
        {
            "mc_v": mc_v.cpu(), "mc_q": mc_q.cpu(), "mc_adv": mc_adv.cpu(),
            "iql_v": iql_v.cpu(), "iql_q": iql_q.cpu(), "iql_adv": iql_adv.cpu(),
            "metrics": metrics, "per_timestep": per_t, "args": vars(args),
        },
        save_path,
    )
    print(f"Results saved to {save_path}")
    return metrics


if __name__ == "__main__":
    import tyro

    args = tyro.cli(Args)

    print("=== IQL SARSA On-Policy Test ===")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  tau={args.iql_tau}, gamma={args.gamma}, tau_polyak={args.tau_polyak}")
    print(f"  Envs={args.num_envs}, Steps={args.num_steps}, MC={args.mc_samples}")
    print(f"  Epochs={args.iql_epochs}, patience={args.iql_patience}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if args.load_data:
        print(f"\nLoading cached data from {args.load_data}")
        data = torch.load(args.load_data, map_location="cpu")
        print(f"  {data['num_steps']} steps × {data['num_envs']} envs, gamma={data['gamma']}")
    else:
        data = collect_and_mc(args, device)

    train_and_eval(data, args, device)
