"""Quick test: TD+EMA with early stopping vs without, data scaling only.
Only runs TD+EMA method to verify if early stopping fixes the overfitting issue.

Usage:
  python -u -m RL.td_ema_earlystop_test --gamma 0.99 --td_grad_steps 100000
"""

import copy
import os
import random
import sys
import time
from dataclasses import dataclass

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from scipy.stats import pearsonr, spearmanr
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from data.data_collection.ppo import Agent, layer_init


@dataclass
class Args:
    checkpoint: str = "runs/pickcube_ppo/ckpt_76_logstd-1.5.pt"
    optimal_checkpoint: str = "runs/pickcube_ppo/ckpt_301.pt"
    gamma: float = 0.99
    num_envs: int = 100
    num_steps: int = 50
    max_episode_steps: int = 50
    mc_samples: int = 16
    td_grad_steps: int = 100000
    td_reward_scale: float = 10.0
    ema_tau: float = 0.005
    critic_layers: int = 3
    lr: float = 3e-4
    minibatch_size: int = 1000
    patience: int = 20
    rollout_counts: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100)
    seed: int = 1
    env_id: str = "PickCube-v1"
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"


if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda")
    max_rollouts = max(args.rollout_counts)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_kwargs = dict(
        obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda",
        reward_mode=args.reward_mode, control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
    )

    T, E = args.num_steps, args.num_envs

    # ── Setup ──
    envs = gym.make(args.env_id, num_envs=args.num_envs, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=False, record_metrics=True)
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    action_dim = int(np.prod(envs.single_action_space.shape))
    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()
    optimal_agent = Agent(envs).to(device)
    optimal_agent.load_state_dict(torch.load(args.optimal_checkpoint, map_location=device))
    optimal_agent.eval()

    def clip_action(a):
        return torch.clamp(a.detach(), action_low, action_high)

    def make_critic(hidden_dim=256):
        layers = [layer_init(nn.Linear(obs_dim, hidden_dim)), nn.Tanh()]
        for _ in range(args.critic_layers - 1):
            layers += [layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh()]
        layers.append(layer_init(nn.Linear(hidden_dim, 1)))
        return nn.Sequential(*layers).to(device)

    def _clone_state(sd):
        if isinstance(sd, dict):
            return {k: _clone_state(v) for k, v in sd.items()}
        return sd.clone()

    def _expand_state(sd, repeats):
        if isinstance(sd, dict):
            return {k: _expand_state(v, repeats) for k, v in sd.items()}
        if isinstance(sd, torch.Tensor) and sd.dim() > 0:
            return sd.repeat_interleave(repeats, dim=0)
        return sd

    # ── Phase 1: Collect rollouts ──
    print(f"Phase 1: Collecting {max_rollouts} rollouts...")
    sys.stdout.flush()
    t0 = time.time()
    data_pool = []

    for ri in range(max_rollouts):
        roll_obs = torch.zeros(T, E, obs_dim, device=device)
        roll_rewards = torch.zeros(T, E, device=device)
        roll_dones = torch.zeros(T, E, device=device)
        saved_states = [] if ri == 0 else None

        next_obs, _ = envs.reset(seed=args.seed + ri)
        next_done = torch.zeros(E, device=device)

        with torch.no_grad():
            for step in range(T):
                if ri == 0:
                    saved_states.append(_clone_state(envs.base_env.get_state_dict()))
                roll_obs[step] = next_obs
                roll_dones[step] = next_done
                action = agent.get_action(next_obs, deterministic=False)
                next_obs, reward, term, trunc, _ = envs.step(clip_action(action))
                next_done = (term | trunc).float()
                roll_rewards[step] = reward.view(-1)

        data_pool.append(dict(
            obs=roll_obs, rewards=roll_rewards, dones=roll_dones,
            next_obs=next_obs.clone(), next_done=next_done.clone(),
            saved_states=saved_states,
        ))
        if (ri + 1) % 20 == 0 or ri + 1 == max_rollouts:
            print(f"  {ri + 1}/{max_rollouts}")
            sys.stdout.flush()

    print(f"  Phase 1 done ({time.time() - t0:.1f}s)")

    # ── Phase 2: MC16 ground truth ──
    samples_per_env = 2 * args.mc_samples
    num_mc_envs = E * samples_per_env
    print(f"\nPhase 2: MC16 ground truth ({num_mc_envs} mc_envs)...")
    sys.stdout.flush()
    t0 = time.time()

    mc_envs_raw = gym.make(args.env_id, num_envs=num_mc_envs, **env_kwargs)
    if isinstance(mc_envs_raw.action_space, gym.spaces.Dict):
        mc_envs_raw = FlattenActionSpaceWrapper(mc_envs_raw)
    mc_envs = ManiSkillVectorEnv(mc_envs_raw, num_mc_envs, ignore_terminations=False, record_metrics=False)
    mc_zero_action = torch.zeros(num_mc_envs, *envs.single_action_space.shape, device=device)

    opt_indices = []
    on_indices = []
    for i in range(E):
        base = i * samples_per_env
        opt_indices.extend(range(base, base + args.mc_samples))
        on_indices.extend(range(base + args.mc_samples, base + 2 * args.mc_samples))
    opt_indices = torch.tensor(opt_indices, device=device, dtype=torch.long)
    on_indices = torch.tensor(on_indices, device=device, dtype=torch.long)

    eval_saved_states = data_pool[0]['saved_states']
    on_mc16 = torch.zeros(T, E, device=device)

    def _restore_mc_state(sd, seed=None):
        mc_envs.reset(seed=seed if seed is not None else args.seed)
        mc_envs.base_env.set_state_dict(sd)
        mc_envs.base_env.step(mc_zero_action)
        mc_envs.base_env.set_state_dict(sd)
        mc_envs.base_env._elapsed_steps[:] = 0
        return mc_envs.base_env.get_obs()

    with torch.no_grad():
        for t in range(T):
            expanded = _expand_state(eval_saved_states[t], samples_per_env)
            mc_obs = _restore_mc_state(expanded, seed=args.seed + 1000 + t)
            env_done = torch.zeros(num_mc_envs, device=device).bool()
            all_rews = []
            for _ in range(args.max_episode_steps):
                if env_done.all():
                    break
                a = torch.zeros(num_mc_envs, action_dim, device=device)
                a[opt_indices] = optimal_agent.get_action(mc_obs[opt_indices], deterministic=False)
                a[on_indices] = agent.get_action(mc_obs[on_indices], deterministic=False)
                mc_obs, rew, term, trunc, _ = mc_envs.step(clip_action(a))
                all_rews.append(rew.view(-1) * (~env_done).float())
                env_done = env_done | (term | trunc).view(-1).bool()
            ret = torch.zeros(num_mc_envs, device=device)
            for s in reversed(range(len(all_rews))):
                ret = all_rews[s] + args.gamma * ret
            ret = ret.view(E, samples_per_env)
            on_mc16[t] = ret[:, args.mc_samples:2 * args.mc_samples].mean(dim=1)
            if (t + 1) % 10 == 0:
                print(f"  MC16 step {t + 1}/{T}")
                sys.stdout.flush()

    mc_envs.close()
    del mc_envs, mc_envs_raw
    data_pool[0]['saved_states'] = None
    torch.cuda.empty_cache()

    print(f"  On-policy MC16: mean={on_mc16.mean():.4f}, std={on_mc16.std():.4f}")
    print(f"  Phase 2 done ({time.time() - t0:.1f}s)")

    eval_obs = data_pool[0]['obs']
    envs.close()
    del envs, agent, optimal_agent
    torch.cuda.empty_cache()

    def evaluate(critic):
        with torch.no_grad():
            v = critic(eval_obs.reshape(-1, obs_dim)).view(-1).cpu().numpy()
        g = on_mc16.reshape(-1).cpu().numpy()
        return pearsonr(v, g)[0], spearmanr(v, g).correlation

    def combine(n):
        d = data_pool[:n]
        return (
            torch.cat([x['obs'] for x in d], dim=1),
            torch.cat([x['rewards'] for x in d], dim=1),
            torch.cat([x['dones'] for x in d], dim=1),
            torch.cat([x['next_obs'] for x in d], dim=0),
            torch.cat([x['next_done'] for x in d], dim=0),
        )

    def prep_flat(obs, rewards, dones, next_obs, next_done):
        Tl, El, D = obs.shape
        flat_s = obs.reshape(-1, D)
        flat_r = rewards.reshape(-1) * args.td_reward_scale
        flat_ns = torch.zeros_like(obs)
        flat_ns[:-1] = obs[1:]
        flat_ns[-1] = next_obs
        flat_ns = flat_ns.reshape(-1, D)
        flat_d = torch.zeros_like(rewards)
        flat_d[:-1] = dones[1:]
        flat_d[-1] = next_done
        flat_d = flat_d.reshape(-1)
        return flat_s, flat_r, flat_ns, flat_d

    def unscale(critic):
        with torch.no_grad():
            critic[-1].weight.div_(args.td_reward_scale)
            critic[-1].bias.div_(args.td_reward_scale)

    # ── TD+EMA without early stopping (fixed grad steps) ──
    def train_ema_fixed(flat_s, flat_r, flat_ns, flat_d):
        N = flat_s.shape[0]
        mb = min(args.minibatch_size, N)
        critic = make_critic()
        critic_target = make_critic()
        critic_target.load_state_dict(critic.state_dict())
        opt = optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)
        critic.train()
        step = 0
        while step < args.td_grad_steps:
            perm = torch.randperm(N, device=device)
            for start in range(0, N, mb):
                idx = perm[start:start + mb]
                with torch.no_grad():
                    target = flat_r[idx] + args.gamma * critic_target(flat_ns[idx]).view(-1) * (1 - flat_d[idx])
                loss = 0.5 * ((critic(flat_s[idx]).view(-1) - target) ** 2).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5); opt.step()
                with torch.no_grad():
                    for p, pt in zip(critic.parameters(), critic_target.parameters()):
                        pt.data.mul_(1 - args.ema_tau).add_(p.data, alpha=args.ema_tau)
                step += 1
                if step >= args.td_grad_steps:
                    break
        unscale(critic)
        return critic

    # ── TD+EMA with early stopping ──
    def train_ema_earlystop(flat_s, flat_r, flat_ns, flat_d):
        N = flat_s.shape[0]
        mb = min(args.minibatch_size, N)

        # train/val split
        perm_all = torch.randperm(N, device=device)
        val_size = max(mb, int(N * 0.1))
        val_idx = perm_all[:val_size]
        train_idx = perm_all[val_size:]
        N_train = train_idx.shape[0]

        critic = make_critic()
        critic_target = make_critic()
        critic_target.load_state_dict(critic.state_dict())
        opt = optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        max_epochs = args.td_grad_steps // max(1, N_train // mb)
        max_epochs = max(max_epochs, 200)

        critic.train()
        for epoch in range(max_epochs):
            train_perm = train_idx[torch.randperm(N_train, device=device)]
            for start in range(0, N_train, mb):
                idx = train_perm[start:start + mb]
                with torch.no_grad():
                    target = flat_r[idx] + args.gamma * critic_target(flat_ns[idx]).view(-1) * (1 - flat_d[idx])
                loss = 0.5 * ((critic(flat_s[idx]).view(-1) - target) ** 2).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5); opt.step()
                with torch.no_grad():
                    for p, pt in zip(critic.parameters(), critic_target.parameters()):
                        pt.data.mul_(1 - args.ema_tau).add_(p.data, alpha=args.ema_tau)

            # Val loss
            critic.eval()
            with torch.no_grad():
                vt = flat_r[val_idx] + args.gamma * critic_target(flat_ns[val_idx]).view(-1) * (1 - flat_d[val_idx])
                vp = critic(flat_s[val_idx]).view(-1)
                val_loss = 0.5 * ((vp - vt) ** 2).mean().item()
            critic.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in critic.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"      early stop epoch {epoch + 1}/{max_epochs}")
                    break

        if best_state is not None:
            critic.load_state_dict(best_state)
        unscale(critic)
        return critic

    # ── Run ──
    print(f"\n{'=' * 70}")
    print(f"TD+EMA Data Scaling: fixed {args.td_grad_steps} grad steps, patience={args.patience}")
    print(f"{'=' * 70}\n")

    results = {}
    for N_roll in args.rollout_counts:
        trans = N_roll * E * T
        steps_per_ep = trans // args.minibatch_size
        epochs_100k = args.td_grad_steps // max(1, steps_per_ep)
        print(f"--- N={N_roll} ({trans:,} trans, {steps_per_ep} steps/ep, 100k={epochs_100k} epochs) ---")
        sys.stdout.flush()

        obs, rewards, dones, next_obs, next_done = combine(N_roll)
        flat_s, flat_r, flat_ns, flat_d = prep_flat(obs, rewards, dones, next_obs, next_done)
        del obs, rewards, dones, next_obs, next_done

        # Without early stopping
        t0 = time.time()
        c = train_ema_fixed(flat_s, flat_r, flat_ns, flat_d)
        r_fixed, rho_fixed = evaluate(c)
        t_fixed = time.time() - t0
        del c

        # With early stopping
        t0 = time.time()
        c = train_ema_earlystop(flat_s, flat_r, flat_ns, flat_d)
        r_es, rho_es = evaluate(c)
        t_es = time.time() - t0
        del c

        print(f"  Fixed:      r={r_fixed:.4f} rho={rho_fixed:.4f} ({t_fixed:.1f}s)")
        print(f"  EarlyStop:  r={r_es:.4f} rho={rho_es:.4f} ({t_es:.1f}s)")
        sys.stdout.flush()

        results[N_roll] = dict(fixed_r=r_fixed, fixed_rho=rho_fixed,
                               es_r=r_es, es_rho=rho_es)
        del flat_s, flat_r, flat_ns, flat_d
        torch.cuda.empty_cache()

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: TD+EMA on-policy r")
    print(f"{'=' * 70}")
    print(f"| {'N':>5} | {'Fixed r':>8} | {'ES r':>8} | {'delta':>8} |")
    print(f"|-------|----------|----------|----------|")
    for n in args.rollout_counts:
        d = results[n]
        delta = d['es_r'] - d['fixed_r']
        print(f"| {n:>5} | {d['fixed_r']:>8.4f} | {d['es_r']:>8.4f} | {delta:>+8.4f} |")
    print(f"{'=' * 70}")
