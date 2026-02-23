"""Quick test: GAE vs GAE+rs10 data scaling.

Usage:
  python -u -m RL.gae_rs_test --gamma 0.99
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
    gae_lambda: float = 0.95
    num_envs: int = 100
    num_steps: int = 50
    max_episode_steps: int = 50
    mc_samples: int = 16
    gae_iters: int = 5
    gae_epochs: int = 100
    critic_layers: int = 3
    lr: float = 3e-4
    minibatch_size: int = 1000
    rollout_counts: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100)
    hidden_dims: tuple[int, ...] = (64, 128, 256, 512)
    reward_scales: tuple[float, ...] = (1.0, 10.0)
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

    # ── Phase 2: MC16 ground truth (on-policy only for speed) ──
    samples_per_env = args.mc_samples  # on-policy only
    num_mc_envs = E * samples_per_env
    print(f"\nPhase 2: On-policy MC16 ground truth ({num_mc_envs} mc_envs)...")
    sys.stdout.flush()
    t0 = time.time()

    mc_envs_raw = gym.make(args.env_id, num_envs=num_mc_envs, **env_kwargs)
    if isinstance(mc_envs_raw.action_space, gym.spaces.Dict):
        mc_envs_raw = FlattenActionSpaceWrapper(mc_envs_raw)
    mc_envs = ManiSkillVectorEnv(mc_envs_raw, num_mc_envs, ignore_terminations=False, record_metrics=False)
    mc_zero_action = torch.zeros(num_mc_envs, *envs.single_action_space.shape, device=device)

    def _restore_mc_state(sd, seed=None):
        mc_envs.reset(seed=seed if seed is not None else args.seed)
        mc_envs.base_env.set_state_dict(sd)
        mc_envs.base_env.step(mc_zero_action)
        mc_envs.base_env.set_state_dict(sd)
        mc_envs.base_env._elapsed_steps[:] = 0
        return mc_envs.base_env.get_obs()

    eval_saved_states = data_pool[0]['saved_states']
    on_mc16 = torch.zeros(T, E, device=device)

    with torch.no_grad():
        for t in range(T):
            expanded = _expand_state(eval_saved_states[t], samples_per_env)
            mc_obs = _restore_mc_state(expanded, seed=args.seed + 1000 + t)
            env_done = torch.zeros(num_mc_envs, device=device).bool()
            all_rews = []
            for _ in range(args.max_episode_steps):
                if env_done.all():
                    break
                a = agent.get_action(mc_obs, deterministic=False)
                mc_obs, rew, term, trunc, _ = mc_envs.step(clip_action(a))
                all_rews.append(rew.view(-1) * (~env_done).float())
                env_done = env_done | (term | trunc).view(-1).bool()
            ret = torch.zeros(num_mc_envs, device=device)
            for s in reversed(range(len(all_rews))):
                ret = all_rews[s] + args.gamma * ret
            on_mc16[t] = ret.view(E, samples_per_env).mean(dim=1)
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

    # ── GAE training with reward_scale ──
    def train_gae(obs, rewards, dones, next_obs, next_done,
                  reward_scale=1.0, hidden_dim=256):
        Tl, El, D = obs.shape
        critic = make_critic(hidden_dim)
        opt = optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)
        flat_obs = obs.reshape(-1, D)
        N = flat_obs.shape[0]
        mb = min(args.minibatch_size, N)

        scaled_rewards = rewards * reward_scale

        for _ in range(args.gae_iters):
            with torch.no_grad():
                values = torch.stack([critic(obs[t]).flatten() for t in range(Tl)])
                nv = critic(next_obs).reshape(1, -1)
                adv = torch.zeros_like(rewards)
                lastgaelam = 0
                for t in reversed(range(Tl)):
                    if t == Tl - 1:
                        nnd, nvs = 1.0 - next_done, nv
                    else:
                        nnd, nvs = 1.0 - dones[t + 1], values[t + 1]
                    delta = scaled_rewards[t] + args.gamma * nnd * nvs - values[t]
                    adv[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nnd * lastgaelam
                flat_ret = (adv + values).reshape(-1)

            critic.train()
            for _ in range(args.gae_epochs):
                perm = torch.randperm(N, device=device)
                for start in range(0, N, mb):
                    idx = perm[start:start + mb]
                    loss = 0.5 * ((critic(flat_obs[idx]).view(-1) - flat_ret[idx]) ** 2).mean()
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), 0.5); opt.step()

        # Unscale output
        if reward_scale != 1.0:
            with torch.no_grad():
                critic[-1].weight.div_(reward_scale)
                critic[-1].bias.div_(reward_scale)
        return critic

    # ── Run data scaling ──
    rs_labels = [f"GAE_rs{int(rs)}" for rs in args.reward_scales]
    print(f"\n{'=' * 70}")
    print(f"GAE Data Scaling: reward_scales={args.reward_scales}")
    print(f"  GAE: {args.gae_iters} iters x {args.gae_epochs} epochs, lambda={args.gae_lambda}")
    print(f"{'=' * 70}\n")

    data_results = {}
    for N_roll in args.rollout_counts:
        trans = N_roll * E * T
        print(f"--- N={N_roll} ({trans:,} trans) ---")
        sys.stdout.flush()

        obs, rewards, dones, next_obs, next_done = combine(N_roll)
        row = {}
        for rs in args.reward_scales:
            t0 = time.time()
            c = train_gae(obs, rewards, dones, next_obs, next_done,
                          reward_scale=rs, hidden_dim=256)
            r_on, rho_on = evaluate(c)
            label = f"GAE_rs{int(rs)}"
            row[label] = (r_on, rho_on)
            print(f"  {label}: r={r_on:.4f} rho={rho_on:.4f} ({time.time()-t0:.1f}s)")
            del c
        data_results[N_roll] = row
        del obs, rewards, dones, next_obs, next_done
        torch.cuda.empty_cache()
        sys.stdout.flush()

    # ── Run network scaling ──
    print(f"\n{'=' * 70}")
    print(f"GAE Network Scaling (rollouts={max_rollouts})")
    print(f"{'=' * 70}\n")

    obs_all, rew_all, don_all, nobs_all, nd_all = combine(max_rollouts)
    net_results = {}
    for hd in args.hidden_dims:
        print(f"--- hidden={hd} ---")
        sys.stdout.flush()
        row = {}
        for rs in args.reward_scales:
            t0 = time.time()
            c = train_gae(obs_all, rew_all, don_all, nobs_all, nd_all,
                          reward_scale=rs, hidden_dim=hd)
            r_on, rho_on = evaluate(c)
            label = f"GAE_rs{int(rs)}"
            row[label] = (r_on, rho_on)
            print(f"  {label}: r={r_on:.4f} rho={rho_on:.4f} ({time.time()-t0:.1f}s)")
            del c
        net_results[hd] = row
        torch.cuda.empty_cache()
        sys.stdout.flush()

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"DATA SCALING (hidden=256) — Pearson r")
    print(f"{'=' * 70}")
    header = f"| {'N':>5} |"
    for label in rs_labels:
        header += f" {label:>10} |"
    print(header)
    print("|-------|" + "------------|" * len(rs_labels))
    for n in args.rollout_counts:
        row = f"| {n:>5} |"
        for label in rs_labels:
            row += f" {data_results[n][label][0]:>10.4f} |"
        print(row)

    print(f"\n{'=' * 70}")
    print(f"NETWORK SCALING (rollouts={max_rollouts}) — Pearson r")
    print(f"{'=' * 70}")
    header = f"| {'H':>5} |"
    for label in rs_labels:
        header += f" {label:>10} |"
    print(header)
    print("|-------|" + "------------|" * len(rs_labels))
    for h in args.hidden_dims:
        row = f"| {h:>5} |"
        for label in rs_labels:
            row += f" {net_results[h][label][0]:>10.4f} |"
        print(row)
    print(f"{'=' * 70}")
