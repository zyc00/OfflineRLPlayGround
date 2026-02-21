"""V Scaling Analysis: GAE vs TD(0) quality as a function of offline data size.

Compares iterative GAE and TD(0) for learning V from behavior policy rollouts.
Evaluates V quality via Pearson r and Spearman ρ against trajectory MC returns.

Usage:
  python -u -m RL.v_scaling_analysis
  python -u -m RL.v_scaling_analysis --gamma 0.8
"""

import random
import time
from dataclasses import dataclass, field
from typing import Optional

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
    gamma: float = 0.99
    gae_lambda: float = 0.9
    num_envs: int = 100
    num_steps: int = 50
    max_episode_steps: int = 50
    rollout_counts: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100)
    gae_iters: int = 5
    gae_epochs: int = 100
    td_epochs: int = 200
    td_reward_scale: float = 10.0
    critic_hidden: int = 256
    critic_layers: int = 3
    lr: float = 3e-4
    minibatch_size: int = 1000
    seed: int = 1
    env_id: str = "PickCube-v1"
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"


if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda")
    max_rollouts = max(args.rollout_counts)
    obs_dim = None  # set after env creation

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Env setup ──
    env_kwargs = dict(
        obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda",
        reward_mode=args.reward_mode, control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
    )
    envs = gym.make(args.env_id, num_envs=args.num_envs, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(
        envs, args.num_envs, ignore_terminations=False, record_metrics=True,
    )
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    # ── Agent ──
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()

    def clip_action(a):
        return torch.clamp(a.detach(), action_low, action_high)

    def make_critic():
        layers = [layer_init(nn.Linear(obs_dim, args.critic_hidden)), nn.Tanh()]
        for _ in range(args.critic_layers - 1):
            layers += [layer_init(nn.Linear(args.critic_hidden, args.critic_hidden)), nn.Tanh()]
        layers.append(layer_init(nn.Linear(args.critic_hidden, 1)))
        return nn.Sequential(*layers).to(device)

    # ── Collect data pool ──
    T, E = args.num_steps, args.num_envs
    print(f"Collecting {max_rollouts} rollouts ({max_rollouts * E * T:,} transitions)...")
    data_pool = []

    for ri in range(max_rollouts):
        roll_obs = torch.zeros(T, E, obs_dim, device=device)
        roll_rewards = torch.zeros(T, E, device=device)
        roll_dones = torch.zeros(T, E, device=device)

        next_obs, _ = envs.reset(seed=args.seed + ri)
        next_done = torch.zeros(E, device=device)

        with torch.no_grad():
            for step in range(T):
                roll_obs[step] = next_obs
                roll_dones[step] = next_done
                action = agent.get_action(next_obs, deterministic=False)
                next_obs, reward, term, trunc, infos = envs.step(clip_action(action))
                next_done = (term | trunc).float()
                roll_rewards[step] = reward.view(-1)

        # MC returns (ground truth)
        mc = torch.zeros(T, E, device=device)
        future = torch.zeros(E, device=device)
        for t in reversed(range(T)):
            if t == T - 1:
                mask = 1.0 - next_done
            else:
                mask = 1.0 - roll_dones[t + 1]
            future = roll_rewards[t] + args.gamma * future * mask
            mc[t] = future

        data_pool.append(dict(
            obs=roll_obs, rewards=roll_rewards, dones=roll_dones,
            next_obs=next_obs.clone(), next_done=next_done.clone(),
            mc_returns=mc,
        ))

        if (ri + 1) % 20 == 0 or ri + 1 == max_rollouts:
            print(f"  {ri + 1}/{max_rollouts} rollouts collected")

    # ── Combine rollouts along env axis ──
    def combine(n):
        d = data_pool[:n]
        return (
            torch.cat([x['obs'] for x in d], dim=1),
            torch.cat([x['rewards'] for x in d], dim=1),
            torch.cat([x['dones'] for x in d], dim=1),
            torch.cat([x['next_obs'] for x in d], dim=0),
            torch.cat([x['next_done'] for x in d], dim=0),
            torch.cat([x['mc_returns'] for x in d], dim=1),
        )

    # ── GAE training ──
    def train_gae(obs, rewards, dones, next_obs, next_done):
        Tl, El, D = obs.shape
        critic = make_critic()
        opt = optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)
        flat_obs = obs.reshape(-1, D)
        N = flat_obs.shape[0]
        mb = min(args.minibatch_size, N)

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
                    delta = rewards[t] + args.gamma * nnd * nvs - values[t]
                    adv[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nnd * lastgaelam
                flat_ret = (adv + values).reshape(-1)

            critic.train()
            for _ in range(args.gae_epochs):
                perm = torch.randperm(N, device=device)
                for start in range(0, N, mb):
                    idx = perm[start:start + mb]
                    loss = 0.5 * ((critic(flat_obs[idx]).view(-1) - flat_ret[idx]) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                    opt.step()
        return critic

    # ── TD(0) training ──
    def train_td(obs, rewards, dones, next_obs, next_done, reward_scale=1.0):
        Tl, El, D = obs.shape
        critic = make_critic()
        opt = optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)

        flat_s = obs.reshape(-1, D)
        flat_r = rewards.reshape(-1) * reward_scale

        flat_ns = torch.zeros_like(obs)
        flat_ns[:-1] = obs[1:]
        flat_ns[-1] = next_obs
        flat_ns = flat_ns.reshape(-1, D)

        flat_d = torch.zeros_like(rewards)
        flat_d[:-1] = dones[1:]
        flat_d[-1] = next_done
        flat_d = flat_d.reshape(-1)

        N = flat_s.shape[0]
        mb = min(args.minibatch_size, N)

        critic.train()
        for _ in range(args.td_epochs):
            perm = torch.randperm(N, device=device)
            for start in range(0, N, mb):
                idx = perm[start:start + mb]
                with torch.no_grad():
                    target = flat_r[idx] + args.gamma * critic(flat_ns[idx]).view(-1) * (1 - flat_d[idx])
                loss = 0.5 * ((critic(flat_s[idx]).view(-1) - target) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                opt.step()

        if reward_scale != 1.0:
            with torch.no_grad():
                critic[-1].weight.div_(reward_scale)
                critic[-1].bias.div_(reward_scale)
        return critic

    # ── Evaluate ──
    def evaluate(critic, obs, mc_gt):
        with torch.no_grad():
            v = critic(obs.reshape(-1, obs.shape[-1])).view(-1).cpu().numpy()
        g = mc_gt.reshape(-1).cpu().numpy()
        return pearsonr(v, g)[0], spearmanr(v, g).correlation

    # ── Run experiments ──
    print(f"\n{'=' * 70}")
    print(f"V Scaling: GAE(iter={args.gae_iters}) vs TD(0) vs TD(0,rs={args.td_reward_scale}) | γ={args.gamma}")
    print(f"{'=' * 70}")

    results = []
    for N in args.rollout_counts:
        trans = N * E * T
        print(f"\n--- N={N} rollouts ({trans:,} transitions) ---")
        obs, rewards, dones, next_obs, next_done, mc_gt = combine(N)

        t0 = time.time()
        c_gae = train_gae(obs, rewards, dones, next_obs, next_done)
        dt_gae = time.time() - t0
        r_gae, rho_gae = evaluate(c_gae, obs, mc_gt)
        print(f"  GAE:        r={r_gae:.4f}  ρ={rho_gae:.4f}  ({dt_gae:.1f}s)")

        t0 = time.time()
        c_td = train_td(obs, rewards, dones, next_obs, next_done, reward_scale=1.0)
        dt_td = time.time() - t0
        r_td, rho_td = evaluate(c_td, obs, mc_gt)
        print(f"  TD(0):      r={r_td:.4f}  ρ={rho_td:.4f}  ({dt_td:.1f}s)")

        t0 = time.time()
        rs = args.td_reward_scale
        c_td_rs = train_td(obs, rewards, dones, next_obs, next_done, reward_scale=rs)
        dt_td_rs = time.time() - t0
        r_tdrs, rho_tdrs = evaluate(c_td_rs, obs, mc_gt)
        print(f"  TD(0,rs{int(rs)}): r={r_tdrs:.4f}  ρ={rho_tdrs:.4f}  ({dt_td_rs:.1f}s)")

        results.append((N, trans, r_gae, rho_gae, r_td, rho_td, r_tdrs, rho_tdrs))
        del obs, rewards, dones, next_obs, next_done, mc_gt, c_gae, c_td, c_td_rs
        torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'=' * 80}")
    print(f"| Rollouts | Trans  | GAE r  | GAE ρ  | TD r   | TD ρ   | TD+rs r | TD+rs ρ |")
    print(f"|----------|--------|--------|--------|--------|--------|---------|---------|")
    for N, trans, rg, rhog, rt, rhot, rtrs, rhotrs in results:
        print(f"| {N:>8} | {trans:>6,} | {rg:.4f} | {rhog:.4f} | {rt:.4f} | {rhot:.4f} | {rtrs:.4f}  | {rhotrs:.4f}  |")
    print(f"{'=' * 80}")

    envs.close()
