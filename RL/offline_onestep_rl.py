"""One-step offline RL: Compare V learning methods for policy improvement.

Collects a large offline dataset, learns V via different methods (MC1, TD, TD+EMA, GAE),
computes GAE advantages from each V, then does AWR policy update and evaluates.

Tests whether better V quality (TD+EMA r=0.82 vs MC1 r=0.36 at γ=0.99) translates
to better downstream policy improvement.

Usage:
  python -u -m RL.offline_onestep_rl --gamma 0.99 --num_rollouts 200
"""

import copy
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from data.data_collection.ppo import Agent, layer_init


@dataclass
class Args:
    checkpoint: str = "runs/pickcube_ppo/ckpt_76_logstd-1.5.pt"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_envs: int = 100
    num_eval_envs: int = 128
    num_steps: int = 50
    max_episode_steps: int = 50
    num_rollouts: int = 200
    # V learning
    gae_v_lambda: float = 0.95
    """GAE lambda for V learning (iterative GAE method)"""
    gae_iters: int = 5
    gae_epochs: int = 100
    td_epochs: int = 200
    mc_epochs: int = 500
    td_reward_scale: float = 10.0
    ema_tau: float = 0.005
    critic_hidden: int = 256
    critic_layers: int = 3
    critic_lr: float = 3e-4
    critic_minibatch: int = 1000
    # AWR
    awr_beta: float = 1.0
    awr_max_weight: float = 20.0
    norm_adv: bool = True
    learning_rate: float = 3e-4
    update_epochs: int = 4
    num_minibatches: int = 32
    num_iterations: int = 100
    eval_freq: int = 5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    # Misc
    seed: int = 1
    env_id: str = "PickCube-v1"
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"


if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    T, E = args.num_steps, args.num_envs

    # ── Env setup ──
    env_kwargs = dict(
        obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda",
        reward_mode=args.reward_mode, control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
    )
    envs = gym.make(args.env_id, num_envs=E, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    envs = ManiSkillVectorEnv(envs, E, ignore_terminations=False, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=False, record_metrics=True)

    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    act_shape = envs.single_action_space.shape
    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(a):
        return torch.clamp(a.detach(), action_low, action_high)

    # ── Agent ──
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()

    def make_critic():
        layers = [layer_init(nn.Linear(obs_dim, args.critic_hidden)), nn.Tanh()]
        for _ in range(args.critic_layers - 1):
            layers += [layer_init(nn.Linear(args.critic_hidden, args.critic_hidden)), nn.Tanh()]
        layers.append(layer_init(nn.Linear(args.critic_hidden, 1)))
        return nn.Sequential(*layers).to(device)

    # ══════════════════════════════════════════════════════════════════
    #  1. Collect offline data
    # ══════════════════════════════════════════════════════════════════
    NR = args.num_rollouts
    total_trans = NR * E * T
    print(f"=== One-Step Offline RL: {NR} rollouts ({total_trans:,} transitions) ===")
    print(f"  γ={args.gamma}, GAE λ={args.gae_lambda}, AWR β={args.awr_beta}")
    print(f"  V methods: MC1, TD, TD+rs{int(args.td_reward_scale)}, TD+EMA, GAE(iter{args.gae_iters})")

    # Pre-allocate combined storage
    all_obs = torch.zeros(T, NR * E, obs_dim, device=device)
    all_actions = torch.zeros(T, NR * E, *act_shape, device=device)
    all_rewards = torch.zeros(T, NR * E, device=device)
    all_dones = torch.zeros(T, NR * E, device=device)
    all_next_obs = torch.zeros(NR * E, obs_dim, device=device)
    all_next_done = torch.zeros(NR * E, device=device)
    all_mc1 = torch.zeros(T, NR * E, device=device)

    print(f"\nCollecting {NR} rollouts...")
    t0 = time.time()

    for ri in range(NR):
        sl = slice(ri * E, (ri + 1) * E)
        next_obs, _ = envs.reset(seed=args.seed + ri)
        next_done = torch.zeros(E, device=device)

        with torch.no_grad():
            for step in range(T):
                all_obs[step, sl] = next_obs
                all_dones[step, sl] = next_done
                action = agent.get_action(next_obs, deterministic=False)
                action_clipped = clip_action(action)
                all_actions[step, sl] = action_clipped
                next_obs, reward, term, trunc, _ = envs.step(action_clipped)
                next_done = (term | trunc).float()
                all_rewards[step, sl] = reward.view(-1)

        all_next_obs[sl] = next_obs
        all_next_done[sl] = next_done

        # MC1 returns (trajectory)
        future = torch.zeros(E, device=device)
        for t in reversed(range(T)):
            if t == T - 1:
                mask = 1.0 - next_done
            else:
                mask = 1.0 - all_dones[t + 1, sl]
            future = all_rewards[t, sl] + args.gamma * future * mask
            all_mc1[t, sl] = future

        if (ri + 1) % 50 == 0 or ri + 1 == NR:
            print(f"  {ri + 1}/{NR} rollouts collected")

    envs.close()
    dt_collect = time.time() - t0
    print(f"  Data collection: {dt_collect:.1f}s")

    total_E = NR * E  # combined env dimension

    # ══════════════════════════════════════════════════════════════════
    #  2. Learn V (standalone critic per method)
    # ══════════════════════════════════════════════════════════════════
    print(f"\nLearning V functions...")

    def train_mc(obs, mc_returns, epochs):
        Tl, El, D = obs.shape
        critic = make_critic()
        opt = optim.Adam(critic.parameters(), lr=args.critic_lr, eps=1e-5)
        flat_obs = obs.reshape(-1, D)
        flat_ret = mc_returns.reshape(-1)
        N = flat_obs.shape[0]
        mb = min(args.critic_minibatch, N)
        critic.train()
        for _ in range(epochs):
            perm = torch.randperm(N, device=device)
            for start in range(0, N, mb):
                idx = perm[start:start + mb]
                loss = 0.5 * ((critic(flat_obs[idx]).view(-1) - flat_ret[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                opt.step()
        return critic

    def train_td(obs, rewards, dones, next_obs, next_done, reward_scale=1.0, epochs=200):
        Tl, El, D = obs.shape
        critic = make_critic()
        opt = optim.Adam(critic.parameters(), lr=args.critic_lr, eps=1e-5)
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
        mb = min(args.critic_minibatch, N)
        critic.train()
        for _ in range(epochs):
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

    def train_td_ema(obs, rewards, dones, next_obs, next_done,
                     reward_scale=1.0, ema_tau=0.005, epochs=200):
        Tl, El, D = obs.shape
        critic = make_critic()
        critic_target = make_critic()
        critic_target.load_state_dict(critic.state_dict())
        opt = optim.Adam(critic.parameters(), lr=args.critic_lr, eps=1e-5)
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
        mb = min(args.critic_minibatch, N)
        critic.train()
        for _ in range(epochs):
            perm = torch.randperm(N, device=device)
            for start in range(0, N, mb):
                idx = perm[start:start + mb]
                with torch.no_grad():
                    target = flat_r[idx] + args.gamma * critic_target(flat_ns[idx]).view(-1) * (1 - flat_d[idx])
                loss = 0.5 * ((critic(flat_s[idx]).view(-1) - target) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    for p, pt in zip(critic.parameters(), critic_target.parameters()):
                        pt.data.mul_(1 - ema_tau).add_(p.data, alpha=ema_tau)
        if reward_scale != 1.0:
            with torch.no_grad():
                critic[-1].weight.div_(reward_scale)
                critic[-1].bias.div_(reward_scale)
        return critic

    def train_gae(obs, rewards, dones, next_obs, next_done, gae_lambda):
        Tl, El, D = obs.shape
        critic = make_critic()
        opt = optim.Adam(critic.parameters(), lr=args.critic_lr, eps=1e-5)
        flat_obs = obs.reshape(-1, D)
        N = flat_obs.shape[0]
        mb = min(args.critic_minibatch, N)
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
                    adv[t] = lastgaelam = delta + args.gamma * gae_lambda * nnd * lastgaelam
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

    # Train all V methods
    methods = {}

    t0 = time.time()
    methods["MC1"] = train_mc(all_obs, all_mc1, args.mc_epochs)
    print(f"  MC1:     {time.time() - t0:.1f}s")

    t0 = time.time()
    methods["TD"] = train_td(all_obs, all_rewards, all_dones, all_next_obs, all_next_done,
                             reward_scale=1.0, epochs=args.td_epochs)
    print(f"  TD:      {time.time() - t0:.1f}s")

    rs = args.td_reward_scale
    t0 = time.time()
    methods[f"TD+rs{int(rs)}"] = train_td(all_obs, all_rewards, all_dones, all_next_obs, all_next_done,
                                          reward_scale=rs, epochs=args.td_epochs)
    print(f"  TD+rs:   {time.time() - t0:.1f}s")

    t0 = time.time()
    methods["TD+EMA"] = train_td_ema(all_obs, all_rewards, all_dones, all_next_obs, all_next_done,
                                     reward_scale=rs, ema_tau=args.ema_tau, epochs=args.td_epochs)
    print(f"  TD+EMA:  {time.time() - t0:.1f}s")

    t0 = time.time()
    methods["GAE"] = train_gae(all_obs, all_rewards, all_dones, all_next_obs, all_next_done,
                               gae_lambda=args.gae_v_lambda)
    print(f"  GAE:     {time.time() - t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════
    #  3. Compute GAE advantages per method
    # ══════════════════════════════════════════════════════════════════
    print(f"\nComputing GAE(λ={args.gae_lambda}) advantages per method...")

    def compute_advantages(critic):
        with torch.no_grad():
            values = torch.stack([critic(all_obs[t]).flatten() for t in range(T)])
            nv = critic(all_next_obs).reshape(1, -1)
            adv = torch.zeros_like(all_rewards)
            lastgaelam = 0
            for t in reversed(range(T)):
                if t == T - 1:
                    nnd, nvs = 1.0 - all_next_done, nv
                else:
                    nnd, nvs = 1.0 - all_dones[t + 1], values[t + 1]
                delta = all_rewards[t] + args.gamma * nnd * nvs - values[t]
                adv[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nnd * lastgaelam
        return adv

    method_advantages = {}
    for name, critic in methods.items():
        critic.eval()
        adv = compute_advantages(critic)
        method_advantages[name] = adv
        pos_pct = (adv > 0).float().mean().item()
        print(f"  {name:10s}: mean={adv.mean().item():.4f}, std={adv.std().item():.4f}, pos%={pos_pct:.1%}")

    # Flatten data for AWR
    b_obs = all_obs.reshape((-1,) + envs.single_observation_space.shape)
    b_actions = all_actions.reshape((-1,) + act_shape)
    batch_size = b_obs.shape[0]
    minibatch_size = batch_size // args.num_minibatches

    print(f"\n  Batch: {batch_size:,}, Minibatch: {minibatch_size:,}")

    # ══════════════════════════════════════════════════════════════════
    #  4. Eval helper
    # ══════════════════════════════════════════════════════════════════
    def eval_policy(policy):
        policy.eval()
        eval_obs, _ = eval_envs.reset()
        eval_metrics = defaultdict(list)
        for _ in range(args.max_episode_steps):
            with torch.no_grad():
                eval_obs, _, term, trunc, infos = eval_envs.step(
                    policy.get_action(eval_obs, deterministic=True)
                )
                if "final_info" in infos:
                    mask = infos["_final_info"]
                    for k, v in infos["final_info"]["episode"].items():
                        eval_metrics[k].append(v[mask])
        sr_vals = eval_metrics.get("success_once", [])
        return torch.cat(sr_vals).float().mean().item() if sr_vals else 0.0

    # Baseline SR
    baseline_sr = eval_policy(agent)
    print(f"\nBaseline SR (behavior policy): {baseline_sr:.1%}")

    # ══════════════════════════════════════════════════════════════════
    #  5. AWR policy update per method
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"AWR Training: {args.num_iterations} iterations × {args.update_epochs} epochs")
    print(f"{'=' * 70}")

    all_results = {}

    for method_name, adv in method_advantages.items():
        print(f"\n--- {method_name} ---")

        # Compute AWR weights
        b_adv = adv.reshape(-1)
        if args.norm_adv:
            b_adv_norm = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        else:
            b_adv_norm = b_adv
        b_weights = torch.exp(b_adv_norm / args.awr_beta)
        b_weights = torch.clamp(b_weights, max=args.awr_max_weight)
        print(f"  Weights: mean={b_weights.mean().item():.2f}, max={b_weights.max().item():.2f}")

        # Clone behavior policy
        policy = Agent(envs).to(device)
        policy.load_state_dict(agent.state_dict())

        # Actor-only optimizer
        actor_params = list(policy.actor_mean.parameters()) + [policy.actor_logstd]
        optimizer = optim.Adam(actor_params, lr=args.learning_rate, eps=1e-5)

        b_inds = np.arange(batch_size)
        sr_curve = []

        for iteration in range(1, args.num_iterations + 1):
            # Eval
            if iteration == 1 or iteration % args.eval_freq == 0 or iteration == args.num_iterations:
                sr = eval_policy(policy)
                sr_curve.append((iteration, sr))
                print(f"  Iter {iteration:3d}: SR={sr:.1%}")

            # AWR update
            policy.train()
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    mb_inds = b_inds[start:start + minibatch_size]
                    _, newlogprob, entropy, _ = policy.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    mb_weights = b_weights[mb_inds]
                    mb_weights = mb_weights / mb_weights.sum() * len(mb_weights)
                    policy_loss = -(mb_weights.detach() * newlogprob).mean()
                    entropy_loss = entropy.mean()
                    loss = policy_loss - args.ent_coef * entropy_loss
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(actor_params, args.max_grad_norm)
                    optimizer.step()

        # Final eval
        final_sr = eval_policy(policy)
        if not sr_curve or sr_curve[-1][0] != args.num_iterations:
            sr_curve.append((args.num_iterations, final_sr))

        peak_sr = max(sr for _, sr in sr_curve)
        all_results[method_name] = dict(peak=peak_sr, final=final_sr, curve=sr_curve)
        print(f"  Peak SR={peak_sr:.1%}, Final SR={final_sr:.1%}")

        del policy, optimizer
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"Summary | γ={args.gamma}, GAE λ={args.gae_lambda}, AWR β={args.awr_beta}")
    print(f"  Offline data: {NR} rollouts ({total_trans:,} transitions)")
    print(f"  Baseline SR: {baseline_sr:.1%}")
    print(f"{'=' * 60}")
    print(f"| {'Method':12s} | Peak SR | Final SR |")
    print(f"|{'-' * 14}|---------|----------|")
    for name, res in all_results.items():
        print(f"| {name:12s} | {res['peak']:6.1%} | {res['final']:7.1%} |")
    print(f"{'=' * 60}")

    # Per-iteration table
    print(f"\nPer-iteration SR:")
    iters_set = sorted(set(it for res in all_results.values() for it, _ in res['curve']))
    header = f"| {'Iter':>4s} |" + "|".join(f" {n:>10s} " for n in all_results) + "|"
    print(header)
    print("|" + "-" * 6 + "|" + "|".join("-" * 12 for _ in all_results) + "|")
    for it in iters_set:
        row = f"| {it:4d} |"
        for name, res in all_results.items():
            sr_at_it = next((sr for i, sr in res['curve'] if i == it), None)
            row += f" {sr_at_it:9.1%}  |" if sr_at_it is not None else f" {'—':>9s}  |"
        print(row)

    eval_envs.close()
