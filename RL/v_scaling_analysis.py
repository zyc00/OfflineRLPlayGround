"""V Scaling Analysis: GAE vs TD(0) quality as a function of offline data size.

Compares iterative GAE and TD(0) for learning V from behavior policy rollouts.
Evaluates V quality via Pearson r and Spearman ρ against MC16 ground truth.

MC16 is computed once on a fixed eval set (first rollout), then cached.
All methods are evaluated on this same eval set.

Usage:
  python -u -m RL.v_scaling_analysis
  python -u -m RL.v_scaling_analysis --gamma 0.8
"""

import random
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
    gamma: float = 0.99
    gae_lambda: float = 0.9
    num_envs: int = 100
    num_steps: int = 50
    max_episode_steps: int = 50
    rollout_counts: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100)
    mc_samples: int = 16
    gae_iters: int = 5
    gae_epochs: int = 100
    td_epochs: int = 200
    td_reward_scale: float = 10.0
    ema_tau: float = 0.005
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
    obs_dim = None

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

    # ── State save/restore helpers ──
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

    # ── Collect data pool ──
    T, E = args.num_steps, args.num_envs
    print(f"Collecting {max_rollouts} rollouts ({max_rollouts * E * T:,} transitions)...")
    data_pool = []

    for ri in range(max_rollouts):
        roll_obs = torch.zeros(T, E, obs_dim, device=device)
        roll_rewards = torch.zeros(T, E, device=device)
        roll_dones = torch.zeros(T, E, device=device)

        # Save states only for rollout 0 (eval set)
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
                next_obs, reward, term, trunc, infos = envs.step(clip_action(action))
                next_done = (term | trunc).float()
                roll_rewards[step] = reward.view(-1)

        # MC1 returns (trajectory)
        mc1 = torch.zeros(T, E, device=device)
        future = torch.zeros(E, device=device)
        for t in reversed(range(T)):
            if t == T - 1:
                mask = 1.0 - next_done
            else:
                mask = 1.0 - roll_dones[t + 1]
            future = roll_rewards[t] + args.gamma * future * mask
            mc1[t] = future

        data_pool.append(dict(
            obs=roll_obs, rewards=roll_rewards, dones=roll_dones,
            next_obs=next_obs.clone(), next_done=next_done.clone(),
            mc1_returns=mc1,
            saved_states=saved_states,  # only non-None for ri=0
        ))

        if (ri + 1) % 20 == 0 or ri + 1 == max_rollouts:
            print(f"  {ri + 1}/{max_rollouts} rollouts collected")

    # ── Compute MC16 ground truth on eval set (rollout 0) ──
    print(f"\nComputing MC{args.mc_samples} ground truth for eval set ({E * T:,} states)...")
    num_mc_envs = E * args.mc_samples
    mc_envs_raw = gym.make(args.env_id, num_envs=num_mc_envs, **env_kwargs)
    if isinstance(mc_envs_raw.action_space, gym.spaces.Dict):
        mc_envs_raw = FlattenActionSpaceWrapper(mc_envs_raw)
    mc_envs = ManiSkillVectorEnv(
        mc_envs_raw, num_mc_envs, ignore_terminations=False, record_metrics=True,
    )
    mc_zero_action = torch.zeros(
        num_mc_envs, *envs.single_action_space.shape, device=device
    )

    def _restore_mc_state(state_dict, seed=None):
        mc_envs.reset(seed=seed if seed is not None else args.seed)
        mc_envs.base_env.set_state_dict(state_dict)
        mc_envs.base_env.step(mc_zero_action)  # PhysX contact warmup
        mc_envs.base_env.set_state_dict(state_dict)
        mc_envs.base_env._elapsed_steps[:] = 0
        return mc_envs.base_env.get_obs()

    eval_saved_states = data_pool[0]['saved_states']
    eval_mc16 = torch.zeros(T, E, device=device)

    with torch.no_grad():
        for t in range(T):
            expanded = _expand_state(eval_saved_states[t], args.mc_samples)
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

            eval_mc16[t] = ret.view(E, args.mc_samples).mean(dim=1)

            if (t + 1) % 10 == 0 or t + 1 == T:
                print(f"  MC{args.mc_samples} step {t + 1}/{T}")

    mc_envs.close()
    # Free saved states
    data_pool[0]['saved_states'] = None
    del eval_saved_states

    # Eval set: rollout 0's obs and MC16 returns
    eval_obs = data_pool[0]['obs']  # (T, E, D)
    eval_mc1 = data_pool[0]['mc1_returns']  # (T, E)

    print(f"  MC1 vs MC16 correlation: r={pearsonr(eval_mc1.reshape(-1).cpu().numpy(), eval_mc16.reshape(-1).cpu().numpy())[0]:.4f}")

    # ── Combine rollouts along env axis (for training) ──
    def combine(n):
        d = data_pool[:n]
        return (
            torch.cat([x['obs'] for x in d], dim=1),
            torch.cat([x['rewards'] for x in d], dim=1),
            torch.cat([x['dones'] for x in d], dim=1),
            torch.cat([x['next_obs'] for x in d], dim=0),
            torch.cat([x['next_done'] for x in d], dim=0),
            torch.cat([x['mc1_returns'] for x in d], dim=1),
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

    # ── TD(0) with EMA target network (like IQL) ──
    def train_td_ema(obs, rewards, dones, next_obs, next_done,
                     reward_scale=1.0, ema_tau=0.005):
        Tl, El, D = obs.shape
        critic = make_critic()
        critic_target = make_critic()
        critic_target.load_state_dict(critic.state_dict())
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

    # ── MC1 regression: directly regress on trajectory MC returns ──
    def train_mc(obs, mc_returns):
        Tl, El, D = obs.shape
        critic = make_critic()
        opt = optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)
        flat_obs = obs.reshape(-1, D)
        flat_ret = mc_returns.reshape(-1)
        N = flat_obs.shape[0]
        mb = min(args.minibatch_size, N)

        critic.train()
        for _ in range(args.gae_iters * args.gae_epochs):
            perm = torch.randperm(N, device=device)
            for start in range(0, N, mb):
                idx = perm[start:start + mb]
                loss = 0.5 * ((critic(flat_obs[idx]).view(-1) - flat_ret[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                opt.step()
        return critic

    # ── Evaluate on fixed eval set (rollout 0) against MC16 ──
    def evaluate(critic, eval_obs, mc16_gt):
        with torch.no_grad():
            v = critic(eval_obs.reshape(-1, eval_obs.shape[-1])).view(-1).cpu().numpy()
        g = mc16_gt.reshape(-1).cpu().numpy()
        return pearsonr(v, g)[0], spearmanr(v, g).correlation

    # ── Run experiments ──
    print(f"\n{'=' * 70}")
    print(f"V Scaling (eval=MC{args.mc_samples} on rollout0): GAE vs TD vs MC1 | γ={args.gamma} λ={args.gae_lambda}")
    print(f"{'=' * 70}")

    results = []
    for N in args.rollout_counts:
        trans = N * E * T
        print(f"\n--- N={N} rollouts ({trans:,} transitions) ---")
        obs, rewards, dones, next_obs, next_done, mc1_gt = combine(N)

        t0 = time.time()
        c_mc = train_mc(obs, mc1_gt)
        dt_mc = time.time() - t0
        r_mc, rho_mc = evaluate(c_mc, eval_obs, eval_mc16)
        print(f"  MC1:        r={r_mc:.4f}  ρ={rho_mc:.4f}  ({dt_mc:.1f}s)")

        t0 = time.time()
        c_gae = train_gae(obs, rewards, dones, next_obs, next_done)
        dt_gae = time.time() - t0
        r_gae, rho_gae = evaluate(c_gae, eval_obs, eval_mc16)
        print(f"  GAE:        r={r_gae:.4f}  ρ={rho_gae:.4f}  ({dt_gae:.1f}s)")

        t0 = time.time()
        c_td = train_td(obs, rewards, dones, next_obs, next_done, reward_scale=1.0)
        dt_td = time.time() - t0
        r_td, rho_td = evaluate(c_td, eval_obs, eval_mc16)
        print(f"  TD(0):      r={r_td:.4f}  ρ={rho_td:.4f}  ({dt_td:.1f}s)")

        t0 = time.time()
        rs = args.td_reward_scale
        c_td_rs = train_td(obs, rewards, dones, next_obs, next_done, reward_scale=rs)
        dt_td_rs = time.time() - t0
        r_tdrs, rho_tdrs = evaluate(c_td_rs, eval_obs, eval_mc16)
        print(f"  TD+rs{int(rs)}:    r={r_tdrs:.4f}  ρ={rho_tdrs:.4f}  ({dt_td_rs:.1f}s)")

        t0 = time.time()
        c_ema = train_td_ema(obs, rewards, dones, next_obs, next_done, reward_scale=rs, ema_tau=args.ema_tau)
        dt_ema = time.time() - t0
        r_ema, rho_ema = evaluate(c_ema, eval_obs, eval_mc16)
        print(f"  TD+EMA:     r={r_ema:.4f}  ρ={rho_ema:.4f}  ({dt_ema:.1f}s)")

        results.append((N, trans, r_mc, rho_mc, r_gae, rho_gae, r_td, rho_td, r_tdrs, rho_tdrs, r_ema, rho_ema))
        del obs, rewards, dones, next_obs, next_done, mc1_gt, c_mc, c_gae, c_td, c_td_rs, c_ema
        torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'=' * 105}")
    print(f"Eval against MC{args.mc_samples} | γ={args.gamma} λ={args.gae_lambda}")
    print(f"| Rollouts | Trans   | MC1 r  | MC1 ρ  | GAE r  | GAE ρ  | TD r   | TD ρ   | TD+rs r | TD+rs ρ | EMA r  | EMA ρ  |")
    print(f"|----------|---------|--------|--------|--------|--------|--------|--------|---------|---------|--------|--------|")
    for N, trans, rmc, rhomc, rg, rhog, rt, rhot, rtrs, rhotrs, rema, rhoema in results:
        print(f"| {N:>8} | {trans:>7,} | {rmc:.4f} | {rhomc:.4f} | {rg:.4f} | {rhog:.4f} | {rt:.4f} | {rhot:.4f} | {rtrs:.4f}  | {rhotrs:.4f}  | {rema:.4f} | {rhoema:.4f} |")
    print(f"{'=' * 105}")

    envs.close()
