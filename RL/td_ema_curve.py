"""Training curve: V quality vs epoch for TD+EMA, MC1, GAE, IQL across data sizes.
Logs Pearson r against on-policy MC16 at regular intervals during training.

Usage:
  python -u -m RL.td_ema_curve --gamma 0.99
  python -u -m RL.td_ema_curve --gamma 0.99 --methods TD+EMA IQL
"""

import copy
import os
import random
import sys
import time
from dataclasses import dataclass

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from scipy.stats import pearsonr
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from data.data_collection.ppo import Agent, layer_init


def expectile_loss(diff, tau):
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return (weight * (diff ** 2)).mean()


@dataclass
class Args:
    checkpoint: str = "runs/pickcube_ppo/ckpt_76_logstd-1.5.pt"
    optimal_checkpoint: str = "runs/pickcube_ppo/ckpt_301.pt"
    gamma: float = 0.99
    num_envs: int = 100
    num_steps: int = 50
    max_episode_steps: int = 50
    mc_samples: int = 16
    td_reward_scale: float = 10.0
    ema_tau: float = 0.005
    gae_lambda: float = 0.95
    gae_iters: int = 5
    gae_epochs: int = 100
    critic_layers: int = 3
    lr: float = 3e-4
    minibatch_size: int = 1000
    td_epochs: int = 500
    """Max epochs for all data sizes."""
    eval_every: int = 5
    """Evaluate V quality every N epochs."""
    rollout_counts: tuple[int, ...] = (1, 5, 10, 20, 50, 100)
    methods: tuple[str, ...] = ("TD+EMA", "MC1", "GAE")
    # n-step TD
    td_n_step: int = 1
    """N-step for TD+EMA. 1=standard TD(0), 10=10-step, etc."""
    # IQL
    expectile_tau: float = 0.5
    """Expectile for IQL V loss. 0.5=mean, >0.5=upper expectile."""
    iql_td_n: int = 10
    """N-step TD for IQL Q target. 1=standard TD(0), 50=full MC."""
    seed: int = 1
    env_id: str = "PickCube-v1"
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"
    output: str = "runs/td_ema_curve.png"


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

    def make_q_net(hidden_dim=256):
        layers = [layer_init(nn.Linear(obs_dim + action_dim, hidden_dim)), nn.Tanh()]
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

    # ── Phase 1: Collect rollouts (store actions for IQL) ──
    print(f"Phase 1: Collecting {max_rollouts} rollouts...")
    sys.stdout.flush()
    t0 = time.time()
    data_pool = []

    for ri in range(max_rollouts):
        roll_obs = torch.zeros(T, E, obs_dim, device=device)
        roll_actions = torch.zeros(T, E, action_dim, device=device)
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
                roll_actions[step] = action
                next_obs, reward, term, trunc, _ = envs.step(clip_action(action))
                next_done = (term | trunc).float()
                roll_rewards[step] = reward.view(-1)

        # MC1 returns
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
            obs=roll_obs, actions=roll_actions, rewards=roll_rewards, dones=roll_dones,
            next_obs=next_obs.clone(), next_done=next_done.clone(),
            mc1_returns=mc1, saved_states=saved_states,
        ))
        if (ri + 1) % 20 == 0 or ri + 1 == max_rollouts:
            print(f"  {ri + 1}/{max_rollouts}")
            sys.stdout.flush()

    print(f"  Phase 1 done ({time.time() - t0:.1f}s)")

    # ── Phase 2: On-policy MC16 ground truth ──
    samples_per_env = args.mc_samples
    num_mc_envs = E * samples_per_env
    print(f"\nPhase 2: On-policy MC16 ({num_mc_envs} mc_envs)...")
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
    mc16_flat = on_mc16.reshape(-1).cpu().numpy()

    envs.close()
    del envs, agent, optimal_agent
    torch.cuda.empty_cache()

    def combine(n):
        d = data_pool[:n]
        return (
            torch.cat([x['obs'] for x in d], dim=1),
            torch.cat([x['actions'] for x in d], dim=1),
            torch.cat([x['rewards'] for x in d], dim=1),
            torch.cat([x['dones'] for x in d], dim=1),
            torch.cat([x['next_obs'] for x in d], dim=0),
            torch.cat([x['next_done'] for x in d], dim=0),
            torch.cat([x['mc1_returns'] for x in d], dim=1),
        )

    def compute_r(critic, scale=1.0):
        with torch.no_grad():
            v = critic(eval_obs.reshape(-1, obs_dim)).view(-1).cpu().numpy() / scale
        return pearsonr(v, mc16_flat)[0]

    # ── TD+EMA with periodic eval (supports n-step) ──
    def train_td_ema(obs, actions, rewards, dones, next_obs, next_done, mc1_ret):
        Tl, El, D = obs.shape
        rs = args.td_reward_scale
        n = args.td_n_step

        flat_s = obs.reshape(-1, D)

        if n == 1:
            # Standard 1-step TD
            flat_r = rewards.reshape(-1) * rs
            flat_ns = torch.zeros_like(obs)
            flat_ns[:-1] = obs[1:]
            flat_ns[-1] = next_obs
            flat_ns = flat_ns.reshape(-1, D)
            flat_d = torch.zeros_like(rewards)
            flat_d[:-1] = dones[1:]
            flat_d[-1] = next_done
            flat_d = flat_d.reshape(-1)
            flat_nstep_ret = flat_r
            flat_boot_obs = flat_ns
            flat_boot_mask = args.gamma * (1 - flat_d)
        else:
            # N-step TD: target = sum_{k=0}^{n-1} gamma^k * r_{t+k} + gamma^n * V(s_{t+n})
            nstep_ret = torch.zeros(Tl, El, device=device)
            nstep_boot_obs = torch.zeros(Tl, El, D, device=device)
            nstep_boot_mask = torch.zeros(Tl, El, device=device)

            for t in range(Tl):
                cumul = torch.zeros(El, device=device)
                discount = torch.ones(El, device=device)
                alive = torch.ones(El, device=device)
                for k in range(n):
                    step = t + k
                    if step < Tl:
                        cumul += discount * alive * rewards[step]
                        if k < n - 1 and step + 1 < Tl:
                            alive *= (1.0 - dones[step + 1])
                        elif k < n - 1 and step + 1 == Tl:
                            alive *= (1.0 - next_done)
                        discount *= args.gamma
                    else:
                        break
                nstep_ret[t] = cumul * rs
                boot_step = min(t + n, Tl)
                if boot_step < Tl:
                    nstep_boot_obs[t] = obs[boot_step]
                else:
                    nstep_boot_obs[t] = next_obs
                nstep_boot_mask[t] = discount * alive

            flat_nstep_ret = nstep_ret.reshape(-1)
            flat_boot_obs = nstep_boot_obs.reshape(-1, D)
            flat_boot_mask = nstep_boot_mask.reshape(-1)

        N = flat_s.shape[0]
        mb = min(args.minibatch_size, N)

        critic = make_critic()
        critic_target = make_critic()
        critic_target.load_state_dict(critic.state_dict())
        opt = optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)

        epochs, r_values, grad_steps_log = [], [], []
        total_steps = 0

        for epoch in range(args.td_epochs):
            critic.train()
            perm = torch.randperm(N, device=device)
            for start in range(0, N, mb):
                idx = perm[start:start + mb]
                with torch.no_grad():
                    v_boot = critic_target(flat_boot_obs[idx]).view(-1)
                    target = flat_nstep_ret[idx] + flat_boot_mask[idx] * v_boot
                loss = 0.5 * ((critic(flat_s[idx]).view(-1) - target) ** 2).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5); opt.step()
                with torch.no_grad():
                    for p, pt in zip(critic.parameters(), critic_target.parameters()):
                        pt.data.mul_(1 - args.ema_tau).add_(p.data, alpha=args.ema_tau)
                total_steps += 1

            if (epoch + 1) % args.eval_every == 0 or epoch == 0:
                critic.eval()
                r = compute_r(critic, args.td_reward_scale)
                epochs.append(epoch + 1)
                r_values.append(r)
                grad_steps_log.append(total_steps)

        return epochs, r_values, grad_steps_log

    # ── MC1 with periodic eval ──
    def train_mc1(obs, actions, rewards, dones, next_obs, next_done, mc1_ret):
        D = obs.shape[-1]
        flat_obs = obs.reshape(-1, D)
        flat_ret = mc1_ret.reshape(-1)
        N = flat_obs.shape[0]
        mb = min(args.minibatch_size, N)

        critic = make_critic()
        opt = optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)

        epochs, r_values, grad_steps_log = [], [], []
        total_steps = 0

        for epoch in range(args.td_epochs):
            critic.train()
            perm = torch.randperm(N, device=device)
            for start in range(0, N, mb):
                idx = perm[start:start + mb]
                loss = 0.5 * ((critic(flat_obs[idx]).view(-1) - flat_ret[idx]) ** 2).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5); opt.step()
                total_steps += 1

            if (epoch + 1) % args.eval_every == 0 or epoch == 0:
                critic.eval()
                r = compute_r(critic)
                epochs.append(epoch + 1)
                r_values.append(r)
                grad_steps_log.append(total_steps)

        return epochs, r_values, grad_steps_log

    # ── GAE with periodic eval ──
    def train_gae(obs, actions, rewards, dones, next_obs, next_done, mc1_ret):
        Tl, El, D = obs.shape
        flat_obs = obs.reshape(-1, D)
        N = flat_obs.shape[0]
        mb = min(args.minibatch_size, N)

        critic = make_critic()
        opt = optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)

        epochs, r_values, grad_steps_log = [], [], []
        total_steps = 0
        total_epoch = 0

        for gae_iter in range(args.gae_iters):
            # Recompute GAE targets
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

            # Fit critic to GAE targets
            for ep in range(args.gae_epochs):
                critic.train()
                perm = torch.randperm(N, device=device)
                for start in range(0, N, mb):
                    idx = perm[start:start + mb]
                    loss = 0.5 * ((critic(flat_obs[idx]).view(-1) - flat_ret[idx]) ** 2).mean()
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), 0.5); opt.step()
                    total_steps += 1
                total_epoch += 1

                if total_epoch % args.eval_every == 0 or total_epoch == 1:
                    critic.eval()
                    r = compute_r(critic)
                    epochs.append(total_epoch)
                    r_values.append(r)
                    grad_steps_log.append(total_steps)

        return epochs, r_values, grad_steps_log

    # ── IQL with n-step Q target + periodic eval ──
    def train_iql(obs, actions, rewards, dones, next_obs, next_done, mc1_ret):
        Tl, El, D = obs.shape
        rs = args.td_reward_scale
        tau = args.expectile_tau
        n = args.iql_td_n

        flat_s = obs.reshape(-1, D)
        flat_a = actions.reshape(-1, action_dim)

        # Precompute n-step returns and bootstrap info
        # For each (t, env): nstep_ret = sum_{k=0}^{n-1} gamma^k * r_{t+k}
        #                     boot_obs = s_{t+n}
        #                     boot_mask = gamma^n * (alive through n steps)
        nstep_ret = torch.zeros(Tl, El, device=device)
        nstep_boot_obs = torch.zeros(Tl, El, D, device=device)
        nstep_boot_mask = torch.zeros(Tl, El, device=device)

        for t in range(Tl):
            cumul = torch.zeros(El, device=device)
            discount = torch.ones(El, device=device)
            alive = torch.ones(El, device=device)

            for k in range(n):
                step = t + k
                if step < Tl:
                    cumul += discount * alive * rewards[step]
                    if k < n - 1 and step + 1 < Tl:
                        alive *= (1.0 - dones[step + 1])
                    elif k < n - 1 and step + 1 == Tl:
                        alive *= (1.0 - next_done)
                    discount *= args.gamma
                else:
                    break

            nstep_ret[t] = cumul * rs

            boot_step = min(t + n, Tl)
            if boot_step < Tl:
                nstep_boot_obs[t] = obs[boot_step]
            else:
                nstep_boot_obs[t] = next_obs

            nstep_boot_mask[t] = discount * alive

        flat_nstep_ret = nstep_ret.reshape(-1)
        flat_boot_obs = nstep_boot_obs.reshape(-1, D)
        flat_boot_mask = nstep_boot_mask.reshape(-1)

        N = flat_s.shape[0]
        mb = min(args.minibatch_size, N)

        q_net = make_q_net()
        q_target = make_q_net()
        q_target.load_state_dict(q_net.state_dict())
        v_net = make_critic()

        q_opt = optim.Adam(q_net.parameters(), lr=args.lr, eps=1e-5)
        v_opt = optim.Adam(v_net.parameters(), lr=args.lr, eps=1e-5)

        epochs, r_values, grad_steps_log = [], [], []
        total_steps = 0

        for epoch in range(args.td_epochs):
            q_net.train(); v_net.train()
            perm = torch.randperm(N, device=device)
            for start in range(0, N, mb):
                idx = perm[start:start + mb]
                s_b = flat_s[idx]
                a_b = flat_a[idx]
                sa_b = torch.cat([s_b, a_b], dim=-1)

                # Q loss: n-step TD target
                # Q(s,a) → sum gamma^k r_{t+k} * rs + gamma^n * V(s_{t+n}) * alive
                with torch.no_grad():
                    v_boot = v_net(flat_boot_obs[idx]).view(-1)
                    q_tgt = flat_nstep_ret[idx] + flat_boot_mask[idx] * v_boot
                q_pred = q_net(sa_b).view(-1)
                q_loss = 0.5 * ((q_pred - q_tgt) ** 2).mean()
                q_opt.zero_grad(); q_loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 0.5); q_opt.step()

                # V loss: expectile regression on Q_target(s,a) - V(s)
                with torch.no_grad():
                    q_val = q_target(sa_b).view(-1)
                v_pred = v_net(s_b).view(-1)
                v_loss = expectile_loss(q_val - v_pred, tau)
                v_opt.zero_grad(); v_loss.backward()
                nn.utils.clip_grad_norm_(v_net.parameters(), 0.5); v_opt.step()

                # Polyak update Q_target
                with torch.no_grad():
                    for p, pt in zip(q_net.parameters(), q_target.parameters()):
                        pt.data.mul_(1 - args.ema_tau).add_(p.data, alpha=args.ema_tau)
                total_steps += 1

            if (epoch + 1) % args.eval_every == 0 or epoch == 0:
                v_net.eval()
                r = compute_r(v_net, rs)
                epochs.append(epoch + 1)
                r_values.append(r)
                grad_steps_log.append(total_steps)

        return epochs, r_values, grad_steps_log

    # ── Method dispatch ──
    method_fns = {
        "TD+EMA": train_td_ema,
        "MC1": train_mc1,
        "GAE": train_gae,
        "IQL": train_iql,
    }

    # ── Run for each data size x method ──
    print(f"\n{'=' * 70}")
    print(f"Training Curves: methods={args.methods}")
    print(f"  TD+EMA: {args.td_epochs} epochs, rs={args.td_reward_scale}, ema_tau={args.ema_tau}, n_step={args.td_n_step}")
    print(f"  MC1: {args.td_epochs} epochs")
    print(f"  GAE: {args.gae_iters} iters x {args.gae_epochs} epochs = {args.gae_iters * args.gae_epochs} total")
    if "IQL" in args.methods:
        print(f"  IQL: {args.td_epochs} epochs, rs={args.td_reward_scale}, tau={args.expectile_tau}, td_n={args.iql_td_n}")
    print(f"  eval every {args.eval_every} epochs")
    print(f"{'=' * 70}\n")

    # all_curves[(N, method)] = {epochs, r_values, grad_steps, peak_r, ...}
    all_curves = {}
    for N_roll in args.rollout_counts:
        trans = N_roll * E * T
        steps_per_ep = trans // args.minibatch_size
        print(f"--- N={N_roll} ({trans:,} trans, {steps_per_ep} steps/ep) ---")
        sys.stdout.flush()

        obs, actions, rewards, dones, next_obs, next_done, mc1_ret = combine(N_roll)

        for method in args.methods:
            t0 = time.time()
            fn = method_fns[method]
            epochs, r_vals, grad_steps = fn(obs, actions, rewards, dones, next_obs, next_done, mc1_ret)
            peak_r = max(r_vals)
            peak_idx = r_vals.index(peak_r)
            peak_ep = epochs[peak_idx]
            peak_steps = grad_steps[peak_idx]
            final_r = r_vals[-1]
            print(f"  {method:>7}: peak={peak_r:.4f}@ep{peak_ep}({peak_steps}steps) final={final_r:.4f} ({time.time()-t0:.1f}s)")
            sys.stdout.flush()

            all_curves[(N_roll, method)] = dict(
                epochs=epochs, r_values=r_vals, grad_steps=grad_steps,
                peak_r=peak_r, peak_epoch=peak_ep, peak_steps=peak_steps, final_r=final_r)

        del obs, actions, rewards, dones, next_obs, next_done, mc1_ret
        torch.cuda.empty_cache()

    # ── Summary table ──
    print(f"\n{'=' * 70}")
    print(f"SUMMARY — peak r (peak_epoch)")
    print(f"{'=' * 70}")
    header = f"| {'N':>5} |"
    for m in args.methods:
        header += f" {m:>20} |"
    print(header)
    print("|-------|" + "----------------------|" * len(args.methods))
    for n in args.rollout_counts:
        row = f"| {n:>5} |"
        for m in args.methods:
            c = all_curves[(n, m)]
            row += f" {c['peak_r']:.4f}@ep{c['peak_epoch']:<5} |"
        print(row)

    # ── Plot ──
    fig, axes = plt.subplots(1, len(args.methods), figsize=(6 * len(args.methods), 5))
    if len(args.methods) == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(args.rollout_counts)))

    for ax, method in zip(axes, args.methods):
        for i, n in enumerate(args.rollout_counts):
            c = all_curves[(n, method)]
            ax.plot(c['epochs'], c['r_values'], color=colors[i], lw=1.5,
                    label=f"N={n} (pk={c['peak_r']:.2f}@{c['peak_epoch']})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Pearson r (vs on-policy MC16)")
        ax.set_title(method)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 0.9)

    plt.suptitle(f"V Training Curves | gamma={args.gamma} | hidden=256",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {args.output}")

    torch.save(all_curves, args.output.replace('.png', '.pt'))
    print(f"Saved raw data to {args.output.replace('.png', '.pt')}")
