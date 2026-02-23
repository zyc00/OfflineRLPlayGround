"""IQL training curve: V quality vs epoch, with TD-N for Q updates + reward scaling.
Compare: IQL vanilla, IQL+rs10, IQL+rs10+TD-N, and TD+EMA baseline.

Usage:
  python -u -m RL.iql_curve --gamma 0.99
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


@dataclass
class Args:
    checkpoint: str = "runs/pickcube_ppo/ckpt_76_logstd-1.5.pt"
    optimal_checkpoint: str = "runs/pickcube_ppo/ckpt_301.pt"
    gamma: float = 0.99
    num_envs: int = 100
    num_steps: int = 50
    max_episode_steps: int = 50
    mc_samples: int = 16
    # IQL
    iql_epochs: int = 500
    iql_lr: float = 3e-4
    iql_batch_size: int = 1000
    expectile_tau: float = 0.7
    ema_tau: float = 0.005
    td_n_steps: tuple[int, ...] = (1, 5, 10, 50)
    """N-step TD for Q target. 1=standard TD(0), 50=full MC."""
    reward_scales: tuple[float, ...] = (1.0, 10.0)
    # Shared
    critic_layers: int = 3
    hidden_dim: int = 256
    eval_every: int = 5
    num_rollouts: int = 100
    seed: int = 1
    env_id: str = "PickCube-v1"
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"
    output: str = "runs/iql_curve.png"


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, n_layers=3):
        super().__init__()
        layers = [layer_init(nn.Linear(state_dim + action_dim, hidden_dim)), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh()]
        layers.append(layer_init(nn.Linear(hidden_dim, 1)))
        self.net = nn.Sequential(*layers)

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


def expectile_loss(diff, tau):
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return (weight * (diff ** 2)).mean()


if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda")

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

    def make_v_net():
        layers = [layer_init(nn.Linear(obs_dim, args.hidden_dim)), nn.Tanh()]
        for _ in range(args.critic_layers - 1):
            layers += [layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)), nn.Tanh()]
        layers.append(layer_init(nn.Linear(args.hidden_dim, 1)))
        return nn.Sequential(*layers).to(device)

    def make_critic():
        return make_v_net()  # same architecture

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

    # ── Phase 1: Collect rollouts (keep trajectory structure) ──
    print(f"Phase 1: Collecting {args.num_rollouts} rollouts...")
    sys.stdout.flush()
    t0 = time.time()

    # Store as (T, E*num_rollouts, D) to keep trajectory structure
    all_obs = torch.zeros(T, E * args.num_rollouts, obs_dim, device=device)
    all_actions = torch.zeros(T, E * args.num_rollouts, action_dim, device=device)
    all_rewards = torch.zeros(T, E * args.num_rollouts, device=device)
    all_dones = torch.zeros(T, E * args.num_rollouts, device=device)
    all_next_obs = torch.zeros(args.num_rollouts, E, obs_dim, device=device)
    all_next_done = torch.zeros(args.num_rollouts, E, device=device)
    saved_states_0 = []

    for ri in range(args.num_rollouts):
        next_obs, _ = envs.reset(seed=args.seed + ri)
        next_done = torch.zeros(E, device=device)

        with torch.no_grad():
            for step in range(T):
                if ri == 0:
                    saved_states_0.append(_clone_state(envs.base_env.get_state_dict()))
                col = slice(ri * E, (ri + 1) * E)
                all_obs[step, col] = next_obs
                all_dones[step, col] = next_done
                action = agent.get_action(next_obs, deterministic=False)
                all_actions[step, col] = action
                next_obs, reward, term, trunc, _ = envs.step(clip_action(action))
                next_done = (term | trunc).float()
                all_rewards[step, col] = reward.view(-1)

        all_next_obs[ri] = next_obs
        all_next_done[ri] = next_done

        if (ri + 1) % 20 == 0 or ri + 1 == args.num_rollouts:
            print(f"  {ri + 1}/{args.num_rollouts}")
            sys.stdout.flush()

    # Reshape next_obs/next_done to match trajectory structure
    final_next_obs = all_next_obs.reshape(-1, obs_dim)  # (E*num_rollouts, D)
    final_next_done = all_next_done.reshape(-1)  # (E*num_rollouts,)

    print(f"  Phase 1 done ({time.time() - t0:.1f}s)")
    print(f"  Data shape: obs={all_obs.shape}, rewards={all_rewards.shape}")

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

    on_mc16 = torch.zeros(T, E, device=device)

    with torch.no_grad():
        for t in range(T):
            expanded = _expand_state(saved_states_0[t], samples_per_env)
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
    del mc_envs, mc_envs_raw, saved_states_0
    torch.cuda.empty_cache()

    print(f"  On-policy MC16: mean={on_mc16.mean():.4f}, std={on_mc16.std():.4f}")
    print(f"  Phase 2 done ({time.time() - t0:.1f}s)")

    eval_obs = all_obs[:, :E, :]  # rollout 0 states (T, E, D)
    mc16_flat = on_mc16.reshape(-1).cpu().numpy()

    envs.close()
    del envs, agent, optimal_agent
    torch.cuda.empty_cache()

    def compute_r(critic, scale=1.0):
        with torch.no_grad():
            v = critic(eval_obs.reshape(-1, obs_dim)).view(-1).cpu().numpy() / scale
        return pearsonr(v, mc16_flat)[0]

    # ── Precompute n-step Q targets for all n values ──
    # For each (t, env), compute: sum_{k=0}^{n-1} gamma^k * r_{t+k} + gamma^n * V(s_{t+n})
    # The V(s_{t+n}) part depends on the current V network, so we store n-step RETURNS
    # (without the bootstrap) and add V bootstrap at training time.
    #
    # n_step_ret[n][(t, env)] = sum_{k=0}^{n-1} gamma^k * r_{t+k}
    # n_step_ns[n][(t, env)] = s_{t+n}  (the state to bootstrap from)
    # n_step_nd[n][(t, env)] = done at step t+n-1 (whether episode ended before n steps)
    # n_step_mask[n][(t, env)] = gamma^n * (1 - done_before_n)

    E_total = E * args.num_rollouts

    print(f"\nPhase 3: Precomputing n-step returns...")
    t0 = time.time()

    # Build next_obs for each step: obs[t+1] for t<T-1, final_next_obs for t=T-1
    next_obs_all = torch.zeros_like(all_obs)
    next_obs_all[:-1] = all_obs[1:]
    next_obs_all[-1] = final_next_obs.unsqueeze(0)

    next_done_all = torch.zeros_like(all_rewards)
    next_done_all[:-1] = all_dones[1:]
    next_done_all[-1] = final_next_done.unsqueeze(0)

    nstep_data = {}
    for n in args.td_n_steps:
        # For each (t, env), compute n-step partial return and bootstrap state
        nstep_ret = torch.zeros(T, E_total, device=device)
        nstep_bootstrap_obs = torch.zeros(T, E_total, obs_dim, device=device)
        nstep_bootstrap_mask = torch.zeros(T, E_total, device=device)  # gamma^n * (not done)

        for t in range(T):
            cumul = torch.zeros(E_total, device=device)
            discount = torch.ones(E_total, device=device)
            alive = torch.ones(E_total, device=device)  # track if episode is still going

            for k in range(n):
                step = t + k
                if step < T:
                    cumul += discount * alive * all_rewards[step]
                    if k < n - 1 and step + 1 < T:
                        alive *= (1.0 - all_dones[step + 1])
                    elif k < n - 1 and step + 1 == T:
                        alive *= (1.0 - final_next_done)
                    discount *= args.gamma
                else:
                    # Past end of trajectory — use final state
                    break

            nstep_ret[t] = cumul

            # Bootstrap state: s_{t+n} if t+n < T, else final_next_obs
            boot_step = min(t + n, T)
            if boot_step < T:
                nstep_bootstrap_obs[t] = all_obs[boot_step]
            else:
                nstep_bootstrap_obs[t] = final_next_obs

            # Bootstrap mask: gamma^n * alive
            nstep_bootstrap_mask[t] = discount * alive

        nstep_data[n] = dict(
            ret=nstep_ret,           # (T, E_total)
            boot_obs=nstep_bootstrap_obs,  # (T, E_total, D)
            boot_mask=nstep_bootstrap_mask, # (T, E_total) = gamma^n * (1-done)
        )
        print(f"  TD-{n}: ret range [{nstep_ret.min():.4f}, {nstep_ret.max():.4f}], "
              f"mask mean={nstep_bootstrap_mask.mean():.4f}")

    print(f"  Phase 3 done ({time.time() - t0:.1f}s)")

    # ── Flatten for training ──
    flat_s = all_obs.reshape(-1, obs_dim)         # (T*E_total, D)
    flat_a = all_actions.reshape(-1, action_dim)   # (T*E_total, A)
    N = flat_s.shape[0]

    # ── TD+EMA baseline (for reference) ──
    def train_td_ema_curve(reward_scale=10.0):
        flat_r = all_rewards.reshape(-1) * reward_scale
        flat_ns = next_obs_all.reshape(-1, obs_dim)
        flat_nd = next_done_all.reshape(-1)
        mb = min(args.iql_batch_size, N)

        critic = make_critic()
        critic_target = copy.deepcopy(critic)
        opt = optim.Adam(critic.parameters(), lr=args.iql_lr, eps=1e-5)

        epochs, r_values = [], []
        for epoch in range(args.iql_epochs):
            critic.train()
            perm = torch.randperm(N, device=device)
            for start in range(0, N, mb):
                idx = perm[start:start + mb]
                with torch.no_grad():
                    target = flat_r[idx] + args.gamma * critic_target(flat_ns[idx]).view(-1) * (1 - flat_nd[idx])
                loss = 0.5 * ((critic(flat_s[idx]).view(-1) - target) ** 2).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5); opt.step()
                with torch.no_grad():
                    for p, pt in zip(critic.parameters(), critic_target.parameters()):
                        pt.data.mul_(1 - args.ema_tau).add_(p.data, alpha=args.ema_tau)

            if (epoch + 1) % args.eval_every == 0 or epoch == 0:
                critic.eval()
                r = compute_r(critic, reward_scale)
                epochs.append(epoch + 1)
                r_values.append(r)

        return epochs, r_values

    # ── IQL with n-step Q + reward scaling ──
    def train_iql_curve(reward_scale=1.0, td_n=1, tau=0.7):
        nd = nstep_data[td_n]
        flat_nstep_ret = nd['ret'].reshape(-1) * reward_scale   # (N,)
        flat_boot_obs = nd['boot_obs'].reshape(-1, obs_dim)     # (N, D)
        flat_boot_mask = nd['boot_mask'].reshape(-1)             # (N,)

        mb = min(args.iql_batch_size, N)

        q_net = QNetwork(obs_dim, action_dim, args.hidden_dim, args.critic_layers).to(device)
        q_target = copy.deepcopy(q_net)
        v_net = make_v_net()

        q_opt = optim.Adam(q_net.parameters(), lr=args.iql_lr, eps=1e-5)
        v_opt = optim.Adam(v_net.parameters(), lr=args.iql_lr, eps=1e-5)

        epochs, r_values = [], []

        for epoch in range(args.iql_epochs):
            q_net.train(); v_net.train()
            perm = torch.randperm(N, device=device)

            for start in range(0, N, mb):
                idx = perm[start:start + mb]
                s = flat_s[idx]
                a = flat_a[idx]

                # Q update: n-step TD target
                # Q_target = sum_{k=0}^{n-1} gamma^k * r_{t+k} + gamma^n * V(s_{t+n})
                with torch.no_grad():
                    v_boot = v_net(flat_boot_obs[idx]).squeeze(-1)
                    q_tgt = flat_nstep_ret[idx] + flat_boot_mask[idx] * v_boot
                q_pred = q_net(s, a).squeeze(-1)
                q_loss = 0.5 * ((q_pred - q_tgt) ** 2).mean()
                q_opt.zero_grad(); q_loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 0.5); q_opt.step()

                # V update: expectile regression on Q_target(s, a)
                with torch.no_grad():
                    q_val = q_target(s, a).squeeze(-1)
                v_pred = v_net(s).squeeze(-1)
                v_loss = expectile_loss(q_val - v_pred, tau)
                v_opt.zero_grad(); v_loss.backward()
                nn.utils.clip_grad_norm_(v_net.parameters(), 0.5); v_opt.step()

                # EMA update for Q target
                with torch.no_grad():
                    for p, pt in zip(q_net.parameters(), q_target.parameters()):
                        pt.data.mul_(1 - args.ema_tau).add_(p.data, alpha=args.ema_tau)

            if (epoch + 1) % args.eval_every == 0 or epoch == 0:
                v_net.eval()
                r = compute_r(v_net, reward_scale)
                epochs.append(epoch + 1)
                r_values.append(r)

        return epochs, r_values

    # ── Run experiments ──
    print(f"\n{'=' * 70}")
    print(f"IQL Curve Experiments: {args.iql_epochs} epochs, eval every {args.eval_every}")
    print(f"  N={args.num_rollouts} rollouts, hidden={args.hidden_dim}")
    print(f"  td_n_steps={args.td_n_steps}, reward_scales={args.reward_scales}")
    print(f"{'=' * 70}\n")

    results = {}

    # TD+EMA baseline
    print("--- TD+EMA rs=10 (baseline) ---")
    t0 = time.time()
    ep, rv = train_td_ema_curve(reward_scale=10.0)
    peak = max(rv); peak_ep = ep[rv.index(peak)]
    print(f"  peak={peak:.4f}@ep{peak_ep}, final={rv[-1]:.4f} ({time.time()-t0:.1f}s)")
    results["TD+EMA_rs10"] = dict(epochs=ep, r_values=rv, peak_r=peak, peak_ep=peak_ep, final_r=rv[-1])

    # IQL variants
    for rs in args.reward_scales:
        for td_n in args.td_n_steps:
            label = f"IQL_rs{int(rs)}_n{td_n}"
            print(f"--- {label} ---")
            sys.stdout.flush()
            t0 = time.time()
            ep, rv = train_iql_curve(reward_scale=rs, td_n=td_n, tau=args.expectile_tau)
            peak = max(rv); peak_ep = ep[rv.index(peak)]
            print(f"  peak={peak:.4f}@ep{peak_ep}, final={rv[-1]:.4f} ({time.time()-t0:.1f}s)")
            results[label] = dict(epochs=ep, r_values=rv, peak_r=peak, peak_ep=peak_ep, final_r=rv[-1])
            sys.stdout.flush()

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"| {'Method':>20} | {'peak_r':>7} | {'peak_ep':>8} | {'final_r':>7} |")
    print(f"|----------------------|---------|----------|---------|")
    for label in results:
        c = results[label]
        print(f"| {label:>20} | {c['peak_r']:>7.4f} | {c['peak_ep']:>8} | {c['final_r']:>7.4f} |")

    # ── Plot ──
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # TD+EMA in black dashed
    c = results["TD+EMA_rs10"]
    ax.plot(c['epochs'], c['r_values'], 'k--', lw=2,
            label=f"TD+EMA rs10 (pk={c['peak_r']:.3f})")

    # IQL variants with colors
    cmap = plt.cm.tab10
    ci = 0
    for label in results:
        if label.startswith("IQL"):
            c = results[label]
            ax.plot(c['epochs'], c['r_values'], color=cmap(ci), lw=1.5,
                    label=f"{label} (pk={c['peak_r']:.3f})")
            ci += 1

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pearson r (vs on-policy MC16)")
    ax.set_title(f"IQL V Quality: TD-N + Reward Scaling | N={args.num_rollouts}, gamma={args.gamma}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {args.output}")

    torch.save(results, args.output.replace('.png', '.pt'))
    print(f"Saved raw data to {args.output.replace('.png', '.pt')}")
