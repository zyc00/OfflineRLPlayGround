"""Per-timestep V error analysis: bias, variance, MAE vs episode timestep.

Hypothesis: states later in the episode (further from root) have higher
V prediction error because the trajectory tree diverges.

Usage:
  python -u -m RL.v_timestep_analysis --gamma 0.99
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
    # Training
    epochs: int = 500
    lr: float = 3e-4
    batch_size: int = 1000
    eval_every: int = 5
    # IQL
    expectile_tau: float = 0.7
    ema_tau: float = 0.005
    iql_beta: float = 10.0
    # Architecture
    critic_layers: int = 3
    hidden_dim: int = 256
    num_rollouts: int = 100
    seed: int = 1
    env_id: str = "PickCube-v1"
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"
    output: str = "runs/v_timestep_analysis.png"


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
    print(f"Phase 1: Collecting {args.num_rollouts} rollouts...")
    sys.stdout.flush()
    t0 = time.time()

    all_obs = torch.zeros(T, E * args.num_rollouts, obs_dim, device=device)
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
                next_obs, reward, term, trunc, _ = envs.step(clip_action(action))
                next_done = (term | trunc).float()
                all_rewards[step, col] = reward.view(-1)

        all_next_obs[ri] = next_obs
        all_next_done[ri] = next_done

        if (ri + 1) % 20 == 0 or ri + 1 == args.num_rollouts:
            print(f"  {ri + 1}/{args.num_rollouts}")
            sys.stdout.flush()

    final_next_obs = all_next_obs.reshape(-1, obs_dim)
    final_next_done = all_next_done.reshape(-1)

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

    eval_obs = all_obs[:, :E, :]  # (T, E, D) — rollout 0 states

    envs.close()
    del envs, agent, optimal_agent
    torch.cuda.empty_cache()

    # ── Phase 3: Prepare TD data ──
    E_total = E * args.num_rollouts

    next_obs_all = torch.zeros_like(all_obs)
    next_obs_all[:-1] = all_obs[1:]
    next_obs_all[-1] = final_next_obs.unsqueeze(0)

    next_done_all = torch.zeros_like(all_rewards)
    next_done_all[:-1] = all_dones[1:]
    next_done_all[-1] = final_next_done.unsqueeze(0)

    # ── Flatten training data ──
    flat_s = all_obs.reshape(-1, obs_dim)
    N = flat_s.shape[0]

    # ── Training functions (save best critic) ──

    def train_td_ema(reward_scale=10.0):
        """Train TD+EMA, return best critic (at peak r)."""
        flat_r = all_rewards.reshape(-1) * reward_scale
        flat_ns = next_obs_all.reshape(-1, obs_dim)
        flat_nd = next_done_all.reshape(-1)
        mb = min(args.batch_size, N)

        critic = make_v_net()
        critic_target = copy.deepcopy(critic)
        opt = optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)

        best_r, best_sd = -999, None
        for epoch in range(args.epochs):
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

            if (epoch + 1) % args.eval_every == 0:
                critic.eval()
                with torch.no_grad():
                    v = critic(eval_obs.reshape(-1, obs_dim)).view(-1).cpu().numpy() / reward_scale
                r = pearsonr(v, on_mc16.reshape(-1).cpu().numpy())[0]
                if r > best_r:
                    best_r = r
                    best_sd = {k: v.clone() for k, v in critic.state_dict().items()}
                    best_ep = epoch + 1

        # Restore best and unscale
        critic.load_state_dict(best_sd)
        with torch.no_grad():
            critic[-1].weight.div_(reward_scale)
            critic[-1].bias.div_(reward_scale)
        critic.eval()
        return critic, best_r, best_ep

    # ── Phase 4: Train methods ──
    print(f"\n{'=' * 70}")
    print("Phase 4: Training methods (saving best critic)")
    print(f"{'=' * 70}\n")

    methods = {}

    print("--- TD+EMA rs=10 ---")
    t0 = time.time()
    critic, r, ep = train_td_ema(reward_scale=10.0)
    print(f"  peak r={r:.4f} @ epoch {ep} ({time.time()-t0:.1f}s)")
    methods["TD+EMA"] = critic

    # ── Phase 5: Extract trajectories from rollout 0 ──
    print(f"\n{'=' * 70}")
    print("Phase 5: Extracting trajectories from rollout 0")
    print(f"{'=' * 70}\n")

    mc16_np = on_mc16.cpu().numpy()  # (T, E)
    dones_0 = all_dones[:, :E].cpu().numpy()  # (T, E) — rollout 0 dones

    # Extract individual trajectories
    # dones[t, e] = 1 means env e was in a done state at step t (reset happened)
    # So the state at t is the FIRST state of a new episode
    trajectories = []  # list of (env_id, start_t, end_t, length)
    for e in range(E):
        traj_start = 0
        for t in range(1, T):
            if dones_0[t, e] == 1.0:
                # Trajectory ended with done: steps [traj_start, t-1]
                length = t - traj_start
                trajectories.append((e, traj_start, t - 1, length))
                traj_start = t
        # Last trajectory: [traj_start, T-1] — only keep if it's a full episode
        # (started at t=0, i.e. never reset) or was truncated at max_episode_steps
        length = T - traj_start
        if traj_start == 0 or length >= args.max_episode_steps:
            trajectories.append((e, traj_start, T - 1, length))
        # Otherwise it's a fragment at rollout end — discard

    print(f"  Found {len(trajectories)} trajectories")
    traj_lengths = [tr[3] for tr in trajectories]
    print(f"  Length: min={min(traj_lengths)}, max={max(traj_lengths)}, "
          f"mean={np.mean(traj_lengths):.1f}, median={np.median(traj_lengths):.1f}")

    # Build arrays: for each (traj, step), store normalized_t, mc16, obs_index
    all_norm_t = []
    all_mc16_vals = []
    all_obs_indices = []  # (t, e) pairs for eval_obs lookup

    for (e, t_start, t_end, length) in trajectories:
        for t in range(t_start, t_end + 1):
            norm_t = (t - t_start) / (length - 1) if length > 1 else 0.5
            all_norm_t.append(norm_t)
            all_mc16_vals.append(mc16_np[t, e])
            all_obs_indices.append((t, e))

    all_norm_t = np.array(all_norm_t)
    all_mc16_vals = np.array(all_mc16_vals)
    print(f"  Total state-points: {len(all_norm_t)}")

    # Gather obs for these indices
    traj_obs = torch.stack([eval_obs[t, e] for (t, e) in all_obs_indices]).to(device)  # (N_pts, D)

    # ── Phase 6: Per-trajectory normalized timestep analysis ──
    print(f"\n{'=' * 70}")
    print("Phase 6: Per-trajectory normalized timestep analysis")
    print(f"{'=' * 70}\n")

    # Bin normalized timesteps
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = np.digitize(all_norm_t, bin_edges[1:-1])  # 0 to n_bins-1

    # Ground truth stats per bin
    mc16_bin_mean = np.array([all_mc16_vals[bin_idx == b].mean() for b in range(n_bins)])
    mc16_bin_std = np.array([all_mc16_vals[bin_idx == b].std() for b in range(n_bins)])
    bin_counts = np.array([np.sum(bin_idx == b) for b in range(n_bins)])

    print("MC16 ground truth per normalized timestep bin:")
    print(f"  {'bin':>5} | {'n_pts':>6} | {'mean':>8} | {'std':>8}")
    print(f"  ------|--------|----------|--------")
    for b in range(n_bins):
        print(f"  {bin_centers[b]:>5.2f} | {bin_counts[b]:>6} | {mc16_bin_mean[b]:>8.4f} | {mc16_bin_std[b]:>8.4f}")

    # Per-method per-bin metrics
    ts_results = {}
    for name, critic in methods.items():
        with torch.no_grad():
            v_pred = critic(traj_obs).view(-1).cpu().numpy()

        error = v_pred - all_mc16_vals

        bias_bins = np.array([error[bin_idx == b].mean() for b in range(n_bins)])
        std_bins = np.array([error[bin_idx == b].std() for b in range(n_bins)])
        mae_bins = np.array([np.abs(error[bin_idx == b]).mean() for b in range(n_bins)])
        mc16_std_bins = np.array([all_mc16_vals[bin_idx == b].std() for b in range(n_bins)])

        # Per-bin correlation
        r_bins = []
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() > 10 and all_mc16_vals[mask].std() > 1e-8:
                r_bins.append(pearsonr(v_pred[mask], all_mc16_vals[mask])[0])
            else:
                r_bins.append(np.nan)
        r_bins = np.array(r_bins)

        ts_results[name] = dict(
            bias=bias_bins, error_std=std_bins, mae=mae_bins, per_bin_r=r_bins,
        )

        print(f"\n{name}: per-bin summary")
        print(f"  {'bin':>5} | {'bias':>8} | {'err_std':>8} | {'MAE':>8} | {'r':>8}")
        print(f"  ------|----------|----------|----------|--------")
        for b in range(n_bins):
            print(f"  {bin_centers[b]:>5.2f} | {bias_bins[b]:>8.4f} | {std_bins[b]:>8.4f} | {mae_bins[b]:>8.4f} | {r_bins[b]:>8.4f}")

    # ── Phase 7: Plot ──
    colors = {"TD+EMA": "tab:blue"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # [0,0] MC16 ground truth: mean ± std per bin
    ax = axes[0, 0]
    ax.plot(bin_centers, mc16_bin_mean, 'k-o', lw=2, label='MC16 mean')
    ax.fill_between(bin_centers, mc16_bin_mean - mc16_bin_std, mc16_bin_mean + mc16_bin_std,
                    alpha=0.3, color='gray', label='MC16 ±1σ')
    ax.set_xlabel("Normalized Timestep (0=start, 1=end)")
    ax.set_ylabel("V (MC16)")
    ax.set_title("Ground Truth V per Trajectory Phase")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # [0,1] Bias per bin
    ax = axes[0, 1]
    for name in methods:
        ax.plot(bin_centers, ts_results[name]['bias'], '-o', color=colors[name], lw=2, label=name)
    ax.axhline(0, color='gray', ls='--', lw=0.5)
    ax.set_xlabel("Normalized Timestep")
    ax.set_ylabel("Bias (V_pred - MC16)")
    ax.set_title("Bias vs Trajectory Phase")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # [0,2] Error std per bin
    ax = axes[0, 2]
    for name in methods:
        ax.plot(bin_centers, ts_results[name]['error_std'], '-o', color=colors[name], lw=2, label=f"{name} err std")
    ax.plot(bin_centers, mc16_bin_std, 'k--o', lw=1.5, label="MC16 std")
    ax.set_xlabel("Normalized Timestep")
    ax.set_ylabel("Std")
    ax.set_title("Error Std vs Trajectory Phase")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # [1,0] MAE per bin
    ax = axes[1, 0]
    for name in methods:
        ax.plot(bin_centers, ts_results[name]['mae'], '-o', color=colors[name], lw=2, label=name)
    ax.set_xlabel("Normalized Timestep")
    ax.set_ylabel("MAE")
    ax.set_title("MAE vs Trajectory Phase")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # [1,1] Per-bin Pearson r
    ax = axes[1, 1]
    for name in methods:
        ax.plot(bin_centers, ts_results[name]['per_bin_r'], '-o', color=colors[name], lw=2, label=name)
    ax.set_xlabel("Normalized Timestep")
    ax.set_ylabel("Pearson r")
    ax.set_title("Pearson r vs Trajectory Phase")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # [1,2] Trajectory length distribution
    ax = axes[1, 2]
    ax.hist(traj_lengths, bins=range(1, T + 2), edgecolor='black', alpha=0.7)
    ax.set_xlabel("Trajectory Length")
    ax.set_ylabel("Count")
    ax.set_title(f"Trajectory Length Distribution (n={len(trajectories)})")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Per-Trajectory V Error Analysis | N={args.num_rollouts}, γ={args.gamma}", fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {args.output}")

    # Save raw data
    save_path = args.output.replace('.png', '.pt')
    torch.save(dict(
        bin_centers=bin_centers, bin_counts=bin_counts,
        mc16_bin_mean=mc16_bin_mean, mc16_bin_std=mc16_bin_std,
        ts_results={k: v for k, v in ts_results.items()},
        trajectories=trajectories, traj_lengths=traj_lengths,
        args=vars(args),
    ), save_path)
    print(f"Saved data to {save_path}")
