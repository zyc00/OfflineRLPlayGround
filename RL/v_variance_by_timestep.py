"""Quick diagnostic: plot Var[V(s)] by episode percentage using MC16 ground truth.

Measures V variance in two frames:
  1. By absolute timestep (raw rollout index)
  2. By episode percentage (normalized within each episode)

With partial reset, a single rollout may contain multiple episodes per env.
Episode-percentage view correctly handles varying episode lengths.

Usage:
  python -u -m RL.v_variance_by_timestep --gamma 0.99
"""

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
import tyro
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from data.data_collection.ppo import Agent


@dataclass
class Args:
    checkpoint: str = "runs/pickcube_ppo/ckpt_76_logstd-1.5.pt"
    gamma: float = 0.99
    num_envs: int = 100
    num_steps: int = 50
    max_episode_steps: int = 50
    mc_samples: int = 16
    seed: int = 1
    env_id: str = "PickCube-v1"
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"
    output: str = "runs/v_variance_by_episode_pct.png"


if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda")
    T, E = args.num_steps, args.num_envs

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_kwargs = dict(
        obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda",
        reward_mode=args.reward_mode, control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
    )

    # ── Setup ──
    envs = gym.make(args.env_id, num_envs=E, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(envs, E, ignore_terminations=False, record_metrics=True)
    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()

    def clip_action(a):
        return torch.clamp(a.detach(), action_low, action_high)

    def _clone_state_cpu(sd):
        if isinstance(sd, dict):
            return {k: _clone_state_cpu(v) for k, v in sd.items()}
        if isinstance(sd, torch.Tensor):
            return sd.cpu().clone()
        return sd

    def _expand_state(sd, repeats):
        if isinstance(sd, dict):
            return {k: _expand_state(v, repeats) for k, v in sd.items()}
        if isinstance(sd, torch.Tensor) and sd.dim() > 0:
            return sd.repeat_interleave(repeats, dim=0)
        return sd

    def _state_to_device(sd, dev):
        if isinstance(sd, dict):
            return {k: _state_to_device(v, dev) for k, v in sd.items()}
        if isinstance(sd, torch.Tensor):
            return sd.to(dev)
        return sd

    # ── Phase 1: Collect 1 rollout, save all states ──
    print(f"Phase 1: Collecting 1 rollout, saving states...")
    sys.stdout.flush()
    t0 = time.time()

    states = [None] * T
    roll_rewards = torch.zeros(T, E, device=device)
    roll_dones = torch.zeros(T, E, device=device)

    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(E, device=device)

    with torch.no_grad():
        for step in range(T):
            states[step] = _clone_state_cpu(envs.base_env.get_state_dict())
            roll_dones[step] = next_done
            action = agent.get_action(next_obs, deterministic=False)
            next_obs, reward, term, trunc, _ = envs.step(clip_action(action))
            next_done = (term | trunc).float()
            roll_rewards[step] = reward.view(-1)

    print(f"  Done ({time.time() - t0:.1f}s)")

    # ── Phase 1b: Identify episodes and compute episode percentage ──
    print(f"\nPhase 1b: Identifying episode segments...")
    dones_np = roll_dones.cpu().numpy()  # (T, E), dones[t]=1 means new episode starts at t

    # ep_pct[t, e] = position of state (t,e) within its episode, in [0, 1)
    ep_pct = np.full((T, E), np.nan)
    # ep_len[t, e] = length of the episode that state (t,e) belongs to
    ep_len = np.full((T, E), np.nan)
    # Track all episodes: (env_id, t_start, length, completed)
    all_episodes = []

    for e in range(E):
        # Find episode boundaries
        t_start = 0
        for t in range(1, T):
            if dones_np[t, e] > 0.5:  # new episode starts at t
                length = t - t_start
                for s in range(t_start, t):
                    ep_pct[s, e] = (s - t_start) / length
                    ep_len[s, e] = length
                all_episodes.append((e, t_start, length, True))
                t_start = t
        # Last segment (may be truncated by rollout end)
        length = T - t_start
        for s in range(t_start, T):
            ep_pct[s, e] = (s - t_start) / length
            ep_len[s, e] = length
        all_episodes.append((e, t_start, length, False))

    ep_lengths = [ep[2] for ep in all_episodes]
    completed_lengths = [ep[2] for ep in all_episodes if ep[3]]
    print(f"  Total episodes: {len(all_episodes)} ({len(completed_lengths)} completed)")
    print(f"  Episode lengths: mean={np.mean(ep_lengths):.1f}, "
          f"min={np.min(ep_lengths)}, max={np.max(ep_lengths)}")
    if completed_lengths:
        print(f"  Completed episode lengths: mean={np.mean(completed_lengths):.1f}, "
              f"min={np.min(completed_lengths)}, max={np.max(completed_lengths)}")

    # ── Phase 2: MC16 at each timestep ──
    num_mc_envs = E * args.mc_samples
    print(f"\nPhase 2: MC16 ground truth ({num_mc_envs} mc_envs)...")
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
    on_mc16_var = torch.zeros(T, E, device=device)  # per-state MC variance

    with torch.no_grad():
        for t in range(T):
            expanded = _expand_state(_state_to_device(states[t], device), args.mc_samples)
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
            ret_per_state = ret.view(E, args.mc_samples)
            on_mc16[t] = ret_per_state.mean(dim=1)
            on_mc16_var[t] = ret_per_state.var(dim=1)
            if (t + 1) % 10 == 0:
                print(f"  MC16 step {t + 1}/{T}")
                sys.stdout.flush()

    mc_envs.close()
    del mc_envs, mc_envs_raw
    torch.cuda.empty_cache()

    print(f"  MC16: mean={on_mc16.mean():.4f}, std={on_mc16.std():.4f}")
    print(f"  Phase 2 done ({time.time() - t0:.1f}s)")

    # ── Phase 3: Compute statistics in both frames ──
    mc16_np = on_mc16.cpu().numpy()  # (T, E)

    # --- Frame 1: by absolute timestep ---
    var_by_t = mc16_np.var(axis=1)
    mean_by_t = mc16_np.mean(axis=1)
    std_by_t = mc16_np.std(axis=1)
    q25_by_t = np.percentile(mc16_np, 25, axis=1)
    q75_by_t = np.percentile(mc16_np, 75, axis=1)

    # --- Frame 2: by episode percentage ---
    n_bins = 20
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    var_by_pct = np.zeros(n_bins)
    mean_by_pct = np.zeros(n_bins)
    std_by_pct = np.zeros(n_bins)
    q25_by_pct = np.zeros(n_bins)
    q75_by_pct = np.zeros(n_bins)
    count_by_pct = np.zeros(n_bins, dtype=int)

    # Per-state MC variance by episode %
    mc_var_np = on_mc16_var.cpu().numpy()  # (T, E)
    flat_mc_var = mc_var_np.ravel()
    mc_var_mean_by_pct = np.zeros(n_bins)
    mc_var_std_by_pct = np.zeros(n_bins)

    flat_pct = ep_pct.ravel()
    flat_mc16 = mc16_np.ravel()
    flat_eplen = ep_len.ravel()

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (flat_pct >= lo) & (flat_pct <= hi)
        else:
            mask = (flat_pct >= lo) & (flat_pct < hi)
        vals = flat_mc16[mask]
        var_vals = flat_mc_var[mask]
        count_by_pct[b] = len(vals)
        if len(vals) > 1:
            var_by_pct[b] = vals.var()
            mean_by_pct[b] = vals.mean()
            std_by_pct[b] = vals.std()
            q25_by_pct[b] = np.percentile(vals, 25)
            q75_by_pct[b] = np.percentile(vals, 75)
            mc_var_mean_by_pct[b] = var_vals.mean()
            mc_var_std_by_pct[b] = var_vals.std()

    # --- Print tables ---
    print(f"\n{'=' * 60}")
    print(f"V variance by TIMESTEP (gamma={args.gamma})")
    print(f"{'=' * 60}")
    print(f"{'t':>4} | {'mean V':>8} | {'std V':>8} | {'var V':>8}")
    print(f"-----|----------|----------|----------")
    for t in range(0, T, 5):
        print(f"{t:>4} | {mean_by_t[t]:>8.4f} | {std_by_t[t]:>8.4f} | {var_by_t[t]:>8.4f}")

    peak_var_t = np.argmax(var_by_t)
    print(f"\nPeak variance at t={peak_var_t} (var={var_by_t[peak_var_t]:.4f})")

    print(f"\n{'=' * 80}")
    print(f"V value & MC variance by EPISODE PERCENTAGE (gamma={args.gamma})")
    print(f"{'=' * 80}")
    print(f"{'pct':>6} | {'mean V':>8} | {'std V':>8} | {'Var[V]':>8} | {'MC var':>10} | {'count':>6}")
    print(f"-------|----------|----------|----------|------------|-------")
    for b in range(n_bins):
        pct_label = f"{bin_edges[b]*100:.0f}-{bin_edges[b+1]*100:.0f}%"
        print(f"{pct_label:>6} | {mean_by_pct[b]:>8.4f} | {std_by_pct[b]:>8.4f} | "
              f"{var_by_pct[b]:>8.4f} | {mc_var_mean_by_pct[b]:>10.6f} | {count_by_pct[b]:>6}")

    peak_var_bin = np.argmax(var_by_pct)
    peak_mc_var_bin = np.argmax(mc_var_mean_by_pct)
    print(f"\nPeak cross-env Var[V] at {bin_edges[peak_var_bin]*100:.0f}-{bin_edges[peak_var_bin+1]*100:.0f}% "
          f"(var={var_by_pct[peak_var_bin]:.4f})")
    print(f"Peak MC variance at {bin_edges[peak_mc_var_bin]*100:.0f}-{bin_edges[peak_mc_var_bin+1]*100:.0f}% "
          f"(mc_var={mc_var_mean_by_pct[peak_mc_var_bin]:.6f})")

    # ── Phase 4: Plot (3×2) ──
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # [0,0] Variance by timestep
    ax = axes[0, 0]
    ax.plot(range(T), var_by_t, 'b-', lw=2)
    ax.axvline(peak_var_t, color='red', ls='--', alpha=0.7, label=f'peak t={peak_var_t}')
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Var[V(s)]")
    ax.set_title("V Variance by Timestep")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [0,1] Variance by episode percentage
    ax = axes[0, 1]
    ax.bar(bin_centers * 100, var_by_pct, width=100/n_bins * 0.85, color='steelblue',
           edgecolor='black', linewidth=0.5)
    pct_peak_x = bin_centers[peak_var_bin] * 100
    ax.axvline(pct_peak_x, color='red', ls='--', alpha=0.7,
               label=f'peak {bin_edges[peak_var_bin]*100:.0f}-{bin_edges[peak_var_bin+1]*100:.0f}%')
    ax.set_xlabel("Episode Percentage (%)")
    ax.set_ylabel("Var[V(s)]")
    ax.set_title("V Variance by Episode %")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # [1,0] Mean V + IQR by timestep
    ax = axes[1, 0]
    ax.plot(range(T), mean_by_t, 'b-', lw=2, label='mean')
    ax.fill_between(range(T), q25_by_t, q75_by_t, alpha=0.3, color='blue', label='IQR')
    ax.set_xlabel("Timestep")
    ax.set_ylabel("V(s)")
    ax.set_title("V Distribution by Timestep")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [1,1] Mean V + IQR by episode percentage
    ax = axes[1, 1]
    ax.plot(bin_centers * 100, mean_by_pct, 'b-', lw=2, label='mean')
    ax.fill_between(bin_centers * 100, q25_by_pct, q75_by_pct, alpha=0.3, color='blue', label='IQR')
    ax.set_xlabel("Episode Percentage (%)")
    ax.set_ylabel("V(s)")
    ax.set_title("V Distribution by Episode %")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [2,0] Episode length distribution
    ax = axes[2, 0]
    ax.hist(ep_lengths, bins=range(0, T + 2), color='orange', alpha=0.7, edgecolor='black')
    if completed_lengths:
        ax.hist(completed_lengths, bins=range(0, T + 2), color='green', alpha=0.5,
                edgecolor='black', label='completed')
    ax.set_xlabel("Episode Length (steps)")
    ax.set_ylabel("Count")
    ax.set_title(f"Episode Length Distribution (n={len(all_episodes)})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [2,1] Sample count per episode-% bin
    ax = axes[2, 1]
    ax.bar(bin_centers * 100, count_by_pct, width=100/n_bins * 0.85, color='gray',
           edgecolor='black', linewidth=0.5)
    ax.set_xlabel("Episode Percentage (%)")
    ax.set_ylabel("Number of States")
    ax.set_title("State Count per Episode % Bin")
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f"V(s) Variance: Timestep vs Episode % | gamma={args.gamma} | MC16 | {args.env_id}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {args.output}")

    # Save data
    save_path = args.output.replace('.png', '.pt')
    torch.save(dict(
        mc16=on_mc16.cpu(),
        ep_pct=ep_pct,
        ep_len=ep_len,
        all_episodes=all_episodes,
        # By timestep
        var_by_t=var_by_t,
        mean_by_t=mean_by_t,
        # By episode percentage
        var_by_pct=var_by_pct,
        mean_by_pct=mean_by_pct,
        count_by_pct=count_by_pct,
        bin_edges=bin_edges,
        args=vars(args),
    ), save_path)
    print(f"Saved data to {save_path}")

    envs.close()
