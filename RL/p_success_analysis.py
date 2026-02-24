"""Compute P(success|s) directly from MC16 rollouts (no discount needed).

For each state in a rollout, runs 16 MC rollouts and counts how many succeed.
P(success|s) = n_success / 16. Also computes discounted V for comparison.

IMPORTANT: Only analyzes FIRST episodes per env (no truncated post-reset fragments).

Usage:
  # PickCube
  python -u -m RL.p_success_analysis --env_id PickCube-v1 \
    --checkpoint runs/pickcube_ppo/ckpt_76_logstd-1.5.pt \
    --max_episode_steps 50 --num_steps 50 --gamma 0.99

  # PegInsertion
  python -u -m RL.p_success_analysis --env_id PegInsertionSide-v1 \
    --checkpoint runs/peginsertion_ppo_ema99/ckpt_231.pt \
    --max_episode_steps 100 --num_steps 100 --gamma 0.97
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
    output: str = "runs/p_success_analysis.png"


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

    # ── Phase 1: Collect 1 rollout ──
    print(f"Phase 1: Collecting rollout ({E} envs, {T} steps)...")
    sys.stdout.flush()
    t0 = time.time()

    states = [None] * T
    roll_obs = torch.zeros(T, E, envs.single_observation_space.shape[0], device=device)
    roll_dones = torch.zeros(T, E, device=device)
    roll_rewards = torch.zeros(T, E, device=device)

    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(E, device=device)

    with torch.no_grad():
        for step in range(T):
            states[step] = _clone_state_cpu(envs.base_env.get_state_dict())
            roll_obs[step] = next_obs
            roll_dones[step] = next_done
            action = agent.get_action(next_obs, deterministic=False)
            next_obs, reward, term, trunc, _ = envs.step(clip_action(action))
            next_done = (term | trunc).float()
            roll_rewards[step] = reward.view(-1)

    print(f"  Done ({time.time() - t0:.1f}s)")

    # ── Phase 1b: Identify first episodes only ──
    dones_np = roll_dones.cpu().numpy()
    rewards_np = roll_rewards.cpu().numpy()

    # ep_info[e] = (length, success) for the FIRST episode of each env
    first_ep_end = {}  # env -> end timestep (exclusive)
    for e in range(E):
        for t in range(1, T):
            if dones_np[t, e] > 0.5:
                first_ep_end[e] = t
                break
        if e not in first_ep_end:
            first_ep_end[e] = T  # didn't terminate

    # Build first-episode mask and episode %
    first_ep_mask = np.zeros((T, E), dtype=bool)
    ep_pct = np.full((T, E), np.nan)
    first_ep_success = {}

    for e in range(E):
        end = first_ep_end[e]
        # Check success: any reward > 0 in first episode
        success = any(rewards_np[t, e] > 0.5 for t in range(end))
        first_ep_success[e] = success
        for t in range(end):
            first_ep_mask[t, e] = True
            ep_pct[t, e] = t / end  # 0 to (end-1)/end

    n_success = sum(first_ep_success.values())
    success_lens = [first_ep_end[e] for e in range(E) if first_ep_success[e]]
    fail_lens = [first_ep_end[e] for e in range(E) if not first_ep_success[e]]

    print(f"\nFirst episodes: {E}")
    print(f"  Success: {n_success} ({n_success/E*100:.1f}%)")
    if success_lens:
        print(f"  Success ep lens: mean={np.mean(success_lens):.1f}, min={min(success_lens)}, max={max(success_lens)}")
    if fail_lens:
        print(f"  Fail ep lens: mean={np.mean(fail_lens):.1f}, min={min(fail_lens)}, max={max(fail_lens)}")

    # ── Phase 2: MC16 — compute both discounted V and P(success) ──
    num_mc_envs = E * args.mc_samples
    print(f"\nPhase 2: MC16 ({num_mc_envs} mc_envs)...")
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

    # Only compute for first-episode states
    on_V = torch.full((T, E), float('nan'), device=device)       # discounted V
    on_P = torch.full((T, E), float('nan'), device=device)       # P(success)
    on_P_var = torch.full((T, E), float('nan'), device=device)   # Bernoulli variance

    # Which timesteps have any first-episode states?
    active_steps = sorted(set(t for t in range(T) for e in range(E) if first_ep_mask[t, e]))

    with torch.no_grad():
        for idx, t in enumerate(active_steps):
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

            # Discounted return
            ret = torch.zeros(num_mc_envs, device=device)
            for s in reversed(range(len(all_rews))):
                ret = all_rews[s] + args.gamma * ret
            ret_per_state = ret.view(E, args.mc_samples)
            on_V[t] = ret_per_state.mean(dim=1)

            # P(success) = fraction of MC rollouts with any reward > 0
            total_rew_per_rollout = sum(all_rews)  # (num_mc_envs,)
            success_per_rollout = (total_rew_per_rollout > 0.5).float().view(E, args.mc_samples)
            on_P[t] = success_per_rollout.mean(dim=1)
            on_P_var[t] = success_per_rollout.var(dim=1)

            if (idx + 1) % 10 == 0:
                print(f"  MC16 step {idx + 1}/{len(active_steps)}")
                sys.stdout.flush()

    mc_envs.close()
    del mc_envs, mc_envs_raw
    torch.cuda.empty_cache()
    print(f"  Phase 2 done ({time.time() - t0:.1f}s)")

    # ── Phase 3: Analyze by episode % (first episodes only) ──
    V_np = on_V.cpu().numpy()
    P_np = on_P.cpu().numpy()

    n_bins = 20
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    def bin_stats(values, mask):
        flat_pct = ep_pct[mask]
        flat_val = values[mask]
        mean_v = np.zeros(n_bins)
        var_v = np.zeros(n_bins)
        count = np.zeros(n_bins, dtype=int)
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            sel = (flat_pct >= lo) & (flat_pct <= hi) if b == n_bins - 1 else (flat_pct >= lo) & (flat_pct < hi)
            vals = flat_val[sel]
            count[b] = len(vals)
            if len(vals) > 1:
                mean_v[b] = vals.mean()
                var_v[b] = vals.var()
        return mean_v, var_v, count

    # All first episodes
    all_V_mean, all_V_var, all_count = bin_stats(V_np, first_ep_mask)
    all_P_mean, all_P_var, all_P_count = bin_stats(P_np, first_ep_mask)

    # Success first episodes
    succ_mask = np.zeros((T, E), dtype=bool)
    fail_mask = np.zeros((T, E), dtype=bool)
    for e in range(E):
        end = first_ep_end[e]
        for t in range(end):
            if first_ep_success[e]:
                succ_mask[t, e] = True
            else:
                fail_mask[t, e] = True

    succ_P_mean, succ_P_var, succ_count = bin_stats(P_np, succ_mask)
    fail_P_mean, fail_P_var, fail_count = bin_stats(P_np, fail_mask)
    succ_V_mean, _, _ = bin_stats(V_np, succ_mask)
    fail_V_mean, _, _ = bin_stats(V_np, fail_mask)

    # ── Print tables ──
    print(f"\n{'='*90}")
    print(f"P(success) and V by episode % — FIRST EPISODES ONLY")
    print(f"{'='*90}")
    print(f"{'ep%':>8} | {'mean P':>8} | {'Var[P]':>8} | {'mean V':>8} | "
          f"{'P_succ':>8} | {'P_fail':>8} | {'V_succ':>8} | {'V_fail':>8} | {'count':>5}")
    print(f"---------|----------|----------|----------|----------|----------|----------|----------|------")
    for b in range(n_bins):
        pct = f"{bin_edges[b]*100:.0f}-{bin_edges[b+1]*100:.0f}%"
        print(f"{pct:>8} | {all_P_mean[b]:>8.4f} | {all_P_var[b]:>8.4f} | {all_V_mean[b]:>8.4f} | "
              f"{succ_P_mean[b]:>8.4f} | {fail_P_mean[b]:>8.4f} | {succ_V_mean[b]:>8.4f} | {fail_V_mean[b]:>8.4f} | {all_count[b]:>5}")

    peak_varp = np.argmax(all_P_var)
    closest_05 = np.argmin(np.abs(all_P_mean - 0.5))
    print(f"\nPeak Var[P] at {bin_edges[peak_varp]*100:.0f}-{bin_edges[peak_varp+1]*100:.0f}%")
    print(f"P closest to 0.5 at {bin_edges[closest_05]*100:.0f}-{bin_edges[closest_05+1]*100:.0f}% (P={all_P_mean[closest_05]:.4f})")

    # ── Phase 4: Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # [0,0] V vs P
    ax = axes[0, 0]
    ax.plot(bin_centers * 100, all_V_mean, 'b-o', ms=4, lw=2, label='V(s) (discounted)')
    ax.plot(bin_centers * 100, all_P_mean, 'r-s', ms=4, lw=2, label='P(success|s) (MC16)')
    ax.axhline(0.5, color='gray', ls='--', alpha=0.5, label='P=0.5')
    ax.axhline(n_success / E, color='green', ls=':', alpha=0.5, label=f'Overall SR={n_success/E*100:.0f}%')
    ax.set_xlabel("Episode %")
    ax.set_ylabel("Value")
    ax.set_title("V(s) vs P(success|s)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # [0,1] Var[P]
    ax = axes[0, 1]
    colors = ['red' if b == peak_varp else 'coral' for b in range(n_bins)]
    ax.bar(bin_centers * 100, all_P_var, width=4.2, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel("Episode %")
    ax.set_ylabel("Var[P(success)]")
    ax.set_title(f"Cross-env Var[P] — peak at {bin_centers[peak_varp]*100:.0f}%")
    ax.grid(True, alpha=0.3, axis='y')

    # [1,0] P by success/fail trajectories
    ax = axes[1, 0]
    ax.plot(bin_centers * 100, all_P_mean, 'k-o', ms=4, lw=2, label='All')
    ax.plot(bin_centers * 100, succ_P_mean, 'g-s', ms=4, lw=2, label=f'Success ({sum(succ_count)} states)')
    ax.plot(bin_centers * 100, fail_P_mean, 'r-^', ms=4, lw=2, label=f'Fail ({sum(fail_count)} states)')
    ax.axhline(0.5, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel("Episode %")
    ax.set_ylabel("P(success|s)")
    ax.set_title("P(success) — Success vs Fail Trajectories")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # [1,1] P distribution at key bins
    ax = axes[1, 1]
    flat_pct_first = ep_pct[first_ep_mask]
    flat_P_first = P_np[first_ep_mask]
    early_b, mid_b, late_b = 1, n_bins // 2, n_bins - 2
    for bidx, color, lbl in [
        (early_b, 'blue', f'{bin_edges[early_b]*100:.0f}-{bin_edges[early_b+1]*100:.0f}% (early)'),
        (mid_b, 'orange', f'{bin_edges[mid_b]*100:.0f}-{bin_edges[mid_b+1]*100:.0f}% (mid)'),
        (late_b, 'red', f'{bin_edges[late_b]*100:.0f}-{bin_edges[late_b+1]*100:.0f}% (late)')
    ]:
        lo, hi = bin_edges[bidx], bin_edges[bidx + 1]
        sel = (flat_pct_first >= lo) & (flat_pct_first < hi) if bidx < n_bins - 1 else (flat_pct_first >= lo)
        vals = flat_P_first[sel]
        if len(vals) > 0:
            ax.hist(vals, bins=17, range=(0, 1), alpha=0.4, color=color, label=lbl,
                    density=True, edgecolor='black', linewidth=0.3)
    ax.axvline(0.5, color='gray', ls='--', lw=2, alpha=0.5)
    ax.set_xlabel("P(success|s)")
    ax.set_ylabel("Density")
    ax.set_title("P Distribution at Key Episode %")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    env_name = args.env_id.split("-")[0]
    fig.suptitle(f"{env_name}: P(success|s) from MC16 — First Episodes Only | gamma={args.gamma}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {args.output}")

    # Save data
    save_path = args.output.replace('.png', '.pt')
    torch.save(dict(
        V=on_V.cpu(), P=on_P.cpu(), P_var=on_P_var.cpu(),
        ep_pct=ep_pct, first_ep_mask=first_ep_mask,
        first_ep_end=first_ep_end, first_ep_success=first_ep_success,
        all_P_mean=all_P_mean, all_P_var=all_P_var,
        succ_P_mean=succ_P_mean, fail_P_mean=fail_P_mean,
        args=vars(args),
    ), save_path)
    print(f"Saved data: {save_path}")

    envs.close()
