"""Analyze HOW decisive states diverge: same s₀, different x_T → different outcomes.

For a few selected decisive states, run many MC samples, save full trajectories,
and analyze:
1. At which step do success/fail trajectories diverge?
2. What's the action variance at each decision step?
3. Where is the "fork point" — the critical phase where outcome is determined?

Usage:
  python analysis_trajectory_divergence.py \
    --ckpt runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt \
    --num_states 200 --mc_samples 32 --ddim_steps 10
"""
import copy
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

sys.path.insert(0, os.path.dirname(__file__))
from DPPO.model.unet_wrapper import DiffusionUNet
from DPPO.model.diffusion import DiffusionModel


@dataclass
class Args:
    ckpt: str = "runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt"
    env_id: str = "PegInsertionSide-v1"
    control_mode: str = "pd_joint_delta_pos"
    num_states: int = 200
    mc_samples: int = 32
    max_episode_steps: int = 200
    seed: int = 0
    zero_qvel: bool = False
    ddim_steps: int = 10
    ddim_eta: float = 1.0
    output_dir: str = "runs/analysis_divergence"


def load_model(args, device):
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if "pretrain_args" in ckpt and ckpt["pretrain_args"] is not None:
        arch_args = ckpt["pretrain_args"]
    else:
        arch_args = ckpt["args"]
    ckpt_args = ckpt["args"]
    obs_dim = ckpt["obs_dim"]
    act_dim = ckpt["action_dim"]
    cond_steps = arch_args.get("cond_steps", 2)
    horizon_steps = arch_args.get("horizon_steps", 16)
    act_steps = arch_args.get("act_steps", 8)

    network = DiffusionUNet(
        action_dim=act_dim, horizon_steps=horizon_steps,
        cond_dim=obs_dim * cond_steps,
        diffusion_step_embed_dim=arch_args.get("diffusion_step_embed_dim", 64),
        down_dims=arch_args.get("unet_dims", [64, 128, 256]),
        n_groups=arch_args.get("n_groups", 8),
    )
    model = DiffusionModel(
        network=network, horizon_steps=horizon_steps,
        obs_dim=obs_dim, action_dim=act_dim, device=device,
        denoising_steps=arch_args.get("denoising_steps", 100),
        denoised_clip_value=1.0, randn_clip_value=10,
        final_action_clip_value=1.0, predict_epsilon=True,
        base_eta=args.ddim_eta,
    )
    state_key = "ema" if "ema" in ckpt else "model"
    raw_sd = ckpt[state_key]
    if any(k.startswith("actor_ft.") for k in raw_sd):
        remapped = {}
        for k, v in raw_sd.items():
            if k.startswith("actor_ft.unet."):
                remapped["network.unet." + k[len("actor_ft.unet."):]] = v
            elif k.startswith("actor_ft."):
                remapped["network." + k[len("actor_ft."):]] = v
            elif not k.startswith(("actor.", "critic.", "ddim_")):
                remapped[k] = v
        raw_sd = remapped
    model.load_state_dict(raw_sd, strict=False)
    if torch.isnan(model.eta.eta_logit.data).any() or torch.isinf(model.eta.eta_logit.data).any():
        model.eta.eta_logit.data = torch.atanh(torch.tensor([0.999]))
    model.eval()

    if ckpt_args.get("zero_qvel", False) and not args.zero_qvel:
        args.zero_qvel = True

    return model, obs_dim, act_dim, cond_steps, horizon_steps, act_steps


def main():
    args = tyro.cli(Args)
    device = torch.device("cuda")
    os.makedirs(args.output_dir, exist_ok=True)

    model, obs_dim, act_dim, cond_steps, horizon_steps, act_steps = load_model(args, device)
    act_offset = cond_steps - 1

    print(f"Loaded: {args.ckpt}")
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}, zero_qvel={args.zero_qvel}")
    print(f"  DDIM steps={args.ddim_steps}, deterministic=True")

    # ===== Phase 1: Find decisive states using MC rollouts =====
    # Use num_states envs, run mc_samples MC rollouts to get P(success|s₀)
    N = args.num_states
    MC = args.mc_samples
    total_envs = N * MC  # each state gets MC parallel envs

    print(f"\nPhase 1: Finding decisive states ({N} states × {MC} MC)...")
    env = gym.make(
        args.env_id, num_envs=total_envs, obs_mode="state",
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
        sim_backend="gpu", reward_mode="sparse",
    )
    env = ManiSkillVectorEnv(env, total_envs, ignore_terminations=True, record_metrics=True)

    obs_init, _ = env.reset(seed=args.seed)
    obs_init = obs_init.float().to(device)

    # Replicate states: for state i, copy env[i*MC] → env[i*MC+1..i*MC+MC-1]
    state = env.unwrapped.get_state_dict()
    for key in state:
        if isinstance(state[key], torch.Tensor) and state[key].shape[0] == total_envs:
            for s in range(N):
                base = s * MC
                state[key][base+1:base+MC] = state[key][base:base+1].expand(
                    MC-1, *state[key].shape[1:])
    env.unwrapped.set_state_dict(state)

    # Save initial obs (one per state, from the base env)
    obs_raw = env.unwrapped.get_obs().float().to(device)
    obs_init_per_state = obs_raw[::MC].cpu().numpy()  # (N, obs_dim)
    saved_state = copy.deepcopy(env.unwrapped.get_state_dict())

    # Run rollout and track per-step data
    n_decision_steps = args.max_episode_steps // act_steps + 1

    # Storage: per-env, per-step tracking
    # We want: for each env, the tcp_pos (obs dims 18:21) and peg_pos (obs dims 25:28) at each step
    max_track_steps = args.max_episode_steps
    tcp_traj = torch.zeros(total_envs, max_track_steps, 3, device=device)
    peg_traj = torch.zeros(total_envs, max_track_steps, 3, device=device)
    action_traj = torch.zeros(total_envs, max_track_steps, act_dim, device=device)
    step_counter = torch.zeros(total_envs, dtype=torch.long, device=device)

    obs_history = obs_raw.unsqueeze(1).repeat(1, cond_steps, 1)
    success = torch.zeros(total_envs, dtype=torch.bool, device=device)
    done = torch.zeros(total_envs, dtype=torch.bool, device=device)
    success_step = torch.full((total_envs,), -1, dtype=torch.long, device=device)
    global_step = 0

    t0 = time.time()
    for step_block in range(n_decision_steps):
        if done.all():
            break
        obs_cond = obs_history
        if args.zero_qvel:
            obs_cond = obs_cond.clone()
            obs_cond[..., 9:18] = 0.0
        cond = {"state": obs_cond}
        with torch.no_grad():
            samples = model(cond, deterministic=True, ddim_steps=args.ddim_steps)
        actions = samples.trajectories

        for a_idx in range(act_steps):
            if global_step >= max_track_steps:
                break
            act_i = act_offset + a_idx
            action = actions[:, min(act_i, actions.shape[1] - 1)]

            # Record trajectory data before stepping
            alive = ~done
            tcp_traj[alive, global_step] = obs_history[alive, -1, 18:21]
            peg_traj[alive, global_step] = obs_history[alive, -1, 25:28]
            action_traj[alive, global_step] = action[alive]
            step_counter[alive] = global_step + 1

            obs_new, rew, term, trunc, _ = env.step(action)
            obs_new = obs_new.float().to(device)
            got_reward = rew.float() > 0.5
            newly_success = got_reward & ~done & ~success
            success_step[newly_success] = global_step
            success = success | (got_reward & ~done)
            done = done | term | trunc

            reset_mask = term | trunc
            if reset_mask.any():
                obs_history[reset_mask] = obs_new[reset_mask].unsqueeze(1).repeat(1, cond_steps, 1)
            not_reset = ~reset_mask
            if not_reset.any():
                obs_history[not_reset] = torch.cat(
                    [obs_history[not_reset, 1:], obs_new[not_reset].unsqueeze(1)], dim=1)
            global_step += 1

    env.close()
    elapsed = time.time() - t0

    # Reshape to (N, MC)
    success_mat = success.cpu().numpy().reshape(N, MC)
    success_step_mat = success_step.cpu().numpy().reshape(N, MC)
    p_success = success_mat.mean(axis=1)

    sr = p_success.mean()
    print(f"Phase 1 done in {elapsed:.0f}s. SR={sr:.1%}")

    # Categorize states
    decisive_mask = (p_success > 0.1) & (p_success < 0.9)
    n_decisive = decisive_mask.sum()
    print(f"  Decisive states: {n_decisive} (P in (0.1, 0.9))")

    # ===== Phase 2: Analyze trajectory divergence for decisive states =====
    print(f"\nPhase 2: Analyzing trajectory divergence...")

    tcp_np = tcp_traj.cpu().numpy().reshape(N, MC, max_track_steps, 3)
    peg_np = peg_traj.cpu().numpy().reshape(N, MC, max_track_steps, 3)
    action_np = action_traj.cpu().numpy().reshape(N, MC, max_track_steps, act_dim)
    step_count_np = step_counter.cpu().numpy().reshape(N, MC)

    # For all decisive states, compute per-step statistics
    decisive_idx = np.where(decisive_mask)[0]

    # Aggregate across all decisive states: at each step, compare success vs fail trajectories
    # Metric: mean pairwise distance between success and fail tcp trajectories
    max_steps_used = int(step_count_np[decisive_idx].max())

    # Per-step metrics aggregated over decisive states
    n_steps_analyze = min(max_steps_used, max_track_steps)
    tcp_std_succ = np.zeros(n_steps_analyze)   # mean TCP position std among successful rollouts
    tcp_std_fail = np.zeros(n_steps_analyze)
    tcp_std_all = np.zeros(n_steps_analyze)
    action_std_all = np.zeros(n_steps_analyze)
    peg_std_all = np.zeros(n_steps_analyze)

    # Per-step: mean distance between success and fail TCP positions
    tcp_succ_fail_dist = np.zeros(n_steps_analyze)
    n_contributing = np.zeros(n_steps_analyze)  # how many states contribute at each step

    for si in decisive_idx:
        succ_mc = success_mat[si]  # (MC,) bool
        n_succ = succ_mc.sum()
        n_fail = MC - n_succ
        if n_succ == 0 or n_fail == 0:
            continue

        max_step = int(step_count_np[si].max())

        for t in range(min(max_step, n_steps_analyze)):
            tcp_t = tcp_np[si, :, t]  # (MC, 3)
            peg_t = peg_np[si, :, t]
            act_t = action_np[si, :, t]

            # Overall std
            tcp_std_all[t] += np.mean(np.std(tcp_t, axis=0))
            peg_std_all[t] += np.mean(np.std(peg_t, axis=0))
            action_std_all[t] += np.mean(np.std(act_t, axis=0))

            # Succ vs fail mean distance
            tcp_succ_mean = tcp_t[succ_mc.astype(bool)].mean(axis=0)
            tcp_fail_mean = tcp_t[~succ_mc.astype(bool)].mean(axis=0)
            tcp_succ_fail_dist[t] += np.linalg.norm(tcp_succ_mean - tcp_fail_mean)

            # Within-group std
            tcp_std_succ[t] += np.mean(np.std(tcp_t[succ_mc.astype(bool)], axis=0))
            tcp_std_fail[t] += np.mean(np.std(tcp_t[~succ_mc.astype(bool)], axis=0))

            n_contributing[t] += 1

    # Average over contributing states
    mask_valid = n_contributing > 0
    tcp_std_all[mask_valid] /= n_contributing[mask_valid]
    peg_std_all[mask_valid] /= n_contributing[mask_valid]
    action_std_all[mask_valid] /= n_contributing[mask_valid]
    tcp_succ_fail_dist[mask_valid] /= n_contributing[mask_valid]
    tcp_std_succ[mask_valid] /= n_contributing[mask_valid]
    tcp_std_fail[mask_valid] /= n_contributing[mask_valid]

    steps = np.arange(n_steps_analyze)

    # Print key findings
    print(f"\n--- Trajectory divergence analysis ({n_decisive} decisive states) ---")

    # Find the step where succ-fail distance peaks
    peak_step = np.argmax(tcp_succ_fail_dist)
    print(f"  Peak TCP succ-fail distance at step {peak_step}: {tcp_succ_fail_dist[peak_step]:.4f}")
    print(f"  Action std at step 0: {action_std_all[0]:.4f}")
    print(f"  Action std at step {peak_step}: {action_std_all[peak_step]:.4f}")
    print(f"  TCP std at step 0: {tcp_std_all[0]:.6f}")
    print(f"  TCP std at step 50: {tcp_std_all[min(50, n_steps_analyze-1)]:.6f}")
    print(f"  TCP std at step 100: {tcp_std_all[min(100, n_steps_analyze-1)]:.6f}")
    print(f"  TCP std at step 150: {tcp_std_all[min(150, n_steps_analyze-1)]:.6f}")

    # Find "fork point": where does succ-fail distance first become > within-group std?
    combined_within_std = (tcp_std_succ + tcp_std_fail) / 2
    fork_ratio = np.zeros(n_steps_analyze)
    fork_ratio[combined_within_std > 1e-8] = (
        tcp_succ_fail_dist[combined_within_std > 1e-8] /
        combined_within_std[combined_within_std > 1e-8]
    )
    fork_candidates = np.where(fork_ratio > 2.0)[0]  # distance > 2x within-group std
    if len(fork_candidates) > 0:
        fork_step = fork_candidates[0]
        print(f"  Fork point (succ-fail dist > 2x within-group std): step {fork_step}")
        print(f"    = {fork_step / args.max_episode_steps:.0%} of episode")
    else:
        fork_step = -1
        print(f"  No clear fork point found (gradual divergence)")

    # ===== Phase 3: Single-state deep dive =====
    # Pick 3 representative decisive states: P≈0.3, P≈0.5, P≈0.7
    targets = [0.3, 0.5, 0.7]
    selected = []
    for target_p in targets:
        dists = np.abs(p_success[decisive_idx] - target_p)
        best = decisive_idx[np.argmin(dists)]
        selected.append((best, p_success[best]))
        print(f"\n  Selected state {best}: P={p_success[best]:.2f} (target {target_p})")
        succ_mc = success_mat[best].astype(bool)
        succ_steps = success_step_mat[best][succ_mc]
        print(f"    Success: {succ_mc.sum()}/{MC}, success steps: "
              f"mean={succ_steps.mean():.0f}, min={succ_steps.min()}, max={succ_steps.max()}")

    # ===== Plots =====
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: TCP position std over time (all MC samples)
    ax = axes[0, 0]
    ax.plot(steps, tcp_std_all, 'b-', label='TCP pos std', linewidth=1.5)
    ax.plot(steps, peg_std_all, 'r-', label='Peg pos std', linewidth=1.5)
    if fork_step > 0:
        ax.axvline(fork_step, color='k', ls='--', alpha=0.5, label=f'Fork @ step {fork_step}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean std across MC samples (m)')
    ax.set_title('Position Divergence Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Action std over time
    ax = axes[0, 1]
    ax.plot(steps, action_std_all, 'g-', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean action std across MC samples')
    ax.set_title('Action Variance Over Time')
    ax.grid(True, alpha=0.3)

    # Plot 3: Succ vs fail TCP distance
    ax = axes[0, 2]
    ax.plot(steps, tcp_succ_fail_dist, 'purple', linewidth=2, label='Succ-Fail mean dist')
    ax.plot(steps, tcp_std_succ, 'g--', alpha=0.7, label='Within-succ std')
    ax.plot(steps, tcp_std_fail, 'r--', alpha=0.7, label='Within-fail std')
    if fork_step > 0:
        ax.axvline(fork_step, color='k', ls='--', alpha=0.5, label=f'Fork @ step {fork_step}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Distance / Std (m)')
    ax.set_title('Success vs Failure: TCP Separation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plots 4-6: Single state deep dives (TCP XY trajectories)
    for plot_i, (si, p_val) in enumerate(selected):
        ax = axes[1, plot_i]
        succ_mc = success_mat[si].astype(bool)
        max_step = int(step_count_np[si].max())

        # Plot TCP XY trajectory for each MC sample
        for j in range(MC):
            t_end = int(step_count_np[si, j])
            x = tcp_np[si, j, :t_end, 0]  # tcp_x
            y = tcp_np[si, j, :t_end, 1]  # tcp_y
            color = 'green' if succ_mc[j] else 'red'
            alpha = 0.4
            ax.plot(x, y, color=color, alpha=alpha, linewidth=0.8)
            # Mark start and end
            if j == 0:
                ax.plot(x[0], y[0], 'ko', markersize=6, zorder=5)
            if succ_mc[j]:
                ax.plot(x[-1], y[-1], 'g^', markersize=4, alpha=0.5)
            else:
                ax.plot(x[-1], y[-1], 'rx', markersize=4, alpha=0.5)

        # Mark hole position
        hole_x = obs_init_per_state[si, 35]
        hole_y = obs_init_per_state[si, 36]
        ax.plot(hole_x, hole_y, 'b*', markersize=12, zorder=10, label='Hole')

        ax.set_xlabel('TCP X')
        ax.set_ylabel('TCP Y')
        ax.set_title(f'State {si}: P={p_val:.2f} '
                     f'({succ_mc.sum()}/{MC} succ)\n'
                     f'Green=success, Red=fail')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.suptitle(f'Trajectory Divergence Analysis ({n_decisive} decisive states, MC{MC})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, 'trajectory_divergence.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {plot_path}")

    # ===== Additional plot: Fork ratio over time =====
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, fork_ratio, 'b-', linewidth=1.5)
    ax.axhline(2.0, color='r', ls='--', alpha=0.5, label='Threshold (2x)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Succ-Fail dist / Within-group std')
    ax.set_title('Fork Ratio: When Do Success and Failure Trajectories Separate?')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(fork_ratio.max() * 1.1, 3))
    plt.tight_layout()
    plot2_path = os.path.join(args.output_dir, 'fork_ratio.png')
    plt.savefig(plot2_path, dpi=150)
    plt.close()
    print(f"Saved: {plot2_path}")

    # Save data
    np.savez(
        os.path.join(args.output_dir, 'divergence_data.npz'),
        p_success=p_success, decisive_idx=decisive_idx,
        tcp_std_all=tcp_std_all, peg_std_all=peg_std_all,
        action_std_all=action_std_all,
        tcp_succ_fail_dist=tcp_succ_fail_dist,
        tcp_std_succ=tcp_std_succ, tcp_std_fail=tcp_std_fail,
        fork_ratio=fork_ratio,
        obs_init_per_state=obs_init_per_state,
    )
    print(f"Saved data to {os.path.join(args.output_dir, 'divergence_data.npz')}")


if __name__ == "__main__":
    main()
