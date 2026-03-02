"""Analyze deterministic DP failure: separate frac_zero / decisive / frac_one states,
then analyze what obs features distinguish them.

Step 1: MC rollouts to get P(success|s₀) per state (reuse dp_p_success_gpu.py logic)
Step 2: Categorize states into zero/decisive/one
Step 3: Feature analysis on each group

Usage:
  python analysis_deterministic_failure_v2.py \
    --ckpt runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt \
    --num_states 500 --mc_samples 16 --ddim_steps 10
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
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

sys.path.insert(0, os.path.dirname(__file__))
from DPPO.model.unet_wrapper import DiffusionUNet
from DPPO.model.diffusion import DiffusionModel


OBS_LABELS = [
    "qpos_j1", "qpos_j2", "qpos_j3", "qpos_j4", "qpos_j5", "qpos_j6", "qpos_j7",
    "qpos_finger1", "qpos_finger2",
    "qvel_j1", "qvel_j2", "qvel_j3", "qvel_j4", "qvel_j5", "qvel_j6", "qvel_j7",
    "qvel_finger1", "qvel_finger2",
    "tcp_x", "tcp_y", "tcp_z",
    "tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw",
    "peg_x", "peg_y", "peg_z",
    "peg_qx", "peg_qy", "peg_qz", "peg_qw",
    "peg_hsize_x", "peg_hsize_y", "peg_hsize_z",
    "hole_x", "hole_y", "hole_z",
    "hole_qx", "hole_qy", "hole_qz", "hole_qw",
    "hole_radius",
]

# Indices for geometric features (skip qpos/qvel which are near-constant at reset)
GEOM_DIMS = list(range(18, 43))  # tcp_pos through hole_radius
GEOM_LABELS = OBS_LABELS[18:]


@dataclass
class Args:
    ckpt: str = "runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt"
    env_id: str = "PegInsertionSide-v1"
    control_mode: str = "pd_joint_delta_pos"
    num_states: int = 500
    mc_samples: int = 16
    max_episode_steps: int = 200
    seed: int = 0
    zero_qvel: bool = False
    ddim_steps: int = 10
    ddim_eta: float = 1.0
    output_dir: str = "runs/analysis_deterministic_v2"


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
    print(f"  deterministic=True, DDIM steps={args.ddim_steps}")
    print(f"  MC rollout: {args.num_states} states × {args.mc_samples} samples")

    # ===== Create GPU envs =====
    N = args.num_states
    env = gym.make(
        args.env_id, num_envs=N, obs_mode="state",
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
        sim_backend="gpu", reward_mode="sparse",
    )
    env = ManiSkillVectorEnv(env, N, ignore_terminations=True, record_metrics=True)

    # Reset and save initial states
    obs_init, _ = env.reset(seed=args.seed)
    obs_init = obs_init.float().to(device)
    obs_init_raw = obs_init.cpu().numpy()
    saved_state = copy.deepcopy(env.unwrapped.get_state_dict())

    n_steps_per_ep = args.max_episode_steps // act_steps + 1
    p_success = np.zeros(N)
    t0 = time.time()

    # ===== MC rollouts (deterministic policy, GPU simulation noise) =====
    for mc in range(args.mc_samples):
        env.unwrapped.set_state_dict(copy.deepcopy(saved_state))
        obs_raw = env.unwrapped.get_obs()
        obs = obs_raw.float().to(device) if isinstance(obs_raw, torch.Tensor) else obs_raw

        obs_history = obs.unsqueeze(1).repeat(1, cond_steps, 1)
        success = torch.zeros(N, dtype=torch.bool, device=device)
        done = torch.zeros(N, dtype=torch.bool, device=device)

        for step_block in range(n_steps_per_ep):
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
                act_i = act_offset + a_idx
                action = actions[:, min(act_i, actions.shape[1] - 1)]
                obs_new, rew, term, trunc, _ = env.step(action)
                obs_new = obs_new.float().to(device)
                got_reward = rew.float() > 0.5
                success = success | (got_reward & ~done)
                done = done | term | trunc
                reset_mask = term | trunc
                if reset_mask.any():
                    obs_history[reset_mask] = obs_new[reset_mask].unsqueeze(1).repeat(1, cond_steps, 1)
                not_reset = ~reset_mask
                if not_reset.any():
                    obs_history[not_reset] = torch.cat(
                        [obs_history[not_reset, 1:], obs_new[not_reset].unsqueeze(1)], dim=1
                    )

        p_success += success.cpu().numpy()
        elapsed = time.time() - t0
        mc_sr = p_success / (mc + 1)
        print(f"  MC {mc+1}/{args.mc_samples}: SR={mc_sr.mean():.1%}, "
              f"frac_zero={(mc_sr==0).mean():.1%}, frac_one={(mc_sr==1).mean():.1%}, "
              f"time={elapsed:.0f}s")

    env.close()

    # ===== Categorize states =====
    p = p_success / args.mc_samples
    frac_zero = p == 0
    frac_one = p == 1
    decisive = (p > 0) & (p < 1)
    # Further split decisive into low/mid/high
    decisive_low = (p > 0) & (p <= 0.3)
    decisive_mid = (p > 0.3) & (p < 0.7)
    decisive_high = (p >= 0.7) & (p < 1)

    print(f"\n{'='*70}")
    print(f"  Coverage: {N} states, MC{args.mc_samples}")
    print(f"  SR = {p.mean():.1%}")
    print(f"  frac_zero  (P=0)       = {frac_zero.mean():.1%}  ({frac_zero.sum()} states)")
    print(f"  frac_one   (P=1)       = {frac_one.mean():.1%}  ({frac_one.sum()} states)")
    print(f"  decisive   (0<P<1)     = {decisive.mean():.1%}  ({decisive.sum()} states)")
    print(f"    low      (0<P≤0.3)   = {decisive_low.mean():.1%}  ({decisive_low.sum()} states)")
    print(f"    mid      (0.3<P<0.7) = {decisive_mid.mean():.1%}  ({decisive_mid.sum()} states)")
    print(f"    high     (0.7≤P<1)   = {decisive_high.mean():.1%}  ({decisive_high.sum()} states)")
    print(f"{'='*70}")

    # Save data
    np.savez(
        os.path.join(args.output_dir, "mc_rollout_data.npz"),
        obs_init=obs_init_raw, p_success=p, p_success_raw=p_success,
        mc_samples=args.mc_samples,
    )

    # ===== Analyze: what distinguishes frac_zero from the rest? =====
    obs = obs_init_raw

    # Derived geometric features
    peg_pos = obs[:, 25:28]
    hole_pos = obs[:, 35:38]
    rel_pos = peg_pos - hole_pos
    peg_hole_dist = np.linalg.norm(rel_pos, axis=1)
    peg_quat = obs[:, 28:32]
    hole_quat = obs[:, 38:42]
    # Angle between peg and hole orientation (via quaternion dot product)
    quat_dot = np.abs(np.sum(peg_quat * hole_quat, axis=1))  # |q1·q2|, 1=aligned
    peg_half = obs[:, 32:35]
    peg_radius = peg_half[:, 1]  # y dim = cross-section radius
    peg_length = peg_half[:, 0]  # x dim = half-length
    hole_radius = obs[:, 42]

    derived = {
        "peg_hole_dist": peg_hole_dist,
        "rel_x": rel_pos[:, 0],
        "rel_y": rel_pos[:, 1],
        "rel_z": rel_pos[:, 2],
        "quat_align": quat_dot,
        "peg_length": peg_length,
        "peg_radius": peg_radius,
    }

    # ===== Analysis 1: frac_zero vs rest =====
    print(f"\n--- Analysis 1: frac_zero ({frac_zero.sum()}) vs rest ({(~frac_zero).sum()}) ---")
    _compare_groups(obs, derived, frac_zero, ~frac_zero, "ZERO", "REST")

    # ===== Analysis 2: frac_zero vs frac_one =====
    if frac_one.sum() > 5:
        print(f"\n--- Analysis 2: frac_zero ({frac_zero.sum()}) vs frac_one ({frac_one.sum()}) ---")
        mask_both = frac_zero | frac_one
        _compare_groups(obs[mask_both], {k: v[mask_both] for k, v in derived.items()},
                        frac_zero[mask_both], frac_one[mask_both], "ZERO", "ONE")

    # ===== Analysis 3: P(success) correlation with features =====
    print(f"\n--- Analysis 3: Correlation of P(success) with features ---")
    print(f"  {'Feature':>20} {'Pearson r':>10} {'p-value':>10}")
    print(f"  {'-'*42}")
    corrs = {}
    for name, vals in derived.items():
        r, pval = stats.pearsonr(vals, p)
        corrs[name] = r
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {name:>20} {r:+10.3f} {pval:10.2e} {sig}")

    # Also check raw obs geometric dims
    print(f"\n  Raw obs geometric dims:")
    for d in GEOM_DIMS:
        r, pval = stats.pearsonr(obs[:, d], p)
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        if abs(r) > 0.08:
            print(f"  dim {d:2d} {OBS_LABELS[d]:>16} {r:+10.3f} {pval:10.2e} {sig}")

    # ===== Analysis 4: Decision boundary visualization =====
    # What if we predict P(success) from just peg_radius + peg_hole_dist?
    print(f"\n--- Analysis 4: P(success) vs (peg_radius, peg_hole_dist) ---")
    # Bin by peg_radius and show mean P per bin
    radius_bins = np.linspace(peg_radius.min(), peg_radius.max(), 8)
    print(f"  {'Peg radius bin':>20} {'N':>5} {'mean P':>8} {'frac_zero':>10} {'frac_one':>10}")
    for i in range(len(radius_bins) - 1):
        mask = (peg_radius >= radius_bins[i]) & (peg_radius < radius_bins[i+1])
        if mask.sum() == 0:
            continue
        print(f"  [{radius_bins[i]:.4f}, {radius_bins[i+1]:.4f}) {mask.sum():5d} "
              f"{p[mask].mean():8.3f} {frac_zero[mask].mean():10.1%} {frac_one[mask].mean():10.1%}")

    dist_bins = np.linspace(peg_hole_dist.min(), peg_hole_dist.max(), 8)
    print(f"\n  {'Peg-hole dist bin':>20} {'N':>5} {'mean P':>8} {'frac_zero':>10} {'frac_one':>10}")
    for i in range(len(dist_bins) - 1):
        mask = (peg_hole_dist >= dist_bins[i]) & (peg_hole_dist < dist_bins[i+1])
        if mask.sum() == 0:
            continue
        print(f"  [{dist_bins[i]:.3f}, {dist_bins[i+1]:.3f}) {mask.sum():5d} "
              f"{p[mask].mean():8.3f} {frac_zero[mask].mean():10.1%} {frac_one[mask].mean():10.1%}")

    # ===== Plots =====
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: P(success) histogram
    ax = axes[0, 0]
    bins = np.linspace(0, 1, 18)
    ax.hist(p, bins=bins, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(p.mean(), color="red", ls="--", lw=2, label=f"SR={p.mean():.1%}")
    ax.set_xlabel("P(success|s₀)")
    ax.set_ylabel("Count")
    ax.set_title("Coverage Distribution")
    ax.legend()

    # Plot 2: P(success) vs peg_radius — scatter
    ax = axes[0, 1]
    ax.scatter(peg_radius, p, c=p, cmap="RdYlGn", s=15, alpha=0.6, edgecolors="none")
    ax.set_xlabel("Peg radius")
    ax.set_ylabel("P(success|s₀)")
    ax.set_title(f"P(success) vs Peg Radius (r={corrs['peg_radius']:.3f})")
    ax.set_ylim(-0.05, 1.05)

    # Plot 3: P(success) vs peg-hole distance
    ax = axes[0, 2]
    ax.scatter(peg_hole_dist, p, c=p, cmap="RdYlGn", s=15, alpha=0.6, edgecolors="none")
    ax.set_xlabel("Peg-Hole distance")
    ax.set_ylabel("P(success|s₀)")
    ax.set_title(f"P(success) vs Distance (r={corrs['peg_hole_dist']:.3f})")
    ax.set_ylim(-0.05, 1.05)

    # Plot 4: P(success) vs quat alignment
    ax = axes[1, 0]
    ax.scatter(quat_dot, p, c=p, cmap="RdYlGn", s=15, alpha=0.6, edgecolors="none")
    ax.set_xlabel("|peg_quat · hole_quat| (1=aligned)")
    ax.set_ylabel("P(success|s₀)")
    ax.set_title(f"P(success) vs Orientation Alignment (r={corrs['quat_align']:.3f})")
    ax.set_ylim(-0.05, 1.05)

    # Plot 5: 2D — peg_radius vs peg_hole_dist, colored by P(success)
    ax = axes[1, 1]
    sc = ax.scatter(peg_radius, peg_hole_dist, c=p, cmap="RdYlGn", s=20, alpha=0.7, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="P(success)")
    # Mark frac_zero
    ax.scatter(peg_radius[frac_zero], peg_hole_dist[frac_zero],
               c="black", marker="x", s=30, label=f"frac_zero ({frac_zero.sum()})")
    ax.set_xlabel("Peg radius")
    ax.set_ylabel("Peg-Hole distance")
    ax.set_title("2D: Radius × Distance → P(success)")
    ax.legend(fontsize=8)

    # Plot 6: 2D — peg_radius vs quat_align, colored by P(success)
    ax = axes[1, 2]
    sc = ax.scatter(peg_radius, quat_dot, c=p, cmap="RdYlGn", s=20, alpha=0.7, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="P(success)")
    ax.scatter(peg_radius[frac_zero], quat_dot[frac_zero],
               c="black", marker="x", s=30, label=f"frac_zero ({frac_zero.sum()})")
    ax.set_xlabel("Peg radius")
    ax.set_ylabel("|peg_quat · hole_quat|")
    ax.set_title("2D: Radius × Alignment → P(success)")
    ax.legend(fontsize=8)

    plt.suptitle(f"Deterministic DP Coverage Analysis (SR={p.mean():.1%}, N={N}, MC{args.mc_samples})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "coverage_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {plot_path}")


def _compare_groups(obs, derived, mask_a, mask_b, name_a, name_b):
    """Compare obs features between two groups."""
    n_a = mask_a.sum()
    n_b = mask_b.sum()
    if n_a < 3 or n_b < 3:
        print(f"  Skipping: too few samples ({n_a} vs {n_b})")
        return

    print(f"\n  Derived features:")
    print(f"  {'Feature':>20} {'mean_'+name_a:>12} {'mean_'+name_b:>12} {'diff':>10} {'t-stat':>8} {'sig':>4}")
    for fname, vals in derived.items():
        va = vals[mask_a]
        vb = vals[mask_b]
        if va.std() < 1e-12 and vb.std() < 1e-12:
            continue
        t, pval = stats.ttest_ind(va, vb, equal_var=False)
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        diff = va.mean() - vb.mean()
        print(f"  {fname:>20} {va.mean():12.4f} {vb.mean():12.4f} {diff:+10.4f} {t:8.2f} {sig:>4}")


if __name__ == "__main__":
    main()
