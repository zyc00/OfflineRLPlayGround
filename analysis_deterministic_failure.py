"""Analyze what initial state features determine success/failure for deterministic DP.

Runs deterministic DDIM rollouts on many initial states, saves obs + success labels,
then analyzes correlations between obs dimensions and success.

For PegInsertionSide-v1 (pd_joint_delta_pos), obs is 43D:
  0-6:   arm qpos (7 joints)
  7-8:   gripper qpos (2 fingers)
  9-15:  arm qvel (7 joints)
  16-17: gripper qvel (2 fingers)
  18-20: TCP position (x,y,z)
  21-24: TCP orientation (quat xyzw)
  25-27: Peg position (x,y,z)
  28-31: Peg orientation (quat xyzw)
  32-34: Peg half-sizes (x,y,z)
  35-37: Box hole position (x,y,z)
  38-41: Box hole orientation (quat xyzw)
  42:    Hole radius

Usage:
  python analysis_deterministic_failure.py \
    --ckpt runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt \
    --num_states 1000 --ddim_steps 10
"""
import copy
import os
import sys
import time
from dataclasses import dataclass, field
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


@dataclass
class Args:
    ckpt: str = "runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt"
    env_id: str = "PegInsertionSide-v1"
    control_mode: str = "pd_joint_delta_pos"
    num_states: int = 1000
    max_episode_steps: int = 200
    seed: int = 0
    zero_qvel: bool = False
    ddim_steps: int = 10
    ddim_eta: float = 1.0
    output_dir: str = "runs/analysis_deterministic"


def main():
    args = tyro.cli(Args)
    device = torch.device("cuda")
    os.makedirs(args.output_dir, exist_ok=True)

    # ===== Load model =====
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
    act_offset = cond_steps - 1

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

    print(f"Loaded: {args.ckpt}")
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}, zero_qvel={args.zero_qvel}")
    print(f"  DDIM steps={args.ddim_steps}, deterministic=True")

    # ===== Create GPU envs =====
    N = args.num_states
    env = gym.make(
        args.env_id, num_envs=N, obs_mode="state",
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
        sim_backend="gpu", reward_mode="sparse",
    )
    env = ManiSkillVectorEnv(env, N, ignore_terminations=True, record_metrics=True)

    obs_init, _ = env.reset(seed=args.seed)
    obs_init = obs_init.float().to(device)

    # Save raw initial observations (before zero_qvel)
    obs_init_raw = obs_init.cpu().numpy()

    # Also save intermediate trajectory data for failure analysis
    n_steps_per_ep = args.max_episode_steps // act_steps + 1

    # Run deterministic rollout
    obs_history = obs_init.unsqueeze(1).repeat(1, cond_steps, 1)
    success = torch.zeros(N, dtype=torch.bool, device=device)
    done = torch.zeros(N, dtype=torch.bool, device=device)
    # Track when each env succeeds (step number)
    success_step = torch.full((N,), -1, dtype=torch.long, device=device)
    # Track trajectory: save obs at each decision step
    traj_obs = []  # list of (N, obs_dim) tensors
    traj_actions = []  # list of (N, act_dim) tensors (first action of each chunk)
    total_steps = 0

    print(f"\nRunning deterministic rollout on {N} states...")
    t0 = time.time()

    for step_block in range(n_steps_per_ep):
        if done.all():
            break

        obs_cond = obs_history
        if args.zero_qvel:
            obs_cond = obs_cond.clone()
            obs_cond[..., 9:18] = 0.0
        cond = {"state": obs_cond}

        with torch.no_grad():
            samples = model(
                cond, deterministic=True,
                ddim_steps=args.ddim_steps,
            )
        actions = samples.trajectories

        # Save trajectory info
        traj_obs.append(obs_history[:, -1].cpu())  # latest obs
        traj_actions.append(actions[:, act_offset].cpu())  # first action in chunk

        for a_idx in range(act_steps):
            act_i = act_offset + a_idx
            action = actions[:, min(act_i, actions.shape[1] - 1)]
            obs_new, rew, term, trunc, _ = env.step(action)
            obs_new = obs_new.float().to(device)
            total_steps += 1

            got_reward = rew.float() > 0.5
            newly_success = got_reward & ~done & ~success
            success_step[newly_success] = total_steps
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

    elapsed = time.time() - t0
    success_np = success.cpu().numpy()
    success_step_np = success_step.cpu().numpy()
    sr = success_np.mean()
    print(f"Done in {elapsed:.1f}s. SR={sr:.1%} ({success_np.sum()}/{N})")

    # ===== Save data =====
    data_path = os.path.join(args.output_dir, "deterministic_rollout_data.npz")
    np.savez(
        data_path,
        obs_init=obs_init_raw,  # (N, obs_dim) raw initial obs
        success=success_np,  # (N,) bool
        success_step=success_step_np,  # (N,) int, -1 if failed
        sr=sr,
    )
    print(f"Saved data to {data_path}")

    env.close()

    # ===== Analysis =====
    analyze(obs_init_raw, success_np, success_step_np, args.output_dir, obs_dim)


def analyze(obs_init, success, success_step, output_dir, obs_dim):
    """Analyze what initial state features predict success/failure."""
    N = len(success)
    sr = success.mean()
    n_succ = success.sum()
    n_fail = N - n_succ
    print(f"\n{'='*70}")
    print(f"  ANALYSIS: {N} states, SR={sr:.1%} ({n_succ} success, {n_fail} fail)")
    print(f"{'='*70}")

    obs_s = obs_init[success]
    obs_f = obs_init[~success]

    # 1. Per-dimension mean/std comparison
    print(f"\n--- Per-dimension statistics (success vs fail) ---")
    print(f"{'Dim':>4} {'Label':>16} {'mean_S':>8} {'mean_F':>8} {'diff':>8} {'std_S':>8} {'std_F':>8} {'t-stat':>8} {'signif':>6}")
    print("-" * 90)

    labels = OBS_LABELS if obs_dim <= len(OBS_LABELS) else [f"dim{i}" for i in range(obs_dim)]
    t_stats = np.zeros(obs_dim)
    p_values = np.zeros(obs_dim)

    from scipy import stats

    for d in range(obs_dim):
        ms = obs_s[:, d].mean()
        mf = obs_f[:, d].mean()
        ss = obs_s[:, d].std()
        sf = obs_f[:, d].std()
        diff = ms - mf

        # Welch's t-test
        if ss > 1e-10 or sf > 1e-10:
            t, p = stats.ttest_ind(obs_s[:, d], obs_f[:, d], equal_var=False)
        else:
            t, p = 0.0, 1.0
        t_stats[d] = t
        p_values[d] = p

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if p < 0.05:
            print(f"{d:4d} {labels[d]:>16} {ms:8.4f} {mf:8.4f} {diff:+8.4f} {ss:8.4f} {sf:8.4f} {t:8.2f} {sig:>6}")

    # 2. Top features by |t-stat|
    print(f"\n--- Top 10 features by |t-statistic| ---")
    top_idx = np.argsort(np.abs(t_stats))[::-1][:10]
    for rank, d in enumerate(top_idx):
        ms = obs_s[:, d].mean()
        mf = obs_f[:, d].mean()
        print(f"  #{rank+1}: dim {d:2d} ({labels[d]:>16}) | mean_S={ms:.4f}, mean_F={mf:.4f}, "
              f"diff={ms-mf:+.4f}, t={t_stats[d]:.2f}, p={p_values[d]:.2e}")

    # 3. Derived features: relative peg-hole geometry
    print(f"\n--- Derived geometric features ---")

    # Peg position relative to hole
    peg_pos = obs_init[:, 25:28]  # (N,3)
    hole_pos = obs_init[:, 35:38]  # (N,3)
    rel_pos = peg_pos - hole_pos  # peg relative to hole
    peg_hole_dist = np.linalg.norm(rel_pos, axis=1)

    # Peg orientation (quaternion) - extract z-rotation angle
    peg_quat = obs_init[:, 28:32]  # qx,qy,qz,qw
    hole_quat = obs_init[:, 38:42]

    # Simple: use qz component as proxy for z-rotation
    peg_qz = peg_quat[:, 2]
    hole_qz = hole_quat[:, 2]
    rot_diff = np.abs(peg_qz - hole_qz)

    # Peg size
    peg_half = obs_init[:, 32:35]
    hole_radius = obs_init[:, 42]

    # Clearance: hole_radius - peg_half_y (cross section)
    clearance = hole_radius - peg_half[:, 1]

    derived = {
        "peg_hole_dist": peg_hole_dist,
        "rel_x": rel_pos[:, 0],
        "rel_y": rel_pos[:, 1],
        "rel_z": rel_pos[:, 2],
        "rot_diff_qz": rot_diff,
        "peg_length": peg_half[:, 0],
        "peg_radius": peg_half[:, 1],
        "clearance": clearance,
    }

    for name, vals in derived.items():
        vs = vals[success]
        vf = vals[~success]
        t, p = stats.ttest_ind(vs, vf, equal_var=False) if vs.std() > 1e-10 else (0.0, 1.0)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {name:>20}: mean_S={vs.mean():.4f}±{vs.std():.4f}, "
              f"mean_F={vf.mean():.4f}±{vf.std():.4f}, t={t:.2f} {sig}")

    # 4. Logistic regression: which features predict success?
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = obs_init.copy()
    # Add derived features
    X_derived = np.column_stack([v for v in derived.values()])
    X_all = np.hstack([X, X_derived])
    feature_names = labels[:obs_dim] + list(derived.keys())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    y = success.astype(int)

    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X_scaled, y)
    train_acc = lr.score(X_scaled, y)
    print(f"\n--- Logistic Regression (all features) ---")
    print(f"  Train accuracy: {train_acc:.1%}")

    coefs = lr.coef_[0]
    top_pos = np.argsort(coefs)[::-1][:10]
    top_neg = np.argsort(coefs)[:10]

    print(f"\n  Top 10 POSITIVE predictors (more → more likely to succeed):")
    for i, idx in enumerate(top_pos):
        print(f"    #{i+1}: {feature_names[idx]:>20} coef={coefs[idx]:+.3f}")

    print(f"\n  Top 10 NEGATIVE predictors (more → more likely to fail):")
    for i, idx in enumerate(top_neg):
        print(f"    #{i+1}: {feature_names[idx]:>20} coef={coefs[idx]:+.3f}")

    # 5. Plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Peg-hole distance vs success
    ax = axes[0, 0]
    ax.hist(peg_hole_dist[success], bins=30, alpha=0.6, color="green", label="Success", density=True)
    ax.hist(peg_hole_dist[~success], bins=30, alpha=0.6, color="red", label="Fail", density=True)
    ax.set_xlabel("Peg-Hole distance")
    ax.set_ylabel("Density")
    ax.set_title("Peg-Hole Distance")
    ax.legend()

    # Plot 2: Relative Y (peg_y - hole_y) vs success
    ax = axes[0, 1]
    ax.hist(rel_pos[success, 1], bins=30, alpha=0.6, color="green", label="Success", density=True)
    ax.hist(rel_pos[~success, 1], bins=30, alpha=0.6, color="red", label="Fail", density=True)
    ax.set_xlabel("Peg Y - Hole Y")
    ax.set_title("Relative Y Position")
    ax.legend()

    # Plot 3: Rotation difference
    ax = axes[0, 2]
    ax.hist(rot_diff[success], bins=30, alpha=0.6, color="green", label="Success", density=True)
    ax.hist(rot_diff[~success], bins=30, alpha=0.6, color="red", label="Fail", density=True)
    ax.set_xlabel("|peg_qz - hole_qz|")
    ax.set_title("Rotation Difference (qz proxy)")
    ax.legend()

    # Plot 4: Peg size vs success
    ax = axes[1, 0]
    ax.hist(peg_half[success, 0], bins=30, alpha=0.6, color="green", label="Success", density=True)
    ax.hist(peg_half[~success, 0], bins=30, alpha=0.6, color="red", label="Fail", density=True)
    ax.set_xlabel("Peg half-length")
    ax.set_title("Peg Size (half-length)")
    ax.legend()

    # Plot 5: Clearance vs success
    ax = axes[1, 1]
    ax.hist(clearance[success], bins=30, alpha=0.6, color="green", label="Success", density=True)
    ax.hist(clearance[~success], bins=30, alpha=0.6, color="red", label="Fail", density=True)
    ax.set_xlabel("Hole radius - Peg radius")
    ax.set_title("Clearance")
    ax.legend()

    # Plot 6: 2D scatter of top 2 derived features
    ax = axes[1, 2]
    # Use |coef| to pick top 2 derived features
    derived_coefs = coefs[obs_dim:]
    derived_names = list(derived.keys())
    top2 = np.argsort(np.abs(derived_coefs))[::-1][:2]
    dx = list(derived.values())[top2[0]]
    dy = list(derived.values())[top2[1]]
    ax.scatter(dx[success], dy[success], c="green", alpha=0.3, s=10, label="Success")
    ax.scatter(dx[~success], dy[~success], c="red", alpha=0.3, s=10, label="Fail")
    ax.set_xlabel(derived_names[top2[0]])
    ax.set_ylabel(derived_names[top2[1]])
    ax.set_title(f"Top 2 Derived Features")
    ax.legend()

    plt.suptitle(f"Deterministic DP Failure Analysis (SR={sr:.1%}, N={N})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "failure_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot: {plot_path}")

    # 6. Success step distribution (how quickly do successful envs finish?)
    succ_steps = success_step[success]
    if len(succ_steps) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(succ_steps, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
        ax.axvline(np.median(succ_steps), color="red", ls="--", label=f"Median={np.median(succ_steps):.0f}")
        ax.set_xlabel("Step at success")
        ax.set_ylabel("Count")
        ax.set_title(f"Success Step Distribution (N={len(succ_steps)})")
        ax.legend()
        plt.tight_layout()
        plot2_path = os.path.join(output_dir, "success_step_dist.png")
        plt.savefig(plot2_path, dpi=150)
        plt.close()
        print(f"Saved plot: {plot2_path}")
        print(f"  Success step: median={np.median(succ_steps):.0f}, "
              f"mean={np.mean(succ_steps):.0f}, min={np.min(succ_steps)}, max={np.max(succ_steps)}")


if __name__ == "__main__":
    main()
