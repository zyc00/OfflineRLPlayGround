"""Analyze DP vs MIP predictions on val set states.

Loads demo data, uses last 100 trajectories as val set.
Runs both models in eval mode (deterministic) on val states.
Compares predicted actions vs ground truth, focusing on insertion phase.

Usage:
    python scripts/analyze_val_predictions.py \
      --dp_ckpt runs/dppo_pretrain/dp_peg_split900/best.pt \
      --mip_ckpt runs/mip_pretrain/mip_peg_split900/best.pt \
      --output /tmp/val_analysis
"""
import os
import sys
import argparse
import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from DPPO.model.unet_wrapper import DiffusionUNet
from DPPO.model.diffusion import DiffusionModel
from MultiGaussian.models.multi_gaussian import MIPPolicy
from MultiGaussian.models.mip_unet import MIPUNetPolicy


def load_dp_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch_args = ckpt.get("pretrain_args", ckpt["args"])
    if "pretrain_args" in ckpt and ckpt["pretrain_args"] is not None:
        arch_args = ckpt["pretrain_args"]
    else:
        arch_args = ckpt["args"]
    obs_dim = ckpt["obs_dim"]
    act_dim = ckpt["action_dim"]
    cond_steps = arch_args.get("cond_steps", 2)
    horizon_steps = arch_args.get("horizon_steps", 16)

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
        final_action_clip_value=1.0,
        predict_epsilon=arch_args.get("predict_epsilon", True),
        mip_noise=arch_args.get("mip_noise", False),
    )
    if arch_args.get("fixed_t_points"):
        model.fixed_t_points = torch.tensor(arch_args["fixed_t_points"], dtype=torch.long)
    state_key = "ema" if "ema" in ckpt else "model"
    model.load_state_dict(ckpt[state_key], strict=False)
    if torch.isnan(model.eta.eta_logit.data).any():
        model.eta.eta_logit.data = torch.atanh(torch.tensor([0.999]))
    model.eval()
    return model, cond_steps, horizon_steps


def load_mip_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    network_type = args.get("network_type", "mlp")
    if network_type == "unet":
        model = MIPUNetPolicy(
            input_dim=ckpt["obs_dim"], action_dim=ckpt["action_dim"],
            cond_steps=args.get("cond_steps", 1),
            horizon_steps=args.get("horizon_steps", 1),
            t_star=args.get("t_star", 0.9),
        ).to(device)
    else:
        model = MIPPolicy(
            input_dim=ckpt["obs_dim"], action_dim=ckpt["action_dim"],
            cond_steps=args.get("cond_steps", 1),
            horizon_steps=args.get("horizon_steps", 1),
            t_star=args.get("t_star", 0.9),
            dropout=args.get("dropout", 0.1),
            emb_dim=args.get("emb_dim", 512),
            n_layers=args.get("n_layers", 6),
            predict_epsilon=args.get("predict_epsilon", False),
        ).to(device)
    state_key = "ema" if "ema" in ckpt else "model"
    model.load_state_dict(ckpt[state_key])
    model.eval()
    return model, args.get("cond_steps", 1), args.get("horizon_steps", 1)


def load_demo_data(demo_path, cond_steps=2, horizon_steps=16, zero_qvel=True):
    """Load raw demo trajectories, return list of (obs_seq, action_seq) per traj."""
    with h5py.File(demo_path, "r") as f:
        traj_keys = sorted([k for k in f.keys() if k.startswith("traj_")],
                          key=lambda x: int(x.split("_")[1]))
        trajs = []
        for tk in traj_keys:
            obs = f[tk]["obs"][:]       # (T+1, obs_dim)
            actions = f[tk]["actions"][:] # (T, act_dim)
            trajs.append({"obs": obs, "actions": actions})
    return trajs


def extract_samples(trajs, cond_steps, horizon_steps, zero_qvel=True):
    """Extract (cond_obs, action_horizon) samples from trajectories.

    Returns:
        obs: (N, cond_steps, obs_dim) - conditioning observations
        actions: (N, horizon_steps, act_dim) - ground truth action chunks
        traj_idx: (N,) - which trajectory each sample comes from
        step_idx: (N,) - which step in the trajectory
        traj_len: (N,) - total length of the source trajectory
    """
    all_obs = []
    all_actions = []
    all_traj_idx = []
    all_step_idx = []
    all_traj_len = []

    for ti, traj in enumerate(trajs):
        obs = traj["obs"]      # (T+1, obs_dim)
        act = traj["actions"]  # (T, act_dim)
        T = len(act)

        for t in range(T - horizon_steps + 1):
            # Conditioning obs: [t, t+1, ..., t+cond_steps-1]
            # We need cond_steps consecutive obs ending at t+cond_steps-1
            if t + cond_steps - 1 >= len(obs):
                continue
            obs_cond = obs[t:t+cond_steps]  # (cond_steps, obs_dim)

            # Action horizon starting from t
            act_horizon = act[t:t+horizon_steps]  # (horizon_steps, act_dim)
            if len(act_horizon) < horizon_steps:
                # Pad with last action (gripper) + zeros (arm)
                pad_len = horizon_steps - len(act_horizon)
                last_act = act_horizon[-1:].copy()
                last_act[:, :-1] = 0  # zero arm, keep gripper
                padding = np.tile(last_act, (pad_len, 1))
                act_horizon = np.concatenate([act_horizon, padding])

            if zero_qvel:
                obs_cond = obs_cond.copy()
                obs_cond[..., 9:18] = 0.0

            all_obs.append(obs_cond)
            all_actions.append(act_horizon)
            all_traj_idx.append(ti)
            all_step_idx.append(t)
            all_traj_len.append(T)

    return {
        "obs": np.array(all_obs),
        "actions": np.array(all_actions),
        "traj_idx": np.array(all_traj_idx),
        "step_idx": np.array(all_step_idx),
        "traj_len": np.array(all_traj_len),
    }


@torch.no_grad()
def predict_dp(model, obs_batch, cond_steps):
    """DP deterministic prediction. obs_batch: (B, cond_steps, obs_dim)"""
    cond = {"state": obs_batch}
    samples = model(cond, deterministic=True, ddim_steps=10)
    return samples.trajectories  # (B, horizon, act_dim)


@torch.no_grad()
def predict_mip(model, obs_batch, cond_steps):
    """MIP deterministic prediction. obs_batch: (B, cond_steps, obs_dim)"""
    return model.predict(obs_batch)  # (B, horizon, act_dim) or (B, act_dim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp_ckpt", required=True)
    parser.add_argument("--mip_ckpt", required=True)
    parser.add_argument("--demo_path", default="/home/jigu/.maniskill/demos/PegInsertionSide-v1/policy_from_mp_states/trajectory.state.pd_joint_delta_pos.h5")
    parser.add_argument("--n_train", type=int, default=900)
    parser.add_argument("--output", default="/tmp/val_analysis")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda")
    os.makedirs(args.output, exist_ok=True)

    # Load models
    print("Loading DP model...")
    dp_model, dp_cond, dp_horizon = load_dp_model(args.dp_ckpt, device)
    print("Loading MIP model...")
    mip_model, mip_cond, mip_horizon = load_mip_model(args.mip_ckpt, device)

    cond_steps = max(dp_cond, mip_cond)  # should be 2 for both
    horizon_steps = max(dp_horizon, mip_horizon)  # should be 16 for both
    print(f"  cond_steps={cond_steps}, horizon_steps={horizon_steps}")

    # Load data
    print(f"Loading demos from {args.demo_path}...")
    trajs = load_demo_data(args.demo_path, cond_steps, horizon_steps)
    n_total = len(trajs)
    n_train = args.n_train
    n_val = n_total - n_train
    print(f"  {n_total} trajs total, {n_train} train, {n_val} val")

    train_trajs = trajs[:n_train]
    val_trajs = trajs[n_train:]

    # Extract val samples
    print("Extracting val samples...")
    val_data = extract_samples(val_trajs, cond_steps, horizon_steps, zero_qvel=True)
    N = len(val_data["obs"])
    print(f"  {N} val samples from {n_val} trajectories")

    # Also extract train samples for reference
    print("Extracting train samples (for reference)...")
    train_data = extract_samples(train_trajs, cond_steps, horizon_steps, zero_qvel=True)
    print(f"  {len(train_data['obs'])} train samples from {n_train} trajectories")

    # Run predictions in batches
    print("Running DP predictions on val set...")
    dp_preds = []
    for i in range(0, N, args.batch_size):
        batch = torch.tensor(val_data["obs"][i:i+args.batch_size], dtype=torch.float32, device=device)
        pred = predict_dp(dp_model, batch, cond_steps)
        dp_preds.append(pred.cpu().numpy())
    dp_preds = np.concatenate(dp_preds, axis=0)  # (N, horizon, act_dim)

    print("Running MIP predictions on val set...")
    mip_preds = []
    for i in range(0, N, args.batch_size):
        batch = torch.tensor(val_data["obs"][i:i+args.batch_size], dtype=torch.float32, device=device)
        pred = predict_mip(mip_model, batch, cond_steps)
        if pred.dim() == 2:
            pred = pred.unsqueeze(1)
        mip_preds.append(pred.cpu().numpy())
    mip_preds = np.concatenate(mip_preds, axis=0)

    gt_actions = val_data["actions"]  # (N, horizon, act_dim)
    act_dim = gt_actions.shape[2]

    # Compute errors
    dp_err = ((dp_preds - gt_actions) ** 2).mean(axis=2)   # (N, horizon)
    mip_err = ((mip_preds - gt_actions) ** 2).mean(axis=2)  # (N, horizon)
    dp_mse = dp_err.mean(axis=1)   # (N,)
    mip_mse = mip_err.mean(axis=1)  # (N,)

    # Episode phase: step_idx / traj_len
    phase = val_data["step_idx"] / val_data["traj_len"]  # 0=start, 1=end

    print(f"\n{'='*60}")
    print(f"Val Set Prediction Analysis ({N} samples)")
    print(f"{'='*60}")
    print(f"  DP  MSE (all):  {dp_mse.mean():.6f}")
    print(f"  MIP MSE (all):  {mip_mse.mean():.6f}")
    print(f"  Ratio MIP/DP:   {mip_mse.mean() / dp_mse.mean():.2f}x")

    # By phase
    phase_bins = [(0, 0.25, "approach (0-25%)"),
                  (0.25, 0.5, "pre-insert (25-50%)"),
                  (0.5, 0.75, "insert (50-75%)"),
                  (0.75, 1.01, "post-insert (75-100%)")]

    print(f"\n--- MSE by Episode Phase ---")
    print(f"  {'Phase':<25s} {'DP MSE':>10s} {'MIP MSE':>10s} {'Ratio':>8s} {'N':>6s}")
    phase_stats = []
    for lo, hi, label in phase_bins:
        mask = (phase >= lo) & (phase < hi)
        n_phase = mask.sum()
        if n_phase > 0:
            dp_phase = dp_mse[mask].mean()
            mip_phase = mip_mse[mask].mean()
            ratio = mip_phase / dp_phase if dp_phase > 0 else float('inf')
            print(f"  {label:<25s} {dp_phase:10.6f} {mip_phase:10.6f} {ratio:8.2f}x {n_phase:6d}")
            phase_stats.append((label, dp_phase, mip_phase, n_phase))

    # Per-joint analysis
    dp_joint_err = ((dp_preds - gt_actions) ** 2).mean(axis=(0, 1))  # (act_dim,)
    mip_joint_err = ((mip_preds - gt_actions) ** 2).mean(axis=(0, 1))

    print(f"\n--- Per-Joint MSE ---")
    print(f"  {'Joint':>6s} {'DP MSE':>10s} {'MIP MSE':>10s} {'Ratio':>8s}")
    for j in range(act_dim):
        dp_j = ((dp_preds[:, :, j] - gt_actions[:, :, j]) ** 2).mean()
        mip_j = ((mip_preds[:, :, j] - gt_actions[:, :, j]) ** 2).mean()
        ratio = mip_j / dp_j if dp_j > 0 else float('inf')
        print(f"  j{j:d}:    {dp_j:10.6f} {mip_j:10.6f} {ratio:8.2f}x")

    # Insertion phase deep dive (phase 50-100%)
    insert_mask = phase >= 0.5
    insert_dp = dp_preds[insert_mask]
    insert_mip = mip_preds[insert_mask]
    insert_gt = gt_actions[insert_mask]
    insert_obs = val_data["obs"][insert_mask]

    print(f"\n--- Insertion Phase (>50%) Deep Dive ({insert_mask.sum()} samples) ---")

    # Action bias: mean(pred - gt) per joint
    dp_bias = (insert_dp - insert_gt).mean(axis=(0, 1))
    mip_bias = (insert_mip - insert_gt).mean(axis=(0, 1))
    print(f"\n  Action Bias (mean pred - gt):")
    print(f"  {'Joint':>6s} {'DP bias':>10s} {'MIP bias':>10s}")
    for j in range(act_dim):
        print(f"  j{j:d}:    {dp_bias[j]:+10.5f} {mip_bias[j]:+10.5f}")

    # Action variance: how diverse are predictions across insertion states
    dp_insert_std = insert_dp.std(axis=0).mean(axis=0)  # (act_dim,)
    mip_insert_std = insert_mip.std(axis=0).mean(axis=0)
    gt_insert_std = insert_gt.std(axis=0).mean(axis=0)
    print(f"\n  Action Std (across samples, averaged over horizon):")
    print(f"  {'Joint':>6s} {'GT std':>10s} {'DP std':>10s} {'MIP std':>10s} {'DP/GT':>8s} {'MIP/GT':>8s}")
    for j in range(act_dim):
        dp_ratio = dp_insert_std[j] / gt_insert_std[j] if gt_insert_std[j] > 0 else 0
        mip_ratio = mip_insert_std[j] / gt_insert_std[j] if gt_insert_std[j] > 0 else 0
        print(f"  j{j:d}:    {gt_insert_std[j]:10.5f} {dp_insert_std[j]:10.5f} {mip_insert_std[j]:10.5f} "
              f"{dp_ratio:8.2f} {mip_ratio:8.2f}")

    # Correlation: how well does pred track GT variation
    print(f"\n  Prediction-GT Correlation (per joint, insertion phase, first action in horizon):")
    print(f"  {'Joint':>6s} {'DP corr':>10s} {'MIP corr':>10s}")
    for j in range(act_dim):
        gt_j = insert_gt[:, 0, j]  # first action step
        dp_j = insert_dp[:, 0, j]
        mip_j = insert_mip[:, 0, j]
        dp_corr = np.corrcoef(gt_j, dp_j)[0, 1] if gt_j.std() > 1e-8 else 0
        mip_corr = np.corrcoef(gt_j, mip_j)[0, 1] if gt_j.std() > 1e-8 else 0
        print(f"  j{j:d}:    {dp_corr:10.4f} {mip_corr:10.4f}")

    # Training data comparison: what does training data look like at insertion phase
    train_phase = train_data["step_idx"] / train_data["traj_len"]
    train_insert_mask = train_phase >= 0.5
    train_insert_acts = train_data["actions"][train_insert_mask]

    print(f"\n  Training Data at Insertion Phase ({train_insert_mask.sum()} samples):")
    print(f"  {'Joint':>6s} {'train mean':>10s} {'train std':>10s} {'DP mean':>10s} {'MIP mean':>10s}")
    for j in range(act_dim):
        tm = train_insert_acts[:, 0, j].mean()
        ts = train_insert_acts[:, 0, j].std()
        dm = insert_dp[:, 0, j].mean()
        mm = insert_mip[:, 0, j].mean()
        print(f"  j{j:d}:    {tm:+10.5f} {ts:10.5f} {dm:+10.5f} {mm:+10.5f}")

    # ========== PLOTS ==========
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. MSE by phase
    ax = axes[0, 0]
    labels = [s[0] for s in phase_stats]
    dp_vals = [s[1] for s in phase_stats]
    mip_vals = [s[2] for s in phase_stats]
    x = np.arange(len(labels))
    ax.bar(x - 0.15, dp_vals, width=0.3, label="DP", alpha=0.7, color="steelblue")
    ax.bar(x + 0.15, mip_vals, width=0.3, label="MIP", alpha=0.7, color="coral")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=8)
    ax.set_ylabel("MSE"); ax.set_title("MSE by Episode Phase"); ax.legend()

    # 2. Per-joint MSE
    ax = axes[0, 1]
    x_j = np.arange(act_dim)
    ax.bar(x_j - 0.15, dp_joint_err, width=0.3, label="DP", alpha=0.7, color="steelblue")
    ax.bar(x_j + 0.15, mip_joint_err, width=0.3, label="MIP", alpha=0.7, color="coral")
    ax.set_xticks(x_j); ax.set_xticklabels([f"j{i}" for i in range(act_dim)])
    ax.set_ylabel("MSE"); ax.set_title("Per-Joint MSE"); ax.legend()

    # 3. MSE scatter: DP vs MIP per sample
    ax = axes[0, 2]
    ax.scatter(dp_mse, mip_mse, alpha=0.1, s=5, c=phase, cmap="coolwarm")
    max_val = max(dp_mse.max(), mip_mse.max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
    ax.set_xlabel("DP MSE"); ax.set_ylabel("MIP MSE")
    ax.set_title("Per-sample MSE (color=phase)")
    ax.set_aspect("equal")

    # 4. Action bias at insertion phase
    ax = axes[1, 0]
    ax.bar(x_j - 0.15, dp_bias, width=0.3, label="DP", alpha=0.7, color="steelblue")
    ax.bar(x_j + 0.15, mip_bias, width=0.3, label="MIP", alpha=0.7, color="coral")
    ax.set_xticks(x_j); ax.set_xticklabels([f"j{i}" for i in range(act_dim)])
    ax.set_ylabel("Bias (pred - GT)"); ax.set_title("Action Bias (Insertion Phase)"); ax.legend()
    ax.axhline(0, color="black", lw=0.5)

    # 5. MSE as function of phase (continuous)
    ax = axes[1, 1]
    # Bin by phase
    n_bins = 20
    phase_edges = np.linspace(0, 1, n_bins + 1)
    dp_curve = []
    mip_curve = []
    phase_centers = []
    for i in range(n_bins):
        mask = (phase >= phase_edges[i]) & (phase < phase_edges[i+1])
        if mask.sum() > 0:
            dp_curve.append(dp_mse[mask].mean())
            mip_curve.append(mip_mse[mask].mean())
            phase_centers.append((phase_edges[i] + phase_edges[i+1]) / 2)
    ax.plot(phase_centers, dp_curve, 'o-', label="DP", color="steelblue", markersize=4)
    ax.plot(phase_centers, mip_curve, 'o-', label="MIP", color="coral", markersize=4)
    ax.set_xlabel("Episode Phase"); ax.set_ylabel("MSE")
    ax.set_title("MSE vs Episode Phase"); ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(0.5, color="gray", ls="--", alpha=0.5, label="insertion start")

    # 6. Example: first action prediction for a specific joint at insertion
    ax = axes[1, 2]
    j_show = 0  # Show joint 0
    gt_j = insert_gt[:200, 0, j_show]
    dp_j = insert_dp[:200, 0, j_show]
    mip_j = insert_mip[:200, 0, j_show]
    sort_idx = np.argsort(gt_j)
    ax.plot(gt_j[sort_idx], 'k-', label="GT", alpha=0.7, lw=1)
    ax.plot(dp_j[sort_idx], 'b.', label="DP", alpha=0.3, markersize=2)
    ax.plot(mip_j[sort_idx], 'r.', label="MIP", alpha=0.3, markersize=2)
    ax.set_xlabel("Sample (sorted by GT)"); ax.set_ylabel(f"Action j{j_show}")
    ax.set_title(f"Prediction vs GT (j{j_show}, insertion)"); ax.legend(fontsize=8)

    plt.suptitle("DP vs MIP: Val Set Prediction Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(args.output, "val_prediction_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot: {plot_path}")

    # Save data
    np.savez(os.path.join(args.output, "val_analysis_data.npz"),
             dp_preds=dp_preds, mip_preds=mip_preds, gt_actions=gt_actions,
             phase=phase, dp_mse=dp_mse, mip_mse=mip_mse,
             traj_idx=val_data["traj_idx"], step_idx=val_data["step_idx"])
    print(f"Data: {os.path.join(args.output, 'val_analysis_data.npz')}")


if __name__ == "__main__":
    main()
