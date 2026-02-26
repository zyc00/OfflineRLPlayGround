"""
Diagnostic: check if DPPO pretrained model can reconstruct demo actions.

Tests:
1. Feed demo obs → model → compare generated actions to demo actions
2. Check normalization round-trip consistency
3. Visualize generated vs demo action distributions
"""

import torch
import numpy as np
import os
from DPPO.dataset import DPPODataset
from DPPO.model.mlp import DiffusionMLP
from DPPO.model.diffusion import DiffusionModel


def main():
    ckpt_path = "runs/dppo_pretrain/dppo_pretrain_PickCube-v1_T20_H4/best.pt"
    demo_path = os.path.expanduser(
        "~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]
    print(f"Checkpoint args: denoising_steps={args['denoising_steps']}, "
          f"horizon_steps={args['horizon_steps']}, act_steps={args['act_steps']}")

    # Load dataset
    dataset = DPPODataset(
        data_path=demo_path,
        horizon_steps=args["horizon_steps"],
        cond_steps=args.get("cond_steps", 1),
    )

    # Build model
    obs_dim = ckpt["obs_dim"]
    action_dim = ckpt["action_dim"]
    cond_dim = obs_dim * args.get("cond_steps", 1)

    network = DiffusionMLP(
        action_dim=action_dim,
        horizon_steps=args["horizon_steps"],
        cond_dim=cond_dim,
        time_dim=args.get("time_dim", 16),
        mlp_dims=args.get("mlp_dims", [512, 512, 512]),
        activation_type=args.get("activation_type", "Mish"),
        residual_style=args.get("residual_style", True),
    )
    model = DiffusionModel(
        network=network,
        horizon_steps=args["horizon_steps"],
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        denoising_steps=args["denoising_steps"],
        denoised_clip_value=1.0,
        randn_clip_value=10,
        final_action_clip_value=1.0,
        predict_epsilon=True,
    )

    # Load EMA weights
    model.load_state_dict(ckpt["ema"])
    model.eval()
    print(f"Loaded EMA model from {ckpt_path}")

    obs_mean = ckpt["obs_mean"].to(device)
    obs_std = ckpt["obs_std"].to(device)
    action_min = ckpt["action_min"].to(device)
    action_max = ckpt["action_max"].to(device)

    # --- Test 1: Normalization round-trip ---
    print("\n=== Test 1: Normalization round-trip ===")
    sample_batch = dataset[0]
    raw_act = dataset.act_list[0][0]  # First action of first trajectory
    norm_act = dataset.normalize_action(raw_act)
    unnorm_act = dataset.unnormalize_action(norm_act)
    print(f"  Raw action:        {raw_act.numpy().round(4)}")
    print(f"  Normalized [-1,1]: {norm_act.numpy().round(4)}")
    print(f"  Unnormalized:      {unnorm_act.numpy().round(4)}")
    print(f"  Round-trip error:  {(raw_act - unnorm_act).abs().max().item():.2e}")

    # Check normalized action range in training data
    all_norm_acts = []
    for i in range(min(100, len(dataset))):
        batch = dataset[i]
        all_norm_acts.append(batch["actions"])
    all_norm_acts = torch.stack(all_norm_acts)
    print(f"\n  Normalized action stats (training data):")
    print(f"    min: {all_norm_acts.min(dim=0).values.min(dim=0).values.numpy().round(3)}")
    print(f"    max: {all_norm_acts.max(dim=0).values.max(dim=0).values.numpy().round(3)}")
    print(f"    mean: {all_norm_acts.mean(dim=(0,1)).numpy().round(3)}")
    print(f"    std: {all_norm_acts.std(dim=(0,1)).numpy().round(3)}")

    # --- Test 2: Model reconstruction from demo obs ---
    print("\n=== Test 2: Model reconstruction from demo obs ===")
    # Take 50 random samples from dataset, generate actions, compare
    torch.manual_seed(42)
    n_test = 200
    indices = torch.randperm(len(dataset))[:n_test]

    demo_actions = []
    gen_actions = []
    for idx in indices:
        batch = dataset[idx.item()]
        act_gt = batch["actions"].to(device)  # (H, A) normalized
        obs_cond = batch["cond"]["state"].unsqueeze(0).to(device)  # (1, To, Do)
        cond = {"state": obs_cond}

        # Generate action via full denoising
        with torch.no_grad():
            sample = model(cond, deterministic=True)
        act_gen = sample.trajectories[0]  # (H, A) normalized

        demo_actions.append(act_gt.cpu())
        gen_actions.append(act_gen.cpu())

    demo_actions = torch.stack(demo_actions)  # (N, H, A)
    gen_actions = torch.stack(gen_actions)  # (N, H, A)

    # Per-dim stats (first action step only)
    demo_a0 = demo_actions[:, 0, :]  # (N, A)
    gen_a0 = gen_actions[:, 0, :]

    print(f"\n  First action step (normalized [-1,1]):")
    print(f"  {'Dim':>4s}  {'Demo Mean':>10s}  {'Gen Mean':>10s}  {'Demo Std':>10s}  {'Gen Std':>10s}  {'MSE':>10s}  {'Corr':>10s}")
    for d in range(action_dim):
        dm = demo_a0[:, d].mean().item()
        gm = gen_a0[:, d].mean().item()
        ds = demo_a0[:, d].std().item()
        gs = gen_a0[:, d].std().item()
        mse = ((demo_a0[:, d] - gen_a0[:, d]) ** 2).mean().item()
        # Correlation
        if ds > 1e-6 and gs > 1e-6:
            corr = torch.corrcoef(torch.stack([demo_a0[:, d], gen_a0[:, d]]))[0, 1].item()
        else:
            corr = 0.0
        print(f"  {d:>4d}  {dm:>10.4f}  {gm:>10.4f}  {ds:>10.4f}  {gs:>10.4f}  {mse:>10.4f}  {corr:>10.4f}")

    total_mse = ((demo_actions - gen_actions) ** 2).mean().item()
    print(f"\n  Total MSE (normalized): {total_mse:.6f}")

    # Denormalized comparison
    demo_denorm = dataset.unnormalize_action(demo_actions)
    gen_denorm = dataset.unnormalize_action(gen_actions)
    total_mse_raw = ((demo_denorm - gen_denorm) ** 2).mean().item()
    print(f"  Total MSE (raw): {total_mse_raw:.6f}")

    # --- Test 3: Check denoising chain quality ---
    print("\n=== Test 3: Denoising chain quality ===")
    # Take one sample, run denoising, check intermediate quality
    batch = dataset[0]
    obs_cond = batch["cond"]["state"].unsqueeze(0).to(device)
    act_gt = batch["actions"].unsqueeze(0).to(device)  # (1, H, A)
    cond = {"state": obs_cond}

    # Forward diffusion: q(x_t | x_0) at various t
    print(f"  Forward diffusion q(x_t|x_0) and reconstruction quality:")
    for t_val in [0, 1, 5, 10, 15, 19]:
        t = torch.tensor([t_val], device=device)
        noise = torch.randn_like(act_gt)
        x_t = model.q_sample(act_gt, t, noise)
        # Predict noise
        noise_pred = model.network(x_t, t, cond=cond)
        noise_mse = ((noise_pred - noise) ** 2).mean().item()
        # Reconstruct x_0
        x_recon = (
            model.sqrt_recip_alphas_cumprod[t_val] * x_t
            - model.sqrt_recipm1_alphas_cumprod[t_val] * noise_pred
        )
        recon_mse = ((x_recon - act_gt) ** 2).mean().item()
        snr = (model.alphas_cumprod[t_val] / (1 - model.alphas_cumprod[t_val])).item()
        print(f"    t={t_val:>2d}: SNR={snr:>8.3f}, noise_MSE={noise_mse:.6f}, recon_MSE={recon_mse:.6f}")

    # --- Test 4: Multiple sample comparison ---
    print("\n=== Test 4: Sample variance (multiple generations from same obs) ===")
    batch = dataset[100]
    obs_cond = batch["cond"]["state"].unsqueeze(0).to(device)
    cond = {"state": obs_cond}
    act_gt = batch["actions"]

    samples = []
    for _ in range(20):
        with torch.no_grad():
            s = model(cond, deterministic=True)
        samples.append(s.trajectories[0, 0].cpu())  # first action step
    samples = torch.stack(samples)  # (20, A)
    print(f"  Demo action [0]: {act_gt[0].numpy().round(4)}")
    print(f"  Gen mean:        {samples.mean(0).numpy().round(4)}")
    print(f"  Gen std:         {samples.std(0).numpy().round(4)}")
    print(f"  MSE(mean, demo): {((samples.mean(0) - act_gt[0]) ** 2).mean().item():.6f}")

    # --- Test 5: Compare noise schedule ---
    print("\n=== Test 5: Noise schedule comparison ===")
    print(f"  Denoising steps: {model.denoising_steps}")
    print(f"  alphas_cumprod: {model.alphas_cumprod.cpu().numpy().round(4)}")
    print(f"  betas: {model.betas.cpu().numpy().round(4)}")
    print(f"  sqrt_alphas_cumprod[0]: {model.sqrt_alphas_cumprod[0].item():.4f}")
    print(f"  sqrt_alphas_cumprod[-1]: {model.sqrt_alphas_cumprod[-1].item():.4f}")
    print(f"  sqrt_one_minus_alphas_cumprod[0]: {model.sqrt_one_minus_alphas_cumprod[0].item():.4f}")
    print(f"  sqrt_one_minus_alphas_cumprod[-1]: {model.sqrt_one_minus_alphas_cumprod[-1].item():.4f}")


if __name__ == "__main__":
    main()
