"""
Per-dimension Lipschitz analysis for different policy types (DP, MIP, Gaussian).

For each obs dimension d, estimates:
    L_d = E_x[ ||pi(x + delta * e_d) - pi(x)|| / delta ]

This measures how sensitive the policy output is to perturbation along each obs dimension.
"""

import argparse
import torch
import numpy as np
import h5py
import os


def load_dp_model(ckpt_path, device):
    """Load DP (diffusion policy) model from checkpoint."""
    from DPPO.model.unet_wrapper import DiffusionUNet
    from DPPO.model.mlp import DiffusionMLP
    from DPPO.model.diffusion import DiffusionModel

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    obs_dim = ckpt["obs_dim"]
    action_dim = ckpt["action_dim"]
    cond_steps = args.get("cond_steps", 2)
    horizon_steps = args.get("horizon_steps", 16)
    denoising_steps = args.get("denoising_steps", 100)

    if args.get("network_type") == "unet":
        network = DiffusionUNet(
            action_dim=action_dim, horizon_steps=horizon_steps,
            cond_dim=obs_dim * cond_steps,
            diffusion_step_embed_dim=args.get("diffusion_step_embed_dim", 64),
            down_dims=args.get("unet_dims", [64, 128, 256]),
            n_groups=args.get("n_groups", 8),
        )
    else:
        network = DiffusionMLP(
            action_dim=action_dim, horizon_steps=horizon_steps,
            cond_dim=obs_dim * cond_steps,
        )

    model = DiffusionModel(
        network=network, horizon_steps=horizon_steps,
        obs_dim=obs_dim, action_dim=action_dim, device=device,
        denoising_steps=denoising_steps, denoised_clip_value=1.0,
        randn_clip_value=10, final_action_clip_value=1.0, predict_epsilon=True,
    )
    model.load_state_dict(ckpt["ema"], strict=False)
    # Fix NaN eta
    if hasattr(model, 'eta') and torch.isnan(model.eta.eta_logit.data).any():
        model.eta.eta_logit.data = torch.atanh(torch.tensor([0.999], device=device))
    model.eval()

    norm_info = {
        "obs_min": ckpt.get("obs_min"), "obs_max": ckpt.get("obs_max"),
        "action_min": ckpt.get("action_min"), "action_max": ckpt.get("action_max"),
        "no_obs_norm": ckpt.get("no_obs_norm", False),
        "no_action_norm": ckpt.get("no_action_norm", False),
    }

    # Pre-generate fixed noise for deterministic Lipschitz comparison
    _fixed_noise = None

    def predict_fn(obs_batch):
        """obs_batch: (B, obs_dim) raw obs. Returns (B, action_dim) raw actions."""
        nonlocal _fixed_noise
        B = obs_batch.shape[0]
        # Normalize obs
        if not norm_info["no_obs_norm"]:
            o_lo = norm_info["obs_min"].to(device)
            o_hi = norm_info["obs_max"].to(device)
            obs_n = (obs_batch - o_lo) / (o_hi - o_lo + 1e-8) * 2.0 - 1.0
        else:
            obs_n = obs_batch
        # Build cond
        cond = {"state": obs_n.unsqueeze(1).expand(-1, cond_steps, -1)}
        # Use fixed seed so same noise is used for base and perturbed obs
        torch.manual_seed(42)
        samples = model(cond, deterministic=True, ddim_steps=10)
        act = samples.trajectories[:, 0]  # first action step
        # Denormalize
        if not norm_info["no_action_norm"]:
            a_lo = norm_info["action_min"].to(device)
            a_hi = norm_info["action_max"].to(device)
            act = (act + 1.0) / 2.0 * (a_hi - a_lo) + a_lo
        return act

    return predict_fn, ckpt


def load_mip_model(ckpt_path, device):
    """Load MIP model from checkpoint."""
    from MultiGaussian.models.multi_gaussian import MIPPolicy

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]

    cond_steps = args.get("cond_steps", 1)
    horizon_steps = args.get("horizon_steps", 1)

    model = MIPPolicy(
        input_dim=ckpt["obs_dim"], action_dim=ckpt["action_dim"],
        cond_steps=cond_steps,
        horizon_steps=horizon_steps,
        t_star=args.get("t_star", 0.9),
        dropout=args.get("dropout", 0.1),
        emb_dim=args.get("emb_dim", 512),
        n_layers=args.get("n_layers", 6),
        predict_epsilon=args.get("predict_epsilon", False),
    ).to(device)
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()

    norm_info = {
        "obs_min": ckpt.get("obs_min"), "obs_max": ckpt.get("obs_max"),
        "action_min": ckpt.get("action_min"), "action_max": ckpt.get("action_max"),
        "no_obs_norm": ckpt.get("no_obs_norm", False),
        "no_action_norm": ckpt.get("no_action_norm", False),
    }

    def predict_fn(obs_batch):
        B = obs_batch.shape[0]
        if not norm_info["no_obs_norm"]:
            o_lo = norm_info["obs_min"].to(device)
            o_hi = norm_info["obs_max"].to(device)
            obs_n = (obs_batch - o_lo) / (o_hi - o_lo + 1e-8) * 2.0 - 1.0
        else:
            obs_n = obs_batch
        # Handle cond_steps > 1: stack same obs
        if cond_steps > 1:
            obs_n = obs_n.unsqueeze(1).expand(-1, cond_steps, -1)
        act = model.predict(obs_n)
        # Handle horizon_steps > 1: take first action
        if horizon_steps > 1:
            act = act[:, 0]
        if not norm_info["no_action_norm"]:
            a_lo = norm_info["action_min"].to(device)
            a_hi = norm_info["action_max"].to(device)
            act = (act + 1.0) / 2.0 * (a_hi - a_lo) + a_lo
        return act

    return predict_fn, ckpt


def load_gaussian_model(ckpt_path, device):
    """Load Gaussian (MLP BC) model from checkpoint."""
    from MultiGaussian.models.gaussian import GaussianPolicy

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]

    cond_steps = args.get("cond_steps", 1)
    horizon_steps = args.get("horizon_steps", 1)

    model = GaussianPolicy(
        input_dim=ckpt["obs_dim"], action_dim=ckpt["action_dim"],
        hidden_dims=args.get("hidden_dims", [256, 256]),
        activation=args.get("activation", "ReLU"),
        sigma_init=args.get("sigma_init", -1.5),
        cond_steps=cond_steps,
        horizon_steps=horizon_steps,
    ).to(device)
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()

    norm_info = {
        "obs_min": ckpt.get("obs_min"), "obs_max": ckpt.get("obs_max"),
        "action_min": ckpt.get("action_min"), "action_max": ckpt.get("action_max"),
        "no_obs_norm": ckpt.get("no_obs_norm", False),
        "no_action_norm": ckpt.get("no_action_norm", False),
    }

    def predict_fn(obs_batch):
        B = obs_batch.shape[0]
        if not norm_info["no_obs_norm"]:
            o_lo = norm_info["obs_min"].to(device)
            o_hi = norm_info["obs_max"].to(device)
            obs_n = (obs_batch - o_lo) / (o_hi - o_lo + 1e-8) * 2.0 - 1.0
        else:
            obs_n = obs_batch
        if cond_steps > 1:
            obs_n = obs_n.unsqueeze(1).expand(-1, cond_steps, -1)
        mean, _ = model(obs_n)  # deterministic: use mean
        if horizon_steps > 1:
            act = mean[:, 0]
        else:
            act = mean
        if not norm_info["no_action_norm"]:
            a_lo = norm_info["action_min"].to(device)
            a_hi = norm_info["action_max"].to(device)
            act = (act + 1.0) / 2.0 * (a_hi - a_lo) + a_lo
        return act

    return predict_fn, ckpt


def load_lknn_model(ckpt_path, demo_path, zero_qvel, k_eval, device):
    """Load learned KNN model from checkpoint."""
    from scripts.learned_knn_policy import (ObsEncoder, SoftKNNPolicy,
                                            SoftKNNPolicyWithResidual, load_data)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    lknn_args = ckpt["args"]

    # Reload training data for the memory bank
    train_obs, train_act, norm_info = load_data(
        demo_path, horizon_steps=lknn_args["horizon_steps"],
        cond_steps=lknn_args.get("cond_steps", 1),
        zero_qvel=zero_qvel,
        num_demos=lknn_args.get("num_demos"),
        device=device,
    )

    encoder = ObsEncoder(ckpt["obs_flat_dim"], feat_dim=lknn_args["feat_dim"],
                         hidden_dim=lknn_args["hidden_dim"],
                         n_layers=lknn_args["n_layers"])
    encoder.load_state_dict(ckpt["encoder"])

    if lknn_args.get("residual", False) and "residual_head" in ckpt:
        policy = SoftKNNPolicyWithResidual(
            encoder, train_obs, train_act,
            temperature=lknn_args["temperature"],
            residual_hidden_dim=lknn_args.get("residual_hidden_dim", 128),
            max_residual_norm=lknn_args.get("max_residual_norm", 0.1),
        ).to(device)
        policy.residual_head.load_state_dict(ckpt["residual_head"])
    else:
        policy = SoftKNNPolicy(encoder, train_obs, train_act,
                               temperature=lknn_args["temperature"]).to(device)
    policy.eval()
    policy.encode_train()

    o_lo = norm_info["obs_min"].to(device)
    o_hi = norm_info["obs_max"].to(device)
    o_range = (o_hi - o_lo).clamp(min=1e-8)
    a_lo = norm_info["action_min"].to(device)
    a_hi = norm_info["action_max"].to(device)
    a_range = (a_hi - a_lo).clamp(min=1e-8)

    horizon_steps = lknn_args["horizon_steps"]
    cond_steps = lknn_args.get("cond_steps", 1)

    def predict_fn(obs_batch):
        B = obs_batch.shape[0]
        # Normalize obs
        obs_n = (obs_batch - o_lo) / o_range * 2.0 - 1.0
        if cond_steps > 1:
            obs_n = obs_n.unsqueeze(1).expand(-1, cond_steps, -1).reshape(B, -1)
        act = policy.predict(obs_n, k=k_eval)
        if horizon_steps > 1:
            act = act[:, 0]
        # Denormalize
        act = (act + 1.0) / 2.0 * a_range + a_lo
        return act

    return predict_fn


def load_obs_from_demos(demo_path, num_obs=500, zero_qvel=False):
    """Load a batch of observations from demo H5 file."""
    with h5py.File(demo_path, "r") as f:
        traj_keys = sorted([k for k in f.keys() if k.startswith("traj_")],
                           key=lambda k: int(k.split("_")[1]))
        all_obs = []
        for tk in traj_keys:
            obs = f[tk]["obs"][:]
            # Sample from middle of trajectory (avoid edge states)
            L = len(obs)
            mid = obs[L // 4: 3 * L // 4]
            all_obs.append(mid)
            if sum(len(o) for o in all_obs) >= num_obs:
                break
        all_obs = np.concatenate(all_obs, axis=0)[:num_obs]

    obs_t = torch.from_numpy(all_obs).float()
    if zero_qvel:
        obs_t[:, 9:18] = 0.0
    return obs_t


@torch.no_grad()
def compute_lipschitz(predict_fn, obs_batch, delta=1e-3, device="cuda"):
    """Compute per-dimension Lipschitz constant.

    Returns:
        lip_mean: (obs_dim,) mean sensitivity per dimension
        lip_std: (obs_dim,) std per dimension
        lip_max: (obs_dim,) max (Lipschitz lower bound) per dimension
    """
    obs_batch = obs_batch.to(device)
    B, D = obs_batch.shape

    # Base action
    act_base = predict_fn(obs_batch)  # (B, act_dim)

    lip_all = []
    for d in range(D):
        obs_pert = obs_batch.clone()
        obs_pert[:, d] += delta

        act_pert = predict_fn(obs_pert)
        # L2 norm of action change / delta
        act_diff = (act_pert - act_base).norm(dim=-1) / delta  # (B,)
        lip_all.append(act_diff.cpu())

    lip_all = torch.stack(lip_all, dim=1)  # (B, D)
    return (lip_all.mean(dim=0).numpy(),
            lip_all.std(dim=0).numpy(),
            lip_all.max(dim=0).values.numpy())


def get_obs_dim_names(obs_dim):
    """Best-effort obs dimension names for PegInsertionSide (43D state)."""
    if obs_dim == 43:
        names = (
            [f"qpos_{i}" for i in range(9)] +
            [f"qvel_{i}" for i in range(9)] +
            [f"peg_pose_{i}" for i in range(7)] +
            [f"box_pose_{i}" for i in range(7)] +
            [f"peg_half_{i}" for i in range(3)] +
            [f"box_half_{i}" for i in range(3)] +
            [f"offset_{i}" for i in range(5)]  # remaining dims
        )
        return names[:obs_dim]
    elif obs_dim == 42:
        names = (
            [f"qpos_{i}" for i in range(9)] +
            [f"qvel_{i}" for i in range(9)] +
            [f"tcp_pose_{i}" for i in range(7)] +
            [f"cube_pose_{i}" for i in range(7)] +
            [f"goal_pos_{i}" for i in range(3)] +
            [f"extra_{i}" for i in range(7)]
        )
        return names[:obs_dim]
    return [f"dim_{i}" for i in range(obs_dim)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp_ckpt", type=str, default=None, help="DP checkpoint path")
    parser.add_argument("--mip_ckpt", type=str, default=None, help="MIP checkpoint path")
    parser.add_argument("--gaussian_ckpt", type=str, default=None, help="Gaussian MLP checkpoint path")
    parser.add_argument("--lknn_ckpt", type=str, default=None, help="Learned KNN checkpoint path")
    parser.add_argument("--lknn_k", type=int, default=10, help="K for learned KNN eval")
    parser.add_argument("--demo_path", type=str, required=True, help="Demo H5 for obs sampling")
    parser.add_argument("--num_obs", type=int, default=500)
    parser.add_argument("--delta", type=float, default=1e-3)
    parser.add_argument("--zero_qvel", action="store_true")
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load obs
    obs_batch = load_obs_from_demos(args.demo_path, args.num_obs, args.zero_qvel)
    obs_dim = obs_batch.shape[1]
    dim_names = get_obs_dim_names(obs_dim)
    print(f"Loaded {len(obs_batch)} obs (dim={obs_dim}), delta={args.delta}")

    results = {}

    if args.dp_ckpt:
        print(f"\n=== DP: {args.dp_ckpt} ===")
        predict_fn, _ = load_dp_model(args.dp_ckpt, device)
        lip_mean, lip_std, lip_max = compute_lipschitz(predict_fn, obs_batch, args.delta, device)
        results["dp"] = (lip_mean, lip_std, lip_max)

    if args.mip_ckpt:
        print(f"\n=== MIP: {args.mip_ckpt} ===")
        predict_fn, _ = load_mip_model(args.mip_ckpt, device)
        lip_mean, lip_std, lip_max = compute_lipschitz(predict_fn, obs_batch, args.delta, device)
        results["mip"] = (lip_mean, lip_std, lip_max)

    if args.gaussian_ckpt:
        print(f"\n=== Gaussian: {args.gaussian_ckpt} ===")
        predict_fn, _ = load_gaussian_model(args.gaussian_ckpt, device)
        lip_mean, lip_std, lip_max = compute_lipschitz(predict_fn, obs_batch, args.delta, device)
        results["gaussian"] = (lip_mean, lip_std, lip_max)

    if args.lknn_ckpt:
        print(f"\n=== Learned KNN: {args.lknn_ckpt} ===")
        predict_fn = load_lknn_model(args.lknn_ckpt, args.demo_path,
                                     args.zero_qvel, args.lknn_k, device)
        lip_mean, lip_std, lip_max = compute_lipschitz(predict_fn, obs_batch, args.delta, device)
        results["lknn"] = (lip_mean, lip_std, lip_max)

    # Print comparison table (mean and max)
    print(f"\n{'dim':<16}", end="")
    for name in results:
        print(f"  {name:>8} mean  {name:>8} max", end="")
    print()
    print("-" * (16 + 22 * len(results)))

    for d in range(obs_dim):
        print(f"{dim_names[d]:<16}", end="")
        for name in results:
            m, mx = results[name][0][d], results[name][2][d]
            print(f"  {m:10.2f}  {mx:10.2f}", end="")
        print()

    # Summary
    print(f"\n{'--- Summary ---':<16}", end="")
    for name in results:
        print(f"  {name:>8} mean  {name:>8} max", end="")
    print()
    print(f"{'mean over dims':<16}", end="")
    for name in results:
        print(f"  {results[name][0].mean():10.2f}  {results[name][2].mean():10.2f}", end="")
    print()
    print(f"{'worst dim':<16}", end="")
    for name in results:
        idx_m = results[name][0].argmax()
        idx_x = results[name][2].argmax()
        print(f"  {results[name][0][idx_m]:8.1f}({dim_names[idx_m][:6]:>6})  {results[name][2][idx_x]:8.1f}({dim_names[idx_x][:6]:>6})", end="")
    print()

    if args.save_path:
        np.savez(args.save_path, dim_names=dim_names,
                 **{f"{k}_mean": v[0] for k, v in results.items()},
                 **{f"{k}_std": v[1] for k, v in results.items()},
                 **{f"{k}_max": v[2] for k, v in results.items()})
        print(f"\nSaved to {args.save_path}")


if __name__ == "__main__":
    main()
