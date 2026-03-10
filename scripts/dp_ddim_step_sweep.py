"""Level 1 Bridge Ablation: DP DDIM step sweep.

Evaluates an existing DP checkpoint with different DDIM step counts
{1, 2, 3, 5, 10, 20} to see how inference step count affects performance.
No retraining needed — same checkpoint, varying inference steps.
"""
import sys
import time
import torch
import gymnasium as gym
import mani_skill.envs

from DPPO.model.unet_wrapper import DiffusionUNet
from DPPO.model.diffusion import DiffusionModel
from DPPO.eval_cpu import evaluate_cpu_model


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("ckpt_path")
    p.add_argument("--ddim_steps", type=int, nargs="+", default=[1, 2, 3, 5, 10, 20])
    p.add_argument("--n_episodes", type=int, default=100)
    p.add_argument("--env_id", default="PegInsertionSide-v1")
    p.add_argument("--control_mode", default="pd_joint_delta_pos")
    p.add_argument("--max_episode_steps", type=int, default=200)
    p.add_argument("--zero_qvel", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda")
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    is_finetune = "model" in ckpt and "ema" not in ckpt
    ckpt_args = ckpt.get("pretrain_args", ckpt["args"]) if is_finetune else ckpt["args"]

    obs_dim = ckpt["obs_dim"]
    action_dim = ckpt["action_dim"]
    cond_steps = ckpt_args.get("cond_steps", 2)
    horizon_steps = ckpt_args.get("horizon_steps", 16)
    act_steps = ckpt_args.get("act_steps", 8)
    denoising_steps = ckpt_args.get("denoising_steps", 100)

    network = DiffusionUNet(
        action_dim=action_dim,
        horizon_steps=horizon_steps,
        cond_dim=obs_dim * cond_steps,
        diffusion_step_embed_dim=ckpt_args.get("diffusion_step_embed_dim", 64),
        down_dims=ckpt_args.get("unet_dims", [64, 128, 256]),
        n_groups=ckpt_args.get("n_groups", 8),
    )

    model = DiffusionModel(
        network=network,
        horizon_steps=horizon_steps,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        denoising_steps=denoising_steps,
        denoised_clip_value=1.0,
        randn_clip_value=10,
        final_action_clip_value=1.0,
        predict_epsilon=True,
    )

    if is_finetune:
        model_sd = ckpt["model"]
        ft_sd = {}
        for k, v in model_sd.items():
            if k.startswith("actor_ft."):
                ft_sd["network." + k[len("actor_ft."):]] = v
        model.load_state_dict(ft_sd, strict=False)
    else:
        model.load_state_dict(ckpt["ema"], strict=False)

    if hasattr(model, 'eta') and torch.isnan(model.eta.eta_logit.data).any():
        model.eta.eta_logit.data = torch.atanh(torch.tensor([0.999], device=device))
    model.eval()

    no_obs_norm = ckpt.get("no_obs_norm", False)
    no_action_norm = ckpt.get("no_action_norm", False)
    act_offset = cond_steps - 1 if ckpt_args.get("network_type") == "unet" else 0

    zero_qvel = args.zero_qvel or ckpt_args.get("zero_qvel", False)

    print(f"Checkpoint: {args.ckpt_path}")
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}, denoising_steps={denoising_steps}")
    print(f"  cond_steps={cond_steps}, horizon_steps={horizon_steps}, act_steps={act_steps}")
    print(f"  zero_qvel={zero_qvel}, no_obs_norm={no_obs_norm}, no_action_norm={no_action_norm}")
    print(f"  act_offset={act_offset}")
    print(f"\nSweeping DDIM steps: {args.ddim_steps}")
    print(f"Env: {args.env_id}, control={args.control_mode}, max_steps={args.max_episode_steps}")
    print(f"Episodes per config: {args.n_episodes}\n")

    results = []
    for steps in args.ddim_steps:
        print(f"{'='*60}")
        print(f"DDIM steps = {steps}")
        print(f"{'='*60}")
        t0 = time.time()
        metrics = evaluate_cpu_model(
            n_episodes=args.n_episodes,
            model=model,
            device=device,
            act_steps=act_steps,
            cond_steps=cond_steps,
            env_id=args.env_id,
            control_mode=args.control_mode,
            max_episode_steps=args.max_episode_steps,
            obs_min=ckpt.get("obs_min"),
            obs_max=ckpt.get("obs_max"),
            action_min=ckpt.get("action_min"),
            action_max=ckpt.get("action_max"),
            no_obs_norm=no_obs_norm,
            no_action_norm=no_action_norm,
            act_offset=act_offset,
            ddim_steps=steps,
            zero_qvel=zero_qvel,
        )
        elapsed = time.time() - t0
        sr = metrics["success_once"]
        results.append((steps, sr, elapsed))
        print(f"  DDIM-{steps}: SR={sr:.1%}, time={elapsed:.0f}s\n")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY: DP DDIM Step Sweep")
    print("=" * 60)
    print(f"{'DDIM Steps':>12} {'SR':>8} {'Time':>8}")
    print("-" * 32)
    for steps, sr, elapsed in results:
        print(f"{steps:>12} {sr:>7.1%} {elapsed:>7.0f}s")
    print("-" * 32)


if __name__ == "__main__":
    main()
