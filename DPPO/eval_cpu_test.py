"""
Quick CPU eval test for DPPO pretrained model.
Compares CPU (physx_cpu) vs GPU (physx_cuda) eval to check backend mismatch.
"""

import torch
import numpy as np
import os
import gymnasium as gym
import mani_skill.envs  # noqa

from DPPO.model.mlp import DiffusionMLP
from DPPO.model.diffusion import DiffusionModel


def eval_cpu(model, device, obs_mean, obs_std, action_min, action_max,
             cond_steps=1, act_steps=4, n_episodes=100,
             env_id="PickCube-v1", control_mode="pd_joint_delta_pos",
             max_episode_steps=100):
    """Evaluate on CPU env (physx_cpu), single env sequentially."""
    model.eval()

    a_lo = action_min.to(device)
    a_hi = action_max.to(device)

    success_once_list = []
    success_at_end_list = []

    env = gym.make(
        env_id,
        obs_mode="state",
        render_mode="rgb_array",
        control_mode=control_mode,
        max_episode_steps=max_episode_steps,
    )

    for ep in range(n_episodes):
        obs, _ = env.reset()
        if isinstance(obs, dict):
            obs = np.concatenate([v.flatten() for v in obs.values()])
        if isinstance(obs, torch.Tensor):
            obs = obs.float().to(device)
        else:
            obs = torch.from_numpy(obs).float().to(device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # (1, obs_dim)
        elif obs.dim() == 2 and obs.shape[0] == 1:
            pass  # already (1, obs_dim)
        else:
            obs = obs.view(1, -1)
        obs_history = obs.unsqueeze(1).repeat(1, cond_steps, 1)  # (1, To, Do)

        ever_success = False
        for step in range(max_episode_steps):
            obs_norm = (obs_history - obs_mean) / obs_std
            cond = {"state": obs_norm}

            with torch.no_grad():
                samples = model(cond, deterministic=True)
            action_chunk = samples.trajectories  # (1, H, A)
            action_chunk = (action_chunk + 1.0) / 2.0 * (a_hi - a_lo) + a_lo

            for a_idx in range(min(act_steps, action_chunk.shape[1])):
                action = action_chunk[0, a_idx].cpu().numpy()
                obs_new, reward, terminated, truncated, info = env.step(action)

                if isinstance(obs_new, dict):
                    obs_new = np.concatenate([v.flatten() for v in obs_new.values()])
                if isinstance(obs_new, torch.Tensor):
                    obs_new_t = obs_new.float().to(device)
                else:
                    obs_new_t = torch.from_numpy(obs_new).float().to(device)
                if obs_new_t.dim() == 1:
                    obs_new_t = obs_new_t.unsqueeze(0)
                obs_history = torch.cat([obs_history[:, 1:], obs_new_t.unsqueeze(1)], dim=1)

                if info.get("success", False):
                    ever_success = True

                if terminated or truncated:
                    break
            if terminated or truncated:
                break

        success_at_end_list.append(float(info.get("success", False)))
        success_once_list.append(float(ever_success))

    env.close()

    return {
        "success_at_end": np.mean(success_at_end_list),
        "success_once": np.mean(success_once_list),
        "n_episodes": n_episodes,
    }


def eval_gpu(model, device, obs_mean, obs_std, action_min, action_max,
             cond_steps=1, act_steps=4, n_episodes=100,
             env_id="PickCube-v1", control_mode="pd_joint_delta_pos",
             max_episode_steps=100):
    """Evaluate on GPU env (physx_cuda)."""
    from DPPO.evaluate import evaluate_gpu
    return evaluate_gpu(
        n_episodes=n_episodes,
        model=model,
        device=device,
        act_steps=act_steps,
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_min=action_min,
        action_max=action_max,
        cond_steps=cond_steps,
        env_id=env_id,
        num_envs=min(n_episodes, 100),
        control_mode=control_mode,
        max_episode_steps=max_episode_steps,
    )


def main():
    ckpt_path = "runs/dppo_pretrain/dppo_pretrain_PickCube-v1_T20_H4/best.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]

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
    model.load_state_dict(ckpt["ema"], strict=False)
    model.eval()

    obs_mean = ckpt["obs_mean"].to(device)
    obs_std = ckpt["obs_std"].to(device)
    action_min = ckpt["action_min"].to(device)
    action_max = ckpt["action_max"].to(device)

    print("=== CPU eval (physx_cpu) ===")
    cpu_metrics = eval_cpu(
        model, device, obs_mean, obs_std, action_min, action_max,
        cond_steps=args.get("cond_steps", 1),
        act_steps=args["act_steps"],
    )
    print(f"  success_at_end={cpu_metrics['success_at_end']:.3f}, "
          f"success_once={cpu_metrics['success_once']:.3f} "
          f"(n={cpu_metrics['n_episodes']})")

    print("\n=== GPU eval (physx_cuda) ===")
    gpu_metrics = eval_gpu(
        model, device, obs_mean, obs_std, action_min, action_max,
        cond_steps=args.get("cond_steps", 1),
        act_steps=args["act_steps"],
    )
    print(f"  success_at_end={gpu_metrics['success_at_end']:.3f}, "
          f"success_once={gpu_metrics['success_once']:.3f} "
          f"(n={gpu_metrics['n_episodes']})")


if __name__ == "__main__":
    main()
