"""Save evaluation videos for Diffusion Policy (success & failure cases).

Usage:
  # Deterministic eval, save 10 episodes
  python dp_eval_video.py --ckpt runs/dppo_pretrain/dppo_25traj_aligned_v2/best.pt \
    --num-episodes 10 --output-dir runs/videos/dppo_25traj

  # With exploration noise
  python dp_eval_video.py --ckpt runs/dppo_pretrain/dppo_25traj_aligned_v2/best.pt \
    --num-episodes 10 --min-sampling-denoising-std 0.01 --ddim-steps 10
"""
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import tyro
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack

sys.path.insert(0, os.path.dirname(__file__))
from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


@dataclass
class Args:
    ckpt: str = "runs/dppo_pretrain/dppo_25traj_aligned_v2/best.pt"
    env_id: str = "PickCube-v1"
    control_mode: str = "pd_ee_delta_pos"
    num_episodes: int = 10
    max_episode_steps: int = 100
    seed: int = 0
    min_sampling_denoising_std: Optional[float] = None
    ddim_steps: Optional[int] = None
    ddim_eta: float = 1.0
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16
    diffusion_step_embed_dim: int = 64
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    n_groups: int = 8
    output_dir: Optional[str] = None
    render_mode: str = "rgb_array"
    video_fps: int = 20


def main():
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Output dir
    if args.output_dir is None:
        ckpt_name = os.path.basename(os.path.dirname(args.ckpt))
        noise_tag = "det" if args.min_sampling_denoising_std is None else f"std{args.min_sampling_denoising_std}"
        args.output_dir = f"runs/videos/{ckpt_name}_{noise_tag}"
    os.makedirs(os.path.join(args.output_dir, "success"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "failure"), exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    if "ema" in ckpt and "ema_agent" not in ckpt:
        # DPPO format
        from DPPO.model.unet_wrapper import DiffusionUNet
        from DPPO.model.diffusion import DiffusionModel

        ckpt_args = ckpt["args"]
        obs_dim, act_dim = ckpt["obs_dim"], ckpt["action_dim"]
        cond_steps = ckpt_args.get("cond_steps", args.obs_horizon)
        horizon_steps = ckpt_args.get("horizon_steps", args.pred_horizon)
        act_steps = ckpt_args.get("act_steps", args.act_horizon)
        act_offset = cond_steps - 1

        network = DiffusionUNet(
            action_dim=act_dim, horizon_steps=horizon_steps,
            cond_dim=obs_dim * cond_steps,
            diffusion_step_embed_dim=ckpt_args.get("diffusion_step_embed_dim", 64),
            down_dims=ckpt_args.get("unet_dims", [64, 128, 256]),
            n_groups=ckpt_args.get("n_groups", 8),
        )
        model = DiffusionModel(
            network=network, horizon_steps=horizon_steps,
            obs_dim=obs_dim, action_dim=act_dim, device=device,
            denoising_steps=ckpt_args.get("denoising_steps", 100),
            denoised_clip_value=1.0, randn_clip_value=10,
            final_action_clip_value=1.0, predict_epsilon=True,
            base_eta=args.ddim_eta,
        )
        model.load_state_dict(ckpt["ema"], strict=False)
        model.eval()
        min_std = args.min_sampling_denoising_std
        deterministic = (min_std is None)

        def get_action(obs_seq):
            cond = {"state": obs_seq}
            samples = model(cond, deterministic=deterministic,
                            min_sampling_denoising_std=min_std,
                            ddim_steps=args.ddim_steps)
            return samples.trajectories[:, act_offset:act_offset + act_steps]

        print(f"Loaded DPPO checkpoint: {args.ckpt}")
        print(f"  obs_dim={obs_dim}, act_dim={act_dim}, cond={cond_steps}, "
              f"horizon={horizon_steps}, act={act_steps}")
    else:
        raise ValueError(f"Unknown checkpoint format, keys: {list(ckpt.keys())}")

    noise_desc = "deterministic" if deterministic else f"std={min_std}, ddim={args.ddim_steps}"
    print(f"  Sampling: {noise_desc}")
    print(f"  Env: {args.env_id}, {args.control_mode}, max_steps={args.max_episode_steps}")
    print(f"  Output: {args.output_dir}")
    print()

    # Create env with video recording
    env = gym.make(args.env_id, obs_mode="state", render_mode=args.render_mode,
                   reward_mode="sparse", control_mode=args.control_mode,
                   max_episode_steps=args.max_episode_steps, reconfiguration_freq=1)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = FrameStack(env, num_stack=args.obs_horizon)
    env = CPUGymWrapper(env, ignore_terminations=False, record_metrics=True)

    n_success = 0
    n_failure = 0

    for ep in range(args.num_episodes):
        seed = args.seed + ep
        obs, _ = env.reset(seed=seed)
        frames = []
        done = False
        success = False
        step = 0

        while step < args.max_episode_steps and not done:
            # Render
            frame = env.render()
            if frame is not None:
                frames.append(frame)

            # Get action
            obs_t = torch.from_numpy(np.array(obs)).float().unsqueeze(0).to(device)
            action_seq = get_action(obs_t)  # (1, act_horizon, act_dim)
            action_np = action_seq.cpu().numpy()[0]

            # Execute action chunk
            for a_idx in range(action_np.shape[0]):
                if step >= args.max_episode_steps or done:
                    break
                obs, r, term, trunc, info = env.step(action_np[a_idx])
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
                step += 1
                if r > 0.5:
                    success = True
                    done = True
                elif term or trunc:
                    done = True

        # Save video
        if len(frames) > 0:
            import imageio
            tag = "success" if success else "failure"
            if success:
                n_success += 1
            else:
                n_failure += 1
            fname = f"seed{seed:03d}_{tag}_{step}steps.mp4"
            fpath = os.path.join(args.output_dir, tag, fname)
            imageio.mimwrite(fpath, frames, fps=args.video_fps)
            print(f"  ep {ep}: seed={seed}, {tag}, {step} steps → {fpath}")
        else:
            print(f"  ep {ep}: seed={seed}, no frames captured")
            if success:
                n_success += 1
            else:
                n_failure += 1

    env.close()
    print(f"\nDone: {n_success} success, {n_failure} failure")
    print(f"Videos saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
