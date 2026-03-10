"""Save evaluation videos for MIP policy (success & failure cases).

Usage:
  python -m MultiGaussian.eval_mip_video --ckpt runs/mip_pretrain/mip_PegInsertion_chunk16_v2/ckpt_50000.pt \
    --num-episodes 10
"""
import os
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs  # noqa: F401
import imageio
from collections import deque
from dataclasses import dataclass
from typing import Optional

import tyro
from mani_skill.utils.wrappers import CPUGymWrapper

from MultiGaussian.eval_mip_cpu import _load_model_from_ckpt


@dataclass
class Args:
    ckpt: str = ""
    env_id: Optional[str] = None
    control_mode: Optional[str] = None
    max_episode_steps: Optional[int] = None
    num_episodes: int = 10
    seed: int = 0
    output_dir: Optional[str] = None
    video_fps: int = 20


def main():
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ckpt = _load_model_from_ckpt(args.ckpt, device)
    ckpt_args = ckpt["args"]

    env_id = args.env_id or ckpt_args.get("env_id", "PickCube-v1")
    control_mode = args.control_mode or ckpt_args.get("control_mode", "pd_ee_delta_pos")
    max_episode_steps = args.max_episode_steps or ckpt_args.get("max_episode_steps", 100)
    no_obs_norm = ckpt.get("no_obs_norm", False)
    no_action_norm = ckpt.get("no_action_norm", False)
    zero_qvel = ckpt_args.get("zero_qvel", False)
    cond_steps = ckpt_args.get("cond_steps", 1)
    horizon_steps = ckpt_args.get("horizon_steps", 1)
    act_steps = ckpt_args.get("act_steps", 1)

    if not no_obs_norm:
        o_lo = ckpt["obs_min"].to(device)
        o_hi = ckpt["obs_max"].to(device)
    if not no_action_norm:
        a_lo = ckpt["action_min"].to(device)
        a_hi = ckpt["action_max"].to(device)

    if args.output_dir is None:
        ckpt_name = os.path.basename(os.path.dirname(args.ckpt))
        step = ckpt_args.get("step", os.path.basename(args.ckpt).replace(".pt", ""))
        args.output_dir = f"runs/videos/{ckpt_name}_iter{step}"
    os.makedirs(os.path.join(args.output_dir, "success"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "failure"), exist_ok=True)

    print(f"MIP eval video: {env_id}, {control_mode}, max_steps={max_episode_steps}")
    print(f"  cond={cond_steps}, horizon={horizon_steps}, act={act_steps}, zero_qvel={zero_qvel}")
    print(f"  Output: {args.output_dir}")

    env = gym.make(env_id, obs_mode="state", render_mode="rgb_array",
                   reward_mode="sparse", control_mode=control_mode,
                   max_episode_steps=max_episode_steps, reconfiguration_freq=1)
    env = CPUGymWrapper(env, ignore_terminations=False, record_metrics=True)

    n_success = 0
    n_failure = 0

    for ep in range(args.num_episodes):
        seed = args.seed + ep
        obs, _ = env.reset(seed=seed)
        obs = torch.from_numpy(np.array(obs)).float().unsqueeze(0).to(device)

        obs_buffer = deque(maxlen=cond_steps)
        for _ in range(cond_steps):
            obs_buffer.append(obs.clone())

        frames = []
        done = False
        success = False
        step = 0

        while step < max_episode_steps and not done:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

            if cond_steps == 1:
                cond_obs = obs_buffer[-1]
            else:
                cond_obs = torch.stack(list(obs_buffer), dim=1)

            if zero_qvel:
                cond_obs[..., 9:18] = 0.0

            if no_obs_norm:
                obs_norm = cond_obs
            else:
                obs_norm = (cond_obs - o_lo) / (o_hi - o_lo + 1e-8) * 2.0 - 1.0

            actions_chunk = model.predict(obs_norm)

            if not no_action_norm:
                actions_chunk = (actions_chunk + 1.0) / 2.0 * (a_hi - a_lo) + a_lo

            n_exec = min(act_steps, max_episode_steps - step) if horizon_steps > 1 else 1
            for t in range(n_exec):
                if step >= max_episode_steps or done:
                    break
                if horizon_steps > 1:
                    action = actions_chunk[:, t]
                else:
                    action = actions_chunk

                obs_np, r, term, trunc, info = env.step(action.cpu().numpy()[0])
                obs = torch.from_numpy(np.array(obs_np)).float().unsqueeze(0).to(device)
                obs_buffer.append(obs.clone())

                frame = env.render()
                if frame is not None:
                    frames.append(frame)
                step += 1

                if r > 0.5:
                    success = True
                    done = True
                elif term or trunc:
                    done = True

        tag = "success" if success else "failure"
        if success:
            n_success += 1
        else:
            n_failure += 1

        if len(frames) > 0:
            fname = f"seed{seed:03d}_{tag}_{step}steps.mp4"
            fpath = os.path.join(args.output_dir, tag, fname)
            imageio.mimwrite(fpath, frames, fps=args.video_fps)
            print(f"  ep {ep}: seed={seed}, {tag}, {step} steps -> {fpath}")
        else:
            print(f"  ep {ep}: seed={seed}, {tag}, {step} steps (no frames)")

    env.close()
    print(f"\nDone: {n_success} success, {n_failure} failure out of {args.num_episodes}")
    print(f"Videos saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
