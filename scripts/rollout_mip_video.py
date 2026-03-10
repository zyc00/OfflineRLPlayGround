"""Rollout MIP policy and save videos.

Usage:
    python scripts/rollout_mip_video.py \
      --ckpt runs/mip_pretrain/mip_peg_split900/best.pt \
      --num_episodes 50 --output /tmp/mip_videos
"""
import os
import sys
import argparse
import numpy as np
import torch
import gymnasium as gym
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mani_skill.envs
import DPPO.peg_insertion_easy
from mani_skill.utils.wrappers import CPUGymWrapper

from MultiGaussian.models.multi_gaussian import MIPPolicy
from MultiGaussian.models.mip_unet import MIPUNetPolicy


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
    return model, ckpt


@torch.no_grad()
def rollout_and_record(model, ckpt, n_episodes, env_id, control_mode,
                       max_episode_steps, device, output_dir):
    args = ckpt["args"]
    cond_steps = args.get("cond_steps", 1)
    horizon_steps = args.get("horizon_steps", 1)
    act_steps = args.get("act_steps", 1)
    zero_qvel = args.get("zero_qvel", False)
    no_obs_norm = ckpt.get("no_obs_norm", False)
    no_action_norm = ckpt.get("no_action_norm", False)

    if not no_obs_norm:
        o_lo = ckpt["obs_min"].to(device)
        o_hi = ckpt["obs_max"].to(device)
    if not no_action_norm:
        a_lo = ckpt["action_min"].to(device)
        a_hi = ckpt["action_max"].to(device)

    os.makedirs(output_dir, exist_ok=True)
    successes = []

    for ep in range(n_episodes):
        env = gym.make(env_id, obs_mode="state", control_mode=control_mode,
                       render_mode="rgb_array", max_episode_steps=max_episode_steps,
                       reconfiguration_freq=1)
        env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)

        obs, _ = env.reset(seed=ep)
        obs = torch.from_numpy(obs).float().to(device).unsqueeze(0)  # (1, obs_dim)
        obs_buffer = deque(maxlen=cond_steps)
        for _ in range(cond_steps):
            obs_buffer.append(obs.clone())

        frames = []
        success = False
        step = 0

        while step < max_episode_steps:
            if cond_steps == 1:
                cond_obs = obs_buffer[-1]
            else:
                cond_obs = torch.stack(list(obs_buffer), dim=1)  # (1, cond, obs_dim)

            cond_obs_proc = cond_obs.clone()
            if zero_qvel:
                cond_obs_proc[..., 9:18] = 0.0
            if not no_obs_norm:
                cond_obs_proc = (cond_obs_proc - o_lo) / (o_hi - o_lo + 1e-8) * 2.0 - 1.0

            actions_chunk = model.predict(cond_obs_proc)
            if not no_action_norm:
                actions_chunk = (actions_chunk + 1.0) / 2.0 * (a_hi - a_lo) + a_lo

            n_exec = min(act_steps, max_episode_steps - step) if horizon_steps > 1 else 1
            for t_idx in range(n_exec):
                if horizon_steps > 1:
                    action = actions_chunk[:, t_idx]
                else:
                    action = actions_chunk
                action_np = action.cpu().numpy().squeeze(0)

                obs_np, rew, terminated, truncated, info = env.step(action_np)
                frames.append(env.render())
                obs = torch.from_numpy(obs_np).float().to(device).unsqueeze(0)
                obs_buffer.append(obs.clone())
                step += 1

                if info.get("success", False):
                    success = True

                if terminated or truncated:
                    break
            if terminated or truncated:
                # Capture final frame
                frames.append(env.render())
                break

        env.close()
        successes.append(success)

        # Save video
        tag = "S" if success else "F"
        video_path = os.path.join(output_dir, f"ep{ep:03d}_{tag}.mp4")
        save_video(frames, video_path)
        print(f"  ep {ep:3d}: {'SUCCESS' if success else 'FAIL':7s} ({step:3d} steps) → {video_path}")

    sr = np.mean(successes)
    print(f"\nTotal: {sum(successes)}/{n_episodes} = {sr:.1%}")
    return successes


def save_video(frames, path, fps=20):
    """Save frames as mp4 video."""
    import cv2
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--env_id", default="PegInsertionSide-v1")
    parser.add_argument("--control_mode", default="pd_joint_delta_pos")
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--output", default="/tmp/mip_videos")
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"Loading {args.ckpt}...")
    model, ckpt = load_mip_model(args.ckpt, device)
    print(f"Rolling out {args.num_episodes} episodes...")
    rollout_and_record(model, ckpt, args.num_episodes, args.env_id,
                       args.control_mode, args.max_episode_steps, device, args.output)


if __name__ == "__main__":
    main()
