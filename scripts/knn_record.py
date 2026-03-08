"""Record KNN policy rollout videos."""

import argparse
import torch
import numpy as np
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers import CPUGymWrapper, RecordEpisode
from scripts.knn_policy import build_knn_from_demos
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_path", type=str, required=True)
    parser.add_argument("--env_id", type=str, default="PegInsertionSide-v1")
    parser.add_argument("--control_mode", type=str, default="pd_joint_delta_pos")
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--horizon_steps", type=int, default=16)
    parser.add_argument("--cond_steps", type=int, default=1)
    parser.add_argument("--act_steps", type=int, default=8)
    parser.add_argument("--zero_qvel", action="store_true")
    parser.add_argument("--n_episodes", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="/tmp/knn_videos")
    args = parser.parse_args()

    args.demo_path = os.path.expanduser(args.demo_path)
    os.makedirs(args.output_dir, exist_ok=True)

    policy = build_knn_from_demos(
        args.demo_path, horizon_steps=args.horizon_steps,
        cond_steps=args.cond_steps, k=args.k,
        weighting='inverse_distance', normalize=True,
        zero_qvel=args.zero_qvel,
    )

    env = gym.make(args.env_id, obs_mode="state", control_mode=args.control_mode,
                   render_mode="rgb_array", max_episode_steps=args.max_episode_steps,
                   reconfiguration_freq=1)
    env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
    env = RecordEpisode(env, output_dir=args.output_dir, save_trajectory=False,
                        info_on_video=True, max_steps_per_video=args.max_episode_steps)

    for ep in range(args.n_episodes):
        obs, _ = env.reset()
        obs = torch.from_numpy(obs).float().unsqueeze(0)  # (1, obs_dim)

        step = 0
        truncated = False
        while step < args.max_episode_steps and not truncated:
            cond_obs = obs.clone()
            if args.zero_qvel:
                cond_obs[..., 9:18] = 0.0

            act_chunk = policy.predict(cond_obs)  # (1, horizon, act_dim)

            n_exec = min(args.act_steps, args.max_episode_steps - step)
            for t in range(n_exec):
                action = act_chunk[0, t].numpy() if args.horizon_steps > 1 else act_chunk[0].numpy()
                obs_np, rew, terminated, truncated, info = env.step(action)
                obs = torch.from_numpy(obs_np).float().unsqueeze(0)
                step += 1
                if truncated:
                    break

        print(f"Episode {ep}: done at step {step}")

    env.close()
    print(f"\nVideos saved to {args.output_dir}")
    # List files
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith('.mp4'):
            fpath = os.path.join(args.output_dir, f)
            size = os.path.getsize(fpath) / 1024
            print(f"  {f} ({size:.0f} KB)")


if __name__ == "__main__":
    main()
