"""Collect Push-T demos by rolling out a PPO checkpoint.

Saves state obs + actions, and optionally RGB images from env.render().
"""
import os
import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import mani_skill.envs
from tqdm import tqdm


class PPOActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs):
        return self.actor_mean(obs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="~/.maniskill/demos/PushT-v1/rl/ppo_pd_ee_delta_pos_ckpt.pt")
    parser.add_argument("--env_id", default="PushT-v1")
    parser.add_argument("--control_mode", default="pd_ee_delta_pos")
    parser.add_argument("--max_episode_steps", type=int, default=100)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--num_trajs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--save_rgb", action="store_true", help="Save RGB images from env.render()")
    parser.add_argument("--render_size", type=int, default=128, help="Render image size")
    args = parser.parse_args()

    args.ckpt = os.path.expanduser(args.ckpt)
    if args.output is None:
        suffix = "rgb" if args.save_rgb else "state"
        args.output = os.path.expanduser(
            f"~/.maniskill/demos/{args.env_id}/rl/trajectory.{suffix}.{args.control_mode}.collected.h5"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    obs_dim = sd["actor_mean.0.weight"].shape[1]
    act_dim = sd["actor_mean.6.weight"].shape[0]
    print(f"obs_dim={obs_dim}, act_dim={act_dim}")

    actor = PPOActor(obs_dim, act_dim).to(device)
    actor_sd = {k: v for k, v in sd.items() if k.startswith("actor_mean.")}
    actor.load_state_dict({"actor_mean." + k.split("actor_mean.")[1]: v for k, v in actor_sd.items()})
    actor.eval()

    # Create vectorized env on GPU for speed
    render_kwargs = {}
    if args.save_rgb:
        render_kwargs["human_render_camera_configs"] = {"width": args.render_size, "height": args.render_size}
    env = gym.make(args.env_id, obs_mode="state", control_mode=args.control_mode,
                   render_mode="rgb_array", max_episode_steps=args.max_episode_steps,
                   num_envs=args.num_envs, sim_backend="physx_cuda",
                   reconfiguration_freq=1, **render_kwargs)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    f = h5py.File(args.output, "w")

    traj_id = 0
    pbar = tqdm(total=args.num_trajs, desc="Collecting")

    # Storage for current episodes
    obs_buf = [[] for _ in range(args.num_envs)]
    act_buf = [[] for _ in range(args.num_envs)]
    img_buf = [[] for _ in range(args.num_envs)] if args.save_rgb else None

    obs, _ = env.reset(seed=args.seed)
    for i in range(args.num_envs):
        obs_buf[i].append(obs[i].cpu().numpy())

    if args.save_rgb:
        imgs = env.render().cpu().numpy()  # (num_envs, H, W, 3)
        for i in range(args.num_envs):
            img_buf[i].append(imgs[i])

    # Track which envs just reset (to avoid saving 1-step truncated episodes)
    just_reset = torch.zeros(args.num_envs, dtype=torch.bool)

    while traj_id < args.num_trajs:
        with torch.no_grad():
            action = actor(obs)
            action = action.clamp(-1, 1)

        next_obs, reward, terminated, truncated, info = env.step(action)

        for i in range(args.num_envs):
            act_buf[i].append(action[i].cpu().numpy())

        if args.save_rgb:
            imgs = env.render().cpu().numpy()
            for i in range(args.num_envs):
                img_buf[i].append(imgs[i])

        done = terminated | truncated
        if done.any():
            done_indices = done.nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
            if isinstance(done_indices, int):
                done_indices = [done_indices]

            for idx in done_indices:
                if traj_id >= args.num_trajs:
                    break

                # Skip 1-step episodes from immediate post-reset termination
                if len(act_buf[idx]) <= 1 and just_reset[idx]:
                    obs_buf[idx] = []
                    act_buf[idx] = []
                    if args.save_rgb:
                        img_buf[idx] = []
                    continue

                obs_arr = np.array(obs_buf[idx])
                act_arr = np.array(act_buf[idx])
                success = bool(info.get("success", torch.zeros(1))[idx])

                grp = f.create_group(f"traj_{traj_id}")
                grp.create_dataset("obs", data=obs_arr)
                grp.create_dataset("actions", data=act_arr)
                if args.save_rgb:
                    img_arr = np.array(img_buf[idx])  # (T+1, H, W, 3) uint8
                    grp.create_dataset("rgb", data=img_arr, compression="gzip", compression_opts=4)
                grp.attrs["success"] = success
                grp.attrs["length"] = len(act_arr)

                traj_id += 1
                pbar.update(1)

                obs_buf[idx] = []
                act_buf[idx] = []
                if args.save_rgb:
                    img_buf[idx] = []

        # After processing done, start new episodes for done envs
        just_reset = done.cpu()
        for i in range(args.num_envs):
            if done[i]:
                # New episode starts with the auto-reset obs
                obs_buf[i].append(next_obs[i].cpu().numpy())
                if args.save_rgb:
                    img_buf[i].append(imgs[i])
            else:
                obs_buf[i].append(next_obs[i].cpu().numpy())

        obs = next_obs

    f.close()
    pbar.close()
    env.close()

    # Summary
    f = h5py.File(args.output, "r")
    n = len([k for k in f.keys() if k.startswith("traj_")])
    successes = sum(1 for i in range(n) if f[f"traj_{i}"].attrs.get("success", False))
    lengths = [f[f"traj_{i}"].attrs["length"] for i in range(n)]
    print(f"\nSaved {n} trajs to {args.output}")
    print(f"Success: {successes}/{n} = {successes/n*100:.1f}%")
    print(f"Length: mean={np.mean(lengths):.1f}, min={min(lengths)}, max={max(lengths)}")
    print(f"obs shape: {f['traj_0/obs'].shape}, actions shape: {f['traj_0/actions'].shape}")
    if args.save_rgb and "rgb" in f["traj_0"]:
        print(f"rgb shape: {f['traj_0/rgb'].shape}, dtype: {f['traj_0/rgb'].dtype}")
    f.close()


if __name__ == "__main__":
    main()
