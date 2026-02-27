"""Collect demo data on GPU env using a pretrained diffusion policy.

Usage:
    python -m DPPO.collect_gpu_demos \
        --ckpt runs/dppo_pretrain/dppo_1000traj_unet_5k/best.pt \
        --num_demos 1000 --num_envs 200
"""
import os
import argparse
import numpy as np
import torch
import h5py
import gymnasium as gym
import mani_skill.envs
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from DPPO.model.unet_wrapper import DiffusionUNet
from DPPO.model.diffusion import DiffusionModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--num_demos", type=int, default=1000)
    parser.add_argument("--num_envs", type=int, default=200)
    parser.add_argument("--env_id", default="PickCube-v1")
    parser.add_argument("--control_mode", default="pd_ee_delta_pos")
    parser.add_argument("--max_episode_steps", type=int, default=100)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device("cuda")

    # Load model
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ckpt_args = ckpt["args"]
    obs_dim, action_dim = ckpt["obs_dim"], ckpt["action_dim"]
    cond_steps = ckpt_args["cond_steps"]
    horizon_steps = ckpt_args["horizon_steps"]
    act_steps = ckpt_args["act_steps"]
    T = ckpt_args["denoising_steps"]
    act_offset = cond_steps - 1 if ckpt_args.get("network_type") == "unet" else 0

    network = DiffusionUNet(
        action_dim=action_dim, horizon_steps=horizon_steps,
        cond_dim=obs_dim * cond_steps,
        diffusion_step_embed_dim=ckpt_args.get("diffusion_step_embed_dim", 64),
        down_dims=ckpt_args.get("unet_dims", [64, 128, 256]),
        n_groups=ckpt_args.get("n_groups", 8),
    )
    model = DiffusionModel(
        network=network, horizon_steps=horizon_steps,
        obs_dim=obs_dim, action_dim=action_dim, device=device,
        denoising_steps=T, denoised_clip_value=1.0,
        randn_clip_value=10, final_action_clip_value=1.0, predict_epsilon=True,
    )
    model.load_state_dict(ckpt["ema"], strict=False)
    model.eval()
    print(f"Loaded model: T={T}, H={horizon_steps}, cond={cond_steps}, act={act_steps}")

    # Create GPU env
    env = gym.make(args.env_id, num_envs=args.num_envs, obs_mode="state",
                   control_mode=args.control_mode, max_episode_steps=args.max_episode_steps,
                   sim_backend="gpu", reward_mode="sparse")
    env = ManiSkillVectorEnv(env, args.num_envs, ignore_terminations=False, record_metrics=True)
    print(f"GPU env: {args.num_envs} envs, max_steps={args.max_episode_steps}")

    # Collect trajectories
    # Per-env buffers: obs (T+1, obs_dim), actions (T, action_dim)
    env_obs_bufs = [[] for _ in range(args.num_envs)]
    env_act_bufs = [[] for _ in range(args.num_envs)]
    successful_trajs = []  # list of (obs_array, act_array)

    obs, _ = env.reset()
    obs = obs.float()
    obs_history = obs.unsqueeze(1).repeat(1, cond_steps, 1)

    # Record initial obs for each env
    for i in range(args.num_envs):
        env_obs_bufs[i].append(obs[i].cpu().numpy())

    n_collected = 0
    n_total_episodes = 0
    step = 0

    while n_collected < args.num_demos:
        cond = {"state": obs_history}
        with torch.no_grad():
            samples = model(cond, deterministic=True)
        action_chunk = samples.trajectories

        for a_idx in range(act_steps):
            act_idx = act_offset + a_idx
            action = action_chunk[:, min(act_idx, action_chunk.shape[1] - 1)]

            obs_new, reward, terminated, truncated, info = env.step(action)
            obs_new = obs_new.float()

            # Record actions and next obs for each env
            for i in range(args.num_envs):
                env_act_bufs[i].append(action[i].cpu().numpy())
                env_obs_bufs[i].append(obs_new[i].cpu().numpy())

            # Check for episode completions
            done_mask = terminated | truncated
            if done_mask.any():
                for i in done_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
                    success = reward[i].item() > 0.5
                    n_total_episodes += 1

                    if success and n_collected < args.num_demos:
                        # Save this trajectory (exclude the reset obs at the end)
                        traj_obs = np.array(env_obs_bufs[i][:-1])  # (T, obs_dim) — obs before last reset
                        traj_act = np.array(env_act_bufs[i][:-1])  # (T-1, action_dim)
                        # Actually: obs has T+1 entries (init + each step), actions has T entries
                        # But we appended obs after each step AND after reset
                        # The last obs is from the reset env, not from this episode
                        # So: obs[:-1] = episode obs (init + T steps), actions[:-1] removes last action too
                        # Let's be more careful:
                        traj_obs = np.array(env_obs_bufs[i])  # includes post-reset obs at end
                        traj_act = np.array(env_act_bufs[i])
                        # The episode has len(traj_act) actions and len(traj_obs) obs
                        # Last obs is from reset, so episode obs = traj_obs[:-1], episode acts = traj_act[:-1]? No...
                        # Actually: for each action step, we append action then obs_new
                        # If the env terminates, obs_new is the reset obs
                        # So the last obs in buf is from reset, and the last action caused termination
                        # Episode: obs[0..T], acts[0..T-1], where obs[T] is the terminal obs
                        # But obs[T] here is actually the reset obs, not the true terminal obs
                        # For demo purposes, we want: obs[0..T-1] (states visited), acts[0..T-1] (actions taken)
                        # Where T = number of actions taken
                        n_acts = len(traj_act)
                        # obs has n_acts+1 entries (init + one per step), last is reset obs
                        # We want obs[0..n_acts-1] as the obs before each action, and obs[n_acts] excluded (reset)
                        ep_obs = traj_obs[:n_acts]  # (n_acts, obs_dim) — obs before each action
                        ep_act = traj_act  # (n_acts, action_dim)
                        successful_trajs.append((ep_obs, ep_act))
                        n_collected += 1

                    # Reset buffer for this env
                    env_obs_bufs[i] = [obs_new[i].cpu().numpy()]
                    env_act_bufs[i] = []

            # Update obs history
            reset_mask = terminated | truncated
            if reset_mask.any():
                obs_history[reset_mask] = obs_new[reset_mask].unsqueeze(1).repeat(1, cond_steps, 1)
            obs_history[~reset_mask] = torch.cat(
                [obs_history[~reset_mask, 1:], obs_new[~reset_mask].unsqueeze(1)], dim=1)

        step += 1
        if step % 50 == 0:
            print(f"  step={step}, collected={n_collected}/{args.num_demos}, "
                  f"total_eps={n_total_episodes}, SR={n_collected/max(n_total_episodes,1):.3f}")

    env.close()
    print(f"Collected {n_collected} successful demos from {n_total_episodes} episodes "
          f"(SR={n_collected/n_total_episodes:.3f})")

    # Print trajectory length stats
    lengths = [t[1].shape[0] for t in successful_trajs]
    print(f"Trajectory lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.1f}, median={np.median(lengths):.1f}")

    # Save as h5 (same format as ManiSkill demos)
    if args.output is None:
        args.output = os.path.expanduser(
            f"~/.maniskill/demos/{args.env_id}/gpu_collected/"
            f"trajectory.state.{args.control_mode}.physx_cuda.h5"
        )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with h5py.File(args.output, "w") as f:
        for i, (obs, act) in enumerate(successful_trajs):
            grp = f.create_group(f"traj_{i}")
            grp.create_dataset("obs", data=obs.astype(np.float32))
            grp.create_dataset("actions", data=act.astype(np.float32))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
