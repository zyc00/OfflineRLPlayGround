"""Render successful episodes as videos for visual inspection.

Saves per-episode mp4 videos so you can see where the critical decision
(grasping, insertion, etc.) happens within the episode.

Usage:
  # PickCube
  python -u -m RL.render_episodes --env_id PickCube-v1 \
    --checkpoint runs/pickcube_ppo/ckpt_76_logstd-1.5.pt \
    --max_episode_steps 50 --n_success 3

  # PegInsertion
  python -u -m RL.render_episodes --env_id PegInsertionSide-v1 \
    --checkpoint runs/peginsertion_ppo/ckpt_76.pt \
    --max_episode_steps 100 --n_success 3
"""

import os
import sys
from dataclasses import dataclass

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import tyro
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper

from data.data_collection.ppo import Agent


@dataclass
class Args:
    checkpoint: str = "runs/pickcube_ppo/ckpt_76_logstd-1.5.pt"
    env_id: str = "PickCube-v1"
    max_episode_steps: int = 50
    n_success: int = 3      # how many successful episodes to save
    n_fail: int = 1          # how many failed episodes to save (for comparison)
    max_attempts: int = 100  # max episodes to try
    seed: int = 1
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"
    output_dir: str = "runs/episode_renders"


if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda")

    env_name = args.env_id.split("-")[0].lower()
    out_dir = os.path.join(args.output_dir, env_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── Setup env (1 env, CPU for rendering) ──
    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="cpu",
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)

    agent = Agent(env).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()

    action_low = torch.from_numpy(env.single_action_space.low).to(device)
    action_high = torch.from_numpy(env.single_action_space.high).to(device)

    def clip_action(a):
        return torch.clamp(a.detach(), action_low, action_high)

    success_episodes = []
    fail_episodes = []
    ep_idx = 0

    print(f"Collecting episodes for {args.env_id}...")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Target: {args.n_success} success + {args.n_fail} fail")
    sys.stdout.flush()

    obs, _ = env.reset(seed=args.seed)

    while ep_idx < args.max_attempts:
        if len(success_episodes) >= args.n_success and len(fail_episodes) >= args.n_fail:
            break

        frames = []
        rewards = []
        obs, _ = env.reset()

        for step in range(args.max_episode_steps):
            # Render
            frame = env.render()
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            if frame.ndim == 4:
                frame = frame[0]  # (H, W, C)
            frames.append(frame)

            # Step
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=device).float()
                if obs_t.dim() == 1:
                    obs_t = obs_t.unsqueeze(0)
                action = agent.get_action(obs_t, deterministic=False)
            obs, reward, term, trunc, info = env.step(clip_action(action))
            r = float(reward) if np.isscalar(reward) else float(reward.item()) if hasattr(reward, 'item') else float(np.array(reward).flat[0])
            rewards.append(r)

            done = bool(term) or bool(trunc) if np.isscalar(term) else bool(term.any()) or bool(trunc.any())
            if done:
                break

        ep_idx += 1
        success = any(r > 0.5 for r in rewards)
        ep_len = len(frames)

        # Find reward step
        reward_step = None
        for i, r in enumerate(rewards):
            if r > 0.5:
                reward_step = i
                break

        label = "SUCCESS" if success else "FAIL"
        print(f"  Episode {ep_idx}: {label}, len={ep_len}"
              + (f", reward at step {reward_step} ({reward_step/ep_len*100:.0f}%)" if reward_step is not None else ""))
        sys.stdout.flush()

        ep_data = dict(frames=frames, rewards=rewards, success=success,
                       length=ep_len, reward_step=reward_step)

        if success and len(success_episodes) < args.n_success:
            success_episodes.append(ep_data)
        elif not success and len(fail_episodes) < args.n_fail:
            fail_episodes.append(ep_data)

    env.close()

    # ── Save videos as mp4 ──
    try:
        import imageio
        has_imageio = True
    except ImportError:
        has_imageio = False
        print("WARNING: imageio not installed, saving as .npy instead")

    all_eps = [(ep, "success") for ep in success_episodes] + [(ep, "fail") for ep in fail_episodes]

    for i, (ep, label) in enumerate(all_eps):
        fname = f"{env_name}_{label}_{i}"

        if has_imageio:
            video_path = os.path.join(out_dir, f"{fname}.mp4")
            writer = imageio.get_writer(video_path, fps=10)
            for frame in ep["frames"]:
                if frame.dtype != np.uint8:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                writer.append_data(frame)
            writer.close()
            print(f"Saved: {video_path} (len={ep['length']}, reward_step={ep['reward_step']})")
        else:
            npy_path = os.path.join(out_dir, f"{fname}.npy")
            np.save(npy_path, np.stack(ep["frames"]))
            print(f"Saved: {npy_path}")

    # ── Also save a summary image grid: key frames from each success episode ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if success_episodes:
        n_cols = 8  # frames per episode to show
        n_rows = len(success_episodes)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
        if n_rows == 1:
            axes = [axes]

        for row, ep in enumerate(success_episodes):
            ep_len = ep["length"]
            rs = ep["reward_step"]

            # Pick frames: evenly spaced + reward step
            if rs is not None:
                # Show: start, a few before reward, reward step, a few after, end
                key_steps = sorted(set([
                    0,
                    max(0, rs - 5),
                    max(0, rs - 2),
                    max(0, rs - 1),
                    rs,
                    min(ep_len - 1, rs + 1),
                    min(ep_len - 1, rs + 3),
                    ep_len - 1,
                ]))
            else:
                key_steps = [int(i * (ep_len - 1) / (n_cols - 1)) for i in range(n_cols)]

            # Trim to n_cols
            if len(key_steps) > n_cols:
                key_steps = key_steps[:n_cols]
            while len(key_steps) < n_cols:
                key_steps.append(key_steps[-1])

            for col, step_idx in enumerate(key_steps):
                ax = axes[row][col]
                frame = ep["frames"][step_idx]
                if frame.dtype != np.uint8:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                ax.imshow(frame)
                pct = step_idx / ep_len * 100
                title = f"t={step_idx} ({pct:.0f}%)"
                if rs is not None and step_idx == rs:
                    title += "\nREWARD"
                    ax.set_title(title, fontsize=8, color="red", fontweight="bold")
                else:
                    ax.set_title(title, fontsize=8)
                ax.axis("off")

        fig.suptitle(f"{args.env_id} — Successful Episodes (key frames around reward)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        grid_path = os.path.join(out_dir, f"{env_name}_keyframes.png")
        plt.savefig(grid_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nSaved keyframe grid: {grid_path}")

    # ── Print summary ──
    print(f"\n{'='*50}")
    print(f"Summary for {args.env_id}")
    print(f"{'='*50}")
    print(f"Attempts: {ep_idx}")
    print(f"Successes: {len(success_episodes)}, Fails: {len(fail_episodes)}")
    if success_episodes:
        reward_pcts = [ep["reward_step"] / ep["length"] * 100
                       for ep in success_episodes if ep["reward_step"] is not None]
        if reward_pcts:
            print(f"Reward step (episode %): {[f'{p:.0f}%' for p in reward_pcts]}")
            print(f"  Mean: {np.mean(reward_pcts):.1f}%")
            print(f"  Min:  {np.min(reward_pcts):.1f}%")
            print(f"  Max:  {np.max(reward_pcts):.1f}%")
