"""Replay script to verify a collected .pt dataset.

Loads the dataset, prints summary statistics, then re-runs the policy
from the same checkpoint with video recording for visual verification.
"""

import os
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import tyro
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import envs
from data.data_collection.ppo import Agent


@dataclass
class Args:
    dataset: str = "data/datasets/pickcube_expert_eval.pt"
    """path to the .pt dataset to verify"""
    checkpoint: str = "runs/pickcube_ppo/ckpt_301.pt"
    """path to the PPO checkpoint (for live replay with video)"""
    env_id: str = "PickCube-v2"
    """the id of the environment"""
    num_eval_episodes: int = 10
    """number of episodes to replay live with video"""
    max_episode_steps: int = 50
    """max steps per episode"""
    deterministic: bool = False
    """use deterministic actions for live replay"""
    control_mode: str = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
    output_dir: str = "data/datasets/replay_videos"
    """directory to save replay videos"""
    seed: int = 1
    """random seed"""
    cuda: bool = True
    """if toggled, cuda will be enabled"""


def print_tensor_stats(name: str, t: torch.Tensor):
    """Print shape/dtype/min/max/mean for a tensor."""
    print(
        f"  {name:20s}: shape={str(list(t.shape)):20s} dtype={t.dtype}  "
        f"min={t.float().min():.4f}  max={t.float().max():.4f}  "
        f"mean={t.float().mean():.4f}"
    )


def print_dataset_stats(path: str):
    """Load a .pt dataset and print summary statistics."""
    print(f"\n{'=' * 60}")
    print(f"Dataset: {path}")
    print(f"{'=' * 60}")

    data = torch.load(path)
    print(f"Keys: {list(data.keys())}")

    def print_value(key: str, val, indent: int = 0):
        prefix = "  " * indent
        if isinstance(val, dict):
            print(f"{prefix}{key}:")
            for sub_key, sub_val in val.items():
                print_value(sub_key, sub_val, indent + 1)
        elif isinstance(val, torch.Tensor):
            print_tensor_stats(f"{prefix}{key}", val)
        else:
            print(f"{prefix}{key:20s}: type={type(val).__name__}")

    for key, val in data.items():
        print_value(key, val, indent=1)

    obs = data["obs"]
    rewards = data["rewards"]
    dones = data["dones"]
    terminated = data.get("terminated")

    # obs may be a dict {"state": ..., "rgb": ...} or a flat tensor
    if isinstance(obs, dict):
        num_transitions = obs["state"].shape[0]
    else:
        num_transitions = obs.shape[0]
    num_episodes = int(dones.sum().item())

    print("\n--- Summary ---")
    print(f"Total transitions: {num_transitions}")
    print(f"Total episodes:    {num_episodes}")
    if num_episodes > 0:
        print(f"Avg transitions/episode: {num_transitions / num_episodes:.1f}")

    # Reconstruct per-episode rewards
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    current_reward = 0.0
    current_length = 0
    for i in range(num_transitions):
        current_reward += rewards[i].item()
        current_length += 1
        if dones[i] > 0.5:
            episode_rewards.append(current_reward)
            episode_lengths.append(current_length)
            if terminated is not None:
                episode_successes.append(terminated[i].item() > 0.5)
            current_reward = 0.0
            current_length = 0

    if episode_rewards:
        ep_r = np.array(episode_rewards)
        ep_l = np.array(episode_lengths)
        print("\n--- Per-Episode Stats ---")
        print(
            f"Episode return:  mean={ep_r.mean():.4f}  std={ep_r.std():.4f}  "
            f"min={ep_r.min():.4f}  max={ep_r.max():.4f}"
        )
        print(
            f"Episode length:  mean={ep_l.mean():.1f}  std={ep_l.std():.1f}  "
            f"min={ep_l.min()}  max={ep_l.max()}"
        )
        if episode_successes:
            success_rate = np.mean(episode_successes)
            print(
                f"Success rate (terminated): {success_rate:.2%} "
                f"({int(sum(episode_successes))}/{len(episode_successes)})"
            )

    return data


def live_replay(args: Args):
    """Run the policy live in the environment with video recording."""
    print(f"\n{'=' * 60}")
    print(f"Live replay: {args.num_eval_episodes} episodes from {args.checkpoint}")
    print(f"{'=' * 60}")

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    sim_backend = "physx_cuda" if device.type == "cuda" else "cpu"
    num_envs = 1  # single env for clear per-episode videos

    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend=sim_backend,
        control_mode=args.control_mode,
    )
    eval_envs = gym.make(args.env_id, num_envs=num_envs, **env_kwargs)
    if isinstance(eval_envs.action_space, gym.spaces.Dict):
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    os.makedirs(args.output_dir, exist_ok=True)
    eval_envs = RecordEpisode(
        eval_envs,
        output_dir=args.output_dir,
        save_trajectory=False,
        max_steps_per_video=args.max_episode_steps,
        video_fps=30,
    )
    eval_envs = ManiSkillVectorEnv(
        eval_envs, num_envs, ignore_terminations=False, record_metrics=True
    )

    agent = Agent(eval_envs).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()

    action_low = torch.from_numpy(eval_envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(eval_envs.single_action_space.high).to(device)

    episode_rewards = []
    episode_successes = []
    episodes_done = 0

    obs, _ = eval_envs.reset(seed=args.seed)
    current_reward = 0.0

    while episodes_done < args.num_eval_episodes:
        with torch.no_grad():
            action = agent.get_action(obs, deterministic=args.deterministic)
        action = torch.clamp(action.detach(), action_low, action_high)
        obs, reward, terminated, truncated, _ = eval_envs.step(action)
        current_reward += reward.item()
        done = terminated | truncated

        if done.any():
            episode_rewards.append(current_reward)
            episode_successes.append(bool(terminated.any()))
            episodes_done += 1
            print(
                f"  Episode {episodes_done}/{args.num_eval_episodes}: "
                f"return={current_reward:.4f}  "
                f"success={bool(terminated.any())}"
            )
            current_reward = 0.0

    eval_envs.close()

    ep_r = np.array(episode_rewards)
    print("\n--- Live Replay Summary ---")
    print(f"Episode return:  mean={ep_r.mean():.4f}  std={ep_r.std():.4f}")
    print(
        f"Success rate:    {np.mean(episode_successes):.2%} "
        f"({int(sum(episode_successes))}/{len(episode_successes)})"
    )
    print(f"Videos saved to: {args.output_dir}")


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Phase 1: Dataset statistics
    if os.path.exists(args.dataset):
        print_dataset_stats(args.dataset)
    else:
        print(f"Dataset not found: {args.dataset} (skipping stats)")

    # Phase 2: Live replay with video
    if os.path.exists(args.checkpoint):
        live_replay(args)
    else:
        print(f"Checkpoint not found: {args.checkpoint} (skipping live replay)")
