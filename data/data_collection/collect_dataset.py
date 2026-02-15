import os
import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import tyro
from mani_skill.utils.wrappers.flatten import (
    FlattenActionSpaceWrapper,
    FlattenRGBDObservationWrapper,
)
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import envs
from data.data_collection.ppo import Agent


@dataclass
class Args:
    checkpoint: str = "runs/pickcube_ppo/ckpt_301.pt"
    """path to a pretrained PPO checkpoint file"""
    env_id: str = "PickCube-v2"
    """the id of the environment"""
    num_envs: int = 16
    """number of parallel environments for collection"""
    num_episodes: int = 1000
    """total number of episodes to collect (may slightly overshoot)"""
    deterministic: bool = False
    """use deterministic (mean) actions instead of sampling"""
    seed: int = 1
    """random seed"""
    output: str = "data/datasets/pickcube_expert.pt"
    """output path for the dataset file"""
    control_mode: str = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
    cuda: bool = True
    """if toggled, cuda will be enabled"""
    obs_mode: str = "rgb"
    """observation mode for the environment (e.g., 'state', 'rgbd')"""
    reward_mode: str = "sparse"
    """reward mode for the environment (e.g., 'sparse', 'dense')"""


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    sim_backend = "physx_cuda" if device.type == "cuda" else "cpu"
    num_envs = args.num_envs if device.type == "cuda" else 1

    is_visual = args.obs_mode != "state"

    # Environment setup
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        sim_backend=sim_backend,
        control_mode=args.control_mode,
        reward_mode=args.reward_mode,
    )
    envs = gym.make(args.env_id, num_envs=num_envs, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    if is_visual:
        envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=True)
    envs = ManiSkillVectorEnv(
        envs, num_envs, ignore_terminations=False, record_metrics=True
    )

    # Load agent
    if is_visual:
        # Create temporary state-only env for Agent init (obs space shape must match checkpoint)
        state_env_kwargs = dict(
            obs_mode="state",
            render_mode="rgb_array",
            sim_backend=sim_backend,
            control_mode=args.control_mode,
            reward_mode=args.reward_mode,
        )
        state_envs = gym.make(args.env_id, num_envs=1, **state_env_kwargs)
        if isinstance(state_envs.action_space, gym.spaces.Dict):
            state_envs = FlattenActionSpaceWrapper(state_envs)
        state_envs = ManiSkillVectorEnv(
            state_envs, 1, ignore_terminations=False, record_metrics=False
        )
        agent = Agent(state_envs).to(device)
        state_envs.close()
    else:
        agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()

    # Action clipping (matches ppo.py)
    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(action: torch.Tensor):
        return torch.clamp(action.detach(), action_low, action_high)

    # Access base env for state capture
    base_env = envs.unwrapped

    # Collection storage
    all_obs = []
    all_next_obs = []
    all_actions = []
    all_rewards = []
    all_dones = []
    all_terminated = []
    all_truncated = []
    all_log_probs = []
    all_values = []
    all_env_states = []

    episodes_collected = 0
    obs, _ = envs.reset(seed=args.seed)

    print(
        f"Collecting {args.num_episodes} episodes from {args.checkpoint} "
        f"({'deterministic' if args.deterministic else 'stochastic'} policy, "
        f"obs_mode={args.obs_mode})"
    )

    def _concat_nested(items):
        """Recursively concatenate a list of nested dicts of tensors along dim 0."""
        if isinstance(items[0], dict):
            return {k: _concat_nested([it[k] for it in items]) for k in items[0]}
        return torch.cat(items, dim=0)

    while episodes_collected < args.num_episodes:
        # Capture simulator state before stepping (matches current obs).
        # Tensors have shape (num_envs, D); after concatenation across steps
        # they become (total_transitions, D), matching obs/actions layout.
        # To restore transition i: slice [i:i+1] from each leaf tensor and
        # pass the resulting dict to base_env.set_state_dict().
        env_state = base_env.get_state_dict()
        all_env_states.append(
            {
                k: (
                    v.cpu().clone()
                    if isinstance(v, torch.Tensor)
                    else {sk: sv.cpu().clone() for sk, sv in v.items()}
                )
                for k, v in env_state.items()
            }
        )

        # Extract state for agent input
        state_obs = obs["state"] if is_visual else obs

        with torch.no_grad():
            if args.deterministic:
                action = agent.get_action(state_obs, deterministic=True)
                _, log_prob, _, value = agent.get_action_and_value(
                    state_obs, action=action
                )
            else:
                action, log_prob, _, value = agent.get_action_and_value(state_obs)

        clipped_action = clip_action(action)
        next_obs, reward, terminated, truncated, info = envs.step(clipped_action)
        done = terminated | truncated

        # For finished envs, next_obs is already the reset obs.
        # Use final_observation for the true terminal next_obs.
        if is_visual:
            actual_next_state = next_obs["state"].clone()
            actual_next_rgb = next_obs["rgb"].clone()
            if "final_observation" in info:
                mask = info["_final_info"]
                actual_next_state[mask] = info["final_observation"]["state"][mask]
                actual_next_rgb[mask] = info["final_observation"]["rgb"][mask]

            all_obs.append(
                {"state": state_obs.cpu().numpy(), "rgb": obs["rgb"].cpu().numpy()}
            )
            all_next_obs.append(
                {
                    "state": actual_next_state.cpu().numpy(),
                    "rgb": actual_next_rgb.cpu().numpy(),
                }
            )
        else:
            actual_next_obs = next_obs.clone()
            if "final_observation" in info:
                mask = info["_final_info"]
                actual_next_obs[mask] = info["final_observation"][mask]

            all_obs.append(obs.cpu().numpy())
            all_next_obs.append(actual_next_obs.cpu().numpy())

        # Store transitions (CPU numpy to save GPU memory)
        all_actions.append(clipped_action.cpu().numpy())
        all_rewards.append(reward.cpu().numpy())
        all_dones.append(done.cpu().numpy().astype(np.float32))
        all_terminated.append(terminated.cpu().numpy().astype(np.float32))
        all_truncated.append(truncated.cpu().numpy().astype(np.float32))
        all_log_probs.append(log_prob.cpu().numpy())
        all_values.append(value.squeeze(-1).cpu().numpy())

        # Count finished episodes
        if done.any():
            new_episodes = int(done.sum().item())
            episodes_collected += new_episodes
            print(f"Episodes collected: {episodes_collected}/{args.num_episodes}")

        obs = next_obs

    # Concatenate collected data and convert to torch tensors
    if is_visual:
        obs_data = {
            "state": torch.from_numpy(
                np.concatenate([o["state"] for o in all_obs], axis=0)
            ),
            "rgb": torch.from_numpy(
                np.concatenate([o["rgb"] for o in all_obs], axis=0)
            ),
        }
        next_obs_data = {
            "state": torch.from_numpy(
                np.concatenate([o["state"] for o in all_next_obs], axis=0)
            ),
            "rgb": torch.from_numpy(
                np.concatenate([o["rgb"] for o in all_next_obs], axis=0)
            ),
        }
    else:
        obs_data = torch.from_numpy(np.concatenate(all_obs, axis=0))
        next_obs_data = torch.from_numpy(np.concatenate(all_next_obs, axis=0))

    dataset = {
        "obs": obs_data,
        "next_obs": next_obs_data,
        "actions": torch.from_numpy(np.concatenate(all_actions, axis=0)),
        "rewards": torch.from_numpy(np.concatenate(all_rewards, axis=0)),
        "dones": torch.from_numpy(np.concatenate(all_dones, axis=0)),
        "terminated": torch.from_numpy(np.concatenate(all_terminated, axis=0)),
        "truncated": torch.from_numpy(np.concatenate(all_truncated, axis=0)),
        "log_probs": torch.from_numpy(np.concatenate(all_log_probs, axis=0)),
        "values": torch.from_numpy(np.concatenate(all_values, axis=0)),
        "env_states": _concat_nested(all_env_states),
    }

    # Save dataset
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(dataset, args.output)
    num_transitions = obs_data["state"].shape[0] if is_visual else obs_data.shape[0]
    print(
        f"Saved {num_transitions} transitions "
        f"from {episodes_collected} episodes to {args.output}"
    )
    if is_visual:
        print(f"  state: {obs_data['state'].shape}, rgb: {obs_data['rgb'].shape}")

    envs.close()
