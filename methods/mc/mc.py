import math
import os
import random
from dataclasses import dataclass

import gymnasium as gym
import mani_skill.envs
import numpy as np
import torch
import tyro
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from tqdm import tqdm

from data.data_collection.ppo import Agent
from data.offline_dataset import OfflineRLDataset


def _replicate_state(state_dict, n):
    """Replicate a (1, ...) state dict to (n, ...)."""
    if isinstance(state_dict, dict):
        return {k: _replicate_state(v, n) for k, v in state_dict.items()}
    return state_dict.repeat(n, *([1] * (state_dict.ndim - 1)))


@dataclass
class Args:
    checkpoint: str = "runs/pickcube_ppo/ckpt_301.pt"
    """path to a pretrained PPO checkpoint file"""
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    num_envs: int = 1
    """number of parallel environments for MC rollouts"""
    seed: int = 1
    """random seed"""
    dataset_path: str = "data/datasets/pickcube_expert_eval.pt"
    """output path for the .pt dataset file"""
    control_mode: str = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
    cuda: bool = True
    """if toggled, cuda will be enabled"""
    max_steps: int = 50
    """maximum number of steps per episode"""
    sample_iters: int = 10
    """total MC estimation iterations (divided across parallel envs)"""
    gamma: float = 0.8
    """discount factor for MC estimation"""
    reward_mode: str = "sparse"
    """reward mode for the environment (e.g., 'sparse', 'dense')"""
    num_random_actions: int = 3
    """number of random actions to sample per state for Q(s, a_random) estimation"""


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    sim_backend = "physx_cuda" if device.type == "cuda" else "cpu"
    num_envs = args.num_envs if device.type == "cuda" else 1

    # Environment setup (matches ppo.py but simpler)
    env_kwargs = dict(
        obs_mode="state",
        render_mode="sensors",
        sim_backend=sim_backend,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        max_episode_steps=args.max_steps,
    )
    envs = gym.make(args.env_id, num_envs=num_envs, **env_kwargs)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(
        envs, num_envs, ignore_terminations=False, record_metrics=True
    )

    # Load agent
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()

    # Action clipping (matches ppo.py)
    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(action: torch.Tensor):
        return torch.clamp(action.detach(), action_low, action_high)

    # Pre-allocate zero action for contact warm-up
    _zero_action = torch.zeros(
        num_envs, envs.single_action_space.shape[0], device=device
    )

    _IS_GRASPED_IDX = 18  # index of is_grasped in flattened obs

    def _restore_state_with_contacts(env_state, seed, is_grasped=None):
        """Reset env, set state, and warm up physics contacts.

        After set_state_dict() PhysX contact forces are stale from the
        previous reset(). Stepping once with a zero action forces PhysX to
        recompute contacts, then we restore the exact poses/velocities.

        If *is_grasped* is provided (scalar tensor from the dataset), it
        overrides the contact-based flag in the returned observation.
        PhysX contact detection is history-dependent, so borderline grasp
        states may get the wrong binary flag after a state restore.
        """
        envs.reset(seed=seed)
        envs.base_env.set_state_dict(env_state)
        envs.base_env.step(_zero_action)  # populate contacts
        envs.base_env.set_state_dict(env_state)  # restore exact state
        envs.base_env._elapsed_steps[:] = 0
        obs = envs.base_env.get_obs()
        if is_grasped is not None:
            obs[:, _IS_GRASPED_IDX] = is_grasped
        return obs

    # Load dataset
    dataset = OfflineRLDataset([args.dataset_path], False, False)

    num_rounds = math.ceil(args.sample_iters / num_envs)
    print(
        f"MC estimation: {args.sample_iters} iters = "
        f"{num_rounds} rounds x {num_envs} envs "
        f"({num_rounds * num_envs} total samples)"
    )

    values = []
    action_values = []
    advantages = []
    random_action_values_all = []
    random_actions_all = []
    for data in tqdm(dataset):
        env_state = _replicate_state(dataset.get_env_state(data["idx"]), num_envs)

        # --- V(s): roll out from state following policy ---
        return_to_gos = []
        is_grasped = data["obs"]["state"][_IS_GRASPED_IDX]
        for mc_round in range(num_rounds):
            next_obs = _restore_state_with_contacts(
                env_state,
                args.seed + mc_round,
                is_grasped=is_grasped,
            )

            env_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
            first_done_step = torch.full(
                (num_envs,), -1, dtype=torch.long, device=device
            )
            all_rewards = []
            step = 0

            while not env_done.all():
                action, _, _, _ = agent.get_action_and_value(next_obs)
                # action = agent.get_action(next_obs, deterministic=True)
                action = clip_action(action)
                next_obs, reward, terminated, truncated, info = envs.step(action)
                all_rewards.append(reward.view(-1))
                newly_done = (terminated | truncated).view(-1) & ~env_done
                first_done_step[newly_done] = step
                env_done = env_done | newly_done
                step += 1

            # Backward pass: only accumulate up to each env's termination
            all_rewards = torch.stack(all_rewards, dim=0)  # (T, num_envs)
            return_to_go = torch.zeros(num_envs, device=device)
            for t in reversed(range(step)):
                active = t <= first_done_step
                return_to_go = torch.where(
                    active,
                    all_rewards[t] + args.gamma * return_to_go,
                    return_to_go,
                )
            return_to_gos.extend(return_to_go.cpu().tolist())

        values.append(np.mean(return_to_gos))

        # --- Q(s,a): take dataset action first, then follow policy ---
        return_to_gos = []
        for mc_round in range(num_rounds):
            _restore_state_with_contacts(
                env_state,
                args.seed + mc_round,
                is_grasped=is_grasped,
            )

            env_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
            first_done_step = torch.full(
                (num_envs,), -1, dtype=torch.long, device=device
            )
            all_rewards = []
            step = 0

            # First step: dataset action for all envs
            action = data["action"].unsqueeze(0).to(device).expand(num_envs, -1)
            action = clip_action(action)
            next_obs, reward, terminated, truncated, info = envs.step(action)
            all_rewards.append(reward.view(-1))
            newly_done = (terminated | truncated).view(-1)
            first_done_step[newly_done] = step
            env_done = newly_done.clone()
            step += 1

            # Subsequent steps: follow policy (deterministic, matching V(s))
            while not env_done.all():
                action, _, _, _ = agent.get_action_and_value(next_obs)
                # action = agent.get_action(next_obs, deterministic=True)
                action = clip_action(action)
                next_obs, reward, terminated, truncated, info = envs.step(action)
                all_rewards.append(reward.view(-1))
                newly_done = (terminated | truncated).view(-1) & ~env_done
                first_done_step[newly_done] = step
                env_done = env_done | newly_done
                step += 1

            # Backward pass: only accumulate up to each env's termination
            all_rewards = torch.stack(all_rewards, dim=0)  # (T, num_envs)
            return_to_go = torch.zeros(num_envs, device=device)
            for t in reversed(range(step)):
                active = t <= first_done_step
                return_to_go = torch.where(
                    active,
                    all_rewards[t] + args.gamma * return_to_go,
                    return_to_go,
                )
            return_to_gos.extend(return_to_go.cpu().tolist())

        action_values.append(np.mean(return_to_gos))

        # Advantage = Q(s, a) - V(s)
        advantages.append(action_values[-1] - values[-1])

        # --- Q(s, a_random): sample random actions, MC estimate each ---
        state_random_qvals = []
        state_random_acts = []
        for _ in range(args.num_random_actions):
            # Sample one random action for this round
            raw_action = envs.single_action_space.sample()
            rand_action = torch.from_numpy(raw_action).to(device)
            rand_action = clip_action(rand_action)
            state_random_acts.append(rand_action.cpu())

            # MC estimate Q(s, a_random) with num_rounds rollouts
            return_to_gos = []
            for mc_round in range(num_rounds):
                _restore_state_with_contacts(
                    env_state,
                    args.seed + mc_round,
                    is_grasped=is_grasped,
                )

                env_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
                first_done_step = torch.full(
                    (num_envs,), -1, dtype=torch.long, device=device
                )
                all_rewards = []
                step = 0

                # First step: same random action for all envs
                action = rand_action.unsqueeze(0).expand(num_envs, -1)
                next_obs, reward, terminated, truncated, info = envs.step(action)
                all_rewards.append(reward.view(-1))
                newly_done = (terminated | truncated).view(-1)
                first_done_step[newly_done] = step
                env_done = newly_done.clone()
                step += 1

                # Subsequent steps: follow policy
                while not env_done.all():
                    action, _, _, _ = agent.get_action_and_value(next_obs)
                    action = clip_action(action)
                    next_obs, reward, terminated, truncated, info = envs.step(action)
                    all_rewards.append(reward.view(-1))
                    newly_done = (terminated | truncated).view(-1) & ~env_done
                    first_done_step[newly_done] = step
                    env_done = env_done | newly_done
                    step += 1

                # Backward pass
                all_rewards = torch.stack(all_rewards, dim=0)
                return_to_go = torch.zeros(num_envs, device=device)
                for t in reversed(range(step)):
                    active = t <= first_done_step
                    return_to_go = torch.where(
                        active,
                        all_rewards[t] + args.gamma * return_to_go,
                        return_to_go,
                    )
                return_to_gos.extend(return_to_go.cpu().tolist())

            state_random_qvals.append(np.mean(return_to_gos))

        random_action_values_all.append(state_random_qvals)
        random_actions_all.append(torch.stack(state_random_acts, dim=0))

    # Aggregate random action results
    random_action_values = torch.tensor(random_action_values_all)  # (N, num_random_actions)
    random_actions = torch.stack(random_actions_all, dim=0)  # (N, num_random_actions, action_dim)
    mean_random_action_values = random_action_values.mean(dim=1)  # (N,)

    # Save results
    results = {
        "values": torch.tensor(values),
        "action_values": torch.tensor(action_values),
        "advantages": torch.tensor(advantages),
        "random_action_values": random_action_values,
        "random_actions": random_actions,
        "mean_random_action_values": mean_random_action_values,
    }
    save_path = os.path.join(
        os.path.dirname(args.dataset_path),
        f"mc_estimates_gamma{args.gamma}_iters{args.sample_iters}.pt",
    )
    torch.save(results, save_path)
    print(f"Saved MC estimates to {save_path}")
    print(f"  Values:        mean={np.mean(values):.4f}, std={np.std(values):.4f}")
    print(
        f"  Action values: mean={np.mean(action_values):.4f}, std={np.std(action_values):.4f}"
    )
    print(
        f"  Advantages:    mean={np.mean(advantages):.4f}, std={np.std(advantages):.4f}"
    )
    print(
        f"  Random action values (mean over {args.num_random_actions} actions): "
        f"mean={mean_random_action_values.mean():.4f}, std={mean_random_action_values.std():.4f}"
    )
    dataset_av = torch.tensor(action_values)
    dataset_better = (dataset_av > mean_random_action_values).float().mean()
    print(f"  P(Q_dataset > mean Q_random): {dataset_better:.4f}")
    frac_better = (dataset_av.unsqueeze(1) > random_action_values).float().mean(dim=1)
    print(
        f"  Per-state frac(Q_dataset > Q_random_k): "
        f"mean={frac_better.mean():.4f}, std={frac_better.std():.4f}"
    )

    # Close envs
    envs.close()
