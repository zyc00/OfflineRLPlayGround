import math
import os
import random
from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
import mani_skill.envs
import numpy as np
import torch
import torch.nn as nn
import tyro
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from tqdm import tqdm

from data.data_collection.ppo import Agent
from data.offline_dataset import OfflineRLDataset
from methods.gae.gae_online import Critic, _make_obs


# ---------------------------------------------------------------------------
# Helpers (adapted from mc.py)
# ---------------------------------------------------------------------------


def _replicate_state(state_dict, n):
    """Replicate a (1, ...) state dict to (n, ...)."""
    if isinstance(state_dict, dict):
        return {k: _replicate_state(v, n) for k, v in state_dict.items()}
    return state_dict.repeat(n, *([1] * (state_dict.ndim - 1)))


# ---------------------------------------------------------------------------
# Batched GAE returns — single forward pass over flat tensors, then
# per-trajectory GAE trace on CPU.  Avoids thousands of tiny GPU ops.
# ---------------------------------------------------------------------------


def _batched_forward(critic, obs, device, batch_size=4096):
    """Run critic on a flat obs tensor in batches, return CPU values."""
    N = obs.shape[0] if not isinstance(obs, dict) else next(iter(obs.values())).shape[0]
    values = torch.zeros(N)
    critic.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            if isinstance(obs, dict):
                batch = {k: v[start:end].to(device) for k, v in obs.items()}
            else:
                batch = obs[start:end].to(device)
            values[start:end] = critic(batch).squeeze(-1).cpu()
    return values


def _compute_gae_returns_batched(
    critic, trajectories, all_obs, all_next_obs, gamma, gae_lambda, device,
):
    """Compute GAE returns with batched value predictions."""
    N = all_obs.shape[0] if not isinstance(all_obs, dict) else next(iter(all_obs.values())).shape[0]

    all_v = _batched_forward(critic, all_obs, device)
    all_v_next = _batched_forward(critic, all_next_obs, device)

    all_returns = []
    for traj in trajectories:
        idx = traj["flat_indices"]
        v = all_v[idx]
        v_next = all_v_next[idx]
        rewards = traj["rewards"]
        terminated = traj["terminated"]
        dones = traj["dones"]

        deltas = rewards + gamma * v_next * (1.0 - terminated) - v

        traj_len = rewards.shape[0]
        advantages = torch.zeros(traj_len)
        lastgaelam = 0.0
        for t in reversed(range(traj_len)):
            not_done = 1.0 - dones[t]
            advantages[t] = lastgaelam = (
                deltas[t] + gamma * gae_lambda * not_done * lastgaelam
            )

        all_returns.append(advantages + v)

    return torch.cat(all_returns, dim=0)


def train_critic_batched(
    all_obs, all_next_obs, trajectories, obs_mode, state_dim,
    sample_rgb, device, args,
):
    """Train a critic using iterative GAE targets with batched forward passes."""
    critic = Critic(obs_mode, state_dim=state_dim, sample_rgb=sample_rgb).to(device)
    optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=args.critic_lr,
        eps=1e-5,
        weight_decay=args.critic_weight_decay,
    )

    N = all_obs.shape[0] if not isinstance(all_obs, dict) else next(iter(all_obs.values())).shape[0]

    for gae_iter in range(1, args.num_gae_iterations + 1):
        gae_returns = _compute_gae_returns_batched(
            critic, trajectories, all_obs, all_next_obs,
            args.gamma, args.gae_lambda, device,
        )

        critic.train()
        total_loss = 0.0
        total_batches = 0
        for _epoch in range(args.critic_update_epochs):
            indices = torch.randperm(N)
            for start in range(0, N, args.critic_batch_size):
                batch_idx = indices[start : start + args.critic_batch_size]
                if isinstance(all_obs, dict):
                    batch_obs = {k: v[batch_idx].to(device) for k, v in all_obs.items()}
                else:
                    batch_obs = all_obs[batch_idx].to(device)
                batch_returns = gae_returns[batch_idx].to(device)

                pred = critic(batch_obs).squeeze(-1)
                loss = 0.5 * ((pred - batch_returns) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        if gae_iter % 10 == 0 or gae_iter == 1:
            print(
                f"  GAE iter {gae_iter}/{args.num_gae_iterations}: "
                f"loss={avg_loss:.6f}, "
                f"returns mean={gae_returns.mean():.4f}, "
                f"std={gae_returns.std():.4f}"
            )

    critic.eval()
    return critic


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
    eval_dataset_path: str = "data/datasets/pickcube_expert_eval.pt"
    """path to the evaluation .pt dataset file"""
    control_mode: str = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
    cuda: bool = True
    """if toggled, cuda will be enabled"""
    max_steps: int = 50
    """maximum number of steps per episode"""
    sample_iters: int = 10
    """number of MC trajectories per eval state (divided across parallel envs)"""
    gamma: float = 0.8
    """discount factor"""
    gae_lambda: float = 0.95
    """lambda for generalized advantage estimation"""
    reward_mode: str = "sparse"
    """reward mode for the environment"""
    dataset_num_envs: int = 16
    """number of parallel envs used when collecting the dataset"""
    obs_mode: Literal["state", "rgb", "state+rgb"] = "state"
    """observation mode for the critic"""
    num_gae_iterations: int = 50
    """number of outer GAE iterations (recompute targets each iteration)"""
    critic_update_epochs: int = 4
    """number of inner epochs per GAE iteration"""
    critic_lr: float = 3e-4
    """learning rate for critic training"""
    critic_batch_size: int = 256
    """minibatch size for critic training"""
    critic_weight_decay: float = 1e-4
    """weight decay (L2 regularization) for critic training"""


def _cache_path(args):
    """Deterministic cache filename based on sampling parameters."""
    base = os.path.dirname(args.eval_dataset_path)
    return os.path.join(
        base,
        f"mc_gae_cache_iters{args.sample_iters}_seed{args.seed}.pt",
    )


def sample_trajectories(args, device):
    """Sample n trajectories per eval state using the environment + PPO policy.

    Returns (sampled_trajectories, eval_indices, N, state_dim).
    """
    sim_backend = "physx_cuda" if device.type == "cuda" else "cpu"
    num_envs = args.num_envs if device.type == "cuda" else 1

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

    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()

    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(action: torch.Tensor):
        return torch.clamp(action.detach(), action_low, action_high)

    _zero_action = torch.zeros(
        num_envs, envs.single_action_space.shape[0], device=device
    )

    _IS_GRASPED_IDX = 18

    def _restore_state_with_contacts(env_state, seed, is_grasped=None):
        envs.reset(seed=seed)
        envs.base_env.set_state_dict(env_state)
        envs.base_env.step(_zero_action)
        envs.base_env.set_state_dict(env_state)
        envs.base_env._elapsed_steps[:] = 0
        obs = envs.base_env.get_obs()
        if is_grasped is not None:
            obs[:, _IS_GRASPED_IDX] = is_grasped
        return obs

    dataset = OfflineRLDataset([args.eval_dataset_path], False, False)
    N = len(dataset)
    state_dim = dataset.state.shape[1]

    num_rounds = math.ceil(args.sample_iters / num_envs)
    total_samples_per_state = num_rounds * num_envs
    print(
        f"MC-GAE: {args.sample_iters} iters = "
        f"{num_rounds} rounds x {num_envs} envs "
        f"({total_samples_per_state} total samples per state)"
    )

    sampled_trajectories = []
    eval_indices = []

    for data in tqdm(dataset, desc="Sampling trajectories"):
        data_idx = data["idx"]
        env_state = _replicate_state(dataset.get_env_state(data_idx), num_envs)
        is_grasped = data["obs"]["state"][_IS_GRASPED_IDX]

        for mc_round in range(num_rounds):
            obs_t = _restore_state_with_contacts(
                env_state, args.seed + mc_round, is_grasped=is_grasped
            )

            action = data["action"].unsqueeze(0).to(device).expand(num_envs, -1)
            action = clip_action(action)
            next_obs, reward, terminated, truncated, info = envs.step(action)

            step_states = [obs_t.clone()]
            step_next_states = [next_obs.clone()]
            step_rewards = [reward.view(-1)]
            step_terminated = [terminated.view(-1).float()]
            step_dones = [(terminated | truncated).view(-1).float()]

            env_done = (terminated | truncated).view(-1)
            first_done_step = torch.full(
                (num_envs,), -1, dtype=torch.long, device=device
            )
            first_done_step[env_done] = 0
            step = 1

            while not env_done.all():
                prev_obs = next_obs.clone()
                action, _, _, _ = agent.get_action_and_value(next_obs)
                action = clip_action(action)
                next_obs, reward, terminated, truncated, info = envs.step(action)

                step_states.append(prev_obs)
                step_next_states.append(next_obs.clone())
                step_rewards.append(reward.view(-1))
                step_terminated.append(terminated.view(-1).float())
                step_dones.append((terminated | truncated).view(-1).float())

                newly_done = (terminated | truncated).view(-1) & ~env_done
                first_done_step[newly_done] = step
                env_done = env_done | newly_done
                step += 1

            all_states = torch.stack(step_states, dim=0)
            all_next_states = torch.stack(step_next_states, dim=0)
            all_rewards = torch.stack(step_rewards, dim=0)
            all_terminated = torch.stack(step_terminated, dim=0)
            all_dones = torch.stack(step_dones, dim=0)

            for env_idx in range(num_envs):
                traj_len = first_done_step[env_idx].item() + 1
                traj = {
                    "states": all_states[:traj_len, env_idx].cpu(),
                    "next_states": all_next_states[:traj_len, env_idx].cpu(),
                    "rewards": all_rewards[:traj_len, env_idx].cpu(),
                    "dones": all_dones[:traj_len, env_idx].cpu(),
                    "terminated": all_terminated[:traj_len, env_idx].cpu(),
                    "flat_indices": None,
                }
                sampled_trajectories.append(traj)
                eval_indices.append(data_idx)

    envs.close()
    del envs, agent
    torch.cuda.empty_cache()

    traj_lens = [t["states"].shape[0] for t in sampled_trajectories]
    print(
        f"Sampled {len(sampled_trajectories)} trajectories "
        f"({total_samples_per_state} per state x {N} states)"
    )
    print(
        f"  Trajectory lengths: min={min(traj_lens)}, max={max(traj_lens)}, "
        f"mean={sum(traj_lens) / len(traj_lens):.1f}"
    )

    # Save cache
    cache = _cache_path(args)
    torch.save(
        {"trajectories": sampled_trajectories, "eval_indices": eval_indices,
         "N": N, "state_dim": state_dim},
        cache,
    )
    print(f"Cached sampled trajectories to {cache}")

    return sampled_trajectories, eval_indices, N, state_dim


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # -------------------------------------------------------------------
    # 1. Sample trajectories (or load from cache)
    # -------------------------------------------------------------------
    cache = _cache_path(args)
    if os.path.exists(cache):
        print(f"Loading cached trajectories from {cache}")
        cached = torch.load(cache, weights_only=False)
        sampled_trajectories = cached["trajectories"]
        eval_indices = cached["eval_indices"]
        N = cached["N"]
        state_dim = cached["state_dim"]
        traj_lens = [t["states"].shape[0] for t in sampled_trajectories]
        print(
            f"  {len(sampled_trajectories)} trajectories, "
            f"lengths: min={min(traj_lens)}, max={max(traj_lens)}, "
            f"mean={sum(traj_lens) / len(traj_lens):.1f}"
        )
    else:
        sampled_trajectories, eval_indices, N, state_dim = sample_trajectories(
            args, device
        )

    # -------------------------------------------------------------------
    # 2. Prepare flat observation tensors and assign flat_indices
    # -------------------------------------------------------------------
    offset = 0
    for traj in sampled_trajectories:
        traj_len = traj["states"].shape[0]
        traj["flat_indices"] = torch.arange(offset, offset + traj_len)
        offset += traj_len

    total_transitions = offset
    print(f"Total sampled transitions: {total_transitions}")

    all_states = torch.cat([t["states"] for t in sampled_trajectories], dim=0)
    all_next_states = torch.cat(
        [t["next_states"] for t in sampled_trajectories], dim=0
    )

    all_rgbs = None
    all_next_rgbs = None
    sample_rgb = None
    if args.obs_mode in ("rgb", "state+rgb"):
        all_rgbs = torch.cat([t["rgbs"] for t in sampled_trajectories], dim=0)
        all_next_rgbs = torch.cat(
            [t["next_rgbs"] for t in sampled_trajectories], dim=0
        )
        sample_rgb = all_rgbs[:1].cpu()

    all_obs = _make_obs(args.obs_mode, all_states, all_rgbs)
    all_next_obs = _make_obs(args.obs_mode, all_next_states, all_next_rgbs)

    # -------------------------------------------------------------------
    # 3. Train critic on sampled trajectories (batched)
    # -------------------------------------------------------------------
    print(
        f"\nTraining critic on {len(sampled_trajectories)} sampled trajectories "
        f"({total_transitions} transitions)..."
    )
    critic = train_critic_batched(
        all_obs,
        all_next_obs,
        sampled_trajectories,
        args.obs_mode,
        state_dim,
        sample_rgb,
        device,
        args,
    )

    # -------------------------------------------------------------------
    # 4. Compute final GAE per trajectory (batched), average per eval transition
    # -------------------------------------------------------------------
    print("Computing GAE per sampled trajectory...")
    all_v = _batched_forward(critic, all_obs, device)
    all_v_next = _batched_forward(critic, all_next_obs, device)

    # Free flat tensors
    del all_states, all_next_states, all_rgbs, all_next_rgbs, all_obs, all_next_obs

    traj_first_advantages = []
    traj_first_values = []

    for traj in sampled_trajectories:
        idx = traj["flat_indices"]
        v = all_v[idx]
        v_next = all_v_next[idx]
        rewards = traj["rewards"]
        terminated = traj["terminated"]
        dones = traj["dones"]

        deltas = rewards + args.gamma * v_next * (1.0 - terminated) - v

        traj_len = rewards.shape[0]
        advantages = torch.zeros(traj_len)
        lastgaelam = 0.0
        for t in reversed(range(traj_len)):
            not_done = 1.0 - dones[t]
            advantages[t] = lastgaelam = (
                deltas[t] + args.gamma * args.gae_lambda * not_done * lastgaelam
            )

        traj_first_advantages.append(advantages[0].item())
        traj_first_values.append(v[0].item())

    # -------------------------------------------------------------------
    # 5. Average across n trajectories per eval transition
    # -------------------------------------------------------------------
    flat_advantages = torch.zeros(N)
    flat_values = torch.zeros(N)
    counts = torch.zeros(N)

    for k, eval_idx in enumerate(eval_indices):
        flat_advantages[eval_idx] += traj_first_advantages[k]
        flat_values[eval_idx] += traj_first_values[k]
        counts[eval_idx] += 1

    flat_advantages /= counts
    flat_values /= counts
    flat_action_values = flat_advantages + flat_values

    # -------------------------------------------------------------------
    # 6. Save results
    # -------------------------------------------------------------------
    results = {
        "values": flat_values,
        "action_values": flat_action_values,
        "advantages": flat_advantages,
    }
    save_path = os.path.join(
        os.path.dirname(args.eval_dataset_path),
        f"mc_gae_estimates_gamma{args.gamma}_lambda{args.gae_lambda}"
        f"_iters{args.sample_iters}.pt",
    )
    torch.save(results, save_path)
    print(f"\nSaved MC-GAE estimates to {save_path}")
    print(
        f"  Values:        mean={flat_values.mean():.4f}, "
        f"std={flat_values.std():.4f}"
    )
    print(
        f"  Action values: mean={flat_action_values.mean():.4f}, "
        f"std={flat_action_values.std():.4f}"
    )
    print(
        f"  Advantages:    mean={flat_advantages.mean():.4f}, "
        f"std={flat_advantages.std():.4f}"
    )
