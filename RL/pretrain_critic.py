"""Pretrain a critic (value function) using expert policy rollouts.

Collects trajectories with an expert policy in sparse reward env,
computes MC returns, and trains a critic network via supervised regression.

The pretrained critic can then be loaded into ppo_finetune.py for
sample-efficient finetuning (no critic warmup needed).

Usage:
  python -m RL.pretrain_critic
  python -m RL.pretrain_critic --num_rollouts 50 --epochs 200
"""

import os
import random
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from data.data_collection.ppo import Agent, layer_init


@dataclass
class Args:
    expert_checkpoint: str = "runs/pickcube_ppo/ckpt_301.pt"
    """expert policy checkpoint for data collection"""
    env_id: str = "PickCube-v1"
    num_envs: int = 512
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"
    max_episode_steps: int = 50
    gamma: float = 0.8
    seed: int = 1
    cuda: bool = True

    # Data collection
    num_rollouts: int = 20
    """number of full rollouts (each = num_steps * num_envs transitions)"""
    num_steps: int = 50

    # Training
    epochs: int = 100
    batch_size: int = 4096
    learning_rate: float = 1e-3

    # Output
    output_path: Optional[str] = None


if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.output_path is None:
        args.output_path = "runs/pretrained_critic_sparse.pt"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ── Environment ───────────────────────────────────────────────────
    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="physx_cuda" if device.type == "cuda" else "cpu",
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
    )
    envs = gym.make(args.env_id, num_envs=args.num_envs, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(
        envs, args.num_envs,
        ignore_terminations=False,
        record_metrics=True,
    )

    # ── Load expert ───────────────────────────────────────────────────
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.expert_checkpoint, map_location=device))
    agent.eval()
    print(f"Loaded expert: {args.expert_checkpoint}")

    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(a):
        return torch.clamp(a.detach(), action_low, action_high)

    # ── Collect data ──────────────────────────────────────────────────
    print(f"Collecting {args.num_rollouts} rollouts "
          f"({args.num_envs} envs × {args.num_steps} steps each)...")

    all_obs = []
    all_returns = []

    next_obs, _ = envs.reset(seed=args.seed)

    for rollout_idx in range(args.num_rollouts):
        obs_buf = torch.zeros(
            (args.num_steps, args.num_envs) + envs.single_observation_space.shape,
            device=device,
        )
        reward_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
        done_buf = torch.zeros((args.num_steps, args.num_envs), device=device)

        for step in range(args.num_steps):
            obs_buf[step] = next_obs
            with torch.no_grad():
                action = agent.get_action(next_obs, deterministic=False)
            next_obs, reward, term, trunc, _ = envs.step(clip_action(action))
            reward_buf[step] = reward.view(-1)
            done_buf[step] = (term | trunc).float()

        # MC returns (backward pass, reset at episode boundaries)
        returns = torch.zeros((args.num_steps, args.num_envs), device=device)
        ret = torch.zeros(args.num_envs, device=device)
        for t in reversed(range(args.num_steps)):
            ret = reward_buf[t] + args.gamma * ret * (1.0 - done_buf[t])
            returns[t] = ret

        all_obs.append(obs_buf.reshape(-1, obs_buf.shape[-1]).cpu())
        all_returns.append(returns.reshape(-1).cpu())

        print(f"  Rollout {rollout_idx+1}/{args.num_rollouts}: "
              f"mean_return={returns.mean():.4f}, "
              f"nonzero={(returns > 0).float().mean():.1%}")

    envs.close()

    # ── Prepare dataset ───────────────────────────────────────────────
    all_obs = torch.cat(all_obs, dim=0)
    all_returns = torch.cat(all_returns, dim=0)
    N = len(all_obs)
    print(f"\nTotal data: {N} state-return pairs")
    print(f"Return stats: mean={all_returns.mean():.4f}, "
          f"std={all_returns.std():.4f}, "
          f"min={all_returns.min():.4f}, max={all_returns.max():.4f}")

    # Train/val split
    perm = torch.randperm(N)
    val_size = min(N // 10, 10000)
    train_obs = all_obs[perm[val_size:]].to(device)
    train_returns = all_returns[perm[val_size:]].to(device)
    val_obs = all_obs[perm[:val_size]].to(device)
    val_returns = all_returns[perm[:val_size]].to(device)

    # ── Train critic ──────────────────────────────────────────────────
    obs_dim = all_obs.shape[1]
    critic = nn.Sequential(
        layer_init(nn.Linear(obs_dim, 256)),
        nn.Tanh(),
        layer_init(nn.Linear(256, 256)),
        nn.Tanh(),
        layer_init(nn.Linear(256, 256)),
        nn.Tanh(),
        layer_init(nn.Linear(256, 1)),
    ).to(device)

    optimizer = optim.Adam(critic.parameters(), lr=args.learning_rate)

    print(f"\nTraining critic: {len(train_obs)} train, {len(val_obs)} val")

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        critic.train()
        perm_train = torch.randperm(len(train_obs), device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_obs), args.batch_size):
            idx = perm_train[i : i + args.batch_size]
            pred = critic(train_obs[idx]).squeeze(-1)
            loss = ((pred - train_returns[idx]) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        critic.eval()
        with torch.no_grad():
            val_pred = critic(val_obs).squeeze(-1)
            val_loss = ((val_pred - val_returns) ** 2).mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in critic.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}: "
                  f"train={epoch_loss / n_batches:.6f}, val={val_loss:.6f}")

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.save(best_state, args.output_path)
    print(f"\nSaved pretrained critic to {args.output_path}")
    print(f"Best val MSE: {best_val_loss:.6f}")
