"""Behavior Cloning from ManiSkill motion planning demonstrations.

Usage:
    # First convert demos (one-time):
    python -m mani_skill.trajectory.replay_trajectory \
      --traj-path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.h5 \
      -o state -c pd_joint_delta_pos --save-traj --num-envs 1

    # Then train:
    python -u -m RL.bc_train \
      --demo_path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.state.pd_joint_delta_pos.h5
"""

import json
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)

    def get_action(self, x, deterministic=False):
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        return torch.distributions.Normal(action_mean, action_std).sample()

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


@dataclass
class Args:
    demo_path: str = ""
    """path to converted h5 demo file"""
    env_id: str = "StackCube-v1"
    """environment id"""
    num_demos: int = -1
    """number of demos to use (-1 = all)"""
    batch_size: int = 1024
    """training batch size"""
    lr: float = 3e-4
    """learning rate"""
    num_epochs: int = 1000
    """number of training epochs"""
    eval_freq: int = 50
    """evaluate every N epochs"""
    num_eval_envs: int = 128
    """number of parallel eval environments"""
    max_episode_steps: int = 200
    """max episode steps for eval (MP demos can be long)"""
    seed: int = 1
    """random seed"""
    output_dir: str = "runs/bc_stackcube"
    """output directory for checkpoints and logs"""
    logstd_init: float = -0.5
    """initial log std for the policy"""
    obs_noise: float = 0.0
    """Gaussian noise std added to obs during training (0=off)"""


def load_demos(path, num_demos=-1):
    """Load obs and actions from h5 or npz file."""
    if path.endswith(".npz"):
        data = np.load(path)
        all_obs = data["obs"]
        all_actions = data["actions"]
        print(f"Loaded npz: {len(all_obs)} transitions")
        print(f"  Obs shape: {all_obs.shape}, Actions shape: {all_actions.shape}")
        return torch.tensor(all_obs, dtype=torch.float32), torch.tensor(all_actions, dtype=torch.float32)

    # h5 path
    json_path = path.replace(".h5", ".json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            meta = json.load(f)
        n_episodes = len(meta["episodes"])
    else:
        with h5py.File(path, "r") as f:
            n_episodes = len([k for k in f.keys() if k.startswith("traj_")])

    if num_demos > 0:
        n_episodes = min(n_episodes, num_demos)

    all_obs = []
    all_actions = []
    ep_lengths = []

    with h5py.File(path, "r") as f:
        for i in range(n_episodes):
            traj = f[f"traj_{i}"]
            obs = traj["obs"][:]       # (T+1, obs_dim) — includes final obs
            actions = traj["actions"][:] # (T, act_dim)
            T = actions.shape[0]
            all_obs.append(obs[:T])     # drop final obs, align with actions
            all_actions.append(actions)
            ep_lengths.append(T)

    all_obs = np.concatenate(all_obs, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)

    print(f"Loaded {n_episodes} demos, {len(all_obs)} transitions")
    print(f"  Episode lengths: min={min(ep_lengths)}, max={max(ep_lengths)}, "
          f"mean={np.mean(ep_lengths):.1f}, median={np.median(ep_lengths):.0f}")
    print(f"  Obs shape: {all_obs.shape}, Actions shape: {all_actions.shape}")

    return torch.tensor(all_obs, dtype=torch.float32), torch.tensor(all_actions, dtype=torch.float32)


def evaluate(agent, env_id, num_envs, max_episode_steps, device):
    """Rollout in env with deterministic policy, return success rate."""
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda")
    eval_envs = gym.make(
        env_id,
        num_envs=num_envs,
        control_mode="pd_joint_delta_pos",
        max_episode_steps=max_episode_steps,
        **env_kwargs,
    )
    if isinstance(eval_envs.action_space, gym.spaces.Dict):
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    eval_envs = ManiSkillVectorEnv(
        eval_envs, num_envs, ignore_terminations=False, record_metrics=True,
    )

    obs, _ = eval_envs.reset()
    successes = []
    num_episodes = 0
    for _ in range(max_episode_steps):
        with torch.no_grad():
            action = agent.get_action(obs, deterministic=True)
        obs, rew, terminations, truncations, infos = eval_envs.step(action)
        if "final_info" in infos:
            mask = infos["_final_info"]
            num_episodes += mask.sum().item()
            successes.append(infos["final_info"]["success"][mask].float())

    eval_envs.close()

    if successes:
        success_rate = torch.cat(successes).mean().item()
    else:
        success_rate = 0.0
    return success_rate, num_episodes


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    all_obs, all_actions = load_demos(args.demo_path, args.num_demos)
    obs_dim = all_obs.shape[1]
    act_dim = all_actions.shape[1]
    print(f"obs_dim={obs_dim}, act_dim={act_dim}")

    # Create agent
    agent = Agent(obs_dim, act_dim).to(device)
    agent.actor_logstd.data.fill_(args.logstd_init)
    optimizer = optim.Adam(agent.actor_mean.parameters(), lr=args.lr)

    # DataLoader
    dataset = TensorDataset(all_obs, all_actions)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Logging
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(args.output_dir)

    best_sr = -1.0
    best_epoch = -1

    print(f"\nTraining BC for {args.num_epochs} epochs, {len(dataloader)} batches/epoch")
    print(f"Output: {args.output_dir}\n")

    for epoch in range(1, args.num_epochs + 1):
        agent.train()
        epoch_loss = 0.0
        n_batches = 0
        for obs_batch, act_batch in dataloader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            if args.obs_noise > 0:
                obs_batch = obs_batch + torch.randn_like(obs_batch) * args.obs_noise

            pred = agent.actor_mean(obs_batch)
            loss = F.mse_loss(pred, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        writer.add_scalar("train/loss", avg_loss, epoch)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | loss={avg_loss:.6f}")

        # Evaluate
        if epoch % args.eval_freq == 0 or epoch == 1:
            agent.eval()
            sr, n_ep = evaluate(agent, args.env_id, args.num_eval_envs,
                                args.max_episode_steps, device)
            writer.add_scalar("eval/success_rate", sr, epoch)
            print(f"  EVAL epoch={epoch}: success_rate={sr:.3f} ({n_ep} episodes)")

            # Save checkpoint
            ckpt_path = os.path.join(args.output_dir, f"ckpt_{epoch}.pt")
            torch.save(agent.state_dict(), ckpt_path)

            if sr > best_sr:
                best_sr = sr
                best_epoch = epoch
                best_path = os.path.join(args.output_dir, "best.pt")
                torch.save(agent.state_dict(), best_path)
                print(f"  NEW BEST: {sr:.3f} at epoch {epoch} -> {best_path}")

    # Save final
    final_path = os.path.join(args.output_dir, "final.pt")
    torch.save(agent.state_dict(), final_path)
    print(f"\nTraining complete. Best SR={best_sr:.3f} at epoch {best_epoch}")
    print(f"Best checkpoint: {os.path.join(args.output_dir, 'best.pt')}")
    writer.close()
