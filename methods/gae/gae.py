import os
import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import tyro

from data.offline_dataset import OfflineRLDataset


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Critic(nn.Module):
    """Value function network supporting state, rgb, and state+rgb inputs.

    Architecture follows ManiSkill PPO baselines:
      - state:     3x256 Tanh MLP  (matches ppo.py Agent.critic)
      - rgb:       NatureCNN (conv 32→64→64, fc→256) + value head (512→1)
      - state+rgb: NatureCNN rgb encoder (→256) ∥ state encoder (→256)
                   → concat (512) → value head (512→1)
    """

    def __init__(
        self,
        obs_mode: str,
        state_dim: int = 0,
        sample_rgb: torch.Tensor | None = None,
    ):
        super().__init__()
        self.obs_mode = obs_mode

        if obs_mode == "state":
            self.net = nn.Sequential(
                layer_init(nn.Linear(state_dim, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 1)),
            )
        else:
            extractors = {}
            feature_dim = 0

            # RGB encoder (NatureCNN)
            in_channels = sample_rgb.shape[-1]  # (N, H, W, C)
            cnn = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                test_img = sample_rgb[:1].float().permute(0, 3, 1, 2)
                n_flatten = cnn(test_img).shape[1]
            fc = nn.Sequential(nn.Linear(n_flatten, 256), nn.ReLU())
            extractors["rgb"] = nn.Sequential(cnn, fc)
            feature_dim += 256

            # Optional state encoder
            if obs_mode == "state+rgb":
                extractors["state"] = nn.Linear(state_dim, 256)
                feature_dim += 256

            self.extractors = nn.ModuleDict(extractors)
            self.value_head = nn.Sequential(
                layer_init(nn.Linear(feature_dim, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, 1)),
            )

    def forward(self, obs):
        """Forward pass.

        Args:
            obs: For state mode, a (B, state_dim) tensor.
                 For rgb/state+rgb modes, a dict with "rgb" (B,H,W,C uint8)
                 and optionally "state" (B, state_dim) tensors.
        """
        if self.obs_mode == "state":
            return self.net(obs)

        features = []
        for key, extractor in self.extractors.items():
            x = obs[key]
            if key == "rgb":
                x = x.float().permute(0, 3, 1, 2) / 255.0
            features.append(extractor(x))
        return self.value_head(torch.cat(features, dim=1))


# ---------------------------------------------------------------------------
# Observation helpers — build / batch obs dicts based on obs_mode
# ---------------------------------------------------------------------------


def _make_obs(obs_mode, states, rgbs):
    """Build the observation input for a given mode."""
    if obs_mode == "state":
        return states
    if obs_mode == "rgb":
        return {"rgb": rgbs}
    return {"state": states, "rgb": rgbs}


def _batch_obs(obs, indices, device):
    """Index and move an observation (tensor or dict) to device."""
    if isinstance(obs, dict):
        return {k: v[indices].to(device) for k, v in obs.items()}
    return obs[indices].to(device)


# ---------------------------------------------------------------------------
# GAE returns computation
# ---------------------------------------------------------------------------


def _compute_gae_returns(critic, trajectories, obs_mode, gamma, gae_lambda, device):
    """Compute GAE-based returns (advantages + values) for all trajectories.

    Returns a flat tensor of returns, concatenated in trajectory order.
    """
    all_returns = []
    critic.eval()
    for traj in trajectories:
        rewards = traj["rewards"].to(device)
        terminated = traj["terminated"].to(device)
        dones = traj["dones"].to(device)
        traj_len = rewards.shape[0]

        obs = _make_obs(obs_mode, traj["states"], traj["rgbs"])
        next_obs = _make_obs(obs_mode, traj["next_states"], traj["next_rgbs"])

        with torch.no_grad():
            if isinstance(obs, dict):
                obs_dev = {k: v.to(device) for k, v in obs.items()}
                next_obs_dev = {k: v.to(device) for k, v in next_obs.items()}
            else:
                obs_dev = obs.to(device)
                next_obs_dev = next_obs.to(device)

            v = critic(obs_dev).squeeze(-1)
            v_next = critic(next_obs_dev).squeeze(-1)

        deltas = rewards + gamma * v_next * (1.0 - terminated) - v

        advantages = torch.zeros(traj_len, device=device)
        lastgaelam = 0.0
        for t in reversed(range(traj_len)):
            not_done = 1.0 - dones[t]
            advantages[t] = lastgaelam = (
                deltas[t] + gamma * gae_lambda * not_done * lastgaelam
            )

        all_returns.append((advantages + v).cpu())

    return torch.cat(all_returns, dim=0)


# ---------------------------------------------------------------------------
# Critic training
# ---------------------------------------------------------------------------


def train_critic(
    obs,
    trajectories: list,
    obs_mode: str,
    state_dim: int,
    sample_rgb: torch.Tensor | None,
    device: torch.device,
    args: "Args",
) -> Critic:
    """Train a critic using iterative GAE targets (like PPO value updates).

    Each outer iteration recomputes GAE returns with the current critic,
    then trains the critic on those frozen targets for K inner epochs
    (mirroring PPO's update_epochs for the value function).
    """
    critic = Critic(obs_mode, state_dim=state_dim, sample_rgb=sample_rgb).to(device)
    optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=args.critic_lr,
        eps=1e-5,
        weight_decay=args.critic_weight_decay,
    )

    N = sum(t["states"].shape[0] for t in trajectories)

    for gae_iter in range(1, args.num_gae_iterations + 1):
        # Recompute GAE returns with current critic (frozen targets)
        gae_returns = _compute_gae_returns(
            critic, trajectories, obs_mode,
            args.gamma, args.gae_lambda, device,
        )

        # Train critic on GAE returns for K epochs (like PPO update_epochs)
        critic.train()
        total_loss = 0.0
        total_batches = 0
        for _epoch in range(args.critic_update_epochs):
            indices = torch.randperm(N)
            for start in range(0, N, args.critic_batch_size):
                batch_idx = indices[start : start + args.critic_batch_size]
                batch_obs = _batch_obs(obs, batch_idx, device)
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
    seed: int = 1
    """random seed"""
    train_dataset_path: str = "data/datasets/pickcube_expert.pt"
    """path to the training .pt dataset file (used to fit the critic)"""
    eval_dataset_path: str = "data/datasets/pickcube_expert_eval.pt"
    """path to the evaluation .pt dataset file (GAE computed on this)"""
    cuda: bool = True
    """if toggled, cuda will be enabled"""
    gamma: float = 0.8
    """discount factor"""
    gae_lambda: float = 0.95
    """lambda for generalized advantage estimation"""
    dataset_num_envs: int = 16
    """number of parallel envs used when collecting the datasets"""
    obs_mode: Literal["state", "rgb", "state+rgb"] = "state"
    """observation mode for the critic: state, rgb, or state+rgb"""
    num_gae_iterations: int = 50
    """number of outer GAE iterations (recompute targets each iteration)"""
    critic_update_epochs: int = 4
    """number of inner epochs per GAE iteration (like PPO update_epochs)"""
    critic_lr: float = 3e-4
    """learning rate for critic training"""
    critic_batch_size: int = 256
    """minibatch size for critic training"""
    critic_weight_decay: float = 1e-4
    """weight decay (L2 regularization) for critic training"""


if __name__ == "__main__":
    args = tyro.cli(Args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ---------------------------------------------------------------
    # 1. Load both datasets and extract trajectories
    # ---------------------------------------------------------------
    print(f"Loading training dataset: {args.train_dataset_path}")
    train_dataset = OfflineRLDataset([args.train_dataset_path], False, False)
    state_dim = train_dataset.state.shape[1]

    print(
        f"Extracting training trajectories "
        f"(num_envs={args.dataset_num_envs}, gamma={args.gamma})"
    )
    train_trajectories = train_dataset.extract_trajectories(
        num_envs=args.dataset_num_envs, gamma=args.gamma
    )
    traj_lens = [t["states"].shape[0] for t in train_trajectories]
    print(
        f"  Found {len(train_trajectories)} trajectories, "
        f"lengths: min={min(traj_lens)}, max={max(traj_lens)}, "
        f"mean={sum(traj_lens) / len(traj_lens):.1f}"
    )

    print(f"\nLoading eval dataset: {args.eval_dataset_path}")
    eval_dataset = OfflineRLDataset([args.eval_dataset_path], False, False)
    N = len(eval_dataset)

    print(
        f"Extracting eval trajectories "
        f"(num_envs={args.dataset_num_envs}, gamma={args.gamma})"
    )
    eval_trajectories = eval_dataset.extract_trajectories(
        num_envs=args.dataset_num_envs, gamma=args.gamma
    )
    traj_lens = [t["states"].shape[0] for t in eval_trajectories]
    print(
        f"  Found {len(eval_trajectories)} trajectories, "
        f"lengths: min={min(traj_lens)}, max={max(traj_lens)}, "
        f"mean={sum(traj_lens) / len(traj_lens):.1f}"
    )

    # ---------------------------------------------------------------
    # 2. Train critic with iterative GAE targets
    # ---------------------------------------------------------------
    all_trajectories = train_trajectories + eval_trajectories
    train_size = sum(t["states"].shape[0] for t in all_trajectories)
    print(
        f"\nCritic fitting: {len(all_trajectories)} trajectories, "
        f"{train_size} transitions"
    )

    all_states = torch.cat([t["states"] for t in all_trajectories], dim=0)

    all_rgbs = None
    sample_rgb = None
    if args.obs_mode in ("rgb", "state+rgb"):
        all_rgbs = torch.cat([t["rgbs"] for t in all_trajectories], dim=0)
        sample_rgb = all_rgbs[:1].cpu()

    all_obs = _make_obs(args.obs_mode, all_states, all_rgbs)

    print(f"Training critic with GAE supervision (obs_mode={args.obs_mode})...")
    critic = train_critic(
        all_obs,
        all_trajectories,
        args.obs_mode,
        state_dim,
        sample_rgb,
        device,
        args,
    )

    # Free combined training data
    del train_dataset, train_trajectories, all_states
    del all_rgbs, all_obs, all_trajectories

    # ---------------------------------------------------------------
    # 3. Compute GAE on the evaluation dataset
    # ---------------------------------------------------------------
    print("Computing GAE per trajectory...")
    flat_values = torch.zeros(N)
    flat_advantages = torch.zeros(N)

    for traj in eval_trajectories:
        rewards = traj["rewards"].to(device)
        terminated = traj["terminated"].to(device)
        dones = traj["dones"].to(device)
        flat_idx = traj["flat_indices"]
        traj_len = rewards.shape[0]

        obs = _make_obs(args.obs_mode, traj["states"], traj["rgbs"])
        next_obs = _make_obs(args.obs_mode, traj["next_states"], traj["next_rgbs"])

        with torch.no_grad():
            if isinstance(obs, dict):
                obs_dev = {k: v.to(device) for k, v in obs.items()}
                next_obs_dev = {k: v.to(device) for k, v in next_obs.items()}
            else:
                obs_dev = obs.to(device)
                next_obs_dev = next_obs.to(device)

            v = critic(obs_dev).squeeze(-1)
            v_next = critic(next_obs_dev).squeeze(-1)

        # TD residuals
        deltas = rewards + args.gamma * v_next * (1.0 - terminated) - v

        # GAE backward pass within this trajectory
        advantages = torch.zeros(traj_len, device=device)
        lastgaelam = 0.0
        for t in reversed(range(traj_len)):
            not_done = 1.0 - dones[t]
            advantages[t] = lastgaelam = (
                deltas[t] + args.gamma * args.gae_lambda * not_done * lastgaelam
            )

        flat_values[flat_idx] = v.cpu()
        flat_advantages[flat_idx] = advantages.cpu()

    flat_action_values = flat_advantages + flat_values

    # Save results
    results = {
        "values": flat_values,
        "action_values": flat_action_values,
        "advantages": flat_advantages,
    }
    save_path = os.path.join(
        os.path.dirname(args.eval_dataset_path),
        f"gae_estimates_gamma{args.gamma}_lambda{args.gae_lambda}.pt",
    )
    torch.save(results, save_path)
    print(f"\nSaved GAE estimates to {save_path}")
    print(
        f"  Values:        mean={flat_values.mean():.4f}, std={flat_values.std():.4f}"
    )
    print(
        f"  Action values: mean={flat_action_values.mean():.4f}, "
        f"std={flat_action_values.std():.4f}"
    )
    print(
        f"  Advantages:    mean={flat_advantages.mean():.4f}, "
        f"std={flat_advantages.std():.4f}"
    )
