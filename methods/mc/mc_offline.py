"""Offline MC: Q(s,a) = G_t directly from trajectory returns.

No critic, no rollouts — just the discounted sum of observed rewards
from each transition to the end of its episode.
"""

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import tyro

from data.offline_dataset import OfflineRLDataset


@dataclass
class Args:
    seed: int = 1
    """random seed"""
    dataset_path: str = "data/datasets/pickcube_expert_eval.pt"
    """path to the .pt dataset file"""
    gamma: float = 0.8
    """discount factor"""
    dataset_num_envs: int = 16
    """number of parallel envs used when collecting the dataset"""


if __name__ == "__main__":
    args = tyro.cli(Args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---------------------------------------------------------------
    # Load dataset and extract trajectories (computes MC returns)
    # ---------------------------------------------------------------
    print(f"Loading dataset: {args.dataset_path}")
    dataset = OfflineRLDataset([args.dataset_path], False, False)
    N = len(dataset)

    print(
        f"Extracting trajectories "
        f"(num_envs={args.dataset_num_envs}, gamma={args.gamma})"
    )
    trajectories = dataset.extract_trajectories(
        num_envs=args.dataset_num_envs, gamma=args.gamma
    )
    traj_lens = [t["states"].shape[0] for t in trajectories]
    print(
        f"  Found {len(trajectories)} trajectories, "
        f"lengths: min={min(traj_lens)}, max={max(traj_lens)}, "
        f"mean={sum(traj_lens) / len(traj_lens):.1f}"
    )

    # ---------------------------------------------------------------
    # Scatter trajectory returns back to flat ordering
    # ---------------------------------------------------------------
    flat_action_values = torch.zeros(N)

    for traj in trajectories:
        flat_action_values[traj["flat_indices"]] = traj["mc_returns"]

    # Save results (same format as mc.py and gae.py)
    results = {
        "action_values": flat_action_values,
    }
    save_path = os.path.join(
        os.path.dirname(args.dataset_path),
        f"mc_offline_estimates_gamma{args.gamma}.pt",
    )
    torch.save(results, save_path)
    print(f"\nSaved offline MC estimates to {save_path}")
    print(
        f"  Action values: mean={flat_action_values.mean():.4f}, "
        f"std={flat_action_values.std():.4f}"
    )
