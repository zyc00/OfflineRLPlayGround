"""
Dataset for DPPO: loads ManiSkill H5 demos into DPPO-compatible format.

Action normalization: bbox (per-dim min/max → [-1, 1])
Obs normalization: min-max (per-dim min/max → [-1, 1]), matching DPPO paper
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from diffusion_policy.utils import load_demo_dataset


class DPPODataset(Dataset):
    """Loads ManiSkill H5 demos into (action_chunk, cond) pairs for diffusion training."""

    def __init__(
        self,
        data_path,
        horizon_steps,
        cond_steps=1,
        num_traj=None,
        device="cpu",
        no_obs_norm=False,
        no_action_norm=False,
    ):
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps
        self.device = device
        self.no_obs_norm = no_obs_norm
        self.no_action_norm = no_action_norm

        # Load trajectories (not concatenated)
        trajectories = load_demo_dataset(
            data_path,
            keys=["observations", "actions"],
            num_traj=num_traj,
            concat=False,
        )

        self.obs_list = [torch.from_numpy(o).float() for o in trajectories["observations"]]
        self.act_list = [torch.from_numpy(a).float() for a in trajectories["actions"]]
        self.num_traj = len(self.act_list)

        # Build index: (traj_idx, start_within_traj)
        # Match dp_train slicing: pad_before = cond_steps-1, pad_after = horizon_steps-cond_steps
        pad_before = self.cond_steps - 1
        pad_after = self.horizon_steps - self.cond_steps
        self.slices = []
        for traj_idx in range(self.num_traj):
            L = self.act_list[traj_idx].shape[0]
            for start in range(-pad_before, L - self.horizon_steps + pad_after):
                self.slices.append((traj_idx, start))

        # Compute normalization stats
        all_obs = torch.cat(self.obs_list, dim=0)
        all_act = torch.cat(self.act_list, dim=0)

        # Obs: min-max normalization (per-dim min/max → [-1, 1]), matching DPPO paper
        self.obs_min = all_obs.min(dim=0).values
        self.obs_max = all_obs.max(dim=0).values
        # Small margin to avoid exact boundary values
        obs_range = self.obs_max - self.obs_min
        obs_margin = 0.01 * obs_range.clamp(min=1e-6)
        self.obs_min = self.obs_min - obs_margin
        self.obs_max = self.obs_max + obs_margin

        # Keep z-score stats for backward compatibility (checkpoint loading)
        self.obs_mean = all_obs.mean(dim=0)
        self.obs_std = all_obs.std(dim=0).clamp(min=1e-6)

        # Actions: bbox normalization (per-dim min/max → [-1, 1])
        self.action_min = all_act.min(dim=0).values
        self.action_max = all_act.max(dim=0).values
        # Small margin to avoid exact boundary values
        action_range = self.action_max - self.action_min
        margin = 0.01 * action_range.clamp(min=1e-6)
        self.action_min = self.action_min - margin
        self.action_max = self.action_max + margin

        self.obs_dim = all_obs.shape[-1]
        self.action_dim = all_act.shape[-1]

        print(f"DPPODataset: {self.num_traj} trajs, {len(self.slices)} samples, "
              f"obs_dim={self.obs_dim}, action_dim={self.action_dim}")
        print(f"  obs_min:  {self.obs_min.numpy().round(3)}")
        print(f"  obs_max:  {self.obs_max.numpy().round(3)}")
        print(f"  action_min: {self.action_min.numpy().round(3)}")
        print(f"  action_max: {self.action_max.numpy().round(3)}")

    def normalize_obs(self, obs):
        """Min-max normalize obs: [obs_min, obs_max] → [-1, 1]"""
        lo = self.obs_min.to(obs.device)
        hi = self.obs_max.to(obs.device)
        return (obs - lo) / (hi - lo + 1e-8) * 2.0 - 1.0

    def unnormalize_obs(self, obs):
        """Min-max denormalize obs: [-1, 1] → [obs_min, obs_max]"""
        lo = self.obs_min.to(obs.device)
        hi = self.obs_max.to(obs.device)
        return (obs + 1.0) / 2.0 * (hi - lo) + lo

    def normalize_action(self, action):
        """Bbox normalize: [action_min, action_max] → [-1, 1]"""
        lo = self.action_min.to(action.device)
        hi = self.action_max.to(action.device)
        return (action - lo) / (hi - lo) * 2.0 - 1.0

    def unnormalize_action(self, action):
        """Bbox denormalize: [-1, 1] → [action_min, action_max]"""
        lo = self.action_min.to(action.device)
        hi = self.action_max.to(action.device)
        return (action + 1.0) / 2.0 * (hi - lo) + lo

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        traj_idx, start = self.slices[idx]
        obs_traj = self.obs_list[traj_idx]
        act_traj = self.act_list[traj_idx]
        L = act_traj.shape[0]

        # Get obs conditioning: cond_steps frames starting at `start` (matching dp_train)
        # dp_train convention: obs = [obs[start], obs[start+1], ...obs[start+cond_steps-1]]
        obs_seq = obs_traj[max(0, start):start + self.cond_steps]
        # Pad before if start < 0
        if start < 0:
            obs_seq = torch.cat([obs_seq[0:1].expand(-start, -1), obs_seq], dim=0)

        # Get action chunk: horizon_steps frames starting at `start`
        act_start = max(0, start)
        act_end = start + self.horizon_steps
        act_seq = act_traj[act_start:min(act_end, L)]

        # Pad before (start < 0): repeat first action
        if start < 0:
            n_pad_before = -start
            act_seq = torch.cat([act_seq[0:1].expand(n_pad_before, -1), act_seq], dim=0)

        # Pad after (act_end > L): zeros for arm + last gripper (matching dp_train)
        if act_end > L:
            n_pad = act_end - L
            gripper_val = act_traj[-1, -1:]  # (1,)
            pad_arm = torch.zeros(self.action_dim - 1)
            pad_action = torch.cat([pad_arm, gripper_val], dim=0)  # (action_dim,)
            act_seq = torch.cat([act_seq, pad_action.unsqueeze(0).expand(n_pad, -1)], dim=0)

        # Normalize (optional)
        if not self.no_obs_norm:
            obs_seq = self.normalize_obs(obs_seq)
        if not self.no_action_norm:
            act_seq = self.normalize_action(act_seq)

        return {
            "actions": act_seq,  # (horizon_steps, action_dim)
            "cond": {"state": obs_seq},  # (cond_steps, obs_dim)
        }
