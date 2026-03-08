"""
Dataset for DPPO: loads ManiSkill H5 demos into DPPO-compatible format.

Action normalization: bbox (per-dim min/max → [-1, 1])
Obs normalization: min-max (per-dim min/max → [-1, 1]), matching DPPO paper
"""

import h5py
import numpy as np
import torch
import torch.nn.functional as F
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


class DPPOImageDataset(Dataset):
    """Loads ManiSkill rgbd H5 demos for image-based diffusion policy training.

    H5 structure (from `replay_trajectory -o rgbd`):
        traj_N/obs/sensor_data/{camera}/rgb: (T, H, W, 3) uint8
        traj_N/obs/agent/qpos: (T, 9) float32
        traj_N/obs/agent/qvel: (T, 9) float32
        traj_N/obs/extra/tcp_pose: (T, 7) float32
        traj_N/actions: (T-1, action_dim) float32
    """

    def __init__(
        self,
        data_path,
        horizon_steps,
        cond_steps=1,
        img_cond_steps=1,
        num_traj=None,
        camera_names=("base_camera",),
        state_keys=("agent/qpos", "agent/qvel"),
        img_size=128,
        no_action_norm=False,
        no_state_norm=False,
    ):
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps
        self.img_cond_steps = img_cond_steps
        self.camera_names = list(camera_names)
        self.num_img = len(camera_names)
        self.img_size = img_size
        self.no_action_norm = no_action_norm
        self.no_state_norm = no_state_norm

        # Load from H5
        self.rgb_list = []  # list of (T, num_cameras, C, H, W) uint8
        self.state_list = []  # list of (T, state_dim) float32
        self.act_list = []  # list of (T-1, action_dim) float32

        with h5py.File(data_path, "r") as f:
            traj_keys = sorted(
                [k for k in f.keys() if k.startswith("traj_")],
                key=lambda k: int(k.split("_")[1]),
            )
            if num_traj is not None:
                traj_keys = traj_keys[:num_traj]

            for tk in traj_keys:
                traj = f[tk]
                # Actions
                actions = torch.from_numpy(traj["actions"][:]).float()
                self.act_list.append(actions)

                # RGB images: (T, H, W, 3) -> (T, 3, H, W) per camera, stack cameras
                cam_imgs = []
                for cam in self.camera_names:
                    rgb = traj[f"obs/sensor_data/{cam}/rgb"][:]  # (T, H, W, 3) uint8
                    rgb = torch.from_numpy(rgb).permute(0, 3, 1, 2)  # (T, 3, H, W)
                    cam_imgs.append(rgb)
                # Stack cameras: (T, num_cam, 3, H, W)
                if len(cam_imgs) == 1:
                    stacked = cam_imgs[0].unsqueeze(1)
                else:
                    stacked = torch.stack(cam_imgs, dim=1)
                self.rgb_list.append(stacked)

                # Proprioceptive state
                state_parts = []
                for sk in state_keys:
                    data = traj[f"obs/{sk}"][:]
                    if data.ndim == 1:
                        data = data[:, None]
                    state_parts.append(torch.from_numpy(data).float())
                state = torch.cat(state_parts, dim=-1)
                self.state_list.append(state)

        self.num_traj = len(self.act_list)
        self.state_dim = self.state_list[0].shape[-1]
        self.action_dim = self.act_list[0].shape[-1]
        self.obs_dim = self.state_dim  # For compatibility

        # Build sample index
        pad_before = self.cond_steps - 1
        pad_after = self.horizon_steps - self.cond_steps
        self.slices = []
        for traj_idx in range(self.num_traj):
            L = self.act_list[traj_idx].shape[0]
            for start in range(-pad_before, L - self.horizon_steps + pad_after):
                self.slices.append((traj_idx, start))

        # Normalization stats (state and action only, not images)
        all_state = torch.cat(self.state_list, dim=0)
        all_act = torch.cat(self.act_list, dim=0)

        # State: min-max normalization
        self.state_min = all_state.min(dim=0).values
        self.state_max = all_state.max(dim=0).values
        state_range = self.state_max - self.state_min
        state_margin = 0.01 * state_range.clamp(min=1e-6)
        self.state_min = self.state_min - state_margin
        self.state_max = self.state_max + state_margin

        # For compatibility with state-only code paths
        self.obs_min = self.state_min
        self.obs_max = self.state_max

        # Action: bbox normalization
        self.action_min = all_act.min(dim=0).values
        self.action_max = all_act.max(dim=0).values
        action_range = self.action_max - self.action_min
        margin = 0.01 * action_range.clamp(min=1e-6)
        self.action_min = self.action_min - margin
        self.action_max = self.action_max + margin

        # Original image size
        orig_h = self.rgb_list[0].shape[-2]
        self._need_resize = (img_size != orig_h)

        print(f"DPPOImageDataset: {self.num_traj} trajs, {len(self.slices)} samples")
        print(f"  state_dim={self.state_dim}, action_dim={self.action_dim}")
        print(f"  cameras={self.camera_names}, img_size={img_size} (orig {orig_h})")
        print(f"  state_min: {self.state_min.numpy().round(3)}")
        print(f"  state_max: {self.state_max.numpy().round(3)}")

    def normalize_state(self, state):
        lo, hi = self.state_min.to(state.device), self.state_max.to(state.device)
        return (state - lo) / (hi - lo + 1e-8) * 2.0 - 1.0

    def normalize_action(self, action):
        lo, hi = self.action_min.to(action.device), self.action_max.to(action.device)
        return (action - lo) / (hi - lo) * 2.0 - 1.0

    def unnormalize_action(self, action):
        lo, hi = self.action_min.to(action.device), self.action_max.to(action.device)
        return (action + 1.0) / 2.0 * (hi - lo) + lo

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        traj_idx, start = self.slices[idx]
        act_traj = self.act_list[traj_idx]
        state_traj = self.state_list[traj_idx]
        rgb_traj = self.rgb_list[traj_idx]
        L = act_traj.shape[0]

        # -- Obs conditioning (cond_steps frames starting at `start`) --
        state_seq = state_traj[max(0, start):start + self.cond_steps]
        rgb_seq = rgb_traj[max(0, start):start + self.cond_steps]
        if start < 0:
            n_pad = -start
            state_seq = torch.cat([state_seq[0:1].expand(n_pad, -1), state_seq], dim=0)
            rgb_seq = torch.cat([rgb_seq[0:1].expand(n_pad, *rgb_seq.shape[1:]), rgb_seq], dim=0)

        # -- Action chunk (horizon_steps frames starting at `start`) --
        act_start = max(0, start)
        act_end = start + self.horizon_steps
        act_seq = act_traj[act_start:min(act_end, L)]
        if start < 0:
            n_pad = -start
            act_seq = torch.cat([act_seq[0:1].expand(n_pad, -1), act_seq], dim=0)
        if act_end > L:
            n_pad = act_end - L
            gripper_val = act_traj[-1, -1:]
            pad_arm = torch.zeros(self.action_dim - 1)
            pad_action = torch.cat([pad_arm, gripper_val], dim=0)
            act_seq = torch.cat([act_seq, pad_action.unsqueeze(0).expand(n_pad, -1)], dim=0)

        # Normalize
        if not self.no_state_norm:
            state_seq = self.normalize_state(state_seq)
        if not self.no_action_norm:
            act_seq = self.normalize_action(act_seq)

        # RGB: (cond_steps, num_cam, 3, H, W) -> for single camera: (cond_steps, 3, H, W)
        if self.num_img == 1:
            rgb_seq = rgb_seq.squeeze(1)  # (cond_steps, 3, H, W)
        # Resize if needed
        if self._need_resize:
            shape = rgb_seq.shape
            flat = rgb_seq.reshape(-1, *shape[-3:]).float()
            flat = F.interpolate(flat, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
            rgb_seq = flat.to(torch.uint8).reshape(*shape[:-2], self.img_size, self.img_size)

        return {
            "actions": act_seq,
            "cond": {
                "state": state_seq,  # (cond_steps, state_dim)
                "rgb": rgb_seq,      # (cond_steps, [num_cam,] 3, H, W) uint8
            },
        }
