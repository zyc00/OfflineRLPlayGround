"""
Evaluation for image-based diffusion policy using GPU-parallel ManiSkill envs.
"""

import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import mani_skill.envs  # noqa
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def _extract_obs(obs, camera_names, state_keys, img_size, device):
    """Extract RGB images and proprioceptive state from ManiSkill dict obs.

    Args:
        obs: dict observation from ManiSkill (rgbd mode)
        camera_names: list of camera names
        state_keys: list of state keys (e.g. ["agent/qpos", "agent/qvel"])
        img_size: target image resolution
        device: torch device

    Returns:
        rgb: (B, num_cam, 3, H, W) uint8 on device
        state: (B, state_dim) float32 on device
    """
    B = None
    # Extract state
    state_parts = []
    for key in state_keys:
        parts = key.split("/")
        val = obs
        for p in parts:
            val = val[p]
        if val.ndim == 1:
            val = val.unsqueeze(-1)
        val = val.float().to(device)
        state_parts.append(val)
        if B is None:
            B = val.shape[0]
    state = torch.cat(state_parts, dim=-1)  # (B, state_dim)

    # Extract RGB images
    cam_imgs = []
    for cam in camera_names:
        rgb = obs["sensor_data"][cam]["rgb"]  # (B, H, W, 3) uint8
        rgb = rgb.permute(0, 3, 1, 2).to(device)  # (B, 3, H, W)
        if rgb.shape[-1] != img_size:
            rgb = F.interpolate(rgb.float(), size=(img_size, img_size),
                                mode="bilinear", align_corners=False).to(torch.uint8)
        cam_imgs.append(rgb)

    if len(cam_imgs) == 1:
        rgb = cam_imgs[0]  # (B, 3, H, W)
    else:
        rgb = torch.stack(cam_imgs, dim=1)  # (B, num_cam, 3, H, W)

    return rgb, state


@torch.no_grad()
def evaluate_image_gpu(
    n_episodes,
    model,
    device,
    act_steps,
    cond_steps=1,
    env_id="PickCube-v1",
    num_envs=100,
    control_mode="pd_ee_delta_pos",
    max_episode_steps=100,
    camera_names=("base_camera",),
    state_keys=("agent/qpos", "agent/qvel"),
    img_size=128,
    state_min=None,
    state_max=None,
    action_min=None,
    action_max=None,
    no_state_norm=False,
    no_action_norm=False,
    act_offset=0,
):
    """Evaluate image-based diffusion policy with GPU-parallel ManiSkill envs."""
    model.eval()

    envs = gym.make(
        env_id,
        num_envs=num_envs,
        obs_mode="rgbd",
        render_mode="rgb_array",
        control_mode=control_mode,
        max_episode_steps=max_episode_steps,
        sim_backend="physx_cuda",
        reconfiguration_freq=1,
    )
    envs = ManiSkillVectorEnv(envs, num_envs, ignore_terminations=True, record_metrics=True)

    obs, _ = envs.reset()

    if not no_action_norm:
        a_lo = action_min.to(device)
        a_hi = action_max.to(device)
    if not no_state_norm:
        s_lo = state_min.to(device)
        s_hi = state_max.to(device)

    # Initialize history buffers
    rgb0, state0 = _extract_obs(obs, camera_names, state_keys, img_size, device)
    num_cam = len(camera_names)

    # RGB history: (B, cond_steps, [num_cam,] 3, H, W)
    if num_cam == 1:
        rgb_history = rgb0.unsqueeze(1).repeat(1, cond_steps, 1, 1, 1)  # (B, T, 3, H, W)
    else:
        rgb_history = rgb0.unsqueeze(1).repeat(1, cond_steps, 1, 1, 1, 1)  # (B, T, num_cam, 3, H, W)

    # State history: (B, cond_steps, state_dim)
    state_history = state0.unsqueeze(1).repeat(1, cond_steps, 1)

    success_at_end_list = []
    success_once_list = []

    for _ in range(max_episode_steps):
        # Normalize state
        if no_state_norm:
            state_cond = state_history
        else:
            state_cond = (state_history - s_lo) / (s_hi - s_lo + 1e-8) * 2.0 - 1.0

        cond = {"state": state_cond, "rgb": rgb_history}

        samples = model(cond, deterministic=True)
        action_chunk = samples.trajectories

        if not no_action_norm:
            action_chunk = (action_chunk + 1.0) / 2.0 * (a_hi - a_lo) + a_lo

        for a_idx in range(act_offset, min(act_offset + act_steps, action_chunk.shape[1])):
            action = action_chunk[:, a_idx]
            obs_new, reward, terminated, truncated, info = envs.step(action)

            rgb_new, state_new = _extract_obs(obs_new, camera_names, state_keys, img_size, device)

            # Update histories
            state_history = torch.cat([state_history[:, 1:], state_new.unsqueeze(1)], dim=1)
            if num_cam == 1:
                rgb_history = torch.cat([rgb_history[:, 1:], rgb_new.unsqueeze(1)], dim=1)
            else:
                rgb_history = torch.cat([rgb_history[:, 1:], rgb_new.unsqueeze(1)], dim=1)

        if truncated.any():
            fi = info.get("final_info", {})
            ep = fi.get("episode", {})
            if "success_at_end" in ep:
                mask = info.get("_final_info", truncated)
                sa = ep["success_at_end"][mask].float().cpu().numpy()
                so = ep["success_once"][mask].float().cpu().numpy()
                success_at_end_list.append(sa)
                success_once_list.append(so)
            break

    envs.close()

    if success_at_end_list:
        sa_all = np.concatenate(success_at_end_list)
        so_all = np.concatenate(success_once_list)
    else:
        sa_all = np.array([0.0])
        so_all = np.array([0.0])

    return {
        "success_at_end": sa_all.mean(),
        "success_once": so_all.mean(),
        "n_episodes": len(sa_all),
    }
