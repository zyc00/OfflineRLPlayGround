"""
Evaluation loop for diffusion policy with action chunking.
"""

import torch
import numpy as np
import gymnasium as gym
import mani_skill.envs  # noqa: register envs
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


@torch.no_grad()
def evaluate_gpu(
    n_episodes,
    model,
    device,
    act_steps,
    obs_min,
    obs_max,
    action_min,
    action_max,
    cond_steps=1,
    env_id="PickCube-v1",
    num_envs=100,
    control_mode="pd_joint_delta_pos",
    max_episode_steps=100,
    no_obs_norm=False,
    no_action_norm=False,
    act_offset=0,
    zero_qvel=False,
):
    """
    Evaluate diffusion policy using GPU-parallel envs (ManiSkillVectorEnv).

    Creates envs internally, runs n_episodes, closes, returns metrics.
    """
    model.eval()

    # Create GPU vectorized env
    envs = gym.make(
        env_id,
        num_envs=num_envs,
        obs_mode="state",
        render_mode="rgb_array",
        control_mode=control_mode,
        max_episode_steps=max_episode_steps,
        sim_backend="physx_cuda",
        reconfiguration_freq=1,
    )
    envs = ManiSkillVectorEnv(envs, num_envs, ignore_terminations=True, record_metrics=True)

    obs, _ = envs.reset()
    obs = obs.float().to(device)  # (num_envs, obs_dim)

    obs_dim = obs.shape[-1]
    obs_history = obs.unsqueeze(1).repeat(1, cond_steps, 1)

    if not no_action_norm:
        a_lo = action_min.to(device)
        a_hi = action_max.to(device)

    if not no_obs_norm:
        o_lo = obs_min.to(device)
        o_hi = obs_max.to(device)

    success_at_end_list = []
    success_once_list = []
    episodes_done = 0

    for _ in range(max_episode_steps):
        if zero_qvel:
            obs_history[..., 9:18] = 0.0

        # Normalize obs (min-max → [-1, 1])
        if no_obs_norm:
            cond = {"state": obs_history}
        else:
            obs_norm = (obs_history - o_lo) / (o_hi - o_lo + 1e-8) * 2.0 - 1.0
            cond = {"state": obs_norm}

        # Diffusion inference
        samples = model(cond, deterministic=True)
        action_chunk = samples.trajectories  # (num_envs, horizon_steps, action_dim)

        # Denormalize actions
        if not no_action_norm:
            action_chunk = (action_chunk + 1.0) / 2.0 * (a_hi - a_lo) + a_lo
        # else: raw actions in [-1, 1], clipped by diffusion model

        # Execute act_steps (with optional offset for UNet pred_horizon > act_horizon)
        for a_idx in range(act_offset, min(act_offset + act_steps, action_chunk.shape[1])):
            action = action_chunk[:, a_idx]
            obs_new, reward, terminated, truncated, info = envs.step(action)
            obs_new = obs_new.float().to(device)
            obs_history = torch.cat([obs_history[:, 1:], obs_new.unsqueeze(1)], dim=1)

        # Check truncation — with ignore_terminations=True, all envs truncate together
        if truncated.any():
            # ManiSkillVectorEnv: info["final_info"]["episode"] is a dict of tensors
            fi = info.get("final_info", {})
            ep = fi.get("episode", {})
            if "success_at_end" in ep:
                mask = info.get("_final_info", truncated)
                sa = ep["success_at_end"][mask].float().cpu().numpy()
                so = ep["success_once"][mask].float().cpu().numpy()
                success_at_end_list.append(sa)
                success_once_list.append(so)
                episodes_done += len(sa)
            break  # All envs truncate together

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
