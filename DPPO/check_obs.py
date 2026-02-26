"""
Quick check: compare obs format from env vs demo data.
Also checks if the model produces reasonable actions for env obs.
"""
import torch
import numpy as np
import gymnasium as gym
import mani_skill.envs  # noqa
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from DPPO.dataset import DPPODataset
import os


def main():
    demo_path = os.path.expanduser(
        "~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5"
    )
    device = torch.device("cuda")

    # Load dataset
    dataset = DPPODataset(
        data_path=demo_path, horizon_steps=4, cond_steps=1,
    )

    # Get first demo obs (raw, not normalized)
    demo_obs_0 = dataset.obs_list[0][0]  # First obs of first trajectory
    print(f"Demo obs shape: {demo_obs_0.shape}")
    print(f"Demo obs[0] (first 10 dims): {demo_obs_0[:10].numpy().round(4)}")
    print(f"Demo obs[0] (last 10 dims): {demo_obs_0[-10:].numpy().round(4)}")
    print(f"Demo obs range: [{demo_obs_0.min():.4f}, {demo_obs_0.max():.4f}]")

    # Get env obs (GPU)
    envs = gym.make(
        "PickCube-v1",
        num_envs=1,
        obs_mode="state",
        render_mode="rgb_array",
        control_mode="pd_joint_delta_pos",
        max_episode_steps=100,
        sim_backend="physx_cuda",
    )
    envs = ManiSkillVectorEnv(envs, 1, ignore_terminations=True, record_metrics=True)
    obs, _ = envs.reset()
    obs = obs.float()

    print(f"\nEnv obs type: {type(obs)}")
    print(f"Env obs shape: {obs.shape}")
    print(f"Env obs[0] (first 10 dims): {obs[0, :10].cpu().numpy().round(4)}")
    print(f"Env obs[0] (last 10 dims): {obs[0, -10:].cpu().numpy().round(4)}")
    print(f"Env obs range: [{obs.min():.4f}, {obs.max():.4f}]")

    # Compare obs statistics
    all_demo_obs = torch.cat(dataset.obs_list, dim=0)
    print(f"\nDemo obs stats (all trajs):")
    print(f"  mean (first 10): {all_demo_obs[:, :10].mean(0).numpy().round(4)}")
    print(f"  std (first 10): {all_demo_obs[:, :10].std(0).numpy().round(4)}")
    print(f"  mean (last 10): {all_demo_obs[:, -10:].mean(0).numpy().round(4)}")
    print(f"  std (last 10): {all_demo_obs[:, -10:].std(0).numpy().round(4)}")

    # Check: are the first 10 obs values in similar ranges?
    print(f"\nObs dimension labels (guessed):")
    print(f"  Dims 0-8: likely agent qpos (joint positions)")
    print(f"  Dims 9-17: likely agent qvel (joint velocities)")
    print(f"  Dims 18+: likely object/goal state")

    # Step with zero action to check obs format
    zero_action = torch.zeros(1, 8, device=obs.device)
    obs2, _, _, _, _ = envs.step(zero_action)
    obs2 = obs2.float()
    print(f"\nObs after zero action shape: {obs2.shape}")
    print(f"Obs change (first 10): {(obs2[0, :10] - obs[0, :10]).cpu().numpy().round(6)}")

    envs.close()


if __name__ == "__main__":
    main()
