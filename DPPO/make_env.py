"""
Environment factory for DPPO training and evaluation.

Supports both CPU (SyncVectorEnv) and GPU (ManiSkill vectorized) backends.
GPU envs run all environments in parallel on GPU — much faster for training.
CPU envs are used for accurate evaluation (cuda can underestimate by ~20%).

GPU env pattern follows ManiSkill official PPO example:
  envs = gym.make(env_id, num_envs=N, sim_backend="gpu", ...)
  envs = ManiSkillVectorEnv(envs, N, ignore_terminations=False, record_metrics=True)
"""

import mani_skill.envs  # noqa: F401  register envs
import gymnasium as gym
import numpy as np
from mani_skill.utils.wrappers import CPUGymWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


class SeedPoolWrapper(gym.Wrapper):
    """Override reset() to sample from a pre-computed pool of seeds."""
    def __init__(self, env, seed_pool):
        super().__init__(env)
        self.seed_pool = np.asarray(seed_pool)

    def reset(self, **kwargs):
        kwargs["seed"] = int(np.random.choice(self.seed_pool))
        return super().reset(**kwargs)


def make_eval_envs(
    env_id,
    num_envs,
    sim_backend="physx_cpu",
    control_mode="pd_joint_delta_pos",
    max_episode_steps=100,
    seed=0,
):
    """Create CPU-based eval envs with ignore_terminations=True."""

    def cpu_make_env(env_id, seed):
        def thunk():
            env = gym.make(
                env_id,
                obs_mode="state",
                render_mode="rgb_array",
                control_mode=control_mode,
                max_episode_steps=max_episode_steps,
                reconfiguration_freq=1,
            )
            env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk

    envs = gym.vector.SyncVectorEnv(
        [cpu_make_env(env_id, seed + i) for i in range(num_envs)]
    )
    return envs


def make_train_envs(
    env_id,
    num_envs,
    sim_backend="gpu",
    control_mode="pd_ee_delta_pos",
    max_episode_steps=100,
    seed=0,
    seed_pool=None,
):
    """Create envs for RL rollout.

    sim_backend="gpu": ManiSkill GPU vectorized env (fast, returns CUDA tensors).
                       Uses ManiSkillVectorEnv wrapper for proper partial reset
                       and episode metrics (matching official PPO example).
    sim_backend="cpu": CPU SyncVectorEnv (slow, returns numpy arrays).
    """
    if sim_backend == "gpu":
        env = gym.make(
            env_id,
            num_envs=num_envs,
            obs_mode="state",
            control_mode=control_mode,
            max_episode_steps=max_episode_steps,
            sim_backend="gpu",
            reward_mode="sparse",
        )
        # ManiSkillVectorEnv handles partial reset + episode metrics
        # ignore_terminations=False → auto-reset on success (partial reset)
        env = ManiSkillVectorEnv(env, num_envs,
                                 ignore_terminations=False,
                                 record_metrics=True)
        return env
    else:
        def cpu_make_env(env_id, seed):
            def thunk():
                env = gym.make(
                    env_id,
                    obs_mode="state",
                    render_mode="rgb_array",
                    control_mode=control_mode,
                    max_episode_steps=max_episode_steps,
                    reconfiguration_freq=1,
                    reward_mode="sparse",
                )
                if seed_pool is not None:
                    env = SeedPoolWrapper(env, seed_pool)
                env = CPUGymWrapper(env, ignore_terminations=False, record_metrics=True)
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env
            return thunk

        envs = gym.vector.SyncVectorEnv(
            [cpu_make_env(env_id, seed + i) for i in range(num_envs)]
        )
        return envs
