"""
Environment factory for DPPO training and evaluation.

Uses CPUGymWrapper from ManiSkill to properly convert ManiSkill's
batched tensor outputs to numpy scalars for gymnasium compatibility.
"""

import mani_skill.envs  # noqa: F401  register envs
import gymnasium as gym
from mani_skill.utils.wrappers import CPUGymWrapper


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
    sim_backend="physx_cpu",
    control_mode="pd_joint_delta_pos",
    max_episode_steps=100,
    seed=0,
):
    """Create CPU envs for RL rollout with ignore_terminations=False."""

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
            env = CPUGymWrapper(env, ignore_terminations=False, record_metrics=True)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk

    envs = gym.vector.SyncVectorEnv(
        [cpu_make_env(env_id, seed + i) for i in range(num_envs)]
    )
    return envs
