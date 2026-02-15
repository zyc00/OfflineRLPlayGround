import gymnasium as gym


def make_env(env_name, num_envs, reconfiguration_freq, max_steps, **kwargs):
    return gym.make(
        env_name,
        num_envs=num_envs,
        max_episode_steps=max_steps,
        reconfiguration_freq=reconfiguration_freq,
        **kwargs,
    )
