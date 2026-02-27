"""
Running reward scaler — ported from irom-princeton/dppo.

Normalizes rewards by the running std of discounted returns, then clips.
"""

import numpy as np
import torch


class RunningRewardScaler:
    """Normalizes rewards by running std of discounted returns."""

    def __init__(self, gamma=0.99, clip_max=10.0, eps=1e-8):
        self.gamma = gamma
        self.clip_max = clip_max
        self.eps = eps
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 0

    def update_and_scale(self, rewards, dones):
        """
        Compute discounted returns, update running stats, scale rewards.

        Args:
            rewards: (n_steps, n_envs) tensor
            dones: (n_steps, n_envs) tensor (0/1)

        Returns:
            scaled_rewards: (n_steps, n_envs) tensor
        """
        n_steps, n_envs = rewards.shape
        device = rewards.device

        # Compute backward discounted returns
        discounted = torch.zeros_like(rewards)
        running_sum = torch.zeros(n_envs, device=device)
        for t in reversed(range(n_steps)):
            running_sum = rewards[t] + self.gamma * (1.0 - dones[t]) * running_sum
            discounted[t] = running_sum

        # Update running mean/variance of discounted returns
        flat = discounted.cpu().numpy().flatten()
        batch_mean = flat.mean()
        batch_var = flat.var()
        batch_count = len(flat)

        # Welford's online update
        delta = batch_mean - self.running_mean
        total = self.count + batch_count
        self.running_mean += delta * batch_count / max(total, 1)
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / max(total, 1)
        self.running_var = m2 / max(total, 1)
        self.count = total

        # Scale rewards by running std
        std = np.sqrt(self.running_var + self.eps)
        scaled = rewards / std

        # Clip
        if self.clip_max is not None:
            scaled = scaled.clamp(-self.clip_max, self.clip_max)

        return scaled
