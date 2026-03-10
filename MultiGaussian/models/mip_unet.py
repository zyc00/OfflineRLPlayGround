"""MIP with UNet backbone (same UNet as Diffusion Policy).

Uses DP's ConditionalUnet1D as the network, but trained with MIP's
2-pass loss and inference instead of iterative diffusion denoising.

This isolates the effect of network architecture (UNet vs MLP) from
the algorithm (2-pass vs 100-step diffusion).
"""

import torch
import torch.nn as nn
from diffusion_policy.conditional_unet1d import ConditionalUnet1D


class MIPUNetPolicy(nn.Module):
    """MIP policy using DP's 1D UNet backbone.

    UNet forward: (noisy_action, timestep, obs_cond) -> predicted_action
    MIP training: 2 losses at t=0 and t=t*
    MIP inference: 2-pass deterministic
    """

    def __init__(self, input_dim, action_dim, cond_steps=1, horizon_steps=1,
                 t_star=0.9, down_dims=[64, 128, 256],
                 diffusion_step_embed_dim=64, kernel_size=5, n_groups=8):
        super().__init__()
        assert 0.0 < t_star <= 1.0
        self.cond_steps = cond_steps
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.t_star = t_star

        global_cond_dim = input_dim * cond_steps

        self.unet = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )

    def _make_timestep(self, t_val, B, device):
        """Convert scalar t to integer timestep for UNet's sinusoidal embedding."""
        # UNet expects integer-like timesteps (0..999 in DP).
        # Map MIP's t in [0, 1] to [0, 999].
        t_int = int(round(t_val * 999)) if isinstance(t_val, float) else t_val
        return torch.full((B,), t_int, device=device, dtype=torch.long)

    def forward(self, obs, I_t, t):
        """Single forward: UNet(I_t, timestep=t, obs_cond).

        obs: (B, cond_steps, input_dim) or (B, input_dim)
        I_t: (B, horizon_steps, action_dim) or (B, action_dim)
        t: float in [0, 1]
        """
        B = obs.shape[0]

        # Flatten obs for global conditioning
        obs_flat = obs.reshape(B, -1)

        # Ensure I_t is (B, T, action_dim)
        if I_t.dim() == 2:
            I_t = I_t.unsqueeze(1)

        timestep = self._make_timestep(t, B, obs.device)
        out = self.unet(I_t, timestep, global_cond=obs_flat)

        if self.horizon_steps > 1:
            return out  # (B, T, action_dim)
        return out.squeeze(1)  # (B, action_dim)

    def compute_loss(self, obs, actions):
        """MIP 2-pass loss with official scaling."""
        t_star = self.t_star

        # Step 1: predict from zeros at t=0
        I_0 = torch.zeros_like(actions)
        pred_t0 = self.forward(obs, I_0, 0.0)
        diff0 = (pred_t0 - actions) / t_star
        loss_t0 = torch.mean(torch.sum(diff0 * diff0, dim=-1))

        # Step 2: predict from noised action at t=t*
        noise = torch.randn_like(actions)
        I_tstar = actions + (1.0 - t_star) * noise
        pred_tstar = self.forward(obs, I_tstar, t_star)
        diff1 = (pred_tstar - actions) / (1.0 - t_star)
        loss_tstar = torch.mean(torch.sum(diff1 * diff1, dim=-1))

        total = loss_t0 + loss_tstar
        return total, loss_t0, loss_tstar

    @torch.no_grad()
    def predict(self, obs):
        """2-pass deterministic inference."""
        B = obs.shape[0]
        if self.horizon_steps > 1:
            act_shape = (B, self.horizon_steps, self.action_dim)
        else:
            act_shape = (B, self.action_dim)

        I_0 = torch.zeros(act_shape, device=obs.device, dtype=obs.dtype)
        a_hat = self.forward(obs, I_0, 0.0)
        a_hat = self.forward(obs, a_hat, self.t_star)
        return a_hat
