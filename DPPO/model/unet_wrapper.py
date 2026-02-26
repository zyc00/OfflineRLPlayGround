"""
Wrapper to make ConditionalUnet1D compatible with DiffusionModel's network interface.

DiffusionModel calls network(x, time, cond=cond) where:
- x: (B, Ta, Da) noisy actions
- time: (B,) diffusion timestep
- cond: dict with "state": (B, To, Do)

ConditionalUnet1D expects forward(sample, timestep, global_cond) where:
- sample: (B, T, input_dim) noisy actions
- timestep: (B,) or int
- global_cond: (B, global_cond_dim) flattened obs
"""

import torch
import torch.nn as nn
from diffusion_policy.conditional_unet1d import ConditionalUnet1D


class DiffusionUNet(nn.Module):
    """Wraps ConditionalUnet1D to match DiffusionModel's network interface."""

    def __init__(
        self,
        action_dim,
        horizon_steps,
        cond_dim,
        diffusion_step_embed_dim=64,
        down_dims=[64, 128, 256],
        kernel_size=5,
        n_groups=8,
    ):
        super().__init__()
        self.unet = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )

    def forward(self, x, time, cond, **kwargs):
        """
        x: (B, Ta, Da) noisy actions
        time: (B,) or (B, 1) diffusion timestep
        cond: dict with "state": (B, To, Do)
        returns: (B, Ta, Da) predicted noise
        """
        # Flatten obs conditioning
        state = cond["state"]
        B = state.shape[0]
        global_cond = state.reshape(B, -1)  # (B, To * Do)

        # Squeeze time if needed
        if time.dim() > 1:
            time = time.squeeze(-1)

        return self.unet(sample=x, timestep=time, global_cond=global_cond)
