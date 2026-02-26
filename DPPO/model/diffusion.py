"""
Base Gaussian diffusion model with DDPM sampling.

Ported from https://github.com/irom-princeton/dppo
"""

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from collections import namedtuple

Sample = namedtuple("Sample", "trajectories chains")


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def make_timesteps(batch_size, i, device):
    return torch.full((batch_size,), i, device=device, dtype=torch.long)


class DiffusionModel(nn.Module):
    def __init__(
        self,
        network,
        horizon_steps,
        obs_dim,
        action_dim,
        device="cpu",
        denoised_clip_value=1.0,
        randn_clip_value=10,
        final_action_clip_value=None,
        denoising_steps=100,
        predict_epsilon=True,
    ):
        super().__init__()
        self.device = device
        self.horizon_steps = horizon_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.denoising_steps = int(denoising_steps)
        self.predict_epsilon = predict_epsilon
        self.denoised_clip_value = denoised_clip_value
        self.final_action_clip_value = final_action_clip_value
        self.randn_clip_value = randn_clip_value

        self.network = network.to(device)

        # DDPM parameters: cosine beta schedule
        self.betas = cosine_beta_schedule(denoising_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), self.alphas_cumprod[:-1]]
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # Posterior variance
        self.ddpm_var = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.ddpm_logvar_clipped = torch.log(torch.clamp(self.ddpm_var, min=1e-20))

        # Posterior mean coefficients
        self.ddpm_mu_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.ddpm_mu_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def p_mean_var(self, x, t, cond, network_override=None):
        if network_override is not None:
            noise = network_override(x, t, cond=cond)
        else:
            noise = self.network(x, t, cond=cond)

        if self.predict_epsilon:
            x_recon = (
                extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * noise
            )
        else:
            x_recon = noise

        if self.denoised_clip_value is not None:
            x_recon.clamp_(-self.denoised_clip_value, self.denoised_clip_value)

        mu = (
            extract(self.ddpm_mu_coef1, t, x.shape) * x_recon
            + extract(self.ddpm_mu_coef2, t, x.shape) * x
        )
        logvar = extract(self.ddpm_logvar_clipped, t, x.shape)
        return mu, logvar

    @torch.no_grad()
    def forward(self, cond, deterministic=True, min_sampling_denoising_std=None):
        """
        Full denoising chain for sampling actions.

        Args:
            cond: dict with key "state": (B, To, Do)
            deterministic: if True, use minimal noise (eval mode)
            min_sampling_denoising_std: if set and not deterministic, clip std
                to this minimum at each denoising step (DPPO exploration noise)
        Returns:
            Sample(trajectories=(B, Ta, Da), chains=None)
        """
        device = self.betas.device
        sample_data = cond["state"]
        B = len(sample_data)

        x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
        t_all = list(reversed(range(self.denoising_steps)))
        for t in t_all:
            t_b = make_timesteps(B, t, device)
            mean, logvar = self.p_mean_var(x=x, t=t_b, cond=cond)
            std = torch.exp(0.5 * logvar)
            if deterministic and t == 0:
                std = torch.zeros_like(std)
            elif deterministic:
                std = torch.clip(std, min=1e-3)
            else:
                # Exploration mode: enforce minimum std floor
                min_std = min_sampling_denoising_std or 0.1
                if t == 0:
                    std = torch.zeros_like(std)
                else:
                    std = torch.clip(std, min=min_std)
            noise = torch.randn_like(x).clamp_(
                -self.randn_clip_value, self.randn_clip_value
            )
            x = mean + std * noise
            if self.final_action_clip_value is not None and t == 0:
                x = torch.clamp(
                    x, -self.final_action_clip_value, self.final_action_clip_value
                )
        return Sample(x, None)

    # --- Supervised training ---

    def loss(self, x, *args):
        batch_size = len(x)
        t = torch.randint(
            0, self.denoising_steps, (batch_size,), device=x.device
        ).long()
        return self.p_losses(x, *args, t)

    def p_losses(self, x_start, cond, t):
        """
        Noise prediction loss: E_{t,x0,eps} [||eps - eps_theta(sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps, t)||^2]
        """
        device = x_start.device
        noise = torch.randn_like(x_start, device=device)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.network(x_noisy, t, cond=cond)
        if self.predict_epsilon:
            return F.mse_loss(x_recon, noise, reduction="mean")
        else:
            return F.mse_loss(x_recon, x_start, reduction="mean")

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start, device=x_start.device)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
