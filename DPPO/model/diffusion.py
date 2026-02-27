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


class EtaFixed(nn.Module):
    """Fixed eta schedule for DDIM sampling. Matches irom-princeton/dppo exactly.

    Maps a learnable logit through tanh to [min_eta, max_eta].
    With base_eta=1, min_eta=0.1, max_eta=1.0, returns constant 1.0.
    """
    def __init__(self, base_eta=1.0, min_eta=0.1, max_eta=1.0):
        super().__init__()
        self.eta_logit = nn.Parameter(torch.ones(1))
        self.min = min_eta
        self.max = max_eta
        # Initialize logit so that tanh(logit) maps to base_eta
        # Clamp to avoid inf (atanh(1.0)=inf → NaN via AdamW weight decay)
        normalized = 2 * (base_eta - min_eta) / (max_eta - min_eta) - 1
        normalized = min(max(normalized, -0.999), 0.999)
        self.eta_logit.data = torch.atanh(torch.tensor([normalized]))

    def forward(self, cond):
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = len(sample_data)
        device = sample_data.device
        eta_normalized = torch.tanh(self.eta_logit)
        eta = 0.5 * (eta_normalized + 1) * (self.max - self.min) + self.min
        return torch.full((B, 1), eta.item()).to(device)


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
        base_eta=1.0,
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
        self.eta = EtaFixed(base_eta=base_eta)

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
    def forward(self, cond, deterministic=True, min_sampling_denoising_std=None,
                ddim_steps=None):
        """
        Denoising chain for sampling actions. Supports DDPM (full steps) and DDIM.

        Args:
            cond: dict with key "state": (B, To, Do)
            deterministic: if True, use minimal noise (eval mode)
            min_sampling_denoising_std: if not deterministic, clip noise std
                to this minimum (DPPO exploration). Default 0.1.
            ddim_steps: if set, use DDIM with this many steps instead of full DDPM
        """
        device = self.betas.device
        B = len(cond["state"])
        x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)

        if ddim_steps is not None and ddim_steps < self.denoising_steps:
            return self._forward_ddim(x, cond, ddim_steps, deterministic,
                                      min_sampling_denoising_std)
        else:
            return self._forward_ddpm(x, cond, deterministic,
                                      min_sampling_denoising_std)

    def _forward_ddpm(self, x, cond, deterministic, min_sampling_denoising_std):
        """Full DDPM sampling. Aligned with DPPO VPGDiffusion.forward().

        Std handling (matches VPGDiffusion exactly):
          - deterministic, t==0: std=0
          - deterministic, t>0:  std=clip(posterior_std, min=1e-3)
          - stochastic:          std=clip(posterior_std, min=min_sampling_denoising_std)
        """
        min_std = min_sampling_denoising_std or 0.1
        t_all = list(reversed(range(self.denoising_steps)))
        for i, t in enumerate(t_all):
            t_b = make_timesteps(x.shape[0], t, x.device)
            mean, logvar = self.p_mean_var(x=x, t=t_b, cond=cond)
            std = torch.exp(0.5 * logvar)

            if deterministic and t == 0:
                std = torch.zeros_like(std)
            elif deterministic:
                std = torch.clip(std, min=1e-3)
            else:
                std = torch.clip(std, min=min_std)

            noise = torch.randn_like(x).clamp_(
                -self.randn_clip_value, self.randn_clip_value
            )
            x = mean + std * noise

            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch.clamp(
                    x, -self.final_action_clip_value, self.final_action_clip_value
                )
        return Sample(x, None)

    def _forward_ddim(self, x, cond, ddim_steps, deterministic,
                      min_sampling_denoising_std):
        """DDIM sampling. Aligned with DPPO (irom-princeton/dppo).

        Uses self.eta (EtaFixed module) to compute per-step sigma, matching
        diffusion_vpg.py p_mean_var (DDIM branch) and forward loop exactly.

        eta controls the mean/variance tradeoff:
          eta=0: deterministic DDIM (noise_pred at full strength, sigma=0)
          eta=1: DDPM-like (noise_pred reduced by sigma², sigma = DDPM posterior)

        DPPO uses eta=1 (EtaFixed base_eta=1) so that the mean properly accounts
        for the injected noise.

        Std handling (matches VPGDiffusion.forward()):
          - deterministic: etas=0 (bypasses eta module), std=0
          - stochastic:    etas=self.eta(cond), std=clip(sigma, min=min_sampling_denoising_std)
        """
        T = self.denoising_steps
        B = x.shape[0]
        device = x.device
        min_std = min_sampling_denoising_std or 0.1

        # Compute etas from module (matches VPG: deterministic → zeros, else → eta(cond))
        if deterministic:
            etas = torch.zeros((B, 1), device=device)
        else:
            etas = self.eta(cond)  # (B, 1)
        etas = etas.unsqueeze(1)  # (B, 1, 1) for broadcasting with (B, H, Da)

        # Timestep schedule (matches reference: "leading" style uniform discretization)
        step_ratio = T // ddim_steps
        ddim_t = torch.arange(0, ddim_steps, device=device) * step_ratio

        # Precompute alpha schedules (ascending order first)
        ddim_alphas = self.alphas_cumprod[ddim_t].clone()
        ddim_alphas_prev = torch.cat([
            torch.tensor([1.0], device=device),
            self.alphas_cumprod[ddim_t[:-1]],
        ])
        ddim_sqrt_one_minus_alphas = (1.0 - ddim_alphas) ** 0.5

        # Flip to descending order (denoise from high noise to clean)
        ddim_t = torch.flip(ddim_t, [0])
        ddim_alphas = torch.flip(ddim_alphas, [0])
        ddim_alphas_prev = torch.flip(ddim_alphas_prev, [0])
        ddim_sqrt_one_minus_alphas = torch.flip(ddim_sqrt_one_minus_alphas, [0])

        for i in range(ddim_steps):
            t_b = make_timesteps(B, ddim_t[i].item(), device)

            # Predict noise
            noise_pred = self.network(x, t_b, cond=cond)

            # Predict x_0: x₀ = (xₜ - √(1-ᾱₜ) ε) / √ᾱₜ
            alpha = ddim_alphas[i]
            alpha_prev = ddim_alphas_prev[i]
            sqrt_one_minus_alpha = ddim_sqrt_one_minus_alphas[i]

            x_recon = (x - sqrt_one_minus_alpha * noise_pred) / (alpha ** 0.5)

            # Clip x_0 and recompute noise for consistency (matches original)
            if self.denoised_clip_value is not None:
                x_recon.clamp_(-self.denoised_clip_value, self.denoised_clip_value)
                noise_pred = (x - (alpha ** 0.5) * x_recon) / sqrt_one_minus_alpha

            # DDIM sigma: etas * sqrt((1-α_prev)/(1-α) * (1-α/α_prev))
            # Matches VPG p_mean_var DDIM branch exactly
            sigma = (
                etas
                * ((1 - alpha_prev) / (1 - alpha)
                   * (1 - alpha / alpha_prev)).clamp(min=0).sqrt()
            ).clamp_(min=1e-10)

            # DDIM mean: μ = √ᾱₜ₋₁ x₀ + √(1-ᾱₜ₋₁-σ²) ε
            dir_xt_coef = (1.0 - alpha_prev - sigma ** 2).clamp_(min=0).sqrt()
            mean = (alpha_prev ** 0.5) * x_recon + dir_xt_coef * noise_pred

            # Std: clip sigma to min_sampling_denoising_std (matches VPG forward)
            if deterministic:
                std = torch.zeros_like(x)
            else:
                std = torch.clip(sigma, min=min_std)

            noise = torch.randn_like(x).clamp_(
                -self.randn_clip_value, self.randn_clip_value
            )
            x = mean + std * noise

            if self.final_action_clip_value is not None and i == ddim_steps - 1:
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
