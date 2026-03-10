"""
MIP-K: MIP with variable number of denoising steps.

Option A (MIPKShared): Single shared MLP denoiser, K steps.
Option B (MIPKIndependent): K separate MLP denoisers.
Option C (MIPKSharedUNet): Single shared UNet denoiser, K steps.
Option D (MIPKIndependentUNet): K separate UNet denoisers.

Both generalize the original 2-step MIP. At K=2, MIPKShared with t_star=0.9
uses t=[0, 0.9], matching original MIP (modulo loss weighting).
"""

import torch
import torch.nn as nn
from MultiGaussian.models.multi_gaussian import MIPPolicy
from MultiGaussian.models.mip_unet import MIPUNetPolicy


class MIPKShared(MIPPolicy):
    """MIP-K with shared denoiser (Option A).

    Same architecture as MIPPolicy. Training: sample random timestep from
    K evenly-spaced points in [0, t_star]. Inference: cascade K steps.
    """

    def __init__(self, K=2, **kwargs):
        super().__init__(**kwargs)
        self.K = K
        if K == 1:
            ts = torch.tensor([0.0])
        else:
            ts = torch.linspace(0, self.t_star, K)
        self.register_buffer('t_schedule', ts)

    def compute_loss(self, obs, actions):
        """Random timestep training (like DP). Uniform MSE loss."""
        B = obs.shape[0]
        k = torch.randint(0, self.K, (B,), device=obs.device)
        t = self.t_schedule[k]  # (B,)

        noise = torch.randn_like(actions)
        noise_level = 1.0 - t  # (B,)

        # Reshape for broadcasting: actions is (B, D) or (B, H, D)
        nl = noise_level
        is_k0 = (k == 0)
        for _ in range(actions.dim() - 1):
            nl = nl.unsqueeze(-1)
            is_k0 = is_k0.unsqueeze(-1)

        # k=0: zeros input; k>0: action + (1-t_k)*noise
        I_t = torch.where(is_k0, torch.zeros_like(actions), actions + nl * noise)

        pred = self.forward(obs, I_t, t)
        loss = ((pred - actions) ** 2).mean()
        return loss, loss, loss  # (total, t0_placeholder, tstar_placeholder)

    @torch.no_grad()
    def predict(self, obs):
        """K-step cascade from zeros."""
        B = obs.shape[0]
        if self.horizon_steps > 1:
            a = torch.zeros(B, self.horizon_steps, self.action_dim,
                            device=obs.device, dtype=obs.dtype)
        else:
            a = torch.zeros(B, self.action_dim, device=obs.device, dtype=obs.dtype)

        for k in range(self.K):
            a = self.forward(obs, a, self.t_schedule[k].item())
        return a


class MIPKIndependent(nn.Module):
    """MIP-K with independent denoisers (Option B).

    K separate MIPPolicy networks. Training: sample random k, train only
    that network. Inference: cascade through all K networks.
    """

    def __init__(self, K=2, input_dim=43, action_dim=8,
                 cond_steps=1, horizon_steps=1, t_star=0.9,
                 emb_dim=512, n_layers=6, dropout=0.1,
                 timestep_emb_dim=128, max_freq=100.0,
                 predict_epsilon=False, **kwargs):
        super().__init__()
        self.K = K
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.cond_steps = cond_steps

        net_kwargs = dict(
            input_dim=input_dim, action_dim=action_dim,
            cond_steps=cond_steps, horizon_steps=horizon_steps,
            t_star=t_star, dropout=dropout, emb_dim=emb_dim,
            n_layers=n_layers, timestep_emb_dim=timestep_emb_dim,
            max_freq=max_freq, predict_epsilon=predict_epsilon,
        )
        self.denoisers = nn.ModuleList([MIPPolicy(**net_kwargs) for _ in range(K)])

        if K == 1:
            ts = torch.tensor([0.0])
        else:
            ts = torch.linspace(0, t_star, K)
        self.register_buffer('t_schedule', ts)

    def compute_loss_k(self, obs, actions, k):
        """Compute loss for a specific denoiser k."""
        t_k = self.t_schedule[k].item()
        noise_level = 1.0 - t_k

        if k == 0:
            I_t = torch.zeros_like(actions)
        else:
            noise = torch.randn_like(actions)
            I_t = actions + noise_level * noise

        pred = self.denoisers[k].forward(obs, I_t, t_k)
        loss = ((pred - actions) ** 2).mean()
        return loss

    @torch.no_grad()
    def predict(self, obs):
        """K-step cascade through all denoisers."""
        B = obs.shape[0]
        if self.horizon_steps > 1:
            a = torch.zeros(B, self.horizon_steps, self.action_dim,
                            device=obs.device, dtype=obs.dtype)
        else:
            a = torch.zeros(B, self.action_dim, device=obs.device, dtype=obs.dtype)

        for k in range(self.K):
            a = self.denoisers[k].forward(obs, a, self.t_schedule[k].item())
        return a


class MIPKSharedUNet(MIPUNetPolicy):
    """MIP-K with shared UNet denoiser (Option C).

    Same as MIPKShared but uses UNet backbone instead of MLP.
    """

    def __init__(self, K=2, **kwargs):
        super().__init__(**kwargs)
        self.K = K
        if K == 1:
            ts = torch.tensor([0.0])
        else:
            ts = torch.linspace(0, self.t_star, K)
        self.register_buffer('t_schedule', ts)

    def compute_loss(self, obs, actions):
        """Random timestep training with per-sample t."""
        B = obs.shape[0]
        k = torch.randint(0, self.K, (B,), device=obs.device)
        t = self.t_schedule[k]  # (B,)

        noise = torch.randn_like(actions)
        noise_level = 1.0 - t  # (B,)

        nl = noise_level
        is_k0 = (k == 0)
        for _ in range(actions.dim() - 1):
            nl = nl.unsqueeze(-1)
            is_k0 = is_k0.unsqueeze(-1)

        I_t = torch.where(is_k0, torch.zeros_like(actions), actions + nl * noise)

        # UNet forward with per-sample timesteps
        obs_flat = obs.reshape(B, -1)
        if I_t.dim() == 2:
            I_t_3d = I_t.unsqueeze(1)
        else:
            I_t_3d = I_t
        timesteps = (t * 999).round().long()
        out = self.unet(I_t_3d, timesteps, global_cond=obs_flat)
        pred = out if self.horizon_steps > 1 else out.squeeze(1)

        loss = ((pred - actions) ** 2).mean()
        return loss, loss, loss

    @torch.no_grad()
    def predict(self, obs):
        """K-step cascade from zeros."""
        B = obs.shape[0]
        if self.horizon_steps > 1:
            a = torch.zeros(B, self.horizon_steps, self.action_dim,
                            device=obs.device, dtype=obs.dtype)
        else:
            a = torch.zeros(B, self.action_dim, device=obs.device, dtype=obs.dtype)

        for k in range(self.K):
            a = self.forward(obs, a, self.t_schedule[k].item())
        return a


class MIPKIndependentUNet(nn.Module):
    """MIP-K with independent UNet denoisers (Option D).

    K separate MIPUNetPolicy networks.
    """

    def __init__(self, K=2, input_dim=43, action_dim=8,
                 cond_steps=1, horizon_steps=1, t_star=0.9,
                 down_dims=[64, 128, 256], diffusion_step_embed_dim=64,
                 kernel_size=5, n_groups=8, **kwargs):
        super().__init__()
        self.K = K
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.cond_steps = cond_steps

        net_kwargs = dict(
            input_dim=input_dim, action_dim=action_dim,
            cond_steps=cond_steps, horizon_steps=horizon_steps,
            t_star=t_star, down_dims=down_dims,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            kernel_size=kernel_size, n_groups=n_groups,
        )
        self.denoisers = nn.ModuleList([MIPUNetPolicy(**net_kwargs) for _ in range(K)])

        if K == 1:
            ts = torch.tensor([0.0])
        else:
            ts = torch.linspace(0, t_star, K)
        self.register_buffer('t_schedule', ts)

    def compute_loss_k(self, obs, actions, k):
        """Compute loss for a specific denoiser k."""
        t_k = self.t_schedule[k].item()

        if k == 0:
            I_t = torch.zeros_like(actions)
        else:
            noise = torch.randn_like(actions)
            noise_level = 1.0 - t_k
            I_t = actions + noise_level * noise

        pred = self.denoisers[k].forward(obs, I_t, t_k)
        loss = ((pred - actions) ** 2).mean()
        return loss

    @torch.no_grad()
    def predict(self, obs):
        """K-step cascade through all denoisers."""
        B = obs.shape[0]
        if self.horizon_steps > 1:
            a = torch.zeros(B, self.horizon_steps, self.action_dim,
                            device=obs.device, dtype=obs.dtype)
        else:
            a = torch.zeros(B, self.action_dim, device=obs.device, dtype=obs.dtype)

        for k in range(self.K):
            a = self.denoisers[k].forward(obs, a, self.t_schedule[k].item())
        return a
