"""
Minimal Iterative Policy (MIP) from "Much Ado About Noising" (arxiv 2512.01809).

Aligned with official implementation: https://github.com/simchowitzlabpublic/much-ado-about-noising

Key differences from our initial implementation (now fixed):
1. Network: ResidualBlock with LayerNorm + GELU + 4x expansion, orthogonal init
2. Time embedding: sinusoidal frequency embedding (concat), not FiLM
3. Loss scaling: divide by t_star and (1-t_star), making refinement loss 9x heavier
4. Noised target: act_t = act + (1-t*)*noise (not t*·act + (1-t*)·noise)
5. Default: emb_dim=512, n_layers=6, dropout=0.1
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm, GELU, 4x expansion, and dropout."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
        self.dropout2 = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.orthogonal_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0)

    def forward(self, x):
        h = self.norm1(x)
        h = self.linear1(h)
        h = self.act(h)
        h = self.dropout1(h)
        h = self.norm2(h)
        h = self.linear2(h)
        h = self.dropout2(h)
        return x + h


class MIPPolicy(nn.Module):
    """Minimal Iterative Policy aligned with official implementation.

    Architecture: sinusoidal time embedding + residual MLP blocks.
    Training: two losses with 1/t* and 1/(1-t*) weighting.
    Inference: strict two-pass deterministic.
    """

    def __init__(self, input_dim, action_dim, hidden_dims=None,
                 cond_steps=1, horizon_steps=1, t_star=0.9, dropout=0.1,
                 emb_dim=512, n_layers=6, timestep_emb_dim=128, max_freq=100.0,
                 predict_epsilon=False):
        super().__init__()
        assert 0.0 < t_star <= 1.0, f"t_star must be in (0, 1], got {t_star}"
        self.cond_steps = cond_steps
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.t_star = t_star
        self.timestep_emb_dim = timestep_emb_dim
        self.predict_epsilon = predict_epsilon

        # Override hidden_dims with emb_dim/n_layers if provided
        if hidden_dims is not None:
            emb_dim = hidden_dims[0]
            n_layers = len(hidden_dims)

        obs_flat_dim = input_dim * cond_steps
        act_flat_dim = action_dim * horizon_steps

        # Sinusoidal time embedding (uniform frequencies)
        assert timestep_emb_dim % 2 == 0
        num_freq = timestep_emb_dim // 2
        self.register_buffer("frequencies", torch.linspace(0, max_freq, num_freq))

        # Input: [act_flat, s_emb, t_emb, obs_flat]
        input_dim_total = act_flat_dim + 2 * timestep_emb_dim + obs_flat_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim_total, emb_dim)
        self.input_norm = nn.LayerNorm(emb_dim)
        self.input_act = nn.GELU()

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(emb_dim, dropout=dropout) for _ in range(n_layers)
        ])

        # Output
        self.out_norm = nn.LayerNorm(emb_dim)
        self.out_proj = nn.Linear(emb_dim, act_flat_dim)

        # Orthogonal init for projections
        nn.init.orthogonal_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0)
        nn.init.orthogonal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0)

    def _embed_time(self, t_val, B, device, dtype):
        """Sinusoidal time embedding."""
        if isinstance(t_val, (int, float)):
            t = torch.full((B, 1), float(t_val), device=device, dtype=dtype)
        else:
            t = t_val.unsqueeze(-1) if t_val.dim() == 1 else t_val
        angles = t * self.frequencies.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(self, obs, I_t, t):
        """Single forward pass: network(I_t, s=t, t=t, obs).

        Following official: get_velocity passes t for both s and t args.
        """
        B = obs.shape[0]

        obs_flat = obs.reshape(B, -1) if obs.dim() == 3 else obs
        I_flat = I_t.reshape(B, -1) if I_t.dim() == 3 else I_t

        # Time embeddings (s=t, t=t for velocity mode, matching official get_velocity)
        t_emb = self._embed_time(t, B, obs.device, obs.dtype)
        s_emb = t_emb  # s = t for get_velocity

        # Concat: [I_t, s_emb, t_emb, obs]
        x = torch.cat([I_flat, s_emb, t_emb, obs_flat], dim=-1)

        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_act(x)

        # Residual blocks
        for block in self.blocks:
            x = block(x)

        # Output
        out = self.out_proj(self.out_norm(x))

        if self.horizon_steps > 1:
            return out.reshape(B, self.horizon_steps, self.action_dim)
        return out

    def compute_loss(self, obs, actions):
        """MIP loss. Supports both x-prediction and epsilon-prediction.

        x-prediction (original):
            loss0 = mean(||pred_0 - act||^2 / t_star^2)
            loss1 = mean(||pred_1 - act||^2 / (1-t_star)^2)

        epsilon-prediction (diffusion-style):
            Step 1: I_0 = act + noise, predict noise. loss0 = mean(||eps_hat - noise||^2)
            Step 2: I_t* = act + (1-t*)*noise, predict noise. loss1 = mean(||eps_hat - noise||^2)
        """
        t_star = self.t_star

        if self.predict_epsilon:
            # Step 1: full noise at t=0
            noise_0 = torch.randn_like(actions)
            I_0 = actions + 1.0 * noise_0  # at t=0, noise_scale = (1-0) = 1
            eps_hat_0 = self.forward(obs, I_0, 0.0)
            loss_t0 = torch.mean(torch.sum((eps_hat_0 - noise_0) ** 2, dim=-1))

            # Step 2: small noise at t=t*
            noise_1 = torch.randn_like(actions)
            I_tstar = actions + (1.0 - t_star) * noise_1
            eps_hat_1 = self.forward(obs, I_tstar, t_star)
            loss_tstar = torch.mean(torch.sum((eps_hat_1 - noise_1) ** 2, dim=-1))
        else:
            # Original x-prediction
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
        """Two-pass inference.

        x-prediction: a0 = net(o, 0, 0); a = net(o, a0, t*)
        eps-prediction: a0 = I_0 - net(o, I_0, 0); a = a0 - (1-t*) * net(o, a0, t*)
        """
        B = obs.shape[0]
        if self.horizon_steps > 1:
            act_shape = (B, self.horizon_steps, self.action_dim)
        else:
            act_shape = (B, self.action_dim)

        if self.predict_epsilon:
            # Pass 1: start from zeros, predict eps, recover x
            # At t=0, noise_scale=1.0, so x = I_0 - 1.0 * eps_hat
            I_0 = torch.zeros(act_shape, device=obs.device, dtype=obs.dtype)
            eps_hat = self.forward(obs, I_0, 0.0)
            a_hat = I_0 - 1.0 * eps_hat

            # Pass 2: refine, predict residual eps
            eps_hat = self.forward(obs, a_hat, self.t_star)
            a_hat = a_hat - (1.0 - self.t_star) * eps_hat
        else:
            # Original x-prediction
            I_0 = torch.zeros(act_shape, device=obs.device, dtype=obs.dtype)
            a_hat = self.forward(obs, I_0, 0.0)
            a_hat = self.forward(obs, a_hat, self.t_star)

        return a_hat
