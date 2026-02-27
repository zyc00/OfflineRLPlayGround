"""
VPGDiffusion + PPODiffusion: Diffusion policy with PPO finetuning.

Aligned with https://github.com/irom-princeton/dppo

Supports both DDPM and DDIM denoising for RL rollout and logprob computation.
For sparse reward tasks, DDIM (5 steps) is recommended over DDPM (100 steps).
"""

import copy
import math
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from DPPO.model.diffusion import DiffusionModel, Sample, extract, make_timesteps


class VPGDiffusion(DiffusionModel):
    """Diffusion model with dual actor for RL finetuning.

    Supports both DDPM and DDIM denoising:
    - DDPM: full T denoising steps, last ft_denoising_steps use actor_ft
    - DDIM: ddim_steps fast denoising, last ft_denoising_steps use actor_ft
    """

    def __init__(
        self,
        actor,
        critic,
        ft_denoising_steps,
        min_sampling_denoising_std=0.1,
        min_logprob_denoising_std=0.1,
        use_ddim=False,
        ddim_steps=None,
        network_path=None,
        **kwargs,
    ):
        super().__init__(network=actor, **kwargs)
        self.ft_denoising_steps = ft_denoising_steps
        self.min_sampling_denoising_std = min_sampling_denoising_std
        self.min_logprob_denoising_std = min_logprob_denoising_std
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps

        if use_ddim:
            assert ddim_steps is not None
            assert ft_denoising_steps <= ddim_steps
            self._setup_ddim_schedule(ddim_steps)
        else:
            assert ft_denoising_steps <= self.denoising_steps

        # Rename network to actor, create finetuning copy
        self.actor = self.network
        self.actor_ft = copy.deepcopy(self.actor)

        # Freeze original actor
        for param in self.actor.parameters():
            param.requires_grad = False

        # Value function
        self.critic = critic.to(self.device)

        # Load pretrained weights if provided
        if network_path is not None:
            checkpoint = torch.load(
                network_path, map_location=self.device, weights_only=True
            )
            if "ema" in checkpoint:
                self.load_state_dict(checkpoint["ema"], strict=False)
            else:
                self.load_state_dict(checkpoint["model"], strict=False)
            # Copy to actor_ft and re-freeze original
            self.actor_ft.load_state_dict(self.actor.state_dict())
            for param in self.actor.parameters():
                param.requires_grad = False
        # Sanitize eta_logit: atanh(1.0)=inf → NaN via AdamW weight decay
        self._sanitize_eta()

    def _sanitize_eta(self):
        """Fix NaN/inf eta_logit from checkpoints trained with AdamW weight decay."""
        logit = self.eta.eta_logit.data
        if torch.isnan(logit).any() or torch.isinf(logit).any():
            # Reset to atanh(0.999) ≈ 3.8, which maps to eta ≈ 1.0
            self.eta.eta_logit.data = torch.atanh(torch.tensor([0.999]))

    def _setup_ddim_schedule(self, ddim_steps):
        """Precompute DDIM timestep schedule and alpha values."""
        T = self.denoising_steps
        device = self.alphas_cumprod.device
        # Ascending order (matches reference: "leading" style uniform discretization)
        step_ratio = T // ddim_steps
        ddim_t = torch.arange(0, ddim_steps, device=device) * step_ratio
        ddim_alphas = self.alphas_cumprod[ddim_t].clone()
        ddim_alphas_prev = torch.cat([
            torch.tensor([1.0], device=device),
            self.alphas_cumprod[ddim_t[:-1]],
        ])
        # Flip to descending (high noise → clean)
        self.register_buffer("ddim_t", torch.flip(ddim_t, [0]))
        self.register_buffer("ddim_alphas", torch.flip(ddim_alphas, [0]))
        self.register_buffer("ddim_alphas_prev", torch.flip(ddim_alphas_prev, [0]))
        self.register_buffer("ddim_sqrt_one_minus_alphas",
                             torch.flip((1.0 - ddim_alphas) ** 0.5, [0]))

    def _predict_noise_routed(self, x, t, cond, use_base_policy=False, ddim_step_idx=None):
        """Get noise prediction, routing between frozen actor and actor_ft.

        Args:
            x: (B, Ta, Da) noisy actions
            t: (B,) DDPM timesteps
            cond: dict with state: (B, To, Do)
            use_base_policy: if True, use frozen actor for all steps
            ddim_step_idx: if set, use DDIM step index for routing (not DDPM timestep)
        Returns:
            noise_pred: (B, Ta, Da)
        """
        # Start with base actor for all
        noise = self.actor(x, t, cond=cond)

        # Determine which samples should use actor_ft
        if self.use_ddim and ddim_step_idx is not None:
            # DDIM: finetune the last ft_denoising_steps DDIM steps
            # ddim_step_idx: 0 = first DDIM step (highest noise), ddim_steps-1 = last
            ft_threshold = self.ddim_steps - self.ft_denoising_steps
            ft_indices = torch.where(ddim_step_idx >= ft_threshold)[0]
        else:
            # DDPM: finetune steps t < ft_denoising_steps
            ft_indices = torch.where(t < self.ft_denoising_steps)[0]

        actor = self.actor if use_base_policy else self.actor_ft

        if len(ft_indices) > 0:
            cond_ft = {key: cond[key][ft_indices] for key in cond}
            noise_ft = actor(x[ft_indices], t[ft_indices], cond=cond_ft)
            noise[ft_indices] = noise_ft

        return noise

    def p_mean_var(self, x, t, cond, use_base_policy=False, deterministic=False):
        """DDPM mean/var with actor routing. Used for DDPM forward and logprobs."""
        noise = self._predict_noise_routed(x, t, cond, use_base_policy)

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

    def _ddim_mean_sigma(self, x, t, cond, alpha, alpha_prev, sqrt_one_minus_alpha,
                         etas, use_base_policy=False, ddim_step_idx=None):
        """Compute DDIM mean and sigma for one step.

        Returns: (mean, sigma, x_recon)
        """
        noise_pred = self._predict_noise_routed(x, t, cond, use_base_policy, ddim_step_idx)

        # Predict x_0
        x_recon = (x - sqrt_one_minus_alpha * noise_pred) / (alpha ** 0.5)
        if self.denoised_clip_value is not None:
            x_recon.clamp_(-self.denoised_clip_value, self.denoised_clip_value)
            noise_pred = (x - (alpha ** 0.5) * x_recon) / sqrt_one_minus_alpha

        # DDIM sigma
        sigma = (
            etas * ((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev))
            .clamp(min=0).sqrt()
        ).clamp_(min=1e-10)

        # DDIM mean
        dir_xt_coef = (1.0 - alpha_prev - sigma ** 2).clamp_(min=0).sqrt()
        mean = (alpha_prev ** 0.5) * x_recon + dir_xt_coef * noise_pred

        return mean, sigma

    @torch.no_grad()
    def forward(
        self,
        cond,
        deterministic=False,
        return_chain=True,
        use_base_policy=False,
    ):
        """
        Forward pass with chain tracking for RL training.

        Dispatches to DDPM or DDIM based on self.use_ddim.

        Returns:
            Sample(trajectories=(B, Ta, Da), chains=(B, K+1, Ta, Da) or None)
        """
        if self.use_ddim:
            return self._forward_ddim(cond, deterministic, return_chain, use_base_policy)
        else:
            return self._forward_ddpm(cond, deterministic, return_chain, use_base_policy)

    def _forward_ddpm(self, cond, deterministic, return_chain, use_base_policy):
        """DDPM forward with chain tracking."""
        device = self.betas.device
        B = len(cond["state"])
        min_std = self.min_sampling_denoising_std

        x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
        t_all = list(reversed(range(self.denoising_steps)))

        chain = [] if return_chain else None
        if self.ft_denoising_steps == self.denoising_steps and return_chain:
            chain.append(x)

        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t, device)
            mean, logvar = self.p_mean_var(
                x=x, t=t_b, cond=cond,
                use_base_policy=use_base_policy,
            )
            std = torch.exp(0.5 * logvar)

            if deterministic and t == 0:
                std = torch.zeros_like(std)
            elif deterministic:
                std = torch.clip(std, min=1e-3)
            else:
                std = torch.clip(std, min=min_std)

            noise = torch.randn_like(x).clamp_(-self.randn_clip_value, self.randn_clip_value)
            x = mean + std * noise

            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch.clamp(x, -self.final_action_clip_value, self.final_action_clip_value)

            if return_chain and t <= self.ft_denoising_steps:
                chain.append(x)

        if return_chain:
            chain = torch.stack(chain, dim=1)
        return Sample(x, chain)

    def _forward_ddim(self, cond, deterministic, return_chain, use_base_policy):
        """DDIM forward with chain tracking."""
        device = self.betas.device
        B = len(cond["state"])
        min_std = self.min_sampling_denoising_std

        x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)

        # Compute etas
        if deterministic:
            etas = torch.zeros((B, 1, 1), device=device)
        else:
            etas = self.eta(cond).unsqueeze(1)  # (B, 1, 1)

        chain = [] if return_chain else None
        ft_start = self.ddim_steps - self.ft_denoising_steps

        if ft_start == 0 and return_chain:
            chain.append(x)

        for i in range(self.ddim_steps):
            t_b = make_timesteps(B, self.ddim_t[i].item(), device)
            ddim_idx = make_timesteps(B, i, device)

            alpha = self.ddim_alphas[i]
            alpha_prev = self.ddim_alphas_prev[i]
            sqrt_one_minus = self.ddim_sqrt_one_minus_alphas[i]

            mean, sigma = self._ddim_mean_sigma(
                x, t_b, cond, alpha, alpha_prev, sqrt_one_minus, etas,
                use_base_policy, ddim_idx,
            )

            if deterministic:
                std = torch.zeros_like(x)
            else:
                std = torch.clip(sigma, min=min_std)

            noise = torch.randn_like(x).clamp_(-self.randn_clip_value, self.randn_clip_value)
            x = mean + std * noise

            if self.final_action_clip_value is not None and i == self.ddim_steps - 1:
                x = torch.clamp(x, -self.final_action_clip_value, self.final_action_clip_value)

            if return_chain and i >= max(ft_start - 1, 0):
                chain.append(x)

        if return_chain:
            chain = torch.stack(chain, dim=1)
        return Sample(x, chain)

    def get_logprobs(self, cond, chains, get_ent=False, use_base_policy=False):
        """
        Calculate logprobs of the entire denoising chain.

        Dispatches to DDPM or DDIM logprob computation.

        Args:
            cond: dict with state: (B, To, Do)
            chains: (B, K+1, Ta, Da)
        Returns:
            logprobs: (B*K, Ta, Da)
        """
        if self.use_ddim:
            return self._get_logprobs_ddim(cond, chains, get_ent, use_base_policy)
        else:
            return self._get_logprobs_ddpm(cond, chains, get_ent, use_base_policy)

    def _get_logprobs_ddpm(self, cond, chains, get_ent, use_base_policy):
        """DDPM logprobs using p_mean_var."""
        K = self.ft_denoising_steps
        B = chains.shape[0]

        cond_rep = {
            key: cond[key].unsqueeze(1)
            .repeat(1, K, *(1,) * (cond[key].ndim - 1))
            .flatten(start_dim=0, end_dim=1)
            for key in cond
        }

        t_single = torch.arange(start=K - 1, end=-1, step=-1, device=self.device)
        t_all = t_single.repeat(B, 1).flatten()

        chains_prev = chains[:, :-1].reshape(-1, self.horizon_steps, self.action_dim)
        chains_next = chains[:, 1:].reshape(-1, self.horizon_steps, self.action_dim)

        next_mean, logvar = self.p_mean_var(
            chains_prev, t_all, cond=cond_rep, use_base_policy=use_base_policy,
        )
        std = torch.exp(0.5 * logvar)
        std = torch.clip(std, min=self.min_logprob_denoising_std)
        dist = Normal(next_mean, std)
        log_prob = dist.log_prob(chains_next)

        if get_ent:
            return log_prob, dist.entropy()
        return log_prob

    def _get_logprobs_ddim(self, cond, chains, get_ent, use_base_policy):
        """DDIM logprobs using DDIM mean/sigma."""
        K = self.ft_denoising_steps
        B = chains.shape[0]
        ft_start = self.ddim_steps - K

        cond_rep = {
            key: cond[key].unsqueeze(1)
            .repeat(1, K, *(1,) * (cond[key].ndim - 1))
            .flatten(start_dim=0, end_dim=1)
            for key in cond
        }

        # DDIM step indices for the finetuned steps
        ddim_idx_single = torch.arange(ft_start, self.ddim_steps, device=self.device)
        ddim_idx_all = ddim_idx_single.repeat(B, 1).flatten()  # (B*K,)
        t_all = self.ddim_t[ddim_idx_all]  # Map to DDPM timesteps

        # Etas (use same for logprobs — non-deterministic)
        etas = self.eta(cond)  # (B, 1)
        etas_rep = etas.unsqueeze(1).expand(-1, K, -1).reshape(B * K, 1, 1)  # (B*K, 1, 1)

        chains_prev = chains[:, :-1].reshape(-1, self.horizon_steps, self.action_dim)
        chains_next = chains[:, 1:].reshape(-1, self.horizon_steps, self.action_dim)

        alpha = self.ddim_alphas[ddim_idx_all]
        alpha_prev = self.ddim_alphas_prev[ddim_idx_all]
        sqrt_one_minus = self.ddim_sqrt_one_minus_alphas[ddim_idx_all]

        mean, sigma = self._ddim_mean_sigma(
            chains_prev, t_all, cond_rep,
            alpha.unsqueeze(-1).unsqueeze(-1),
            alpha_prev.unsqueeze(-1).unsqueeze(-1),
            sqrt_one_minus.unsqueeze(-1).unsqueeze(-1),
            etas_rep, use_base_policy, ddim_idx_all,
        )

        std = torch.clip(sigma, min=self.min_logprob_denoising_std)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(chains_next)

        if get_ent:
            return log_prob, dist.entropy()
        return log_prob

    def get_logprobs_subsample(
        self,
        cond,
        chains_prev,
        chains_next,
        denoising_inds,
        get_ent=False,
        use_base_policy=False,
    ):
        """
        Calculate logprobs for subsampled denoising steps (used in PPO minibatch).

        Args:
            cond: dict with state: (B, To, Do)
            chains_prev: (B, Ta, Da)
            chains_next: (B, Ta, Da)
            denoising_inds: (B,) indices into ft_denoising_steps (0=first ft step)
        Returns:
            logprobs: (B, Ta, Da)
        """
        if self.use_ddim:
            return self._get_logprobs_subsample_ddim(
                cond, chains_prev, chains_next, denoising_inds, get_ent, use_base_policy)
        else:
            return self._get_logprobs_subsample_ddpm(
                cond, chains_prev, chains_next, denoising_inds, get_ent, use_base_policy)

    def _get_logprobs_subsample_ddpm(self, cond, chains_prev, chains_next,
                                      denoising_inds, get_ent, use_base_policy):
        """DDPM logprobs for subsampled denoising steps."""
        t_single = torch.arange(
            start=self.ft_denoising_steps - 1, end=-1, step=-1, device=self.device,
        )
        t_all = t_single[denoising_inds]

        next_mean, logvar = self.p_mean_var(
            chains_prev, t_all, cond=cond, use_base_policy=use_base_policy,
        )
        std = torch.exp(0.5 * logvar)
        std = torch.clip(std, min=self.min_logprob_denoising_std)
        dist = Normal(next_mean, std)
        log_prob = dist.log_prob(chains_next)

        if get_ent:
            return log_prob, dist.entropy()
        return log_prob

    def _get_logprobs_subsample_ddim(self, cond, chains_prev, chains_next,
                                      denoising_inds, get_ent, use_base_policy):
        """DDIM logprobs for subsampled denoising steps."""
        ft_start = self.ddim_steps - self.ft_denoising_steps
        ddim_idx = (ft_start + denoising_inds).to(self.device)
        t_all = self.ddim_t[ddim_idx]

        etas = self.eta(cond).unsqueeze(1)  # (B, 1, 1)

        alpha = self.ddim_alphas[ddim_idx].unsqueeze(-1).unsqueeze(-1)
        alpha_prev = self.ddim_alphas_prev[ddim_idx].unsqueeze(-1).unsqueeze(-1)
        sqrt_one_minus = self.ddim_sqrt_one_minus_alphas[ddim_idx].unsqueeze(-1).unsqueeze(-1)

        mean, sigma = self._ddim_mean_sigma(
            chains_prev, t_all, cond, alpha, alpha_prev, sqrt_one_minus, etas,
            use_base_policy, ddim_idx,
        )

        std = torch.clip(sigma, min=self.min_logprob_denoising_std)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(chains_next)

        if get_ent:
            return log_prob, dist.entropy()
        return log_prob


class PPODiffusion(VPGDiffusion):
    """PPO-based finetuning of diffusion policy."""

    def __init__(
        self,
        gamma_denoising: float,
        clip_ploss_coef: float,
        clip_ploss_coef_base: float = 1e-3,
        clip_ploss_coef_rate: float = 3,
        clip_vloss_coef=None,
        clip_advantage_lower_quantile: float = 0,
        clip_advantage_upper_quantile: float = 1,
        norm_adv: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm_adv = norm_adv
        self.clip_ploss_coef = clip_ploss_coef
        self.clip_ploss_coef_base = clip_ploss_coef_base
        self.clip_ploss_coef_rate = clip_ploss_coef_rate
        self.clip_vloss_coef = clip_vloss_coef
        self.gamma_denoising = gamma_denoising
        self.clip_advantage_lower_quantile = clip_advantage_lower_quantile
        self.clip_advantage_upper_quantile = clip_advantage_upper_quantile

    def loss(
        self,
        obs,
        chains_prev,
        chains_next,
        denoising_inds,
        returns,
        oldvalues,
        advantages,
        oldlogprobs,
        reward_horizon=4,
        act_offset=0,
    ):
        """
        PPO loss over denoising chain transitions.

        Args:
            obs: dict with state: (B, To, Do)
            chains_prev, chains_next: (B, Ta, Da)
            denoising_inds: (B,) indices into ft_denoising_steps
            returns, oldvalues, advantages: (B,)
            oldlogprobs: (B, Ta, Da)
            reward_horizon: action steps that backpropagate gradient
        Returns:
            (pg_loss, entropy_loss, v_loss, clipfrac, approx_kl, ratio_mean)
        """
        newlogprobs, entropy = self.get_logprobs_subsample(
            obs, chains_prev, chains_next, denoising_inds, get_ent=True,
        )
        entropy_loss = -entropy.mean()
        # No clamping — with low min_logprob_std matching sampling std,
        # per-element logprobs can exceed 2. Clamping kills ratio sensitivity.
        # The logratio (new - old) is bounded even without clamping.

        # Only backpropagate through executed action steps (positions act_offset..act_offset+reward_horizon-1)
        newlogprobs = newlogprobs[:, act_offset:act_offset + reward_horizon, :]
        oldlogprobs = oldlogprobs[:, act_offset:act_offset + reward_horizon, :]

        # Aggregate logprobs: mean over action dims and horizon
        newlogprobs = newlogprobs.mean(dim=(-1, -2)).view(-1)
        oldlogprobs = oldlogprobs.mean(dim=(-1, -2)).view(-1)

        # Normalize advantages
        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Clip advantages by quantile
        adv_min = torch.quantile(advantages, self.clip_advantage_lower_quantile)
        adv_max = torch.quantile(advantages, self.clip_advantage_upper_quantile)
        advantages = advantages.clamp(min=adv_min, max=adv_max)

        # Denoising discount: gamma^(K-i-1) for each denoising step
        discount = torch.tensor(
            [self.gamma_denoising ** (self.ft_denoising_steps - i - 1)
             for i in denoising_inds]
        ).to(self.device)
        advantages = advantages * discount

        # PPO ratio
        logratio = newlogprobs - oldlogprobs
        ratio = logratio.exp()

        # Exponentially interpolated clip coefficient across denoising steps
        t = (denoising_inds.float() / max(self.ft_denoising_steps - 1, 1)).to(
            self.device
        )
        if self.ft_denoising_steps > 1:
            clip_ploss_coef = self.clip_ploss_coef_base + (
                self.clip_ploss_coef - self.clip_ploss_coef_base
            ) * (torch.exp(self.clip_ploss_coef_rate * t) - 1) / (
                math.exp(self.clip_ploss_coef_rate) - 1
            )
        else:
            clip_ploss_coef = t

        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > clip_ploss_coef).float().mean().item()

        # Clipped surrogate loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio, 1 - clip_ploss_coef, 1 + clip_ploss_coef
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalues = self.critic(obs).view(-1)
        if self.clip_vloss_coef is not None:
            v_loss_unclipped = (newvalues - returns) ** 2
            v_clipped = oldvalues + torch.clamp(
                newvalues - oldvalues,
                -self.clip_vloss_coef,
                self.clip_vloss_coef,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((newvalues - returns) ** 2).mean()

        return (
            pg_loss,
            entropy_loss,
            v_loss,
            clipfrac,
            approx_kl.item(),
            ratio.mean().item(),
        )
