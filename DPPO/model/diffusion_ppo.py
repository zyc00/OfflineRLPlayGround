"""
VPGDiffusion + PPODiffusion: Diffusion policy with PPO finetuning.

Ported from https://github.com/irom-princeton/dppo
"""

import copy
import math
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from DPPO.model.diffusion import DiffusionModel, Sample, extract, make_timesteps


class VPGDiffusion(DiffusionModel):
    """Diffusion model with dual actor for RL finetuning."""

    def __init__(
        self,
        actor,
        critic,
        ft_denoising_steps,
        min_sampling_denoising_std=0.1,
        min_logprob_denoising_std=0.1,
        network_path=None,
        **kwargs,
    ):
        super().__init__(network=actor, **kwargs)
        assert ft_denoising_steps <= self.denoising_steps
        self.ft_denoising_steps = ft_denoising_steps
        self.min_sampling_denoising_std = min_sampling_denoising_std
        self.min_logprob_denoising_std = min_logprob_denoising_std

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
            # Re-freeze original actor after loading
            for param in self.actor.parameters():
                param.requires_grad = False

    def p_mean_var(self, x, t, cond, use_base_policy=False, deterministic=False):
        """Overridden: routes denoising steps between frozen and finetuned actor."""
        noise = self.actor(x, t, cond=cond)

        # Determine which samples use finetuned actor
        ft_indices = torch.where(t < self.ft_denoising_steps)[0]
        actor = self.actor if use_base_policy else self.actor_ft

        if len(ft_indices) > 0:
            cond_ft = {key: cond[key][ft_indices] for key in cond}
            noise_ft = actor(x[ft_indices], t[ft_indices], cond=cond_ft)
            noise[ft_indices] = noise_ft

        # Predict x_0
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
    def forward(
        self,
        cond,
        deterministic=False,
        return_chain=True,
        use_base_policy=False,
    ):
        """
        Forward pass with chain tracking for RL training.

        Returns:
            Sample(trajectories=(B, Ta, Da), chains=(B, K+1, Ta, Da))
        """
        device = self.betas.device
        sample_data = cond["state"]
        B = len(sample_data)
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
                deterministic=deterministic,
            )
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

            if return_chain and t <= self.ft_denoising_steps:
                chain.append(x)

        if return_chain:
            chain = torch.stack(chain, dim=1)
        return Sample(x, chain)

    def get_logprobs(self, cond, chains, get_ent=False, use_base_policy=False):
        """
        Calculate logprobs of the entire denoising chain.

        Args:
            cond: dict with state: (B, To, Do)
            chains: (B, K+1, Ta, Da)
        Returns:
            logprobs: (B*K, Ta, Da)
        """
        # Repeat cond for each denoising step
        cond_rep = {
            key: cond[key]
            .unsqueeze(1)
            .repeat(1, self.ft_denoising_steps, *(1,) * (cond[key].ndim - 1))
            .flatten(start_dim=0, end_dim=1)
            for key in cond
        }

        # Build timestep indices: ft_denoising_steps-1, ..., 0, repeated B times
        t_single = torch.arange(
            start=self.ft_denoising_steps - 1, end=-1, step=-1, device=self.device,
        )
        t_all = t_single.repeat(chains.shape[0], 1).flatten()

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
            denoising_inds: (B,) indices into ft_denoising_steps
        Returns:
            logprobs: (B, Ta, Da)
        """
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
        newlogprobs = newlogprobs.clamp(min=-5, max=2)
        oldlogprobs = oldlogprobs.clamp(min=-5, max=2)

        # Only backpropagate through executed action steps
        newlogprobs = newlogprobs[:, :reward_horizon, :]
        oldlogprobs = oldlogprobs[:, :reward_horizon, :]

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
