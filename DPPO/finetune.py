"""
DPPO RL Finetune: PPO-based finetuning of pretrained diffusion policy.

Aligned with https://github.com/irom-princeton/dppo (the reference implementation).

Key design: The last ft_denoising_steps of the denoising chain use a trainable
actor_ft (initialized from pretrained). Earlier steps use the frozen pretrained actor.
PPO treats each denoising step as an MDP transition.

Usage:
    python -m DPPO.finetune --pretrain_checkpoint runs/dppo_pretrain/dppo_1000traj_unet_5k/ckpt_1500.pt
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, List
from torch.utils.tensorboard import SummaryWriter

import sys
import tyro

from DPPO.model.mlp import DiffusionMLP
from DPPO.model.unet_wrapper import DiffusionUNet
from DPPO.model.critic import CriticObs
from DPPO.model.diffusion_ppo import PPODiffusion
from DPPO.make_env import make_train_envs
from DPPO.reward_scaler import RunningRewardScaler


@dataclass
class Args:
    # Pretrained checkpoint
    pretrain_checkpoint: str = "runs/dppo_pretrain/dppo_1000traj_unet_T20_5k/best.pt"

    # Environment
    env_id: str = "PickCube-v1"
    control_mode: str = "pd_ee_delta_pos"
    max_episode_steps: int = 100
    n_envs: int = 200
    sim_backend: str = "gpu"  # "gpu" for fast CUDA envs, "cpu" for CPU

    # DPPO specific (overridden from checkpoint at runtime)
    ft_denoising_steps: int = 10
    denoising_steps: int = 20
    horizon_steps: int = 16
    cond_steps: int = 2
    act_steps: int = 8

    # DDIM (disabled by default: DDPM-trained models need retraining for DDIM)
    use_ddim: bool = False
    ddim_steps: int = 5

    # Critic
    critic_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    critic_activation: str = "Mish"

    # RL
    n_train_itr: int = 301
    n_steps: int = 200
    gamma: float = 0.999
    gae_lambda: float = 0.95
    reward_scale_running: bool = True
    reward_scale_const: float = 1.0
    update_epochs: int = 10
    minibatch_size: int = 500
    target_kl: float = 1.0
    grad_accumulate: int = 1

    # PPO
    clip_ploss_coef: float = 0.01
    clip_ploss_coef_base: float = 1e-3
    clip_ploss_coef_rate: float = 3.0
    clip_vloss_coef: Optional[float] = None
    gamma_denoising: float = 0.99
    norm_adv: bool = True
    clip_adv_min: Optional[float] = None  # Clamp advantages >= this. 0 = only positive advantages.
    vf_coef: float = 0.5

    # Optimization
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    max_grad_norm: float = 1.0
    n_critic_warmup_itr: int = 2

    # Exploration noise
    min_sampling_denoising_std: float = 0.01
    min_logprob_denoising_std: float = 0.1

    # Eval
    eval_freq: int = 10
    eval_n_rounds: int = 3
    """Number of eval rounds (each round = n_envs episodes). Total eval episodes = eval_n_rounds * n_envs."""

    # Seed pool filtering (train only on easy states)
    seed_pool_path: Optional[str] = None
    """Path to .npz from dp_p_success_cpu.py with p_success and seeds arrays."""
    seed_pool_threshold: float = 0.5
    """P(success) threshold: only train on seeds with P > threshold."""

    # Logging
    exp_name: Optional[str] = None
    seed: int = 0
    save_dir: str = "runs/dppo_finetune"


def main():
    args = tyro.cli(Args)

    if args.exp_name is None:
        args.exp_name = (
            f"dppo_ft_{args.env_id}_T{args.denoising_steps}_K{args.ft_denoising_steps}"
            f"_envs{args.n_envs}_steps{args.n_steps}"
        )

    run_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained checkpoint
    ckpt = torch.load(args.pretrain_checkpoint, map_location=device, weights_only=False)
    obs_dim = ckpt["obs_dim"]
    action_dim = ckpt["action_dim"]

    # Read normalization flags
    no_obs_norm = ckpt.get("no_obs_norm", False)
    no_action_norm = ckpt.get("no_action_norm", False)

    # Load normalization stats (only used when normalization is enabled)
    obs_min = obs_max = action_min = action_max = None
    if not no_obs_norm:
        obs_min = ckpt["obs_min"].to(device)
        obs_max = ckpt["obs_max"].to(device)
    if not no_action_norm:
        action_min = ckpt["action_min"].to(device)
        action_max = ckpt["action_max"].to(device)

    # Override architecture params from checkpoint
    pretrain_args = ckpt.get("args", {})
    args.denoising_steps = pretrain_args.get("denoising_steps", args.denoising_steps)
    args.horizon_steps = pretrain_args.get("horizon_steps", args.horizon_steps)
    args.cond_steps = pretrain_args.get("cond_steps", args.cond_steps)
    args.act_steps = pretrain_args.get("act_steps", args.act_steps)
    network_type = pretrain_args.get("network_type", "mlp")

    # Validate control_mode matches pretrained checkpoint
    ckpt_control_mode = pretrain_args.get("control_mode")
    if ckpt_control_mode is not None and ckpt_control_mode != args.control_mode:
        raise ValueError(
            f"control_mode mismatch: checkpoint was trained with '{ckpt_control_mode}' "
            f"but finetune is configured with '{args.control_mode}'. "
            f"Pass --control_mode {ckpt_control_mode} to fix."
        )

    cond_dim = obs_dim * args.cond_steps
    # UNet predicts for full horizon; executable actions start at cond_steps-1
    act_offset = args.cond_steps - 1 if network_type == "unet" else 0

    print(f"Pretrained: {args.pretrain_checkpoint}")
    print(f"  network_type={network_type}, obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"  denoising_steps={args.denoising_steps}, ft_denoising_steps={args.ft_denoising_steps}")
    print(f"  horizon_steps={args.horizon_steps}, cond_steps={args.cond_steps}, act_steps={args.act_steps}")
    print(f"  act_offset={act_offset}, no_obs_norm={no_obs_norm}, no_action_norm={no_action_norm}")

    # Build actor network (match pretrained architecture)
    if network_type == "unet":
        actor = DiffusionUNet(
            action_dim=action_dim,
            horizon_steps=args.horizon_steps,
            cond_dim=cond_dim,
            diffusion_step_embed_dim=pretrain_args.get("diffusion_step_embed_dim", 64),
            down_dims=pretrain_args.get("unet_dims", [64, 128, 256]),
            n_groups=pretrain_args.get("n_groups", 8),
        )
    else:
        actor = DiffusionMLP(
            action_dim=action_dim,
            horizon_steps=args.horizon_steps,
            cond_dim=cond_dim,
            time_dim=pretrain_args.get("time_dim", 16),
            mlp_dims=pretrain_args.get("mlp_dims", [512, 512, 512]),
            activation_type=pretrain_args.get("activation_type", "Mish"),
            residual_style=pretrain_args.get("residual_style", True),
        )

    # Build critic V(s)
    critic = CriticObs(
        cond_dim=cond_dim,
        mlp_dims=args.critic_dims,
        activation_type=args.critic_activation,
        residual_style=True,
    )

    # Build PPODiffusion model (randn_clip_value=3 per original DPPO)
    model = PPODiffusion(
        actor=actor,
        critic=critic,
        ft_denoising_steps=args.ft_denoising_steps,
        min_sampling_denoising_std=args.min_sampling_denoising_std,
        min_logprob_denoising_std=args.min_logprob_denoising_std,
        use_ddim=args.use_ddim,
        ddim_steps=args.ddim_steps if args.use_ddim else None,
        horizon_steps=args.horizon_steps,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        denoising_steps=args.denoising_steps,
        denoised_clip_value=1.0,
        randn_clip_value=3,
        final_action_clip_value=1.0,
        predict_epsilon=True,
        gamma_denoising=args.gamma_denoising,
        clip_ploss_coef=args.clip_ploss_coef,
        clip_ploss_coef_base=args.clip_ploss_coef_base,
        clip_ploss_coef_rate=args.clip_ploss_coef_rate,
        clip_vloss_coef=args.clip_vloss_coef,
        norm_adv=args.norm_adv,
    )

    # Load pretrained weights into actor (via self.network alias)
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"], strict=False)
    else:
        model.load_state_dict(ckpt["model"], strict=False)

    # Fix NaN eta_logit from checkpoints trained with AdamW weight decay
    model._sanitize_eta()

    # CRITICAL: load_state_dict only updates self.network/self.actor (same object).
    # self.actor_ft was deepcopied BEFORE loading, so it still has random weights.
    # Must copy pretrained weights to actor_ft so finetuning starts from pretrained.
    model.actor_ft.load_state_dict(model.actor.state_dict())

    # Freeze original actor
    for param in model.actor.parameters():
        param.requires_grad = False

    n_actor_params = sum(p.numel() for p in model.actor_ft.parameters() if p.requires_grad)
    n_critic_params = sum(p.numel() for p in model.critic.parameters() if p.requires_grad)
    print(f"Finetuned actor params: {n_actor_params:,}, Critic params: {n_critic_params:,}")

    # Optimizers: separate for actor and critic
    actor_optimizer = torch.optim.Adam(
        model.actor_ft.parameters(), lr=args.actor_lr,
    )
    critic_optimizer = torch.optim.Adam(
        model.critic.parameters(), lr=args.critic_lr,
    )

    # Running reward scaler (normalizes rewards by running std of discounted returns)
    reward_scaler = RunningRewardScaler(gamma=args.gamma) if args.reward_scale_running else None

    # Load seed pool for filtered training
    seed_pool = None
    if args.seed_pool_path is not None:
        data = np.load(args.seed_pool_path)
        p_success = data["p_success"]
        seeds = data["seeds"]
        mask = p_success > args.seed_pool_threshold
        seed_pool = seeds[mask]
        print(f"Seed pool: {len(seed_pool)}/{len(seeds)} seeds with P(success)>{args.seed_pool_threshold}")
        if len(seed_pool) == 0:
            raise ValueError(f"No seeds pass threshold {args.seed_pool_threshold}. "
                             f"Max P(success)={p_success.max():.3f}")
        if args.sim_backend == "gpu":
            print("WARNING: seed_pool only works with CPU envs (sim_backend=cpu)")
            seed_pool = None

    # Train environments (obs_history managed manually, no FrameStack)
    use_gpu_env = args.sim_backend == "gpu"
    train_envs = make_train_envs(
        env_id=args.env_id,
        num_envs=args.n_envs,
        sim_backend=args.sim_backend,
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        seed_pool=seed_pool,
    )

    n_decision_steps = args.n_steps
    K = args.ft_denoising_steps
    assert K > 1, (
        f"ft_denoising_steps must be > 1 (got {K}). "
        f"K=1 causes PPO clip coefficient to degenerate to 0."
    )

    def normalize_obs(obs):
        """Apply obs normalization (min-max to [-1,1]) if enabled."""
        if no_obs_norm:
            return obs
        return (obs - obs_min) / (obs_max - obs_min + 1e-8) * 2.0 - 1.0

    def denormalize_actions(actions):
        """Denormalize actions from [-1,1] to env space if enabled."""
        if no_action_norm:
            return actions
        return (actions + 1.0) / 2.0 * (action_max - action_min) + action_min

    @torch.no_grad()
    def evaluate_gpu_inline(n_rounds=5):
        """Eval using training env: reset + deterministic rollout, track success_once."""
        model.eval()
        total_success = 0
        total_eps = 0
        for _ in range(n_rounds):
            obs_r, _ = train_envs.reset()
            if isinstance(obs_r, np.ndarray):
                obs_r = torch.from_numpy(obs_r).float().to(device)
            else:
                obs_r = obs_r.float().to(device)
            obs_h = obs_r.unsqueeze(1).repeat(1, args.cond_steps, 1)
            success_once = torch.zeros(args.n_envs, dtype=torch.bool, device=device)
            for step in range(args.max_episode_steps // args.act_steps + 1):
                cond_eval = {"state": normalize_obs(obs_h)}
                samples_eval = model(cond_eval, deterministic=True)
                ac_eval = denormalize_actions(samples_eval.trajectories)
                for a_idx in range(args.act_steps):
                    act_idx = act_offset + a_idx
                    action_eval = ac_eval[:, min(act_idx, ac_eval.shape[1] - 1)]
                    if use_gpu_env:
                        obs_new_eval, rew_eval, term_eval, trunc_eval, _ = train_envs.step(action_eval)
                        obs_new_eval = obs_new_eval.float()
                        rew_eval = rew_eval.float()
                        term_eval = term_eval.bool()
                        trunc_eval = trunc_eval.bool()
                    else:
                        obs_new_eval, rew_eval, term_eval, trunc_eval, _ = train_envs.step(action_eval.cpu().numpy())
                        obs_new_eval = torch.from_numpy(np.array(obs_new_eval)).float().to(device)
                        rew_eval = torch.from_numpy(np.array(rew_eval)).float().to(device)
                        term_eval = torch.from_numpy(np.array(term_eval)).bool().to(device)
                        trunc_eval = torch.from_numpy(np.array(trunc_eval)).bool().to(device)
                    success_once |= (rew_eval > 0.5).bool()
                    rm = term_eval | trunc_eval
                    if rm.any():
                        obs_h[rm] = obs_new_eval[rm].unsqueeze(1).repeat(1, args.cond_steps, 1)
                    obs_h[~rm] = torch.cat([obs_h[~rm, 1:], obs_new_eval[~rm].unsqueeze(1)], dim=1)
            total_success += success_once.sum().item()
            total_eps += args.n_envs
        return total_success / max(total_eps, 1)

    best_sr = -1.0
    global_step = 0

    batch_size = args.n_envs * n_decision_steps
    print(f"\nStarting DPPO finetuning for {args.n_train_itr} iterations...")
    print(f"  n_envs={args.n_envs}, n_steps={n_decision_steps}, act_steps={args.act_steps}, sim_backend={args.sim_backend}")
    print(f"  batch_size={batch_size}, total_samples={batch_size * K}")
    print(f"  gamma={args.gamma}, gamma_denoising={args.gamma_denoising}")
    if args.use_ddim:
        print(f"  DDIM: ddim_steps={args.ddim_steps}, ft_denoising_steps={args.ft_denoising_steps}")
    else:
        print(f"  DDPM: denoising_steps={args.denoising_steps}, ft_denoising_steps={args.ft_denoising_steps}")
    print(f"  min_sampling_std={args.min_sampling_denoising_std}, min_logprob_std={args.min_logprob_denoising_std}")
    print(f"  reward_scale_running={args.reward_scale_running}, reward_scale_const={args.reward_scale_const}")
    print(f"  grad_accumulate={args.grad_accumulate}, actor_lr={args.actor_lr}")
    if args.n_critic_warmup_itr > 0:
        print(f"  critic warmup: {args.n_critic_warmup_itr} iterations (actor frozen)")

    obs_raw, _ = train_envs.reset()
    if isinstance(obs_raw, np.ndarray):
        obs_raw = torch.from_numpy(obs_raw).float().to(device)
    else:
        obs_raw = obs_raw.float().to(device)

    # Obs history buffer: (n_envs, cond_steps, obs_dim)
    obs_history = obs_raw.unsqueeze(1).repeat(1, args.cond_steps, 1)

    for iteration in range(1, args.n_train_itr + 1):
        t_start = time.time()
        model.eval()
        is_warmup = iteration <= args.n_critic_warmup_itr

        # ===== 1. ROLLOUT =====
        obs_trajs = []          # (n_steps, n_envs, cond_steps, obs_dim)
        chains_trajs = []       # (n_steps, n_envs, K+1, horizon_steps, action_dim)
        reward_trajs = []       # (n_steps, n_envs)
        done_trajs = []         # (n_steps, n_envs) terminated | truncated
        value_trajs = []        # (n_steps, n_envs)

        n_success_rollout = 0

        for step in range(n_decision_steps):
            obs_norm = normalize_obs(obs_history)
            cond = {"state": obs_norm}

            with torch.no_grad():
                value = model.critic(cond).squeeze(-1)
                samples = model(cond, deterministic=False, return_chain=True)
                action_chunk = samples.trajectories
                chains = samples.chains

            obs_trajs.append(obs_norm.clone())
            chains_trajs.append(chains.clone())
            value_trajs.append(value.clone())

            # Denormalize actions for env
            action_chunk_denorm = denormalize_actions(action_chunk)

            # Execute act_steps in env (starting from act_offset)
            step_reward = torch.zeros(args.n_envs, device=device)
            step_done = torch.zeros(args.n_envs, dtype=torch.bool, device=device)

            for a_idx in range(args.act_steps):
                act_idx = act_offset + a_idx
                if act_idx < action_chunk_denorm.shape[1]:
                    action = action_chunk_denorm[:, act_idx]
                else:
                    action = action_chunk_denorm[:, -1]

                if use_gpu_env:
                    obs_new, reward, terminated, truncated, info = train_envs.step(action)
                    obs_new = obs_new.float()
                    reward_t = reward.float()
                    term_t = terminated.bool()
                    trunc_t = truncated.bool()
                else:
                    action_np = action.cpu().numpy()
                    obs_new, reward, terminated, truncated, info = train_envs.step(action_np)
                    obs_new = torch.from_numpy(obs_new).float().to(device)
                    reward_t = torch.from_numpy(np.array(reward)).float().to(device)
                    term_t = torch.from_numpy(np.array(terminated)).bool().to(device)
                    trunc_t = torch.from_numpy(np.array(truncated)).bool().to(device)

                step_reward += reward_t * (~step_done).float()
                step_done = step_done | term_t | trunc_t
                n_success_rollout += (reward_t > 0.5).sum().item()

                # Reset obs_history for envs that terminated/truncated
                reset_mask = term_t | trunc_t
                if reset_mask.any():
                    # For reset envs, fill obs_history with the new initial obs
                    obs_history[reset_mask] = obs_new[reset_mask].unsqueeze(1).repeat(1, args.cond_steps, 1)
                # Normal shift for non-reset envs
                obs_history[~reset_mask] = torch.cat(
                    [obs_history[~reset_mask, 1:], obs_new[~reset_mask].unsqueeze(1)], dim=1
                )
                global_step += args.n_envs

            reward_trajs.append(step_reward)
            done_trajs.append(step_done.float())

        rewards = torch.stack(reward_trajs)     # (n_steps, n_envs)
        dones = torch.stack(done_trajs)         # (n_steps, n_envs)
        values = torch.stack(value_trajs)       # (n_steps, n_envs)

        # ===== 2. REWARD SCALING =====
        if reward_scaler is not None:
            rewards = reward_scaler.update_and_scale(rewards, dones)
        rewards = rewards * args.reward_scale_const

        # ===== 3. GAE COMPUTATION =====
        with torch.no_grad():
            obs_norm_last = normalize_obs(obs_history)
            next_value = model.critic({"state": obs_norm_last}).squeeze(-1)

        advantages = torch.zeros_like(rewards)
        lastgaelam = 0

        for t in reversed(range(n_decision_steps)):
            if t == n_decision_steps - 1:
                next_not_done = 1.0 - dones[t]
                nextvalues = next_value
            else:
                next_not_done = 1.0 - dones[t]
                nextvalues = values[t + 1]

            delta = rewards[t] + args.gamma * nextvalues * next_not_done - values[t]
            advantages[t] = lastgaelam = (
                delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam
            )

        returns = advantages + values

        # ===== 4. COMPUTE LOGPROBS (after rollout, before PPO update) =====
        # Process in batches to prevent OOM (MLP is small, so use larger batches)
        all_logprobs = []
        logprob_batch_size = 2048
        obs_stacked = torch.stack(obs_trajs)       # (n_steps, n_envs, cond_steps, obs_dim)
        chains_stacked = torch.stack(chains_trajs)  # (n_steps, n_envs, K+1, horizon, act_dim)

        for i in range(0, n_decision_steps * args.n_envs, logprob_batch_size):
            end = min(i + logprob_batch_size, n_decision_steps * args.n_envs)
            step_inds = torch.arange(i, end)
            s_idx = step_inds // args.n_envs
            e_idx = step_inds % args.n_envs

            batch_obs = obs_stacked[s_idx, e_idx]      # (B, cond_steps, obs_dim)
            batch_chains = chains_stacked[s_idx, e_idx]  # (B, K+1, horizon, act_dim)

            with torch.no_grad():
                batch_lp = model.get_logprobs({"state": batch_obs}, batch_chains)
                # (B*K, horizon, act_dim) -> (B, K, horizon, act_dim)
                batch_lp = batch_lp.reshape(end - i, K, args.horizon_steps, action_dim)
            all_logprobs.append(batch_lp)

        all_logprobs_flat = torch.cat(all_logprobs, dim=0)  # (N, K, horizon, act_dim)

        # ===== 5. PPO UPDATE =====
        N = n_decision_steps * args.n_envs
        b_obs = obs_stacked.reshape(N, args.cond_steps, obs_dim)
        b_chains = chains_stacked.reshape(N, K + 1, args.horizon_steps, action_dim)
        b_logprobs = all_logprobs_flat  # (N, K, horizon, act_dim)
        b_advantages = advantages.reshape(-1)
        if args.clip_adv_min is not None:
            b_advantages = torch.clamp(b_advantages, min=args.clip_adv_min)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        total_samples = N * K

        model.train()
        grad_acc = args.grad_accumulate
        for epoch in range(args.update_epochs):
            perm = torch.randperm(total_samples)

            epoch_pg_loss = 0.0
            epoch_v_loss = 0.0
            epoch_kl = 0.0
            epoch_ratio_mean = 0.0
            n_mb = 0
            acc_count = 0
            early_stop = False

            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

            for mb_start in range(0, total_samples, args.minibatch_size):
                mb_inds = perm[mb_start : mb_start + args.minibatch_size]
                if len(mb_inds) == 0:
                    continue

                sample_inds = mb_inds // K
                denoise_inds = mb_inds % K

                mb_obs = {"state": b_obs[sample_inds]}
                mb_chains_prev = b_chains[sample_inds, denoise_inds]
                mb_chains_next = b_chains[sample_inds, denoise_inds + 1]
                mb_old_logprobs = b_logprobs[sample_inds, denoise_inds]
                mb_advantages = b_advantages[sample_inds]
                mb_returns = b_returns[sample_inds]
                mb_oldvalues = b_values[sample_inds]

                pg_loss, entropy_loss, v_loss, clipfrac, approx_kl, ratio_mean = model.loss(
                    obs=mb_obs,
                    chains_prev=mb_chains_prev,
                    chains_next=mb_chains_next,
                    denoising_inds=denoise_inds,
                    returns=mb_returns,
                    oldvalues=mb_oldvalues,
                    advantages=mb_advantages,
                    oldlogprobs=mb_old_logprobs,
                    reward_horizon=args.act_steps,
                    act_offset=act_offset,
                )

                # During critic warmup, only update critic
                if is_warmup:
                    loss = args.vf_coef * v_loss / grad_acc
                else:
                    loss = (pg_loss + args.vf_coef * v_loss) / grad_acc

                loss.backward()
                acc_count += 1

                # Step optimizers after accumulating grad_accumulate batches
                if acc_count >= grad_acc:
                    if not is_warmup:
                        nn.utils.clip_grad_norm_(model.actor_ft.parameters(), args.max_grad_norm)
                    nn.utils.clip_grad_norm_(model.critic.parameters(), args.max_grad_norm)

                    if not is_warmup:
                        actor_optimizer.step()
                    critic_optimizer.step()

                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()
                    acc_count = 0

                epoch_pg_loss += pg_loss.item()
                epoch_v_loss += v_loss.item()
                epoch_kl += approx_kl
                epoch_ratio_mean += ratio_mean
                n_mb += 1

                if not is_warmup and args.target_kl is not None and approx_kl > args.target_kl:
                    early_stop = True
                    break

            # Flush remaining accumulated gradients
            if acc_count > 0:
                if not is_warmup:
                    nn.utils.clip_grad_norm_(model.actor_ft.parameters(), args.max_grad_norm)
                nn.utils.clip_grad_norm_(model.critic.parameters(), args.max_grad_norm)
                if not is_warmup:
                    actor_optimizer.step()
                critic_optimizer.step()
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()

            if early_stop:
                break

        # ===== LOGGING =====
        avg_reward = rewards.sum(0).mean().item()
        avg_pg_loss = epoch_pg_loss / max(n_mb, 1)
        avg_v_loss = epoch_v_loss / max(n_mb, 1)
        avg_kl = epoch_kl / max(n_mb, 1)
        avg_ratio = epoch_ratio_mean / max(n_mb, 1)

        # Advantage stats
        adv_mean = b_advantages.mean().item()
        adv_std = b_advantages.std().item()
        adv_pos_frac = (b_advantages > 0).float().mean().item()

        writer.add_scalar("train/reward", avg_reward, iteration)
        writer.add_scalar("train/pg_loss", avg_pg_loss, iteration)
        writer.add_scalar("train/v_loss", avg_v_loss, iteration)
        writer.add_scalar("train/approx_kl", avg_kl, iteration)
        writer.add_scalar("train/rollout_successes", n_success_rollout, iteration)
        writer.add_scalar("train/global_step", global_step, iteration)
        writer.add_scalar("train/ratio_mean", avg_ratio, iteration)
        writer.add_scalar("train/adv_pos_frac", adv_pos_frac, iteration)

        t_elapsed = time.time() - t_start

        if iteration % 5 == 0 or iteration <= 5:
            warmup_tag = " [WARMUP]" if is_warmup else ""
            print(
                f"Iter {iteration}/{args.n_train_itr}{warmup_tag} | "
                f"r={avg_reward:.3f} | succ={n_success_rollout} | pg={avg_pg_loss:.6f} | "
                f"v={avg_v_loss:.4f} | kl={avg_kl:.6f} | ratio={avg_ratio:.4f} | "
                f"adv: mean={adv_mean:.4f} std={adv_std:.4f} pos={adv_pos_frac:.2f} | "
                f"ep={epoch+1} | time={t_elapsed:.1f}s"
            )

        # ===== EVALUATE =====
        if iteration == 1 or iteration % args.eval_freq == 0:
            sr_once = evaluate_gpu_inline(n_rounds=args.eval_n_rounds)
            writer.add_scalar("eval/success_once", sr_once, iteration)
            print(f"  EVAL @ iter {iteration}: gpu_sr={sr_once:.3f}")

            # Reset training env state after eval (eval reuses train_envs)
            obs_raw_r, _ = train_envs.reset()
            if isinstance(obs_raw_r, np.ndarray):
                obs_raw_r = torch.from_numpy(obs_raw_r).float().to(device)
            else:
                obs_raw_r = obs_raw_r.float().to(device)
            obs_history[:] = obs_raw_r.unsqueeze(1).repeat(1, args.cond_steps, 1)

            sr = sr_once
            if sr > best_sr:
                best_sr = sr
                save_ckpt = {
                    "model": model.state_dict(),
                    "obs_min": obs_min,
                    "obs_max": obs_max,
                    "action_min": action_min,
                    "action_max": action_max,
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "args": vars(args),
                    "iteration": iteration,
                    "no_obs_norm": no_obs_norm,
                    "no_action_norm": no_action_norm,
                    "network_type": network_type,
                    "pretrain_args": pretrain_args,
                }
                torch.save(save_ckpt, os.path.join(run_dir, "best.pt"))
                print(f"  Saved best checkpoint (sr_once={best_sr:.3f})")

    # Save final
    save_ckpt = {
        "model": model.state_dict(),
        "obs_min": obs_min,
        "obs_max": obs_max,
        "action_min": action_min,
        "action_max": action_max,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "args": vars(args),
        "iteration": args.n_train_itr,
        "no_obs_norm": no_obs_norm,
        "no_action_norm": no_action_norm,
        "network_type": network_type,
        "pretrain_args": pretrain_args,
    }
    torch.save(save_ckpt, os.path.join(run_dir, "final.pt"))
    print(f"Finetuning complete. Best sr_once={best_sr:.3f}")

    train_envs.close()
    writer.close()


if __name__ == "__main__":
    main()
