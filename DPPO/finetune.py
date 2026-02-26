"""
DPPO RL Finetune: PPO-based finetuning of pretrained diffusion policy.

Usage:
    python -m DPPO.finetune --pretrain_checkpoint runs/dppo_pretrain/best.pt --env_id PickCube-v1
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, List
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

import tyro

from DPPO.model.mlp import DiffusionMLP
from DPPO.model.critic import CriticObs
from DPPO.model.diffusion_ppo import PPODiffusion
from DPPO.make_env import make_train_envs, make_eval_envs
from DPPO.evaluate import evaluate


@dataclass
class Args:
    # Pretrained checkpoint
    pretrain_checkpoint: str = "runs/dppo_pretrain/best.pt"

    # Environment
    env_id: str = "PickCube-v1"
    control_mode: str = "pd_joint_delta_pos"
    max_episode_steps: int = 100
    n_envs: int = 50
    n_eval_envs: int = 10
    sim_backend: str = "physx_cpu"

    # DPPO specific
    ft_denoising_steps: int = 10
    denoising_steps: int = 20
    horizon_steps: int = 4
    cond_steps: int = 1
    act_steps: int = 4

    # Network (must match pretrained)
    mlp_dims: List[int] = field(default_factory=lambda: [512, 512, 512])
    time_dim: int = 16
    activation_type: str = "Mish"
    residual_style: bool = True

    # Critic
    critic_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    critic_activation: str = "Mish"

    # RL
    n_train_itr: int = 500
    n_steps: int = 25
    gamma: float = 0.99
    gae_lambda: float = 0.95
    reward_scale: float = 1.0
    update_epochs: int = 5
    minibatch_size: int = 256
    target_kl: float = 1.0

    # PPO
    clip_ploss_coef: float = 0.01
    clip_ploss_coef_base: float = 1e-3
    clip_ploss_coef_rate: float = 3.0
    clip_vloss_coef: Optional[float] = None
    gamma_denoising: float = 1.0
    norm_adv: bool = True
    vf_coef: float = 0.5

    # Optimization
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    max_grad_norm: float = 1.0

    # Exploration
    min_sampling_denoising_std: float = 0.1
    min_logprob_denoising_std: float = 0.1

    # Eval
    eval_freq: int = 10
    num_eval_episodes: int = 100

    # Logging
    exp_name: Optional[str] = None
    seed: int = 0
    save_dir: str = "runs/dppo_finetune"


def main():
    args = tyro.cli(Args)

    if args.exp_name is None:
        args.exp_name = (
            f"dppo_ft_{args.env_id}_K{args.ft_denoising_steps}"
            f"_clip{args.clip_ploss_coef}_envs{args.n_envs}"
        )

    run_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    # Load pretrained checkpoint
    ckpt = torch.load(args.pretrain_checkpoint, map_location=device, weights_only=False)
    obs_dim = ckpt["obs_dim"]
    action_dim = ckpt["action_dim"]
    obs_mean = ckpt["obs_mean"].to(device)
    obs_std = ckpt["obs_std"].to(device)
    action_min = ckpt["action_min"].to(device)
    action_max = ckpt["action_max"].to(device)

    # Override network architecture from checkpoint to ensure compatibility
    pretrain_args = ckpt.get("args", {})
    for key in ["denoising_steps", "horizon_steps", "cond_steps",
                 "mlp_dims", "time_dim", "activation_type", "residual_style"]:
        if key in pretrain_args:
            setattr(args, key, pretrain_args[key])

    cond_dim = obs_dim * args.cond_steps
    print(f"obs_dim={obs_dim}, action_dim={action_dim}, cond_dim={cond_dim}")
    print(f"denoising_steps={args.denoising_steps}, ft_denoising_steps={args.ft_denoising_steps}")
    print(f"horizon_steps={args.horizon_steps}, act_steps={args.act_steps}")

    # Build actor network (same architecture as pretrained)
    actor = DiffusionMLP(
        action_dim=action_dim,
        horizon_steps=args.horizon_steps,
        cond_dim=cond_dim,
        time_dim=args.time_dim,
        mlp_dims=args.mlp_dims,
        activation_type=args.activation_type,
        residual_style=args.residual_style,
    )

    # Build critic
    critic = CriticObs(
        cond_dim=cond_dim,
        mlp_dims=args.critic_dims,
        activation_type=args.critic_activation,
        residual_style=False,
    )

    # Build PPODiffusion model
    model = PPODiffusion(
        actor=actor,
        critic=critic,
        ft_denoising_steps=args.ft_denoising_steps,
        min_sampling_denoising_std=args.min_sampling_denoising_std,
        min_logprob_denoising_std=args.min_logprob_denoising_std,
        horizon_steps=args.horizon_steps,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        denoising_steps=args.denoising_steps,
        denoised_clip_value=1.0,
        randn_clip_value=10,
        final_action_clip_value=1.0,
        predict_epsilon=True,
        # PPO specific
        gamma_denoising=args.gamma_denoising,
        clip_ploss_coef=args.clip_ploss_coef,
        clip_ploss_coef_base=args.clip_ploss_coef_base,
        clip_ploss_coef_rate=args.clip_ploss_coef_rate,
        clip_vloss_coef=args.clip_vloss_coef,
        norm_adv=args.norm_adv,
    )

    # Load pretrained weights into the model (actor + actor_ft)
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"], strict=False)
    else:
        model.load_state_dict(ckpt["model"], strict=False)
    # Re-freeze original actor
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

    # Environments
    train_envs = make_train_envs(
        env_id=args.env_id,
        num_envs=args.n_envs,
        sim_backend=args.sim_backend,
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
    )
    eval_envs = make_eval_envs(
        env_id=args.env_id,
        num_envs=args.n_eval_envs,
        sim_backend=args.sim_backend,
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed + 2000,
    )

    # Decision steps per rollout: each decision = act_steps env steps
    n_decision_steps = args.n_steps
    K = args.ft_denoising_steps

    # Storage
    all_obs = []          # (n_steps, n_envs, cond_steps, obs_dim)
    all_chains = []       # (n_steps, n_envs, K+1, horizon_steps, action_dim)
    all_rewards = []      # (n_steps, n_envs)
    all_dones = []        # (n_steps, n_envs)
    all_values = []       # (n_steps, n_envs)
    all_logprobs = []     # (n_steps, n_envs, K, horizon_steps, action_dim)

    best_sr = -1.0
    global_step = 0

    print(f"Starting DPPO finetuning for {args.n_train_itr} iterations...")
    print(f"  n_envs={args.n_envs}, n_steps={n_decision_steps}, act_steps={args.act_steps}")
    print(f"  batch_size per iter = {args.n_envs * n_decision_steps * K}")

    obs_raw, _ = train_envs.reset()
    obs_raw = torch.from_numpy(obs_raw).float().to(device) if isinstance(obs_raw, np.ndarray) else obs_raw.float().to(device)

    # Obs history buffer
    obs_history = obs_raw.unsqueeze(1).repeat(1, args.cond_steps, 1)

    for iteration in range(1, args.n_train_itr + 1):
        t_start = time.time()
        model.eval()

        # Clear storage
        all_obs.clear()
        all_chains.clear()
        all_rewards.clear()
        all_dones.clear()
        all_values.clear()
        all_logprobs.clear()

        # 1. ROLLOUT
        for step in range(n_decision_steps):
            # Normalize obs
            obs_norm = (obs_history - obs_mean) / obs_std
            cond = {"state": obs_norm}  # (n_envs, cond_steps, obs_dim)

            with torch.no_grad():
                # Get value
                value = model.critic(cond).squeeze(-1)  # (n_envs,)
                # Sample with chain
                samples = model(cond, deterministic=False, return_chain=True)
                action_chunk = samples.trajectories  # (n_envs, horizon_steps, action_dim)
                chains = samples.chains  # (n_envs, K+1, horizon_steps, action_dim)

                # Get logprobs for the chain
                logprobs = model.get_logprobs(cond, chains)
                # (n_envs * K, horizon_steps, action_dim) -> reshape
                logprobs = logprobs.reshape(
                    args.n_envs, K, args.horizon_steps, action_dim
                )

            # Store
            all_obs.append(obs_norm.clone())
            all_chains.append(chains.clone())
            all_values.append(value.clone())
            all_logprobs.append(logprobs.clone())

            # Denormalize actions for env
            action_chunk_denorm = (action_chunk + 1.0) / 2.0 * (action_max - action_min) + action_min

            # Execute act_steps in env
            step_reward = torch.zeros(args.n_envs, device=device)
            step_done = torch.zeros(args.n_envs, dtype=torch.bool, device=device)

            for a_idx in range(args.act_steps):
                if a_idx < action_chunk_denorm.shape[1]:
                    action = action_chunk_denorm[:, a_idx]
                else:
                    action = action_chunk_denorm[:, -1]

                action_np = action.cpu().numpy()
                obs_new, reward, terminated, truncated, info = train_envs.step(action_np)
                obs_new = torch.from_numpy(obs_new).float().to(device) if isinstance(obs_new, np.ndarray) else obs_new.float().to(device)
                reward_t = torch.from_numpy(np.array(reward)).float().to(device) if isinstance(reward, np.ndarray) else torch.tensor(reward).float().to(device)
                term_t = torch.from_numpy(np.array(terminated)).bool().to(device) if isinstance(terminated, np.ndarray) else torch.tensor(terminated).bool().to(device)
                trunc_t = torch.from_numpy(np.array(truncated)).bool().to(device) if isinstance(truncated, np.ndarray) else torch.tensor(truncated).bool().to(device)

                # Accumulate reward (don't count rewards after done)
                step_reward += reward_t * args.reward_scale * (~step_done).float()
                step_done = step_done | term_t | trunc_t

                # Update obs history
                obs_history = torch.cat([obs_history[:, 1:], obs_new.unsqueeze(1)], dim=1)
                global_step += args.n_envs

            all_rewards.append(step_reward)
            all_dones.append(step_done.float())

        # 2. GAE COMPUTATION
        # Bootstrap value from last obs
        with torch.no_grad():
            obs_norm_last = (obs_history - obs_mean) / obs_std
            cond_last = {"state": obs_norm_last}
            next_value = model.critic(cond_last).squeeze(-1)  # (n_envs,)

        rewards = torch.stack(all_rewards)     # (n_steps, n_envs)
        dones = torch.stack(all_dones)         # (n_steps, n_envs)
        values = torch.stack(all_values)       # (n_steps, n_envs)

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

        # 3. PPO UPDATE
        # Flatten: (n_steps, n_envs, ...) -> (n_steps * n_envs, ...)
        b_obs = torch.stack(all_obs).reshape(-1, args.cond_steps, obs_dim)           # (N, cond_steps, obs_dim)
        b_chains = torch.stack(all_chains).reshape(-1, K + 1, args.horizon_steps, action_dim)
        b_logprobs = torch.stack(all_logprobs).reshape(-1, K, args.horizon_steps, action_dim)
        b_advantages = advantages.reshape(-1)           # (N,)
        b_returns = returns.reshape(-1)                 # (N,)
        b_values = values.reshape(-1)                   # (N,)

        N = b_obs.shape[0]  # n_steps * n_envs

        # Expand across denoising steps: each (obs, advantage, return, value) is
        # paired with each denoising step
        # Total samples: N * K
        total_samples = N * K

        model.train()
        for epoch in range(args.update_epochs):
            # Generate indices over (sample_idx, denoising_idx)
            perm = torch.randperm(total_samples)

            epoch_pg_loss = 0.0
            epoch_v_loss = 0.0
            epoch_kl = 0.0
            n_mb = 0
            early_stop = False

            for mb_start in range(0, total_samples, args.minibatch_size):
                mb_inds = perm[mb_start : mb_start + args.minibatch_size]
                if len(mb_inds) == 0:
                    continue

                # Unravel: sample_idx and denoising_idx
                sample_inds = mb_inds // K
                denoise_inds = mb_inds % K

                # Gather data
                mb_obs = {"state": b_obs[sample_inds]}
                mb_chains_prev = b_chains[sample_inds, denoise_inds]       # (mb, Ta, Da)
                mb_chains_next = b_chains[sample_inds, denoise_inds + 1]   # (mb, Ta, Da)
                mb_old_logprobs = b_logprobs[sample_inds, denoise_inds]    # (mb, Ta, Da)
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
                )

                loss = pg_loss + args.vf_coef * v_loss

                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.actor_ft.parameters(), args.max_grad_norm)
                nn.utils.clip_grad_norm_(model.critic.parameters(), args.max_grad_norm)
                actor_optimizer.step()
                critic_optimizer.step()

                epoch_pg_loss += pg_loss.item()
                epoch_v_loss += v_loss.item()
                epoch_kl += approx_kl
                n_mb += 1

                if args.target_kl is not None and approx_kl > args.target_kl:
                    early_stop = True
                    break

            if early_stop:
                break

        # Logging
        avg_reward = rewards.sum(0).mean().item()
        avg_pg_loss = epoch_pg_loss / max(n_mb, 1)
        avg_v_loss = epoch_v_loss / max(n_mb, 1)
        avg_kl = epoch_kl / max(n_mb, 1)

        writer.add_scalar("train/reward", avg_reward, iteration)
        writer.add_scalar("train/pg_loss", avg_pg_loss, iteration)
        writer.add_scalar("train/v_loss", avg_v_loss, iteration)
        writer.add_scalar("train/approx_kl", avg_kl, iteration)
        writer.add_scalar("train/global_step", global_step, iteration)

        t_elapsed = time.time() - t_start

        if iteration % 10 == 0:
            print(
                f"Iter {iteration}/{args.n_train_itr} | "
                f"reward={avg_reward:.3f} | pg={avg_pg_loss:.4f} | "
                f"v={avg_v_loss:.4f} | kl={avg_kl:.4f} | "
                f"time={t_elapsed:.1f}s"
            )

        # 4. EVALUATE
        if iteration == 1 or iteration % args.eval_freq == 0:
            model.eval()
            eval_metrics = evaluate(
                n_episodes=args.num_eval_episodes,
                model=model,
                eval_envs=eval_envs,
                device=device,
                act_steps=args.act_steps,
                obs_mean=obs_mean,
                obs_std=obs_std,
                action_min=action_min,
                action_max=action_max,
                cond_steps=args.cond_steps,
                max_episode_steps=args.max_episode_steps,
            )

            sr = eval_metrics.get("success_at_end", np.array([0])).mean()
            sr_once = eval_metrics.get("success_once", np.array([0])).mean()
            writer.add_scalar("eval/success_at_end", sr, iteration)
            writer.add_scalar("eval/success_once", sr_once, iteration)
            print(f"  EVAL @ iter {iteration}: success_at_end={sr:.3f}, success_once={sr_once:.3f}")

            if sr_once > best_sr:
                best_sr = sr_once
                ckpt = {
                    "model": model.state_dict(),
                    "obs_mean": obs_mean,
                    "obs_std": obs_std,
                    "action_min": action_min,
                    "action_max": action_max,
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "args": vars(args),
                    "iteration": iteration,
                }
                torch.save(ckpt, os.path.join(run_dir, "best.pt"))
                print(f"  Saved best checkpoint (sr_once={best_sr:.3f})")

    # Save final
    ckpt = {
        "model": model.state_dict(),
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "action_min": action_min,
        "action_max": action_max,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "args": vars(args),
        "iteration": args.n_train_itr,
    }
    torch.save(ckpt, os.path.join(run_dir, "final.pt"))
    print(f"Finetuning complete. Best sr_once={best_sr:.3f}")

    train_envs.close()
    eval_envs.close()
    writer.close()


if __name__ == "__main__":
    main()
