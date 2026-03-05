"""
REINFORCE finetuning with rollout sampling (no importance sampling ratio).

Keeps the same rollout/eval structure as finetune_bc.py, but updates policy with
policy-gradient objective on diffusion sub-steps:
    L = - E[ log pi_theta(x_{k+1} | x_k, s) * R_t ]

Where each decision-step return R_t is shared across its K denoising sub-steps.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from torch.utils.tensorboard import SummaryWriter

import tyro

from DPPO.model.mlp import DiffusionMLP
from DPPO.model.unet_wrapper import DiffusionUNet
from DPPO.model.critic import CriticObs
from DPPO.model.diffusion_ppo import PPODiffusion
from DPPO.make_env import make_train_envs
from DPPO.reward_scaler import RunningRewardScaler


@dataclass
class Args:
    pretrain_checkpoint: str = ""

    # Environment
    env_id: str = "PegInsertionSide-v1"
    control_mode: str = "pd_joint_delta_pos"
    max_episode_steps: int = 200
    n_envs: int = 500
    sim_backend: str = "gpu"

    # Architecture (overridden from checkpoint)
    denoising_steps: int = 100
    horizon_steps: int = 16
    cond_steps: int = 2
    act_steps: int = 8

    # DDIM
    use_ddim: bool = True
    ddim_steps: int = 10

    # Rollout
    n_train_itr: int = 20
    n_steps: int = 25
    gamma: float = 0.99
    reward_scale_running: bool = True
    reward_scale_const: float = 1.0

    # REINFORCE update
    update_epochs: int = 5
    minibatch_size: int = 2500
    norm_returns: bool = True
    binarize_returns: bool = False  # If True, use R=1{return>0}
    gamma_denoising: float = 1.0  # 1.0 means no denoising-step discount

    # Rollout/logprob noise config
    min_sampling_denoising_std: float = 0.01
    min_logprob_denoising_std: float = 0.01

    # Optimization
    actor_lr: float = 3e-6
    max_grad_norm: float = 1.0

    # Augmentation
    zero_qvel: bool = False

    # Eval
    eval_freq: int = 1
    eval_n_rounds: int = 3

    # Logging
    exp_name: Optional[str] = None
    seed: int = 0
    save_dir: str = "runs/dppo_finetune"


def main():
    args = tyro.cli(Args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.pretrain_checkpoint, map_location=device, weights_only=False)
    obs_dim = ckpt["obs_dim"]
    action_dim = ckpt["action_dim"]

    no_obs_norm = ckpt.get("no_obs_norm", False)
    no_action_norm = ckpt.get("no_action_norm", False)

    obs_min = obs_max = action_min = action_max = None
    if not no_obs_norm:
        obs_min = ckpt["obs_min"].to(device)
        obs_max = ckpt["obs_max"].to(device)
    if not no_action_norm:
        action_min = ckpt["action_min"].to(device)
        action_max = ckpt["action_max"].to(device)

    pretrain_args = ckpt.get("args", {})
    args.denoising_steps = pretrain_args.get("denoising_steps", args.denoising_steps)
    args.horizon_steps = pretrain_args.get("horizon_steps", args.horizon_steps)
    args.cond_steps = pretrain_args.get("cond_steps", args.cond_steps)
    args.act_steps = pretrain_args.get("act_steps", args.act_steps)
    network_type = pretrain_args.get("network_type", "mlp")

    if pretrain_args.get("zero_qvel", False) and not args.zero_qvel:
        args.zero_qvel = True
        print("  Inherited zero_qvel=True from pretrain checkpoint")

    cond_dim = obs_dim * args.cond_steps
    act_offset = args.cond_steps - 1 if network_type == "unet" else 0

    if args.exp_name is None:
        args.exp_name = f"reinforce_{args.env_id}_envs{args.n_envs}"

    run_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)

    print(f"Pretrained: {args.pretrain_checkpoint}")
    print(f"  network_type={network_type}, obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"  denoising_steps={args.denoising_steps}, ft_denoising_steps={args.denoising_steps if not args.use_ddim else args.ddim_steps}")
    print(f"  horizon_steps={args.horizon_steps}, cond_steps={args.cond_steps}, act_steps={args.act_steps}, act_offset={act_offset}")
    print(f"  zero_qvel={args.zero_qvel}")

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

    # Dummy critic: PPODiffusion expects one, but REINFORCE update does not use it.
    critic = CriticObs(
        cond_dim=cond_dim,
        mlp_dims=[256, 256, 256],
        activation_type="Mish",
        residual_style=True,
    )

    model = PPODiffusion(
        actor=actor,
        critic=critic,
        ft_denoising_steps=args.ddim_steps if args.use_ddim else args.denoising_steps,
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
        clip_ploss_coef=0.02,
        clip_ploss_coef_base=1e-3,
        clip_ploss_coef_rate=3.0,
        norm_adv=False,
    )

    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"], strict=False)
    else:
        model.load_state_dict(ckpt["model"], strict=False)

    model._sanitize_eta()
    model.actor_ft.load_state_dict(model.actor.state_dict())
    for p in model.actor.parameters():
        p.requires_grad = False

    n_actor_params = sum(p.numel() for p in model.actor_ft.parameters() if p.requires_grad)
    print(f"Finetuned actor params: {n_actor_params:,} (REINFORCE no-IS)")

    optimizer = torch.optim.Adam(model.actor_ft.parameters(), lr=args.actor_lr)
    reward_scaler = RunningRewardScaler(gamma=args.gamma) if args.reward_scale_running else None

    K = model.ft_denoising_steps
    n_decision_steps = args.n_steps

    train_envs = make_train_envs(
        env_id=args.env_id,
        num_envs=args.n_envs,
        sim_backend=args.sim_backend,
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
    )

    def normalize_obs(obs):
        if args.zero_qvel:
            obs = obs.clone()
            obs[..., 9:18] = 0.0
        if no_obs_norm:
            return obs
        return (obs - obs_min) / (obs_max - obs_min + 1e-8) * 2.0 - 1.0

    def denormalize_actions(actions):
        if no_action_norm:
            return actions
        return (actions + 1.0) / 2.0 * (action_max - action_min) + action_min

    @torch.no_grad()
    def evaluate_gpu_inline(n_rounds=5):
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
            ep_done = torch.zeros(args.n_envs, dtype=torch.bool, device=device)
            for _ in range(args.max_episode_steps // args.act_steps + 1):
                cond_eval = {"state": normalize_obs(obs_h)}
                samples_eval = model(cond_eval, deterministic=True, return_chain=False)
                ac_eval = denormalize_actions(samples_eval.trajectories)
                for a_idx in range(args.act_steps):
                    act_idx = act_offset + a_idx
                    action_eval = ac_eval[:, min(act_idx, ac_eval.shape[1] - 1)]
                    obs_new_eval, rew_eval, term_eval, trunc_eval, _ = train_envs.step(action_eval)
                    obs_new_eval = obs_new_eval.float()
                    rew_eval = rew_eval.float()
                    term_eval = term_eval.bool()
                    trunc_eval = trunc_eval.bool()
                    success_once |= (rew_eval > 0.5).bool() & ~ep_done
                    rm = term_eval | trunc_eval
                    ep_done = ep_done | rm
                    if rm.any():
                        obs_h[rm] = obs_new_eval[rm].unsqueeze(1).repeat(1, args.cond_steps, 1)
                    obs_h[~rm] = torch.cat([obs_h[~rm, 1:], obs_new_eval[~rm].unsqueeze(1)], dim=1)
                if ep_done.all():
                    break
            total_success += success_once.sum().item()
            total_eps += args.n_envs
        return total_success / max(total_eps, 1)

    best_sr = -1.0
    global_step = 0

    print(f"\nStarting REINFORCE(no-IS) finetuning for {args.n_train_itr} iterations...")
    print(f"  n_envs={args.n_envs}, n_steps={n_decision_steps}, act_steps={args.act_steps}, K={K}")
    print(f"  gamma={args.gamma}, update_epochs={args.update_epochs}, minibatch_size={args.minibatch_size}")
    print(
        f"  norm_returns={args.norm_returns}, binarize_returns={args.binarize_returns}, "
        f"gamma_denoising={args.gamma_denoising}"
    )

    obs_raw, _ = train_envs.reset()
    if isinstance(obs_raw, np.ndarray):
        obs_raw = torch.from_numpy(obs_raw).float().to(device)
    else:
        obs_raw = obs_raw.float().to(device)
    obs_history = obs_raw.unsqueeze(1).repeat(1, args.cond_steps, 1)

    for iteration in range(1, args.n_train_itr + 1):
        t_start = time.time()
        model.eval()

        # ===== 1. ROLLOUT =====
        obs_trajs = []
        chains_trajs = []
        reward_trajs = []
        done_trajs = []
        n_success_rollout = 0

        for _ in range(n_decision_steps):
            obs_norm = normalize_obs(obs_history)
            cond = {"state": obs_norm}

            with torch.no_grad():
                samples = model(cond, deterministic=False, return_chain=True)
                action_chunk = samples.trajectories
                chains = samples.chains

            obs_trajs.append(obs_norm.clone())
            chains_trajs.append(chains.clone())

            action_chunk_denorm = denormalize_actions(action_chunk)
            step_reward = torch.zeros(args.n_envs, device=device)
            step_done = torch.zeros(args.n_envs, dtype=torch.bool, device=device)

            for a_idx in range(args.act_steps):
                act_idx = act_offset + a_idx
                action = action_chunk_denorm[:, min(act_idx, action_chunk_denorm.shape[1] - 1)]

                if step_done.any():
                    action = action.clone()
                    action[step_done] = 0.0

                obs_new, reward, terminated, truncated, _ = train_envs.step(action)
                obs_new = obs_new.float()
                reward_t = reward.float()
                term_t = terminated.bool()
                trunc_t = truncated.bool()

                newly_done = (term_t | trunc_t) & ~step_done
                step_reward += reward_t * (~step_done).float()
                n_success_rollout += ((reward_t > 0.5) & newly_done).sum().item()
                step_done = step_done | term_t | trunc_t

                reset_mask = term_t | trunc_t
                if reset_mask.any():
                    obs_history[reset_mask] = obs_new[reset_mask].unsqueeze(1).repeat(1, args.cond_steps, 1)
                obs_history[~reset_mask] = torch.cat(
                    [obs_history[~reset_mask, 1:], obs_new[~reset_mask].unsqueeze(1)], dim=1
                )
                global_step += args.n_envs

            reward_trajs.append(step_reward)
            done_trajs.append(step_done.float())

        rewards = torch.stack(reward_trajs)
        dones = torch.stack(done_trajs)

        if reward_scaler is not None:
            rewards = reward_scaler.update_and_scale(rewards, dones)
        rewards = rewards * args.reward_scale_const

        # ===== 2. DISCOUNTED RETURNS =====
        returns = torch.zeros_like(rewards)
        running_return = torch.zeros(args.n_envs, device=device)
        for t in reversed(range(n_decision_steps)):
            running_return = rewards[t] + args.gamma * running_return * (1.0 - dones[t])
            returns[t] = running_return

        obs_stacked = torch.stack(obs_trajs)
        chains_stacked = torch.stack(chains_trajs)

        N = n_decision_steps * args.n_envs
        b_obs = obs_stacked.reshape(N, args.cond_steps, obs_dim)
        b_chains = chains_stacked.reshape(N, K + 1, args.horizon_steps, action_dim)
        b_returns = returns.reshape(-1)

        if args.binarize_returns:
            b_returns = (b_returns > 0).float()
        if args.norm_returns:
            b_returns = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-8)

        # ===== 3. REINFORCE UPDATE (NO IS) =====
        model.train()
        total_loss = 0.0
        total_logp = 0.0
        n_mb = 0
        grad_norm_last = 0.0

        total_samples = N * K
        for _ in range(args.update_epochs):
            perm = torch.randperm(total_samples, device=device)
            for mb_start in range(0, total_samples, args.minibatch_size):
                mb_inds = perm[mb_start:mb_start + args.minibatch_size]
                if len(mb_inds) == 0:
                    continue

                sample_inds = mb_inds // K
                denoise_inds = mb_inds % K

                mb_obs = {"state": b_obs[sample_inds]}
                mb_chains_prev = b_chains[sample_inds, denoise_inds]
                mb_chains_next = b_chains[sample_inds, denoise_inds + 1]
                mb_returns = b_returns[sample_inds]

                logp, _ = model.get_logprobs_subsample(
                    mb_obs, mb_chains_prev, mb_chains_next, denoise_inds, get_ent=True
                )
                logp = logp[:, act_offset:act_offset + args.act_steps, :].mean(dim=(-1, -2))

                if args.gamma_denoising != 1.0:
                    denoise_discount = torch.pow(
                        torch.tensor(args.gamma_denoising, device=device),
                        (K - denoise_inds - 1).float(),
                    )
                    mb_returns_eff = mb_returns * denoise_discount
                else:
                    mb_returns_eff = mb_returns

                loss = -(logp * mb_returns_eff).mean()

                optimizer.zero_grad()
                loss.backward()
                grad_norm_last = nn.utils.clip_grad_norm_(model.actor_ft.parameters(), args.max_grad_norm)
                optimizer.step()

                total_loss += loss.item()
                total_logp += logp.mean().item()
                n_mb += 1

        # ===== 4. LOGGING =====
        avg_reward = rewards.sum(0).mean().item()
        avg_loss = total_loss / max(n_mb, 1)
        avg_logp = total_logp / max(n_mb, 1)
        ret_mean = returns.mean().item()
        ret_std = returns.std().item()
        ret_pos_frac = (returns > 0).float().mean().item()

        writer.add_scalar("train/reward", avg_reward, iteration)
        writer.add_scalar("train/reinforce_loss", avg_loss, iteration)
        writer.add_scalar("train/logp_mean", avg_logp, iteration)
        writer.add_scalar("train/rollout_ep_successes", n_success_rollout, iteration)
        writer.add_scalar("train/ret_mean", ret_mean, iteration)
        writer.add_scalar("train/ret_std", ret_std, iteration)
        writer.add_scalar("train/ret_pos_frac", ret_pos_frac, iteration)
        writer.add_scalar("train/global_step", global_step, iteration)

        t_elapsed = time.time() - t_start
        print(
            f"Iter {iteration}/{args.n_train_itr} | "
            f"r={avg_reward:.3f} | succ={n_success_rollout} | "
            f"reinforce={avg_loss:.6f} | logp={avg_logp:.4f} | "
            f"grad={float(grad_norm_last):.2f} | "
            f"ret: mean={ret_mean:.4f} std={ret_std:.4f} pos={ret_pos_frac:.2f} | "
            f"time={t_elapsed:.1f}s"
        )

        # ===== 5. EVALUATE =====
        if iteration == 1 or iteration % args.eval_freq == 0:
            sr_once = evaluate_gpu_inline(n_rounds=args.eval_n_rounds)
            writer.add_scalar("eval/success_once", sr_once, iteration)
            print(f"  EVAL @ iter {iteration}: gpu_sr={sr_once:.3f}")

            obs_raw_r, _ = train_envs.reset()
            if isinstance(obs_raw_r, np.ndarray):
                obs_raw_r = torch.from_numpy(obs_raw_r).float().to(device)
            else:
                obs_raw_r = obs_raw_r.float().to(device)
            obs_history[:] = obs_raw_r.unsqueeze(1).repeat(1, args.cond_steps, 1)

            if sr_once > best_sr:
                best_sr = sr_once
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

    print(f"Finetuning complete. Best sr_once={best_sr:.3f}")
    train_envs.close()
    writer.close()


if __name__ == "__main__":
    main()
