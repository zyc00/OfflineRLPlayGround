"""
Weighted-BC finetuning with rollout sampling.

Keep on-policy rollout collection, but replace policy-gradient update with
weighted diffusion BC update:
    min E[w(s,a) * diffusion_mse(a | s)]
where w is derived from rollout return.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from torch.utils.tensorboard import SummaryWriter

import tyro

from DPPO.model.mlp import DiffusionMLP
from DPPO.model.unet_wrapper import DiffusionUNet
from DPPO.model.diffusion import DiffusionModel
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

    # DDIM for rollout/eval
    use_ddim: bool = True
    ddim_steps: int = 10

    # Rollout
    n_train_itr: int = 20
    n_steps: int = 25
    gamma: float = 0.99
    reward_scale_running: bool = True
    reward_scale_const: float = 1.0

    # Weighted BC update
    update_epochs: int = 5
    minibatch_size: int = 2500
    norm_returns: bool = True
    # Weighting mode for BC update:
    # - return_pos: w=max(R,0) (default, current behavior)
    # - return: w=R (can be negative; generally less stable)
    # - posneg_return: split positive/negative return into two losses:
    #       L = L_pos - neg_loss_coef * L_neg
    # - binary_pos: w=1{R>0} (filtered-BC style)
    # - uniform: w=1 (plain BC on rollout batch)
    # - exp_return: w=exp(R / weight_temperature)
    weight_mode: str = "return_pos"
    weight_temperature: float = 1.0
    neg_loss_coef: float = 0.25  # used only by posneg_return
    # Optional data-distribution reweighting (NOT policy-ratio / IS):
    # - none: no correction
    # - inv_success_freq: inverse class-frequency by success label (R>0)
    # This correction multiplies the base BC weights.
    dist_reweight_mode: str = "none"
    dist_reweight_power: float = 1.0
    dist_reweight_max: float = 10.0

    # Rollout noise
    min_sampling_denoising_std: float = 0.01

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
        args.exp_name = f"wbc_{args.env_id}_envs{args.n_envs}"

    run_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)

    print(f"Pretrained: {args.pretrain_checkpoint}")
    print(f"  network_type={network_type}, obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"  denoising_steps={args.denoising_steps}, horizon_steps={args.horizon_steps}")
    print(f"  cond_steps={args.cond_steps}, act_steps={args.act_steps}, act_offset={act_offset}")
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

    model = DiffusionModel(
        network=actor,
        horizon_steps=args.horizon_steps,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        denoising_steps=args.denoising_steps,
        denoised_clip_value=1.0,
        randn_clip_value=3,
        final_action_clip_value=1.0,
        predict_epsilon=True,
    )

    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"], strict=False)
    else:
        model.load_state_dict(ckpt["model"], strict=False)

    n_actor_params = sum(p.numel() for p in model.network.parameters() if p.requires_grad)
    print(f"Finetuned actor params: {n_actor_params:,} (weighted BC)")

    optimizer = torch.optim.Adam(model.network.parameters(), lr=args.actor_lr)
    reward_scaler = RunningRewardScaler(gamma=args.gamma) if args.reward_scale_running else None

    n_decision_steps = args.n_steps
    use_gpu_env = args.sim_backend == "gpu"

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
        ddim_eval = args.ddim_steps if args.use_ddim else None
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
                samples_eval = model(cond_eval, deterministic=True, ddim_steps=ddim_eval)
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
    batch_size = args.n_envs * n_decision_steps

    print(f"\nStarting weighted-BC finetuning for {args.n_train_itr} iterations...")
    print(f"  n_envs={args.n_envs}, n_steps={n_decision_steps}, act_steps={args.act_steps}")
    print(f"  batch_size={batch_size}, gamma={args.gamma}")
    print(f"  update_epochs={args.update_epochs}, minibatch_size={args.minibatch_size}")
    print(f"  norm_returns={args.norm_returns}, reward_scale_running={args.reward_scale_running}")
    print(
        f"  weight_mode={args.weight_mode}, weight_temperature={args.weight_temperature}, "
        f"neg_loss_coef={args.neg_loss_coef}"
    )
    print(
        f"  dist_reweight_mode={args.dist_reweight_mode}, power={args.dist_reweight_power}, "
        f"max={args.dist_reweight_max}"
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
        action_trajs = []
        reward_trajs = []
        done_trajs = []
        n_success_rollout = 0

        ddim_rollout = args.ddim_steps if args.use_ddim else None
        for _ in range(n_decision_steps):
            obs_norm = normalize_obs(obs_history)
            cond = {"state": obs_norm}

            with torch.no_grad():
                samples = model(
                    cond,
                    deterministic=False,
                    min_sampling_denoising_std=args.min_sampling_denoising_std,
                    ddim_steps=ddim_rollout,
                )
                action_chunk = samples.trajectories

            obs_trajs.append(obs_norm.clone())
            action_trajs.append(action_chunk.clone())

            action_chunk_denorm = denormalize_actions(action_chunk)
            step_reward = torch.zeros(args.n_envs, device=device)
            step_done = torch.zeros(args.n_envs, dtype=torch.bool, device=device)

            for a_idx in range(args.act_steps):
                act_idx = act_offset + a_idx
                if act_idx < action_chunk_denorm.shape[1]:
                    action = action_chunk_denorm[:, act_idx]
                else:
                    action = action_chunk_denorm[:, -1]

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

        # ===== 2. DISCOUNTED RETURNS AS WEIGHTS =====
        returns = torch.zeros_like(rewards)
        running_return = torch.zeros(args.n_envs, device=device)
        for t in reversed(range(n_decision_steps)):
            running_return = rewards[t] + args.gamma * running_return * (1.0 - dones[t])
            returns[t] = running_return

        obs_stacked = torch.stack(obs_trajs)
        action_stacked = torch.stack(action_trajs)
        N = n_decision_steps * args.n_envs

        b_obs = obs_stacked.reshape(N, args.cond_steps, obs_dim)
        b_actions = action_stacked.reshape(N, args.horizon_steps, action_dim)
        b_returns = returns.reshape(-1)

        # Build per-sample BC weights from return
        if args.weight_mode == "return_pos":
            b_weights = b_returns.clamp(min=0.0)
        elif args.weight_mode == "return":
            b_weights = b_returns
        elif args.weight_mode == "posneg_return":
            # Success → +1, failure → -1 (binary contrastive)
            b_weights = 2.0 * (b_returns > 0).float() - 1.0
        elif args.weight_mode == "binary_pos":
            b_weights = (b_returns > 0).float()
        elif args.weight_mode == "uniform":
            b_weights = torch.ones_like(b_returns)
        elif args.weight_mode == "exp_return":
            temp = max(args.weight_temperature, 1e-6)
            b_weights = torch.exp((b_returns / temp).clamp(min=-20.0, max=20.0))
        else:
            raise ValueError(
                f"Unknown weight_mode={args.weight_mode}. "
                "Choose from: return_pos, return, binary_pos, uniform, exp_return."
            )

        if args.norm_returns:
            b_weights = b_weights / (b_weights.mean() + 1e-8)

        # Optional correction for dataset sampling bias (not policy-ratio).
        dist_corr = torch.ones_like(b_weights)
        if args.dist_reweight_mode == "none":
            pass
        elif args.dist_reweight_mode == "inv_success_freq":
            is_pos = (b_returns > 0).float()
            pos_frac = is_pos.mean()
            neg_frac = 1.0 - pos_frac
            w_pos = 1.0 / (pos_frac + 1e-8)
            w_neg = 1.0 / (neg_frac + 1e-8)
            dist_corr = torch.where(is_pos > 0.5, w_pos, w_neg)
        else:
            raise ValueError(
                f"Unknown dist_reweight_mode={args.dist_reweight_mode}. "
                "Choose from: none, inv_success_freq."
            )

        if args.dist_reweight_power != 1.0:
            dist_corr = dist_corr.pow(args.dist_reweight_power)
        if args.dist_reweight_max is not None and args.dist_reweight_max > 0:
            dist_corr = dist_corr.clamp(max=args.dist_reweight_max)
        # Keep overall update scale stable.
        dist_corr = dist_corr / (dist_corr.mean() + 1e-8)
        b_weights = b_weights * dist_corr

        # ===== 3. WEIGHTED BC UPDATE =====
        model.train()
        total_loss = 0.0
        n_mb = 0
        grad_norm_last = 0.0

        for _ in range(args.update_epochs):
            perm = torch.randperm(N, device=device)
            for mb_start in range(0, N, args.minibatch_size):
                mb_inds = perm[mb_start:mb_start + args.minibatch_size]
                if len(mb_inds) == 0:
                    continue

                mb_obs = {"state": b_obs[mb_inds]}
                mb_actions = b_actions[mb_inds]
                mb_weights = b_weights[mb_inds]

                t = torch.randint(0, model.denoising_steps, (len(mb_inds),), device=device).long()
                noise = torch.randn_like(mb_actions)
                x_noisy = model.q_sample(x_start=mb_actions, t=t, noise=noise)
                x_pred = model.network(x_noisy, t, cond=mb_obs)
                target = noise if model.predict_epsilon else mb_actions

                per_sample = F.mse_loss(x_pred, target, reduction="none").mean(dim=(1, 2))
                if args.weight_mode == "posneg_return":
                    # Stabilized positive/negative decomposition:
                    #   L = mean_pos(per_sample, w+) - lambda * mean_neg(per_sample, w-)
                    w_pos = mb_weights.clamp(min=0.0)
                    w_neg = (-mb_weights).clamp(min=0.0)
                    pos_denom = w_pos.sum() + 1e-8
                    neg_denom = w_neg.sum() + 1e-8
                    pos_loss = (per_sample * w_pos).sum() / pos_denom
                    neg_loss = (per_sample * w_neg).sum() / neg_denom
                    loss = pos_loss - args.neg_loss_coef * neg_loss
                else:
                    loss = (per_sample * mb_weights).mean()

                optimizer.zero_grad()
                loss.backward()
                grad_norm_last = nn.utils.clip_grad_norm_(model.network.parameters(), args.max_grad_norm)
                optimizer.step()

                total_loss += loss.item()
                n_mb += 1

        # ===== 4. LOGGING =====
        avg_reward = rewards.sum(0).mean().item()
        avg_bc_loss = total_loss / max(n_mb, 1)
        ret_mean = b_returns.mean().item()
        ret_std = b_returns.std().item()
        ret_pos_frac = (b_returns > 0).float().mean().item()
        weight_mean = b_weights.mean().item()
        weight_max = b_weights.max().item()
        corr_mean = dist_corr.mean().item()
        corr_max = dist_corr.max().item()

        writer.add_scalar("train/reward", avg_reward, iteration)
        writer.add_scalar("train/weighted_bc_loss", avg_bc_loss, iteration)
        writer.add_scalar("train/weight_mean", weight_mean, iteration)
        writer.add_scalar("train/weight_max", weight_max, iteration)
        writer.add_scalar("train/dist_corr_mean", corr_mean, iteration)
        writer.add_scalar("train/dist_corr_max", corr_max, iteration)
        writer.add_scalar("train/rollout_ep_successes", n_success_rollout, iteration)
        writer.add_scalar("train/global_step", global_step, iteration)

        t_elapsed = time.time() - t_start
        print(
            f"Iter {iteration}/{args.n_train_itr} | "
            f"r={avg_reward:.3f} | succ={n_success_rollout} | "
            f"wbc={avg_bc_loss:.6f} | "
            f"w_mean={weight_mean:.3f} w_max={weight_max:.3f} | "
            f"corr_mean={corr_mean:.3f} corr_max={corr_max:.3f} | "
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
