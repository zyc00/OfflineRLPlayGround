"""
Bridge ablation trainer: progressively bridge REINFORCE -> DPPO.

Starts from finetune_reinforce-style rollout/update and adds optional components:
- continuous returns
- denoising discount
- value baseline (TD(1))
- GAE
- advantage normalization
- critic warmup
- PPO clipped objective via model.loss(...)

Use --bridge_stage presets for reproducible ablation runs.
"""

import os
import time
import math
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


STAGES = [
    "s0_reinforce_is_matched",
    "s1_continuous_return",
    "s2_gamma_denoising",
    "s3_value_baseline_td1",
    "s3b_value_baseline_td1_warmup",
    "s4_gae",
    "s4b_gae_warmup",
    "s5_norm_adv",
    "s6_critic_warmup",
    "s7_full_dppo_equiv",
    # PPO -> REINFORCE policy-extraction bridge (finetune.py-aligned start)
    "p0_ppo_clip_full",
    "p1_reinforce_is_gae",
    "p2_reinforce_is_td",
    "p3_reinforce_is_nobaseline",
    "p4_reinforce_is_final",
]


@dataclass
class Args:
    pretrain_checkpoint: str = ""

    # Stage preset
    bridge_stage: str = "s0_reinforce_is_matched"

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

    # Update
    update_epochs: int = 10
    minibatch_size: int = 2500
    gamma_denoising: float = 1.0

    # Return shaping
    binarize_returns: bool = True
    norm_returns: bool = False

    # Policy objective
    policy_objective: str = "reinforce_is"  # reinforce_is | ppo_clip
    use_is: bool = True
    is_clip_ratio: float = 0.2
    reinforce_use_ppo_clip_schedule: bool = False

    # Value / baseline
    use_value_baseline: bool = False
    use_gae: bool = False
    gae_lambda: float = 0.95
    norm_adv: bool = False
    vf_coef: float = 0.5
    n_critic_warmup_itr: int = 0

    # PPO params
    clip_ploss_coef: float = 0.02
    clip_ploss_coef_base: float = 1e-3
    clip_ploss_coef_rate: float = 3.0
    target_kl: Optional[float] = 1.0

    # Noise config
    min_sampling_denoising_std: float = 0.01
    min_logprob_denoising_std: float = 0.01

    # Optimization
    actor_lr: float = 3e-6
    critic_lr: float = 1e-3
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


def apply_stage_preset(args: Args):
    if args.bridge_stage not in STAGES:
        raise ValueError(f"Unknown bridge_stage={args.bridge_stage}. Choose from: {STAGES}")

    # S0 baseline
    args.policy_objective = "reinforce_is"
    args.use_is = True
    args.is_clip_ratio = 0.2
    args.reinforce_use_ppo_clip_schedule = False
    args.binarize_returns = True
    args.norm_returns = False
    args.gamma_denoising = 1.0
    args.use_value_baseline = False
    args.use_gae = False
    args.norm_adv = False
    args.n_critic_warmup_itr = 0
    args.vf_coef = 0.5

    if args.bridge_stage in [
        "s1_continuous_return",
        "s2_gamma_denoising",
        "s3_value_baseline_td1",
        "s3b_value_baseline_td1_warmup",
        "s4_gae",
        "s5_norm_adv",
        "s6_critic_warmup",
        "s7_full_dppo_equiv",
    ]:
        args.binarize_returns = False
        args.norm_returns = True

    if args.bridge_stage in [
        "s2_gamma_denoising",
        "s3_value_baseline_td1",
        "s3b_value_baseline_td1_warmup",
        "s4_gae",
        "s5_norm_adv",
        "s6_critic_warmup",
        "s7_full_dppo_equiv",
    ]:
        args.gamma_denoising = 0.99

    if args.bridge_stage in [
        "s3_value_baseline_td1",
        "s3b_value_baseline_td1_warmup",
        "s4_gae",
        "s5_norm_adv",
        "s6_critic_warmup",
        "s7_full_dppo_equiv",
    ]:
        args.use_value_baseline = True

    if args.bridge_stage in ["s4_gae", "s4b_gae_warmup", "s5_norm_adv", "s6_critic_warmup", "s7_full_dppo_equiv"]:
        args.use_gae = True

    if args.bridge_stage in ["s5_norm_adv", "s6_critic_warmup", "s7_full_dppo_equiv"]:
        args.norm_adv = True

    # Align with run_dppo_finetune.sh: any stage that learns critic uses 5 iters warmup.
    if args.use_value_baseline:
        args.n_critic_warmup_itr = 5

    if args.bridge_stage == "s7_full_dppo_equiv":
        args.policy_objective = "ppo_clip"
        args.use_is = True
        args.is_clip_ratio = 0.2

    # ===== PPO -> REINFORCE extraction bridge =====
    if args.bridge_stage.startswith("p"):
        # p0: mimic finetune.py training setup as closely as possible.
        args.policy_objective = "ppo_clip"
        args.use_is = True
        args.is_clip_ratio = 0.2
        args.binarize_returns = False
        args.norm_returns = True
        args.gamma_denoising = 0.99
        args.use_value_baseline = True
        args.use_gae = True
        args.norm_adv = True
        args.vf_coef = 0.5
        args.n_critic_warmup_itr = 5

        # p1: only switch policy extraction from PPO-clip to IS-REINFORCE.
        if args.bridge_stage in ["p1_reinforce_is_gae", "p2_reinforce_is_td", "p3_reinforce_is_nobaseline", "p4_reinforce_is_final"]:
            args.policy_objective = "reinforce_is"
        if args.bridge_stage == "p1_reinforce_is_gae":
            # Match PPO's trust-region scale more closely for the first bridge step.
            args.is_clip_ratio = 0.02
            args.reinforce_use_ppo_clip_schedule = True

        # p2: remove GAE (keep baseline/value learning).
        if args.bridge_stage in ["p2_reinforce_is_td", "p3_reinforce_is_nobaseline", "p4_reinforce_is_final"]:
            args.use_gae = False
            args.norm_adv = False

        # p3: remove critic baseline/value module.
        if args.bridge_stage in ["p3_reinforce_is_nobaseline", "p4_reinforce_is_final"]:
            args.use_value_baseline = False
            args.n_critic_warmup_itr = 0
            args.vf_coef = 0.0

        # p4: match finetune_reinforce-style return signal.
        if args.bridge_stage == "p4_reinforce_is_final":
            args.binarize_returns = True
            args.norm_returns = False
            args.gamma_denoising = 1.0


def main():
    args = tyro.cli(Args)
    apply_stage_preset(args)

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
        args.exp_name = f"bridge_{args.bridge_stage}_seed{args.seed}"

    run_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)

    print(f"Pretrained: {args.pretrain_checkpoint}")
    print(f"  stage={args.bridge_stage}, objective={args.policy_objective}")
    print(f"  network_type={network_type}, obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"  denoising_steps={args.denoising_steps}, ft_denoising_steps={args.ddim_steps if args.use_ddim else args.denoising_steps}")
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
        clip_ploss_coef=args.clip_ploss_coef,
        clip_ploss_coef_base=args.clip_ploss_coef_base,
        clip_ploss_coef_rate=args.clip_ploss_coef_rate,
        norm_adv=args.norm_adv,
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
    n_critic_params = sum(p.numel() for p in model.critic.parameters() if p.requires_grad)
    print(f"Finetuned actor params: {n_actor_params:,}, critic params: {n_critic_params:,}")

    actor_optimizer = torch.optim.Adam(model.actor_ft.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=args.critic_lr)
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

    def get_reinforce_clip_coef(denoise_inds):
        if not args.reinforce_use_ppo_clip_schedule:
            return torch.full_like(denoise_inds.float(), args.is_clip_ratio, device=device)

        t = (denoise_inds.float() / max(K - 1, 1)).to(device)
        if K > 1:
            return args.clip_ploss_coef_base + (
                args.clip_ploss_coef - args.clip_ploss_coef_base
            ) * (torch.exp(args.clip_ploss_coef_rate * t) - 1) / (
                math.exp(args.clip_ploss_coef_rate) - 1
            )
        return t

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

    print(f"\nStarting bridge ablation for {args.n_train_itr} iterations...")
    print(f"  n_envs={args.n_envs}, n_steps={n_decision_steps}, act_steps={args.act_steps}, K={K}")
    print(f"  gamma={args.gamma}, update_epochs={args.update_epochs}, minibatch_size={args.minibatch_size}")
    print(
        f"  binarize_returns={args.binarize_returns}, norm_returns={args.norm_returns}, "
        f"gamma_denoising={args.gamma_denoising}"
    )
    print(
        f"  use_value_baseline={args.use_value_baseline}, use_gae={args.use_gae}, "
        f"norm_adv={args.norm_adv}, warmup={args.n_critic_warmup_itr}"
    )
    print(
        f"  policy_objective={args.policy_objective}, use_is={args.use_is}, "
        f"is_clip_ratio={args.is_clip_ratio}, reinforce_ppo_clip_schedule={args.reinforce_use_ppo_clip_schedule}, "
        f"target_kl={args.target_kl}"
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
        is_warmup = args.use_value_baseline and (iteration <= args.n_critic_warmup_itr)

        # ===== 1. ROLLOUT =====
        obs_trajs = []
        chains_trajs = []
        reward_trajs = []
        done_trajs = []
        value_trajs = []
        n_success_rollout = 0

        for _ in range(n_decision_steps):
            obs_norm = normalize_obs(obs_history)
            cond = {"state": obs_norm}

            with torch.no_grad():
                samples = model(cond, deterministic=False, return_chain=True)
                action_chunk = samples.trajectories
                chains = samples.chains
                if args.use_value_baseline:
                    value = model.critic(cond).squeeze(-1)
                else:
                    value = torch.zeros(args.n_envs, device=device)

            obs_trajs.append(obs_norm.clone())
            chains_trajs.append(chains.clone())
            value_trajs.append(value.clone())

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
        values = torch.stack(value_trajs)

        if reward_scaler is not None:
            rewards = reward_scaler.update_and_scale(rewards, dones)
        rewards = rewards * args.reward_scale_const

        # ===== 2. RETURNS / ADVANTAGES =====
        returns = torch.zeros_like(rewards)
        running_return = torch.zeros(args.n_envs, device=device)
        for t in reversed(range(n_decision_steps)):
            running_return = rewards[t] + args.gamma * running_return * (1.0 - dones[t])
            returns[t] = running_return

        if args.binarize_returns:
            returns = (returns > 0).float()
        if args.norm_returns:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        if args.use_value_baseline:
            with torch.no_grad():
                obs_norm_last = normalize_obs(obs_history)
                next_value = model.critic({"state": obs_norm_last}).squeeze(-1)

            if args.use_gae:
                advantages = torch.zeros_like(rewards)
                lastgaelam = torch.zeros(args.n_envs, device=device)
                for t in reversed(range(n_decision_steps)):
                    if t == n_decision_steps - 1:
                        nextvalues = next_value
                    else:
                        nextvalues = values[t + 1]
                    next_not_done = 1.0 - dones[t]
                    delta = rewards[t] + args.gamma * nextvalues * next_not_done - values[t]
                    lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam
                    advantages[t] = lastgaelam
                returns_v = advantages + values
            else:
                advantages = returns - values
                returns_v = returns
        else:
            advantages = returns
            returns_v = returns

        obs_stacked = torch.stack(obs_trajs)
        chains_stacked = torch.stack(chains_trajs)

        N = n_decision_steps * args.n_envs
        b_obs = obs_stacked.reshape(N, args.cond_steps, obs_dim)
        b_chains = chains_stacked.reshape(N, K + 1, args.horizon_steps, action_dim)
        b_advantages = advantages.reshape(-1)
        b_returns_v = returns_v.reshape(-1)
        b_values = values.reshape(-1)

        if args.use_is or args.policy_objective == "ppo_clip":
            model.actor.load_state_dict(model.actor_ft.state_dict())

        # ===== 3. UPDATE =====
        model.train()
        total_loss = 0.0
        total_pg = 0.0
        total_v = 0.0
        total_logp = 0.0
        total_ratio = 0.0
        total_clipfrac = 0.0
        total_kl = 0.0
        n_mb = 0
        grad_norm_actor_last = 0.0
        grad_norm_critic_last = 0.0

        total_samples = N * K
        early_stop = False
        early_stop_reason = ""
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
                mb_adv = b_advantages[sample_inds]
                mb_ret_v = b_returns_v[sample_inds]
                mb_oldv = b_values[sample_inds]

                if args.policy_objective == "ppo_clip":
                    with torch.no_grad():
                        old_lp = model.get_logprobs_subsample(
                            mb_obs, mb_chains_prev, mb_chains_next, denoise_inds, get_ent=False, use_base_policy=True
                        )
                    pg_loss, _, v_loss, clipfrac, approx_kl, ratio_mean = model.loss(
                        obs=mb_obs,
                        chains_prev=mb_chains_prev,
                        chains_next=mb_chains_next,
                        denoising_inds=denoise_inds,
                        returns=mb_ret_v,
                        oldvalues=mb_oldv,
                        advantages=mb_adv,
                        oldlogprobs=old_lp,
                        reward_horizon=args.act_steps,
                        act_offset=act_offset,
                    )
                    if is_warmup and args.use_value_baseline:
                        loss = args.vf_coef * v_loss
                    else:
                        loss = pg_loss + (args.vf_coef * v_loss if args.use_value_baseline else 0.0)

                    total_pg += pg_loss.item()
                    total_v += v_loss.item()
                    total_ratio += ratio_mean
                    total_clipfrac += clipfrac
                    total_kl += approx_kl
                else:
                    # REINFORCE-style stages apply advantage normalization and
                    # denoising-step discount explicitly here. PPO stages do
                    # this inside model.loss(...), matching finetune.py.
                    if args.norm_adv:
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    if args.gamma_denoising != 1.0:
                        denoise_discount = torch.pow(
                            torch.tensor(args.gamma_denoising, device=device),
                            (K - denoise_inds - 1).float(),
                        )
                        mb_adv_eff = mb_adv * denoise_discount
                    else:
                        mb_adv_eff = mb_adv

                    lp_new, _ = model.get_logprobs_subsample(
                        mb_obs, mb_chains_prev, mb_chains_next, denoise_inds, get_ent=True
                    )
                    lp_new = lp_new[:, act_offset:act_offset + args.act_steps, :].mean(dim=(-1, -2))
                    if args.use_is:
                        with torch.no_grad():
                            lp_old = model.get_logprobs_subsample(
                                mb_obs, mb_chains_prev, mb_chains_next, denoise_inds, get_ent=False, use_base_policy=True
                            )
                            lp_old = lp_old[:, act_offset:act_offset + args.act_steps, :].mean(dim=(-1, -2))
                        ratio = (lp_new - lp_old).exp()
                        if args.is_clip_ratio > 0:
                            clip_coef = get_reinforce_clip_coef(denoise_inds)
                            ratio_clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                            obj = torch.minimum(ratio * mb_adv_eff, ratio_clipped * mb_adv_eff)
                            clipfrac = ((ratio - ratio_clipped).abs() > 1e-12).float().mean().item()
                        else:
                            obj = ratio * mb_adv_eff
                            clipfrac = 0.0
                        pg_loss = -obj.mean()
                        total_ratio += ratio.mean().item()
                        total_clipfrac += clipfrac
                        total_kl += (((ratio - 1) - (lp_new - lp_old)).mean().item())
                    else:
                        pg_loss = -(lp_new * mb_adv_eff).mean()

                    if args.use_value_baseline:
                        v_pred = model.critic(mb_obs).squeeze(-1)
                        v_loss = 0.5 * ((v_pred - mb_ret_v) ** 2).mean()
                    else:
                        v_loss = torch.zeros([], device=device)

                    if is_warmup and args.use_value_baseline:
                        loss = args.vf_coef * v_loss
                    else:
                        loss = pg_loss + (args.vf_coef * v_loss if args.use_value_baseline else 0.0)

                    total_pg += pg_loss.item()
                    total_v += v_loss.item()
                    total_logp += lp_new.mean().item()

                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                loss.backward()

                if (not is_warmup) or (not args.use_value_baseline):
                    grad_norm_actor_last = float(nn.utils.clip_grad_norm_(model.actor_ft.parameters(), args.max_grad_norm))
                    actor_optimizer.step()
                if args.use_value_baseline:
                    grad_norm_critic_last = float(nn.utils.clip_grad_norm_(model.critic.parameters(), args.max_grad_norm))
                    critic_optimizer.step()

                total_loss += loss.item()
                n_mb += 1

                if args.target_kl is not None and (args.policy_objective == "ppo_clip" or args.use_is):
                    avg_kl_so_far = total_kl / max(n_mb, 1)
                    if avg_kl_so_far > args.target_kl:
                        early_stop = True
                        early_stop_reason = f"target_kl={args.target_kl} avg_kl={avg_kl_so_far:.6f}"
                        break
            if early_stop:
                break

        # ===== 4. LOGGING =====
        avg_reward = rewards.sum(0).mean().item()
        avg_loss = total_loss / max(n_mb, 1)
        avg_pg = total_pg / max(n_mb, 1)
        avg_v = total_v / max(n_mb, 1)
        avg_logp = total_logp / max(n_mb, 1) if total_logp != 0 else 0.0
        avg_ratio = total_ratio / max(n_mb, 1) if total_ratio != 0 else 1.0
        avg_clipfrac = total_clipfrac / max(n_mb, 1) if total_clipfrac != 0 else 0.0
        avg_kl = total_kl / max(n_mb, 1) if total_kl != 0 else 0.0
        adv_mean = b_advantages.mean().item()
        adv_std = b_advantages.std().item()
        adv_pos_frac = (b_advantages > 0).float().mean().item()

        writer.add_scalar("train/reward", avg_reward, iteration)
        writer.add_scalar("train/loss", avg_loss, iteration)
        writer.add_scalar("train/pg_loss", avg_pg, iteration)
        writer.add_scalar("train/v_loss", avg_v, iteration)
        writer.add_scalar("train/logp_mean", avg_logp, iteration)
        writer.add_scalar("train/ratio_mean", avg_ratio, iteration)
        writer.add_scalar("train/clipfrac", avg_clipfrac, iteration)
        writer.add_scalar("train/approx_kl", avg_kl, iteration)
        writer.add_scalar("train/adv_mean", adv_mean, iteration)
        writer.add_scalar("train/adv_std", adv_std, iteration)
        writer.add_scalar("train/adv_pos_frac", adv_pos_frac, iteration)
        writer.add_scalar("train/rollout_ep_successes", n_success_rollout, iteration)
        writer.add_scalar("train/global_step", global_step, iteration)

        t_elapsed = time.time() - t_start
        warmup_tag = " [WARMUP]" if is_warmup else ""
        print(
            f"Iter {iteration}/{args.n_train_itr}{warmup_tag} | "
            f"r={avg_reward:.3f} | succ={n_success_rollout} | "
            f"loss={avg_loss:.6f} pg={avg_pg:.6f} v={avg_v:.6f} | "
            f"logp={avg_logp:.4f} ratio={avg_ratio:.4f} clipfrac={avg_clipfrac:.3f} kl={avg_kl:.6f} | "
            f"adv: mean={adv_mean:.4f} std={adv_std:.4f} pos={adv_pos_frac:.2f} | "
            f"grad_a={grad_norm_actor_last:.2f} grad_c={grad_norm_critic_last:.2f} | "
            f"time={t_elapsed:.1f}s"
        )
        if early_stop:
            print(f"  Early stop update: {early_stop_reason}")

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
