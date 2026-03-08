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
import sys
import time
import math
from collections import deque
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
    "p4b_reinforce_binary_schedclip",
    "p4c_reinforce_binary_normadv_schedclip",
    "p4d_reinforce_pm1_schedclip",
    "p4e_reinforce_pm1_normadv_schedclip",
    "p4f_awr_is_pm1_histmix",
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
    pm1_returns: bool = False
    norm_returns: bool = False

    # Policy objective
    policy_objective: str = "reinforce_is"  # reinforce_is | ppo_clip | awr_is_wbc
    use_is: bool = True
    is_clip_ratio: float = 0.2
    is_weight_max: float = 0.0
    awr_kl_coef: float = 0.0
    reinforce_use_ppo_clip_schedule: bool = False

    # Value / baseline
    use_value_baseline: bool = False
    use_gae: bool = False
    gae_lambda: float = 0.95
    norm_adv: bool = False
    vf_coef: float = 0.5
    n_critic_warmup_itr: int = 0
    actor_return_warmup_itr: int = 0

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
    offpolicy_history_iters: int = 0
    offpolicy_mix_ratio: float = 0.0
    offpolicy_precompute_mb: int = 4096

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
    def cli_overrode(flag: str) -> bool:
        candidates = {f"--{flag}", f"--{flag.replace('_', '-')}"}
        for arg in sys.argv[1:]:
            head = arg.split("=", 1)[0]
            if head in candidates:
                return True
        return False

    keep_gamma_denoising = cli_overrode("gamma_denoising")
    keep_is_clip_ratio = cli_overrode("is_clip_ratio")
    keep_is_weight_max = cli_overrode("is_weight_max")
    keep_awr_kl_coef = cli_overrode("awr_kl_coef")
    keep_critic_warmup = cli_overrode("n_critic_warmup_itr")
    keep_actor_return_warmup = cli_overrode("actor_return_warmup_itr")

    if args.bridge_stage not in STAGES:
        raise ValueError(f"Unknown bridge_stage={args.bridge_stage}. Choose from: {STAGES}")

    # S0 baseline
    args.policy_objective = "reinforce_is"
    args.use_is = True
    if not keep_is_clip_ratio:
        args.is_clip_ratio = 0.2
    args.reinforce_use_ppo_clip_schedule = False
    args.binarize_returns = True
    args.norm_returns = False
    if not keep_gamma_denoising:
        args.gamma_denoising = 1.0
    args.use_value_baseline = False
    args.use_gae = False
    args.norm_adv = False
    if not keep_critic_warmup:
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
        if not keep_gamma_denoising:
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
    if args.use_value_baseline and not keep_critic_warmup:
        args.n_critic_warmup_itr = 5

    if args.bridge_stage == "s7_full_dppo_equiv":
        args.policy_objective = "ppo_clip"
        args.use_is = True
        if not keep_is_clip_ratio:
            args.is_clip_ratio = 0.2

    # ===== PPO -> REINFORCE extraction bridge =====
    if args.bridge_stage.startswith("p"):
        # p0: mimic finetune.py training setup as closely as possible.
        args.policy_objective = "ppo_clip"
        args.use_is = True
        if not keep_is_clip_ratio:
            args.is_clip_ratio = 0.2
        args.binarize_returns = False
        args.norm_returns = True
        if not keep_gamma_denoising:
            args.gamma_denoising = 0.99
        args.use_value_baseline = True
        args.use_gae = True
        args.norm_adv = True
        args.vf_coef = 0.5
        if not keep_critic_warmup:
            args.n_critic_warmup_itr = 5

        # p1: only switch policy extraction from PPO-clip to IS-REINFORCE.
        if args.bridge_stage in ["p1_reinforce_is_gae", "p2_reinforce_is_td", "p3_reinforce_is_nobaseline", "p4_reinforce_is_final", "p4b_reinforce_binary_schedclip"]:
            args.policy_objective = "reinforce_is"
        if args.bridge_stage in ["p4c_reinforce_binary_normadv_schedclip", "p4d_reinforce_pm1_schedclip", "p4e_reinforce_pm1_normadv_schedclip", "p4f_awr_is_pm1_histmix"]:
            args.policy_objective = "reinforce_is"
        if args.bridge_stage == "p1_reinforce_is_gae":
            # Match PPO's trust-region scale more closely for the first bridge step.
            if not keep_is_clip_ratio:
                args.is_clip_ratio = 0.02
            args.reinforce_use_ppo_clip_schedule = True

        # p2: remove GAE (keep baseline/value learning).
        if args.bridge_stage in ["p2_reinforce_is_td", "p3_reinforce_is_nobaseline", "p4_reinforce_is_final", "p4b_reinforce_binary_schedclip"]:
            args.use_gae = False
            args.norm_adv = False
        if args.bridge_stage in ["p4c_reinforce_binary_normadv_schedclip", "p4d_reinforce_pm1_schedclip", "p4e_reinforce_pm1_normadv_schedclip", "p4f_awr_is_pm1_histmix"]:
            args.use_gae = False

        # p3: remove critic baseline/value module.
        if args.bridge_stage in ["p3_reinforce_is_nobaseline", "p4_reinforce_is_final", "p4b_reinforce_binary_schedclip"]:
            args.use_value_baseline = False
            if not keep_critic_warmup:
                args.n_critic_warmup_itr = 0
            args.vf_coef = 0.0
        if args.bridge_stage in ["p4c_reinforce_binary_normadv_schedclip", "p4d_reinforce_pm1_schedclip", "p4e_reinforce_pm1_normadv_schedclip", "p4f_awr_is_pm1_histmix"]:
            args.use_value_baseline = False
            if not keep_critic_warmup:
                args.n_critic_warmup_itr = 0
            args.vf_coef = 0.0

        # p4: match finetune_reinforce-style return signal.
        if args.bridge_stage in ["p4_reinforce_is_final", "p4b_reinforce_binary_schedclip"]:
            args.binarize_returns = True
            args.norm_returns = False
            if not keep_gamma_denoising:
                args.gamma_denoising = 1.0
        if args.bridge_stage == "p4b_reinforce_binary_schedclip":
            if not keep_is_clip_ratio:
                args.is_clip_ratio = 0.02
            args.reinforce_use_ppo_clip_schedule = True
        if args.bridge_stage in ["p4c_reinforce_binary_normadv_schedclip", "p4d_reinforce_pm1_schedclip", "p4e_reinforce_pm1_normadv_schedclip", "p4f_awr_is_pm1_histmix"]:
            args.binarize_returns = True
            args.norm_returns = False
            if not keep_gamma_denoising:
                args.gamma_denoising = 1.0
            if not keep_is_clip_ratio:
                args.is_clip_ratio = 0.02
            args.reinforce_use_ppo_clip_schedule = True
        if args.bridge_stage in ["p4c_reinforce_binary_normadv_schedclip", "p4e_reinforce_pm1_normadv_schedclip"]:
            args.norm_adv = True
        else:
            args.pm1_returns = False
        if args.bridge_stage in ["p4d_reinforce_pm1_schedclip", "p4f_awr_is_pm1_histmix"]:
            args.norm_adv = False
        if args.bridge_stage in ["p4d_reinforce_pm1_schedclip", "p4e_reinforce_pm1_normadv_schedclip", "p4f_awr_is_pm1_histmix"]:
            args.pm1_returns = True
        if args.bridge_stage == "p4f_awr_is_pm1_histmix":
            args.policy_objective = "awr_is_wbc"
            args.use_is = True
            args.reinforce_use_ppo_clip_schedule = False
            if not keep_is_weight_max:
                args.is_weight_max = 10.0
            if not keep_awr_kl_coef:
                args.awr_kl_coef = 1.0


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

    def build_reinforce_flat_dataset(obs_flat, chains_flat, adv_flat, ret_v_flat, oldv_flat):
        n_decisions = obs_flat.shape[0]
        sample_inds = torch.arange(n_decisions, device=device).repeat_interleave(K)
        denoise_inds = torch.arange(K, device=device).repeat(n_decisions)
        return {
            "obs": obs_flat[sample_inds].cpu(),
            "prev": chains_flat[sample_inds, denoise_inds].cpu(),
            "next": chains_flat[sample_inds, denoise_inds + 1].cpu(),
            "denoise": denoise_inds.cpu(),
            "adv": adv_flat[sample_inds].cpu(),
            "ret_v": ret_v_flat[sample_inds].cpu(),
            "oldv": oldv_flat[sample_inds].cpu(),
        }

    @torch.no_grad()
    def precompute_behavior_logp(flat_ds):
        n_flat = flat_ds["denoise"].shape[0]
        behavior_lp = []
        for start in range(0, n_flat, args.offpolicy_precompute_mb):
            end = min(start + args.offpolicy_precompute_mb, n_flat)
            mb_obs = {"state": flat_ds["obs"][start:end].to(device)}
            mb_prev = flat_ds["prev"][start:end].to(device)
            mb_next = flat_ds["next"][start:end].to(device)
            mb_denoise = flat_ds["denoise"][start:end].to(device)
            lp = model.get_logprobs_subsample(
                mb_obs, mb_prev, mb_next, mb_denoise, get_ent=False, use_base_policy=True
            )
            lp = lp[:, act_offset:act_offset + args.act_steps, :].mean(dim=(-1, -2))
            behavior_lp.append(lp.cpu())
        flat_ds["behavior_lp"] = torch.cat(behavior_lp, dim=0)
        return flat_ds

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
        f"  binarize_returns={args.binarize_returns}, pm1_returns={args.pm1_returns}, norm_returns={args.norm_returns}, "
        f"gamma_denoising={args.gamma_denoising}"
    )
    print(
        f"  use_value_baseline={args.use_value_baseline}, use_gae={args.use_gae}, "
        f"norm_adv={args.norm_adv}, warmup={args.n_critic_warmup_itr}, "
        f"actor_return_warmup={args.actor_return_warmup_itr}"
    )
    print(
        f"  policy_objective={args.policy_objective}, use_is={args.use_is}, "
        f"is_clip_ratio={args.is_clip_ratio}, reinforce_ppo_clip_schedule={args.reinforce_use_ppo_clip_schedule}, "
        f"target_kl={args.target_kl}"
    )
    print(f"  is_weight_max={args.is_weight_max}, awr_kl_coef={args.awr_kl_coef}")
    print(
        f"  offpolicy_history_iters={args.offpolicy_history_iters}, "
        f"offpolicy_mix_ratio={args.offpolicy_mix_ratio}"
    )

    use_reinforce_history = (
        args.policy_objective in {"reinforce_is", "awr_is_wbc"}
        and args.offpolicy_history_iters > 0
        and args.offpolicy_mix_ratio > 0
    )
    history_buffer = deque(maxlen=args.offpolicy_history_iters) if use_reinforce_history else None

    obs_raw, _ = train_envs.reset()
    if isinstance(obs_raw, np.ndarray):
        obs_raw = torch.from_numpy(obs_raw).float().to(device)
    else:
        obs_raw = obs_raw.float().to(device)
    obs_history = obs_raw.unsqueeze(1).repeat(1, args.cond_steps, 1)

    for iteration in range(1, args.n_train_itr + 1):
        t_start = time.time()
        model.eval()
        is_return_actor_warmup = args.use_value_baseline and (iteration <= args.actor_return_warmup_itr)
        is_warmup = (
            args.use_value_baseline
            and (iteration <= args.n_critic_warmup_itr)
            and not is_return_actor_warmup
        )

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
            if args.pm1_returns:
                returns = returns * 2.0 - 1.0
        if args.norm_returns:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        if args.use_value_baseline:
            if is_return_actor_warmup:
                advantages = returns
                returns_v = returns
            else:
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

        current_flat = None
        history_cat = None
        history_pool_samples = 0
        behavior_logp_mean = 0.0
        use_flat_replay = use_reinforce_history or args.policy_objective == "awr_is_wbc"
        if use_flat_replay:
            current_flat = build_reinforce_flat_dataset(
                b_obs, b_chains, b_advantages, b_returns_v, b_values
            )
            current_flat["reward_signal"] = returns.reshape(-1).repeat_interleave(K).cpu()
            current_flat = precompute_behavior_logp(current_flat)
            behavior_logp_mean = current_flat["behavior_lp"].mean().item()
            if history_buffer:
                history_cat = {
                    k: torch.cat([item[k] for item in history_buffer], dim=0)
                    for k in history_buffer[0].keys()
                }
                history_pool_samples = history_cat["denoise"].shape[0]
            writer.add_scalar("train/log_prob_behavior_mean", behavior_logp_mean, iteration)

        # ===== 3. UPDATE =====
        model.train()
        total_loss = 0.0
        total_pg = 0.0
        total_v = 0.0
        total_logp = 0.0
        total_ratio = 0.0
        total_clipfrac = 0.0
        total_kl = 0.0
        total_kl_penalty = 0.0
        total_is_weight = 0.0
        n_mb = 0
        grad_norm_actor_last = 0.0
        grad_norm_critic_last = 0.0

        total_samples = N * K
        onpolicy_mb_size = args.minibatch_size
        history_mb_size = 0
        if use_reinforce_history and history_pool_samples > 0:
            history_mb_size = min(
                int(round(args.minibatch_size * args.offpolicy_mix_ratio)),
                args.minibatch_size - 1,
            )
            onpolicy_mb_size = max(args.minibatch_size - history_mb_size, 1)

        early_stop = False
        early_stop_reason = ""
        for _ in range(args.update_epochs):
            perm_device = "cpu" if use_flat_replay else device
            perm = torch.randperm(total_samples, device=perm_device)
            mb_stride = onpolicy_mb_size if use_flat_replay else args.minibatch_size
            for mb_start in range(0, total_samples, mb_stride):
                mb_inds = perm[mb_start:mb_start + mb_stride]
                if len(mb_inds) == 0:
                    continue

                if use_flat_replay:
                    cur_idx = mb_inds.long()
                    mb_obs_state = current_flat["obs"][cur_idx]
                    mb_chains_prev = current_flat["prev"][cur_idx]
                    mb_chains_next = current_flat["next"][cur_idx]
                    denoise_inds = current_flat["denoise"][cur_idx]
                    mb_adv = current_flat["adv"][cur_idx]
                    mb_ret_v = current_flat["ret_v"][cur_idx]
                    mb_oldv = current_flat["oldv"][cur_idx]
                    mb_behavior_lp = current_flat["behavior_lp"][cur_idx]
                    mb_reward_signal = current_flat["reward_signal"][cur_idx]

                    if history_mb_size > 0 and history_cat is not None:
                        hist_idx = torch.randint(0, history_pool_samples, (history_mb_size,))
                        mb_obs_state = torch.cat([mb_obs_state, history_cat["obs"][hist_idx]], dim=0)
                        mb_chains_prev = torch.cat([mb_chains_prev, history_cat["prev"][hist_idx]], dim=0)
                        mb_chains_next = torch.cat([mb_chains_next, history_cat["next"][hist_idx]], dim=0)
                        denoise_inds = torch.cat([denoise_inds, history_cat["denoise"][hist_idx]], dim=0)
                        mb_adv = torch.cat([mb_adv, history_cat["adv"][hist_idx]], dim=0)
                        mb_ret_v = torch.cat([mb_ret_v, history_cat["ret_v"][hist_idx]], dim=0)
                        mb_oldv = torch.cat([mb_oldv, history_cat["oldv"][hist_idx]], dim=0)
                        mb_behavior_lp = torch.cat([mb_behavior_lp, history_cat["behavior_lp"][hist_idx]], dim=0)
                        mb_reward_signal = torch.cat([mb_reward_signal, history_cat["reward_signal"][hist_idx]], dim=0)

                    mb_obs = {"state": mb_obs_state.to(device)}
                    mb_chains_prev = mb_chains_prev.to(device)
                    mb_chains_next = mb_chains_next.to(device)
                    denoise_inds = denoise_inds.to(device)
                    mb_adv = mb_adv.to(device)
                    mb_ret_v = mb_ret_v.to(device)
                    mb_oldv = mb_oldv.to(device)
                    mb_behavior_lp = mb_behavior_lp.to(device)
                    mb_reward_signal = mb_reward_signal.to(device)
                else:
                    sample_inds = mb_inds // K
                    denoise_inds = mb_inds % K

                    mb_obs = {"state": b_obs[sample_inds]}
                    mb_chains_prev = b_chains[sample_inds, denoise_inds]
                    mb_chains_next = b_chains[sample_inds, denoise_inds + 1]
                    mb_adv = b_advantages[sample_inds]
                    mb_ret_v = b_returns_v[sample_inds]
                    mb_oldv = b_values[sample_inds]
                    mb_behavior_lp = None
                    mb_reward_signal = None

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
                elif args.policy_objective == "awr_is_wbc":
                    lp_new, _ = model.get_logprobs_subsample(
                        mb_obs, mb_chains_prev, mb_chains_next, denoise_inds, get_ent=True
                    )
                    lp_new = lp_new[:, act_offset:act_offset + args.act_steps, :].mean(dim=(-1, -2))

                    with torch.no_grad():
                        lp_old = model.get_logprobs_subsample(
                            mb_obs, mb_chains_prev, mb_chains_next, denoise_inds, get_ent=False, use_base_policy=True
                        )
                        lp_old = lp_old[:, act_offset:act_offset + args.act_steps, :].mean(dim=(-1, -2))

                    is_weight = (lp_old - mb_behavior_lp).exp()
                    if args.is_weight_max > 0:
                        is_weight = is_weight.clamp(min=0.0, max=args.is_weight_max)

                    sr_old = mb_reward_signal.mean()
                    mb_adv_eff = mb_reward_signal - sr_old
                    logratio = lp_new - lp_old
                    ratio = logratio.exp()
                    approx_kl = ((ratio - 1) - logratio).mean()

                    awr_loss = -(is_weight * mb_adv_eff * lp_new).mean()
                    pg_loss = awr_loss
                    kl_penalty = args.awr_kl_coef * approx_kl

                    if args.use_value_baseline:
                        v_pred = model.critic(mb_obs).squeeze(-1)
                        v_loss = 0.5 * ((v_pred - mb_ret_v) ** 2).mean()
                    else:
                        v_loss = torch.zeros([], device=device)

                    if is_warmup and args.use_value_baseline:
                        loss = args.vf_coef * v_loss
                    else:
                        loss = pg_loss + kl_penalty + (args.vf_coef * v_loss if args.use_value_baseline else 0.0)

                    total_pg += pg_loss.item()
                    total_v += v_loss.item()
                    total_logp += lp_new.mean().item()
                    total_ratio += is_weight.mean().item()
                    total_is_weight += is_weight.mean().item()
                    total_clipfrac += 0.0
                    total_kl += approx_kl.item()
                    total_kl_penalty += kl_penalty.item()
                    writer.add_scalar("train/is_weight_mean", is_weight.mean().item(), global_step)
                    writer.add_scalar("train/sr_old_batch", sr_old.item(), global_step)
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
                        # For replay samples, correct behavior->old mismatch
                        # separately from the PPO-style old->new ratio.
                        if mb_behavior_lp is not None:
                            is_weight = (lp_old - mb_behavior_lp).exp()
                            if args.is_weight_max > 0:
                                is_weight = is_weight.clamp(min=0.0, max=args.is_weight_max)
                        else:
                            is_weight = torch.ones_like(lp_old)

                        ratio = (lp_new - lp_old).exp()
                        if args.is_clip_ratio > 0:
                            clip_coef = get_reinforce_clip_coef(denoise_inds)
                            ratio_clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                            obj_core = torch.minimum(ratio * mb_adv_eff, ratio_clipped * mb_adv_eff)
                            clipfrac = ((ratio - ratio_clipped).abs() > 1e-12).float().mean().item()
                        else:
                            obj_core = ratio * mb_adv_eff
                            clipfrac = 0.0
                        obj = is_weight * obj_core
                        pg_loss = -obj.mean()
                        total_ratio += ratio.mean().item()
                        total_is_weight += is_weight.mean().item()
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
        avg_kl_penalty = total_kl_penalty / max(n_mb, 1) if total_kl_penalty != 0 else 0.0
        avg_is_weight = total_is_weight / max(n_mb, 1) if total_is_weight != 0 else 1.0
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
        writer.add_scalar("train/kl_penalty", avg_kl_penalty, iteration)
        writer.add_scalar("train/is_weight_mean_epoch", avg_is_weight, iteration)
        writer.add_scalar("train/adv_mean", adv_mean, iteration)
        writer.add_scalar("train/adv_std", adv_std, iteration)
        writer.add_scalar("train/adv_pos_frac", adv_pos_frac, iteration)
        writer.add_scalar("train/rollout_ep_successes", n_success_rollout, iteration)
        writer.add_scalar("train/global_step", global_step, iteration)
        writer.add_scalar("train/history_pool_samples", history_pool_samples, iteration)
        writer.add_scalar("train/log_prob_behavior_mean", behavior_logp_mean, iteration)

        t_elapsed = time.time() - t_start
        if is_return_actor_warmup:
            warmup_tag = " [RET-WARMUP]"
        elif is_warmup:
            warmup_tag = " [WARMUP]"
        else:
            warmup_tag = ""
        print(
            f"Iter {iteration}/{args.n_train_itr}{warmup_tag} | "
            f"r={avg_reward:.3f} | succ={n_success_rollout} | "
            f"loss={avg_loss:.6f} pg={avg_pg:.6f} v={avg_v:.6f} | "
            f"logp={avg_logp:.4f} logp_beta={behavior_logp_mean:.4f} isw={avg_is_weight:.4f} ratio={avg_ratio:.4f} clipfrac={avg_clipfrac:.3f} kl={avg_kl:.6f} klp={avg_kl_penalty:.6f} | "
            f"adv: mean={adv_mean:.4f} std={adv_std:.4f} pos={adv_pos_frac:.2f} | "
            f"hist={history_pool_samples} | "
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

        if use_reinforce_history:
            history_buffer.append(current_flat)

    print(f"Finetuning complete. Best sr_once={best_sr:.3f}")
    train_envs.close()
    writer.close()


if __name__ == "__main__":
    main()
