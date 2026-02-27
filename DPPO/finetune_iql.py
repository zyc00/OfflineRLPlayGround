"""
DPPO RL Finetune with IQL Advantage Estimation.

Instead of GAE (which requires iterative critic learning via PPO's v_loss),
this uses IQL to train Q(s,a) and V(s) on each iteration's rollout data,
then computes advantages as Q-V or GAE with IQL's V.

Known risk: IQL Q may not rank actions well (Issue #8 in CLAUDE.md).
Use --advantage_mode gae as fallback (GAE with IQL's V for bootstrapping).

Usage:
    python -m DPPO.finetune_iql \
        --pretrain_checkpoint runs/dppo_pretrain/best.pt \
        --env_id PickCube-v1 --control_mode pd_ee_delta_pos \
        --n_envs 100 --n_steps 100 \
        --ft_denoising_steps 10 --use_ddim --ddim_steps 10 \
        --iql_epochs 100 --iql_reward_scale 10.0 --advantage_mode qv
"""

import copy
import os
import time
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, List
from torch.utils.tensorboard import SummaryWriter

import tyro

from DPPO.model.mlp import DiffusionMLP
from DPPO.model.unet_wrapper import DiffusionUNet
from DPPO.model.critic import CriticObs
from DPPO.model.diffusion_ppo import PPODiffusion
from DPPO.make_env import make_train_envs
from DPPO.reward_scaler import RunningRewardScaler


class QNetworkDPPO(nn.Module):
    """Q(s, a_env) for DPPO IQL advantage estimation.

    Input: flattened obs_history (cond_dim) + first env action (action_dim).
    Output: scalar Q-value.
    """

    def __init__(self, cond_dim, action_dim, mlp_dims=[256, 256, 256]):
        super().__init__()
        dims = [cond_dim + action_dim] + mlp_dims + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, state_flat, action):
        """
        state_flat: (B, cond_dim) flattened normalized obs_history
        action: (B, action_dim) first env action (denormalized)
        Returns: (B, 1)
        """
        return self.net(torch.cat([state_flat, action], dim=-1))


def expectile_loss(diff, tau):
    """Asymmetric squared loss for IQL value learning."""
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return (weight * (diff ** 2)).mean()


def train_iql_on_rollout(
    obs, actions, rewards, next_obs, dones,
    q_net, q_target, v_net,
    q_optimizer, v_optimizer, args, device,
):
    """Train IQL Q(s,a) and V(s) on one iteration's rollout data.

    Standard IQL: Q learns Bellman backup, V learns expectile of Q_target,
    Q_target updated via Polyak averaging.

    Rewards are scaled by iql_reward_scale internally (Issue #13).

    Args:
        obs: (N, cond_dim) flattened normalized obs
        actions: (N, action_dim) first env actions (denormalized)
        rewards: (N,) raw rewards
        next_obs: (N, cond_dim) flattened normalized next obs
        dones: (N,) float done flags
        q_net, q_target: QNetworkDPPO
        v_net: CriticObs (model.critic), accepts flat tensor
        q_optimizer, v_optimizer: optimizers
    Returns:
        dict with q_loss, v_loss stats
    """
    N = obs.shape[0]
    scaled_rewards = rewards * args.iql_reward_scale

    total_q_loss = 0.0
    total_v_loss = 0.0
    n_updates = 0

    for epoch in range(args.iql_epochs):
        perm = torch.randperm(N, device=device)

        for start in range(0, N, args.iql_batch_size):
            idx = perm[start : start + args.iql_batch_size]

            s = obs[idx]
            a = actions[idx]
            r = scaled_rewards[idx]
            ns = next_obs[idx]
            d = dones[idx]

            # Q loss: MSE to r*scale + gamma * V(s') * (1-done)
            with torch.no_grad():
                v_next = v_net(ns).squeeze(-1)
                q_bellman = r + args.gamma * v_next * (1.0 - d)

            q_pred = q_net(s, a).squeeze(-1)
            q_loss = 0.5 * ((q_pred - q_bellman) ** 2).mean()

            q_optimizer.zero_grad()
            q_loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), args.iql_grad_clip)
            q_optimizer.step()

            # V loss: expectile regression against Q_target
            with torch.no_grad():
                q_val = q_target(s, a).squeeze(-1)

            v_pred = v_net(s).squeeze(-1)
            v_loss = expectile_loss(q_val - v_pred, args.iql_expectile_tau)

            v_optimizer.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(v_net.parameters(), args.iql_grad_clip)
            v_optimizer.step()

            # Polyak update Q_target
            with torch.no_grad():
                for p, pt in zip(q_net.parameters(), q_target.parameters()):
                    pt.data.mul_(1.0 - args.iql_polyak_tau).add_(
                        p.data, alpha=args.iql_polyak_tau
                    )

            total_q_loss += q_loss.item()
            total_v_loss += v_loss.item()
            n_updates += 1

    return {
        "q_loss": total_q_loss / max(n_updates, 1),
        "v_loss": total_v_loss / max(n_updates, 1),
    }


@dataclass
class Args:
    # Pretrained checkpoint
    pretrain_checkpoint: str = "runs/dppo_pretrain/dppo_1000traj_unet_T20_5k/best.pt"

    # Environment
    env_id: str = "PickCube-v1"
    control_mode: str = "pd_ee_delta_pos"
    max_episode_steps: int = 100
    n_envs: int = 100
    n_eval_envs: int = 10
    sim_backend: str = "gpu"

    # DPPO specific (overridden from checkpoint at runtime)
    ft_denoising_steps: int = 10
    denoising_steps: int = 20
    horizon_steps: int = 16
    cond_steps: int = 2
    act_steps: int = 8

    # DDIM
    use_ddim: bool = False
    ddim_steps: int = 5

    # Critic (V network, shared between IQL and PPODiffusion)
    critic_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    critic_activation: str = "Mish"

    # IQL
    iql_epochs: int = 100
    iql_batch_size: int = 256
    iql_lr: float = 3e-4
    iql_reward_scale: float = 10.0
    iql_expectile_tau: float = 0.7
    iql_polyak_tau: float = 0.005
    iql_grad_clip: float = 0.5
    iql_q_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    advantage_mode: str = "qv"  # "qv" = Q-V, "gae" = GAE with IQL's V

    # RL
    n_train_itr: int = 301
    n_steps: int = 100
    gamma: float = 0.999
    gae_lambda: float = 0.95
    reward_scale_running: bool = False  # Disabled: IQL uses iql_reward_scale
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
    clip_adv_min: Optional[float] = None
    vf_coef: float = 0.0  # IQL trains V; set > 0 to also use PPO v_loss

    # Optimization
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3  # Only effective if vf_coef > 0
    max_grad_norm: float = 1.0
    n_critic_warmup_itr: int = 0

    # Exploration noise
    min_sampling_denoising_std: float = 0.01
    min_logprob_denoising_std: float = 0.1

    # Eval
    eval_freq: int = 10
    num_eval_episodes: int = 100

    # Logging
    exp_name: Optional[str] = None
    seed: int = 0
    save_dir: str = "runs/dppo_finetune_iql"


def main():
    args = tyro.cli(Args)

    if args.exp_name is None:
        args.exp_name = (
            f"dppo_ft_iql_{args.env_id}_K{args.ft_denoising_steps}"
            f"_{args.advantage_mode}_envs{args.n_envs}_steps{args.n_steps}"
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

    no_obs_norm = ckpt.get("no_obs_norm", False)
    no_action_norm = ckpt.get("no_action_norm", False)

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

    cond_dim = obs_dim * args.cond_steps
    act_offset = args.cond_steps - 1 if network_type == "unet" else 0

    print(f"Pretrained: {args.pretrain_checkpoint}")
    print(f"  network_type={network_type}, obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"  denoising_steps={args.denoising_steps}, ft_denoising_steps={args.ft_denoising_steps}")
    print(f"  horizon_steps={args.horizon_steps}, cond_steps={args.cond_steps}, act_steps={args.act_steps}")
    print(f"  act_offset={act_offset}, no_obs_norm={no_obs_norm}, no_action_norm={no_action_norm}")
    print(f"  advantage_mode={args.advantage_mode}, iql_epochs={args.iql_epochs}")
    print(f"  iql_reward_scale={args.iql_reward_scale}, iql_expectile_tau={args.iql_expectile_tau}")

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

    # Build critic V(s) — shared with IQL
    critic = CriticObs(
        cond_dim=cond_dim,
        mlp_dims=args.critic_dims,
        activation_type=args.critic_activation,
        residual_style=True,
    )

    # Build PPODiffusion model
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

    # Load pretrained weights into actor
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"], strict=False)
    else:
        model.load_state_dict(ckpt["model"], strict=False)

    model._sanitize_eta()
    model.actor_ft.load_state_dict(model.actor.state_dict())

    for param in model.actor.parameters():
        param.requires_grad = False

    # Build Q network for IQL
    q_net = QNetworkDPPO(cond_dim, action_dim, mlp_dims=args.iql_q_dims).to(device)
    q_target = copy.deepcopy(q_net)
    for param in q_target.parameters():
        param.requires_grad = False

    n_actor_params = sum(p.numel() for p in model.actor_ft.parameters() if p.requires_grad)
    n_critic_params = sum(p.numel() for p in model.critic.parameters() if p.requires_grad)
    n_q_params = sum(p.numel() for p in q_net.parameters() if p.requires_grad)
    print(f"Actor params: {n_actor_params:,}, Critic(V) params: {n_critic_params:,}, Q params: {n_q_params:,}")

    # Optimizers
    actor_optimizer = torch.optim.Adam(model.actor_ft.parameters(), lr=args.actor_lr)
    # critic_optimizer: only used if vf_coef > 0 (PPO also updates V)
    critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=args.critic_lr)
    # IQL optimizers
    q_optimizer = torch.optim.Adam(q_net.parameters(), lr=args.iql_lr)
    v_optimizer = torch.optim.Adam(model.critic.parameters(), lr=args.iql_lr)

    reward_scaler = RunningRewardScaler(gamma=args.gamma) if args.reward_scale_running else None

    # Train environments
    use_gpu_env = args.sim_backend == "gpu"
    train_envs = make_train_envs(
        env_id=args.env_id,
        num_envs=args.n_envs,
        sim_backend=args.sim_backend,
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
    )

    n_decision_steps = args.n_steps
    K = args.ft_denoising_steps

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
        """Eval using training GPU env: reset + deterministic rollout."""
        model.eval()
        total_success = 0
        total_eps = 0
        for _ in range(n_rounds):
            obs_r, _ = train_envs.reset()
            obs_r = obs_r.float().to(device) if not isinstance(obs_r, torch.Tensor) else obs_r.float()
            obs_h = obs_r.unsqueeze(1).repeat(1, args.cond_steps, 1)
            for step in range(args.max_episode_steps // args.act_steps + 1):
                cond_eval = {"state": normalize_obs(obs_h)}
                samples_eval = model(cond_eval, deterministic=True)
                ac_eval = denormalize_actions(samples_eval.trajectories)
                for a_idx in range(args.act_steps):
                    act_idx = act_offset + a_idx
                    action_eval = ac_eval[:, min(act_idx, ac_eval.shape[1] - 1)]
                    obs_new_eval, rew_eval, term_eval, trunc_eval, _ = train_envs.step(action_eval)
                    obs_new_eval = obs_new_eval.float()
                    total_success += (rew_eval.float() > 0.5).sum().item()
                    rm = term_eval | trunc_eval
                    if rm.any():
                        obs_h[rm] = obs_new_eval[rm].unsqueeze(1).repeat(1, args.cond_steps, 1)
                    obs_h[~rm] = torch.cat([obs_h[~rm, 1:], obs_new_eval[~rm].unsqueeze(1)], dim=1)
            total_eps += args.n_envs
        return total_success / max(total_eps, 1)

    best_sr = -1.0
    global_step = 0

    batch_size = args.n_envs * n_decision_steps
    print(f"\nStarting DPPO-IQL finetuning for {args.n_train_itr} iterations...")
    print(f"  n_envs={args.n_envs}, n_steps={n_decision_steps}, act_steps={args.act_steps}, sim_backend={args.sim_backend}")
    print(f"  batch_size={batch_size}, total_ppo_samples={batch_size * K}")
    print(f"  gamma={args.gamma}, gamma_denoising={args.gamma_denoising}")
    print(f"  advantage_mode={args.advantage_mode}")
    print(f"  iql: epochs={args.iql_epochs}, batch={args.iql_batch_size}, lr={args.iql_lr}")
    print(f"  iql: reward_scale={args.iql_reward_scale}, tau={args.iql_expectile_tau}")
    if args.use_ddim:
        print(f"  DDIM: ddim_steps={args.ddim_steps}, ft_denoising_steps={args.ft_denoising_steps}")
    else:
        print(f"  DDPM: denoising_steps={args.denoising_steps}, ft_denoising_steps={args.ft_denoising_steps}")
    print(f"  min_sampling_std={args.min_sampling_denoising_std}, min_logprob_std={args.min_logprob_denoising_std}")
    print(f"  vf_coef={args.vf_coef}, actor_lr={args.actor_lr}")
    if args.n_critic_warmup_itr > 0:
        print(f"  critic warmup: {args.n_critic_warmup_itr} iterations (actor frozen, IQL trains V)")

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
        obs_trajs = []            # (n_steps, n_envs, cond_steps, obs_dim) normalized
        chains_trajs = []         # (n_steps, n_envs, K+1, horizon_steps, action_dim)
        reward_trajs = []         # (n_steps, n_envs)
        done_trajs = []           # (n_steps, n_envs)
        first_actions_trajs = []  # (n_steps, n_envs, action_dim) denormalized

        n_success_rollout = 0

        for step in range(n_decision_steps):
            obs_norm = normalize_obs(obs_history)
            cond = {"state": obs_norm}

            with torch.no_grad():
                samples = model(cond, deterministic=False, return_chain=True)
                action_chunk = samples.trajectories
                chains = samples.chains

            obs_trajs.append(obs_norm.clone())
            chains_trajs.append(chains.clone())

            # Denormalize actions for env
            action_chunk_denorm = denormalize_actions(action_chunk)

            # Collect first env action for IQL Q(s, a)
            first_actions_trajs.append(action_chunk_denorm[:, act_offset].clone())

            # Execute act_steps in env
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

                reset_mask = term_t | trunc_t
                if reset_mask.any():
                    obs_history[reset_mask] = obs_new[reset_mask].unsqueeze(1).repeat(1, args.cond_steps, 1)
                obs_history[~reset_mask] = torch.cat(
                    [obs_history[~reset_mask, 1:], obs_new[~reset_mask].unsqueeze(1)], dim=1
                )
                global_step += args.n_envs

            reward_trajs.append(step_reward)
            done_trajs.append(step_done.float())

        rewards = torch.stack(reward_trajs)     # (n_steps, n_envs)
        dones = torch.stack(done_trajs)         # (n_steps, n_envs)

        # ===== 2. REWARD SCALING (optional, disabled by default) =====
        if reward_scaler is not None:
            rewards = reward_scaler.update_and_scale(rewards, dones)
        rewards = rewards * args.reward_scale_const

        # ===== 3. IQL TRAINING =====
        obs_stacked = torch.stack(obs_trajs)       # (n_steps, n_envs, cond_steps, obs_dim)
        chains_stacked = torch.stack(chains_trajs)  # (n_steps, n_envs, K+1, horizon, act_dim)
        N = n_decision_steps * args.n_envs

        # Prepare IQL transition data
        iql_obs = obs_stacked.reshape(N, cond_dim)

        first_actions_stacked = torch.stack(first_actions_trajs)  # (n_steps, n_envs, action_dim)
        iql_actions = first_actions_stacked.reshape(N, action_dim)

        iql_rewards = rewards.reshape(N)
        iql_dones = dones.reshape(N)

        # Next obs: shift by 1, last step uses current obs_history
        next_obs_stacked = torch.zeros_like(obs_stacked)
        next_obs_stacked[:-1] = obs_stacked[1:]
        next_obs_stacked[-1] = normalize_obs(obs_history)
        iql_next_obs = next_obs_stacked.reshape(N, cond_dim)

        iql_stats = train_iql_on_rollout(
            iql_obs, iql_actions, iql_rewards, iql_next_obs, iql_dones,
            q_net, q_target, model.critic,
            q_optimizer, v_optimizer, args, device,
        )

        # ===== 3b. COMPUTE ADVANTAGES =====
        with torch.no_grad():
            if args.advantage_mode == "qv":
                # Q-V advantage (both in scaled reward space)
                q_values = q_net(iql_obs, iql_actions).squeeze(-1)  # (N,)
                v_values = model.critic(iql_obs).squeeze(-1)        # (N,)
                advantages_flat = q_values - v_values
                returns_flat = q_values    # for v_loss (unused with vf_coef=0)
                values_flat = v_values

                advantages = advantages_flat.reshape(n_decision_steps, args.n_envs)
                returns = returns_flat.reshape(n_decision_steps, args.n_envs)
                values = values_flat.reshape(n_decision_steps, args.n_envs)

            elif args.advantage_mode == "gae":
                # GAE with IQL-trained V for bootstrapping
                # Use scaled rewards to match V's scale (V trained on scaled rewards)
                v_all = model.critic(iql_obs).squeeze(-1).reshape(
                    n_decision_steps, args.n_envs
                )
                obs_last_flat = normalize_obs(obs_history).reshape(args.n_envs, cond_dim)
                v_next_bootstrap = model.critic(obs_last_flat).squeeze(-1)

                scaled_rewards = rewards * args.iql_reward_scale

                advantages = torch.zeros_like(rewards)
                lastgaelam = 0.0
                for t in reversed(range(n_decision_steps)):
                    next_not_done = 1.0 - dones[t]
                    if t == n_decision_steps - 1:
                        nextvalues = v_next_bootstrap
                    else:
                        nextvalues = v_all[t + 1]
                    delta = scaled_rewards[t] + args.gamma * nextvalues * next_not_done - v_all[t]
                    advantages[t] = lastgaelam = (
                        delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam
                    )

                returns = advantages + v_all
                values = v_all
            else:
                raise ValueError(f"Unknown advantage_mode: {args.advantage_mode}")

        # ===== 4. COMPUTE LOGPROBS =====
        all_logprobs = []
        logprob_batch_size = 2048

        for i in range(0, N, logprob_batch_size):
            end = min(i + logprob_batch_size, N)
            step_inds = torch.arange(i, end)
            s_idx = step_inds // args.n_envs
            e_idx = step_inds % args.n_envs

            batch_obs = obs_stacked[s_idx, e_idx]
            batch_chains = chains_stacked[s_idx, e_idx]

            with torch.no_grad():
                batch_lp = model.get_logprobs({"state": batch_obs}, batch_chains)
                batch_lp = batch_lp.reshape(end - i, K, args.horizon_steps, action_dim)
            all_logprobs.append(batch_lp)

        all_logprobs_flat = torch.cat(all_logprobs, dim=0)  # (N, K, horizon, act_dim)

        # ===== 5. PPO UPDATE =====
        b_obs = obs_stacked.reshape(N, args.cond_steps, obs_dim)
        b_chains = chains_stacked.reshape(N, K + 1, args.horizon_steps, action_dim)
        b_logprobs = all_logprobs_flat
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
            if args.vf_coef > 0:
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

                if is_warmup:
                    loss = args.vf_coef * v_loss / grad_acc
                else:
                    loss = (pg_loss + args.vf_coef * v_loss) / grad_acc

                loss.backward()
                acc_count += 1

                if acc_count >= grad_acc:
                    if not is_warmup:
                        nn.utils.clip_grad_norm_(model.actor_ft.parameters(), args.max_grad_norm)
                    if args.vf_coef > 0:
                        nn.utils.clip_grad_norm_(model.critic.parameters(), args.max_grad_norm)

                    if not is_warmup:
                        actor_optimizer.step()
                    if args.vf_coef > 0:
                        critic_optimizer.step()

                    actor_optimizer.zero_grad()
                    if args.vf_coef > 0:
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
                if args.vf_coef > 0:
                    nn.utils.clip_grad_norm_(model.critic.parameters(), args.max_grad_norm)
                if not is_warmup:
                    actor_optimizer.step()
                if args.vf_coef > 0:
                    critic_optimizer.step()
                actor_optimizer.zero_grad()
                if args.vf_coef > 0:
                    critic_optimizer.zero_grad()

            if early_stop:
                break

        # ===== LOGGING =====
        avg_reward = rewards.sum(0).mean().item()
        avg_pg_loss = epoch_pg_loss / max(n_mb, 1)
        avg_v_loss = epoch_v_loss / max(n_mb, 1)
        avg_kl = epoch_kl / max(n_mb, 1)
        avg_ratio = epoch_ratio_mean / max(n_mb, 1)

        # Advantage stats (unscaled for interpretability)
        adv_unscaled = b_advantages / args.iql_reward_scale
        adv_mean = adv_unscaled.mean().item()
        adv_std = adv_unscaled.std().item()
        adv_pos_frac = (b_advantages > 0).float().mean().item()

        writer.add_scalar("train/reward", avg_reward, iteration)
        writer.add_scalar("train/pg_loss", avg_pg_loss, iteration)
        writer.add_scalar("train/v_loss", avg_v_loss, iteration)
        writer.add_scalar("train/approx_kl", avg_kl, iteration)
        writer.add_scalar("train/rollout_successes", n_success_rollout, iteration)
        writer.add_scalar("train/global_step", global_step, iteration)
        writer.add_scalar("train/ratio_mean", avg_ratio, iteration)
        writer.add_scalar("train/adv_pos_frac", adv_pos_frac, iteration)
        writer.add_scalar("train/adv_mean", adv_mean, iteration)
        writer.add_scalar("train/adv_std", adv_std, iteration)
        writer.add_scalar("iql/q_loss", iql_stats["q_loss"], iteration)
        writer.add_scalar("iql/v_loss", iql_stats["v_loss"], iteration)

        t_elapsed = time.time() - t_start

        if iteration % 5 == 0 or iteration <= 5:
            warmup_tag = " [WARMUP]" if is_warmup else ""
            print(
                f"Iter {iteration}/{args.n_train_itr}{warmup_tag} | "
                f"r={avg_reward:.3f} | succ={n_success_rollout} | pg={avg_pg_loss:.6f} | "
                f"kl={avg_kl:.6f} | ratio={avg_ratio:.4f} | "
                f"adv: mean={adv_mean:.4f} std={adv_std:.4f} pos={adv_pos_frac:.2f} | "
                f"iql: q={iql_stats['q_loss']:.4f} v={iql_stats['v_loss']:.4f} | "
                f"ep={epoch+1} | time={t_elapsed:.1f}s"
            )

        # ===== EVALUATE =====
        if iteration == 1 or iteration % args.eval_freq == 0:
            sr_once = evaluate_gpu_inline(n_rounds=3)
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
                    "q_net": q_net.state_dict(),
                    "q_target": q_target.state_dict(),
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
        "q_net": q_net.state_dict(),
        "q_target": q_target.state_dict(),
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
