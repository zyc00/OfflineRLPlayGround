"""Critic analysis for PegInsertion diffusion policy.

Collects on-policy rollout data from pretrained DP, then trains V(s) with
different methods and compares against MC1 ground truth.

Methods: MC1 regression, TD(0), TD+EMA, GAE (iterative)
Sweep:   reward_scale (1, 10), data size (1, 5, 10 rollouts)

Usage:
    python -u -m DPPO.critic_analysis \
      --pretrain_checkpoint runs/dppo_pretrain/dppo_pretrain_PegInsertionSide-v1_T100_H16_1M/best.pt \
      --n_rollouts 10 --td_epochs 500
"""

import copy
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Optional
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tyro

from DPPO.model.mlp import DiffusionMLP
from DPPO.model.unet_wrapper import DiffusionUNet
from DPPO.model.diffusion import DiffusionModel
from DPPO.make_env import make_train_envs


@dataclass
class Args:
    pretrain_checkpoint: str = "runs/dppo_pretrain/dppo_pretrain_PegInsertionSide-v1_T100_H16_1M/best.pt"

    # Environment
    env_id: str = "PegInsertionSide-v1"
    control_mode: str = "pd_joint_delta_pos"
    max_episode_steps: int = 200
    n_envs: int = 100
    sim_backend: str = "cpu"

    # Architecture (overridden from checkpoint)
    denoising_steps: int = 100
    horizon_steps: int = 16
    cond_steps: int = 2
    act_steps: int = 8

    # DDIM
    use_ddim: bool = True
    ddim_steps: int = 10

    # Rollout
    n_rollouts: int = 3   # Number of rollout rounds (each = n_envs * n_steps samples)
    n_steps: int = 25     # Decision steps per rollout

    # Sampling noise
    min_sampling_denoising_std: float = 0.01

    # V training
    gamma: float = 0.999
    gae_lambda: float = 0.95
    lr: float = 3e-4
    minibatch_size: int = 1000
    td_epochs: int = 500
    gae_iters: int = 5
    gae_epochs: int = 100
    ema_tau: float = 0.005
    patience: int = 20

    # Sweep
    reward_scales: tuple = (1.0, 10.0, 100.0)
    hidden_dim: int = 256
    critic_layers: int = 3

    # Convergence tracking
    eval_freq: int = 10   # Evaluate r(V,MC1) every N epochs
    save_dir: str = "runs/critic_analysis"

    seed: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def main():
    args = tyro.cli(Args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== Load pretrained DP model =====
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

    cond_dim = obs_dim * args.cond_steps
    act_offset = args.cond_steps - 1 if network_type == "unet" else 0

    if network_type == "unet":
        network = DiffusionUNet(
            action_dim=action_dim, horizon_steps=args.horizon_steps,
            cond_dim=cond_dim,
            diffusion_step_embed_dim=pretrain_args.get("diffusion_step_embed_dim", 64),
            down_dims=pretrain_args.get("unet_dims", [64, 128, 256]),
            n_groups=pretrain_args.get("n_groups", 8),
        )
    else:
        network = DiffusionMLP(
            action_dim=action_dim, horizon_steps=args.horizon_steps,
            cond_dim=cond_dim,
            time_dim=pretrain_args.get("time_dim", 16),
            mlp_dims=pretrain_args.get("mlp_dims", [512, 512, 512]),
            activation_type=pretrain_args.get("activation_type", "Mish"),
            residual_style=pretrain_args.get("residual_style", True),
        )

    model = DiffusionModel(
        network=network, horizon_steps=args.horizon_steps,
        obs_dim=obs_dim, action_dim=action_dim, device=device,
        denoising_steps=args.denoising_steps,
        denoised_clip_value=1.0, randn_clip_value=3,
        final_action_clip_value=1.0, predict_epsilon=True,
    )
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"], strict=False)
    else:
        model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"Loaded DP model: {args.pretrain_checkpoint}")
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}, cond_dim={cond_dim}")

    def normalize_obs(obs):
        if no_obs_norm:
            return obs
        return (obs - obs_min) / (obs_max - obs_min + 1e-8) * 2.0 - 1.0

    def denormalize_actions(actions):
        if no_action_norm:
            return actions
        return (actions + 1.0) / 2.0 * (action_max - action_min) + action_min

    # ===== Create environments =====
    use_gpu_env = args.sim_backend == "gpu"
    train_envs = make_train_envs(
        env_id=args.env_id, num_envs=args.n_envs,
        sim_backend=args.sim_backend, control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps, seed=args.seed,
    )

    # ===== Phase 1: Collect rollout data =====
    print(f"\n{'='*70}")
    print(f"Phase 1: Collecting {args.n_rollouts} rollouts "
          f"({args.n_rollouts * args.n_envs * args.n_steps} samples)")
    print(f"{'='*70}")

    all_obs = []       # list of (n_steps, n_envs, cond_steps, obs_dim)
    all_rewards = []   # list of (n_steps, n_envs)
    all_dones = []     # list of (n_steps, n_envs)
    all_next_obs = []  # list of (n_envs, cond_steps, obs_dim) — last obs

    total_reward_events = 0

    for rollout_idx in range(args.n_rollouts):
        t0 = time.time()
        obs_raw, _ = train_envs.reset()
        if isinstance(obs_raw, np.ndarray):
            obs_raw = torch.from_numpy(obs_raw).float().to(device)
        else:
            obs_raw = obs_raw.float().to(device)
        obs_history = obs_raw.unsqueeze(1).repeat(1, args.cond_steps, 1)

        obs_trajs = []
        reward_trajs = []
        done_trajs = []
        n_succ = 0

        for step in range(args.n_steps):
            obs_norm = normalize_obs(obs_history)
            cond = {"state": obs_norm}

            with torch.no_grad():
                if args.use_ddim:
                    samples = model(cond, deterministic=False,
                                    min_sampling_denoising_std=args.min_sampling_denoising_std,
                                    ddim_steps=args.ddim_steps)
                else:
                    samples = model(cond, deterministic=False,
                                    min_sampling_denoising_std=args.min_sampling_denoising_std)
                action_chunk = samples.trajectories

            obs_trajs.append(obs_norm.clone())
            action_chunk_denorm = denormalize_actions(action_chunk)

            step_reward = torch.zeros(args.n_envs, device=device)
            step_done = torch.zeros(args.n_envs, dtype=torch.bool, device=device)

            for a_idx in range(args.act_steps):
                act_idx = act_offset + a_idx
                if act_idx < action_chunk_denorm.shape[1]:
                    action = action_chunk_denorm[:, act_idx]
                else:
                    action = action_chunk_denorm[:, -1]

                if use_gpu_env:
                    obs_new, reward, terminated, truncated, _ = train_envs.step(action)
                    obs_new = obs_new.float()
                    reward_t = reward.float()
                    term_t = terminated.bool()
                    trunc_t = truncated.bool()
                else:
                    obs_new, reward, terminated, truncated, _ = train_envs.step(action.cpu().numpy())
                    obs_new = torch.from_numpy(obs_new).float().to(device)
                    reward_t = torch.from_numpy(np.array(reward)).float().to(device)
                    term_t = torch.from_numpy(np.array(terminated)).bool().to(device)
                    trunc_t = torch.from_numpy(np.array(truncated)).bool().to(device)

                step_reward += reward_t * (~step_done).float()
                step_done = step_done | term_t | trunc_t
                n_succ += (reward_t > 0.5).sum().item()

                reset_mask = term_t | trunc_t
                if reset_mask.any():
                    obs_history[reset_mask] = obs_new[reset_mask].unsqueeze(1).repeat(1, args.cond_steps, 1)
                obs_history[~reset_mask] = torch.cat(
                    [obs_history[~reset_mask, 1:], obs_new[~reset_mask].unsqueeze(1)], dim=1
                )

            reward_trajs.append(step_reward)
            done_trajs.append(step_done.float())

        # Store this rollout
        obs_t = torch.stack(obs_trajs)       # (T, E, cond_steps, obs_dim)
        rew_t = torch.stack(reward_trajs)    # (T, E)
        don_t = torch.stack(done_trajs)      # (T, E)
        next_obs_t = normalize_obs(obs_history)  # (E, cond_steps, obs_dim)

        all_obs.append(obs_t)
        all_rewards.append(rew_t)
        all_dones.append(don_t)
        all_next_obs.append(next_obs_t)

        total_reward_events += n_succ
        elapsed = time.time() - t0
        print(f"  Rollout {rollout_idx+1}/{args.n_rollouts}: "
              f"reward_events={n_succ}, time={elapsed:.1f}s")

    print(f"Total reward events: {total_reward_events} "
          f"across {args.n_rollouts} rollouts x {args.n_envs} envs")

    # ===== Phase 2: Compute MC1 returns =====
    print(f"\n{'='*70}")
    print(f"Phase 2: Computing MC1 returns (gamma={args.gamma})")
    print(f"{'='*70}")

    def compute_mc1(rewards, dones):
        """Compute MC1 backward returns for a single rollout, respecting episode boundaries."""
        T, E = rewards.shape
        mc1 = torch.zeros_like(rewards)
        running = torch.zeros(E, device=rewards.device)
        for t in reversed(range(T)):
            running = rewards[t] + args.gamma * (1.0 - dones[t]) * running
            mc1[t] = running
        return mc1

    # Compute MC1 per rollout (no cross-rollout leakage), then concatenate
    mc1_per_roll = [compute_mc1(r, d) for r, d in zip(all_rewards, all_dones)]

    # Concatenate for training (flat sample access)
    full_obs = torch.cat(all_obs, dim=0)
    full_rew = torch.cat(all_rewards, dim=0)
    full_don = torch.cat(all_dones, dim=0)
    full_mc1 = torch.cat(mc1_per_roll, dim=0)

    T_full, E = full_rew.shape
    N_total = T_full * E
    print(f"  Data: T={T_full}, E={E}, N={N_total}")
    print(f"  MC1 return: mean={full_mc1.mean():.3f}, std={full_mc1.std():.3f}, "
          f"min={full_mc1.min():.3f}, max={full_mc1.max():.3f}")
    print(f"  Rewards: mean={full_rew.mean():.4f}, sum_per_env={full_rew.sum(0).mean():.3f}")

    # ===== Phase 3: Train V with different methods =====
    print(f"\n{'='*70}")
    print(f"Phase 3: V learning methods comparison")
    print(f"{'='*70}")

    D = cond_dim  # flattened obs input dim

    def make_critic(hidden_dim=256, n_layers=3):
        layers = []
        dims = [D] + [hidden_dim] * n_layers + [1]
        for i in range(len(dims) - 1):
            layers.append(layer_init(nn.Linear(dims[i], dims[i+1]),
                                     std=np.sqrt(2) if i < len(dims)-2 else 1.0))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers).to(device)

    def eval_critic(critic, obs, mc1_gt):
        """Evaluate critic V(s) vs MC1 ground truth."""
        critic.eval()
        with torch.no_grad():
            T, E = obs.shape[:2]
            flat_obs = obs.reshape(-1, args.cond_steps, obs_dim).reshape(-1, D)
            v = critic(flat_obs).view(-1).cpu().numpy()
        gt = mc1_gt.reshape(-1).cpu().numpy()
        r_val = pearsonr(v, gt)[0] if v.std() > 1e-8 and gt.std() > 1e-8 else 0.0
        rho_val = spearmanr(v, gt).correlation if v.std() > 1e-8 and gt.std() > 1e-8 else 0.0
        ratio = v.std() / (gt.std() + 1e-8)
        return r_val, rho_val, ratio, v.mean(), v.std()

    def eval_convergence(critic, rs):
        """Quick eval of r(V,MC1) — undo reward scaling mentally or literally."""
        critic.eval()
        with torch.no_grad():
            flat = full_obs.reshape(-1, args.cond_steps, obs_dim).reshape(-1, D)
            v = critic(flat).view(-1)
            if rs != 1.0:
                v = v / rs
            v = v.cpu().numpy()
        gt = full_mc1.reshape(-1).cpu().numpy()
        if v.std() > 1e-8 and gt.std() > 1e-8:
            return pearsonr(v, gt)[0], spearmanr(v, gt).correlation
        return 0.0, 0.0

    # --- Method 1: MC1 regression ---
    def train_mc1(obs, mc1_returns, reward_scale=1.0, label="MC1"):
        T, E = obs.shape[:2]
        critic = make_critic()
        opt = optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)
        flat_obs = obs.reshape(-1, args.cond_steps, obs_dim).reshape(-1, D)
        flat_ret = mc1_returns.reshape(-1) * reward_scale
        N = flat_obs.shape[0]
        mb = min(args.minibatch_size, N)

        # Train/val split
        perm = torch.randperm(N, device=device)
        val_size = max(mb, int(N * 0.1))
        val_idx, train_idx = perm[:val_size], perm[val_size:]
        N_train = train_idx.shape[0]

        best_val = float("inf")
        best_state = None
        no_improve = 0
        max_epochs = max(args.td_epochs, 200)

        history = []  # (epoch, r, rho)

        for epoch in range(max_epochs):
            critic.train()
            tp = train_idx[torch.randperm(N_train, device=device)]
            for start in range(0, N_train, mb):
                idx = tp[start:start+mb]
                loss = 0.5 * ((critic(flat_obs[idx]).view(-1) - flat_ret[idx]) ** 2).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5); opt.step()

            critic.eval()
            with torch.no_grad():
                vl = 0.5 * ((critic(flat_obs[val_idx]).view(-1) - flat_ret[val_idx]) ** 2).mean().item()
            if vl < best_val:
                best_val = vl
                best_state = {k: v.clone() for k, v in critic.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"    {label} early stop @ epoch {epoch+1}")
                    break

            if (epoch + 1) % args.eval_freq == 0:
                r_val, rho_val = eval_convergence(critic, reward_scale)
                history.append((epoch + 1, r_val, rho_val))
                print(f"    {label} epoch {epoch+1}: r={r_val:.3f}, rho={rho_val:.3f}, val_loss={vl:.6f}")

        if best_state:
            critic.load_state_dict(best_state)
        # Undo reward scaling in last layer
        if reward_scale != 1.0:
            with torch.no_grad():
                for m in critic.modules():
                    if isinstance(m, nn.Linear):
                        last_linear = m
                last_linear.weight.div_(reward_scale)
                last_linear.bias.div_(reward_scale)
        return critic, history

    def build_td_tuples(reward_scale=1.0):
        """Build (s, r, s', done) tuples per rollout with correct done alignment, then concat."""
        all_s, all_r, all_ns, all_d = [], [], [], []
        for obs_r, rew_r, don_r, nobs_r in zip(all_obs, all_rewards, all_dones, all_next_obs):
            T_r, E_r = rew_r.shape
            s = obs_r.reshape(-1, args.cond_steps, obs_dim).reshape(-1, D)
            r = rew_r.reshape(-1) * reward_scale
            # Next states: obs[t+1] for t < T-1, nobs for t = T-1
            ns = torch.zeros(T_r, E_r, args.cond_steps, obs_dim, device=device)
            ns[:-1] = obs_r[1:]
            ns[-1] = nobs_r
            ns = ns.reshape(-1, D)
            # Done for transition t = dones[t] (NOT dones[t+1])
            d = don_r.reshape(-1)
            all_s.append(s); all_r.append(r); all_ns.append(ns); all_d.append(d)
        return torch.cat(all_s), torch.cat(all_r), torch.cat(all_ns), torch.cat(all_d)

    # --- Method 2: TD(0) ---
    def train_td(reward_scale=1.0, label="TD"):
        critic = make_critic()
        opt = optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)

        flat_s, flat_r, flat_ns, flat_d = build_td_tuples(reward_scale)

        N = flat_s.shape[0]
        mb = min(args.minibatch_size, N)
        perm = torch.randperm(N, device=device)
        val_size = max(mb, int(N * 0.1))
        val_idx, train_idx = perm[:val_size], perm[val_size:]
        N_train = train_idx.shape[0]

        best_val = float("inf")
        best_state = None
        no_improve = 0
        max_epochs = max(args.td_epochs, 200)

        history = []

        for epoch in range(max_epochs):
            critic.train()
            tp = train_idx[torch.randperm(N_train, device=device)]
            for start in range(0, N_train, mb):
                idx = tp[start:start+mb]
                with torch.no_grad():
                    target = flat_r[idx] + args.gamma * critic(flat_ns[idx]).view(-1) * (1 - flat_d[idx])
                loss = 0.5 * ((critic(flat_s[idx]).view(-1) - target) ** 2).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5); opt.step()

            critic.eval()
            with torch.no_grad():
                vt = flat_r[val_idx] + args.gamma * critic(flat_ns[val_idx]).view(-1) * (1 - flat_d[val_idx])
                vl = 0.5 * ((critic(flat_s[val_idx]).view(-1) - vt) ** 2).mean().item()
            if vl < best_val:
                best_val = vl
                best_state = {k: v.clone() for k, v in critic.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"    {label} early stop @ epoch {epoch+1}")
                    break

            if (epoch + 1) % args.eval_freq == 0:
                r_val, rho_val = eval_convergence(critic, reward_scale)
                history.append((epoch + 1, r_val, rho_val))
                print(f"    {label} epoch {epoch+1}: r={r_val:.3f}, rho={rho_val:.3f}, val_loss={vl:.6f}")

        if best_state:
            critic.load_state_dict(best_state)
        if reward_scale != 1.0:
            with torch.no_grad():
                for m in critic.modules():
                    if isinstance(m, nn.Linear):
                        last_linear = m
                last_linear.weight.div_(reward_scale)
                last_linear.bias.div_(reward_scale)
        return critic, history

    # --- Method 3: TD+EMA ---
    def train_td_ema(reward_scale=1.0, label="TD+EMA"):
        critic = make_critic()
        critic_target = make_critic()
        critic_target.load_state_dict(critic.state_dict())
        opt = optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)

        flat_s, flat_r, flat_ns, flat_d = build_td_tuples(reward_scale)

        N = flat_s.shape[0]
        mb = min(args.minibatch_size, N)
        perm = torch.randperm(N, device=device)
        val_size = max(mb, int(N * 0.1))
        val_idx, train_idx = perm[:val_size], perm[val_size:]
        N_train = train_idx.shape[0]

        best_val = float("inf")
        best_state = None
        no_improve = 0
        max_epochs = max(args.td_epochs, 200)

        history = []

        for epoch in range(max_epochs):
            critic.train()
            tp = train_idx[torch.randperm(N_train, device=device)]
            for start in range(0, N_train, mb):
                idx = tp[start:start+mb]
                with torch.no_grad():
                    target = flat_r[idx] + args.gamma * critic_target(flat_ns[idx]).view(-1) * (1 - flat_d[idx])
                loss = 0.5 * ((critic(flat_s[idx]).view(-1) - target) ** 2).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5); opt.step()
                with torch.no_grad():
                    for p, pt in zip(critic.parameters(), critic_target.parameters()):
                        pt.data.mul_(1 - args.ema_tau).add_(p.data, alpha=args.ema_tau)

            critic.eval()
            with torch.no_grad():
                vt = flat_r[val_idx] + args.gamma * critic_target(flat_ns[val_idx]).view(-1) * (1 - flat_d[val_idx])
                vl = 0.5 * ((critic(flat_s[val_idx]).view(-1) - vt) ** 2).mean().item()
            if vl < best_val:
                best_val = vl
                best_state = {k: v.clone() for k, v in critic.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"    {label} early stop @ epoch {epoch+1}")
                    break

            if (epoch + 1) % args.eval_freq == 0:
                r_val, rho_val = eval_convergence(critic, reward_scale)
                history.append((epoch + 1, r_val, rho_val))
                print(f"    {label} epoch {epoch+1}: r={r_val:.3f}, rho={rho_val:.3f}, val_loss={vl:.6f}")

        if best_state:
            critic.load_state_dict(best_state)
        if reward_scale != 1.0:
            with torch.no_grad():
                for m in critic.modules():
                    if isinstance(m, nn.Linear):
                        last_linear = m
                last_linear.weight.div_(reward_scale)
                last_linear.bias.div_(reward_scale)
        return critic, history

    # --- Method 4: GAE (iterative) ---
    def train_gae(label="GAE"):
        critic = make_critic()
        opt = optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)
        flat_obs = full_obs.reshape(-1, args.cond_steps, obs_dim).reshape(-1, D)
        N = flat_obs.shape[0]
        mb = min(args.minibatch_size, N)

        history = []
        cumulative_epoch = 0

        for gae_iter in range(args.gae_iters):
            # Compute GAE per rollout (no cross-rollout leakage)
            adv_list = []
            val_list = []
            with torch.no_grad():
                for obs_r, rew_r, don_r, nobs_r in zip(all_obs, all_rewards, all_dones, all_next_obs):
                    T_r, E_r = rew_r.shape
                    obs_flat_2d = obs_r.reshape(T_r * E_r, args.cond_steps, obs_dim).reshape(T_r * E_r, D)
                    values = critic(obs_flat_2d).view(T_r, E_r)
                    nobs_flat = nobs_r.reshape(E_r, -1)
                    nv = critic(nobs_flat).view(E_r)

                    adv = torch.zeros_like(rew_r)
                    lastgaelam = 0
                    for t in reversed(range(T_r)):
                        if t == T_r - 1:
                            nvs = nv
                        else:
                            nvs = values[t + 1]
                        nnd = 1.0 - don_r[t]
                        delta = rew_r[t] + args.gamma * nnd * nvs - values[t]
                        adv[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nnd * lastgaelam
                    adv_list.append(adv)
                    val_list.append(values)

            all_adv = torch.cat(adv_list, dim=0)
            all_val = torch.cat(val_list, dim=0)
            flat_ret = (all_adv + all_val).reshape(-1)

            # Fit critic to GAE returns
            critic.train()
            for ep in range(args.gae_epochs):
                perm = torch.randperm(N, device=device)
                for start in range(0, N, mb):
                    idx = perm[start:start+mb]
                    loss = 0.5 * ((critic(flat_obs[idx]).view(-1) - flat_ret[idx]) ** 2).mean()
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), 0.5); opt.step()
                cumulative_epoch += 1

                if cumulative_epoch % args.eval_freq == 0:
                    r_val, rho_val = eval_convergence(critic, 1.0)
                    history.append((cumulative_epoch, r_val, rho_val))
                    print(f"    {label} gae_iter={gae_iter+1} epoch {ep+1}/{args.gae_epochs} "
                          f"(total={cumulative_epoch}): r={r_val:.3f}, rho={rho_val:.3f}")

            # Log per-GAE-iter summary
            r_val, rho_val, ratio, v_mean, v_std = eval_critic(
                critic, full_obs, full_mc1)
            print(f"    {label} iter {gae_iter+1}/{args.gae_iters}: "
                  f"r={r_val:.3f}, rho={rho_val:.3f}, V_std/MC1_std={ratio:.3f}")

        return critic, history

    # ===== Run all methods =====
    results = {}
    all_histories = {}  # label -> [(epoch, r, rho), ...]

    obs = full_obs
    mc1 = full_mc1

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\nTraining with {N_total} samples ({args.n_rollouts} rollouts)")

    for rs in args.reward_scales:
        rs_label = f"rs{int(rs)}" if rs == int(rs) else f"rs{rs}"

        # MC1 regression
        label = f"MC1_{rs_label}"
        print(f"\n--- {label} ---")
        t0 = time.time()
        c, hist = train_mc1(obs, mc1, reward_scale=rs, label=label)
        r, rho, ratio, vm, vs = eval_critic(c, obs, mc1)
        results[label] = (r, rho, ratio, vm, vs)
        all_histories[label] = hist
        print(f"  {label}: r={r:.3f}, rho={rho:.3f}, V_std/MC1_std={ratio:.3f}, "
              f"V(mean={vm:.3f}, std={vs:.3f}), time={time.time()-t0:.1f}s")

        # TD(0) — uses build_td_tuples internally (per-rollout, correct done alignment)
        label = f"TD_{rs_label}"
        print(f"\n--- {label} ---")
        t0 = time.time()
        c, hist = train_td(reward_scale=rs, label=label)
        r, rho, ratio, vm, vs = eval_critic(c, obs, mc1)
        results[label] = (r, rho, ratio, vm, vs)
        all_histories[label] = hist
        print(f"  {label}: r={r:.3f}, rho={rho:.3f}, V_std/MC1_std={ratio:.3f}, "
              f"V(mean={vm:.3f}, std={vs:.3f}), time={time.time()-t0:.1f}s")

        # TD+EMA
        label = f"TD+EMA_{rs_label}"
        print(f"\n--- {label} ---")
        t0 = time.time()
        c, hist = train_td_ema(reward_scale=rs, label=label)
        r, rho, ratio, vm, vs = eval_critic(c, obs, mc1)
        results[label] = (r, rho, ratio, vm, vs)
        all_histories[label] = hist
        print(f"  {label}: r={r:.3f}, rho={rho:.3f}, V_std/MC1_std={ratio:.3f}, "
              f"V(mean={vm:.3f}, std={vs:.3f}), time={time.time()-t0:.1f}s")

    # GAE (no reward_scale needed — iterative, per-rollout internally)
    label = "GAE"
    print(f"\n--- {label} ---")
    t0 = time.time()
    c, hist = train_gae(label=label)
    r, rho, ratio, vm, vs = eval_critic(c, obs, mc1)
    results[label] = (r, rho, ratio, vm, vs)
    all_histories[label] = hist
    print(f"  {label}: r={r:.3f}, rho={rho:.3f}, V_std/MC1_std={ratio:.3f}, "
          f"V(mean={vm:.3f}, std={vs:.3f}), time={time.time()-t0:.1f}s")

    # ===== Summary =====
    print(f"\n{'='*70}")
    print(f"SUMMARY: V quality on PegInsertion (gamma={args.gamma}, "
          f"N={N_total}, {args.n_rollouts} rollouts)")
    print(f"{'='*70}")
    print(f"{'Method':<20s} {'r(V,MC1)':>10s} {'rho(V,MC1)':>12s} "
          f"{'V_std/MC1':>10s} {'V_mean':>8s} {'V_std':>8s}")
    print("-" * 70)
    for label, (r, rho, ratio, vm, vs) in sorted(results.items()):
        print(f"{label:<20s} {r:>10.3f} {rho:>12.3f} {ratio:>10.3f} "
              f"{vm:>8.3f} {vs:>8.3f}")

    # ===== Plot convergence curves =====
    print(f"\nSaving convergence plots to {args.save_dir}/")

    # Group by method type for cleaner plots
    method_groups = {
        "MC1": [k for k in all_histories if k.startswith("MC1_")],
        "TD": [k for k in all_histories if k.startswith("TD_") and "EMA" not in k],
        "TD+EMA": [k for k in all_histories if k.startswith("TD+EMA_")],
        "GAE": [k for k in all_histories if k == "GAE"],
    }
    colors = {"rs1": "C0", "rs10": "C1", "rs100": "C2"}

    # Plot 1: r(V,MC1) convergence — all methods
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, hist in all_histories.items():
        if not hist:
            continue
        epochs = [h[0] for h in hist]
        rs = [h[1] for h in hist]
        rhos = [h[2] for h in hist]
        axes[0].plot(epochs, rs, label=label, alpha=0.8)
        axes[1].plot(epochs, rhos, label=label, alpha=0.8)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Pearson r(V, MC1)")
    axes[0].set_title("V quality convergence (Pearson r)")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.2, 1.0)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Spearman rho(V, MC1)")
    axes[1].set_title("V quality convergence (Spearman rho)")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.2, 1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "convergence_all.png"), dpi=150)
    plt.close()

    # Plot 2: Per-method-type subplots (cleaner comparison across reward scales)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (mtype, labels) in zip(axes.flat, method_groups.items()):
        for label in labels:
            hist = all_histories.get(label, [])
            if not hist:
                continue
            epochs = [h[0] for h in hist]
            rs = [h[1] for h in hist]
            ax.plot(epochs, rs, label=label, alpha=0.8, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Pearson r(V, MC1)")
        ax.set_title(f"{mtype}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.2, 1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "convergence_by_method.png"), dpi=150)
    plt.close()

    # Save raw data
    np.savez(os.path.join(args.save_dir, "convergence_data.npz"),
             **{label: np.array(hist) for label, hist in all_histories.items() if hist})
    print("Done.")

    train_envs.close()


if __name__ == "__main__":
    main()
