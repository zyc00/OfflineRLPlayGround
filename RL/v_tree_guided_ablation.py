"""Guided Tree Sampling Ablation — TD Error vs P(success) Predictor vs Uniform.

Compares branch point selection strategies (all same transition budget):
1. uniform: random branches (baseline)
2. td_weighted: branches weighted by |TD error| from V₀(gamma)
3. p_uncertain: branches from states where P₀(s) ≈ 0.5 (max uncertainty)
4. p_bernoulli: branches weighted by P₀(s)*(1-P₀(s)) (Bernoulli variance)

V₀ = TD+EMA with target gamma, trained on seed data
P₀ = TD+EMA with gamma=1.0, trained on seed data → P₀(s) ≈ P(success|s)

Both predictors use ONLY existing data (no MC16 oracle).

Usage:
  python -u -m RL.v_tree_guided_ablation --gamma 0.99
"""

import copy
import os
import random
import sys
import time
from dataclasses import dataclass

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from scipy.stats import pearsonr
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from data.data_collection.ppo import Agent, layer_init


@dataclass
class Args:
    checkpoint: str = "runs/pickcube_ppo/ckpt_76_logstd-1.5.pt"
    gamma: float = 0.99
    num_envs: int = 100
    num_steps: int = 50
    max_episode_steps: int = 50
    mc_samples: int = 16
    # Ablation grid — N=1 means "0.5 seed + 0.5 branch" (same total as 1 rollout)
    N_values: tuple[int, ...] = (1, 2, 3, 5)
    strategies: tuple[str, ...] = ("uniform", "td_weighted", "td_topk", "p_bernoulli", "p_topk",
                                    "ens_weighted", "ens_topk",
                                    "ens_td_mean_topk", "ens_td_var_topk")
    topk_frac: float = 0.2
    ensemble_K: int = 5
    # Training
    td_reward_scale: float = 10.0
    ema_tau: float = 0.005
    epochs: int = 2000
    v0_epochs: int = 500
    p0_epochs: int = 2000  # P₀ needs more epochs (gamma=1, long bootstrap chain)
    lr: float = 3e-4
    batch_size: int = 1000
    eval_every: int = 5
    critic_layers: int = 3
    hidden_dim: int = 256
    seed: int = 1
    env_id: str = "PickCube-v1"
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"
    output: str = "runs/v_tree_guided_ablation.png"


if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda")

    max_N = max(args.N_values)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_kwargs = dict(
        obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda",
        reward_mode=args.reward_mode, control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
    )

    T, E = args.num_steps, args.num_envs

    # ── Setup ──
    envs = gym.make(args.env_id, num_envs=E, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(envs, E, ignore_terminations=False, record_metrics=True)
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()

    def clip_action(a):
        return torch.clamp(a.detach(), action_low, action_high)

    rs = args.td_reward_scale

    def make_v_net():
        layers = [layer_init(nn.Linear(obs_dim, args.hidden_dim)), nn.Tanh()]
        for _ in range(args.critic_layers - 1):
            layers += [layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)), nn.Tanh()]
        layers.append(layer_init(nn.Linear(args.hidden_dim, 1)))
        return nn.Sequential(*layers).to(device)

    def _clone_state_cpu(sd):
        if isinstance(sd, dict):
            return {k: _clone_state_cpu(v) for k, v in sd.items()}
        if isinstance(sd, torch.Tensor):
            return sd.cpu().clone()
        return sd

    def _state_to_device(sd, dev):
        if isinstance(sd, dict):
            return {k: _state_to_device(v, dev) for k, v in sd.items()}
        if isinstance(sd, torch.Tensor):
            return sd.to(dev)
        return sd

    zero_action = torch.zeros(E, *envs.single_action_space.shape, device=device)

    def _restore_state(sd, elapsed_steps=None, seed=None):
        envs.reset(seed=seed if seed is not None else args.seed)
        sd_gpu = _state_to_device(sd, device)
        envs.base_env.set_state_dict(sd_gpu)
        envs.base_env.step(zero_action)
        envs.base_env.set_state_dict(sd_gpu)
        if elapsed_steps is not None:
            envs.base_env._elapsed_steps[:] = elapsed_steps
        else:
            envs.base_env._elapsed_steps[:] = 0
        return envs.base_env.get_obs()

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1: Collect rollouts, save states for rollout 0
    # ══════════════════════════════════════════════════════════════════════
    print(f"Phase 1: Collecting {max_N} rollouts (saving states for rollout 0)...")
    sys.stdout.flush()
    t0 = time.time()

    seed_states = [None] * T
    data_pool = []

    for ri in range(max_N):
        roll_obs = torch.zeros(T, E, obs_dim, device=device)
        roll_rewards = torch.zeros(T, E, device=device)
        roll_dones = torch.zeros(T, E, device=device)

        next_obs, _ = envs.reset(seed=args.seed + ri)
        next_done = torch.zeros(E, device=device)

        with torch.no_grad():
            for step in range(T):
                if ri == 0:
                    seed_states[step] = _clone_state_cpu(envs.base_env.get_state_dict())
                roll_obs[step] = next_obs
                roll_dones[step] = next_done
                action = agent.get_action(next_obs, deterministic=False)
                next_obs, reward, term, trunc, _ = envs.step(clip_action(action))
                next_done = (term | trunc).float()
                roll_rewards[step] = reward.view(-1)

        data_pool.append(dict(
            obs=roll_obs, rewards=roll_rewards, dones=roll_dones,
            next_obs=next_obs.clone(), next_done=next_done.clone(),
        ))

    print(f"  Phase 1 done ({time.time() - t0:.1f}s)")

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2: MC16 ground truth (for evaluation only)
    # ══════════════════════════════════════════════════════════════════════
    num_mc_envs = E * args.mc_samples
    print(f"\nPhase 2: MC16 ({num_mc_envs} mc_envs)...")
    sys.stdout.flush()
    t0 = time.time()

    mc_envs_raw = gym.make(args.env_id, num_envs=num_mc_envs, **env_kwargs)
    if isinstance(mc_envs_raw.action_space, gym.spaces.Dict):
        mc_envs_raw = FlattenActionSpaceWrapper(mc_envs_raw)
    mc_envs = ManiSkillVectorEnv(mc_envs_raw, num_mc_envs, ignore_terminations=False, record_metrics=False)
    mc_zero_action = torch.zeros(num_mc_envs, *envs.single_action_space.shape, device=device)

    def _expand_state(sd, repeats):
        if isinstance(sd, dict):
            return {k: _expand_state(v, repeats) for k, v in sd.items()}
        if isinstance(sd, torch.Tensor) and sd.dim() > 0:
            return sd.repeat_interleave(repeats, dim=0)
        return sd

    def _restore_mc_state(sd, seed=None):
        mc_envs.reset(seed=seed if seed is not None else args.seed)
        mc_envs.base_env.set_state_dict(sd)
        mc_envs.base_env.step(mc_zero_action)
        mc_envs.base_env.set_state_dict(sd)
        mc_envs.base_env._elapsed_steps[:] = 0
        return mc_envs.base_env.get_obs()

    on_mc16 = torch.zeros(T, E, device=device)
    with torch.no_grad():
        for t in range(T):
            expanded = _expand_state(_state_to_device(seed_states[t], device), args.mc_samples)
            mc_obs = _restore_mc_state(expanded, seed=args.seed + 1000 + t)
            env_done = torch.zeros(num_mc_envs, device=device).bool()
            all_rews = []
            for _ in range(args.max_episode_steps):
                if env_done.all():
                    break
                a = agent.get_action(mc_obs, deterministic=False)
                mc_obs, rew, term, trunc, _ = mc_envs.step(clip_action(a))
                all_rews.append(rew.view(-1) * (~env_done).float())
                env_done = env_done | (term | trunc).view(-1).bool()
            ret = torch.zeros(num_mc_envs, device=device)
            for s in reversed(range(len(all_rews))):
                ret = all_rews[s] + args.gamma * ret
            on_mc16[t] = ret.view(E, args.mc_samples).mean(dim=1)
            if (t + 1) % 10 == 0:
                print(f"  MC16 step {t + 1}/{T}")
                sys.stdout.flush()

    mc_envs.close()
    del mc_envs, mc_envs_raw
    torch.cuda.empty_cache()

    eval_obs = data_pool[0]['obs']
    mc16_flat = on_mc16.reshape(-1).cpu().numpy()
    print(f"  MC16: mean={on_mc16.mean():.4f}, std={on_mc16.std():.4f}")
    print(f"  Phase 2 done ({time.time() - t0:.1f}s)")

    # ══════════════════════════════════════════════════════════════════════
    # Phase 3: Build per-rollout flat TD data
    # ══════════════════════════════════════════════════════════════════════
    print(f"\nPhase 3: Building per-rollout flat TD data...")
    t0 = time.time()

    per_rollout_flat = []
    for ri in range(max_N):
        d = data_pool[ri]
        obs_ri = d['obs']
        rew_ri = d['rewards']
        done_ri = d['dones']
        ns_ri = torch.zeros_like(obs_ri)
        ns_ri[:-1] = obs_ri[1:]
        ns_ri[-1] = d['next_obs']
        nd_ri = torch.zeros_like(rew_ri)
        nd_ri[:-1] = done_ri[1:]
        nd_ri[-1] = d['next_done']
        per_rollout_flat.append((
            obs_ri.reshape(-1, obs_dim),
            rew_ri.reshape(-1),
            ns_ri.reshape(-1, obs_dim),
            nd_ri.reshape(-1),
        ))

    del data_pool
    torch.cuda.empty_cache()

    def build_rollout_td(n_rollouts):
        return tuple(
            torch.cat([per_rollout_flat[ri][k] for ri in range(n_rollouts)])
            for k in range(4)
        )

    print(f"  {max_N} rollouts, {max_N * E * T:,} total transitions")
    print(f"  Phase 3 done ({time.time() - t0:.1f}s)")

    # ══════════════════════════════════════════════════════════════════════
    # Phase 4: Train V₀ (gamma) and P₀ (gamma=1) on seed data
    # ══════════════════════════════════════════════════════════════════════
    def compute_r(critic, scale=1.0):
        with torch.no_grad():
            v = critic(eval_obs.reshape(-1, obs_dim)).view(-1).cpu().numpy() / scale
        return pearsonr(v, mc16_flat)[0]

    def train_td_ema_custom(flat_s, flat_r, flat_ns, flat_nd, gamma, reward_scale,
                            n_epochs=None, label=""):
        """Train TD+EMA with custom gamma and reward_scale."""
        if n_epochs is None:
            n_epochs = args.epochs
        N = flat_s.shape[0]
        mb = min(args.batch_size, N)
        scaled_r = flat_r * reward_scale

        critic = make_v_net()
        critic_target = copy.deepcopy(critic)
        opt = optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)

        best_r, best_ep = -999, 0

        for epoch in range(n_epochs):
            critic.train()
            perm = torch.randperm(N, device=device)
            for start in range(0, N, mb):
                idx = perm[start:start + mb]
                with torch.no_grad():
                    target = scaled_r[idx] + gamma * critic_target(flat_ns[idx]).view(-1) * (1 - flat_nd[idx])
                loss = 0.5 * ((critic(flat_s[idx]).view(-1) - target) ** 2).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5); opt.step()
                with torch.no_grad():
                    for p, pt in zip(critic.parameters(), critic_target.parameters()):
                        pt.data.mul_(1 - args.ema_tau).add_(p.data, alpha=args.ema_tau)

            if (epoch + 1) % args.eval_every == 0 or epoch == 0:
                critic.eval()
                r = compute_r(critic, reward_scale)
                if r > best_r:
                    best_r = r
                    best_ep = epoch + 1

        return best_r, best_ep, critic

    def train_td_ema(flat_s, flat_r, flat_ns, flat_nd, n_epochs=None, label=""):
        """Train TD+EMA with full logging, return (epochs_log, r_log, best_r, best_ep, critic)."""
        if n_epochs is None:
            n_epochs = args.epochs
        N = flat_s.shape[0]
        mb = min(args.batch_size, N)
        scaled_r = flat_r * rs

        critic = make_v_net()
        critic_target = copy.deepcopy(critic)
        opt = optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)

        epochs_log, r_log = [], []
        best_r, best_ep = -999, 0

        for epoch in range(n_epochs):
            critic.train()
            perm = torch.randperm(N, device=device)
            for start in range(0, N, mb):
                idx = perm[start:start + mb]
                with torch.no_grad():
                    target = scaled_r[idx] + args.gamma * critic_target(flat_ns[idx]).view(-1) * (1 - flat_nd[idx])
                loss = 0.5 * ((critic(flat_s[idx]).view(-1) - target) ** 2).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5); opt.step()
                with torch.no_grad():
                    for p, pt in zip(critic.parameters(), critic_target.parameters()):
                        pt.data.mul_(1 - args.ema_tau).add_(p.data, alpha=args.ema_tau)

            if (epoch + 1) % args.eval_every == 0 or epoch == 0:
                critic.eval()
                r = compute_r(critic, rs)
                epochs_log.append(epoch + 1)
                r_log.append(r)
                if r > best_r:
                    best_r = r
                    best_ep = epoch + 1

        return epochs_log, r_log, best_r, best_ep, critic

    seed_s, seed_r, seed_ns, seed_nd = per_rollout_flat[0]

    # 4a: V₀ with target gamma (for TD error)
    print(f"\nPhase 4a: Training V₀ (gamma={args.gamma}, rs={rs})...")
    sys.stdout.flush()
    t0 = time.time()
    v0_r, v0_ep, v0_critic = train_td_ema_custom(
        seed_s, seed_r, seed_ns, seed_nd,
        gamma=args.gamma, reward_scale=rs, n_epochs=args.v0_epochs, label="V₀"
    )
    print(f"  V₀ peak r={v0_r:.4f} @ ep{v0_ep} ({time.time() - t0:.1f}s)")

    # Compute per-state TD error
    v0_critic.eval()
    with torch.no_grad():
        v_s = v0_critic(seed_s).view(-1)
        v_ns = v0_critic(seed_ns).view(-1)
        td_target = seed_r * rs + args.gamma * v_ns * (1 - seed_nd)
        td_error = (v_s - td_target).abs()
    td_error_2d = td_error.view(T, E).cpu().numpy()
    print(f"  TD error: mean={td_error.mean():.4f}, std={td_error.std():.4f}")

    del v0_critic
    torch.cuda.empty_cache()

    # 4b: P₀ with gamma=1.0 (for P(success) prediction)
    # Use reward_scale=10 to help NN learning (same as V₀), will interpret P = V/rs
    p0_rs = 10.0
    print(f"\nPhase 4b: Training P₀ (gamma=1.0, rs={p0_rs}, epochs={args.p0_epochs})...")
    sys.stdout.flush()
    t0 = time.time()
    p0_r, p0_ep, p0_critic = train_td_ema_custom(
        seed_s, seed_r, seed_ns, seed_nd,
        gamma=1.0, reward_scale=p0_rs, n_epochs=args.p0_epochs, label="P₀"
    )
    print(f"  P₀ peak r={p0_r:.4f} @ ep{p0_ep} ({time.time() - t0:.1f}s)")

    # Compute P(success|s) predictions
    p0_critic.eval()
    with torch.no_grad():
        p_pred = p0_critic(seed_s).view(-1) / p0_rs  # scale back to [0, 1]
        p_pred = p_pred.clamp(0, 1)  # ensure valid probability range
    p_pred_2d = p_pred.view(T, E).cpu().numpy()
    print(f"  P₀ predictions: mean={p_pred.mean():.4f}, std={p_pred.std():.4f}, "
          f"min={p_pred.min():.4f}, max={p_pred.max():.4f}")

    # Show P₀ by timestep
    p_by_t = p_pred_2d.mean(axis=1)
    print(f"\n  P₀(s) by timestep (first/mid/last 3):")
    for t in [0, 1, 2, T//2-1, T//2, T//2+1, T-3, T-2, T-1]:
        if t < T:
            print(f"    t={t}: mean P={p_by_t[t]:.4f}")

    del p0_critic
    torch.cuda.empty_cache()

    # 4c: Ensemble of K independent V networks (for epistemic uncertainty)
    ens_strategies = [s for s in args.strategies if "ens" in s]
    if ens_strategies:
        K = args.ensemble_K
        print(f"\nPhase 4c: Training ensemble of {K} V networks (gamma={args.gamma}, rs={rs}, epochs={args.v0_epochs})...")
        sys.stdout.flush()
        t0 = time.time()

        ensemble_preds = []  # (K, T*E)
        ensemble_td_errors = []  # (K, T*E)
        for ki in range(K):
            torch.manual_seed(args.seed + 7000 + ki)  # different init per member
            _, _, ens_critic = train_td_ema_custom(
                seed_s, seed_r, seed_ns, seed_nd,
                gamma=args.gamma, reward_scale=rs, n_epochs=args.v0_epochs,
                label=f"Ens[{ki}]"
            )
            ens_critic.eval()
            with torch.no_grad():
                pred = ens_critic(seed_s).view(-1)
                pred_ns = ens_critic(seed_ns).view(-1)
                td_target_k = seed_r * rs + args.gamma * pred_ns * (1 - seed_nd)
                td_err_k = (pred - td_target_k).abs().cpu().numpy()
                ensemble_preds.append(pred.cpu().numpy())
                ensemble_td_errors.append(td_err_k)
            del ens_critic
            torch.cuda.empty_cache()

        # Restore RNG state after ensemble training
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        ensemble_preds = np.stack(ensemble_preds)  # (K, T*E)
        ensemble_td_errors = np.stack(ensemble_td_errors)  # (K, T*E)
        ens_var = ensemble_preds.var(axis=0)  # (T*E,)
        ens_var_2d = ens_var.reshape(T, E)
        ens_td_mean = ensemble_td_errors.mean(axis=0)  # (T*E,) — mean TD error across K
        ens_td_mean_2d = ens_td_mean.reshape(T, E)
        ens_td_var = ensemble_td_errors.var(axis=0)  # (T*E,) — Bellman residual variance
        ens_mean_r = np.mean([pearsonr(ensemble_preds[ki], mc16_flat)[0] for ki in range(K)])
        print(f"  Ensemble: mean_member_r={ens_mean_r:.4f}")
        print(f"  Var[V_k]: mean={ens_var.mean():.6f}, std={ens_var.std():.6f}, max={ens_var.max():.6f}")
        print(f"  Mean|TD_err|: mean={ens_td_mean.mean():.4f}, std={ens_td_mean.std():.4f}")
        print(f"  Var[TD_err]: mean={ens_td_var.mean():.6f}, std={ens_td_var.std():.6f}")

        # Correlation between single-V₀ TD error and ensemble-mean TD error
        corr_single_vs_mean = np.corrcoef(td_error_2d.flatten(), ens_td_mean)[0, 1]
        print(f"  Corr(single_TD, mean_TD): {corr_single_vs_mean:.4f}")

        # Concentration analysis
        ens_var_by_t = ens_var_2d.mean(axis=1)
        top20_ens = np.percentile(ens_var, 80)
        ens_top_mask = ens_var_2d >= top20_ens
        q_counts = [ens_top_mask[i*T//4:(i+1)*T//4].sum() for i in range(4)]
        total_top = sum(q_counts)
        print(f"  Ens_var top-20% by quartile: {[f'{c/total_top*100:.1f}%' for c in q_counts]}")

        top20_etd = np.percentile(ens_td_mean, 80)
        etd_top_mask = ens_td_mean_2d >= top20_etd
        q_counts_etd = [etd_top_mask[i*T//4:(i+1)*T//4].sum() for i in range(4)]
        total_top_etd = sum(q_counts_etd)
        print(f"  Mean_TD top-20% by quartile: {[f'{c/total_top_etd*100:.1f}%' for c in q_counts_etd]}")

        corr_td_ens = np.corrcoef(td_error_2d.flatten(), ens_var)[0, 1]
        print(f"  Corr(TD_error, Ens_var): {corr_td_ens:.4f}")
        print(f"  Phase 4c done ({time.time() - t0:.1f}s)")
    else:
        ens_var = None
        ens_var_2d = None
        corr_td_ens = None
        ens_mean_r = None

    # Compute P-based sampling weights
    # p_uncertain: |P - 0.5| → smallest = most uncertain → weight = 1/(|P-0.5| + eps)
    p_uncertainty = np.abs(p_pred_2d - 0.5)  # (T, E), low = more uncertain
    # p_bernoulli: P*(1-P) → largest = most uncertain
    p_bernoulli = p_pred_2d * (1 - p_pred_2d)  # (T, E), high = more uncertain

    print(f"\n  Uncertainty stats:")
    print(f"    |P-0.5| mean={p_uncertainty.mean():.4f}, min={p_uncertainty.min():.4f}")
    print(f"    P*(1-P) mean={p_bernoulli.mean():.4f}, max={p_bernoulli.max():.4f}")

    # Print TD error vs P uncertainty correlation
    td_flat = td_error_2d.flatten()
    bern_flat = p_bernoulli.flatten()
    corr = np.corrcoef(td_flat, bern_flat)[0, 1]
    print(f"\n  Correlation(TD_error, Bernoulli_var): {corr:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # Phase 5: Collect branches with different strategies
    # ══════════════════════════════════════════════════════════════════════
    def _assemble_composite_state(branch_points):
        """Build composite state dict from seed_states (always seed 0)."""
        ref = seed_states[0]

        def _assemble_inner(ref_val, key_path):
            if isinstance(ref_val, dict):
                return {k: _assemble_inner(v, key_path + [k]) for k, v in ref_val.items()}
            if isinstance(ref_val, torch.Tensor) and ref_val.dim() > 0:
                slices = []
                for (t, ei) in branch_points:
                    val = seed_states[t]
                    for k in key_path:
                        val = val[k]
                    slices.append(val[ei:ei+1])
                return torch.cat(slices, dim=0).to(device)
            if isinstance(ref_val, torch.Tensor):
                return ref_val.to(device)
            return ref_val

        return _assemble_inner(ref, [])

    def _make_weights(strategy):
        """Return (T*E,) sampling weights for given strategy."""
        if strategy == "uniform":
            return None  # handled separately
        elif strategy == "td_weighted":
            w = td_error_2d.flatten().copy()
            w = w / w.sum()
            return w
        elif strategy == "td_topk":
            flat_td = td_error_2d.flatten()
            k = max(1, int(len(flat_td) * args.topk_frac))
            topk_idx = np.argsort(flat_td)[-k:]
            w = np.zeros(len(flat_td))
            w[topk_idx] = 1.0 / k
            return w
        elif strategy == "p_uncertain":
            # Weight inversely proportional to |P - 0.5|
            eps = 0.01
            w = 1.0 / (p_uncertainty.flatten() + eps)
            w = w / w.sum()
            return w
        elif strategy == "p_bernoulli":
            # Weight proportional to P*(1-P)
            w = p_bernoulli.flatten().copy()
            w = np.maximum(w, 1e-8)
            w = w / w.sum()
            return w
        elif strategy == "p_topk":
            flat_bern = p_bernoulli.flatten()
            k = max(1, int(len(flat_bern) * args.topk_frac))
            topk_idx = np.argsort(flat_bern)[-k:]
            w = np.zeros(len(flat_bern))
            w[topk_idx] = 1.0 / k
            return w
        elif strategy == "ens_weighted":
            w = ens_var.copy()
            w = w / w.sum()
            return w
        elif strategy == "ens_topk":
            k = max(1, int(len(ens_var) * args.topk_frac))
            topk_idx = np.argsort(ens_var)[-k:]
            w = np.zeros(len(ens_var))
            w[topk_idx] = 1.0 / k
            return w
        elif strategy == "ens_td_mean_topk":
            k = max(1, int(len(ens_td_mean) * args.topk_frac))
            topk_idx = np.argsort(ens_td_mean)[-k:]
            w = np.zeros(len(ens_td_mean))
            w[topk_idx] = 1.0 / k
            return w
        elif strategy == "ens_td_var_topk":
            k = max(1, int(len(ens_td_var) * args.topk_frac))
            topk_idx = np.argsort(ens_td_var)[-k:]
            w = np.zeros(len(ens_td_var))
            w[topk_idx] = 1.0 / k
            return w
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def collect_branches(strategy, budget, rng_seed_offset=0):
        """Collect branch transitions using given strategy up to budget."""
        branch_chunks = [[] for _ in range(4)]
        branch_total = 0
        rng = np.random.RandomState(args.seed + 999 + rng_seed_offset)
        n_batches = 0
        weights = _make_weights(strategy)
        branch_t_hist = np.zeros(T)

        with torch.no_grad():
            while branch_total < budget:
                if strategy == "uniform":
                    bp_timesteps = rng.randint(0, T, size=E)
                    bp_envs = rng.randint(0, E, size=E)
                else:
                    flat_idx = rng.choice(T * E, size=E, p=weights)
                    bp_timesteps = flat_idx // E
                    bp_envs = flat_idx % E

                branch_points = list(zip(bp_timesteps.tolist(), bp_envs.tolist()))
                for t in bp_timesteps:
                    branch_t_hist[t] += 1

                composite = _assemble_composite_state(branch_points)
                _restore_state(composite, seed=args.seed + 5000 + rng_seed_offset * 10000 + n_batches)
                elapsed = torch.tensor(bp_timesteps, dtype=torch.int32, device=device)
                envs.base_env._elapsed_steps[:] = elapsed

                next_obs = envs.base_env.get_obs()
                batch_obs, batch_ns, batch_rew, batch_nd = [], [], [], []
                active = torch.ones(E, device=device).bool()

                for step_i in range(T):
                    if not active.any():
                        break
                    obs_t = next_obs.clone()
                    action = agent.get_action(next_obs, deterministic=False)
                    next_obs, reward, term, trunc, _ = envs.step(clip_action(action))
                    done = (term | trunc).float()
                    batch_obs.append(obs_t)
                    batch_rew.append(reward.view(-1) * active.float())
                    batch_ns.append(next_obs.clone())
                    batch_nd.append(done)
                    newly_done = (term | trunc).view(-1).bool()
                    active = active & ~newly_done

                if len(batch_obs) == 0:
                    n_batches += 1
                    continue

                b_obs = torch.stack(batch_obs)
                b_rew = torch.stack(batch_rew)
                b_ns = torch.stack(batch_ns)
                b_nd = torch.stack(batch_nd)

                active_mask = torch.ones_like(b_rew)
                running = torch.ones(E, device=device).bool()
                for si in range(b_obs.shape[0]):
                    active_mask[si] = running.float()
                    running = running & (b_nd[si] == 0)

                flat_mask = active_mask.reshape(-1).bool()
                branch_chunks[0].append(b_obs.reshape(-1, obs_dim)[flat_mask])
                branch_chunks[1].append(b_rew.reshape(-1)[flat_mask])
                branch_chunks[2].append(b_ns.reshape(-1, obs_dim)[flat_mask])
                branch_chunks[3].append(b_nd.reshape(-1)[flat_mask])

                branch_total += flat_mask.sum().item()
                n_batches += 1

        branch_flat = tuple(torch.cat(c) for c in branch_chunks)
        print(f"    {strategy}: {branch_flat[0].shape[0]:,} trans from {n_batches} batches")
        return branch_flat, branch_t_hist

    # ══════════════════════════════════════════════════════════════════════
    # Phase 6: Run all experiments
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"Phase 6: Training experiments")
    print(f"  N_values = {args.N_values}")
    print(f"  strategies = {args.strategies}")
    print(f"  epochs = {args.epochs}")
    print(f"{'=' * 70}\n")

    all_results = {}
    rollout_results = {}
    branch_histograms = {}

    for N in args.N_values:
        target_trans = N * E * T

        # N=1 special case: split 1 rollout into 0.5 seed + 0.5 branches
        # Reduce envs (not steps!) to preserve full episode length
        if N == 1:
            half_E = E // 2  # 50 envs
            # seed_s is (T*E, D), reshape to (T, E, D) and take first half_E envs
            idx = []
            for t in range(T):
                idx.extend(range(t * E, t * E + half_E))
            idx = torch.tensor(idx, device=device)
            cur_seed_s = seed_s[idx]
            cur_seed_r = seed_r[idx]
            cur_seed_ns = seed_ns[idx]
            cur_seed_nd = seed_nd[idx]
            seed_trans = half_E * T  # 2500
            branch_budget = target_trans - seed_trans
        else:
            cur_seed_s, cur_seed_r = seed_s, seed_r
            cur_seed_ns, cur_seed_nd = seed_ns, seed_nd
            seed_trans = E * T
            branch_budget = target_trans - seed_trans

        print(f"{'─' * 60}")
        print(f"N={N} ({target_trans:,} trans, seed={seed_trans:,}, branch={branch_budget:,})")
        print(f"{'─' * 60}")

        # Rollout baseline
        print(f"  Rollout baseline...")
        r_s, r_r, r_ns, r_nd = build_rollout_td(max(N, 1))
        # For N=1, just use the 1 rollout
        r_s, r_r = r_s[:target_trans], r_r[:target_trans]
        r_ns, r_nd = r_ns[:target_trans], r_nd[:target_trans]
        t0 = time.time()
        ep_log, r_log, pk_r, pk_ep, _ = train_td_ema(r_s, r_r, r_ns, r_nd, label=f"Roll N={N}")
        print(f"    peak r={pk_r:.4f} @ ep{pk_ep} ({time.time()-t0:.1f}s)")
        rollout_results[N] = dict(epochs=ep_log, r_log=r_log, peak_r=pk_r, peak_ep=pk_ep)
        del r_s, r_r, r_ns, r_nd
        torch.cuda.empty_cache()

        # Tree strategies
        for si, strategy in enumerate(args.strategies):
            print(f"  Tree [{strategy}]...")
            sys.stdout.flush()

            t0 = time.time()
            branch_flat, branch_hist = collect_branches(
                strategy, branch_budget, rng_seed_offset=si * 100 + N
            )
            collect_time = time.time() - t0
            branch_histograms[(N, strategy)] = branch_hist

            tree_s = torch.cat([cur_seed_s, branch_flat[0][:branch_budget]])
            tree_r = torch.cat([cur_seed_r, branch_flat[1][:branch_budget]])
            tree_ns = torch.cat([cur_seed_ns, branch_flat[2][:branch_budget]])
            tree_nd = torch.cat([cur_seed_nd, branch_flat[3][:branch_budget]])

            t0 = time.time()
            ep_log, r_log, pk_r, pk_ep, _ = train_td_ema(
                tree_s, tree_r, tree_ns, tree_nd, label=f"Tree {strategy} N={N}"
            )
            train_time = time.time() - t0
            print(f"    peak r={pk_r:.4f} @ ep{pk_ep} "
                  f"(collect={collect_time:.1f}s, train={train_time:.1f}s, "
                  f"trans={tree_s.shape[0]:,})")

            all_results[(N, strategy)] = dict(
                epochs=ep_log, r_log=r_log, peak_r=pk_r, peak_ep=pk_ep,
                n_trans=tree_s.shape[0],
            )

            del tree_s, tree_r, tree_ns, tree_nd, branch_flat
            torch.cuda.empty_cache()
            sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════════════
    # Phase 7: Summary
    # ══════════════════════════════════════════════════════════════════════
    strategies = list(args.strategies)
    print(f"\n{'=' * 70}")
    print(f"SUMMARY — peak Pearson r (vs MC16)")
    print(f"V₀ r={v0_r:.4f}, P₀ r={p0_r:.4f}")
    print(f"Corr(TD_error, Bernoulli_var)={corr:.4f}")
    print(f"{'=' * 70}")

    header = f"| {'N':>5} | {'Rollout':>8} |"
    for s in strategies:
        header += f" {s:>13} |"
    print(header)
    sep = f"|-------|----------|"
    for _ in strategies:
        sep += f"---------------|"
    print(sep)

    for N in args.N_values:
        row = f"| {N:>5} | {rollout_results[N]['peak_r']:>8.4f} |"
        for s in strategies:
            if (N, s) in all_results:
                row += f" {all_results[(N, s)]['peak_r']:>13.4f} |"
            else:
                row += f" {'—':>13} |"
        print(row)

    # Delta vs rollout
    print(f"\nDelta vs Rollout:")
    for N in args.N_values:
        row = f"| {N:>5} |"
        for s in strategies:
            if (N, s) in all_results:
                delta = all_results[(N, s)]['peak_r'] - rollout_results[N]['peak_r']
                row += f" {delta:>+13.4f} |"
            else:
                row += f" {'—':>13} |"
        print(row)

    # Delta vs uniform
    if "uniform" in strategies:
        print(f"\nDelta vs Uniform Tree:")
        for N in args.N_values:
            row = f"| {N:>5} |"
            u_r = all_results.get((N, "uniform"), {}).get("peak_r", None)
            for s in strategies:
                if s == "uniform":
                    row += f" {'—':>13} |"
                elif (N, s) in all_results and u_r is not None:
                    delta = all_results[(N, s)]['peak_r'] - u_r
                    row += f" {delta:>+13.4f} |"
                else:
                    row += f" {'—':>13} |"
            print(row)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 8: Plot (2×3)
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    strat_colors = {
        'uniform': 'blue', 'td_weighted': 'red', 'td_topk': 'darkred',
        'p_uncertain': 'green', 'p_bernoulli': 'purple', 'p_topk': 'darkviolet',
        'ens_weighted': 'orange', 'ens_topk': 'darkorange',
        'ens_td_mean_topk': 'teal', 'ens_td_var_topk': 'olive',
    }

    # [0,0] Peak r vs N
    ax = axes[0, 0]
    Ns = list(args.N_values)
    ax.plot(Ns, [rollout_results[N]['peak_r'] for N in Ns], 'k--o', lw=2, label='Rollout')
    for s in strategies:
        peaks = [all_results[(N, s)]['peak_r'] for N in Ns if (N, s) in all_results]
        ns_list = [N for N in Ns if (N, s) in all_results]
        ax.plot(ns_list, peaks, 's-', color=strat_colors.get(s, 'gray'), lw=1.5, label=s)
    ax.set_xlabel("N (data budget)")
    ax.set_ylabel("Peak Pearson r")
    ax.set_title("Peak r vs Data Budget")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [0,1] Training curves for mid N
    ax = axes[0, 1]
    N_plot = args.N_values[len(args.N_values) // 2]
    if N_plot in rollout_results:
        rr = rollout_results[N_plot]
        ax.plot(rr['epochs'], rr['r_log'], 'k-', lw=2, alpha=0.7,
                label=f"Rollout (pk={rr['peak_r']:.3f})")
    for s in strategies:
        if (N_plot, s) in all_results:
            res = all_results[(N_plot, s)]
            ax.plot(res['epochs'], res['r_log'], '-', color=strat_colors.get(s, 'gray'),
                    lw=1.2, alpha=0.7, label=f"{s} (pk={res['peak_r']:.3f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pearson r")
    ax.set_title(f"Training Curves (N={N_plot})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # [0,2] Branch point histograms
    ax = axes[0, 2]
    timesteps = np.arange(T)
    for s in strategies:
        if (N_plot, s) in branch_histograms:
            hist = branch_histograms[(N_plot, s)]
            hist_norm = hist / hist.sum() if hist.sum() > 0 else hist
            ax.plot(timesteps, hist_norm, '-', color=strat_colors.get(s, 'gray'),
                    lw=1.5, alpha=0.8, label=s)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Branch freq (norm)")
    ax.set_title(f"Branch Distribution (N={N_plot})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [1,0] TD error profile
    ax = axes[1, 0]
    td_by_t = td_error_2d.mean(axis=1)
    ax.plot(timesteps, td_by_t, 'r-', lw=2, label='TD error mean')
    ax.fill_between(timesteps, td_error_2d.min(axis=1), td_error_2d.max(axis=1),
                     alpha=0.15, color='red')
    ax.set_xlabel("Timestep")
    ax.set_ylabel("|TD error|")
    ax.set_title(f"TD Error Profile (V₀ r={v0_r:.3f})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # [1,1] P₀ profile
    ax = axes[1, 1]
    ax.plot(timesteps, p_by_t, 'g-', lw=2, label='P₀(s) mean')
    ax.fill_between(timesteps, p_pred_2d.min(axis=1), p_pred_2d.max(axis=1),
                     alpha=0.15, color='green')
    ax.axhline(0.5, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("P(success|s)")
    ax.set_title(f"P₀ Predictions (r={p0_r:.3f})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # [1,2] TD error vs P scatter
    ax = axes[1, 2]
    subsample = np.random.choice(len(td_flat), size=min(2000, len(td_flat)), replace=False)
    ax.scatter(bern_flat[subsample], td_flat[subsample], alpha=0.2, s=5, c='black')
    ax.set_xlabel("P*(1-P) (Bernoulli var)")
    ax.set_ylabel("|TD error|")
    ax.set_title(f"TD Error vs P Uncertainty (corr={corr:.3f})")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Guided Branching Ablation | {args.env_id} | γ={args.gamma} | rs={rs}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {args.output}")

    save_path = args.output.replace('.png', '.pt')
    torch.save(dict(
        rollout_results=rollout_results,
        all_results={str(k): v for k, v in all_results.items()},
        branch_histograms={str(k): v for k, v in branch_histograms.items()},
        td_error_2d=td_error_2d,
        p_pred_2d=p_pred_2d,
        ens_var_2d=ens_var_2d,
        v0_r=v0_r, p0_r=p0_r, corr_td_p=corr,
        corr_td_ens=corr_td_ens, ens_mean_r=ens_mean_r,
        args=vars(args),
    ), save_path)
    print(f"Saved data: {save_path}")

    envs.close()
