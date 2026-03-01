"""Test: Does P(success) distribution width affect critic learning difficulty?

Hypothesis: When all states have P(success) in a narrow range [0.4, 0.8],
V(s) is nearly constant, making critic regression trivially easy and
producing cleaner advantages.  When P spans the full [0, 1], V varies
widely, critic regression is harder, and advantages carry more state noise.

Uses pretrained DP model + coverage data to define two seed groups:
  - Narrow: seeds with P ∈ [narrow_lo, narrow_hi]
  - Wide: same number of seeds sampled uniformly from full [0, 1]
Collects on-policy rollouts from each, trains identical critic MLPs,
then compares V quality, advantage quality, and advantage SNR.

Usage:
    python -u -m DPPO.critic_distribution_test \
      --pretrain_checkpoint runs/dppo_pretrain/dppo_pretrain_PegInsertionSide-v1_T100_H16_1M/best.pt \
      --coverage_path runs/coverage_pretrained_ddim10_500seeds.npz \
      --narrow_lo 0.4 --narrow_hi 0.8 \
      --n_rollouts 5 --n_steps 25 --n_envs 79 \
      --td_epochs 500 --save_dir runs/critic_distribution_test
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
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
    coverage_path: str = "runs/coverage_pretrained_ddim10_500seeds.npz"

    # P(success) range for "narrow" group
    narrow_lo: float = 0.4
    narrow_hi: float = 0.8

    # Environment
    env_id: str = "PegInsertionSide-v1"
    control_mode: str = "pd_joint_delta_pos"
    max_episode_steps: int = 200
    n_envs: int = 79
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
    n_rollouts: int = 5
    n_steps: int = 25

    # Sampling noise
    min_sampling_denoising_std: float = 0.01

    # V training
    gamma: float = 0.999
    gae_lambda: float = 0.95
    lr: float = 3e-4
    minibatch_size: int = 1000
    td_epochs: int = 500
    patience: int = 20
    hidden_dim: int = 256
    critic_layers: int = 3

    # Reward scale sweep
    reward_scales: tuple = (1.0, 10.0)

    eval_freq: int = 10
    save_dir: str = "runs/critic_distribution_test"
    seed: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def load_dp_model(args, device):
    """Load pretrained diffusion policy model. Returns model + normalization helpers."""
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

    act_offset = args.cond_steps - 1 if network_type == "unet" else 0

    def normalize_obs(obs):
        if no_obs_norm:
            return obs
        return (obs - obs_min) / (obs_max - obs_min + 1e-8) * 2.0 - 1.0

    def denormalize_actions(actions):
        if no_action_norm:
            return actions
        return (actions + 1.0) / 2.0 * (action_max - action_min) + action_min

    return model, obs_dim, action_dim, cond_dim, act_offset, normalize_obs, denormalize_actions


def collect_rollouts(args, model, envs, obs_dim, cond_dim, act_offset,
                     normalize_obs, denormalize_actions, device, n_envs):
    """Collect n_rollouts of data.

    Returns per-rollout lists (NOT concatenated) to avoid cross-rollout
    MC1/GAE leakage — each rollout starts with envs.reset() and must be
    treated as an independent segment for backward return computation.
    """
    assert args.sim_backend == "cpu", "This script only supports CPU envs (SeedPoolWrapper is CPU-only)"

    all_obs = []       # list of (T, E, cond_steps, obs_dim)
    all_rewards = []   # list of (T, E)
    all_dones = []     # list of (T, E)
    all_next_obs = []  # list of (E, cond_steps, obs_dim)
    all_seed_ids = []  # list of (T, E)

    total_reward_events = 0

    for rollout_idx in range(args.n_rollouts):
        t0 = time.time()
        obs_raw, info = envs.reset()
        obs_raw = torch.from_numpy(obs_raw).float().to(device)
        obs_history = obs_raw.unsqueeze(1).repeat(1, args.cond_steps, 1)

        # Track episode boundaries for per-episode SNR.
        # episode_id changes when an env terminates/truncates and resets.
        current_episode_id = torch.arange(n_envs, device=device) + rollout_idx * n_envs * 10

        obs_trajs = []
        reward_trajs = []
        done_trajs = []
        seed_id_trajs = []
        n_reward_events = 0

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
            seed_id_trajs.append(current_episode_id.clone())
            action_chunk_denorm = denormalize_actions(action_chunk)

            step_reward = torch.zeros(n_envs, device=device)
            step_done = torch.zeros(n_envs, dtype=torch.bool, device=device)

            for a_idx in range(args.act_steps):
                act_idx = act_offset + a_idx
                if act_idx < action_chunk_denorm.shape[1]:
                    action = action_chunk_denorm[:, act_idx]
                else:
                    action = action_chunk_denorm[:, -1]

                obs_new, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
                obs_new = torch.from_numpy(obs_new).float().to(device)
                reward_t = torch.from_numpy(np.array(reward)).float().to(device)
                term_t = torch.from_numpy(np.array(terminated)).bool().to(device)
                trunc_t = torch.from_numpy(np.array(truncated)).bool().to(device)

                step_reward += reward_t * (~step_done).float()
                step_done = step_done | term_t | trunc_t
                n_reward_events += (reward_t > 0.5).sum().item()

                reset_mask = term_t | trunc_t
                if reset_mask.any():
                    obs_history[reset_mask] = obs_new[reset_mask].unsqueeze(1).repeat(1, args.cond_steps, 1)
                    current_episode_id[reset_mask] += 10000
                obs_history[~reset_mask] = torch.cat(
                    [obs_history[~reset_mask, 1:], obs_new[~reset_mask].unsqueeze(1)], dim=1
                )

            reward_trajs.append(step_reward)
            done_trajs.append(step_done.float())

        obs_t = torch.stack(obs_trajs)         # (T, E, cond_steps, obs_dim)
        rew_t = torch.stack(reward_trajs)      # (T, E)
        don_t = torch.stack(done_trajs)        # (T, E)
        next_obs_t = normalize_obs(obs_history)  # (E, cond_steps, obs_dim)
        seed_ids_t = torch.stack(seed_id_trajs)  # (T, E)

        all_obs.append(obs_t)
        all_rewards.append(rew_t)
        all_dones.append(don_t)
        all_next_obs.append(next_obs_t)
        all_seed_ids.append(seed_ids_t)

        total_reward_events += n_reward_events
        elapsed = time.time() - t0
        print(f"    Rollout {rollout_idx+1}/{args.n_rollouts}: "
              f"reward_events={n_reward_events}, time={elapsed:.1f}s")

    print(f"    Total reward events: {total_reward_events} "
          f"across {args.n_rollouts} rollouts x {n_envs} envs")

    return all_obs, all_rewards, all_dones, all_next_obs, all_seed_ids


def compute_mc1(rewards, dones, gamma):
    """MC1 backward returns respecting episode boundaries."""
    T, E = rewards.shape
    mc1 = torch.zeros_like(rewards)
    running = torch.zeros(E, device=rewards.device)
    for t in reversed(range(T)):
        running = rewards[t] + gamma * (1.0 - dones[t]) * running
        mc1[t] = running
    return mc1


def compute_mc1_per_rollout(all_rewards, all_dones, gamma):
    """Compute MC1 returns per rollout (no cross-rollout leakage), then concatenate."""
    mc1_list = []
    for rew, don in zip(all_rewards, all_dones):
        mc1_list.append(compute_mc1(rew, don, gamma))
    return mc1_list


def compute_gae_per_rollout(all_obs, all_rewards, all_dones, all_next_obs,
                            critic, gamma, gae_lambda, obs_dim, cond_steps, D, device):
    """Compute GAE per rollout (no cross-rollout leakage), then concatenate."""
    adv_list = []
    val_list = []
    for obs, rew, don, nobs in zip(all_obs, all_rewards, all_dones, all_next_obs):
        T, E = rew.shape
        with torch.no_grad():
            flat_obs = obs.reshape(T * E, cond_steps, obs_dim).reshape(T * E, D)
            values = critic(flat_obs).view(T, E)
            nobs_flat = nobs.reshape(E, -1)
            nv = critic(nobs_flat).view(E)

            adv = torch.zeros_like(rew)
            lastgaelam = 0.0
            for t in reversed(range(T)):
                if t == T - 1:
                    next_values = nv
                else:
                    next_values = values[t + 1]
                nnd = 1.0 - don[t]
                delta = rew[t] + gamma * nnd * next_values - values[t]
                adv[t] = lastgaelam = delta + gamma * gae_lambda * nnd * lastgaelam
        adv_list.append(adv)
        val_list.append(values)
    return torch.cat(adv_list, dim=0), torch.cat(val_list, dim=0)


def main():
    args = tyro.cli(Args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # ===== Load coverage data and define seed groups =====
    print(f"{'='*70}")
    print(f"Loading coverage data from {args.coverage_path}")
    print(f"{'='*70}")

    cov = np.load(args.coverage_path)
    p_success = cov["p_success"]
    seeds = cov["seeds"]

    # Narrow group: seeds with P in [narrow_lo, narrow_hi]
    narrow_mask = (p_success >= args.narrow_lo) & (p_success < args.narrow_hi)
    narrow_seeds = seeds[narrow_mask]
    narrow_p = p_success[narrow_mask]
    n_narrow = len(narrow_seeds)

    if n_narrow < 2:
        print(f"ERROR: Only {n_narrow} seeds in [{args.narrow_lo}, {args.narrow_hi}). "
              f"Need at least 2. Widen the range or use different coverage data.")
        return

    print(f"Narrow group: P ∈ [{args.narrow_lo}, {args.narrow_hi})")
    print(f"  {n_narrow} seeds, mean P={narrow_p.mean():.3f}, "
          f"std P={narrow_p.std():.3f}, range=[{narrow_p.min():.3f}, {narrow_p.max():.3f}]")

    # Wide group: same size, sampled uniformly from all seeds
    rng = np.random.RandomState(args.seed + 42)
    wide_idx = rng.choice(len(seeds), size=n_narrow, replace=False)
    wide_seeds = seeds[wide_idx]
    wide_p = p_success[wide_idx]

    print(f"Wide group: {n_narrow} seeds from full [0, 1] range")
    print(f"  mean P={wide_p.mean():.3f}, std P={wide_p.std():.3f}, "
          f"range=[{wide_p.min():.3f}, {wide_p.max():.3f}]")

    # Adjust n_envs to match group size
    actual_n_envs = min(args.n_envs, n_narrow)
    print(f"\nUsing n_envs={actual_n_envs} (min of requested {args.n_envs} and group size {n_narrow})")

    # ===== Load pretrained DP model =====
    print(f"\n{'='*70}")
    print(f"Loading pretrained DP model")
    print(f"{'='*70}")

    model, obs_dim, action_dim, cond_dim, act_offset, normalize_obs, denormalize_actions = \
        load_dp_model(args, device)
    D = cond_dim  # flattened obs input dim

    print(f"  obs_dim={obs_dim}, action_dim={action_dim}, cond_dim={cond_dim}")

    # ===== Helper: make and train critic =====
    def make_critic():
        layers = []
        dims = [D] + [args.hidden_dim] * args.critic_layers + [1]
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
        if v.std() < 1e-8 or gt.std() < 1e-8:
            return 0.0, 0.0, 0.0, v.mean(), v.std()
        r_val = pearsonr(v, gt)[0]
        rho_val = spearmanr(v, gt).correlation
        ratio = v.std() / (gt.std() + 1e-8)
        return r_val, rho_val, ratio, v.mean(), v.std()

    def train_mc1(obs, mc1_returns, reward_scale=1.0, label="MC1"):
        """Train critic via MC1 regression with early stopping."""
        T, E = obs.shape[:2]
        critic = make_critic()
        opt = optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)
        flat_obs = obs.reshape(-1, args.cond_steps, obs_dim).reshape(-1, D)
        flat_ret = mc1_returns.reshape(-1) * reward_scale
        N = flat_obs.shape[0]
        mb = min(args.minibatch_size, N)

        perm = torch.randperm(N, device=device)
        val_size = max(mb, int(N * 0.1))
        val_idx, train_idx = perm[:val_size], perm[val_size:]
        N_train = train_idx.shape[0]

        best_val = float("inf")
        best_state = None
        no_improve = 0
        max_epochs = max(args.td_epochs, 200)

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
                    print(f"      {label} early stop @ epoch {epoch+1}")
                    break

            if (epoch + 1) % args.eval_freq == 0:
                # r/rho are scale-invariant, so eval on scaled critic vs unscaled MC1 is fine
                r_val, rho_val, _, _, _ = eval_critic(critic, obs, mc1_returns)
                print(f"      {label} epoch {epoch+1}: r={r_val:.3f}, rho={rho_val:.3f}, val_loss={vl:.6f}")

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
        return critic

    # ===== Run experiment for each group =====
    groups = {
        "narrow": narrow_seeds,
        "wide": wide_seeds,
    }

    group_results = {}

    for group_name, group_seeds in groups.items():
        print(f"\n{'='*70}")
        print(f"Group: {group_name.upper()} ({len(group_seeds)} seeds)")
        print(f"{'='*70}")

        # Create envs with seed pool
        envs = make_train_envs(
            env_id=args.env_id, num_envs=actual_n_envs,
            sim_backend=args.sim_backend, control_mode=args.control_mode,
            max_episode_steps=args.max_episode_steps, seed=args.seed,
            seed_pool=group_seeds.tolist(),
        )

        # Collect rollouts (returns per-rollout lists, not concatenated)
        print(f"\n  Phase 1: Collecting {args.n_rollouts} rollouts")
        roll_obs, roll_rew, roll_don, roll_nobs, roll_sids = collect_rollouts(
            args, model, envs, obs_dim, cond_dim, act_offset,
            normalize_obs, denormalize_actions, device, actual_n_envs,
        )

        envs.close()

        # Compute MC1 returns per rollout (no cross-rollout leakage), then concat
        mc1_per_roll = compute_mc1_per_rollout(roll_rew, roll_don, args.gamma)

        # Concatenate for critic training (training uses flat samples, not sequences)
        obs = torch.cat(roll_obs, dim=0)       # (T_total, E, cond, obs_dim)
        rew = torch.cat(roll_rew, dim=0)       # (T_total, E)
        don = torch.cat(roll_don, dim=0)       # (T_total, E)
        mc1 = torch.cat(mc1_per_roll, dim=0)   # (T_total, E)
        seed_ids = torch.cat(roll_sids, dim=0)  # (T_total, E)

        T_total, E = rew.shape
        N_total = T_total * E

        print(f"\n  Phase 2: MC1 returns (computed per rollout)")
        print(f"    Data: T={T_total}, E={E}, N={N_total}")
        print(f"    MC1: mean={mc1.mean():.4f}, std={mc1.std():.4f}, "
              f"min={mc1.min():.4f}, max={mc1.max():.4f}")
        total_rew = sum(r.sum().item() for r in roll_rew)
        print(f"    Rewards: mean={rew.mean():.4f}, total={total_rew:.1f}")

        # Train critics with different reward scales
        print(f"\n  Phase 3: Training critics")
        critics = {}
        for rs in args.reward_scales:
            rs_label = f"rs{int(rs)}" if rs == int(rs) else f"rs{rs}"
            label = f"MC1_{rs_label}"
            print(f"\n    --- {label} ---")
            t0 = time.time()
            critic = train_mc1(obs, mc1, reward_scale=rs, label=label)
            r, rho, ratio, vm, vs = eval_critic(critic, obs, mc1)
            critics[label] = critic
            print(f"    {label}: r={r:.3f}, rho={rho:.3f}, V_std/MC1_std={ratio:.3f}, "
                  f"V(mean={vm:.4f}, std={vs:.4f}), time={time.time()-t0:.1f}s")

        # Advantage analysis with best critic
        print(f"\n  Phase 4: Advantage analysis")
        last_rs = args.reward_scales[-1]
        best_rs_label = f"MC1_rs{int(last_rs)}" if last_rs == int(last_rs) else f"MC1_rs{last_rs}"
        best_critic = critics[best_rs_label]

        # GAE per rollout (no cross-rollout leakage)
        gae_adv, gae_values = compute_gae_per_rollout(
            roll_obs, roll_rew, roll_don, roll_nobs,
            best_critic, args.gamma, args.gae_lambda,
            obs_dim, args.cond_steps, D, device,
        )

        # MC advantages (ground truth): MC1_return - mean(MC1)
        mc_adv = mc1 - mc1.mean()

        # Flatten for correlation
        gae_flat = gae_adv.reshape(-1).cpu().numpy()
        mc_flat = mc_adv.reshape(-1).cpu().numpy()

        # Filter out near-zero entries (states where nothing happened)
        valid = np.abs(mc_flat) > 1e-8
        if valid.sum() > 100:
            adv_r = pearsonr(gae_flat[valid], mc_flat[valid])[0]
            adv_rho = spearmanr(gae_flat[valid], mc_flat[valid]).correlation
        else:
            adv_r = adv_rho = 0.0

        adv_pos_frac = (gae_flat > 0).mean()
        adv_mean = gae_flat.mean()
        adv_std = gae_flat.std()

        print(f"    GAE adv: mean={adv_mean:.4f}, std={adv_std:.4f}, pos_frac={adv_pos_frac:.3f}")
        print(f"    r(GAE, MC_adv)={adv_r:.3f}, rho(GAE, MC_adv)={adv_rho:.3f}")

        # Per-episode SNR: variance of advantages within vs across episodes
        seed_flat = seed_ids.reshape(-1).cpu().numpy()
        unique_episodes = np.unique(seed_flat)

        within_vars = []
        episode_means = []
        for ep_id in unique_episodes:
            mask = seed_flat == ep_id
            if mask.sum() < 2:
                continue
            ep_adv = gae_flat[mask]
            within_vars.append(ep_adv.var())
            episode_means.append(ep_adv.mean())

        if len(within_vars) > 2:
            mean_within_var = np.mean(within_vars)
            across_var = np.var(episode_means)
            # SNR = action signal / state noise
            # within_var = Var(A|episode) = action-dependent signal
            # across_var = Var(E[A|episode]) = state-dependent noise
            # Higher SNR = advantages carry more action info relative to state info
            snr = mean_within_var / (across_var + 1e-10)
            print(f"    Per-episode SNR: within_var={mean_within_var:.6f}, "
                  f"across_var={across_var:.6f}, SNR={snr:.3f}")
        else:
            snr = mean_within_var = across_var = float("nan")
            print(f"    Per-episode SNR: insufficient episodes")

        # Store results
        mc1_np = mc1.reshape(-1).cpu().numpy()

        critic_metrics = {}
        for label, critic in critics.items():
            r, rho, ratio, vm, vs = eval_critic(critic, obs, mc1)
            critic_metrics[label] = {"r": r, "rho": rho, "ratio": ratio, "v_mean": vm, "v_std": vs}

        group_results[group_name] = {
            "n_seeds": len(group_seeds),
            "mc1_mean": mc1_np.mean(),
            "mc1_std": mc1_np.std(),
            "mc1_min": mc1_np.min(),
            "mc1_max": mc1_np.max(),
            "critic_metrics": critic_metrics,
            "adv_r": adv_r,
            "adv_rho": adv_rho,
            "adv_pos_frac": adv_pos_frac,
            "adv_mean": adv_mean,
            "adv_std": adv_std,
            "snr": snr,
            "within_var": mean_within_var,
            "across_var": across_var,
            "reward_per_step": total_rew / N_total,
        }

    # ===== Print comparison =====
    print(f"\n{'='*70}")
    print(f"COMPARISON: Narrow [{args.narrow_lo}, {args.narrow_hi}) vs Wide [0, 1]")
    print(f"{'='*70}")

    # V(s) statistics
    print(f"\n--- V(s) = MC1 return statistics ---")
    print(f"{'Metric':<25s} {'Narrow':>12s} {'Wide':>12s}")
    print("-" * 50)
    for key in ["mc1_mean", "mc1_std", "mc1_min", "mc1_max", "reward_per_step"]:
        n_val = group_results["narrow"][key]
        w_val = group_results["wide"][key]
        print(f"{key:<25s} {n_val:>12.4f} {w_val:>12.4f}")

    # Critic quality
    print(f"\n--- Critic V quality (r, rho vs MC1 ground truth) ---")
    print(f"{'Method':<20s} {'Narrow r':>10s} {'Wide r':>10s} "
          f"{'Narrow rho':>12s} {'Wide rho':>12s} "
          f"{'Narrow ratio':>14s} {'Wide ratio':>14s}")
    print("-" * 82)
    all_labels = sorted(set(
        list(group_results["narrow"]["critic_metrics"].keys()) +
        list(group_results["wide"]["critic_metrics"].keys())
    ))
    for label in all_labels:
        nm = group_results["narrow"]["critic_metrics"].get(label, {})
        wm = group_results["wide"]["critic_metrics"].get(label, {})
        print(f"{label:<20s} "
              f"{nm.get('r', 0):.3f}{'':>5s} {wm.get('r', 0):.3f}{'':>5s} "
              f"{nm.get('rho', 0):.3f}{'':>7s} {wm.get('rho', 0):.3f}{'':>7s} "
              f"{nm.get('ratio', 0):.3f}{'':>9s} {wm.get('ratio', 0):.3f}")

    # Advantage quality
    print(f"\n--- Advantage quality ---")
    print(f"{'Metric':<25s} {'Narrow':>12s} {'Wide':>12s}")
    print("-" * 50)
    for key in ["adv_r", "adv_rho", "adv_pos_frac", "adv_mean", "adv_std",
                "snr", "within_var", "across_var"]:
        n_val = group_results["narrow"][key]
        w_val = group_results["wide"][key]
        fmt = ".6f" if "var" in key else ".3f"
        print(f"{key:<25s} {n_val:>12{fmt}} {w_val:>12{fmt}}")

    # ===== Save results =====
    save_data = {
        "args": vars(args),
        "narrow_seeds": narrow_seeds,
        "narrow_p": narrow_p,
        "wide_seeds": wide_seeds,
        "wide_p": wide_p,
    }
    for gname, gres in group_results.items():
        for key, val in gres.items():
            if key == "critic_metrics":
                for label, metrics in val.items():
                    for mk, mv in metrics.items():
                        save_data[f"{gname}_{label}_{mk}"] = mv
            else:
                save_data[f"{gname}_{key}"] = val

    np.savez(os.path.join(args.save_dir, "results.npz"), **save_data)

    # ===== Plot: P(success) distributions of both groups =====
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(narrow_p, bins=20, alpha=0.7, color="C0", label="Narrow", edgecolor="black")
    axes[0].hist(wide_p, bins=20, alpha=0.5, color="C1", label="Wide", edgecolor="black")
    axes[0].set_xlabel("P(success)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Seed P(success) distributions")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bar chart of key metrics
    metrics_to_plot = ["adv_rho", "snr"]
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    narrow_vals = [group_results["narrow"][m] for m in metrics_to_plot]
    wide_vals = [group_results["wide"][m] for m in metrics_to_plot]
    axes[1].bar(x - width/2, narrow_vals, width, label="Narrow", color="C0")
    axes[1].bar(x + width/2, wide_vals, width, label="Wide", color="C1")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics_to_plot)
    axes[1].set_title("Advantage quality comparison")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "comparison.png"), dpi=150)
    plt.close()
    print(f"\nPlots saved to {args.save_dir}/comparison.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
