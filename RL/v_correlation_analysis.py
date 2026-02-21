"""V(s) Correlation: Optimal MC V vs On-policy MC V vs IQL V.

Compares value functions evaluated at the same states:
  - V^{pi*}_{MC16}(s):  MC16 estimate using optimal policy (ckpt_301)
  - V^{pi*}_{MC1}(s):   single-sample MC (optimal)
  - V^{pi_on}_{MC16}(s): MC16 estimate using rollout policy (ckpt_76)
  - V^{pi_on}_{MC1}(s):  single-sample MC (on-policy)
  - V_IQL(s): IQL V trained on offline multi-checkpoint data

Pipeline:
  1. Rollout with ckpt_76, save env states
  2. MC re-rollout from each state (optimal + on-policy) -> V values
  3. Collect offline data (multi-checkpoint)
  4. Train IQL (offline only) -> V_IQL
  5. Scatter plots + correlation matrix

Usage:
  python -u -m RL.v_correlation_analysis
  python -u -m RL.v_correlation_analysis --cache_path runs/v_corr_cache.pt
"""

import os
import random
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from data.data_collection.ppo import Agent
from methods.iql.iql import train_iql, compute_nstep_targets


@dataclass
class Args:
    checkpoint: str = "runs/pickcube_ppo/ckpt_76_logstd-1.5.pt"
    """rollout policy (det SR=43.8%)"""
    optimal_checkpoint: str = "runs/pickcube_ppo/ckpt_301.pt"
    """optimal policy for MC re-rollouts"""

    # MC re-rollout
    mc_samples: int = 16
    num_envs: int = 128
    num_mc_envs: int = 0
    """0 = auto (num_envs * 2 * mc_samples)"""
    num_steps: int = 800
    max_episode_steps: int = 50
    gamma: float = 0.8
    reward_scale: float = 1.0

    # IQL training
    iql_expectile_tau: float = 0.7
    iql_epochs: int = 200
    iql_lr: float = 3e-4
    iql_batch_size: int = 256
    iql_nstep: int = 1
    iql_patience: int = 50
    iql_max_transitions: int = 0

    # Offline data
    iql_data_checkpoints: tuple[str, ...] = (
        "runs/pickcube_ppo/ckpt_1.pt",
        "runs/pickcube_ppo/ckpt_51.pt",
        "runs/pickcube_ppo/ckpt_101.pt",
        "runs/pickcube_ppo/ckpt_201.pt",
        "runs/pickcube_ppo/ckpt_301.pt",
    )
    iql_episodes_per_ckpt: int = 200
    iql_offline_num_envs: int = 512

    # Environment
    env_id: str = "PickCube-v1"
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"
    seed: int = 1
    cuda: bool = True

    # Cache
    cache_path: Optional[str] = None
    """Cache MC re-rollout data. If exists, skip Steps 1-2."""

    # Output
    output: str = "runs/v_correlation.png"
    save_data: bool = True


def collect_offline_data(checkpoints, episodes_per_ckpt, env_id, num_envs,
                         device, env_kwargs, reward_scale):
    """Collect mixed-quality offline data from multiple policy checkpoints."""
    envs = gym.make(env_id, num_envs=num_envs, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(
        envs, num_envs, ignore_terminations=False, record_metrics=True,
    )

    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(a):
        return torch.clamp(a.detach(), action_low, action_high)

    all_trajectories = []

    for ckpt_path in checkpoints:
        print(f"  Collecting {episodes_per_ckpt} episodes from {ckpt_path}...")
        sys.stdout.flush()
        collector = Agent(envs).to(device)
        collector.load_state_dict(torch.load(ckpt_path, map_location=device))
        collector.eval()

        step_obs, step_actions, step_rewards = [], [], []
        step_next_obs, step_dones = [], []
        episodes_collected = 0
        obs, _ = envs.reset()

        while episodes_collected < episodes_per_ckpt:
            with torch.no_grad():
                action = collector.get_action(obs, deterministic=False)
            next_obs, reward, term, trunc, infos = envs.step(clip_action(action))
            done = (term | trunc).float()

            step_obs.append(obs.cpu())
            step_actions.append(action.cpu())
            step_rewards.append((reward.view(-1) * reward_scale).cpu())
            step_next_obs.append(next_obs.cpu())
            step_dones.append(done.view(-1).cpu())

            if "final_info" in infos:
                episodes_collected += infos["_final_info"].sum().item()
            obs = next_obs

        del collector
        torch.cuda.empty_cache()

        ckpt_obs = torch.stack(step_obs)
        ckpt_actions = torch.stack(step_actions)
        ckpt_rewards = torch.stack(step_rewards)
        ckpt_next_obs = torch.stack(step_next_obs)
        ckpt_dones = torch.stack(step_dones)
        T = ckpt_obs.shape[0]

        for env_idx in range(num_envs):
            env_dns = ckpt_dones[:, env_idx]
            ep_start = 0
            for t in range(T):
                if env_dns[t] > 0.5 or t == T - 1:
                    all_trajectories.append({
                        "states": ckpt_obs[ep_start:t+1, env_idx],
                        "actions": ckpt_actions[ep_start:t+1, env_idx],
                        "next_states": ckpt_next_obs[ep_start:t+1, env_idx],
                        "rewards": ckpt_rewards[ep_start:t+1, env_idx],
                        "dones": ckpt_dones[ep_start:t+1, env_idx],
                    })
                    ep_start = t + 1

        print(f"    {int(episodes_collected)} episodes, "
              f"{T * num_envs} transitions")
        sys.stdout.flush()
        del step_obs, step_actions, step_rewards, step_next_obs, step_dones
        del ckpt_obs, ckpt_actions, ckpt_rewards, ckpt_next_obs, ckpt_dones

    envs.close()
    return all_trajectories


def train_and_eval_v(trajectories, eval_obs, device, args, label):
    """Train IQL on trajectories, return V predictions at eval_obs."""
    print(f"\n  Training IQL [{label}]...")
    sys.stdout.flush()

    if args.iql_max_transitions > 0:
        total = sum(t["states"].shape[0] for t in trajectories)
        if total > args.iql_max_transitions:
            random.shuffle(trajectories)
            kept, count = [], 0
            for t in trajectories:
                kept.append(t)
                count += t["states"].shape[0]
                if count >= args.iql_max_transitions:
                    break
            trajectories = kept
            print(f"    Subsampled: {total} -> {count} transitions")

    flat_s = torch.cat([t["states"] for t in trajectories])
    flat_a = torch.cat([t["actions"] for t in trajectories])
    flat_r = torch.cat([t["rewards"] for t in trajectories])
    flat_ns = torch.cat([t["next_states"] for t in trajectories])
    flat_d = torch.cat([t["dones"] for t in trajectories])

    total = flat_s.shape[0]
    print(f"    Data: {total} transitions, {len(trajectories)} trajectories")
    sys.stdout.flush()

    iql_args = SimpleNamespace(
        lr=args.iql_lr, weight_decay=1e-4, epochs=args.iql_epochs,
        batch_size=args.iql_batch_size, gamma=args.gamma,
        expectile_tau=args.iql_expectile_tau, tau_polyak=0.005,
        patience=args.iql_patience, grad_clip=0.5,
    )

    nstep_kw = {}
    if args.iql_nstep > 1:
        nret, boot_s, ndisc = compute_nstep_targets(
            trajectories, args.iql_nstep, args.gamma
        )
        nstep_kw = dict(
            nstep_returns=nret, bootstrap_states=boot_s, nstep_discounts=ndisc
        )

    q_net, v_net = train_iql(
        flat_s, flat_a, flat_r, flat_ns, flat_d,
        device, iql_args, **nstep_kw,
    )

    num_steps, num_envs = eval_obs.shape[0], eval_obs.shape[1]
    v_pred = torch.zeros(num_steps, num_envs, device=device)
    v_net.eval()
    with torch.no_grad():
        for t in range(num_steps):
            v_pred[t] = v_net(eval_obs[t]).squeeze(-1)

    print(f"    {label} V: mean={v_pred.mean():.4f}, std={v_pred.std():.4f}")
    sys.stdout.flush()

    del q_net, v_net, flat_s, flat_a, flat_r, flat_ns, flat_d
    torch.cuda.empty_cache()
    return v_pred


if __name__ == "__main__":
    import tyro
    args = tyro.cli(Args)

    if args.num_mc_envs == 0:
        args.num_mc_envs = args.num_envs * 2 * args.mc_samples

    samples_per_env = args.num_mc_envs // args.num_envs
    assert args.num_mc_envs % args.num_envs == 0
    assert samples_per_env >= 2 * args.mc_samples

    print(f"=== V(s) Correlation Analysis ===")
    print(f"  Rollout policy: {args.checkpoint}")
    print(f"  Optimal policy: {args.optimal_checkpoint}")
    print(f"  MC samples: {args.mc_samples}, gamma: {args.gamma}")
    print(f"  Envs: {args.num_envs}, MC envs: {args.num_mc_envs}, "
          f"Steps: {args.num_steps}")
    if args.cache_path:
        print(f"  Cache: {args.cache_path}")
    sys.stdout.flush()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )

    env_kwargs = dict(
        obs_mode="state", render_mode="rgb_array",
        sim_backend="physx_cuda" if device.type == "cuda" else "cpu",
        reward_mode=args.reward_mode, control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
    )

    use_cache = args.cache_path and os.path.exists(args.cache_path)

    if use_cache:
        # ══════════════════════════════════════════════════════════════
        #  Load cached MC data
        # ══════════════════════════════════════════════════════════════
        print(f"\n  Loading cache from {args.cache_path}...")
        cache = torch.load(args.cache_path, map_location=device)
        obs = cache["obs"]
        opt_v_all = cache["opt_v_all"]
        on_v_all = cache["on_v_all"]
        cache_rs = cache.get("reward_scale", 1.0)

        if abs(args.reward_scale - cache_rs) > 1e-8:
            sf = args.reward_scale / cache_rs
            print(f"  Rescaling: {cache_rs} -> {args.reward_scale} (x{sf})")
            opt_v_all *= sf
            on_v_all *= sf

        opt_v = opt_v_all.mean(dim=-1)
        on_v = on_v_all.mean(dim=-1)
        opt_v1 = opt_v_all[:, :, 0]
        on_v1 = on_v_all[:, :, 0]

        mc_k = opt_v_all.shape[-1]
        print(f"  obs={list(obs.shape)}, per-replica samples={mc_k}")
        print(f"  Opt V (MC{mc_k}): mean={opt_v.mean():.4f}, "
              f"std={opt_v.std():.4f}")
        print(f"  On  V (MC{mc_k}): mean={on_v.mean():.4f}, "
              f"std={on_v.std():.4f}")
        sys.stdout.flush()
        del cache

    else:
        # ══════════════════════════════════════════════════════════════
        #  Step 1: Rollout with initial policy, saving states
        # ══════════════════════════════════════════════════════════════
        envs = gym.make(args.env_id, num_envs=args.num_envs, **env_kwargs)
        mc_envs_raw = gym.make(
            args.env_id, num_envs=args.num_mc_envs, **env_kwargs
        )

        if isinstance(envs.action_space, gym.spaces.Dict):
            envs = FlattenActionSpaceWrapper(envs)
            mc_envs_raw = FlattenActionSpaceWrapper(mc_envs_raw)

        envs = ManiSkillVectorEnv(
            envs, args.num_envs,
            ignore_terminations=False, record_metrics=True,
        )
        mc_envs = ManiSkillVectorEnv(
            mc_envs_raw, args.num_mc_envs,
            ignore_terminations=False, record_metrics=False,
        )

        max_ep_steps = gym_utils.find_max_episode_steps_value(envs._env)

        agent = Agent(envs).to(device)
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
        agent.eval()
        print(f"  Loaded rollout policy: {args.checkpoint}")

        optimal_agent = Agent(envs).to(device)
        optimal_agent.load_state_dict(
            torch.load(args.optimal_checkpoint, map_location=device)
        )
        optimal_agent.eval()
        print(f"  Loaded optimal policy: {args.optimal_checkpoint}")

        action_low = torch.from_numpy(envs.single_action_space.low).to(device)
        action_high = torch.from_numpy(envs.single_action_space.high).to(device)

        def clip_action(a):
            return torch.clamp(a.detach(), action_low, action_high)

        _mc_zero_action = torch.zeros(
            args.num_mc_envs, *envs.single_action_space.shape, device=device
        )

        def _clone_state(sd):
            if isinstance(sd, dict):
                return {k: _clone_state(v) for k, v in sd.items()}
            return sd.clone()

        def _expand_state(sd, repeats):
            if isinstance(sd, dict):
                return {k: _expand_state(v, repeats) for k, v in sd.items()}
            if isinstance(sd, torch.Tensor) and sd.dim() > 0:
                return sd.repeat_interleave(repeats, dim=0)
            return sd

        def _restore_mc_state(sd, seed=None):
            mc_envs.reset(seed=seed if seed is not None else args.seed)
            mc_envs.base_env.set_state_dict(sd)
            mc_envs.base_env.step(_mc_zero_action)
            mc_envs.base_env.set_state_dict(sd)
            mc_envs.base_env._elapsed_steps[:] = 0
            return mc_envs.base_env.get_obs()

        # Precompute replica indices
        opt_indices = []
        on_indices = []
        for i in range(args.num_envs):
            base = i * samples_per_env
            opt_indices.extend(range(base, base + args.mc_samples))
            on_indices.extend(
                range(base + args.mc_samples, base + 2 * args.mc_samples)
            )
        opt_indices = torch.tensor(opt_indices, device=device, dtype=torch.long)
        on_indices = torch.tensor(on_indices, device=device, dtype=torch.long)

        print("\nStep 1: Rolling out with initial policy...")
        sys.stdout.flush()
        rollout_t0 = time.time()

        obs = torch.zeros(
            (args.num_steps, args.num_envs)
            + envs.single_observation_space.shape,
            device=device,
        )
        saved_states = []

        next_obs, _ = envs.reset(seed=args.seed)
        with torch.no_grad():
            for step in range(args.num_steps):
                saved_states.append(
                    _clone_state(envs.base_env.get_state_dict())
                )
                obs[step] = next_obs
                action = agent.get_action(next_obs, deterministic=False)
                next_obs, _, _, _, _ = envs.step(clip_action(action))

        print(f"  Rollout time: {time.time() - rollout_t0:.1f}s")
        sys.stdout.flush()

        # ══════════════════════════════════════════════════════════════
        #  Step 2: MC re-rollout -> per-replica returns
        # ══════════════════════════════════════════════════════════════
        print(f"\nStep 2: MC{args.mc_samples} re-rollout "
              f"(optimal + on-policy)...")
        sys.stdout.flush()
        rerollout_t0 = time.time()

        opt_v_all = torch.zeros(
            (args.num_steps, args.num_envs, args.mc_samples), device=device
        )
        on_v_all = torch.zeros(
            (args.num_steps, args.num_envs, args.mc_samples), device=device
        )

        with torch.no_grad():
            for t in tqdm(range(args.num_steps), desc="  MC re-rollout"):
                expanded = _expand_state(saved_states[t], samples_per_env)
                mc_obs = _restore_mc_state(expanded, seed=args.seed + t)

                first_actions = torch.zeros(
                    args.num_mc_envs,
                    *envs.single_action_space.shape,
                    device=device,
                )
                first_actions[opt_indices] = optimal_agent.get_action(
                    mc_obs[opt_indices], deterministic=False
                )
                first_actions[on_indices] = agent.get_action(
                    mc_obs[on_indices], deterministic=False
                )

                mc_obs, rew, term, trunc, _ = mc_envs.step(
                    clip_action(first_actions)
                )
                all_rews = [(rew.view(-1) * args.reward_scale).cpu()]
                env_done = (term | trunc).view(-1).bool()

                for _ in range(max_ep_steps - 1):
                    if env_done.all():
                        break
                    a = torch.zeros_like(first_actions)
                    a[opt_indices] = optimal_agent.get_action(
                        mc_obs[opt_indices], deterministic=False
                    )
                    a[on_indices] = agent.get_action(
                        mc_obs[on_indices], deterministic=False
                    )
                    mc_obs, rew, term, trunc, _ = mc_envs.step(clip_action(a))
                    all_rews.append(
                        (rew.view(-1) * args.reward_scale
                         * (~env_done).float()).cpu()
                    )
                    env_done = env_done | (term | trunc).view(-1).bool()

                # Discounted returns
                ret = torch.zeros(args.num_mc_envs, device=device)
                for s in reversed(range(len(all_rews))):
                    ret = all_rews[s].to(device) + args.gamma * ret
                ret = ret.view(args.num_envs, samples_per_env)
                opt_v_all[t] = ret[:, :args.mc_samples]
                on_v_all[t] = ret[:, args.mc_samples:2*args.mc_samples]
                del all_rews

        rerollout_time = time.time() - rerollout_t0
        opt_v = opt_v_all.mean(dim=2)
        on_v = on_v_all.mean(dim=2)
        opt_v1 = opt_v_all[:, :, 0]
        on_v1 = on_v_all[:, :, 0]

        print(f"  MC re-rollout time: {rerollout_time:.1f}s")
        print(f"  Opt V (MC16): mean={opt_v.mean():.4f}, "
              f"std={opt_v.std():.4f}")
        print(f"  On  V (MC16): mean={on_v.mean():.4f}, "
              f"std={on_v.std():.4f}")
        print(f"  Opt V (MC1):  mean={opt_v1.mean():.4f}, "
              f"std={opt_v1.std():.4f}")
        print(f"  On  V (MC1):  mean={on_v1.mean():.4f}, "
              f"std={on_v1.std():.4f}")
        sys.stdout.flush()

        # Save cache (just returns + obs, no trajectories)
        if args.cache_path:
            os.makedirs(os.path.dirname(args.cache_path) or ".", exist_ok=True)
            torch.save({
                "obs": obs.cpu(),
                "opt_v_all": opt_v_all.cpu(),
                "on_v_all": on_v_all.cpu(),
                "reward_scale": args.reward_scale,
            }, args.cache_path)
            print(f"  Cached to {args.cache_path}")
            sys.stdout.flush()

        del optimal_agent, agent, saved_states
        mc_envs.close()
        envs.close()
        del mc_envs, envs
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════
    #  Step 3: Collect offline data + train IQL
    # ══════════════════════════════════════════════════════════════════
    print(f"\nStep 3: Collecting offline data...")
    sys.stdout.flush()
    offline_t0 = time.time()
    offline_trajectories = collect_offline_data(
        args.iql_data_checkpoints, args.iql_episodes_per_ckpt,
        args.env_id, args.iql_offline_num_envs, device, env_kwargs,
        args.reward_scale,
    )
    n_offline = sum(t["states"].shape[0] for t in offline_trajectories)
    print(f"  Offline: {len(offline_trajectories)} trajectories, "
          f"{n_offline} transitions ({time.time()-offline_t0:.1f}s)")
    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════════
    #  Step 4: Train IQL (offline only)
    # ══════════════════════════════════════════════════════════════════
    iql_v = train_and_eval_v(
        offline_trajectories, obs, device, args, "offline"
    )
    del offline_trajectories
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════
    #  Step 5: Correlation analysis + plots
    # ══════════════════════════════════════════════════════════════════
    print("\nStep 5: Computing correlations...")
    sys.stdout.flush()

    vals = {
        r"$V^{\pi^*}_{MC16}$": opt_v.cpu().flatten().numpy(),
        r"$V^{\pi^*}_{MC1}$": opt_v1.cpu().flatten().numpy(),
        r"$V^{\pi_{on}}_{MC16}$": on_v.cpu().flatten().numpy(),
        r"$V^{\pi_{on}}_{MC1}$": on_v1.cpu().flatten().numpy(),
        r"$V_{IQL}$": iql_v.cpu().flatten().numpy(),
    }
    keys = list(vals.keys())
    n_v = len(keys)

    r_matrix = np.ones((n_v, n_v))
    rho_matrix = np.ones((n_v, n_v))
    for i in range(n_v):
        for j in range(i + 1, n_v):
            r_matrix[i, j] = r_matrix[j, i] = pearsonr(
                vals[keys[i]], vals[keys[j]]
            )[0]
            rho_matrix[i, j] = rho_matrix[j, i] = spearmanr(
                vals[keys[i]], vals[keys[j]]
            )[0]

    print(f"\n{'='*70}")
    print("  Pearson r:")
    header = "                  " + "  ".join(f"{k:>16s}" for k in keys)
    print(header)
    for i, k in enumerate(keys):
        row = f"  {k:>16s}"
        for j in range(n_v):
            row += f"  {r_matrix[i,j]:>16.4f}"
        print(row)

    print(f"\n  Spearman rho:")
    print(header)
    for i, k in enumerate(keys):
        row = f"  {k:>16s}"
        for j in range(n_v):
            row += f"  {rho_matrix[i,j]:>16.4f}"
        print(row)
    print(f"{'='*70}")
    sys.stdout.flush()

    if args.save_data:
        data_path = args.output.replace(".png", ".pt")
        torch.save({
            "opt_v": opt_v.cpu(), "on_v": on_v.cpu(),
            "opt_v1": opt_v1.cpu(), "on_v1": on_v1.cpu(),
            "iql_v": iql_v.cpu(), "obs": obs.cpu(),
            "r_matrix": r_matrix, "rho_matrix": rho_matrix,
            "args": vars(args),
        }, data_path)
        print(f"  Saved data to {data_path}")

    # ── Scatter plots ─────────────────────────────────────────────────
    print("\nGenerating plots...")

    pairs = [
        (0, 1),  # opt MC16 vs opt MC1
        (2, 3),  # on MC16 vs on MC1
        (0, 2),  # opt MC16 vs on MC16
        (0, 4),  # opt MC16 vs IQL
        (2, 4),  # on MC16 vs IQL
        (1, 4),  # opt MC1 vs IQL
    ]

    fig, axes = plt.subplots(2, 4, figsize=(28, 14))
    axes = axes.flatten()

    for idx, (xi, yi) in enumerate(pairs):
        ax = axes[idx]
        xv, yv = vals[keys[xi]], vals[keys[yi]]
        ax.scatter(xv, yv, alpha=0.05, s=3, edgecolors="none")
        lo = min(xv.min(), yv.min())
        hi = max(xv.max(), yv.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        r = r_matrix[xi, yi]
        rho = rho_matrix[xi, yi]
        ax.set_xlabel(keys[xi], fontsize=9)
        ax.set_ylabel(keys[yi], fontsize=9)
        ax.set_title(f"r={r:.4f}, rho={rho:.4f}", fontsize=10)
        ax.grid(True, alpha=0.3)

    # Correlation heatmap
    ax = axes[len(pairs)]
    short_labels = [r"$V^*_{16}$", r"$V^*_1$",
                    r"$V^{on}_{16}$", r"$V^{on}_1$", r"$V_{IQL}$"]
    im = ax.imshow(r_matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(n_v))
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_yticks(range(n_v))
    ax.set_yticklabels(short_labels, fontsize=8)
    for i in range(n_v):
        for j in range(n_v):
            ax.text(j, i, f"{r_matrix[i,j]:.3f}", ha="center", va="center",
                    fontsize=9, fontweight="bold" if i != j else "normal")
    ax.set_title("Pearson r matrix")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(len(pairs) + 1, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"V(s) Correlation: Optimal MC vs On-policy MC vs IQL\n"
        f"(mc={args.mc_samples}, gamma={args.gamma}, "
        f"tau={args.iql_expectile_tau})",
        fontsize=13,
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\n  Saved figure to {args.output}")
    plt.close(fig)
