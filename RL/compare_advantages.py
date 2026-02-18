"""Compare three advantage estimation methods on the same (s, a) data.

1. MC optimal:   A^{π*}(s,a) = Q^{π*}(s,a) - V^{π*}(s)  (optimal policy MC re-rollout)
2. MC on-policy: A^π(s,a)    = Q^π(s,a)    - V^π(s)       (current policy MC re-rollout)
3. IQL:          A_IQL(s,a)  = Q_IQL(s,a)  - V_IQL(s)      (IQL network estimates)

Outputs scatter plots comparing ranking consistency across methods.

Usage:
  python -m RL.compare_advantages
  python -m RL.compare_advantages --mc_samples 32 --iql_expectile_tau 0.9
"""

import os
import random
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tyro
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from data.data_collection.ppo import Agent
from methods.iql.iql import QNetwork, train_iql, compute_nstep_targets


@dataclass
class Args:
    checkpoint: str = "runs/pickcube_ppo/ckpt_101.pt"
    """initial policy for rollout and on-policy MC"""
    optimal_checkpoint: str = "runs/pickcube_ppo/ckpt_301.pt"
    """optimal policy for optimal MC re-rollout"""

    # IQL
    iql_data_checkpoints: tuple[str, ...] = (
        "runs/pickcube_ppo/ckpt_1.pt",
        "runs/pickcube_ppo/ckpt_51.pt",
        "runs/pickcube_ppo/ckpt_101.pt",
        "runs/pickcube_ppo/ckpt_201.pt",
        "runs/pickcube_ppo/ckpt_301.pt",
    )
    iql_episodes_per_ckpt: int = 200
    iql_expectile_tau: float = 0.7
    iql_epochs: int = 200
    iql_lr: float = 3e-4
    iql_batch_size: int = 256
    iql_nstep: int = 1
    iql_patience: int = 50

    # Environment
    env_id: str = "PickCube-v1"
    num_envs: int = 100
    num_mc_envs: int = 0
    """MC re-rollout envs. 0 = auto"""
    mc_samples: int = 16
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"
    max_episode_steps: int = 50
    gamma: float = 0.8
    reward_scale: float = 1.0

    # Data collection
    num_steps: int = 50
    """rollout length"""
    seed: int = 1
    cuda: bool = True

    # Output
    output: str = "runs/advantage_comparison.png"


# ── IQL data collection (from iql_awr_offline.py) ────────────────────────

def collect_iql_data(checkpoints, episodes_per_ckpt, envs, num_envs, device,
                     clip_action_fn, reward_scale):
    """Collect mixed-quality offline data from multiple policy checkpoints."""
    all_trajectories = []

    for ckpt_path in checkpoints:
        print(f"  Collecting {episodes_per_ckpt} episodes from {ckpt_path}...")

        collector = Agent(envs).to(device)
        collector.load_state_dict(torch.load(ckpt_path, map_location=device))
        collector.eval()

        step_obs = []
        step_actions = []
        step_rewards = []
        step_next_obs = []
        step_dones = []

        episodes_collected = 0
        obs, _ = envs.reset()

        while episodes_collected < episodes_per_ckpt:
            with torch.no_grad():
                action = collector.get_action(obs, deterministic=False)
            next_obs, reward, term, trunc, infos = envs.step(
                clip_action_fn(action)
            )
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
            env_obs = ckpt_obs[:, env_idx]
            env_acts = ckpt_actions[:, env_idx]
            env_rews = ckpt_rewards[:, env_idx]
            env_nobs = ckpt_next_obs[:, env_idx]
            env_dns = ckpt_dones[:, env_idx]

            ep_start = 0
            for t in range(T):
                if env_dns[t] > 0.5 or t == T - 1:
                    all_trajectories.append({
                        "states": env_obs[ep_start : t + 1],
                        "actions": env_acts[ep_start : t + 1],
                        "next_states": env_nobs[ep_start : t + 1],
                        "rewards": env_rews[ep_start : t + 1],
                        "dones": env_dns[ep_start : t + 1],
                    })
                    ep_start = t + 1

        print(f"    {int(episodes_collected)} episodes, {T} steps, "
              f"{T * num_envs} transitions")

        del step_obs, step_actions, step_rewards, step_next_obs, step_dones
        del ckpt_obs, ckpt_actions, ckpt_rewards, ckpt_next_obs, ckpt_dones

    return all_trajectories


# ── Per-timestep min-max normalization ────────────────────────────────────

def per_timestep_normalize(adv):
    """adv: (num_steps, num_envs) -> per-timestep min-max normalize to [0, 1]."""
    adv_norm = torch.zeros_like(adv)
    for t in range(adv.shape[0]):
        row = adv[t]
        rmin, rmax = row.min(), row.max()
        if rmax - rmin > 1e-8:
            adv_norm[t] = (row - rmin) / (rmax - rmin)
        else:
            adv_norm[t] = 0.5
    return adv_norm


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.num_mc_envs == 0:
        args.num_mc_envs = args.num_envs * 2 * args.mc_samples

    samples_per_env = args.num_mc_envs // args.num_envs
    assert args.num_mc_envs % args.num_envs == 0
    assert samples_per_env >= 2 * args.mc_samples

    print(f"=== Advantage Comparison: MC optimal vs MC on-policy vs IQL ===")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Optimal: {args.optimal_checkpoint}")
    print(f"  MC samples: {args.mc_samples}, envs: {args.num_envs}, "
          f"mc_envs: {args.num_mc_envs}")
    print(f"  IQL: tau={args.iql_expectile_tau}, epochs={args.iql_epochs}")

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ── Environment setup ─────────────────────────────────────────────
    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="physx_cuda" if device.type == "cuda" else "cpu",
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
    )

    envs = gym.make(args.env_id, num_envs=args.num_envs, **env_kwargs)
    mc_envs_raw = gym.make(args.env_id, num_envs=args.num_mc_envs, **env_kwargs)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        mc_envs_raw = FlattenActionSpaceWrapper(mc_envs_raw)

    envs = ManiSkillVectorEnv(
        envs, args.num_envs,
        ignore_terminations=False,
        record_metrics=True,
    )
    mc_envs = ManiSkillVectorEnv(
        mc_envs_raw, args.num_mc_envs,
        ignore_terminations=False,
        record_metrics=False,
    )

    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(a):
        return torch.clamp(a.detach(), action_low, action_high)

    # ── Load agents ───────────────────────────────────────────────────
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()
    print(f"  Loaded agent: {args.checkpoint}")

    optimal_agent = Agent(envs).to(device)
    optimal_agent.load_state_dict(torch.load(args.optimal_checkpoint, map_location=device))
    optimal_agent.eval()
    print(f"  Loaded optimal: {args.optimal_checkpoint}")

    # ── Rollout env helpers ───────────────────────────────────────────
    _zero_action = torch.zeros(
        args.num_envs, *envs.single_action_space.shape, device=device
    )

    def _clone_state(state_dict):
        if isinstance(state_dict, dict):
            return {k: _clone_state(v) for k, v in state_dict.items()}
        return state_dict.clone()

    # ── MC env helpers ────────────────────────────────────────────────
    _mc_zero_action = torch.zeros(
        args.num_mc_envs, *envs.single_action_space.shape, device=device
    )

    def _expand_state(state_dict, repeats):
        if isinstance(state_dict, dict):
            return {k: _expand_state(v, repeats) for k, v in state_dict.items()}
        if isinstance(state_dict, torch.Tensor) and state_dict.dim() > 0:
            return state_dict.repeat_interleave(repeats, dim=0)
        return state_dict

    def _restore_mc_state(state_dict, seed=None):
        mc_envs.reset(seed=seed if seed is not None else args.seed)
        mc_envs.base_env.set_state_dict(state_dict)
        mc_envs.base_env.step(_mc_zero_action)
        mc_envs.base_env.set_state_dict(state_dict)
        mc_envs.base_env._elapsed_steps[:] = 0
        return mc_envs.base_env.get_obs()

    # V-replica indices
    v_indices = []
    for i in range(args.num_envs):
        base = i * samples_per_env
        v_indices.extend(range(base + args.mc_samples, base + 2 * args.mc_samples))
    v_indices = torch.tensor(v_indices, device=device, dtype=torch.long)

    # ══════════════════════════════════════════════════════════════════
    #  Phase 1: IQL data collection + training
    # ══════════════════════════════════════════════════════════════════
    print("\n--- Phase 1: IQL data collection ---")
    iql_t0 = time.time()

    trajectories = collect_iql_data(
        args.iql_data_checkpoints, args.iql_episodes_per_ckpt,
        envs, args.num_envs, device, clip_action, args.reward_scale,
    )

    flat_states = torch.cat([t["states"] for t in trajectories])
    flat_actions = torch.cat([t["actions"] for t in trajectories])
    flat_rewards = torch.cat([t["rewards"] for t in trajectories])
    flat_next_states = torch.cat([t["next_states"] for t in trajectories])
    flat_dones = torch.cat([t["dones"] for t in trajectories])

    total_transitions = flat_states.shape[0]
    print(f"  Total: {total_transitions} transitions, {len(trajectories)} trajectories")
    print(f"  Collection time: {time.time() - iql_t0:.1f}s")

    print("\n--- Phase 2: IQL training ---")
    iql_train_t0 = time.time()

    iql_args = SimpleNamespace(
        lr=args.iql_lr,
        weight_decay=1e-4,
        epochs=args.iql_epochs,
        batch_size=args.iql_batch_size,
        gamma=args.gamma,
        expectile_tau=args.iql_expectile_tau,
        tau_polyak=0.005,
        patience=args.iql_patience,
        grad_clip=0.5,
    )

    nstep_kw = {}
    if args.iql_nstep > 1:
        print(f"  Computing {args.iql_nstep}-step TD targets...")
        nret, boot_s, ndisc = compute_nstep_targets(
            trajectories, args.iql_nstep, args.gamma
        )
        nstep_kw = dict(
            nstep_returns=nret, bootstrap_states=boot_s, nstep_discounts=ndisc
        )

    q_net, v_net = train_iql(
        flat_states, flat_actions, flat_rewards, flat_next_states, flat_dones,
        device, iql_args, **nstep_kw,
    )

    print(f"  IQL training time: {time.time() - iql_train_t0:.1f}s")

    del trajectories, flat_states, flat_actions, flat_rewards
    del flat_next_states, flat_dones, nstep_kw
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════
    #  Phase 3: Rollout with ckpt_101 + save states
    # ══════════════════════════════════════════════════════════════════
    print("\n--- Phase 3: Rollout with initial policy ---")
    rollout_t0 = time.time()

    obs_buf = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape,
        device=device,
    )
    act_buf = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape,
        device=device,
    )

    saved_states = []
    next_obs, _ = envs.reset(seed=args.seed)

    for step in range(args.num_steps):
        saved_states.append(_clone_state(envs.base_env.get_state_dict()))
        obs_buf[step] = next_obs

        with torch.no_grad():
            action = agent.get_action(next_obs, deterministic=False)
        act_buf[step] = action

        next_obs, reward, term, trunc, infos = envs.step(clip_action(action))

    print(f"  Rollout: {args.num_steps} steps x {args.num_envs} envs = "
          f"{args.num_steps * args.num_envs} (s,a) pairs")
    print(f"  Rollout time: {time.time() - rollout_t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════
    #  Phase 4: MC re-rollout (optimal + on-policy)
    # ══════════════════════════════════════════════════════════════════

    def mc_rerollout(rollout_agent, label):
        """Run parallel MC re-rollout using rollout_agent for both Q and V."""
        mc_q = torch.zeros((args.num_steps, args.num_envs), device=device)
        mc_v = torch.zeros((args.num_steps, args.num_envs), device=device)

        with torch.no_grad():
            for t in tqdm(range(args.num_steps), desc=f"  MC re-rollout ({label})", leave=False):
                expanded_state = _expand_state(saved_states[t], samples_per_env)
                mc_obs = _restore_mc_state(expanded_state, seed=args.seed + t)

                # Build first actions
                first_actions = torch.zeros(
                    args.num_mc_envs, *envs.single_action_space.shape, device=device
                )

                # Q replicas: use the rollout action
                for i in range(args.num_envs):
                    base = i * samples_per_env
                    first_actions[base : base + args.mc_samples] = act_buf[t][i]

                # V replicas: sample from rollout_agent
                v_obs = mc_obs[v_indices]
                first_actions[v_indices] = rollout_agent.get_action(
                    v_obs, deterministic=False
                )

                # Step all mc_envs
                mc_obs, rew, term, trunc, _ = mc_envs.step(clip_action(first_actions))
                all_rews = [rew.view(-1) * args.reward_scale]
                env_done = (term | trunc).view(-1).bool()

                # Follow rollout_agent until done
                for _ in range(args.max_episode_steps - 1):
                    if env_done.all():
                        break
                    a = rollout_agent.get_action(mc_obs, deterministic=False)
                    mc_obs, rew, term, trunc, _ = mc_envs.step(clip_action(a))
                    all_rews.append(
                        rew.view(-1) * args.reward_scale * (~env_done).float()
                    )
                    env_done = env_done | (term | trunc).view(-1).bool()

                # Discounted returns
                ret = torch.zeros(args.num_mc_envs, device=device)
                for s in reversed(range(len(all_rews))):
                    ret = all_rews[s] + args.gamma * ret

                ret = ret.view(args.num_envs, samples_per_env)
                mc_q[t] = ret[:, :args.mc_samples].mean(dim=1)
                mc_v[t] = ret[:, args.mc_samples : 2 * args.mc_samples].mean(dim=1)

        return mc_q, mc_v

    print("\n--- Phase 4a: MC re-rollout (optimal) ---")
    t0 = time.time()
    mc_q_opt, mc_v_opt = mc_rerollout(optimal_agent, "optimal")
    adv_opt = mc_q_opt - mc_v_opt
    print(f"  Time: {time.time() - t0:.1f}s")
    print(f"  Q_opt: mean={mc_q_opt.mean():.4f}, V_opt: mean={mc_v_opt.mean():.4f}")
    print(f"  A_opt: mean={adv_opt.mean():.4f}, std={adv_opt.std():.4f}")

    print("\n--- Phase 4b: MC re-rollout (on-policy) ---")
    t0 = time.time()
    mc_q_onp, mc_v_onp = mc_rerollout(agent, "on-policy")
    adv_onp = mc_q_onp - mc_v_onp
    print(f"  Time: {time.time() - t0:.1f}s")
    print(f"  Q_onp: mean={mc_q_onp.mean():.4f}, V_onp: mean={mc_v_onp.mean():.4f}")
    print(f"  A_onp: mean={adv_onp.mean():.4f}, std={adv_onp.std():.4f}")

    # ══════════════════════════════════════════════════════════════════
    #  Phase 5: IQL advantage on rollout data
    # ══════════════════════════════════════════════════════════════════
    print("\n--- Phase 5: IQL advantage ---")
    b_obs = obs_buf.reshape((-1,) + envs.single_observation_space.shape)
    b_actions = act_buf.reshape((-1,) + envs.single_action_space.shape)
    batch_size = args.num_steps * args.num_envs

    with torch.no_grad():
        iql_q_list, iql_v_list = [], []
        for start in range(0, batch_size, 1024):
            end = min(start + 1024, batch_size)
            s = b_obs[start:end]
            a = b_actions[start:end]
            iql_q_list.append(q_net(s, a).squeeze(-1))
            iql_v_list.append(v_net(s).squeeze(-1))
        iql_q_flat = torch.cat(iql_q_list)
        iql_v_flat = torch.cat(iql_v_list)

    adv_iql = (iql_q_flat - iql_v_flat).view(args.num_steps, args.num_envs)
    print(f"  Q_IQL: mean={iql_q_flat.mean():.4f}, V_IQL: mean={iql_v_flat.mean():.4f}")
    print(f"  A_IQL: mean={adv_iql.mean():.4f}, std={adv_iql.std():.4f}")

    del q_net, v_net, iql_q_list, iql_v_list, iql_q_flat, iql_v_flat
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════
    #  Phase 6: Normalize + plot
    # ══════════════════════════════════════════════════════════════════
    print("\n--- Phase 6: Plotting ---")

    # Move to CPU for plotting
    adv_opt_cpu = adv_opt.cpu()
    adv_onp_cpu = adv_onp.cpu()
    adv_iql_cpu = adv_iql.cpu()

    # Raw (unnormalized) flat
    raw_opt = adv_opt_cpu.flatten().numpy()
    raw_onp = adv_onp_cpu.flatten().numpy()
    raw_iql = adv_iql_cpu.flatten().numpy()

    # Per-timestep normalized
    norm_opt = per_timestep_normalize(adv_opt_cpu)
    norm_onp = per_timestep_normalize(adv_onp_cpu)
    norm_iql = per_timestep_normalize(adv_iql_cpu)

    flat_opt = norm_opt.flatten().numpy()
    flat_onp = norm_onp.flatten().numpy()
    flat_iql = norm_iql.flatten().numpy()

    # ── Correlations ──────────────────────────────────────────────────
    def compute_corr(x, y):
        r, _ = pearsonr(x, y)
        rho, _ = spearmanr(x, y)
        return r, rho

    # Normalized correlations
    r_opt_iql, rho_opt_iql = compute_corr(flat_opt, flat_iql)
    r_opt_onp, rho_opt_onp = compute_corr(flat_opt, flat_onp)
    r_onp_iql, rho_onp_iql = compute_corr(flat_onp, flat_iql)

    # Raw correlations
    raw_r_opt_iql, raw_rho_opt_iql = compute_corr(raw_opt, raw_iql)
    raw_r_opt_onp, raw_rho_opt_onp = compute_corr(raw_opt, raw_onp)
    raw_r_onp_iql, raw_rho_onp_iql = compute_corr(raw_onp, raw_iql)

    # ── Per-timestep correlations ─────────────────────────────────────
    per_t_rho_opt_iql = []
    per_t_rho_opt_onp = []
    per_t_rho_onp_iql = []
    for t in range(args.num_steps):
        r1, _ = spearmanr(adv_opt_cpu[t].numpy(), adv_iql_cpu[t].numpy())
        r2, _ = spearmanr(adv_opt_cpu[t].numpy(), adv_onp_cpu[t].numpy())
        r3, _ = spearmanr(adv_onp_cpu[t].numpy(), adv_iql_cpu[t].numpy())
        per_t_rho_opt_iql.append(r1)
        per_t_rho_opt_onp.append(r2)
        per_t_rho_onp_iql.append(r3)

    # ── Print summary ─────────────────────────────────────────────────
    print("\n=== Correlation Summary ===")
    print(f"{'Pair':<30} {'Pearson r':>10} {'Spearman ρ':>12} "
          f"{'Raw r':>10} {'Raw ρ':>10}")
    print("-" * 76)
    print(f"{'MC Optimal vs IQL':<30} {r_opt_iql:>10.4f} {rho_opt_iql:>12.4f} "
          f"{raw_r_opt_iql:>10.4f} {raw_rho_opt_iql:>10.4f}")
    print(f"{'MC Optimal vs MC On-policy':<30} {r_opt_onp:>10.4f} {rho_opt_onp:>12.4f} "
          f"{raw_r_opt_onp:>10.4f} {raw_rho_opt_onp:>10.4f}")
    print(f"{'MC On-policy vs IQL':<30} {r_onp_iql:>10.4f} {rho_onp_iql:>12.4f} "
          f"{raw_r_onp_iql:>10.4f} {raw_rho_onp_iql:>10.4f}")

    print(f"\n=== Advantage Statistics ===")
    for name, raw in [("MC Optimal", raw_opt), ("MC On-policy", raw_onp), ("IQL", raw_iql)]:
        print(f"  {name:<15}: mean={raw.mean():.4f}, std={raw.std():.4f}, "
              f"pos%={(raw > 0).mean():.1%}")

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))

    scatter_kw = dict(alpha=0.1, s=4, edgecolors="none")

    def plot_scatter(ax, x, y, xlabel, ylabel, title, r, rho, raw_r, raw_rho):
        ax.scatter(x, y, **scatter_kw)
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\n"
                     f"Normalized: r={r:.3f}, ρ={rho:.3f}\n"
                     f"Raw: r={raw_r:.3f}, ρ={raw_rho:.3f}",
                     fontsize=10)
        ax.set_aspect("equal")

    # (0,0): MC optimal vs IQL
    plot_scatter(axes[0, 0], flat_opt, flat_iql,
                 r"MC Optimal $A^{\pi^*}$", r"IQL $A_{IQL}$",
                 "MC Optimal vs IQL",
                 r_opt_iql, rho_opt_iql, raw_r_opt_iql, raw_rho_opt_iql)

    # (0,1): MC optimal vs MC on-policy
    plot_scatter(axes[0, 1], flat_opt, flat_onp,
                 r"MC Optimal $A^{\pi^*}$", r"MC On-policy $A^{\pi}$",
                 "MC Optimal vs MC On-policy",
                 r_opt_onp, rho_opt_onp, raw_r_opt_onp, raw_rho_opt_onp)

    # (0,2): MC on-policy vs IQL
    plot_scatter(axes[0, 2], flat_onp, flat_iql,
                 r"MC On-policy $A^{\pi}$", r"IQL $A_{IQL}$",
                 "MC On-policy vs IQL",
                 r_onp_iql, rho_onp_iql, raw_r_onp_iql, raw_rho_onp_iql)

    # (1,0): Per-timestep Spearman ρ
    ax = axes[1, 0]
    ts = np.arange(args.num_steps)
    ax.plot(ts, per_t_rho_opt_iql, label="Optimal vs IQL", marker=".", ms=3)
    ax.plot(ts, per_t_rho_opt_onp, label="Optimal vs On-policy", marker=".", ms=3)
    ax.plot(ts, per_t_rho_onp_iql, label="On-policy vs IQL", marker=".", ms=3)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(r"Spearman $\rho$")
    ax.set_title("Per-timestep Rank Correlation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 1)

    # (1,1): Summary statistics table
    ax = axes[1, 1]
    ax.axis("off")
    table_data = [
        ["Metric", "MC Optimal", "MC On-policy", "IQL"],
        ["Mean (raw)", f"{raw_opt.mean():.4f}", f"{raw_onp.mean():.4f}", f"{raw_iql.mean():.4f}"],
        ["Std (raw)", f"{raw_opt.std():.4f}", f"{raw_onp.std():.4f}", f"{raw_iql.std():.4f}"],
        ["Pos% (raw)", f"{(raw_opt > 0).mean():.1%}", f"{(raw_onp > 0).mean():.1%}", f"{(raw_iql > 0).mean():.1%}"],
        ["", "", "", ""],
        ["Pair", "Pearson r", "Spearman ρ", ""],
        [" Opt vs IQL (norm)", f"{r_opt_iql:.4f}", f"{rho_opt_iql:.4f}", ""],
        [" Opt vs OnP (norm)", f"{r_opt_onp:.4f}", f"{rho_opt_onp:.4f}", ""],
        [" OnP vs IQL (norm)", f"{r_onp_iql:.4f}", f"{rho_onp_iql:.4f}", ""],
        [" Opt vs IQL (raw)", f"{raw_r_opt_iql:.4f}", f"{raw_rho_opt_iql:.4f}", ""],
        [" Opt vs OnP (raw)", f"{raw_r_opt_onp:.4f}", f"{raw_rho_opt_onp:.4f}", ""],
        [" OnP vs IQL (raw)", f"{raw_r_onp_iql:.4f}", f"{raw_rho_onp_iql:.4f}", ""],
    ]
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)
    # Bold header row
    for j in range(4):
        table[0, j].set_text_props(fontweight="bold")
        table[5, j].set_text_props(fontweight="bold")
    ax.set_title("Summary Statistics", fontsize=10)

    # (1,2): Raw advantage distributions (histograms)
    ax = axes[1, 2]
    bins = 100
    ax.hist(raw_opt, bins=bins, alpha=0.4, label="MC Optimal", density=True)
    ax.hist(raw_onp, bins=bins, alpha=0.4, label="MC On-policy", density=True)
    ax.hist(raw_iql, bins=bins, alpha=0.4, label="IQL", density=True)
    ax.set_xlabel("Advantage (raw)")
    ax.set_ylabel("Density")
    ax.set_title("Raw Advantage Distributions")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Advantage Comparison: {args.checkpoint}\n"
        f"MC{args.mc_samples} samples, γ={args.gamma}, IQL τ={args.iql_expectile_tau}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {args.output}")

    # ── Save raw data ─────────────────────────────────────────────────
    data_path = args.output.replace(".png", "_data.pt")
    torch.save({
        "adv_opt": adv_opt_cpu,
        "adv_onp": adv_onp_cpu,
        "adv_iql": adv_iql_cpu,
        "adv_opt_norm": norm_opt,
        "adv_onp_norm": norm_onp,
        "adv_iql_norm": norm_iql,
        "per_t_rho_opt_iql": per_t_rho_opt_iql,
        "per_t_rho_opt_onp": per_t_rho_opt_onp,
        "per_t_rho_onp_iql": per_t_rho_onp_iql,
        "args": vars(args),
    }, data_path)
    print(f"Saved raw data to {data_path}")

    # ── Cleanup ───────────────────────────────────────────────────────
    envs.close()
    mc_envs.close()
    plt.close(fig)
    print("\nDone!")
