"""TD Error Guided Tree Sampling — Does branching from high-TD-error states help?

Compares branch point selection strategies (all same transition budget):
1. rollout: N full rollouts (baseline)
2. uniform: 1 seed + uniform random branches
3. td_weighted: 1 seed + branches weighted by |TD error| from V₀
4. td_topk: 1 seed + branches from top 20% |TD error| states only

V₀ for TD error is trained on seed rollout (1 rollout = E*T transitions).

Usage:
  python -u -m RL.v_tree_td_guided --gamma 0.99
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
    # Ablation grid
    N_values: tuple[int, ...] = (5, 10, 20)
    strategies: tuple[str, ...] = ("uniform", "td_weighted", "td_topk")
    topk_frac: float = 0.2  # fraction of states for td_topk
    # Training
    td_reward_scale: float = 10.0
    ema_tau: float = 0.005
    epochs: int = 2000
    v0_epochs: int = 500  # epochs for V₀ training
    lr: float = 3e-4
    batch_size: int = 1000
    eval_every: int = 5
    critic_layers: int = 3
    hidden_dim: int = 256
    seed: int = 1
    env_id: str = "PickCube-v1"
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"
    output: str = "runs/v_tree_td_guided.png"


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

    seed_states = [None] * T  # seed_states[t] = state dict for rollout 0
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
    # Phase 2: MC16 ground truth (from rollout 0 states)
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

    eval_obs = data_pool[0]['obs']  # (T, E, D)
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
    # Phase 4: Train V₀ on seed rollout → compute TD errors
    # ══════════════════════════════════════════════════════════════════════
    def compute_r(critic, scale=1.0):
        with torch.no_grad():
            v = critic(eval_obs.reshape(-1, obs_dim)).view(-1).cpu().numpy() / scale
        return pearsonr(v, mc16_flat)[0]

    def train_td_ema(flat_s, flat_r, flat_ns, flat_nd, n_epochs=None, label=""):
        """Train TD+EMA, return (epochs_log, r_log, best_r, best_epoch, critic)."""
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

    print(f"\nPhase 4: Training V₀ on seed rollout (1 rollout, {E * T:,} transitions)...")
    sys.stdout.flush()
    t0 = time.time()

    seed_s, seed_r, seed_ns, seed_nd = per_rollout_flat[0]
    _, _, v0_r, v0_ep, v0_critic = train_td_ema(
        seed_s, seed_r, seed_ns, seed_nd,
        n_epochs=args.v0_epochs, label="V₀"
    )
    print(f"  V₀ peak r={v0_r:.4f} @ ep{v0_ep} ({time.time() - t0:.1f}s)")

    # Compute per-state TD error for all states in seed rollout
    v0_critic.eval()
    with torch.no_grad():
        v_s = v0_critic(seed_s).view(-1)  # V₀(s) for each transition
        v_ns = v0_critic(seed_ns).view(-1)  # V₀(s') for each transition
        scaled_r_seed = seed_r * rs
        td_target = scaled_r_seed + args.gamma * v_ns * (1 - seed_nd)
        td_error = (v_s - td_target).abs()  # |TD error| per transition

    # Reshape to (T, E) for branch point selection
    td_error_2d = td_error.view(T, E).cpu().numpy()

    print(f"\n  TD error stats:")
    print(f"    mean={td_error.mean():.4f}, std={td_error.std():.4f}")
    print(f"    min={td_error.min():.4f}, max={td_error.max():.4f}")
    print(f"    median={td_error.median():.4f}")

    # Show TD error by timestep
    td_by_t = td_error_2d.mean(axis=1)
    print(f"\n  TD error by timestep (top 5):")
    top_t = np.argsort(td_by_t)[::-1][:5]
    for t in top_t:
        print(f"    t={t}: mean={td_by_t[t]:.4f}")

    del v0_critic  # free memory
    torch.cuda.empty_cache()

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

    def collect_branches(strategy, budget, rng_seed_offset=0):
        """Collect branch transitions using given strategy up to budget."""
        branch_chunks = [[] for _ in range(4)]
        branch_total = 0
        rng = np.random.RandomState(args.seed + 999 + rng_seed_offset)
        n_batches = 0

        # Precompute sampling weights for td strategies
        if strategy == "td_weighted":
            flat_weights = td_error_2d.flatten()
            flat_weights = flat_weights / flat_weights.sum()
        elif strategy == "td_topk":
            flat_td = td_error_2d.flatten()
            k = max(1, int(len(flat_td) * args.topk_frac))
            topk_idx = np.argsort(flat_td)[-k:]
            # Uniform over top-k
            flat_weights = np.zeros(len(flat_td))
            flat_weights[topk_idx] = 1.0 / k

        branch_t_hist = np.zeros(T)  # track where we branch from

        with torch.no_grad():
            while branch_total < budget:
                # Select branch points based on strategy
                if strategy == "uniform":
                    bp_timesteps = rng.randint(0, T, size=E)
                    bp_envs = rng.randint(0, E, size=E)
                elif strategy in ("td_weighted", "td_topk"):
                    flat_idx = rng.choice(T * E, size=E, p=flat_weights)
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

    all_results = {}  # (N, strategy) -> dict
    rollout_results = {}  # N -> dict
    branch_histograms = {}  # (N, strategy) -> np.array

    for N in args.N_values:
        target_trans = N * E * T
        seed_trans = E * T  # 1 seed rollout
        branch_budget = target_trans - seed_trans

        print(f"{'─' * 60}")
        print(f"N={N} ({target_trans:,} transitions, branch_budget={branch_budget:,})")
        print(f"{'─' * 60}")

        # 6a: Rollout baseline
        print(f"  Rollout baseline...")
        r_s, r_r, r_ns, r_nd = build_rollout_td(N)
        t0 = time.time()
        ep_log, r_log, pk_r, pk_ep, _ = train_td_ema(r_s, r_r, r_ns, r_nd, label=f"Roll N={N}")
        print(f"    peak r={pk_r:.4f} @ ep{pk_ep} ({time.time()-t0:.1f}s)")
        rollout_results[N] = dict(epochs=ep_log, r_log=r_log, peak_r=pk_r, peak_ep=pk_ep)
        del r_s, r_r, r_ns, r_nd
        torch.cuda.empty_cache()

        # 6b-6d: Tree strategies
        for si, strategy in enumerate(args.strategies):
            print(f"  Tree [{strategy}]...")
            sys.stdout.flush()

            # Collect branches
            t0 = time.time()
            branch_flat, branch_hist = collect_branches(
                strategy, branch_budget, rng_seed_offset=si * 100 + N
            )
            collect_time = time.time() - t0
            branch_histograms[(N, strategy)] = branch_hist

            # Combine seed + branches
            tree_s = torch.cat([seed_s, branch_flat[0][:branch_budget]])
            tree_r = torch.cat([seed_r, branch_flat[1][:branch_budget]])
            tree_ns = torch.cat([seed_ns, branch_flat[2][:branch_budget]])
            tree_nd = torch.cat([seed_nd, branch_flat[3][:branch_budget]])

            # Train
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
    print(f"\n{'=' * 70}")
    print(f"SUMMARY — peak Pearson r (vs MC16)")
    print(f"{'=' * 70}")

    strategies = list(args.strategies)
    header = f"| {'N':>5} | {'Rollout':>8} |"
    for s in strategies:
        header += f" {s:>12} |"
    print(header)
    sep = f"|-------|----------|"
    for _ in strategies:
        sep += f"--------------|"
    print(sep)

    for N in args.N_values:
        row = f"| {N:>5} | {rollout_results[N]['peak_r']:>8.4f} |"
        for s in strategies:
            if (N, s) in all_results:
                row += f" {all_results[(N, s)]['peak_r']:>12.4f} |"
            else:
                row += f" {'—':>12} |"
        print(row)

    # Delta table
    print(f"\nDelta vs Rollout:")
    header2 = f"| {'N':>5} |"
    for s in strategies:
        header2 += f" {s:>12} |"
    print(header2)
    sep2 = f"|-------|"
    for _ in strategies:
        sep2 += f"--------------|"
    print(sep2)

    for N in args.N_values:
        row = f"| {N:>5} |"
        for s in strategies:
            if (N, s) in all_results:
                delta = all_results[(N, s)]['peak_r'] - rollout_results[N]['peak_r']
                row += f" {delta:>+12.4f} |"
            else:
                row += f" {'—':>12} |"
        print(row)

    # Delta vs uniform tree
    if "uniform" in strategies:
        print(f"\nDelta vs Uniform Tree:")
        header3 = f"| {'N':>5} |"
        other_strats = [s for s in strategies if s != "uniform"]
        for s in other_strats:
            header3 += f" {s:>12} |"
        print(header3)
        for N in args.N_values:
            row = f"| {N:>5} |"
            uniform_r = all_results.get((N, "uniform"), {}).get("peak_r", None)
            for s in other_strats:
                if (N, s) in all_results and uniform_r is not None:
                    delta = all_results[(N, s)]['peak_r'] - uniform_r
                    row += f" {delta:>+12.4f} |"
                else:
                    row += f" {'—':>12} |"
            print(row)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 8: Plot
    # ══════════════════════════════════════════════════════════════════════
    n_strats = len(strategies)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # [0,0] Peak r vs N
    ax = axes[0, 0]
    Ns = list(args.N_values)
    ax.plot(Ns, [rollout_results[N]['peak_r'] for N in Ns], 'k--o', lw=2, label='Rollout')
    strat_colors = {'uniform': 'blue', 'td_weighted': 'red', 'td_topk': 'orange'}
    for s in strategies:
        peaks = [all_results[(N, s)]['peak_r'] for N in Ns if (N, s) in all_results]
        ns_list = [N for N in Ns if (N, s) in all_results]
        ax.plot(ns_list, peaks, 's-', color=strat_colors.get(s, 'gray'), lw=1.5, label=s)
    ax.set_xlabel("N (data budget in rollouts)")
    ax.set_ylabel("Peak Pearson r (vs MC16)")
    ax.set_title("Peak r vs Data Budget")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # [0,1] Training curves for N=10
    ax = axes[0, 1]
    N_plot = 10 if 10 in args.N_values else args.N_values[len(args.N_values)//2]
    if N_plot in rollout_results:
        rr = rollout_results[N_plot]
        ax.plot(rr['epochs'], rr['r_log'], 'k-', lw=2, alpha=0.7,
                label=f"Rollout (pk={rr['peak_r']:.3f})")
    for s in strategies:
        if (N_plot, s) in all_results:
            res = all_results[(N_plot, s)]
            ax.plot(res['epochs'], res['r_log'], '-', color=strat_colors.get(s, 'gray'),
                    lw=1.5, alpha=0.7, label=f"{s} (pk={res['peak_r']:.3f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pearson r")
    ax.set_title(f"Training Curves (N={N_plot})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [1,0] Branch point histograms (for N=10)
    ax = axes[1, 0]
    timesteps = np.arange(T)
    for s in strategies:
        if (N_plot, s) in branch_histograms:
            hist = branch_histograms[(N_plot, s)]
            hist_norm = hist / hist.sum() if hist.sum() > 0 else hist
            ax.plot(timesteps, hist_norm, '-', color=strat_colors.get(s, 'gray'),
                    lw=1.5, alpha=0.8, label=s)
    # Also show TD error profile
    td_by_t_norm = td_by_t / td_by_t.sum()
    ax.plot(timesteps, td_by_t_norm, 'k--', lw=1, alpha=0.5, label='TD error profile')
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Branch frequency (normalized)")
    ax.set_title(f"Branch Point Distribution (N={N_plot})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [1,1] TD error heatmap (T × E)
    ax = axes[1, 1]
    im = ax.imshow(td_error_2d, aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_xlabel("Environment index")
    ax.set_ylabel("Timestep")
    ax.set_title(f"TD Error (V₀, seed rollout) | V₀ r={v0_r:.3f}")
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"TD Error Guided Tree Sampling | {args.env_id} | γ={args.gamma} | rs={rs}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {args.output}")

    # Save data
    save_path = args.output.replace('.png', '.pt')
    torch.save(dict(
        rollout_results=rollout_results,
        all_results={str(k): v for k, v in all_results.items()},
        branch_histograms={str(k): v for k, v in branch_histograms.items()},
        td_error_2d=td_error_2d,
        v0_r=v0_r,
        args=vars(args),
    ), save_path)
    print(f"Saved data: {save_path}")

    envs.close()
