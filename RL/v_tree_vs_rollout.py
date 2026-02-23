"""Tree Sampling vs Rollout Sampling for TD V Learning — Scaling Analysis.

Compares two data collection strategies for offline TD(0)+EMA V learning
across multiple data sizes:
- Rollout sampling: start all trajectories from envs.reset()
- Tree sampling: branch from intermediate states of seed rollouts

Both use the same total transition budget at each data size.
Evaluated against MC16 ground truth.

Usage:
  python -u -m RL.v_tree_vs_rollout --gamma 0.99
  python -u -m RL.v_tree_vs_rollout --gamma 0.99 --rollout_counts '(1,2,3,5,10)'
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
    # Scaling
    rollout_counts: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 50, 100)
    n_seed: int = 1
    """Number of seed rollouts for tree state pool (counted toward tree budget)."""
    # Training
    td_reward_scale: float = 10.0
    ema_tau: float = 0.005
    epochs: int = 500
    lr: float = 3e-4
    batch_size: int = 1000
    eval_every: int = 5
    critic_layers: int = 3
    hidden_dim: int = 256
    seed: int = 1
    env_id: str = "PickCube-v1"
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"
    output: str = "runs/v_tree_vs_rollout.png"


if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda")
    max_rollouts = max(args.rollout_counts)

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
    action_dim = int(np.prod(envs.single_action_space.shape))
    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()

    def clip_action(a):
        return torch.clamp(a.detach(), action_low, action_high)

    def make_v_net():
        layers = [layer_init(nn.Linear(obs_dim, args.hidden_dim)), nn.Tanh()]
        for _ in range(args.critic_layers - 1):
            layers += [layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)), nn.Tanh()]
        layers.append(layer_init(nn.Linear(args.hidden_dim, 1)))
        return nn.Sequential(*layers).to(device)

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

    zero_action = torch.zeros(E, *envs.single_action_space.shape, device=device)

    def _restore_state(sd, elapsed_steps=None, seed=None):
        envs.reset(seed=seed if seed is not None else args.seed)
        envs.base_env.set_state_dict(sd)
        envs.base_env.step(zero_action)
        envs.base_env.set_state_dict(sd)
        if elapsed_steps is not None:
            envs.base_env._elapsed_steps[:] = elapsed_steps
        else:
            envs.base_env._elapsed_steps[:] = 0
        return envs.base_env.get_obs()

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1: Collect all rollouts (max_rollouts), saving states for seeds
    # ══════════════════════════════════════════════════════════════════════
    print(f"Phase 1: Collecting {max_rollouts} rollouts (saving states for first {args.n_seed})...")
    sys.stdout.flush()
    t0 = time.time()

    # seed_states[ri][t] = state dict for all E envs at (rollout ri, timestep t)
    seed_states = [[None] * T for _ in range(args.n_seed)]
    data_pool = []  # list of per-rollout dicts

    for ri in range(max_rollouts):
        roll_obs = torch.zeros(T, E, obs_dim, device=device)
        roll_rewards = torch.zeros(T, E, device=device)
        roll_dones = torch.zeros(T, E, device=device)

        next_obs, _ = envs.reset(seed=args.seed + ri)
        next_done = torch.zeros(E, device=device)

        with torch.no_grad():
            for step in range(T):
                if ri < args.n_seed:
                    seed_states[ri][step] = _clone_state(envs.base_env.get_state_dict())
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
        if (ri + 1) % 20 == 0 or ri + 1 == max_rollouts:
            print(f"  {ri + 1}/{max_rollouts}")
            sys.stdout.flush()

    print(f"  Phase 1 done ({time.time() - t0:.1f}s)")

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2: On-policy MC16 ground truth (from rollout 0 states)
    # ══════════════════════════════════════════════════════════════════════
    samples_per_env = args.mc_samples
    num_mc_envs = E * samples_per_env
    print(f"\nPhase 2: On-policy MC16 ({num_mc_envs} mc_envs)...")
    sys.stdout.flush()
    t0 = time.time()

    mc_envs_raw = gym.make(args.env_id, num_envs=num_mc_envs, **env_kwargs)
    if isinstance(mc_envs_raw.action_space, gym.spaces.Dict):
        mc_envs_raw = FlattenActionSpaceWrapper(mc_envs_raw)
    mc_envs = ManiSkillVectorEnv(mc_envs_raw, num_mc_envs, ignore_terminations=False, record_metrics=False)
    mc_zero_action = torch.zeros(num_mc_envs, *envs.single_action_space.shape, device=device)

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
            expanded = _expand_state(seed_states[0][t], samples_per_env)
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
            on_mc16[t] = ret.view(E, samples_per_env).mean(dim=1)
            if (t + 1) % 10 == 0:
                print(f"  MC16 step {t + 1}/{T}")
                sys.stdout.flush()

    mc_envs.close()
    del mc_envs, mc_envs_raw
    torch.cuda.empty_cache()

    eval_obs = data_pool[0]['obs']  # (T, E, D)
    mc16_flat = on_mc16.reshape(-1).cpu().numpy()
    print(f"  On-policy MC16: mean={on_mc16.mean():.4f}, std={on_mc16.std():.4f}")
    print(f"  Phase 2 done ({time.time() - t0:.1f}s)")

    # ══════════════════════════════════════════════════════════════════════
    # Phase 3: Collect tree branch data (enough for max budget)
    # ══════════════════════════════════════════════════════════════════════
    max_target = max_rollouts * E * T
    seed_trans = args.n_seed * E * T
    max_branch_trans = max_target - seed_trans

    print(f"\nPhase 3: Collecting tree branch data (target={max_branch_trans:,} branch transitions)...")
    sys.stdout.flush()
    t0 = time.time()

    def _assemble_composite_state(branch_points):
        ref = seed_states[0][0]

        def _assemble_inner(ref_val, key_path):
            if isinstance(ref_val, dict):
                return {k: _assemble_inner(v, key_path + [k]) for k, v in ref_val.items()}
            if isinstance(ref_val, torch.Tensor) and ref_val.dim() > 0:
                slices = []
                for (si, t, ei) in branch_points:
                    src = seed_states[si][t]
                    val = src
                    for k in key_path:
                        val = val[k]
                    slices.append(val[ei:ei+1])
                return torch.cat(slices, dim=0)
            return ref_val

        return _assemble_inner(ref, [])

    # Collect branch transitions into a flat pool
    branch_s_chunks = []
    branch_r_chunks = []
    branch_ns_chunks = []
    branch_nd_chunks = []
    branch_trans_total = 0

    rng = np.random.RandomState(args.seed + 999)
    n_batches = 0

    with torch.no_grad():
        while branch_trans_total < max_branch_trans:
            bp_timesteps = rng.randint(0, T, size=E)
            bp_seeds = rng.randint(0, args.n_seed, size=E)
            bp_envs = rng.randint(0, E, size=E)
            branch_points = list(zip(bp_seeds.tolist(), bp_timesteps.tolist(), bp_envs.tolist()))

            composite = _assemble_composite_state(branch_points)
            _restore_state(composite, seed=args.seed + 5000 + n_batches)
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
                continue

            b_obs = torch.stack(batch_obs)
            b_rew = torch.stack(batch_rew)
            b_ns = torch.stack(batch_ns)
            b_nd = torch.stack(batch_nd)

            # Active mask
            active_mask = torch.ones_like(b_rew)
            running = torch.ones(E, device=device).bool()
            for si in range(b_obs.shape[0]):
                active_mask[si] = running.float()
                running = running & (b_nd[si] == 0)

            flat_mask = active_mask.reshape(-1).bool()
            branch_s_chunks.append(b_obs.reshape(-1, obs_dim)[flat_mask])
            branch_r_chunks.append(b_rew.reshape(-1)[flat_mask])
            branch_ns_chunks.append(b_ns.reshape(-1, obs_dim)[flat_mask])
            branch_nd_chunks.append(b_nd.reshape(-1)[flat_mask])

            branch_trans_total += flat_mask.sum().item()
            n_batches += 1

            if n_batches % 50 == 0:
                print(f"  Batch {n_batches}: {branch_trans_total:,}/{max_branch_trans:,}")
                sys.stdout.flush()

    # Concatenate all branch data
    branch_flat_s = torch.cat(branch_s_chunks)
    branch_flat_r = torch.cat(branch_r_chunks)
    branch_flat_ns = torch.cat(branch_ns_chunks)
    branch_flat_nd = torch.cat(branch_nd_chunks)

    print(f"  Branch pool: {branch_flat_s.shape[0]:,} transitions from {n_batches} batches")
    print(f"  Phase 3 done ({time.time() - t0:.1f}s)")

    del branch_s_chunks, branch_r_chunks, branch_ns_chunks, branch_nd_chunks
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════
    # Prepare seed rollout TD data (for tree: included in every size)
    # ══════════════════════════════════════════════════════════════════════
    seed_flat_s_list, seed_flat_r_list, seed_flat_ns_list, seed_flat_nd_list = [], [], [], []
    for ri in range(args.n_seed):
        d = data_pool[ri]
        obs_ri = d['obs']       # (T, E, D)
        rew_ri = d['rewards']   # (T, E)
        done_ri = d['dones']    # (T, E)
        ns_ri = torch.zeros_like(obs_ri)
        ns_ri[:-1] = obs_ri[1:]
        ns_ri[-1] = d['next_obs']
        nd_ri = torch.zeros_like(rew_ri)
        nd_ri[:-1] = done_ri[1:]
        nd_ri[-1] = d['next_done']

        seed_flat_s_list.append(obs_ri.reshape(-1, obs_dim))
        seed_flat_r_list.append(rew_ri.reshape(-1))
        seed_flat_ns_list.append(ns_ri.reshape(-1, obs_dim))
        seed_flat_nd_list.append(nd_ri.reshape(-1))

    seed_flat_s = torch.cat(seed_flat_s_list)
    seed_flat_r = torch.cat(seed_flat_r_list)
    seed_flat_ns = torch.cat(seed_flat_ns_list)
    seed_flat_nd = torch.cat(seed_flat_nd_list)

    # Free seed states (no longer needed)
    del seed_states
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════
    # Helper: build rollout TD data for N rollouts
    # ══════════════════════════════════════════════════════════════════════
    def build_rollout_td(n_rollouts):
        """Build flat (s, r, ns, nd) from first n_rollouts in data_pool."""
        s_list, r_list, ns_list, nd_list = [], [], [], []
        for ri in range(n_rollouts):
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

            s_list.append(obs_ri.reshape(-1, obs_dim))
            r_list.append(rew_ri.reshape(-1))
            ns_list.append(ns_ri.reshape(-1, obs_dim))
            nd_list.append(nd_ri.reshape(-1))
        return torch.cat(s_list), torch.cat(r_list), torch.cat(ns_list), torch.cat(nd_list)

    def build_tree_td(n_rollouts):
        """Build flat (s, r, ns, nd) for tree: n_seed seed rollouts + branches up to budget."""
        target = n_rollouts * E * T
        # Start with seed data
        if seed_trans >= target:
            # Budget fits entirely in seed data — subsample
            return seed_flat_s[:target], seed_flat_r[:target], seed_flat_ns[:target], seed_flat_nd[:target]
        # Seed + branches
        branch_needed = target - seed_trans
        branch_n = min(branch_needed, branch_flat_s.shape[0])
        return (
            torch.cat([seed_flat_s, branch_flat_s[:branch_n]]),
            torch.cat([seed_flat_r, branch_flat_r[:branch_n]]),
            torch.cat([seed_flat_ns, branch_flat_ns[:branch_n]]),
            torch.cat([seed_flat_nd, branch_flat_nd[:branch_n]]),
        )

    # ══════════════════════════════════════════════════════════════════════
    # Phase 4: Train TD+EMA at each data size
    # ══════════════════════════════════════════════════════════════════════
    rs = args.td_reward_scale

    def compute_r(critic, scale=1.0):
        with torch.no_grad():
            v = critic(eval_obs.reshape(-1, obs_dim)).view(-1).cpu().numpy() / scale
        return pearsonr(v, mc16_flat)[0]

    def train_td_ema(flat_s, flat_r, flat_ns, flat_nd, label=""):
        """Train TD+EMA, return (epochs_log, r_log, best_r, best_epoch)."""
        N = flat_s.shape[0]
        mb = min(args.batch_size, N)
        scaled_r = flat_r * rs

        critic = make_v_net()
        critic_target = copy.deepcopy(critic)
        opt = optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)

        epochs_log, r_log = [], []
        best_r, best_ep = -999, 0

        for epoch in range(args.epochs):
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

        return epochs_log, r_log, best_r, best_ep

    print(f"\n{'=' * 70}")
    print(f"Phase 4: Training TD+EMA rs={rs} across data sizes")
    print(f"  rollout_counts = {args.rollout_counts}")
    print(f"  n_seed = {args.n_seed} (seed trans = {seed_trans:,})")
    print(f"  epochs = {args.epochs}")
    print(f"{'=' * 70}\n")

    results = {}  # (N, method) -> {epochs, r_log, peak_r, peak_ep, n_trans}

    for N_roll in args.rollout_counts:
        target = N_roll * E * T
        print(f"--- N={N_roll} ({target:,} transitions) ---")
        sys.stdout.flush()

        # Rollout
        r_s, r_r, r_ns, r_nd = build_rollout_td(N_roll)
        t0 = time.time()
        ep_log, r_log, pk_r, pk_ep = train_td_ema(r_s, r_r, r_ns, r_nd, f"Roll N={N_roll}")
        print(f"  Rollout: peak r={pk_r:.4f} @ ep{pk_ep} ({r_s.shape[0]:,} trans, {time.time()-t0:.1f}s)")
        results[(N_roll, "Rollout")] = dict(epochs=ep_log, r_log=r_log, peak_r=pk_r, peak_ep=pk_ep, n_trans=r_s.shape[0])
        del r_s, r_r, r_ns, r_nd

        # Tree
        t_s, t_r, t_ns, t_nd = build_tree_td(N_roll)
        t0 = time.time()
        ep_log, r_log, pk_r, pk_ep = train_td_ema(t_s, t_r, t_ns, t_nd, f"Tree N={N_roll}")
        print(f"  Tree:    peak r={pk_r:.4f} @ ep{pk_ep} ({t_s.shape[0]:,} trans, {time.time()-t0:.1f}s)")
        results[(N_roll, "Tree")] = dict(epochs=ep_log, r_log=r_log, peak_r=pk_r, peak_ep=pk_ep, n_trans=t_s.shape[0])
        del t_s, t_r, t_ns, t_nd

        torch.cuda.empty_cache()
        sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════════════
    # Summary Table
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"SUMMARY — peak Pearson r")
    print(f"{'=' * 70}")
    print(f"| {'N':>5} | {'Trans':>9} | {'Rollout r':>10} | {'Tree r':>10} | {'Delta':>8} | {'Tree trans':>10} |")
    print(f"|-------|-----------|------------|------------|----------|------------|")
    for N_roll in args.rollout_counts:
        rr = results[(N_roll, "Rollout")]
        tr = results[(N_roll, "Tree")]
        delta = tr['peak_r'] - rr['peak_r']
        print(f"| {N_roll:>5} | {rr['n_trans']:>9,} | {rr['peak_r']:>10.4f} | {tr['peak_r']:>10.4f} | {delta:>+8.4f} | {tr['n_trans']:>10,} |")

    # ══════════════════════════════════════════════════════════════════════
    # Plot
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # [0] Scaling curve: peak r vs N
    ax = axes[0]
    Ns = list(args.rollout_counts)
    roll_peaks = [results[(n, "Rollout")]['peak_r'] for n in Ns]
    tree_peaks = [results[(n, "Tree")]['peak_r'] for n in Ns]
    ax.plot(Ns, roll_peaks, 'o-', color='tab:blue', lw=2, label='Rollout')
    ax.plot(Ns, tree_peaks, 's-', color='tab:orange', lw=2, label='Tree')
    ax.set_xlabel("N rollouts (data size)")
    ax.set_ylabel("Peak Pearson r (vs MC16)")
    ax.set_title(f"V Quality Scaling: Tree vs Rollout (n_seed={args.n_seed})")
    ax.set_xscale('log')
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # [1] Training curves for selected sizes
    ax = axes[1]
    highlight = [n for n in [1, 3, 10, 100] if n in args.rollout_counts]
    colors_map = plt.cm.viridis(np.linspace(0.15, 0.85, len(highlight)))
    for i, n in enumerate(highlight):
        rr = results[(n, "Rollout")]
        tr = results[(n, "Tree")]
        ax.plot(rr['epochs'], rr['r_log'], '-', color=colors_map[i], lw=1.5, alpha=0.7,
                label=f"Roll N={n} (pk={rr['peak_r']:.2f})")
        ax.plot(tr['epochs'], tr['r_log'], '--', color=colors_map[i], lw=1.5, alpha=0.7,
                label=f"Tree N={n} (pk={tr['peak_r']:.2f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pearson r")
    ax.set_title("Training Curves (solid=Rollout, dashed=Tree)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Tree vs Rollout Scaling | γ={args.gamma} | rs={rs} | n_seed={args.n_seed}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {args.output}")

    save_path = args.output.replace('.png', '.pt')
    torch.save(dict(results=results, args=vars(args)), save_path)
    print(f"Saved data to {save_path}")

    envs.close()
