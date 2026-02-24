"""Branch Point Ablation (V-bin + Variance-bin) for Tree Sampling

Branches from states grouped by MC16 ground truth V value OR MC variance.
Tests which V-value/variance region benefits most from resampling.

V bins:
  v0_20:   V in [0, 0.2)   — likely failure states
  v20_40:  V in [0.2, 0.4) — uncertain, leaning failure
  v40_60:  V in [0.4, 0.6) — most uncertain
  v60_100: V in [0.6, 1.0] — likely success states

Variance bins (by MC16 return variance percentiles):
  var_lo:  bottom 25% variance — outcome is predictable
  var_mid: middle 50% variance — moderate uncertainty
  var_hi:  top 25% variance   — outcome is highly uncertain

  uniform: all states — baseline for both

Usage:
  python -u -m RL.v_tree_branch_ablation_vbin --gamma 0.99
  python -u -m RL.v_tree_branch_ablation_vbin --env_id PegInsertionSide-v1 \
    --checkpoint runs/peginsertion_ppo_ema99/ckpt_231.pt \
    --gamma 0.97 --num_steps 100 --max_episode_steps 100 \
    --output runs/v_tree_branch_ablation_vbin_peg.png
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


VBINS = {
    "v0_20":   (0.0, 0.2),
    "v20_40":  (0.2, 0.4),
    "v40_60":  (0.4, 0.6),
    "v60_100": (0.6, 1.01),  # slightly above 1 to include V=1.0
    "uniform": (0.0, 1.01),
}


@dataclass
class Args:
    checkpoint: str = "runs/pickcube_ppo/ckpt_76_logstd-1.5.pt"
    gamma: float = 0.99
    num_envs: int = 100
    num_steps: int = 50
    max_episode_steps: int = 50
    mc_samples: int = 16
    # Ablation grid
    N_values: tuple[int, ...] = (2, 3, 5)
    vbins: tuple[str, ...] = ("v0_20", "v20_40", "v40_60", "v60_100", "var_lo", "var_mid", "var_hi", "uniform")
    # Training
    td_n: int = 1  # n-step TD (1=TD(0), 10=TD(10), etc.)
    td_reward_scale: float = 10.0
    ema_tau: float = 0.005
    epochs: int = 2000
    lr: float = 3e-4
    batch_size: int = 1000
    eval_every: int = 5
    critic_layers: int = 3
    hidden_dim: int = 256
    seed: int = 1
    env_id: str = "PickCube-v1"
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"
    output: str = "runs/v_tree_branch_ablation_vbin.png"
    mc_cache: str = ""


if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda")

    max_N = max(args.N_values)
    save_state_rollouts = 1
    max_rollouts = max_N

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

    def _clone_state_cpu(sd):
        if isinstance(sd, dict):
            return {k: _clone_state_cpu(v) for k, v in sd.items()}
        if isinstance(sd, torch.Tensor):
            return sd.cpu().clone()
        return sd

    def _expand_state(sd, repeats):
        if isinstance(sd, dict):
            return {k: _expand_state(v, repeats) for k, v in sd.items()}
        if isinstance(sd, torch.Tensor) and sd.dim() > 0:
            return sd.repeat_interleave(repeats, dim=0)
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

    def _state_to_device(sd, dev):
        if isinstance(sd, dict):
            return {k: _state_to_device(v, dev) for k, v in sd.items()}
        if isinstance(sd, torch.Tensor):
            return sd.to(dev)
        return sd

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1: Collect rollouts, saving states for rollout 0
    # ══════════════════════════════════════════════════════════════════════
    print(f"Phase 1: Collecting {max_rollouts} rollouts (saving states for rollout 0)...")
    sys.stdout.flush()
    t0 = time.time()

    seed_states = [[None] * T for _ in range(save_state_rollouts)]
    data_pool = []

    for ri in range(max_rollouts):
        roll_obs = torch.zeros(T, E, obs_dim, device=device)
        roll_rewards = torch.zeros(T, E, device=device)
        roll_dones = torch.zeros(T, E, device=device)

        next_obs, _ = envs.reset(seed=args.seed + ri)
        next_done = torch.zeros(E, device=device)

        with torch.no_grad():
            for step in range(T):
                if ri < save_state_rollouts:
                    seed_states[ri][step] = _clone_state_cpu(envs.base_env.get_state_dict())
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

    # ── Phase 1b: Identify episode segments & step-in-episode ──
    print(f"\nPhase 1b: Identifying episode segments in rollout 0...")
    dones_np = data_pool[0]['dones'].cpu().numpy()

    episodes_by_env = [[] for _ in range(E)]
    step_in_ep = np.zeros((T, E), dtype=int)

    for e in range(E):
        t_start = 0
        for t in range(1, T):
            if dones_np[t, e] > 0.5:
                episodes_by_env[e].append((t_start, t - t_start))
                for s in range(t_start, t):
                    step_in_ep[s, e] = s - t_start
                t_start = t
        episodes_by_env[e].append((t_start, T - t_start))
        for s in range(t_start, T):
            step_in_ep[s, e] = s - t_start

    all_ep_lengths = [l for eps in episodes_by_env for (_, l) in eps]
    n_completed = sum(1 for eps in episodes_by_env for i, _ in enumerate(eps) if i < len(eps) - 1)
    print(f"  Episodes: {len(all_ep_lengths)} ({n_completed} completed), "
          f"mean length={np.mean(all_ep_lengths):.1f}")

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2: MC16 ground truth (with caching)
    # ══════════════════════════════════════════════════════════════════════
    eval_obs = data_pool[0]['obs']

    ckpt_base = os.path.splitext(os.path.basename(args.checkpoint))[0]
    mc_cache_path = args.mc_cache or (
        f"runs/mc16_cache_{args.env_id}_{ckpt_base}_g{args.gamma}_s{args.seed}"
        f"_E{E}_T{T}_mc{args.mc_samples}.pt"
    )

    need_recompute = False
    if os.path.exists(mc_cache_path):
        cached = torch.load(mc_cache_path, map_location=device)
        if 'mc16_var' in cached:
            print(f"\nPhase 2: Loading MC16 from cache: {mc_cache_path}")
            on_mc16 = cached['mc16'].to(device)
            on_mc16_var = cached['mc16_var'].to(device)
            print(f"  MC16: mean={on_mc16.mean():.4f}, std={on_mc16.std():.4f}")
            print(f"  MC16 var: mean={on_mc16_var.mean():.4f}, std={on_mc16_var.std():.4f}")
        else:
            print(f"\nPhase 2: Cache exists but missing mc16_var, recomputing...")
            need_recompute = True
    else:
        need_recompute = True

    if need_recompute:
        samples_per_env = args.mc_samples
        num_mc_envs = E * samples_per_env
        print(f"\nPhase 2: Computing MC16 ({num_mc_envs} mc_envs)...")
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
        on_mc16_var = torch.zeros(T, E, device=device)

        with torch.no_grad():
            for t in range(T):
                expanded = _expand_state(_state_to_device(seed_states[0][t], device), samples_per_env)
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
                ret_per_state = ret.view(E, samples_per_env)
                on_mc16[t] = ret_per_state.mean(dim=1)
                on_mc16_var[t] = ret_per_state.var(dim=1)
                if (t + 1) % 10 == 0:
                    print(f"  MC16 step {t + 1}/{T}")
                    sys.stdout.flush()

        mc_envs.close()
        del mc_envs, mc_envs_raw
        torch.cuda.empty_cache()

        print(f"  MC16: mean={on_mc16.mean():.4f}, std={on_mc16.std():.4f}")
        print(f"  MC16 var: mean={on_mc16_var.mean():.4f}, std={on_mc16_var.std():.4f}")
        print(f"  Phase 2 done ({time.time() - t0:.1f}s)")

        os.makedirs(os.path.dirname(mc_cache_path) or ".", exist_ok=True)
        torch.save({'mc16': on_mc16.cpu(), 'mc16_var': on_mc16_var.cpu()}, mc_cache_path)
        print(f"  Saved MC16 cache to {mc_cache_path}")

    mc16_flat = on_mc16.reshape(-1).cpu().numpy()
    mc16_np = on_mc16.cpu().numpy()  # (T, E)
    mc16_var_np = on_mc16_var.cpu().numpy()  # (T, E)

    # ── Phase 2b: Compute variance bin thresholds ──
    mc16_var_flat = mc16_var_np.ravel()
    var_p25 = np.percentile(mc16_var_flat, 25)
    var_p75 = np.percentile(mc16_var_flat, 75)
    var_max = mc16_var_flat.max() + 0.001

    # Dynamic variance bins based on percentiles
    VARBINS = {
        "var_lo":  (0.0, var_p25),
        "var_mid": (var_p25, var_p75),
        "var_hi":  (var_p75, var_max),
    }
    print(f"\nPhase 2b: Building V-bin & variance-bin state indices...")
    print(f"  MC16 variance percentiles: p25={var_p25:.6f}, p75={var_p75:.6f}, max={var_max:.6f}")

    # ── Build eligible state lists per bin ──
    eligible_states = {}  # bin_name -> list of (t, e)
    for vbin_name in args.vbins:
        pairs = []
        if vbin_name in VBINS:
            # V-value bin
            v_lo, v_hi = VBINS[vbin_name]
            for t in range(T):
                for e in range(E):
                    v = mc16_np[t, e]
                    if v_lo <= v < v_hi:
                        pairs.append((t, e))
        elif vbin_name in VARBINS:
            # Variance bin
            var_lo, var_hi = VARBINS[vbin_name]
            for t in range(T):
                for e in range(E):
                    var_val = mc16_var_np[t, e]
                    if var_lo <= var_val < var_hi:
                        pairs.append((t, e))
        elif vbin_name == "uniform":
            for t in range(T):
                for e in range(E):
                    pairs.append((t, e))
        eligible_states[vbin_name] = pairs

    # Print V-value bin distribution
    v_bins_in_args = [vb for vb in args.vbins if vb in VBINS]
    var_bins_in_args = [vb for vb in args.vbins if vb in VARBINS]

    print(f"\n  V-value bins ({T * E} total states):")
    print(f"  {'Bin':>10} | {'Range':>12} | {'Count':>6} | {'Frac':>6} | {'Mean V':>8} | {'Std V':>8}")
    print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}")
    for vbin_name in v_bins_in_args + ["uniform"]:
        pairs = eligible_states.get(vbin_name, [])
        n = len(pairs)
        if n > 0:
            vals = [mc16_np[t, e] for t, e in pairs]
            if vbin_name in VBINS:
                v_lo, v_hi = VBINS[vbin_name]
                label = f"[{v_lo:.1f}, {min(v_hi,1.0):.1f}{']' if v_hi>1 else ')'}"
            else:
                label = "[all]"
            print(f"  {vbin_name:>10} | {label:>12} | {n:>6} | {n/(T*E):>5.1%} | {np.mean(vals):>8.4f} | {np.std(vals):>8.4f}")

    # Print variance bin distribution
    print(f"\n  Variance bins ({T * E} total states):")
    print(f"  {'Bin':>10} | {'Var Range':>16} | {'Count':>6} | {'Frac':>6} | {'Mean Var':>10} | {'Mean V':>8}")
    print(f"  {'-'*10}-+-{'-'*16}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*8}")
    for vbin_name in var_bins_in_args + ["uniform"]:
        pairs = eligible_states.get(vbin_name, [])
        n = len(pairs)
        if n > 0:
            var_vals = [mc16_var_np[t, e] for t, e in pairs]
            v_vals = [mc16_np[t, e] for t, e in pairs]
            if vbin_name in VARBINS:
                vr_lo, vr_hi = VARBINS[vbin_name]
                label = f"[{vr_lo:.4f}, {vr_hi:.4f})"
            else:
                label = "[all]"
            print(f"  {vbin_name:>10} | {label:>16} | {n:>6} | {n/(T*E):>5.1%} | {np.mean(var_vals):>10.6f} | {np.mean(v_vals):>8.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # Phase 3: Build per-rollout flat TD data
    # ══════════════════════════════════════════════════════════════════════
    print(f"\nPhase 3: Building per-rollout flat TD data...")
    sys.stdout.flush()
    t0 = time.time()

    per_rollout_flat = []
    for ri in range(max_rollouts):
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

    def get_seed_flat():
        return per_rollout_flat[0]

    def build_rollout_td(n_rollouts):
        return tuple(
            torch.cat([per_rollout_flat[ri][k] for ri in range(n_rollouts)])
            for k in range(4)
        )

    print(f"  {max_rollouts} rollouts, {max_rollouts * E * T:,} transitions")
    print(f"  Phase 3 done ({time.time() - t0:.1f}s)")

    # ══════════════════════════════════════════════════════════════════════
    # Training helpers
    # ══════════════════════════════════════════════════════════════════════
    rs = args.td_reward_scale

    def compute_r(critic, scale=1.0):
        with torch.no_grad():
            v = critic(eval_obs.reshape(-1, obs_dim)).view(-1).cpu().numpy() / scale
        return pearsonr(v, mc16_flat)[0]

    def train_td_ema(flat_s, flat_r, flat_ns, flat_nd, label=""):
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

    # ══════════════════════════════════════════════════════════════════════
    # Phase 4: Rollout baselines
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"Phase 4: Rollout baselines")
    print(f"  N_values = {args.N_values}, vbins = {args.vbins}")
    print(f"  epochs = {args.epochs}")
    print(f"{'=' * 70}\n")

    rollout_results = {}
    for N in args.N_values:
        target = N * E * T
        print(f"--- Rollout N={N} ({target:,} transitions) ---")
        sys.stdout.flush()
        r_s, r_r, r_ns, r_nd = build_rollout_td(N)
        t0 = time.time()
        ep_log, r_log, pk_r, pk_ep = train_td_ema(r_s, r_r, r_ns, r_nd, f"Roll N={N}")
        print(f"  peak r={pk_r:.4f} @ ep{pk_ep} ({r_s.shape[0]:,} trans, {time.time()-t0:.1f}s)")
        rollout_results[N] = dict(epochs=ep_log, r_log=r_log, peak_r=pk_r, peak_ep=pk_ep, n_trans=r_s.shape[0])
        del r_s, r_r, r_ns, r_nd
        torch.cuda.empty_cache()
        sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════════════
    # Phase 5: V-bin branch ablation
    # ══════════════════════════════════════════════════════════════════════
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
                return torch.cat(slices, dim=0).to(device)
            if isinstance(ref_val, torch.Tensor):
                return ref_val.to(device)
            return ref_val

        return _assemble_inner(ref, [])

    def sample_branch_points(rng, eligible):
        """Sample E branch points from eligible (t, e) pairs."""
        bp_timesteps = np.zeros(E, dtype=int)
        bp_envs = np.zeros(E, dtype=int)
        bp_elapsed = np.zeros(E, dtype=int)
        bp_v_values = np.zeros(E, dtype=float)

        for i in range(E):
            idx = rng.randint(0, len(eligible))
            t, e = eligible[idx]
            bp_timesteps[i] = t
            bp_envs[i] = e
            bp_elapsed[i] = step_in_ep[t, e]
            bp_v_values[i] = mc16_np[t, e]

        return bp_timesteps, bp_envs, bp_elapsed, bp_v_values

    def collect_branches(vbin_name, budget, bin_idx):
        eligible = eligible_states[vbin_name]
        branch_s_chunks, branch_r_chunks = [], []
        branch_ns_chunks, branch_nd_chunks = [], []
        branch_total = 0
        rng = np.random.RandomState(args.seed + 999 + bin_idx)
        n_batches = 0
        branch_lengths = []
        branch_v_values_all = []

        with torch.no_grad():
            while branch_total < budget:
                bp_timesteps, bp_envs, bp_elapsed, bp_v = sample_branch_points(rng, eligible)
                branch_points = [(0, int(bp_timesteps[i]), int(bp_envs[i])) for i in range(E)]
                branch_v_values_all.extend(bp_v.tolist())

                composite = _assemble_composite_state(branch_points)
                _restore_state(composite, seed=args.seed + 5000 + bin_idx * 10000 + n_batches)
                elapsed = torch.tensor(bp_elapsed, dtype=torch.int32, device=device)
                envs.base_env._elapsed_steps[:] = elapsed

                next_obs = envs.base_env.get_obs()
                batch_obs, batch_ns, batch_rew, batch_nd = [], [], [], []
                active = torch.ones(E, device=device).bool()

                for step_i in range(args.max_episode_steps):
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
                branch_s_chunks.append(b_obs.reshape(-1, obs_dim)[flat_mask])
                branch_r_chunks.append(b_rew.reshape(-1)[flat_mask])
                branch_ns_chunks.append(b_ns.reshape(-1, obs_dim)[flat_mask])
                branch_nd_chunks.append(b_nd.reshape(-1)[flat_mask])

                per_env_trans = active_mask.sum(dim=0)
                branch_lengths.extend(per_env_trans.cpu().tolist())

                branch_total += flat_mask.sum().item()
                n_batches += 1

                if n_batches % 50 == 0:
                    print(f"    Batch {n_batches}: {branch_total:,}/{budget:,}")
                    sys.stdout.flush()

        branch_flat = (
            torch.cat(branch_s_chunks),
            torch.cat(branch_r_chunks),
            torch.cat(branch_ns_chunks),
            torch.cat(branch_nd_chunks),
        )

        avg_len = np.mean(branch_lengths) if branch_lengths else 0
        avg_v = np.mean(branch_v_values_all) if branch_v_values_all else 0
        print(f"    Pool: {branch_flat[0].shape[0]:,} trans, {len(branch_lengths)} branches, "
              f"avg len={avg_len:.1f}, avg V={avg_v:.3f}")

        diagnostics = dict(
            n_branches=len(branch_lengths),
            avg_branch_length=avg_len,
            avg_branch_v=avg_v,
            branch_v_values=branch_v_values_all,
        )
        return branch_flat, diagnostics

    print(f"\n{'=' * 70}")
    print(f"Phase 5: V-bin branch ablation")
    print(f"{'=' * 70}\n")

    tree_results = {}
    representative_N = min(args.N_values, key=lambda n: abs(n - 3))
    tree_curves_repr = {}

    seed_s, seed_r, seed_ns, seed_nd = get_seed_flat()
    seed_trans = E * T

    vbin_list = [vb for vb in args.vbins if len(eligible_states[vb]) > 0]
    skipped = [vb for vb in args.vbins if len(eligible_states[vb]) == 0]
    if skipped:
        print(f"  Skipping empty bins: {skipped}\n")

    for bi, vbin_name in enumerate(vbin_list):
        v_lo, v_hi = VBINS[vbin_name]
        n_eligible = len(eligible_states[vbin_name])
        max_branch_budget = max(args.N_values) * E * T - seed_trans
        print(f"  V-bin: {vbin_name} [{v_lo:.1f}, {min(v_hi,1.0):.1f}] "
              f"({n_eligible} eligible states, budget={max_branch_budget:,})")
        sys.stdout.flush()
        t0_vbin = time.time()

        branch_flat, diagnostics = collect_branches(vbin_name, max_branch_budget, bi)

        for N in args.N_values:
            target_trans = N * E * T
            branch_needed = target_trans - seed_trans
            branch_n = min(branch_needed, branch_flat[0].shape[0])

            tree_s = torch.cat([seed_s, branch_flat[0][:branch_n]])
            tree_r = torch.cat([seed_r, branch_flat[1][:branch_n]])
            tree_ns = torch.cat([seed_ns, branch_flat[2][:branch_n]])
            tree_nd = torch.cat([seed_nd, branch_flat[3][:branch_n]])

            print(f"    N={N} ({tree_s.shape[0]:,} trans)...", end=" ")
            sys.stdout.flush()

            t0 = time.time()
            ep_log, r_log, pk_r, pk_ep = train_td_ema(tree_s, tree_r, tree_ns, tree_nd)
            print(f"peak r={pk_r:.4f} @ ep{pk_ep} ({time.time()-t0:.1f}s)")
            tree_results[(N, vbin_name)] = dict(
                epochs=ep_log, r_log=r_log, peak_r=pk_r, peak_ep=pk_ep,
                n_trans=tree_s.shape[0], diagnostics=diagnostics,
            )
            if N == representative_N:
                tree_curves_repr[vbin_name] = dict(epochs=ep_log, r_log=r_log, peak_r=pk_r)

            del tree_s, tree_r, tree_ns, tree_nd
            torch.cuda.empty_cache()
            sys.stdout.flush()

        del branch_flat
        torch.cuda.empty_cache()
        print(f"  {vbin_name} done ({time.time() - t0_vbin:.1f}s)\n")
        sys.stdout.flush()

    del seed_s, seed_r, seed_ns, seed_nd

    # ══════════════════════════════════════════════════════════════════════
    # Phase 6: Summary
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"SUMMARY — peak Pearson r (V-bin branching)")
    print(f"{'=' * 70}")

    header = f"| {'N':>5} | {'Rollout':>8} |"
    for vb in vbin_list:
        header += f" {vb:>8} |"
    print(header)
    sep = f"|-------|----------|"
    for _ in vbin_list:
        sep += "----------|"
    print(sep)

    for N in args.N_values:
        row = f"| {N:>5} | {rollout_results[N]['peak_r']:>8.4f} |"
        for vb in vbin_list:
            if (N, vb) in tree_results:
                row += f" {tree_results[(N, vb)]['peak_r']:>8.4f} |"
            else:
                row += f"      — |"
        print(row)

    print(f"\nDelta (tree - rollout):")
    header2 = f"| {'N':>5} |"
    for vb in vbin_list:
        header2 += f" {vb:>8} |"
    print(header2)
    print(sep[7:])  # reuse separator without first column

    for N in args.N_values:
        row = f"| {N:>5} |"
        for vb in vbin_list:
            if (N, vb) in tree_results:
                delta = tree_results[(N, vb)]['peak_r'] - rollout_results[N]['peak_r']
                row += f" {delta:>+8.4f} |"
            else:
                row += f"      — |"
        print(row)

    print(f"\nBranch diagnostics (at max N={max(args.N_values)}):")
    print(f"| {'Bin':>8} | {'Eligible':>8} | {'Branches':>8} | {'Avg Len':>8} | {'Avg V':>8} |")
    print(f"|----------|----------|----------|----------|----------|")
    for vb in vbin_list:
        key = (max(args.N_values), vb)
        if key in tree_results:
            diag = tree_results[key]['diagnostics']
            print(f"| {vb:>8} | {len(eligible_states[vb]):>8} | {diag['n_branches']:>8,} | "
                  f"{diag['avg_branch_length']:>8.1f} | {diag['avg_branch_v']:>8.3f} |")

    # ══════════════════════════════════════════════════════════════════════
    # Phase 6b: Plot (2×2)
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    bin_colors = {
        "v0_20": "#3498db", "v20_40": "#2ecc71",
        "v40_60": "#f39c12", "v60_100": "#e74c3c",
        "var_lo": "#9b59b6", "var_mid": "#1abc9c", "var_hi": "#e67e22",
        "uniform": "#95a5a6",
    }

    # [0,0] Peak r vs N
    ax = axes[0, 0]
    Ns = list(args.N_values)
    roll_peaks = [rollout_results[N]['peak_r'] for N in Ns]
    ax.plot(Ns, roll_peaks, 'k--o', lw=2, label='Rollout', zorder=10)
    for vb in vbin_list:
        vb_Ns = [N for N in Ns if (N, vb) in tree_results]
        vb_peaks = [tree_results[(N, vb)]['peak_r'] for N in vb_Ns]
        if vb_Ns:
            ax.plot(vb_Ns, vb_peaks, 's-', color=bin_colors.get(vb, 'gray'),
                    lw=1.5, label=f'tree-{vb}')
    ax.set_xlabel("N (data budget in rollouts)")
    ax.set_ylabel("Peak Pearson r (vs MC16)")
    ax.set_title("Peak r vs N (V-bin branching)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # [0,1] Delta bar chart
    ax = axes[0, 1]
    bar_width = 0.15
    x = np.arange(len(Ns))
    for i, vb in enumerate(vbin_list):
        deltas = []
        for N in Ns:
            if (N, vb) in tree_results:
                deltas.append(tree_results[(N, vb)]['peak_r'] - rollout_results[N]['peak_r'])
            else:
                deltas.append(0)
        ax.bar(x + i * bar_width, deltas, bar_width, label=vb,
               color=bin_colors.get(vb, 'gray'), alpha=0.8)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel("N")
    ax.set_ylabel("Delta r (tree - rollout)")
    ax.set_title("Improvement over Rollout")
    ax.set_xticks(x + bar_width * (len(vbin_list) - 1) / 2)
    ax.set_xticklabels([str(N) for N in Ns])
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # [1,0] Training curves
    ax = axes[1, 0]
    if representative_N in rollout_results:
        rr = rollout_results[representative_N]
        ax.plot(rr['epochs'], rr['r_log'], 'k-', lw=2, alpha=0.7,
                label=f"Rollout (pk={rr['peak_r']:.3f})")
    for vb in vbin_list:
        if vb in tree_curves_repr:
            tc = tree_curves_repr[vb]
            ax.plot(tc['epochs'], tc['r_log'], '-', color=bin_colors.get(vb, 'gray'),
                    lw=1.2, alpha=0.7, label=f"{vb} (pk={tc['peak_r']:.3f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pearson r")
    ax.set_title(f"Training Curves (N={representative_N})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # [1,1] V distribution of branch points
    ax = axes[1, 1]
    for vb in vbin_list:
        key = (max(args.N_values), vb)
        if key in tree_results:
            vs = tree_results[key]['diagnostics']['branch_v_values']
            ax.hist(vs, bins=np.linspace(0, 1, 21), alpha=0.4,
                    color=bin_colors.get(vb, 'gray'), label=vb, density=True)
    # Also show overall V distribution
    ax.hist(mc16_np.ravel(), bins=np.linspace(0, 1, 21), alpha=0.3,
            color='black', label='all states', density=True, histtype='step', lw=2)
    ax.set_xlabel("V value at branch point")
    ax.set_ylabel("Density")
    ax.set_title("Branch Point V Distribution")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"V-bin Branch Ablation | {args.env_id} | gamma={args.gamma} | rs={rs}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {args.output}")

    save_path = args.output.replace('.png', '.pt')
    save_tree_results = {}
    for key, val in tree_results.items():
        save_val = {k: v for k, v in val.items() if k != 'diagnostics'}
        diag = val['diagnostics']
        save_val['diagnostics'] = dict(
            n_branches=diag['n_branches'],
            avg_branch_length=diag['avg_branch_length'],
            avg_branch_v=diag['avg_branch_v'],
        )
        save_tree_results[key] = save_val

    torch.save(dict(
        rollout_results=rollout_results,
        tree_results=save_tree_results,
        eligible_counts={vb: len(eligible_states[vb]) for vb in args.vbins},
        args=vars(args),
    ), save_path)
    print(f"Saved data to {save_path}")

    envs.close()
