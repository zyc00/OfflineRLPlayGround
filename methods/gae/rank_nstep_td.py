"""Ablation: n-step TD advantage and simple averaging vs GAE.

Compare different ways of combining TD errors for action ranking:

  1. n-step TD:  A^(n) = sum_{l=0}^{n-1} gamma^l delta_l
                       = r_0 + gamma*r_1 + ... + gamma^{n-1}*r_{n-1} + gamma^n*V(s_n) - V(s_0)

  2. Simple average of n-step advantages (uniform weighting):
     A_avg(n_max) = (1/n_max) * sum_{n=1}^{n_max} A^(n)
                  = sum_{l=0}^{n_max-1} [(n_max-l)/n_max] * gamma^l * delta_l

  3. GAE (exponential weighting):
     A_GAE(lam) = sum_{l=0}^{T-1} (gamma*lam)^l * delta_l

  All are sample-based (computed from trajectories, averaged over M rollouts).
  No neural network regression involved.

  Weight comparison for delta_l:
    n-step TD(n):     gamma^l           if l < n, else 0
    Simple avg(n):    (n-l)/n * gamma^l if l < n, else 0
    GAE(lam):         (gamma*lam)^l     (exponential decay)

Usage:
  python -m methods.gae.rank_nstep_td
"""

import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tyro
from scipy import stats as sp_stats

from data.offline_dataset import OfflineRLDataset
from methods.gae.gae import Critic
from methods.gae.rank_iql_debug import v_eval, mc_returns, ranking_metrics


# =====================================================================
# Config
# =====================================================================


@dataclass
class Args:
    seed: int = 1
    cuda: bool = True
    gamma: float = 0.8
    gae_lambda: float = 0.95

    cache_path: str = "data/datasets/rank_cache_K8_M10_seed1.pt"
    train_dataset_path: str = "data/datasets/pickcube_expert.pt"
    eval_dataset_path: str = "data/datasets/pickcube_expert_eval.pt"
    dataset_num_envs: int = 16

    # V(s) training
    v_epochs: int = 100
    v_lr: float = 3e-4
    v_batch_size: int = 256

    # n-step values to test
    nsteps: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 50)
    """n values for n-step TD and simple average"""


# =====================================================================
# V(s) training on MC returns
# =====================================================================


def train_v_mc(trajectories, state_dim, gamma, device, args):
    """Train V(s) by MSE regression on MC returns."""
    all_s, all_G = [], []
    for traj in trajectories:
        all_s.append(traj["states"])
        all_G.append(mc_returns(traj["rewards"], gamma))
    all_s = torch.cat(all_s)
    all_G = torch.cat(all_G)
    N = all_s.shape[0]

    print(f"  Training V(s) on {N} (state, MC return) pairs...")
    critic = Critic("state", state_dim=state_dim).to(device)
    optimizer = torch.optim.Adam(
        critic.parameters(), lr=args.v_lr, eps=1e-5, weight_decay=1e-4,
    )

    for epoch in range(1, args.v_epochs + 1):
        idx = torch.randperm(N)
        total_loss, n_batch = 0.0, 0
        critic.train()
        for start in range(0, N, args.v_batch_size):
            batch = idx[start : start + args.v_batch_size]
            pred = critic(all_s[batch].to(device)).squeeze(-1)
            loss = 0.5 * ((pred - all_G[batch].to(device)) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        if epoch == 1 or epoch % 20 == 0:
            print(f"    Epoch {epoch}/{args.v_epochs}: "
                  f"loss={total_loss / n_batch:.6f}")

    critic.eval()
    return critic


# =====================================================================
# Compute n-step TD and simple-average advantages from trajectories
# =====================================================================


def compute_nstep_advantages(v_net, trajectories, traj_map, N, K,
                             gamma, nsteps, gae_lambda, device):
    """Compute multiple advantage estimates from trajectories.

    For each trajectory, pre-compute all TD errors delta_l, then combine
    them in different ways.

    Args:
        nsteps: list of n values to test

    Returns dict of {method_name: (N, K) tensor}:
        "TD(n)":     n-step TD advantage (first-step only)
        "Avg(n)":    simple average of 1..n step advantages
        "GAE(lam)":  standard GAE
        "MC":        full MC return - V(s_0) (= n-step TD with n=T)
    """
    # Batch-evaluate V on all states
    all_s = torch.cat([t["states"] for t in trajectories])
    all_ns = torch.cat([t["next_states"] for t in trajectories])
    all_v = v_eval(v_net, all_s, device)
    all_v_next = v_eval(v_net, all_ns, device)

    # Initialize accumulators for each method
    method_names = []
    for n in nsteps:
        method_names.append(f"TD({n})")
        method_names.append(f"Avg({n})")
    method_names.append(f"GAE({gae_lambda})")
    method_names.append("MC-V")

    adv_sums = {m: torch.zeros(N, K) for m in method_names}
    counts = torch.zeros(N, K)

    offset = 0
    for i, traj in enumerate(trajectories):
        T = traj["states"].shape[0]
        v = all_v[offset : offset + T]
        v_next = all_v_next[offset : offset + T]
        rewards = traj["rewards"]
        terminated = traj["terminated"]
        dones = traj["dones"]
        offset += T

        # TD errors: delta_l = r_l + gamma * V(s_{l+1}) * (1-term) - V(s_l)
        delta = rewards + gamma * v_next * (1.0 - terminated) - v

        si, ai = traj_map[i]

        # --- n-step TD: A^(n) = sum_{l=0}^{n-1} gamma^l * delta_l ---
        # Pre-compute cumulative: cum[l] = sum_{j=0}^{l} gamma^j * delta_j
        gamma_powers = gamma ** torch.arange(T, dtype=torch.float32)
        weighted_delta = gamma_powers * delta
        cum = torch.cumsum(weighted_delta, dim=0)  # cum[l] = A^(l+1)

        for n in nsteps:
            n_eff = min(n, T)

            # TD(n): A^(n) = cum[n-1]
            td_n = cum[n_eff - 1].item()
            adv_sums[f"TD({n})"][si, ai] += td_n

            # Avg(n): (1/n) * sum_{k=1}^{n} A^(k) = (1/n) * sum_{k=0}^{n-1} cum[k]
            avg_n = cum[:n_eff].sum().item() / n_eff
            adv_sums[f"Avg({n})"][si, ai] += avg_n

        # --- GAE: sum (gamma*lam)^l * delta_l ---
        gae_val = 0.0
        for t in reversed(range(T)):
            gae_val = delta[t] + gamma * gae_lambda * (1.0 - dones[t]) * gae_val
        adv_sums[f"GAE({gae_lambda})"][si, ai] += gae_val.item()

        # --- MC - V(s_0): full trajectory return minus V ---
        mc_ret = 0.0
        for t in reversed(range(T)):
            mc_ret = rewards[t].item() + gamma * mc_ret
        adv_sums["MC-V"][si, ai] += mc_ret - v[0].item()

        counts[si, ai] += 1

    counts = counts.clamp(min=1)
    return {m: adv_sums[m] / counts for m in method_names}


# =====================================================================
# Ranking comparison
# =====================================================================


def spearman_vs_mc(mc_adv, other_adv, valid):
    """Per-state Spearman rho and top-1 agreement vs MC."""
    rhos, top1s = [], []
    N = mc_adv.shape[0]
    for i in range(N):
        if not valid[i]:
            continue
        rho, t1 = ranking_metrics(mc_adv[i], other_adv[i])
        rhos.append(rho)
        top1s.append(t1)
    rhos = np.array(rhos)
    top1s = np.array(top1s, dtype=float)
    return np.nanmean(rhos), np.nanmedian(rhos), np.mean(top1s)


# =====================================================================
# Main
# =====================================================================


if __name__ == "__main__":
    args = tyro.cli(Args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # =================================================================
    # 1. Load data
    # =================================================================
    print(f"Loading cache: {args.cache_path}")
    cache = torch.load(args.cache_path, weights_only=False)

    v_mc = cache["v_mc"]
    q_mc = cache["q_mc"]
    sampled_actions = cache["sampled_actions"]
    trajectories = cache["trajectories"]
    traj_map = cache["traj_to_state_action"]
    N = cache["N"]
    state_dim = cache["state_dim"]
    K = sampled_actions.shape[1]

    mc_adv = (q_mc - v_mc.unsqueeze(1)).numpy()
    valid = np.array([mc_adv[i].std() > 1e-8 for i in range(N)])
    n_valid = int(valid.sum())

    # Trajectory length stats
    traj_lens = [t["states"].shape[0] for t in trajectories]
    n_terminated = sum(1 for t in trajectories if t["terminated"][-1] > 0.5)
    n_truncated = len(trajectories) - n_terminated
    print(f"  {N} states, K={K}, {len(trajectories)} trajectories, "
          f"{n_valid} valid")
    print(f"  Trajectory lengths: min={min(traj_lens)}, max={max(traj_lens)}, "
          f"mean={np.mean(traj_lens):.1f}")
    print(f"  Terminated: {n_terminated}, Truncated: {n_truncated}")

    # =================================================================
    # 2. Train V(s) on MC returns
    # =================================================================
    print(f"\n{'=' * 60}")
    print("Train V(s) on MC returns")
    print(f"{'=' * 60}")

    train_dataset = OfflineRLDataset([args.train_dataset_path], False, False)
    train_trajs = train_dataset.extract_trajectories(
        num_envs=args.dataset_num_envs, gamma=args.gamma
    )
    del train_dataset

    v_net = train_v_mc(train_trajs, state_dim, args.gamma, device, args)
    del train_trajs

    eval_states = OfflineRLDataset([args.eval_dataset_path], False, False).state
    v_pred = v_eval(v_net, eval_states, device)
    r_v, _ = sp_stats.pearsonr(v_mc.numpy(), v_pred.numpy())
    print(f"  V quality: Pearson r={r_v:.4f}")

    # =================================================================
    # 3. Compute all advantage estimates
    # =================================================================
    print(f"\n{'=' * 60}")
    print(f"Compute advantages: n-step TD, simple avg, GAE")
    print(f"  n values: {args.nsteps}")
    print(f"{'=' * 60}")

    all_advs = compute_nstep_advantages(
        v_net, trajectories, traj_map, N, K,
        args.gamma, list(args.nsteps), args.gae_lambda, device,
    )

    # --- Diagnostic: per-trajectory TD(T) vs MC-V ---
    max_n = max(args.nsteps)
    td_max = all_advs[f"TD({max_n})"]
    mc_v = all_advs["MC-V"]
    diff = (td_max - mc_v).abs()
    print(f"\n  Diagnostic: TD({max_n}) vs MC-V")
    print(f"    Max abs diff:  {diff.max():.6f}")
    print(f"    Mean abs diff: {diff.mean():.6f}")
    if diff.max() > 1e-4:
        worst = diff.argmax()
        si, ai = worst // K, worst % K
        print(f"    Worst (state={si}, action={ai}): "
              f"TD={td_max[si, ai]:.6f}, MC-V={mc_v[si, ai]:.6f}")

    # =================================================================
    # 4. Rollout averaging ablation: MC, TD(50), GAE with M rollouts
    # =================================================================
    # Count rollouts per (state, action)
    rollout_counts = torch.zeros(N, K, dtype=torch.long)
    for si, ai in traj_map:
        rollout_counts[si, ai] += 1
    max_rollouts = int(rollout_counts.max())
    print(f"\n{'=' * 60}")
    print(f"Rollout averaging ablation ({max_rollouts} rollouts per (s,a))")
    print(f"{'=' * 60}")

    if max_rollouts > 1:
        # Pre-compute per-trajectory: MC return, TD(50) advantage, GAE advantage
        # Then group by (si, ai) for subset averaging
        all_s_flat = torch.cat([t["states"] for t in trajectories])
        all_ns_flat = torch.cat([t["next_states"] for t in trajectories])
        all_v_flat = v_eval(v_net, all_s_flat, device)
        all_vnext_flat = v_eval(v_net, all_ns_flat, device)

        mc_per_sa = [[[] for _ in range(K)] for _ in range(N)]
        gae_per_sa = [[[] for _ in range(K)] for _ in range(N)]
        td_nsteps_ablation = [5, 10, 20, 50]
        td_per_sa = {n: [[[] for _ in range(K)] for _ in range(N)]
                     for n in td_nsteps_ablation}

        offset = 0
        for i, traj in enumerate(trajectories):
            si, ai = traj_map[i]
            T = traj["states"].shape[0]
            rewards = traj["rewards"]
            terminated = traj["terminated"]
            dones = traj["dones"]
            v = all_v_flat[offset : offset + T]
            v_next = all_vnext_flat[offset : offset + T]
            offset += T

            # MC return
            ret = 0.0
            for t in reversed(range(T)):
                ret = rewards[t].item() + args.gamma * ret
            mc_per_sa[si][ai].append(ret)

            # TD errors (use dones to zero bootstrap for truncated too)
            delta = rewards + args.gamma * v_next * (1.0 - dones) - v

            # TD(n) advantages via cumulative sum
            gamma_powers = args.gamma ** torch.arange(T, dtype=torch.float32)
            cum = torch.cumsum(gamma_powers * delta, dim=0)
            for n in td_nsteps_ablation:
                n_eff = min(n, T)
                td_per_sa[n][si][ai].append(cum[n_eff - 1].item())

            # GAE advantage
            gae_val = 0.0
            for t in reversed(range(T)):
                gae_val = delta[t] + args.gamma * args.gae_lambda * (1.0 - dones[t]) * gae_val
            gae_per_sa[si][ai].append(gae_val.item())

        # Test different numbers of rollouts
        ms_to_test = sorted(set(
            [1, 2, 4, 8, max_rollouts] + [max_rollouts // 2]
        ))
        ms_to_test = [m for m in ms_to_test if 1 <= m <= max_rollouts]

        # Build method list: MC, TD(5), TD(10), TD(20), TD(50), GAE
        method_list = [("MC", mc_per_sa)]
        for n in td_nsteps_ablation:
            method_list.append((f"TD({n})", td_per_sa[n]))
        method_list.append(("GAE", gae_per_sa))

        # Header
        header_names = [name for name, _ in method_list]
        print(f"\n  {'M':<4}", end="")
        for name in header_names:
            print(f" | {name:>8}", end="")
        print()
        print(f"  {'─' * (6 + 11 * len(header_names))}")

        def pearson_vs_mc(mc_adv, other_adv, valid):
            """Per-state Pearson r vs MC."""
            rs = []
            for i in range(mc_adv.shape[0]):
                if not valid[i]:
                    continue
                r, _ = sp_stats.pearsonr(mc_adv[i], other_adv[i])
                rs.append(r)
            return np.nanmean(rs)

        # Compute advantages for each (method, M) combination
        adv_cache = {}
        for m in ms_to_test:
            for name, per_sa in method_list:
                adv_m = np.zeros((N, K))
                for si in range(N):
                    for ai in range(K):
                        vals = per_sa[si][ai][:m]
                        adv_m[si, ai] = np.mean(vals) if vals else 0.0
                if name == "MC":
                    adv_m = adv_m - v_mc.numpy()[:, None]
                adv_cache[(name, m)] = adv_m

        # Table 1: Spearman rho
        print("  [Spearman rho]")
        for m in ms_to_test:
            row = f"  {m:<4}"
            for name, _ in method_list:
                rho_mean, _, _ = spearman_vs_mc(mc_adv, adv_cache[(name, m)], valid)
                row += f" | {rho_mean:>8.3f}"
            print(row)

        # Table 2: Pearson r
        print(f"\n  {'M':<4}", end="")
        for name in header_names:
            print(f" | {name:>8}", end="")
        print()
        print(f"  {'─' * (6 + 11 * len(header_names))}")
        print("  [Pearson r]")
        for m in ms_to_test:
            row = f"  {m:<4}"
            for name, _ in method_list:
                r_mean = pearson_vs_mc(mc_adv, adv_cache[(name, m)], valid)
                row += f" | {r_mean:>8.3f}"
            print(row)

    # =================================================================
    # 5. Ranking comparison
    # =================================================================
    print(f"\n{'=' * 60}")
    print("RANKING vs MC (Spearman rho)")
    print(f"  Valid states: {n_valid}/{N}")
    print(f"{'=' * 60}")

    print(f"\n  {'Method':<18} {'rho mean':>10} {'rho med':>10} {'top-1':>8}")
    print(f"  {'─' * 50}")

    results = {}
    for name, adv in sorted(all_advs.items()):
        rho_mean, rho_med, top1 = spearman_vs_mc(mc_adv, adv.numpy(), valid)
        results[name] = (rho_mean, rho_med, top1)
        print(f"  {name:<18} {rho_mean:>10.3f} {rho_med:>10.3f} {top1:>7.1%}")

    # =================================================================
    # 6. Summary: TD(n) vs Avg(n) vs GAE
    # =================================================================
    print(f"\n{'=' * 60}")
    print("SUMMARY: n-step comparison")
    print(f"{'=' * 60}")

    print(f"\n  {'n':<6} {'TD(n) rho':>12} {'Avg(n) rho':>12} {'GAE rho':>12}")
    print(f"  {'─' * 44}")
    gae_rho = results[f"GAE({args.gae_lambda})"][0]
    for n in args.nsteps:
        td_rho = results[f"TD({n})"][0]
        avg_rho = results[f"Avg({n})"][0]
        print(f"  {n:<6} {td_rho:>12.3f} {avg_rho:>12.3f} {gae_rho:>12.3f}")

    print(f"\n  MC-V (n=T):  rho = {results['MC-V'][0]:.3f}")
    print(f"  GAE({args.gae_lambda}):   rho = {gae_rho:.3f}")

    # =================================================================
    # 7. Plot
    # =================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("n-step TD vs Simple Average vs GAE: Action Ranking Quality",
                 fontsize=13)

    ns = list(args.nsteps)

    # --- Plot 1: rho vs n ---
    ax = axes[0]
    td_rhos = [results[f"TD({n})"][0] for n in ns]
    avg_rhos = [results[f"Avg({n})"][0] for n in ns]
    ax.plot(ns, td_rhos, "o-", label="TD(n)", markersize=6)
    ax.plot(ns, avg_rhos, "s-", label="Avg(n)", markersize=6)
    ax.axhline(gae_rho, color="red", ls="--", lw=1.5,
               label=f"GAE(λ={args.gae_lambda})")
    ax.axhline(results["MC-V"][0], color="green", ls=":",
               lw=1.5, label="MC-V (n=T)")
    ax.set_xlabel("n (number of steps)")
    ax.set_ylabel("Spearman ρ vs MC")
    ax.set_title("Ranking Quality vs n")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

    # --- Plot 2: weight profiles ---
    ax = axes[1]
    T_example = int(np.mean(traj_lens))
    ls = np.arange(T_example)
    gamma = args.gamma

    # GAE weights
    gae_weights = (gamma * args.gae_lambda) ** ls
    ax.plot(ls, gae_weights / gae_weights.sum(),
            label=f"GAE(λ={args.gae_lambda})", lw=2)

    # n-step TD weights for a few n values
    for n in [1, 5, 20]:
        if n > T_example:
            continue
        w = np.zeros(T_example)
        w[:n] = gamma ** np.arange(n)
        ax.plot(ls, w / w.sum(), label=f"TD({n})", ls="--")

    # Simple average weights for a few n values
    for n in [5, 20]:
        if n > T_example:
            continue
        w = np.zeros(T_example)
        for l in range(n):
            w[l] = (n - l) / n * gamma ** l
        ax.plot(ls, w / w.sum(), label=f"Avg({n})", ls=":")

    ax.set_xlabel("Step l (distance from t=0)")
    ax.set_ylabel("Normalized weight on δ_l")
    ax.set_title("Weight Profiles on TD Errors")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, min(30, T_example))

    # --- Plot 3: top-1 agreement vs n ---
    ax = axes[2]
    td_top1 = [results[f"TD({n})"][2] for n in ns]
    avg_top1 = [results[f"Avg({n})"][2] for n in ns]
    ax.plot(ns, td_top1, "o-", label="TD(n)", markersize=6)
    ax.plot(ns, avg_top1, "s-", label="Avg(n)", markersize=6)
    ax.axhline(results[f"GAE({args.gae_lambda})"][2], color="red", ls="--",
               lw=1.5, label=f"GAE(λ={args.gae_lambda})")
    ax.axhline(results["MC-V"][2], color="green", ls=":", lw=1.5,
               label="MC-V (n=T)")
    ax.set_xlabel("n (number of steps)")
    ax.set_ylabel("Top-1 Agreement vs MC")
    ax.set_title("Top-1 Agreement vs n")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

    fig.tight_layout()
    save_path = "data/datasets/rank_nstep_td.png"
    fig.savefig(save_path, dpi=150)
    print(f"\nSaved figure to {save_path}")
