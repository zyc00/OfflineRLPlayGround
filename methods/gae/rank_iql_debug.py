"""Debug: Why IQL advantage ranking fails despite SARSA = GAE in theory.

For each eval state s, K=8 actions are sampled from the policy. Each method
estimates A(s, a_k) and we compare the per-state action rankings.

Ablation table:
  Method          V(s) source        A(s,a) computation
  ──────────────  ─────────────────  ──────────────────────────────────
  MC              MC rollouts        Q_mc - V_mc  (ground truth)
  GAE             MC-supervised V    GAE(lam=0.95) from trajectories
  GAE(lam=0)      MC-supervised V    delta_0 = r + gV(s') - V(s)
  IQL             IQL joint train    Q_net(s,a) - V_net(s)
  IQL>traj        IQL's V_net        GAE(lam=0.95) from trajectories
  IQL>traj(lam=0) IQL's V_net        delta_0 = r + gV(s') - V(s)

Key diagnostic:
  - IQL>traj ~ GAE    -> Q-network destroys the ranking (V is fine)
  - IQL>traj << GAE   -> IQL's V is also broken
  - GAE vs GAE(lam=0) -> Does multi-step help?

Usage:
  python -m methods.gae.rank_iql_debug
"""

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import tyro
from scipy import stats as sp_stats

from data.offline_dataset import OfflineRLDataset
from methods.gae.gae import Critic, layer_init  # same Critic used by IQL
from methods.iql.iql import QNetwork, train_iql, compute_nstep_targets
from methods.iql.iql import Args as IQLArgs


# =====================================================================
# Config
# =====================================================================


@dataclass
class Args:
    seed: int = 1
    cuda: bool = True
    gamma: float = 0.8
    gae_lambda: float = 0.95

    # Data paths
    cache_path: str = "data/datasets/rank_cache_K8_M10_seed1.pt"
    """cached MC rollout data (from rank_mc_vs_gae.py)"""
    train_dataset_path: str = "data/datasets/pickcube_expert.pt"
    eval_dataset_path: str = "data/datasets/pickcube_expert_eval.pt"
    dataset_num_envs: int = 16

    # V(s) training (MC-supervised, for GAE)
    v_epochs: int = 100
    v_lr: float = 3e-4
    v_batch_size: int = 256
    v_weight_decay: float = 1e-4

    # IQL training
    iql_tau: float = 0.5
    """expectile (0.5 = SARSA, should match GAE in theory)"""
    iql_epochs: int = 200
    iql_lr: float = 3e-4
    iql_batch_size: int = 256
    iql_nstep: int = 10
    iql_patience: int = 100


# =====================================================================
# Helpers
# =====================================================================


@torch.no_grad()
def v_eval(v_net, states, device, batch_size=4096):
    """Evaluate V(s) in batches. Works with any nn.Module: state -> scalar."""
    v_net.eval()
    N = states.shape[0]
    out = torch.zeros(N)
    for i in range(0, N, batch_size):
        j = min(i + batch_size, N)
        out[i:j] = v_net(states[i:j].to(device)).squeeze(-1).cpu()
    return out


def mc_returns(rewards, gamma):
    """Compute discounted MC returns G_t = sum_k gamma^k r_{t+k}."""
    T = len(rewards)
    G = torch.zeros(T)
    running = 0.0
    for t in reversed(range(T)):
        running = rewards[t].item() + gamma * running
        G[t] = running
    return G


# =====================================================================
# Step A: Train V(s) by regressing on MC returns (same as GAE pipeline)
# =====================================================================


def train_v_mc(trajectories, state_dim, gamma, device, args):
    """Train V(s) by MSE regression on MC returns from trajectories."""
    all_s, all_G = [], []
    for traj in trajectories:
        all_s.append(traj["states"])
        all_G.append(mc_returns(traj["rewards"], gamma))
    all_s = torch.cat(all_s)
    all_G = torch.cat(all_G)
    N = all_s.shape[0]

    print(f"  Training on {N} (state, MC return) pairs...")
    critic = Critic("state", state_dim=state_dim).to(device)
    optimizer = torch.optim.Adam(
        critic.parameters(), lr=args.v_lr, eps=1e-5,
        weight_decay=args.v_weight_decay,
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
            print(f"    Epoch {epoch}/{args.v_epochs}: loss={total_loss / n_batch:.6f}")

    critic.eval()
    return critic


# =====================================================================
# Step B: Compute GAE advantages from trajectories (works with ANY V)
# =====================================================================


def compute_gae(v_net, trajectories, traj_map, N, K, gamma, lam, device):
    """Compute first-step GAE advantage for each (state, action) pair.

    This is the core function: it takes ANY trained V(s) network and computes
    advantages from trajectory data. No Q-network is involved.

    For each trajectory starting at (s_i, a_k):
        delta_t = r_t + gamma * V(s_{t+1}) * (1 - term_t) - V(s_t)
        A_t = delta_t + gamma * lam * (1 - done_t) * A_{t+1}   (backward)
        -> return A_0

    When lam=0: A = delta_0 = r_0 + gamma*V(s_1) - V(s_0).
    This is exactly what SARSA computes: Q(s,a) - V(s) = [r + gamma*V(s')] - V(s).

    Args:
        v_net:  Any nn.Module that maps states to scalar values.
        trajectories: List of dicts with states, next_states, rewards, etc.
        traj_map: List of (state_idx, action_idx) per trajectory.
        N, K: Number of eval states and sampled actions per state.
        gamma, lam: Discount factor and GAE lambda.

    Returns:
        (N, K) tensor of advantages, averaged over rollouts per (state, action).
    """
    # Batch-evaluate V on all trajectory states
    all_s = torch.cat([t["states"] for t in trajectories])
    all_ns = torch.cat([t["next_states"] for t in trajectories])
    all_v = v_eval(v_net, all_s, device)
    all_v_next = v_eval(v_net, all_ns, device)

    adv_sum = torch.zeros(N, K)
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

        # TD errors: delta_t = r_t + gamma * V(s_{t+1}) * (1-term) - V(s_t)
        delta = rewards + gamma * v_next * (1.0 - terminated) - v

        # GAE backward pass
        gae_val = 0.0
        advantages = torch.zeros(T)
        for t in reversed(range(T)):
            gae_val = delta[t] + gamma * lam * (1.0 - dones[t]) * gae_val
            advantages[t] = gae_val

        si, ai = traj_map[i]
        adv_sum[si, ai] += advantages[0].item()
        counts[si, ai] += 1

    return adv_sum / counts.clamp(min=1)


# =====================================================================
# Step C: IQL training and standard eval
# =====================================================================


def prepare_iql_data(train_dataset, train_trajs, rollout_trajs):
    """Flatten trajectory data into IQL training tensors.

    Combines:
      - Training dataset trajectories (actions from dataset via flat_indices)
      - Rollout trajectories (actions stored in trajectory dicts)

    Returns:
        (states, actions, rewards, next_states, terminated) flat tensors
        all_trajs: combined trajectory list (for n-step target computation)
    """
    all_s, all_a, all_r, all_ns, all_term = [], [], [], [], []

    # Training dataset: actions come from dataset, not trajectory dict
    for t in train_trajs:
        all_s.append(t["states"])
        all_a.append(train_dataset.actions[t["flat_indices"]])
        all_r.append(t["rewards"])
        all_ns.append(t["next_states"])
        all_term.append(t["terminated"])

    # Rollout trajectories: already have actions stored
    for t in rollout_trajs:
        all_s.append(t["states"])
        all_a.append(t["actions"])
        all_r.append(t["rewards"])
        all_ns.append(t["next_states"])
        all_term.append(t["terminated"])

    flat = (
        torch.cat(all_s),
        torch.cat(all_a),
        torch.cat(all_r),
        torch.cat(all_ns),
        torch.cat(all_term),
    )

    # Combined trajectory list (same order as flat data, for n-step computation)
    all_trajs = list(train_trajs) + list(rollout_trajs)

    return *flat, all_trajs


def eval_iql_advantages(q_net, v_net, eval_states, sampled_actions, device):
    """Standard IQL eval: A(s, a_k) = Q_net(s, a_k) - V_net(s).

    This uses the Q-network for action-dependent values.
    """
    N, K, _ = sampled_actions.shape
    adv = torch.zeros(N, K)

    q_net.eval()
    v_net.eval()
    with torch.no_grad():
        for i in range(0, N, 4096):
            j = min(i + 4096, N)
            s = eval_states[i:j].to(device)
            v = v_net(s).squeeze(-1)
            for k in range(K):
                a = sampled_actions[i:j, k].to(device)
                q = q_net(s, a).squeeze(-1)
                adv[i:j, k] = (q - v).cpu()
    return adv


# =====================================================================
# Ranking comparison
# =====================================================================


def ranking_metrics(adv_a, adv_b):
    """Spearman rho and top-1 agreement between two advantage vectors."""
    rho, _ = sp_stats.spearmanr(adv_a, adv_b)
    top1 = int(adv_a.argmax() == adv_b.argmax())
    return rho, top1


def compare_all(methods):
    """Print pairwise ranking comparison table.

    Args:
        methods: dict of {name: (N, K) numpy array of advantages}
                 Must include "MC" as the ground truth reference.
    """
    names = list(methods.keys())
    mc = methods["MC"]
    N, K = mc.shape

    # Skip states where MC has zero variance (no signal)
    valid = np.array([mc[i].std() > 1e-8 for i in range(N)])
    n_valid = int(valid.sum())
    print(f"  Valid states: {n_valid}/{N}\n")

    # Header
    print(f"  {'Method A':<18} {'Method B':<18} {'Spearman rho':>14} {'Top-1':>8}")
    print(f"  {'─' * 62}")

    # All pairs
    for i, n1 in enumerate(names):
        for n2 in names[i + 1 :]:
            rhos, top1s = [], []
            for s in range(N):
                if not valid[s]:
                    continue
                rho, t1 = ranking_metrics(methods[n1][s], methods[n2][s])
                rhos.append(rho)
                top1s.append(t1)
            rhos = np.array(rhos)
            top1s = np.array(top1s, dtype=float)
            print(
                f"  {n1:<18} {n2:<18} "
                f"{np.nanmean(rhos):>6.3f} (med {np.nanmedian(rhos):.3f}) "
                f"{np.mean(top1s):>6.1%}"
            )


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
    # 1. Load cached MC rollout data
    # =================================================================
    print(f"Loading cache: {args.cache_path}")
    cache = torch.load(args.cache_path, weights_only=False)

    v_mc = cache["v_mc"]                        # (N,)
    q_mc = cache["q_mc"]                        # (N, K)
    sampled_actions = cache["sampled_actions"]   # (N, K, action_dim)
    trajectories = cache["trajectories"]         # list of traj dicts
    traj_map = cache["traj_to_state_action"]     # [(state_idx, action_idx), ...]
    N = cache["N"]
    state_dim = cache["state_dim"]
    K = sampled_actions.shape[1]

    mc_adv = q_mc - v_mc.unsqueeze(1)  # (N, K)
    print(f"  {N} states, K={K} actions, {len(trajectories)} trajectories")
    print(f"  MC A(s,a): mean={mc_adv.mean():.4f}, std={mc_adv.std():.4f}")

    # =================================================================
    # 2. Train V on MC returns (GAE's approach)
    # =================================================================
    print(f"\n{'=' * 60}")
    print("STEP A: Train V(s) on MC returns (for GAE)")
    print(f"{'=' * 60}")

    train_dataset = OfflineRLDataset([args.train_dataset_path], False, False)
    train_trajs = train_dataset.extract_trajectories(
        num_envs=args.dataset_num_envs, gamma=args.gamma
    )
    n_train_trans = sum(t["states"].shape[0] for t in train_trajs)
    print(f"  Training dataset: {len(train_trajs)} trajectories, {n_train_trans} transitions")

    v_gae_net = train_v_mc(train_trajs, state_dim, args.gamma, device, args)

    # =================================================================
    # 3. Compute GAE advantages (two lambda values)
    # =================================================================
    print(f"\n{'=' * 60}")
    print("STEP B: GAE advantages (using MC-supervised V)")
    print(f"{'=' * 60}")

    adv_gae = compute_gae(
        v_gae_net, trajectories, traj_map, N, K,
        args.gamma, args.gae_lambda, device,
    )
    adv_gae_0 = compute_gae(
        v_gae_net, trajectories, traj_map, N, K,
        args.gamma, 0.0, device,
    )
    print(f"  GAE(lam={args.gae_lambda}): mean={adv_gae.mean():.4f}, std={adv_gae.std():.4f}")
    print(f"  GAE(lam=0):      mean={adv_gae_0.mean():.4f}, std={adv_gae_0.std():.4f}")

    # =================================================================
    # 4. Train IQL
    # =================================================================
    print(f"\n{'=' * 60}")
    print(f"STEP C: Train IQL (tau={args.iql_tau}, nstep={args.iql_nstep})")
    print(f"{'=' * 60}")

    states, actions, rewards, next_states, terminated, all_trajs = \
        prepare_iql_data(train_dataset, train_trajs, trajectories)
    n_total = states.shape[0]
    print(f"  IQL training data: {n_total} transitions "
          f"({n_train_trans} dataset + {n_total - n_train_trans} rollout)")

    # N-step targets
    nstep_kw = {}
    if args.iql_nstep > 1:
        print(f"  Computing {args.iql_nstep}-step targets "
              f"from {len(all_trajs)} trajectories...")
        nret, boot_s, ndisc = compute_nstep_targets(
            all_trajs, args.iql_nstep, args.gamma,
        )
        nstep_kw = dict(
            nstep_returns=nret, bootstrap_states=boot_s, nstep_discounts=ndisc,
        )
        frac = (ndisc > 0).float().mean()
        print(f"  n-step returns: mean={nret.mean():.4f}, "
              f"bootstrapped={frac:.1%}")

    iql_args = IQLArgs(
        gamma=args.gamma,
        expectile_tau=args.iql_tau,
        epochs=args.iql_epochs,
        lr=args.iql_lr,
        batch_size=args.iql_batch_size,
        patience=args.iql_patience,
        nstep=args.iql_nstep,
    )
    q_net, v_iql_net = train_iql(
        states, actions, rewards, next_states, terminated,
        device, iql_args, **nstep_kw,
    )

    # =================================================================
    # 5. Compute all advantages
    # =================================================================
    print(f"\n{'=' * 60}")
    print("STEP D: Compute advantages (all methods)")
    print(f"{'=' * 60}")

    eval_states = OfflineRLDataset([args.eval_dataset_path], False, False).state

    # IQL standard: A = Q_net(s,a) - V_net(s)
    adv_iql = eval_iql_advantages(
        q_net, v_iql_net, eval_states, sampled_actions, device,
    )
    print(f"  IQL:              mean={adv_iql.mean():.4f}, std={adv_iql.std():.4f}")

    # IQL>traj: bypass Q-net, use IQL's V on trajectories
    adv_iql_traj = compute_gae(
        v_iql_net, trajectories, traj_map, N, K,
        args.gamma, args.gae_lambda, device,
    )
    adv_iql_traj_0 = compute_gae(
        v_iql_net, trajectories, traj_map, N, K,
        args.gamma, 0.0, device,
    )
    print(f"  IQL>traj(lam=0.95): mean={adv_iql_traj.mean():.4f}, "
          f"std={adv_iql_traj.std():.4f}")
    print(f"  IQL>traj(lam=0):    mean={adv_iql_traj_0.mean():.4f}, "
          f"std={adv_iql_traj_0.std():.4f}")

    # =================================================================
    # 6. V(s) quality check
    # =================================================================
    print(f"\n{'=' * 60}")
    print("V(s) comparison on eval states")
    print(f"{'=' * 60}")

    v_gae_vals = v_eval(v_gae_net, eval_states, device)
    v_iql_vals = v_eval(v_iql_net, eval_states, device)

    r_gae, _ = sp_stats.pearsonr(v_mc.numpy(), v_gae_vals.numpy())
    r_iql, _ = sp_stats.pearsonr(v_mc.numpy(), v_iql_vals.numpy())
    r_cross, _ = sp_stats.pearsonr(v_gae_vals.numpy(), v_iql_vals.numpy())

    print(f"  MC V:  mean={v_mc.mean():.4f}, std={v_mc.std():.4f}")
    print(f"  GAE V: mean={v_gae_vals.mean():.4f}, std={v_gae_vals.std():.4f}"
          f"  (Pearson vs MC: {r_gae:.4f})")
    print(f"  IQL V: mean={v_iql_vals.mean():.4f}, std={v_iql_vals.std():.4f}"
          f"  (Pearson vs MC: {r_iql:.4f})")
    print(f"  GAE V vs IQL V: Pearson r = {r_cross:.4f}")

    # =================================================================
    # 7. Ranking comparison
    # =================================================================
    print(f"\n{'=' * 60}")
    print("RANKING COMPARISON")
    print(f"{'=' * 60}")

    methods = {
        "MC": mc_adv.numpy(),
        "GAE": adv_gae.numpy(),
        "GAE(lam=0)": adv_gae_0.numpy(),
        "IQL": adv_iql.numpy(),
        "IQL>traj": adv_iql_traj.numpy(),
        "IQL>traj(lam=0)": adv_iql_traj_0.numpy(),
    }

    compare_all(methods)

    # =================================================================
    # 8. Focused summary
    # =================================================================
    print(f"\n{'=' * 60}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'=' * 60}")
    print()
    print("  Q1: Does the Q-network hurt?")
    print("      Compare IQL vs IQL>traj (same V, different A computation)")
    print()
    print("  Q2: Is IQL's V as good as MC-supervised V?")
    print("      Compare IQL>traj vs GAE (different V, same A computation)")
    print()
    print("  Q3: Does multi-step GAE help?")
    print("      Compare GAE vs GAE(lam=0)")
    print()
