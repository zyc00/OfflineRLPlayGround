"""Ablation: Does a stronger neural network fix IQL action ranking?

Tests whether increasing network width (hidden_dim) reduces V(s) per-state
error enough to make 1-step advantages usable, and whether a larger Q-network
can resolve within-state action differences.

For each hidden_dim in [256, 512, 1024]:
  1. Train V(s) on MC returns -> measure per-state V error
  2. GAE(lam=0): does better V fix 1-step ranking?
  3. GAE(lam=0.95): does better V improve multi-step ranking?
  4. IQL: does larger Q-net resolve within-state action differences?
  5. IQL>traj: is IQL's larger V better for trajectory-based ranking?

Usage:
  python -m methods.gae.rank_network_size
"""

import copy
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tyro
from scipy import stats as sp_stats

from data.offline_dataset import OfflineRLDataset
from methods.gae.gae import layer_init
from methods.iql.iql import compute_nstep_targets
from methods.gae.rank_iql_debug import (
    v_eval, mc_returns, compute_gae, prepare_iql_data, ranking_metrics,
)


# =====================================================================
# Networks with configurable width
# =====================================================================


class VNet(nn.Module):
    """V(s) network: 3-layer Tanh MLP with configurable hidden_dim."""

    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1)),
        )

    def forward(self, x):
        return self.net(x)


class QNet(nn.Module):
    """Q(s, a) network: 3-layer Tanh MLP with configurable hidden_dim."""

    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim + action_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1)),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


# =====================================================================
# V(s) training on MC returns
# =====================================================================


def train_v_mc(trajectories, state_dim, hidden_dim, gamma, device,
               epochs=100, lr=3e-4, batch_size=256):
    """Train V(s) on MC returns with configurable network width."""
    all_s, all_G = [], []
    for traj in trajectories:
        all_s.append(traj["states"])
        all_G.append(mc_returns(traj["rewards"], gamma))
    all_s = torch.cat(all_s)
    all_G = torch.cat(all_G)
    N = all_s.shape[0]

    v_net = VNet(state_dim, hidden_dim).to(device)
    n_params = sum(p.numel() for p in v_net.parameters())
    print(f"    V({hidden_dim}): {n_params:,} params, {N:,} training samples")

    optimizer = torch.optim.Adam(
        v_net.parameters(), lr=lr, eps=1e-5, weight_decay=1e-4,
    )

    for epoch in range(1, epochs + 1):
        idx = torch.randperm(N)
        total_loss, n_batch = 0.0, 0
        v_net.train()
        for start in range(0, N, batch_size):
            bi = idx[start : start + batch_size]
            pred = v_net(all_s[bi].to(device)).squeeze(-1)
            loss = 0.5 * ((pred - all_G[bi].to(device)) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(v_net.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"      Epoch {epoch}/{epochs}: loss={total_loss / n_batch:.6f}")

    v_net.eval()
    return v_net


# =====================================================================
# IQL training with configurable network size
# =====================================================================


def train_iql_sized(states, actions, rewards, next_states, terminated,
                    state_dim, action_dim, hidden_dim, device,
                    gamma=0.8, tau=0.5, epochs=200, lr=3e-4, batch_size=256,
                    tau_polyak=0.005, grad_clip=0.5,
                    nstep_returns=None, bootstrap_states=None,
                    nstep_discounts=None):
    """Train IQL Q(s,a) and V(s) with configurable hidden_dim."""
    q_net = QNet(state_dim, action_dim, hidden_dim).to(device)
    q_target = copy.deepcopy(q_net)
    v_net = VNet(state_dim, hidden_dim).to(device)

    q_params = sum(p.numel() for p in q_net.parameters())
    v_params = sum(p.numel() for p in v_net.parameters())
    print(f"    Q({hidden_dim}): {q_params:,} params, "
          f"V({hidden_dim}): {v_params:,} params")

    q_opt = torch.optim.Adam(
        q_net.parameters(), lr=lr, eps=1e-5, weight_decay=1e-4,
    )
    v_opt = torch.optim.Adam(
        v_net.parameters(), lr=lr, eps=1e-5, weight_decay=1e-4,
    )
    q_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        q_opt, T_max=epochs, eta_min=1e-5,
    )
    v_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        v_opt, T_max=epochs, eta_min=1e-5,
    )

    use_nstep = nstep_returns is not None
    N = states.shape[0]

    # Track per-epoch metrics
    history = {
        "q_loss": [], "v_loss": [],
        "q_grad_norm": [], "v_grad_norm": [],
    }

    for epoch in range(1, epochs + 1):
        q_net.train()
        v_net.train()
        idx = torch.randperm(N)
        epoch_q, epoch_v, n_batch = 0.0, 0.0, 0
        epoch_q_grad, epoch_v_grad = 0.0, 0.0

        for start in range(0, N, batch_size):
            bi = idx[start : start + batch_size]
            s = states[bi].to(device)
            a = actions[bi].to(device)

            # Q-loss: TD target using V
            with torch.no_grad():
                if use_nstep:
                    boot_v = v_net(bootstrap_states[bi].to(device)).squeeze(-1)
                    target = (nstep_returns[bi].to(device)
                              + nstep_discounts[bi].to(device) * boot_v)
                else:
                    v_next = v_net(next_states[bi].to(device)).squeeze(-1)
                    target = (rewards[bi].to(device)
                              + gamma * v_next * (1 - terminated[bi].to(device)))

            q_pred = q_net(s, a).squeeze(-1)
            q_loss = 0.5 * ((q_pred - target) ** 2).mean()

            q_opt.zero_grad()
            q_loss.backward()
            # Record grad norm BEFORE clipping
            q_grad = torch.nn.utils.clip_grad_norm_(
                q_net.parameters(), grad_clip,
            ).item()
            q_opt.step()

            # V-loss: expectile regression against target Q
            with torch.no_grad():
                q_tgt_val = q_target(s, a).squeeze(-1)
            v_pred = v_net(s).squeeze(-1)
            diff = q_tgt_val - v_pred
            weight = torch.where(diff > 0, tau, 1.0 - tau)
            v_loss = (weight * diff ** 2).mean()

            v_opt.zero_grad()
            v_loss.backward()
            # Record grad norm BEFORE clipping
            v_grad = torch.nn.utils.clip_grad_norm_(
                v_net.parameters(), grad_clip,
            ).item()
            v_opt.step()

            # Polyak update
            with torch.no_grad():
                for p, pt in zip(q_net.parameters(), q_target.parameters()):
                    pt.data.mul_(1 - tau_polyak).add_(p.data, alpha=tau_polyak)

            epoch_q += q_loss.item()
            epoch_v += v_loss.item()
            epoch_q_grad += q_grad
            epoch_v_grad += v_grad
            n_batch += 1

        q_sched.step()
        v_sched.step()

        history["q_loss"].append(epoch_q / n_batch)
        history["v_loss"].append(epoch_v / n_batch)
        history["q_grad_norm"].append(epoch_q_grad / n_batch)
        history["v_grad_norm"].append(epoch_v_grad / n_batch)

        if epoch == 1 or epoch % 50 == 0:
            print(f"      Epoch {epoch}/{epochs}: "
                  f"q_loss={epoch_q / n_batch:.6f}, "
                  f"v_loss={epoch_v / n_batch:.6f}, "
                  f"q_grad={epoch_q_grad / n_batch:.4f}, "
                  f"v_grad={epoch_v_grad / n_batch:.4f}")

    q_net.eval()
    v_net.eval()
    return q_net, v_net, history


# =====================================================================
# Evaluation helpers
# =====================================================================


def eval_iql_adv(q_net, v_net, eval_states, sampled_actions, device):
    """Standard IQL: A = Q(s,a) - V(s)."""
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


def spearman_vs_mc(mc_adv, other_adv, valid):
    """Mean per-state Spearman rho against MC (only valid states)."""
    rhos = []
    for i in range(mc_adv.shape[0]):
        if not valid[i]:
            continue
        rho, _ = sp_stats.spearmanr(mc_adv[i], other_adv[i])
        rhos.append(rho)
    rhos = np.array(rhos)
    return np.nanmean(rhos), np.nanmedian(rhos)


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

    hidden_dims: tuple[int, ...] = (256, 512, 1024)
    """network widths to test"""

    v_epochs: int = 100
    iql_tau: float = 0.5
    iql_epochs: int = 200
    iql_nstep: int = 10


# =====================================================================
# Main
# =====================================================================


if __name__ == "__main__":
    args = tyro.cli(Args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ── Load data ─────────────────────────────────────────────────────
    print("Loading cached MC rollout data...")
    cache = torch.load(args.cache_path, weights_only=False)
    v_mc = cache["v_mc"]
    q_mc = cache["q_mc"]
    sampled_actions = cache["sampled_actions"]
    trajectories = cache["trajectories"]
    traj_map = cache["traj_to_state_action"]
    N = cache["N"]
    state_dim = cache["state_dim"]
    K = sampled_actions.shape[1]
    action_dim = sampled_actions.shape[2]

    mc_adv = (q_mc - v_mc.unsqueeze(1)).numpy()
    valid = np.array([mc_adv[i].std() > 1e-8 for i in range(N)])
    n_valid = int(valid.sum())
    print(f"  {N} states, K={K}, {len(trajectories)} trajectories, "
          f"{n_valid} valid")

    eval_states = OfflineRLDataset([args.eval_dataset_path], False, False).state

    # Prepare IQL data (train dataset + rollout trajectories)
    train_dataset = OfflineRLDataset([args.train_dataset_path], False, False)
    train_trajs = train_dataset.extract_trajectories(
        num_envs=args.dataset_num_envs, gamma=args.gamma,
    )
    states, actions, rewards, next_states, terminated, all_trajs = \
        prepare_iql_data(train_dataset, train_trajs, trajectories)
    n_iql = states.shape[0]
    print(f"  IQL data: {n_iql:,} transitions")

    # N-step targets (computed once, shared across sizes)
    nstep_kw = {}
    if args.iql_nstep > 1:
        print(f"  Computing {args.iql_nstep}-step targets...")
        nret, boot_s, ndisc = compute_nstep_targets(
            all_trajs, args.iql_nstep, args.gamma,
        )
        nstep_kw = dict(
            nstep_returns=nret, bootstrap_states=boot_s,
            nstep_discounts=ndisc,
        )
        print(f"    done. bootstrapped={((ndisc > 0).float().mean()):.1%}")

    # ── Run ablation ──────────────────────────────────────────────────
    # Collect results: {hidden_dim: {method_name: (mean_rho, med_rho)}}
    summary = {}
    all_histories = {}  # {hidden_dim: history dict}

    for hdim in args.hidden_dims:
        print(f"\n{'=' * 60}")
        print(f"HIDDEN DIM = {hdim}")
        print(f"{'=' * 60}")

        # Reset seed for fair comparison
        torch.manual_seed(args.seed)

        # ── Train V on MC returns ─────────────────────────────────────
        print(f"\n  [V training] MC return supervision:")
        v_net = train_v_mc(
            train_trajs, state_dim, hdim, args.gamma, device,
            epochs=args.v_epochs,
        )

        # V error
        v_pred = v_eval(v_net, eval_states, device)
        v_mc_np = v_mc.numpy()
        v_pred_np = v_pred.numpy()
        r_val, _ = sp_stats.pearsonr(v_mc_np, v_pred_np)
        mae = np.mean(np.abs(v_pred_np - v_mc_np))
        print(f"    V error: Pearson r={r_val:.4f}, MAE={mae:.4f}")

        # GAE advantages
        adv_gae = compute_gae(
            v_net, trajectories, traj_map, N, K,
            args.gamma, args.gae_lambda, device,
        )
        adv_gae_0 = compute_gae(
            v_net, trajectories, traj_map, N, K,
            args.gamma, 0.0, device,
        )

        rho_gae, med_gae = spearman_vs_mc(mc_adv, adv_gae.numpy(), valid)
        rho_gae0, med_gae0 = spearman_vs_mc(mc_adv, adv_gae_0.numpy(), valid)

        # ── Train IQL ─────────────────────────────────────────────────
        print(f"\n  [IQL training] tau={args.iql_tau}, nstep={args.iql_nstep}:")
        torch.manual_seed(args.seed)
        q_net, v_iql, hist = train_iql_sized(
            states, actions, rewards, next_states, terminated,
            state_dim, action_dim, hdim, device,
            gamma=args.gamma, tau=args.iql_tau, epochs=args.iql_epochs,
            **nstep_kw,
        )
        all_histories[hdim] = hist

        # IQL V error
        v_iql_pred = v_eval(v_iql, eval_states, device)
        r_iql, _ = sp_stats.pearsonr(v_mc_np, v_iql_pred.numpy())
        mae_iql = np.mean(np.abs(v_iql_pred.numpy() - v_mc_np))
        print(f"    IQL V error: Pearson r={r_iql:.4f}, MAE={mae_iql:.4f}")

        # IQL advantages
        adv_iql = eval_iql_adv(
            q_net, v_iql, eval_states, sampled_actions, device,
        )
        adv_iql_traj = compute_gae(
            v_iql, trajectories, traj_map, N, K,
            args.gamma, args.gae_lambda, device,
        )
        adv_iql_0 = compute_gae(
            v_iql, trajectories, traj_map, N, K,
            args.gamma, 0.0, device,
        )

        rho_iql, med_iql = spearman_vs_mc(mc_adv, adv_iql.numpy(), valid)
        rho_itraj, med_itraj = spearman_vs_mc(mc_adv, adv_iql_traj.numpy(), valid)
        rho_itraj0, med_itraj0 = spearman_vs_mc(mc_adv, adv_iql_0.numpy(), valid)

        # ── Print results for this size ───────────────────────────────
        print(f"\n  Results (hidden_dim={hdim}):")
        print(f"    {'Method':<22} {'rho mean':>10} {'rho med':>10}")
        print(f"    {'─' * 44}")
        print(f"    {'GAE(lam=0.95)':<22} {rho_gae:>10.3f} {med_gae:>10.3f}")
        print(f"    {'GAE(lam=0)':<22} {rho_gae0:>10.3f} {med_gae0:>10.3f}")
        print(f"    {'IQL (Q-net)':<22} {rho_iql:>10.3f} {med_iql:>10.3f}")
        print(f"    {'IQL>traj(lam=0.95)':<22} {rho_itraj:>10.3f} {med_itraj:>10.3f}")
        print(f"    {'IQL>traj(lam=0)':<22} {rho_itraj0:>10.3f} {med_itraj0:>10.3f}")

        summary[hdim] = {
            "V_r": r_val, "V_mae": mae,
            "IQL_V_r": r_iql, "IQL_V_mae": mae_iql,
            "GAE": rho_gae, "GAE(lam=0)": rho_gae0,
            "IQL": rho_iql,
            "IQL>traj": rho_itraj, "IQL>traj(lam=0)": rho_itraj0,
        }

        del v_net, q_net, v_iql
        torch.cuda.empty_cache()

    # ── Summary table ─────────────────────────────────────────────────
    dims = list(summary.keys())

    print(f"\n{'=' * 60}")
    print("SUMMARY: Network Size vs Ranking Quality")
    print(f"{'=' * 60}")

    print(f"\n  V(s) quality (Pearson r vs MC / MAE):")
    print(f"    {'hidden_dim':<12} {'MC-sup V':>16} {'IQL V':>16}")
    print(f"    {'─' * 46}")
    for d in dims:
        s = summary[d]
        print(f"    {d:<12} {s['V_r']:.4f} / {s['V_mae']:.4f}"
              f"  {s['IQL_V_r']:.4f} / {s['IQL_V_mae']:.4f}")

    print(f"\n  Action ranking (Spearman rho vs MC):")
    print(f"    {'hidden_dim':<12} {'GAE':>8} {'GAE(0)':>8} "
          f"{'IQL':>8} {'IQL>t':>8} {'IQL>t(0)':>8}")
    print(f"    {'─' * 58}")
    for d in dims:
        s = summary[d]
        print(f"    {d:<12} {s['GAE']:>8.3f} {s['GAE(lam=0)']:>8.3f} "
              f"{s['IQL']:>8.3f} {s['IQL>traj']:>8.3f} "
              f"{s['IQL>traj(lam=0)']:>8.3f}")

    # ── Plot gradient norms ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("IQL Training: Loss and Gradient L2 Norm by Network Size", fontsize=14)

    for hdim, hist in all_histories.items():
        epochs_x = np.arange(1, len(hist["q_loss"]) + 1)
        label = f"dim={hdim}"
        axes[0, 0].plot(epochs_x, hist["q_loss"], label=label)
        axes[0, 1].plot(epochs_x, hist["v_loss"], label=label)
        axes[1, 0].plot(epochs_x, hist["q_grad_norm"], label=label)
        axes[1, 1].plot(epochs_x, hist["v_grad_norm"], label=label)

    axes[0, 0].set_title("Q Loss")
    axes[0, 0].set_ylabel("MSE loss")
    axes[0, 1].set_title("V Loss")
    axes[0, 1].set_ylabel("Expectile loss")
    axes[1, 0].set_title("Q Gradient L2 Norm")
    axes[1, 0].set_ylabel("Grad norm")
    axes[1, 1].set_title("V Gradient L2 Norm")
    axes[1, 1].set_ylabel("Grad norm")

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    save_path = "data/datasets/rank_network_size_grad_norms.png"
    fig.savefig(save_path, dpi=150)
    print(f"Saved gradient norm plot to {save_path}")
