"""Ablation: Is the problem TD targets, NN regression, or both?

We know sample-based GAE(lam=0.95) ranks actions well (rho~0.93 vs MC).
This script tests whether training a neural network via MSE on the SAME
targets preserves that ranking quality.

Experimental setup:
  1. Train V(s) on MC returns (frozen, known good from prior experiments)
  2. Use this V to compute per-transition targets from trajectories:
     - TD1 target:  A_td1(s,a) = delta_0 = r + gamma*V(s') - V(s)
     - GAE target:  A_gae(s,a) = sum_t (gamma*lam)^t delta_t  (first-step GAE)
     - MC target:   Q_mc(s,a) = sum_t gamma^t r_t  (from rollout)
  3. Train separate networks via MSE regression on these targets:
     - A_net(s,a) on TD1 targets       -> "NN(TD1)"
     - A_net(s,a) on GAE targets       -> "NN(GAE)"
     - Q_net(s,a) on MC Q targets      -> "NN(Q_MC)"
     - A_net(s,a) on MC A targets      -> "NN(A_MC)"  (A = Q_mc - V_mc)
  4. Compare rankings:
     - Sample-based baselines:  MC, GAE(lam=0.95), GAE(lam=0)
     - Network predictions:     NN(TD1), NN(GAE), NN(Q_MC), NN(A_MC)

Key insight:
  If sample-based GAE works but NN(GAE) fails, the problem is NN regression.
  If NN(GAE) works but NN(TD1) fails, the problem is TD1 targets.
  If NN(Q_MC) fails but NN(A_MC) works, the problem is Q-scale SNR.

Usage:
  python -m methods.gae.rank_nn_regression
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
from methods.gae.gae import layer_init
from methods.gae.rank_iql_debug import (
    v_eval, mc_returns, compute_gae, ranking_metrics,
)


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
    cache_path: str = "data/datasets/rank_cache_K8_M1_seed1.pt"
    train_dataset_path: str = "data/datasets/pickcube_expert.pt"
    eval_dataset_path: str = "data/datasets/pickcube_expert_eval.pt"
    dataset_num_envs: int = 16

    # V(s) training (MC-supervised, frozen after training)
    v_epochs: int = 100
    v_lr: float = 3e-4
    v_batch_size: int = 256

    # A/Q network regression
    reg_epochs: int = 200
    reg_lr: float = 3e-4
    reg_batch_size: int = 256
    reg_hidden_dim: int = 256


# =====================================================================
# Networks
# =====================================================================


class AdvNet(nn.Module):
    """A(s, a) network for advantage regression."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
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
# V(s) training on MC returns (same as rank_iql_debug)
# =====================================================================


def train_v_mc(trajectories, state_dim, gamma, device, args):
    """Train V(s) by MSE regression on MC returns."""
    from methods.gae.gae import Critic

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
# Compute per-(state, action) regression targets from trajectories
# =====================================================================


def compute_regression_targets(v_net, trajectories, traj_map, N, K,
                               gamma, gae_lambda, device):
    """Compute TD1, GAE, and MC-Q targets for each (state, action) pair.

    For each trajectory starting at (s_i, a_k):
      - TD1:  delta_0 = r_0 + gamma*V(s_1)*(1-term_0) - V(s_0)
      - GAE:  sum_t (gamma*lam)^t delta_t  (first-step advantage)
      - MC-Q: sum_t gamma^t r_t  (discounted return of entire trajectory)

    Returns averaged over M rollouts per (state, action) pair.

    Returns:
        td1_targets:  (N, K) tensor
        gae_targets:  (N, K) tensor
        mcq_targets:  (N, K) tensor  (MC return = Q_mc estimate)
    """
    # Batch-evaluate V on all trajectory states
    all_s = torch.cat([t["states"] for t in trajectories])
    all_ns = torch.cat([t["next_states"] for t in trajectories])
    all_v = v_eval(v_net, all_s, device)
    all_v_next = v_eval(v_net, all_ns, device)

    td1_sum = torch.zeros(N, K)
    gae_sum = torch.zeros(N, K)
    mcq_sum = torch.zeros(N, K)
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

        # TD errors
        delta = rewards + gamma * v_next * (1.0 - terminated) - v

        # TD1: just delta_0
        td1_val = delta[0].item()

        # GAE: backward pass
        gae_val = 0.0
        for t in reversed(range(T)):
            gae_val = delta[t] + gamma * gae_lambda * (1.0 - dones[t]) * gae_val
        gae_val = gae_val.item()

        # MC return (Q_mc): sum gamma^t r_t
        mc_ret = 0.0
        for t in reversed(range(T)):
            mc_ret = rewards[t].item() + gamma * mc_ret

        si, ai = traj_map[i]
        td1_sum[si, ai] += td1_val
        gae_sum[si, ai] += gae_val
        mcq_sum[si, ai] += mc_ret
        counts[si, ai] += 1

    counts = counts.clamp(min=1)
    return td1_sum / counts, gae_sum / counts, mcq_sum / counts


# =====================================================================
# Train A/Q network via MSE regression
# =====================================================================


def train_regression(train_states, train_actions, train_targets,
                     state_dim, action_dim, hidden_dim, device, args,
                     label=""):
    """Train an A(s,a) network by MSE on pre-computed targets.

    Args:
        train_states:  (M, state_dim) tensor
        train_actions: (M, action_dim) tensor
        train_targets: (M,) tensor of scalar targets
        label: string for logging

    Returns:
        trained AdvNet, training history dict
    """
    net = AdvNet(state_dim, action_dim, hidden_dim).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    N = train_states.shape[0]
    print(f"    {label}: {n_params:,} params, {N:,} samples, "
          f"target mean={train_targets.mean():.4f}, std={train_targets.std():.4f}")

    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.reg_lr, eps=1e-5, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.reg_epochs, eta_min=1e-5,
    )

    history = {"loss": [], "grad_norm": []}

    for epoch in range(1, args.reg_epochs + 1):
        idx = torch.randperm(N)
        epoch_loss, epoch_grad, n_batch = 0.0, 0.0, 0
        net.train()

        for start in range(0, N, args.reg_batch_size):
            bi = idx[start : start + args.reg_batch_size]
            s = train_states[bi].to(device)
            a = train_actions[bi].to(device)
            tgt = train_targets[bi].to(device)

            pred = net(s, a).squeeze(-1)
            loss = 0.5 * ((pred - tgt) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            grad = nn.utils.clip_grad_norm_(net.parameters(), 0.5).item()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_grad += grad
            n_batch += 1

        scheduler.step()
        history["loss"].append(epoch_loss / n_batch)
        history["grad_norm"].append(epoch_grad / n_batch)

        if epoch == 1 or epoch % 50 == 0:
            print(f"      Epoch {epoch}/{args.reg_epochs}: "
                  f"loss={epoch_loss / n_batch:.6f}, "
                  f"grad={epoch_grad / n_batch:.4f}")

    net.eval()
    return net, history


# =====================================================================
# Evaluate network predictions on eval (state, action) pairs
# =====================================================================


@torch.no_grad()
def eval_net_advantages(net, eval_states, sampled_actions, device):
    """Evaluate A_net(s, a_k) for all eval states and sampled actions.

    Returns (N, K) tensor.
    """
    N, K, _ = sampled_actions.shape
    adv = torch.zeros(N, K)
    net.eval()
    for i in range(0, N, 4096):
        j = min(i + 4096, N)
        s = eval_states[i:j].to(device)
        for k in range(K):
            a = sampled_actions[i:j, k].to(device)
            adv[i:j, k] = net(s, a).squeeze(-1).cpu()
    return adv


@torch.no_grad()
def eval_qnet_advantages(net, v_net, eval_states, sampled_actions, device):
    """Evaluate A = Q_net(s,a) - V(s) for all eval states and sampled actions.

    Returns (N, K) tensor.
    """
    N, K, _ = sampled_actions.shape
    adv = torch.zeros(N, K)
    net.eval()
    v_net.eval()
    for i in range(0, N, 4096):
        j = min(i + 4096, N)
        s = eval_states[i:j].to(device)
        v = v_net(s).squeeze(-1)
        for k in range(K):
            a = sampled_actions[i:j, k].to(device)
            q = net(s, a).squeeze(-1)
            adv[i:j, k] = (q - v).cpu()
    return adv


# =====================================================================
# Ranking comparison (reused from rank_iql_debug)
# =====================================================================


def compare_all(methods):
    """Print pairwise ranking comparison table."""
    names = list(methods.keys())
    mc = methods["MC"]
    N, K = mc.shape

    valid = np.array([mc[i].std() > 1e-8 for i in range(N)])
    n_valid = int(valid.sum())
    print(f"  Valid states: {n_valid}/{N}\n")

    print(f"  {'Method A':<18} {'Method B':<18} {'Spearman rho':>14} {'Top-1':>8}")
    print(f"  {'─' * 62}")

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
    action_dim = sampled_actions.shape[2]

    mc_adv = q_mc - v_mc.unsqueeze(1)  # (N, K)
    print(f"  {N} states, K={K} actions, {len(trajectories)} trajectories")
    print(f"  MC A(s,a): mean={mc_adv.mean():.4f}, std={mc_adv.std():.4f}")

    eval_states = OfflineRLDataset([args.eval_dataset_path], False, False).state

    # =================================================================
    # 2. Train V on MC returns (frozen for all subsequent steps)
    # =================================================================
    print(f"\n{'=' * 60}")
    print("STEP 1: Train V(s) on MC returns (frozen after this)")
    print(f"{'=' * 60}")

    train_dataset = OfflineRLDataset([args.train_dataset_path], False, False)
    train_trajs = train_dataset.extract_trajectories(
        num_envs=args.dataset_num_envs, gamma=args.gamma
    )
    del train_dataset

    v_net = train_v_mc(train_trajs, state_dim, args.gamma, device, args)

    # Check V quality
    v_pred = v_eval(v_net, eval_states, device)
    r_v, _ = sp_stats.pearsonr(v_mc.numpy(), v_pred.numpy())
    mae_v = np.mean(np.abs(v_pred.numpy() - v_mc.numpy()))
    print(f"  V quality: Pearson r={r_v:.4f}, MAE={mae_v:.4f}")

    # =================================================================
    # 3. Compute sample-based baselines (no NN involved)
    # =================================================================
    print(f"\n{'=' * 60}")
    print("STEP 2: Sample-based baselines (from trajectories, no NN)")
    print(f"{'=' * 60}")

    adv_gae = compute_gae(
        v_net, trajectories, traj_map, N, K,
        args.gamma, args.gae_lambda, device,
    )
    adv_td1 = compute_gae(
        v_net, trajectories, traj_map, N, K,
        args.gamma, 0.0, device,
    )
    print(f"  Sample GAE(lam={args.gae_lambda}): "
          f"mean={adv_gae.mean():.4f}, std={adv_gae.std():.4f}")
    print(f"  Sample TD1 (lam=0):      "
          f"mean={adv_td1.mean():.4f}, std={adv_td1.std():.4f}")

    # =================================================================
    # 4. Compute regression targets from trajectories
    # =================================================================
    print(f"\n{'=' * 60}")
    print("STEP 3: Compute per-(s,a) regression targets")
    print(f"{'=' * 60}")

    td1_tgt, gae_tgt, mcq_tgt = compute_regression_targets(
        v_net, trajectories, traj_map, N, K,
        args.gamma, args.gae_lambda, device,
    )
    mca_tgt = mcq_tgt - v_mc.unsqueeze(1)  # A_mc = Q_mc - V_mc (using sample V_mc)

    print(f"  TD1 targets:  mean={td1_tgt.mean():.4f}, std={td1_tgt.std():.4f}")
    print(f"  GAE targets:  mean={gae_tgt.mean():.4f}, std={gae_tgt.std():.4f}")
    print(f"  MC-Q targets: mean={mcq_tgt.mean():.4f}, std={mcq_tgt.std():.4f}")
    print(f"  MC-A targets: mean={mca_tgt.mean():.4f}, std={mca_tgt.std():.4f}")

    # Flatten (N, K) -> (N*K,) for regression training
    flat_states = eval_states.unsqueeze(1).expand(-1, K, -1).reshape(-1, state_dim)
    flat_actions = sampled_actions.reshape(-1, action_dim)

    # =================================================================
    # 5. Train networks via MSE regression on each target type
    # =================================================================
    print(f"\n{'=' * 60}")
    print("STEP 4: Train A/Q networks via MSE regression")
    print(f"{'=' * 60}")

    hdim = args.reg_hidden_dim
    all_histories = {}

    # NN(TD1): A-net trained on TD1 targets
    print(f"\n  --- NN(TD1): Train A(s,a) on TD1 targets ---")
    torch.manual_seed(args.seed)
    net_td1, hist_td1 = train_regression(
        flat_states, flat_actions, td1_tgt.reshape(-1),
        state_dim, action_dim, hdim, device, args, label="NN(TD1)",
    )
    all_histories["NN(TD1)"] = hist_td1

    # NN(GAE): A-net trained on GAE(lam=0.95) targets
    print(f"\n  --- NN(GAE): Train A(s,a) on GAE targets ---")
    torch.manual_seed(args.seed)
    net_gae, hist_gae = train_regression(
        flat_states, flat_actions, gae_tgt.reshape(-1),
        state_dim, action_dim, hdim, device, args, label="NN(GAE)",
    )
    all_histories["NN(GAE)"] = hist_gae

    # NN(Q_MC): Q-net trained on MC return targets (Q-scale)
    print(f"\n  --- NN(Q_MC): Train Q(s,a) on MC-Q targets ---")
    torch.manual_seed(args.seed)
    net_qmc, hist_qmc = train_regression(
        flat_states, flat_actions, mcq_tgt.reshape(-1),
        state_dim, action_dim, hdim, device, args, label="NN(Q_MC)",
    )
    all_histories["NN(Q_MC)"] = hist_qmc

    # NN(A_MC): A-net trained on MC advantage targets (A-scale)
    print(f"\n  --- NN(A_MC): Train A(s,a) on MC-A targets ---")
    torch.manual_seed(args.seed)
    net_amc, hist_amc = train_regression(
        flat_states, flat_actions, mca_tgt.reshape(-1),
        state_dim, action_dim, hdim, device, args, label="NN(A_MC)",
    )
    all_histories["NN(A_MC)"] = hist_amc

    # =================================================================
    # 6. Evaluate all network predictions
    # =================================================================
    print(f"\n{'=' * 60}")
    print("STEP 5: Evaluate network predictions")
    print(f"{'=' * 60}")

    # NN(TD1) and NN(GAE) predict advantage directly
    adv_nn_td1 = eval_net_advantages(net_td1, eval_states, sampled_actions, device)
    adv_nn_gae = eval_net_advantages(net_gae, eval_states, sampled_actions, device)

    # NN(Q_MC): predict Q, use A = Q_net(s,a) - V(s) with frozen V
    adv_nn_qmc = eval_qnet_advantages(
        net_qmc, v_net, eval_states, sampled_actions, device,
    )

    # NN(A_MC): predict advantage directly
    adv_nn_amc = eval_net_advantages(net_amc, eval_states, sampled_actions, device)

    print(f"  NN(TD1):  mean={adv_nn_td1.mean():.4f}, std={adv_nn_td1.std():.4f}")
    print(f"  NN(GAE):  mean={adv_nn_gae.mean():.4f}, std={adv_nn_gae.std():.4f}")
    print(f"  NN(Q_MC): mean={adv_nn_qmc.mean():.4f}, std={adv_nn_qmc.std():.4f}")
    print(f"  NN(A_MC): mean={adv_nn_amc.mean():.4f}, std={adv_nn_amc.std():.4f}")

    # =================================================================
    # 7. Ranking comparison
    # =================================================================
    print(f"\n{'=' * 60}")
    print("RANKING COMPARISON (all vs MC)")
    print(f"{'=' * 60}")

    methods = {
        "MC": mc_adv.numpy(),
        "Sample GAE": adv_gae.numpy(),
        "Sample TD1": adv_td1.numpy(),
        "NN(TD1)": adv_nn_td1.numpy(),
        "NN(GAE)": adv_nn_gae.numpy(),
        "NN(Q_MC)": adv_nn_qmc.numpy(),
        "NN(A_MC)": adv_nn_amc.numpy(),
    }

    compare_all(methods)

    # =================================================================
    # 8. Diagnosis
    # =================================================================
    print(f"\n{'=' * 60}")
    print("DIAGNOSIS")
    print(f"{'=' * 60}")
    print()
    print("  Q1: Does NN regression destroy GAE ranking?")
    print("      Compare 'Sample GAE' vs 'NN(GAE)' (same targets, +/- NN)")
    print()
    print("  Q2: Is TD1 the problem or NN the problem?")
    print("      Compare 'Sample TD1' vs 'NN(TD1)'")
    print()
    print("  Q3: Does Q-scale SNR hurt NN regression?")
    print("      Compare 'NN(Q_MC)' vs 'NN(A_MC)' (same info, different scale)")
    print()
    print("  Q4: Can a NN learn advantage at all?")
    print("      'NN(A_MC)' uses ground-truth MC advantages as targets")
    print()

    # =================================================================
    # 9. Plot: training curves + target-vs-prediction scatter
    # =================================================================
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("NN Regression Ablation: Training Curves and Predictions",
                 fontsize=14)

    nn_methods = ["NN(TD1)", "NN(GAE)", "NN(Q_MC)", "NN(A_MC)"]
    nn_preds = [adv_nn_td1, adv_nn_gae, adv_nn_qmc, adv_nn_amc]
    nn_targets = [td1_tgt, gae_tgt, mcq_tgt, mca_tgt]
    target_names = ["TD1 targets", "GAE targets", "MC-Q targets", "MC-A targets"]

    # Row 0: training loss
    for col, name in enumerate(nn_methods):
        ax = axes[0, col]
        hist = all_histories[name]
        epochs_x = np.arange(1, len(hist["loss"]) + 1)
        ax.plot(epochs_x, hist["loss"], label="loss", color="C0")
        ax.set_title(f"{name}: Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.grid(alpha=0.3)

        # Add grad norm on twin axis
        ax2 = ax.twinx()
        ax2.plot(epochs_x, hist["grad_norm"], label="grad norm",
                 color="C1", alpha=0.6)
        ax2.set_ylabel("Grad L2 Norm", color="C1")

    # Row 1: target vs prediction scatter (per-state-action)
    mc_adv_np = mc_adv.numpy()
    valid = np.array([mc_adv_np[i].std() > 1e-8 for i in range(N)])

    for col, (name, pred, tgt, tgt_name) in enumerate(
        zip(nn_methods, nn_preds, nn_targets, target_names)
    ):
        ax = axes[1, col]

        # Scatter: NN prediction vs MC advantage (ground truth)
        x = mc_adv_np.flatten()
        y = pred.numpy().flatten()
        ax.scatter(x, y, alpha=0.15, s=8, edgecolors="none", color="C0")

        # y=x line
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "r--", lw=1)

        r_pool, _ = sp_stats.pearsonr(x, y)

        # Per-state Spearman vs MC
        rhos = []
        for i in range(N):
            if not valid[i]:
                continue
            rho, _ = sp_stats.spearmanr(mc_adv_np[i], pred.numpy()[i])
            rhos.append(rho)
        rho_mean = np.nanmean(rhos)

        ax.set_xlabel("MC advantage (ground truth)")
        ax.set_ylabel(f"{name} prediction")
        ax.set_title(f"{name}\nr={r_pool:.3f}, ρ={rho_mean:.3f}")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    save_path = "data/datasets/rank_nn_regression.png"
    fig.savefig(save_path, dpi=150)
    print(f"\nSaved figure to {save_path}")
