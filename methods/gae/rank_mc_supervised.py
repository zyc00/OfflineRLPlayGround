"""Train Q(s,a) and V(s) supervised on MC return targets, compare advantage rankings.

Sanity check: if supervised regression on ground-truth MC returns can recover
correct advantage rankings, function approximation is not the bottleneck.

Compares three methods against MC ground truth:
  - GAE:           uses the MC-trained V(s) + GAE formula (TD errors)
  - MC_supervised: uses MC-trained Q(s,a) - V(s) directly

Both GAE and MC_supervised share the same V(s) trained on trajectory MC returns.
This isolates the effect of the GAE formula vs direct Q regression.

Requires cached rollout data from rank_mc_vs_gae.py.
"""

import os
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tyro
from scipy import stats as sp_stats

from data.offline_dataset import OfflineRLDataset
from methods.gae.gae_online import Critic
from methods.gae.rank_mc_vs_gae import (
    _cache_path,
    _compute_mc_returns,
    compute_gae_advantages,
    compute_ranking_metrics,
)
from methods.iql.iql import QNetwork


@dataclass
class Args:
    eval_dataset_path: str = "data/datasets/pickcube_expert_eval.pt"
    """path to the evaluation .pt dataset file"""
    seed: int = 1
    """random seed"""
    cuda: bool = True
    """if toggled, cuda will be enabled"""
    gamma: float = 0.8
    """discount factor"""
    gae_lambda: float = 0.95
    """lambda for generalized advantage estimation"""
    num_sampled_actions: int = 8
    """K: must match the cached rollout data"""
    num_mc_rollouts: int = 10
    """M: must match the cached rollout data"""

    # Training hyperparameters
    lr: float = 3e-4
    """learning rate"""
    epochs: int = 200
    """number of training epochs"""
    batch_size: int = 256
    """minibatch size"""
    weight_decay: float = 1e-4
    """weight decay (L2 regularization)"""
    grad_clip: float = 0.5
    """max gradient norm"""
    patience: int = 50
    """early stopping patience"""

    output: str = ""
    """save figure to this path (default: auto-generated)"""


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_q_mc(eval_states, sampled_actions, q_mc, state_dim, action_dim,
               device, args):
    """Train Q(s,a) by regressing on MC Q-returns.

    Training data: (state_i, action_k) -> q_mc[i, k] for all i, k.
    """
    N, K = q_mc.shape
    states_flat = eval_states.unsqueeze(1).expand(-1, K, -1).reshape(N * K, state_dim)
    actions_flat = sampled_actions.reshape(N * K, action_dim)
    targets_flat = q_mc.reshape(N * K)

    total = N * K
    perm = torch.randperm(total)
    val_size = max(1, int(total * 0.1))
    train_idx = perm[val_size:]
    val_idx = perm[:val_size]
    train_size = train_idx.shape[0]

    q_net = QNetwork(state_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(
        q_net.parameters(), lr=args.lr, eps=1e-5, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5,
    )

    val_s = states_flat[val_idx].to(device)
    val_a = actions_flat[val_idx].to(device)
    val_t = targets_flat[val_idx].to(device)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    print(f"\nTraining Q(s,a) on {train_size} MC Q-return pairs...")

    for epoch in range(args.epochs):
        q_net.train()
        indices = train_idx[torch.randperm(train_size)]
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, train_size, args.batch_size):
            batch_idx = indices[start : start + args.batch_size]
            s = states_flat[batch_idx].to(device)
            a = actions_flat[batch_idx].to(device)
            t = targets_flat[batch_idx].to(device)

            pred = q_net(s, a).squeeze(-1)
            loss = 0.5 * ((pred - t) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()

        q_net.eval()
        with torch.no_grad():
            val_pred = q_net(val_s, val_a).squeeze(-1)
            val_loss = 0.5 * ((val_pred - val_t) ** 2).mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in q_net.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 20 == 0 or epoch == 0:
            avg = epoch_loss / max(num_batches, 1)
            print(
                f"  Epoch {epoch + 1}/{args.epochs}: "
                f"train_loss={avg:.6f}, val_loss={val_loss:.6f}"
            )

    if best_state is not None:
        q_net.load_state_dict(best_state)
    q_net.eval()
    return q_net


def train_v_mc(trajectories, state_dim, gamma, device, args):
    """Train V(s) by regressing on MC returns from trajectory data."""
    all_states = []
    all_returns = []
    for traj in trajectories:
        all_states.append(traj["states"])
        all_returns.append(_compute_mc_returns(traj["rewards"], gamma))
    all_states = torch.cat(all_states, dim=0)
    all_returns = torch.cat(all_returns, dim=0)

    total = all_states.shape[0]
    perm = torch.randperm(total)
    val_size = max(1, int(total * 0.1))
    train_idx = perm[val_size:]
    val_idx = perm[:val_size]
    train_size = train_idx.shape[0]

    v_net = Critic("state", state_dim=state_dim).to(device)
    optimizer = torch.optim.Adam(
        v_net.parameters(), lr=args.lr, eps=1e-5, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5,
    )

    val_s = all_states[val_idx].to(device)
    val_t = all_returns[val_idx].to(device)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    print(f"\nTraining V(s) on {train_size} MC return pairs...")

    for epoch in range(args.epochs):
        v_net.train()
        indices = train_idx[torch.randperm(train_size)]
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, train_size, args.batch_size):
            batch_idx = indices[start : start + args.batch_size]
            s = all_states[batch_idx].to(device)
            t = all_returns[batch_idx].to(device)

            pred = v_net(s).squeeze(-1)
            loss = 0.5 * ((pred - t) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(v_net.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()

        v_net.eval()
        with torch.no_grad():
            val_pred = v_net(val_s).squeeze(-1)
            val_loss = 0.5 * ((val_pred - val_t) ** 2).mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in v_net.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 20 == 0 or epoch == 0:
            avg = epoch_loss / max(num_batches, 1)
            print(
                f"  Epoch {epoch + 1}/{args.epochs}: "
                f"train_loss={avg:.6f}, val_loss={val_loss:.6f}"
            )

    if best_state is not None:
        v_net.load_state_dict(best_state)
    v_net.eval()
    return v_net


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(methods_dict, metrics, pred_q, q_mc, pred_v, v_mc,
                 save_path):
    """Plot regression quality and 3-way ranking comparison."""
    pairs = list(metrics["pairs"].keys())
    n_pairs = len(pairs)

    fig = plt.figure(figsize=(6 * max(n_pairs, 3), 18), constrained_layout=True)
    gs = fig.add_gridspec(3, max(n_pairs, 3))

    # --- Row 0: Regression quality (Q, V, Advantage scatter) ---
    ax = fig.add_subplot(gs[0, 0])
    x, y = q_mc.numpy().flatten(), pred_q.numpy().flatten()
    ax.scatter(x, y, alpha=0.15, s=8, edgecolors="none")
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "r--", lw=1)
    r, _ = sp_stats.pearsonr(x, y)
    ax.set_xlabel("MC Q(s,a)")
    ax.set_ylabel("Predicted Q(s,a)")
    ax.set_title(f"Q regression (r={r:.3f})")

    ax = fig.add_subplot(gs[0, 1])
    x, y = v_mc.numpy(), pred_v.numpy()
    ax.scatter(x, y, alpha=0.3, s=12, edgecolors="none")
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "r--", lw=1)
    r, _ = sp_stats.pearsonr(x, y)
    ax.set_xlabel("MC V(s)")
    ax.set_ylabel("Predicted V(s)")
    ax.set_title(f"V regression (r={r:.3f})")

    mc_adv = methods_dict["MC"]
    sup_adv = methods_dict["MC_supervised"]
    ax = fig.add_subplot(gs[0, 2])
    x, y = mc_adv.flatten(), sup_adv.flatten()
    ax.scatter(x, y, alpha=0.15, s=8, edgecolors="none")
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "r--", lw=1)
    r, _ = sp_stats.pearsonr(x, y)
    ax.set_xlabel("MC advantage")
    ax.set_ylabel("MC_supervised advantage")
    ax.set_title(f"Advantage (r={r:.3f})")

    # --- Row 1: Spearman rho histograms for each pair ---
    for col, pair_key in enumerate(pairs):
        n1, n2 = pair_key.split("_vs_")
        ax = fig.add_subplot(gs[1, col])
        rhos = metrics["pairs"][pair_key]["spearman_rhos"]
        ax.hist(rhos, bins=30, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axvline(np.mean(rhos), color="r", ls="--", lw=1.5,
                   label=f"mean={np.mean(rhos):.3f}")
        ax.axvline(np.median(rhos), color="orange", ls="--", lw=1.5,
                   label=f"median={np.median(rhos):.3f}")
        ax.set_xlabel("Spearman rho")
        ax.set_ylabel("Count")
        ax.set_title(f"Per-state Spearman rho: {n1} vs {n2}")
        ax.legend(fontsize=9)

    # --- Row 2: Example state + Summary table ---
    # Example state (near median Spearman for first pair)
    first_pair = pairs[0]
    rhos_first = metrics["pairs"][first_pair]["spearman_rhos"]
    median_rho = np.median(rhos_first)
    example_idx = np.argmin(np.abs(rhos_first - median_rho))
    valid_indices = np.where(metrics["valid_mask"])[0]
    orig_idx = valid_indices[example_idx]
    K = mc_adv.shape[1]

    ax = fig.add_subplot(gs[2, 0])
    x = np.arange(K)
    names = list(methods_dict.keys())
    n_methods = len(names)
    width = 0.8 / n_methods
    for m_idx, name in enumerate(names):
        offset = (m_idx - (n_methods - 1) / 2) * width
        ax.bar(x + offset, methods_dict[name][orig_idx], width,
               label=name, alpha=0.8)
    ax.set_xlabel("Action index")
    ax.set_ylabel("Advantage")
    n1, n2 = first_pair.split("_vs_")
    ax.set_title(
        f"Example state {orig_idx} "
        f"({n1}-{n2} rho={rhos_first[example_idx]:.3f})"
    )
    ax.legend(fontsize=9)
    ax.set_xticks(x)

    # Summary table
    ax = fig.add_subplot(gs[2, 1:])
    ax.axis("off")
    col_labels = ["Metric"] + [k.replace("_vs_", " vs ") for k in pairs]
    rows = []
    for metric_name, metric_key, fmt in [
        ("Spearman rho (mean)", "spearman_rhos", ".3f"),
        ("Spearman rho (median)", "spearman_rhos", ".3f"),
        ("Kendall tau (mean)", "kendall_taus", ".3f"),
        ("Top-1 agreement", "top1_agrees", ".3f"),
        ("Concordance (mean)", "concordances", ".3f"),
    ]:
        row = [metric_name]
        for pair_key in pairs:
            vals = metrics["pairs"][pair_key][metric_key]
            if "median" in metric_name:
                row.append(f"{np.median(vals):{fmt}}")
            else:
                row.append(f"{np.mean(vals):{fmt}}")
        rows.append(row)
    # Add Pearson r (pooled) row
    row = ["Pearson r (pooled)"]
    for pair_key in pairs:
        n1, n2 = pair_key.split("_vs_")
        r, _ = sp_stats.pearsonr(
            methods_dict[n1].flatten(), methods_dict[n2].flatten()
        )
        row.append(f"{r:.3f}")
    rows.append(row)
    rows.append(
        ["Valid states"]
        + [f"{metrics['num_valid']}/{metrics['num_total']}"] * n_pairs
    )

    table = ax.table(
        cellText=rows, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    ax.set_title("Summary", fontsize=11, pad=10)

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    args = tyro.cli(Args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )

    # -------------------------------------------------------------------
    # 1. Load cached rollout data
    # -------------------------------------------------------------------
    cache_file = _cache_path(args)
    if not os.path.exists(cache_file):
        print(f"Cache not found: {cache_file}")
        print("Run rank_mc_vs_gae.py first to collect MC rollouts.")
        exit(1)

    print(f"Loading cached rollout data from {cache_file}")
    cache = torch.load(cache_file, weights_only=False)

    v_mc = cache["v_mc"]
    q_mc = cache["q_mc"]
    sampled_actions = cache["sampled_actions"]
    trajectories = cache["trajectories"]
    traj_to_state_action = cache["traj_to_state_action"]
    N = cache["N"]
    state_dim = cache["state_dim"]
    K = args.num_sampled_actions
    action_dim = sampled_actions.shape[2]

    mc_advantages = q_mc - v_mc.unsqueeze(1)  # (N, K)

    eval_dataset = OfflineRLDataset([args.eval_dataset_path], False, False)
    eval_states = eval_dataset.state  # (N, state_dim)

    print(f"N={N}, K={K}, state_dim={state_dim}, action_dim={action_dim}")
    print(f"Trajectories: {len(trajectories)}")
    print(
        f"MC advantages: mean={mc_advantages.mean():.4f}, "
        f"std={mc_advantages.std():.4f}"
    )

    # -------------------------------------------------------------------
    # 2. Train V(s) on MC trajectory returns (shared by GAE & MC_supervised)
    # -------------------------------------------------------------------
    v_net = train_v_mc(trajectories, state_dim, args.gamma, device, args)

    # -------------------------------------------------------------------
    # 3. Compute GAE advantages using the trained V
    # -------------------------------------------------------------------
    print("\nComputing GAE advantages...")
    gae_advantages = compute_gae_advantages(
        v_net, trajectories, traj_to_state_action,
        N, K, args.gamma, args.gae_lambda, device,
    )
    print(
        f"  GAE A(s,a): mean={gae_advantages.mean():.4f}, "
        f"std={gae_advantages.std():.4f}"
    )

    # -------------------------------------------------------------------
    # 4. Train Q(s,a) on MC Q-returns
    # -------------------------------------------------------------------
    q_net = train_q_mc(
        eval_states, sampled_actions, q_mc,
        state_dim, action_dim, device, args,
    )

    # -------------------------------------------------------------------
    # 5. Evaluate MC_supervised advantages: Q_pred - V_pred
    # -------------------------------------------------------------------
    print("\nEvaluating MC_supervised advantages...")
    pred_q = torch.zeros(N, K)
    pred_v = torch.zeros(N)

    with torch.no_grad():
        for start in range(0, N, args.batch_size):
            end = min(start + args.batch_size, N)
            s = eval_states[start:end].to(device)
            pred_v[start:end] = v_net(s).squeeze(-1).cpu()
            for k in range(K):
                a = sampled_actions[start:end, k].to(device)
                pred_q[start:end, k] = q_net(s, a).squeeze(-1).cpu()

    pred_advantages = pred_q - pred_v.unsqueeze(1)

    # -------------------------------------------------------------------
    # 6. Regression quality
    # -------------------------------------------------------------------
    q_mse = ((pred_q - q_mc) ** 2).mean().item()
    v_mse = ((pred_v - v_mc) ** 2).mean().item()
    a_mse = ((pred_advantages - mc_advantages) ** 2).mean().item()
    q_r = np.corrcoef(pred_q.numpy().flatten(), q_mc.numpy().flatten())[0, 1]
    v_r = np.corrcoef(pred_v.numpy().flatten(), v_mc.numpy().flatten())[0, 1]
    a_r = np.corrcoef(
        pred_advantages.numpy().flatten(), mc_advantages.numpy().flatten()
    )[0, 1]

    print(f"\nRegression quality:")
    print(f"  Q(s,a):  MSE={q_mse:.6f},  Pearson r={q_r:.4f}")
    print(f"  V(s):    MSE={v_mse:.6f},  Pearson r={v_r:.4f}")
    print(f"  A(s,a):  MSE={a_mse:.6f},  Pearson r={a_r:.4f}")

    print(f"\nPredicted stats:")
    print(f"  Q: mean={pred_q.mean():.4f}, std={pred_q.std():.4f}")
    print(f"  V: mean={pred_v.mean():.4f}, std={pred_v.std():.4f}")
    print(f"  A: mean={pred_advantages.mean():.4f}, std={pred_advantages.std():.4f}")

    # -------------------------------------------------------------------
    # 7. Ranking comparison: MC vs GAE vs MC_supervised
    # -------------------------------------------------------------------
    methods_dict = {
        "MC": mc_advantages.numpy(),
        "GAE": gae_advantages.numpy(),
        "MC_supervised": pred_advantages.numpy(),
    }

    print("\nComputing ranking metrics...")
    metrics = compute_ranking_metrics(methods_dict)

    print(f"\n{'=' * 60}")
    print("RANKING COMPARISON: MC vs GAE vs MC_supervised")
    print(f"{'=' * 60}")
    print(
        f"Valid states (MC has variance): "
        f"{metrics['num_valid']}/{metrics['num_total']}"
    )
    for pair_key, pm in metrics["pairs"].items():
        label = pair_key.replace("_vs_", " vs ")
        print(f"\n  {label}:")
        print(
            f"    Spearman rho:  mean={pm['spearman_rhos'].mean():.3f}, "
            f"median={np.median(pm['spearman_rhos']):.3f}"
        )
        print(
            f"    Kendall tau:   mean={pm['kendall_taus'].mean():.3f}, "
            f"median={np.median(pm['kendall_taus']):.3f}"
        )
        print(f"    Top-1 agree:   {pm['top1_agrees'].mean():.3f}")
        print(f"    Concordance:   {pm['concordances'].mean():.3f}")
    print(f"{'=' * 60}")

    # -------------------------------------------------------------------
    # 8. Save results and plot
    # -------------------------------------------------------------------
    results = {
        "mc_advantages": mc_advantages,
        "gae_advantages": gae_advantages,
        "pred_advantages": pred_advantages,
        "pred_q": pred_q,
        "pred_v": pred_v,
        "q_mc": q_mc,
        "v_mc": v_mc,
        "metrics": metrics,
    }
    save_path = os.path.join(
        os.path.dirname(args.eval_dataset_path),
        f"rank_mc_supervised_K{K}_M{args.num_mc_rollouts}"
        f"_gamma{args.gamma}_lambda{args.gae_lambda}.pt",
    )
    torch.save(results, save_path)
    print(f"\nSaved results to {save_path}")

    fig_path = args.output or os.path.join(
        os.path.dirname(args.eval_dataset_path),
        f"rank_mc_supervised_K{K}_M{args.num_mc_rollouts}"
        f"_gamma{args.gamma}_lambda{args.gae_lambda}.png",
    )
    plot_results(
        methods_dict, metrics, pred_q, q_mc, pred_v, v_mc, fig_path,
    )
