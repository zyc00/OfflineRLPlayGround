"""Train Q(s,a) and V(s) by regressing on MC ground truth, evaluate ranking.

Trains on the eval set's MC-computed Q and V values directly (no TD, no IQL).
Tests whether the learned Q can preserve within-state action ranking.

Key metrics:
  - Q/V regression quality (Pearson r, MAE)
  - Q per-state ranking (does Q_nn rank actions correctly within each state?)
  - A = Q_nn - V_nn ranking vs MC ground truth
  - Isolates Q vs V error contribution

Usage:
  python -m methods.gae.rank_qv_regression
  python -m methods.gae.rank_qv_regression --hidden_dim 512 --epochs 500
"""

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import tyro
from scipy import stats as sp_stats

from methods.gae.gae import layer_init
from methods.gae.rank_iql_debug import v_eval, compute_gae


# ── Networks ──────────────────────────────────────────────────────────


def _build_mlp(in_dim, hidden_dim, num_layers, layer_norm=False):
    layers = [layer_init(nn.Linear(in_dim, hidden_dim))]
    if layer_norm:
        layers.append(nn.LayerNorm(hidden_dim))
    layers.append(nn.Tanh())
    for _ in range(num_layers - 1):
        layers.append(layer_init(nn.Linear(hidden_dim, hidden_dim)))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Tanh())
    layers.append(layer_init(nn.Linear(hidden_dim, 1)))
    return nn.Sequential(*layers)


class VNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_layers=3, layer_norm=False):
        super().__init__()
        self.net = _build_mlp(state_dim, hidden_dim, num_layers, layer_norm)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers=3,
                 layer_norm=False, action_repeat=1):
        super().__init__()
        self.action_repeat = action_repeat
        self.net = _build_mlp(state_dim + action_dim * action_repeat,
                              hidden_dim, num_layers, layer_norm)

    def forward(self, state, action):
        if self.action_repeat > 1:
            action = action.repeat(1, self.action_repeat)
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


# ── Training ──────────────────────────────────────────────────────────


def train_net(net, data, targets, device, epochs, lr, batch_size, label="",
              is_q=False):
    """Train a network via MSE regression. Returns training losses."""
    N = targets.shape[0]
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    losses = []
    for epoch in range(1, epochs + 1):
        idx = torch.randperm(N)
        total_loss, n_batch = 0.0, 0
        net.train()
        for start in range(0, N, batch_size):
            bi = idx[start:start + batch_size]
            if is_q:
                pred = net(data[0][bi].to(device), data[1][bi].to(device))
            else:
                pred = net(data[bi].to(device))
            loss = 0.5 * ((pred - targets[bi].to(device)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            total_loss += loss.item()
            n_batch += 1
        sched.step()
        avg = total_loss / n_batch
        losses.append(avg)
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(f"    {label} epoch {epoch}/{epochs}: loss={avg:.6f}")

    net.eval()
    return losses


# ── Evaluation ────────────────────────────────────────────────────────


@torch.no_grad()
def eval_q(q_net, states, actions, device):
    """Evaluate Q(s, a_k) for all (N, K) pairs. Returns (N, K) numpy."""
    N, K, _ = actions.shape
    q = np.zeros((N, K))
    for i in range(0, N, 512):
        j = min(i + 512, N)
        s = states[i:j].to(device)
        for k in range(K):
            a = actions[i:j, k].to(device)
            q[i:j, k] = q_net(s, a).cpu().numpy()
    return q


@torch.no_grad()
def eval_v(v_net, states, device):
    """Evaluate V(s) for N states. Returns (N,) numpy."""
    v = []
    for i in range(0, states.shape[0], 4096):
        j = min(i + 4096, states.shape[0])
        v.append(v_net(states[i:j].to(device)).cpu().numpy())
    return np.concatenate(v)


def spearman_ranking(pred, mc_adv, valid):
    """Per-state mean Spearman rho of pred vs mc_adv."""
    rhos = []
    for i in range(mc_adv.shape[0]):
        if not valid[i]:
            continue
        rho, _ = sp_stats.spearmanr(mc_adv[i], pred[i])
        rhos.append(rho)
    rhos = np.array(rhos)
    return np.nanmean(rhos), np.nanmedian(rhos)


# ── Config ────────────────────────────────────────────────────────────


@dataclass
class Args:
    seed: int = 1
    cuda: bool = True
    gamma: float = 0.8
    gae_lambda: float = 0.95

    cache_path: str = "data/datasets/rank_cache_ckpt_301_K8_M16_seed1.pt"

    hidden_dim: int = 256
    num_layers: int = 3
    layer_norm: bool = False
    action_repeat: int = 1
    """repeat action vector N times in Q input to amplify action signal"""
    normalize: bool = False
    """normalize states and actions using offline dataset statistics"""
    offline_data_path: str = "data/datasets/pickcube_expert.pt"
    """path to offline dataset for computing normalization stats"""
    scale_factor: float = 1.0
    """scale Q and V targets by this factor before training (amplifies signal)"""
    epochs: int = 200
    v_lr: float = 3e-4
    q_lr: float = 3e-4
    batch_size: int = 256


# ── Main ──────────────────────────────────────────────────────────────


if __name__ == "__main__":
    args = tyro.cli(Args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Load data
    cache = torch.load(args.cache_path, weights_only=False)
    v_mc = cache["v_mc"]                        # (N,)
    q_mc = cache["q_mc"]                        # (N, K)
    sampled_actions = cache["sampled_actions"]   # (N, K, action_dim)
    trajectories = cache["trajectories"]
    traj_map = cache["traj_to_state_action"]
    N = cache["N"]
    state_dim = cache["state_dim"]
    K = sampled_actions.shape[1]
    action_dim = sampled_actions.shape[2]

    eval_states = cache.get("eval_states")
    if eval_states is None:
        from data.offline_dataset import OfflineRLDataset
        eval_states = OfflineRLDataset(
            ["data/datasets/pickcube_expert_eval.pt"], False, False
        ).state

    # ── Normalization ──────────────────────────────────────────────────
    if args.normalize:
        from data.offline_dataset import OfflineRLDataset
        offline_ds = OfflineRLDataset([args.offline_data_path], False, False)
        s_mean = offline_ds.state.mean(dim=0)
        s_std = offline_ds.state.std(dim=0).clamp(min=1e-6)
        a_min = offline_ds.actions.min(dim=0).values
        a_max = offline_ds.actions.max(dim=0).values
        a_range = (a_max - a_min).clamp(min=1e-6)
        del offline_ds
        print(f"\nNormalization (from {args.offline_data_path}):")
        print(f"  state: z-score  mean=[{s_mean.min():.3f}, {s_mean.max():.3f}]  "
              f"std=[{s_std.min():.3f}, {s_std.max():.3f}]")
        print(f"  action: min-max  min=[{a_min.min():.3f}, {a_min.max():.3f}]  "
              f"max=[{a_max.min():.3f}, {a_max.max():.3f}]")

        eval_states = (eval_states - s_mean) / s_std
        sampled_actions = (sampled_actions - a_min) / a_range * 2 - 1  # -> [-1, 1]
        # Normalize trajectory states/actions for GAE evaluation
        for traj in trajectories:
            traj["states"] = (traj["states"] - s_mean) / s_std
            traj["next_states"] = (traj["next_states"] - s_mean) / s_std
            traj["actions"] = (traj["actions"] - a_min) / a_range * 2 - 1

    # ── Scale targets ────────────────────────────────────────────────
    sf = args.scale_factor
    if sf != 1.0:
        v_mc = v_mc * sf
        q_mc = q_mc * sf
        for traj in trajectories:
            traj["rewards"] = traj["rewards"] * sf
        print(f"\nScale factor: {sf}x  (V, Q targets and rewards scaled)")

    mc_adv = (q_mc - v_mc.unsqueeze(1)).numpy()  # (N, K)
    valid = np.array([mc_adv[i].std() > 1e-8 for i in range(N)])

    print(f"Data: {N} states, K={K} actions, {int(valid.sum())} valid")
    print(f"  V:  mean={v_mc.mean():.4f}  std={v_mc.std():.4f}")
    print(f"  Q:  mean={q_mc.mean():.4f}  std={q_mc.std():.4f}")
    print(f"  A:  mean={np.mean(mc_adv):.4f}  std={np.std(mc_adv):.4f}")
    print(f"  SNR: Q_std / A_std = {q_mc.std() / np.std(mc_adv):.1f}x")
    ln_str = ", layer_norm" if args.layer_norm else ""
    ar_str = f", action_repeat={args.action_repeat}" if args.action_repeat > 1 else ""
    norm_str = ", normalized" if args.normalize else ""
    sf_str = f", scale={args.scale_factor}x" if args.scale_factor != 1.0 else ""
    print(f"\nNetwork: hidden_dim={args.hidden_dim}, num_layers={args.num_layers}{ln_str}{ar_str}{norm_str}{sf_str}, epochs={args.epochs}")

    # ── Train V ───────────────────────────────────────────────────────
    print(f"\n[V] Training on {N} (state, V_mc) pairs")
    torch.manual_seed(args.seed)
    v_net = VNet(state_dim, args.hidden_dim, args.num_layers, args.layer_norm).to(device)
    n_params_v = sum(p.numel() for p in v_net.parameters())
    print(f"    V params: {n_params_v:,}")
    train_net(v_net, eval_states, v_mc, device,
              args.epochs, args.v_lr, args.batch_size, label="V")

    # ── Train Q ───────────────────────────────────────────────────────
    flat_s = eval_states.unsqueeze(1).expand(-1, K, -1).reshape(-1, state_dim)
    flat_a = sampled_actions.reshape(-1, action_dim)
    flat_q = q_mc.reshape(-1)

    print(f"\n[Q] Training on {N * K} (state, action, Q_mc) pairs")
    torch.manual_seed(args.seed)
    q_net = QNet(state_dim, action_dim, args.hidden_dim, args.num_layers,
                 args.layer_norm, args.action_repeat).to(device)
    n_params_q = sum(p.numel() for p in q_net.parameters())
    print(f"    Q params: {n_params_q:,}")
    train_net(q_net, (flat_s, flat_a), flat_q, device,
              args.epochs, args.q_lr, args.batch_size, label="Q", is_q=True)

    # ── Evaluate ──────────────────────────────────────────────────────
    v_pred = eval_v(v_net, eval_states, device)          # (N,)
    q_pred = eval_q(q_net, eval_states, sampled_actions, device)  # (N, K)
    v_mc_np = v_mc.numpy()
    q_mc_np = q_mc.numpy()

    # Regression quality
    r_v, _ = sp_stats.pearsonr(v_mc_np, v_pred)
    mae_v = np.mean(np.abs(v_pred - v_mc_np))
    r_q, _ = sp_stats.pearsonr(q_mc_np.flatten(), q_pred.flatten())
    mae_q = np.mean(np.abs(q_pred - q_mc_np))

    # Per-state Q ranking (Pearson: does Q_nn preserve action order within state?)
    q_ps_r = [sp_stats.pearsonr(q_mc_np[i], q_pred[i])[0]
              for i in range(N) if valid[i]]

    print(f"\n{'='*60}")
    print(f"REGRESSION QUALITY")
    print(f"{'='*60}")
    print(f"  V:  Pearson r={r_v:.4f}   MAE={mae_v:.4f}")
    print(f"  Q:  Pearson r={r_q:.4f}   MAE={mae_q:.4f}   (pooled)")
    print(f"  Q per-state:  mean r={np.nanmean(q_ps_r):.4f}  "
          f"med r={np.nanmedian(q_ps_r):.4f}")
    print(f"  Q error / A signal: {mae_q:.4f} / {np.std(mc_adv):.4f} "
          f"= {mae_q / np.std(mc_adv):.1f}x")

    # Action ranking
    v_broad = np.broadcast_to(v_pred[:, None], (N, K))
    v_mc_broad = np.broadcast_to(v_mc_np[:, None], (N, K))

    rho_both, med_both = spearman_ranking(q_pred - v_broad, mc_adv, valid)
    rho_q_err, med_q_err = spearman_ranking(q_pred - v_mc_broad, mc_adv, valid)
    rho_v_err, med_v_err = spearman_ranking(q_mc_np - v_broad, mc_adv, valid)

    # GAE with learned V (trajectory-based, for reference)
    adv_gae = compute_gae(v_net, trajectories, traj_map, N, K,
                          args.gamma, args.gae_lambda, device).numpy()
    rho_gae, med_gae = spearman_ranking(adv_gae, mc_adv, valid)

    adv_td1 = compute_gae(v_net, trajectories, traj_map, N, K,
                          args.gamma, 0.0, device).numpy()
    rho_td1, med_td1 = spearman_ranking(adv_td1, mc_adv, valid)

    print(f"\n{'='*60}")
    print(f"ACTION RANKING (Spearman ρ vs MC)")
    print(f"{'='*60}")
    print(f"  {'Method':<30} {'mean':>8} {'median':>8}")
    print(f"  {'─'*48}")
    print(f"  {'Q_nn - V_nn':<30} {rho_both:>8.3f} {med_both:>8.3f}")
    print(f"  {'Q_nn - V_mc  (isolate Q)':<30} {rho_q_err:>8.3f} {med_q_err:>8.3f}")
    print(f"  {'Q_mc - V_nn  (isolate V)':<30} {rho_v_err:>8.3f} {med_v_err:>8.3f}")
    td1_label = "TD1: r+γV'-V"
    print(f"  {td1_label:<30} {rho_td1:>8.3f} {med_td1:>8.3f}")
    print(f"  {f'GAE(λ={args.gae_lambda})':<30} {rho_gae:>8.3f} {med_gae:>8.3f}")
