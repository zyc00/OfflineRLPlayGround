"""Does a larger network fix Q/V regression for action ranking?

Directly regresses Q(s,a) on MC-Q and V(s) on MC-V at different network
widths, then isolates whether Q or V error breaks the ranking:

  A = Q_nn - V_nn    (both learned — realistic)
  A = Q_nn - V_mc    (perfect V, learned Q — isolates Q error)
  A = Q_mc - V_nn    (perfect Q, learned V — isolates V error)

Also tests Q-V time accumulation: instead of single-step Q(s,a)-V(s),
accumulate along trajectories like GAE does with TD errors:
  A_accum = Σ_t (γλ)^t [Q_nn(s_t, a_t) - V_nn(s_t)]
vs standard GAE:
  A_gae   = Σ_t (γλ)^t [r_t + γV(s_{t+1})(1-term) - V(s_t)]

Usage:
  python -m methods.gae.rank_qv_size
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


@dataclass
class Args:
    seed: int = 1
    cuda: bool = True
    gamma: float = 0.8

    cache_path: str = "data/datasets/rank_cache_K8_M10_seed1.pt"

    hidden_dims: tuple[int, ...] = (256, 512, 1024)
    epochs: int = 200
    lr: float = 3e-4
    batch_size: int = 256
    gae_lambda: float = 0.95


class VNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1)),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim + action_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1)),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


def train_v(states, targets, state_dim, hidden_dim, device, args):
    """Train V(s) on MC-V targets."""
    N = states.shape[0]
    net = VNet(state_dim, hidden_dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    for epoch in range(1, args.epochs + 1):
        idx = torch.randperm(N)
        total_loss, n_batch = 0.0, 0
        net.train()
        for start in range(0, N, args.batch_size):
            bi = idx[start:start + args.batch_size]
            pred = net(states[bi].to(device))
            loss = 0.5 * ((pred - targets[bi].to(device)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            total_loss += loss.item()
            n_batch += 1
        sched.step()
        if epoch == 1 or epoch % 50 == 0:
            print(f"      Epoch {epoch}/{args.epochs}: loss={total_loss / n_batch:.6f}")
    net.eval()
    return net


def train_q(states, actions, targets, state_dim, action_dim, hidden_dim, device, args):
    """Train Q(s,a) on MC-Q targets."""
    N = states.shape[0]
    net = QNet(state_dim, action_dim, hidden_dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    for epoch in range(1, args.epochs + 1):
        idx = torch.randperm(N)
        total_loss, n_batch = 0.0, 0
        net.train()
        for start in range(0, N, args.batch_size):
            bi = idx[start:start + args.batch_size]
            pred = net(states[bi].to(device), actions[bi].to(device))
            loss = 0.5 * ((pred - targets[bi].to(device)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            total_loss += loss.item()
            n_batch += 1
        sched.step()
        if epoch == 1 or epoch % 50 == 0:
            print(f"      Epoch {epoch}/{args.epochs}: loss={total_loss / n_batch:.6f}")
    net.eval()
    return net


@torch.no_grad()
def compute_qv_accum(q_net, v_net, trajectories, traj_map, N, K,
                     gamma, lam, device):
    """Accumulate Q(s_t,a_t) - V(s_t) along trajectories like GAE.

    For each trajectory starting at (s_i, a_k):
        δ_t = Q_nn(s_t, a_t) - V_nn(s_t)
        A_t = δ_t + γλ(1-done_t) * A_{t+1}   (backward)
        -> return A_0
    """
    q_net.eval()
    v_net.eval()

    # Batch compute Q and V for all trajectory steps
    all_s = torch.cat([t["states"] for t in trajectories])
    all_a = torch.cat([t["actions"] for t in trajectories])

    # Evaluate in batches
    all_qv_delta = []
    for start in range(0, all_s.shape[0], 4096):
        end = min(start + 4096, all_s.shape[0])
        s = all_s[start:end].to(device)
        a = all_a[start:end].to(device)
        q = q_net(s, a)
        v = v_net(s)
        all_qv_delta.append((q - v).cpu())
    all_qv_delta = torch.cat(all_qv_delta)

    adv_sum = torch.zeros(N, K)
    counts = torch.zeros(N, K)

    offset = 0
    for i, traj in enumerate(trajectories):
        T = traj["states"].shape[0]
        delta = all_qv_delta[offset:offset + T]
        dones = traj["dones"]
        offset += T

        # GAE-style backward accumulation of Q-V deltas
        gae_val = 0.0
        for t in reversed(range(T)):
            gae_val = delta[t] + gamma * lam * (1.0 - dones[t]) * gae_val
        # gae_val is now A_0

        si, ai = traj_map[i]
        adv_sum[si, ai] += gae_val.item()
        counts[si, ai] += 1

    return adv_sum / counts.clamp(min=1)


@torch.no_grad()
def eval_ranking(q_vals, v_vals, mc_adv, valid):
    """Compute per-state Spearman rho of (q - v) vs mc_adv."""
    adv = q_vals - v_vals
    rhos = []
    for i in range(mc_adv.shape[0]):
        if not valid[i]:
            continue
        rho, _ = sp_stats.spearmanr(mc_adv[i], adv[i])
        rhos.append(rho)
    rhos = np.array(rhos)
    return np.nanmean(rhos), np.nanmedian(rhos)


if __name__ == "__main__":
    args = tyro.cli(Args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Load cache
    print("Loading cache...")
    cache = torch.load(args.cache_path, weights_only=False)
    v_mc = cache["v_mc"]                        # (N,)
    q_mc = cache["q_mc"]                        # (N, K)
    sampled_actions = cache["sampled_actions"]   # (N, K, action_dim)
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

    trajectories = cache["trajectories"]
    traj_map = cache["traj_to_state_action"]

    mc_adv = (q_mc - v_mc.unsqueeze(1)).numpy()
    valid = np.array([mc_adv[i].std() > 1e-8 for i in range(N)])
    n_valid = int(valid.sum())
    print(f"  {N} states, K={K}, {n_valid} valid, {len(trajectories)} trajectories")
    print(f"  MC Q: mean={q_mc.mean():.4f}, std={q_mc.std():.4f}")
    print(f"  MC V: mean={v_mc.mean():.4f}, std={v_mc.std():.4f}")
    print(f"  MC A: mean={np.mean(mc_adv):.4f}, std={np.std(mc_adv):.4f}")

    # Flatten for Q training: (N*K) samples
    flat_states = eval_states.unsqueeze(1).expand(-1, K, -1).reshape(-1, state_dim)
    flat_actions = sampled_actions.reshape(-1, action_dim)
    flat_q_targets = q_mc.reshape(-1)

    # V training: (N) samples
    v_states = eval_states
    v_targets = v_mc

    summary = {}

    for hdim in args.hidden_dims:
        print(f"\n{'=' * 60}")
        print(f"HIDDEN DIM = {hdim}")
        print(f"{'=' * 60}")

        # ── Train V(s) on MC-V ────────────────────────────────────
        print(f"\n  [V] Training on MC-V ({N} samples):")
        torch.manual_seed(args.seed)
        v_net = train_v(v_states, v_targets, state_dim, hdim, device, args)

        # V error
        with torch.no_grad():
            v_pred = v_net(eval_states.to(device)).cpu().numpy()
        v_mc_np = v_mc.numpy()
        r_v, _ = sp_stats.pearsonr(v_mc_np, v_pred)
        mae_v = np.mean(np.abs(v_pred - v_mc_np))
        v_per_state_err = np.abs(v_pred - v_mc_np)
        print(f"    V Pearson r={r_v:.4f}, MAE={mae_v:.6f}, "
              f"max_err={v_per_state_err.max():.4f}")

        # ── Train Q(s,a) on MC-Q ──────────────────────────────────
        print(f"\n  [Q] Training on MC-Q ({N * K} samples):")
        torch.manual_seed(args.seed)
        q_net = train_q(flat_states, flat_actions, flat_q_targets,
                        state_dim, action_dim, hdim, device, args)

        # Q error
        with torch.no_grad():
            q_pred = torch.zeros(N, K)
            for i in range(0, N, 512):
                j = min(i + 512, N)
                s = eval_states[i:j].to(device)
                for k in range(K):
                    a = sampled_actions[i:j, k].to(device)
                    q_pred[i:j, k] = q_net(s, a).cpu()
        q_mc_np = q_mc.numpy()
        q_pred_np = q_pred.numpy()
        r_q, _ = sp_stats.pearsonr(q_mc_np.flatten(), q_pred_np.flatten())
        mae_q = np.mean(np.abs(q_pred_np - q_mc_np))

        # Per-state Q error: for each state, how well does Q rank actions?
        q_per_state_r = []
        for i in range(N):
            if valid[i]:
                rho, _ = sp_stats.pearsonr(q_mc_np[i], q_pred_np[i])
                q_per_state_r.append(rho)
        q_per_state_r = np.array(q_per_state_r)
        print(f"    Q Pearson r(pooled)={r_q:.4f}, MAE={mae_q:.6f}")
        print(f"    Q per-state Pearson r: mean={np.nanmean(q_per_state_r):.4f}, "
              f"med={np.nanmedian(q_per_state_r):.4f}")

        # ── Ranking: Q_nn - V_nn ──────────────────────────────────
        v_for_rank = np.broadcast_to(v_pred[:, None], (N, K))
        rho_both, med_both = eval_ranking(
            q_pred_np, v_for_rank, mc_adv, valid,
        )

        # ── Ranking: Q_nn - V_mc (isolate Q error) ───────────────
        v_mc_broad = np.broadcast_to(v_mc_np[:, None], (N, K))
        rho_q_only, med_q_only = eval_ranking(
            q_pred_np, v_mc_broad, mc_adv, valid,
        )

        # ── Ranking: Q_mc - V_nn (isolate V error) ───────────────
        rho_v_only, med_v_only = eval_ranking(
            q_mc_np, v_for_rank, mc_adv, valid,
        )

        # ── Ranking: Q_mc - V_mc (oracle, should be 1.0) ─────────
        rho_oracle, med_oracle = eval_ranking(
            q_mc_np, v_mc_broad, mc_adv, valid,
        )

        # ── Q-V time accumulation ─────────────────────────────────
        # Accumulate Q_nn(s_t,a_t) - V_nn(s_t) along trajectories
        print(f"\n  [Q-V accum] Accumulating Q_nn-V_nn along trajectories...")
        adv_qv_accum = compute_qv_accum(
            q_net, v_net, trajectories, traj_map, N, K,
            args.gamma, args.gae_lambda, device,
        )
        rho_qv_accum, med_qv_accum = eval_ranking(
            adv_qv_accum.numpy(),
            np.zeros_like(adv_qv_accum.numpy()),  # V already subtracted
            mc_adv, valid,
        )

        # ── Standard GAE with V_nn (for comparison) ──────────────
        adv_gae = compute_gae(
            v_net, trajectories, traj_map, N, K,
            args.gamma, args.gae_lambda, device,
        )
        rho_gae, med_gae = eval_ranking(
            adv_gae.numpy(),
            np.zeros_like(adv_gae.numpy()),
            mc_adv, valid,
        )

        # ── TD1 with V_nn ────────────────────────────────────────
        adv_td1 = compute_gae(
            v_net, trajectories, traj_map, N, K,
            args.gamma, 0.0, device,
        )
        rho_td1, med_td1 = eval_ranking(
            adv_td1.numpy(),
            np.zeros_like(adv_td1.numpy()),
            mc_adv, valid,
        )

        print(f"\n  Ranking (Spearman rho vs MC):")
        print(f"    {'Method':<30} {'mean':>8} {'median':>8}")
        print(f"    {'─' * 48}")
        print(f"    {'Q_mc - V_mc (oracle)':<30} {rho_oracle:>8.3f} {med_oracle:>8.3f}")
        print(f"    {'Q_nn - V_nn (single step)':<30} {rho_both:>8.3f} {med_both:>8.3f}")
        print(f"    {'Q_nn - V_mc (Q error only)':<30} {rho_q_only:>8.3f} {med_q_only:>8.3f}")
        print(f"    {'Q_mc - V_nn (V error only)':<30} {rho_v_only:>8.3f} {med_v_only:>8.3f}")
        print(f"    {'TD1: r+γV(s\')-V(s)':<30} {rho_td1:>8.3f} {med_td1:>8.3f}")
        print(f"    {'GAE(λ=0.95) with V_nn':<30} {rho_gae:>8.3f} {med_gae:>8.3f}")
        print(f"    {'Q-V accum (λ=0.95)':<30} {rho_qv_accum:>8.3f} {med_qv_accum:>8.3f}")

        summary[hdim] = {
            "V_r": r_v, "V_mae": mae_v,
            "Q_r": r_q, "Q_mae": mae_q,
            "Q_per_state_r": np.nanmean(q_per_state_r),
            "both": rho_both, "q_only": rho_q_only,
            "v_only": rho_v_only, "oracle": rho_oracle,
            "td1": rho_td1, "gae": rho_gae, "qv_accum": rho_qv_accum,
        }

        del v_net, q_net
        torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────
    dims = list(summary.keys())

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    print(f"\n  Regression quality:")
    print(f"    {'dim':<8} {'V_r':>8} {'V_MAE':>8} {'Q_r':>8} {'Q_MAE':>8} {'Q_per_s':>8}")
    print(f"    {'─' * 48}")
    for d in dims:
        s = summary[d]
        print(f"    {d:<8} {s['V_r']:>8.4f} {s['V_mae']:>8.4f} "
              f"{s['Q_r']:>8.4f} {s['Q_mae']:>8.4f} {s['Q_per_state_r']:>8.4f}")

    print(f"\n  Action ranking (Spearman rho vs MC):")
    print(f"    {'dim':<8} {'oracle':>8} {'Q-V':>8} {'Q_nn-V_mc':>10} "
          f"{'Q_mc-V_nn':>10} {'TD1':>8} {'GAE':>8} {'QV_acc':>8}")
    print(f"    {'─' * 72}")
    for d in dims:
        s = summary[d]
        print(f"    {d:<8} {s['oracle']:>8.3f} {s['both']:>8.3f} "
              f"{s['q_only']:>10.3f} {s['v_only']:>10.3f} "
              f"{s['td1']:>8.3f} {s['gae']:>8.3f} {s['qv_accum']:>8.3f}")

    print(f"\n  Key comparison:")
    print(f"    GAE:    accumulate r+γV(s')-V(s) over trajectory → uses V only")
    print(f"    QV_acc: accumulate Q(s,a)-V(s) over trajectory  → uses Q+V")
    print(f"    If QV_acc ≈ GAE → Q-V carries same info as TD error when accumulated")
    print(f"    If QV_acc ≈ Q-V → accumulation doesn't help Q-V noise")
