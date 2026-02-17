"""Tune IQL with architecture knobs to match MC/GAE action ranking.

Applies tuning tricks from rank_qv_regression (normalize, scale_factor,
action_repeat, deeper/wider nets, layer_norm, separate LRs) to IQL's
actual TD-based training procedure.

Key question: can these tricks make IQL's TD training match MC regression
or GAE quality for within-state action ranking?

Usage:
  python -m methods.gae.rank_iql_tune
  python -m methods.gae.rank_iql_tune --normalize --scale_factor 20 \
    --hidden_dim 512 --num_layers 10 --action_repeat 8 --epochs 4000
  python -m methods.gae.rank_iql_tune --nstep 10 --normalize
"""

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import tyro
from scipy import stats as sp_stats

from data.offline_dataset import OfflineRLDataset
from methods.gae.rank_iql_debug import compute_gae
from methods.gae.rank_qv_regression import (
    QNet,
    VNet,
    eval_q,
    eval_v,
    spearman_ranking,
    train_net,
)
from methods.iql.iql import compute_nstep_targets, expectile_loss


# ── IQL Training ─────────────────────────────────────────────────────


def train_iql_tuned(
    states, actions, rewards, next_states, terminated,
    state_dim, action_dim, device, args,
    nstep_returns=None, bootstrap_states=None, nstep_discounts=None,
    v_pretrained=None,
):
    """Train IQL Q+V with tuned architecture. Returns (q_net, v_net).

    If v_pretrained is provided, V starts from those weights instead of random.
    """
    use_nstep = nstep_returns is not None
    N = states.shape[0]

    torch.manual_seed(args.seed)
    q_net = QNet(state_dim, action_dim, args.hidden_dim, args.num_layers,
                 args.layer_norm, args.action_repeat).to(device)
    q_target = copy.deepcopy(q_net)
    v_net = VNet(state_dim, args.hidden_dim, args.num_layers,
                 args.layer_norm).to(device)
    if v_pretrained is not None:
        v_net.load_state_dict(v_pretrained.state_dict())
        print("    V initialized from pre-trained weights")

    freeze_v = args.freeze_v
    if freeze_v:
        for p in v_net.parameters():
            p.requires_grad = False
        print("    V is FROZEN (only Q trains)")

    n_params_q = sum(p.numel() for p in q_net.parameters())
    n_params_v = sum(p.numel() for p in v_net.parameters())
    print(f"    Q params: {n_params_q:,}  V params: {n_params_v:,}")

    q_opt = torch.optim.Adam(q_net.parameters(), lr=args.q_lr, eps=1e-5,
                             weight_decay=1e-4)
    if not freeze_v:
        v_opt = torch.optim.Adam(v_net.parameters(), lr=args.v_lr, eps=1e-5,
                                 weight_decay=1e-4)
        v_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            v_opt, T_max=args.epochs, eta_min=1e-5)
    q_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        q_opt, T_max=args.epochs, eta_min=1e-5)

    # Train/val split
    perm = torch.randperm(N)
    val_size = max(1, int(N * 0.1))
    train_idx = perm[val_size:]
    val_idx = perm[:val_size]
    train_size = train_idx.shape[0]

    val_s = states[val_idx].to(device)
    val_a = actions[val_idx].to(device)
    val_r = rewards[val_idx].to(device)
    val_ns = next_states[val_idx].to(device)
    val_term = terminated[val_idx].to(device)
    if use_nstep:
        val_nret = nstep_returns[val_idx].to(device)
        val_boot = bootstrap_states[val_idx].to(device)
        val_disc = nstep_discounts[val_idx].to(device)

    best_val_loss = float("inf")
    best_q_state = None
    best_v_state = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        q_net.train()
        v_net.train()
        indices = train_idx[torch.randperm(train_size)]
        epoch_q, epoch_v, n_batch = 0.0, 0.0, 0

        for start in range(0, train_size, args.batch_size):
            bi = indices[start:start + args.batch_size]
            s = states[bi].to(device)
            a = actions[bi].to(device)
            r = rewards[bi].to(device)
            ns = next_states[bi].to(device)
            term = terminated[bi].to(device)

            # Q loss: TD backup using V
            with torch.no_grad():
                if use_nstep:
                    v_boot = v_net(bootstrap_states[bi].to(device))
                    q_tgt = nstep_returns[bi].to(device) + \
                            nstep_discounts[bi].to(device) * v_boot
                else:
                    v_next = v_net(ns)
                    q_tgt = r + args.gamma * v_next * (1.0 - term)
            q_pred = q_net(s, a)
            q_loss = 0.5 * ((q_pred - q_tgt) ** 2).mean()

            q_opt.zero_grad()
            q_loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 0.5)
            q_opt.step()

            # V loss: expectile regression against target Q
            if not freeze_v:
                with torch.no_grad():
                    q_val = q_target(s, a)
                v_pred = v_net(s)
                v_loss = expectile_loss(q_val - v_pred, args.iql_tau)

                v_opt.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(v_net.parameters(), 0.5)
                v_opt.step()
                epoch_v += v_loss.item()

            # Polyak update target Q
            with torch.no_grad():
                for p, pt in zip(q_net.parameters(), q_target.parameters()):
                    pt.data.mul_(1.0 - args.tau_polyak).add_(
                        p.data, alpha=args.tau_polyak)

            epoch_q += q_loss.item()
            n_batch += 1

        q_sched.step()
        if not freeze_v:
            v_sched.step()
        avg_q = epoch_q / n_batch
        avg_v = epoch_v / n_batch if not freeze_v else 0.0

        # Validation
        q_net.eval()
        with torch.no_grad():
            if use_nstep:
                vb = v_net(val_boot)
                vq_tgt = val_nret + val_disc * vb
            else:
                vq_tgt = val_r + args.gamma * v_net(val_ns) * (1.0 - val_term)
            vq_pred = q_net(val_s, val_a)
            val_q_loss = 0.5 * ((vq_pred - vq_tgt) ** 2).mean().item()

            val_v_loss = 0.0
            if not freeze_v:
                vq_for_v = q_target(val_s, val_a)
                vv_pred = v_net(val_s)
                diff = vq_for_v - vv_pred
                weight = torch.where(diff > 0, args.iql_tau,
                                     1.0 - args.iql_tau)
                val_v_loss = (weight * (diff ** 2)).mean().item()

        val_total = val_q_loss + val_v_loss
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_q_state = {k: v.clone() for k, v in q_net.state_dict().items()}
            best_v_state = {k: v.clone() for k, v in v_net.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"    Early stopping at epoch {epoch}")
                break

        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            print(f"    epoch {epoch}/{args.epochs}: "
                  f"q={avg_q:.6f} v={avg_v:.6f} "
                  f"val_q={val_q_loss:.6f} val_v={val_v_loss:.6f}")

    if best_q_state is not None:
        q_net.load_state_dict(best_q_state)
    if best_v_state is not None:
        v_net.load_state_dict(best_v_state)
    q_net.eval()
    v_net.eval()
    return q_net, v_net


# ── Config ───────────────────────────────────────────────────────────


@dataclass
class Args:
    seed: int = 1
    cuda: bool = True
    gamma: float = 0.8
    gae_lambda: float = 0.95

    cache_path: str = "data/datasets/rank_cache_K8_M10_seed1.pt"
    train_dataset_path: str = "data/datasets/pickcube_expert.pt"
    dataset_num_envs: int = 16

    # Architecture
    hidden_dim: int = 256
    num_layers: int = 3
    layer_norm: bool = False
    action_repeat: int = 1
    """repeat action vector N times in Q input to amplify action signal"""

    # Data processing
    normalize: bool = False
    """normalize states (z-score) and actions (min-max -> [-1,1])"""
    offline_data_path: str = "data/datasets/pickcube_expert.pt"
    """path to offline dataset for computing normalization stats"""
    scale_factor: float = 1.0
    """scale rewards by this factor (amplifies advantage signal)"""

    # IQL-specific
    iql_tau: float = 0.5
    """expectile (0.5 = SARSA, >0.5 = optimistic)"""
    tau_polyak: float = 0.005
    """Polyak averaging rate for target Q network"""
    nstep: int = 1
    """n-step TD return (1 = standard, >1 = multi-step)"""
    patience: int = 100
    """early stopping patience (0 = disabled)"""

    # V pre-training
    pretrain_v: bool = False
    """pre-train V on MC returns before IQL joint training"""
    v_pretrain_epochs: int = 500
    """epochs for V pre-training on MC targets"""
    freeze_v: bool = False
    """freeze V during IQL (only train Q against fixed V targets)"""

    # Training
    epochs: int = 200
    q_lr: float = 3e-4
    v_lr: float = 3e-4
    batch_size: int = 256


# ── Main ─────────────────────────────────────────────────────────────


if __name__ == "__main__":
    args = tyro.cli(Args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ── Load eval cache ───────────────────────────────────────────────
    cache = torch.load(args.cache_path, weights_only=False)
    v_mc = cache["v_mc"]                        # (N,)
    q_mc = cache["q_mc"]                        # (N, K)
    sampled_actions = cache["sampled_actions"]   # (N, K, action_dim)
    rollout_trajs = cache["trajectories"]
    traj_map = cache["traj_to_state_action"]
    N = cache["N"]
    state_dim = cache["state_dim"]
    K = sampled_actions.shape[1]
    action_dim = sampled_actions.shape[2]

    eval_states = cache.get("eval_states")
    if eval_states is None:
        eval_states = OfflineRLDataset(
            ["data/datasets/pickcube_expert_eval.pt"], False, False
        ).state

    # ── Load train dataset + trajectories ─────────────────────────────
    train_ds = OfflineRLDataset([args.train_dataset_path], False, False)
    train_trajs = train_ds.extract_trajectories(
        num_envs=args.dataset_num_envs, gamma=args.gamma)

    # ── Normalization ─────────────────────────────────────────────────
    if args.normalize:
        offline_ds = OfflineRLDataset([args.offline_data_path], False, False)
        s_mean = offline_ds.state.mean(dim=0)
        s_std = offline_ds.state.std(dim=0).clamp(min=1e-6)
        a_min = offline_ds.actions.min(dim=0).values
        a_max = offline_ds.actions.max(dim=0).values
        a_range = (a_max - a_min).clamp(min=1e-6)
        del offline_ds
        print(f"Normalization (from {args.offline_data_path}):")
        print(f"  state: z-score  mean=[{s_mean.min():.3f}, {s_mean.max():.3f}]  "
              f"std=[{s_std.min():.3f}, {s_std.max():.3f}]")
        print(f"  action: min-max  min=[{a_min.min():.3f}, {a_min.max():.3f}]  "
              f"max=[{a_max.min():.3f}, {a_max.max():.3f}]")

        # Eval data
        eval_states = (eval_states - s_mean) / s_std
        sampled_actions = (sampled_actions - a_min) / a_range * 2 - 1

        # Rollout trajectories
        for traj in rollout_trajs:
            traj["states"] = (traj["states"] - s_mean) / s_std
            traj["next_states"] = (traj["next_states"] - s_mean) / s_std
            traj["actions"] = (traj["actions"] - a_min) / a_range * 2 - 1

        # Train trajectories (actions from dataset via flat_indices)
        for traj in train_trajs:
            traj["states"] = (traj["states"] - s_mean) / s_std
            traj["next_states"] = (traj["next_states"] - s_mean) / s_std
            traj["actions"] = (
                train_ds.actions[traj["flat_indices"]] - a_min
            ) / a_range * 2 - 1
    else:
        # Extract actions from dataset into trajectory dicts
        for traj in train_trajs:
            traj["actions"] = train_ds.actions[traj["flat_indices"]]

    del train_ds  # free memory

    # ── Scale rewards ─────────────────────────────────────────────────
    sf = args.scale_factor
    if sf != 1.0:
        v_mc = v_mc * sf
        q_mc = q_mc * sf
        for traj in rollout_trajs:
            traj["rewards"] = traj["rewards"] * sf
        for traj in train_trajs:
            traj["rewards"] = traj["rewards"] * sf
        print(f"Scale factor: {sf}x  (rewards scaled)")

    # ── Flatten training data ─────────────────────────────────────────
    all_trajs = list(train_trajs) + list(rollout_trajs)
    flat_s = torch.cat([t["states"] for t in all_trajs])
    flat_a = torch.cat([t["actions"] for t in all_trajs])
    flat_r = torch.cat([t["rewards"] for t in all_trajs])
    flat_ns = torch.cat([t["next_states"] for t in all_trajs])
    flat_term = torch.cat([t["terminated"] for t in all_trajs])
    n_train = flat_s.shape[0]

    # N-step targets
    nstep_kw = {}
    if args.nstep > 1:
        nret, boot_s, ndisc = compute_nstep_targets(
            all_trajs, args.nstep, args.gamma)
        nstep_kw = dict(nstep_returns=nret, bootstrap_states=boot_s,
                        nstep_discounts=ndisc)

    # ── Print config ──────────────────────────────────────────────────
    mc_adv = (q_mc - v_mc.unsqueeze(1)).numpy()
    valid = np.array([mc_adv[i].std() > 1e-8 for i in range(N)])

    print(f"\nData: {N} eval states, K={K}, {int(valid.sum())} valid, "
          f"{n_train} IQL training transitions")
    print(f"  V:  mean={v_mc.mean():.4f}  std={v_mc.std():.4f}")
    print(f"  Q:  mean={q_mc.mean():.4f}  std={q_mc.std():.4f}")
    print(f"  A:  mean={np.mean(mc_adv):.4f}  std={np.std(mc_adv):.4f}")
    print(f"  SNR: Q_std / A_std = {q_mc.std() / np.std(mc_adv):.1f}x")

    ln_str = ", layer_norm" if args.layer_norm else ""
    ar_str = f", action_repeat={args.action_repeat}" if args.action_repeat > 1 else ""
    norm_str = ", normalized" if args.normalize else ""
    sf_str = f", scale={sf}x" if sf != 1.0 else ""
    ns_str = f", nstep={args.nstep}" if args.nstep > 1 else ""
    pt_str = f", pretrain_v={args.v_pretrain_epochs}ep" if args.pretrain_v else ""
    fz_str = ", freeze_v" if args.freeze_v else ""
    print(f"\nIQL: tau={args.iql_tau}, polyak={args.tau_polyak}{ns_str}, "
          f"patience={args.patience}{pt_str}{fz_str}")
    print(f"Network: hidden={args.hidden_dim}, layers={args.num_layers}"
          f"{ln_str}{ar_str}{norm_str}{sf_str}")
    print(f"Training: epochs={args.epochs}, q_lr={args.q_lr}, "
          f"v_lr={args.v_lr}, batch={args.batch_size}")

    # ── Pre-train V (optional) ────────────────────────────────────────
    v_pre = None
    if args.pretrain_v:
        print(f"\n[V pretrain] Training V on MC returns "
              f"({N} states, {args.v_pretrain_epochs} epochs)")
        torch.manual_seed(args.seed)
        v_pre = VNet(state_dim, args.hidden_dim, args.num_layers,
                     args.layer_norm).to(device)
        train_net(v_pre, eval_states, v_mc, device,
                  args.v_pretrain_epochs, args.v_lr, args.batch_size,
                  label="V_pre")

    # ── Train IQL ─────────────────────────────────────────────────────
    print(f"\n[IQL] Training on {n_train} transitions")
    q_net, v_net = train_iql_tuned(
        flat_s, flat_a, flat_r, flat_ns, flat_term,
        state_dim, action_dim, device, args,
        **nstep_kw, v_pretrained=v_pre)

    # ── Evaluate ──────────────────────────────────────────────────────
    v_pred = eval_v(v_net, eval_states, device)
    q_pred = eval_q(q_net, eval_states, sampled_actions, device)
    v_mc_np = v_mc.numpy()
    q_mc_np = q_mc.numpy()

    # Regression quality
    r_v, _ = sp_stats.pearsonr(v_mc_np, v_pred)
    mae_v = np.mean(np.abs(v_pred - v_mc_np))
    r_q, _ = sp_stats.pearsonr(q_mc_np.flatten(), q_pred.flatten())
    mae_q = np.mean(np.abs(q_pred - q_mc_np))

    q_ps_r = [sp_stats.pearsonr(q_mc_np[i], q_pred[i])[0]
              for i in range(N) if valid[i]]

    print(f"\n{'='*60}")
    print(f"REGRESSION QUALITY (vs MC ground truth)")
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

    # GAE with IQL's V (trajectory-based)
    adv_gae = compute_gae(v_net, rollout_trajs, traj_map, N, K,
                          args.gamma, args.gae_lambda, device).numpy()
    rho_gae, med_gae = spearman_ranking(adv_gae, mc_adv, valid)

    adv_td1 = compute_gae(v_net, rollout_trajs, traj_map, N, K,
                          args.gamma, 0.0, device).numpy()
    rho_td1, med_td1 = spearman_ranking(adv_td1, mc_adv, valid)

    print(f"\n{'='*60}")
    print(f"ACTION RANKING (Spearman ρ vs MC)")
    print(f"{'='*60}")
    print(f"  {'Method':<30} {'mean':>8} {'median':>8}")
    print(f"  {'─'*48}")
    print(f"  {'Q_nn - V_nn  (IQL)':<30} {rho_both:>8.3f} {med_both:>8.3f}")
    print(f"  {'Q_nn - V_mc  (isolate Q)':<30} {rho_q_err:>8.3f} {med_q_err:>8.3f}")
    print(f"  {'Q_mc - V_nn  (isolate V)':<30} {rho_v_err:>8.3f} {med_v_err:>8.3f}")
    td1_label = "TD1: r+γV'-V  (IQL V)"
    print(f"  {td1_label:<30} {rho_td1:>8.3f} {med_td1:>8.3f}")
    gae_label = f"GAE(λ={args.gae_lambda})  (IQL V)"
    print(f"  {gae_label:<30} {rho_gae:>8.3f} {med_gae:>8.3f}")
