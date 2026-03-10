"""
MIP Step Count Ablation on PegInsertionSide-v1.

Tests: does increasing MIP's iteration count from 1→20 monotonically increase SR?
  Option A (shared MLP denoiser): one MLP network, K timesteps
  Option B (independent MLP denoisers): K separate MLP networks
  Option C (shared UNet denoiser): one UNet network, K timesteps
  Option D (independent UNet denoisers): K separate UNet networks

Usage:
    PYTHONPATH=. python -u scripts/mip_step_ablation.py --options A B --k_values 1 2 5 10 20
    PYTHONPATH=. python -u scripts/mip_step_ablation.py --options C D --k_values 1 2 5 10 20
    PYTHONPATH=. python -u scripts/mip_step_ablation.py --k_values 2 --options A --total_iters 50000  # quick test
"""

import os
import copy
import time
import random
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

from diffusers.training_utils import EMAModel as DiffusersEMA
from diffusers.optimization import get_scheduler

from DPPO.dataset import DPPODataset
from MultiGaussian.models.mip_k import (
    MIPKShared, MIPKIndependent,
    MIPKSharedUNet, MIPKIndependentUNet,
)
from MultiGaussian.eval_mip_cpu import evaluate_mip_cpu


OPTION_NAMES = {
    'A': 'MLP shared',
    'B': 'MLP independent',
    'C': 'UNet shared',
    'D': 'UNet independent',
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--k_values', type=int, nargs='+', default=[1, 2, 3, 5, 10, 15, 20])
    p.add_argument('--options', nargs='+', default=['A', 'B'],
                   choices=['A', 'B', 'C', 'D'])
    p.add_argument('--total_iters', type=int, default=100_000)
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--eval_freq', type=int, default=10_000)
    p.add_argument('--log_freq', type=int, default=1000)
    p.add_argument('--n_eval', type=int, default=100)
    p.add_argument('--num_eval_envs', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_dir', type=str, default='runs/mip_step_ablation')

    # MLP architecture (match official_v3 best)
    p.add_argument('--emb_dim', type=int, default=512)
    p.add_argument('--n_layers', type=int, default=6)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--t_star', type=float, default=0.9)

    # Action chunking (match official_v3: cond=2, horizon=16, act=8)
    p.add_argument('--cond_steps', type=int, default=2)
    p.add_argument('--horizon_steps', type=int, default=16)
    p.add_argument('--act_steps', type=int, default=8)

    # Task
    p.add_argument('--env_id', type=str, default='PegInsertionSide-v1')
    p.add_argument('--control_mode', type=str, default='pd_joint_delta_pos')
    p.add_argument('--max_episode_steps', type=int, default=200)
    p.add_argument('--demo_path', type=str, default=(
        '~/.maniskill/demos/PegInsertionSide-v1/policy_from_mp_states/'
        'trajectory.state.pd_joint_delta_pos.h5'))
    return p.parse_args()


def load_data(args, device):
    """Load dataset and preload to GPU."""
    dataset = DPPODataset(
        data_path=os.path.expanduser(args.demo_path),
        horizon_steps=args.horizon_steps, cond_steps=args.cond_steps,
        no_obs_norm=False, no_action_norm=False,
    )
    val_frac = 0.1
    n_val_traj = max(1, int(dataset.num_traj * val_frac))
    n_train_traj = dataset.num_traj - n_val_traj
    train_indices = [i for i, (tj, _) in enumerate(dataset.slices) if tj < n_train_traj]
    val_indices = [i for i, (tj, _) in enumerate(dataset.slices) if tj >= n_train_traj]

    def _preload(indices):
        obs_l, act_l = [], []
        for idx in indices:
            s = dataset[idx]
            obs_l.append(s["cond"]["state"])
            act_l.append(s["actions"])
        obs_t = torch.stack(obs_l).to(device)  # (N, cond_steps, obs_dim)
        act_t = torch.stack(act_l).to(device)  # (N, horizon_steps, action_dim)
        obs_t[..., 9:18] = 0.0  # zero_qvel
        return obs_t, act_t

    print("Pre-loading data...")
    train_obs, train_act = _preload(train_indices)
    val_obs, val_act = _preload(val_indices)
    print(f"  Train: {train_obs.shape}, Val: {val_obs.shape}")
    return train_obs, train_act, val_obs, val_act, dataset


def make_model(option, K, obs_dim, action_dim, args, device):
    """Create model for given option and K."""
    if option in ('A', 'B'):
        # MLP variants
        kw = dict(
            input_dim=obs_dim, action_dim=action_dim,
            cond_steps=args.cond_steps, horizon_steps=args.horizon_steps,
            t_star=args.t_star,
            dropout=args.dropout, emb_dim=args.emb_dim,
            n_layers=args.n_layers,
        )
        if option == 'A':
            model = MIPKShared(K=K, **kw).to(device)
        else:
            model = MIPKIndependent(K=K, **kw).to(device)
    else:
        # UNet variants
        kw = dict(
            input_dim=obs_dim, action_dim=action_dim,
            cond_steps=args.cond_steps, horizon_steps=args.horizon_steps,
            t_star=args.t_star,
        )
        if option == 'C':
            model = MIPKSharedUNet(K=K, **kw).to(device)
        else:
            model = MIPKIndependentUNet(K=K, **kw).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [{option} K={K}] {OPTION_NAMES[option]}, {n_params:,} params")
    return model


def _is_shared(option):
    return option in ('A', 'C')


def train_shared(model, K, train_obs, train_act, val_obs, val_act, dataset, args, device, label):
    """Train shared denoiser (Option A or C)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    lr_sched = get_scheduler('cosine', optimizer=optimizer,
                             num_warmup_steps=500,
                             num_training_steps=args.total_iters)
    ema = DiffusersEMA(parameters=model.parameters(), power=0.75)
    ema_model = copy.deepcopy(model)

    n_train = train_obs.shape[0]
    obs_min = dataset.obs_min.to(device)
    obs_max = dataset.obs_max.to(device)
    action_min = dataset.action_min.to(device)
    action_max = dataset.action_max.to(device)

    best_sr, eval_history = -1.0, []
    t0 = time.time()

    for it in range(1, args.total_iters + 1):
        model.train()
        idx = torch.randint(n_train, (args.batch_size,), device=device)
        loss, _, _ = model.compute_loss(train_obs[idx], train_act[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_sched.step()
        ema.step(model.parameters())

        if it % args.log_freq == 0:
            with torch.no_grad():
                vl, _, _ = model.compute_loss(val_obs, val_act)
            elapsed = time.time() - t0
            print(f"  [{label}] iter {it}: loss={loss.item():.6f}, val={vl.item():.6f}, {elapsed:.0f}s")

        if it % args.eval_freq == 0 or it == args.total_iters:
            ema.copy_to(ema_model.parameters())
            ema_model.eval()
            metrics = evaluate_mip_cpu(
                model=ema_model, device=device,
                n_episodes=args.n_eval, env_id=args.env_id,
                control_mode=args.control_mode,
                max_episode_steps=args.max_episode_steps,
                num_envs=args.num_eval_envs,
                obs_min=obs_min, obs_max=obs_max,
                action_min=action_min, action_max=action_max,
                no_obs_norm=False, no_action_norm=False,
                zero_qvel=True, cond_steps=args.cond_steps,
                horizon_steps=args.horizon_steps, act_steps=args.act_steps,
            )
            sr = metrics["success_once"]
            eval_history.append((it, sr))
            if sr > best_sr:
                best_sr = sr
            print(f"  [{label}] eval @ {it}: SR={sr:.1%} (best={best_sr:.1%})")

    return best_sr, eval_history


def train_independent(model, K, train_obs, train_act, val_obs, val_act, dataset, args, device, label):
    """Train independent denoisers (Option B or D)."""
    # K separate optimizers and EMAs
    optimizers = [
        torch.optim.AdamW(model.denoisers[k].parameters(), lr=args.lr, weight_decay=1e-6)
        for k in range(K)
    ]
    lr_scheds = [
        get_scheduler('cosine', optimizer=optimizers[k],
                      num_warmup_steps=500,
                      num_training_steps=args.total_iters)
        for k in range(K)
    ]
    emas = [
        DiffusersEMA(parameters=model.denoisers[k].parameters(), power=0.75)
        for k in range(K)
    ]
    ema_model = copy.deepcopy(model)

    n_train = train_obs.shape[0]
    obs_min = dataset.obs_min.to(device)
    obs_max = dataset.obs_max.to(device)
    action_min = dataset.action_min.to(device)
    action_max = dataset.action_max.to(device)

    best_sr, eval_history = -1.0, []
    step_counts = [0] * K  # track per-denoiser updates
    t0 = time.time()

    for it in range(1, args.total_iters + 1):
        model.train()
        k = random.randint(0, K - 1)

        idx = torch.randint(n_train, (args.batch_size,), device=device)
        loss = model.compute_loss_k(train_obs[idx], train_act[idx], k)

        optimizers[k].zero_grad()
        loss.backward()
        optimizers[k].step()
        lr_scheds[k].step()
        emas[k].step(model.denoisers[k].parameters())
        step_counts[k] += 1

        if it % args.log_freq == 0:
            elapsed = time.time() - t0
            print(f"  [{label}] iter {it}: loss={loss.item():.6f}, "
                  f"k={k}, steps/denoiser≈{it//K}, {elapsed:.0f}s")

        if it % args.eval_freq == 0 or it == args.total_iters:
            # Copy all EMAs
            for kk in range(K):
                emas[kk].copy_to(ema_model.denoisers[kk].parameters())
            ema_model.eval()
            metrics = evaluate_mip_cpu(
                model=ema_model, device=device,
                n_episodes=args.n_eval, env_id=args.env_id,
                control_mode=args.control_mode,
                max_episode_steps=args.max_episode_steps,
                num_envs=args.num_eval_envs,
                obs_min=obs_min, obs_max=obs_max,
                action_min=action_min, action_max=action_max,
                no_obs_norm=False, no_action_norm=False,
                zero_qvel=True, cond_steps=args.cond_steps,
                horizon_steps=args.horizon_steps, act_steps=args.act_steps,
            )
            sr = metrics["success_once"]
            eval_history.append((it, sr))
            if sr > best_sr:
                best_sr = sr
            print(f"  [{label}] eval @ {it}: SR={sr:.1%} (best={best_sr:.1%})")

    print(f"  [{label}] step_counts per denoiser: {step_counts}")
    return best_sr, eval_history


def plot_results(results, out_dir, options_run):
    """Generate plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    styles = {
        'A': ('o-', 'tab:blue'),
        'B': ('s--', 'tab:orange'),
        'C': ('^-', 'tab:green'),
        'D': ('D--', 'tab:red'),
    }

    # Figure 1: SR vs K
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for option in options_run:
        marker, color = styles[option]
        ks = sorted([k for (opt, k) in results if opt == option])
        srs = [results[(option, k)][0] * 100 for k in ks]
        if ks:
            ax.plot(ks, srs, marker, color=color,
                    label=f'{option}: {OPTION_NAMES[option]}',
                    markersize=8, linewidth=2)

    ax.axhline(y=47, color='gray', linestyle=':', alpha=0.7, label='MIP-2 MLP (47%)')
    ax.axhline(y=77, color='purple', linestyle=':', alpha=0.7, label='DP UNet DDIM-10 (77%)')

    ax.set_xlabel('Number of denoising steps (K)', fontsize=13)
    ax.set_ylabel('Success Rate (%)', fontsize=13)
    ax.set_title('MIP Step Count Ablation on PegInsertionSide', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xticks(sorted(set(k for _, k in results)))
    ax.set_ylim(-2, 100)
    ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, 'step_ablation.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved plot: {path}")
    plt.close(fig)

    # Figure 2: training curves per option
    n_opts = len(options_run)
    fig2, axes = plt.subplots(1, n_opts, figsize=(7 * n_opts, 5))
    if n_opts == 1:
        axes = [axes]
    for ax_idx, option in enumerate(options_run):
        ax = axes[ax_idx]
        for (opt, K), (best_sr, history) in sorted(results.items()):
            if opt != option or not history:
                continue
            iters, srs = zip(*history)
            ax.plot(iters, [s * 100 for s in srs], 'o-', label=f'K={K}', markersize=4)
        ax.set_xlabel('Training iteration')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title(f'{option}: {OPTION_NAMES[option]}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig2.tight_layout()
    path2 = os.path.join(out_dir, 'training_curves.png')
    fig2.savefig(path2, dpi=150)
    print(f"Saved plot: {path2}")
    plt.close(fig2)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Config: cond={args.cond_steps}, horizon={args.horizon_steps}, "
          f"act={args.act_steps}, lr={args.lr}, t_star={args.t_star}")

    train_obs, train_act, val_obs, val_act, dataset = load_data(args, device)
    obs_dim = dataset.obs_dim
    action_dim = dataset.action_dim

    results = {}  # {(option, K): (best_sr, eval_history)}

    for K in args.k_values:
        for option in args.options:
            label = f"{option}_K{K}"
            print(f"\n{'='*60}")
            print(f"Training {label} ({OPTION_NAMES[option]})")
            print(f"{'='*60}")

            model = make_model(option, K, obs_dim, action_dim, args, device)

            if _is_shared(option):
                best_sr, history = train_shared(
                    model, K, train_obs, train_act, val_obs, val_act,
                    dataset, args, device, label)
            else:
                best_sr, history = train_independent(
                    model, K, train_obs, train_act, val_obs, val_act,
                    dataset, args, device, label)

            results[(option, K)] = (best_sr, history)
            print(f"  [{label}] DONE: best SR = {best_sr:.1%}")

            # Free memory
            del model
            torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    header = f"{'K':>4s}"
    for opt in args.options:
        header += f"  {OPTION_NAMES[opt]:>20s}"
    print(header)
    print("-" * len(header))
    for K in args.k_values:
        row = f"{K:>4d}"
        for opt in args.options:
            sr = f"{results[(opt, K)][0]:.1%}" if (opt, K) in results else "—"
            row += f"  {sr:>20s}"
        print(row)

    # Save results
    save_data = {
        f"{opt}_K{K}": {"best_sr": sr, "history": hist}
        for (opt, K), (sr, hist) in results.items()
    }
    json_path = os.path.join(args.out_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {json_path}")

    plot_results(results, args.out_dir, args.options)


if __name__ == "__main__":
    main()
