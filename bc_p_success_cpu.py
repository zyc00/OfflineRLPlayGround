"""P(success|s₀) distribution for BC policies on PickCube-v1, physx_cpu.

Supports MC sampling with action noise to analyze how noise affects
the bimodality of deterministic BC policies.

Usage:
  # Deterministic (MC1 sufficient)
  python bc_p_success_cpu.py --ckpt runs/lff_fix_mlp2_s1/checkpoints/final.pt

  # With noise sweep (MC16 needed)
  python bc_p_success_cpu.py --ckpt runs/lff_fix_mlp2_s1/checkpoints/final.pt \
    --noise-levels 0.0 0.01 0.03 0.05 0.1 0.2 --mc-samples 16

  # Custom control mode
  python bc_p_success_cpu.py --ckpt runs/bc_rl_joint_mlp2_s1/checkpoints/final.pt \
    --control-mode pd_joint_delta_pos --mc-samples 16 --noise-levels 0.0 0.05
"""
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tyro
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers import CPUGymWrapper


@dataclass
class Args:
    ckpt: str = "runs/lff_fix_mlp2_s1/checkpoints/final.pt"
    """path to BC checkpoint (must have 'actor' key with actor_mean.* weights)"""
    control_mode: str = "pd_ee_delta_pos"
    """control mode for the environment"""
    num_states: int = 500
    """number of initial states to evaluate"""
    mc_samples: int = 1
    """MC samples per initial state (1 for deterministic, 16+ for stochastic)"""
    noise_levels: List[float] = field(default_factory=lambda: [0.0])
    """action noise std levels to sweep"""
    max_episode_steps: int = 100
    """max steps per episode"""
    seed: int = 0
    """starting seed for initial states (states use seeds 0..num_states-1)"""
    output: Optional[str] = None
    """output plot path (default: runs/bc_p_success_cpu.png)"""


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        return self.actor_mean(x)


def run_mc(actor, env, num_states, mc_samples, noise_std, max_steps, device, seed_offset=0):
    """Run MC rollouts and return per-state P(success) array."""
    p_success = np.zeros(num_states)
    t0 = time.time()

    with torch.no_grad():
        for i in range(num_states):
            mc_success = 0
            for mc in range(mc_samples):
                obs, _ = env.reset(seed=seed_offset + i)
                for step in range(max_steps):
                    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                    action = actor(obs_t)
                    if noise_std > 0:
                        action = action + noise_std * torch.randn_like(action)
                    obs, rew, term, trunc, _ = env.step(action.cpu().numpy().squeeze(0))
                    if rew > 0.5:
                        mc_success += 1
                        break
                    if term or trunc:
                        break
            p_success[i] = mc_success / mc_samples

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                sr = p_success[:i+1].mean()
                print(f"  {i+1}/{num_states} done ({elapsed:.0f}s), running SR={sr:.1%}", flush=True)

    return p_success


if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create env
    env = gym.make("PickCube-v1", obs_mode="state", render_mode="rgb_array",
                   reward_mode="sparse", control_mode=args.control_mode,
                   max_episode_steps=args.max_episode_steps, reconfiguration_freq=1)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = CPUGymWrapper(env, ignore_terminations=False, record_metrics=True)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Load checkpoint
    actor = Actor(obs_dim, act_dim).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    # Filter out actor_logstd if present (BC checkpoints may include it)
    sd = {k: v for k, v in ckpt["actor"].items() if k.startswith("actor_mean.")}
    actor.load_state_dict(sd)
    actor.eval()
    print(f"Loaded: {args.ckpt}")
    print(f"Env: PickCube-v1, {args.control_mode}, max_steps={args.max_episode_steps}")
    print(f"States: {args.num_states}, MC: {args.mc_samples}, Noise: {args.noise_levels}")
    print()

    # Run for each noise level
    results = {}
    for noise_std in args.noise_levels:
        label = f"noise={noise_std:.3f}"
        print(f"{'='*60}")
        print(f"  {label} (MC{args.mc_samples}, {args.num_states} states)")
        print(f"{'='*60}")

        p = run_mc(actor, env, args.num_states, args.mc_samples, noise_std,
                   args.max_episode_steps, device, seed_offset=args.seed)
        results[noise_std] = p

        sr = p.mean()
        fz = (p == 0).mean()
        fo = (p == 1).mean()
        fd = ((p > 0.1) & (p < 0.9)).mean()
        print(f"  SR={sr:.1%} | frac_zero={fz:.1%} | frac_one={fo:.1%} | frac_decisive={fd:.1%}")
        print()

    env.close()

    # Summary table
    print(f"{'='*70}")
    print(f"  Summary: P(success|s₀) distribution by noise level")
    print(f"{'='*70}")
    print(f"{'noise_std':>10} | {'SR':>7} | {'frac_zero':>10} | {'frac_one':>10} | {'frac_decisive':>14}")
    print(f"{'-'*10}-+-{'-'*7}-+-{'-'*10}-+-{'-'*10}-+-{'-'*14}")
    for noise_std, p in results.items():
        sr = p.mean()
        fz = (p == 0).mean()
        fo = (p == 1).mean()
        fd = ((p > 0.1) & (p < 0.9)).mean()
        print(f"{noise_std:>10.3f} | {sr:>6.1%} | {fz:>9.1%} | {fo:>9.1%} | {fd:>13.1%}")

    # Plot
    n_plots = len(results)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), squeeze=False)
    axes = axes[0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_plots))

    for idx, (noise_std, p) in enumerate(results.items()):
        ax = axes[idx]
        bins = np.linspace(0, 1, 18)  # 17 bins for MC16
        ax.hist(p, bins=bins, color=colors[idx], edgecolor="black", linewidth=0.5, alpha=0.8)
        ax.axvline(p.mean(), color="red", ls="--", lw=2, label=f"SR={p.mean():.1%}")

        fz = (p == 0).mean()
        fo = (p == 1).mean()
        fd = ((p > 0.1) & (p < 0.9)).mean()
        textstr = (f"frac_zero:     {fz:.1%}\n"
                   f"frac_one:      {fo:.1%}\n"
                   f"frac_decisive: {fd:.1%}\n"
                   f"SR:            {p.mean():.1%}")
        ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", horizontalalignment="right",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))
        ax.set_xlabel("P(success|s₀)")
        ax.set_ylabel(f"Count (out of {args.num_states})")
        ax.set_title(f"noise_std={noise_std:.3f}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xlim(-0.03, 1.03)

    fig.suptitle(f"BC P(success|s₀) — MC{args.mc_samples}, physx_cpu, {args.control_mode}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = args.output or "runs/bc_p_success_cpu.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {out_path}")
