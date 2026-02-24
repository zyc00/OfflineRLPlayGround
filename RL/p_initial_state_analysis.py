"""Compare P(success|s₀) distributions across initial states for PickCube vs StackCube.

For each task: reset 500 envs → 500 diverse initial states → MC16 from each → P(success|s₀).
If StackCube's distribution is bimodal (clustered at 0 and 1), advantages are uninformative
regardless of MC samples.

Usage:
  python -u -m RL.p_initial_state_analysis
"""

import os
import random
import sys
import time
from dataclasses import dataclass

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from data.data_collection.ppo import Agent


@dataclass
class Args:
    pick_checkpoint: str = "runs/pickcube_ppo/ckpt_76_logstd-1.5.pt"
    stack_checkpoint: str = "runs/stackcube_ppo/ckpt_481.pt"
    gamma: float = 0.8
    num_envs: int = 500
    max_episode_steps: int = 50
    mc_samples: int = 16
    seed: int = 1
    output: str = "runs/p_initial_state_analysis.png"


def _clone_state_cpu(sd):
    if isinstance(sd, dict):
        return {k: _clone_state_cpu(v) for k, v in sd.items()}
    if isinstance(sd, torch.Tensor):
        return sd.cpu().clone()
    return sd


def _expand_state(sd, repeats):
    if isinstance(sd, dict):
        return {k: _expand_state(v, repeats) for k, v in sd.items()}
    if isinstance(sd, torch.Tensor) and sd.dim() > 0:
        return sd.repeat_interleave(repeats, dim=0)
    return sd


def _state_to_device(sd, dev):
    if isinstance(sd, dict):
        return {k: _state_to_device(v, dev) for k, v in sd.items()}
    if isinstance(sd, torch.Tensor):
        return sd.to(dev)
    return sd


def analyze_task(env_id, checkpoint, args, device):
    """Run MC16 from initial states and return per-state P(success) array."""
    E = args.num_envs
    M = args.mc_samples
    num_mc_envs = E * M

    env_kwargs = dict(
        obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda",
        reward_mode="sparse", control_mode="pd_joint_delta_pos",
        max_episode_steps=args.max_episode_steps,
    )

    # Create envs for reset
    print(f"\n{'='*60}")
    print(f"  {env_id} — checkpoint: {os.path.basename(checkpoint)}")
    print(f"{'='*60}")

    envs = gym.make(env_id, num_envs=E, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(envs, E, ignore_terminations=False, record_metrics=False)
    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(checkpoint, map_location=device))
    agent.eval()

    def clip_action(a):
        return torch.clamp(a.detach(), action_low, action_high)

    # Step 1: Reset → save initial states
    print(f"  Resetting {E} envs...")
    sys.stdout.flush()
    envs.reset(seed=args.seed)
    initial_state = _clone_state_cpu(envs.base_env.get_state_dict())
    envs.close()
    del envs

    # Step 2: Create MC envs
    print(f"  Creating MC envs ({num_mc_envs})...")
    sys.stdout.flush()
    mc_envs_raw = gym.make(env_id, num_envs=num_mc_envs, **env_kwargs)
    if isinstance(mc_envs_raw.action_space, gym.spaces.Dict):
        mc_envs_raw = FlattenActionSpaceWrapper(mc_envs_raw)
    mc_envs = ManiSkillVectorEnv(mc_envs_raw, num_mc_envs, ignore_terminations=False, record_metrics=False)
    mc_zero_action = torch.zeros(num_mc_envs, *mc_envs.single_action_space.shape, device=device)

    # Need a new agent for mc_envs (different env instance)
    mc_agent = Agent(mc_envs).to(device)
    mc_agent.load_state_dict(torch.load(checkpoint, map_location=device))
    mc_agent.eval()

    mc_action_low = torch.from_numpy(mc_envs.single_action_space.low).to(device)
    mc_action_high = torch.from_numpy(mc_envs.single_action_space.high).to(device)

    def mc_clip_action(a):
        return torch.clamp(a.detach(), mc_action_low, mc_action_high)

    # Step 3: Expand initial state × mc_samples and rollout
    print(f"  Running MC{M} rollouts from {E} initial states...")
    sys.stdout.flush()
    t0 = time.time()

    expanded = _expand_state(_state_to_device(initial_state, device), M)

    # Restore state
    mc_envs.reset(seed=args.seed + 999)
    mc_envs.base_env.set_state_dict(expanded)
    mc_envs.base_env.step(mc_zero_action)
    mc_envs.base_env.set_state_dict(expanded)
    mc_envs.base_env._elapsed_steps[:] = 0
    mc_obs = mc_envs.base_env.get_obs()

    env_done = torch.zeros(num_mc_envs, device=device).bool()
    all_rews = []

    with torch.no_grad():
        for step in range(args.max_episode_steps):
            if env_done.all():
                break
            a = mc_agent.get_action(mc_obs, deterministic=False)
            mc_obs, rew, term, trunc, _ = mc_envs.step(mc_clip_action(a))
            all_rews.append(rew.view(-1) * (~env_done).float())
            env_done = env_done | (term | trunc).view(-1).bool()

    elapsed = time.time() - t0
    print(f"  MC rollout done ({elapsed:.1f}s, {len(all_rews)} steps)")

    # Step 4: Compute per-initial-state P(success) and MC return std
    total_rew = sum(all_rews)  # (num_mc_envs,)
    success_per_rollout = (total_rew > 0.5).float().view(E, M)
    p_success = success_per_rollout.mean(dim=1).cpu().numpy()  # (E,)

    # Discounted return per rollout
    ret = torch.zeros(num_mc_envs, device=device)
    for s in reversed(range(len(all_rews))):
        ret = all_rews[s] + args.gamma * ret
    ret_per_state = ret.view(E, M)
    mc_return_mean = ret_per_state.mean(dim=1).cpu().numpy()
    mc_return_std = ret_per_state.std(dim=1).cpu().numpy()

    mc_envs.close()
    del mc_envs, mc_envs_raw, mc_agent
    torch.cuda.empty_cache()

    # Step 5: Compute statistics
    overall_sr = p_success.mean()
    frac_zero = (p_success == 0.0).mean()
    frac_one = (p_success == 1.0).mean()
    frac_decisive = ((p_success > 0.1) & (p_success < 0.9)).mean()
    mean_mc_std = mc_return_std.mean()

    print(f"\n  Results for {env_id}:")
    print(f"    Overall SR:      {overall_sr*100:.1f}%")
    print(f"    frac_zero (P=0): {frac_zero*100:.1f}%")
    print(f"    frac_one  (P=1): {frac_one*100:.1f}%")
    print(f"    frac_decisive:   {frac_decisive*100:.1f}%  (0.1 < P < 0.9)")
    print(f"    mean MC std:     {mean_mc_std:.4f}")
    print(f"    P(success) mean: {p_success.mean():.4f}, std: {p_success.std():.4f}")
    print(f"    MC return  mean: {mc_return_mean.mean():.4f}, std across states: {mc_return_mean.std():.4f}")

    return dict(
        p_success=p_success,
        mc_return_mean=mc_return_mean,
        mc_return_std=mc_return_std,
        overall_sr=overall_sr,
        frac_zero=frac_zero,
        frac_one=frac_one,
        frac_decisive=frac_decisive,
        mean_mc_std=mean_mc_std,
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tasks = [
        ("PickCube-v1", args.pick_checkpoint, "PickCube"),
        ("StackCube-v1", args.stack_checkpoint, "StackCube"),
    ]

    results = {}
    for env_id, ckpt, name in tasks:
        results[name] = analyze_task(env_id, ckpt, args, device)

    # ── Comparison table ──
    print(f"\n{'='*70}")
    print(f"  COMPARISON: Initial State P(success|s₀) Distribution")
    print(f"{'='*70}")
    print(f"{'Metric':<25} | {'PickCube':>12} | {'StackCube':>12}")
    print(f"{'-'*25}-+-{'-'*12}-+-{'-'*12}")
    for key, label in [
        ("overall_sr", "Overall SR"),
        ("frac_zero", "frac_zero (P=0)"),
        ("frac_one", "frac_one (P=1)"),
        ("frac_decisive", "frac_decisive"),
        ("mean_mc_std", "mean MC std"),
    ]:
        v1 = results["PickCube"][key]
        v2 = results["StackCube"][key]
        if key == "mean_mc_std":
            print(f"{label:<25} | {v1:>12.4f} | {v2:>12.4f}")
        else:
            print(f"{label:<25} | {v1*100:>11.1f}% | {v2*100:>11.1f}%")
    # P stats
    for name in ["PickCube", "StackCube"]:
        r = results[name]
        p = r["p_success"]
        print(f"\n  {name} P distribution: mean={p.mean():.3f}, std={p.std():.3f}, "
              f"median={np.median(p):.3f}, min={p.min():.3f}, max={p.max():.3f}")

    # ── Plot (1×2) ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (name, color) in enumerate([("PickCube", "steelblue"), ("StackCube", "coral")]):
        ax = axes[idx]
        r = results[name]
        p = r["p_success"]

        # Histogram with 17 bins (0, 1/16, 2/16, ..., 16/16)
        bins = np.linspace(0, 1, 18)
        ax.hist(p, bins=bins, color=color, edgecolor="black", linewidth=0.5, alpha=0.8)
        ax.axvline(p.mean(), color="black", ls="--", lw=2, label=f"mean SR={p.mean()*100:.1f}%")
        ax.axvline(0.5, color="gray", ls=":", lw=1.5, alpha=0.6)

        # Annotations
        textstr = (
            f"frac_zero (P=0): {r['frac_zero']*100:.1f}%\n"
            f"frac_one  (P=1): {r['frac_one']*100:.1f}%\n"
            f"frac_decisive:   {r['frac_decisive']*100:.1f}%\n"
            f"overall SR:      {r['overall_sr']*100:.1f}%"
        )
        ax.text(0.98, 0.97, textstr, transform=ax.transAxes,
                fontsize=9, verticalalignment="top", horizontalalignment="right",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))

        ax.set_xlabel("P(success|s₀)", fontsize=11)
        ax.set_ylabel("Count (out of 500 initial states)", fontsize=11)
        ax.set_title(f"{name}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xlim(-0.03, 1.03)

    fig.suptitle(
        f"Initial State Success Rate Distribution — MC{args.mc_samples}, gamma={args.gamma}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {args.output}")

    # Save data
    save_path = args.output.replace(".png", ".pt")
    torch.save(dict(results=results, args=vars(args)), save_path)
    print(f"Saved data: {save_path}")
