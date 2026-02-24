"""Quick sweep: measure within-state MC std for different logstd values on StackCube.

Finds the logstd that matches PickCube's within-state std (~0.33).

Usage:
  python -u -m RL.mc_within_std_sweep
"""

import random
import sys
import time
from dataclasses import dataclass, field

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import tyro
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from data.data_collection.ppo import Agent


@dataclass
class Args:
    base_checkpoint: str = "runs/stackcube_ppo/ckpt_481.pt"
    logstd_values: tuple[float, ...] = (-0.5, -0.7, -1.0, -1.25, -1.5)
    gamma: float = 0.99
    num_envs: int = 100
    num_steps: int = 50
    max_episode_steps: int = 50
    mc_samples: int = 16
    seed: int = 1
    env_id: str = "StackCube-v1"
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"


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


if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda")
    T, E = args.num_steps, args.num_envs

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_kwargs = dict(
        obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda",
        reward_mode=args.reward_mode, control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
    )

    # Create envs
    envs = gym.make(args.env_id, num_envs=E, **env_kwargs)
    envs = ManiSkillVectorEnv(envs, E, ignore_terminations=False, record_metrics=True)
    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(a):
        return torch.clamp(a.detach(), action_low, action_high)

    # MC envs
    num_mc_envs = E * args.mc_samples
    mc_envs_raw = gym.make(args.env_id, num_envs=num_mc_envs, **env_kwargs)
    mc_envs = ManiSkillVectorEnv(mc_envs_raw, num_mc_envs, ignore_terminations=False, record_metrics=False)
    mc_zero_action = torch.zeros(num_mc_envs, *envs.single_action_space.shape, device=device)

    def _restore_mc_state(sd, seed=None):
        mc_envs.reset(seed=seed if seed is not None else args.seed)
        mc_envs.base_env.set_state_dict(sd)
        mc_envs.base_env.step(mc_zero_action)
        mc_envs.base_env.set_state_dict(sd)
        mc_envs.base_env._elapsed_steps[:] = 0
        return mc_envs.base_env.get_obs()

    # Load base checkpoint
    base_ckpt = torch.load(args.base_checkpoint, map_location=device)

    agent = Agent(envs).to(device)

    print(f"{'='*70}")
    print(f"Within-state MC std sweep for {args.env_id}")
    print(f"Base checkpoint: {args.base_checkpoint}")
    print(f"logstd values: {args.logstd_values}")
    print(f"MC samples: {args.mc_samples}, gamma: {args.gamma}")
    print(f"{'='*70}\n")

    results = []

    for logstd_val in args.logstd_values:
        t0 = time.time()
        print(f"─── logstd = {logstd_val:.2f} (action std = {np.exp(logstd_val):.4f}) ───")

        # Load checkpoint and override logstd
        ckpt = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in base_ckpt.items()}
        ckpt["actor_logstd"] = torch.full_like(ckpt["actor_logstd"], logstd_val)
        agent.load_state_dict(ckpt)
        agent.eval()

        # Phase 1: Collect 1 rollout
        states = [None] * T
        total_successes = 0
        total_episodes = 0

        next_obs, _ = envs.reset(seed=args.seed)
        next_done = torch.zeros(E, device=device)

        with torch.no_grad():
            for step in range(T):
                states[step] = _clone_state_cpu(envs.base_env.get_state_dict())
                action = agent.get_action(next_obs, deterministic=False)
                next_obs, reward, term, trunc, info = envs.step(clip_action(action))
                done = (term | trunc).view(-1)
                total_successes += reward.view(-1)[done.bool()].sum().item()
                total_episodes += done.sum().item()
                next_done = done.float()

        sr = total_successes / max(total_episodes, 1) * 100

        # Phase 2: MC16 on all states
        mc_returns = torch.zeros(T, E, args.mc_samples, device=device)

        with torch.no_grad():
            for t in range(T):
                expanded = _expand_state(_state_to_device(states[t], device), args.mc_samples)
                mc_obs = _restore_mc_state(expanded, seed=args.seed + 1000 + t)
                env_done = torch.zeros(num_mc_envs, device=device).bool()
                all_rews = []
                for _ in range(args.max_episode_steps):
                    if env_done.all():
                        break
                    a = agent.get_action(mc_obs, deterministic=False)
                    mc_obs, rew, term, trunc, _ = mc_envs.step(clip_action(a))
                    all_rews.append(rew.view(-1) * (~env_done).float())
                    env_done = env_done | (term | trunc).view(-1).bool()
                ret = torch.zeros(num_mc_envs, device=device)
                for s in reversed(range(len(all_rews))):
                    ret = all_rews[s] + args.gamma * ret
                mc_returns[t] = ret.view(E, args.mc_samples)

        # Compute stats
        mc16_mean = mc_returns.mean(dim=2)  # (T, E) - mean across 16 samples
        within_var = mc_returns.var(dim=2)   # (T, E) - variance within each state
        within_std_per_state = within_var.sqrt()  # (T, E)

        mean_within_std = within_std_per_state.mean().item()
        between_std = mc16_mean.std().item()
        snr = between_std / max(mean_within_std, 1e-8)

        # MC1 vs MC16 correlation
        mc1 = mc_returns[:, :, 0].reshape(-1)  # first sample
        mc16 = mc16_mean.reshape(-1)
        corr = torch.corrcoef(torch.stack([mc1, mc16]))[0, 1].item()

        # Fraction of states with very low variance
        frac_low = (within_std_per_state < 0.01).float().mean().item() * 100

        elapsed = time.time() - t0
        print(f"  SR={sr:.1f}%  within_std={mean_within_std:.4f}  between_std={between_std:.4f}  "
              f"SNR={snr:.2f}  Corr(MC1,MC16)={corr:.3f}  low_var={frac_low:.1f}%  ({elapsed:.1f}s)")
        sys.stdout.flush()

        results.append(dict(
            logstd=logstd_val, action_std=np.exp(logstd_val),
            sr=sr, within_std=mean_within_std, between_std=between_std,
            snr=snr, corr_mc1_mc16=corr, frac_low_var=frac_low,
        ))

    # Summary table
    print(f"\n{'='*90}")
    print(f"SUMMARY: Within-state MC std sweep for {args.env_id}")
    print(f"{'='*90}")
    print(f"{'logstd':>8} | {'act_std':>8} | {'SR%':>6} | {'within_std':>11} | {'between_std':>12} | {'SNR':>6} | {'Corr':>6} | {'low_var%':>8}")
    print(f"{'─'*8}─┼─{'─'*8}─┼─{'─'*6}─┼─{'─'*11}─┼─{'─'*12}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*8}")
    for r in results:
        print(f"{r['logstd']:>8.2f} | {r['action_std']:>8.4f} | {r['sr']:>6.1f} | {r['within_std']:>11.4f} | "
              f"{r['between_std']:>12.4f} | {r['snr']:>6.2f} | {r['corr_mc1_mc16']:>6.3f} | {r['frac_low_var']:>8.1f}")
    print(f"\nTarget: within_std ≈ 0.330 (PickCube logstd=-1.5)")

    envs.close()
    mc_envs.close()
