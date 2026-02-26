"""P(success|s₀) distribution for Diffusion Policy on PickCube-v1, physx_cpu.

DP is inherently stochastic (diffusion sampling noise), so MC>1 is meaningful
even without adding external noise.

Supports both dp_train and DPPO checkpoint formats (auto-detected).

Usage:
  # dp_train checkpoint
  python dp_p_success_cpu.py --ckpt runs/dp_pickcube_mp_ee_30traj/checkpoints/best_eval_success_once.pt \
    --num-states 100 --mc-samples 16

  # DPPO checkpoint
  python dp_p_success_cpu.py --ckpt runs/dppo_pretrain/dppo_25traj_aligned_v2/best.pt \
    --num-states 100 --mc-samples 16
"""
import os
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
from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack

# Need dp_train Agent and ConditionalUnet1D
sys.path.insert(0, os.path.dirname(__file__))
from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


@dataclass
class Args:
    ckpt: str = "runs/dp_pickcube_mp_ee_30traj/checkpoints/best_eval_success_once.pt"
    """path to DP checkpoint"""
    env_id: str = "PickCube-v1"
    control_mode: str = "pd_ee_delta_pos"
    num_states: int = 50
    mc_samples: int = 16
    max_episode_steps: int = 100
    seed: int = 0
    min_sampling_denoising_std: Optional[float] = None
    """If set, use stochastic sampling with this min std (DPPO exploration noise). Default: deterministic."""
    # DP architecture args (must match training)
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16
    diffusion_step_embed_dim: int = 64
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    n_groups: int = 8
    output: Optional[str] = None


class DPAgent(nn.Module):
    """Minimal DP agent for inference only."""
    def __init__(self, obs_dim, act_dim, args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        self.act_dim = act_dim

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=act_dim,
            global_cond_dim=args.obs_horizon * obs_dim,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon',
        )

    def get_action(self, obs_seq):
        """obs_seq: (B, obs_horizon, obs_dim) -> (B, act_horizon, act_dim)"""
        B = obs_seq.shape[0]
        with torch.no_grad():
            obs_cond = obs_seq.flatten(start_dim=1)
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=obs_seq.device
            )
            for k in self.noise_scheduler.timesteps:
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq, timestep=k, global_cond=obs_cond,
                )
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=noisy_action_seq,
                ).prev_sample
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]


def run_mc(agent, num_states, mc_samples, max_steps, device, seed_offset=0,
           env_id="PickCube-v1", control_mode="pd_ee_delta_pos", obs_horizon=2):
    """Run MC rollouts with vectorized envs (mc_samples parallel envs per state)."""
    p_success = np.zeros(num_states)
    t0 = time.time()

    # Create mc_samples parallel envs
    def make_env(seed):
        def thunk():
            e = gym.make(env_id, obs_mode="state", render_mode="rgb_array",
                         reward_mode="sparse", control_mode=control_mode,
                         max_episode_steps=max_steps, reconfiguration_freq=1)
            if isinstance(e.action_space, gym.spaces.Dict):
                e = FlattenActionSpaceWrapper(e)
            e = FrameStack(e, num_stack=obs_horizon)
            e = CPUGymWrapper(e, ignore_terminations=False, record_metrics=True)
            return e
        return thunk

    envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(mc_samples)])

    for i in range(num_states):
        # Reset all mc_samples envs to the same initial state
        obs_list = []
        for j in range(mc_samples):
            obs_j, _ = envs.envs[j].reset(seed=seed_offset + i)
            obs_list.append(obs_j)
        obs = np.stack(obs_list)  # (mc_samples, obs_horizon, obs_dim)

        done = np.zeros(mc_samples, dtype=bool)
        success = np.zeros(mc_samples, dtype=bool)
        step = 0

        while step < max_steps and not done.all():
            obs_t = torch.from_numpy(obs).float().to(device)
            action_seq = agent.get_action(obs_t)  # (mc_samples, act_horizon, act_dim)
            action_np = action_seq.cpu().numpy()

            for a_idx in range(action_np.shape[1]):
                if step >= max_steps or done.all():
                    break
                # Step each env individually (SyncVectorEnv doesn't support per-env masking)
                obs_list = []
                for j in range(mc_samples):
                    if done[j]:
                        obs_list.append(obs[j])
                        continue
                    o, r, term, trunc, _ = envs.envs[j].step(action_np[j, a_idx])
                    obs_list.append(o)
                    if r > 0.5:
                        success[j] = True
                        done[j] = True
                    elif term or trunc:
                        done[j] = True
                obs = np.stack(obs_list)
                step += 1

        p_success[i] = success.sum() / mc_samples

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            sr = p_success[:i+1].mean()
            fz = (p_success[:i+1] == 0).mean()
            print(f"  {i+1}/{num_states} done ({elapsed:.0f}s), SR={sr:.1%}, frac_zero={fz:.1%}", flush=True)

    envs.close()
    return p_success


if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint (auto-detect dp_train vs DPPO format)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    if "ema_agent" in ckpt:
        # dp_train format — probe obs_dim from env
        tmp_env = gym.make(args.env_id, obs_mode="state", render_mode="rgb_array",
                           reward_mode="sparse", control_mode=args.control_mode,
                           max_episode_steps=args.max_episode_steps)
        if isinstance(tmp_env.action_space, gym.spaces.Dict):
            tmp_env = FlattenActionSpaceWrapper(tmp_env)
        tmp_env = FrameStack(tmp_env, num_stack=args.obs_horizon)
        tmp_env = CPUGymWrapper(tmp_env, ignore_terminations=False, record_metrics=True)
        obs_dim = tmp_env.observation_space.shape[-1]
        act_dim = tmp_env.action_space.shape[0]
        tmp_env.close()
        print(f"obs_dim={obs_dim}, act_dim={act_dim}")
    elif "ema" in ckpt:
        # DPPO format — read dims from checkpoint
        obs_dim = ckpt["obs_dim"]
        act_dim = ckpt["action_dim"]
        print(f"obs_dim={obs_dim}, act_dim={act_dim}")

    if "ema_agent" in ckpt:
        # dp_train format
        agent = DPAgent(obs_dim, act_dim, args).to(device)
        agent.load_state_dict(ckpt["ema_agent"])
        agent.eval()
        print(f"Loaded dp_train checkpoint: {args.ckpt}")
    elif "ema" in ckpt:
        # DPPO format
        from DPPO.model.unet_wrapper import DiffusionUNet
        from DPPO.model.diffusion import DiffusionModel
        ckpt_args = ckpt["args"]
        cond_steps = ckpt_args.get("cond_steps", args.obs_horizon)
        horizon_steps = ckpt_args.get("horizon_steps", args.pred_horizon)
        act_steps = ckpt_args.get("act_steps", args.act_horizon)
        network = DiffusionUNet(
            action_dim=act_dim, horizon_steps=horizon_steps,
            cond_dim=obs_dim * cond_steps,
            diffusion_step_embed_dim=ckpt_args.get("diffusion_step_embed_dim", 64),
            down_dims=ckpt_args.get("unet_dims", [64, 128, 256]),
            n_groups=ckpt_args.get("n_groups", 8),
        )
        dppo_model = DiffusionModel(
            network=network, horizon_steps=horizon_steps,
            obs_dim=obs_dim, action_dim=act_dim, device=device,
            denoising_steps=ckpt_args.get("denoising_steps", 100),
            denoised_clip_value=1.0, randn_clip_value=10,
            final_action_clip_value=1.0, predict_epsilon=True,
        )
        dppo_model.load_state_dict(ckpt["ema"])
        dppo_model.eval()
        act_offset = cond_steps - 1
        min_std = args.min_sampling_denoising_std

        class DPPOAgentWrapper:
            def __init__(self, model, act_offset, act_steps, min_std):
                self.model = model
                self.act_offset = act_offset
                self.act_steps = act_steps
                self.min_std = min_std
            def get_action(self, obs_seq):
                cond = {"state": obs_seq}
                deterministic = (self.min_std is None)
                samples = self.model(cond, deterministic=deterministic,
                                     min_sampling_denoising_std=self.min_std)
                return samples.trajectories[:, self.act_offset:self.act_offset + self.act_steps]
        agent = DPPOAgentWrapper(dppo_model, act_offset, act_steps, min_std)
        if min_std is not None:
            print(f"Loaded DPPO checkpoint: {args.ckpt} (min_std={min_std})")
        else:
            print(f"Loaded DPPO checkpoint: {args.ckpt} (deterministic)")
    else:
        raise ValueError(f"Unknown checkpoint format, keys: {list(ckpt.keys())}")
    print(f"Env: {args.env_id}, {args.control_mode}, max_steps={args.max_episode_steps}")
    print(f"States: {args.num_states}, MC: {args.mc_samples}")
    print()

    # Run P(success) analysis (vectorized MC)
    p = run_mc(agent, args.num_states, args.mc_samples, args.max_episode_steps,
               device, seed_offset=args.seed,
               env_id=args.env_id, control_mode=args.control_mode,
               obs_horizon=args.obs_horizon)

    sr = p.mean()
    fz = (p == 0).mean()
    fo = (p == 1).mean()
    fd = ((p > 0.1) & (p < 0.9)).mean()

    print(f"\n{'='*60}")
    print(f"  P(success|s₀) Distribution — DP MC{args.mc_samples}")
    print(f"{'='*60}")
    print(f"  SR         = {sr:.1%}")
    print(f"  frac_zero  = {fz:.1%}")
    print(f"  frac_one   = {fo:.1%}")
    print(f"  frac_decisive = {fd:.1%}")
    print(f"  mean P     = {p.mean():.3f}")
    print(f"  std P      = {p.std():.3f}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    bins = np.linspace(0, 1, 18)
    ax.hist(p, bins=bins, color="steelblue", edgecolor="black", linewidth=0.5, alpha=0.8)
    ax.axvline(sr, color="red", ls="--", lw=2, label=f"SR={sr:.1%}")

    textstr = (f"frac_zero:     {fz:.1%}\n"
               f"frac_one:      {fo:.1%}\n"
               f"frac_decisive: {fd:.1%}\n"
               f"SR:            {sr:.1%}")
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))
    ax.set_xlabel("P(success|s₀)")
    ax.set_ylabel(f"Count (out of {args.num_states})")
    ax.set_title(f"DP P(success|s₀) — MC{args.mc_samples}, {args.env_id}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xlim(-0.03, 1.03)

    plt.tight_layout()
    out_path = args.output or "runs/dp_p_success_cpu.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {out_path}")
