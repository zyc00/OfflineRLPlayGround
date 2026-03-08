"""P(success|s₀) coverage analysis for Diffusion Policy on GPU.

Uses ManiSkill GPU vectorized envs with state save/restore for MC rollouts.
Each initial state is tested mc_samples times — stochasticity comes from
diffusion sampling noise (DDIM with min_std > 0).

Usage:
  python dp_p_success_gpu.py \
    --ckpt runs/dppo_pretrain/dppo_200traj_unet_T100_5k/ckpt_3000.pt \
    --num_states 100 --mc_samples 16 \
    --ddim_steps 10 --min_sampling_denoising_std 0.01
"""
import copy
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import DPPO.peg_insertion_easy  # noqa: F401  register easy peg env
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
import h5py
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

sys.path.insert(0, os.path.dirname(__file__))
from DPPO.model.unet_wrapper import DiffusionUNet
from DPPO.model.diffusion import DiffusionModel


@dataclass
class Args:
    ckpt: str = "runs/dppo_pretrain/dppo_200traj_unet_T100_5k/ckpt_3000.pt"
    env_id: str = "PickCube-v1"
    control_mode: str = "pd_ee_delta_pos"
    num_states: int = 100
    mc_samples: int = 16
    max_episode_steps: int = 100
    seed: int = 0
    min_sampling_denoising_std: float = 0.01
    deterministic: bool = False
    """Use deterministic DDIM (no noise). Overrides min_sampling_denoising_std."""
    zero_qvel: bool = False
    """Zero out qvel dims (9:18). Auto-inherited from checkpoint if stored."""
    ddim_steps: int = 10
    ddim_eta: float = 1.0
    output: Optional[str] = None
    export_states_path: Optional[str] = None
    export_min_p: float = 0.0
    export_max_p: float = 0.3


def main():
    args = tyro.cli(Args)
    device = torch.device("cuda")

    # Load checkpoint (supports both pretrain 'ema' and finetune 'model' formats)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    assert "ema" in ckpt or "model" in ckpt, \
        f"Expected DPPO checkpoint with 'ema' or 'model' key, got: {list(ckpt.keys())}"

    # For finetuned checkpoints, architecture args may be in pretrain_args
    if "pretrain_args" in ckpt and ckpt["pretrain_args"] is not None:
        arch_args = ckpt["pretrain_args"]
    else:
        arch_args = ckpt["args"]
    ckpt_args = ckpt["args"]
    obs_dim = ckpt["obs_dim"]
    act_dim = ckpt["action_dim"]
    cond_steps = arch_args.get("cond_steps", 2)
    horizon_steps = arch_args.get("horizon_steps", 16)
    act_steps = arch_args.get("act_steps", 8)
    act_offset = cond_steps - 1

    network = DiffusionUNet(
        action_dim=act_dim, horizon_steps=horizon_steps,
        cond_dim=obs_dim * cond_steps,
        diffusion_step_embed_dim=arch_args.get("diffusion_step_embed_dim", 64),
        down_dims=arch_args.get("unet_dims", [64, 128, 256]),
        n_groups=arch_args.get("n_groups", 8),
    )
    model = DiffusionModel(
        network=network, horizon_steps=horizon_steps,
        obs_dim=obs_dim, action_dim=act_dim, device=device,
        denoising_steps=arch_args.get("denoising_steps", 100),
        denoised_clip_value=1.0, randn_clip_value=10,
        final_action_clip_value=1.0, predict_epsilon=True,
        base_eta=args.ddim_eta,
    )
    state_key = "ema" if "ema" in ckpt else "model"
    raw_sd = ckpt[state_key]
    # Finetuned checkpoints store actor weights under 'actor_ft.unet.*'
    # but DiffusionModel expects 'network.unet.*'. Remap if needed.
    if any(k.startswith("actor_ft.") for k in raw_sd):
        remapped = {}
        for k, v in raw_sd.items():
            if k.startswith("actor_ft.unet."):
                remapped["network.unet." + k[len("actor_ft.unet."):]] = v
            elif k.startswith("actor_ft."):
                remapped["network." + k[len("actor_ft."):]] = v
            elif not k.startswith(("actor.", "critic.", "ddim_")):
                remapped[k] = v
        raw_sd = remapped
    model.load_state_dict(raw_sd, strict=False)
    if torch.isnan(model.eta.eta_logit.data).any() or torch.isinf(model.eta.eta_logit.data).any():
        model.eta.eta_logit.data = torch.atanh(torch.tensor([0.999]))
    model.eval()

    # Auto-inherit zero_qvel from checkpoint
    if ckpt_args.get("zero_qvel", False) and not args.zero_qvel:
        args.zero_qvel = True

    print(f"Loaded: {args.ckpt}")
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}, cond_steps={cond_steps}")
    print(f"  horizon={horizon_steps}, act_steps={act_steps}, act_offset={act_offset}")
    print(f"  DDIM steps={args.ddim_steps}, min_std={args.min_sampling_denoising_std}, deterministic={args.deterministic}")
    if args.zero_qvel:
        print(f"  zero_qvel=True (dims 9:18 zeroed)")
    print(f"Coverage: {args.num_states} states x {args.mc_samples} MC samples")
    print()

    # Create GPU envs
    N = args.num_states
    env = gym.make(
        args.env_id, num_envs=N, obs_mode="state",
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
        sim_backend="gpu", reward_mode="sparse",
    )
    env = ManiSkillVectorEnv(env, N, ignore_terminations=True, record_metrics=True)

    # Reset to get N initial states and save
    obs_init, _ = env.reset(seed=args.seed)
    obs_init = obs_init.float().to(device)
    saved_state = copy.deepcopy(env.unwrapped.get_state_dict())

    p_success = np.zeros(N)
    n_steps_per_ep = args.max_episode_steps // act_steps + 1
    t0 = time.time()

    for mc in range(args.mc_samples):
        # Restore initial states
        env.unwrapped.set_state_dict(copy.deepcopy(saved_state))
        obs_raw = env.unwrapped.get_obs()
        if isinstance(obs_raw, torch.Tensor):
            obs = obs_raw.float().to(device)
        else:
            obs = obs_raw

        obs_history = obs.unsqueeze(1).repeat(1, cond_steps, 1)
        success = torch.zeros(N, dtype=torch.bool, device=device)
        done = torch.zeros(N, dtype=torch.bool, device=device)

        for step_block in range(n_steps_per_ep):
            if done.all():
                break
            obs_cond = obs_history
            if args.zero_qvel:
                obs_cond = obs_cond.clone()
                obs_cond[..., 9:18] = 0.0
            cond = {"state": obs_cond}
            with torch.no_grad():
                samples = model(
                    cond, deterministic=args.deterministic,
                    min_sampling_denoising_std=args.min_sampling_denoising_std,
                    ddim_steps=args.ddim_steps,
                )
            actions = samples.trajectories  # (N, horizon, act_dim)

            for a_idx in range(act_steps):
                act_i = act_offset + a_idx
                action = actions[:, min(act_i, actions.shape[1] - 1)]
                obs_new, rew, term, trunc, _ = env.step(action)
                obs_new = obs_new.float().to(device)

                # Track success (ignore_terminations=True, so env doesn't auto-reset)
                got_reward = rew.float() > 0.5
                success = success | (got_reward & ~done)
                done = done | term | trunc

                # Update obs history
                reset_mask = term | trunc
                if reset_mask.any():
                    obs_history[reset_mask] = obs_new[reset_mask].unsqueeze(1).repeat(1, cond_steps, 1)
                not_reset = ~reset_mask
                if not_reset.any():
                    obs_history[not_reset] = torch.cat(
                        [obs_history[not_reset, 1:], obs_new[not_reset].unsqueeze(1)], dim=1
                    )

        p_success += success.cpu().numpy()
        elapsed = time.time() - t0
        mc_sr = p_success / (mc + 1)
        print(f"  MC {mc+1}/{args.mc_samples}: SR={mc_sr.mean():.1%}, "
              f"frac_zero={(mc_sr==0).mean():.1%}, time={elapsed:.0f}s", flush=True)

    env.close()

    # Final P(success) per state
    p = p_success / args.mc_samples
    sr = p.mean()
    fz = (p == 0).mean()
    fo = (p == 1).mean()
    fd = ((p > 0.1) & (p < 0.9)).mean()

    print(f"\n{'='*60}")
    print(f"  P(success|s₀) — GPU DDIM-{args.ddim_steps}, min_std={args.min_sampling_denoising_std}")
    print(f"  MC{args.mc_samples}, {args.num_states} states")
    print(f"{'='*60}")
    print(f"  SR            = {sr:.1%}")
    print(f"  frac_zero     = {fz:.1%}")
    print(f"  frac_one      = {fo:.1%}")
    print(f"  frac_decisive = {fd:.1%}")
    print(f"  mean P        = {p.mean():.3f}")
    print(f"  std P         = {p.std():.3f}")
    print(f"  median P      = {np.median(p):.3f}")

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
    ax.set_ylabel(f"Count (out of {N})")
    ax.set_title(f"GPU Coverage — DDIM-{args.ddim_steps}, std={args.min_sampling_denoising_std}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xlim(-0.03, 1.03)
    plt.tight_layout()

    out_path = args.output or f"runs/dp_p_success_gpu_ddim{args.ddim_steps}_std{args.min_sampling_denoising_std}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    npz_path = out_path.replace(".png", ".npz")
    np.savez(npz_path, p_success=p, mc_samples=args.mc_samples, num_states=N)
    print(f"\nSaved plot: {out_path}")
    print(f"Saved data: {npz_path}")

    if args.export_states_path:
        export_path = args.export_states_path
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        select = (p >= args.export_min_p) & (p < args.export_max_p)
        sel_inds = np.nonzero(select)[0]
        with h5py.File(export_path, "w") as f:
            for out_idx, src_idx in enumerate(sel_inds):
                grp = f.create_group(f"traj_{out_idx}")
                es = grp.create_group("env_states")
                actors = es.create_group("actors")
                articulations = es.create_group("articulations")
                for name, tensor in saved_state["actors"].items():
                    arr = tensor[src_idx:src_idx + 1].detach().cpu().numpy().astype(np.float32)
                    actors.create_dataset(name, data=arr)
                for name, tensor in saved_state["articulations"].items():
                    arr = tensor[src_idx:src_idx + 1].detach().cpu().numpy().astype(np.float32)
                    articulations.create_dataset(name, data=arr)
                grp.attrs["orig_index"] = int(src_idx)
                grp.attrs["p_success"] = float(p[src_idx])
        print(
            f"Exported {len(sel_inds)} initial states with "
            f"{args.export_min_p:.3f} <= p < {args.export_max_p:.3f} to {export_path}"
        )


if __name__ == "__main__":
    main()
