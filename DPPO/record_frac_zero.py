"""Record videos of frac_zero states with both finetuned and pretrain policies.

Uses the same eval logic as dp_p_success_gpu.py (batch GPU, succeed_once tracking).

Usage:
  python -m DPPO.record_frac_zero \
    --ft_ckpt runs/dppo_finetune/mc_d500_c04_r2/best.pt \
    --pretrain_ckpt runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt \
    --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos \
    --max_episode_steps 2000 --zero_qvel \
    --num_states 200 --max_videos 10
"""
import copy
import os
import sys
from dataclasses import dataclass

import gymnasium as gym
import imageio
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from DPPO.model.unet_wrapper import DiffusionUNet
from DPPO.model.diffusion import DiffusionModel


@dataclass
class Args:
    ft_ckpt: str = ""
    pretrain_ckpt: str = ""
    env_id: str = "PegInsertionSide-v1"
    control_mode: str = "pd_joint_delta_pos"
    num_states: int = 200
    max_episode_steps: int = 2000
    max_videos: int = 10
    seed: int = 0
    zero_qvel: bool = False
    ddim_steps: int = 10
    min_sampling_denoising_std: float = 0.01
    output_dir: str = "runs/frac_zero_videos"


def load_model(ckpt_path, device):
    """Load a DPPO checkpoint (pretrain or finetuned) into a DiffusionModel."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "pretrain_args" in ckpt and ckpt["pretrain_args"] is not None:
        arch_args = ckpt["pretrain_args"]
    else:
        arch_args = ckpt["args"]

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
    )

    raw_sd = ckpt.get("ema") or ckpt["model"]
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

    # Detect DDIM from finetuned checkpoint
    ft_ddim_steps = None
    is_finetune = any(k.startswith("actor_ft.") for k in (ckpt.get("ema") or ckpt["model"]))
    if is_finetune and ckpt["args"].get("use_ddim"):
        ft_ddim_steps = ckpt["args"].get("ddim_steps", 10)

    return model, {
        "obs_dim": obs_dim, "act_dim": act_dim,
        "cond_steps": cond_steps, "horizon_steps": horizon_steps,
        "act_steps": act_steps, "act_offset": act_offset,
        "ft_ddim_steps": ft_ddim_steps,
    }


@torch.no_grad()
def find_failing_states(model, info, args, device):
    """Batch eval on GPU (same logic as dp_p_success_gpu.py). Returns fail indices + saved states."""
    N = args.num_states
    env = gym.make(
        args.env_id, num_envs=N, obs_mode="state",
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
        sim_backend="gpu", reward_mode="sparse",
    )
    env = ManiSkillVectorEnv(env, N, ignore_terminations=True, record_metrics=True)

    obs, _ = env.reset(seed=args.seed)
    obs = obs.float().to(device)
    saved_state = copy.deepcopy(env.unwrapped.get_state_dict())

    cond_steps = info["cond_steps"]
    act_steps = info["act_steps"]
    act_offset = info["act_offset"]
    ddim_steps = info.get("ft_ddim_steps") or args.ddim_steps
    n_steps_per_ep = args.max_episode_steps // act_steps + 1

    obs_history = obs.unsqueeze(1).repeat(1, cond_steps, 1)
    success = torch.zeros(N, dtype=torch.bool, device=device)
    done = torch.zeros(N, dtype=torch.bool, device=device)

    print(f"Phase 1: Finding failing states ({N} envs, {args.max_episode_steps} steps, deterministic)...")
    for step_block in range(n_steps_per_ep):
        if done.all():
            break
        obs_cond = obs_history
        if args.zero_qvel:
            obs_cond = obs_cond.clone()
            obs_cond[..., 9:18] = 0.0
        cond = {"state": obs_cond}
        samples = model(
            cond, deterministic=True,
            min_sampling_denoising_std=args.min_sampling_denoising_std,
            ddim_steps=ddim_steps,
        )
        actions = samples.trajectories

        for a_idx in range(act_steps):
            act_i = act_offset + a_idx
            action = actions[:, min(act_i, actions.shape[1] - 1)]
            obs_new, rew, term, trunc, _ = env.step(action)
            obs_new = obs_new.float().to(device)

            # succeed_once tracking (same as dp_p_success_gpu.py)
            got_reward = rew.float() > 0.5
            success = success | (got_reward & ~done)
            done = done | term | trunc

            reset_mask = term | trunc
            if reset_mask.any():
                obs_history[reset_mask] = obs_new[reset_mask].unsqueeze(1).repeat(1, cond_steps, 1)
            not_reset = ~reset_mask
            if not_reset.any():
                obs_history[not_reset] = torch.cat(
                    [obs_history[not_reset, 1:], obs_new[not_reset].unsqueeze(1)], dim=1
                )

    env.close()

    fail_mask = ~success.cpu().numpy()
    fail_idxs = np.where(fail_mask)[0]
    print(f"  SR={success.float().mean():.1%}, {len(fail_idxs)} failures")

    # Extract per-env states
    fail_states = []
    for idx in fail_idxs[:args.max_videos]:
        state = {}
        for k in saved_state:
            state[k] = {}
            for n in saved_state[k]:
                state[k][n] = saved_state[k][n][idx:idx+1].clone()
        fail_states.append((int(idx), state))

    return fail_states


@torch.no_grad()
def record_video(model, info, state_dict, args, device):
    """Record one episode on GPU (1 env), rendering every raw step. Returns (success, frames)."""
    env = gym.make(
        args.env_id, num_envs=1, obs_mode="state",
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
        sim_backend="gpu", reward_mode="sparse",
        render_mode="rgb_array",
    )
    env = ManiSkillVectorEnv(env, 1, ignore_terminations=True, record_metrics=True)

    obs, _ = env.reset()
    env.unwrapped.set_state_dict(copy.deepcopy(state_dict))
    obs = env.unwrapped.get_obs().float().to(device)

    cond_steps = info["cond_steps"]
    act_steps = info["act_steps"]
    act_offset = info["act_offset"]
    ddim_steps = info.get("ft_ddim_steps") or args.ddim_steps
    n_steps_per_ep = args.max_episode_steps // act_steps + 1

    obs_history = obs.unsqueeze(1).repeat(1, cond_steps, 1)
    success = False
    done = False
    frames = []

    for step_block in range(n_steps_per_ep):
        if done:
            break
        obs_cond = obs_history
        if args.zero_qvel:
            obs_cond = obs_cond.clone()
            obs_cond[..., 9:18] = 0.0
        cond = {"state": obs_cond}
        samples = model(
            cond, deterministic=True,
            min_sampling_denoising_std=args.min_sampling_denoising_std,
            ddim_steps=ddim_steps,
        )
        actions = samples.trajectories

        for a_idx in range(act_steps):
            act_i = act_offset + a_idx
            action = actions[:, min(act_i, actions.shape[1] - 1)]
            obs_new, rew, term, trunc, _ = env.step(action)
            obs_new = obs_new.float().to(device)

            # Render every raw step
            frame = env.unwrapped.render()
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            if frame.ndim == 4:
                frame = frame[0]
            frames.append(frame)

            if rew.float().item() > 0.5:
                success = True
            if term.any() or trunc.any():
                done = True
                obs_history = obs_new.unsqueeze(1).repeat(1, cond_steps, 1)
                break
            obs_history = torch.cat(
                [obs_history[:, 1:], obs_new.unsqueeze(1)], dim=1
            )

    env.close()
    return success, frames


def main():
    import tyro
    args = tyro.cli(Args)
    device = torch.device("cuda")

    assert args.ft_ckpt, "--ft_ckpt required"
    assert args.pretrain_ckpt, "--pretrain_ckpt required"

    # Load finetuned model
    print("Loading finetuned model...")
    ft_model, ft_info = load_model(args.ft_ckpt, device)

    # Auto-inherit zero_qvel
    ft_ckpt = torch.load(args.ft_ckpt, map_location="cpu", weights_only=False)
    if ft_ckpt["args"].get("zero_qvel", False):
        args.zero_qvel = True
    del ft_ckpt

    # Phase 1: Find failing states (batch, same as dp_p_success_gpu.py)
    fail_states = find_failing_states(ft_model, ft_info, args, device)
    if not fail_states:
        print("No failing states found!")
        return

    # Save states
    os.makedirs(args.output_dir, exist_ok=True)
    states_path = os.path.join(args.output_dir, "fail_states.pt")
    torch.save(fail_states, states_path)
    print(f"Saved {len(fail_states)} states to {states_path}")

    # Load pretrain model
    print("\nLoading pretrain model...")
    pt_model, pt_info = load_model(args.pretrain_ckpt, device)
    pt_info["ft_ddim_steps"] = args.ddim_steps

    # Phase 2: Record videos for each failing state
    for i, (state_idx, state_dict) in enumerate(fail_states):
        print(f"\nState {i+1}/{len(fail_states)} (env {state_idx}):")

        # Finetuned video
        ft_success, ft_frames = record_video(ft_model, ft_info, state_dict, args, device)
        ft_path = os.path.join(args.output_dir, f"state{state_idx}_finetuned.mp4")
        imageio.mimsave(ft_path, ft_frames, fps=20)
        print(f"  [finetuned] {len(ft_frames)} frames, succeed_once={ft_success} -> {ft_path}")

        # Pretrain video
        pt_success, pt_frames = record_video(pt_model, pt_info, state_dict, args, device)
        pt_path = os.path.join(args.output_dir, f"state{state_idx}_pretrain.mp4")
        imageio.mimsave(pt_path, pt_frames, fps=20)
        print(f"  [pretrain] {len(pt_frames)} frames, succeed_once={pt_success} -> {pt_path}")

    print(f"\nDone! {len(fail_states)} x 2 videos saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
