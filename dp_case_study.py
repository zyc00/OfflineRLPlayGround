"""Case study: Compare success vs failure from the SAME initial state.

For decisive states (0 < P(success) < 1), run N rollouts with stochastic policy
from the same initial state, and save success/failure videos side by side.

Also records per-step observations and actions for analysis.

Usage:
  python dp_case_study.py --ckpt runs/dppo_finetune/dppo_ft_peg_conservative/best.pt \
    --env-id PegInsertionSide-v1 --control-mode pd_joint_delta_pos \
    --n-rollouts 20 --n-seeds 50 --min-sampling-denoising-std 0.01 --ddim-steps 10
"""
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import tyro
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack

sys.path.insert(0, os.path.dirname(__file__))


@dataclass
class Args:
    ckpt: str = "runs/dppo_finetune/dppo_ft_peg_conservative/best.pt"
    env_id: str = "PegInsertionSide-v1"
    control_mode: str = "pd_joint_delta_pos"
    max_episode_steps: int = 200
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16

    # Stochastic sampling
    min_sampling_denoising_std: float = 0.01
    ddim_steps: int = 10

    # Case study params
    n_seeds: int = 50       # Number of initial states to screen
    n_rollouts: int = 20    # Rollouts per seed to estimate P(success)
    n_videos: int = 3       # Videos to save per category (success/failure) per decisive seed
    seed_start: int = 0

    output_dir: str = "runs/case_study"
    video_fps: int = 20


def main():
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    from DPPO.model.unet_wrapper import DiffusionUNet
    from DPPO.model.diffusion import DiffusionModel

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    is_finetuned = "model" in ckpt and "ema" not in ckpt

    ckpt_args = ckpt["args"]
    obs_dim, act_dim = ckpt["obs_dim"], ckpt["action_dim"]
    cond_steps = ckpt_args.get("cond_steps", args.obs_horizon)
    horizon_steps = ckpt_args.get("horizon_steps", args.pred_horizon)
    act_steps = ckpt_args.get("act_steps", args.act_horizon)
    act_offset = cond_steps - 1
    denoising_steps = ckpt_args.get("denoising_steps", 100)

    network = DiffusionUNet(
        action_dim=act_dim, horizon_steps=horizon_steps,
        cond_dim=obs_dim * cond_steps,
        diffusion_step_embed_dim=ckpt_args.get("diffusion_step_embed_dim", 64),
        down_dims=ckpt_args.get("unet_dims", [64, 128, 256]),
        n_groups=ckpt_args.get("n_groups", 8),
    )
    model = DiffusionModel(
        network=network, horizon_steps=horizon_steps,
        obs_dim=obs_dim, action_dim=act_dim, device=device,
        denoising_steps=denoising_steps,
        denoised_clip_value=1.0, randn_clip_value=10,
        final_action_clip_value=1.0, predict_epsilon=True,
    )

    if is_finetuned:
        sd = ckpt["model"]
        remapped = {k.replace("actor_ft.", "network.", 1): v
                    for k, v in sd.items() if k.startswith("actor_ft.")}
        model.load_state_dict(remapped, strict=False)
        print(f"Loaded finetuned checkpoint ({len(remapped)} keys)")
    else:
        model.load_state_dict(ckpt["ema"], strict=False)
        print(f"Loaded pretrained checkpoint")
    model.eval()

    min_std = args.min_sampling_denoising_std

    def get_action(obs_seq):
        cond = {"state": obs_seq}
        samples = model(cond, deterministic=False,
                        min_sampling_denoising_std=min_std,
                        ddim_steps=args.ddim_steps)
        return samples.trajectories[:, act_offset:act_offset + act_steps]

    # Create env
    env = gym.make(args.env_id, obs_mode="state", render_mode="rgb_array",
                   reward_mode="sparse", control_mode=args.control_mode,
                   max_episode_steps=args.max_episode_steps, reconfiguration_freq=1)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = FrameStack(env, num_stack=cond_steps)
    env = CPUGymWrapper(env, ignore_terminations=False, record_metrics=True)

    print(f"\nPhase 1: Screening {args.n_seeds} seeds × {args.n_rollouts} rollouts")
    print(f"  env={args.env_id}, min_std={min_std}, ddim={args.ddim_steps}")

    # Phase 1: Screen seeds to find decisive ones
    seed_results = {}
    for seed_idx in range(args.n_seeds):
        env_seed = args.seed_start + seed_idx
        successes = 0
        for r in range(args.n_rollouts):
            obs, _ = env.reset(seed=env_seed)
            done = False
            success = False
            step = 0
            while step < args.max_episode_steps and not done:
                obs_t = torch.from_numpy(np.array(obs)).float().unsqueeze(0).to(device)
                action_seq = get_action(obs_t)
                action_np = action_seq.cpu().numpy()[0]
                for a_idx in range(action_np.shape[0]):
                    if step >= args.max_episode_steps or done:
                        break
                    obs, rew, term, trunc, info = env.step(action_np[a_idx])
                    step += 1
                    if rew > 0.5:
                        success = True
                        done = True
                    elif term or trunc:
                        done = True
            if success:
                successes += 1
        p = successes / args.n_rollouts
        seed_results[env_seed] = p
        tag = "DECISIVE" if 0.1 < p < 0.9 else ("success" if p >= 0.9 else "zero")
        if seed_idx % 10 == 0 or tag == "DECISIVE":
            print(f"  seed={env_seed}: P(success)={p:.2f} [{tag}]")

    # Categorize
    decisive = {s: p for s, p in seed_results.items() if 0.1 < p < 0.9}
    always_success = {s: p for s, p in seed_results.items() if p >= 0.9}
    always_fail = {s: p for s, p in seed_results.items() if p <= 0.1}

    print(f"\nResults: {len(always_success)} always_success, {len(decisive)} decisive, {len(always_fail)} always_fail")
    if decisive:
        print(f"Decisive seeds: {dict(sorted(decisive.items(), key=lambda x: x[1]))}")

    # Phase 2: Record videos for decisive seeds
    if not decisive:
        print("No decisive seeds found. Try different seed range or adjust min_std.")
        env.close()
        return

    import imageio
    os.makedirs(args.output_dir, exist_ok=True)

    # Pick top decisive seeds (closest to 0.5)
    sorted_decisive = sorted(decisive.items(), key=lambda x: abs(x[1] - 0.5))
    target_seeds = [s for s, p in sorted_decisive[:5]]

    print(f"\nPhase 2: Recording videos for decisive seeds: {target_seeds}")

    for env_seed in target_seeds:
        p = decisive[env_seed]
        seed_dir = os.path.join(args.output_dir, f"seed{env_seed:03d}_p{p:.2f}")
        os.makedirs(os.path.join(seed_dir, "success"), exist_ok=True)
        os.makedirs(os.path.join(seed_dir, "failure"), exist_ok=True)

        n_succ_saved = 0
        n_fail_saved = 0
        all_trajs = []  # (success, obs_list, action_list, steps)

        for r in range(args.n_rollouts * 2):  # Extra rollouts to get enough videos
            if n_succ_saved >= args.n_videos and n_fail_saved >= args.n_videos:
                break

            obs, _ = env.reset(seed=env_seed)
            frames = []
            obs_history = []
            action_history = []
            done = False
            success = False
            step = 0

            while step < args.max_episode_steps and not done:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

                obs_np = np.array(obs)
                obs_history.append(obs_np.copy())
                obs_t = torch.from_numpy(obs_np).float().unsqueeze(0).to(device)
                action_seq = get_action(obs_t)
                action_np = action_seq.cpu().numpy()[0]

                for a_idx in range(action_np.shape[0]):
                    if step >= args.max_episode_steps or done:
                        break
                    action_history.append(action_np[a_idx].copy())
                    obs, rew, term, trunc, info = env.step(action_np[a_idx])
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                    step += 1
                    if rew > 0.5:
                        success = True
                        done = True
                    elif term or trunc:
                        done = True

            tag = "success" if success else "failure"
            n_saved = n_succ_saved if success else n_fail_saved

            if n_saved < args.n_videos and len(frames) > 0:
                fname = f"rollout{r:02d}_{tag}_{step}steps.mp4"
                fpath = os.path.join(seed_dir, tag, fname)
                imageio.mimwrite(fpath, frames, fps=args.video_fps)
                if success:
                    n_succ_saved += 1
                else:
                    n_fail_saved += 1

            all_trajs.append({
                "success": success,
                "steps": step,
                "obs": np.array(obs_history),
                "actions": np.array(action_history),
            })

        # Save trajectory data for analysis
        np.savez(os.path.join(seed_dir, "trajectories.npz"),
                 p_success=p,
                 env_seed=env_seed,
                 n_trajs=len(all_trajs),
                 successes=[t["success"] for t in all_trajs],
                 steps=[t["steps"] for t in all_trajs])

        print(f"  seed={env_seed} (P={p:.2f}): saved {n_succ_saved} success + {n_fail_saved} failure videos")

        # Quick divergence analysis: compare obs trajectories
        succ_trajs = [t for t in all_trajs if t["success"]]
        fail_trajs = [t for t in all_trajs if not t["success"]]

        if succ_trajs and fail_trajs:
            # Find first divergence point: compare obs at each decision step
            min_len = min(
                min(t["obs"].shape[0] for t in succ_trajs),
                min(t["obs"].shape[0] for t in fail_trajs)
            )
            # Use first trajectory of each as representative
            s_obs = succ_trajs[0]["obs"][:min_len]
            f_obs = fail_trajs[0]["obs"][:min_len]
            s_act = succ_trajs[0]["actions"]
            f_act = fail_trajs[0]["actions"]

            # Per-step obs difference
            obs_diff = np.abs(s_obs - f_obs).mean(axis=-1)  # (T,) or (T, cond_steps)
            if obs_diff.ndim > 1:
                obs_diff = obs_diff.mean(axis=-1)

            # Find first step where difference exceeds threshold
            threshold = 0.01
            diverge_steps = np.where(obs_diff > threshold)[0]
            if len(diverge_steps) > 0:
                first_diverge = diverge_steps[0]
                print(f"    First obs divergence at decision step {first_diverge} "
                      f"(obs_diff={obs_diff[first_diverge]:.4f})")
                print(f"    Success: {succ_trajs[0]['steps']} steps, "
                      f"Failure: {fail_trajs[0]['steps']} steps")
            else:
                print(f"    Obs never diverges > {threshold} in first {min_len} decision steps")

            # Compare action differences at early steps
            act_min_len = min(s_act.shape[0], f_act.shape[0], 50)
            act_diff = np.abs(s_act[:act_min_len] - f_act[:act_min_len]).mean(axis=-1)
            early_act_diff = act_diff[:16].mean()
            print(f"    Mean action diff (first 16 steps): {early_act_diff:.6f}")
            print(f"    Mean action diff (steps 16-{act_min_len}): {act_diff[16:].mean():.6f}")

    env.close()
    print(f"\nDone. Videos and data saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
