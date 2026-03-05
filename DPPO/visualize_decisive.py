"""Visualize decisive states: find initial states where policy both succeeds and fails.

For each decisive state (P(success) between p_low and p_high), records
a success video and a failure video from the same initial state.

Usage:
    python -m DPPO.visualize_decisive \
        --ckpt runs/dppo_finetune/mc_d500_c04_r2/best.pt \
        --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos \
        --max_episode_steps 200 --zero_qvel \
        --mc_samples 16 --num_candidates 100 --max_videos 10
"""
import os
import copy
import argparse
import numpy as np
import torch
import imageio
import time
import gymnasium as gym
import mani_skill.envs


def load_model(ckpt_path, device):
    """Load diffusion model from pretrain or finetuned checkpoint."""
    from DPPO.model.unet_wrapper import DiffusionUNet
    from DPPO.model.mlp import DiffusionMLP
    from DPPO.model.diffusion import DiffusionModel

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    is_finetune = "model" in ckpt and "ema" not in ckpt
    args = ckpt.get("pretrain_args", ckpt["args"]) if is_finetune else ckpt["args"]

    obs_dim = ckpt["obs_dim"]
    action_dim = ckpt["action_dim"]
    cond_steps = args.get("cond_steps", 2)
    horizon_steps = args.get("horizon_steps", 16)
    act_steps = args.get("act_steps", 8)
    T = args.get("denoising_steps", 100)
    network_type = args.get("network_type", "unet")
    act_offset = cond_steps - 1 if network_type == "unet" else 0

    if network_type == "unet":
        network = DiffusionUNet(
            action_dim=action_dim, horizon_steps=horizon_steps,
            cond_dim=obs_dim * cond_steps,
            diffusion_step_embed_dim=args.get("diffusion_step_embed_dim", 64),
            down_dims=args.get("unet_dims", [64, 128, 256]),
            n_groups=args.get("n_groups", 8),
        )
    else:
        network = DiffusionMLP(
            action_dim=action_dim, horizon_steps=horizon_steps,
            cond_dim=obs_dim * cond_steps,
        )

    model = DiffusionModel(
        network=network, horizon_steps=horizon_steps,
        obs_dim=obs_dim, action_dim=action_dim, device=device,
        denoising_steps=T, denoised_clip_value=1.0,
        randn_clip_value=3, final_action_clip_value=1.0, predict_epsilon=True,
    )

    if is_finetune:
        sd = ckpt["model"]
        remapped = {k.replace("actor_ft.", "network.", 1): v
                    for k, v in sd.items() if k.startswith("actor_ft.")}
        model.load_state_dict(remapped, strict=False)
        ft_args = ckpt["args"]
        ddim_steps = ft_args.get("ddim_steps", 10) if ft_args.get("use_ddim") else None
        print(f"Loaded finetuned checkpoint (iter {ckpt.get('iteration', '?')})")
    else:
        model.load_state_dict(ckpt["ema"], strict=False)
        ddim_steps = None
        print("Loaded pretrain EMA checkpoint")

    if hasattr(model, 'eta') and torch.isnan(model.eta.eta_logit.data).any():
        model.eta.eta_logit.data = torch.atanh(torch.tensor([0.999], device=device))

    model.eval()
    return model, dict(obs_dim=obs_dim, action_dim=action_dim, cond_steps=cond_steps,
                       act_steps=act_steps, act_offset=act_offset, ddim_steps=ddim_steps)


@torch.no_grad()
def rollout_episode(env, model, info, device, ddim_steps, min_std, zero_qvel,
                    max_steps, record=False, deterministic=False):
    """Roll out one episode from current env state.

    Returns (success: bool, frames: list[np.ndarray] or []).
    """
    cond_steps = info["cond_steps"]
    act_steps = info["act_steps"]
    act_offset = info["act_offset"]

    obs_raw = env.unwrapped.get_obs()
    if isinstance(obs_raw, dict):
        obs_raw = obs_raw.get("state", obs_raw.get("obs"))
    obs = obs_raw.float().to(device) if isinstance(obs_raw, torch.Tensor) \
        else torch.from_numpy(np.asarray(obs_raw, dtype=np.float32)).to(device)
    if obs.dim() == 1:
        obs = obs.unsqueeze(0)  # (1, obs_dim)

    obs_hist = obs.unsqueeze(1).repeat(1, cond_steps, 1)  # (1, C, D)

    frames = []
    success = False
    step = 0

    def _capture_frame():
        if not record:
            return
        f = env.render()
        if f is not None:
            f_np = f.cpu().numpy() if isinstance(f, torch.Tensor) else np.asarray(f)
            if f_np.ndim == 4:
                f_np = f_np[0]
            frames.append(f_np.astype(np.uint8))

    _capture_frame()  # initial frame

    while step < max_steps:
        obs_in = obs_hist.clone()
        if zero_qvel:
            obs_in[..., 9:18] = 0.0

        samples = model({"state": obs_in}, deterministic=deterministic,
                        min_sampling_denoising_std=min_std, ddim_steps=ddim_steps)
        actions = samples.trajectories  # (1, H, act_dim)

        for a_idx in range(act_steps):
            if step >= max_steps:
                break
            a_i = min(act_offset + a_idx, actions.shape[1] - 1)
            action = actions[:, a_i]  # (1, act_dim) — keep batch dim for env

            obs_new, reward, terminated, truncated, _ = env.step(action)
            step += 1
            _capture_frame()  # render every env step for smooth video

            obs_t = obs_new.float().to(device) if isinstance(obs_new, torch.Tensor) \
                else torch.from_numpy(np.asarray(obs_new, dtype=np.float32)).to(device)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
            obs_hist = torch.cat([obs_hist[:, 1:], obs_t.unsqueeze(1)], dim=1)

            # Handle reward (may be tensor or scalar)
            r = reward.item() if hasattr(reward, 'item') else float(reward)
            if r > 0.5:
                success = True

            term = terminated.item() if hasattr(terminated, 'item') else bool(terminated)
            trunc = truncated.item() if hasattr(truncated, 'item') else bool(truncated)
            if term or trunc:
                return success, frames

    return success, frames


def _deepcopy_state(state_dict):
    """Deep copy a ManiSkill state dict (may contain tensors)."""
    out = {}
    for k, v in state_dict.items():
        if isinstance(v, dict):
            out[k] = {kk: vv.clone() if isinstance(vv, torch.Tensor) else copy.deepcopy(vv)
                       for kk, vv in v.items()}
        elif isinstance(v, torch.Tensor):
            out[k] = v.clone()
        else:
            out[k] = copy.deepcopy(v)
    return out


def main():
    p = argparse.ArgumentParser(description="Record videos of decisive states")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--env_id", default="PegInsertionSide-v1")
    p.add_argument("--control_mode", default="pd_joint_delta_pos")
    p.add_argument("--max_episode_steps", type=int, default=200)
    p.add_argument("--zero_qvel", action="store_true")
    p.add_argument("--ddim_steps", type=int, default=10)
    p.add_argument("--min_sampling_std", type=float, default=0.01)
    p.add_argument("--mc_samples", type=int, default=16,
                   help="MC rollouts per state for classification")
    p.add_argument("--num_candidates", type=int, default=200,
                   help="Initial states to screen")
    p.add_argument("--max_videos", type=int, default=10,
                   help="Max decisive states to record")
    p.add_argument("--output_dir", default="runs/decisive_videos")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--p_low", type=float, default=0.15)
    p.add_argument("--p_high", type=float, default=0.85)
    p.add_argument("--deterministic", action="store_true",
                   help="Record deterministic rollouts (no MC classification, "
                        "just save success/failure videos)")
    p.add_argument("--max_success", type=int, default=5,
                   help="(deterministic mode) max success videos to save")
    p.add_argument("--max_failure", type=int, default=5,
                   help="(deterministic mode) max failure videos to save")
    p.add_argument("--sim_backend", default="cpu",
                   choices=["cpu", "gpu"],
                   help="Simulation backend (gpu for accurate eval of GPU-trained policies)")
    args = p.parse_args()

    device = torch.device("cuda")
    model, model_info = load_model(args.ckpt, device)
    ddim_steps = args.ddim_steps or model_info["ddim_steps"]
    print(f"Model: cond={model_info['cond_steps']}, act={model_info['act_steps']}, "
          f"ddim={ddim_steps}, zero_qvel={args.zero_qvel}")

    # ManiSkill3: even single env has batch dim=1
    env_kwargs = dict(obs_mode="state", control_mode=args.control_mode,
                      max_episode_steps=args.max_episode_steps, render_mode="rgb_array",
                      reconfiguration_freq=0 if args.sim_backend == "cpu" else 1,
                      reward_mode="sparse")
    if args.sim_backend == "gpu":
        env_kwargs["num_envs"] = 1
        env_kwargs["sim_backend"] = "gpu"
    env = gym.make(args.env_id, **env_kwargs)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.deterministic:
        _run_deterministic(env, model, model_info, device, args, ddim_steps)
    else:
        _run_stochastic(env, model, model_info, device, args, ddim_steps)

    env.close()


def _run_deterministic(env, model, model_info, device, args, ddim_steps):
    """Record deterministic rollouts, save success and failure videos separately."""
    n_succ_saved, n_fail_saved = 0, 0
    n_succ_total, n_fail_total = 0, 0
    use_gpu = args.sim_backend == "gpu"
    t0 = time.time()

    print(f"\nDeterministic mode ({args.sim_backend}): recording up to "
          f"{args.max_success} success + {args.max_failure} failure videos\n")

    for si in range(args.num_candidates):
        need_succ = n_succ_saved < args.max_success
        need_fail = n_fail_saved < args.max_failure
        if not need_succ and not need_fail:
            break

        if use_gpu:
            env.reset()
        else:
            env.reset(options={"reconfigure": True})
            state = _deepcopy_state(env.unwrapped.get_state_dict())

        if use_gpu:
            # GPU: just record directly (rendering is fast, no state restore needed)
            s, frames = rollout_episode(env, model, model_info, device,
                                        ddim_steps, args.min_sampling_std,
                                        args.zero_qvel, args.max_episode_steps,
                                        record=True, deterministic=True)
            s_check = s
        else:
            # CPU: pre-check without rendering, then re-render if needed
            env.reset()
            env.unwrapped.set_state_dict(state)
            s_check, _ = rollout_episode(env, model, model_info, device,
                                         ddim_steps, args.min_sampling_std,
                                         args.zero_qvel, args.max_episode_steps,
                                         deterministic=True)
            frames = None

        if s_check:
            n_succ_total += 1
        else:
            n_fail_total += 1

        should_record = (s_check and need_succ) or (not s_check and need_fail)
        if should_record:
            if not use_gpu:
                # CPU: re-run with rendering
                env.reset()
                env.unwrapped.set_state_dict(state)
                s, frames = rollout_episode(env, model, model_info, device,
                                            ddim_steps, args.min_sampling_std,
                                            args.zero_qvel, args.max_episode_steps,
                                            record=True, deterministic=True)
            if frames:
                outcome = "success" if s else "failure"
                idx = n_succ_saved if s else n_fail_saved
                fp = os.path.join(args.output_dir, f"det_{outcome}_{idx:03d}.mp4")
                imageio.mimsave(fp, frames, fps=args.fps)
                if s:
                    n_succ_saved += 1
                else:
                    n_fail_saved += 1
                elapsed = time.time() - t0
                print(f"  State {si}: {outcome} ({len(frames)} frames)  "
                      f"[saved: {n_succ_saved}S/{n_fail_saved}F, "
                      f"total: {n_succ_total}S/{n_fail_total}F]  ({elapsed:.0f}s)")
        elif (si + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  Screened {si+1}  "
                  f"[saved: {n_succ_saved}S/{n_fail_saved}F, "
                  f"total: {n_succ_total}S/{n_fail_total}F]  ({elapsed:.0f}s)")

    total = n_succ_total + n_fail_total
    print(f"\nDone: {total} episodes, SR={n_succ_total/max(total,1):.1%}")
    print(f"Saved: {n_succ_saved} success + {n_fail_saved} failure videos")
    print(f"Videos: {args.output_dir}/")


def _run_stochastic(env, model, model_info, device, args, ddim_steps):
    """Original mode: MC classification + record decisive states."""
    print(f"\nScreening {args.num_candidates} states with MC{args.mc_samples}")
    print(f"Decisive range: [{args.p_low}, {args.p_high}]")
    print(f"Will record up to {args.max_videos} decisive states\n")

    n_decisive = 0
    all_p = []
    t0 = time.time()

    for si in range(args.num_candidates):
        # New scene config (randomized peg/box shapes)
        env.reset(options={"reconfigure": True})
        state = _deepcopy_state(env.unwrapped.get_state_dict())

        # --- Phase 1: Classify via MC rollouts (no rendering) ---
        n_succ = 0
        for _ in range(args.mc_samples):
            env.reset()  # same shapes (reconfiguration_freq=0), resets step counter
            env.unwrapped.set_state_dict(state)
            s, _ = rollout_episode(env, model, model_info, device,
                                   ddim_steps, args.min_sampling_std,
                                   args.zero_qvel, args.max_episode_steps)
            n_succ += int(s)

        p_succ = n_succ / args.mc_samples
        all_p.append(p_succ)

        if args.p_low <= p_succ <= args.p_high:
            # --- Phase 2: Record one success + one failure video ---
            got_s, got_f = False, False
            for _ in range(30):
                if got_s and got_f:
                    break
                env.reset()
                env.unwrapped.set_state_dict(state)
                s, frames = rollout_episode(env, model, model_info, device,
                                            ddim_steps, args.min_sampling_std,
                                            args.zero_qvel, args.max_episode_steps,
                                            record=True)
                if not frames:
                    continue
                tag = f"state{si:03d}_p{p_succ:.2f}"
                if s and not got_s:
                    fp = os.path.join(args.output_dir, f"{tag}_success.mp4")
                    imageio.mimsave(fp, frames, fps=args.fps)
                    got_s = True
                    print(f"    saved {fp} ({len(frames)} frames)")
                elif not s and not got_f:
                    fp = os.path.join(args.output_dir, f"{tag}_failure.mp4")
                    imageio.mimsave(fp, frames, fps=args.fps)
                    got_f = True
                    print(f"    saved {fp} ({len(frames)} frames)")

            n_decisive += 1
            elapsed = time.time() - t0
            print(f"  State {si}: P={p_succ:.2f}  DECISIVE #{n_decisive}  "
                  f"[{n_succ}/{args.mc_samples}]  ({elapsed:.0f}s)")

            if n_decisive >= args.max_videos:
                break
        else:
            if (si + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  Screened {si+1}, found {n_decisive} decisive  ({elapsed:.0f}s)")

    all_p = np.array(all_p)
    print(f"\nDone: screened {len(all_p)} states, {n_decisive} decisive")
    print(f"P distribution: mean={all_p.mean():.3f}, "
          f"P=0: {(all_p == 0).sum()}, P=1: {(all_p == 1).sum()}, "
          f"decisive: {((all_p >= args.p_low) & (all_p <= args.p_high)).sum()}")
    print(f"Videos saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
