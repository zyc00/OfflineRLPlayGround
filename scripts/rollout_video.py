"""Rollout DP or MIP policy, save videos, and classify failures.

Usage:
    # DP
    python scripts/rollout_video.py --type dp \
      --ckpt runs/dppo_pretrain/dp_peg_split900/best.pt \
      --num_episodes 50 --output /tmp/dp_videos

    # MIP
    python scripts/rollout_video.py --type mip \
      --ckpt runs/mip_pretrain/mip_peg_split900/best.pt \
      --num_episodes 50 --output /tmp/mip_videos
"""
import os
import sys
import argparse
import numpy as np
import torch
import gymnasium as gym
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mani_skill.envs
import DPPO.peg_insertion_easy
from mani_skill.utils.wrappers import CPUGymWrapper


def load_dp_model(ckpt_path, device):
    from DPPO.model.unet_wrapper import DiffusionUNet
    from DPPO.model.diffusion import DiffusionModel

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
        final_action_clip_value=1.0,
        predict_epsilon=arch_args.get("predict_epsilon", True),
        mip_noise=arch_args.get("mip_noise", False),
    )
    if arch_args.get("fixed_t_points"):
        model.fixed_t_points = torch.tensor(arch_args["fixed_t_points"], dtype=torch.long)
    state_key = "ema" if "ema" in ckpt else "model"
    model.load_state_dict(ckpt[state_key], strict=False)
    if torch.isnan(model.eta.eta_logit.data).any():
        model.eta.eta_logit.data = torch.atanh(torch.tensor([0.999]))
    model.eval()

    act_offset = cond_steps - 1

    class DPWrapper:
        def __init__(self, m, offset):
            self.model = m
            self.act_offset = offset

        @torch.no_grad()
        def predict(self, cond_obs):
            cond = {"state": cond_obs}
            samples = self.model(cond, deterministic=True, ddim_steps=10)
            return samples.trajectories[:, self.act_offset:]  # (B, act_steps, act_dim)

    wrapper = DPWrapper(model, act_offset)
    ckpt["args"]["cond_steps"] = cond_steps
    ckpt["args"]["horizon_steps"] = horizon_steps
    ckpt["args"]["act_steps"] = act_steps
    return wrapper, ckpt


def load_mip_model(ckpt_path, device):
    from MultiGaussian.models.multi_gaussian import MIPPolicy
    from MultiGaussian.models.mip_unet import MIPUNetPolicy

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    network_type = args.get("network_type", "mlp")
    if network_type == "unet":
        model = MIPUNetPolicy(
            input_dim=ckpt["obs_dim"], action_dim=ckpt["action_dim"],
            cond_steps=args.get("cond_steps", 1),
            horizon_steps=args.get("horizon_steps", 1),
            t_star=args.get("t_star", 0.9),
        ).to(device)
    else:
        model = MIPPolicy(
            input_dim=ckpt["obs_dim"], action_dim=ckpt["action_dim"],
            cond_steps=args.get("cond_steps", 1),
            horizon_steps=args.get("horizon_steps", 1),
            t_star=args.get("t_star", 0.9),
            dropout=args.get("dropout", 0.1),
            emb_dim=args.get("emb_dim", 512),
            n_layers=args.get("n_layers", 6),
            predict_epsilon=args.get("predict_epsilon", False),
        ).to(device)
    state_key = "ema" if "ema" in ckpt else "model"
    model.load_state_dict(ckpt[state_key])
    model.eval()
    return model, ckpt


def save_video(frames, path, fps=20):
    import cv2
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def convert_h264(src_dir, dst_dir):
    """Convert mp4v videos to H.264 for VSCode playback."""
    import subprocess
    os.makedirs(dst_dir, exist_ok=True)
    for f in sorted(os.listdir(src_dir)):
        if f.endswith(".mp4"):
            subprocess.run(
                ["ffmpeg", "-y", "-i", os.path.join(src_dir, f),
                 "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                 "-loglevel", "error", os.path.join(dst_dir, f)],
                check=True,
            )


@torch.no_grad()
def rollout_and_analyze(model, ckpt, n_episodes, env_id, control_mode,
                        max_episode_steps, device, output_dir):
    args = ckpt["args"]
    cond_steps = args.get("cond_steps", 1)
    horizon_steps = args.get("horizon_steps", 1)
    act_steps = args.get("act_steps", 1)
    zero_qvel = args.get("zero_qvel", False)
    no_obs_norm = ckpt.get("no_obs_norm", False)
    no_action_norm = ckpt.get("no_action_norm", False)

    if not no_obs_norm:
        o_lo = ckpt["obs_min"].to(device)
        o_hi = ckpt["obs_max"].to(device)
    if not no_action_norm:
        a_lo = ckpt["action_min"].to(device)
        a_hi = ckpt["action_max"].to(device)

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for ep in range(n_episodes):
        env = gym.make(env_id, obs_mode="state", control_mode=control_mode,
                       render_mode="rgb_array", max_episode_steps=max_episode_steps,
                       reconfiguration_freq=1)
        env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)

        obs, _ = env.reset(seed=ep)
        obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)
        obs_buffer = deque(maxlen=cond_steps)
        for _ in range(cond_steps):
            obs_buffer.append(obs_t.clone())

        frames = []
        actions_log = []
        peg_head_log = []
        success = False
        step = 0

        while step < max_episode_steps:
            if cond_steps == 1:
                cond_obs = obs_buffer[-1]
            else:
                cond_obs = torch.stack(list(obs_buffer), dim=1)

            cond_obs_proc = cond_obs.clone()
            if zero_qvel:
                cond_obs_proc[..., 9:18] = 0.0
            if not no_obs_norm:
                cond_obs_proc = (cond_obs_proc - o_lo) / (o_hi - o_lo + 1e-8) * 2.0 - 1.0

            actions_chunk = model.predict(cond_obs_proc)
            if not no_action_norm:
                actions_chunk = (actions_chunk + 1.0) / 2.0 * (a_hi - a_lo) + a_lo

            n_exec = min(act_steps, max_episode_steps - step) if horizon_steps > 1 else 1
            for t_idx in range(n_exec):
                if horizon_steps > 1:
                    action = actions_chunk[:, t_idx]
                else:
                    action = actions_chunk
                action_np = action.cpu().numpy().squeeze(0)
                actions_log.append(action_np.copy())

                obs_np, rew, terminated, truncated, info = env.step(action_np)
                frames.append(env.render())
                peg_head_log.append(info["peg_head_pos_at_hole"].copy())
                obs_t = torch.from_numpy(obs_np).float().to(device).unsqueeze(0)
                obs_buffer.append(obs_t.clone())
                step += 1

                if info.get("success", False):
                    success = True
                if terminated or truncated:
                    break
            if terminated or truncated:
                frames.append(env.render())
                break

        env.close()

        # Save video
        tag = "S" if success else "F"
        video_path = os.path.join(output_dir, f"ep{ep:03d}_{tag}.mp4")
        save_video(frames, video_path)
        print(f"  ep {ep:3d}: {'SUCCESS' if success else 'FAIL':7s} ({step:3d} steps)")

        # Analyze
        actions_arr = np.array(actions_log)
        peg_arr = np.array(peg_head_log)
        grip = actions_arr[:, 7]
        peg_disp = np.linalg.norm(peg_arr - peg_arr[0], axis=1)

        grip_close_step = -1
        for s in range(len(grip) - 10):
            if grip[s:s+10].mean() < -0.5:
                grip_close_step = s
                break

        results.append({
            "ep": ep, "success": success, "steps": step,
            "max_peg_disp": peg_disp.max(),
            "grip_close_step": grip_close_step,
            "grip_mean_0_30": grip[:30].mean(),
            "grip_mean_30_80": grip[30:80].mean() if len(grip) > 30 else 0,
        })

    # Summary
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    never_grasped = [r for r in failures if r["max_peg_disp"] < 0.05]
    grasped_failed = [r for r in failures if r["max_peg_disp"] >= 0.05]

    print(f"\n{'='*60}")
    print(f"Total: {len(successes)}/{n_episodes} = {len(successes)/n_episodes:.0%}")
    print(f"\nFailure breakdown ({len(failures)} total):")
    if failures:
        print(f"  Never grasped (peg disp < 0.05): {len(never_grasped)} ({len(never_grasped)/len(failures):.0%})")
        print(f"  Grasped but failed insertion:     {len(grasped_failed)} ({len(grasped_failed)/len(failures):.0%})")
    print(f"\nGripper action mean:")
    print(f"  Successes:     steps[0:30]={np.mean([r['grip_mean_0_30'] for r in successes]):.3f}, "
          f"steps[30:80]={np.mean([r['grip_mean_30_80'] for r in successes]):.3f}")
    if never_grasped:
        print(f"  Never-grasped: steps[0:30]={np.mean([r['grip_mean_0_30'] for r in never_grasped]):.3f}, "
              f"steps[30:80]={np.mean([r['grip_mean_30_80'] for r in never_grasped]):.3f}")
    if grasped_failed:
        print(f"  Grasped-fail:  steps[0:30]={np.mean([r['grip_mean_0_30'] for r in grasped_failed]):.3f}, "
              f"steps[30:80]={np.mean([r['grip_mean_30_80'] for r in grasped_failed]):.3f}")

    # Convert to H.264
    h264_dir = output_dir + "_h264"
    print(f"\nConverting to H.264: {h264_dir}")
    convert_h264(output_dir, h264_dir)
    print("Done.")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", required=True, choices=["dp", "mip"])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--env_id", default="PegInsertionSide-v1")
    parser.add_argument("--control_mode", default="pd_joint_delta_pos")
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"Loading {args.type.upper()} from {args.ckpt}...")
    if args.type == "dp":
        model, ckpt = load_dp_model(args.ckpt, device)
    else:
        model, ckpt = load_mip_model(args.ckpt, device)

    # Auto-set zero_qvel for PegInsertion
    if "PegInsertion" in args.env_id:
        ckpt["args"]["zero_qvel"] = True

    print(f"Rolling out {args.num_episodes} episodes...")
    rollout_and_analyze(model, ckpt, args.num_episodes, args.env_id,
                        args.control_mode, args.max_episode_steps, device, args.output)


if __name__ == "__main__":
    main()
