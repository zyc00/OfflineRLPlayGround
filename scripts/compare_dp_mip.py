"""Compare DP vs MIP policy trajectories on the same initial states (CPU eval).

Uses CPU envs with reconfiguration_freq=1 so each episode gets a new random initial state.
Both policies are evaluated on the same sequence of episodes (same seeds).

Usage:
    python scripts/compare_dp_mip.py \
      --dp_ckpt runs/dppo_pretrain/dppo_pretrain_peg_policy_demos_200k/best.pt \
      --mip_ckpt runs/mip_pretrain/mip_peg_chunk16/best.pt \
      --num_episodes 200 --output /tmp/dp_mip_compare
"""
import os
import sys
import argparse
import numpy as np
import torch
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mani_skill.envs
import DPPO.peg_insertion_easy
from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack

from DPPO.model.unet_wrapper import DiffusionUNet
from DPPO.model.diffusion import DiffusionModel
from MultiGaussian.models.multi_gaussian import MIPPolicy
from MultiGaussian.models.mip_unet import MIPUNetPolicy


def load_dp_model(ckpt_path, device):
    """Load a DPPO-format checkpoint."""
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
    if arch_args.get("cascade_start"):
        model.cascade_start = arch_args["cascade_start"]

    state_key = "ema" if "ema" in ckpt else "model"
    raw_sd = ckpt[state_key]
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
    if torch.isnan(model.eta.eta_logit.data).any():
        model.eta.eta_logit.data = torch.atanh(torch.tensor([0.999]))
    model.eval()
    return model, ckpt


def load_mip_model(ckpt_path, device):
    """Load a MIP-format checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]

    network_type = args.get("network_type", "mlp")
    if network_type == "unet":
        model = MIPUNetPolicy(
            input_dim=ckpt["obs_dim"],
            action_dim=ckpt["action_dim"],
            cond_steps=args.get("cond_steps", 1),
            horizon_steps=args.get("horizon_steps", 1),
            t_star=args.get("t_star", 0.9),
        ).to(device)
    else:
        model = MIPPolicy(
            input_dim=ckpt["obs_dim"],
            action_dim=ckpt["action_dim"],
            cond_steps=args.get("cond_steps", 1),
            horizon_steps=args.get("horizon_steps", 1),
            t_star=args.get("t_star", 0.9),
            dropout=args.get("dropout", 0.1),
            emb_dim=args.get("emb_dim", 512),
            n_layers=args.get("n_layers", 6),
            predict_epsilon=args.get("predict_epsilon", False),
        ).to(device)

    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def collect_trajectories_dp(model, ckpt, n_episodes, env_id, control_mode,
                             max_episode_steps, device, num_envs=10):
    """Collect trajectories from DP model (CPU envs with FrameStack)."""
    args = ckpt.get("pretrain_args", ckpt["args"]) if "pretrain_args" in ckpt else ckpt["args"]
    cond_steps = args.get("cond_steps", 2)
    act_steps = args.get("act_steps", 8)
    act_offset = cond_steps - 1

    def make_env(seed):
        def thunk():
            env = gym.make(env_id, obs_mode="state", control_mode=control_mode,
                          render_mode="rgb_array", max_episode_steps=max_episode_steps,
                          reconfiguration_freq=1)
            env = FrameStack(env, num_stack=cond_steps)
            env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
            return env
        return thunk
    envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(num_envs)])

    all_trajs = []
    eps_done = 0

    while eps_done < n_episodes:
        obs, _ = envs.reset()
        obs = torch.from_numpy(obs).float().to(device)
        traj_obs = [[] for _ in range(num_envs)]
        traj_act = [[] for _ in range(num_envs)]
        traj_rew = [[] for _ in range(num_envs)]

        for step in range(max_episode_steps):
            obs_cond = obs.clone()
            obs_cond[..., 9:18] = 0.0  # zero_qvel
            cond = {"state": obs_cond}
            samples = model(cond, deterministic=True, ddim_steps=10)
            action_chunk = samples.trajectories
            action_np = action_chunk[:, act_offset:act_offset+act_steps].cpu().numpy()

            for e in range(num_envs):
                traj_obs[e].append(obs[e, -1].cpu().numpy())  # last frame

            for a_idx in range(action_np.shape[1]):
                for e in range(num_envs):
                    traj_act[e].append(action_np[e, a_idx])
                obs_np, rew, terminated, truncated, info = envs.step(action_np[:, a_idx])
                for e in range(num_envs):
                    traj_rew[e].append(rew[e])
                if truncated.any():
                    break

            obs = torch.from_numpy(obs_np).float().to(device)
            if truncated.any():
                for fi_idx, fi in enumerate(info.get("final_info", [])):
                    if fi and "episode" in fi:
                        all_trajs.append({
                            "obs": np.array(traj_obs[fi_idx]),
                            "actions": np.array(traj_act[fi_idx]),
                            "rewards": np.array(traj_rew[fi_idx]),
                            "success": fi["episode"]["success_once"],
                        })
                eps_done += num_envs
                break

    envs.close()
    return all_trajs[:n_episodes]


@torch.no_grad()
def collect_trajectories_mip(model, ckpt, n_episodes, env_id, control_mode,
                              max_episode_steps, device, num_envs=10):
    """Collect trajectories from MIP model (CPU envs)."""
    args = ckpt["args"]
    cond_steps = args.get("cond_steps", 1)
    horizon_steps = args.get("horizon_steps", 1)
    act_steps = args.get("act_steps", 1)
    no_obs_norm = ckpt.get("no_obs_norm", False)
    no_action_norm = ckpt.get("no_action_norm", False)

    if not no_obs_norm:
        o_lo = ckpt["obs_min"].to(device)
        o_hi = ckpt["obs_max"].to(device)
    if not no_action_norm:
        a_lo = ckpt["action_min"].to(device)
        a_hi = ckpt["action_max"].to(device)

    def make_env(seed):
        def thunk():
            env = gym.make(env_id, obs_mode="state", control_mode=control_mode,
                          render_mode="rgb_array", max_episode_steps=max_episode_steps,
                          reconfiguration_freq=1)
            env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
            return env
        return thunk
    envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(num_envs)])

    all_trajs = []
    eps_done = 0

    while eps_done < n_episodes:
        obs, _ = envs.reset()
        obs = torch.from_numpy(obs).float().to(device)
        obs_buffer = deque(maxlen=cond_steps)
        for _ in range(cond_steps):
            obs_buffer.append(obs.clone())

        traj_obs = [[] for _ in range(num_envs)]
        traj_act = [[] for _ in range(num_envs)]
        traj_rew = [[] for _ in range(num_envs)]

        step = 0
        done = False
        while step < max_episode_steps and not done:
            if cond_steps == 1:
                cond_obs = obs_buffer[-1]
            else:
                cond_obs = torch.stack(list(obs_buffer), dim=1)

            cond_obs_proc = cond_obs.clone()
            cond_obs_proc[..., 9:18] = 0.0  # zero_qvel

            if not no_obs_norm:
                cond_obs_proc = (cond_obs_proc - o_lo) / (o_hi - o_lo + 1e-8) * 2.0 - 1.0

            actions_chunk = model.predict(cond_obs_proc)
            if not no_action_norm:
                actions_chunk = (actions_chunk + 1.0) / 2.0 * (a_hi - a_lo) + a_lo

            for e in range(num_envs):
                traj_obs[e].append(obs_buffer[-1][e].cpu().numpy())

            n_exec = min(act_steps, max_episode_steps - step) if horizon_steps > 1 else 1
            for t_idx in range(n_exec):
                if horizon_steps > 1:
                    action = actions_chunk[:, t_idx]
                else:
                    action = actions_chunk
                action_np = action.cpu().numpy()
                for e in range(num_envs):
                    traj_act[e].append(action_np[e])

                obs_np, rew, terminated, truncated, info = envs.step(action_np)
                obs = torch.from_numpy(obs_np).float().to(device)
                obs_buffer.append(obs.clone())
                for e in range(num_envs):
                    traj_rew[e].append(rew[e])
                step += 1

                if truncated.any():
                    for fi_idx, fi in enumerate(info.get("final_info", [])):
                        if fi and "episode" in fi:
                            all_trajs.append({
                                "obs": np.array(traj_obs[fi_idx]),
                                "actions": np.array(traj_act[fi_idx]),
                                "rewards": np.array(traj_rew[fi_idx]),
                                "success": fi["episode"]["success_once"],
                            })
                    eps_done += num_envs
                    done = True
                    break

    envs.close()
    return all_trajs[:n_episodes]


def analyze(dp_trajs, mip_trajs, output_dir):
    """Analyze and compare trajectories."""
    n = min(len(dp_trajs), len(mip_trajs))
    dp_sr = np.mean([t["success"] for t in dp_trajs[:n]])
    mip_sr = np.mean([t["success"] for t in mip_trajs[:n]])

    print(f"\n{'='*60}")
    print(f"DP vs MIP: {n} episodes each (CPU eval, deterministic)")
    print(f"{'='*60}")
    print(f"  DP SR:  {dp_sr:.1%} ({sum(t['success'] for t in dp_trajs[:n])}/{n})")
    print(f"  MIP SR: {mip_sr:.1%} ({sum(t['success'] for t in mip_trajs[:n])}/{n})")

    # Success/failure categories (not paired — different initial states)
    dp_succ = [t for t in dp_trajs[:n] if t["success"]]
    dp_fail = [t for t in dp_trajs[:n] if not t["success"]]
    mip_succ = [t for t in mip_trajs[:n] if t["success"]]
    mip_fail = [t for t in mip_trajs[:n] if not t["success"]]

    # Episode length analysis
    dp_succ_lens = [len(t["actions"]) for t in dp_succ]
    mip_succ_lens = [len(t["actions"]) for t in mip_succ]
    dp_fail_lens = [len(t["actions"]) for t in dp_fail]
    mip_fail_lens = [len(t["actions"]) for t in mip_fail]

    print(f"\n--- Episode Length ---")
    if dp_succ_lens:
        print(f"  DP  success: {np.mean(dp_succ_lens):.1f} ± {np.std(dp_succ_lens):.1f} steps "
              f"(min={min(dp_succ_lens)}, max={max(dp_succ_lens)})")
    if mip_succ_lens:
        print(f"  MIP success: {np.mean(mip_succ_lens):.1f} ± {np.std(mip_succ_lens):.1f} steps "
              f"(min={min(mip_succ_lens)}, max={max(mip_succ_lens)})")

    # Action statistics
    dp_all_actions = np.concatenate([t["actions"] for t in dp_trajs[:n]])
    mip_all_actions = np.concatenate([t["actions"] for t in mip_trajs[:n]])
    act_dim = dp_all_actions.shape[1]

    print(f"\n--- Action Statistics (all steps) ---")
    print(f"  {'Joint':>6s}  {'DP mean':>8s} {'DP std':>8s}  {'MIP mean':>8s} {'MIP std':>8s}  {'Δmean':>8s}")
    for j in range(act_dim):
        dm, ds = dp_all_actions[:, j].mean(), dp_all_actions[:, j].std()
        mm, ms = mip_all_actions[:, j].mean(), mip_all_actions[:, j].std()
        print(f"  j{j:d}:    {dm:+8.4f} {ds:8.4f}  {mm:+8.4f} {ms:8.4f}  {mm-dm:+8.4f}")

    # Action norm over time (for successful episodes)
    print(f"\n--- Action Norm Over Time (successful episodes) ---")
    max_len = 200
    dp_act_norms = np.full((len(dp_succ), max_len), np.nan)
    mip_act_norms = np.full((len(mip_succ), max_len), np.nan)
    for i, t in enumerate(dp_succ):
        L = min(len(t["actions"]), max_len)
        dp_act_norms[i, :L] = np.linalg.norm(t["actions"][:L], axis=1)
    for i, t in enumerate(mip_succ):
        L = min(len(t["actions"]), max_len)
        mip_act_norms[i, :L] = np.linalg.norm(t["actions"][:L], axis=1)

    for t in [0, 10, 25, 50, 75, 100, 150]:
        dp_val = np.nanmean(dp_act_norms[:, t]) if t < max_len else 0
        mip_val = np.nanmean(mip_act_norms[:, t]) if t < max_len else 0
        dp_cnt = np.sum(~np.isnan(dp_act_norms[:, t]))
        mip_cnt = np.sum(~np.isnan(mip_act_norms[:, t]))
        print(f"  step {t:3d}: DP |a|={dp_val:.4f} (n={dp_cnt}), MIP |a|={mip_val:.4f} (n={mip_cnt})")

    # Gripper action analysis (last joint, j7)
    dp_gripper = dp_all_actions[:, -1]
    mip_gripper = mip_all_actions[:, -1]
    print(f"\n--- Gripper Action (j7) ---")
    print(f"  DP:  mean={dp_gripper.mean():+.4f}, std={dp_gripper.std():.4f}, "
          f"frac>0={( dp_gripper > 0).mean():.1%}, frac<0={(dp_gripper < 0).mean():.1%}")
    print(f"  MIP: mean={mip_gripper.mean():+.4f}, std={mip_gripper.std():.4f}, "
          f"frac>0={(mip_gripper > 0).mean():.1%}, frac<0={(mip_gripper < 0).mean():.1%}")

    # Success vs failure action comparison
    if dp_succ and dp_fail:
        dp_succ_acts = np.concatenate([t["actions"] for t in dp_succ])
        dp_fail_acts = np.concatenate([t["actions"] for t in dp_fail])
        print(f"\n--- DP Success vs Failure Actions ---")
        print(f"  {'Joint':>6s}  {'succ mean':>9s} {'fail mean':>9s}  {'Δ':>8s}")
        for j in range(act_dim):
            sm = dp_succ_acts[:, j].mean()
            fm = dp_fail_acts[:, j].mean()
            print(f"  j{j:d}:    {sm:+9.4f} {fm:+9.4f}  {sm-fm:+8.4f}")

    if mip_succ and mip_fail:
        mip_succ_acts = np.concatenate([t["actions"] for t in mip_succ])
        mip_fail_acts = np.concatenate([t["actions"] for t in mip_fail])
        print(f"\n--- MIP Success vs Failure Actions ---")
        print(f"  {'Joint':>6s}  {'succ mean':>9s} {'fail mean':>9s}  {'Δ':>8s}")
        for j in range(act_dim):
            sm = mip_succ_acts[:, j].mean()
            fm = mip_fail_acts[:, j].mean()
            print(f"  j{j:d}:    {sm:+9.4f} {fm:+9.4f}  {sm-fm:+8.4f}")

    # Per-phase action variance (higher variance = less precise)
    print(f"\n--- Action Variance by Phase (successful episodes) ---")
    phases = [(0, 50), (50, 100), (100, 150), (150, 200)]
    for lo, hi in phases:
        dp_phase = []
        mip_phase = []
        for t in dp_succ:
            a = t["actions"]
            if len(a) > lo:
                dp_phase.append(a[lo:min(hi, len(a))])
        for t in mip_succ:
            a = t["actions"]
            if len(a) > lo:
                mip_phase.append(a[lo:min(hi, len(a))])
        if dp_phase and mip_phase:
            dp_var = np.concatenate(dp_phase).var(axis=0).mean()
            mip_var = np.concatenate(mip_phase).var(axis=0).mean()
            print(f"  step {lo:3d}-{hi:3d}: DP var={dp_var:.5f}, MIP var={mip_var:.5f}, "
                  f"ratio={mip_var/dp_var:.2f}x")

    # ========== PLOTS ==========
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Success rate bar
    ax = axes[0, 0]
    ax.bar(["DP", "MIP"], [dp_sr, mip_sr], color=["steelblue", "coral"], alpha=0.8, edgecolor="black")
    ax.set_ylabel("Success Rate")
    ax.set_title("Overall Success Rate")
    ax.set_ylim(0, 1)
    for i, v in enumerate([dp_sr, mip_sr]):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=12, fontweight="bold")

    # 2. Episode length histogram (successful)
    ax = axes[0, 1]
    if dp_succ_lens:
        ax.hist(dp_succ_lens, bins=20, alpha=0.6, label=f"DP ({len(dp_succ)})", color="steelblue")
    if mip_succ_lens:
        ax.hist(mip_succ_lens, bins=20, alpha=0.6, label=f"MIP ({len(mip_succ)})", color="coral")
    ax.set_xlabel("Episode Length (steps)")
    ax.set_ylabel("Count")
    ax.set_title("Successful Episode Lengths")
    ax.legend()

    # 3. Per-joint action std
    ax = axes[0, 2]
    dp_std = dp_all_actions.std(axis=0)
    mip_std = mip_all_actions.std(axis=0)
    x = np.arange(act_dim)
    ax.bar(x - 0.15, dp_std, width=0.3, label="DP", alpha=0.7, color="steelblue")
    ax.bar(x + 0.15, mip_std, width=0.3, label="MIP", alpha=0.7, color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels([f"j{i}" for i in range(act_dim)])
    ax.set_ylabel("Action Std")
    ax.set_title("Per-Joint Action Variability")
    ax.legend()

    # 4. Action norm over time (successful)
    ax = axes[1, 0]
    if dp_succ:
        ax.plot(np.nanmean(dp_act_norms, axis=0), color="steelblue", label="DP succ", lw=1.5)
    if mip_succ:
        ax.plot(np.nanmean(mip_act_norms, axis=0), color="coral", label="MIP succ", lw=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("|action|")
    ax.set_title("Action Magnitude (Successful Episodes)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Gripper action distribution
    ax = axes[1, 1]
    ax.hist(dp_gripper, bins=50, alpha=0.5, label="DP", color="steelblue", density=True)
    ax.hist(mip_gripper, bins=50, alpha=0.5, label="MIP", color="coral", density=True)
    ax.set_xlabel("Gripper action (j7)")
    ax.set_ylabel("Density")
    ax.set_title("Gripper Action Distribution")
    ax.legend()

    # 6. Per-joint mean action comparison
    ax = axes[1, 2]
    dp_mean = dp_all_actions.mean(axis=0)
    mip_mean = mip_all_actions.mean(axis=0)
    ax.bar(x - 0.15, dp_mean, width=0.3, label="DP", alpha=0.7, color="steelblue")
    ax.bar(x + 0.15, mip_mean, width=0.3, label="MIP", alpha=0.7, color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels([f"j{i}" for i in range(act_dim)])
    ax.set_ylabel("Mean Action")
    ax.set_title("Per-Joint Mean Action")
    ax.legend()
    ax.axhline(0, color="black", lw=0.5)

    plt.suptitle(f"DP ({dp_sr:.0%}) vs MIP ({mip_sr:.0%}) — PegInsertionSide (CPU eval)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "dp_vs_mip_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot: {plot_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp_ckpt", required=True)
    parser.add_argument("--mip_ckpt", required=True)
    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument("--env_id", default="PegInsertionSide-v1")
    parser.add_argument("--control_mode", default="pd_joint_delta_pos")
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--output", default="/tmp/dp_mip_compare")
    args = parser.parse_args()

    device = torch.device("cuda")
    os.makedirs(args.output, exist_ok=True)

    print("Loading DP model...")
    dp_model, dp_ckpt = load_dp_model(args.dp_ckpt, device)
    print("Loading MIP model...")
    mip_model, mip_ckpt = load_mip_model(args.mip_ckpt, device)

    print(f"\nCollecting {args.num_episodes} DP trajectories (CPU)...")
    dp_trajs = collect_trajectories_dp(dp_model, dp_ckpt, args.num_episodes,
                                        args.env_id, args.control_mode,
                                        args.max_episode_steps, device)
    dp_sr = np.mean([t["success"] for t in dp_trajs])
    print(f"  DP SR: {dp_sr:.1%}")

    print(f"Collecting {args.num_episodes} MIP trajectories (CPU)...")
    mip_trajs = collect_trajectories_mip(mip_model, mip_ckpt, args.num_episodes,
                                          args.env_id, args.control_mode,
                                          args.max_episode_steps, device)
    mip_sr = np.mean([t["success"] for t in mip_trajs])
    print(f"  MIP SR: {mip_sr:.1%}")

    analyze(dp_trajs, mip_trajs, args.output)


if __name__ == "__main__":
    main()
