"""Analyze: distance from eval init state to training demos → P(success).

Init state randomness is 5D: cube_xy (2D) + goal_xyz (3D). Cube z is fixed at 0.02.
Computes nearest-demo distance in 5D, cube_xy-only, and goal_xyz-only spaces.

Usage:
  python analyze_init_similarity.py
  python analyze_init_similarity.py --num-train-demos 50 --num-states 100
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py
import torch
from dataclasses import dataclass
from typing import Optional
import tyro
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack
from scipy.stats import spearmanr, pearsonr, pointbiserialr
from dp_p_success_cpu import run_mc
from DPPO.model.unet_wrapper import DiffusionUNet
from DPPO.model.diffusion import DiffusionModel


@dataclass
class Args:
    ckpt: str = "runs/dppo_pretrain/dppo_25traj_aligned_v2/best.pt"
    demo_path: str = "~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5"
    num_train_demos: int = 25
    num_states: int = 50
    mc_samples: int = 16
    max_episode_steps: int = 100
    env_id: str = "PickCube-v1"
    control_mode: str = "pd_ee_delta_pos"
    obs_horizon: int = 2
    seed: int = 0
    output: Optional[str] = None


# Flat obs dim indices for PickCube-v1 state obs (42D, pd_ee_delta_pos)
OBJ_X, OBJ_Y = 29, 30
GOAL_X, GOAL_Y, GOAL_Z = 26, 27, 28
INIT_DIMS = [OBJ_X, OBJ_Y, GOAL_X, GOAL_Y, GOAL_Z]
DIM_NAMES = ["cube_x", "cube_y", "goal_x", "goal_y", "goal_z"]


def extract_init_features(obs):
    """Extract 5D init features from flat obs."""
    return np.array([obs[d] for d in INIT_DIMS])


def main():
    args = tyro.cli(Args)
    args.demo_path = os.path.expanduser(args.demo_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 1) Demo init features (first N = training set) ===
    f = h5py.File(args.demo_path, "r")
    traj_keys = sorted([k for k in f.keys() if k.startswith("traj_")],
                       key=lambda x: int(x.split("_")[-1]))
    demo_feats_train = []
    demo_feats_all = []
    for i, tk in enumerate(traj_keys):
        obs0 = f[tk]["obs"][0]
        feat = extract_init_features(obs0)
        demo_feats_all.append(feat)
        if i < args.num_train_demos:
            demo_feats_train.append(feat)
    f.close()
    demo_feats_train = np.array(demo_feats_train)  # (N_train, 5)
    demo_feats_all = np.array(demo_feats_all)      # (1000, 5)
    print(f"Training demos: {len(demo_feats_train)}, All demos: {len(demo_feats_all)}")

    # === 2) Eval init features ===
    env = gym.make(args.env_id, obs_mode="state", render_mode="rgb_array",
                   reward_mode="sparse", control_mode=args.control_mode,
                   max_episode_steps=args.max_episode_steps, reconfiguration_freq=1)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = FrameStack(env, num_stack=args.obs_horizon)
    env = CPUGymWrapper(env, ignore_terminations=False, record_metrics=True)

    eval_feats = []
    for s in range(args.num_states):
        obs, _ = env.reset(seed=args.seed + s)
        obs_last = np.array(obs[-1]) if obs.ndim > 1 else np.array(obs)
        eval_feats.append(extract_init_features(obs_last))
    env.close()
    eval_feats = np.array(eval_feats)  # (N_states, 5)

    # === 3) Distances (5D, cube_xy, goal_xyz) ===
    dist_5d = np.array([np.linalg.norm(demo_feats_train - eval_feats[i], axis=1).min()
                         for i in range(args.num_states)])
    dist_obj = np.array([np.linalg.norm(demo_feats_train[:, :2] - eval_feats[i, :2], axis=1).min()
                          for i in range(args.num_states)])
    dist_goal = np.array([np.linalg.norm(demo_feats_train[:, 2:] - eval_feats[i, 2:], axis=1).min()
                           for i in range(args.num_states)])
    print(f"Dist 5D:      min={dist_5d.min():.4f}, max={dist_5d.max():.4f}, mean={dist_5d.mean():.4f}")
    print(f"Dist cube_xy: min={dist_obj.min():.4f}, max={dist_obj.max():.4f}, mean={dist_obj.mean():.4f}")
    print(f"Dist goal:    min={dist_goal.min():.4f}, max={dist_goal.max():.4f}, mean={dist_goal.mean():.4f}")

    # === 4) Load model ===
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ckpt_args = ckpt["args"]
    obs_dim, act_dim = ckpt["obs_dim"], ckpt["action_dim"]
    cond_steps = ckpt_args.get("cond_steps", 2)
    horizon_steps = ckpt_args.get("horizon_steps", 16)
    act_steps = ckpt_args.get("act_steps", 8)
    act_offset = cond_steps - 1

    network = DiffusionUNet(
        action_dim=act_dim, horizon_steps=horizon_steps,
        cond_dim=obs_dim * cond_steps,
        diffusion_step_embed_dim=64, down_dims=[64, 128, 256], n_groups=8,
    )
    model = DiffusionModel(
        network=network, horizon_steps=horizon_steps,
        obs_dim=obs_dim, action_dim=act_dim, device=device,
        denoising_steps=100, denoised_clip_value=1.0,
        randn_clip_value=10, final_action_clip_value=1.0,
        predict_epsilon=True, base_eta=1.0,
    )
    model.load_state_dict(ckpt["ema"], strict=False)
    model.eval()

    class Agent:
        def __init__(self, m, det, min_std, ddim):
            self.m, self.det, self.min_std, self.ddim = m, det, min_std, ddim
        def get_action(self, obs_seq):
            cond = {"state": obs_seq}
            s = self.m(cond, deterministic=self.det,
                       min_sampling_denoising_std=self.min_std, ddim_steps=self.ddim)
            return s.trajectories[:, act_offset:act_offset + act_steps]

    # === 5) Run coverage ===
    mc_kwargs = dict(num_states=args.num_states, mc_samples=args.mc_samples,
                     max_steps=args.max_episode_steps, device=device,
                     seed_offset=args.seed, env_id=args.env_id,
                     control_mode=args.control_mode, obs_horizon=args.obs_horizon)

    print("\n--- Deterministic DDPM100 ---")
    p_det = run_mc(Agent(model, True, None, None), **mc_kwargs)

    print("\n--- DDIM10 eta=1 std=0.01 ---")
    p_std = run_mc(Agent(model, False, 0.01, 10), **mc_kwargs)

    # === 6) Correlations ===
    print(f"\n{'='*70}")
    print(f"  Per-dimension correlation with P(success)")
    print(f"{'='*70}")
    for label, p in [("Det DDPM100", p_det), ("DDIM10 std=0.01", p_std)]:
        print(f"\n  {label}:")
        print(f"    {'dim':<10} {'rho':>8} {'p-val':>10} {'mean_hi':>10} {'mean_lo':>10}")
        print(f"    {'-'*50}")
        for i, name in enumerate(DIM_NAMES):
            rho, pv = spearmanr(eval_feats[:, i], p)
            hi = eval_feats[p > 0.5, i].mean() if (p > 0.5).sum() > 0 else float('nan')
            lo = eval_feats[p <= 0.5, i].mean() if (p <= 0.5).sum() > 0 else float('nan')
            print(f"    {name:<10} {rho:>8.3f} {pv:>10.4f} {hi:>10.4f} {lo:>10.4f}")

    print(f"\n{'='*70}")
    print(f"  Distance-to-nearest-demo vs P(success)")
    print(f"{'='*70}")
    print(f"  {'Setting':<25} {'dist_type':<12} {'Spearman':>10} {'p-val':>10} {'Pearson':>10} {'p-val':>10}")
    for label, p in [("Det DDPM100", p_det), ("DDIM10 std=0.01", p_std)]:
        for dname, d in [("5D", dist_5d), ("cube_xy", dist_obj), ("goal_xyz", dist_goal)]:
            rho, pv = spearmanr(d, p)
            r, pr = pearsonr(d, p)
            print(f"  {label:<25} {dname:<12} {rho:>10.3f} {pv:>10.4f} {r:>10.3f} {pr:>10.4f}")

    # === 7) Plot ===
    fig = plt.figure(figsize=(18, 14))

    # Layout: 3 rows x 3 cols
    # Row 0: Det — 5D dist, cube_xy dist, goal_xyz dist
    # Row 1: DDIM — 5D dist, cube_xy dist, goal_xyz dist
    # Row 2: 2D scatter maps (cube_xy and goal_xy colored by P(success))

    for row, (p, label, color) in enumerate([
        (p_det, "Deterministic DDPM100", "steelblue"),
        (p_std, "DDIM10 eta=1 std=0.01", "coral"),
    ]):
        for col, (d, dname) in enumerate([
            (dist_5d, "5D (cube_xy+goal_xyz)"),
            (dist_obj, "cube_xy only"),
            (dist_goal, "goal_xyz only"),
        ]):
            ax = fig.add_subplot(3, 3, row * 3 + col + 1)
            rho, pv = spearmanr(d, p)
            r, pr = pearsonr(d, p)
            ax.scatter(d, p, c=color, s=50, edgecolors="black", linewidth=0.5, alpha=0.8)
            z = np.polyfit(d, p, 1)
            x_line = np.linspace(d.min(), d.max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), "k--", alpha=0.5, linewidth=1.5)
            ax.set_xlabel(f"dist to nearest train demo ({dname})")
            ax.set_ylabel("P(success|s\u2080)")
            ax.set_title(f"{label}\n\u03c1={rho:.3f} (p={pv:.3f}), r={r:.3f}")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)

    # Row 2: 2D scatter maps
    for col, (p, label) in enumerate([(p_det, "Deterministic"), (p_std, "DDIM10 std=0.01")]):
        # Cube xy
        ax = fig.add_subplot(3, 3, 7 + col)
        ax.scatter(demo_feats_all[:, 0], demo_feats_all[:, 1],
                   c="lightgrey", s=6, alpha=0.3, label="all demos")
        ax.scatter(demo_feats_train[:, 0], demo_feats_train[:, 1],
                   c="blue", s=40, marker="x", linewidth=1.5, zorder=4,
                   label=f"train ({args.num_train_demos})")
        sc = ax.scatter(eval_feats[:, 0], eval_feats[:, 1], c=p, cmap="RdYlGn",
                        vmin=0, vmax=1, s=80, edgecolors="black", linewidth=0.5, zorder=5)
        plt.colorbar(sc, ax=ax, label="P(success)")
        ax.set_xlabel("cube x")
        ax.set_ylabel("cube y")
        ax.set_title(f"{label}: cube xy")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_xlim(-0.12, 0.12)
        ax.set_ylim(-0.12, 0.12)
        ax.set_aspect("equal")

    # Goal z vs P(success) — potentially dominant factor
    ax = fig.add_subplot(3, 3, 9)
    for p, label, color in [(p_det, "Det", "steelblue"), (p_std, "DDIM10", "coral")]:
        rho, pv = spearmanr(eval_feats[:, 4], p)
        ax.scatter(eval_feats[:, 4], p, c=color, s=50, edgecolors="black",
                   linewidth=0.5, alpha=0.7, label=f"{label} (\u03c1={rho:.2f})")
    ax.set_xlabel("goal z")
    ax.set_ylabel("P(success|s\u2080)")
    ax.set_title("goal_z vs P(success)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"DP Init State \u2192 Nearest Training Demo Distance vs Success\n"
        f"PickCube {args.num_train_demos}-traj pretrain, "
        f"{args.num_states} states, MC{args.mc_samples}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    out = args.output or f"runs/dp_init_similarity_vs_success_{args.num_train_demos}traj.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
