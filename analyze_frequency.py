"""Analyze spatial frequency characteristics of obs→action mapping in ManiSkill BC datasets.

Three analyses:
1. State-Action Distance Scatter — global Lipschitz structure
2. KNN Sweep per State Dimension — per-dim frequency profile
3. Trajectory Roughness — temporal smoothness along trajectories
"""

import argparse
import json
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def load_h5_data(data):
    out = {}
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


def load_dataset(demo_path, num_demos=None):
    """Load demo dataset, returning stacked arrays and per-trajectory lists."""
    f = h5py.File(demo_path, "r")
    json_path = demo_path.replace(".h5", ".json")
    with open(json_path, "r") as jf:
        json_data = json.load(jf)
    episodes = json_data["episodes"]

    if num_demos is None:
        num_demos = len(episodes)
    else:
        num_demos = min(num_demos, len(episodes))

    obs_list = []
    act_list = []
    print(f"Loading {num_demos} episodes from {demo_path}")
    for i in tqdm(range(num_demos)):
        eps = episodes[i]
        traj = load_h5_data(f[f"traj_{eps['episode_id']}"])
        obs_list.append(traj["obs"][:-1])
        act_list.append(traj["actions"])

    all_obs = np.vstack(obs_list)
    all_actions = np.vstack(act_list)
    print(f"Loaded {all_obs.shape[0]} transitions, obs_dim={all_obs.shape[1]}, action_dim={all_actions.shape[1]}")
    return all_obs, all_actions, obs_list, act_list


# ── Analysis 1: State-Action Distance Scatter ──────────────────────────────


def analysis1_distance_scatter(all_obs, all_actions, output_dir, num_pairs=50000):
    print("\n=== Analysis 1: State-Action Distance Scatter ===")
    n = len(all_obs)
    rng = np.random.default_rng(42)
    idx1 = rng.integers(0, n, size=num_pairs)
    idx2 = rng.integers(0, n, size=num_pairs)
    # avoid self-pairs
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]

    ds = np.linalg.norm(all_obs[idx1] - all_obs[idx2], axis=1)
    da = np.linalg.norm(all_actions[idx1] - all_actions[idx2], axis=1)

    eps = 1e-8
    lipschitz = da / (ds + eps)

    print(f"  ||Δs|| range: [{ds.min():.4f}, {ds.max():.4f}], mean={ds.mean():.4f}")
    print(f"  ||Δa|| range: [{da.min():.4f}, {da.max():.4f}], mean={da.mean():.4f}")
    print(f"  Lipschitz ratio mean={lipschitz.mean():.4f}, median={np.median(lipschitz):.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Subplot 1: scatter
    ax = axes[0]
    subsample = min(10000, len(ds))
    sidx = rng.choice(len(ds), subsample, replace=False)
    ax.scatter(ds[sidx], da[sidx], alpha=0.1, s=2, c="steelblue")
    ax.set_xlabel("||Δs||")
    ax.set_ylabel("||Δa||")
    ax.set_title("State-Action Distance Pairs")

    # Subplot 2: binned mean ± std
    ax = axes[1]
    bins = np.linspace(ds.min(), np.percentile(ds, 99), 21)
    bin_idx = np.digitize(ds, bins) - 1
    bin_means, bin_stds, bin_centers = [], [], []
    for b in range(len(bins) - 1):
        m = bin_idx == b
        if m.sum() > 10:
            bin_means.append(da[m].mean())
            bin_stds.append(da[m].std())
            bin_centers.append((bins[b] + bins[b + 1]) / 2)
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    bin_centers = np.array(bin_centers)
    ax.plot(bin_centers, bin_means, "o-", color="steelblue")
    ax.fill_between(bin_centers, bin_means - bin_stds, bin_means + bin_stds, alpha=0.2, color="steelblue")
    ax.set_xlabel("||Δs|| (binned)")
    ax.set_ylabel("||Δa|| (mean ± std)")
    ax.set_title("Binned Action Distance")

    # Subplot 3: Lipschitz histogram
    ax = axes[2]
    clip_val = np.percentile(lipschitz, 99)
    lip_clipped = lipschitz[lipschitz <= clip_val]
    ax.hist(lip_clipped, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(np.median(lipschitz), color="red", linestyle="--", label=f"median={np.median(lipschitz):.3f}")
    ax.set_xlabel("||Δa|| / ||Δs||")
    ax.set_ylabel("Count")
    ax.set_title("Local Lipschitz Distribution")
    ax.legend()

    fig.suptitle("Analysis 1: State-Action Distance Structure", fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig1_distance_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Analysis 2: KNN Sweep per State Dimension ──────────────────────────────


def analysis2_knn_sweep(all_obs, all_actions, output_dir, k=20, n_sweep=200):
    print("\n=== Analysis 2: KNN Sweep per State Dimension ===")
    obs_var = np.var(all_obs, axis=0)
    top_dims = np.argsort(obs_var)[::-1][:6]
    print(f"  Top-6 variance dims: {top_dims.tolist()}")
    print(f"  Variances: {obs_var[top_dims].tolist()}")

    anchor = np.median(all_obs, axis=0)

    print(f"  Fitting KNN (k={k}) on {len(all_obs)} samples...")
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(all_obs)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for i, dim in enumerate(top_dims):
        ax = axes[i // 3, i % 3]
        lo = np.percentile(all_obs[:, dim], 5)
        hi = np.percentile(all_obs[:, dim], 95)
        sweep_vals = np.linspace(lo, hi, n_sweep)

        # build query points: anchor repeated, with swept dim varying
        queries = np.tile(anchor, (n_sweep, 1))
        queries[:, dim] = sweep_vals

        dists, idxs = nn.kneighbors(queries)
        # action statistics from neighbors
        act_means = []
        act_stds = []
        for j in range(n_sweep):
            neighbor_actions = all_actions[idxs[j]]
            act_norms = np.linalg.norm(neighbor_actions, axis=1)
            act_means.append(act_norms.mean())
            act_stds.append(act_norms.std())
        act_means = np.array(act_means)
        act_stds = np.array(act_stds)

        ax.plot(sweep_vals, act_means, color="steelblue", linewidth=1.5)
        ax.fill_between(sweep_vals, act_means - act_stds, act_means + act_stds,
                         alpha=0.2, color="steelblue")
        ax.set_xlabel(f"obs[{dim}]")
        ax.set_ylabel("||action|| (KNN mean ± std)")
        ax.set_title(f"Dim {dim} (var={obs_var[dim]:.4f})")

        # roughness metric: mean absolute diff of consecutive means
        roughness = np.mean(np.abs(np.diff(act_means)))
        ax.text(0.02, 0.98, f"roughness={roughness:.4f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))

    fig.suptitle("Analysis 2: KNN Action Sweep per State Dimension", fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig2_knn_sweep.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Analysis 3: Trajectory Roughness ───────────────────────────────────────


def analysis3_trajectory_roughness(obs_list, act_list, output_dir, num_trajs=10):
    print("\n=== Analysis 3: Trajectory Roughness ===")
    num_trajs = min(num_trajs, len(obs_list))
    eps = 1e-8

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, num_trajs))

    all_lipschitz = []
    for i in range(num_trajs):
        obs = obs_list[i]
        act = act_list[i]
        ds = np.linalg.norm(np.diff(obs, axis=0), axis=1)
        da = np.linalg.norm(np.diff(act, axis=0), axis=1)
        lip = da / (ds + eps)
        all_lipschitz.append(lip)
        steps = np.arange(len(da))

        axes[0].plot(steps, da, color=colors[i], alpha=0.6, linewidth=0.8)
        axes[1].plot(steps, lip, color=colors[i], alpha=0.6, linewidth=0.8)

    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("||Δa||")
    axes[0].set_title("Action Change per Step")

    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("||Δa|| / ||Δs||")
    axes[1].set_title("Local Lipschitz Along Trajectory")

    all_lip = np.concatenate(all_lipschitz)
    clip_val = np.percentile(all_lip, 99)
    lip_clipped = all_lip[all_lip <= clip_val]
    axes[2].hist(lip_clipped, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    axes[2].axvline(np.median(all_lip), color="red", linestyle="--",
                     label=f"median={np.median(all_lip):.3f}")
    axes[2].set_xlabel("||Δa|| / ||Δs||")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Roughness Distribution (all trajs)")
    axes[2].legend()

    print(f"  Analyzed {num_trajs} trajectories")
    print(f"  Lipschitz: mean={all_lip.mean():.4f}, median={np.median(all_lip):.4f}, "
          f"p95={np.percentile(all_lip, 95):.4f}")

    fig.suptitle("Analysis 3: Trajectory Roughness", fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig3_trajectory_roughness.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Analyze spatial frequency of obs→action mapping")
    parser.add_argument("--demo_path", type=str, required=True, help="Path to .h5 demo file")
    parser.add_argument("--num_demos", type=int, default=None, help="Number of demos to load (default: all)")
    parser.add_argument("--output_dir", type=str, default="analysis_output", help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_obs, all_actions, obs_list, act_list = load_dataset(args.demo_path, args.num_demos)

    analysis1_distance_scatter(all_obs, all_actions, args.output_dir)
    analysis2_knn_sweep(all_obs, all_actions, args.output_dir)
    analysis3_trajectory_roughness(obs_list, act_list, args.output_dir)

    print("\nDone. All figures saved to:", args.output_dir)


if __name__ == "__main__":
    main()
