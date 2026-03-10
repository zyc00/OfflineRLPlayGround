"""
k-NN policy in PCA action space for PegInsertionSide.

1. PCA decompose all training action chunks (16×8=128D → 20D)
2. At eval, find k nearest neighbors in obs space, interpolate PCA coords
3. Reconstruct action chunk, execute first 8 steps
4. Save rollout videos
"""
import os
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers import CPUGymWrapper
from collections import deque
import imageio

from DPPO.dataset import DPPODataset


def build_knn_index(dataset, cond_steps, horizon_steps, n_pca=20, k=10):
    """Build PCA + k-NN index from training data."""
    # Extract all (obs_window, action_chunk) pairs efficiently from raw trajectories
    pad_before = cond_steps - 1

    all_obs = []
    all_act = []
    for traj_idx, start in dataset.slices:
        obs_traj = dataset.obs_list[traj_idx]
        act_traj = dataset.act_list[traj_idx]
        L = act_traj.shape[0]

        # Obs window: [start, start+cond_steps), clamped
        obs_indices = [max(0, min(start + i, L)) for i in range(cond_steps)]
        obs_window = torch.stack([obs_traj[j] for j in obs_indices])  # (cond, 43)

        # Action chunk: [start+pad_before, start+pad_before+horizon], padded
        act_start = start + pad_before
        act_indices = range(act_start, act_start + horizon_steps)
        acts = []
        for j in act_indices:
            if 0 <= j < L:
                acts.append(act_traj[j])
            elif j >= L:
                a = act_traj[-1].clone()
                a[:-1] = 0.0  # zero arm, keep gripper
                acts.append(a)
            else:
                acts.append(act_traj[0])
        act_chunk = torch.stack(acts)  # (horizon, 8)

        all_obs.append(obs_window.numpy().flatten())  # (cond*43,)
        all_act.append(act_chunk.numpy().flatten())  # (horizon*8,)

    all_obs = np.array(all_obs, dtype=np.float32)
    all_act = np.array(all_act, dtype=np.float32)
    print(f"Training data: {all_obs.shape[0]} samples, obs={all_obs.shape[1]}D, act={all_act.shape[1]}D")

    # PCA on actions
    pca = PCA(n_components=n_pca)
    all_act_pca = pca.fit_transform(all_act)
    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"PCA: {n_pca} components explain {explained:.1f}% variance")

    # Normalize obs for k-NN (z-score)
    obs_mean = all_obs.mean(axis=0)
    obs_std = all_obs.std(axis=0)
    obs_std[obs_std < 1e-8] = 1.0
    all_obs_norm = (all_obs - obs_mean) / obs_std

    # Build k-NN
    nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric='euclidean')
    nn.fit(all_obs_norm)
    print(f"k-NN index built (k={k}, {all_obs_norm.shape[0]} points)")

    return pca, nn, all_act_pca, obs_mean, obs_std


def knn_pca_predict(obs_window, pca, nn, all_act_pca, obs_mean, obs_std,
                    horizon_steps, action_dim):
    """Predict action chunk via k-NN interpolation in PCA space."""
    obs_flat = obs_window.flatten().reshape(1, -1)
    obs_norm = (obs_flat - obs_mean) / obs_std

    distances, indices = nn.kneighbors(obs_norm)
    distances = distances[0]
    indices = indices[0]

    # Inverse distance weighting
    if distances.min() < 1e-10:
        weights = np.zeros(len(distances))
        weights[distances.argmin()] = 1.0
    else:
        inv_dist = 1.0 / distances
        weights = inv_dist / inv_dist.sum()

    # Interpolate in PCA space
    neighbor_pca = all_act_pca[indices]  # (k, n_pca)
    interp_pca = (weights[:, None] * neighbor_pca).sum(axis=0)  # (n_pca,)

    # Reconstruct
    act_flat = pca.inverse_transform(interp_pca.reshape(1, -1))[0]
    return act_flat.reshape(horizon_steps, action_dim)


def main():
    cond_steps = 2
    horizon_steps = 16
    act_steps = 8
    action_dim = 8
    max_ep_steps = 200
    n_pca = 20
    k = 10
    n_eval = 100
    n_videos = 3
    out_dir = '/tmp/knn_videos_peg'
    os.makedirs(out_dir, exist_ok=True)

    # Load dataset
    dataset = DPPODataset(
        data_path=os.path.expanduser(
            '~/.maniskill/demos/PegInsertionSide-v1/policy_from_mp_states/'
            'trajectory.state.pd_joint_delta_pos.h5'),
        horizon_steps=horizon_steps,
        cond_steps=cond_steps,
        no_obs_norm=True,
        no_action_norm=True,
    )

    # Build index
    pca, nn, all_act_pca, obs_mean, obs_std = build_knn_index(
        dataset, cond_steps, horizon_steps, n_pca=n_pca, k=k)

    # Eval
    env = gym.make('PegInsertionSide-v1', obs_mode="state",
                    control_mode="pd_joint_delta_pos",
                    render_mode="rgb_array", max_episode_steps=max_ep_steps,
                    reconfiguration_freq=1)
    env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)

    results = []
    video_count = 0

    for ep in range(n_eval):
        obs, _ = env.reset()
        obs_buffer = deque(maxlen=cond_steps)
        for _ in range(cond_steps):
            obs_buffer.append(obs.copy())

        frames = []
        recording = video_count < n_videos
        if recording:
            frames.append(env.render())

        success = False
        step = 0

        while step < max_ep_steps:
            # Build obs window
            obs_window = np.stack(list(obs_buffer), axis=0)  # (2, 43)

            # k-NN PCA predict
            act_chunk = knn_pca_predict(
                obs_window, pca, nn, all_act_pca, obs_mean, obs_std,
                horizon_steps, action_dim)

            # Execute act_steps
            n_exec = min(act_steps, max_ep_steps - step)
            for t in range(n_exec):
                action = act_chunk[t]
                obs, rew, terminated, truncated, info = env.step(action)
                obs_buffer.append(obs.copy())
                step += 1

                if recording:
                    frames.append(env.render())

                if info.get("success", False):
                    success = True

        results.append(success)

        if recording and len(frames) > 1:
            video_path = os.path.join(out_dir, f'knn_pca_ep{ep}.mp4')
            imageio.mimwrite(video_path, frames, fps=20)
            print(f"  Saved video: {video_path} ({'SUCCESS' if success else 'FAIL'})")
            video_count += 1

        if (ep + 1) % 10 == 0:
            sr_so_far = np.mean(results)
            print(f"Episode {ep+1}/{n_eval}: SR={sr_so_far:.1%} "
                  f"(this ep: {'OK' if success else 'FAIL'})")

    env.close()
    sr = np.mean(results)
    print(f"\n=== k-NN PCA Policy (k={k}, PCA={n_pca}) ===")
    print(f"Episodes: {n_eval}, Success Rate: {sr:.1%}")
    print(f"Videos saved to {out_dir}")


if __name__ == "__main__":
    main()
