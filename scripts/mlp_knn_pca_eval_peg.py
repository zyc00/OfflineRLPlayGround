"""
MLP-projected k-NN policy in PCA action space for PegInsertionSide.

1. Fit PCA on training action chunks (16×8=128D → 20D)
2. Train MLP: obs_flat → action_pca (learns obs→action projection)
3. At eval: MLP hidden embedding → k-NN in embedding space → interpolate PCA actions
4. Also compare: MLP-only (no k-NN) and raw k-NN (no MLP)

"不要把自身include进去" = leave-one-out for training eval.
"""
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers import CPUGymWrapper
from collections import deque
import imageio

from DPPO.dataset import DPPODataset

# ---------- Config ----------
COND_STEPS = 2
HORIZON_STEPS = 16
ACT_STEPS = 8
ACTION_DIM = 8
MAX_EP_STEPS = 200
N_PCA = 20
K = 10
HIDDEN_DIM = 512
N_HIDDEN = 4
TRAIN_ITERS = 30000
BATCH_SIZE = 1024
LR = 1e-3
N_EVAL = 100
N_VIDEOS = 3
OUT_DIR = '/tmp/knn_videos_peg'


class ObsActionMLP(nn.Module):
    def __init__(self, obs_dim, out_dim, hidden_dim=512, n_hidden=4):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(n_hidden - 1):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.head(self.backbone(x))

    def embed(self, x):
        """Return penultimate hidden representation."""
        return self.backbone(x)


def load_training_data(dataset):
    """Extract all (obs_window_flat, action_chunk_flat) from dataset."""
    pad_before = COND_STEPS - 1
    all_obs, all_act = [], []
    for traj_idx, start in dataset.slices:
        obs_traj = dataset.obs_list[traj_idx]
        act_traj = dataset.act_list[traj_idx]
        L = act_traj.shape[0]

        obs_indices = [max(0, min(start + i, L)) for i in range(COND_STEPS)]
        obs_window = torch.stack([obs_traj[j] for j in obs_indices])

        act_start = start + pad_before
        acts = []
        for j in range(act_start, act_start + HORIZON_STEPS):
            if 0 <= j < L:
                acts.append(act_traj[j])
            elif j >= L:
                a = act_traj[-1].clone()
                a[:-1] = 0.0
                acts.append(a)
            else:
                acts.append(act_traj[0])

        all_obs.append(obs_window.numpy().flatten())
        all_act.append(torch.stack(acts).numpy().flatten())

    return np.array(all_obs, dtype=np.float32), np.array(all_act, dtype=np.float32)


def train_mlp(all_obs, all_act_pca, device, val_frac=0.1):
    """Train MLP: obs_flat → action_pca."""
    N = len(all_obs)
    n_val = int(N * val_frac)
    perm = np.random.RandomState(42).permutation(N)
    train_idx, val_idx = perm[n_val:], perm[:n_val]

    train_obs = torch.from_numpy(all_obs[train_idx]).to(device)
    train_act = torch.from_numpy(all_act_pca[train_idx]).to(device)
    val_obs = torch.from_numpy(all_obs[val_idx]).to(device)
    val_act = torch.from_numpy(all_act_pca[val_idx]).to(device)

    obs_dim = all_obs.shape[1]
    out_dim = all_act_pca.shape[1]
    model = ObsActionMLP(obs_dim, out_dim, HIDDEN_DIM, N_HIDDEN).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MLP: {obs_dim}→{HIDDEN_DIM}×{N_HIDDEN-1}→{out_dim}, {n_params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, TRAIN_ITERS)

    best_val = float('inf')
    best_state = None
    n_train = len(train_obs)

    for it in range(1, TRAIN_ITERS + 1):
        model.train()
        idx = torch.randint(n_train, (BATCH_SIZE,), device=device)
        pred = model(train_obs[idx])
        loss = ((pred - train_act[idx]) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if it % 2000 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(val_obs)
                val_loss = ((val_pred - val_act) ** 2).mean().item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  iter {it}/{TRAIN_ITERS}: train={loss.item():.6f}, val={val_loss:.6f}"
                  f" (best_val={best_val:.6f})")

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    print(f"MLP trained. Best val loss: {best_val:.6f}")
    return model


def eval_policy(mode, env, pca, mlp, nn_index, all_act_pca, device,
                obs_mean_t, obs_std_t, n_eval, n_videos, out_dir):
    """Evaluate a policy mode. Returns (sr, video_paths).

    Modes:
      'mlp': MLP prediction → PCA inverse → execute
      'mlp_knn': MLP embed → k-NN in embed space → interpolate PCA → execute
    """
    results = []
    video_paths = []
    video_count = 0

    for ep in range(n_eval):
        obs, _ = env.reset()
        obs_buffer = deque(maxlen=COND_STEPS)
        for _ in range(COND_STEPS):
            obs_buffer.append(obs.copy())

        frames = []
        recording = video_count < n_videos
        if recording:
            frames.append(env.render())

        success = False
        step = 0

        while step < MAX_EP_STEPS:
            obs_window = np.stack(list(obs_buffer), axis=0).flatten()  # (86,)
            obs_t = torch.from_numpy(obs_window).float().unsqueeze(0).to(device)

            with torch.no_grad():
                if mode == 'mlp':
                    pred_pca = mlp(obs_t).cpu().numpy()[0]
                    act_pca = pred_pca
                elif mode == 'mlp_knn':
                    # Get hidden embedding
                    embed = mlp.embed(obs_t).cpu().numpy()
                    # Normalize with stored stats
                    embed_norm = (embed - obs_mean_t) / obs_std_t
                    dists, idxs = nn_index.kneighbors(embed_norm)
                    dists, idxs = dists[0], idxs[0]
                    if dists.min() < 1e-10:
                        w = np.zeros(K)
                        w[dists.argmin()] = 1.0
                    else:
                        inv_d = 1.0 / dists
                        w = inv_d / inv_d.sum()
                    act_pca = (w[:, None] * all_act_pca[idxs]).sum(axis=0)

            act_flat = pca.inverse_transform(act_pca.reshape(1, -1))[0]
            act_chunk = act_flat.reshape(HORIZON_STEPS, ACTION_DIM)

            n_exec = min(ACT_STEPS, MAX_EP_STEPS - step)
            for t in range(n_exec):
                obs, rew, terminated, truncated, info = env.step(act_chunk[t])
                obs_buffer.append(obs.copy())
                step += 1
                if recording:
                    frames.append(env.render())
                if info.get("success", False):
                    success = True

        results.append(success)

        if recording and len(frames) > 1:
            tag = 'ok' if success else 'fail'
            vpath = os.path.join(out_dir, f'{mode}_ep{ep}_{tag}.mp4')
            imageio.mimwrite(vpath, frames, fps=20)
            video_paths.append(vpath)
            video_count += 1

        if (ep + 1) % 20 == 0:
            print(f"  [{mode}] ep {ep+1}/{n_eval}: SR={np.mean(results):.1%}")

    sr = np.mean(results)
    return sr, video_paths


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load data ---
    dataset = DPPODataset(
        data_path=os.path.expanduser(
            '~/.maniskill/demos/PegInsertionSide-v1/policy_from_mp_states/'
            'trajectory.state.pd_joint_delta_pos.h5'),
        horizon_steps=HORIZON_STEPS, cond_steps=COND_STEPS,
        no_obs_norm=True, no_action_norm=True,
    )
    print("Extracting training pairs...")
    all_obs, all_act = load_training_data(dataset)
    N = len(all_obs)
    print(f"  {N} samples, obs={all_obs.shape[1]}D, act={all_act.shape[1]}D")

    # --- PCA on actions ---
    pca = PCA(n_components=N_PCA)
    all_act_pca = pca.fit_transform(all_act)
    print(f"PCA: {N_PCA} components → {pca.explained_variance_ratio_.sum()*100:.1f}% variance")

    # --- Train MLP ---
    print("\nTraining MLP (obs → action_pca)...")
    mlp = train_mlp(all_obs, all_act_pca, device)

    # --- Build k-NN on MLP hidden embeddings ---
    print("\nBuilding k-NN index on MLP hidden embeddings...")
    all_obs_t = torch.from_numpy(all_obs).float().to(device)
    with torch.no_grad():
        # Process in batches to avoid OOM
        embeds = []
        for i in range(0, N, 4096):
            embeds.append(mlp.embed(all_obs_t[i:i+4096]).cpu().numpy())
        all_embeds = np.concatenate(embeds, axis=0)  # (N, hidden_dim)

    # Normalize embeddings for k-NN
    embed_mean = all_embeds.mean(axis=0)
    embed_std = all_embeds.std(axis=0)
    embed_std[embed_std < 1e-8] = 1.0
    all_embeds_norm = (all_embeds - embed_mean) / embed_std

    nn_index = NearestNeighbors(n_neighbors=K, algorithm='ball_tree')
    nn_index.fit(all_embeds_norm)
    print(f"  k-NN built: {N} points, {all_embeds.shape[1]}D embeddings, k={K}")

    # --- Eval ---
    env = gym.make('PegInsertionSide-v1', obs_mode="state",
                    control_mode="pd_joint_delta_pos", render_mode="rgb_array",
                    max_episode_steps=MAX_EP_STEPS, reconfiguration_freq=1)
    env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)

    for mode in ['mlp', 'mlp_knn']:
        print(f"\n=== Evaluating: {mode} ===")
        sr, vpaths = eval_policy(
            mode, env, pca, mlp, nn_index, all_act_pca, device,
            embed_mean, embed_std, N_EVAL, N_VIDEOS, OUT_DIR)
        print(f"  {mode}: SR = {sr:.1%} ({int(sr*N_EVAL)}/{N_EVAL})")
        for vp in vpaths:
            print(f"  Video: {vp}")

    env.close()


if __name__ == "__main__":
    main()
