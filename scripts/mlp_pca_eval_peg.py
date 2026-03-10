"""
MLP: zscore+PCA(obs) → PCA(action) for PegInsertionSide.

1. Z-score normalize obs, then PCA (so small-std task-critical dims aren't lost)
2. PCA on actions (128D → 20D)
3. Train MLP: obs_pca → action_pca
4. Eval with 3 videos
"""
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers import CPUGymWrapper
from collections import deque
import imageio

from DPPO.dataset import DPPODataset

COND_STEPS = 2
HORIZON_STEPS = 16
ACT_STEPS = 8
ACTION_DIM = 8
MAX_EP_STEPS = 200
N_ACT_PCA = 20


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=1024, n_layers=8):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_data(dataset):
    pad_before = COND_STEPS - 1
    all_obs, all_act = [], []
    for traj_idx, start in dataset.slices:
        obs_traj = dataset.obs_list[traj_idx]
        act_traj = dataset.act_list[traj_idx]
        L = act_traj.shape[0]
        obs_idx = [max(0, min(start + i, L)) for i in range(COND_STEPS)]
        obs_w = torch.stack([obs_traj[j] for j in obs_idx])
        acts = []
        for j in range(start + pad_before, start + pad_before + HORIZON_STEPS):
            if 0 <= j < L:
                acts.append(act_traj[j])
            elif j >= L:
                a = act_traj[-1].clone(); a[:-1] = 0.0; acts.append(a)
            else:
                acts.append(act_traj[0])
        all_obs.append(obs_w.numpy().flatten())
        all_act.append(torch.stack(acts).numpy().flatten())
    return np.array(all_obs, np.float32), np.array(all_act, np.float32)


def main():
    device = torch.device('cuda')
    out_dir = '/tmp/knn_videos_peg'
    os.makedirs(out_dir, exist_ok=True)

    dataset = DPPODataset(
        data_path=os.path.expanduser(
            '~/.maniskill/demos/PegInsertionSide-v1/policy_from_mp_states/'
            'trajectory.state.pd_joint_delta_pos.h5'),
        horizon_steps=HORIZON_STEPS, cond_steps=COND_STEPS,
        no_obs_norm=True, no_action_norm=True)

    print("Loading data...")
    all_obs, all_act = load_data(dataset)
    N = len(all_obs)
    print(f"{N} samples, obs={all_obs.shape[1]}D, act={all_act.shape[1]}D")

    # PCA on actions
    act_pca = PCA(n_components=N_ACT_PCA)
    all_act_pc = act_pca.fit_transform(all_act)
    print(f"Action PCA: {N_ACT_PCA} comp → {act_pca.explained_variance_ratio_.sum()*100:.1f}%")

    # Z-score normalize obs, remove constant dims, then PCA
    obs_stds = all_obs.std(axis=0)
    nonconst_mask = obs_stds > 1e-8
    print(f"Obs: {nonconst_mask.sum()} non-constant dims (removed {(~nonconst_mask).sum()} constant)")

    scaler = StandardScaler()
    obs_nonconst = all_obs[:, nonconst_mask]
    obs_scaled = scaler.fit_transform(obs_nonconst)

    obs_pca = PCA(n_components=0.995)
    all_obs_pc = obs_pca.fit_transform(obs_scaled)
    n_obs_pca = all_obs_pc.shape[1]
    print(f"Obs pipeline: {all_obs.shape[1]}D → remove const → zscore → PCA → {n_obs_pca}D "
          f"({obs_pca.explained_variance_ratio_.sum()*100:.2f}%)")

    # Train/val split
    rng = np.random.RandomState(42)
    perm = rng.permutation(N)
    n_val = int(N * 0.1)
    tr_idx, va_idx = perm[n_val:], perm[:n_val]

    tr_obs = torch.from_numpy(all_obs_pc[tr_idx]).to(device)
    tr_act = torch.from_numpy(all_act_pc[tr_idx]).to(device)
    va_obs = torch.from_numpy(all_obs_pc[va_idx]).to(device)
    va_act = torch.from_numpy(all_act_pc[va_idx]).to(device)

    # MLP: obs_pca → action_pca
    model = MLP(n_obs_pca, N_ACT_PCA, hidden=1024, n_layers=8).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MLP: {n_obs_pca}→1024×7→{N_ACT_PCA}, {n_params:,} params")

    total_iters = 100000
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_iters)

    best_val, best_state = float('inf'), None
    n_tr = len(tr_obs)
    for it in range(1, total_iters + 1):
        model.train()
        idx = torch.randint(n_tr, (1024,), device=device)
        loss = ((model(tr_obs[idx]) - tr_act[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        if it % 10000 == 0:
            model.eval()
            with torch.no_grad():
                vl = ((model(va_obs) - va_act) ** 2).mean().item()
            if vl < best_val:
                best_val = vl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  iter {it}: train={loss.item():.6f}, val={vl:.6f} (best={best_val:.6f})")

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    # Eval
    env = gym.make('PegInsertionSide-v1', obs_mode="state",
                    control_mode="pd_joint_delta_pos", render_mode="rgb_array",
                    max_episode_steps=MAX_EP_STEPS, reconfiguration_freq=1)
    env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)

    def preprocess_obs(obs_flat):
        """Apply same pipeline: select non-const dims → z-score → PCA."""
        obs_nc = obs_flat[:, nonconst_mask] if obs_flat.ndim == 2 else obs_flat[nonconst_mask].reshape(1, -1)
        obs_sc = scaler.transform(obs_nc)
        return obs_pca.transform(obs_sc)

    results, vcount = [], 0
    for ep in range(100):
        obs, _ = env.reset()
        buf = deque(maxlen=COND_STEPS)
        for _ in range(COND_STEPS):
            buf.append(obs.copy())

        frames = []
        rec = vcount < 3
        if rec:
            frames.append(env.render())
        success, step = False, 0

        while step < MAX_EP_STEPS:
            obs_flat = np.stack(list(buf), axis=0).flatten()
            obs_pc = preprocess_obs(obs_flat)
            with torch.no_grad():
                pred_act_pc = model(
                    torch.from_numpy(obs_pc).float().to(device)
                ).cpu().numpy()[0]
            act_chunk = act_pca.inverse_transform(
                pred_act_pc.reshape(1, -1))[0].reshape(HORIZON_STEPS, ACTION_DIM)

            for t in range(min(ACT_STEPS, MAX_EP_STEPS - step)):
                obs, _, _, _, info = env.step(act_chunk[t])
                buf.append(obs.copy())
                step += 1
                if rec:
                    frames.append(env.render())
                if info.get("success", False):
                    success = True

        results.append(success)
        if rec and frames:
            tag = 'ok' if success else 'fail'
            vp = os.path.join(out_dir, f'mlp_zscore_pca_ep{ep}_{tag}.mp4')
            imageio.mimwrite(vp, frames, fps=20)
            print(f"  Video: {vp}")
            vcount += 1
        if (ep + 1) % 20 == 0:
            print(f"ep {ep+1}: SR={np.mean(results):.1%}")

    env.close()
    print(f"\nzscore+PCA(obs)→MLP→PCA(act) policy: SR = {np.mean(results):.1%}")


if __name__ == "__main__":
    main()
