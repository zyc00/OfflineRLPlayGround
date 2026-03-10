"""
PickCube MLP ablations: 4 configs comparing PCA/norm on obs and act sides.

A) raw obs → raw act (no PCA at all)
B) raw obs → PCA(act)
C) raw obs → norm_PCA(act)  (z-score actions before PCA)
D) zscore+PCA(obs) → PCA(act)  [already done = 76%, included for completeness]
"""
import os, sys
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
MAX_EP_STEPS = 100
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


def train_mlp(tr_obs, tr_act, va_obs, va_act, in_dim, out_dim, device, label):
    model = MLP(in_dim, out_dim, hidden=1024, n_layers=8).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [{label}] MLP: {in_dim}→1024×7→{out_dim}, {n_params:,} params")

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
            print(f"  [{label}] iter {it}: train={loss.item():.6f}, val={vl:.6f} (best={best_val:.6f})")

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    return model


def eval_policy(model, env, obs_preprocess, act_postprocess, label, action_dim,
                out_dir, n_eval=100, n_videos=3):
    results, vcount = [], 0
    for ep in range(n_eval):
        obs, _ = env.reset()
        buf = deque(maxlen=COND_STEPS)
        for _ in range(COND_STEPS):
            buf.append(obs.copy())

        frames = []
        rec = vcount < n_videos
        if rec:
            frames.append(env.render())
        success, step = False, 0

        while step < MAX_EP_STEPS:
            obs_flat = np.stack(list(buf), axis=0).flatten()
            obs_in = obs_preprocess(obs_flat)
            with torch.no_grad():
                pred = model(obs_in).cpu().numpy()[0]
            act_chunk = act_postprocess(pred).reshape(HORIZON_STEPS, action_dim)

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
            vp = os.path.join(out_dir, f'{label}_ep{ep}_{tag}.mp4')
            imageio.mimwrite(vp, frames, fps=20)
            vcount += 1
        if (ep + 1) % 20 == 0:
            print(f"  [{label}] ep {ep+1}: SR={np.mean(results):.1%}")

    sr = np.mean(results)
    print(f"  [{label}] FINAL: SR={sr:.1%}")
    return sr


def main():
    device = torch.device('cuda')
    out_dir = '/tmp/knn_videos_pickcube'
    os.makedirs(out_dir, exist_ok=True)

    dataset = DPPODataset(
        data_path=os.path.expanduser(
            '~/.maniskill/demos/PickCube-v1/motionplanning/'
            'trajectory.state.pd_ee_delta_pos.physx_cpu.h5'),
        horizon_steps=HORIZON_STEPS, cond_steps=COND_STEPS,
        no_obs_norm=True, no_action_norm=True)

    print("Loading data...")
    all_obs, all_act = load_data(dataset)
    action_dim = dataset.action_dim
    N = len(all_obs)
    print(f"{N} samples, obs={all_obs.shape[1]}D, act={all_act.shape[1]}D, action_dim={action_dim}")

    # --- Precompute all representations ---

    # Action PCA
    act_pca = PCA(n_components=N_ACT_PCA)
    all_act_pc = act_pca.fit_transform(all_act)
    print(f"Action PCA: {N_ACT_PCA} comp → {act_pca.explained_variance_ratio_.sum()*100:.1f}%")

    # Action norm+PCA (z-score actions first)
    act_scaler = StandardScaler()
    all_act_scaled = act_scaler.fit_transform(all_act)
    act_norm_pca = PCA(n_components=N_ACT_PCA)
    all_act_norm_pc = act_norm_pca.fit_transform(all_act_scaled)
    print(f"Action norm+PCA: {N_ACT_PCA} comp → {act_norm_pca.explained_variance_ratio_.sum()*100:.1f}%")

    # Obs zscore+PCA
    obs_stds = all_obs.std(axis=0)
    nonconst_mask = obs_stds > 1e-8
    obs_scaler = StandardScaler()
    obs_nonconst = all_obs[:, nonconst_mask]
    obs_scaled = obs_scaler.fit_transform(obs_nonconst)
    obs_pca = PCA(n_components=0.995)
    all_obs_pc = obs_pca.fit_transform(obs_scaled)
    n_obs_pca = all_obs_pc.shape[1]
    print(f"Obs zscore+PCA: {all_obs.shape[1]}D → {n_obs_pca}D ({obs_pca.explained_variance_ratio_.sum()*100:.2f}%)")

    # --- Train/val split ---
    rng = np.random.RandomState(42)
    perm = rng.permutation(N)
    n_val = int(N * 0.1)
    tr_idx, va_idx = perm[n_val:], perm[:n_val]

    # --- Env ---
    env = gym.make('PickCube-v1', obs_mode="state",
                    control_mode="pd_ee_delta_pos", render_mode="rgb_array",
                    max_episode_steps=MAX_EP_STEPS, reconfiguration_freq=1)
    env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)

    results_table = {}

    # ===== A) raw obs → raw act =====
    label = "A_raw_raw"
    print(f"\n=== {label}: raw obs ({all_obs.shape[1]}D) → raw act ({all_act.shape[1]}D) ===")
    tr_o = torch.from_numpy(all_obs[tr_idx]).to(device)
    tr_a = torch.from_numpy(all_act[tr_idx]).to(device)
    va_o = torch.from_numpy(all_obs[va_idx]).to(device)
    va_a = torch.from_numpy(all_act[va_idx]).to(device)
    model_a = train_mlp(tr_o, tr_a, va_o, va_a, all_obs.shape[1], all_act.shape[1], device, label)

    def obs_pre_a(obs_flat):
        return torch.from_numpy(obs_flat).float().unsqueeze(0).to(device)
    def act_post_a(pred):
        return pred

    sr_a = eval_policy(model_a, env, obs_pre_a, act_post_a, label, action_dim, out_dir)
    results_table[label] = sr_a
    del model_a, tr_o, tr_a, va_o, va_a
    torch.cuda.empty_cache()

    # ===== B) raw obs → PCA(act) =====
    label = "B_raw_pca"
    print(f"\n=== {label}: raw obs ({all_obs.shape[1]}D) → PCA act ({N_ACT_PCA}D) ===")
    tr_o = torch.from_numpy(all_obs[tr_idx]).to(device)
    tr_a = torch.from_numpy(all_act_pc[tr_idx]).to(device)
    va_o = torch.from_numpy(all_obs[va_idx]).to(device)
    va_a = torch.from_numpy(all_act_pc[va_idx]).to(device)
    model_b = train_mlp(tr_o, tr_a, va_o, va_a, all_obs.shape[1], N_ACT_PCA, device, label)

    def obs_pre_b(obs_flat):
        return torch.from_numpy(obs_flat).float().unsqueeze(0).to(device)
    def act_post_b(pred):
        return act_pca.inverse_transform(pred.reshape(1, -1))[0]

    sr_b = eval_policy(model_b, env, obs_pre_b, act_post_b, label, action_dim, out_dir)
    results_table[label] = sr_b
    del model_b, tr_o, tr_a, va_o, va_a
    torch.cuda.empty_cache()

    # ===== C) raw obs → norm_PCA(act) =====
    label = "C_raw_normpca"
    print(f"\n=== {label}: raw obs ({all_obs.shape[1]}D) → zscore+PCA act ({N_ACT_PCA}D) ===")
    tr_o = torch.from_numpy(all_obs[tr_idx]).to(device)
    tr_a = torch.from_numpy(all_act_norm_pc[tr_idx]).to(device)
    va_o = torch.from_numpy(all_obs[va_idx]).to(device)
    va_a = torch.from_numpy(all_act_norm_pc[va_idx]).to(device)
    model_c = train_mlp(tr_o, tr_a, va_o, va_a, all_obs.shape[1], N_ACT_PCA, device, label)

    def obs_pre_c(obs_flat):
        return torch.from_numpy(obs_flat).float().unsqueeze(0).to(device)
    def act_post_c(pred):
        # inverse PCA then inverse z-score
        act_scaled = act_norm_pca.inverse_transform(pred.reshape(1, -1))[0]
        return act_scaler.inverse_transform(act_scaled.reshape(1, -1))[0]

    sr_c = eval_policy(model_c, env, obs_pre_c, act_post_c, label, action_dim, out_dir)
    results_table[label] = sr_c
    del model_c, tr_o, tr_a, va_o, va_a
    torch.cuda.empty_cache()

    # ===== D) zscore+PCA(obs) → PCA(act) =====
    label = "D_normpca_pca"
    print(f"\n=== {label}: zscore+PCA obs ({n_obs_pca}D) → PCA act ({N_ACT_PCA}D) ===")
    tr_o = torch.from_numpy(all_obs_pc[tr_idx]).to(device)
    tr_a = torch.from_numpy(all_act_pc[tr_idx]).to(device)
    va_o = torch.from_numpy(all_obs_pc[va_idx]).to(device)
    va_a = torch.from_numpy(all_act_pc[va_idx]).to(device)
    model_d = train_mlp(tr_o, tr_a, va_o, va_a, n_obs_pca, N_ACT_PCA, device, label)

    def obs_pre_d(obs_flat):
        obs_nc = obs_flat[nonconst_mask].reshape(1, -1)
        obs_sc = obs_scaler.transform(obs_nc)
        obs_pc = obs_pca.transform(obs_sc)
        return torch.from_numpy(obs_pc).float().to(device)
    def act_post_d(pred):
        return act_pca.inverse_transform(pred.reshape(1, -1))[0]

    sr_d = eval_policy(model_d, env, obs_pre_d, act_post_d, label, action_dim, out_dir)
    results_table[label] = sr_d

    env.close()

    # Summary
    print("\n" + "="*60)
    print("PickCube MLP Ablation Results")
    print("="*60)
    print(f"  A) raw obs → raw act:              {results_table['A_raw_raw']:.1%}")
    print(f"  B) raw obs → PCA(act):             {results_table['B_raw_pca']:.1%}")
    print(f"  C) raw obs → zscore+PCA(act):      {results_table['C_raw_normpca']:.1%}")
    print(f"  D) zscore+PCA(obs) → PCA(act):     {results_table['D_normpca_pca']:.1%}")
    print("="*60)


if __name__ == "__main__":
    main()
