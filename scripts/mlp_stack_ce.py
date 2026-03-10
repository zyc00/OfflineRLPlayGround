"""StackCube: C) raw→zscore+PCA(act) and E) zscore+PCA(obs)→zscore+PCA(act)."""
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
from DPPO.dataset import DPPODataset

COND_STEPS = 2; HORIZON_STEPS = 16; ACT_STEPS = 8; MAX_EP_STEPS = 300; N_ACT_PCA = 20

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

def train_and_eval(label, tr_obs, tr_act, va_obs, va_act, in_dim, out_dim, device,
                   obs_pre_fn, act_post_fn, env, action_dim):
    model = MLP(in_dim, out_dim, hidden=1024, n_layers=8).to(device)
    print(f"  [{label}] MLP: {in_dim}→1024×7→{out_dim}, {sum(p.numel() for p in model.parameters()):,} params")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 100000)
    best_val, best_state = float('inf'), None
    n_tr = len(tr_obs)
    for it in range(1, 100001):
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

    results = []
    for ep in range(100):
        obs, _ = env.reset()
        buf = deque(maxlen=COND_STEPS)
        for _ in range(COND_STEPS):
            buf.append(obs.copy())
        success, step = False, 0
        while step < MAX_EP_STEPS:
            obs_flat = np.stack(list(buf), axis=0).flatten()
            obs_in = obs_pre_fn(obs_flat)
            with torch.no_grad():
                pred = model(obs_in).cpu().numpy()[0]
            act_chunk = act_post_fn(pred).reshape(HORIZON_STEPS, action_dim)
            for t in range(min(ACT_STEPS, MAX_EP_STEPS - step)):
                obs, _, _, _, info = env.step(act_chunk[t])
                buf.append(obs.copy())
                step += 1
                if info.get("success", False):
                    success = True
        results.append(success)
        if (ep + 1) % 20 == 0:
            print(f"  [{label}] ep {ep+1}: SR={np.mean(results):.1%}")
    sr = np.mean(results)
    print(f"  [{label}] FINAL: SR={sr:.1%}")
    return sr

def main():
    device = torch.device('cuda')
    dataset = DPPODataset(
        data_path=os.path.expanduser(
            '~/.maniskill/demos/StackCube-v1/motionplanning/'
            'trajectory.state.pd_joint_delta_pos.physx_cpu.h5'),
        horizon_steps=HORIZON_STEPS, cond_steps=COND_STEPS,
        no_obs_norm=True, no_action_norm=True)
    all_obs, all_act = load_data(dataset)
    action_dim = dataset.action_dim
    N = len(all_obs)
    print(f"{N} samples, obs={all_obs.shape[1]}D, act={all_act.shape[1]}D")

    # Act zscore+PCA
    act_scaler = StandardScaler()
    all_act_scaled = act_scaler.fit_transform(all_act)
    act_pca = PCA(n_components=N_ACT_PCA)
    all_act_pc = act_pca.fit_transform(all_act_scaled)
    print(f"Act zscore+PCA: {all_act.shape[1]}D → {N_ACT_PCA}D ({act_pca.explained_variance_ratio_.sum()*100:.1f}%)")

    # Obs zscore+PCA
    obs_stds = all_obs.std(axis=0)
    nonconst_mask = obs_stds > 1e-8
    obs_scaler = StandardScaler()
    obs_scaled = obs_scaler.fit_transform(all_obs[:, nonconst_mask])
    obs_pca = PCA(n_components=0.995)
    all_obs_pc = obs_pca.fit_transform(obs_scaled)
    n_obs_pca = all_obs_pc.shape[1]
    print(f"Obs zscore+PCA: {all_obs.shape[1]}D → {n_obs_pca}D ({obs_pca.explained_variance_ratio_.sum()*100:.2f}%)")

    # Split
    rng = np.random.RandomState(42)
    perm = rng.permutation(N)
    n_val = int(N * 0.1)
    tr_idx, va_idx = perm[n_val:], perm[:n_val]

    env = gym.make('StackCube-v1', obs_mode="state", control_mode="pd_joint_delta_pos",
                    render_mode="rgb_array", max_episode_steps=MAX_EP_STEPS,
                    reconfiguration_freq=1, sim_backend="physx_cpu")
    env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)

    def act_post(pred):
        return act_scaler.inverse_transform(act_pca.inverse_transform(pred.reshape(1, -1)))[0]

    # C) raw obs → zscore+PCA(act)
    print("\n=== C) raw obs → zscore+PCA(act) ===")
    sr_c = train_and_eval("C",
        torch.from_numpy(all_obs[tr_idx]).to(device),
        torch.from_numpy(all_act_pc[tr_idx]).to(device),
        torch.from_numpy(all_obs[va_idx]).to(device),
        torch.from_numpy(all_act_pc[va_idx]).to(device),
        all_obs.shape[1], N_ACT_PCA, device,
        lambda o: torch.from_numpy(o).float().unsqueeze(0).to(device),
        act_post, env, action_dim)

    # E) zscore+PCA(obs) → zscore+PCA(act)
    print("\n=== E) zscore+PCA(obs) → zscore+PCA(act) ===")
    def obs_pre_e(obs_flat):
        obs_nc = obs_flat[nonconst_mask].reshape(1, -1)
        obs_pc = obs_pca.transform(obs_scaler.transform(obs_nc))
        return torch.from_numpy(obs_pc).float().to(device)
    sr_e = train_and_eval("E",
        torch.from_numpy(all_obs_pc[tr_idx]).to(device),
        torch.from_numpy(all_act_pc[tr_idx]).to(device),
        torch.from_numpy(all_obs_pc[va_idx]).to(device),
        torch.from_numpy(all_act_pc[va_idx]).to(device),
        n_obs_pca, N_ACT_PCA, device,
        obs_pre_e, act_post, env, action_dim)

    env.close()
    print(f"\nStackCube: C={sr_c:.1%}, E={sr_e:.1%}")

if __name__ == "__main__":
    main()
