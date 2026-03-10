"""
MIP with PCA-preprocessed obs on PegInsertionSide.

Matches best MIP config (official_v3): dropout=0.1, obs/act normalization, zero_qvel,
cond=1, horizon=1, act=1 — but replaces raw obs with PCA(obs).
"""
import os, copy, time, random
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from sklearn.decomposition import PCA
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers import CPUGymWrapper
from diffusers.training_utils import EMAModel as DiffusersEMA

from DPPO.dataset import DPPODataset
from MultiGaussian.models.multi_gaussian import MIPPolicy


def main():
    device = torch.device('cuda')
    seed = 0
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # --- Config (match official_v3 best) ---
    env_id = 'PegInsertionSide-v1'
    control_mode = 'pd_joint_delta_pos'
    max_ep_steps = 200
    cond_steps, horizon_steps, act_steps = 1, 1, 1
    emb_dim, n_layers, dropout = 512, 6, 0.1
    t_star = 0.9
    total_iters = 100_000
    batch_size = 1024
    lr = 1e-4
    eval_freq = 10000
    log_freq = 200

    # --- Load data ---
    dataset = DPPODataset(
        data_path=os.path.expanduser(
            '~/.maniskill/demos/PegInsertionSide-v1/policy_from_mp_states/'
            'trajectory.state.pd_joint_delta_pos.h5'),
        horizon_steps=horizon_steps, cond_steps=cond_steps,
        no_obs_norm=False, no_action_norm=False)  # keep normalization like official_v3

    # Pre-load
    val_frac = 0.1
    n_val_traj = max(1, int(dataset.num_traj * val_frac))
    n_train_traj = dataset.num_traj - n_val_traj
    train_indices = [i for i, (tj, _) in enumerate(dataset.slices) if tj < n_train_traj]
    val_indices = [i for i, (tj, _) in enumerate(dataset.slices) if tj >= n_train_traj]

    def _preload(indices):
        obs_l, act_l = [], []
        for idx in indices:
            s = dataset[idx]
            obs_l.append(s["cond"]["state"])
            act_l.append(s["actions"])
        obs_t = torch.stack(obs_l)
        act_t = torch.stack(act_l)
        if cond_steps == 1: obs_t = obs_t.squeeze(1)
        if horizon_steps == 1: act_t = act_t.squeeze(1)
        obs_t[..., 9:18] = 0.0  # zero_qvel
        return obs_t, act_t

    print("Pre-loading...")
    train_obs, train_act = _preload(train_indices)
    val_obs, val_act = _preload(val_indices)
    print(f"Train: {train_obs.shape}, Val: {val_obs.shape}")

    # --- PCA on obs ---
    obs_np = train_obs.numpy()
    obs_pca = PCA(n_components=0.995)
    obs_pca.fit(obs_np)
    n_obs_pc = obs_pca.n_components_
    print(f"Obs PCA: {train_obs.shape[1]}D → {n_obs_pc}D "
          f"({obs_pca.explained_variance_ratio_.sum()*100:.2f}%)")

    # Transform
    train_obs_pc = torch.from_numpy(obs_pca.transform(obs_np)).float().to(device)
    val_obs_pc = torch.from_numpy(obs_pca.transform(val_obs.numpy())).float().to(device)
    train_act = train_act.to(device)
    val_act = val_act.to(device)

    action_dim = dataset.action_dim
    n_train = train_obs_pc.shape[0]

    # --- Build MIP with PCA obs dim ---
    model = MIPPolicy(
        input_dim=n_obs_pc,  # PCA obs dim instead of raw
        action_dim=action_dim,
        cond_steps=1, horizon_steps=1,
        t_star=t_star, dropout=dropout,
        emb_dim=emb_dim, n_layers=n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MIP: input_dim={n_obs_pc}, {n_params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    from diffusers.optimization import get_scheduler
    lr_sched = get_scheduler('cosine', optimizer=optimizer,
                             num_warmup_steps=500, num_training_steps=total_iters)
    ema = DiffusersEMA(parameters=model.parameters(), power=0.75)
    ema_model = copy.deepcopy(model)

    # Norm stats for action denormalization at eval
    action_min = dataset.action_min.to(device)
    action_max = dataset.action_max.to(device)

    best_sr, t0 = -1.0, time.time()
    print(f"Training {total_iters} iters...")

    for it in range(1, total_iters + 1):
        model.train()
        idx = torch.randint(n_train, (batch_size,), device=device)
        loss, l0, lt = model.compute_loss(train_obs_pc[idx], train_act[idx])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        lr_sched.step(); ema.step(model.parameters())

        if it % log_freq == 0:
            with torch.no_grad():
                vl, vl0, vlt = model.compute_loss(val_obs_pc, val_act)
            elapsed = time.time() - t0
            print(f"Iter {it}: loss={loss.item():.6f} (t0={l0.item():.6f}, t*={lt.item():.6f}), "
                  f"val={vl.item():.6f}, {elapsed:.0f}s")

        if it % eval_freq == 0 or it == total_iters:
            ema.copy_to(ema_model.parameters())
            ema_model.eval()
            sr = _eval_cpu(ema_model, obs_pca, action_min, action_max, device,
                           env_id, control_mode, max_ep_steps)
            print(f"  Eval @ {it}: SR={sr:.1%}")
            if sr > best_sr:
                best_sr = sr
                print(f"  New best: {sr:.1%}")

    print(f"\nDone. Best SR: {best_sr:.1%}")


@torch.no_grad()
def _eval_cpu(model, obs_pca, action_min, action_max, device,
              env_id, control_mode, max_ep_steps, n_episodes=100, num_envs=10):
    """Eval MIP with PCA obs preprocessing."""
    model.eval()
    a_lo, a_hi = action_min, action_max

    def make_env(seed):
        def thunk():
            env = gym.make(env_id, obs_mode="state", control_mode=control_mode,
                           render_mode="rgb_array", max_episode_steps=max_ep_steps,
                           reconfiguration_freq=1)
            return CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
        return thunk

    envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(num_envs)])
    eps_done, so_list, sa_list = 0, [], []

    while eps_done < n_episodes:
        obs_np, _ = envs.reset()
        step, done = 0, False
        while step < max_ep_steps and not done:
            obs = obs_np.copy()
            obs[:, 9:18] = 0.0  # zero_qvel
            # PCA transform
            obs_pc = obs_pca.transform(obs)
            obs_t = torch.from_numpy(obs_pc).float().to(device)
            act = model.predict(obs_t)
            # Denormalize action
            act = (act + 1.0) / 2.0 * (a_hi - a_lo) + a_lo
            obs_np, rew, terminated, truncated, info = envs.step(act.cpu().numpy())
            step += 1
            if truncated.any():
                for fi in info.get("final_info", []):
                    if fi and "episode" in fi:
                        so_list.append(fi["episode"]["success_once"])
                        sa_list.append(fi["episode"]["success_at_end"])
                eps_done += num_envs
                done = True

    envs.close()
    return np.mean(so_list[:n_episodes])


if __name__ == "__main__":
    main()
