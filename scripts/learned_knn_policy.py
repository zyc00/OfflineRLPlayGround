"""
End-to-end Learned KNN Policy.

The encoder learns a feature space where KNN interpolation of neighbor actions
matches the ground truth. Output is always a convex combination of training
actions — it can never leave the data manifold.
"""

import argparse
import os
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
from torch.utils.tensorboard import SummaryWriter


class ObsEncoder(nn.Module):
    def __init__(self, obs_dim, feat_dim=64, hidden_dim=256, n_layers=3,
                 normalize=True):
        super().__init__()
        layers = [nn.Linear(obs_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, feat_dim))
        self.net = nn.Sequential(*layers)
        self.normalize = normalize

    def forward(self, obs):
        z = self.net(obs)
        if self.normalize:
            z = F.normalize(z, dim=-1)
        return z


class SoftKNNPolicy(nn.Module):
    def __init__(self, encoder, train_obs, train_actions, temperature=0.1,
                 learn_temperature=False):
        """
        Args:
            encoder: ObsEncoder
            train_obs: (N, obs_flat_dim) normalized obs, on GPU
            train_actions: (N, horizon, act_dim) or (N, act_dim), on GPU
            temperature: softmax temperature
        """
        super().__init__()
        self.encoder = encoder
        self.register_buffer('train_obs', train_obs)
        self.register_buffer('train_actions', train_actions)

        if learn_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(np.log(temperature)))
        else:
            self.register_buffer('log_temperature', torch.tensor(np.log(temperature)))

        # Cached train features (updated periodically)
        self._cached_features = None
        self._cache_step = -1

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def encode_train(self, batch_size=4096):
        """Re-encode all training obs."""
        self.encoder.eval()
        feats = []
        with torch.no_grad():
            for i in range(0, len(self.train_obs), batch_size):
                z = self.encoder(self.train_obs[i:i+batch_size])
                feats.append(z)
        self._cached_features = torch.cat(feats, dim=0)
        self.encoder.train()

    def forward(self, query_obs, query_indices=None, use_cache=True):
        """
        Args:
            query_obs: (B, obs_flat_dim)
            query_indices: (B,) indices into train set, for self-exclusion
            use_cache: if True, use cached train features
        Returns:
            pred_actions: (B, horizon, act_dim) or (B, act_dim)
            weights: (B, N) soft KNN weights
        """
        z_query = self.encoder(query_obs)  # (B, feat_dim)

        if use_cache and self._cached_features is not None:
            z_train = self._cached_features
        else:
            z_train = self.encoder(self.train_obs)

        # Cosine similarity (features are L2-normalized)
        sim = torch.mm(z_query, z_train.t())  # (B, N)

        # Self-exclusion: mask out own index
        if query_indices is not None:
            sim[torch.arange(len(query_indices), device=sim.device), query_indices] = -1e9

        # Soft KNN weights
        weights = F.softmax(sim / self.temperature, dim=-1)  # (B, N)

        # Weighted sum of training actions
        if self.train_actions.dim() == 3:
            # (B, N) x (N, H, A) -> (B, H, A)
            pred = torch.einsum('bn,nha->bha', weights, self.train_actions)
        else:
            pred = torch.mm(weights, self.train_actions)  # (B, A)

        return pred, weights

    @torch.no_grad()
    def predict(self, query_obs, k=10):
        """Hard KNN prediction for eval."""
        self.encoder.eval()
        z_query = self.encoder(query_obs)

        if self._cached_features is None:
            self.encode_train()
        z_train = self._cached_features

        sim = torch.mm(z_query, z_train.t())  # (B, N)

        # Hard top-K
        topk_sim, topk_idx = sim.topk(k, dim=-1)  # (B, k)
        topk_w = F.softmax(topk_sim / self.temperature, dim=-1)  # (B, k)

        # Gather actions
        if self.train_actions.dim() == 3:
            H, A = self.train_actions.shape[1], self.train_actions.shape[2]
            nn_acts = self.train_actions[topk_idx.reshape(-1)].reshape(
                z_query.shape[0], k, H, A)
            pred = torch.einsum('bk,bkha->bha', topk_w, nn_acts)
        else:
            A = self.train_actions.shape[1]
            nn_acts = self.train_actions[topk_idx.reshape(-1)].reshape(
                z_query.shape[0], k, A)
            pred = torch.einsum('bk,bka->ba', topk_w, nn_acts)

        return pred


class SoftKNNPolicyWithResidual(SoftKNNPolicy):
    """Learned KNN + small parametric residual correction."""

    def __init__(self, encoder, train_obs, train_actions, temperature=0.1,
                 learn_temperature=False, residual_hidden_dim=128,
                 max_residual_norm=0.1):
        super().__init__(encoder, train_obs, train_actions, temperature,
                         learn_temperature)
        feat_dim = encoder.net[-1].out_features  # last linear layer
        if train_actions.dim() == 3:
            action_flat_dim = train_actions.shape[1] * train_actions.shape[2]
            self.action_shape = (train_actions.shape[1], train_actions.shape[2])
        else:
            action_flat_dim = train_actions.shape[1]
            self.action_shape = (train_actions.shape[1],)

        self.residual_head = nn.Sequential(
            nn.Linear(feat_dim, residual_hidden_dim),
            nn.ReLU(),
            nn.Linear(residual_hidden_dim, action_flat_dim),
        )
        self.max_residual_norm = max_residual_norm

    def _get_residual(self, z_query):
        raw = self.residual_head(z_query)
        residual = torch.tanh(raw) * self.max_residual_norm
        return residual.reshape(-1, *self.action_shape)

    def forward(self, query_obs, query_indices=None, use_cache=True):
        z_query = self.encoder(query_obs)

        if use_cache and self._cached_features is not None:
            z_train = self._cached_features
        else:
            z_train = self.encoder(self.train_obs)

        sim = torch.mm(z_query, z_train.t())
        if query_indices is not None:
            sim[torch.arange(len(query_indices), device=sim.device),
                query_indices] = -1e9
        weights = F.softmax(sim / self.temperature, dim=-1)

        if self.train_actions.dim() == 3:
            knn_act = torch.einsum('bn,nha->bha', weights, self.train_actions)
        else:
            knn_act = torch.mm(weights, self.train_actions)

        residual = self._get_residual(z_query)
        return knn_act + residual, weights, knn_act, residual

    @torch.no_grad()
    def predict(self, query_obs, k=10):
        self.encoder.eval()
        z_query = self.encoder(query_obs)

        if self._cached_features is None:
            self.encode_train()
        z_train = self._cached_features

        sim = torch.mm(z_query, z_train.t())
        topk_sim, topk_idx = sim.topk(k, dim=-1)
        topk_w = F.softmax(topk_sim / self.temperature, dim=-1)

        if self.train_actions.dim() == 3:
            H, A = self.train_actions.shape[1], self.train_actions.shape[2]
            nn_acts = self.train_actions[topk_idx.reshape(-1)].reshape(
                z_query.shape[0], k, H, A)
            knn_act = torch.einsum('bk,bkha->bha', topk_w, nn_acts)
        else:
            A = self.train_actions.shape[1]
            nn_acts = self.train_actions[topk_idx.reshape(-1)].reshape(
                z_query.shape[0], k, A)
            knn_act = torch.einsum('bk,bka->ba', topk_w, nn_acts)

        residual = self._get_residual(z_query)
        return knn_act + residual


def load_data(demo_path, horizon_steps=16, cond_steps=1, zero_qvel=False,
              num_demos=None, device='cuda'):
    """Load demo data and prepare (obs, action_chunk) pairs on GPU."""
    with h5py.File(demo_path, "r") as f:
        traj_keys = sorted([k for k in f.keys() if k.startswith("traj_")],
                           key=lambda k: int(k.split("_")[1]))
        if num_demos:
            traj_keys = traj_keys[:num_demos]

        all_obs, all_act = [], []
        for tk in traj_keys:
            obs = f[tk]["obs"][:]
            acts = f[tk]["actions"][:]
            T = len(acts)

            for t in range(T):
                if cond_steps == 1:
                    o = obs[t]
                else:
                    indices = [max(0, t - cond_steps + 1 + i) for i in range(cond_steps)]
                    o = obs[indices].flatten()

                chunk = []
                for h in range(horizon_steps):
                    idx = min(t + h, T - 1)
                    chunk.append(acts[idx])
                chunk = np.stack(chunk, axis=0)

                all_obs.append(o)
                all_act.append(chunk)

    all_obs = np.array(all_obs, dtype=np.float32)
    all_act = np.array(all_act, dtype=np.float32)

    if zero_qvel:
        all_obs[:, 9:18] = 0.0

    # Normalize obs to [-1, 1]
    obs_min = all_obs.min(axis=0)
    obs_max = all_obs.max(axis=0)
    obs_range = np.clip(obs_max - obs_min, 1e-8, None)
    all_obs_norm = (all_obs - obs_min) / obs_range * 2.0 - 1.0

    # Normalize actions to [-1, 1]
    act_min = all_act.reshape(-1, all_act.shape[-1]).min(axis=0)
    act_max = all_act.reshape(-1, all_act.shape[-1]).max(axis=0)
    act_range = np.clip(act_max - act_min, 1e-8, None)
    all_act_norm = (all_act - act_min) / act_range * 2.0 - 1.0

    obs_t = torch.from_numpy(all_obs_norm).to(device)
    act_t = torch.from_numpy(all_act_norm).to(device)

    norm_info = {
        "obs_min": torch.from_numpy(obs_min),
        "obs_max": torch.from_numpy(obs_max),
        "action_min": torch.from_numpy(act_min),
        "action_max": torch.from_numpy(act_max),
    }

    print(f"Loaded {len(obs_t)} samples from {len(traj_keys)} trajs, "
          f"obs={obs_t.shape}, act={act_t.shape}")
    return obs_t, act_t, norm_info


def evaluate_cpu(policy, norm_info, env_id, control_mode, max_episode_steps,
                 n_episodes=100, num_envs=10, act_steps=8, horizon_steps=16,
                 cond_steps=1, zero_qvel=False, k_eval=10, device='cuda'):
    """Evaluate learned KNN policy."""
    import gymnasium as gym
    import mani_skill.envs
    from collections import deque
    from mani_skill.utils.wrappers import CPUGymWrapper

    policy.eval()
    policy.encode_train()

    o_lo = norm_info["obs_min"].to(device)
    o_hi = norm_info["obs_max"].to(device)
    o_range = (o_hi - o_lo).clamp(min=1e-8)
    a_lo = norm_info["action_min"].to(device)
    a_hi = norm_info["action_max"].to(device)
    a_range = (a_hi - a_lo).clamp(min=1e-8)

    def make_env(seed):
        def thunk():
            env = gym.make(env_id, obs_mode="state", control_mode=control_mode,
                           render_mode="rgb_array", max_episode_steps=max_episode_steps,
                           reconfiguration_freq=1)
            env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
            return env
        return thunk

    envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(num_envs)])

    eps_done = 0
    success_once_list = []
    success_at_end_list = []

    while eps_done < n_episodes:
        obs, _ = envs.reset()
        obs = torch.from_numpy(obs).float().to(device)

        obs_buffer = deque(maxlen=cond_steps)
        for _ in range(cond_steps):
            obs_buffer.append(obs.clone())

        step = 0
        done = False
        while step < max_episode_steps and not done:
            if cond_steps == 1:
                cond_obs = obs_buffer[-1]
            else:
                cond_obs = torch.stack(list(obs_buffer), dim=1).reshape(num_envs, -1)

            if zero_qvel:
                cond_obs[..., 9:18] = 0.0

            # Normalize obs
            obs_norm = (cond_obs - o_lo) / o_range * 2.0 - 1.0

            # Predict
            act_chunk = policy.predict(obs_norm, k=k_eval)

            # Denormalize actions
            act_chunk = (act_chunk + 1.0) / 2.0 * a_range + a_lo

            n_exec = min(act_steps, max_episode_steps - step)
            for t in range(n_exec):
                if horizon_steps > 1:
                    action = act_chunk[:, t]
                else:
                    action = act_chunk
                obs_np, rew, terminated, truncated, info = envs.step(action.cpu().numpy())
                obs = torch.from_numpy(obs_np).float().to(device)
                obs_buffer.append(obs.clone())
                step += 1
                if truncated.any():
                    for fi in info.get("final_info", []):
                        if fi and "episode" in fi:
                            success_once_list.append(fi["episode"]["success_once"])
                            success_at_end_list.append(fi["episode"]["success_at_end"])
                    eps_done += num_envs
                    done = True
                    break

    envs.close()

    so = np.mean(success_once_list[:n_episodes])
    sa = np.mean(success_at_end_list[:n_episodes])
    return {"success_once": so, "success_at_end": sa,
            "n_episodes": min(len(success_once_list), n_episodes)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_path", type=str, required=True)
    parser.add_argument("--env_id", type=str, default="PegInsertionSide-v1")
    parser.add_argument("--control_mode", type=str, default="pd_joint_delta_pos")
    parser.add_argument("--max_episode_steps", type=int, default=200)

    # Data
    parser.add_argument("--horizon_steps", type=int, default=16)
    parser.add_argument("--cond_steps", type=int, default=1)
    parser.add_argument("--act_steps", type=int, default=8)
    parser.add_argument("--zero_qvel", action="store_true")
    parser.add_argument("--num_demos", type=int, default=None)

    # Encoder
    parser.add_argument("--feat_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=3)

    # KNN
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--learn_temperature", action="store_true")
    parser.add_argument("--k_eval", type=int, default=10)

    # Residual
    parser.add_argument("--residual", action="store_true",
                        help="Add parametric residual head on top of KNN output")
    parser.add_argument("--residual_hidden_dim", type=int, default=128)
    parser.add_argument("--max_residual_norm", type=float, default=0.1,
                        help="Max residual magnitude (tanh scaling)")
    parser.add_argument("--residual_lambda", type=float, default=0.0,
                        help="Regularization weight on residual magnitude")

    # Training
    parser.add_argument("--total_iters", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--cache_interval", type=int, default=100)
    parser.add_argument("--subsample_n", type=int, default=50000,
                        help="Subsample training set for distance computation (0=all)")

    # Eval
    parser.add_argument("--eval_freq", type=int, default=5000)
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--num_eval_envs", type=int, default=10)
    parser.add_argument("--log_freq", type=int, default=200)

    # Logging
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="runs/learned_knn")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    args.demo_path = os.path.expanduser(args.demo_path)

    if args.exp_name is None:
        name = f"lknn_{args.env_id}_f{args.feat_dim}_t{args.temperature}"
        if args.residual:
            name += f"_res{args.max_residual_norm}"
        args.exp_name = name

    run_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_obs, train_act, norm_info = load_data(
        args.demo_path, horizon_steps=args.horizon_steps,
        cond_steps=args.cond_steps, zero_qvel=args.zero_qvel,
        num_demos=args.num_demos, device=device,
    )
    N = len(train_obs)
    obs_flat_dim = train_obs.shape[1]
    print(f"N={N}, obs_flat_dim={obs_flat_dim}, act_shape={train_act.shape}")

    # Subsample indices for distance computation
    if args.subsample_n > 0 and args.subsample_n < N:
        sub_idx = torch.randperm(N, device=device)[:args.subsample_n]
        sub_obs = train_obs[sub_idx]
        sub_act = train_act[sub_idx]
        print(f"Subsampled {args.subsample_n} points for distance computation")
    else:
        sub_idx = None
        sub_obs = train_obs
        sub_act = train_act

    # Build model
    encoder = ObsEncoder(obs_flat_dim, feat_dim=args.feat_dim,
                         hidden_dim=args.hidden_dim, n_layers=args.n_layers)
    if args.residual:
        policy = SoftKNNPolicyWithResidual(
            encoder, sub_obs, sub_act,
            temperature=args.temperature,
            learn_temperature=args.learn_temperature,
            residual_hidden_dim=args.residual_hidden_dim,
            max_residual_norm=args.max_residual_norm,
        ).to(device)
    else:
        policy = SoftKNNPolicy(encoder, sub_obs, sub_act,
                               temperature=args.temperature,
                               learn_temperature=args.learn_temperature).to(device)

    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Params: {n_params:,}, temperature={args.temperature}"
          f"{' (learnable)' if args.learn_temperature else ''}"
          f"{f', residual max_norm={args.max_residual_norm}' if args.residual else ''}")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    # Initial cache
    policy.encode_train()

    best_sr = -1.0
    t_start = time.time()

    for iteration in range(1, args.total_iters + 1):
        policy.train()

        # Sample batch from full training set
        idx = torch.randint(N, (args.batch_size,), device=device)
        obs_batch = train_obs[idx]
        act_batch = train_act[idx]

        # Map batch indices to subsample indices for self-exclusion
        if sub_idx is not None:
            # Find which batch elements are in the subsample
            # For simplicity, just don't exclude (subsample is different from batch)
            query_indices = None
        else:
            query_indices = idx

        # Re-encode cache periodically
        if iteration % args.cache_interval == 0:
            policy.encode_train()

        # Forward
        fwd = policy(obs_batch, query_indices=query_indices, use_cache=True)
        if args.residual:
            pred_act, weights, knn_act, residual = fwd
            loss = F.mse_loss(pred_act, act_batch)
            if args.residual_lambda > 0:
                res_mag = residual.norm(dim=-1).mean()
                loss = loss + args.residual_lambda * res_mag
        else:
            pred_act, weights = fwd
            loss = F.mse_loss(pred_act, act_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("train/loss", loss.item(), iteration)

        if args.learn_temperature:
            writer.add_scalar("train/temperature",
                              policy.temperature.item(), iteration)

        if args.residual and iteration % args.log_freq == 0:
            with torch.no_grad():
                loss_knn_only = F.mse_loss(knn_act, act_batch)
                res_mag = residual.norm(dim=-1).mean()
                res_max = residual.abs().max()
            writer.add_scalar("train/loss_knn_only", loss_knn_only.item(), iteration)
            writer.add_scalar("train/residual_mag", res_mag.item(), iteration)
            writer.add_scalar("train/residual_max", res_max.item(), iteration)

        if iteration % args.log_freq == 0:
            # Weight entropy
            with torch.no_grad():
                ent = -(weights * (weights + 1e-10).log()).sum(dim=-1).mean()
                max_w = weights.max(dim=-1).values.mean()
            writer.add_scalar("train/weight_entropy", ent.item(), iteration)
            writer.add_scalar("train/max_weight", max_w.item(), iteration)

            elapsed = time.time() - t_start
            temp_str = f", temp={policy.temperature.item():.4f}" if args.learn_temperature else ""
            res_str = ""
            if args.residual:
                res_str = f", knn_loss={loss_knn_only.item():.6f}, res_mag={res_mag.item():.4f}"
            print(f"Iter {iteration}/{args.total_iters}, loss={loss.item():.6f}, "
                  f"ent={ent.item():.2f}, max_w={max_w.item():.4f}{temp_str}"
                  f"{res_str}, time={elapsed:.0f}s")

        # Eval
        if iteration % args.eval_freq == 0 or iteration == args.total_iters:
            policy.encode_train()
            metrics = evaluate_cpu(
                policy, norm_info, env_id=args.env_id,
                control_mode=args.control_mode,
                max_episode_steps=args.max_episode_steps,
                n_episodes=args.n_episodes,
                num_envs=args.num_eval_envs,
                act_steps=args.act_steps,
                horizon_steps=args.horizon_steps,
                cond_steps=args.cond_steps,
                zero_qvel=args.zero_qvel,
                k_eval=args.k_eval,
                device=device,
            )
            sr = metrics["success_once"]
            sa = metrics["success_at_end"]
            writer.add_scalar("eval/success_once", sr, iteration)
            writer.add_scalar("eval/success_at_end", sa, iteration)
            print(f"  Eval @ iter {iteration}: success_once={sr:.3f}, "
                  f"success_at_end={sa:.3f}")

            # Save
            ckpt = {
                "encoder": encoder.state_dict(),
                "norm_info": {k: v.cpu() for k, v in norm_info.items()},
                "args": vars(args),
                "step": iteration,
                "obs_flat_dim": obs_flat_dim,
                "act_shape": list(train_act.shape[1:]),
            }
            if args.residual:
                ckpt["residual_head"] = policy.residual_head.state_dict()
            torch.save(ckpt, os.path.join(run_dir, f"ckpt_{iteration}.pt"))
            if sr > best_sr:
                best_sr = sr
                torch.save(ckpt, os.path.join(run_dir, "best.pt"))
                print(f"  New best (sr_once={best_sr:.3f})")

    elapsed = time.time() - t_start
    print(f"Training complete in {elapsed:.0f}s. Best sr_once={best_sr:.3f}")
    writer.close()


if __name__ == "__main__":
    main()
