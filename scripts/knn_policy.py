"""
KNN Policy: nearest-neighbor baseline for imitation learning.

No training — just store demo (obs, action_chunk) pairs and do KNN lookup at inference.
"""

import argparse
import torch
import numpy as np
import h5py
import time
from sklearn.neighbors import BallTree


class KNNPolicy:
    def __init__(self, obs_all, act_all, k=5, weighting='inverse_distance',
                 normalize=True, cond_steps=1):
        """
        Args:
            obs_all: (N, obs_dim) numpy array of all obs
            act_all: (N, horizon, act_dim) or (N, act_dim) numpy array of action chunks
            k: number of nearest neighbors
            weighting: 'uniform' or 'inverse_distance'
            normalize: z-score normalize obs before distance computation
            cond_steps: if >1, obs is (N, cond_steps, obs_dim), flatten for distance
        """
        self.k = k
        self.weighting = weighting
        self.cond_steps = cond_steps

        # Flatten cond_steps into obs
        if obs_all.ndim == 3:
            obs_all = obs_all.reshape(obs_all.shape[0], -1)
        self.obs_all = obs_all
        self.act_all = act_all

        # Normalize
        self.normalize = normalize
        if normalize:
            self.obs_mean = obs_all.mean(axis=0)
            self.obs_std = obs_all.std(axis=0).clip(min=1e-8)
            obs_normed = (obs_all - self.obs_mean) / self.obs_std
        else:
            obs_normed = obs_all

        self.tree = BallTree(obs_normed, metric='euclidean')

    def predict(self, obs_batch):
        """
        Args:
            obs_batch: (B, obs_dim) torch tensor on any device
        Returns:
            actions: (B, act_dim) torch tensor (first action of chunk)
        """
        device = obs_batch.device
        obs_np = obs_batch.cpu().numpy()

        if obs_np.ndim == 3:
            obs_np = obs_np.reshape(obs_np.shape[0], -1)

        if self.normalize:
            obs_np = (obs_np - self.obs_mean) / self.obs_std

        dists, indices = self.tree.query(obs_np, k=self.k)  # (B, k)

        B = obs_np.shape[0]
        actions = np.zeros((B,) + self.act_all.shape[1:], dtype=np.float32)

        for i in range(B):
            nn_acts = self.act_all[indices[i]]  # (k, ...) action chunks
            if self.weighting == 'inverse_distance':
                w = 1.0 / (dists[i] + 1e-8)
                w = w / w.sum()
            else:
                w = np.ones(self.k) / self.k

            # Weighted average
            for j in range(self.k):
                actions[i] += w[j] * nn_acts[j]

        return torch.from_numpy(actions).to(device)


def build_knn_from_demos(demo_path, horizon_steps=16, cond_steps=1,
                         k=5, weighting='inverse_distance', normalize=True,
                         zero_qvel=False, num_demos=None):
    """Build KNN policy from H5 demo file."""
    with h5py.File(demo_path, "r") as f:
        traj_keys = sorted([k for k in f.keys() if k.startswith("traj_")],
                           key=lambda k: int(k.split("_")[1]))
        if num_demos:
            traj_keys = traj_keys[:num_demos]

        all_obs, all_act = [], []
        for tk in traj_keys:
            obs = f[tk]["obs"][:]   # (T, obs_dim)
            acts = f[tk]["actions"][:]  # (T-1, act_dim) or (T, act_dim)
            T = len(acts)

            for t in range(T):
                # Obs: cond_steps frames
                if cond_steps == 1:
                    o = obs[t]
                else:
                    indices = [max(0, t - cond_steps + 1 + i) for i in range(cond_steps)]
                    o = obs[indices]  # (cond_steps, obs_dim)

                # Action chunk
                chunk = []
                for h in range(horizon_steps):
                    idx = min(t + h, T - 1)
                    chunk.append(acts[idx])
                chunk = np.stack(chunk, axis=0)  # (horizon, act_dim)

                all_obs.append(o)
                all_act.append(chunk)

    all_obs = np.array(all_obs, dtype=np.float32)
    all_act = np.array(all_act, dtype=np.float32)

    if zero_qvel:
        if all_obs.ndim == 3:
            all_obs[:, :, 9:18] = 0.0
        else:
            all_obs[:, 9:18] = 0.0

    print(f"KNN dataset: {len(all_obs)} samples from {len(traj_keys)} trajs, "
          f"obs={all_obs.shape}, act={all_act.shape}")

    policy = KNNPolicy(all_obs, all_act, k=k, weighting=weighting,
                       normalize=normalize, cond_steps=cond_steps)
    return policy


def evaluate_knn_cpu(policy, env_id, control_mode, max_episode_steps,
                     n_episodes=100, num_envs=10, act_steps=1,
                     horizon_steps=16, cond_steps=1, zero_qvel=False):
    """Evaluate KNN policy using CPU envs."""
    import gymnasium as gym
    import mani_skill.envs
    from collections import deque
    from mani_skill.utils.wrappers import CPUGymWrapper

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
        obs = torch.from_numpy(obs).float()

        obs_buffer = deque(maxlen=cond_steps)
        for _ in range(cond_steps):
            obs_buffer.append(obs.clone())

        step = 0
        done = False
        while step < max_episode_steps and not done:
            if cond_steps == 1:
                cond_obs = obs_buffer[-1]
            else:
                cond_obs = torch.stack(list(obs_buffer), dim=1)

            if zero_qvel:
                cond_obs[..., 9:18] = 0.0

            act_chunk = policy.predict(cond_obs)  # (B, horizon, act_dim)

            n_exec = min(act_steps, max_episode_steps - step)
            for t in range(n_exec):
                if horizon_steps > 1:
                    action = act_chunk[:, t].numpy()
                else:
                    action = act_chunk.numpy()

                obs_np, rew, terminated, truncated, info = envs.step(action)
                obs = torch.from_numpy(obs_np).float()
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
    return {"success_once": so, "success_at_end": sa, "n_episodes": min(len(success_once_list), n_episodes)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_path", type=str, required=True)
    parser.add_argument("--env_id", type=str, default="PegInsertionSide-v1")
    parser.add_argument("--control_mode", type=str, default="pd_joint_delta_pos")
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    parser.add_argument("--horizon_steps", type=int, default=16)
    parser.add_argument("--cond_steps", type=int, default=1)
    parser.add_argument("--act_steps", type=int, default=8)
    parser.add_argument("--weighting", type=str, default="inverse_distance",
                        choices=["uniform", "inverse_distance"])
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--zero_qvel", action="store_true")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--num_envs", type=int, default=10)
    parser.add_argument("--num_demos", type=int, default=None)
    # Lipschitz mode
    parser.add_argument("--lipschitz", action="store_true",
                        help="Compute Lipschitz instead of eval")
    parser.add_argument("--num_obs", type=int, default=500)
    parser.add_argument("--delta", type=float, default=1e-3)
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    if args.no_normalize:
        args.normalize = False

    import os
    args.demo_path = os.path.expanduser(args.demo_path)

    if args.lipschitz:
        # Build KNN and compute Lipschitz
        policy = build_knn_from_demos(
            args.demo_path, horizon_steps=args.horizon_steps,
            cond_steps=args.cond_steps, k=args.k[0],
            weighting=args.weighting, normalize=args.normalize,
            zero_qvel=args.zero_qvel, num_demos=args.num_demos,
        )

        # Load obs
        from scripts.lipschitz_analysis import load_obs_from_demos, compute_lipschitz, get_obs_dim_names
        obs_batch = load_obs_from_demos(args.demo_path, args.num_obs, args.zero_qvel)
        obs_dim = obs_batch.shape[1]
        dim_names = get_obs_dim_names(obs_dim)
        device = "cpu"

        def predict_fn(obs_b):
            # obs_b: (B, obs_dim) on device
            if args.cond_steps > 1:
                obs_b = obs_b.unsqueeze(1).expand(-1, args.cond_steps, -1)
            act = policy.predict(obs_b)
            if args.horizon_steps > 1:
                act = act[:, 0]
            return act

        lip_mean, lip_std = compute_lipschitz(predict_fn, obs_batch, args.delta, device)

        print(f"\nKNN (k={args.k[0]}) Lipschitz per dimension:")
        for d in range(obs_dim):
            print(f"  {dim_names[d]:<16} {lip_mean[d]:10.4f} ± {lip_std[d]:.3f}")
        print(f"\nmean L = {lip_mean.mean():.4f}, max L = {lip_mean.max():.4f} ({dim_names[lip_mean.argmax()]})")

        if args.save_path:
            np.savez(args.save_path, dim_names=dim_names,
                     knn_mean=lip_mean, knn_std=lip_std)
            print(f"Saved to {args.save_path}")
        return

    # Eval mode: sweep over k values
    for k in args.k:
        print(f"\n{'='*50}")
        print(f"KNN k={k}, weighting={args.weighting}, normalize={args.normalize}")
        print(f"{'='*50}")

        policy = build_knn_from_demos(
            args.demo_path, horizon_steps=args.horizon_steps,
            cond_steps=args.cond_steps, k=k,
            weighting=args.weighting, normalize=args.normalize,
            zero_qvel=args.zero_qvel, num_demos=args.num_demos,
        )

        t0 = time.time()
        metrics = evaluate_knn_cpu(
            policy, env_id=args.env_id, control_mode=args.control_mode,
            max_episode_steps=args.max_episode_steps,
            n_episodes=args.n_episodes, num_envs=args.num_envs,
            act_steps=args.act_steps, horizon_steps=args.horizon_steps,
            cond_steps=args.cond_steps, zero_qvel=args.zero_qvel,
        )
        elapsed = time.time() - t0

        print(f"  k={k}: success_once={metrics['success_once']:.3f}, "
              f"success_at_end={metrics['success_at_end']:.3f} "
              f"({metrics['n_episodes']} eps, {elapsed:.0f}s)")


if __name__ == "__main__":
    main()
