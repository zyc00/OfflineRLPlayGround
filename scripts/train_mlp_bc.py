"""Train MLP BC with action chunking on PegInsertionSide-v1.

Same data split, normalization, and eval as DP/MIP for fair comparison.

Usage:
    python scripts/train_mlp_bc.py \
      --demo_path ~/.maniskill/demos/PegInsertionSide-v1/policy_from_mp_states/trajectory.state.pd_joint_delta_pos.h5 \
      --n_train 900 --total_iters 100000 \
      --output runs/mlp_bc/mlp_peg_split900
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import h5py
import gymnasium as gym
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mani_skill.envs
import DPPO.peg_insertion_easy
from mani_skill.utils.wrappers import CPUGymWrapper


class MLPPolicy(nn.Module):
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


def load_data(demo_path, n_train, cond_steps, horizon_steps, zero_qvel=True):
    """Load demos and split into train/val, returning (obs_cond, action_chunk) pairs."""
    with h5py.File(demo_path, "r") as f:
        traj_keys = sorted([k for k in f.keys() if k.startswith("traj_")],
                          key=lambda x: int(x.split("_")[1]))
        trajs = []
        for tk in traj_keys:
            trajs.append({
                "obs": f[tk]["obs"][:],
                "actions": f[tk]["actions"][:],
            })

    train_trajs = trajs[:n_train]
    val_trajs = trajs[n_train:]

    def extract(traj_list):
        all_obs = []
        all_act = []
        for traj in traj_list:
            obs = traj["obs"]      # (T+1, obs_dim)
            act = traj["actions"]  # (T, act_dim)
            T = len(act)
            for t in range(T - horizon_steps + 1):
                # Conditioning obs: [t, t+1, ..., t+cond_steps-1]
                obs_cond = obs[t:t + cond_steps]  # (cond_steps, obs_dim)
                if zero_qvel:
                    obs_cond = obs_cond.copy()
                    obs_cond[..., 9:18] = 0.0

                # Action chunk
                act_chunk = act[t:t + horizon_steps]  # (horizon_steps, act_dim)
                if len(act_chunk) < horizon_steps:
                    pad_len = horizon_steps - len(act_chunk)
                    last_act = act_chunk[-1:].copy()
                    last_act[:, :-1] = 0
                    act_chunk = np.concatenate([act_chunk, np.tile(last_act, (pad_len, 1))])

                all_obs.append(obs_cond.flatten())
                all_act.append(act_chunk.flatten())

        return np.array(all_obs, dtype=np.float32), np.array(all_act, dtype=np.float32)

    train_obs, train_act = extract(train_trajs)
    val_obs, val_act = extract(val_trajs)
    return train_obs, train_act, val_obs, val_act


@torch.no_grad()
def evaluate(model, device, obs_dim, action_dim, cond_steps, horizon_steps,
             act_steps, max_episode_steps, n_episodes=100):
    model.eval()
    env = gym.make("PegInsertionSide-v1", obs_mode="state",
                   control_mode="pd_joint_delta_pos", render_mode="rgb_array",
                   max_episode_steps=max_episode_steps, reconfiguration_freq=1)
    env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)

    successes = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        buf = deque(maxlen=cond_steps)
        for _ in range(cond_steps):
            buf.append(obs.copy())
        success = False
        step = 0

        while step < max_episode_steps:
            obs_flat = np.stack(list(buf), axis=0).copy()
            obs_flat[..., 9:18] = 0.0  # zero_qvel
            obs_in = torch.from_numpy(obs_flat.flatten()).float().unsqueeze(0).to(device)
            pred = model(obs_in).cpu().numpy()[0]
            act_chunk = pred.reshape(horizon_steps, action_dim)

            for t in range(min(act_steps, max_episode_steps - step)):
                obs, _, term, trunc, info = env.step(act_chunk[t])
                buf.append(obs.copy())
                step += 1
                if info.get("success", False):
                    success = True
                if term or trunc:
                    break
            if term or trunc:
                break

        successes.append(success)

    env.close()
    return np.mean(successes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_path", default=os.path.expanduser(
        "~/.maniskill/demos/PegInsertionSide-v1/policy_from_mp_states/"
        "trajectory.state.pd_joint_delta_pos.h5"))
    parser.add_argument("--n_train", type=int, default=900)
    parser.add_argument("--cond_steps", type=int, default=2)
    parser.add_argument("--horizon_steps", type=int, default=16)
    parser.add_argument("--act_steps", type=int, default=8)
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--total_iters", type=int, default=100000)
    parser.add_argument("--eval_freq", type=int, default=10000)
    parser.add_argument("--n_eval_episodes", type=int, default=50)
    parser.add_argument("--output", default="runs/mlp_bc/mlp_peg_split900")
    args = parser.parse_args()

    device = torch.device("cuda")
    os.makedirs(args.output, exist_ok=True)

    # Load data
    print(f"Loading data from {args.demo_path}...")
    train_obs, train_act, val_obs, val_act = load_data(
        args.demo_path, args.n_train, args.cond_steps, args.horizon_steps)
    obs_dim = train_obs.shape[1] // args.cond_steps
    action_dim = train_act.shape[1] // args.horizon_steps
    print(f"  Train: {len(train_obs)} samples, Val: {len(val_obs)} samples")
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"  input_dim={train_obs.shape[1]}, output_dim={train_act.shape[1]}")

    train_obs_t = torch.from_numpy(train_obs).to(device)
    train_act_t = torch.from_numpy(train_act).to(device)
    val_obs_t = torch.from_numpy(val_obs).to(device)
    val_act_t = torch.from_numpy(val_act).to(device)

    # Model
    in_dim = train_obs.shape[1]
    out_dim = train_act.shape[1]
    model = MLPPolicy(in_dim, out_dim, hidden=args.hidden, n_layers=args.n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  MLP: {in_dim}→{args.hidden}×{args.n_layers-1}→{out_dim}, {n_params:,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.total_iters)

    best_sr = 0.0
    best_val_loss = float("inf")
    n_tr = len(train_obs_t)

    for it in range(1, args.total_iters + 1):
        model.train()
        idx = torch.randint(n_tr, (args.batch_size,), device=device)
        pred = model(train_obs_t[idx])
        loss = ((pred - train_act_t[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if it % args.eval_freq == 0 or it == 1:
            model.eval()
            with torch.no_grad():
                val_loss = ((model(val_obs_t) - val_act_t) ** 2).mean().item()

            sr = evaluate(model, device, obs_dim, action_dim,
                         args.cond_steps, args.horizon_steps, args.act_steps,
                         args.max_episode_steps, n_episodes=args.n_eval_episodes)

            improved = ""
            if sr > best_sr or (sr == best_sr and val_loss < best_val_loss):
                best_sr = sr
                best_val_loss = val_loss
                torch.save({
                    "model": model.state_dict(),
                    "args": vars(args),
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "iteration": it,
                    "sr": sr,
                    "val_loss": val_loss,
                }, os.path.join(args.output, "best.pt"))
                improved = " ★"

            print(f"  iter {it:6d}: train_loss={loss.item():.6f}, val_loss={val_loss:.6f}, "
                  f"SR={sr:.0%} (best={best_sr:.0%}){improved}", flush=True)

            # Also save periodic checkpoint
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "iteration": it,
                "sr": sr,
                "val_loss": val_loss,
            }, os.path.join(args.output, f"ckpt_{it}.pt"))

    print(f"\nDone. Best SR={best_sr:.0%}")
    print(f"Checkpoints: {args.output}")


if __name__ == "__main__":
    main()
