"""Pretrain IQL Q(s,a) and V(s) from expert policy rollouts.

Collects trajectories with expert policy in sparse reward env,
trains Q and V via IQL (tau=0.5 = SARSA by default).

Usage:
  python -m RL.pretrain_iql
  python -m RL.pretrain_iql --expectile_tau 0.7
"""

import copy
import os
import random
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from data.data_collection.ppo import Agent, layer_init
from methods.iql.iql import QNetwork, expectile_loss


@dataclass
class Args:
    expert_checkpoint: str = "runs/pickcube_ppo/ckpt_301.pt"
    """expert policy checkpoint for data collection"""
    env_id: str = "PickCube-v1"
    num_envs: int = 512
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"
    max_episode_steps: int = 50
    gamma: float = 0.8
    seed: int = 1
    cuda: bool = True

    # Data collection
    num_rollouts: int = 20
    """number of full rollouts (each = num_steps * num_envs transitions)"""
    num_steps: int = 50

    # IQL training
    expectile_tau: float = 0.5
    """0.5 = SARSA (mean), >0.5 = IQL (optimistic)"""
    tau_polyak: float = 0.005
    epochs: int = 200
    batch_size: int = 4096
    learning_rate: float = 1e-3
    grad_clip: float = 0.5

    # Output
    output_dir: Optional[str] = None


if __name__ == "__main__":
    args = tyro.cli(Args)

    tau_str = f"tau{args.expectile_tau}"
    if args.output_dir is None:
        args.output_dir = f"runs/pretrained_iql_{tau_str}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ── Environment ───────────────────────────────────────────────────
    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="physx_cuda" if device.type == "cuda" else "cpu",
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
    )
    envs = gym.make(args.env_id, num_envs=args.num_envs, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(
        envs, args.num_envs,
        ignore_terminations=False,
        record_metrics=True,
    )

    # ── Load expert ───────────────────────────────────────────────────
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.expert_checkpoint, map_location=device))
    agent.eval()
    print(f"Loaded expert: {args.expert_checkpoint}")

    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(a):
        return torch.clamp(a.detach(), action_low, action_high)

    # ── Collect data ──────────────────────────────────────────────────
    obs_dim = np.array(envs.single_observation_space.shape).prod()
    act_dim = np.prod(envs.single_action_space.shape)
    act_shape = envs.single_action_space.shape

    print(f"Collecting {args.num_rollouts} rollouts "
          f"({args.num_envs} envs × {args.num_steps} steps each)...")

    all_obs = []
    all_actions = []
    all_rewards = []
    all_next_obs = []
    all_dones = []

    next_obs, _ = envs.reset(seed=args.seed)

    for rollout_idx in range(args.num_rollouts):
        obs_buf = torch.zeros(
            (args.num_steps, args.num_envs, obs_dim), device=device,
        )
        act_buf = torch.zeros(
            (args.num_steps, args.num_envs) + act_shape, device=device,
        )
        rew_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
        next_obs_buf = torch.zeros(
            (args.num_steps, args.num_envs, obs_dim), device=device,
        )
        done_buf = torch.zeros((args.num_steps, args.num_envs), device=device)

        for step in range(args.num_steps):
            obs_buf[step] = next_obs
            with torch.no_grad():
                action = agent.get_action(next_obs, deterministic=False)
            clipped = clip_action(action)
            next_obs, reward, term, trunc, infos = envs.step(clipped)

            act_buf[step] = clipped
            rew_buf[step] = reward.view(-1)
            done_buf[step] = (term | trunc).float()

            # Store true next obs (before auto-reset)
            next_obs_buf[step] = next_obs.clone()
            if "final_info" in infos:
                done_mask = infos["_final_info"]
                next_obs_buf[step, done_mask] = infos["final_observation"][done_mask]

        n_success = (rew_buf > 0).any(dim=0).float().mean()
        all_obs.append(obs_buf.reshape(-1, obs_dim).cpu())
        all_actions.append(act_buf.reshape(-1, act_dim).cpu())
        all_rewards.append(rew_buf.reshape(-1).cpu())
        all_next_obs.append(next_obs_buf.reshape(-1, obs_dim).cpu())
        all_dones.append(done_buf.reshape(-1).cpu())

        print(f"  Rollout {rollout_idx+1}/{args.num_rollouts}: "
              f"success_rate={n_success:.1%}, "
              f"total_reward={rew_buf.sum():.0f}")

    envs.close()

    # ── Prepare dataset ───────────────────────────────────────────────
    all_obs = torch.cat(all_obs, dim=0)
    all_actions = torch.cat(all_actions, dim=0)
    all_rewards = torch.cat(all_rewards, dim=0)
    all_next_obs = torch.cat(all_next_obs, dim=0)
    all_dones = torch.cat(all_dones, dim=0)
    N = len(all_obs)

    print(f"\nTotal data: {N} transitions")
    print(f"  Rewards: {(all_rewards > 0).sum().item()} positive / {N}")
    print(f"  Dones: {(all_dones > 0).sum().item()} / {N}")

    # Train/val split
    perm = torch.randperm(N)
    val_size = min(N // 10, 10000)
    train_idx = perm[val_size:]
    val_idx = perm[:val_size]

    # ── Train IQL ─────────────────────────────────────────────────────
    q_net = QNetwork(obs_dim, act_dim).to(device)
    q_target = copy.deepcopy(q_net)
    v_net = nn.Sequential(
        layer_init(nn.Linear(obs_dim, 256)),
        nn.Tanh(),
        layer_init(nn.Linear(256, 256)),
        nn.Tanh(),
        layer_init(nn.Linear(256, 256)),
        nn.Tanh(),
        layer_init(nn.Linear(256, 1)),
    ).to(device)

    q_optimizer = optim.Adam(q_net.parameters(), lr=args.learning_rate)
    v_optimizer = optim.Adam(v_net.parameters(), lr=args.learning_rate)

    print(f"\nTraining IQL (tau={args.expectile_tau}): "
          f"{len(train_idx)} train, {len(val_idx)} val")

    best_val_loss = float("inf")
    best_q_state = None
    best_v_state = None

    for epoch in range(args.epochs):
        q_net.train()
        v_net.train()
        indices = train_idx[torch.randperm(len(train_idx))]
        epoch_q_loss = 0.0
        epoch_v_loss = 0.0
        n_batches = 0

        for start in range(0, len(train_idx), args.batch_size):
            mb = indices[start : start + args.batch_size]
            s = all_obs[mb].to(device)
            a = all_actions[mb].to(device)
            r = all_rewards[mb].to(device)
            ns = all_next_obs[mb].to(device)
            d = all_dones[mb].to(device)

            # Q loss: Q(s,a) → r + γ V(s') (1 - done)
            with torch.no_grad():
                v_next = v_net(ns).squeeze(-1)
                q_tgt = r + args.gamma * v_next * (1.0 - d)
            q_pred = q_net(s, a).squeeze(-1)
            q_loss = 0.5 * ((q_pred - q_tgt) ** 2).mean()

            q_optimizer.zero_grad()
            q_loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), args.grad_clip)
            q_optimizer.step()

            # V loss: expectile regression against target Q
            with torch.no_grad():
                q_val = q_target(s, a).squeeze(-1)
            v_pred = v_net(s).squeeze(-1)
            v_loss = expectile_loss(q_val - v_pred, args.expectile_tau)

            v_optimizer.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(v_net.parameters(), args.grad_clip)
            v_optimizer.step()

            # Polyak update target Q
            with torch.no_grad():
                for p, pt in zip(q_net.parameters(), q_target.parameters()):
                    pt.data.mul_(1.0 - args.tau_polyak).add_(p.data, alpha=args.tau_polyak)

            epoch_q_loss += q_loss.item()
            epoch_v_loss += v_loss.item()
            n_batches += 1

        # Validation
        q_net.eval()
        v_net.eval()
        with torch.no_grad():
            vs = all_obs[val_idx].to(device)
            va = all_actions[val_idx].to(device)
            vr = all_rewards[val_idx].to(device)
            vns = all_next_obs[val_idx].to(device)
            vd = all_dones[val_idx].to(device)

            v_next_val = v_net(vns).squeeze(-1)
            q_tgt_val = vr + args.gamma * v_next_val * (1.0 - vd)
            q_pred_val = q_net(vs, va).squeeze(-1)
            val_q_loss = 0.5 * ((q_pred_val - q_tgt_val) ** 2).mean().item()

            q_for_v = q_target(vs, va).squeeze(-1)
            v_pred_val = v_net(vs).squeeze(-1)
            val_v_loss = expectile_loss(
                q_for_v - v_pred_val, args.expectile_tau
            ).item()

        val_total = val_q_loss + val_v_loss
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_q_state = {k: v.cpu().clone() for k, v in q_net.state_dict().items()}
            best_v_state = {k: v.cpu().clone() for k, v in v_net.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}: "
                  f"q={epoch_q_loss/n_batches:.6f}, v={epoch_v_loss/n_batches:.6f}, "
                  f"val_q={val_q_loss:.6f}, val_v={val_v_loss:.6f}")

    # Load best
    q_net.load_state_dict(best_q_state)
    v_net.load_state_dict(best_v_state)
    q_net.eval()
    v_net.eval()

    # Summary
    with torch.no_grad():
        all_q, all_v = [], []
        for start in range(0, N, args.batch_size):
            end = min(start + args.batch_size, N)
            s = all_obs[start:end].to(device)
            a = all_actions[start:end].to(device)
            all_q.append(q_net(s, a).squeeze(-1).cpu())
            all_v.append(v_net(s).squeeze(-1).cpu())
        all_q = torch.cat(all_q)
        all_v = torch.cat(all_v)
        all_adv = all_q - all_v
    print(f"\n  Q(s,a): mean={all_q.mean():.4f}, std={all_q.std():.4f}")
    print(f"  V(s):   mean={all_v.mean():.4f}, std={all_v.std():.4f}")
    print(f"  A(s,a): mean={all_adv.mean():.4f}, std={all_adv.std():.4f}")

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(best_q_state, os.path.join(args.output_dir, "q_net.pt"))
    torch.save(best_v_state, os.path.join(args.output_dir, "v_net.pt"))
    print(f"\nSaved to {args.output_dir}/")
    print(f"  Best val loss: {best_val_loss:.6f}")
