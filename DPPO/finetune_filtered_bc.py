"""
DPPO Filtered BC Finetuning: iterative rollout → filter successes → BC on replay queue.

Tests whether "BC on your own successes" suffices vs full RL (PPO with advantages,
logprobs, critic, etc.). Uses a replay queue that evolves from offline demos to
on-policy successes.

Usage:
    python -m DPPO.finetune_filtered_bc \
      --pretrain_checkpoint runs/dppo_pretrain/.../best.pt \
      --demo_path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5 \
      --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos \
      --n_envs 100 --n_steps 100 --max_episode_steps 200 \
      --use_ddim --ddim_steps 10 \
      --n_train_itr 50 --eval_freq 5
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, List
from torch.utils.tensorboard import SummaryWriter

import tyro

from DPPO.model.mlp import DiffusionMLP
from DPPO.model.unet_wrapper import DiffusionUNet
from DPPO.model.diffusion import DiffusionModel
from DPPO.dataset import DPPODataset
from DPPO.make_env import make_train_envs


class ReplayQueue:
    """FIFO buffer of (obs_cond, action_chunk) pairs on CPU."""

    def __init__(self, max_size, cond_shape, action_shape):
        """
        Args:
            max_size: maximum number of entries
            cond_shape: (cond_steps, obs_dim)
            action_shape: (horizon_steps, action_dim)
        """
        self.max_size = max_size
        self.obs = torch.zeros(max_size, *cond_shape)
        self.actions = torch.zeros(max_size, *action_shape)
        self._size = 0
        self._ptr = 0

    def add(self, obs_batch, action_batch):
        """Add a batch of (obs, action) pairs. Circular FIFO write.

        Args:
            obs_batch: (N, cond_steps, obs_dim)
            action_batch: (N, horizon_steps, action_dim)
        """
        n = obs_batch.shape[0]
        if n == 0:
            return
        obs_cpu = obs_batch.detach().cpu()
        act_cpu = action_batch.detach().cpu()

        for i in range(n):
            self.obs[self._ptr] = obs_cpu[i]
            self.actions[self._ptr] = act_cpu[i]
            self._ptr = (self._ptr + 1) % self.max_size
            self._size = min(self._size + 1, self.max_size)

    def sample(self, batch_size, device):
        """Random sample transferred to GPU.

        Returns:
            obs: (batch_size, cond_steps, obs_dim) on device
            actions: (batch_size, horizon_steps, action_dim) on device
        """
        indices = torch.randint(0, self._size, (batch_size,))
        return self.obs[indices].to(device), self.actions[indices].to(device)

    @property
    def size(self):
        return self._size


@dataclass
class Args:
    # Pretrained checkpoint
    pretrain_checkpoint: str = "runs/dppo_pretrain/best.pt"
    demo_path: Optional[str] = None  # Offline demos for queue initialization

    # Environment
    env_id: str = "PegInsertionSide-v1"
    control_mode: str = "pd_joint_delta_pos"
    max_episode_steps: int = 200
    n_envs: int = 100
    sim_backend: str = "gpu"

    # Architecture (overridden from checkpoint)
    denoising_steps: int = 100
    horizon_steps: int = 16
    cond_steps: int = 2
    act_steps: int = 8

    # DDIM for fast rollout
    use_ddim: bool = False
    ddim_steps: int = 10

    # Rollout
    n_train_itr: int = 50
    n_steps: int = 100  # Decision steps per rollout
    gamma: float = 0.999  # For return computation / filtering

    # BC update
    bc_batch_size: int = 256
    bc_gradient_steps: int = 200
    bc_lr: float = 1e-4
    max_grad_norm: float = 1.0

    # Replay
    replay_max_size: int = 200000

    # Sampling noise
    min_sampling_denoising_std: float = 0.01

    # Eval
    eval_freq: int = 5
    num_eval_episodes: int = 100

    # Logging
    exp_name: Optional[str] = None
    seed: int = 0
    save_dir: str = "runs/dppo_filtered_bc"


def main():
    args = tyro.cli(Args)

    if args.exp_name is None:
        args.exp_name = (
            f"fbc_{args.env_id}_envs{args.n_envs}_steps{args.n_steps}"
            f"_bcsteps{args.bc_gradient_steps}"
        )

    run_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== Load pretrained checkpoint =====
    ckpt = torch.load(args.pretrain_checkpoint, map_location=device, weights_only=False)
    obs_dim = ckpt["obs_dim"]
    action_dim = ckpt["action_dim"]

    no_obs_norm = ckpt.get("no_obs_norm", False)
    no_action_norm = ckpt.get("no_action_norm", False)

    obs_min = obs_max = action_min = action_max = None
    if not no_obs_norm:
        obs_min = ckpt["obs_min"].to(device)
        obs_max = ckpt["obs_max"].to(device)
    if not no_action_norm:
        action_min = ckpt["action_min"].to(device)
        action_max = ckpt["action_max"].to(device)

    # Override architecture params from checkpoint
    pretrain_args = ckpt.get("args", {})
    args.denoising_steps = pretrain_args.get("denoising_steps", args.denoising_steps)
    args.horizon_steps = pretrain_args.get("horizon_steps", args.horizon_steps)
    args.cond_steps = pretrain_args.get("cond_steps", args.cond_steps)
    args.act_steps = pretrain_args.get("act_steps", args.act_steps)
    network_type = pretrain_args.get("network_type", "mlp")

    cond_dim = obs_dim * args.cond_steps
    act_offset = args.cond_steps - 1 if network_type == "unet" else 0

    print(f"Pretrained: {args.pretrain_checkpoint}")
    print(f"  network_type={network_type}, obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"  denoising_steps={args.denoising_steps}, horizon_steps={args.horizon_steps}")
    print(f"  cond_steps={args.cond_steps}, act_steps={args.act_steps}, act_offset={act_offset}")
    print(f"  no_obs_norm={no_obs_norm}, no_action_norm={no_action_norm}")

    # ===== Build DiffusionModel (no critic, no PPO) =====
    if network_type == "unet":
        actor = DiffusionUNet(
            action_dim=action_dim,
            horizon_steps=args.horizon_steps,
            cond_dim=cond_dim,
            diffusion_step_embed_dim=pretrain_args.get("diffusion_step_embed_dim", 64),
            down_dims=pretrain_args.get("unet_dims", [64, 128, 256]),
            n_groups=pretrain_args.get("n_groups", 8),
        )
    else:
        actor = DiffusionMLP(
            action_dim=action_dim,
            horizon_steps=args.horizon_steps,
            cond_dim=cond_dim,
            time_dim=pretrain_args.get("time_dim", 16),
            mlp_dims=pretrain_args.get("mlp_dims", [512, 512, 512]),
            activation_type=pretrain_args.get("activation_type", "Mish"),
            residual_style=pretrain_args.get("residual_style", True),
        )

    model = DiffusionModel(
        network=actor,
        horizon_steps=args.horizon_steps,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        denoising_steps=args.denoising_steps,
        denoised_clip_value=1.0,
        randn_clip_value=3,
        final_action_clip_value=1.0,
        predict_epsilon=True,
    )

    # Load pretrained weights (EMA preferred)
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"], strict=False)
    else:
        model.load_state_dict(ckpt["model"], strict=False)

    n_params = sum(p.numel() for p in model.network.parameters())
    print(f"Network params: {n_params:,}")

    optimizer = torch.optim.Adam(model.network.parameters(), lr=args.bc_lr)

    # ===== Normalization helpers =====
    def normalize_obs(obs):
        if no_obs_norm:
            return obs
        return (obs - obs_min) / (obs_max - obs_min + 1e-8) * 2.0 - 1.0

    def denormalize_actions(actions):
        if no_action_norm:
            return actions
        return (actions + 1.0) / 2.0 * (action_max - action_min) + action_min

    def normalize_actions(actions):
        if no_action_norm:
            return actions
        return (actions - action_min) / (action_max - action_min + 1e-8) * 2.0 - 1.0

    # ===== Initialize replay queue =====
    cond_shape = (args.cond_steps, obs_dim)
    action_shape = (args.horizon_steps, action_dim)
    queue = ReplayQueue(args.replay_max_size, cond_shape, action_shape)

    # Pre-fill queue from offline demos
    if args.demo_path:
        demo_path = os.path.expanduser(args.demo_path)
        dataset = DPPODataset(
            demo_path, args.horizon_steps, args.cond_steps,
            no_obs_norm=no_obs_norm, no_action_norm=no_action_norm,
        )
        print(f"Loading {len(dataset)} demo samples into replay queue...")
        # Batch-load all demo data
        for i in range(len(dataset)):
            item = dataset[i]
            queue.add(item["cond"]["state"].unsqueeze(0), item["actions"].unsqueeze(0))
        print(f"Queue initialized with {queue.size} demo samples")

    # ===== Create training environments =====
    use_gpu_env = args.sim_backend == "gpu"
    train_envs = make_train_envs(
        env_id=args.env_id,
        num_envs=args.n_envs,
        sim_backend=args.sim_backend,
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
    )

    # ===== Eval function =====
    @torch.no_grad()
    def evaluate_gpu_inline(n_rounds=3):
        model.eval()
        total_success = 0
        total_eps = 0
        for _ in range(n_rounds):
            obs_r, _ = train_envs.reset()
            obs_r = obs_r.float().to(device) if not isinstance(obs_r, torch.Tensor) else obs_r.float()
            obs_h = obs_r.unsqueeze(1).repeat(1, args.cond_steps, 1)
            for step in range(args.max_episode_steps // args.act_steps + 1):
                cond_eval = {"state": normalize_obs(obs_h)}
                ddim = args.ddim_steps if args.use_ddim else None
                samples_eval = model(cond_eval, deterministic=True, ddim_steps=ddim)
                ac_eval = denormalize_actions(samples_eval.trajectories)
                for a_idx in range(args.act_steps):
                    act_idx = act_offset + a_idx
                    action_eval = ac_eval[:, min(act_idx, ac_eval.shape[1] - 1)]
                    obs_new_eval, rew_eval, term_eval, trunc_eval, _ = train_envs.step(action_eval)
                    obs_new_eval = obs_new_eval.float()
                    total_success += (rew_eval.float() > 0.5).sum().item()
                    rm = term_eval | trunc_eval
                    if rm.any():
                        obs_h[rm] = obs_new_eval[rm].unsqueeze(1).repeat(1, args.cond_steps, 1)
                    obs_h[~rm] = torch.cat([obs_h[~rm, 1:], obs_new_eval[~rm].unsqueeze(1)], dim=1)
            total_eps += args.n_envs
        return total_success / max(total_eps, 1)

    # ===== Main loop =====
    n_decision_steps = args.n_steps
    best_sr = -1.0
    global_step = 0

    print(f"\nStarting filtered BC finetuning for {args.n_train_itr} iterations...")
    print(f"  n_envs={args.n_envs}, n_steps={n_decision_steps}, act_steps={args.act_steps}")
    print(f"  bc_batch_size={args.bc_batch_size}, bc_gradient_steps={args.bc_gradient_steps}")
    print(f"  gamma={args.gamma}, replay_max_size={args.replay_max_size}")
    print(f"  queue_size={queue.size} (from demos)")
    if args.use_ddim:
        print(f"  DDIM: ddim_steps={args.ddim_steps}")

    obs_raw, _ = train_envs.reset()
    if isinstance(obs_raw, np.ndarray):
        obs_raw = torch.from_numpy(obs_raw).float().to(device)
    else:
        obs_raw = obs_raw.float().to(device)
    obs_history = obs_raw.unsqueeze(1).repeat(1, args.cond_steps, 1)

    for iteration in range(1, args.n_train_itr + 1):
        t_start = time.time()

        # ===== 1. ROLLOUT =====
        model.eval()
        reward_trajs = []
        done_trajs = []
        obs_norm_trajs = []       # normalized obs for BC: (n_steps, n_envs, cond_steps, obs_dim)
        action_chunk_trajs = []   # normalized action chunks: (n_steps, n_envs, horizon_steps, action_dim)
        n_success_rollout = 0

        for step in range(n_decision_steps):
            obs_norm = normalize_obs(obs_history)
            cond = {"state": obs_norm}

            with torch.no_grad():
                ddim = args.ddim_steps if args.use_ddim else None
                samples = model(
                    cond, deterministic=False,
                    min_sampling_denoising_std=args.min_sampling_denoising_std,
                    ddim_steps=ddim,
                )
                action_chunk = samples.trajectories  # (n_envs, horizon_steps, action_dim) normalized

            obs_norm_trajs.append(obs_norm.clone())
            action_chunk_trajs.append(action_chunk.clone())

            action_chunk_denorm = denormalize_actions(action_chunk)

            step_reward = torch.zeros(args.n_envs, device=device)
            step_done = torch.zeros(args.n_envs, dtype=torch.bool, device=device)

            for a_idx in range(args.act_steps):
                act_idx = act_offset + a_idx
                if act_idx < action_chunk_denorm.shape[1]:
                    action = action_chunk_denorm[:, act_idx]
                else:
                    action = action_chunk_denorm[:, -1]

                if use_gpu_env:
                    obs_new, reward, terminated, truncated, info = train_envs.step(action)
                    obs_new = obs_new.float()
                    reward_t = reward.float()
                    term_t = terminated.bool()
                    trunc_t = truncated.bool()
                else:
                    action_np = action.cpu().numpy()
                    obs_new, reward, terminated, truncated, info = train_envs.step(action_np)
                    obs_new = torch.from_numpy(obs_new).float().to(device)
                    reward_t = torch.from_numpy(np.array(reward)).float().to(device)
                    term_t = torch.from_numpy(np.array(terminated)).bool().to(device)
                    trunc_t = torch.from_numpy(np.array(truncated)).bool().to(device)

                step_reward += reward_t * (~step_done).float()
                step_done = step_done | term_t | trunc_t
                n_success_rollout += (reward_t > 0.5).sum().item()

                reset_mask = term_t | trunc_t
                if reset_mask.any():
                    obs_history[reset_mask] = obs_new[reset_mask].unsqueeze(1).repeat(1, args.cond_steps, 1)
                obs_history[~reset_mask] = torch.cat(
                    [obs_history[~reset_mask, 1:], obs_new[~reset_mask].unsqueeze(1)], dim=1
                )
                global_step += args.n_envs

            reward_trajs.append(step_reward)
            done_trajs.append(step_done.float())

        rewards = torch.stack(reward_trajs)  # (n_steps, n_envs)
        dones = torch.stack(done_trajs)      # (n_steps, n_envs)
        obs_stacked = torch.stack(obs_norm_trajs)          # (n_steps, n_envs, cond_steps, obs_dim)
        action_stacked = torch.stack(action_chunk_trajs)   # (n_steps, n_envs, horizon_steps, action_dim)

        # ===== 2. COMPUTE RETURNS & FILTER =====
        returns = torch.zeros_like(rewards)
        running = torch.zeros(args.n_envs, device=device)
        for t in reversed(range(n_decision_steps)):
            running = rewards[t] + args.gamma * (1 - dones[t]) * running
            returns[t] = running

        # Filter: keep (obs, action_chunk) where return > 0
        success_mask = returns > 0  # (n_steps, n_envs)
        n_success_filtered = success_mask.sum().item()

        if n_success_filtered > 0:
            obs_success = obs_stacked[success_mask]       # (N_succ, cond_steps, obs_dim)
            act_success = action_stacked[success_mask]    # (N_succ, horizon_steps, action_dim)
            queue.add(obs_success, act_success)

        # ===== 3. BC UPDATE =====
        if queue.size >= args.bc_batch_size:
            model.train()
            total_loss = 0.0
            for step in range(args.bc_gradient_steps):
                obs_batch, action_batch = queue.sample(args.bc_batch_size, device)
                loss = model.loss(action_batch, {"state": obs_batch})
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.network.parameters(), args.max_grad_norm)
                optimizer.step()
                total_loss += loss.item()
            avg_bc_loss = total_loss / args.bc_gradient_steps
        else:
            avg_bc_loss = float("nan")
            print(f"  [WARN] Queue size {queue.size} < batch_size {args.bc_batch_size}, skipping BC update")

        # ===== 4. LOGGING =====
        avg_reward = rewards.sum(0).mean().item()
        t_elapsed = time.time() - t_start

        writer.add_scalar("train/bc_loss", avg_bc_loss, iteration)
        writer.add_scalar("train/n_success_filtered", n_success_filtered, iteration)
        writer.add_scalar("train/queue_size", queue.size, iteration)
        writer.add_scalar("train/rollout_successes", n_success_rollout, iteration)
        writer.add_scalar("train/reward", avg_reward, iteration)
        writer.add_scalar("train/global_step", global_step, iteration)

        if iteration % 5 == 0 or iteration <= 5:
            print(
                f"Iter {iteration}/{args.n_train_itr} | "
                f"r={avg_reward:.3f} | succ_rollout={n_success_rollout} | "
                f"succ_filtered={n_success_filtered} | queue={queue.size} | "
                f"bc_loss={avg_bc_loss:.6f} | time={t_elapsed:.1f}s"
            )

        # ===== 5. EVALUATE =====
        if iteration == 1 or iteration % args.eval_freq == 0:
            sr_once = evaluate_gpu_inline(n_rounds=3)
            writer.add_scalar("eval/success_once", sr_once, iteration)
            print(f"  EVAL @ iter {iteration}: gpu_sr={sr_once:.3f}")

            # Reset training env state after eval
            obs_raw_r, _ = train_envs.reset()
            if isinstance(obs_raw_r, np.ndarray):
                obs_raw_r = torch.from_numpy(obs_raw_r).float().to(device)
            else:
                obs_raw_r = obs_raw_r.float().to(device)
            obs_history[:] = obs_raw_r.unsqueeze(1).repeat(1, args.cond_steps, 1)

            if sr_once > best_sr:
                best_sr = sr_once
                save_ckpt = {
                    "model": model.state_dict(),
                    "obs_min": obs_min,
                    "obs_max": obs_max,
                    "action_min": action_min,
                    "action_max": action_max,
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "args": vars(args),
                    "iteration": iteration,
                    "no_obs_norm": no_obs_norm,
                    "no_action_norm": no_action_norm,
                    "network_type": network_type,
                    "pretrain_args": pretrain_args,
                }
                torch.save(save_ckpt, os.path.join(run_dir, "best.pt"))
                print(f"  Saved best checkpoint (sr_once={best_sr:.3f})")

    # Save final
    save_ckpt = {
        "model": model.state_dict(),
        "obs_min": obs_min,
        "obs_max": obs_max,
        "action_min": action_min,
        "action_max": action_max,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "args": vars(args),
        "iteration": args.n_train_itr,
        "no_obs_norm": no_obs_norm,
        "no_action_norm": no_action_norm,
        "network_type": network_type,
        "pretrain_args": pretrain_args,
    }
    torch.save(save_ckpt, os.path.join(run_dir, "final.pt"))
    print(f"Finetuning complete. Best sr_once={best_sr:.3f}")

    train_envs.close()
    writer.close()


if __name__ == "__main__":
    main()
