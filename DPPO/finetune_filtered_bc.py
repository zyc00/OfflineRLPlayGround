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
import csv
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
    demo_ratio: float = 0.0  # Fraction of BC batch from original demos (0=all online, 0.5=50/50)

    # Sampling noise
    min_sampling_denoising_std: float = 0.01

    # Augmentation
    zero_qvel: bool = False  # Zero out qvel dims (9:18) — eliminates GPU/CPU eval gap

    # Cold-start mode
    cold_start: bool = False  # If True, reinitialize model from pretrained weights before each BC update

    # Eval
    eval_freq: int = 5
    num_eval_episodes: int = 100

    # Coverage analysis at end
    coverage_mc_samples: int = 8
    coverage_n_states: int = 200

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

    # Validate control_mode matches pretrained checkpoint
    ckpt_control_mode = pretrain_args.get("control_mode")
    if ckpt_control_mode is not None and ckpt_control_mode != args.control_mode:
        raise ValueError(
            f"control_mode mismatch: checkpoint was trained with '{ckpt_control_mode}' "
            f"but finetune is configured with '{args.control_mode}'. "
            f"Pass --control_mode {ckpt_control_mode} to fix."
        )

    # Inherit zero_qvel from pretrain checkpoint if not explicitly set
    if pretrain_args.get("zero_qvel", False) and not args.zero_qvel:
        args.zero_qvel = True
        print("  Inherited zero_qvel=True from pretrain checkpoint")

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

    # Save pretrained weights for cold_start reinit
    if args.cold_start:
        import copy
        pretrained_state_dict = copy.deepcopy(model.state_dict())
        print("  Cold-start mode: will reinitialize from pretrained weights each iteration")

    optimizer = torch.optim.Adam(model.network.parameters(), lr=args.bc_lr)

    # ===== Normalization helpers =====
    def normalize_obs(obs):
        if args.zero_qvel:
            obs = obs.clone()
            obs[..., 9:18] = 0.0
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

    # ===== Initialize replay queue + demo buffer =====
    cond_shape = (args.cond_steps, obs_dim)
    action_shape = (args.horizon_steps, action_dim)
    queue = ReplayQueue(args.replay_max_size, cond_shape, action_shape)

    # Load demos into a separate fixed buffer (never overwritten by online data)
    demo_buffer = None
    if args.demo_path:
        demo_path = os.path.expanduser(args.demo_path)
        dataset = DPPODataset(
            demo_path, args.horizon_steps, args.cond_steps,
            no_obs_norm=no_obs_norm, no_action_norm=no_action_norm,
        )
        n_demo = len(dataset)
        demo_obs = torch.zeros(n_demo, *cond_shape)
        demo_act = torch.zeros(n_demo, *action_shape)
        for i in range(n_demo):
            item = dataset[i]
            demo_obs[i] = item["cond"]["state"]
            demo_act[i] = item["actions"]
        demo_buffer = (demo_obs, demo_act)
        print(f"Loaded {n_demo} demo samples into fixed demo buffer")
        if args.demo_ratio == 0.0:
            # No mixing requested — seed queue with demos (old behavior)
            for i in range(n_demo):
                queue.add(demo_obs[i:i+1], demo_act[i:i+1])
            print(f"  Seeded queue with {queue.size} demo samples (demo_ratio=0)")
        else:
            print(f"  Demo mixing enabled: {args.demo_ratio:.0%} demo + {1-args.demo_ratio:.0%} online")

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
    ddim_eval = args.ddim_steps if args.use_ddim else None

    @torch.no_grad()
    def evaluate_gpu_inline():
        """Eval using training env: reset + deterministic rollout, track first episode only.

        Runs ceil(num_eval_episodes / n_envs) rounds so total episodes >= num_eval_episodes.
        """
        model.eval()
        n_rounds = max(1, (args.num_eval_episodes + args.n_envs - 1) // args.n_envs)
        total_success = 0
        total_eps = 0
        for _ in range(n_rounds):
            obs_r, _ = train_envs.reset()
            if isinstance(obs_r, np.ndarray):
                obs_r = torch.from_numpy(obs_r).float().to(device)
            else:
                obs_r = obs_r.float().to(device)
            obs_h = obs_r.unsqueeze(1).repeat(1, args.cond_steps, 1)
            success_once = torch.zeros(args.n_envs, dtype=torch.bool, device=device)
            ep_done = torch.zeros(args.n_envs, dtype=torch.bool, device=device)
            for step in range(args.max_episode_steps // args.act_steps + 1):
                cond_eval = {"state": normalize_obs(obs_h)}
                samples_eval = model(cond_eval, deterministic=True, ddim_steps=ddim_eval)
                ac_eval = denormalize_actions(samples_eval.trajectories)
                for a_idx in range(args.act_steps):
                    act_idx = act_offset + a_idx
                    action_eval = ac_eval[:, min(act_idx, ac_eval.shape[1] - 1)]
                    if use_gpu_env:
                        obs_new_eval, rew_eval, term_eval, trunc_eval, _ = train_envs.step(action_eval)
                        obs_new_eval = obs_new_eval.float()
                        rew_eval = rew_eval.float()
                        term_eval = term_eval.bool()
                        trunc_eval = trunc_eval.bool()
                    else:
                        obs_new_eval, rew_eval, term_eval, trunc_eval, _ = train_envs.step(action_eval.cpu().numpy())
                        obs_new_eval = torch.from_numpy(np.array(obs_new_eval)).float().to(device)
                        rew_eval = torch.from_numpy(np.array(rew_eval)).float().to(device)
                        term_eval = torch.from_numpy(np.array(term_eval)).bool().to(device)
                        trunc_eval = torch.from_numpy(np.array(trunc_eval)).bool().to(device)
                    # Only track success for first episode per env slot
                    success_once |= (rew_eval > 0.5).bool() & ~ep_done
                    rm = term_eval | trunc_eval
                    ep_done |= rm
                    if rm.any():
                        obs_h[rm] = obs_new_eval[rm].unsqueeze(1).repeat(1, args.cond_steps, 1)
                    obs_h[~rm] = torch.cat([obs_h[~rm, 1:], obs_new_eval[~rm].unsqueeze(1)], dim=1)
                if ep_done.all():
                    break
            total_success += success_once.sum().item()
            total_eps += args.n_envs
        return total_success / max(total_eps, 1)

    # ===== Main loop =====
    n_decision_steps = args.n_steps
    best_sr = -1.0
    global_step = 0

    # ===== CSV logging =====
    csv_path = os.path.join(run_dir, "results.csv")
    csv_fields = [
        "iteration", "eval_sr", "rollout_success_rate", "n_filtered",
        "queue_size", "bc_loss", "total_trajs", "time",
    ]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()
    csv_file.flush()
    total_trajs_collected = 0  # cumulative successful trajectories added to queue

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
        n_episodes_rollout = 0

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

                # Zero out actions for envs that already terminated this decision step
                if step_done.any():
                    action = action.clone()
                    action[step_done] = 0.0

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

                newly_done = (term_t | trunc_t) & ~step_done
                step_reward += reward_t * (~step_done).float()
                n_episodes_rollout += newly_done.sum().item()
                n_success_rollout += ((reward_t > 0.5) & newly_done).sum().item()
                step_done = step_done | term_t | trunc_t

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

        total_trajs_collected += n_success_filtered

        # ===== 3. BC UPDATE =====
        use_demo_mix = args.demo_ratio > 0 and demo_buffer is not None
        n_demo_per_batch = int(args.bc_batch_size * args.demo_ratio) if use_demo_mix else 0
        n_online_per_batch = args.bc_batch_size - n_demo_per_batch
        min_online_needed = max(n_online_per_batch, 1)

        if queue.size >= min_online_needed or (use_demo_mix and n_online_per_batch == 0):
            # Cold-start: reinitialize model from pretrained weights before training
            if args.cold_start:
                model.load_state_dict(pretrained_state_dict)
                optimizer = torch.optim.Adam(model.network.parameters(), lr=args.bc_lr)
            model.train()
            total_loss = 0.0
            for step in range(args.bc_gradient_steps):
                # Mixed sampling: demo_ratio from demos, rest from online queue
                if use_demo_mix and n_demo_per_batch > 0:
                    demo_idx = torch.randint(0, demo_buffer[0].shape[0], (n_demo_per_batch,))
                    demo_obs_b = demo_buffer[0][demo_idx].to(device)
                    demo_act_b = demo_buffer[1][demo_idx].to(device)
                    if n_online_per_batch > 0 and queue.size > 0:
                        online_obs_b, online_act_b = queue.sample(n_online_per_batch, device)
                        obs_batch = torch.cat([demo_obs_b, online_obs_b], dim=0)
                        action_batch = torch.cat([demo_act_b, online_act_b], dim=0)
                    else:
                        obs_batch, action_batch = demo_obs_b, demo_act_b
                else:
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
        # Rollout SR: successes / completed episodes
        rollout_sr = n_success_rollout / max(n_episodes_rollout, 1)

        writer.add_scalar("train/bc_loss", avg_bc_loss, iteration)
        writer.add_scalar("train/n_success_filtered", n_success_filtered, iteration)
        writer.add_scalar("train/queue_size", queue.size, iteration)
        writer.add_scalar("train/rollout_ep_successes", n_success_rollout, iteration)
        writer.add_scalar("train/rollout_sr", rollout_sr, iteration)
        writer.add_scalar("train/reward", avg_reward, iteration)
        writer.add_scalar("train/global_step", global_step, iteration)
        writer.add_scalar("train/total_trajs", total_trajs_collected, iteration)

        if iteration % 5 == 0 or iteration <= 5:
            print(
                f"Iter {iteration}/{args.n_train_itr} | "
                f"r={avg_reward:.3f} | rollout_sr={rollout_sr:.3f} | "
                f"succ_filtered={n_success_filtered} | queue={queue.size} | "
                f"bc_loss={avg_bc_loss:.6f} | time={t_elapsed:.1f}s"
            )

        # ===== 5. EVALUATE =====
        eval_sr = float("nan")
        if iteration == 1 or iteration % args.eval_freq == 0:
            sr_once = evaluate_gpu_inline()
            eval_sr = sr_once
            writer.add_scalar("eval/success_once", sr_once, iteration)
            print(f"  EVAL @ iter {iteration}: gpu_sr={sr_once:.3f}")

            # Reset training env state after eval (eval reuses train envs)
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
                    "zero_qvel": args.zero_qvel,
                }
                torch.save(save_ckpt, os.path.join(run_dir, "best.pt"))
                print(f"  Saved best checkpoint (sr_once={best_sr:.3f})")

        # Write CSV row every iteration
        csv_writer.writerow({
            "iteration": iteration,
            "eval_sr": f"{eval_sr:.4f}" if not np.isnan(eval_sr) else "",
            "rollout_success_rate": f"{rollout_sr:.4f}",
            "n_filtered": n_success_filtered,
            "queue_size": queue.size,
            "bc_loss": f"{avg_bc_loss:.6f}" if not np.isnan(avg_bc_loss) else "",
            "total_trajs": total_trajs_collected,
            "time": f"{t_elapsed:.1f}",
        })
        csv_file.flush()

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
        "zero_qvel": args.zero_qvel,
    }
    torch.save(save_ckpt, os.path.join(run_dir, "final.pt"))
    csv_file.close()
    print(f"Finetuning complete. Best sr_once={best_sr:.3f}")
    print(f"Results saved to {csv_path}")

    # ===== 6. SUMMARY PLOTS =====
    try:
        _generate_plots(csv_path, run_dir)
    except Exception as e:
        print(f"Warning: plot generation failed: {e}")

    # ===== 7. COVERAGE ANALYSIS =====
    if args.coverage_n_states > 0 and args.coverage_mc_samples > 0:
        try:
            _coverage_analysis(
                model, train_envs, args, normalize_obs, denormalize_actions,
                act_offset, use_gpu_env, device, run_dir,
            )
        except Exception as e:
            print(f"Warning: coverage analysis failed: {e}")

    train_envs.close()
    writer.close()


def _generate_plots(csv_path, run_dir):
    """Generate summary plots from CSV results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k not in data:
                    data[k] = []
                data[k].append(v)

    iterations = [int(x) for x in data["iteration"]]
    rollout_sr = [float(x) for x in data["rollout_success_rate"]]

    # Parse eval_sr (may have empty entries for non-eval iterations)
    eval_iters = []
    eval_srs = []
    for it, sr in zip(iterations, data["eval_sr"]):
        if sr:
            eval_iters.append(it)
            eval_srs.append(float(sr))

    total_trajs = [int(x) for x in data["total_trajs"]]

    # Plot 1: SR vs iteration
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(iterations, rollout_sr, "b-", alpha=0.5, label="Rollout SR")
    if eval_iters:
        ax.plot(eval_iters, eval_srs, "ro-", markersize=4, label="Eval SR")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate vs Iteration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Plot 2: Eval SR vs total trajectories collected
    ax = axes[1]
    if eval_iters:
        eval_total_trajs = []
        for it in eval_iters:
            idx = iterations.index(it)
            eval_total_trajs.append(total_trajs[idx])
        ax.plot(eval_total_trajs, eval_srs, "ro-", markersize=4)
    ax.set_xlabel("Total Filtered Trajectories Collected")
    ax.set_ylabel("Eval Success Rate")
    ax.set_title("Eval SR vs Total Trajectories")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plot_path = os.path.join(run_dir, "sr_vs_iteration.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot: {plot_path}")

    # Plot 3: Rollout stats
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    n_filtered = [int(x) for x in data["n_filtered"]]
    ax.plot(iterations, n_filtered, "g-")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("N Filtered (success samples)")
    ax.set_title("Filtered Samples per Iteration")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    queue_size = [int(x) for x in data["queue_size"]]
    ax.plot(iterations, queue_size, "m-")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Queue Size")
    ax.set_title("Replay Queue Size")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(run_dir, "rollout_stats.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot: {plot_path}")


@torch.no_grad()
def _coverage_analysis(model, envs, args, normalize_obs, denormalize_actions,
                       act_offset, use_gpu_env, device, run_dir):
    """Aggregate SR analysis of final policy over many episodes.

    NOTE: Each episode uses a fresh random initial state. This reports overall SR,
    NOT per-state P(success|s0). For true per-state coverage analysis, use
    dp_p_success_cpu.py which can fix initial states via set_state on CPU envs.
    """
    n_episodes = args.coverage_n_states * args.coverage_mc_samples
    n_envs = args.n_envs
    n_rounds = max(1, (n_episodes + n_envs - 1) // n_envs)
    ddim = args.ddim_steps if args.use_ddim else None

    print(f"\n===== Final SR Analysis: {n_rounds} rounds × {n_envs} envs = {n_rounds * n_envs} episodes =====")

    model.eval()
    all_successes = []

    for r in range(n_rounds):
        obs_r, _ = envs.reset()
        if isinstance(obs_r, np.ndarray):
            obs_r = torch.from_numpy(obs_r).float().to(device)
        else:
            obs_r = obs_r.float().to(device)
        obs_h = obs_r.unsqueeze(1).repeat(1, args.cond_steps, 1)
        success_once = torch.zeros(n_envs, dtype=torch.bool, device=device)
        ep_done = torch.zeros(n_envs, dtype=torch.bool, device=device)

        for step in range(args.max_episode_steps // args.act_steps + 1):
            cond = {"state": normalize_obs(obs_h)}
            samples = model(
                cond, deterministic=True, ddim_steps=ddim,
            )
            ac = denormalize_actions(samples.trajectories)
            for a_idx in range(args.act_steps):
                act_idx = act_offset + a_idx
                action = ac[:, min(act_idx, ac.shape[1] - 1)]
                if ep_done.any():
                    action = action.clone()
                    action[ep_done] = 0.0
                if use_gpu_env:
                    obs_new, rew, term, trunc, _ = envs.step(action)
                    obs_new = obs_new.float()
                    rew = rew.float()
                    term = term.bool()
                    trunc = trunc.bool()
                else:
                    obs_new, rew, term, trunc, _ = envs.step(action.cpu().numpy())
                    obs_new = torch.from_numpy(np.array(obs_new)).float().to(device)
                    rew = torch.from_numpy(np.array(rew)).float().to(device)
                    term = torch.from_numpy(np.array(term)).bool().to(device)
                    trunc = torch.from_numpy(np.array(trunc)).bool().to(device)
                success_once |= (rew > 0.5).bool() & ~ep_done
                rm = term | trunc
                ep_done |= rm
                if rm.any():
                    obs_h[rm] = obs_new[rm].unsqueeze(1).repeat(1, args.cond_steps, 1)
                obs_h[~rm] = torch.cat([obs_h[~rm, 1:], obs_new[~rm].unsqueeze(1)], dim=1)
            if ep_done.all():
                break

        all_successes.extend(success_once.cpu().numpy().tolist())

    all_successes = np.array(all_successes)
    overall_sr = all_successes.mean()
    n_total = len(all_successes)
    # 95% CI using normal approximation
    se = np.sqrt(overall_sr * (1 - overall_sr) / max(n_total, 1))

    print(f"\nFinal SR: {overall_sr:.3f} +/- {1.96*se:.3f} (95% CI, {n_total} episodes)")
    print(f"  Successes: {int(all_successes.sum())} / {n_total}")

    npz_path = os.path.join(run_dir, "final_sr.npz")
    np.savez(
        npz_path,
        successes=all_successes,
        overall_sr=overall_sr,
        se=se,
        n_episodes=n_total,
    )
    print(f"Saved to {npz_path}")
    print("  NOTE: For per-state P(success|s0) coverage, use dp_p_success_cpu.py")


if __name__ == "__main__":
    main()
