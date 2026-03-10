"""
Gaussian Policy IL Pretrain: Train Gaussian policy on ManiSkill demos.

Usage:
    python -m MultiGaussian.pretrain --env_id PickCube-v1 --total_iters 100000
"""

import os
import copy
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, BatchSampler, Subset
from dataclasses import dataclass, field
from typing import Optional, List
from torch.utils.tensorboard import SummaryWriter

import tyro

from MultiGaussian.models.gaussian import GaussianPolicy
from MultiGaussian.eval_cpu import evaluate_gaussian_cpu
from DPPO.dataset import DPPODataset

from diffusers.training_utils import EMAModel as DiffusersEMA


@dataclass
class Args:
    env_id: str = "PickCube-v1"
    demo_path: str = "~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5"
    num_demos: Optional[int] = None
    control_mode: str = "pd_ee_delta_pos"
    max_episode_steps: int = 100

    # Network
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "ReLU"
    sigma_init: float = -1.5  # Initial log_std (not trained with MSE, used at finetuning)

    # Action chunking (ACT-style)
    cond_steps: int = 1   # obs history length
    horizon_steps: int = 1  # predict this many future actions
    act_steps: int = 1    # execute this many actions per prediction

    # Training
    total_iters: int = 100_000
    batch_size: int = 1024
    lr: float = 1e-4
    weight_decay: float = 1e-6
    max_grad_norm: float = 0.0  # 0 = no clipping
    lr_warmup: int = 500
    use_cosine_lr: bool = True
    num_workers: int = 0
    torch_deterministic: bool = True
    loss_type: str = "mse"  # "mse" or "nll"

    # Eval
    eval_freq: int = 5000
    num_eval_episodes: int = 100
    num_eval_envs: int = 10  # CPU eval, fewer envs
    log_freq: int = 200

    # Normalization
    no_obs_norm: bool = False
    no_action_norm: bool = False

    # Logging
    exp_name: Optional[str] = None
    seed: int = 0
    save_dir: str = "runs/gaussian_pretrain"

    # Augmentation
    obs_noise_std: float = 0.0
    zero_qvel: bool = False

    # Resume
    resume: Optional[str] = None


def main():
    args = tyro.cli(Args)
    args.demo_path = os.path.expanduser(args.demo_path)

    if args.exp_name is None:
        args.exp_name = f"gaussian_{args.env_id}_{'x'.join(map(str, args.hidden_dims))}"

    run_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Validate demo control_mode matches args
    import json
    json_path = args.demo_path.replace(".h5", ".json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            demo_info = json.load(f)
        env_kwargs = demo_info.get("env_info", {}).get("env_kwargs", {})
        demo_cm = env_kwargs.get("control_mode")
        if demo_cm is not None and demo_cm != args.control_mode:
            raise ValueError(
                f"Demo control_mode='{demo_cm}' != args.control_mode='{args.control_mode}'. "
                f"Fix --control_mode or use the correct demo file."
            )

    assert args.act_steps <= args.horizon_steps, \
        f"act_steps ({args.act_steps}) must be <= horizon_steps ({args.horizon_steps})"

    dataset = DPPODataset(
        data_path=args.demo_path,
        horizon_steps=args.horizon_steps,
        cond_steps=args.cond_steps,
        num_traj=args.num_demos,
        no_obs_norm=args.no_obs_norm,
        no_action_norm=args.no_action_norm,
    )

    # Train/val split by trajectory (last 10% trajs held out)
    val_frac = 0.1
    n_val_traj = max(1, int(dataset.num_traj * val_frac))
    n_train_traj = dataset.num_traj - n_val_traj
    train_indices = [i for i, (tj, _) in enumerate(dataset.slices) if tj < n_train_traj]
    val_indices = [i for i, (tj, _) in enumerate(dataset.slices) if tj >= n_train_traj]

    obs_dim = dataset.obs_dim
    action_dim = dataset.action_dim

    print(f"obs_dim={obs_dim}, action_dim={action_dim}, "
          f"cond={args.cond_steps}, horizon={args.horizon_steps}, act={args.act_steps}")
    print(f"Dataset: {len(dataset)} samples, train={len(train_indices)}, val={len(val_indices)} "
          f"({n_train_traj}/{n_val_traj} trajs)")

    # Pre-load val set on GPU
    val_obs_list, val_act_list = [], []
    for idx in val_indices:
        sample = dataset[idx]
        val_obs_list.append(sample["cond"]["state"])   # (cond_steps, obs_dim)
        val_act_list.append(sample["actions"])          # (horizon_steps, action_dim)
    val_obs = torch.stack(val_obs_list).to(device)     # (N, cond_steps, obs_dim)
    val_act = torch.stack(val_act_list).to(device)     # (N, horizon_steps, action_dim)
    if args.cond_steps == 1:
        val_obs = val_obs.squeeze(1)
    if args.horizon_steps == 1:
        val_act = val_act.squeeze(1)
    if args.zero_qvel:
        val_obs[..., 9:18] = 0.0
    del val_obs_list, val_act_list

    # Dataloader on train subset only
    from diffusion_policy.utils import IterationBasedBatchSampler
    train_subset = Subset(dataset, train_indices)
    sampler = RandomSampler(train_subset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    dataloader = DataLoader(train_subset, batch_sampler=batch_sampler, num_workers=args.num_workers,
                            pin_memory=(args.num_workers > 0))

    # Build model
    model = GaussianPolicy(
        input_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=args.hidden_dims,
        activation=args.activation,
        sigma_init=args.sigma_init,
        cond_steps=args.cond_steps,
        horizon_steps=args.horizon_steps,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # LR scheduler
    if args.use_cosine_lr:
        from diffusers.optimization import get_scheduler
        lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup,
            num_training_steps=args.total_iters,
        )
    else:
        lr_scheduler = None

    # EMA
    ema = DiffusersEMA(parameters=model.parameters(), power=0.75)
    ema_model = copy.deepcopy(model)

    # Resume
    start_iter = 0
    _lr_scheduler_restored = False
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model"])
        ema_model.load_state_dict(resume_ckpt["ema"])
        start_iter = resume_ckpt.get("step", 0)
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        if "ema_state" in resume_ckpt:
            ema.load_state_dict(resume_ckpt["ema_state"])
            ema.shadow_params = [p.to(device) for p in ema.shadow_params]
        else:
            ema.shadow_params = [p.clone() for p in ema_model.parameters()]
        if "lr_scheduler" in resume_ckpt and lr_scheduler is not None:
            lr_scheduler.load_state_dict(resume_ckpt["lr_scheduler"])
            _lr_scheduler_restored = True
        # Warn if norm stats differ from checkpoint
        for key in ("obs_min", "obs_max", "action_min", "action_max"):
            if key in resume_ckpt:
                ckpt_val = resume_ckpt[key]
                ds_val = getattr(dataset, key)
                if not torch.allclose(ckpt_val, ds_val, atol=1e-5):
                    print(f"  WARNING: {key} differs between checkpoint and current dataset! "
                          f"Norm mismatch will corrupt training/eval.")
        print(f"Resumed from {args.resume} at iter {start_iter}")
        del resume_ckpt

    # Norm stats on GPU for eval
    obs_min = dataset.obs_min.to(device)
    obs_max = dataset.obs_max.to(device)
    action_min = dataset.action_min.to(device)
    action_max = dataset.action_max.to(device)

    # Precompute per-dim obs std for noise augmentation (in the same space as training obs)
    if args.obs_noise_std > 0:
        if not args.no_obs_norm:
            # Dataset returns obs normalized to [-1, 1] via min-max.
            # std(x_norm) = std(x) * 2 / (obs_max - obs_min)
            obs_range = (dataset.obs_max - dataset.obs_min).clamp(min=1e-8)
            obs_std_dev = (dataset.obs_std * 2.0 / obs_range).to(device)
        else:
            obs_std_dev = dataset.obs_std.to(device)
        print(f"Obs noise augmentation: {args.obs_noise_std} * per-dim std (normalized={not args.no_obs_norm})")

    best_sr = -1.0
    t_start = time.time()

    # Fast-forward LR scheduler if resuming
    if start_iter > 0 and lr_scheduler is not None and not _lr_scheduler_restored:
        for _ in range(start_iter):
            lr_scheduler.step()

    print(f"Starting training from iter {start_iter + 1}...")
    for iteration, batch in enumerate(dataloader, 1):
        if iteration <= start_iter:
            continue
        model.train()

        actions = batch["actions"].to(device)       # (B, horizon_steps, action_dim)
        obs = batch["cond"]["state"].to(device)     # (B, cond_steps, obs_dim)
        if args.horizon_steps == 1:
            actions = actions.squeeze(1)
        if args.cond_steps == 1:
            obs = obs.squeeze(1)

        if args.zero_qvel:
            obs[..., 9:18] = 0.0

        if args.obs_noise_std > 0:
            noise = torch.randn_like(obs) * obs_std_dev * args.obs_noise_std
            obs = obs + noise

        mean, std = model(obs)

        if args.loss_type == "nll":
            dist = torch.distributions.Normal(mean, std)
            loss = -dist.log_prob(actions).sum(dim=-1).mean()
        else:
            loss = nn.functional.mse_loss(mean, actions)

        optimizer.zero_grad()
        loss.backward()
        if args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        ema.step(model.parameters())

        writer.add_scalar("train/loss", loss.item(), iteration)

        # Val loss at log_freq
        if iteration % args.log_freq == 0:
            with torch.no_grad():
                val_mean, _ = model(val_obs)
                val_loss = nn.functional.mse_loss(val_mean, val_act).item()
            writer.add_scalar("val/loss", val_loss, iteration)

        if iteration % args.log_freq == 0:
            elapsed = time.time() - t_start
            print(f"Iter {iteration}/{args.total_iters}, loss={loss.item():.6f}, val_loss={val_loss:.6f}, "
                  f"time={elapsed:.0f}s ({iteration/elapsed:.1f} it/s)", flush=True)

        # Evaluate
        if iteration % args.eval_freq == 0 or iteration == args.total_iters:
            ema.copy_to(ema_model.parameters())
            ema_model.eval()

            eval_metrics = evaluate_gaussian_cpu(
                model=ema_model,
                device=device,
                n_episodes=args.num_eval_episodes,
                env_id=args.env_id,
                control_mode=args.control_mode,
                max_episode_steps=args.max_episode_steps,
                num_envs=args.num_eval_envs,
                obs_min=obs_min,
                obs_max=obs_max,
                action_min=action_min,
                action_max=action_max,
                no_obs_norm=args.no_obs_norm,
                no_action_norm=args.no_action_norm,
                zero_qvel=args.zero_qvel,
                cond_steps=args.cond_steps,
                horizon_steps=args.horizon_steps,
                act_steps=args.act_steps,
            )

            sr = eval_metrics.get("success_at_end", 0.0)
            sr_once = eval_metrics.get("success_once", 0.0)
            writer.add_scalar("eval/success_at_end", sr, iteration)
            writer.add_scalar("eval/success_once", sr_once, iteration)
            print(f"  Eval [cpu] @ iter {iteration}: success_at_end={sr:.3f}, success_once={sr_once:.3f}")

            _save_ckpt(model, ema_model, dataset, obs_dim, action_dim, args, iteration,
                       os.path.join(run_dir, f"ckpt_{iteration}.pt"),
                       optimizer=optimizer, ema_state=ema, lr_scheduler=lr_scheduler)

            if sr_once > best_sr:
                best_sr = sr_once
                _save_ckpt(model, ema_model, dataset, obs_dim, action_dim, args, iteration,
                           os.path.join(run_dir, "best.pt"),
                           optimizer=optimizer, ema_state=ema, lr_scheduler=lr_scheduler)
                print(f"  Saved best checkpoint (sr_once={best_sr:.3f})")

    # Save final
    ema.copy_to(ema_model.parameters())
    _save_ckpt(model, ema_model, dataset, obs_dim, action_dim, args, args.total_iters,
               os.path.join(run_dir, "final.pt"),
               optimizer=optimizer, ema_state=ema, lr_scheduler=lr_scheduler)
    elapsed = time.time() - t_start
    print(f"Training complete in {elapsed:.0f}s. Best sr_once={best_sr:.3f}")
    writer.close()


def _save_ckpt(model, ema_model, dataset, obs_dim, action_dim, args, step, path,
               optimizer=None, ema_state=None, lr_scheduler=None):
    ckpt = {
        "model": {k: v.cpu() for k, v in model.state_dict().items()},
        "ema": {k: v.cpu() for k, v in ema_model.state_dict().items()},
        "obs_min": dataset.obs_min,
        "obs_max": dataset.obs_max,
        "action_min": dataset.action_min,
        "action_max": dataset.action_max,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "args": vars(args),
        "step": step,
        "no_obs_norm": args.no_obs_norm,
        "no_action_norm": args.no_action_norm,
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if ema_state is not None:
        ckpt["ema_state"] = ema_state.state_dict()
    if lr_scheduler is not None:
        ckpt["lr_scheduler"] = lr_scheduler.state_dict()
    torch.save(ckpt, path)


if __name__ == "__main__":
    main()
