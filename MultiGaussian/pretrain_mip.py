"""
MIP (Minimal Iterative Policy) IL Pretrain from "Much Ado About Noising" (2512.01809).

Usage:
    python -m MultiGaussian.pretrain_mip --env_id PickCube-v1 --total_iters 100000
"""

import os
import copy
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
from dataclasses import dataclass, field
from typing import Optional, List
from torch.utils.tensorboard import SummaryWriter

import tyro

from MultiGaussian.models.multi_gaussian import MIPPolicy
from MultiGaussian.eval_mip_cpu import evaluate_mip_cpu
from DPPO.dataset import DPPODataset

from diffusers.training_utils import EMAModel as DiffusersEMA


@dataclass
class Args:
    env_id: str = "PickCube-v1"
    demo_path: str = "~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5"
    num_demos: Optional[int] = None
    control_mode: str = "pd_ee_delta_pos"
    max_episode_steps: int = 100

    # Network (official defaults: 512 dim, 6 layers, dropout=0.1)
    network_type: str = "mlp"  # "mlp" or "unet"
    emb_dim: int = 512
    n_layers: int = 6
    dropout: float = 0.0

    # MIP-specific
    t_star: float = 0.9
    predict_epsilon: bool = False  # epsilon-prediction (diffusion-style) vs x-prediction (original)
    mip_k: int = 2  # Number of cascade steps (2=original MIP, >2=MIP-K)

    # Action chunking
    cond_steps: int = 1
    horizon_steps: int = 1
    act_steps: int = 1

    # Training
    total_iters: int = 100_000
    batch_size: int = 1024
    lr: float = 1e-4
    weight_decay: float = 1e-6
    max_grad_norm: float = 0.0
    lr_warmup: int = 500
    use_cosine_lr: bool = True
    num_workers: int = 0
    torch_deterministic: bool = True

    # Eval
    eval_freq: int = 5000
    num_eval_episodes: int = 100
    num_eval_envs: int = 10
    log_freq: int = 200

    # Normalization
    no_obs_norm: bool = False
    no_action_norm: bool = False

    # Logging
    exp_name: Optional[str] = None
    seed: int = 0
    save_dir: str = "runs/mip_pretrain"

    # Augmentation
    obs_noise_std: float = 0.0
    zero_qvel: bool = False

    # Resume
    resume: Optional[str] = None


def main():
    args = tyro.cli(Args)
    args.demo_path = os.path.expanduser(args.demo_path)

    if args.exp_name is None:
        args.exp_name = f"mip_{args.env_id}_{args.emb_dim}x{args.n_layers}"

    run_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Validate demo control_mode
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

    # Train/val split by trajectory
    val_frac = 0.1
    n_val_traj = max(1, int(dataset.num_traj * val_frac))
    n_train_traj = dataset.num_traj - n_val_traj
    train_indices = [i for i, (tj, _) in enumerate(dataset.slices) if tj < n_train_traj]
    val_indices = [i for i, (tj, _) in enumerate(dataset.slices) if tj >= n_train_traj]

    obs_dim = dataset.obs_dim
    action_dim = dataset.action_dim

    print(f"obs_dim={obs_dim}, action_dim={action_dim}, "
          f"cond={args.cond_steps}, horizon={args.horizon_steps}, act={args.act_steps}")
    pred_mode = "epsilon" if args.predict_epsilon else "x"
    print(f"MIP: t_star={args.t_star}, dropout={args.dropout}, predict={pred_mode}")
    print(f"Dataset: {len(dataset)} samples, train={len(train_indices)}, val={len(val_indices)} "
          f"({n_train_traj}/{n_val_traj} trajs)")

    # Pre-load all data on GPU (model is small, avoid DataLoader bottleneck)
    def _preload(indices):
        obs_list, act_list = [], []
        for idx in indices:
            sample = dataset[idx]
            obs_list.append(sample["cond"]["state"])
            act_list.append(sample["actions"])
        obs_t = torch.stack(obs_list).to(device)
        act_t = torch.stack(act_list).to(device)
        if args.cond_steps == 1:
            obs_t = obs_t.squeeze(1)
        if args.horizon_steps == 1:
            act_t = act_t.squeeze(1)
        if args.zero_qvel:
            obs_t[..., 9:18] = 0.0
        return obs_t, act_t

    print("Pre-loading train/val data to GPU...")
    train_obs, train_act = _preload(train_indices)
    val_obs, val_act = _preload(val_indices)
    n_train = train_obs.shape[0]
    print(f"  Train: {train_obs.shape}, Val: {val_obs.shape}")

    # Build model
    if args.network_type == "unet":
        if args.mip_k > 2:
            from MultiGaussian.models.mip_k import MIPKSharedUNet
            model = MIPKSharedUNet(
                K=args.mip_k,
                input_dim=obs_dim,
                action_dim=action_dim,
                cond_steps=args.cond_steps,
                horizon_steps=args.horizon_steps,
                t_star=args.t_star,
            ).to(device)
            print(f"MIP-K={args.mip_k} shared UNet, t_schedule={model.t_schedule.tolist()}")
        else:
            from MultiGaussian.models.mip_unet import MIPUNetPolicy
            model = MIPUNetPolicy(
                input_dim=obs_dim,
                action_dim=action_dim,
                cond_steps=args.cond_steps,
                horizon_steps=args.horizon_steps,
                t_star=args.t_star,
            ).to(device)
    else:
        model = MIPPolicy(
            input_dim=obs_dim,
            action_dim=action_dim,
            cond_steps=args.cond_steps,
            horizon_steps=args.horizon_steps,
            t_star=args.t_star,
            dropout=args.dropout,
            emb_dim=args.emb_dim,
            n_layers=args.n_layers,
            predict_epsilon=args.predict_epsilon,
        ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    if args.network_type == "unet":
        model = torch.compile(model)
        print("torch.compile enabled")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

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
        for key in ("obs_min", "obs_max", "action_min", "action_max"):
            if key in resume_ckpt:
                ckpt_val = resume_ckpt[key]
                ds_val = getattr(dataset, key)
                if not torch.allclose(ckpt_val, ds_val, atol=1e-5):
                    print(f"  WARNING: {key} differs between checkpoint and current dataset!")
        print(f"Resumed from {args.resume} at iter {start_iter}")
        del resume_ckpt

    # Norm stats on GPU
    obs_min = dataset.obs_min.to(device)
    obs_max = dataset.obs_max.to(device)
    action_min = dataset.action_min.to(device)
    action_max = dataset.action_max.to(device)

    if args.obs_noise_std > 0:
        if not args.no_obs_norm:
            obs_range = (dataset.obs_max - dataset.obs_min).clamp(min=1e-8)
            obs_std_dev = (dataset.obs_std * 2.0 / obs_range).to(device)
        else:
            obs_std_dev = dataset.obs_std.to(device)
        print(f"Obs noise augmentation: {args.obs_noise_std} * per-dim std")

    best_sr = -1.0
    t_start = time.time()

    if start_iter > 0 and lr_scheduler is not None and not _lr_scheduler_restored:
        for _ in range(start_iter):
            lr_scheduler.step()

    print(f"Starting training from iter {start_iter + 1}...")
    for iteration in range(1, args.total_iters + 1):
        if iteration <= start_iter:
            continue
        model.train()

        # Random batch from GPU tensors
        idx = torch.randint(n_train, (args.batch_size,), device=device)
        obs = train_obs[idx]
        actions = train_act[idx]

        if args.obs_noise_std > 0:
            noise = torch.randn_like(obs) * obs_std_dev * args.obs_noise_std
            obs = obs + noise

        # MIP loss: two terms at t=0 and t=t*
        loss, loss_t0, loss_tstar = model.compute_loss(obs, actions)

        optimizer.zero_grad()
        loss.backward()
        if args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        ema.step(model.parameters())

        writer.add_scalar("train/loss", loss.item(), iteration)
        writer.add_scalar("train/loss_t0", loss_t0.item(), iteration)
        writer.add_scalar("train/loss_tstar", loss_tstar.item(), iteration)

        # Val loss
        if iteration % args.log_freq == 0:
            with torch.no_grad():
                val_loss, val_t0, val_tstar = model.compute_loss(val_obs, val_act)
            writer.add_scalar("val/loss", val_loss.item(), iteration)
            writer.add_scalar("val/loss_t0", val_t0.item(), iteration)
            writer.add_scalar("val/loss_tstar", val_tstar.item(), iteration)

            elapsed = time.time() - t_start
            print(f"Iter {iteration}/{args.total_iters}, "
                  f"loss={loss.item():.6f} (t0={loss_t0.item():.6f}, t*={loss_tstar.item():.6f}), "
                  f"val={val_loss.item():.6f}, "
                  f"time={elapsed:.0f}s ({iteration/elapsed:.1f} it/s)", flush=True)

        # Evaluate
        if iteration % args.eval_freq == 0 or iteration == args.total_iters:
            ema.copy_to(ema_model.parameters())
            ema_model.eval()

            eval_metrics = evaluate_mip_cpu(
                model=ema_model, device=device,
                n_episodes=args.num_eval_episodes,
                env_id=args.env_id, control_mode=args.control_mode,
                max_episode_steps=args.max_episode_steps,
                num_envs=args.num_eval_envs,
                obs_min=obs_min, obs_max=obs_max,
                action_min=action_min, action_max=action_max,
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
        "model_type": "mip",
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
