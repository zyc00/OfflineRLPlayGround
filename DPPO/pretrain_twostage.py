"""
Two-Stage Diffusion Policy Training.

Stage 1: Unconditional action manifold learning (obs zeroed out).
Stage 2: FiLM conditioning learning (backbone frozen, only cond_encoder trained).

Usage:
    # Stage 1: learn action manifold
    python -m DPPO.pretrain_twostage --stage 1 --total_iters 100000 ...

    # Stage 2: learn FiLM conditioning
    python -m DPPO.pretrain_twostage --stage 2 --stage1_ckpt runs/.../best.pt --total_iters 100000 ...

    # Stage 2 + unfreeze: fine-tune everything after N iters
    python -m DPPO.pretrain_twostage --stage 2 --stage1_ckpt runs/.../best.pt --unfreeze_after 50000 ...
"""

import os
import copy
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from dataclasses import dataclass, field
from typing import Optional, List
from torch.utils.tensorboard import SummaryWriter

import tyro

from DPPO.model.mlp import DiffusionMLP
from DPPO.model.unet_wrapper import DiffusionUNet
from DPPO.model.diffusion import DiffusionModel
from DPPO.dataset import DPPODataset
from DPPO.evaluate import evaluate_gpu
from DPPO.eval_cpu import evaluate_cpu_model
from diffusers.training_utils import EMAModel as DiffusersEMA


@dataclass
class Args:
    env_id: str = "PegInsertionSide-v1"
    demo_path: str = "~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5"
    num_demos: Optional[int] = None
    control_mode: str = "pd_joint_delta_pos"
    max_episode_steps: int = 300

    # Diffusion
    denoising_steps: int = 100
    horizon_steps: int = 16
    cond_steps: int = 2
    act_steps: int = 8

    # Network
    network_type: str = "unet"
    mlp_dims: List[int] = field(default_factory=lambda: [512, 512, 512])
    time_dim: int = 16
    activation_type: str = "Mish"
    residual_style: bool = True
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    diffusion_step_embed_dim: int = 64
    n_groups: int = 8

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
    eval_freq: int = 25000
    num_eval_episodes: int = 100
    num_eval_envs: int = 100
    log_freq: int = 200
    cpu_eval: bool = True

    # Normalization
    no_obs_norm: bool = True
    no_action_norm: bool = True

    # Augmentation
    zero_qvel: bool = True

    # Two-stage
    stage: str = "1"  # "1" (unconditional manifold) or "2" (FiLM only)
    stage1_ckpt: Optional[str] = None  # Required for stage 2
    unfreeze_after: int = 0  # Unfreeze all params after this many iters in stage 2 (0=never)
    stage2_lr: Optional[float] = None  # Override lr for stage 2

    # Logging
    exp_name: Optional[str] = None
    seed: int = 1
    save_dir: str = "runs/dppo_pretrain"


def _freeze_backbone(model):
    """Freeze everything except cond_encoder modules in the UNet."""
    frozen, trainable = 0, 0
    for name, param in model.named_parameters():
        if 'cond_encoder' in name:
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()
    return frozen, trainable


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


def main():
    args = tyro.cli(Args)
    args.demo_path = os.path.expanduser(args.demo_path)

    assert args.stage in ("1", "2"), f"--stage must be '1' or '2', got '{args.stage}'"
    if args.stage == "2":
        assert args.stage1_ckpt, "--stage1_ckpt required for stage 2"

    if args.exp_name is None:
        args.exp_name = f"dp_{args.env_id}_stage{args.stage}"

    run_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, stage={args.stage}")

    # Dataset
    dataset = DPPODataset(
        data_path=args.demo_path,
        horizon_steps=args.horizon_steps,
        cond_steps=args.cond_steps,
        num_traj=args.num_demos,
        no_obs_norm=args.no_obs_norm,
        no_action_norm=args.no_action_norm,
    )

    from diffusion_policy.utils import IterationBasedBatchSampler
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.num_workers,
                            pin_memory=(args.num_workers > 0))

    obs_dim = dataset.obs_dim
    action_dim = dataset.action_dim
    cond_dim = obs_dim * args.cond_steps

    print(f"obs_dim={obs_dim}, action_dim={action_dim}, cond_dim={cond_dim}")
    print(f"Dataset: {len(dataset)} samples, training {args.total_iters} iters")

    # Build network
    if args.network_type == "unet":
        network = DiffusionUNet(
            action_dim=action_dim,
            horizon_steps=args.horizon_steps,
            cond_dim=cond_dim,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
    else:
        network = DiffusionMLP(
            action_dim=action_dim,
            horizon_steps=args.horizon_steps,
            cond_dim=cond_dim,
            time_dim=args.time_dim,
            mlp_dims=args.mlp_dims,
            activation_type=args.activation_type,
            residual_style=args.residual_style,
        )
    model = DiffusionModel(
        network=network,
        horizon_steps=args.horizon_steps,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        denoising_steps=args.denoising_steps,
        denoised_clip_value=1.0,
        randn_clip_value=10,
        final_action_clip_value=1.0,
        predict_epsilon=True,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Stage 2: load stage 1 weights and freeze backbone
    if args.stage == "2":
        s1_ckpt = torch.load(os.path.expanduser(args.stage1_ckpt),
                             map_location=device, weights_only=False)
        model.load_state_dict(s1_ckpt["ema"])
        print(f"Loaded Stage 1 checkpoint from {args.stage1_ckpt}")
        del s1_ckpt
        frozen, trainable = _freeze_backbone(model)
        print(f"Frozen {frozen:,} params, training {trainable:,} params (cond_encoder)")

    effective_lr = args.stage2_lr if (args.stage == "2" and args.stage2_lr) else args.lr
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=effective_lr, weight_decay=args.weight_decay, betas=(0.95, 0.999),
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

    ema = DiffusersEMA(parameters=model.parameters(), power=0.75)
    ema_model = copy.deepcopy(model)

    obs_min = dataset.obs_min.to(device)
    obs_max = dataset.obs_max.to(device)
    action_min = dataset.action_min.to(device)
    action_max = dataset.action_max.to(device)

    best_sr = -1.0
    t_start = time.time()
    unfrozen = False

    print(f"Starting Stage {args.stage} training...")
    for iteration, batch in enumerate(dataloader, 1):
        model.train()
        actions = batch["actions"].to(device)
        cond = {k: v.to(device) for k, v in batch["cond"].items()}

        if args.zero_qvel and "state" in cond:
            cond["state"][..., 9:18] = 0.0

        # Stage 1: zero out obs conditioning
        if args.stage == "1":
            cond["state"] = torch.zeros_like(cond["state"])

        # Stage 2: unfreeze backbone after N iters
        if (args.stage == "2" and args.unfreeze_after > 0
                and iteration == args.unfreeze_after and not unfrozen):
            for param in model.parameters():
                param.requires_grad = True
            unfrozen = True
            unfreeze_lr = effective_lr * 0.1
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=unfreeze_lr,
                weight_decay=args.weight_decay, betas=(0.95, 0.999),
            )
            if args.use_cosine_lr:
                from diffusers.optimization import get_scheduler
                lr_scheduler = get_scheduler(
                    name='cosine', optimizer=optimizer,
                    num_warmup_steps=100,
                    num_training_steps=args.total_iters - iteration,
                )
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Unfroze all at iter {iteration} ({n_train:,} params, lr={unfreeze_lr:.1e})")

        loss = model.loss(actions, cond)
        optimizer.zero_grad()
        loss.backward()
        if args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        ema.step(model.parameters())

        writer.add_scalar("train/loss", loss.item(), iteration)

        if iteration % args.log_freq == 0:
            elapsed = time.time() - t_start
            print(f"Iter {iteration}/{args.total_iters}, loss={loss.item():.6f}, "
                  f"time={elapsed:.0f}s ({iteration/elapsed:.1f} it/s)", flush=True)

        # Evaluate
        if iteration % args.eval_freq == 0 or iteration == args.total_iters:
            ema.copy_to(ema_model.parameters())
            ema_model.eval()

            act_offset = args.cond_steps - 1 if args.network_type == "unet" else 0

            if args.cpu_eval:
                eval_metrics = evaluate_cpu_model(
                    n_episodes=args.num_eval_episodes,
                    model=ema_model,
                    device=device,
                    act_steps=args.act_steps,
                    cond_steps=args.cond_steps,
                    env_id=args.env_id,
                    num_envs=min(args.num_eval_envs, 10),
                    control_mode=args.control_mode,
                    max_episode_steps=args.max_episode_steps,
                    obs_min=obs_min,
                    obs_max=obs_max,
                    action_min=action_min,
                    action_max=action_max,
                    no_obs_norm=args.no_obs_norm,
                    no_action_norm=args.no_action_norm,
                    act_offset=act_offset,
                    zero_qvel=args.zero_qvel,
                )
            else:
                eval_metrics = evaluate_gpu(
                    n_episodes=args.num_eval_episodes,
                    model=ema_model,
                    device=device,
                    act_steps=args.act_steps,
                    obs_min=obs_min,
                    obs_max=obs_max,
                    action_min=action_min,
                    action_max=action_max,
                    cond_steps=args.cond_steps,
                    env_id=args.env_id,
                    num_envs=args.num_eval_envs,
                    control_mode=args.control_mode,
                    max_episode_steps=args.max_episode_steps,
                    no_obs_norm=args.no_obs_norm,
                    no_action_norm=args.no_action_norm,
                    act_offset=act_offset,
                    zero_qvel=args.zero_qvel,
                )

            sr = eval_metrics.get("success_at_end", 0.0)
            sr_once = eval_metrics.get("success_once", 0.0)
            writer.add_scalar("eval/success_at_end", sr, iteration)
            writer.add_scalar("eval/success_once", sr_once, iteration)
            backend = "cpu" if args.cpu_eval else "cuda"
            print(f"  Eval [{backend}] @ iter {iteration}: "
                  f"success_at_end={sr:.3f}, success_once={sr_once:.3f}")

            _save_ckpt(model, ema_model, dataset, obs_dim, action_dim, args, iteration,
                       os.path.join(run_dir, f"ckpt_{iteration}.pt"),
                       optimizer=optimizer, ema_state=ema, lr_scheduler=lr_scheduler)

            if sr_once > best_sr:
                best_sr = sr_once
                _save_ckpt(model, ema_model, dataset, obs_dim, action_dim, args, iteration,
                           os.path.join(run_dir, "best.pt"),
                           optimizer=optimizer, ema_state=ema, lr_scheduler=lr_scheduler)
                print(f"  Saved best (sr_once={best_sr:.3f})")

    ema.copy_to(ema_model.parameters())
    _save_ckpt(model, ema_model, dataset, obs_dim, action_dim, args, args.total_iters,
               os.path.join(run_dir, "final.pt"),
               optimizer=optimizer, ema_state=ema, lr_scheduler=lr_scheduler)
    elapsed = time.time() - t_start
    print(f"Stage {args.stage} complete in {elapsed:.0f}s. Best sr_once={best_sr:.3f}")
    writer.close()


if __name__ == "__main__":
    main()
