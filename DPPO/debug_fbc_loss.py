"""
Diagnose why demo-only BC finetuning degrades SR from 0.84 to ~0.5.

Checks:
1. Checkpoint contents and what gets loaded
2. Loss at initialization (should match pretrain iter 2000)
3. Loss after gradient steps
4. Parameter drift after training
5. Data format consistency (pretrain vs finetune loading path)
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
from DPPO.model.mlp import DiffusionMLP
from DPPO.model.unet_wrapper import DiffusionUNet
from DPPO.model.diffusion import DiffusionModel
from DPPO.dataset import DPPODataset
from torch.utils.data import DataLoader, RandomSampler, BatchSampler

device = torch.device("cuda")

# ===== 1. Load checkpoint =====
ckpt_path = "runs/dppo_pretrain/dppo_200traj_unet_T100_5k/ckpt_2000.pt"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

print("=" * 60)
print("CHECKPOINT CONTENTS")
print("=" * 60)
for k, v in ckpt.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: Tensor {v.shape} {v.dtype}")
    elif isinstance(v, dict):
        print(f"  {k}: dict with {len(v)} keys")
    else:
        print(f"  {k}: {v}")

pretrain_args = ckpt.get("args", {})
print(f"\nPretrain args:")
for k in ["network_type", "denoising_steps", "horizon_steps", "cond_steps", "act_steps",
           "no_obs_norm", "no_action_norm", "lr", "batch_size", "ema_decay",
           "max_grad_norm", "num_demos"]:
    print(f"  {k}: {pretrain_args.get(k, 'NOT SET')}")

no_obs_norm = ckpt.get("no_obs_norm", False)
no_action_norm = ckpt.get("no_action_norm", False)
obs_dim = ckpt["obs_dim"]
action_dim = ckpt["action_dim"]
print(f"\nno_obs_norm={no_obs_norm}, no_action_norm={no_action_norm}")
print(f"obs_dim={obs_dim}, action_dim={action_dim}")

# ===== 2. Build model exactly as pretrain does =====
network_type = pretrain_args.get("network_type", "mlp")
horizon_steps = pretrain_args.get("horizon_steps", 16)
cond_steps = pretrain_args.get("cond_steps", 2)
cond_dim = obs_dim * cond_steps

if network_type == "unet":
    actor = DiffusionUNet(
        action_dim=action_dim,
        horizon_steps=horizon_steps,
        cond_dim=cond_dim,
        diffusion_step_embed_dim=pretrain_args.get("diffusion_step_embed_dim", 64),
        down_dims=pretrain_args.get("unet_dims", [64, 128, 256]),
        n_groups=pretrain_args.get("n_groups", 8),
    )

# Build with pretrain's randn_clip_value=10 (vs finetune's 3)
model_pretrain = DiffusionModel(
    network=copy.deepcopy(actor),
    horizon_steps=horizon_steps,
    obs_dim=obs_dim,
    action_dim=action_dim,
    device=device,
    denoising_steps=pretrain_args.get("denoising_steps", 100),
    denoised_clip_value=1.0,
    randn_clip_value=10,
    final_action_clip_value=1.0,
    predict_epsilon=True,
)

# Build with finetune's randn_clip_value=3
model_finetune = DiffusionModel(
    network=copy.deepcopy(actor),
    horizon_steps=horizon_steps,
    obs_dim=obs_dim,
    action_dim=action_dim,
    device=device,
    denoising_steps=pretrain_args.get("denoising_steps", 100),
    denoised_clip_value=1.0,
    randn_clip_value=3,
    final_action_clip_value=1.0,
    predict_epsilon=True,
)

# ===== 3. Check what strict=False hides =====
print("\n" + "=" * 60)
print("WEIGHT LOADING CHECK")
print("=" * 60)

ema_state = ckpt["ema"]
model_state = ckpt["model"]

print(f"EMA state dict keys: {len(ema_state)}")
print(f"Model state dict keys: {len(model_state)}")

# Check missing/unexpected keys
result = model_pretrain.load_state_dict(ema_state, strict=False)
print(f"\nLoad EMA into model (strict=False):")
print(f"  Missing keys: {result.missing_keys}")
print(f"  Unexpected keys: {result.unexpected_keys}")

# Also check model weights
result2 = model_finetune.load_state_dict(ema_state, strict=False)
print(f"\nLoad EMA into finetune model (strict=False):")
print(f"  Missing keys: {result2.missing_keys}")
print(f"  Unexpected keys: {result2.unexpected_keys}")

# ===== 4. Load demo data both ways =====
print("\n" + "=" * 60)
print("DATA LOADING COMPARISON")
print("=" * 60)

demo_path = os.path.expanduser(
    "~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5"
)

# Way 1: Full 1000 demos (how finetune_filtered_bc loads)
dataset_1000 = DPPODataset(
    demo_path, horizon_steps, cond_steps,
    no_obs_norm=no_obs_norm, no_action_norm=no_action_norm,
)

# Way 2: 200 demos (how pretrain was trained)
num_demos = pretrain_args.get("num_demos", None)
print(f"\nPretrain num_demos: {num_demos}")
dataset_200 = DPPODataset(
    demo_path, horizon_steps, cond_steps,
    num_traj=200 if num_demos is None else num_demos,
    no_obs_norm=no_obs_norm, no_action_norm=no_action_norm,
)

# Compare a few samples
print(f"\nDataset 1000: {len(dataset_1000)} samples")
print(f"Dataset 200:  {len(dataset_200)} samples")

# Check if first 200 trajs' data is identical
item_200 = dataset_200[0]
item_1000 = dataset_1000[0]
print(f"\nSample 0 obs match: {torch.allclose(item_200['cond']['state'], item_1000['cond']['state'])}")
print(f"Sample 0 act match: {torch.allclose(item_200['actions'], item_1000['actions'])}")
print(f"Sample 0 obs range: [{item_200['cond']['state'].min():.4f}, {item_200['cond']['state'].max():.4f}]")
print(f"Sample 0 act range: [{item_200['actions'].min():.4f}, {item_200['actions'].max():.4f}]")

# ===== 5. Compute loss with both datasets =====
print("\n" + "=" * 60)
print("LOSS COMPARISON")
print("=" * 60)

model_pretrain.eval()
model_finetune.eval()

torch.manual_seed(42)

# Compute average loss over 100 batches with pretrain model on pretrain data (200 demos)
losses_pretrain_200 = []
for _ in range(100):
    idxs = torch.randint(0, len(dataset_200), (256,))
    batch_act = torch.stack([dataset_200[i]["actions"] for i in idxs]).to(device)
    batch_obs = torch.stack([dataset_200[i]["cond"]["state"] for i in idxs]).to(device)
    with torch.no_grad():
        loss = model_pretrain.loss(batch_act, {"state": batch_obs})
    losses_pretrain_200.append(loss.item())

# Compute with pretrain model on 1000 demos
losses_pretrain_1000 = []
torch.manual_seed(42)
for _ in range(100):
    idxs = torch.randint(0, len(dataset_1000), (256,))
    batch_act = torch.stack([dataset_1000[i]["actions"] for i in idxs]).to(device)
    batch_obs = torch.stack([dataset_1000[i]["cond"]["state"] for i in idxs]).to(device)
    with torch.no_grad():
        loss = model_pretrain.loss(batch_act, {"state": batch_obs})
    losses_pretrain_1000.append(loss.item())

# Also compute with finetune model (randn_clip=3)
losses_finetune_1000 = []
torch.manual_seed(42)
for _ in range(100):
    idxs = torch.randint(0, len(dataset_1000), (256,))
    batch_act = torch.stack([dataset_1000[i]["actions"] for i in idxs]).to(device)
    batch_obs = torch.stack([dataset_1000[i]["cond"]["state"] for i in idxs]).to(device)
    with torch.no_grad():
        loss = model_finetune.loss(batch_act, {"state": batch_obs})
    losses_finetune_1000.append(loss.item())

print(f"Pretrain model + 200 demo loss:  {np.mean(losses_pretrain_200):.6f} ± {np.std(losses_pretrain_200):.6f}")
print(f"Pretrain model + 1000 demo loss: {np.mean(losses_pretrain_1000):.6f} ± {np.std(losses_pretrain_1000):.6f}")
print(f"Finetune model + 1000 demo loss: {np.mean(losses_finetune_1000):.6f} ± {np.std(losses_finetune_1000):.6f}")

# ===== 6. Check pretrain's TensorBoard loss at iter 2000 =====
print("\n" + "=" * 60)
print("PRETRAIN TB LOSS AT ITER 2000")
print("=" * 60)
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator("runs/dppo_pretrain/dppo_200traj_unet_T100_5k")
    ea.Reload()
    loss_events = ea.Scalars("train/loss")
    for e in loss_events:
        if 1990 <= e.step <= 2010:
            print(f"  train/loss @ iter {e.step}: {e.value:.6f}")
except Exception as ex:
    print(f"  Could not read TB: {ex}")

# ===== 7. Train a few steps and monitor parameter drift =====
print("\n" + "=" * 60)
print("GRADIENT STEP ANALYSIS")
print("=" * 60)

# Save initial parameters
init_params = {k: v.clone() for k, v in model_finetune.named_parameters()}

optimizer = torch.optim.Adam(model_finetune.network.parameters(), lr=1e-6)
model_finetune.train()

for step in range(10):
    idxs = torch.randint(0, len(dataset_1000), (256,))
    batch_act = torch.stack([dataset_1000[i]["actions"] for i in idxs]).to(device)
    batch_obs = torch.stack([dataset_1000[i]["cond"]["state"] for i in idxs]).to(device)
    loss = model_finetune.loss(batch_act, {"state": batch_obs})
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model_finetune.network.parameters(), 1.0)
    optimizer.step()
    if step < 5 or step == 9:
        print(f"  Step {step}: loss={loss.item():.6f}")

# Check parameter drift
print(f"\nParameter drift after 10 steps (lr=1e-6):")
total_drift = 0
total_norm = 0
for name, param in model_finetune.named_parameters():
    drift = (param - init_params[name]).norm().item()
    norm = init_params[name].norm().item()
    total_drift += drift ** 2
    total_norm += norm ** 2
    if drift > 0:
        print(f"  {name}: drift={drift:.8f}, norm={norm:.4f}, ratio={drift/max(norm,1e-10):.8f}")
total_drift = total_drift ** 0.5
total_norm = total_norm ** 0.5
print(f"  TOTAL: drift={total_drift:.8f}, norm={total_norm:.4f}, ratio={total_drift/total_norm:.8f}")

# ===== 8. Now do 200 steps (one "iteration" of finetune) =====
print("\n" + "=" * 60)
print("FULL ITERATION (200 steps) ANALYSIS")
print("=" * 60)

# Reset to initial weights
model_finetune.load_state_dict(ema_state, strict=False)
optimizer = torch.optim.Adam(model_finetune.network.parameters(), lr=1e-6)
model_finetune.train()

losses_during = []
for step in range(200):
    idxs = torch.randint(0, len(dataset_1000), (256,))
    batch_act = torch.stack([dataset_1000[i]["actions"] for i in idxs]).to(device)
    batch_obs = torch.stack([dataset_1000[i]["cond"]["state"] for i in idxs]).to(device)
    loss = model_finetune.loss(batch_act, {"state": batch_obs})
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model_finetune.network.parameters(), 1.0)
    optimizer.step()
    losses_during.append(loss.item())

print(f"Loss: start={losses_during[0]:.6f}, end={losses_during[-1]:.6f}")
print(f"Loss: mean={np.mean(losses_during):.6f}, min={np.min(losses_during):.6f}")

# Total parameter drift after 200 steps
total_drift = 0
total_norm = 0
for name, param in model_finetune.named_parameters():
    drift = (param - init_params[name]).norm().item()
    norm = init_params[name].norm().item()
    total_drift += drift ** 2
    total_norm += norm ** 2
total_drift = total_drift ** 0.5
total_norm = total_norm ** 0.5
print(f"Parameter drift after 200 steps: drift={total_drift:.6f}, norm={total_norm:.4f}, ratio={total_drift/total_norm:.8f}")

# ===== 9. Compare EMA vs non-EMA model weights =====
print("\n" + "=" * 60)
print("EMA vs MODEL WEIGHT COMPARISON")
print("=" * 60)
ema_sd = ckpt["ema"]
model_sd = ckpt["model"]
for key in list(ema_sd.keys())[:5]:
    if key in model_sd:
        diff = (ema_sd[key].float() - model_sd[key].float()).norm().item()
        norm = ema_sd[key].float().norm().item()
        print(f"  {key}: ema-model diff={diff:.6f}, ema_norm={norm:.4f}")
