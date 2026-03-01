"""
Compare evaluate_inline (finetune_filtered_bc style) vs eval_cpu.py on the same checkpoint.
If they give different SR, the eval function is buggy.
"""

import torch
import numpy as np
from DPPO.model.unet_wrapper import DiffusionUNet
from DPPO.model.diffusion import DiffusionModel
from DPPO.make_env import make_train_envs
from DPPO.eval_cpu import evaluate_cpu_model

device = torch.device("cuda")

# Load checkpoint
ckpt_path = "runs/dppo_pretrain/dppo_200traj_unet_T100_5k/ckpt_2000.pt"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
pretrain_args = ckpt.get("args", {})

obs_dim = ckpt["obs_dim"]
action_dim = ckpt["action_dim"]
no_obs_norm = ckpt.get("no_obs_norm", True)
no_action_norm = ckpt.get("no_action_norm", True)

horizon_steps = pretrain_args["horizon_steps"]
cond_steps = pretrain_args["cond_steps"]
act_steps = pretrain_args["act_steps"]
denoising_steps = pretrain_args["denoising_steps"]
cond_dim = obs_dim * cond_steps
act_offset = cond_steps - 1  # UNet convention

# Build model
actor = DiffusionUNet(
    action_dim=action_dim,
    horizon_steps=horizon_steps,
    cond_dim=cond_dim,
    diffusion_step_embed_dim=pretrain_args.get("diffusion_step_embed_dim", 64),
    down_dims=pretrain_args.get("unet_dims", [64, 128, 256]),
    n_groups=pretrain_args.get("n_groups", 8),
)

model = DiffusionModel(
    network=actor,
    horizon_steps=horizon_steps,
    obs_dim=obs_dim,
    action_dim=action_dim,
    device=device,
    denoising_steps=denoising_steps,
    denoised_clip_value=1.0,
    randn_clip_value=3,  # finetune uses 3
    final_action_clip_value=1.0,
    predict_epsilon=True,
)
model.load_state_dict(ckpt["ema"], strict=False)

print(f"Loaded {ckpt_path}")
print(f"act_offset={act_offset}, act_steps={act_steps}, cond_steps={cond_steps}")
print(f"no_obs_norm={no_obs_norm}, no_action_norm={no_action_norm}")

# ===== Test 1: eval_cpu.py (ground truth) =====
print("\n" + "=" * 60)
print("TEST 1: eval_cpu.py (ground truth)")
print("=" * 60)
model.eval()
metrics_cpu = evaluate_cpu_model(
    n_episodes=100,
    model=model,
    device=device,
    act_steps=act_steps,
    cond_steps=cond_steps,
    env_id="PickCube-v1",
    control_mode="pd_ee_delta_pos",
    max_episode_steps=100,
    no_obs_norm=no_obs_norm,
    no_action_norm=no_action_norm,
    act_offset=act_offset,
)
print(f"eval_cpu.py: success_once={metrics_cpu['success_once']:.3f}, "
      f"success_at_end={metrics_cpu['success_at_end']:.3f}")

# ===== Test 2: evaluate_inline style (from finetune_filtered_bc) =====
print("\n" + "=" * 60)
print("TEST 2: evaluate_inline (finetune_filtered_bc style)")
print("=" * 60)

n_envs = 50
max_episode_steps = 200

train_envs = make_train_envs(
    env_id="PickCube-v1",
    num_envs=n_envs,
    sim_backend="cpu",
    control_mode="pd_ee_delta_pos",
    max_episode_steps=max_episode_steps,
    seed=0,
)

use_ddim = True
ddim_steps = 10
n_rounds = 3

@torch.no_grad()
def evaluate_inline():
    model.eval()
    total_success = 0
    total_eps = 0
    for rd in range(n_rounds):
        obs_r, _ = train_envs.reset()
        if isinstance(obs_r, np.ndarray):
            obs_r = torch.from_numpy(obs_r).float().to(device)
        else:
            obs_r = obs_r.float().to(device)
        obs_h = obs_r.unsqueeze(1).repeat(1, cond_steps, 1)
        success_once = torch.zeros(n_envs, dtype=torch.bool, device=device)
        for step in range(max_episode_steps // act_steps + 1):
            cond_eval = {"state": obs_h}  # no_obs_norm=True, identity
            samples_eval = model(cond_eval, deterministic=True, ddim_steps=ddim_steps)
            ac_eval = samples_eval.trajectories  # no_action_norm=True, identity
            for a_idx in range(act_steps):
                act_idx = act_offset + a_idx
                action_eval = ac_eval[:, min(act_idx, ac_eval.shape[1] - 1)]
                action_np = action_eval.cpu().numpy()
                obs_new_eval, rew_eval, term_eval, trunc_eval, _ = train_envs.step(action_np)
                obs_new_eval = torch.from_numpy(obs_new_eval).float().to(device)
                rew_t = torch.from_numpy(np.array(rew_eval)).float().to(device)
                term_t = torch.from_numpy(np.array(term_eval)).bool().to(device)
                trunc_t = torch.from_numpy(np.array(trunc_eval)).bool().to(device)
                success_once |= (rew_t > 0.5).bool()
                rm = term_t | trunc_t
                if rm.any():
                    obs_h[rm] = obs_new_eval[rm].unsqueeze(1).repeat(1, cond_steps, 1)
                obs_h[~rm] = torch.cat([obs_h[~rm, 1:], obs_new_eval[~rm].unsqueeze(1)], dim=1)
        total_success += success_once.sum().item()
        total_eps += n_envs
        print(f"  Round {rd+1}: {success_once.sum().item()}/{n_envs} success")
    return total_success / max(total_eps, 1)

sr = evaluate_inline()
print(f"evaluate_inline: sr={sr:.3f}")

# ===== Test 3: Same as Test 2 but with DDPM (no DDIM) =====
print("\n" + "=" * 60)
print("TEST 3: evaluate_inline with DDPM (no DDIM)")
print("=" * 60)

@torch.no_grad()
def evaluate_inline_ddpm():
    model.eval()
    total_success = 0
    total_eps = 0
    for rd in range(n_rounds):
        obs_r, _ = train_envs.reset()
        if isinstance(obs_r, np.ndarray):
            obs_r = torch.from_numpy(obs_r).float().to(device)
        else:
            obs_r = obs_r.float().to(device)
        obs_h = obs_r.unsqueeze(1).repeat(1, cond_steps, 1)
        success_once = torch.zeros(n_envs, dtype=torch.bool, device=device)
        for step in range(max_episode_steps // act_steps + 1):
            cond_eval = {"state": obs_h}
            samples_eval = model(cond_eval, deterministic=True)  # DDPM, no ddim_steps
            ac_eval = samples_eval.trajectories
            for a_idx in range(act_steps):
                act_idx = act_offset + a_idx
                action_eval = ac_eval[:, min(act_idx, ac_eval.shape[1] - 1)]
                action_np = action_eval.cpu().numpy()
                obs_new_eval, rew_eval, term_eval, trunc_eval, _ = train_envs.step(action_np)
                obs_new_eval = torch.from_numpy(obs_new_eval).float().to(device)
                rew_t = torch.from_numpy(np.array(rew_eval)).float().to(device)
                term_t = torch.from_numpy(np.array(term_eval)).bool().to(device)
                trunc_t = torch.from_numpy(np.array(trunc_eval)).bool().to(device)
                success_once |= (rew_t > 0.5).bool()
                rm = term_t | trunc_t
                if rm.any():
                    obs_h[rm] = obs_new_eval[rm].unsqueeze(1).repeat(1, cond_steps, 1)
                obs_h[~rm] = torch.cat([obs_h[~rm, 1:], obs_new_eval[~rm].unsqueeze(1)], dim=1)
        total_success += success_once.sum().item()
        total_eps += n_envs
        print(f"  Round {rd+1}: {success_once.sum().item()}/{n_envs} success")
    return total_success / max(total_eps, 1)

sr_ddpm = evaluate_inline_ddpm()
print(f"evaluate_inline DDPM: sr={sr_ddpm:.3f}")

# ===== Test 4: eval_cpu.py but with randn_clip=10 (pretrain default) =====
print("\n" + "=" * 60)
print("TEST 4: eval_cpu.py with randn_clip=10")
print("=" * 60)

actor2 = DiffusionUNet(
    action_dim=action_dim,
    horizon_steps=horizon_steps,
    cond_dim=cond_dim,
    diffusion_step_embed_dim=pretrain_args.get("diffusion_step_embed_dim", 64),
    down_dims=pretrain_args.get("unet_dims", [64, 128, 256]),
    n_groups=pretrain_args.get("n_groups", 8),
)
model2 = DiffusionModel(
    network=actor2,
    horizon_steps=horizon_steps,
    obs_dim=obs_dim,
    action_dim=action_dim,
    device=device,
    denoising_steps=denoising_steps,
    denoised_clip_value=1.0,
    randn_clip_value=10,  # pretrain default
    final_action_clip_value=1.0,
    predict_epsilon=True,
)
model2.load_state_dict(ckpt["ema"], strict=False)
model2.eval()

metrics_cpu2 = evaluate_cpu_model(
    n_episodes=100,
    model=model2,
    device=device,
    act_steps=act_steps,
    cond_steps=cond_steps,
    env_id="PickCube-v1",
    control_mode="pd_ee_delta_pos",
    max_episode_steps=100,
    no_obs_norm=no_obs_norm,
    no_action_norm=no_action_norm,
    act_offset=act_offset,
)
print(f"eval_cpu.py (clip=10): success_once={metrics_cpu2['success_once']:.3f}")

train_envs.close()
print("\nDone.")
