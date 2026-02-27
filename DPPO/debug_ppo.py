"""
Diagnostic: check what happens to the policy after ONE PPO update.

1. Load pretrained model
2. Run 1 rollout, check successes
3. Do 1 PPO update (1 epoch, all data)
4. Run another rollout, check successes
5. Compare actions before/after
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, List

from DPPO.model.unet_wrapper import DiffusionUNet
from DPPO.model.critic import CriticObs
from DPPO.model.diffusion_ppo import PPODiffusion
from DPPO.make_env import make_train_envs
from DPPO.reward_scaler import RunningRewardScaler

device = torch.device("cuda")
ckpt_path = "runs/dppo_pretrain/dppo_1000traj_unet_5k/ckpt_1500.pt"

# Load checkpoint
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
obs_dim = ckpt["obs_dim"]
action_dim = ckpt["action_dim"]
pretrain_args = ckpt.get("args", {})

# Build model (DDPM mode, ft_denoising_steps=10)
K = 10
actor = DiffusionUNet(
    action_dim=action_dim, horizon_steps=16,
    cond_dim=obs_dim * 2,
    diffusion_step_embed_dim=pretrain_args.get("diffusion_step_embed_dim", 64),
    down_dims=pretrain_args.get("unet_dims", [64, 128, 256]),
    n_groups=pretrain_args.get("n_groups", 8),
)
critic = CriticObs(cond_dim=obs_dim * 2, mlp_dims=[256, 256, 256],
                    activation_type="Mish", residual_style=False)

model = PPODiffusion(
    actor=actor, critic=critic, ft_denoising_steps=K,
    min_sampling_denoising_std=0.01, min_logprob_denoising_std=0.1,
    use_ddim=False, ddim_steps=None,
    horizon_steps=16, obs_dim=obs_dim, action_dim=action_dim,
    device=device, denoising_steps=100,
    denoised_clip_value=1.0, randn_clip_value=3, final_action_clip_value=1.0,
    predict_epsilon=True, gamma_denoising=0.99,
    clip_ploss_coef=0.01, clip_ploss_coef_base=1e-3, clip_ploss_coef_rate=3.0,
    clip_vloss_coef=None, norm_adv=True,
)

model.load_state_dict(ckpt["ema"], strict=False)
model._sanitize_eta()
model.actor_ft.load_state_dict(model.actor.state_dict())
for p in model.actor.parameters():
    p.requires_grad = False

# Save initial actor_ft state for comparison
initial_params = {n: p.clone() for n, p in model.actor_ft.named_parameters()}

actor_optimizer = torch.optim.Adam(model.actor_ft.parameters(), lr=1e-5)
critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=1e-3)

# Create envs
n_envs = 50
n_steps = 25
act_steps = 8
act_offset = 1
gamma = 0.999
gae_lambda = 0.95

train_envs = make_train_envs(
    env_id="PickCube-v1", num_envs=n_envs,
    control_mode="pd_ee_delta_pos", max_episode_steps=100, seed=0,
)

obs_raw, _ = train_envs.reset()
obs_raw = torch.from_numpy(obs_raw).float().to(device)
obs_history = obs_raw.unsqueeze(1).repeat(1, 2, 1)

reward_scaler = RunningRewardScaler(gamma=gamma)

def do_rollout(model, obs_history, label=""):
    """Do one rollout, return obs_trajs, chains_trajs, rewards, dones, values, obs_history"""
    model.eval()
    obs_trajs, chains_trajs, reward_trajs, done_trajs, value_trajs = [], [], [], [], []
    n_succ = 0
    actions_collected = []

    for step in range(n_steps):
        cond = {"state": obs_history.clone()}
        with torch.no_grad():
            value = model.critic(cond).squeeze(-1)
            samples = model(cond, deterministic=False, return_chain=True)
            action_chunk = samples.trajectories
            chains = samples.chains

        obs_trajs.append(obs_history.clone())
        chains_trajs.append(chains.clone())
        value_trajs.append(value.clone())
        actions_collected.append(action_chunk[:, act_offset:act_offset+act_steps].clone())

        step_reward = torch.zeros(n_envs, device=device)
        step_done = torch.zeros(n_envs, dtype=torch.bool, device=device)

        for a_idx in range(act_steps):
            act_idx = act_offset + a_idx
            action = action_chunk[:, act_idx] if act_idx < action_chunk.shape[1] else action_chunk[:, -1]
            action_np = action.cpu().numpy()
            obs_new, reward, terminated, truncated, info = train_envs.step(action_np)
            obs_new = torch.from_numpy(obs_new).float().to(device)
            reward_t = torch.from_numpy(np.array(reward)).float().to(device)
            term_t = torch.from_numpy(np.array(terminated)).bool().to(device)
            trunc_t = torch.from_numpy(np.array(truncated)).bool().to(device)

            step_reward += reward_t * (~step_done).float()
            step_done = step_done | term_t | trunc_t
            n_succ += (reward_t > 0.5).sum().item()

            # Reset obs_history for terminated envs
            reset_mask = term_t | trunc_t
            if reset_mask.any():
                obs_history[reset_mask] = obs_new[reset_mask].unsqueeze(1).repeat(1, 2, 1)
            obs_history[~reset_mask] = torch.cat(
                [obs_history[~reset_mask, 1:], obs_new[~reset_mask].unsqueeze(1)], dim=1)

        reward_trajs.append(step_reward)
        done_trajs.append(step_done.float())

    rewards = torch.stack(reward_trajs)
    dones = torch.stack(done_trajs)
    values = torch.stack(value_trajs)
    actions_all = torch.cat(actions_collected, dim=0)
    print(f"  {label}: succ={n_succ}, mean_reward={rewards.sum(0).mean():.3f}, "
          f"mean_action_abs={actions_all.abs().mean():.4f}, action_std={actions_all.std():.4f}")
    return obs_trajs, chains_trajs, rewards, dones, values, obs_history


# === ROLLOUT 1: Before any update ===
print("=== Rollout 1 (pretrained) ===")
obs_trajs1, chains_trajs1, rewards1, dones1, values1, obs_history = do_rollout(
    model, obs_history, "pretrained")

# === PPO UPDATE ===
print("\n=== PPO Update (1 epoch) ===")

# Scale rewards
rewards_scaled = reward_scaler.update_and_scale(rewards1, dones1)
print(f"  reward_scale std: {np.sqrt(reward_scaler.running_var):.4f}")
print(f"  scaled rewards range: [{rewards_scaled.min():.3f}, {rewards_scaled.max():.3f}]")

# GAE
with torch.no_grad():
    next_value = model.critic({"state": obs_history}).squeeze(-1)

advantages = torch.zeros_like(rewards_scaled)
lastgaelam = 0
for t in reversed(range(n_steps)):
    if t == n_steps - 1:
        nextvalues = next_value
    else:
        nextvalues = values1[t + 1]
    next_not_done = 1.0 - dones1[t]
    delta = rewards_scaled[t] + gamma * nextvalues * next_not_done - values1[t]
    advantages[t] = lastgaelam = delta + gamma * gae_lambda * next_not_done * lastgaelam

returns = advantages + values1
print(f"  advantages: mean={advantages.mean():.4f}, std={advantages.std():.4f}, "
      f"min={advantages.min():.4f}, max={advantages.max():.4f}")
print(f"  returns: mean={returns.mean():.4f}, std={returns.std():.4f}")

# Compute old logprobs
obs_stacked = torch.stack(obs_trajs1)
chains_stacked = torch.stack(chains_trajs1)
N = n_steps * n_envs

all_logprobs = []
for i in range(0, N, 256):
    end = min(i + 256, N)
    step_inds = torch.arange(i, end)
    s_idx = step_inds // n_envs
    e_idx = step_inds % n_envs
    batch_obs = obs_stacked[s_idx, e_idx]
    batch_chains = chains_stacked[s_idx, e_idx]
    with torch.no_grad():
        batch_lp = model.get_logprobs({"state": batch_obs}, batch_chains)
        batch_lp = batch_lp.reshape(end - i, K, 16, action_dim)
    all_logprobs.append(batch_lp)
all_logprobs = torch.cat(all_logprobs, dim=0)

print(f"  old logprobs: mean={all_logprobs.mean():.4f}, std={all_logprobs.std():.4f}")

# PPO update: 1 epoch
model.train()
b_obs = obs_stacked.reshape(N, 2, obs_dim)
b_chains = chains_stacked.reshape(N, K + 1, 16, action_dim)
b_logprobs = all_logprobs
b_advantages = advantages.reshape(-1)
b_returns = returns.reshape(-1)
b_values = values1.reshape(-1)

total_samples = N * K
perm = torch.randperm(total_samples)
mb_size = 500

total_pg_loss = 0
total_v_loss = 0
total_kl = 0
total_ratio_mean = 0
n_mb = 0

actor_optimizer.zero_grad()
critic_optimizer.zero_grad()

for mb_start in range(0, total_samples, mb_size):
    mb_inds = perm[mb_start:mb_start + mb_size]
    if len(mb_inds) == 0:
        continue

    sample_inds = mb_inds // K
    denoise_inds = mb_inds % K

    mb_obs = {"state": b_obs[sample_inds]}
    mb_chains_prev = b_chains[sample_inds, denoise_inds]
    mb_chains_next = b_chains[sample_inds, denoise_inds + 1]
    mb_old_logprobs = b_logprobs[sample_inds, denoise_inds]
    mb_advantages = b_advantages[sample_inds]
    mb_returns = b_returns[sample_inds]
    mb_oldvalues = b_values[sample_inds]

    pg_loss, entropy_loss, v_loss, clipfrac, approx_kl, ratio_mean = model.loss(
        obs=mb_obs, chains_prev=mb_chains_prev, chains_next=mb_chains_next,
        denoising_inds=denoise_inds, returns=mb_returns, oldvalues=mb_oldvalues,
        advantages=mb_advantages, oldlogprobs=mb_old_logprobs,
        reward_horizon=act_steps,
    )

    loss = pg_loss + 0.5 * v_loss
    loss.backward()

    nn.utils.clip_grad_norm_(model.actor_ft.parameters(), 1.0)
    nn.utils.clip_grad_norm_(model.critic.parameters(), 1.0)
    actor_optimizer.step()
    critic_optimizer.step()
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()

    total_pg_loss += pg_loss.item()
    total_v_loss += v_loss.item()
    total_kl += approx_kl
    total_ratio_mean += ratio_mean
    n_mb += 1

print(f"\n  After 1 epoch ({n_mb} minibatches):")
print(f"  pg_loss={total_pg_loss/n_mb:.6f}, v_loss={total_v_loss/n_mb:.6f}, "
      f"kl={total_kl/n_mb:.6f}, ratio_mean={total_ratio_mean/n_mb:.6f}")

# Check parameter changes
max_diff = 0
total_diff = 0
n_params = 0
for n, p in model.actor_ft.named_parameters():
    diff = (p - initial_params[n]).abs()
    max_diff = max(max_diff, diff.max().item())
    total_diff += diff.sum().item()
    n_params += p.numel()
print(f"  actor_ft param change: max={max_diff:.6f}, mean={total_diff/n_params:.8f}")

# Check new logprobs on same data (should be different from old)
model.eval()
with torch.no_grad():
    new_lp_sample = model.get_logprobs_subsample(
        {"state": b_obs[:100]},
        b_chains[:100, 0],  # first denoising step
        b_chains[:100, 1],
        torch.zeros(100, dtype=torch.long, device=device),
    )
    old_lp_sample = b_logprobs[:100, 0]
    lp_diff = (new_lp_sample - old_lp_sample).abs()
    print(f"  logprob diff (first 100 samples, step 0): mean={lp_diff.mean():.6f}, max={lp_diff.max():.6f}")

# === ROLLOUT 2: After update ===
print("\n=== Rollout 2 (after 1 epoch PPO) ===")
obs_trajs2, chains_trajs2, rewards2, dones2, values2, obs_history = do_rollout(
    model, obs_history, "after_update")

# Compare action distributions
with torch.no_grad():
    # Generate actions for same observations
    test_cond = {"state": obs_stacked[0, :10]}  # First 10 envs from step 0
    model.eval()
    out_new = model(test_cond, deterministic=True)
    print(f"\n  Deterministic action sample (first 10 envs, step 0):")
    print(f"  action mean abs: {out_new.trajectories[:, act_offset:act_offset+act_steps].abs().mean():.4f}")

train_envs.close()
print("\nDone!")
