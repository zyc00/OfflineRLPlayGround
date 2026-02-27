"""
Diagnostic 4: Test different learning rates to find stability threshold.
"""
import numpy as np
import torch
import torch.nn as nn

from DPPO.model.unet_wrapper import DiffusionUNet
from DPPO.model.critic import CriticObs
from DPPO.model.diffusion_ppo import PPODiffusion
from DPPO.make_env import make_train_envs
from DPPO.reward_scaler import RunningRewardScaler

device = torch.device("cuda")
ckpt_path = "runs/dppo_pretrain/dppo_1000traj_unet_5k/ckpt_1500.pt"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
obs_dim = ckpt["obs_dim"]
action_dim = ckpt["action_dim"]
pretrain_args = ckpt.get("args", {})

K = 10
n_envs = 50
act_steps = 8
act_offset = 1
n_steps = 25


def build_model(norm_adv=False):
    actor = DiffusionUNet(
        action_dim=action_dim, horizon_steps=16, cond_dim=obs_dim * 2,
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
        clip_vloss_coef=None, norm_adv=norm_adv,
    )
    model.load_state_dict(ckpt["ema"], strict=False)
    model._sanitize_eta()
    model.actor_ft.load_state_dict(model.actor.state_dict())
    for p in model.actor.parameters():
        p.requires_grad = False
    return model


def do_rollout(model, seed=0):
    envs = make_train_envs(
        env_id="PickCube-v1", num_envs=n_envs,
        control_mode="pd_ee_delta_pos", max_episode_steps=100, seed=seed,
    )
    obs_raw, _ = envs.reset()
    obs_raw = torch.from_numpy(obs_raw).float().to(device)
    obs_history = obs_raw.unsqueeze(1).repeat(1, 2, 1)
    model.eval()
    n_succ = 0
    obs_trajs, chains_trajs, reward_trajs, done_trajs, value_trajs = [], [], [], [], []
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
        step_reward = torch.zeros(n_envs, device=device)
        step_done = torch.zeros(n_envs, dtype=torch.bool, device=device)
        for a_idx in range(act_steps):
            act_idx = act_offset + a_idx
            action = action_chunk[:, act_idx] if act_idx < action_chunk.shape[1] else action_chunk[:, -1]
            obs_new, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            obs_new = torch.from_numpy(obs_new).float().to(device)
            reward_t = torch.from_numpy(np.array(reward)).float().to(device)
            term_t = torch.from_numpy(np.array(terminated)).bool().to(device)
            trunc_t = torch.from_numpy(np.array(truncated)).bool().to(device)
            step_reward += reward_t * (~step_done).float()
            step_done = step_done | term_t | trunc_t
            n_succ += (reward_t > 0.5).sum().item()
            reset_mask = term_t | trunc_t
            if reset_mask.any():
                obs_history[reset_mask] = obs_new[reset_mask].unsqueeze(1).repeat(1, 2, 1)
            obs_history[~reset_mask] = torch.cat(
                [obs_history[~reset_mask, 1:], obs_new[~reset_mask].unsqueeze(1)], dim=1)
        reward_trajs.append(step_reward)
        done_trajs.append(step_done.float())
    envs.close()
    return n_succ, obs_trajs, chains_trajs, torch.stack(reward_trajs), torch.stack(done_trajs), torch.stack(value_trajs)


def do_ppo_update(model, obs_trajs, chains_trajs, rewards, dones, values,
                  n_epochs=1, lr=1e-5):
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(n_steps)):
        nextvalues = values[t + 1] if t < n_steps - 1 else torch.zeros(n_envs, device=device)
        next_not_done = 1.0 - dones[t]
        delta = rewards[t] + 0.999 * nextvalues * next_not_done - values[t]
        advantages[t] = lastgaelam = delta + 0.999 * 0.95 * next_not_done * lastgaelam
    returns = advantages + values

    obs_stacked = torch.stack(obs_trajs)
    chains_stacked = torch.stack(chains_trajs)
    N = n_steps * n_envs
    all_logprobs = []
    for i in range(0, N, 256):
        end = min(i + 256, N)
        step_inds = torch.arange(i, end)
        s_idx = step_inds // n_envs
        e_idx = step_inds % n_envs
        with torch.no_grad():
            batch_lp = model.get_logprobs({"state": obs_stacked[s_idx, e_idx]},
                                           chains_stacked[s_idx, e_idx])
            batch_lp = batch_lp.reshape(end - i, K, 16, action_dim)
        all_logprobs.append(batch_lp)
    all_logprobs = torch.cat(all_logprobs, dim=0)

    b_obs = obs_stacked.reshape(N, 2, obs_dim)
    b_chains = chains_stacked.reshape(N, K + 1, 16, action_dim)
    b_logprobs = all_logprobs
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    actor_opt = torch.optim.Adam(model.actor_ft.parameters(), lr=lr)
    critic_opt = torch.optim.Adam(model.critic.parameters(), lr=1e-3)
    model.train()
    total_samples = N * K
    for epoch in range(n_epochs):
        perm = torch.randperm(total_samples)
        for mb_start in range(0, total_samples, 500):
            mb_inds = perm[mb_start:mb_start + 500]
            if len(mb_inds) == 0:
                continue
            sample_inds = mb_inds // K
            denoise_inds = mb_inds % K
            pg_loss, _, v_loss, _, _, _ = model.loss(
                obs={"state": b_obs[sample_inds]},
                chains_prev=b_chains[sample_inds, denoise_inds],
                chains_next=b_chains[sample_inds, denoise_inds + 1],
                denoising_inds=denoise_inds,
                returns=b_returns[sample_inds],
                oldvalues=b_values[sample_inds],
                advantages=b_advantages[sample_inds],
                oldlogprobs=b_logprobs[sample_inds, denoise_inds],
                reward_horizon=act_steps,
            )
            loss = pg_loss + 0.5 * v_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.actor_ft.parameters(), 1.0)
            nn.utils.clip_grad_norm_(model.critic.parameters(), 1.0)
            actor_opt.step()
            critic_opt.step()
            actor_opt.zero_grad()
            critic_opt.zero_grad()


# Sweep learning rates
print("=== Learning rate sweep (norm_adv=False, no reward scaler, 1 epoch) ===")
print(f"{'lr':>10s}  {'pre':>4s}  {'post_s0':>7s}  {'post_s1':>7s}")
for lr in [0, 1e-8, 1e-7, 1e-6, 1e-5, 3e-5]:
    model = build_model(norm_adv=False)
    n_pre, obs_trajs, chains_trajs, rewards, dones, values = do_rollout(model, seed=42)
    if lr > 0:
        do_ppo_update(model, obs_trajs, chains_trajs, rewards, dones, values,
                      n_epochs=1, lr=lr)
    n_post0, *_ = do_rollout(model, seed=0)
    n_post1, *_ = do_rollout(model, seed=100)
    print(f"  {lr:>10.0e}  {n_pre:>4d}  {n_post0:>7d}  {n_post1:>7d}")

# Also test with norm_adv=True at different LRs
print("\n=== Learning rate sweep (norm_adv=True, no reward scaler, 1 epoch) ===")
print(f"{'lr':>10s}  {'pre':>4s}  {'post_s0':>7s}  {'post_s1':>7s}")
for lr in [0, 1e-8, 1e-7, 1e-6, 1e-5]:
    model = build_model(norm_adv=True)
    n_pre, obs_trajs, chains_trajs, rewards, dones, values = do_rollout(model, seed=42)
    if lr > 0:
        do_ppo_update(model, obs_trajs, chains_trajs, rewards, dones, values,
                      n_epochs=1, lr=lr)
    n_post0, *_ = do_rollout(model, seed=0)
    n_post1, *_ = do_rollout(model, seed=100)
    print(f"  {lr:>10.0e}  {n_pre:>4d}  {n_post0:>7d}  {n_post1:>7d}")

print("\nDone!")
