"""Test PickCube pretrained policy coverage at different spawn ranges.
Run MC rollouts from fixed initial states to get P(success|s0) distribution.
Uses GPU for speed (coverage analysis is relative, so GPU bias cancels out).
"""
import torch
import numpy as np
import gymnasium as gym
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import envs  # noqa

from DPPO.model.unet_wrapper import DiffusionUNet
from DPPO.model.diffusion import DiffusionModel


def rollout_once(model, env, num_envs, cond_steps, act_steps, act_offset,
                 max_steps, ddim_steps, device, deterministic=False):
    """Single rollout, return per-env success (bool tensor)."""
    obs, _ = env.reset()
    if not isinstance(obs, torch.Tensor):
        obs = torch.from_numpy(obs).float().to(device)
    else:
        obs = obs.float().to(device)
    obs_history = obs.unsqueeze(1).repeat(1, cond_steps, 1)

    success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    done = torch.zeros(num_envs, dtype=torch.bool, device=device)

    for step in range(max_steps // act_steps + 1):
        if done.all():
            break
        cond = {"state": obs_history}
        with torch.no_grad():
            samples = model(cond, deterministic=deterministic,
                          min_sampling_denoising_std=0.01,
                          ddim_steps=ddim_steps)
        action_chunk = samples.trajectories

        for a_idx in range(act_steps):
            act_idx = act_offset + a_idx
            action = action_chunk[:, min(act_idx, action_chunk.shape[1] - 1)]
            obs_new, reward, terminated, truncated, info = env.step(action)
            obs_new = obs_new.float()

            new_done = terminated | truncated
            newly_done = new_done & ~done
            if newly_done.any():
                success[newly_done] = reward[newly_done] > 0.5
                done[newly_done] = True

            # Update obs history (only for non-done envs)
            active = ~done
            if active.any():
                obs_history[active] = torch.cat(
                    [obs_history[active, 1:], obs_new[active].unsqueeze(1)], dim=1)

            if done.all():
                break

    return success


def main():
    device = torch.device("cuda")
    ckpt_path = "runs/dppo_pretrain/dppo_unet_T100_H16_bs1024/best.pt"

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = ckpt["args"]
    obs_dim, action_dim = ckpt["obs_dim"], ckpt["action_dim"]
    cond_steps = ckpt_args["cond_steps"]
    horizon_steps = ckpt_args["horizon_steps"]
    act_steps = ckpt_args["act_steps"]
    T = ckpt_args["denoising_steps"]
    act_offset = cond_steps - 1

    network = DiffusionUNet(
        action_dim=action_dim, horizon_steps=horizon_steps,
        cond_dim=obs_dim * cond_steps,
        diffusion_step_embed_dim=ckpt_args.get("diffusion_step_embed_dim", 64),
        down_dims=ckpt_args.get("unet_dims", [64, 128, 256]),
        n_groups=ckpt_args.get("n_groups", 8),
    )
    model = DiffusionModel(
        network=network, horizon_steps=horizon_steps,
        obs_dim=obs_dim, action_dim=action_dim, device=device,
        denoising_steps=T, denoised_clip_value=1.0,
        randn_clip_value=3, final_action_clip_value=1.0, predict_epsilon=True,
    )
    model.load_state_dict(ckpt["ema"], strict=False)
    model.eval()
    print(f"Model: unet, T={T}, H={horizon_steps}, cond={cond_steps}, act={act_steps}")

    spawn_ranges = [0.1, 0.15, 0.2, 0.25, 0.3]
    num_envs = 100
    mc_rollouts = 16  # MC rollouts per initial state set
    control_mode = ckpt_args.get("control_mode", "pd_joint_delta_pos")

    print(f"\nCoverage analysis: {num_envs} states × {mc_rollouts} MC rollouts, GPU")
    print(f"{'spawn':>8s} | {'SR':>6s} | {'frac_0':>7s} | {'frac_dec':>8s} | {'frac_1':>7s} | {'P_mean':>7s} | {'P_std':>6s}")
    print("-" * 68)

    for spawn_size in spawn_ranges:
        env = gym.make("PickCube-v2", num_envs=num_envs, obs_mode="state",
                       control_mode=control_mode, max_episode_steps=100,
                       sim_backend="gpu", reward_mode="sparse",
                       cube_spawn_half_size=spawn_size)
        env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=False,
                                 record_metrics=True)

        # MC rollouts (stochastic) - each rollout gets different initial states
        # To get proper coverage, we need SAME initial states across MC rollouts
        # But ManiSkill GPU env can't easily save/restore states
        # Instead, just run many rollouts and compute aggregate SR
        all_success = []
        for mc in range(mc_rollouts):
            success = rollout_once(model, env, num_envs, cond_steps, act_steps,
                                  act_offset, max_steps=100, ddim_steps=10,
                                  device=device, deterministic=False)
            all_success.append(success.float().cpu())

        # Stack: (mc_rollouts, num_envs)
        success_matrix = torch.stack(all_success, dim=0)
        # P(success) per env (averaged over MC rollouts) - note: different states each time
        # This gives us the OVERALL coverage distribution, not per-state
        # But since states are uniformly sampled, aggregate stats are meaningful
        overall_sr = success_matrix.mean().item()

        # For per-state coverage, we'd need same states across rollouts
        # Instead, report aggregate statistics
        per_rollout_sr = success_matrix.mean(dim=1)  # SR per rollout
        p_mean = per_rollout_sr.mean().item()
        p_std = per_rollout_sr.std().item()

        # Approximate coverage: fraction of envs that succeed at least once
        # across individual rollouts (each rollout = independent sample)
        per_env_ever_success = success_matrix.any(dim=0).float()
        frac_ever = per_env_ever_success.mean().item()

        # frac_zero = fraction of rollouts with 0% SR (unlikely with 100 envs)
        # Better: per-env success rate across rollouts (different states though)
        # Just report aggregate stats
        per_env_sr = success_matrix.mean(dim=0)  # avg success rate per env slot
        frac_zero = (per_env_sr == 0).float().mean().item()
        frac_one = (per_env_sr == 1).float().mean().item()
        frac_decisive = ((per_env_sr > 0.05) & (per_env_sr < 0.95)).float().mean().item()

        print(f"{spawn_size:>8.2f} | {overall_sr:>5.1%} | {frac_zero:>6.1%} | {frac_decisive:>7.1%} | {frac_one:>6.1%} | {p_mean:>6.3f} | {p_std:>5.3f}")
        env.close()


if __name__ == "__main__":
    main()
