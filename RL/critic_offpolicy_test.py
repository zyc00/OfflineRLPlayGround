#!/usr/bin/env python
"""Test: Does off-policy data help on-policy critic estimation?

Runs iterative GAE PPO. At each iteration, trains three critics
and evaluates against MC16 V^π_k ground truth:

  V_on:       GAE returns on current data only (standard PPO)
  V_replay:   GAE returns on all accumulated data
  V_mc16_reg: Regression to MC16 targets (upper bound)

Policy updated with standard PPO using agent's own critic.
"""

import copy
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from scipy import stats
from tqdm import tqdm
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from data.data_collection.ppo import Agent, layer_init


# ── GAE computation on saved trajectory ─────────────────────────────────

def compute_gae_from_traj(critic, traj, gamma, gae_lambda, device):
    """Compute GAE returns for a saved trajectory using the given critic.

    Handles ManiSkill auto-reset boundaries via final_data.
    Returns (gae_returns_flat, obs_flat) both on CPU.
    """
    obs = traj["obs"]  # [T, N, D]
    rewards = traj["rewards"]  # [T, N]
    dones = traj["dones"]  # [T, N]
    next_obs = traj["next_obs"]  # [N, D]
    next_done = traj["next_done"]  # [N]
    final_data = traj["final_data"]

    T, N = rewards.shape

    with torch.no_grad():
        values = torch.zeros(T, N)
        for t in range(T):
            values[t] = critic(obs[t].to(device)).squeeze(-1).cpu()
        next_value = critic(next_obs.to(device)).squeeze(-1).cpu()

        final_values = torch.zeros(T, N)
        for step, env_mask, final_obs in final_data:
            fv = critic(final_obs.to(device)).squeeze(-1).cpu()
            final_values[step, env_mask] = fv

    advantages = torch.zeros(T, N)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_not_done = 1.0 - next_done
            nextvalues = next_value
        else:
            next_not_done = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        real_next_values = next_not_done * nextvalues + final_values[t]
        delta = rewards[t] + gamma * real_next_values - values[t]
        advantages[t] = lastgaelam = (
            delta + gamma * gae_lambda * next_not_done * lastgaelam
        )

    gae_returns = (advantages + values).reshape(-1)
    obs_flat = obs.reshape(-1, obs.shape[-1])
    return gae_returns, obs_flat


# ── Critic training ─────────────────────────────────────────────────────

def train_critic_mse(critic, obs, targets, num_epochs, minibatch_size, lr, device):
    """Train critic on fixed (obs, target) pairs with MSE loss."""
    opt = optim.Adam(critic.parameters(), lr=lr, eps=1e-5)
    N = obs.shape[0]
    critic.train()
    for _ in range(num_epochs):
        perm = torch.randperm(N)
        for start in range(0, N, minibatch_size):
            idx = perm[start : start + minibatch_size]
            pred = critic(obs[idx].to(device)).squeeze(-1)
            loss = 0.5 * ((pred - targets[idx].to(device)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            opt.step()
    critic.eval()
    return critic


# ── Main ────────────────────────────────────────────────────────────────


@dataclass
class Args:
    checkpoint: str = "runs/pickcube_ppo/ckpt_76_logstd-1.5.pt"
    env_id: str = "PickCube-v1"
    num_envs: int = 100
    num_eval_envs: int = 128
    mc_samples: int = 16
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"
    max_episode_steps: int = 50

    gamma: float = 0.99
    gae_lambda: float = 0.9
    learning_rate: float = 3e-4
    num_steps: int = 50
    num_minibatches: int = 5
    update_epochs: int = 100
    """PPO update epochs (policy + critic)"""
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float = 100.0
    norm_adv: bool = True
    reward_scale: float = 1.0

    critic_eval_epochs: int = 50
    """Epochs for training evaluation critics (V_on, V_replay, V_mc16)"""

    total_timesteps: int = 50_000
    eval_freq: int = 1
    seed: int = 1
    cuda: bool = True
    output: str = "runs/critic_offpolicy_test"

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    print("=== Critic Off-Policy Data Test ===")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  γ={args.gamma}, λ={args.gae_lambda}, mc_samples={args.mc_samples}")
    print(f"  Envs={args.num_envs}, Steps={args.num_steps}, Iters={args.num_iterations}")
    print(f"  PPO epochs={args.update_epochs}, Critic eval epochs={args.critic_eval_epochs}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ── Environments ────────────────────────────────────────────────
    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="physx_cuda" if device.type == "cuda" else "cpu",
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
    )
    envs = gym.make(args.env_id, num_envs=args.num_envs, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, **env_kwargs)
    num_mc_envs = args.num_envs * args.mc_samples
    mc_envs_raw = gym.make(args.env_id, num_envs=num_mc_envs, **env_kwargs)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
        mc_envs_raw = FlattenActionSpaceWrapper(mc_envs_raw)

    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=False, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=False, record_metrics=True)
    mc_envs = ManiSkillVectorEnv(mc_envs_raw, num_mc_envs, ignore_terminations=False, record_metrics=False)
    print(f"  MC envs: {num_mc_envs} ({args.num_envs} × {args.mc_samples})")

    # ── Agent ───────────────────────────────────────────────────────
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    agent.critic = nn.Sequential(
        layer_init(nn.Linear(obs_dim, 256)), nn.Tanh(),
        layer_init(nn.Linear(256, 256)), nn.Tanh(),
        layer_init(nn.Linear(256, 256)), nn.Tanh(),
        layer_init(nn.Linear(256, 1)),
    ).to(device)
    print(f"  Critic reset: 3×256")

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(a):
        return torch.clamp(a.detach(), action_low, action_high)

    # ── MC helpers ──────────────────────────────────────────────────
    _mc_zero = torch.zeros(num_mc_envs, *envs.single_action_space.shape, device=device)

    def _clone_state(sd):
        if isinstance(sd, dict):
            return {k: _clone_state(v) for k, v in sd.items()}
        return sd.clone()

    def _expand_state(sd, reps):
        if isinstance(sd, dict):
            return {k: _expand_state(v, reps) for k, v in sd.items()}
        if isinstance(sd, torch.Tensor) and sd.dim() > 0:
            return sd.repeat_interleave(reps, dim=0)
        return sd

    def _restore_mc(sd, seed=None):
        mc_envs.reset(seed=seed if seed is not None else args.seed)
        mc_envs.base_env.set_state_dict(sd)
        mc_envs.base_env.step(_mc_zero)
        mc_envs.base_env.set_state_dict(sd)
        mc_envs.base_env._elapsed_steps[:] = 0
        return mc_envs.base_env.get_obs()

    # ── Storage ─────────────────────────────────────────────────────
    obs_buf = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device
    )
    actions_buf = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device
    )
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    values_buf = torch.zeros((args.num_steps, args.num_envs), device=device)

    replay_buffer = []
    results = []

    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    for iteration in range(1, args.num_iterations + 1):
        iter_t0 = time.time()
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()

        # ── Evaluation ──────────────────────────────────────────
        eval_obs, _ = eval_envs.reset()
        eval_metrics = defaultdict(list)
        for _ in range(args.max_episode_steps):
            with torch.no_grad():
                eval_obs, _, _, _, eval_infos = eval_envs.step(
                    agent.get_action(eval_obs, deterministic=True)
                )
                if "final_info" in eval_infos:
                    mask = eval_infos["_final_info"]
                    for k, v in eval_infos["final_info"]["episode"].items():
                        eval_metrics[k].append(v[mask])
        sr_vals = eval_metrics.get("success_once", [])
        sr = torch.cat(sr_vals).float().mean().item() if sr_vals else 0.0

        # ── Rollout + save env states ───────────────────────────
        saved_states = []
        final_data = []

        for step in range(args.num_steps):
            global_step += args.num_envs
            saved_states.append(_clone_state(envs.base_env.get_state_dict()))
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_buf[step] = value.flatten()
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            next_obs, reward, term, trunc, infos = envs.step(clip_action(action))
            next_done = (term | trunc).float()
            rewards_buf[step] = reward.view(-1) * args.reward_scale

            if "final_info" in infos:
                done_mask = infos["_final_info"]
                final_obs_step = infos["final_observation"][done_mask].cpu().clone()
                final_data.append((step, done_mask.cpu().clone(), final_obs_step))
                with torch.no_grad():
                    final_values[
                        step, torch.arange(args.num_envs, device=device)[done_mask]
                    ] = agent.get_value(infos["final_observation"][done_mask]).view(-1)

        # Save to replay buffer (CPU)
        traj = {
            "obs": obs_buf.cpu().clone(),
            "rewards": rewards_buf.cpu().clone(),
            "dones": dones_buf.cpu().clone(),
            "next_obs": next_obs.cpu().clone(),
            "next_done": next_done.cpu().clone(),
            "final_data": final_data,
        }
        replay_buffer.append(traj)

        # ── MC16 ground truth V^π_k ────────────────────────────
        mc_t0 = time.time()
        mc16_values = torch.zeros(args.num_steps, args.num_envs)

        with torch.no_grad():
            for t in tqdm(range(args.num_steps), desc=f"  MC16 iter {iteration}", leave=False):
                expanded = _expand_state(saved_states[t], args.mc_samples)
                mc_obs = _restore_mc(expanded, seed=args.seed + t)

                # V^π_k: sample from π_k, rollout π_k
                a = agent.get_action(mc_obs, deterministic=False)
                mc_obs, rew, term_, trunc_, _ = mc_envs.step(clip_action(a))
                all_rews = [rew.view(-1) * args.reward_scale]
                env_done = (term_ | trunc_).view(-1).bool()

                for _ in range(args.max_episode_steps - 1):
                    if env_done.all():
                        break
                    a = agent.get_action(mc_obs, deterministic=False)
                    mc_obs, rew, term_, trunc_, _ = mc_envs.step(clip_action(a))
                    all_rews.append(rew.view(-1) * args.reward_scale * (~env_done).float())
                    env_done = env_done | (term_ | trunc_).view(-1).bool()

                ret = torch.zeros(num_mc_envs, device=device)
                for s in reversed(range(len(all_rews))):
                    ret = all_rews[s] + args.gamma * ret
                mc16_values[t] = ret.view(args.num_envs, args.mc_samples).mean(1).cpu()

        mc_time = time.time() - mc_t0
        mc16_flat = mc16_values.reshape(-1)
        del saved_states

        # ── Compute GAE returns for ALL trajectories ────────────
        all_gae_returns = []
        all_gae_obs = []
        for td in replay_buffer:
            gr, go = compute_gae_from_traj(agent.critic, td, args.gamma, args.gae_lambda, device)
            all_gae_returns.append(gr)
            all_gae_obs.append(go)

        cur_returns = all_gae_returns[-1]
        cur_obs_flat = all_gae_obs[-1]
        all_ret_cat = torch.cat(all_gae_returns)
        all_obs_cat = torch.cat(all_gae_obs)

        # ── Train 3 evaluation critics ──────────────────────────
        critic_t0 = time.time()

        v_on = copy.deepcopy(agent.critic)
        train_critic_mse(
            v_on, cur_obs_flat, cur_returns,
            args.critic_eval_epochs, args.minibatch_size, args.learning_rate, device,
        )

        v_replay = copy.deepcopy(agent.critic)
        train_critic_mse(
            v_replay, all_obs_cat, all_ret_cat,
            args.critic_eval_epochs, args.minibatch_size, args.learning_rate, device,
        )

        v_mc16 = copy.deepcopy(agent.critic)
        train_critic_mse(
            v_mc16, cur_obs_flat, mc16_flat,
            args.critic_eval_epochs, args.minibatch_size, args.learning_rate, device,
        )

        critic_time = time.time() - critic_t0

        # ── Evaluate critics on current states ──────────────────
        with torch.no_grad():
            dev_obs = cur_obs_flat.to(device)
            p_on = v_on(dev_obs).squeeze(-1).cpu().numpy()
            p_rep = v_replay(dev_obs).squeeze(-1).cpu().numpy()
            p_mc = v_mc16(dev_obs).squeeze(-1).cpu().numpy()
        gt = mc16_flat.numpy()

        row = {"iter": iteration, "sr": sr, "mc16_mean": gt.mean(), "mc16_std": gt.std()}
        for name, pred in [("V_on", p_on), ("V_replay", p_rep), ("V_mc16_reg", p_mc)]:
            row[f"{name}_r"] = np.corrcoef(pred, gt)[0, 1]
            row[f"{name}_rho"] = stats.spearmanr(pred, gt).statistic
            row[f"{name}_mse"] = float(((pred - gt) ** 2).mean())
            row[f"{name}_bias"] = float((pred - gt).mean())
        results.append(row)

        print(
            f"Iter {iteration}/{args.num_iterations} | SR={sr:.1%} | "
            f"MC16 μ={gt.mean():.4f} σ={gt.std():.4f} | "
            f"MC={mc_time:.0f}s Cr={critic_time:.0f}s"
        )
        for name in ["V_on", "V_replay", "V_mc16_reg"]:
            print(
                f"  {name:12s}: r={row[f'{name}_r']:.4f}  "
                f"ρ={row[f'{name}_rho']:.4f}  "
                f"MSE={row[f'{name}_mse']:.6f}  "
                f"bias={row[f'{name}_bias']:+.4f}"
            )

        # ── Standard PPO update (agent's own critic + policy) ───
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t]
                delta = rewards_buf[t] + args.gamma * real_next_values - values_buf[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam
                )
            returns = advantages + values_buf

        b_obs = obs_buf.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        agent.train()
        for epoch in range(args.update_epochs):
            b_inds = np.arange(args.batch_size)
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = b_inds[start : start + args.minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb], b_actions[mb]
                )
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                if approx_kl > args.target_kl:
                    break

                mb_adv = b_advantages[mb]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb]) ** 2).mean()
                loss = pg_loss - args.ent_coef * entropy.mean() + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            if approx_kl > args.target_kl:
                break

        print(f"  PPO: {epoch+1} epochs, kl={approx_kl:.4f}, total={time.time()-iter_t0:.0f}s")

    # ── Summary table ───────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("SUMMARY: Critic Quality vs MC16 Ground Truth (V^π_k)")
    print("=" * 100)
    hdr = (
        f"{'It':>2} {'SR':>6} {'MC16μ':>7} | "
        f"{'V_on r':>7} {'V_rep r':>7} {'V_mc r':>7} | "
        f"{'V_on ρ':>7} {'V_rep ρ':>7} {'V_mc ρ':>7} | "
        f"{'on bias':>8} {'rep bias':>9}"
    )
    print(hdr)
    print("-" * 100)
    for r in results:
        print(
            f"{r['iter']:2d} {r['sr']:6.1%} {r['mc16_mean']:7.4f} | "
            f"{r['V_on_r']:7.4f} {r['V_replay_r']:7.4f} {r['V_mc16_reg_r']:7.4f} | "
            f"{r['V_on_rho']:7.4f} {r['V_replay_rho']:7.4f} {r['V_mc16_reg_rho']:7.4f} | "
            f"{r['V_on_bias']:+8.4f} {r['V_replay_bias']:+9.4f}"
        )

    # ── Save ────────────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    save_path = os.path.join(args.output, "results.pt")
    torch.save(results, save_path)
    print(f"\nSaved to {save_path}")

    # ── Plot ────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        iters = [r["iter"] for r in results]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for ax, metric, title in zip(
            axes,
            ["r", "mse", "bias"],
            ["Pearson r with MC16", "MSE vs MC16", "Bias (V - MC16)"],
        ):
            for name, color, ls in [
                ("V_on", "tab:blue", "-"),
                ("V_replay", "tab:red", "--"),
                ("V_mc16_reg", "tab:green", ":"),
            ]:
                vals = [r[f"{name}_{metric}"] for r in results]
                ax.plot(iters, vals, color=color, ls=ls, marker="o", ms=4, label=name)
            ax.set_xlabel("Iteration")
            ax.set_title(title)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Off-Policy Data Test (γ={args.gamma}, λ={args.gae_lambda})", fontsize=12)
        plt.tight_layout()
        plot_path = os.path.join(args.output, "critic_quality.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {plot_path}")
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")

    envs.close()
    eval_envs.close()
    mc_envs.close()
