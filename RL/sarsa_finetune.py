"""PPO finetuning with Q-V regression advantage estimation.

Each iteration:
  1. Rollout (collect trajectories)
  2. Compute MC returns (backward discounted sum, λ=1.0)
  3. Regress Q(s,a) and V(s) on MC returns (supervised MSE)
  4. Advantage = Q(s,a) - V(s)
  5. PPO update (actor + critic for bootstrapping)

If Q/V regression is perfect, Q(s,a)-V(s) ≈ G_t - E[G|s_t], theoretically
equivalent to MC1. This tests whether the regression bottleneck matters for
PPO policy improvement.

Usage:
  python -m RL.sarsa_finetune
  python -m RL.sarsa_finetune --regression_epochs 100
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
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from torch.utils.tensorboard import SummaryWriter

from data.data_collection.ppo import Agent, layer_init
from methods.iql.iql import QNetwork


@dataclass
class Args:
    # Finetuning
    checkpoint: str = "runs/pickcube_ppo/ckpt_101.pt"
    """pretrained checkpoint to finetune from"""
    reset_critic: bool = True
    """reset critic weights (needed when finetuning with different reward mode)"""
    critic_checkpoint: Optional[str] = None
    """pretrained critic checkpoint (overrides reset_critic)"""

    # Q-V regression
    regression_epochs: int = 50
    """epochs to regress Q and V on MC returns each iteration"""
    regression_lr: float = 3e-4
    """learning rate for Q/V regression"""
    regression_batch_size: int = 256
    """batch size for Q/V regression"""

    # Environment
    env_id: str = "PickCube-v1"
    num_envs: int = 512
    num_eval_envs: int = 128
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"
    max_episode_steps: int = 50

    # PPO hyperparameters
    gamma: float = 0.8
    learning_rate: float = 3e-4
    num_steps: int = 50
    """rollout length per iteration (= max_episode_steps for full episodes)"""
    num_minibatches: int = 32
    update_epochs: int = 4
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float = 0.1
    norm_adv: bool = True
    reward_scale: float = 1.0

    # Training
    total_timesteps: int = 2_000_000
    eval_freq: int = 5
    """evaluate every N iterations"""
    warmup_iters: int = 0
    """iterations to train critic only (no policy update)."""
    seed: int = 1
    cuda: bool = True

    # Logging
    exp_name: Optional[str] = None
    capture_video: bool = True
    save_model: bool = True

    # Computed at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Always use MC returns (GAE with lambda=1.0)
    gae_lambda = 1.0

    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    if args.exp_name is None:
        args.exp_name = "qv_regression_ppo"
    run_name = f"{args.exp_name}__seed{args.seed}__{int(time.time())}"

    print(f"=== PPO Finetuning (Q-V Regression) ===")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Reward: {args.reward_mode}")
    print(f"  Regression epochs: {args.regression_epochs}, lr: {args.regression_lr}")
    print(f"  Envs: {args.num_envs}, Steps: {args.num_steps}")
    print(f"  Batch: {args.batch_size}, Minibatch: {args.minibatch_size}")
    print(f"  Iterations: {args.num_iterations}")

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ── Environment setup ──────────────────────────────────────────────
    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="physx_cuda" if device.type == "cuda" else "cpu",
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
    )

    envs = gym.make(args.env_id, num_envs=args.num_envs, **env_kwargs)
    eval_envs = gym.make(
        args.env_id, num_envs=args.num_eval_envs, **env_kwargs,
    )

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        print(f"  Saving eval videos to {eval_output_dir}")
        eval_envs = RecordEpisode(
            eval_envs, output_dir=eval_output_dir,
            save_trajectory=False,
            max_steps_per_video=args.max_episode_steps,
            video_fps=30,
        )

    envs = ManiSkillVectorEnv(
        envs, args.num_envs,
        ignore_terminations=False,
        record_metrics=True,
    )
    eval_envs = ManiSkillVectorEnv(
        eval_envs, args.num_eval_envs,
        ignore_terminations=False,
        record_metrics=True,
    )

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)

    # ── Agent setup ────────────────────────────────────────────────────
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"  Loaded checkpoint: {args.checkpoint}")

    if args.critic_checkpoint is not None:
        critic_state = torch.load(args.critic_checkpoint, map_location=device)
        agent.critic.load_state_dict(critic_state)
        print(f"  Loaded pretrained critic: {args.critic_checkpoint}")
    elif args.reset_critic:
        agent.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        ).to(device)
        print("  Critic reset (fresh init for sparse reward)")

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(a):
        return torch.clamp(a.detach(), action_low, action_high)

    # ── Q/V regression networks (separate from Agent's critic) ─────────
    obs_dim = np.array(envs.single_observation_space.shape).prod()
    act_dim = np.prod(envs.single_action_space.shape)

    q_net = QNetwork(obs_dim, act_dim).to(device)
    v_net = nn.Sequential(
        layer_init(nn.Linear(obs_dim, 256)),
        nn.Tanh(),
        layer_init(nn.Linear(256, 256)),
        nn.Tanh(),
        layer_init(nn.Linear(256, 256)),
        nn.Tanh(),
        layer_init(nn.Linear(256, 1)),
    ).to(device)

    # ── Logger ─────────────────────────────────────────────────────────
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )

    # ── Storage ────────────────────────────────────────────────────────
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape,
        device=device,
    )
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape,
        device=device,
    )
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    # ── Training loop ──────────────────────────────────────────────────
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    for iteration in range(1, args.num_iterations + 1):
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()

        # ── Evaluation ─────────────────────────────────────────────
        if iteration == 1 or iteration % args.eval_freq == 0 or iteration == args.num_iterations:
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.max_episode_steps):
                with torch.no_grad():
                    eval_obs, _, eval_term, eval_trunc, eval_infos = eval_envs.step(
                        agent.get_action(eval_obs, deterministic=False)
                    )
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v[mask])

            sr_vals = eval_metrics.get("success_once", [])
            if sr_vals:
                success_rate = torch.cat(sr_vals).float().mean().item()
            else:
                success_rate = 0.0

            print(
                f"Iter {iteration}/{args.num_iterations} | "
                f"step={global_step} | SR={success_rate:.1%} | "
                f"episodes={num_episodes}"
            )
            writer.add_scalar("eval/success_rate", success_rate, global_step)
            for k, v in eval_metrics.items():
                vals = torch.cat(v) if len(v) > 1 else v[0]
                writer.add_scalar(f"eval/{k}", vals.float().mean().item(), global_step)

            if args.save_model:
                os.makedirs(f"runs/{run_name}", exist_ok=True)
                torch.save(
                    agent.state_dict(),
                    f"runs/{run_name}/ckpt_{iteration}.pt",
                )

        # ── Rollout ────────────────────────────────────────────────
        rollout_t0 = time.time()
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(
                clip_action(action)
            )
            next_done = (terminations | truncations).float()
            rewards[step] = reward.view(-1) * args.reward_scale

            if "final_info" in infos:
                done_mask = infos["_final_info"]
                for k, v in infos["final_info"]["episode"].items():
                    writer.add_scalar(
                        f"train/{k}", v[done_mask].float().mean(), global_step
                    )
                with torch.no_grad():
                    final_values[
                        step,
                        torch.arange(args.num_envs, device=device)[done_mask],
                    ] = agent.get_value(
                        infos["final_observation"][done_mask]
                    ).view(-1)

            next_done = (terminations | truncations).float()

        rollout_time = time.time() - rollout_t0

        # ── Compute MC returns (GAE with λ=1.0) ───────────────────
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages_mc = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t]
                delta = rewards[t] + args.gamma * real_next_values - values[t]
                advantages_mc[t] = lastgaelam = (
                    delta + args.gamma * gae_lambda * next_not_done * lastgaelam
                )
            returns = advantages_mc + values  # MC returns (G_t)

        # ── Flatten batch ──────────────────────────────────────────
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # ── Regress Q and V on MC returns ──────────────────────────
        regression_t0 = time.time()
        N = args.batch_size

        # Re-initialize Q/V each iteration (learn from scratch on current data)
        q_net = QNetwork(obs_dim, act_dim).to(device)
        v_net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        ).to(device)

        q_opt = optim.Adam(q_net.parameters(), lr=args.regression_lr, eps=1e-5)
        v_opt = optim.Adam(v_net.parameters(), lr=args.regression_lr, eps=1e-5)

        reg_inds = np.arange(N)
        for reg_epoch in range(args.regression_epochs):
            np.random.shuffle(reg_inds)
            for start in range(0, N, args.regression_batch_size):
                mb = reg_inds[start:start + args.regression_batch_size]
                mb_obs = b_obs[mb]
                mb_act = b_actions[mb]
                mb_ret = b_returns[mb]

                # Q regression
                q_pred = q_net(mb_obs, mb_act).squeeze(-1)
                q_loss = 0.5 * ((q_pred - mb_ret) ** 2).mean()
                q_opt.zero_grad()
                q_loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), args.max_grad_norm)
                q_opt.step()

                # V regression
                v_pred = v_net(mb_obs).squeeze(-1)
                v_loss = 0.5 * ((v_pred - mb_ret) ** 2).mean()
                v_opt.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(v_net.parameters(), args.max_grad_norm)
                v_opt.step()

        q_net.eval()
        v_net.eval()
        regression_time = time.time() - regression_t0

        # Compute Q-V advantages
        with torch.no_grad():
            q_vals = q_net(b_obs, b_actions).squeeze(-1)
            v_vals = v_net(b_obs).squeeze(-1)
            b_advantages = q_vals - v_vals

        # Log regression quality
        writer.add_scalar("regression/q_loss", q_loss.item(), global_step)
        writer.add_scalar("regression/v_loss", v_loss.item(), global_step)
        writer.add_scalar("regression/q_mean", q_vals.mean().item(), global_step)
        writer.add_scalar("regression/v_mean", v_vals.mean().item(), global_step)
        writer.add_scalar("regression/adv_mean", b_advantages.mean().item(), global_step)
        writer.add_scalar("regression/adv_std", b_advantages.std().item(), global_step)
        writer.add_scalar("time/regression", regression_time, global_step)

        # ── Critic warmup: train only value function, skip policy ──
        if iteration <= args.warmup_iters:
            agent.train()
            update_t0 = time.time()
            perm = torch.randperm(args.batch_size, device=device)
            for epoch in range(args.update_epochs):
                for start in range(0, args.batch_size, args.minibatch_size):
                    mb_inds = perm[start : start + args.minibatch_size]
                    newvalue = agent.get_value(b_obs[mb_inds]).view(-1)
                    v_loss_critic = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    optimizer.zero_grad()
                    v_loss_critic.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
            update_time = time.time() - update_t0

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            print(
                f"  [warmup {iteration}/{args.warmup_iters}] "
                f"v_loss={v_loss_critic.item():.6f}, explained_var={explained_var:.4f}"
            )
            writer.add_scalar("losses/value_loss", v_loss_critic.item(), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("time/rollout", rollout_time, global_step)
            writer.add_scalar("time/update", update_time, global_step)
            continue

        # ── PPO update ─────────────────────────────────────────────
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_t0 = time.time()

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (Agent's critic — for bootstrapping next iteration)
                newvalue = newvalue.view(-1)
                v_loss_critic = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Total loss
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss_critic * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        update_time = time.time() - update_t0

        # ── Logging ────────────────────────────────────────────────
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("losses/value_loss", v_loss_critic.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("time/rollout", rollout_time, global_step)
        writer.add_scalar("time/update", update_time, global_step)

    # ── Final save ─────────────────────────────────────────────────────
    if args.save_model:
        torch.save(agent.state_dict(), f"runs/{run_name}/final_ckpt.pt")
        print(f"Final model saved to runs/{run_name}/final_ckpt.pt")

    writer.close()
    envs.close()
    eval_envs.close()
