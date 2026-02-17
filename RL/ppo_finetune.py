"""PPO finetuning with sparse reward for offline-to-online RL.

Finetunes a pretrained policy checkpoint using online PPO.
Supports GAE, MC1, and MC_M (M>1, with state save/restore) advantage estimation.

Usage:
  # GAE (default)
  python -m RL.ppo_finetune

  # MC1
  python -m RL.ppo_finetune --advantage_mode mc --exp_name ppo_mc

  # MC3 (3 re-rollouts per state-action pair)
  python -m RL.ppo_finetune --mc_samples 3

  # 1-env (real-world simulation)
  python -m RL.ppo_finetune --num_envs 1 --num_minibatches 1
"""

import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Optional

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

from data.data_collection.ppo import Agent


@dataclass
class Args:
    # Finetuning
    checkpoint: str = "runs/pickcube_ppo/ckpt_101.pt"
    """pretrained checkpoint to finetune from"""
    advantage_mode: Literal["gae", "mc"] = "gae"
    """advantage estimation: 'gae' (lambda=0.9) or 'mc' (lambda=1.0, pure MC)"""
    mc_samples: int = 1
    """number of MC rollout samples per (s,a). >1 uses state save/restore for re-rollouts."""
    reset_critic: bool = True
    """reset critic weights (needed when finetuning with different reward mode)"""
    critic_checkpoint: Optional[str] = None
    """pretrained critic checkpoint (overrides reset_critic)"""

    # Environment
    env_id: str = "PickCube-v1"
    num_envs: int = 512
    num_eval_envs: int = 128
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"
    max_episode_steps: int = 50

    # PPO hyperparameters
    gamma: float = 0.8
    gae_lambda: float = 0.9
    """GAE lambda (overridden to 1.0 when advantage_mode='mc')"""
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
    """iterations to train critic only (no policy update). Useful with reset_critic."""
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

    # Override gae_lambda for MC mode (including mc_samples > 1)
    if args.advantage_mode == "mc" or args.mc_samples > 1:
        args.gae_lambda = 1.0
        args.advantage_mode = "mc"

    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    if args.exp_name is None:
        if args.mc_samples > 1:
            args.exp_name = f"ppo_mc{args.mc_samples}"
        else:
            args.exp_name = f"ppo_{args.advantage_mode}"
    run_name = f"{args.exp_name}__seed{args.seed}__{int(time.time())}"

    mode_str = f"MC{args.mc_samples}" if args.mc_samples > 1 else args.advantage_mode.upper()
    print(f"=== PPO Finetuning ({mode_str}) ===")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Reward: {args.reward_mode}")
    print(f"  GAE lambda: {args.gae_lambda}, MC samples: {args.mc_samples}")
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
        # Load pretrained critic for sparse reward
        critic_state = torch.load(args.critic_checkpoint, map_location=device)
        agent.critic.load_state_dict(critic_state)
        print(f"  Loaded pretrained critic: {args.critic_checkpoint}")
    elif args.reset_critic:
        # Reinitialize critic — pretrained critic learned dense reward values,
        # which are wrong for sparse reward and cause policy collapse.
        from data.data_collection.ppo import layer_init
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

    # ── MC re-rollout helpers (for mc_samples > 1) ────────────────────
    use_mc_rerollout = args.mc_samples > 1

    if use_mc_rerollout:
        _zero_action = torch.zeros(
            args.num_envs, *envs.single_action_space.shape, device=device
        )

        def _clone_state(state_dict):
            """Deep clone a nested state dict of tensors."""
            if isinstance(state_dict, dict):
                return {k: _clone_state(v) for k, v in state_dict.items()}
            return state_dict.clone()

        def _restore_state(state_dict, seed=None):
            """Restore env state with PhysX contact warmup."""
            envs.reset(seed=seed if seed is not None else args.seed)
            envs.base_env.set_state_dict(state_dict)
            envs.base_env.step(_zero_action)
            envs.base_env.set_state_dict(state_dict)
            envs.base_env._elapsed_steps[:] = 0
            return envs.base_env.get_obs()

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
        if use_mc_rerollout:
            saved_states = []
        for step in range(args.num_steps):
            global_step += args.num_envs
            if use_mc_rerollout:
                saved_states.append(_clone_state(envs.base_env.get_state_dict()))
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

        rollout_time = time.time() - rollout_t0

        # ── Compute advantages ─────────────────────────────────────
        if use_mc_rerollout:
            # MC_M: re-rollout from each (s_t, a_t) M times via state save/restore
            end_state = _clone_state(envs.base_env.get_state_dict())
            end_elapsed = envs.base_env._elapsed_steps.clone()
            end_obs = next_obs.clone()
            end_done = next_done.clone()

            rerollout_t0 = time.time()
            mc_returns = torch.zeros((args.num_steps, args.num_envs), device=device)

            with torch.no_grad():
                for t in range(args.num_steps):
                    step_rets = []
                    for m in range(args.mc_samples):
                        obs_r = _restore_state(saved_states[t], seed=args.seed + m)

                        # Take the same action as in the original rollout
                        obs_r, rew, term, trunc, _ = envs.step(clip_action(actions[t]))
                        ep_rews = [rew.view(-1) * args.reward_scale]
                        env_done = (term | trunc).view(-1).bool()

                        # Follow policy until all envs done
                        for _ in range(args.max_episode_steps - 1):
                            if env_done.all():
                                break
                            a = agent.get_action(obs_r, deterministic=False)
                            obs_r, rew, term, trunc, _ = envs.step(clip_action(a))
                            ep_rews.append(
                                rew.view(-1) * args.reward_scale * (~env_done).float()
                            )
                            env_done = env_done | (term | trunc).view(-1).bool()

                        # Discounted return (backward sum)
                        ret = torch.zeros(args.num_envs, device=device)
                        for s in reversed(range(len(ep_rews))):
                            ret = ep_rews[s] + args.gamma * ret
                        step_rets.append(ret)

                    mc_returns[t] = torch.stack(step_rets).mean(dim=0)

            rerollout_time = time.time() - rerollout_t0

            # Restore end-of-rollout state for next iteration
            _restore_state(end_state, seed=args.seed)
            envs.base_env._elapsed_steps[:] = end_elapsed
            next_obs = end_obs
            next_done = end_done

            advantages = mc_returns - values
            returns = mc_returns

            writer.add_scalar("time/rerollout", rerollout_time, global_step)
        else:
            # Standard GAE/MC1 advantage computation
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards)
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
                    advantages[t] = lastgaelam = (
                        delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam
                    )
                returns = advantages + values

        # ── Flatten batch ──────────────────────────────────────────
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # ── Critic warmup: train only value function, skip policy ──
        if iteration <= args.warmup_iters:
            agent.train()
            update_t0 = time.time()
            for epoch in range(args.update_epochs):
                perm = torch.randperm(args.batch_size, device=device)
                for start in range(0, args.batch_size, args.minibatch_size):
                    mb_inds = perm[start : start + args.minibatch_size]
                    newvalue = agent.get_value(b_obs[mb_inds]).view(-1)
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    optimizer.zero_grad()
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
            update_time = time.time() - update_t0

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            print(
                f"  [warmup {iteration}/{args.warmup_iters}] "
                f"v_loss={v_loss.item():.6f}, explained_var={explained_var:.4f}"
            )
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
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

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Total loss
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

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

        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
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
