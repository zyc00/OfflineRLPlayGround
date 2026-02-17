"""PPO finetuning with IQL/SARSA advantage estimation.

Replaces GAE advantage computation with IQL-learned Q(s,a) and V(s).
With expectile_tau=0.5 (default), IQL reduces to SARSA:
  Q(s,a) = r + γ V(s')     (TD backup)
  V(s)   = E[Q(s,a)]       (mean, not expectile)
  A(s,a) = Q(s,a) - V(s)

The policy update is still PPO-style (clipped PG).

Usage:
  # SARSA (tau=0.5, default)
  python -m RL.iql_finetune

  # IQL with tau=0.7
  python -m RL.iql_finetune --expectile_tau 0.7

  # Match PPO best config
  python -m RL.iql_finetune --num_envs 50 --num_steps 100 --update_epochs 20
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
from methods.iql.iql import QNetwork, expectile_loss


@dataclass
class Args:
    # Finetuning
    checkpoint: str = "runs/pickcube_ppo/ckpt_101.pt"
    """pretrained checkpoint to finetune from"""

    # IQL / SARSA
    expectile_tau: float = 0.5
    """expectile for V loss (0.5 = SARSA / mean, >0.5 = IQL / optimistic)"""
    iql_epochs: int = 50
    """epochs to train Q and V per iteration on the rollout data"""
    iql_lr: float = 3e-4
    """learning rate for Q and V networks"""
    iql_batch_size: int = 256
    """minibatch size for IQL training"""
    tau_polyak: float = 0.005
    """Polyak averaging rate for target Q network"""
    warmup_iters: int = 3
    """iterations to train Q/V without updating the policy (critic warmup)"""
    pretrained_q: Optional[str] = None
    """pretrained Q network checkpoint (skips online IQL training)"""
    pretrained_v: Optional[str] = None
    """pretrained V network checkpoint (skips online IQL training)"""

    # Environment
    env_id: str = "PickCube-v1"
    num_envs: int = 512
    num_eval_envs: int = 128
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"
    max_episode_steps: int = 50

    # PPO hyperparameters (policy update only)
    gamma: float = 0.8
    learning_rate: float = 3e-4
    num_steps: int = 50
    num_minibatches: int = 32
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float = 0.1
    norm_adv: bool = True
    reward_scale: float = 1.0

    # Training
    total_timesteps: int = 2_000_000
    eval_freq: int = 5
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

    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    if args.exp_name is None:
        if args.expectile_tau == 0.5:
            args.exp_name = "sarsa_ppo"
        else:
            args.exp_name = f"iql_tau{args.expectile_tau}_ppo"
    run_name = f"{args.exp_name}__seed{args.seed}__{int(time.time())}"

    tau_str = f"tau={args.expectile_tau}" + (" (SARSA)" if args.expectile_tau == 0.5 else "")
    print(f"=== IQL + PPO Finetuning ({tau_str}) ===")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Reward: {args.reward_mode}")
    print(f"  IQL epochs: {args.iql_epochs}, IQL batch: {args.iql_batch_size}")
    print(f"  Envs: {args.num_envs}, Steps: {args.num_steps}")
    print(f"  Batch: {args.batch_size}, Minibatch: {args.minibatch_size}")
    print(f"  PPO update epochs: {args.update_epochs}")
    print(f"  Warmup iters: {args.warmup_iters}")
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

    # ── Agent setup (actor only used for policy) ───────────────────────
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"  Loaded checkpoint: {args.checkpoint}")

    # Only optimize the actor parameters with PPO
    actor_params = list(agent.actor_mean.parameters()) + [agent.actor_logstd]
    policy_optimizer = optim.Adam(actor_params, lr=args.learning_rate, eps=1e-5)

    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(a):
        return torch.clamp(a.detach(), action_low, action_high)

    # ── IQL networks (Q and V, separate from Agent) ────────────────────
    obs_dim = np.array(envs.single_observation_space.shape).prod()
    act_dim = np.prod(envs.single_action_space.shape)

    q_net = QNetwork(obs_dim, act_dim).to(device)
    q_target = copy.deepcopy(q_net)
    v_net = nn.Sequential(
        layer_init(nn.Linear(obs_dim, 256)),
        nn.Tanh(),
        layer_init(nn.Linear(256, 256)),
        nn.Tanh(),
        layer_init(nn.Linear(256, 256)),
        nn.Tanh(),
        layer_init(nn.Linear(256, 1)),
    ).to(device)

    use_pretrained_qv = args.pretrained_q is not None and args.pretrained_v is not None
    if use_pretrained_qv:
        q_net.load_state_dict(torch.load(args.pretrained_q, map_location=device))
        q_target = copy.deepcopy(q_net)
        v_net.load_state_dict(torch.load(args.pretrained_v, map_location=device))
        q_net.eval()
        v_net.eval()
        print(f"  Loaded pretrained Q: {args.pretrained_q}")
        print(f"  Loaded pretrained V: {args.pretrained_v}")
        q_optimizer = v_optimizer = None
    else:
        q_optimizer = optim.Adam(q_net.parameters(), lr=args.iql_lr, eps=1e-5)
        v_optimizer = optim.Adam(v_net.parameters(), lr=args.iql_lr, eps=1e-5)

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
    # IQL-specific: true next obs and done-after-action
    next_obs_buf = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape,
        device=device,
    )
    done_buf = torch.zeros((args.num_steps, args.num_envs), device=device)

    # ── Training loop ──────────────────────────────────────────────────
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    for iteration in range(1, args.num_iterations + 1):
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

            with torch.no_grad():
                action, logprob, _, _ = agent.get_action_and_value(next_obs)
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(
                clip_action(action)
            )
            done_flag = (terminations | truncations).float()
            rewards[step] = reward.view(-1) * args.reward_scale

            # Store true next observation and done flag
            next_obs_buf[step] = next_obs.clone()
            done_buf[step] = done_flag

            # For done envs, next_obs is already the reset obs.
            # Use final_observation as the true next state before reset.
            if "final_info" in infos:
                done_mask = infos["_final_info"]
                next_obs_buf[step, done_mask] = infos["final_observation"][done_mask]
                for k, v in infos["final_info"]["episode"].items():
                    writer.add_scalar(
                        f"train/{k}", v[done_mask].float().mean(), global_step
                    )

            next_done = done_flag

        rollout_time = time.time() - rollout_t0

        # ── Flatten rollout ────────────────────────────────────────
        b_obs = obs.reshape(-1, obs_dim)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_rewards = rewards.reshape(-1)
        b_next_obs = next_obs_buf.reshape(-1, obs_dim)
        b_done = done_buf.reshape(-1)
        b_logprobs = logprobs.reshape(-1)
        N = args.batch_size

        # ── Train IQL / SARSA on rollout data ──────────────────────
        avg_q_loss = 0.0
        avg_v_loss = 0.0
        iql_time = 0.0

        if not use_pretrained_qv:
            iql_t0 = time.time()
            q_net.train()
            v_net.train()
            iql_inds = np.arange(N)

            for iql_epoch in range(args.iql_epochs):
                np.random.shuffle(iql_inds)
                epoch_q_loss = 0.0
                epoch_v_loss = 0.0
                n_batches = 0

                for start in range(0, N, args.iql_batch_size):
                    mb = iql_inds[start : start + args.iql_batch_size]
                    s = b_obs[mb]
                    a = b_actions[mb]
                    r = b_rewards[mb]
                    ns = b_next_obs[mb]
                    d = b_done[mb]

                    # Q loss: TD backup Q(s,a) → r + γ V(s') (1 - done)
                    with torch.no_grad():
                        v_next = v_net(ns).squeeze(-1)
                        q_target_val = r + args.gamma * v_next * (1.0 - d)
                    q_pred = q_net(s, a).squeeze(-1)
                    q_loss = 0.5 * ((q_pred - q_target_val) ** 2).mean()

                    q_optimizer.zero_grad()
                    q_loss.backward()
                    nn.utils.clip_grad_norm_(q_net.parameters(), args.max_grad_norm)
                    q_optimizer.step()

                    # V loss: expectile regression against target Q
                    with torch.no_grad():
                        q_val = q_target(s, a).squeeze(-1)
                    v_pred = v_net(s).squeeze(-1)
                    v_loss = expectile_loss(q_val - v_pred, args.expectile_tau)

                    v_optimizer.zero_grad()
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(v_net.parameters(), args.max_grad_norm)
                    v_optimizer.step()

                    # Polyak update target Q
                    with torch.no_grad():
                        for p, pt in zip(q_net.parameters(), q_target.parameters()):
                            pt.data.mul_(1.0 - args.tau_polyak).add_(p.data, alpha=args.tau_polyak)

                    epoch_q_loss += q_loss.item()
                    epoch_v_loss += v_loss.item()
                    n_batches += 1

            q_net.eval()
            v_net.eval()
            iql_time = time.time() - iql_t0
            avg_q_loss = epoch_q_loss / max(n_batches, 1)
            avg_v_loss = epoch_v_loss / max(n_batches, 1)

        # ── Compute IQL advantages ─────────────────────────────────
        with torch.no_grad():
            q_vals = q_net(b_obs, b_actions).squeeze(-1)
            v_vals = v_net(b_obs).squeeze(-1)
            b_advantages = q_vals - v_vals
            b_returns = q_vals  # Q(s,a) as the return target

        # ── Skip policy update during warmup (only for online IQL) ─
        if not use_pretrained_qv and iteration <= args.warmup_iters:
            print(
                f"  [warmup {iteration}/{args.warmup_iters}] "
                f"q_loss={avg_q_loss:.6f}, v_loss={avg_v_loss:.6f}, "
                f"adv_mean={b_advantages.mean():.4f}, adv_std={b_advantages.std():.4f}"
            )
            writer.add_scalar("iql/q_loss", avg_q_loss, global_step)
            writer.add_scalar("iql/v_loss", avg_v_loss, global_step)
            writer.add_scalar("iql/q_mean", q_vals.mean().item(), global_step)
            writer.add_scalar("iql/v_mean", v_vals.mean().item(), global_step)
            writer.add_scalar("iql/adv_mean", b_advantages.mean().item(), global_step)
            writer.add_scalar("iql/adv_std", b_advantages.std().item(), global_step)
            writer.add_scalar("time/rollout", rollout_time, global_step)
            writer.add_scalar("time/iql", iql_time, global_step)
            continue

        # ── PPO policy update (actor only) ─────────────────────────
        agent.train()
        ppo_inds = np.arange(N)
        clipfracs = []
        update_t0 = time.time()

        for epoch in range(args.update_epochs):
            np.random.shuffle(ppo_inds)
            for start in range(0, N, args.minibatch_size):
                end = start + args.minibatch_size
                mb = ppo_inds[start:end]

                _, newlogprob, entropy, _ = agent.get_action_and_value(
                    b_obs[mb], b_actions[mb]
                )
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Clipped policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss

                policy_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor_params, args.max_grad_norm)
                policy_optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        update_time = time.time() - update_t0

        # ── Logging ────────────────────────────────────────────────
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("iql/q_loss", avg_q_loss, global_step)
        writer.add_scalar("iql/v_loss", avg_v_loss, global_step)
        writer.add_scalar("iql/q_mean", q_vals.mean().item(), global_step)
        writer.add_scalar("iql/v_mean", v_vals.mean().item(), global_step)
        writer.add_scalar("iql/adv_mean", b_advantages.mean().item(), global_step)
        writer.add_scalar("iql/adv_std", b_advantages.std().item(), global_step)
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("time/rollout", rollout_time, global_step)
        writer.add_scalar("time/iql", iql_time, global_step)
        writer.add_scalar("time/update", update_time, global_step)

    # ── Final save ─────────────────────────────────────────────────────
    if args.save_model:
        torch.save(agent.state_dict(), f"runs/{run_name}/final_ckpt.pt")
        torch.save(q_net.state_dict(), f"runs/{run_name}/q_net.pt")
        torch.save(v_net.state_dict(), f"runs/{run_name}/v_net.pt")
        print(f"Final model saved to runs/{run_name}/")

    writer.close()
    envs.close()
    eval_envs.close()
