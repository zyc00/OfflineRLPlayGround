"""Online iterative RL with replay buffer TD+EMA V learning + AWR policy update.

Tests whether accumulating a replay buffer for better V learning (TD+EMA)
can improve online iterative RL. Each iteration adds 5K transitions to the
replay buffer; by iteration 10 the critic trains on 50K transitions.

The critic is a standalone nn.Sequential with EMA target network, trained
via TD(0) in scaled reward space. Advantages are computed via GAE or
one-step using the unscaled V predictions.

Off-policy handling: "approximate ignore" — no IS correction. TD+EMA on
replay learns V^{pi_mix} ≈ V^{pi_current} since policy changes are small.

Baseline: GAE PPO γ=0.99 = 92.1% (V2 setting)

Usage:
  # GAE advantages (primary)
  python -u -m RL.td_ema_awr_online --gamma 0.99 --advantage_mode gae

  # One-step advantages
  python -u -m RL.td_ema_awr_online --gamma 0.99 --advantage_mode onestep

  # With offline pre-collection
  python -u -m RL.td_ema_awr_online --gamma 0.99 --advantage_mode gae --offline_rollouts 20
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
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from torch.utils.tensorboard import SummaryWriter

from data.data_collection.ppo import Agent, layer_init


@dataclass
class Args:
    # Finetuning
    checkpoint: str = "runs/pickcube_ppo/ckpt_76_logstd-1.5.pt"
    """pretrained checkpoint to finetune from"""

    # Environment
    env_id: str = "PickCube-v1"
    num_envs: int = 100
    num_eval_envs: int = 128
    num_steps: int = 50
    max_episode_steps: int = 50
    total_timesteps: int = 50000
    """total timesteps (10 iterations with default settings)"""
    eval_freq: int = 1
    gamma: float = 0.99
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"

    # TD+EMA V learning
    td_epochs_per_iter: int = 200
    td_target_steps: int = 0
    """If >0, dynamically set td_epochs = td_target_steps * td_batch_size / data_size
    (overrides td_epochs_per_iter). E.g. 7000 keeps ~7000 gradient steps regardless of data size."""
    td_reward_scale: float = 10.0
    ema_tau: float = 0.005
    td_lr: float = 3e-4
    td_batch_size: int = 1000
    critic_hidden_dim: int = 256
    critic_num_layers: int = 3

    # Offline pre-collection
    offline_rollouts: int = 0
    """Number of extra rollouts to collect with behavior policy before online
    iterations. Each rollout = num_envs * num_steps transitions. 0 = pure online."""
    current_data_only: bool = False
    """If True, only train critic on current iteration's on-policy data (no replay
    accumulation). Critic weights still persist (warm start) across iterations."""
    reset_critic_each_iter: bool = False
    """If True, re-initialize critic + EMA target from scratch each iteration."""

    # Advantage computation
    advantage_mode: Literal["gae", "onestep"] = "gae"
    gae_lambda: float = 0.95

    # Policy update
    policy_update: Literal["awr", "ppo"] = "ppo"
    awr_beta: float = 0.5
    awr_max_weight: float = 20.0
    clip_coef: float = 0.2
    target_kl: float = 100.0
    update_epochs: int = 200
    learning_rate: float = 3e-4
    num_minibatches: int = 5
    norm_adv: bool = True
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5

    # Other
    seed: int = 1
    cuda: bool = True
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
        parts = [f"td_ema_awr_{args.advantage_mode}"]
        if args.offline_rollouts > 0:
            parts.append(f"off{args.offline_rollouts}")
        args.exp_name = "_".join(parts)
    run_name = f"{args.exp_name}__seed{args.seed}__{int(time.time())}"

    print(f"=== Online TD+EMA AWR ({args.advantage_mode.upper()}) ===")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Reward: {args.reward_mode}, gamma: {args.gamma}")
    print(f"  TD+EMA: rs={args.td_reward_scale}, tau={args.ema_tau}, epochs/iter={args.td_epochs_per_iter}")
    print(f"  AWR: beta={args.awr_beta}, max_weight={args.awr_max_weight}, epochs={args.update_epochs}")
    print(f"  Envs: {args.num_envs}, Steps: {args.num_steps}")
    print(f"  Batch: {args.batch_size}, Minibatch: {args.minibatch_size}")
    print(f"  Iterations: {args.num_iterations}")
    if args.offline_rollouts > 0:
        print(f"  Offline pre-collection: {args.offline_rollouts} rollouts "
              f"({args.offline_rollouts * args.batch_size:,} transitions)")

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

    obs_dim = int(np.array(envs.single_observation_space.shape).prod())

    # ── Agent setup (actor only — critic is standalone) ────────────────
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"  Loaded checkpoint: {args.checkpoint}")

    # Only optimize actor parameters
    actor_params = list(agent.actor_mean.parameters()) + [agent.actor_logstd]
    optimizer = optim.Adam(actor_params, lr=args.learning_rate, eps=1e-5)
    print(f"  Actor params: {sum(p.numel() for p in actor_params):,}")

    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(a):
        return torch.clamp(a.detach(), action_low, action_high)

    # ── Standalone critic + EMA target ─────────────────────────────────
    def make_critic():
        layers = [layer_init(nn.Linear(obs_dim, args.critic_hidden_dim)), nn.Tanh()]
        for _ in range(args.critic_num_layers - 1):
            layers += [layer_init(nn.Linear(args.critic_hidden_dim, args.critic_hidden_dim)), nn.Tanh()]
        layers.append(layer_init(nn.Linear(args.critic_hidden_dim, 1)))
        return nn.Sequential(*layers).to(device)

    critic = make_critic()
    critic_target = make_critic()
    critic_target.load_state_dict(critic.state_dict())
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.td_lr, eps=1e-5)
    print(f"  Critic params: {sum(p.numel() for p in critic.parameters()):,}")

    def get_v_unscaled(obs_tensor):
        """Get V predictions in original (unscaled) reward space."""
        return critic(obs_tensor).view(-1) / args.td_reward_scale

    # ── Replay buffer ──────────────────────────────────────────────────
    replay_buffer = []  # list of dicts per rollout

    def prepare_td_data():
        """Concatenate all replay data into flat TD tuples."""
        all_obs = torch.cat([d["obs"] for d in replay_buffer], dim=1)  # (T, N*E, D)
        all_rewards = torch.cat([d["rewards"] for d in replay_buffer], dim=1)  # (T, N*E)
        all_dones = torch.cat([d["dones"] for d in replay_buffer], dim=1)  # (T, N*E)
        all_next_obs = torch.cat([d["next_obs"] for d in replay_buffer], dim=0)  # (N*E, D)
        all_next_done = torch.cat([d["next_done"] for d in replay_buffer], dim=0)  # (N*E,)

        T_steps, total_E, D = all_obs.shape
        rs = args.td_reward_scale

        # Build (s, r*rs, s', d) tuples
        flat_s = all_obs.reshape(-1, D)
        flat_r = all_rewards.reshape(-1) * rs

        # Next obs: obs[t+1] for t<T-1, next_obs for t=T-1
        flat_ns = torch.zeros_like(all_obs)
        flat_ns[:-1] = all_obs[1:]
        flat_ns[-1] = all_next_obs
        flat_ns = flat_ns.reshape(-1, D)

        # Next done: dones[t+1] for t<T-1, next_done for t=T-1
        flat_d = torch.zeros_like(all_rewards)
        flat_d[:-1] = all_dones[1:]
        flat_d[-1] = all_next_done
        flat_d = flat_d.reshape(-1)

        return flat_s, flat_r, flat_ns, flat_d

    # ── Logger ─────────────────────────────────────────────────────────
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )

    # ── Storage for current iteration ──────────────────────────────────
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

    # ── Offline pre-collection ─────────────────────────────────────────
    if args.offline_rollouts > 0:
        print(f"\nCollecting {args.offline_rollouts} offline rollouts...")
        off_t0 = time.time()
        for ri in range(args.offline_rollouts):
            off_obs = torch.zeros(
                args.num_steps, args.num_envs, obs_dim, device=device
            )
            off_rewards = torch.zeros(args.num_steps, args.num_envs, device=device)
            off_dones = torch.zeros(args.num_steps, args.num_envs, device=device)

            off_next_obs, _ = envs.reset(seed=args.seed + 10000 + ri)
            off_next_done = torch.zeros(args.num_envs, device=device)

            agent.eval()
            with torch.no_grad():
                for step in range(args.num_steps):
                    off_obs[step] = off_next_obs
                    off_dones[step] = off_next_done
                    action = agent.get_action(off_next_obs, deterministic=False)
                    off_next_obs, reward, term, trunc, _ = envs.step(clip_action(action))
                    off_next_done = (term | trunc).float()
                    off_rewards[step] = reward.view(-1)

            replay_buffer.append(dict(
                obs=off_obs,
                rewards=off_rewards,
                dones=off_dones,
                next_obs=off_next_obs.clone(),
                next_done=off_next_done.clone(),
            ))

            if (ri + 1) % 10 == 0 or ri + 1 == args.offline_rollouts:
                print(f"  {ri + 1}/{args.offline_rollouts} offline rollouts collected")

        off_time = time.time() - off_t0
        total_off = args.offline_rollouts * args.batch_size
        print(f"  Offline collection: {off_time:.1f}s, {total_off:,} transitions")

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
                        agent.get_action(eval_obs, deterministic=True)
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

            replay_trans = sum(
                d["obs"].shape[0] * d["obs"].shape[1] for d in replay_buffer
            )
            print(
                f"Iter {iteration}/{args.num_iterations} | "
                f"step={global_step} | SR={success_rate:.1%} | "
                f"episodes={num_episodes} | replay={replay_trans:,}"
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
        stored_final_obs = {}  # step -> (done_mask, final_observation)
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, _ = agent.get_action_and_value(next_obs)
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(
                clip_action(action)
            )
            next_done = (terminations | truncations).float()
            rewards[step] = reward.view(-1)

            if "final_info" in infos:
                done_mask = infos["_final_info"]
                for k, v in infos["final_info"]["episode"].items():
                    writer.add_scalar(
                        f"train/{k}", v[done_mask].float().mean(), global_step
                    )
                if done_mask.any():
                    stored_final_obs[step] = (
                        done_mask.clone(),
                        infos["final_observation"][done_mask].clone(),
                    )

        rollout_time = time.time() - rollout_t0

        # ── Append to replay buffer ────────────────────────────────
        if args.current_data_only:
            replay_buffer.clear()
        replay_buffer.append(dict(
            obs=obs.clone(),
            rewards=rewards.clone(),
            dones=dones.clone(),
            next_obs=next_obs.clone(),
            next_done=next_done.clone(),
        ))

        # ── TD+EMA training on full replay ─────────────────────────
        if args.reset_critic_each_iter:
            critic = make_critic()
            critic_target = make_critic()
            critic_target.load_state_dict(critic.state_dict())
            critic_optimizer = optim.Adam(critic.parameters(), lr=args.td_lr, eps=1e-5)

        td_t0 = time.time()
        flat_s, flat_r, flat_ns, flat_d = prepare_td_data()
        N = flat_s.shape[0]
        mb = min(args.td_batch_size, N)
        num_batches = max(1, N // mb)

        if args.td_target_steps > 0:
            td_epochs = max(1, args.td_target_steps // num_batches)
        else:
            td_epochs = args.td_epochs_per_iter

        critic.train()
        for td_epoch in range(td_epochs):
            perm = torch.randperm(N, device=device)
            for start in range(0, N, mb):
                idx = perm[start:start + mb]
                with torch.no_grad():
                    target = flat_r[idx] + args.gamma * critic_target(flat_ns[idx]).view(-1) * (1 - flat_d[idx])
                v_pred = critic(flat_s[idx]).view(-1)
                v_loss = 0.5 * ((v_pred - target) ** 2).mean()

                critic_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                critic_optimizer.step()

                with torch.no_grad():
                    for p, pt in zip(critic.parameters(), critic_target.parameters()):
                        pt.data.mul_(1 - args.ema_tau).add_(p.data, alpha=args.ema_tau)

        td_time = time.time() - td_t0

        # Free flat TD data
        del flat_s, flat_r, flat_ns, flat_d

        # ── Compute final_values with UPDATED critic ───────────────
        # For envs that terminated mid-rollout, obs[t+1] is the reset obs.
        # final_values corrects GAE by providing V(true_final_obs).
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        critic.eval()
        with torch.no_grad():
            for step, (mask, fobs) in stored_final_obs.items():
                final_values[
                    step,
                    torch.arange(args.num_envs, device=device)[mask],
                ] = get_v_unscaled(fobs)

        # ── Compute advantages ─────────────────────────────────────
        critic.eval()
        with torch.no_grad():
            # V predictions for current iteration data (unscaled)
            values = torch.zeros(args.num_steps, args.num_envs, device=device)
            for t in range(args.num_steps):
                values[t] = get_v_unscaled(obs[t])
            next_value = get_v_unscaled(next_obs).reshape(1, -1)

        if args.advantage_mode == "gae":
            with torch.no_grad():
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
        else:
            # One-step: A(s,a) = r + gamma * V(s') - V(s)
            with torch.no_grad():
                advantages = torch.zeros_like(rewards)
                for t in range(args.num_steps):
                    if t == args.num_steps - 1:
                        next_not_done = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        next_not_done = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    real_next_values = next_not_done * nextvalues + final_values[t]
                    advantages[t] = rewards[t] + args.gamma * real_next_values - values[t]

        # Log advantage and V stats
        returns = advantages + values
        b_vals = values.reshape(-1).cpu().numpy()
        b_rets = returns.reshape(-1).cpu().numpy()
        var_y = np.var(b_rets)
        explained_var = np.nan if var_y == 0 else 1 - np.var(b_rets - b_vals) / var_y

        writer.add_scalar("charts/advantage_mean", advantages.mean().item(), global_step)
        writer.add_scalar("charts/advantage_std", advantages.std().item(), global_step)
        writer.add_scalar("charts/advantage_pos_frac", (advantages > 0).float().mean().item(), global_step)
        writer.add_scalar("charts/v_mean", values.mean().item(), global_step)
        writer.add_scalar("charts/v_std", values.std().item(), global_step)
        writer.add_scalar("losses/td_v_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/td_epochs", td_epochs, global_step)
        writer.add_scalar("time/td_train", td_time, global_step)
        writer.add_scalar("charts/replay_transitions",
                          sum(d["obs"].shape[0] * d["obs"].shape[1] for d in replay_buffer),
                          global_step)

        # ── Flatten batch ──────────────────────────────────────────
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)

        # ── Policy update ──────────────────────────────────────────
        agent.train()
        b_inds = np.arange(args.batch_size)
        update_t0 = time.time()

        if args.policy_update == "awr":
            # Precompute AWR weights
            if args.norm_adv:
                b_adv_norm = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            else:
                b_adv_norm = b_advantages
            b_weights = torch.clamp(torch.exp(b_adv_norm / args.awr_beta), max=args.awr_max_weight)

            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    mb_inds = b_inds[start:start + args.minibatch_size]
                    _, newlogprob, entropy, _ = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    mb_weights = b_weights[mb_inds]
                    mb_weights = mb_weights / mb_weights.sum() * len(mb_weights)
                    policy_loss = -(mb_weights.detach() * newlogprob).mean()
                    entropy_loss = entropy.mean()
                    loss = policy_loss - args.ent_coef * entropy_loss
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(actor_params, args.max_grad_norm)
                    optimizer.step()

        else:  # PPO
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    mb_inds = b_inds[start:start + args.minibatch_size]
                    _, newlogprob, entropy, _ = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs.append(
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        )

                    if approx_kl > args.target_kl:
                        break

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                    entropy_loss = entropy.mean()
                    loss = policy_loss - args.ent_coef * entropy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(actor_params, args.max_grad_norm)
                    optimizer.step()

                if approx_kl > args.target_kl:
                    break

            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)

        update_time = time.time() - update_t0

        # ── Logging ────────────────────────────────────────────────
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("time/rollout", rollout_time, global_step)
        writer.add_scalar("time/update", update_time, global_step)

    # ── Final save ─────────────────────────────────────────────────────
    if args.save_model:
        torch.save(agent.state_dict(), f"runs/{run_name}/final_ckpt.pt")
        torch.save(critic.state_dict(), f"runs/{run_name}/critic_final.pt")
        print(f"Final model saved to runs/{run_name}/")

    writer.close()
    envs.close()
    eval_envs.close()
