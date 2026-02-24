"""Online iterative RL with IQL (Q-V) advantage estimation + PPO policy update.

Tests whether IQL-style advantage estimation (A = Q(s,a) - V(s)) can work
in the online iterative setting. With expectile_tau=0.5, IQL degenerates to
standard TD, making Q(s,a)-V(s) ≈ r + γV(s') - V(s) (one-step advantage).
With tau>0.5, V learns an upper expectile of Q, giving pessimistic advantages.

Compared to td_ema_awr_online.py (onestep mode), the key difference is that
advantages come from Q(s,a)-V(s) instead of r+γV(s')-V(s). These should be
equivalent at tau=0.5, but IQL allows varying tau to test asymmetric V learning.

Usage:
  # IQL tau=0.5 (should ≈ TD+EMA onestep)
  python -u -m RL.iql_ppo_online --expectile_tau 0.5

  # IQL tau=0.7 (standard IQL)
  python -u -m RL.iql_ppo_online --expectile_tau 0.7

  # With replay accumulation
  python -u -m RL.iql_ppo_online --expectile_tau 0.5
"""

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
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from torch.utils.tensorboard import SummaryWriter

from data.data_collection.ppo import Agent, layer_init


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """Asymmetric squared loss: |tau - 1(diff < 0)| * diff^2."""
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return (weight * (diff ** 2)).mean()


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

    # IQL Q+V learning
    iql_epochs_per_iter: int = 1400
    iql_reward_scale: float = 10.0
    ema_tau: float = 0.005
    """Polyak averaging rate for Q target network"""
    expectile_tau: float = 0.5
    """Expectile for V loss. 0.5 = mean (≈ standard TD), >0.5 = upper expectile (IQL proper)"""
    iql_td_n: int = 10
    """N-step TD for Q target. 1=standard TD(0), 10=10-step."""
    iql_lr: float = 3e-4
    iql_batch_size: int = 1000
    hidden_dim: int = 256
    num_layers: int = 3

    # Replay / critic options
    current_data_only: bool = False
    """If True, only train Q/V on current iteration's on-policy data (no replay)."""
    reset_critic_each_iter: bool = True
    """If True, re-initialize Q, V, Q_target from scratch each iteration."""
    advantage_mode: str = "qv"
    """'qv' = Q(s,a)-V(s), 'onestep' = r+γV(s')-V(s), 'gae' = GAE with IQL's V"""
    gae_lambda: float = 0.95

    # PPO policy update
    clip_coef: float = 0.2
    target_kl: float = 100.0
    update_epochs: int = 100
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
        args.exp_name = f"iql_ppo_tau{args.expectile_tau}"
    run_name = f"{args.exp_name}__seed{args.seed}__{int(time.time())}"

    print(f"=== Online IQL + PPO (tau={args.expectile_tau}) ===")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Reward: {args.reward_mode}, gamma: {args.gamma}")
    print(f"  IQL: rs={args.iql_reward_scale}, ema_tau={args.ema_tau}, "
          f"expectile_tau={args.expectile_tau}, td_n={args.iql_td_n}, epochs/iter={args.iql_epochs_per_iter}")
    print(f"  PPO: clip={args.clip_coef}, epochs={args.update_epochs}")
    print(f"  Envs: {args.num_envs}, Steps: {args.num_steps}")
    print(f"  Batch: {args.batch_size}, Minibatch: {args.minibatch_size}")
    print(f"  Iterations: {args.num_iterations}")
    print(f"  Reset critic each iter: {args.reset_critic_each_iter}")
    print(f"  Current data only: {args.current_data_only}")

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
    act_dim = int(np.array(envs.single_action_space.shape).prod())

    # ── Agent setup (actor only) ───────────────────────────────────────
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"  Loaded checkpoint: {args.checkpoint}")

    actor_params = list(agent.actor_mean.parameters()) + [agent.actor_logstd]
    optimizer = optim.Adam(actor_params, lr=args.learning_rate, eps=1e-5)
    print(f"  Actor params: {sum(p.numel() for p in actor_params):,}")

    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(a):
        return torch.clamp(a.detach(), action_low, action_high)

    # ── IQL networks: Q(s,a), V(s), Q_target(s,a) ─────────────────────
    def make_v_net():
        layers = [layer_init(nn.Linear(obs_dim, args.hidden_dim)), nn.Tanh()]
        for _ in range(args.num_layers - 1):
            layers += [layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)), nn.Tanh()]
        layers.append(layer_init(nn.Linear(args.hidden_dim, 1)))
        return nn.Sequential(*layers).to(device)

    def make_q_net():
        layers = [layer_init(nn.Linear(obs_dim + act_dim, args.hidden_dim)), nn.Tanh()]
        for _ in range(args.num_layers - 1):
            layers += [layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)), nn.Tanh()]
        layers.append(layer_init(nn.Linear(args.hidden_dim, 1)))
        return nn.Sequential(*layers).to(device)

    class QNet(nn.Module):
        """Wrapper to call q_net(obs, action) -> scalar."""
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, obs, action):
            return self.net(torch.cat([obs, action], dim=-1))

    q_net = QNet(make_q_net()).to(device)
    q_target = QNet(make_q_net()).to(device)
    q_target.load_state_dict(q_net.state_dict())
    v_net = make_v_net()

    q_optimizer = optim.Adam(q_net.parameters(), lr=args.iql_lr, eps=1e-5)
    v_optimizer = optim.Adam(v_net.parameters(), lr=args.iql_lr, eps=1e-5)
    print(f"  Q params: {sum(p.numel() for p in q_net.parameters()):,}")
    print(f"  V params: {sum(p.numel() for p in v_net.parameters()):,}")

    # ── Replay buffer (now includes actions) ───────────────────────────
    replay_buffer = []

    def prepare_iql_data():
        """Concatenate all replay data into flat tensors with n-step returns."""
        all_obs = torch.cat([d["obs"] for d in replay_buffer], dim=1)
        all_actions = torch.cat([d["actions"] for d in replay_buffer], dim=1)
        all_rewards = torch.cat([d["rewards"] for d in replay_buffer], dim=1)
        all_dones = torch.cat([d["dones"] for d in replay_buffer], dim=1)
        all_next_obs = torch.cat([d["next_obs"] for d in replay_buffer], dim=0)
        all_next_done = torch.cat([d["next_done"] for d in replay_buffer], dim=0)

        Tl, El, D = all_obs.shape
        rs = args.iql_reward_scale
        n = args.iql_td_n

        flat_s = all_obs.reshape(-1, D)
        flat_a = all_actions.reshape(-1, act_dim)

        # Precompute n-step returns and bootstrap info
        nstep_ret = torch.zeros(Tl, El, device=device)
        nstep_boot_obs = torch.zeros(Tl, El, D, device=device)
        nstep_boot_mask = torch.zeros(Tl, El, device=device)

        for t in range(Tl):
            cumul = torch.zeros(El, device=device)
            discount = torch.ones(El, device=device)
            alive = torch.ones(El, device=device)
            for k in range(n):
                step = t + k
                if step < Tl:
                    cumul += discount * alive * all_rewards[step]
                    if k < n - 1 and step + 1 < Tl:
                        alive *= (1.0 - all_dones[step + 1])
                    elif k < n - 1 and step + 1 == Tl:
                        alive *= (1.0 - all_next_done)
                    discount *= args.gamma
                else:
                    break
            nstep_ret[t] = cumul * rs
            boot_step = min(t + n, Tl)
            if boot_step < Tl:
                nstep_boot_obs[t] = all_obs[boot_step]
            else:
                nstep_boot_obs[t] = all_next_obs
            nstep_boot_mask[t] = discount * alive

        flat_nstep_ret = nstep_ret.reshape(-1)
        flat_boot_obs = nstep_boot_obs.reshape(-1, D)
        flat_boot_mask = nstep_boot_mask.reshape(-1)

        return flat_s, flat_a, flat_nstep_ret, flat_boot_obs, flat_boot_mask

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
        stored_final_obs = {}
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

        # ── Append to replay buffer (includes actions) ────────────
        if args.current_data_only:
            replay_buffer.clear()
        replay_buffer.append(dict(
            obs=obs.clone(),
            actions=actions.clone(),
            rewards=rewards.clone(),
            dones=dones.clone(),
            next_obs=next_obs.clone(),
            next_done=next_done.clone(),
        ))

        # ── IQL training: Q + V on replay data ────────────────────
        if args.reset_critic_each_iter:
            q_net = QNet(make_q_net()).to(device)
            q_target = QNet(make_q_net()).to(device)
            q_target.load_state_dict(q_net.state_dict())
            v_net = make_v_net()
            q_optimizer = optim.Adam(q_net.parameters(), lr=args.iql_lr, eps=1e-5)
            v_optimizer = optim.Adam(v_net.parameters(), lr=args.iql_lr, eps=1e-5)

        iql_t0 = time.time()
        flat_s, flat_a, flat_nstep_ret, flat_boot_obs, flat_boot_mask = prepare_iql_data()
        N = flat_s.shape[0]
        mb = min(args.iql_batch_size, N)

        q_net.train()
        v_net.train()
        for iql_epoch in range(args.iql_epochs_per_iter):
            perm = torch.randperm(N, device=device)
            for start in range(0, N, mb):
                idx = perm[start:start + mb]
                s_b, a_b = flat_s[idx], flat_a[idx]

                # Q loss: Q(s,a) → nstep_ret + γ^n * V(s_{t+n}) * alive_mask
                with torch.no_grad():
                    v_boot = v_net(flat_boot_obs[idx]).view(-1)
                    q_target_val = flat_nstep_ret[idx] + flat_boot_mask[idx] * v_boot
                q_pred = q_net(s_b, a_b).view(-1)
                q_loss = 0.5 * ((q_pred - q_target_val) ** 2).mean()

                q_optimizer.zero_grad()
                q_loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), args.max_grad_norm)
                q_optimizer.step()

                # V loss: expectile regression on Q_target(s,a) - V(s)
                with torch.no_grad():
                    q_val = q_target(s_b, a_b).view(-1)
                v_pred = v_net(s_b).view(-1)
                v_loss = expectile_loss(q_val - v_pred, args.expectile_tau)

                v_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(v_net.parameters(), args.max_grad_norm)
                v_optimizer.step()

                # Polyak update Q_target
                with torch.no_grad():
                    for p, pt in zip(q_net.parameters(), q_target.parameters()):
                        pt.data.mul_(1 - args.ema_tau).add_(p.data, alpha=args.ema_tau)

        iql_time = time.time() - iql_t0

        del flat_s, flat_a, flat_nstep_ret, flat_boot_obs, flat_boot_mask

        # ── Compute advantages on current data ────────────────────
        q_net.eval()
        v_net.eval()
        rs = args.iql_reward_scale
        with torch.no_grad():
            advantages = torch.zeros(args.num_steps, args.num_envs, device=device)
            values = torch.zeros(args.num_steps, args.num_envs, device=device)
            if args.advantage_mode == "qv":
                # A = Q(s,a) - V(s)
                for t in range(args.num_steps):
                    q_val = q_net(obs[t], actions[t]).view(-1) / rs
                    v_val = v_net(obs[t]).view(-1) / rs
                    advantages[t] = q_val - v_val
                    values[t] = v_val
            elif args.advantage_mode == "onestep":
                # A = r + γV(s') - V(s), using IQL's V
                for t in range(args.num_steps):
                    v_val = v_net(obs[t]).view(-1) / rs
                    values[t] = v_val
                    if t < args.num_steps - 1:
                        v_next = v_net(obs[t + 1]).view(-1) / rs
                        d_next = dones[t + 1]
                    else:
                        v_next = v_net(next_obs).view(-1) / rs
                        d_next = next_done
                    advantages[t] = rewards[t] + args.gamma * v_next * (1 - d_next) - v_val
            elif args.advantage_mode == "gae":
                # GAE with IQL's V
                for t in range(args.num_steps):
                    values[t] = v_net(obs[t]).view(-1) / rs
                v_next_final = v_net(next_obs).view(-1) / rs
                lastgaelam = torch.zeros(args.num_envs, device=device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = v_next_final
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    advantages[t] = lastgaelam
            else:
                raise ValueError(f"Unknown advantage_mode: {args.advantage_mode}")

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
        writer.add_scalar("losses/q_loss", q_loss.item(), global_step)
        writer.add_scalar("losses/v_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/iql_epochs", args.iql_epochs_per_iter, global_step)
        writer.add_scalar("time/iql_train", iql_time, global_step)
        writer.add_scalar("charts/replay_transitions",
                          sum(d["obs"].shape[0] * d["obs"].shape[1] for d in replay_buffer),
                          global_step)

        # ── Flatten batch ──────────────────────────────────────────
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)

        # ── PPO policy update ──────────────────────────────────────
        agent.train()
        b_inds = np.arange(args.batch_size)
        update_t0 = time.time()

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

        update_time = time.time() - update_t0

        # ── Logging ────────────────────────────────────────────────
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("time/rollout", rollout_time, global_step)
        writer.add_scalar("time/update", update_time, global_step)

    # ── Final save ─────────────────────────────────────────────────────
    if args.save_model:
        torch.save(agent.state_dict(), f"runs/{run_name}/final_ckpt.pt")
        torch.save(q_net.state_dict(), f"runs/{run_name}/q_final.pt")
        torch.save(v_net.state_dict(), f"runs/{run_name}/v_final.pt")
        print(f"Final model saved to runs/{run_name}/")

    writer.close()
    envs.close()
    eval_envs.close()
