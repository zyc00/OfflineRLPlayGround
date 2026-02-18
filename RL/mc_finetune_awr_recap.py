"""RECAP-style AWR finetuning with MC-estimated Q-V advantage — parallel re-rollout.

Same as mc_finetune_awr_parallel.py but replaces pure AWR loss with
RECAP-style advantage-conditioned loss: good actions (advantage > 0) get
weighted BC (maximize log prob), bad actions get reverse BC (minimize log prob).

Estimates advantage = Q^π*(s_t, a_t) - V^π*(s_t) via MC re-rollouts:
  - Q^π*(s_t, a_t): from s_t take a_t (current policy), then follow π* → MC return
  - V^π*(s_t): from s_t sample action from π*, then follow π* → MC return

No critic is trained. Actor-only RECAP AWR.

Usage:
  # MC5 RECAP AWR parallel (default)
  python -m RL.mc_finetune_awr_recap --mc_samples 5

  # Tuning temperature and recap_alpha
  python -m RL.mc_finetune_awr_recap --mc_samples 5 --awr_beta 0.5 --recap_alpha 1.0
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
from tqdm import tqdm
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
    optimal_checkpoint: str = "runs/pickcube_ppo/ckpt_301.pt"
    """optimal/converged policy for MC re-rollouts"""
    mc_samples: int = 5
    """number of MC re-rollout samples per (s,a) for Q and V each."""

    # Environment
    env_id: str = "PickCube-v1"
    num_envs: int = 512
    num_eval_envs: int = 128
    num_mc_envs: int = 0
    """MC re-rollout envs. 0 = auto (num_envs * 2 * mc_samples). Must be divisible by num_envs."""
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"
    max_episode_steps: int = 50

    # AWR hyperparameters (actor only)
    gamma: float = 0.8
    learning_rate: float = 3e-4
    num_steps: int = 50
    """rollout length per iteration"""
    num_minibatches: int = 32
    update_epochs: int = 4
    awr_beta: float = 1.0
    """AWR temperature. Lower = more greedy, higher = closer to uniform"""
    awr_max_weight: float = 20.0
    """AWR weight clamp upper bound to prevent exp explosion"""
    recap_alpha: float = 1.0
    """Weight for the negative (push-away) loss term. 0 = pure AWR, higher = stronger repulsion."""
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    reward_scale: float = 1.0

    # Training
    total_timesteps: int = 2_000_000
    eval_freq: int = 5
    """evaluate every N iterations"""
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

    # Auto-compute num_mc_envs if not set
    if args.num_mc_envs == 0:
        args.num_mc_envs = args.num_envs * 2 * args.mc_samples

    # Validate MC env allocation
    assert args.num_mc_envs % args.num_envs == 0, (
        f"num_mc_envs ({args.num_mc_envs}) must be divisible by num_envs ({args.num_envs})"
    )
    samples_per_env = args.num_mc_envs // args.num_envs
    assert samples_per_env >= 2 * args.mc_samples, (
        f"samples_per_env ({samples_per_env}) must be >= 2 * mc_samples ({2 * args.mc_samples}). "
        f"Increase num_mc_envs or decrease mc_samples."
    )

    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    if args.exp_name is None:
        args.exp_name = f"mc{args.mc_samples}_awr_recap"
    run_name = f"{args.exp_name}__seed{args.seed}__{int(time.time())}"

    print(f"=== RECAP AWR Finetuning (MC{args.mc_samples}, parallel re-rollout) ===")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Optimal policy: {args.optimal_checkpoint}")
    print(f"  Reward: {args.reward_mode}, gamma: {args.gamma}")
    print(f"  MC samples: {args.mc_samples}, samples_per_env: {samples_per_env}")
    print(f"  AWR beta: {args.awr_beta}, max_weight: {args.awr_max_weight}")
    print(f"  RECAP alpha: {args.recap_alpha}")
    print(f"  Envs: {args.num_envs}, MC envs: {args.num_mc_envs}, Steps: {args.num_steps}")
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
    # MC re-rollout dedicated envs (no video recording needed)
    mc_envs_raw = gym.make(args.env_id, num_envs=args.num_mc_envs, **env_kwargs)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
        mc_envs_raw = FlattenActionSpaceWrapper(mc_envs_raw)

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
    mc_envs = ManiSkillVectorEnv(
        mc_envs_raw, args.num_mc_envs,
        ignore_terminations=False,
        record_metrics=False,
    )

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)

    # ── Agent setup (actor only — critic exists but is not trained) ────
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"  Loaded checkpoint: {args.checkpoint}")

    # Load optimal policy for MC re-rollouts (frozen, not trained)
    optimal_agent = Agent(envs).to(device)
    optimal_agent.load_state_dict(torch.load(args.optimal_checkpoint, map_location=device))
    optimal_agent.eval()
    print(f"  Loaded optimal policy: {args.optimal_checkpoint}")

    # Only optimize actor parameters (actor_mean + actor_logstd)
    actor_params = list(agent.actor_mean.parameters()) + [agent.actor_logstd]
    optimizer = optim.Adam(actor_params, lr=args.learning_rate, eps=1e-5)
    print(f"  Actor params: {sum(p.numel() for p in actor_params):,}")

    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(a):
        return torch.clamp(a.detach(), action_low, action_high)

    # ── Rollout env helpers ────────────────────────────────────────────
    _zero_action = torch.zeros(
        args.num_envs, *envs.single_action_space.shape, device=device
    )

    def _clone_state(state_dict):
        """Deep clone a nested state dict of tensors."""
        if isinstance(state_dict, dict):
            return {k: _clone_state(v) for k, v in state_dict.items()}
        return state_dict.clone()

    def _restore_state(state_dict, seed=None):
        """Restore rollout env state with PhysX contact warmup."""
        envs.reset(seed=seed if seed is not None else args.seed)
        envs.base_env.set_state_dict(state_dict)
        envs.base_env.step(_zero_action)
        envs.base_env.set_state_dict(state_dict)
        envs.base_env._elapsed_steps[:] = 0
        return envs.base_env.get_obs()

    # ── MC env helpers ─────────────────────────────────────────────────
    _mc_zero_action = torch.zeros(
        args.num_mc_envs, *envs.single_action_space.shape, device=device
    )

    def _expand_state(state_dict, repeats):
        """Recursively repeat_interleave all tensors in state dict."""
        if isinstance(state_dict, dict):
            return {k: _expand_state(v, repeats) for k, v in state_dict.items()}
        if isinstance(state_dict, torch.Tensor) and state_dict.dim() > 0:
            return state_dict.repeat_interleave(repeats, dim=0)
        return state_dict

    def _restore_mc_state(state_dict, seed=None):
        """Restore MC env state with PhysX contact warmup."""
        mc_envs.reset(seed=seed if seed is not None else args.seed)
        mc_envs.base_env.set_state_dict(state_dict)
        mc_envs.base_env.step(_mc_zero_action)
        mc_envs.base_env.set_state_dict(state_dict)
        mc_envs.base_env._elapsed_steps[:] = 0
        return mc_envs.base_env.get_obs()

    # Precompute V-replica indices for gathering obs
    v_indices = []
    for i in range(args.num_envs):
        base = i * samples_per_env
        v_indices.extend(range(base + args.mc_samples, base + 2 * args.mc_samples))
    v_indices = torch.tensor(v_indices, device=device, dtype=torch.long)

    # ── Logger ─────────────────────────────────────────────────────────
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )

    # ── Storage (no logprobs needed for AWR) ───────────────────────────
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape,
        device=device,
    )
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape,
        device=device,
    )
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

        # ── Rollout (save states for re-rollout) ──────────────────
        rollout_t0 = time.time()
        saved_states = []
        for step in range(args.num_steps):
            global_step += args.num_envs
            saved_states.append(_clone_state(envs.base_env.get_state_dict()))
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action = agent.get_action(next_obs, deterministic=False)
            actions[step] = action

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

        rollout_time = time.time() - rollout_t0

        # ── Parallel MC re-rollout: estimate Q(s,a) and V(s) ─────
        end_state = _clone_state(envs.base_env.get_state_dict())
        end_elapsed = envs.base_env._elapsed_steps.clone()
        end_obs = next_obs.clone()
        end_done = next_done.clone()

        rerollout_t0 = time.time()
        mc_q = torch.zeros((args.num_steps, args.num_envs), device=device)
        mc_v = torch.zeros((args.num_steps, args.num_envs), device=device)

        with torch.no_grad():
            for t in tqdm(range(args.num_steps), desc=f"  MC re-rollout iter {iteration}", leave=False):
                # 1. Expand num_envs states → num_mc_envs
                expanded_state = _expand_state(saved_states[t], samples_per_env)

                # 2. Restore to mc_envs
                mc_obs = _restore_mc_state(expanded_state, seed=args.seed + t)

                # 3. Build first actions (num_mc_envs, action_dim)
                #    For each env i, replica j:
                #      j < mc_samples:                Q replica → use actions[t][i]
                #      mc_samples <= j < 2*mc_samples: V replica → sample from π*
                #      j >= 2*mc_samples:              idle → zero action
                first_actions = torch.zeros(
                    args.num_mc_envs, *envs.single_action_space.shape, device=device
                )

                # Q replicas: copy current policy's action
                for i in range(args.num_envs):
                    base = i * samples_per_env
                    first_actions[base : base + args.mc_samples] = actions[t][i]

                # V replicas: sample action from optimal policy
                v_obs = mc_obs[v_indices]
                first_actions[v_indices] = optimal_agent.get_action(
                    v_obs, deterministic=False
                )

                # 4. All num_mc_envs step first action in parallel
                mc_obs, rew, term, trunc, _ = mc_envs.step(clip_action(first_actions))
                all_rews = [rew.view(-1) * args.reward_scale]
                env_done = (term | trunc).view(-1).bool()

                # 5. Follow optimal policy until all done
                for _ in range(args.max_episode_steps - 1):
                    if env_done.all():
                        break
                    a = optimal_agent.get_action(mc_obs, deterministic=False)
                    mc_obs, rew, term, trunc, _ = mc_envs.step(clip_action(a))
                    all_rews.append(
                        rew.view(-1) * args.reward_scale * (~env_done).float()
                    )
                    env_done = env_done | (term | trunc).view(-1).bool()

                # 6. Compute discounted returns
                ret = torch.zeros(args.num_mc_envs, device=device)
                for s in reversed(range(len(all_rews))):
                    ret = all_rews[s] + args.gamma * ret

                # 7. Reshape (num_mc_envs,) → (num_envs, samples_per_env) and average
                ret = ret.view(args.num_envs, samples_per_env)
                mc_q[t] = ret[:, :args.mc_samples].mean(dim=1)
                mc_v[t] = ret[:, args.mc_samples : 2 * args.mc_samples].mean(dim=1)

        rerollout_time = time.time() - rerollout_t0

        # Restore end-of-rollout state for next iteration
        _restore_state(end_state, seed=args.seed)
        envs.base_env._elapsed_steps[:] = end_elapsed
        next_obs = end_obs
        next_done = end_done

        # Advantage = Q^π*(s_t, a_t) - V^π*(s_t)
        advantages = mc_q - mc_v

        writer.add_scalar("time/rerollout", rerollout_time, global_step)
        writer.add_scalar("charts/mc_q_mean", mc_q.mean().item(), global_step)
        writer.add_scalar("charts/mc_v_mean", mc_v.mean().item(), global_step)
        writer.add_scalar("charts/advantage_mean", advantages.mean().item(), global_step)
        writer.add_scalar("charts/advantage_std", advantages.std().item(), global_step)

        # ── Flatten batch ──────────────────────────────────────────
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)

        # ── RECAP AWR update (actor only) ──────────────────────────
        agent.train()
        b_inds = np.arange(args.batch_size)
        update_t0 = time.time()

        # Precompute: split into positive (good) and negative (bad) advantages
        if args.norm_adv:
            b_advantages_norm = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        else:
            b_advantages_norm = b_advantages

        # Positive mask: advantage > 0 (good actions to reinforce)
        b_positive = (b_advantages_norm > 0)

        # AWR weights for positive samples only
        b_pos_weights = torch.exp(b_advantages_norm / args.awr_beta)
        b_pos_weights = torch.clamp(b_pos_weights, max=args.awr_max_weight)

        # Negative weights: how bad is the action (larger magnitude = push harder)
        # Use |advantage| so worse actions get pushed away more
        b_neg_weights = torch.exp(b_advantages_norm.abs() / args.awr_beta)
        b_neg_weights = torch.clamp(b_neg_weights, max=args.awr_max_weight)

        # Logging stats
        pos_ratio = b_positive.float().mean().item()
        awr_pos_weight_mean = b_pos_weights[b_positive].mean().item() if b_positive.any() else 0.0
        awr_neg_weight_mean = b_neg_weights[~b_positive].mean().item() if (~b_positive).any() else 0.0

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, _ = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )

                mb_pos = b_positive[mb_inds]
                mb_neg = ~mb_pos

                # Positive loss: weighted BC on good actions (maximize log prob)
                if mb_pos.any():
                    mb_pw = b_pos_weights[mb_inds][mb_pos]
                    mb_pw = mb_pw / mb_pw.sum() * mb_pos.sum()  # normalize within positive
                    pos_loss = -(mb_pw.detach() * newlogprob[mb_pos]).mean()
                else:
                    pos_loss = torch.tensor(0.0, device=device)

                # Negative loss: push away bad actions (minimize log prob)
                if mb_neg.any():
                    mb_nw = b_neg_weights[mb_inds][mb_neg]
                    mb_nw = mb_nw / mb_nw.sum() * mb_neg.sum()  # normalize within negative
                    neg_loss = (mb_nw.detach() * newlogprob[mb_neg]).mean()
                else:
                    neg_loss = torch.tensor(0.0, device=device)

                # Combined loss
                policy_loss = pos_loss + args.recap_alpha * neg_loss

                entropy_loss = entropy.mean()
                loss = policy_loss - args.ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor_params, args.max_grad_norm)
                optimizer.step()

        update_time = time.time() - update_t0

        # ── Logging ────────────────────────────────────────────────
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/pos_loss", pos_loss.item(), global_step)
        writer.add_scalar("losses/neg_loss", neg_loss.item(), global_step)
        writer.add_scalar("charts/pos_ratio", pos_ratio, global_step)
        writer.add_scalar("losses/awr_pos_weight_mean", awr_pos_weight_mean, global_step)
        writer.add_scalar("losses/awr_neg_weight_mean", awr_neg_weight_mean, global_step)
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
    mc_envs.close()
