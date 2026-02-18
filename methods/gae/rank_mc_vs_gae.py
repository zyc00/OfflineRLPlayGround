"""Compare MC vs GAE vs IQL advantage ranking for on-policy sampled actions.

For each state in the eval dataset, sample K actions from the policy,
estimate advantages via MC (ground-truth rollouts), GAE (with a value
function supervised on MC returns), and IQL (offline Q-learning), then
compare the action rankings across all three methods.
"""

import math
import os
import random
from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tyro
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from scipy import stats as sp_stats
from torch.distributions import Normal
from tqdm import tqdm

from data.data_collection.ppo import Agent
from data.offline_dataset import OfflineRLDataset
from methods.gae.gae_online import Critic, _make_obs
from methods.iql.iql import QNetwork, train_iql, compute_nstep_targets
from methods.iql.iql import Args as IQLArgs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _replicate_state(state_dict, n):
    """Replicate a (1, ...) state dict to (n, ...)."""
    if isinstance(state_dict, dict):
        return {k: _replicate_state(v, n) for k, v in state_dict.items()}
    return state_dict.repeat(n, *([1] * (state_dict.ndim - 1)))


def _batched_forward(critic, obs, device, batch_size=4096):
    """Run critic on a flat obs tensor in batches, return CPU values."""
    N = obs.shape[0] if not isinstance(obs, dict) else next(iter(obs.values())).shape[0]
    values = torch.zeros(N)
    critic.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            if isinstance(obs, dict):
                batch = {k: v[start:end].to(device) for k, v in obs.items()}
            else:
                batch = obs[start:end].to(device)
            values[start:end] = critic(batch).squeeze(-1).cpu()
    return values


def _compute_mc_returns(rewards, gamma):
    """Backward pass: compute discounted MC returns for a trajectory."""
    T = rewards.shape[0]
    mc_returns = torch.zeros(T)
    running = 0.0
    for t in reversed(range(T)):
        running = rewards[t].item() + gamma * running
        mc_returns[t] = running
    return mc_returns


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


@dataclass
class Args:
    checkpoint: str = "runs/pickcube_ppo/ckpt_301.pt"
    """path to a pretrained PPO checkpoint file"""
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    num_envs: int = 1
    """number of parallel environments for MC rollouts"""
    seed: int = 1
    """random seed"""
    eval_dataset_path: str = "data/datasets/pickcube_expert_eval.pt"
    """path to the evaluation .pt dataset file"""
    control_mode: str = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
    cuda: bool = True
    """if toggled, cuda will be enabled"""
    max_steps: int = 50
    """maximum number of steps per episode"""
    reward_mode: str = "sparse"
    """reward mode for the environment"""
    dataset_num_envs: int = 16
    """number of parallel envs used when collecting the dataset"""
    obs_mode: Literal["state", "rgb", "state+rgb"] = "state"
    """observation mode for the critic"""
    gamma: float = 0.8
    """discount factor"""
    gae_lambda: float = 0.95
    """lambda for generalized advantage estimation"""

    # Sampling parameters
    num_sampled_actions: int = 8
    """K: number of actions to sample from the policy per state"""
    num_mc_rollouts: int = 10
    """M: number of MC rollouts per (state, action) pair"""

    # Critic training (V supervised on MC returns)
    critic_lr: float = 3e-4
    """learning rate for critic training"""
    critic_epochs: int = 100
    """number of training epochs for V(s) regression"""
    critic_batch_size: int = 256
    """minibatch size for critic training"""
    critic_weight_decay: float = 1e-4
    """weight decay (L2 regularization) for critic training"""

    # Bootstrap GAE parameters (iterative GAE targets, like gae.py / PPO)
    num_gae_iterations: int = 50
    """number of outer GAE iterations for bootstrap training"""
    critic_update_epochs: int = 4
    """number of inner epochs per GAE iteration"""

    # IQL parameters
    train_dataset_path: str = "data/datasets/pickcube_expert.pt"
    """path to the training .pt dataset file (used to train IQL)"""
    expectile_taus: tuple[float, ...] = (0.5, 0.7)
    """expectile tau values for IQL (trains one model per value)"""
    iql_epochs: int = 200
    """number of training epochs for IQL"""
    iql_patience: int = 100
    """early stopping patience for IQL"""
    iql_nstep: int = 10
    """n-step TD return for IQL (1 = standard, >1 = multi-step)"""

    # Output
    output: str = ""
    """save figure to this path (default: auto-generated)"""


def _cache_path(args):
    """Deterministic cache filename based on sampling parameters."""
    base = os.path.dirname(args.eval_dataset_path)
    return os.path.join(
        base,
        f"rank_cache_K{args.num_sampled_actions}"
        f"_M{args.num_mc_rollouts}_seed{args.seed}.pt",
    )


def _rollout_return(envs, agent, first_action, env_state, is_grasped,
                    num_envs, num_rounds, seed, device, gamma, max_steps,
                    restore_fn, clip_fn, store_trajectories=False):
    """Run MC rollouts from a state, optionally taking a specific first action.

    Args:
        first_action: If None, follow policy from the start (for V(s)).
                      Otherwise, take this action first, then follow policy (for Q(s,a)).
        store_trajectories: If True, store full trajectory data for GAE computation.

    Returns:
        mc_returns: list of floats (one per rollout)
        trajectories: list of trajectory dicts (only if store_trajectories=True)
    """
    mc_returns = []
    trajectories = []

    for mc_round in range(num_rounds):
        obs_t = restore_fn(env_state, seed + mc_round, is_grasped=is_grasped)

        step_states = []
        step_next_states = []
        step_actions = []
        step_rewards = []
        step_terminated = []
        step_dones = []

        # First step
        if first_action is not None:
            action = first_action.unsqueeze(0).expand(num_envs, -1)
            action = clip_fn(action)
        else:
            action, _, _, _ = agent.get_action_and_value(obs_t)
            action = clip_fn(action)

        next_obs, reward, terminated, truncated, info = envs.step(action)

        if store_trajectories:
            step_states.append(obs_t.clone())
            step_next_states.append(next_obs.clone())
            step_actions.append(action.clone())

        step_rewards.append(reward.view(-1))
        step_terminated.append(terminated.view(-1).float())
        step_dones.append((terminated | truncated).view(-1).float())

        env_done = (terminated | truncated).view(-1)
        first_done_step = torch.full(
            (num_envs,), -1, dtype=torch.long, device=device
        )
        first_done_step[env_done] = 0
        step = 1

        # Subsequent steps: follow policy
        while not env_done.all():
            prev_obs = next_obs.clone()
            action, _, _, _ = agent.get_action_and_value(next_obs)
            action = clip_fn(action)
            next_obs, reward, terminated, truncated, info = envs.step(action)

            if store_trajectories:
                step_states.append(prev_obs)
                step_next_states.append(next_obs.clone())
                step_actions.append(action.clone())

            step_rewards.append(reward.view(-1))
            step_terminated.append(terminated.view(-1).float())
            step_dones.append((terminated | truncated).view(-1).float())

            newly_done = (terminated | truncated).view(-1) & ~env_done
            first_done_step[newly_done] = step
            env_done = env_done | newly_done
            step += 1

        # Compute MC returns per env
        all_rewards = torch.stack(step_rewards, dim=0)  # (T, num_envs)
        if store_trajectories:
            all_s = torch.stack(step_states, dim=0)      # (T, num_envs, obs_dim)
            all_ns = torch.stack(step_next_states, dim=0)
            all_a = torch.stack(step_actions, dim=0)     # (T, num_envs, action_dim)
            all_t = torch.stack(step_terminated, dim=0)
            all_d = torch.stack(step_dones, dim=0)

        for env_idx in range(num_envs):
            traj_len = first_done_step[env_idx].item() + 1
            env_rewards = all_rewards[:traj_len, env_idx]
            ret = 0.0
            for t in reversed(range(traj_len)):
                ret = env_rewards[t].item() + gamma * ret
            mc_returns.append(ret)

            if store_trajectories:
                trajectories.append({
                    "states": all_s[:traj_len, env_idx].cpu(),
                    "next_states": all_ns[:traj_len, env_idx].cpu(),
                    "actions": all_a[:traj_len, env_idx].cpu(),
                    "rewards": env_rewards.cpu(),
                    "dones": all_d[:traj_len, env_idx].cpu(),
                    "terminated": all_t[:traj_len, env_idx].cpu(),
                })

    return mc_returns, trajectories


def collect_rollouts(args, device):
    """Sample K actions per state, collect MC rollouts and trajectories."""
    sim_backend = "physx_cuda" if device.type == "cuda" else "cpu"
    num_envs = args.num_envs if device.type == "cuda" else 1

    env_kwargs = dict(
        obs_mode="state",
        render_mode="sensors",
        sim_backend=sim_backend,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        max_episode_steps=args.max_steps,
    )
    envs = gym.make(args.env_id, num_envs=num_envs, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(
        envs, num_envs, ignore_terminations=False, record_metrics=True
    )

    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()

    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)
    action_dim = envs.single_action_space.shape[0]

    def clip_action(a):
        return torch.clamp(a.detach(), action_low, action_high)

    _zero_action = torch.zeros(num_envs, action_dim, device=device)
    _IS_GRASPED_IDX = 18

    def _restore_state_with_contacts(env_state, seed, is_grasped=None):
        envs.reset(seed=seed)
        envs.base_env.set_state_dict(env_state)
        envs.base_env.step(_zero_action)
        envs.base_env.set_state_dict(env_state)
        envs.base_env._elapsed_steps[:] = 0
        obs = envs.base_env.get_obs()
        if is_grasped is not None:
            obs[:, _IS_GRASPED_IDX] = is_grasped
        return obs

    dataset = OfflineRLDataset([args.eval_dataset_path], False, False)
    N = len(dataset)
    state_dim = dataset.state.shape[1]

    num_rounds = math.ceil(args.num_mc_rollouts / num_envs)
    K = args.num_sampled_actions
    print(
        f"Rank comparison: K={K} actions, M={args.num_mc_rollouts} rollouts "
        f"({num_rounds} rounds x {num_envs} envs)"
    )

    all_v_mc = []
    all_q_mc = []  # (N, K)
    all_sampled_actions = []  # (N, K, action_dim)
    all_q_trajectories = []
    traj_to_state_action = []  # (state_idx, action_idx) per trajectory

    for data in tqdm(dataset, desc="Collecting rollouts"):
        data_idx = data["idx"]
        env_state = _replicate_state(dataset.get_env_state(data_idx), num_envs)
        is_grasped = data["obs"]["state"][_IS_GRASPED_IDX]

        # Sample K actions from the policy
        obs_for_policy = _restore_state_with_contacts(
            env_state, args.seed, is_grasped=is_grasped
        )
        with torch.no_grad():
            obs_single = obs_for_policy[:1]  # (1, obs_dim)
            obs_k = obs_single.expand(K, -1)
            action_mean = agent.actor_mean(obs_k)
            action_logstd = agent.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            dist = Normal(action_mean, action_std)
            sampled_actions = clip_action(dist.sample())  # (K, action_dim)
        all_sampled_actions.append(sampled_actions.cpu())

        # V(s): policy rollouts
        v_returns, _ = _rollout_return(
            envs, agent, None, env_state, is_grasped,
            num_envs, num_rounds, args.seed, device, args.gamma, args.max_steps,
            _restore_state_with_contacts, clip_action,
            store_trajectories=False,
        )
        all_v_mc.append(np.mean(v_returns))

        # Q(s, a_k) for each sampled action
        state_q = []
        for k in range(K):
            q_returns, q_trajs = _rollout_return(
                envs, agent, sampled_actions[k], env_state, is_grasped,
                num_envs, num_rounds, args.seed, device, args.gamma, args.max_steps,
                _restore_state_with_contacts, clip_action,
                store_trajectories=True,
            )
            state_q.append(np.mean(q_returns))
            for traj in q_trajs:
                all_q_trajectories.append(traj)
                traj_to_state_action.append((data_idx, k))
        all_q_mc.append(state_q)

    envs.close()
    del envs, agent
    torch.cuda.empty_cache()

    v_mc = torch.tensor(all_v_mc)                             # (N,)
    q_mc = torch.tensor(all_q_mc)                             # (N, K)
    sampled_actions = torch.stack(all_sampled_actions, dim=0)  # (N, K, action_dim)

    print(f"Collected {len(all_q_trajectories)} Q-trajectories")
    traj_lens = [t["states"].shape[0] for t in all_q_trajectories]
    print(
        f"  Lengths: min={min(traj_lens)}, max={max(traj_lens)}, "
        f"mean={sum(traj_lens)/len(traj_lens):.1f}"
    )

    cache = {
        "v_mc": v_mc,
        "q_mc": q_mc,
        "sampled_actions": sampled_actions,
        "trajectories": all_q_trajectories,
        "traj_to_state_action": traj_to_state_action,
        "N": N,
        "state_dim": state_dim,
    }
    cache_file = _cache_path(args)
    torch.save(cache, cache_file)
    print(f"Cached rollout data to {cache_file}")

    return cache


# ---------------------------------------------------------------------------
# V(s) training (MC return supervision)
# ---------------------------------------------------------------------------


def train_value_mc(trajectories, state_dim, gamma, device, args):
    """Train V(s) by regressing on MC returns from collected trajectories."""
    # Flatten all (s_t, G_t) pairs
    all_states = []
    all_returns = []
    for traj in trajectories:
        all_states.append(traj["states"])
        all_returns.append(_compute_mc_returns(traj["rewards"], gamma))
    all_states = torch.cat(all_states, dim=0)
    all_returns = torch.cat(all_returns, dim=0)

    N = all_states.shape[0]
    print(f"\nTraining V(s) on {N} transitions (MC return supervision)...")

    critic = Critic("state", state_dim=state_dim).to(device)
    optimizer = torch.optim.Adam(
        critic.parameters(), lr=args.critic_lr, eps=1e-5,
        weight_decay=args.critic_weight_decay,
    )

    for epoch in range(1, args.critic_epochs + 1):
        indices = torch.randperm(N)
        total_loss = 0.0
        total_batches = 0
        critic.train()
        for start in range(0, N, args.critic_batch_size):
            batch_idx = indices[start : start + args.critic_batch_size]
            batch_obs = all_states[batch_idx].to(device)
            batch_ret = all_returns[batch_idx].to(device)

            pred = critic(batch_obs).squeeze(-1)
            loss = 0.5 * ((pred - batch_ret) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        if epoch % 20 == 0 or epoch == 1:
            avg = total_loss / max(total_batches, 1)
            print(f"  Epoch {epoch}/{args.critic_epochs}: loss={avg:.6f}")

    critic.eval()
    return critic


# ---------------------------------------------------------------------------
# GAE advantage computation
# ---------------------------------------------------------------------------


def compute_gae_advantages(critic, trajectories, traj_to_state_action,
                           N, K, gamma, gae_lambda, device):
    """Compute first-step GAE advantage for each Q-rollout trajectory.

    Returns:
        gae_advantages: (N, K) tensor, averaged over rollouts per (state, action).
    """
    # Flatten all obs for batched critic forward pass
    all_obs = torch.cat([t["states"] for t in trajectories], dim=0)
    all_next_obs = torch.cat([t["next_states"] for t in trajectories], dim=0)

    all_v = _batched_forward(critic, all_obs, device)
    all_v_next = _batched_forward(critic, all_next_obs, device)

    # Compute first-step GAE for each trajectory
    gae_adv_sum = torch.zeros(N, K)
    gae_counts = torch.zeros(N, K)

    offset = 0
    for traj_idx, traj in enumerate(trajectories):
        traj_len = traj["states"].shape[0]
        v = all_v[offset : offset + traj_len]
        v_next = all_v_next[offset : offset + traj_len]
        rewards = traj["rewards"]
        terminated = traj["terminated"]
        dones = traj["dones"]
        offset += traj_len

        deltas = rewards + gamma * v_next * (1.0 - terminated) - v

        advantages = torch.zeros(traj_len)
        lastgaelam = 0.0
        for t in reversed(range(traj_len)):
            not_done = 1.0 - dones[t]
            advantages[t] = lastgaelam = (
                deltas[t] + gamma * gae_lambda * not_done * lastgaelam
            )

        state_idx, action_idx = traj_to_state_action[traj_idx]
        gae_adv_sum[state_idx, action_idx] += advantages[0].item()
        gae_counts[state_idx, action_idx] += 1

    gae_advantages = gae_adv_sum / gae_counts.clamp(min=1)
    return gae_advantages


# ---------------------------------------------------------------------------
# V(s) training (iterative GAE bootstrap, like gae.py / PPO)
# ---------------------------------------------------------------------------


def train_value_bootstrap(trajectories, state_dim, gamma, gae_lambda, device, args):
    """Train V(s) using iterative GAE bootstrap targets (like gae.py / PPO).

    Each outer iteration recomputes GAE returns with the current critic,
    then trains the critic on those frozen targets for K inner epochs.
    """
    critic = Critic("state", state_dim=state_dim).to(device)
    optimizer = torch.optim.Adam(
        critic.parameters(), lr=args.critic_lr, eps=1e-5,
        weight_decay=args.critic_weight_decay,
    )

    all_states = torch.cat([t["states"] for t in trajectories], dim=0)
    N = all_states.shape[0]

    for gae_iter in range(1, args.num_gae_iterations + 1):
        # Recompute GAE returns with current critic (frozen targets)
        all_returns = []
        critic.eval()
        for traj in trajectories:
            rewards = traj["rewards"].to(device)
            terminated = traj["terminated"].to(device)
            dones = traj["dones"].to(device)
            traj_len = rewards.shape[0]

            with torch.no_grad():
                v = critic(traj["states"].to(device)).squeeze(-1)
                v_next = critic(traj["next_states"].to(device)).squeeze(-1)

            deltas = rewards + gamma * v_next * (1.0 - terminated) - v

            advantages = torch.zeros(traj_len, device=device)
            lastgaelam = 0.0
            for t in reversed(range(traj_len)):
                not_done = 1.0 - dones[t]
                advantages[t] = lastgaelam = (
                    deltas[t] + gamma * gae_lambda * not_done * lastgaelam
                )

            all_returns.append((advantages + v).cpu())

        gae_returns = torch.cat(all_returns, dim=0)

        # Train critic on GAE returns for K epochs
        critic.train()
        total_loss = 0.0
        total_batches = 0
        for _epoch in range(args.critic_update_epochs):
            indices = torch.randperm(N)
            for start in range(0, N, args.critic_batch_size):
                batch_idx = indices[start : start + args.critic_batch_size]
                batch_obs = all_states[batch_idx].to(device)
                batch_returns = gae_returns[batch_idx].to(device)

                pred = critic(batch_obs).squeeze(-1)
                loss = 0.5 * ((pred - batch_returns) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        if gae_iter % 10 == 0 or gae_iter == 1:
            print(
                f"  GAE iter {gae_iter}/{args.num_gae_iterations}: "
                f"loss={avg_loss:.6f}, "
                f"returns mean={gae_returns.mean():.4f}, "
                f"std={gae_returns.std():.4f}"
            )

    critic.eval()
    return critic


# ---------------------------------------------------------------------------
# IQL advantage computation
# ---------------------------------------------------------------------------


def train_and_eval_iql(eval_dataset, sampled_actions, device, args,
                       extra_trajectories=None):
    """Train IQL for each tau and evaluate advantages for sampled actions.

    Args:
        extra_trajectories: Optional list of trajectory dicts (from MC rollouts)
            with keys "states", "actions", "rewards", "next_states",
            "terminated", "dones".

    Returns:
        dict mapping "IQL(tau)" -> (N, K) tensor of advantages
    """
    # Load training dataset and prepare flat data (once, shared across taus)
    print(f"\nLoading training dataset for IQL: {args.train_dataset_path}")
    train_dataset = OfflineRLDataset([args.train_dataset_path], False, False)

    train_trajectories = train_dataset.extract_trajectories(
        num_envs=args.dataset_num_envs, gamma=args.gamma
    )
    eval_trajectories = eval_dataset.extract_trajectories(
        num_envs=args.dataset_num_envs, gamma=args.gamma
    )

    all_trajectories = train_trajectories + eval_trajectories
    all_states = torch.cat([t["states"] for t in all_trajectories], dim=0)
    all_next_states = torch.cat([t["next_states"] for t in all_trajectories], dim=0)
    all_rewards = torch.cat([t["rewards"] for t in all_trajectories], dim=0)
    all_terminated = torch.cat([t["terminated"] for t in all_trajectories], dim=0)

    train_actions_list = [train_dataset.actions[t["flat_indices"]] for t in train_trajectories]
    eval_actions_list = [eval_dataset.actions[t["flat_indices"]] for t in eval_trajectories]
    all_actions = torch.cat(train_actions_list + eval_actions_list, dim=0)

    n_dataset = all_states.shape[0]

    # Augment with rollout trajectories
    if extra_trajectories is not None:
        extra_states = torch.cat([t["states"] for t in extra_trajectories], dim=0)
        extra_actions = torch.cat([t["actions"] for t in extra_trajectories], dim=0)
        extra_rewards = torch.cat([t["rewards"] for t in extra_trajectories], dim=0)
        extra_next = torch.cat([t["next_states"] for t in extra_trajectories], dim=0)
        extra_term = torch.cat([t["terminated"] for t in extra_trajectories], dim=0)
        all_states = torch.cat([all_states, extra_states], dim=0)
        all_actions = torch.cat([all_actions, extra_actions], dim=0)
        all_rewards = torch.cat([all_rewards, extra_rewards], dim=0)
        all_next_states = torch.cat([all_next_states, extra_next], dim=0)
        all_terminated = torch.cat([all_terminated, extra_term], dim=0)

    print(f"  IQL training data: {all_states.shape[0]} transitions "
          f"({n_dataset} dataset + {all_states.shape[0] - n_dataset} rollout)")

    # Compute n-step TD targets if nstep > 1
    nstep_kw = {}
    if args.iql_nstep > 1:
        # Combine all trajectory sources for n-step computation
        nstep_trajs = list(all_trajectories)
        if extra_trajectories is not None:
            nstep_trajs = nstep_trajs + list(extra_trajectories)
        print(f"  Computing {args.iql_nstep}-step TD targets from "
              f"{len(nstep_trajs)} trajectories...")
        nret, boot_s, ndisc = compute_nstep_targets(
            nstep_trajs, args.iql_nstep, args.gamma
        )
        nstep_kw = dict(
            nstep_returns=nret, bootstrap_states=boot_s, nstep_discounts=ndisc
        )
        frac_boot = (ndisc > 0).float().mean()
        print(f"  n-step returns: mean={nret.mean():.4f}, std={nret.std():.4f}")
        print(f"  Bootstrapped: {frac_boot:.1%} of transitions")

    eval_states = eval_dataset.state
    results = {}

    for tau in args.expectile_taus:
        iql_args = IQLArgs(
            gamma=args.gamma,
            expectile_tau=tau,
            epochs=args.iql_epochs,
            patience=args.iql_patience,
            lr=args.critic_lr,
            batch_size=args.critic_batch_size,
            weight_decay=args.critic_weight_decay,
            nstep=args.iql_nstep,
        )

        print(f"\nTraining IQL (tau={tau}, nstep={args.iql_nstep})...")
        q_net, v_net = train_iql(
            all_states, all_actions, all_rewards, all_next_states, all_terminated,
            device, iql_args, **nstep_kw,
        )

        adv, iql_q, iql_v = _eval_iql_advantages(
            q_net, v_net, eval_states, sampled_actions, device
        )
        label = f"IQL({tau})"
        results[label] = adv
        results[f"{label}_Q"] = iql_q
        results[f"{label}_V"] = iql_v
        print(f"  {label} V(s):   mean={iql_v.mean():.4f}, std={iql_v.std():.4f}")
        print(f"  {label} Q(s,a): mean={iql_q.mean():.4f}, std={iql_q.std():.4f}")
        print(f"  {label} A(s,a): mean={adv.mean():.4f}, std={adv.std():.4f}")

        del q_net, v_net
        torch.cuda.empty_cache()

    del train_dataset, train_trajectories, eval_trajectories, all_trajectories
    del all_states, all_next_states, all_actions, all_rewards, all_terminated

    return results


def _eval_iql_advantages(q_net, v_net, eval_states, sampled_actions, device):
    """Evaluate IQL advantages A(s,a_k) = Q(s,a_k) - V(s) for sampled actions.

    Returns:
        iql_advantages: (N, K) tensor
        iql_q: (N, K) tensor of raw Q values
        iql_v: (N,) tensor of raw V values
    """
    N, K, _ = sampled_actions.shape
    iql_advantages = torch.zeros(N, K)
    iql_q = torch.zeros(N, K)
    iql_v = torch.zeros(N)

    q_net.eval()
    v_net.eval()
    batch_size = 4096
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            s = eval_states[start:end].to(device)          # (B, state_dim)
            v = v_net(s).squeeze(-1)                        # (B,)
            iql_v[start:end] = v.cpu()
            for k in range(K):
                a = sampled_actions[start:end, k].to(device)  # (B, action_dim)
                q = q_net(s, a).squeeze(-1)                   # (B,)
                iql_q[start:end, k] = q.cpu()
                iql_advantages[start:end, k] = (q - v).cpu()

    return iql_advantages, iql_q, iql_v


# ---------------------------------------------------------------------------
# Ranking comparison
# ---------------------------------------------------------------------------


def _pairwise_metrics(adv_a, adv_b, K):
    """Compute ranking metrics between two advantage vectors for one state."""
    rho, _ = sp_stats.spearmanr(adv_a, adv_b)
    tau, _ = sp_stats.kendalltau(adv_a, adv_b)
    top1 = adv_a.argmax() == adv_b.argmax()

    n_concordant = 0
    n_pairs = 0
    for j in range(K):
        for l in range(j + 1, K):
            s_a = np.sign(adv_a[j] - adv_a[l])
            s_b = np.sign(adv_b[j] - adv_b[l])
            if s_a != 0 and s_b != 0:
                n_concordant += int(s_a == s_b)
                n_pairs += 1
    concordance = n_concordant / max(n_pairs, 1)

    return rho, tau, top1, concordance


def compute_ranking_metrics(methods_dict):
    """Compute per-state ranking comparison metrics for all method pairs.

    Args:
        methods_dict: dict of {name: (N, K) numpy array} advantages.
                      Must include "MC" as the reference.

    Returns:
        dict with per-pair metrics and valid_mask
    """
    names = list(methods_dict.keys())
    N, K = methods_dict["MC"].shape
    mc_adv = methods_dict["MC"]

    # Valid mask: MC has variance
    valid_mask = np.array([mc_adv[i].std() > 1e-8 for i in range(N)])

    # Build all pairs
    pairs = []
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            pairs.append((n1, n2))

    pair_metrics = {}
    for n1, n2 in pairs:
        key = f"{n1}_vs_{n2}"
        rhos, taus, top1s, concs = [], [], [], []
        for i in range(N):
            if not valid_mask[i]:
                continue
            rho, tau, top1, conc = _pairwise_metrics(
                methods_dict[n1][i], methods_dict[n2][i], K,
            )
            rhos.append(rho)
            taus.append(tau)
            top1s.append(top1)
            concs.append(conc)
        pair_metrics[key] = {
            "spearman_rhos": np.array(rhos),
            "kendall_taus": np.array(taus),
            "top1_agrees": np.array(top1s),
            "concordances": np.array(concs),
        }

    return {
        "pairs": pair_metrics,
        "valid_mask": valid_mask,
        "num_valid": int(valid_mask.sum()),
        "num_total": N,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(methods_dict, metrics, save_path):
    """Produce a 3-way comparison figure.

    Args:
        methods_dict: {"MC": (N,K), "GAE": (N,K), "IQL": (N,K)} numpy arrays
        metrics: output of compute_ranking_metrics
        save_path: where to save the figure
    """
    names = list(methods_dict.keys())
    pairs = list(metrics["pairs"].keys())
    n_pairs = len(pairs)

    fig = plt.figure(figsize=(6 * n_pairs, 18), constrained_layout=True)
    gs = fig.add_gridspec(3, n_pairs)

    # --- Row 0: Scatter plots for each pair ---
    for col, pair_key in enumerate(pairs):
        n1, n2 = pair_key.split("_vs_")
        ax = fig.add_subplot(gs[0, col])
        x = methods_dict[n1].flatten()
        y = methods_dict[n2].flatten()
        ax.scatter(x, y, alpha=0.15, s=8, edgecolors="none")
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "r--", lw=1)
        ax.set_xlabel(f"{n1} advantage")
        ax.set_ylabel(f"{n2} advantage")
        r, _ = sp_stats.pearsonr(x, y)
        ax.set_title(f"{n1} vs {n2} (Pearson r={r:.3f})")

    # --- Row 1: Spearman ρ histograms for each pair ---
    for col, pair_key in enumerate(pairs):
        n1, n2 = pair_key.split("_vs_")
        ax = fig.add_subplot(gs[1, col])
        rhos = metrics["pairs"][pair_key]["spearman_rhos"]
        ax.hist(rhos, bins=30, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axvline(np.mean(rhos), color="r", ls="--", lw=1.5,
                   label=f"mean={np.mean(rhos):.3f}")
        ax.axvline(np.median(rhos), color="orange", ls="--", lw=1.5,
                   label=f"median={np.median(rhos):.3f}")
        ax.set_xlabel("Spearman ρ")
        ax.set_ylabel("Count")
        ax.set_title(f"Per-state Spearman ρ: {n1} vs {n2}")
        ax.legend(fontsize=9)

    # --- Row 2: Example state + Summary table ---
    # Example state (near median Spearman for MC vs GAE(MC))
    mc_gae_key = [k for k in pairs if k.startswith("MC_vs_GAE")][0]
    rhos_mg = metrics["pairs"][mc_gae_key]["spearman_rhos"]
    median_rho = np.median(rhos_mg)
    example_idx = np.argmin(np.abs(rhos_mg - median_rho))
    valid_indices = np.where(metrics["valid_mask"])[0]
    orig_idx = valid_indices[example_idx]
    K = methods_dict["MC"].shape[1]

    ax = fig.add_subplot(gs[2, 0])
    x = np.arange(K)
    n_methods = len(names)
    width = 0.8 / n_methods
    for m_idx, name in enumerate(names):
        offset = (m_idx - (n_methods - 1) / 2) * width
        ax.bar(x + offset, methods_dict[name][orig_idx], width,
               label=name, alpha=0.8)
    ax.set_xlabel("Action index")
    ax.set_ylabel("Advantage")
    ax.set_title(f"Example state {orig_idx} (MC-GAE ρ={rhos_mg[example_idx]:.3f})")
    ax.legend(fontsize=9)
    ax.set_xticks(x)

    # Summary table spanning remaining columns
    ax = fig.add_subplot(gs[2, 1:])
    ax.axis("off")

    col_labels = ["Metric"] + [k.replace("_vs_", " vs ") for k in pairs]
    rows = []
    for metric_name, metric_key, fmt in [
        ("Spearman ρ (mean)", "spearman_rhos", ".3f"),
        ("Spearman ρ (median)", "spearman_rhos", ".3f"),
        ("Kendall τ (mean)", "kendall_taus", ".3f"),
        ("Top-1 agreement", "top1_agrees", ".3f"),
        ("Concordance (mean)", "concordances", ".3f"),
    ]:
        row = [metric_name]
        for pair_key in pairs:
            vals = metrics["pairs"][pair_key][metric_key]
            if "median" in metric_name:
                row.append(f"{np.median(vals):{fmt}}")
            else:
                row.append(f"{np.mean(vals):{fmt}}")
        rows.append(row)
    # Add Pearson r (pooled) row
    row = ["Pearson r (pooled)"]
    for pair_key in pairs:
        n1, n2 = pair_key.split("_vs_")
        r, _ = sp_stats.pearsonr(
            methods_dict[n1].flatten(), methods_dict[n2].flatten()
        )
        row.append(f"{r:.3f}")
    rows.append(row)
    rows.append(
        ["Valid states"] +
        [f"{metrics['num_valid']}/{metrics['num_total']}"] * n_pairs
    )

    table = ax.table(
        cellText=rows, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    ax.set_title("Summary", fontsize=11, pad=10)

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    args = tyro.cli(Args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # -------------------------------------------------------------------
    # 1. Collect rollouts (or load from cache)
    # -------------------------------------------------------------------
    cache_file = _cache_path(args)
    if os.path.exists(cache_file):
        print(f"Loading cached rollout data from {cache_file}")
        cache = torch.load(cache_file, weights_only=False)
        print(
            f"  {cache['N']} states, {len(cache['trajectories'])} trajectories"
        )
    else:
        cache = collect_rollouts(args, device)

    v_mc = cache["v_mc"]
    q_mc = cache["q_mc"]
    sampled_actions = cache["sampled_actions"]
    trajectories = cache["trajectories"]
    traj_to_state_action = cache["traj_to_state_action"]
    N = cache["N"]
    state_dim = cache["state_dim"]
    K = args.num_sampled_actions

    mc_advantages = q_mc - v_mc.unsqueeze(1)  # (N, K)

    print(f"\nMC estimates:")
    print(f"  V(s):   mean={v_mc.mean():.4f}, std={v_mc.std():.4f}")
    print(f"  Q(s,a): mean={q_mc.mean():.4f}, std={q_mc.std():.4f}")
    print(f"  A(s,a): mean={mc_advantages.mean():.4f}, std={mc_advantages.std():.4f}")

    # -------------------------------------------------------------------
    # 2. Load training dataset for V(s) training
    # -------------------------------------------------------------------
    print(f"\nLoading training dataset: {args.train_dataset_path}")
    train_dataset = OfflineRLDataset([args.train_dataset_path], False, False)
    train_trajectories = train_dataset.extract_trajectories(
        num_envs=args.dataset_num_envs, gamma=args.gamma
    )
    traj_lens = [t["states"].shape[0] for t in train_trajectories]
    print(
        f"  {len(train_trajectories)} trajectories, "
        f"lengths: min={min(traj_lens)}, max={max(traj_lens)}, "
        f"mean={sum(traj_lens)/len(traj_lens):.1f}"
    )
    del train_dataset

    # -------------------------------------------------------------------
    # 2.5 Get timestep for each eval state
    # -------------------------------------------------------------------
    print("\nExtracting timesteps for eval states...")
    eval_dataset_tmp = OfflineRLDataset([args.eval_dataset_path], False, False)
    eval_trajectories_for_ts = eval_dataset_tmp.extract_trajectories(
        num_envs=args.dataset_num_envs, gamma=args.gamma
    )

    eval_timesteps = torch.zeros(N, dtype=torch.long)
    for traj in eval_trajectories_for_ts:
        flat_indices = traj["flat_indices"]
        for local_t, global_idx in enumerate(flat_indices.tolist()):
            eval_timesteps[global_idx] = local_t

    unique_ts = eval_timesteps.unique()
    print(f"  Timesteps: min={eval_timesteps.min().item()}, max={eval_timesteps.max().item()}, "
          f"unique={len(unique_ts)}")
    del eval_dataset_tmp, eval_trajectories_for_ts

    # -------------------------------------------------------------------
    # 3. GAE(MC): Train V(s) on MC returns from training set → GAE adv
    # -------------------------------------------------------------------
    critic_mc = train_value_mc(
        train_trajectories, state_dim, args.gamma, device, args
    )

    # Load eval states (reused for all methods)
    eval_dataset = OfflineRLDataset([args.eval_dataset_path], False, False)
    eval_states = eval_dataset.state

    v_gae_mc = _batched_forward(critic_mc, eval_states, device)
    print(f"  GAE(MC) V(s) on eval: mean={v_gae_mc.mean():.4f}, std={v_gae_mc.std():.4f}")

    print("\nComputing GAE(MC) advantages...")
    gae_mc_advantages = compute_gae_advantages(
        critic_mc, trajectories, traj_to_state_action,
        N, K, args.gamma, args.gae_lambda, device,
    )
    print(
        f"  GAE(MC) A(s,a): mean={gae_mc_advantages.mean():.4f}, "
        f"std={gae_mc_advantages.std():.4f}"
    )

    # -------------------------------------------------------------------
    # 4. GAE(Bootstrap): Train V(s) with iterative GAE targets → GAE adv
    # -------------------------------------------------------------------
    print("\nTraining V(s) with bootstrap GAE targets on training set...")
    critic_boot = train_value_bootstrap(
        train_trajectories, state_dim, args.gamma, args.gae_lambda, device, args,
    )
    del train_trajectories

    v_gae_boot = _batched_forward(critic_boot, eval_states, device)
    print(f"  GAE(Boot) V(s) on eval: mean={v_gae_boot.mean():.4f}, std={v_gae_boot.std():.4f}")

    print("Computing GAE(Bootstrap) advantages...")
    gae_boot_advantages = compute_gae_advantages(
        critic_boot, trajectories, traj_to_state_action,
        N, K, args.gamma, args.gae_lambda, device,
    )
    print(
        f"  GAE(Bootstrap) A(s,a): mean={gae_boot_advantages.mean():.4f}, "
        f"std={gae_boot_advantages.std():.4f}"
    )

    # -------------------------------------------------------------------
    # 5. Single-traj GAE (1 trajectory per (s,a), like gae.py)
    # -------------------------------------------------------------------
    seen = set()
    single_indices = []
    for i, (si, ai) in enumerate(traj_to_state_action):
        if (si, ai) not in seen:
            seen.add((si, ai))
            single_indices.append(i)
    single_trajs = [trajectories[i] for i in single_indices]
    single_map = [traj_to_state_action[i] for i in single_indices]
    print(f"\nSingle-traj subset: {len(single_trajs)} trajectories "
          f"(1 per (s,a) pair, vs {len(trajectories)} total)")

    print("Computing GAE(MC,1traj) advantages...")
    gae_mc_1t = compute_gae_advantages(
        critic_mc, single_trajs, single_map,
        N, K, args.gamma, args.gae_lambda, device,
    )
    print(f"  GAE(MC,1traj) A(s,a): mean={gae_mc_1t.mean():.4f}, "
          f"std={gae_mc_1t.std():.4f}")

    print("Computing GAE(Boot,1traj) advantages...")
    gae_boot_1t = compute_gae_advantages(
        critic_boot, single_trajs, single_map,
        N, K, args.gamma, args.gae_lambda, device,
    )
    print(f"  GAE(Boot,1traj) A(s,a): mean={gae_boot_1t.mean():.4f}, "
          f"std={gae_boot_1t.std():.4f}")

    # -------------------------------------------------------------------
    # 5.5 V(t) timestep baseline
    # -------------------------------------------------------------------
    print("\nComputing V(t) timestep baseline...")

    # V(t): for each timestep, average V_mc over all states at that timestep
    max_timestep = eval_timesteps.max().item() + 1
    v_timestep_values = torch.zeros(max_timestep)
    for t in range(max_timestep):
        mask = (eval_timesteps == t)
        if mask.any():
            v_timestep_values[t] = v_mc[mask].mean()
        else:
            v_timestep_values[t] = v_mc.mean()  # fallback

    # Each eval state's V(t)
    v_t_per_state = v_timestep_values[eval_timesteps]  # (N,)

    # Advantage = Q_mc(s, a) - V(t)
    vt_advantages = q_mc - v_t_per_state.unsqueeze(1)  # (N, K)

    print(f"  V(t) values: mean={v_timestep_values.mean():.4f}, std={v_timestep_values.std():.4f}")
    print(f"  V(t) A(s,a): mean={vt_advantages.mean():.4f}, std={vt_advantages.std():.4f}")

    # Compare V(t) vs V_mc(s)
    v_diff = (v_t_per_state - v_mc).abs()
    print(f"  |V(t) - V_mc(s)|: mean={v_diff.mean():.4f}, max={v_diff.max():.4f}")

    # -------------------------------------------------------------------
    # 6. Train IQL(s) → compute IQL advantages on same sampled actions
    # -------------------------------------------------------------------
    n_rollout_transitions = sum(t["states"].shape[0] for t in trajectories)
    print(f"\nRollout trajectories for IQL: {len(trajectories)} trajectories, "
          f"{n_rollout_transitions} transitions")

    iql_results = train_and_eval_iql(
        eval_dataset, sampled_actions, device, args,
        extra_trajectories=trajectories,
    )

    # -------------------------------------------------------------------
    # 7. V(s) and Q(s,a) comparison across methods
    # -------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("V(s) COMPARISON ON EVAL STATES")
    print(f"{'='*60}")
    print(f"  MC  V(s):       mean={v_mc.mean():.4f}, std={v_mc.std():.4f}")
    print(f"  GAE(MC) V(s):   mean={v_gae_mc.mean():.4f}, std={v_gae_mc.std():.4f}")
    print(f"  GAE(Boot) V(s): mean={v_gae_boot.mean():.4f}, std={v_gae_boot.std():.4f}")
    print(f"  V(t):           mean={v_timestep_values.mean():.4f}, std={v_timestep_values.std():.4f}")
    for label, val in iql_results.items():
        if label.endswith("_V"):
            iql_v = val
            print(f"  {label[:-2]} V(s):  mean={iql_v.mean():.4f}, std={iql_v.std():.4f}")

    # Correlations of V(s) across methods
    v_mc_np = v_mc.numpy()
    v_gae_mc_np = v_gae_mc.numpy()
    v_gae_boot_np = v_gae_boot.numpy()
    r_mc_gaemc, _ = sp_stats.pearsonr(v_mc_np, v_gae_mc_np)
    r_mc_gaeboot, _ = sp_stats.pearsonr(v_mc_np, v_gae_boot_np)
    r_gaemc_gaeboot, _ = sp_stats.pearsonr(v_gae_mc_np, v_gae_boot_np)
    print(f"\n  V(s) Pearson r:")
    print(f"    MC vs GAE(MC):        {r_mc_gaemc:.4f}")
    print(f"    MC vs GAE(Boot):      {r_mc_gaeboot:.4f}")
    print(f"    GAE(MC) vs GAE(Boot): {r_gaemc_gaeboot:.4f}")
    r_mc_vt, _ = sp_stats.pearsonr(v_mc_np, v_t_per_state.numpy())
    print(f"    MC vs V(t):           {r_mc_vt:.4f}")
    for label, val in iql_results.items():
        if label.endswith("_V"):
            iql_v_np = val.numpy()
            r_mc_iql, _ = sp_stats.pearsonr(v_mc_np, iql_v_np)
            print(f"    MC vs {label[:-2]}:       {r_mc_iql:.4f}")

    print(f"\n{'='*60}")
    print("Q(s,a) COMPARISON ON EVAL STATES (sampled actions)")
    print(f"{'='*60}")
    print(f"  MC  Q(s,a):     mean={q_mc.mean():.4f}, std={q_mc.std():.4f}")
    for label, val in iql_results.items():
        if label.endswith("_Q"):
            iql_q = val
            print(f"  {label[:-2]} Q(s,a): mean={iql_q.mean():.4f}, std={iql_q.std():.4f}")
            # Per-state Q correlation with MC
            q_mc_flat = q_mc.numpy().flatten()
            iql_q_flat = iql_q.numpy().flatten()
            r_q, _ = sp_stats.pearsonr(q_mc_flat, iql_q_flat)
            print(f"    Pearson r (pooled Q): {r_q:.4f}")
            # Per-state Spearman on Q rankings
            rhos_q = []
            for i in range(N):
                rho_q, _ = sp_stats.spearmanr(q_mc[i].numpy(), iql_q[i].numpy())
                rhos_q.append(rho_q)
            print(f"    Per-state Spearman ρ (Q ranks): mean={np.mean(rhos_q):.4f}, "
                  f"median={np.median(rhos_q):.4f}")
    print(f"{'='*60}")

    # -------------------------------------------------------------------
    # 8. Compare advantage rankings (all pairs)
    # -------------------------------------------------------------------
    methods_dict = {
        "MC": mc_advantages.numpy(),
        "V(t)": vt_advantages.numpy(),
        "GAE(MC)": gae_mc_advantages.numpy(),
        "GAE(MC,1traj)": gae_mc_1t.numpy(),
        "GAE(Bootstrap)": gae_boot_advantages.numpy(),
        "GAE(Boot,1traj)": gae_boot_1t.numpy(),
    }
    for label, adv in iql_results.items():
        if not label.endswith("_Q") and not label.endswith("_V"):
            methods_dict[label] = adv.numpy()

    print("\nComputing ranking metrics...")
    metrics = compute_ranking_metrics(methods_dict)

    print(f"\n{'='*60}")
    print(f"RANKING COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"Valid states (MC has variance): {metrics['num_valid']}/{metrics['num_total']}")
    for pair_key, pm in metrics["pairs"].items():
        label = pair_key.replace("_vs_", " vs ")
        print(f"\n  {label}:")
        print(f"    Spearman ρ:  mean={pm['spearman_rhos'].mean():.3f}, "
              f"median={np.median(pm['spearman_rhos']):.3f}")
        print(f"    Kendall τ:   mean={pm['kendall_taus'].mean():.3f}, "
              f"median={np.median(pm['kendall_taus']):.3f}")
        print(f"    Top-1 agree: {pm['top1_agrees'].mean():.3f}")
        print(f"    Concordance: {pm['concordances'].mean():.3f}")
    print(f"{'='*60}")

    # -------------------------------------------------------------------
    # 9. Save results and plot
    # -------------------------------------------------------------------
    results = {
        "v_mc": v_mc,
        "q_mc": q_mc,
        "v_gae_mc": v_gae_mc,
        "v_gae_boot": v_gae_boot,
        "mc_advantages": mc_advantages,
        "gae_mc_advantages": gae_mc_advantages,
        "gae_mc_1traj_advantages": gae_mc_1t,
        "gae_boot_advantages": gae_boot_advantages,
        "gae_boot_1traj_advantages": gae_boot_1t,
        "v_timestep_values": v_timestep_values,
        "vt_advantages": vt_advantages,
        "eval_timesteps": eval_timesteps,
        "sampled_actions": sampled_actions,
        "metrics": metrics,
    }
    for label, val in iql_results.items():
        results[f"{label}"] = val
    save_path = os.path.join(
        os.path.dirname(args.eval_dataset_path),
        f"rank_mc_vs_gae_iql_K{K}_M{args.num_mc_rollouts}"
        f"_gamma{args.gamma}_lambda{args.gae_lambda}.pt",
    )
    torch.save(results, save_path)
    print(f"\nSaved results to {save_path}")

    fig_path = args.output or os.path.join(
        os.path.dirname(args.eval_dataset_path),
        f"rank_mc_vs_gae_iql_K{K}_M{args.num_mc_rollouts}"
        f"_gamma{args.gamma}_lambda{args.gae_lambda}.png",
    )
    plot_results(methods_dict, metrics, fig_path)
