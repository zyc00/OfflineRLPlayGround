"""Collect MC rollout data for action ranking experiments.

For each state in the eval dataset:
  1. Sample K actions from the policy
  2. For each action, run M rollouts (take that action first, then follow policy)
  3. Store trajectories and MC returns

Output: a cache .pt file with v_mc, q_mc, sampled_actions, trajectories, etc.

Usage:
  python -m methods.gae.collect_rollouts --num_envs 16 --num_mc_rollouts 1
  python -m methods.gae.collect_rollouts --num_envs 16 --num_mc_rollouts 10
"""

import math
import os
import random
from dataclasses import dataclass

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import tyro
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from torch.distributions import Normal
from tqdm import tqdm

from data.data_collection.ppo import Agent
from data.offline_dataset import OfflineRLDataset


@dataclass
class Args:
    checkpoint: str = "runs/pickcube_ppo/ckpt_301.pt"
    env_id: str = "PickCube-v1"
    num_envs: int = 1
    """number of parallel environments for rollouts"""
    seed: int = 1
    eval_dataset_path: str = "data/datasets/pickcube_expert_eval.pt"
    control_mode: str = "pd_joint_delta_pos"
    cuda: bool = True
    max_steps: int = 50
    reward_mode: str = "sparse"
    gamma: float = 0.8

    num_sampled_actions: int = 8
    """K: actions sampled per state"""
    num_mc_rollouts: int = 1
    """M: rollouts per (state, action) pair"""

    output_dir: str = "data/datasets"


def replicate_state(state_dict, n):
    if isinstance(state_dict, dict):
        return {k: replicate_state(v, n) for k, v in state_dict.items()}
    return state_dict.repeat(n, *([1] * (state_dict.ndim - 1)))


def rollout(envs, agent, first_action, env_state, is_grasped,
            num_envs, num_rounds, seed, device, gamma, restore_fn, clip_fn):
    """Run MC rollouts from a state, return (mc_returns, trajectories).

    Args:
        first_action: Action to take first (for Q), or None (for V).
    """
    mc_returns = []
    trajectories = []

    for rnd in range(num_rounds):
        obs = restore_fn(env_state, seed + rnd, is_grasped=is_grasped)

        step_s, step_ns, step_a = [], [], []
        step_r, step_term, step_done = [], [], []

        # First step
        if first_action is not None:
            action = clip_fn(first_action.unsqueeze(0).expand(num_envs, -1))
        else:
            action = clip_fn(agent.get_action_and_value(obs)[0])

        next_obs, reward, terminated, truncated, _ = envs.step(action)

        store_q = first_action is not None
        if store_q:
            step_s.append(obs.clone())
            step_ns.append(next_obs.clone())
            step_a.append(action.clone())

        step_r.append(reward.view(-1))
        step_term.append(terminated.view(-1).float())
        step_done.append((terminated | truncated).view(-1).float())

        env_done = (terminated | truncated).view(-1)
        done_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        done_step[env_done] = 0
        t = 1

        # Subsequent steps: follow policy
        while not env_done.all():
            prev_obs = next_obs.clone()
            action = clip_fn(agent.get_action_and_value(next_obs)[0])
            next_obs, reward, terminated, truncated, _ = envs.step(action)

            if store_q:
                step_s.append(prev_obs)
                step_ns.append(next_obs.clone())
                step_a.append(action.clone())

            step_r.append(reward.view(-1))
            step_term.append(terminated.view(-1).float())
            step_done.append((terminated | truncated).view(-1).float())

            newly_done = (terminated | truncated).view(-1) & ~env_done
            done_step[newly_done] = t
            env_done = env_done | newly_done
            t += 1

        # Extract per-env returns and trajectories
        all_r = torch.stack(step_r, dim=0)  # (T, num_envs)
        if store_q:
            all_s = torch.stack(step_s, dim=0)
            all_ns = torch.stack(step_ns, dim=0)
            all_a = torch.stack(step_a, dim=0)
            all_tm = torch.stack(step_term, dim=0)
            all_dn = torch.stack(step_done, dim=0)

        for e in range(num_envs):
            T = done_step[e].item() + 1
            rewards = all_r[:T, e]

            # MC return
            ret = 0.0
            for s in reversed(range(T)):
                ret = rewards[s].item() + gamma * ret
            mc_returns.append(ret)

            if store_q:
                trajectories.append({
                    "states": all_s[:T, e].cpu(),
                    "next_states": all_ns[:T, e].cpu(),
                    "actions": all_a[:T, e].cpu(),
                    "rewards": rewards.cpu(),
                    "dones": all_dn[:T, e].cpu(),
                    "terminated": all_tm[:T, e].cpu(),
                })

    return mc_returns, trajectories


if __name__ == "__main__":
    args = tyro.cli(Args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    sim_backend = "physx_cuda" if device.type == "cuda" else "cpu"
    num_envs = args.num_envs if device.type == "cuda" else 1

    # ── Setup env and agent ───────────────────────────────────────────
    env_kwargs = dict(
        obs_mode="state", render_mode="sensors", sim_backend=sim_backend,
        reward_mode=args.reward_mode, control_mode=args.control_mode,
        max_episode_steps=args.max_steps,
    )
    envs = gym.make(args.env_id, num_envs=num_envs, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(envs, num_envs, ignore_terminations=False, record_metrics=True)

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

    def restore_state(env_state, seed, is_grasped=None):
        envs.reset(seed=seed)
        envs.base_env.set_state_dict(env_state)
        envs.base_env.step(_zero_action)
        envs.base_env.set_state_dict(env_state)
        envs.base_env._elapsed_steps[:] = 0
        obs = envs.base_env.get_obs()
        if is_grasped is not None:
            obs[:, _IS_GRASPED_IDX] = is_grasped
        return obs

    # ── Load eval dataset ─────────────────────────────────────────────
    dataset = OfflineRLDataset([args.eval_dataset_path], False, False)
    N = len(dataset)
    state_dim = dataset.state.shape[1]
    K = args.num_sampled_actions
    M = args.num_mc_rollouts
    num_rounds = math.ceil(M / num_envs)

    print(f"Collecting: {N} states, K={K} actions, M={M} rollouts "
          f"({num_rounds} rounds x {num_envs} envs)")

    # ── Collect ───────────────────────────────────────────────────────
    all_v_mc = []
    all_q_mc = []
    all_sampled_actions = []
    all_trajectories = []
    traj_to_state_action = []

    for data in tqdm(dataset, desc="Collecting"):
        idx = data["idx"]
        env_state = replicate_state(dataset.get_env_state(idx), num_envs)
        is_grasped = data["obs"]["state"][_IS_GRASPED_IDX]

        # Sample K actions from the policy
        obs = restore_state(env_state, args.seed, is_grasped=is_grasped)
        with torch.no_grad():
            obs_k = obs[:1].expand(K, -1)
            mean = agent.actor_mean(obs_k)
            std = torch.exp(agent.actor_logstd.expand_as(mean))
            sampled = clip_action(Normal(mean, std).sample())  # (K, action_dim)
        all_sampled_actions.append(sampled.cpu())

        # V(s): policy rollouts (no stored trajectories)
        v_rets, _ = rollout(
            envs, agent, None, env_state, is_grasped,
            num_envs, num_rounds, args.seed, device, args.gamma,
            restore_state, clip_action,
        )
        all_v_mc.append(np.mean(v_rets))

        # Q(s, a_k): rollouts with first action fixed
        state_q = []
        for k in range(K):
            q_rets, q_trajs = rollout(
                envs, agent, sampled[k], env_state, is_grasped,
                num_envs, num_rounds, args.seed, device, args.gamma,
                restore_state, clip_action,
            )
            state_q.append(np.mean(q_rets))
            for traj in q_trajs:
                all_trajectories.append(traj)
                traj_to_state_action.append((idx, k))
        all_q_mc.append(state_q)

    envs.close()
    del envs, agent
    torch.cuda.empty_cache()

    # ── Save ──────────────────────────────────────────────────────────
    v_mc = torch.tensor(all_v_mc)
    q_mc = torch.tensor(all_q_mc)
    sampled_actions = torch.stack(all_sampled_actions, dim=0)

    traj_lens = [t["states"].shape[0] for t in all_trajectories]
    print(f"\nCollected {len(all_trajectories)} trajectories")
    print(f"  Lengths: min={min(traj_lens)}, max={max(traj_lens)}, "
          f"mean={sum(traj_lens)/len(traj_lens):.1f}")
    print(f"  MC V(s): mean={v_mc.mean():.4f}, std={v_mc.std():.4f}")
    print(f"  MC Q(s,a): mean={q_mc.mean():.4f}, std={q_mc.std():.4f}")

    cache = {
        "v_mc": v_mc,
        "q_mc": q_mc,
        "sampled_actions": sampled_actions,
        "trajectories": all_trajectories,
        "traj_to_state_action": traj_to_state_action,
        "N": N,
        "state_dim": state_dim,
    }

    filename = f"rank_cache_K{K}_M{M}_seed{args.seed}.pt"
    path = os.path.join(args.output_dir, filename)
    torch.save(cache, path)
    print(f"Saved to {path}")
