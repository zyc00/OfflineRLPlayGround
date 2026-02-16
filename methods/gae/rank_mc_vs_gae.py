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
from methods.iql.iql import QNetwork, train_iql
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

    # IQL parameters
    train_dataset_path: str = "data/datasets/pickcube_expert.pt"
    """path to the training .pt dataset file (used to train IQL)"""
    expectile_taus: tuple[float, ...] = (0.5, 0.7)
    """expectile tau values for IQL (trains one model per value)"""
    iql_epochs: int = 200
    """number of training epochs for IQL"""
    iql_patience: int = 100
    """early stopping patience for IQL"""

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
# IQL advantage computation
# ---------------------------------------------------------------------------


def train_and_eval_iql(eval_dataset, sampled_actions, device, args):
    """Train IQL for each tau and evaluate advantages for sampled actions.

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

    print(f"  IQL training data: {all_states.shape[0]} transitions")

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
        )

        print(f"\nTraining IQL (tau={tau})...")
        q_net, v_net = train_iql(
            all_states, all_actions, all_rewards, all_next_states, all_terminated,
            device, iql_args,
        )

        adv = _eval_iql_advantages(q_net, v_net, eval_states, sampled_actions, device)
        label = f"IQL({tau})"
        results[label] = adv
        print(f"  {label} A(s,a): mean={adv.mean():.4f}, std={adv.std():.4f}")

        del q_net, v_net
        torch.cuda.empty_cache()

    del train_dataset, train_trajectories, eval_trajectories, all_trajectories
    del all_states, all_next_states, all_actions, all_rewards, all_terminated

    return results


def _eval_iql_advantages(q_net, v_net, eval_states, sampled_actions, device):
    """Evaluate IQL advantages A(s,a_k) = Q(s,a_k) - V(s) for sampled actions."""
    N, K, _ = sampled_actions.shape
    iql_advantages = torch.zeros(N, K)

    q_net.eval()
    v_net.eval()
    batch_size = 4096
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            s = eval_states[start:end].to(device)          # (B, state_dim)
            v = v_net(s).squeeze(-1)                        # (B,)
            for k in range(K):
                a = sampled_actions[start:end, k].to(device)  # (B, action_dim)
                q = q_net(s, a).squeeze(-1)                   # (B,)
                iql_advantages[start:end, k] = (q - v).cpu()

    return iql_advantages


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
    # Example state (near median Spearman for MC vs GAE)
    mc_gae_key = "MC_vs_GAE"
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
    # 2. Train V(s) on MC returns → compute GAE advantages
    # -------------------------------------------------------------------
    critic = train_value_mc(
        trajectories, state_dim, args.gamma, device, args
    )

    print("\nComputing GAE advantages...")
    gae_advantages = compute_gae_advantages(
        critic, trajectories, traj_to_state_action,
        N, K, args.gamma, args.gae_lambda, device,
    )
    print(
        f"  GAE A(s,a): mean={gae_advantages.mean():.4f}, "
        f"std={gae_advantages.std():.4f}"
    )

    # -------------------------------------------------------------------
    # 3. Train IQL(s) → compute IQL advantages on same sampled actions
    # -------------------------------------------------------------------
    eval_dataset = OfflineRLDataset([args.eval_dataset_path], False, False)
    iql_results = train_and_eval_iql(
        eval_dataset, sampled_actions, device, args,
    )

    # -------------------------------------------------------------------
    # 4. Compare rankings (all pairs)
    # -------------------------------------------------------------------
    methods_dict = {"MC": mc_advantages.numpy(), "GAE": gae_advantages.numpy()}
    for label, adv in iql_results.items():
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
    # 5. Save results and plot
    # -------------------------------------------------------------------
    results = {
        "v_mc": v_mc,
        "q_mc": q_mc,
        "mc_advantages": mc_advantages,
        "gae_advantages": gae_advantages,
        "sampled_actions": sampled_actions,
        "metrics": metrics,
    }
    for label, adv in iql_results.items():
        results[f"{label}_advantages"] = adv
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
