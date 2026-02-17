"""Analyze pretrained IQL Q(s,a) and V(s) from expert data.

Evaluates how the pretrained Q/V behave on:
1. Expert policy rollouts (sanity check)
2. Medium policy rollouts (the policy we want to improve)
3. Random actions (baseline)

Key question: does Q(s,a_expert) > Q(s,a_medium) > Q(s,a_random)?
If yes, the pretrained Q can guide policy improvement.

Usage:
  python -m RL.analyze_pretrained_iql
"""

import random
from dataclasses import dataclass

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import tyro
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from data.data_collection.ppo import Agent, layer_init
from methods.iql.iql import QNetwork


@dataclass
class Args:
    q_checkpoint: str = "runs/pretrained_iql_tau0.5/q_net.pt"
    v_checkpoint: str = "runs/pretrained_iql_tau0.5/v_net.pt"
    expert_checkpoint: str = "runs/pickcube_ppo/ckpt_301.pt"
    medium_checkpoint: str = "runs/pickcube_ppo/ckpt_101.pt"
    env_id: str = "PickCube-v1"
    num_envs: int = 512
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"
    max_episode_steps: int = 50
    gamma: float = 0.8
    num_rollouts: int = 5
    num_steps: int = 50
    num_random_actions: int = 10
    """random actions per state for Q(s, a_random) estimation"""
    seed: int = 1
    cuda: bool = True


if __name__ == "__main__":
    args = tyro.cli(Args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ── Environment ───────────────────────────────────────────────────
    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="physx_cuda" if device.type == "cuda" else "cpu",
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
    )
    envs = gym.make(args.env_id, num_envs=args.num_envs, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(
        envs, args.num_envs,
        ignore_terminations=False,
        record_metrics=True,
    )

    obs_dim = np.array(envs.single_observation_space.shape).prod()
    act_dim = np.prod(envs.single_action_space.shape)
    act_shape = envs.single_action_space.shape

    # ── Load pretrained Q and V ───────────────────────────────────────
    q_net = QNetwork(obs_dim, act_dim).to(device)
    q_net.load_state_dict(torch.load(args.q_checkpoint, map_location=device))
    q_net.eval()

    v_net = nn.Sequential(
        layer_init(nn.Linear(obs_dim, 256)),
        nn.Tanh(),
        layer_init(nn.Linear(256, 256)),
        nn.Tanh(),
        layer_init(nn.Linear(256, 256)),
        nn.Tanh(),
        layer_init(nn.Linear(256, 1)),
    ).to(device)
    v_net.load_state_dict(torch.load(args.v_checkpoint, map_location=device))
    v_net.eval()
    print(f"Loaded Q: {args.q_checkpoint}")
    print(f"Loaded V: {args.v_checkpoint}")

    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(a):
        return torch.clamp(a.detach(), action_low, action_high)

    # ── Helper: collect rollouts and evaluate Q/V ─────────────────────
    def collect_and_evaluate(policy, policy_name, num_rollouts):
        """Collect rollouts with a policy and evaluate Q/V on them."""
        all_obs = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_mc_returns = []

        next_obs, _ = envs.reset(seed=args.seed)

        for r_idx in range(num_rollouts):
            obs_buf = torch.zeros((args.num_steps, args.num_envs, obs_dim), device=device)
            act_buf = torch.zeros((args.num_steps, args.num_envs) + act_shape, device=device)
            rew_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
            done_buf = torch.zeros((args.num_steps, args.num_envs), device=device)

            for step in range(args.num_steps):
                obs_buf[step] = next_obs
                with torch.no_grad():
                    action = policy.get_action(next_obs, deterministic=False)
                clipped = clip_action(action)
                next_obs, reward, term, trunc, _ = envs.step(clipped)
                act_buf[step] = clipped
                rew_buf[step] = reward.view(-1)
                done_buf[step] = (term | trunc).float()

            # MC returns
            returns = torch.zeros((args.num_steps, args.num_envs), device=device)
            ret = torch.zeros(args.num_envs, device=device)
            for t in reversed(range(args.num_steps)):
                ret = rew_buf[t] + args.gamma * ret * (1.0 - done_buf[t])
                returns[t] = ret

            all_obs.append(obs_buf.reshape(-1, obs_dim))
            all_actions.append(act_buf.reshape(-1, act_dim))
            all_rewards.append(rew_buf.reshape(-1))
            all_dones.append(done_buf.reshape(-1))
            all_mc_returns.append(returns.reshape(-1))

        all_obs = torch.cat(all_obs)
        all_actions = torch.cat(all_actions)
        all_mc_returns = torch.cat(all_mc_returns)
        N = len(all_obs)

        # Evaluate pretrained Q and V
        with torch.no_grad():
            q_vals = q_net(all_obs, all_actions).squeeze(-1)
            v_vals = v_net(all_obs).squeeze(-1)
            adv_vals = q_vals - v_vals

        # Also evaluate Q on random actions for comparison
        q_random_list = []
        with torch.no_grad():
            for _ in range(args.num_random_actions):
                rand_a = torch.rand(N, act_dim, device=device) * 2 - 1  # uniform [-1, 1]
                q_rand = q_net(all_obs, rand_a).squeeze(-1)
                q_random_list.append(q_rand)
        q_random = torch.stack(q_random_list).mean(dim=0)

        # Per-state: does Q(s,a_policy) > Q(s,a_random)?
        frac_better = (q_vals > q_random).float().mean().item()

        # Correlation with MC returns
        mc_flat = all_mc_returns.cpu()
        q_flat = q_vals.cpu()
        v_flat = v_vals.cpu()

        # Spearman rank correlation
        from scipy.stats import spearmanr
        rho_q_mc, _ = spearmanr(q_flat.numpy(), mc_flat.numpy())
        rho_v_mc, _ = spearmanr(v_flat.numpy(), mc_flat.numpy())

        sr = (all_mc_returns > 0).float().mean().item()

        print(f"\n{'='*60}")
        print(f"  {policy_name} ({N} transitions, SR≈{sr:.1%})")
        print(f"{'='*60}")
        print(f"  MC returns:  mean={mc_flat.mean():.4f}, std={mc_flat.std():.4f}")
        print(f"  Q(s,a):      mean={q_vals.mean():.4f}, std={q_vals.std():.4f}")
        print(f"  V(s):        mean={v_vals.mean():.4f}, std={v_vals.std():.4f}")
        print(f"  A(s,a)=Q-V:  mean={adv_vals.mean():.4f}, std={adv_vals.std():.4f}")
        print(f"  Q(s,a_rand): mean={q_random.mean():.4f}, std={q_random.std():.4f}")
        print(f"  P(Q_policy > Q_random): {frac_better:.1%}")
        print(f"  Spearman ρ(Q, MC_return): {rho_q_mc:.4f}")
        print(f"  Spearman ρ(V, MC_return): {rho_v_mc:.4f}")

        # Breakdown by success/failure
        success_mask = all_mc_returns > 0
        if success_mask.any() and (~success_mask).any():
            print(f"\n  Success states ({success_mask.sum().item()}):")
            print(f"    Q:   mean={q_vals[success_mask].mean():.4f}")
            print(f"    V:   mean={v_vals[success_mask].mean():.4f}")
            print(f"    A:   mean={adv_vals[success_mask].mean():.4f}")
            print(f"  Failure states ({(~success_mask).sum().item()}):")
            print(f"    Q:   mean={q_vals[~success_mask].mean():.4f}")
            print(f"    V:   mean={v_vals[~success_mask].mean():.4f}")
            print(f"    A:   mean={adv_vals[~success_mask].mean():.4f}")

        return {
            "q_vals": q_vals, "v_vals": v_vals, "adv_vals": adv_vals,
            "mc_returns": all_mc_returns, "q_random": q_random,
        }

    # ── Load policies ─────────────────────────────────────────────────
    expert = Agent(envs).to(device)
    expert.load_state_dict(torch.load(args.expert_checkpoint, map_location=device))
    expert.eval()
    print(f"Loaded expert: {args.expert_checkpoint}")

    medium = Agent(envs).to(device)
    medium.load_state_dict(torch.load(args.medium_checkpoint, map_location=device))
    medium.eval()
    print(f"Loaded medium: {args.medium_checkpoint}")

    # ── Evaluate ──────────────────────────────────────────────────────
    expert_results = collect_and_evaluate(expert, "Expert policy (ckpt_301)", args.num_rollouts)
    medium_results = collect_and_evaluate(medium, "Medium policy (ckpt_101)", args.num_rollouts)

    # ── Cross-policy comparison ───────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Cross-policy Summary")
    print(f"{'='*60}")
    print(f"  Expert Q(s,a): {expert_results['q_vals'].mean():.4f} ± {expert_results['q_vals'].std():.4f}")
    print(f"  Medium Q(s,a): {medium_results['q_vals'].mean():.4f} ± {medium_results['q_vals'].std():.4f}")
    print(f"  Random Q(s,a): {medium_results['q_random'].mean():.4f} ± {medium_results['q_random'].std():.4f}")
    print(f"")
    print(f"  Expert A(s,a): {expert_results['adv_vals'].mean():.4f} ± {expert_results['adv_vals'].std():.4f}")
    print(f"  Medium A(s,a): {medium_results['adv_vals'].mean():.4f} ± {medium_results['adv_vals'].std():.4f}")

    envs.close()
