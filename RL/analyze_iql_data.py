"""Figure out what training data makes IQL Q/V work for policy improvement.

Collects data from expert and medium policies, trains IQL on different mixes,
and evaluates which gives useful advantage signal on medium policy states.

Data mixes tested:
  1. expert_only   — only expert (ckpt_301) rollouts
  2. medium_only   — only medium (ckpt_101) rollouts
  3. mixed         — 50/50 expert + medium
  4. multi_ckpt    — data from ckpt_1, ckpt_101, ckpt_301 (full coverage)

Usage:
  python -m RL.analyze_iql_data
"""

import copy
import random
from dataclasses import dataclass

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from scipy.stats import spearmanr

from data.data_collection.ppo import Agent, layer_init
from methods.iql.iql import QNetwork, expectile_loss


@dataclass
class Args:
    expert_checkpoint: str = "runs/pickcube_ppo/ckpt_301.pt"
    medium_checkpoint: str = "runs/pickcube_ppo/ckpt_101.pt"
    beginner_checkpoint: str = "runs/pickcube_ppo/ckpt_1.pt"
    env_id: str = "PickCube-v1"
    num_envs: int = 512
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"
    max_episode_steps: int = 50
    gamma: float = 0.8
    seed: int = 1
    cuda: bool = True

    # Data collection
    num_rollouts: int = 10
    """rollouts per policy"""
    num_steps: int = 50

    # IQL training
    expectile_tau: float = 0.5
    """0.5 = SARSA"""
    tau_polyak: float = 0.005
    epochs: int = 100
    batch_size: int = 4096
    lr: float = 1e-3
    grad_clip: float = 0.5


def collect_data(envs, policy, num_rollouts, num_steps, device, seed, gamma):
    """Collect (s, a, r, s', done, mc_return) from a policy."""
    obs_dim = np.array(envs.single_observation_space.shape).prod()
    act_dim = np.prod(envs.single_action_space.shape)
    act_shape = envs.single_action_space.shape
    num_envs = envs.num_envs

    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    all_s, all_a, all_r, all_ns, all_d, all_mc = [], [], [], [], [], []
    next_obs, _ = envs.reset(seed=seed)

    for _ in range(num_rollouts):
        obs_buf = torch.zeros((num_steps, num_envs, obs_dim), device=device)
        act_buf = torch.zeros((num_steps, num_envs) + act_shape, device=device)
        rew_buf = torch.zeros((num_steps, num_envs), device=device)
        nobs_buf = torch.zeros((num_steps, num_envs, obs_dim), device=device)
        done_buf = torch.zeros((num_steps, num_envs), device=device)

        for step in range(num_steps):
            obs_buf[step] = next_obs
            with torch.no_grad():
                action = policy.get_action(next_obs, deterministic=False)
            clipped = torch.clamp(action.detach(), action_low, action_high)
            next_obs, reward, term, trunc, infos = envs.step(clipped)
            act_buf[step] = clipped
            rew_buf[step] = reward.view(-1)
            done_buf[step] = (term | trunc).float()
            nobs_buf[step] = next_obs.clone()
            if "final_info" in infos:
                mask = infos["_final_info"]
                nobs_buf[step, mask] = infos["final_observation"][mask]

        # MC returns
        returns = torch.zeros((num_steps, num_envs), device=device)
        ret = torch.zeros(num_envs, device=device)
        for t in reversed(range(num_steps)):
            ret = rew_buf[t] + gamma * ret * (1.0 - done_buf[t])
            returns[t] = ret

        all_s.append(obs_buf.reshape(-1, obs_dim).cpu())
        all_a.append(act_buf.reshape(-1, act_dim).cpu())
        all_r.append(rew_buf.reshape(-1).cpu())
        all_ns.append(nobs_buf.reshape(-1, obs_dim).cpu())
        all_d.append(done_buf.reshape(-1).cpu())
        all_mc.append(returns.reshape(-1).cpu())

    return {
        "obs": torch.cat(all_s),
        "actions": torch.cat(all_a),
        "rewards": torch.cat(all_r),
        "next_obs": torch.cat(all_ns),
        "dones": torch.cat(all_d),
        "mc_returns": torch.cat(all_mc),
    }


def train_iql(data, device, args):
    """Train IQL Q+V on given data, return trained networks."""
    obs_dim = data["obs"].shape[1]
    act_dim = data["actions"].shape[1]
    N = len(data["obs"])

    q_net = QNetwork(obs_dim, act_dim).to(device)
    q_target = copy.deepcopy(q_net)
    v_net = nn.Sequential(
        layer_init(nn.Linear(obs_dim, 256)), nn.Tanh(),
        layer_init(nn.Linear(256, 256)), nn.Tanh(),
        layer_init(nn.Linear(256, 256)), nn.Tanh(),
        layer_init(nn.Linear(256, 1)),
    ).to(device)

    q_opt = optim.Adam(q_net.parameters(), lr=args.lr)
    v_opt = optim.Adam(v_net.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        q_net.train()
        v_net.train()
        perm = torch.randperm(N)
        for start in range(0, N, args.batch_size):
            mb = perm[start : start + args.batch_size]
            s = data["obs"][mb].to(device)
            a = data["actions"][mb].to(device)
            r = data["rewards"][mb].to(device)
            ns = data["next_obs"][mb].to(device)
            d = data["dones"][mb].to(device)

            with torch.no_grad():
                v_next = v_net(ns).squeeze(-1)
                q_tgt = r + args.gamma * v_next * (1.0 - d)
            q_pred = q_net(s, a).squeeze(-1)
            q_loss = 0.5 * ((q_pred - q_tgt) ** 2).mean()
            q_opt.zero_grad()
            q_loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), args.grad_clip)
            q_opt.step()

            with torch.no_grad():
                q_val = q_target(s, a).squeeze(-1)
            v_pred = v_net(s).squeeze(-1)
            v_loss = expectile_loss(q_val - v_pred, args.expectile_tau)
            v_opt.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(v_net.parameters(), args.grad_clip)
            v_opt.step()

            with torch.no_grad():
                for p, pt in zip(q_net.parameters(), q_target.parameters()):
                    pt.data.mul_(1.0 - args.tau_polyak).add_(p.data, alpha=args.tau_polyak)

    q_net.eval()
    v_net.eval()
    return q_net, v_net


def evaluate_qv(q_net, v_net, eval_data, device, label, batch_size=4096):
    """Evaluate Q/V on eval data, print diagnostics."""
    N = len(eval_data["obs"])
    all_q, all_v = [], []

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            s = eval_data["obs"][start:end].to(device)
            a = eval_data["actions"][start:end].to(device)
            all_q.append(q_net(s, a).squeeze(-1).cpu())
            all_v.append(v_net(s).squeeze(-1).cpu())

    q = torch.cat(all_q)
    v = torch.cat(all_v)
    adv = q - v
    mc = eval_data["mc_returns"]

    rho_q, _ = spearmanr(q.numpy(), mc.numpy())
    rho_v, _ = spearmanr(v.numpy(), mc.numpy())

    # Q on random actions
    q_rand_list = []
    act_dim = eval_data["actions"].shape[1]
    with torch.no_grad():
        for _ in range(10):
            ra = torch.rand(N, act_dim) * 2 - 1
            qr = []
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                s = eval_data["obs"][start:end].to(device)
                qr.append(q_net(s, ra[start:end].to(device)).squeeze(-1).cpu())
            q_rand_list.append(torch.cat(qr))
    q_rand = torch.stack(q_rand_list).mean(0)

    frac_better = (q > q_rand).float().mean().item()

    # Success vs failure breakdown
    success = mc > 0
    sr = success.float().mean().item()

    print(f"\n  [{label}] (N={N}, SR={sr:.1%})")
    print(f"    MC return:   mean={mc.mean():.4f}")
    print(f"    Q(s,a):      mean={q.mean():.4f}, std={q.std():.4f}")
    print(f"    V(s):        mean={v.mean():.4f}, std={v.std():.4f}")
    print(f"    A=Q-V:       mean={adv.mean():.4f}, std={adv.std():.4f}")
    print(f"    Q(s,a_rand): mean={q_rand.mean():.4f}")
    print(f"    P(Q > Q_rand): {frac_better:.1%}")
    print(f"    ρ(Q, MC):    {rho_q:.4f}")
    print(f"    ρ(V, MC):    {rho_v:.4f}")

    if success.any() and (~success).any():
        print(f"    Success  Q={q[success].mean():.4f}, V={v[success].mean():.4f}, A={adv[success].mean():.4f}")
        print(f"    Failure  Q={q[~success].mean():.4f}, V={v[~success].mean():.4f}, A={adv[~success].mean():.4f}")

    return {"rho_q": rho_q, "rho_v": rho_v, "adv_std": adv.std().item(),
            "frac_better": frac_better, "adv_mean": adv.mean().item()}


if __name__ == "__main__":
    args = tyro.cli(Args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env_kwargs = dict(
        obs_mode="state", render_mode="rgb_array",
        sim_backend="physx_cuda" if device.type == "cuda" else "cpu",
        reward_mode=args.reward_mode, control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
    )
    envs = gym.make(args.env_id, num_envs=args.num_envs, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=False, record_metrics=True)

    # ── Load policies ─────────────────────────────────────────────────
    def load_policy(ckpt):
        p = Agent(envs).to(device)
        p.load_state_dict(torch.load(ckpt, map_location=device))
        p.eval()
        return p

    expert = load_policy(args.expert_checkpoint)
    medium = load_policy(args.medium_checkpoint)
    beginner = load_policy(args.beginner_checkpoint)

    # ── Collect data ──────────────────────────────────────────────────
    print("Collecting expert rollouts...")
    expert_data = collect_data(envs, expert, args.num_rollouts, args.num_steps,
                               device, args.seed, args.gamma)
    print(f"  Expert: {len(expert_data['obs'])} transitions, "
          f"SR={( expert_data['mc_returns'] > 0).float().mean():.1%}")

    print("Collecting medium rollouts...")
    medium_data = collect_data(envs, medium, args.num_rollouts, args.num_steps,
                                device, args.seed + 100, args.gamma)
    print(f"  Medium: {len(medium_data['obs'])} transitions, "
          f"SR={(medium_data['mc_returns'] > 0).float().mean():.1%}")

    print("Collecting beginner rollouts...")
    beginner_data = collect_data(envs, beginner, args.num_rollouts, args.num_steps,
                                  device, args.seed + 200, args.gamma)
    print(f"  Beginner: {len(beginner_data['obs'])} transitions, "
          f"SR={(beginner_data['mc_returns'] > 0).float().mean():.1%}")

    envs.close()

    # ── Build data mixes ──────────────────────────────────────────────
    def merge(*datasets):
        return {k: torch.cat([d[k] for d in datasets]) for k in datasets[0]}

    mixes = {
        "expert_only": expert_data,
        "medium_only": medium_data,
        "expert+medium": merge(expert_data, medium_data),
        "all_three": merge(beginner_data, medium_data, expert_data),
    }

    # ── Train and evaluate each mix ───────────────────────────────────
    results = {}
    for mix_name, mix_data in mixes.items():
        print(f"\n{'='*60}")
        print(f"Training IQL on: {mix_name} ({len(mix_data['obs'])} transitions)")
        print(f"{'='*60}")

        q_net, v_net = train_iql(mix_data, device, args)

        # Evaluate on medium policy data (the target for improvement)
        r = evaluate_qv(q_net, v_net, medium_data, device,
                        f"{mix_name} → eval on medium")
        results[mix_name] = r

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY: IQL evaluated on medium policy states")
    print(f"{'='*60}")
    print(f"  {'Mix':<18} {'ρ(Q,MC)':>8} {'ρ(V,MC)':>8} {'A_std':>8} {'A_mean':>8} {'Q>Qrand':>8}")
    print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for name, r in results.items():
        print(f"  {name:<18} {r['rho_q']:>8.4f} {r['rho_v']:>8.4f} "
              f"{r['adv_std']:>8.4f} {r['adv_mean']:>8.4f} {r['frac_better']:>7.1%}")
