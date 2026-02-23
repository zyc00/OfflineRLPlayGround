"""V Scaling Law: V quality vs data size & network size across all methods.

Methods: MC1, TD+rs10, TD+EMA+rs10, GAE(iter5), IQL
Eval:    On-policy MC16 and Optimal MC16 ground truth
Axes:    Data scaling (vary rollouts, fixed hidden=256)
         Network scaling (vary hidden, fixed rollouts=100)

Usage:
  python -u -m RL.v_scaling_law --gamma 0.99
"""

import copy
import os
import random
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from scipy.stats import pearsonr, spearmanr
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from data.data_collection.ppo import Agent, layer_init


@dataclass
class Args:
    checkpoint: str = "runs/pickcube_ppo/ckpt_76_logstd-1.5.pt"
    """rollout/behavior policy (det SR=43.8%)"""
    optimal_checkpoint: str = "runs/pickcube_ppo/ckpt_301.pt"
    """optimal policy for MC re-rollouts"""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_envs: int = 100
    num_steps: int = 50
    max_episode_steps: int = 50
    mc_samples: int = 16

    # V training hyperparams
    gae_iters: int = 5
    gae_epochs: int = 100
    td_epochs: int = 500
    td_reward_scale: float = 10.0
    ema_tau: float = 0.005
    critic_layers: int = 3
    lr: float = 3e-4
    minibatch_size: int = 1000
    eval_every: int = 5
    """Evaluate V quality against MC16 every N epochs (for peak r tracking)."""

    # Scaling axes
    rollout_counts: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100)
    hidden_dims: tuple[int, ...] = (64, 128, 256, 512)

    # IQL (on-policy, same data as TD/GAE)
    iql_expectile_tau: float = 0.7
    iql_epochs: int = 500
    iql_lr: float = 3e-4
    iql_batch_size: int = 256

    # Environment
    env_id: str = "PickCube-v1"
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"
    seed: int = 1

    # Output
    output: str = "runs/v_scaling_law.png"


# ══════════════════════════════════════════════════════════════════════
#  Helper: IQL with variable hidden_dim (local copy, avoids modifying
#  shared library code)
# ══════════════════════════════════════════════════════════════════════

class QNetworkVar(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim + action_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1)),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


def make_v_net(state_dim, hidden_dim=256):
    return nn.Sequential(
        layer_init(nn.Linear(state_dim, hidden_dim)),
        nn.Tanh(),
        layer_init(nn.Linear(hidden_dim, hidden_dim)),
        nn.Tanh(),
        layer_init(nn.Linear(hidden_dim, hidden_dim)),
        nn.Tanh(),
        layer_init(nn.Linear(hidden_dim, 1)),
    )


def expectile_loss(diff, tau):
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return (weight * (diff ** 2)).mean()


def train_iql_var(states, actions, rewards, next_states, terminated,
                  device, iql_args, hidden_dim=256):
    """Train IQL Q and V networks. Returns only v_net (we only need V)."""
    state_dim = states.shape[1]
    action_dim = actions.shape[1]

    q_net = QNetworkVar(state_dim, action_dim, hidden_dim).to(device)
    q_target = copy.deepcopy(q_net)
    v_net = make_v_net(state_dim, hidden_dim).to(device)

    q_opt = torch.optim.Adam(q_net.parameters(), lr=iql_args.lr, eps=1e-5,
                             weight_decay=1e-4)
    v_opt = torch.optim.Adam(v_net.parameters(), lr=iql_args.lr, eps=1e-5,
                             weight_decay=1e-4)
    q_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        q_opt, T_max=iql_args.epochs, eta_min=1e-5)
    v_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        v_opt, T_max=iql_args.epochs, eta_min=1e-5)

    N = states.shape[0]
    perm = torch.randperm(N)
    val_size = max(1, int(N * 0.1))
    train_idx = perm[val_size:]
    val_idx = perm[:val_size]
    train_size = train_idx.shape[0]
    bs = iql_args.batch_size

    val_s = states[val_idx].to(device)
    val_a = actions[val_idx].to(device)
    val_r = rewards[val_idx].to(device)
    val_ns = next_states[val_idx].to(device)
    val_term = terminated[val_idx].to(device)

    best_val = float("inf")
    best_v_state = None
    no_improve = 0

    for epoch in range(iql_args.epochs):
        q_net.train(); v_net.train()
        indices = train_idx[torch.randperm(train_size)]

        for start in range(0, train_size, bs):
            bidx = indices[start:start + bs]
            s = states[bidx].to(device)
            a = actions[bidx].to(device)
            r = rewards[bidx].to(device)
            ns = next_states[bidx].to(device)
            term = terminated[bidx].to(device)

            with torch.no_grad():
                v_next = v_net(ns).squeeze(-1)
                q_tgt = r + iql_args.gamma * v_next * (1.0 - term)
            q_pred = q_net(s, a).squeeze(-1)
            q_loss = 0.5 * ((q_pred - q_tgt) ** 2).mean()
            q_opt.zero_grad(); q_loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 0.5)
            q_opt.step()

            with torch.no_grad():
                q_val = q_target(s, a).squeeze(-1)
            v_pred = v_net(s).squeeze(-1)
            v_loss = expectile_loss(q_val - v_pred, iql_args.expectile_tau)
            v_opt.zero_grad(); v_loss.backward()
            nn.utils.clip_grad_norm_(v_net.parameters(), 0.5)
            v_opt.step()

            with torch.no_grad():
                for p, pt in zip(q_net.parameters(), q_target.parameters()):
                    pt.data.mul_(1.0 - iql_args.tau_polyak).add_(
                        p.data, alpha=iql_args.tau_polyak)

        q_sched.step(); v_sched.step()

        # Validation
        q_net.eval(); v_net.eval()
        with torch.no_grad():
            vn = v_net(val_ns).squeeze(-1)
            qt = val_r + iql_args.gamma * vn * (1.0 - val_term)
            qp = q_net(val_s, val_a).squeeze(-1)
            vql = 0.5 * ((qp - qt) ** 2).mean().item()

            qv = q_target(val_s, val_a).squeeze(-1)
            vp = v_net(val_s).squeeze(-1)
            vvl = expectile_loss(qv - vp, iql_args.expectile_tau).item()

        vt = vql + vvl
        if vt < best_val:
            best_val = vt
            best_v_state = {k: v.clone() for k, v in v_net.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= iql_args.patience:
                print(f"    IQL early stop epoch {epoch + 1}")
                break

        if (epoch + 1) % 50 == 0:
            print(f"    IQL epoch {epoch + 1}/{iql_args.epochs}")

    if best_v_state is not None:
        v_net.load_state_dict(best_v_state)
    v_net.eval()

    del q_net, q_target, q_opt, v_opt
    torch.cuda.empty_cache()
    return v_net


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda")
    max_rollouts = max(args.rollout_counts)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_kwargs = dict(
        obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda",
        reward_mode=args.reward_mode, control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
    )

    print(f"{'=' * 70}")
    print(f"V Scaling Law | gamma={args.gamma} lambda={args.gae_lambda}")
    print(f"  Behavior: {args.checkpoint}")
    print(f"  Optimal:  {args.optimal_checkpoint}")
    print(f"  Data scaling: rollouts={args.rollout_counts}")
    print(f"  Network scaling: hidden={args.hidden_dims}")
    print(f"{'=' * 70}")
    sys.stdout.flush()

    # ── Phase 0: Env + agents setup ──
    envs = gym.make(args.env_id, num_envs=args.num_envs, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(
        envs, args.num_envs, ignore_terminations=False, record_metrics=True,
    )
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    action_dim = int(np.prod(envs.single_action_space.shape))
    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()

    optimal_agent = Agent(envs).to(device)
    optimal_agent.load_state_dict(
        torch.load(args.optimal_checkpoint, map_location=device))
    optimal_agent.eval()

    def clip_action(a):
        return torch.clamp(a.detach(), action_low, action_high)

    def make_critic(hidden_dim=256):
        layers = [layer_init(nn.Linear(obs_dim, hidden_dim)), nn.Tanh()]
        for _ in range(args.critic_layers - 1):
            layers += [layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh()]
        layers.append(layer_init(nn.Linear(hidden_dim, 1)))
        return nn.Sequential(*layers).to(device)

    # ── State save/restore helpers ──
    def _clone_state(sd):
        if isinstance(sd, dict):
            return {k: _clone_state(v) for k, v in sd.items()}
        return sd.clone()

    def _expand_state(sd, repeats):
        if isinstance(sd, dict):
            return {k: _expand_state(v, repeats) for k, v in sd.items()}
        if isinstance(sd, torch.Tensor) and sd.dim() > 0:
            return sd.repeat_interleave(repeats, dim=0)
        return sd

    # ══════════════════════════════════════════════════════════════════
    #  Phase 1: Collect 100 rollouts (save states for rollout 0)
    # ══════════════════════════════════════════════════════════════════
    T, E = args.num_steps, args.num_envs
    print(f"\nPhase 1: Collecting {max_rollouts} rollouts "
          f"({max_rollouts * E * T:,} transitions)...")
    sys.stdout.flush()
    t0 = time.time()
    data_pool = []

    for ri in range(max_rollouts):
        roll_obs = torch.zeros(T, E, obs_dim, device=device)
        roll_actions = torch.zeros(T, E, action_dim, device=device)
        roll_rewards = torch.zeros(T, E, device=device)
        roll_dones = torch.zeros(T, E, device=device)

        saved_states = [] if ri == 0 else None

        next_obs, _ = envs.reset(seed=args.seed + ri)
        next_done = torch.zeros(E, device=device)

        with torch.no_grad():
            for step in range(T):
                if ri == 0:
                    saved_states.append(_clone_state(envs.base_env.get_state_dict()))
                roll_obs[step] = next_obs
                roll_dones[step] = next_done
                action = agent.get_action(next_obs, deterministic=False)
                roll_actions[step] = action
                next_obs, reward, term, trunc, infos = envs.step(clip_action(action))
                next_done = (term | trunc).float()
                roll_rewards[step] = reward.view(-1)

        # MC1 returns
        mc1 = torch.zeros(T, E, device=device)
        future = torch.zeros(E, device=device)
        for t in reversed(range(T)):
            if t == T - 1:
                mask = 1.0 - next_done
            else:
                mask = 1.0 - roll_dones[t + 1]
            future = roll_rewards[t] + args.gamma * future * mask
            mc1[t] = future

        data_pool.append(dict(
            obs=roll_obs, actions=roll_actions,
            rewards=roll_rewards, dones=roll_dones,
            next_obs=next_obs.clone(), next_done=next_done.clone(),
            mc1_returns=mc1, saved_states=saved_states,
        ))

        if (ri + 1) % 20 == 0 or ri + 1 == max_rollouts:
            print(f"  {ri + 1}/{max_rollouts} rollouts collected")
            sys.stdout.flush()

    print(f"  Phase 1 done ({time.time() - t0:.1f}s)")

    # ══════════════════════════════════════════════════════════════════
    #  Phase 2: Dual MC16 ground truth (interleaved on-policy + optimal)
    # ══════════════════════════════════════════════════════════════════
    samples_per_env = 2 * args.mc_samples  # 16 opt + 16 on-policy
    num_mc_envs = E * samples_per_env
    print(f"\nPhase 2: Computing dual MC{args.mc_samples} ground truth "
          f"({num_mc_envs} mc_envs)...")
    sys.stdout.flush()
    t0 = time.time()

    mc_envs_raw = gym.make(args.env_id, num_envs=num_mc_envs, **env_kwargs)
    if isinstance(mc_envs_raw.action_space, gym.spaces.Dict):
        mc_envs_raw = FlattenActionSpaceWrapper(mc_envs_raw)
    mc_envs = ManiSkillVectorEnv(
        mc_envs_raw, num_mc_envs, ignore_terminations=False, record_metrics=False,
    )
    mc_zero_action = torch.zeros(
        num_mc_envs, *envs.single_action_space.shape, device=device)

    def _restore_mc_state(sd, seed=None):
        mc_envs.reset(seed=seed if seed is not None else args.seed)
        mc_envs.base_env.set_state_dict(sd)
        mc_envs.base_env.step(mc_zero_action)
        mc_envs.base_env.set_state_dict(sd)
        mc_envs.base_env._elapsed_steps[:] = 0
        return mc_envs.base_env.get_obs()

    # Precompute replica indices (per env: first 16 = optimal, next 16 = on-policy)
    opt_indices = []
    on_indices = []
    for i in range(E):
        base = i * samples_per_env
        opt_indices.extend(range(base, base + args.mc_samples))
        on_indices.extend(range(base + args.mc_samples, base + 2 * args.mc_samples))
    opt_indices = torch.tensor(opt_indices, device=device, dtype=torch.long)
    on_indices = torch.tensor(on_indices, device=device, dtype=torch.long)

    eval_saved_states = data_pool[0]['saved_states']
    opt_mc16 = torch.zeros(T, E, device=device)
    on_mc16 = torch.zeros(T, E, device=device)

    with torch.no_grad():
        for t in range(T):
            expanded = _expand_state(eval_saved_states[t], samples_per_env)
            mc_obs = _restore_mc_state(expanded, seed=args.seed + 1000 + t)

            env_done = torch.zeros(num_mc_envs, device=device).bool()
            all_rews = []
            for _ in range(args.max_episode_steps):
                if env_done.all():
                    break
                a = torch.zeros(num_mc_envs, action_dim, device=device)
                a[opt_indices] = optimal_agent.get_action(
                    mc_obs[opt_indices], deterministic=False)
                a[on_indices] = agent.get_action(
                    mc_obs[on_indices], deterministic=False)
                mc_obs, rew, term, trunc, _ = mc_envs.step(clip_action(a))
                all_rews.append(rew.view(-1) * (~env_done).float())
                env_done = env_done | (term | trunc).view(-1).bool()

            ret = torch.zeros(num_mc_envs, device=device)
            for s in reversed(range(len(all_rews))):
                ret = all_rews[s] + args.gamma * ret
            ret = ret.view(E, samples_per_env)
            opt_mc16[t] = ret[:, :args.mc_samples].mean(dim=1)
            on_mc16[t] = ret[:, args.mc_samples:2 * args.mc_samples].mean(dim=1)
            del all_rews

            if (t + 1) % 10 == 0 or t + 1 == T:
                print(f"  MC{args.mc_samples} step {t + 1}/{T}")
                sys.stdout.flush()

    mc_envs.close()
    del mc_envs, mc_envs_raw, mc_zero_action
    data_pool[0]['saved_states'] = None
    del eval_saved_states
    torch.cuda.empty_cache()

    # Sanity check
    on_flat = on_mc16.reshape(-1).cpu().numpy()
    opt_flat = opt_mc16.reshape(-1).cpu().numpy()
    r_on_opt = pearsonr(on_flat, opt_flat)[0]
    print(f"  On-policy MC16: mean={on_mc16.mean():.4f}, std={on_mc16.std():.4f}")
    print(f"  Optimal  MC16: mean={opt_mc16.mean():.4f}, std={opt_mc16.std():.4f}")
    print(f"  On vs Opt MC16 correlation: r={r_on_opt:.4f}")
    print(f"  Phase 2 done ({time.time() - t0:.1f}s)")
    sys.stdout.flush()

    # Eval set
    eval_obs = data_pool[0]['obs']  # (T, E, D)

    # Close rollout envs (not needed anymore for training)
    envs.close()
    del envs, agent, optimal_agent
    torch.cuda.empty_cache()

    # IQL args (used in train_iql_onpolicy via closure)
    iql_args_ns = SimpleNamespace(
        lr=args.iql_lr, weight_decay=1e-4, epochs=args.iql_epochs,
        batch_size=args.iql_batch_size, gamma=args.gamma,
        expectile_tau=args.iql_expectile_tau, tau_polyak=0.005,
        patience=args.iql_patience, grad_clip=0.5,
    )

    # ══════════════════════════════════════════════════════════════════
    #  Helper: evaluate critic on eval set
    # ══════════════════════════════════════════════════════════════════
    def evaluate(critic, mc16_gt):
        """Return (pearson_r, spearman_rho) against mc16_gt on eval set."""
        with torch.no_grad():
            v = critic(eval_obs.reshape(-1, obs_dim)).view(-1).cpu().numpy()
        g = mc16_gt.reshape(-1).cpu().numpy()
        return pearsonr(v, g)[0], spearmanr(v, g).correlation

    # ══════════════════════════════════════════════════════════════════
    #  V training functions (parameterized by hidden_dim)
    # ══════════════════════════════════════════════════════════════════
    def combine(n):
        d = data_pool[:n]
        return (
            torch.cat([x['obs'] for x in d], dim=1),
            torch.cat([x['actions'] for x in d], dim=1),
            torch.cat([x['rewards'] for x in d], dim=1),
            torch.cat([x['dones'] for x in d], dim=1),
            torch.cat([x['next_obs'] for x in d], dim=0),
            torch.cat([x['next_done'] for x in d], dim=0),
            torch.cat([x['mc1_returns'] for x in d], dim=1),
        )

    def train_mc(obs, mc_returns, hidden_dim=256):
        Tl, El, D = obs.shape
        critic = make_critic(hidden_dim)
        opt = optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)
        flat_obs = obs.reshape(-1, D)
        flat_ret = mc_returns.reshape(-1)
        N = flat_obs.shape[0]
        mb = min(args.minibatch_size, N)

        # Train/val split for early stopping
        perm_all = torch.randperm(N, device=device)
        val_size = max(mb, int(N * 0.1))
        val_idx = perm_all[:val_size]
        train_idx = perm_all[val_size:]
        N_train = train_idx.shape[0]

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0
        patience = 20

        total_epochs = args.gae_iters * args.gae_epochs
        max_epochs = total_epochs
        max_epochs = max(max_epochs, 200)

        for epoch in range(max_epochs):
            train_perm = train_idx[torch.randperm(N_train, device=device)]
            for start in range(0, N_train, mb):
                idx = train_perm[start:start + mb]
                loss = 0.5 * ((critic(flat_obs[idx]).view(-1) - flat_ret[idx]) ** 2).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5); opt.step()

            critic.eval()
            with torch.no_grad():
                val_loss = 0.5 * ((critic(flat_obs[val_idx]).view(-1) - flat_ret[val_idx]) ** 2).mean().item()
            critic.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in critic.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"    MC1 early stop epoch {epoch + 1}/{max_epochs}")
                    break

        if best_state is not None:
            critic.load_state_dict(best_state)
        return critic

    def train_td(obs, rewards, dones, next_obs, next_done,
                 reward_scale=1.0, hidden_dim=256):
        Tl, El, D = obs.shape
        critic = make_critic(hidden_dim)
        opt = optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)

        flat_s = obs.reshape(-1, D)
        flat_r = rewards.reshape(-1) * reward_scale

        flat_ns = torch.zeros_like(obs)
        flat_ns[:-1] = obs[1:]
        flat_ns[-1] = next_obs
        flat_ns = flat_ns.reshape(-1, D)

        flat_d = torch.zeros_like(rewards)
        flat_d[:-1] = dones[1:]
        flat_d[-1] = next_done
        flat_d = flat_d.reshape(-1)

        N = flat_s.shape[0]
        mb = min(args.minibatch_size, N)

        # Train/val split for early stopping
        perm_all = torch.randperm(N, device=device)
        val_size = max(mb, int(N * 0.1))
        val_idx = perm_all[:val_size]
        train_idx = perm_all[val_size:]
        N_train = train_idx.shape[0]

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0
        patience = 20

        max_epochs = args.td_epochs
        max_epochs = max(max_epochs, 200)  # at least 200 epochs

        critic.train()
        for epoch in range(max_epochs):
            train_perm = train_idx[torch.randperm(N_train, device=device)]
            for start in range(0, N_train, mb):
                idx = train_perm[start:start + mb]
                with torch.no_grad():
                    target = (flat_r[idx] + args.gamma *
                              critic(flat_ns[idx]).view(-1) * (1 - flat_d[idx]))
                loss = 0.5 * ((critic(flat_s[idx]).view(-1) - target) ** 2).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5); opt.step()

            # Validation every epoch
            critic.eval()
            with torch.no_grad():
                v_target_val = (flat_r[val_idx] + args.gamma *
                                critic(flat_ns[val_idx]).view(-1) * (1 - flat_d[val_idx]))
                v_pred_val = critic(flat_s[val_idx]).view(-1)
                val_loss = 0.5 * ((v_pred_val - v_target_val) ** 2).mean().item()
            critic.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in critic.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"    TD+rs early stop epoch {epoch + 1}/{max_epochs}")
                    break

        if best_state is not None:
            critic.load_state_dict(best_state)

        if reward_scale != 1.0:
            with torch.no_grad():
                critic[-1].weight.div_(reward_scale)
                critic[-1].bias.div_(reward_scale)
        return critic

    def train_td_ema(obs, rewards, dones, next_obs, next_done,
                     reward_scale=1.0, ema_tau=0.005, hidden_dim=256):
        Tl, El, D = obs.shape
        critic = make_critic(hidden_dim)
        critic_target = make_critic(hidden_dim)
        critic_target.load_state_dict(critic.state_dict())
        opt = optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)

        flat_s = obs.reshape(-1, D)
        flat_r = rewards.reshape(-1) * reward_scale

        flat_ns = torch.zeros_like(obs)
        flat_ns[:-1] = obs[1:]
        flat_ns[-1] = next_obs
        flat_ns = flat_ns.reshape(-1, D)

        flat_d = torch.zeros_like(rewards)
        flat_d[:-1] = dones[1:]
        flat_d[-1] = next_done
        flat_d = flat_d.reshape(-1)

        N = flat_s.shape[0]
        mb = min(args.minibatch_size, N)

        # Train/val split for early stopping
        perm_all = torch.randperm(N, device=device)
        val_size = max(mb, int(N * 0.1))
        val_idx = perm_all[:val_size]
        train_idx = perm_all[val_size:]
        N_train = train_idx.shape[0]

        best_val_loss = float("inf")
        best_state = None
        best_target_state = None
        no_improve = 0
        patience = 20

        max_epochs = args.td_epochs
        max_epochs = max(max_epochs, 200)  # at least 200 epochs

        critic.train()
        for epoch in range(max_epochs):
            train_perm = train_idx[torch.randperm(N_train, device=device)]
            for start in range(0, N_train, mb):
                idx = train_perm[start:start + mb]
                with torch.no_grad():
                    target = (flat_r[idx] + args.gamma *
                              critic_target(flat_ns[idx]).view(-1) * (1 - flat_d[idx]))
                loss = 0.5 * ((critic(flat_s[idx]).view(-1) - target) ** 2).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5); opt.step()
                with torch.no_grad():
                    for p, pt in zip(critic.parameters(), critic_target.parameters()):
                        pt.data.mul_(1 - ema_tau).add_(p.data, alpha=ema_tau)

            # Validation every epoch
            critic.eval()
            with torch.no_grad():
                v_target_val = (flat_r[val_idx] + args.gamma *
                                critic_target(flat_ns[val_idx]).view(-1) * (1 - flat_d[val_idx]))
                v_pred_val = critic(flat_s[val_idx]).view(-1)
                val_loss = 0.5 * ((v_pred_val - v_target_val) ** 2).mean().item()
            critic.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in critic.state_dict().items()}
                best_target_state = {k: v.clone() for k, v in critic_target.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"    TD+EMA early stop epoch {epoch + 1}/{max_epochs}")
                    break

        if best_state is not None:
            critic.load_state_dict(best_state)

        if reward_scale != 1.0:
            with torch.no_grad():
                critic[-1].weight.div_(reward_scale)
                critic[-1].bias.div_(reward_scale)
        return critic

    def train_gae(obs, rewards, dones, next_obs, next_done, hidden_dim=256):
        Tl, El, D = obs.shape
        critic = make_critic(hidden_dim)
        opt = optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)
        flat_obs = obs.reshape(-1, D)
        N = flat_obs.shape[0]
        mb = min(args.minibatch_size, N)

        for _ in range(args.gae_iters):
            with torch.no_grad():
                values = torch.stack([critic(obs[t]).flatten() for t in range(Tl)])
                nv = critic(next_obs).reshape(1, -1)
                adv = torch.zeros_like(rewards)
                lastgaelam = 0
                for t in reversed(range(Tl)):
                    if t == Tl - 1:
                        nnd, nvs = 1.0 - next_done, nv
                    else:
                        nnd, nvs = 1.0 - dones[t + 1], values[t + 1]
                    delta = rewards[t] + args.gamma * nnd * nvs - values[t]
                    adv[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nnd * lastgaelam
                flat_ret = (adv + values).reshape(-1)

            critic.train()
            for _ in range(args.gae_epochs):
                perm = torch.randperm(N, device=device)
                for start in range(0, N, mb):
                    idx = perm[start:start + mb]
                    loss = 0.5 * ((critic(flat_obs[idx]).view(-1) - flat_ret[idx]) ** 2).mean()
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), 0.5); opt.step()
        return critic

    def train_iql_onpolicy(obs, actions, rewards, dones, next_obs, next_done,
                           hidden_dim=256):
        """Train IQL on on-policy rollout data (same format as TD/GAE).
        Converts (T, E, ...) rollout format to flat (N, ...) IQL format."""
        Tl, El, D = obs.shape
        A = actions.shape[-1]

        flat_s = obs.reshape(-1, D)
        flat_a = actions.reshape(-1, A)
        flat_r = rewards.reshape(-1)

        # Build next_states: obs[t+1] for t<T-1, next_obs for t=T-1
        flat_ns = torch.zeros_like(obs)
        flat_ns[:-1] = obs[1:]
        flat_ns[-1] = next_obs
        flat_ns = flat_ns.reshape(-1, D)

        # Build done flags for transitions
        flat_d = torch.zeros_like(rewards)
        flat_d[:-1] = dones[1:]
        flat_d[-1] = next_done
        flat_d = flat_d.reshape(-1)

        # Move to CPU for IQL (it handles device internally)
        v_net = train_iql_var(
            flat_s.cpu(), flat_a.cpu(), flat_r.cpu(), flat_ns.cpu(), flat_d.cpu(),
            device, iql_args_ns, hidden_dim=hidden_dim,
        )
        return v_net

    # ══════════════════════════════════════════════════════════════════
    #  Phase 4: Data scaling (hidden=256, vary rollouts)
    # ══════════════════════════════════════════════════════════════════
    METHOD_NAMES = ["MC1", "TD+rs10", "TD+EMA", "GAE", "IQL"]
    NUM_METHODS = len(METHOD_NAMES)

    print(f"\n{'=' * 70}")
    print(f"Phase 4: Data scaling (hidden=256)")
    print(f"{'=' * 70}")
    sys.stdout.flush()

    # data_results[N] = {method: (on_r, on_rho, opt_r, opt_rho)}
    data_results = {}
    for N_roll in args.rollout_counts:
        trans = N_roll * E * T
        steps_per_epoch = trans // args.minibatch_size
        print(f"\n--- N={N_roll} rollouts ({trans:,} trans, {steps_per_epoch} steps/ep, {steps_per_epoch * args.td_epochs} total steps) ---")
        sys.stdout.flush()
        obs, actions, rewards, dones, next_obs, next_done, mc1_gt = combine(N_roll)

        row = {}

        t0 = time.time()
        c = train_mc(obs, mc1_gt, hidden_dim=256)
        row["MC1"] = (*evaluate(c, on_mc16), *evaluate(c, opt_mc16))
        print(f"  MC1:     on_r={row['MC1'][0]:.4f} opt_r={row['MC1'][2]:.4f} ({time.time()-t0:.1f}s)")
        del c

        t0 = time.time()
        c = train_td(obs, rewards, dones, next_obs, next_done,
                     reward_scale=args.td_reward_scale, hidden_dim=256)
        row["TD+rs10"] = (*evaluate(c, on_mc16), *evaluate(c, opt_mc16))
        print(f"  TD+rs10: on_r={row['TD+rs10'][0]:.4f} opt_r={row['TD+rs10'][2]:.4f} ({time.time()-t0:.1f}s)")
        del c

        t0 = time.time()
        c = train_td_ema(obs, rewards, dones, next_obs, next_done,
                         reward_scale=args.td_reward_scale,
                         ema_tau=args.ema_tau, hidden_dim=256)
        row["TD+EMA"] = (*evaluate(c, on_mc16), *evaluate(c, opt_mc16))
        print(f"  TD+EMA:  on_r={row['TD+EMA'][0]:.4f} opt_r={row['TD+EMA'][2]:.4f} ({time.time()-t0:.1f}s)")
        del c

        t0 = time.time()
        c = train_gae(obs, rewards, dones, next_obs, next_done, hidden_dim=256)
        row["GAE"] = (*evaluate(c, on_mc16), *evaluate(c, opt_mc16))
        print(f"  GAE:     on_r={row['GAE'][0]:.4f} opt_r={row['GAE'][2]:.4f} ({time.time()-t0:.1f}s)")
        del c

        t0 = time.time()
        c = train_iql_onpolicy(obs, actions, rewards, dones, next_obs, next_done,
                               hidden_dim=256)
        row["IQL"] = (*evaluate(c, on_mc16), *evaluate(c, opt_mc16))
        print(f"  IQL:     on_r={row['IQL'][0]:.4f} opt_r={row['IQL'][2]:.4f} ({time.time()-t0:.1f}s)")
        del c

        data_results[N_roll] = row
        del obs, actions, rewards, dones, next_obs, next_done, mc1_gt
        torch.cuda.empty_cache()
        sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════════
    #  Phase 5: Network size scaling (rollouts=100, vary hidden)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"Phase 5: Network size scaling (rollouts={max_rollouts})")
    print(f"{'=' * 70}")
    sys.stdout.flush()

    obs_all, act_all, rewards_all, dones_all, next_obs_all, next_done_all, mc1_all = combine(max_rollouts)

    # net_results[hidden] = {method: (on_r, on_rho, opt_r, opt_rho)}
    net_results = {}
    for hd in args.hidden_dims:
        print(f"\n--- hidden={hd} ---")
        sys.stdout.flush()
        row = {}

        t0 = time.time()
        c = train_mc(obs_all, mc1_all, hidden_dim=hd)
        row["MC1"] = (*evaluate(c, on_mc16), *evaluate(c, opt_mc16))
        print(f"  MC1:     on_r={row['MC1'][0]:.4f} opt_r={row['MC1'][2]:.4f} ({time.time()-t0:.1f}s)")
        del c

        t0 = time.time()
        c = train_td(obs_all, rewards_all, dones_all, next_obs_all, next_done_all,
                     reward_scale=args.td_reward_scale, hidden_dim=hd)
        row["TD+rs10"] = (*evaluate(c, on_mc16), *evaluate(c, opt_mc16))
        print(f"  TD+rs10: on_r={row['TD+rs10'][0]:.4f} opt_r={row['TD+rs10'][2]:.4f} ({time.time()-t0:.1f}s)")
        del c

        t0 = time.time()
        c = train_td_ema(obs_all, rewards_all, dones_all, next_obs_all, next_done_all,
                         reward_scale=args.td_reward_scale,
                         ema_tau=args.ema_tau, hidden_dim=hd)
        row["TD+EMA"] = (*evaluate(c, on_mc16), *evaluate(c, opt_mc16))
        print(f"  TD+EMA:  on_r={row['TD+EMA'][0]:.4f} opt_r={row['TD+EMA'][2]:.4f} ({time.time()-t0:.1f}s)")
        del c

        t0 = time.time()
        c = train_gae(obs_all, rewards_all, dones_all, next_obs_all, next_done_all,
                      hidden_dim=hd)
        row["GAE"] = (*evaluate(c, on_mc16), *evaluate(c, opt_mc16))
        print(f"  GAE:     on_r={row['GAE'][0]:.4f} opt_r={row['GAE'][2]:.4f} ({time.time()-t0:.1f}s)")
        del c

        t0 = time.time()
        c = train_iql_onpolicy(obs_all, act_all, rewards_all, dones_all,
                               next_obs_all, next_done_all, hidden_dim=hd)
        row["IQL"] = (*evaluate(c, on_mc16), *evaluate(c, opt_mc16))
        print(f"  IQL:     on_r={row['IQL'][0]:.4f} opt_r={row['IQL'][2]:.4f} ({time.time()-t0:.1f}s)")
        del c

        net_results[hd] = row
        torch.cuda.empty_cache()
        sys.stdout.flush()

    del obs_all, act_all, rewards_all, dones_all, next_obs_all, next_done_all, mc1_all
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════
    #  Phase 6: Print tables + plot + save
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print(f"DATA SCALING (hidden=256) — Pearson r")
    print(f"{'=' * 90}")
    header = f"| {'Rollouts':>8} |"
    for m in METHOD_NAMES:
        header += f" {m+' on':>10} | {m+' opt':>10} |"
    print(header)
    print("|" + "-" * 10 + "|" + (("-" * 12 + "|" + "-" * 12 + "|") * NUM_METHODS))
    for N_roll in args.rollout_counts:
        row = data_results[N_roll]
        line = f"| {N_roll:>8} |"
        for m in METHOD_NAMES:
            on_r, _, opt_r, _ = row[m]
            line += f" {on_r:>10.4f} | {opt_r:>10.4f} |"
        print(line)
    print(f"{'=' * 90}")

    print(f"\nNETWORK SIZE SCALING (rollouts={max_rollouts}) — Pearson r")
    print(f"{'=' * 90}")
    header = f"| {'Hidden':>8} |"
    for m in METHOD_NAMES:
        header += f" {m+' on':>10} | {m+' opt':>10} |"
    print(header)
    print("|" + "-" * 10 + "|" + (("-" * 12 + "|" + "-" * 12 + "|") * NUM_METHODS))
    for hd in args.hidden_dims:
        row = net_results[hd]
        line = f"| {hd:>8} |"
        for m in METHOD_NAMES:
            on_r, _, opt_r, _ = row[m]
            line += f" {on_r:>10.4f} | {opt_r:>10.4f} |"
        print(line)
    print(f"{'=' * 90}")
    sys.stdout.flush()

    # ── Save raw results ──
    save_data = {
        "data_results": data_results,
        "net_results": net_results,
        "rollout_counts": list(args.rollout_counts),
        "hidden_dims": list(args.hidden_dims),
        "method_names": METHOD_NAMES,
        "on_mc16": on_mc16.cpu(),
        "opt_mc16": opt_mc16.cpu(),
        "eval_obs": eval_obs.cpu(),
        "args": vars(args),
    }
    data_path = args.output.replace(".png", ".pt")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(save_data, data_path)
    print(f"\nSaved raw results to {data_path}")

    # ── Plot ──
    colors = {
        "MC1": "#1f77b4",
        "TD+rs10": "#ff7f0e",
        "TD+EMA": "#2ca02c",
        "GAE": "#d62728",
        "IQL": "#9467bd",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Row 1: Data scaling
    rollouts = list(args.rollout_counts)
    for col, (gt_label, gt_idx) in enumerate([
        ("On-policy MC16", 0), ("Optimal MC16", 2)
    ]):
        ax = axes[0, col]
        for m in METHOD_NAMES:
            ys = [data_results[n][m][gt_idx] for n in rollouts]
            ax.plot(rollouts, ys, "o-", color=colors[m], lw=2, label=m)
        ax.set_xscale("log")
        ax.set_xlabel("Rollouts (log scale)")
        ax.set_ylabel("Pearson r")
        ax.set_title(f"Data Scaling — vs {gt_label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.05)

    # Row 2: Network size scaling
    hiddens = list(args.hidden_dims)
    for col, (gt_label, gt_idx) in enumerate([
        ("On-policy MC16", 0), ("Optimal MC16", 2)
    ]):
        ax = axes[1, col]
        for m in METHOD_NAMES:
            ys = [net_results[h][m][gt_idx] for h in hiddens]
            ax.plot(hiddens, ys, "o-", color=colors[m], lw=2, label=m)
        ax.set_xscale("log", base=2)
        ax.set_xticks(hiddens)
        ax.set_xticklabels([str(h) for h in hiddens])
        ax.set_xlabel("Hidden dim (log2 scale)")
        ax.set_ylabel("Pearson r")
        ax.set_title(f"Network Scaling — vs {gt_label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.05)

    fig.suptitle(
        f"V Scaling Law | gamma={args.gamma}, lambda={args.gae_lambda}\n"
        f"TD uses rs={int(args.td_reward_scale)}, EMA tau={args.ema_tau}",
        fontsize=13,
    )
    plt.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {args.output}")
    plt.close(fig)
