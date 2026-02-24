"""Compare V learning methods: TD(0), TD(10), GAE on same rollout data.

Trains V three ways on identical PegInsertion/PickCube rollout data,
evaluates against MC16 ground truth (Pearson r).

Usage:
  python -u -m RL.v_learning_compare --env_id PegInsertionSide-v1 \
    --checkpoint runs/peginsertion_ppo_ema99/ckpt_231.pt \
    --gamma 0.97 --num_steps 100 --max_episode_steps 100
"""

import copy, os, random, sys, time
from dataclasses import dataclass
from typing import Literal

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
from scipy.stats import pearsonr
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from data.data_collection.ppo import Agent, layer_init


@dataclass
class Args:
    checkpoint: str = "runs/peginsertion_ppo_ema99/ckpt_231.pt"
    gamma: float = 0.97
    num_envs: int = 100
    num_steps: int = 100
    max_episode_steps: int = 100
    mc_samples: int = 16
    N_values: tuple[int, ...] = (2, 3, 5)
    # Training
    td_reward_scale: float = 10.0
    ema_tau: float = 0.005
    epochs: int = 2000
    lr: float = 3e-4
    batch_size: int = 1000
    eval_every: int = 5
    critic_layers: int = 3
    hidden_dim: int = 256
    gae_lambda: float = 0.95
    gae_recompute_every: int = 200  # recompute GAE returns every K epochs
    seed: int = 1
    env_id: str = "PegInsertionSide-v1"
    reward_mode: str = "sparse"
    control_mode: str = "pd_joint_delta_pos"
    output: str = "runs/v_learning_compare.png"
    mc_cache: str = ""


if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda")
    T, E = args.num_steps, args.num_envs
    max_N = max(args.N_values)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_kwargs = dict(
        obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda",
        reward_mode=args.reward_mode, control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
    )

    envs = gym.make(args.env_id, num_envs=E, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(envs, E, ignore_terminations=False, record_metrics=True)
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()

    def clip_action(a):
        return torch.clamp(a.detach(), action_low, action_high)

    def make_v_net():
        layers = [layer_init(nn.Linear(obs_dim, args.hidden_dim)), nn.Tanh()]
        for _ in range(args.critic_layers - 1):
            layers += [layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)), nn.Tanh()]
        layers.append(layer_init(nn.Linear(args.hidden_dim, 1)))
        return nn.Sequential(*layers).to(device)

    def _clone_state_cpu(sd):
        if isinstance(sd, dict):
            return {k: _clone_state_cpu(v) for k, v in sd.items()}
        if isinstance(sd, torch.Tensor):
            return sd.cpu().clone()
        return sd

    def _expand_state(sd, repeats):
        if isinstance(sd, dict):
            return {k: _expand_state(v, repeats) for k, v in sd.items()}
        if isinstance(sd, torch.Tensor) and sd.dim() > 0:
            return sd.repeat_interleave(repeats, dim=0)
        return sd

    def _state_to_device(sd, dev):
        if isinstance(sd, dict):
            return {k: _state_to_device(v, dev) for k, v in sd.items()}
        if isinstance(sd, torch.Tensor):
            return sd.to(dev)
        return sd

    # ═══ Phase 1: Collect rollouts ═══
    print(f"Phase 1: Collecting {max_N} rollouts...")
    sys.stdout.flush()
    t0 = time.time()

    rollout_data = []  # list of (obs, rewards, dones, next_obs, next_done)
    seed_states = [None] * T

    for ri in range(max_N):
        roll_obs = torch.zeros(T, E, obs_dim, device=device)
        roll_rewards = torch.zeros(T, E, device=device)
        roll_dones = torch.zeros(T, E, device=device)

        next_obs, _ = envs.reset(seed=args.seed + ri)
        next_done = torch.zeros(E, device=device)

        with torch.no_grad():
            for step in range(T):
                if ri == 0:
                    seed_states[step] = _clone_state_cpu(envs.base_env.get_state_dict())
                roll_obs[step] = next_obs
                roll_dones[step] = next_done
                action = agent.get_action(next_obs, deterministic=False)
                next_obs, reward, term, trunc, _ = envs.step(clip_action(action))
                next_done = (term | trunc).float()
                roll_rewards[step] = reward.view(-1)

        rollout_data.append(dict(
            obs=roll_obs, rewards=roll_rewards, dones=roll_dones,
            next_obs=next_obs.clone(), next_done=next_done.clone(),
        ))

    print(f"  Done ({time.time() - t0:.1f}s)")

    # ═══ Phase 2: MC16 ground truth ═══
    eval_obs = rollout_data[0]['obs']  # (T, E, D)

    ckpt_base = os.path.splitext(os.path.basename(args.checkpoint))[0]
    mc_cache_path = args.mc_cache or (
        f"runs/mc16_cache_{args.env_id}_{ckpt_base}_g{args.gamma}_s{args.seed}"
        f"_E{E}_T{T}_mc{args.mc_samples}.pt"
    )

    if os.path.exists(mc_cache_path):
        print(f"\nPhase 2: Loading MC16 from cache: {mc_cache_path}")
        cached = torch.load(mc_cache_path, map_location=device)
        on_mc16 = cached['mc16'].to(device)
    else:
        num_mc_envs = E * args.mc_samples
        print(f"\nPhase 2: Computing MC16 ({num_mc_envs} mc_envs)...")
        sys.stdout.flush()
        t0 = time.time()

        mc_envs_raw = gym.make(args.env_id, num_envs=num_mc_envs, **env_kwargs)
        if isinstance(mc_envs_raw.action_space, gym.spaces.Dict):
            mc_envs_raw = FlattenActionSpaceWrapper(mc_envs_raw)
        mc_envs = ManiSkillVectorEnv(mc_envs_raw, num_mc_envs, ignore_terminations=False, record_metrics=False)
        mc_zero_action = torch.zeros(num_mc_envs, *envs.single_action_space.shape, device=device)

        def _restore_mc_state(sd, seed=None):
            mc_envs.reset(seed=seed if seed is not None else args.seed)
            mc_envs.base_env.set_state_dict(sd)
            mc_envs.base_env.step(mc_zero_action)
            mc_envs.base_env.set_state_dict(sd)
            mc_envs.base_env._elapsed_steps[:] = 0
            return mc_envs.base_env.get_obs()

        on_mc16 = torch.zeros(T, E, device=device)
        with torch.no_grad():
            for t in range(T):
                expanded = _expand_state(_state_to_device(seed_states[t], device), args.mc_samples)
                mc_obs = _restore_mc_state(expanded, seed=args.seed + 1000 + t)
                env_done = torch.zeros(num_mc_envs, device=device).bool()
                all_rews = []
                for _ in range(args.max_episode_steps):
                    if env_done.all():
                        break
                    a = agent.get_action(mc_obs, deterministic=False)
                    mc_obs, rew, term, trunc, _ = mc_envs.step(clip_action(a))
                    all_rews.append(rew.view(-1) * (~env_done).float())
                    env_done = env_done | (term | trunc).view(-1).bool()
                ret = torch.zeros(num_mc_envs, device=device)
                for s in reversed(range(len(all_rews))):
                    ret = all_rews[s] + args.gamma * ret
                on_mc16[t] = ret.view(E, args.mc_samples).mean(dim=1)
                if (t + 1) % 10 == 0:
                    print(f"  MC16 step {t + 1}/{T}")
                    sys.stdout.flush()

        mc_envs.close()
        del mc_envs, mc_envs_raw
        torch.cuda.empty_cache()
        os.makedirs(os.path.dirname(mc_cache_path) or ".", exist_ok=True)
        torch.save({'mc16': on_mc16.cpu()}, mc_cache_path)
        print(f"  Phase 2 done ({time.time() - t0:.1f}s)")

    mc16_flat = on_mc16.reshape(-1).cpu().numpy()
    print(f"  MC16: mean={on_mc16.mean():.4f}, std={on_mc16.std():.4f}")

    # ═══ Phase 3: Prepare data ═══
    rs = args.td_reward_scale
    gamma = args.gamma

    def compute_r(critic, scale=1.0):
        with torch.no_grad():
            v = critic(eval_obs.reshape(-1, obs_dim)).view(-1).cpu().numpy() / scale
        return pearsonr(v, mc16_flat)[0]

    def build_td0_flat(n_rollouts):
        """Standard TD(0) flat data: (s, r*rs, s', gamma*(1-d))."""
        all_s, all_r, all_ns, all_mask = [], [], [], []
        for ri in range(n_rollouts):
            d = rollout_data[ri]
            obs_ri = d['obs']
            rew_ri = d['rewards']
            done_ri = d['dones']
            ns_ri = torch.zeros_like(obs_ri)
            ns_ri[:-1] = obs_ri[1:]
            ns_ri[-1] = d['next_obs']
            nd_ri = torch.zeros_like(rew_ri)
            nd_ri[:-1] = done_ri[1:]
            nd_ri[-1] = d['next_done']
            all_s.append(obs_ri.reshape(-1, obs_dim))
            all_r.append(rew_ri.reshape(-1) * rs)
            all_ns.append(ns_ri.reshape(-1, obs_dim))
            all_mask.append(gamma * (1 - nd_ri.reshape(-1)))
        return (torch.cat(all_s), torch.cat(all_r),
                torch.cat(all_ns), torch.cat(all_mask))

    def build_nstep_flat(n_rollouts, n_step):
        """N-step TD flat data: (s, G_scaled, s_boot, gamma_n_mask)."""
        all_s, all_G, all_ns, all_gn = [], [], [], []
        for ri in range(n_rollouts):
            d = rollout_data[ri]
            obs_ri = d['obs']       # (T, E, D)
            rew_ri = d['rewards']   # (T, E)
            done_ri = d['dones']    # (T, E)
            next_obs_ri = d['next_obs']   # (E, D)
            next_done_ri = d['next_done'] # (E,)

            # is_terminal[t, e] = 1 means action at t ended the episode
            is_terminal = torch.zeros(T, E, device=device)
            is_terminal[:-1] = done_ri[1:]
            is_terminal[-1] = next_done_ri

            G_seq = torch.zeros(T, E, device=device)
            ns_seq = torch.zeros(T, E, obs_dim, device=device)
            gn_seq = torch.zeros(T, E, device=device)

            for t in range(T):
                G = torch.zeros(E, device=device)
                gk = 1.0
                alive = torch.ones(E, device=device).bool()
                for k in range(n_step):
                    tk = t + k
                    if tk >= T:
                        break
                    G += (gk * rs) * rew_ri[tk] * alive.float()
                    alive = alive & (is_terminal[tk] < 0.5)
                    gk *= gamma

                G_seq[t] = G
                bt = min(t + n_step, T)
                if bt < T:
                    ns_seq[t] = obs_ri[bt]
                else:
                    ns_seq[t] = next_obs_ri
                gn_seq[t] = gk * alive.float()

            all_s.append(obs_ri.reshape(-1, obs_dim))
            all_G.append(G_seq.reshape(-1))
            all_ns.append(ns_seq.reshape(-1, obs_dim))
            all_gn.append(gn_seq.reshape(-1))

        return (torch.cat(all_s), torch.cat(all_G),
                torch.cat(all_ns), torch.cat(all_gn))

    # ═══ Training functions ═══

    def train_td_ema(flat_s, flat_G, flat_ns, flat_mask, label=""):
        """Unified TD training. target = G + mask * V_target(ns)."""
        N = flat_s.shape[0]
        mb = min(args.batch_size, N)
        critic = make_v_net()
        critic_target = copy.deepcopy(critic)
        opt = optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)

        epochs_log, r_log = [], []
        best_r, best_ep = -999, 0

        for epoch in range(args.epochs):
            critic.train()
            perm = torch.randperm(N, device=device)
            for start in range(0, N, mb):
                idx = perm[start:start + mb]
                with torch.no_grad():
                    target = flat_G[idx] + flat_mask[idx] * critic_target(flat_ns[idx]).view(-1)
                loss = 0.5 * ((critic(flat_s[idx]).view(-1) - target) ** 2).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5); opt.step()
                with torch.no_grad():
                    for p, pt in zip(critic.parameters(), critic_target.parameters()):
                        pt.data.mul_(1 - args.ema_tau).add_(p.data, alpha=args.ema_tau)

            if (epoch + 1) % args.eval_every == 0 or epoch == 0:
                critic.eval()
                r = compute_r(critic, rs)
                epochs_log.append(epoch + 1)
                r_log.append(r)
                if r > best_r:
                    best_r = r; best_ep = epoch + 1

        return epochs_log, r_log, best_r, best_ep

    def train_gae_v(n_rollouts, label=""):
        """GAE-style V learning: recompute GAE returns periodically, train V."""
        # Gather sequential data
        all_obs, all_rew, all_done = [], [], []
        all_next_obs, all_next_done = [], []
        for ri in range(n_rollouts):
            d = rollout_data[ri]
            all_obs.append(d['obs'])
            all_rew.append(d['rewards'])
            all_done.append(d['dones'])
            all_next_obs.append(d['next_obs'])
            all_next_done.append(d['next_done'])

        seq_obs = torch.cat(all_obs, dim=1)       # (T, N*E, D)
        seq_rew = torch.cat(all_rew, dim=1)        # (T, N*E)
        seq_done = torch.cat(all_done, dim=1)      # (T, N*E)
        seq_next_obs = torch.cat(all_next_obs, dim=0)   # (N*E, D)
        seq_next_done = torch.cat(all_next_done, dim=0)  # (N*E,)

        Tl, El = seq_rew.shape
        N_total = Tl * El
        flat_obs = seq_obs.reshape(-1, obs_dim)  # (T*N*E, D)

        critic = make_v_net()
        opt = optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)
        mb = min(args.batch_size, N_total)

        epochs_log, r_log = [], []
        best_r, best_ep = -999, 0
        recompute_freq = args.gae_recompute_every

        flat_returns = None

        for epoch in range(args.epochs):
            # Recompute GAE returns periodically
            if epoch % recompute_freq == 0:
                critic.eval()
                with torch.no_grad():
                    # Compute V for all states
                    v_all = torch.zeros(Tl, El, device=device)
                    for t in range(Tl):
                        v_all[t] = critic(seq_obs[t]).view(-1)
                    v_next = critic(seq_next_obs).view(-1)

                    # GAE computation
                    advantages = torch.zeros(Tl, El, device=device)
                    lastgaelam = torch.zeros(El, device=device)

                    for t in reversed(range(Tl)):
                        if t == Tl - 1:
                            nextnonterminal = 1.0 - seq_next_done
                            nextvalues = v_next
                        else:
                            nextnonterminal = 1.0 - seq_done[t + 1]
                            nextvalues = v_all[t + 1]
                        delta = seq_rew[t] + gamma * nextvalues * nextnonterminal - v_all[t]
                        advantages[t] = lastgaelam = delta + gamma * args.gae_lambda * nextnonterminal * lastgaelam

                    returns = advantages + v_all
                    flat_returns = returns.reshape(-1)

            # Train on fixed returns
            critic.train()
            perm = torch.randperm(N_total, device=device)
            for start in range(0, N_total, mb):
                idx = perm[start:start + mb]
                v_pred = critic(flat_obs[idx]).view(-1)
                loss = 0.5 * ((v_pred - flat_returns[idx]) ** 2).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5); opt.step()

            if (epoch + 1) % args.eval_every == 0 or epoch == 0:
                critic.eval()
                r = compute_r(critic, 1.0)  # GAE V is unscaled
                epochs_log.append(epoch + 1)
                r_log.append(r)
                if r > best_r:
                    best_r = r; best_ep = epoch + 1

        return epochs_log, r_log, best_r, best_ep

    # ═══ Phase 4: Run comparisons ═══
    methods = ["TD(0)", "TD(10)", "GAE"]
    results = {}  # (method, N) -> dict

    for N in args.N_values:
        n_trans = N * E * T
        print(f"\n{'='*60}")
        print(f"N={N} ({n_trans:,} transitions)")
        print(f"{'='*60}")

        # TD(0)
        print(f"  Training TD(0)+EMA...")
        sys.stdout.flush()
        t0 = time.time()
        s, r, ns, mask = build_td0_flat(N)
        ep, rl, pk_r, pk_ep = train_td_ema(s, r, ns, mask, f"TD0 N={N}")
        print(f"    peak r={pk_r:.4f} @ ep{pk_ep} ({time.time()-t0:.1f}s)")
        results[("TD(0)", N)] = dict(epochs=ep, r_log=rl, peak_r=pk_r, peak_ep=pk_ep)
        del s, r, ns, mask; torch.cuda.empty_cache()

        # TD(10)
        print(f"  Training TD(10)+EMA...")
        sys.stdout.flush()
        t0 = time.time()
        s, G, ns, gn = build_nstep_flat(N, 10)
        ep, rl, pk_r, pk_ep = train_td_ema(s, G, ns, gn, f"TD10 N={N}")
        print(f"    peak r={pk_r:.4f} @ ep{pk_ep} ({time.time()-t0:.1f}s)")
        results[("TD(10)", N)] = dict(epochs=ep, r_log=rl, peak_r=pk_r, peak_ep=pk_ep)
        del s, G, ns, gn; torch.cuda.empty_cache()

        # GAE
        print(f"  Training GAE (recompute every {args.gae_recompute_every} epochs)...")
        sys.stdout.flush()
        t0 = time.time()
        ep, rl, pk_r, pk_ep = train_gae_v(N, f"GAE N={N}")
        print(f"    peak r={pk_r:.4f} @ ep{pk_ep} ({time.time()-t0:.1f}s)")
        results[("GAE", N)] = dict(epochs=ep, r_log=rl, peak_r=pk_r, peak_ep=pk_ep)
        torch.cuda.empty_cache()

    # ═══ Phase 5: Summary ═══
    print(f"\n{'='*60}")
    print(f"SUMMARY — peak Pearson r (V learning comparison)")
    print(f"{'='*60}")
    header = f"| {'N':>5} |"
    for m in methods:
        header += f" {m:>10} |"
    print(header)
    sep = "|-------|"
    for _ in methods:
        sep += "------------|"
    print(sep)
    for N in args.N_values:
        row = f"| {N:>5} |"
        for m in methods:
            if (m, N) in results:
                row += f" {results[(m, N)]['peak_r']:>10.4f} |"
            else:
                row += f"        — |"
        print(row)

    # ═══ Phase 6: Plot ═══
    colors = {"TD(0)": "#e74c3c", "TD(10)": "#3498db", "GAE": "#2ecc71"}

    fig, axes = plt.subplots(1, len(args.N_values), figsize=(6 * len(args.N_values), 5))
    if len(args.N_values) == 1:
        axes = [axes]

    for i, N in enumerate(args.N_values):
        ax = axes[i]
        for m in methods:
            if (m, N) in results:
                res = results[(m, N)]
                ax.plot(res['epochs'], res['r_log'], '-', color=colors[m],
                        lw=1.5, alpha=0.8, label=f"{m} (pk={res['peak_r']:.3f})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Pearson r (vs MC16)")
        ax.set_title(f"N={N} ({N*E*T:,} transitions)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"V Learning Comparison | {args.env_id} | gamma={args.gamma} | rs={rs}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {args.output}")

    save_path = args.output.replace('.png', '.pt')
    torch.save(dict(results=results, args=vars(args)), save_path)
    print(f"Saved data to {save_path}")

    envs.close()
