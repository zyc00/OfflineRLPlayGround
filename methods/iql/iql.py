"""IQL (Implicit Q-Learning) advantage estimation.

Trains Q(s,a) and V(s) networks on offline data using expectile regression,
then computes advantages A(s,a) = Q(s,a) - V(s) on the eval dataset.

Reference: Kostrikov et al., "Offline Reinforcement Learning with Implicit
Q-Learning", ICLR 2022.
"""

import copy
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import tyro

from data.offline_dataset import OfflineRLDataset
from methods.gae.gae import Critic, layer_init


class QNetwork(nn.Module):
    """Q(s, a) network: 3-layer Tanh MLP on concatenated (state, action)."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim + action_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """Asymmetric squared loss: |tau - 1(diff < 0)| * diff^2."""
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return (weight * (diff**2)).mean()


@dataclass
class Args:
    seed: int = 1
    """random seed"""
    train_dataset_path: str = "data/datasets/pickcube_expert.pt"
    """path to the training .pt dataset file"""
    eval_dataset_path: str = "data/datasets/pickcube_expert_eval.pt"
    """path to the evaluation .pt dataset file (advantages computed on this)"""
    cuda: bool = True
    """if toggled, cuda will be enabled"""
    gamma: float = 0.8
    """discount factor"""
    expectile_tau: float = 0.7
    """expectile parameter for V loss (sweep: 0.5, 0.7, 0.9)"""
    tau_polyak: float = 0.005
    """Polyak averaging rate for target Q network"""
    dataset_num_envs: int = 16
    """number of parallel envs used when collecting the datasets"""
    epochs: int = 200
    """number of training epochs"""
    lr: float = 3e-4
    """learning rate"""
    batch_size: int = 256
    """minibatch size"""
    weight_decay: float = 1e-4
    """weight decay (L2 regularization)"""
    patience: int = 100
    """early stopping patience"""
    grad_clip: float = 0.5
    """max gradient norm"""
    nstep: int = 1
    """n-step TD return (1 = standard 1-step, >1 = multi-step)"""
    num_random_actions: int = 3
    """number of random actions to sample per state for Q(s, a_random) estimation"""


def compute_nstep_targets(trajectories, n, gamma):
    """Compute n-step TD targets from trajectory data.

    For each step t in each trajectory, computes:
      G_t^n = Σ_{k=0}^{m-1} γ^k r_{t+k}   where m = min(n, steps_until_done)
      bootstrap_state = s_{t+m}              (state to evaluate V on)
      discount = γ^m if not done within n steps, else 0

    Returns tensors aligned with torch.cat([t["states"] for t in trajectories]).
    """
    gp = [gamma ** k for k in range(n + 1)]  # precomputed gamma powers

    all_nstep_returns = []
    all_bootstrap_states = []
    all_nstep_discounts = []

    for traj in trajectories:
        rewards = traj["rewards"]
        states = traj["states"]
        next_states = traj["next_states"]
        dones = traj.get("dones", None)
        if dones is None:
            dones = torch.zeros_like(rewards)
            dones[-1] = 1.0
        T = len(rewards)

        # Convert to Python lists for fast scalar access
        r = rewards.tolist()
        d = dones.tolist()

        nret = [0.0] * T
        disc = [0.0] * T
        bidx = [0] * T       # bootstrap index
        btype = [False] * T  # True = states[bidx], False = next_states[bidx]

        for t in range(T):
            G = 0.0
            en = 0
            for k in range(min(n, T - t)):
                G += gp[k] * r[t + k]
                en = k + 1
                if d[t + k]:
                    break
            nret[t] = G
            last = t + en - 1
            if d[last]:
                bidx[t] = last
            elif t + en < T:
                disc[t] = gp[en]
                bidx[t] = t + en
                btype[t] = True
            else:
                disc[t] = gp[min(en, n)]
                bidx[t] = T - 1

        # Vectorized tensor construction
        bidx_t = torch.tensor(bidx, dtype=torch.long)
        btype_t = torch.tensor(btype, dtype=torch.bool).unsqueeze(1)
        boot_s = torch.where(btype_t, states[bidx_t], next_states[bidx_t])

        all_nstep_returns.append(torch.tensor(nret))
        all_bootstrap_states.append(boot_s)
        all_nstep_discounts.append(torch.tensor(disc))

    return (
        torch.cat(all_nstep_returns),
        torch.cat(all_bootstrap_states),
        torch.cat(all_nstep_discounts),
    )


def train_iql(
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    terminated: torch.Tensor,
    device: torch.device,
    args: Args,
    nstep_returns: torch.Tensor | None = None,
    bootstrap_states: torch.Tensor | None = None,
    nstep_discounts: torch.Tensor | None = None,
) -> tuple[QNetwork, Critic]:
    """Train IQL Q and V networks on flat transition data.

    When nstep_returns/bootstrap_states/nstep_discounts are provided,
    uses n-step TD targets for Q: Q(s,a) → G^n + γ^n V(s_{+n}).
    Otherwise falls back to 1-step: Q(s,a) → r + γ V(s').
    """
    use_nstep = nstep_returns is not None
    state_dim = states.shape[1]
    action_dim = actions.shape[1]

    q_net = QNetwork(state_dim, action_dim).to(device)
    q_target = copy.deepcopy(q_net)
    v_net = Critic("state", state_dim=state_dim).to(device)

    q_optimizer = torch.optim.Adam(
        q_net.parameters(), lr=args.lr, eps=1e-5, weight_decay=args.weight_decay
    )
    v_optimizer = torch.optim.Adam(
        v_net.parameters(), lr=args.lr, eps=1e-5, weight_decay=args.weight_decay
    )
    q_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        q_optimizer, T_max=args.epochs, eta_min=1e-5
    )
    v_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        v_optimizer, T_max=args.epochs, eta_min=1e-5
    )

    N = states.shape[0]
    perm = torch.randperm(N)
    val_size = max(1, int(N * 0.1))
    train_idx = perm[val_size:]
    val_idx = perm[:val_size]
    train_size = train_idx.shape[0]

    # Move val data to device
    val_s = states[val_idx].to(device)
    val_a = actions[val_idx].to(device)
    val_r = rewards[val_idx].to(device)
    val_ns = next_states[val_idx].to(device)
    val_term = terminated[val_idx].to(device)
    if use_nstep:
        val_nstep_ret = nstep_returns[val_idx].to(device)
        val_boot_s = bootstrap_states[val_idx].to(device)
        val_nstep_disc = nstep_discounts[val_idx].to(device)

    best_val_loss = float("inf")
    best_q_state = None
    best_v_state = None
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        q_net.train()
        v_net.train()
        indices = train_idx[torch.randperm(train_size)]
        epoch_q_loss = 0.0
        epoch_v_loss = 0.0
        num_batches = 0

        for start in range(0, train_size, args.batch_size):
            batch_idx = indices[start : start + args.batch_size]
            s = states[batch_idx].to(device)
            a = actions[batch_idx].to(device)
            r = rewards[batch_idx].to(device)
            ns = next_states[batch_idx].to(device)
            term = terminated[batch_idx].to(device)

            # --- Q loss: TD backup using V ---
            with torch.no_grad():
                if use_nstep:
                    b_nret = nstep_returns[batch_idx].to(device)
                    b_boot = bootstrap_states[batch_idx].to(device)
                    b_disc = nstep_discounts[batch_idx].to(device)
                    v_boot = v_net(b_boot).squeeze(-1)
                    q_target_val = b_nret + b_disc * v_boot
                else:
                    v_next = v_net(ns).squeeze(-1)
                    q_target_val = r + args.gamma * v_next * (1.0 - term)
            q_pred = q_net(s, a).squeeze(-1)
            q_loss = 0.5 * ((q_pred - q_target_val) ** 2).mean()

            q_optimizer.zero_grad()
            q_loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), args.grad_clip)
            q_optimizer.step()

            # --- V loss: expectile regression against target Q ---
            with torch.no_grad():
                q_val = q_target(s, a).squeeze(-1)
            v_pred = v_net(s).squeeze(-1)
            v_loss = expectile_loss(q_val - v_pred, args.expectile_tau)

            v_optimizer.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(v_net.parameters(), args.grad_clip)
            v_optimizer.step()

            # --- Polyak update target Q ---
            with torch.no_grad():
                for p, p_targ in zip(q_net.parameters(), q_target.parameters()):
                    p_targ.data.mul_(1.0 - args.tau_polyak).add_(
                        p.data, alpha=args.tau_polyak
                    )

            epoch_q_loss += q_loss.item()
            epoch_v_loss += v_loss.item()
            num_batches += 1

        avg_q = epoch_q_loss / num_batches
        avg_v = epoch_v_loss / num_batches
        q_scheduler.step()
        v_scheduler.step()

        # Validation
        q_net.eval()
        v_net.eval()
        with torch.no_grad():
            if use_nstep:
                v_boot_val = v_net(val_boot_s).squeeze(-1)
                q_tgt = val_nstep_ret + val_nstep_disc * v_boot_val
            else:
                v_next_val = v_net(val_ns).squeeze(-1)
                q_tgt = val_r + args.gamma * v_next_val * (1.0 - val_term)
            q_pred_val = q_net(val_s, val_a).squeeze(-1)
            val_q_loss = 0.5 * ((q_pred_val - q_tgt) ** 2).mean().item()

            q_val_for_v = q_target(val_s, val_a).squeeze(-1)
            v_pred_val = v_net(val_s).squeeze(-1)
            diff = q_val_for_v - v_pred_val
            weight = torch.where(diff > 0, args.expectile_tau, 1.0 - args.expectile_tau)
            val_v_loss = (weight * (diff**2)).mean().item()

        val_total = val_q_loss + val_v_loss
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_q_state = {k: v.clone() for k, v in q_net.state_dict().items()}
            best_v_state = {k: v.clone() for k, v in v_net.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1}/{args.epochs}: "
                f"q_loss={avg_q:.6f}, v_loss={avg_v:.6f}, "
                f"val_q={val_q_loss:.6f}, val_v={val_v_loss:.6f}"
            )

    if best_q_state is not None:
        q_net.load_state_dict(best_q_state)
    if best_v_state is not None:
        v_net.load_state_dict(best_v_state)
    q_net.eval()
    v_net.eval()

    # Summary on full dataset
    with torch.no_grad():
        all_q, all_v = [], []
        for start in range(0, N, args.batch_size):
            end = min(start + args.batch_size, N)
            s = states[start:end].to(device)
            a = actions[start:end].to(device)
            all_q.append(q_net(s, a).squeeze(-1).cpu())
            all_v.append(v_net(s).squeeze(-1).cpu())
        all_q = torch.cat(all_q)
        all_v = torch.cat(all_v)
        all_a = all_q - all_v
        print(f"  Q(s,a): mean={all_q.mean():.4f}, std={all_q.std():.4f}")
        print(f"  V(s):   mean={all_v.mean():.4f}, std={all_v.std():.4f}")
        print(f"  A(s,a): mean={all_a.mean():.4f}, std={all_a.std():.4f}")

    return q_net, v_net


if __name__ == "__main__":
    args = tyro.cli(Args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ---------------------------------------------------------------
    # 1. Load datasets and extract trajectories
    # ---------------------------------------------------------------
    print(f"Loading training dataset: {args.train_dataset_path}")
    train_dataset = OfflineRLDataset([args.train_dataset_path], False, False)
    state_dim = train_dataset.state.shape[1]
    action_dim = train_dataset.actions.shape[1]

    print(
        f"Extracting training trajectories "
        f"(num_envs={args.dataset_num_envs}, gamma={args.gamma})"
    )
    train_trajectories = train_dataset.extract_trajectories(
        num_envs=args.dataset_num_envs, gamma=args.gamma
    )
    traj_lens = [t["states"].shape[0] for t in train_trajectories]
    print(
        f"  Found {len(train_trajectories)} trajectories, "
        f"lengths: min={min(traj_lens)}, max={max(traj_lens)}, "
        f"mean={sum(traj_lens) / len(traj_lens):.1f}"
    )

    print(f"\nLoading eval dataset: {args.eval_dataset_path}")
    eval_dataset = OfflineRLDataset([args.eval_dataset_path], False, False)
    N_eval = len(eval_dataset)

    print(
        f"Extracting eval trajectories "
        f"(num_envs={args.dataset_num_envs}, gamma={args.gamma})"
    )
    eval_trajectories = eval_dataset.extract_trajectories(
        num_envs=args.dataset_num_envs, gamma=args.gamma
    )
    traj_lens = [t["states"].shape[0] for t in eval_trajectories]
    print(
        f"  Found {len(eval_trajectories)} trajectories, "
        f"lengths: min={min(traj_lens)}, max={max(traj_lens)}, "
        f"mean={sum(traj_lens) / len(traj_lens):.1f}"
    )

    # ---------------------------------------------------------------
    # 2. Prepare training data (combined train + eval, same as GAE)
    # ---------------------------------------------------------------
    all_trajectories = train_trajectories + eval_trajectories
    total_transitions = sum(t["states"].shape[0] for t in all_trajectories)
    print(
        f"\nIQL training: {len(all_trajectories)} trajectories, "
        f"{total_transitions} transitions"
    )

    all_states = torch.cat([t["states"] for t in all_trajectories], dim=0)
    all_next_states = torch.cat([t["next_states"] for t in all_trajectories], dim=0)
    all_rewards = torch.cat([t["rewards"] for t in all_trajectories], dim=0)
    all_terminated = torch.cat([t["terminated"] for t in all_trajectories], dim=0)

    # Actions come from the flat dataset — reconstruct in trajectory order
    # Train dataset actions
    train_actions_list = []
    for t in train_trajectories:
        train_actions_list.append(train_dataset.actions[t["flat_indices"]])
    # Eval dataset actions
    eval_actions_list = []
    for t in eval_trajectories:
        eval_actions_list.append(eval_dataset.actions[t["flat_indices"]])
    all_actions = torch.cat(train_actions_list + eval_actions_list, dim=0)

    print(f"  Rewards: {all_rewards.sum().item():.0f} positive out of {len(all_rewards)}")

    # ---------------------------------------------------------------
    # 3. Train IQL
    # ---------------------------------------------------------------
    nstep_kw = {}
    if args.nstep > 1:
        print(f"\nComputing {args.nstep}-step TD targets...")
        nret, boot_s, ndisc = compute_nstep_targets(
            all_trajectories, args.nstep, args.gamma
        )
        nstep_kw = dict(
            nstep_returns=nret, bootstrap_states=boot_s, nstep_discounts=ndisc
        )
        print(f"  n-step returns: mean={nret.mean():.4f}, std={nret.std():.4f}")
        frac_boot = (ndisc > 0).float().mean()
        print(f"  Bootstrapped: {frac_boot:.1%} of transitions")

    print(f"\nTraining IQL (expectile_tau={args.expectile_tau}, nstep={args.nstep})...")
    q_net, v_net = train_iql(
        all_states, all_actions, all_rewards, all_next_states, all_terminated,
        device, args, **nstep_kw,
    )

    # Free training data
    del train_dataset, train_trajectories, all_trajectories
    del all_states, all_next_states, all_actions, all_rewards, all_terminated

    # ---------------------------------------------------------------
    # 4. Compute advantages on eval dataset
    # ---------------------------------------------------------------
    print("\nComputing IQL advantages on eval dataset...")
    flat_values = torch.zeros(N_eval)
    flat_action_values = torch.zeros(N_eval)

    with torch.no_grad():
        for traj in eval_trajectories:
            flat_idx = traj["flat_indices"]
            s = traj["states"].to(device)
            a = eval_dataset.actions[flat_idx].to(device)

            q = q_net(s, a).squeeze(-1).cpu()
            v = v_net(s).squeeze(-1).cpu()

            flat_action_values[flat_idx] = q
            flat_values[flat_idx] = v

    flat_advantages = flat_action_values - flat_values

    # --- Q(s, a_random): evaluate random actions directly through Q-network ---
    print(f"Evaluating {args.num_random_actions} random actions per state...")
    random_action_values = torch.zeros(N_eval, args.num_random_actions)
    random_actions = torch.zeros(N_eval, args.num_random_actions, action_dim)

    with torch.no_grad():
        for k in range(args.num_random_actions):
            # Sample uniform random actions in [-1, 1] (standard ManiSkill bounds)
            rand_a = torch.rand(N_eval, action_dim) * 2 - 1
            random_actions[:, k, :] = rand_a

            # Batch evaluate Q(s, a_random)
            for start in range(0, N_eval, args.batch_size):
                end = min(start + args.batch_size, N_eval)
                s = eval_dataset.state[start:end].to(device)
                a = rand_a[start:end].to(device)
                random_action_values[start:end, k] = q_net(s, a).squeeze(-1).cpu()

    mean_random_action_values = random_action_values.mean(dim=1)

    dataset_av = flat_action_values
    frac_better = (dataset_av.unsqueeze(1) > random_action_values).float().mean(dim=1)
    print(
        f"  Random Q: mean={mean_random_action_values.mean():.4f}, "
        f"std={mean_random_action_values.std():.4f}"
    )
    print(
        f"  P(Q_dataset > mean Q_random): "
        f"{(dataset_av > mean_random_action_values).float().mean():.4f}"
    )
    print(
        f"  Per-state frac(Q_dataset > Q_random_k): "
        f"mean={frac_better.mean():.4f}, std={frac_better.std():.4f}"
    )

    # Save results
    results = {
        "values": flat_values,
        "action_values": flat_action_values,
        "advantages": flat_advantages,
        "random_action_values": random_action_values,
        "random_actions": random_actions,
        "mean_random_action_values": mean_random_action_values,
    }
    save_path = os.path.join(
        os.path.dirname(args.eval_dataset_path),
        f"iql_estimates_gamma{args.gamma}_tau{args.expectile_tau}.pt",
    )
    torch.save(results, save_path)
    print(f"\nSaved IQL estimates to {save_path}")
    print(
        f"  Values:        mean={flat_values.mean():.4f}, std={flat_values.std():.4f}"
    )
    print(
        f"  Action values: mean={flat_action_values.mean():.4f}, "
        f"std={flat_action_values.std():.4f}"
    )
    print(
        f"  Advantages:    mean={flat_advantages.mean():.4f}, "
        f"std={flat_advantages.std():.4f}"
    )
