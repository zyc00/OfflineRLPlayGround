"""BC with Gaussian NLL loss — learns both mean and logstd.

Produces a stochastic policy with learnable exploration noise,
potentially giving non-zero frac_decisive for downstream finetuning.

Usage:
  python bc_gaussian.py --env-id PickCube-v1 \
    --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 \
    --control-mode pd_ee_delta_pos --sim-backend cpu --total-iters 100000 --seed 1
"""
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from mani_skill.utils import gym_utils
from mani_skill.utils.io_utils import load_json
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from behavior_cloning.evaluate import evaluate
from behavior_cloning.make_env import make_eval_envs


@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = False

    env_id: str = "PickCube-v1"
    demo_path: str = ""
    num_demos: Optional[int] = None
    total_iters: int = 100_000
    batch_size: int = 1024

    lr: float = 3e-4
    normalize_states: bool = False

    max_episode_steps: Optional[int] = 100
    """Must be >= max demo length. PickCube MP demos avg 78 steps."""
    log_freq: int = 1000
    eval_freq: int = 10000
    save_freq: Optional[int] = None
    num_eval_episodes: int = 100
    num_eval_envs: int = 10
    sim_backend: str = "cpu"
    control_mode: str = "pd_ee_delta_pos"

    # Loss
    loss_type: str = "nll"
    """'nll' (Gaussian NLL, learns logstd) or 'mse' (baseline)"""
    logstd_min: float = -5.0
    logstd_max: float = 0.0
    """Clamp range for logstd. -5 ≈ std=0.007, 0 ≈ std=1.0"""

    # Eval mode
    eval_deterministic: bool = True
    """Use mean action (True) or sample from Gaussian (False) during eval"""


def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


class ManiSkillDataset:
    def __init__(self, dataset_file, load_count=-1, normalize_states=False):
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]

        self.observations = []
        self.actions = []
        if load_count is None:
            load_count = len(self.episodes)
        print(f"Loading {load_count} episodes")

        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)
            self.observations.append(trajectory["obs"][:-1])
            self.actions.append(trajectory["actions"])

        self.observations = np.vstack(self.observations)
        self.actions = np.vstack(self.actions)

        if normalize_states:
            mean, std = np.mean(self.observations), np.std(self.observations)
            self.observations = (self.observations - mean) / std


class GaussianActor(nn.Module):
    """MLP actor with learnable per-dimension logstd."""
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        # 2-hidden-layer MLP (same as MLP2 in bc_official)
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        self.logstd = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        return self.net(state)

    def get_action(self, state, deterministic=False):
        mu = self.net(state)
        if deterministic:
            return mu
        std = self.logstd.clamp(-5, 2).exp()
        return mu + std * torch.randn_like(mu)

    def nll_loss(self, state, action, logstd_min=-5.0, logstd_max=0.0):
        mu = self.net(state)
        logstd = self.logstd.clamp(logstd_min, logstd_max)
        std = logstd.exp()
        var = std ** 2
        log_prob = -0.5 * (((action - mu) ** 2) / var + 2 * logstd + math.log(2 * math.pi))
        return -log_prob.sum(dim=-1).mean()


def save_ckpt(run_name, tag, actor):
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    torch.save({"actor": actor.state_dict()}, f"runs/{run_name}/checkpoints/{tag}.pt")


if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.exp_name is None:
        run_name = f"{args.env_id}__bc_gaussian__{args.loss_type}__s{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    if args.demo_path.endswith(".h5"):
        import json
        json_file = args.demo_path[:-2] + "json"
        with open(json_file, "r") as f:
            demo_info = json.load(f)
            if "control_mode" in demo_info["env_info"]["env_kwargs"]:
                control_mode = demo_info["env_info"]["env_kwargs"]["control_mode"]
            elif "control_mode" in demo_info["episodes"][0]:
                control_mode = demo_info["episodes"][0]["control_mode"]
            else:
                raise Exception("Control mode not found in json")
            assert control_mode == args.control_mode, \
                f"Control mode mismatch: dataset={control_mode}, args={args.control_mode}"

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env_kwargs = dict(
        control_mode=args.control_mode, reward_mode="sparse",
        obs_mode="state", render_mode="rgb_array",
    )
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    envs = make_eval_envs(args.env_id, args.num_eval_envs, args.sim_backend, env_kwargs)

    writer = SummaryWriter(f"runs/{run_name}")

    ds = ManiSkillDataset(args.demo_path, load_count=args.num_demos,
                          normalize_states=args.normalize_states)

    all_obs = torch.from_numpy(ds.observations).float().to(device)
    all_actions = torch.from_numpy(ds.actions).float().to(device)
    n_samples = all_obs.shape[0]

    actor = GaussianActor(
        envs.single_observation_space.shape[0],
        envs.single_action_space.shape[0],
    ).to(device)

    # NLL trains both net and logstd; MSE only trains net
    if args.loss_type == "nll":
        optimizer = optim.Adam(actor.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(actor.net.parameters(), lr=args.lr)

    best_eval_metrics = defaultdict(float)

    perm_offset = n_samples
    for iteration in range(args.total_iters):
        if perm_offset + args.batch_size > n_samples:
            _seed = int(torch.empty((), dtype=torch.int64).random_().item())
            _g = torch.Generator()
            _g.manual_seed(_seed)
            perm = torch.randperm(n_samples, generator=_g)
            perm_offset = 0
        idx = perm[perm_offset:perm_offset + args.batch_size]
        perm_offset += args.batch_size
        obs, action = all_obs[idx], all_actions[idx]

        optimizer.zero_grad()
        if args.loss_type == "nll":
            loss = actor.nll_loss(obs, action, args.logstd_min, args.logstd_max)
        else:
            pred_action = actor(obs)
            loss = F.mse_loss(pred_action, action)
        loss.backward()
        optimizer.step()

        if iteration % args.log_freq == 0:
            logstd_val = actor.logstd.data.clamp(args.logstd_min, args.logstd_max).mean().item()
            print(f"Iter {iteration}, loss: {loss.item():.6f}, logstd: {logstd_val:.3f}")
            writer.add_scalar("losses/total_loss", loss.item(), iteration)
            writer.add_scalar("charts/logstd", logstd_val, iteration)

        if iteration % args.eval_freq == 0:
            actor.eval()
            # Eval deterministic
            def sample_fn_det(obs):
                if isinstance(obs, np.ndarray):
                    obs = torch.from_numpy(obs).float().to(device)
                return actor.get_action(obs, deterministic=True).cpu().numpy()
            # Eval stochastic
            def sample_fn_sto(obs):
                if isinstance(obs, np.ndarray):
                    obs = torch.from_numpy(obs).float().to(device)
                return actor.get_action(obs, deterministic=False).cpu().numpy()

            with torch.no_grad():
                m_det = evaluate(args.num_eval_episodes, sample_fn_det, envs)
                m_sto = evaluate(args.num_eval_episodes, sample_fn_sto, envs)
            actor.train()

            sr_det = np.mean(m_det['success_once'])
            sr_sto = np.mean(m_sto['success_once'])
            print(f"  Eval: det_SR={sr_det:.1%}, sto_SR={sr_sto:.1%}")
            writer.add_scalar("eval/success_once_det", sr_det, iteration)
            writer.add_scalar("eval/success_once_sto", sr_sto, iteration)

            if sr_det > best_eval_metrics['success_once_det']:
                best_eval_metrics['success_once_det'] = sr_det
                save_ckpt(run_name, "best_det", actor)
            if sr_sto > best_eval_metrics['success_once_sto']:
                best_eval_metrics['success_once_sto'] = sr_sto
                save_ckpt(run_name, "best_sto", actor)

        if args.save_freq is not None and iteration % args.save_freq == 0:
            save_ckpt(run_name, str(iteration), actor)

    save_ckpt(run_name, "final", actor)

    # Final eval
    actor.eval()
    def sample_fn_det(obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(device)
        return actor.get_action(obs, deterministic=True).cpu().numpy()
    def sample_fn_sto(obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(device)
        return actor.get_action(obs, deterministic=False).cpu().numpy()
    with torch.no_grad():
        m_det = evaluate(args.num_eval_episodes, sample_fn_det, envs)
        m_sto = evaluate(args.num_eval_episodes, sample_fn_sto, envs)
    print(f"\nFinal: det_SR={np.mean(m_det['success_once']):.1%}, sto_SR={np.mean(m_sto['success_once']):.1%}")
    print(f"Final logstd: {actor.logstd.data.clamp(args.logstd_min, args.logstd_max).tolist()}")

    envs.close()
