"""Test evaluation variance: evaluate the same checkpoint N times.

Measures how much success rate varies due to:
1. Stochastic action sampling (deterministic=False)
2. Environment stochasticity (reset randomness)

Usage:
  python -m RL.eval_variance_test --checkpoint runs/mc16_ckpt101_offline__seed1__1771437172/ckpt_95.pt --num_trials 20
  python -m RL.eval_variance_test --checkpoint runs/mc16_ckpt101_offline__seed1__1771437172/ckpt_95.pt --num_trials 20 --deterministic
"""

import argparse
import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from collections import defaultdict

from data.data_collection.ppo import Agent


def evaluate_once(agent, eval_envs, max_steps, deterministic, seed=None):
    if seed is not None:
        eval_obs, _ = eval_envs.reset(seed=seed)
    else:
        eval_obs, _ = eval_envs.reset()
    eval_metrics = defaultdict(list)
    for _ in range(max_steps):
        with torch.no_grad():
            eval_obs, _, eval_term, eval_trunc, eval_infos = eval_envs.step(
                agent.get_action(eval_obs, deterministic=deterministic)
            )
            if "final_info" in eval_infos:
                mask = eval_infos["_final_info"]
                for k, v in eval_infos["final_info"]["episode"].items():
                    eval_metrics[k].append(v[mask])

    sr_vals = eval_metrics.get("success_once", [])
    if sr_vals:
        return torch.cat(sr_vals).float().mean().item()
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_eval_envs", type=int, default=128)
    parser.add_argument("--max_episode_steps", type=int, default=50)
    parser.add_argument("--num_trials", type=int, default=20)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--fix_seed", action="store_true", help="Fix env reset seed across trials")
    parser.add_argument("--env_id", type=str, default="PickCube-v1")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="physx_cuda" if device.type == "cuda" else "cpu",
        reward_mode="sparse",
        control_mode="pd_joint_delta_pos",
        max_episode_steps=args.max_episode_steps,
    )

    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, **env_kwargs)
    if isinstance(eval_envs.action_space, gym.spaces.Dict):
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    eval_envs = ManiSkillVectorEnv(
        eval_envs, args.num_eval_envs,
        ignore_terminations=False,
        record_metrics=True,
    )

    agent = Agent(eval_envs).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()

    mode = "deterministic" if args.deterministic else "stochastic"
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Eval mode: {mode}, {args.num_eval_envs} envs, {args.num_trials} trials")
    print("-" * 50)

    seed_mode = "fixed seed=42" if args.fix_seed else "random seed"
    print(f"Env seed: {seed_mode}")
    print("-" * 50)

    results = []
    for i in range(args.num_trials):
        seed = 42 if args.fix_seed else None
        sr = evaluate_once(agent, eval_envs, args.max_episode_steps, args.deterministic, seed=seed)
        results.append(sr)
        print(f"  Trial {i+1:2d}: SR = {sr:.1%}")

    results = np.array(results)
    print("-" * 50)
    print(f"Mean:   {results.mean():.2%}")
    print(f"Std:    {results.std():.2%}")
    print(f"Min:    {results.min():.2%}")
    print(f"Max:    {results.max():.2%}")
    print(f"Range:  {results.max() - results.min():.2%}")

    eval_envs.close()


if __name__ == "__main__":
    main()
