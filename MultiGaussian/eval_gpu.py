"""GPU evaluation for Gaussian policy (fast, may underestimate for deterministic policies)."""
import torch
import numpy as np
import gymnasium as gym
import mani_skill.envs
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from MultiGaussian.models.gaussian import GaussianPolicy


@torch.no_grad()
def evaluate_gaussian_gpu(
    model, device, n_episodes, env_id, control_mode, max_episode_steps,
    num_envs, obs_min, obs_max, action_min, action_max,
    no_obs_norm, no_action_norm, zero_qvel,
):
    """Evaluate Gaussian policy using GPU-parallel envs (ManiSkillVectorEnv)."""
    model.eval()

    envs = gym.make(
        env_id,
        num_envs=num_envs,
        obs_mode="state",
        render_mode="rgb_array",
        control_mode=control_mode,
        max_episode_steps=max_episode_steps,
        sim_backend="physx_cuda",
        reconfiguration_freq=1,
    )
    envs = ManiSkillVectorEnv(envs, num_envs, ignore_terminations=True, record_metrics=True)

    if not no_obs_norm:
        o_lo = obs_min.to(device)
        o_hi = obs_max.to(device)
    if not no_action_norm:
        a_lo = action_min.to(device)
        a_hi = action_max.to(device)

    success_at_end_list = []
    success_once_list = []
    eps_done = 0

    while eps_done < n_episodes:
        obs, _ = envs.reset()
        obs = obs.float().to(device)

        for step in range(max_episode_steps):
            if zero_qvel:
                obs[..., 9:18] = 0.0

            if no_obs_norm:
                obs_norm = obs
            else:
                obs_norm = (obs - o_lo) / (o_hi - o_lo + 1e-8) * 2.0 - 1.0

            mean, _std = model(obs_norm)

            if no_action_norm:
                action = mean
            else:
                action = (mean + 1.0) / 2.0 * (a_hi - a_lo) + a_lo

            obs_new, reward, terminated, truncated, info = envs.step(action)
            obs = obs_new.float().to(device)

            if truncated.any():
                fi = info.get("final_info", {})
                ep = fi.get("episode", {})
                if "success_at_end" in ep:
                    mask = info.get("_final_info", truncated)
                    sa = ep["success_at_end"][mask].float().cpu().numpy()
                    so = ep["success_once"][mask].float().cpu().numpy()
                    success_at_end_list.append(sa)
                    success_once_list.append(so)
                eps_done += num_envs
                break

    envs.close()

    if success_at_end_list:
        sa_all = np.concatenate(success_at_end_list)[:n_episodes]
        so_all = np.concatenate(success_once_list)[:n_episodes]
    else:
        sa_all = np.array([0.0])
        so_all = np.array([0.0])

    return {
        "success_at_end": sa_all.mean(),
        "success_once": so_all.mean(),
        "n_episodes": len(sa_all),
    }


def _load_model_from_ckpt(ckpt_path, device):
    """Load GaussianPolicy from checkpoint, return (model, ckpt_dict)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]

    model = GaussianPolicy(
        input_dim=ckpt["obs_dim"],
        action_dim=ckpt["action_dim"],
        hidden_dims=args["hidden_dims"],
        activation=args["activation"],
        sigma_init=args.get("sigma_init", -1.5),
    ).to(device)

    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def evaluate_gpu_ckpt(ckpt_path, n_episodes=100, env_id=None,
                      control_mode=None, max_episode_steps=None):
    """Load checkpoint and evaluate on GPU."""
    device = torch.device("cuda")
    model, ckpt = _load_model_from_ckpt(ckpt_path, device)
    args = ckpt["args"]

    env_id = env_id or args.get("env_id", "PickCube-v1")
    control_mode = control_mode or args.get("control_mode", "pd_ee_delta_pos")
    max_episode_steps = max_episode_steps or args.get("max_episode_steps", 100)
    no_obs_norm = ckpt.get("no_obs_norm", False)
    no_action_norm = ckpt.get("no_action_norm", False)
    zero_qvel = args.get("zero_qvel", False)
    num_envs = min(n_episodes, 200)

    if zero_qvel:
        print(f"  zero_qvel=True (from checkpoint)")

    metrics = evaluate_gaussian_gpu(
        model=model,
        device=device,
        n_episodes=n_episodes,
        env_id=env_id,
        control_mode=control_mode,
        max_episode_steps=max_episode_steps,
        num_envs=num_envs,
        obs_min=ckpt.get("obs_min"),
        obs_max=ckpt.get("obs_max"),
        action_min=ckpt.get("action_min"),
        action_max=ckpt.get("action_max"),
        no_obs_norm=no_obs_norm,
        no_action_norm=no_action_norm,
        zero_qvel=zero_qvel,
    )
    print(f"\nGPU Eval ({metrics['n_episodes']} eps): success_once={metrics['success_once']:.3f}, "
          f"success_at_end={metrics['success_at_end']:.3f}")
    return metrics


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("ckpt_path")
    p.add_argument("--n_episodes", type=int, default=100)
    p.add_argument("--env_id", default=None)
    p.add_argument("--control_mode", default=None)
    p.add_argument("--max_episode_steps", type=int, default=None)
    args = p.parse_args()
    evaluate_gpu_ckpt(args.ckpt_path, args.n_episodes, args.env_id,
                      args.control_mode, args.max_episode_steps)
