"""CPU evaluation for MIP policy."""
import torch
import numpy as np
import gymnasium as gym
import mani_skill.envs
from collections import deque
from mani_skill.utils.wrappers import CPUGymWrapper

from MultiGaussian.models.multi_gaussian import MIPPolicy


@torch.no_grad()
def evaluate_mip_cpu(
    model, device, n_episodes, env_id, control_mode, max_episode_steps,
    num_envs, obs_min, obs_max, action_min, action_max,
    no_obs_norm, no_action_norm, zero_qvel,
    cond_steps=1, horizon_steps=1, act_steps=1,
):
    """Evaluate MIP policy using CPU envs (deterministic: two-pass inference)."""
    model.eval()

    def make_env(seed):
        def thunk():
            env = gym.make(env_id, obs_mode="state", control_mode=control_mode,
                           render_mode="rgb_array", max_episode_steps=max_episode_steps,
                           reconfiguration_freq=1)
            env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
            return env
        return thunk

    envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(num_envs)])

    if not no_obs_norm:
        o_lo = obs_min.to(device)
        o_hi = obs_max.to(device)
    if not no_action_norm:
        a_lo = action_min.to(device)
        a_hi = action_max.to(device)

    eps_done = 0
    success_once_list = []
    success_at_end_list = []

    while eps_done < n_episodes:
        obs, _ = envs.reset()
        obs = torch.from_numpy(obs).float().to(device)

        obs_buffer = deque(maxlen=cond_steps)
        for _ in range(cond_steps):
            obs_buffer.append(obs.clone())

        step = 0
        done = False
        while step < max_episode_steps and not done:
            if cond_steps == 1:
                cond_obs = obs_buffer[-1]
            else:
                cond_obs = torch.stack(list(obs_buffer), dim=1)

            if zero_qvel:
                cond_obs[..., 9:18] = 0.0

            if no_obs_norm:
                obs_norm = cond_obs
            else:
                obs_norm = (cond_obs - o_lo) / (o_hi - o_lo + 1e-8) * 2.0 - 1.0

            # MIP two-pass prediction
            actions_chunk = model.predict(obs_norm)

            # Denormalize
            if no_action_norm:
                pass
            else:
                actions_chunk = (actions_chunk + 1.0) / 2.0 * (a_hi - a_lo) + a_lo

            n_exec = min(act_steps, max_episode_steps - step) if horizon_steps > 1 else 1
            for t in range(n_exec):
                if horizon_steps > 1:
                    action = actions_chunk[:, t]
                else:
                    action = actions_chunk

                obs_np, rew, terminated, truncated, info = envs.step(action.cpu().numpy())
                obs = torch.from_numpy(obs_np).float().to(device)
                obs_buffer.append(obs.clone())
                step += 1

                if truncated.any():
                    for fi in info.get("final_info", []):
                        if fi and "episode" in fi:
                            success_once_list.append(fi["episode"]["success_once"])
                            success_at_end_list.append(fi["episode"]["success_at_end"])
                    eps_done += num_envs
                    done = True
                    break

    envs.close()

    so = np.mean(success_once_list[:n_episodes])
    sa = np.mean(success_at_end_list[:n_episodes])
    return {
        "success_at_end": sa,
        "success_once": so,
        "n_episodes": min(len(success_once_list), n_episodes),
    }


def _load_model_from_ckpt(ckpt_path, device):
    """Load MIPPolicy from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]

    model = MIPPolicy(
        input_dim=ckpt["obs_dim"],
        action_dim=ckpt["action_dim"],
        cond_steps=args.get("cond_steps", 1),
        horizon_steps=args.get("horizon_steps", 1),
        t_star=args.get("t_star", 0.9),
        dropout=args.get("dropout", 0.1),
        emb_dim=args.get("emb_dim", 512),
        n_layers=args.get("n_layers", 6),
        predict_epsilon=args.get("predict_epsilon", False),
    ).to(device)

    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def evaluate_cpu_ckpt(ckpt_path, n_episodes=100, env_id=None,
                      control_mode=None, max_episode_steps=None):
    """Load MIP checkpoint and evaluate on CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = _load_model_from_ckpt(ckpt_path, device)
    args = ckpt["args"]

    env_id = env_id or args.get("env_id", "PickCube-v1")
    control_mode = control_mode or args.get("control_mode", "pd_ee_delta_pos")
    max_episode_steps = max_episode_steps or args.get("max_episode_steps", 100)
    no_obs_norm = ckpt.get("no_obs_norm", False)
    no_action_norm = ckpt.get("no_action_norm", False)
    zero_qvel = args.get("zero_qvel", False)
    cond_steps = args.get("cond_steps", 1)
    horizon_steps = args.get("horizon_steps", 1)
    act_steps = args.get("act_steps", 1)

    if zero_qvel:
        print(f"  zero_qvel=True (from checkpoint)")
    if horizon_steps > 1:
        print(f"  Action chunking: cond={cond_steps}, horizon={horizon_steps}, act={act_steps}")

    metrics = evaluate_mip_cpu(
        model=model, device=device, n_episodes=n_episodes,
        env_id=env_id, control_mode=control_mode,
        max_episode_steps=max_episode_steps, num_envs=10,
        obs_min=ckpt.get("obs_min"), obs_max=ckpt.get("obs_max"),
        action_min=ckpt.get("action_min"), action_max=ckpt.get("action_max"),
        no_obs_norm=no_obs_norm, no_action_norm=no_action_norm,
        zero_qvel=zero_qvel, cond_steps=cond_steps,
        horizon_steps=horizon_steps, act_steps=act_steps,
    )
    print(f"\nCPU Eval ({n_episodes} eps): success_once={metrics['success_once']:.3f}, "
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
    evaluate_cpu_ckpt(args.ckpt_path, args.n_episodes, args.env_id,
                      args.control_mode, args.max_episode_steps)
