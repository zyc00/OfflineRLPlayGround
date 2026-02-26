"""CPU evaluation for diffusion policy (accurate, matches dp_train baseline)."""
import torch
import numpy as np
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack
from tqdm import tqdm


@torch.no_grad()
def evaluate_cpu_model(
    n_episodes,
    model,
    device,
    act_steps,
    cond_steps=1,
    env_id="PickCube-v1",
    num_envs=10,
    control_mode="pd_ee_delta_pos",
    max_episode_steps=100,
    obs_min=None,
    obs_max=None,
    action_min=None,
    action_max=None,
    no_obs_norm=False,
    no_action_norm=False,
    act_offset=0,
):
    """Evaluate diffusion policy using CPU envs with FrameStack (matches dp_train)."""
    model.eval()

    def make_env(seed):
        def thunk():
            env = gym.make(env_id, obs_mode="state", control_mode=control_mode,
                          render_mode="rgb_array", max_episode_steps=max_episode_steps,
                          reconfiguration_freq=1)
            env = FrameStack(env, num_stack=cond_steps)
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

        for step in range(max_episode_steps):
            if no_obs_norm:
                cond = {"state": obs}
            else:
                cond = {"state": (obs - o_lo) / (o_hi - o_lo + 1e-8) * 2.0 - 1.0}

            samples = model(cond, deterministic=True)
            action_chunk = samples.trajectories

            if not no_action_norm:
                action_chunk = (action_chunk + 1.0) / 2.0 * (a_hi - a_lo) + a_lo

            action_np = action_chunk[:, act_offset:act_offset+act_steps].cpu().numpy()
            for a_idx in range(action_np.shape[1]):
                obs_np, rew, terminated, truncated, info = envs.step(action_np[:, a_idx])
                if truncated.any():
                    break

            obs = torch.from_numpy(obs_np).float().to(device)

            if truncated.any():
                for fi in info.get("final_info", []):
                    if fi and "episode" in fi:
                        success_once_list.append(fi["episode"]["success_once"])
                        success_at_end_list.append(fi["episode"]["success_at_end"])
                eps_done += num_envs
                break

    envs.close()

    so = np.mean(success_once_list[:n_episodes])
    sa = np.mean(success_at_end_list[:n_episodes])
    return {
        "success_at_end": sa,
        "success_once": so,
        "n_episodes": min(len(success_once_list), n_episodes),
    }


def evaluate_cpu_ckpt(ckpt_path, n_episodes=100, env_id="PickCube-v1",
                      control_mode="pd_ee_delta_pos", max_episode_steps=100):
    """Load checkpoint and evaluate on CPU."""
    from DPPO.model.unet_wrapper import DiffusionUNet
    from DPPO.model.diffusion import DiffusionModel

    device = torch.device("cuda")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]

    obs_dim = ckpt["obs_dim"]
    action_dim = ckpt["action_dim"]
    cond_steps = args["cond_steps"]
    horizon_steps = args["horizon_steps"]
    act_steps = args["act_steps"]
    denoising_steps = args["denoising_steps"]

    if args.get("network_type") == "unet":
        network = DiffusionUNet(
            action_dim=action_dim,
            horizon_steps=horizon_steps,
            cond_dim=obs_dim * cond_steps,
            diffusion_step_embed_dim=args.get("diffusion_step_embed_dim", 64),
            down_dims=args.get("unet_dims", [64, 128, 256]),
            n_groups=args.get("n_groups", 8),
        )
    else:
        from DPPO.model.mlp import DiffusionMLP
        network = DiffusionMLP(
            action_dim=action_dim,
            horizon_steps=horizon_steps,
            cond_dim=obs_dim * cond_steps,
        )

    model = DiffusionModel(
        network=network,
        horizon_steps=horizon_steps,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        denoising_steps=denoising_steps,
        denoised_clip_value=1.0,
        randn_clip_value=10,
        final_action_clip_value=1.0,
        predict_epsilon=True,
    )
    model.load_state_dict(ckpt["ema"])
    model.eval()

    no_obs_norm = ckpt.get("no_obs_norm", False)
    no_action_norm = ckpt.get("no_action_norm", False)
    act_offset = cond_steps - 1 if args.get("network_type") == "unet" else 0

    metrics = evaluate_cpu_model(
        n_episodes=n_episodes,
        model=model,
        device=device,
        act_steps=act_steps,
        cond_steps=cond_steps,
        env_id=env_id,
        control_mode=control_mode,
        max_episode_steps=max_episode_steps,
        obs_min=ckpt.get("obs_min"),
        obs_max=ckpt.get("obs_max"),
        action_min=ckpt.get("action_min"),
        action_max=ckpt.get("action_max"),
        no_obs_norm=no_obs_norm,
        no_action_norm=no_action_norm,
        act_offset=act_offset,
    )
    print(f"\nCPU Eval ({n_episodes} eps): success_once={metrics['success_once']:.3f}, "
          f"success_at_end={metrics['success_at_end']:.3f}")
    return metrics


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("ckpt_path")
    p.add_argument("--n_episodes", type=int, default=100)
    p.add_argument("--env_id", default="PickCube-v1")
    p.add_argument("--control_mode", default="pd_ee_delta_pos")
    p.add_argument("--max_episode_steps", type=int, default=100)
    args = p.parse_args()
    evaluate_cpu_ckpt(args.ckpt_path, args.n_episodes, args.env_id,
                      args.control_mode, args.max_episode_steps)
