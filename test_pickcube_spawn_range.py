"""Test PickCube pretrained policy with different cube spawn ranges.
Uses eval_cpu.py's evaluation logic with monkey-patched spawn range.
"""
import torch
import numpy as np
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack

from DPPO.eval_cpu import evaluate_cpu_ckpt
from DPPO.model.unet_wrapper import DiffusionUNet
from DPPO.model.diffusion import DiffusionModel


def eval_with_spawn_range(model, device, act_steps, cond_steps, act_offset,
                          control_mode, spawn_size, n_episodes=100,
                          max_episode_steps=100, no_obs_norm=True, no_action_norm=True,
                          num_envs=10):
    """Evaluate using eval_cpu.py style (FrameStack + CPUGymWrapper)."""
    model.eval()

    def make_env(seed):
        def thunk():
            env = gym.make("PickCube-v1", obs_mode="state", control_mode=control_mode,
                          render_mode="rgb_array", max_episode_steps=max_episode_steps,
                          reconfiguration_freq=1)
            # Monkey-patch spawn range
            env.unwrapped.cube_spawn_half_size = spawn_size
            env = FrameStack(env, num_stack=cond_steps)
            env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
            return env
        return thunk

    envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(num_envs)])

    eps_done = 0
    success_once_list = []

    while eps_done < n_episodes:
        obs, _ = envs.reset()
        obs = torch.from_numpy(obs).float().to(device)

        for step in range(max_episode_steps):
            cond = {"state": obs}
            with torch.no_grad():
                samples = model(cond, deterministic=True)
            action_chunk = samples.trajectories
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
                eps_done += num_envs
                break

    envs.close()
    return np.mean(success_once_list[:n_episodes])


def main():
    device = torch.device("cuda")
    # Use pd_ee_delta_pos checkpoint (100% SR at default range)
    ckpt_path = "runs/dppo_pretrain/dppo_unet_ee_emafix2/best.pt"

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ca = ckpt["args"]
    obs_dim, action_dim = ckpt["obs_dim"], ckpt["action_dim"]
    cond_steps = ca["cond_steps"]
    horizon_steps = ca["horizon_steps"]
    act_steps = ca["act_steps"]
    T = ca["denoising_steps"]
    act_offset = cond_steps - 1  # unet
    control_mode = ca.get("control_mode", "pd_ee_delta_pos")

    network = DiffusionUNet(
        action_dim=action_dim, horizon_steps=horizon_steps,
        cond_dim=obs_dim * cond_steps,
        diffusion_step_embed_dim=ca.get("diffusion_step_embed_dim", 64),
        down_dims=ca.get("unet_dims", [64, 128, 256]),
        n_groups=ca.get("n_groups", 8),
    )
    model = DiffusionModel(
        network=network, horizon_steps=horizon_steps,
        obs_dim=obs_dim, action_dim=action_dim, device=device,
        denoising_steps=T, denoised_clip_value=1.0,
        randn_clip_value=10, final_action_clip_value=1.0, predict_epsilon=True,
    )
    model.load_state_dict(ckpt["ema"], strict=False)
    model.eval()
    print(f"Checkpoint: {ckpt_path}")
    print(f"Model: unet, T={T}, H={horizon_steps}, cond={cond_steps}, act={act_steps}")
    print(f"Control mode: {control_mode}, obs_dim={obs_dim}, action_dim={action_dim}")

    spawn_ranges = [0.1, 0.15, 0.2, 0.25, 0.3]
    n_eval = 100

    print(f"\nCPU eval, {n_eval} episodes per range, deterministic, full DDPM")
    print(f"{'spawn_half_size':>16s} | {'SR':>8s}")
    print("-" * 30)

    for spawn_size in spawn_ranges:
        sr = eval_with_spawn_range(
            model, device, act_steps, cond_steps, act_offset,
            control_mode, spawn_size, n_episodes=n_eval,
            max_episode_steps=100, num_envs=10,
        )
        print(f"{spawn_size:>16.2f} | {sr:>7.1%}")


if __name__ == "__main__":
    main()
