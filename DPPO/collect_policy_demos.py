"""Collect policy-generated demos from a pretrained diffusion policy.

Rolls out the policy and saves successful trajectories as H5.

Two modes:
  1. Random initial states (default): reset() randomizes, collect until num_demos reached.
  2. MP initial states (--demo_path): load env_states from MP H5, rollout from each.

Usage:
    # Random initial states
    python -m DPPO.collect_policy_demos \
        --ckpt runs/dppo_pretrain/dppo_pretrain_PegInsertionSide-v1_T100_H16_50k/best.pt \
        --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos \
        --num_demos 1000 --num_envs 100 --max_episode_steps 200 --ddim_steps 10

    # From MP demo initial states (deterministic rollout)
    python -m DPPO.collect_policy_demos \
        --ckpt runs/dppo_finetune/dppo_ft_peg_conservative/best.pt \
        --demo_path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5 \
        --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos \
        --max_episode_steps 200 --deterministic
"""
import os
import argparse
import numpy as np
import torch
import h5py
import time
import gymnasium as gym
import mani_skill.envs
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from DPPO.model.unet_wrapper import DiffusionUNet
from DPPO.model.mlp import DiffusionMLP
from DPPO.model.diffusion import DiffusionModel


def load_initial_states_from_h5(demo_path):
    """Load initial env_states (timestep 0) from an MP demo H5 file.

    Returns list of dicts, each with structure:
        {"actors": {"name": np.array(dim,)}, "articulations": {"name": np.array(dim,)}}
    """
    initial_states = []
    with h5py.File(demo_path, "r") as f:
        traj_keys = sorted(
            [k for k in f.keys() if k.startswith("traj_")],
            key=lambda x: int(x.split("_")[1]),
        )
        for tk in traj_keys:
            es = f[tk]["env_states"]
            state = {"actors": {}, "articulations": {}}
            for name in es["actors"]:
                state["actors"][name] = np.array(es["actors"][name][0], dtype=np.float32)
            for name in es["articulations"]:
                state["articulations"][name] = np.array(es["articulations"][name][0], dtype=np.float32)
            initial_states.append(state)
    return initial_states


def build_batch_state_dict(states, num_envs, device, use_gpu):
    """Stack per-env initial states into a batched state dict for set_state_dict.

    Pads to num_envs by repeating the first state if len(states) < num_envs.
    """
    batch_size = len(states)
    state_dict = {"actors": {}, "articulations": {}}
    for key in ["actors", "articulations"]:
        for name in states[0][key]:
            arrs = [states[j][key][name] for j in range(batch_size)]
            if batch_size < num_envs:
                arrs.extend([arrs[0]] * (num_envs - batch_size))
            stacked = np.stack(arrs, axis=0)  # (num_envs, dim)
            t = torch.from_numpy(stacked).float()
            state_dict[key][name] = t.to(device) if use_gpu else t
    return state_dict


def get_obs_tensor(env, device):
    """Get obs from base env after set_state_dict, returned as (num_envs, obs_dim) on device."""
    obs_raw = env.unwrapped.get_obs()
    if isinstance(obs_raw, dict):
        obs_raw = obs_raw.get("state", obs_raw.get("obs"))
    if isinstance(obs_raw, np.ndarray):
        return torch.from_numpy(obs_raw).float().to(device)
    return obs_raw.float().to(device)


def collect_from_demo_states(args, model, device, cond_steps, act_steps, act_offset):
    """Collect demos by rolling out from MP demo initial states (batch GPU mode)."""
    initial_states = load_initial_states_from_h5(args.demo_path)
    n_total = len(initial_states)
    print(f"Loaded {n_total} initial states from {args.demo_path}")

    use_gpu = args.sim_backend == "gpu"
    num_envs = min(args.num_envs, n_total)

    env = gym.make(
        args.env_id, num_envs=num_envs, obs_mode="state",
        control_mode=args.control_mode, max_episode_steps=args.max_episode_steps,
        sim_backend="gpu" if use_gpu else "cpu", reward_mode="sparse",
    )
    env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=True, record_metrics=True)
    print(f"Env: {num_envs} envs, max_steps={args.max_episode_steps}, backend={args.sim_backend}")

    successful_trajs = []
    n_success = 0
    n_fail = 0
    t_start = time.time()
    max_blocks = args.max_episode_steps // act_steps + 1

    for batch_start in range(0, n_total, num_envs):
        batch_end = min(batch_start + num_envs, n_total)
        batch_size = batch_end - batch_start

        # Reset env (triggers scene reconfiguration)
        env.reset()

        # Set initial states from H5
        state_dict = build_batch_state_dict(
            initial_states[batch_start:batch_end], num_envs, device, use_gpu,
        )
        env.unwrapped.set_state_dict(state_dict)
        obs = get_obs_tensor(env, device)  # (num_envs, obs_dim)
        obs_history = obs.unsqueeze(1).repeat(1, cond_steps, 1)

        # Per-env trajectory buffers (only track batch_size active envs)
        obs_bufs = [[] for _ in range(batch_size)]
        act_bufs = [[] for _ in range(batch_size)]
        done_flags = [False] * batch_size
        success_flags = [False] * batch_size

        for i in range(batch_size):
            obs_bufs[i].append(obs[i].cpu().numpy())

        for block in range(max_blocks):
            if all(done_flags):
                break

            cond = {"state": obs_history}
            with torch.no_grad():
                samples = model(
                    cond, deterministic=args.deterministic,
                    min_sampling_denoising_std=args.min_sampling_std,
                    ddim_steps=args.ddim_steps,
                )
            action_chunk = samples.trajectories  # (num_envs, horizon, act_dim)

            for a_idx in range(act_steps):
                act_i = act_offset + a_idx
                action = action_chunk[:, min(act_i, action_chunk.shape[1] - 1)]

                if use_gpu:
                    obs_new, reward, terminated, truncated, info = env.step(action)
                    obs_new = obs_new.float()
                else:
                    obs_new, reward, terminated, truncated, info = env.step(action.cpu().numpy())
                    obs_new = torch.from_numpy(obs_new).float().to(device)
                    reward = torch.from_numpy(np.asarray(reward)).float().to(device)
                    terminated = torch.from_numpy(np.asarray(terminated)).bool().to(device)
                    truncated = torch.from_numpy(np.asarray(truncated)).bool().to(device)

                # Record per-env
                for i in range(batch_size):
                    if not done_flags[i]:
                        act_bufs[i].append(action[i].cpu().numpy())
                        obs_bufs[i].append(obs_new[i].cpu().numpy())

                        if reward[i].item() > 0.5:
                            success_flags[i] = True
                            done_flags[i] = True
                        elif terminated[i].item() or truncated[i].item():
                            done_flags[i] = True

                # Update obs history (shift left, append new obs)
                obs_history = torch.cat(
                    [obs_history[:, 1:], obs_new.unsqueeze(1)], dim=1,
                )

        # Collect successful trajectories from this batch
        for i in range(batch_size):
            if success_flags[i]:
                n_acts = len(act_bufs[i])
                ep_obs = np.array(obs_bufs[i][:n_acts])
                ep_act = np.array(act_bufs[i])
                successful_trajs.append((ep_obs, ep_act))
                n_success += 1
            else:
                n_fail += 1

        elapsed = time.time() - t_start
        print(
            f"  batch {batch_start // num_envs + 1}: processed {batch_end}/{n_total}, "
            f"success={n_success}, fail={n_fail}, SR={n_success / batch_end:.3f}, "
            f"time={elapsed:.1f}s",
            flush=True,
        )

    env.close()
    return successful_trajs, n_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--demo_path", default=None,
                        help="Path to MP demo H5 file. When set, rolls out from each "
                             "demo's initial env_state instead of random resets.")
    parser.add_argument("--num_demos", type=int, default=1000)
    parser.add_argument("--num_envs", type=int, default=100)
    parser.add_argument("--env_id", default="PegInsertionSide-v1")
    parser.add_argument("--control_mode", default="pd_joint_delta_pos")
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--ddim_steps", type=int, default=None,
                        help="DDIM steps. None=DDPM full denoising (default)")
    parser.add_argument("--sim_backend", default="gpu")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic rollout (no exploration noise)")
    parser.add_argument("--min_sampling_std", type=float, default=0.01)
    parser.add_argument("--num_initial_states", type=int, default=1000,
                        help="For --uniform: total initial states to try (each once)")
    parser.add_argument("--uniform", action="store_true",
                        help="Uniform coverage: try each initial state until success")
    parser.add_argument("--max_retries", type=int, default=50,
                        help="For --uniform: max rollout attempts per initial state")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device("cuda")
    use_gpu = args.sim_backend == "gpu"

    # Load model
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ckpt_args = ckpt["args"]
    obs_dim, action_dim = ckpt["obs_dim"], ckpt["action_dim"]
    cond_steps = ckpt_args["cond_steps"]
    horizon_steps = ckpt_args["horizon_steps"]
    act_steps = ckpt_args["act_steps"]
    T = ckpt_args["denoising_steps"]
    network_type = ckpt_args.get("network_type", "unet")
    act_offset = cond_steps - 1 if network_type == "unet" else 0

    if network_type == "unet":
        network = DiffusionUNet(
            action_dim=action_dim, horizon_steps=horizon_steps,
            cond_dim=obs_dim * cond_steps,
            diffusion_step_embed_dim=ckpt_args.get("diffusion_step_embed_dim", 64),
            down_dims=ckpt_args.get("unet_dims", [64, 128, 256]),
            n_groups=ckpt_args.get("n_groups", 8),
        )
    else:
        network = DiffusionMLP(
            action_dim=action_dim, horizon_steps=horizon_steps,
            cond_dim=obs_dim * cond_steps,
            time_dim=ckpt_args.get("time_dim", 16),
            mlp_dims=ckpt_args.get("mlp_dims", [512, 512, 512]),
            activation_type=ckpt_args.get("activation_type", "Mish"),
            residual_style=ckpt_args.get("residual_style", True),
        )

    model = DiffusionModel(
        network=network, horizon_steps=horizon_steps,
        obs_dim=obs_dim, action_dim=action_dim, device=device,
        denoising_steps=T, denoised_clip_value=1.0,
        randn_clip_value=3, final_action_clip_value=1.0, predict_epsilon=True,
    )

    # Load weights: support both pretrain (EMA) and finetuned (actor_ft remap) formats
    is_finetuned = "model" in ckpt and "ema" not in ckpt
    if is_finetuned:
        sd = ckpt["model"]
        remapped = {}
        for k, v in sd.items():
            if k.startswith("actor_ft."):
                remapped[k.replace("actor_ft.", "network.", 1)] = v
        model.load_state_dict(remapped, strict=False)
        print(f"Loaded FINETUNED checkpoint ({len(remapped)} actor_ft keys remapped)")
    else:
        model.load_state_dict(ckpt["ema"], strict=False)
        print("Loaded pretrain EMA checkpoint")
    model.eval()
    print(f"Model: {network_type}, T={T}, H={horizon_steps}, cond={cond_steps}, act={act_steps}")
    print(f"  DDIM: {args.ddim_steps} steps, deterministic={args.deterministic}, "
          f"min_sampling_std={args.min_sampling_std}")

    # ── Mode: uniform coverage (retry each state until success) ──
    if args.uniform:
        successful_trajs, n_total_episodes = collect_uniform(
            args, model, device, cond_steps, act_steps, act_offset,
            obs_dim=obs_dim,
        )
    # ── Mode: demo initial states ──
    elif args.demo_path:
        args.demo_path = os.path.expanduser(args.demo_path)
        successful_trajs, n_total_episodes = collect_from_demo_states(
            args, model, device, cond_steps, act_steps, act_offset,
        )
    # ── Mode: random initial states (original) ──
    else:
        successful_trajs, n_total_episodes = collect_random(
            args, model, device, cond_steps, act_steps, act_offset, use_gpu,
            obs_dim=obs_dim,
        )

    # Stats
    lengths = [t[1].shape[0] for t in successful_trajs]
    print(f"\nCollected {len(successful_trajs)} demos from {n_total_episodes} episodes "
          f"(SR={len(successful_trajs)/n_total_episodes:.3f})")
    print(f"Trajectory lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.1f}, median={np.median(lengths):.1f}")

    # Save as H5 (compatible with load_demo_dataset: traj_i/obs, traj_i/actions)
    if args.output is None:
        if args.uniform:
            args.output = os.path.expanduser(
                f"~/.maniskill/demos/{args.env_id}/policy_uniform/"
                f"trajectory.state.{args.control_mode}.h5"
            )
        elif args.demo_path:
            args.output = os.path.expanduser(
                f"~/.maniskill/demos/{args.env_id}/policy_from_mp_states/"
                f"trajectory.state.{args.control_mode}.h5"
            )
        else:
            args.output = os.path.expanduser(
                f"~/.maniskill/demos/{args.env_id}/policy_collected/"
                f"trajectory.state.{args.control_mode}.h5"
            )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with h5py.File(args.output, "w") as f:
        for i, (obs, act) in enumerate(successful_trajs):
            grp = f.create_group(f"traj_{i}")
            grp.create_dataset("obs", data=obs.astype(np.float32))
            grp.create_dataset("actions", data=act.astype(np.float32))
    print(f"Saved to {args.output}")


def _make_cpu_env(env_id, control_mode, max_episode_steps, cond_steps):
    """Factory for a single CPU env compatible with SyncVectorEnv."""
    from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack
    def thunk():
        env = gym.make(env_id, obs_mode="state", control_mode=control_mode,
                       max_episode_steps=max_episode_steps, reconfiguration_freq=1,
                       reward_mode="sparse")
        env = FrameStack(env, num_stack=cond_steps)
        env = CPUGymWrapper(env, ignore_terminations=False, record_metrics=True)
        return env
    return thunk


def _extract_raw_obs(obs, obs_dim):
    """Extract single-frame obs from potentially stacked obs.

    Handles both FrameStack 3D (N, cond_steps, obs_dim) and flat 2D (N, cond_dim).
    """
    if obs.dim() == 3:
        return obs[:, -1]  # (N, obs_dim)
    return obs[:, -obs_dim:]  # last obs_dim elements = most recent frame


def collect_random(args, model, device, cond_steps, act_steps, act_offset, use_gpu,
                   obs_dim=None):
    """Original collection mode: random initial states, collect until num_demos."""
    if use_gpu:
        env = gym.make(args.env_id, num_envs=args.num_envs, obs_mode="state",
                       control_mode=args.control_mode, max_episode_steps=args.max_episode_steps,
                       sim_backend="gpu", reward_mode="sparse")
        env = ManiSkillVectorEnv(env, args.num_envs, ignore_terminations=False,
                                 record_metrics=True)
    else:
        # CPU: use SyncVectorEnv with FrameStack (same pattern as eval_cpu.py)
        env = gym.vector.SyncVectorEnv(
            [_make_cpu_env(args.env_id, args.control_mode, args.max_episode_steps, cond_steps)
             for _ in range(args.num_envs)]
        )
    print(f"Env: {args.num_envs} envs, max_steps={args.max_episode_steps}, backend={args.sim_backend}")

    num_envs = args.num_envs
    env_obs_bufs = [[] for _ in range(num_envs)]
    env_act_bufs = [[] for _ in range(num_envs)]
    successful_trajs = []
    n_total_episodes = 0
    t_start = time.time()

    obs, _ = env.reset()
    if not isinstance(obs, torch.Tensor):
        obs = torch.from_numpy(obs).float().to(device)
    else:
        obs = obs.float().to(device)

    if use_gpu:
        # GPU: obs is (N, obs_dim), manually build obs_history (N, cond_steps, obs_dim)
        obs_cond = obs.unsqueeze(1).repeat(1, cond_steps, 1)
    else:
        # CPU + FrameStack: obs is already stacked, model handles via reshape(B, -1)
        obs_cond = obs  # (N, cond_steps, obs_dim) or (N, cond_dim)

    raw = _extract_raw_obs(obs, obs_dim) if not use_gpu else obs
    for i in range(num_envs):
        env_obs_bufs[i].append(raw[i].cpu().numpy())

    step = 0
    while len(successful_trajs) < args.num_demos:
        cond = {"state": obs_cond}
        with torch.no_grad():
            samples = model(cond, deterministic=args.deterministic,
                          min_sampling_denoising_std=args.min_sampling_std,
                          ddim_steps=args.ddim_steps)
        action_chunk = samples.trajectories

        for a_idx in range(act_steps):
            act_idx = act_offset + a_idx
            action = action_chunk[:, min(act_idx, action_chunk.shape[1] - 1)]

            if use_gpu:
                obs_new, reward, terminated, truncated, info = env.step(action)
                obs_new = obs_new.float()
            else:
                obs_new, reward, terminated, truncated, info = env.step(action.cpu().numpy())
                obs_new = torch.from_numpy(obs_new).float().to(device)
                reward = torch.from_numpy(np.asarray(reward)).float().to(device)
                terminated = torch.from_numpy(np.asarray(terminated)).bool().to(device)
                truncated = torch.from_numpy(np.asarray(truncated)).bool().to(device)

            obs_new_raw = obs_new if use_gpu else _extract_raw_obs(obs_new, obs_dim)

            for i in range(num_envs):
                env_act_bufs[i].append(action[i].cpu().numpy())
                env_obs_bufs[i].append(obs_new_raw[i].cpu().numpy())

            done_mask = terminated | truncated
            if done_mask.any():
                for i in done_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
                    success = reward[i].item() > 0.5
                    n_total_episodes += 1

                    if success and len(successful_trajs) < args.num_demos:
                        n_acts = len(env_act_bufs[i])
                        ep_obs = np.array(env_obs_bufs[i][:n_acts])
                        ep_act = np.array(env_act_bufs[i])
                        successful_trajs.append((ep_obs, ep_act))

                    env_obs_bufs[i] = [obs_new_raw[i].cpu().numpy()]
                    env_act_bufs[i] = []

            # Update obs conditioning
            if use_gpu:
                reset_mask = terminated | truncated
                if reset_mask.any():
                    obs_cond[reset_mask] = obs_new[reset_mask].unsqueeze(1).repeat(1, cond_steps, 1)
                obs_cond[~reset_mask] = torch.cat(
                    [obs_cond[~reset_mask, 1:], obs_new[~reset_mask].unsqueeze(1)], dim=1)
            else:
                # FrameStack handles stacking automatically
                obs_cond = obs_new

        step += 1
        if step % 100 == 0 or len(successful_trajs) >= args.num_demos:
            elapsed = time.time() - t_start
            sr = len(successful_trajs) / max(n_total_episodes, 1)
            print(f"  step={step}, collected={len(successful_trajs)}/{args.num_demos}, "
                  f"total_eps={n_total_episodes}, SR={sr:.3f}, time={elapsed:.1f}s",
                  flush=True)

    env.close()
    return successful_trajs, n_total_episodes


def collect_uniform(args, model, device, cond_steps, act_steps, act_offset, obs_dim):
    """Uniform coverage: sample N scene configs, retry each until success.

    Uses single CPU env with FrameStack + CPUGymWrapper.
    reconfiguration_freq=0 so reset() keeps the same geometry.
    For each scene config:
    1. env.reset(options={'reconfigure': True}) → new geometry
    2. Stochastic rollout → if success, save traj
    3. If fail, env.reset() → same geometry, new random pose, clean FrameStack → retry
    """
    from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack

    env = gym.make(args.env_id, obs_mode="state", control_mode=args.control_mode,
                   max_episode_steps=args.max_episode_steps, reconfiguration_freq=0,
                   reward_mode="sparse")
    env = FrameStack(env, num_stack=cond_steps)
    env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)

    n_states = args.num_initial_states
    successful_trajs = []
    total_attempts = 0
    t_start = time.time()
    max_retries = args.max_retries
    print(f"Uniform collection: {n_states} states, max_retries={max_retries}, "
          f"max_steps={args.max_episode_steps}", flush=True)

    for si in range(n_states):
        success = False
        for attempt in range(max_retries):
            total_attempts += 1

            if attempt == 0:
                # New scene config (new geometry)
                obs, _ = env.reset(options={"reconfigure": True})
            else:
                # Same geometry, new random pose, clean FrameStack
                obs, _ = env.reset()

            obs = np.asarray(obs)
            obs_buf = []
            act_buf = []
            # Save initial raw obs (last frame)
            raw_obs = obs[-1] if obs.ndim == 2 else obs[-obs_dim:]
            obs_buf.append(raw_obs.copy())

            done = False
            ep_success = False
            env_step = 0
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)

            while env_step < args.max_episode_steps and not done:
                cond = {"state": obs_t}
                with torch.no_grad():
                    samples = model(cond, deterministic=False,
                                    min_sampling_denoising_std=args.min_sampling_std,
                                    ddim_steps=args.ddim_steps)
                action_chunk = samples.trajectories[0]  # (horizon, act_dim)

                for a_idx in range(act_steps):
                    if env_step >= args.max_episode_steps or done:
                        break
                    act_idx = act_offset + a_idx
                    action_np = action_chunk[min(act_idx, action_chunk.shape[0] - 1)].cpu().numpy()

                    obs_new, reward, terminated, truncated, info = env.step(action_np)
                    obs_new = np.asarray(obs_new)
                    if isinstance(reward, torch.Tensor):
                        reward = reward.item()

                    act_buf.append(action_np)
                    raw_new = obs_new[-1] if obs_new.ndim == 2 else obs_new[-obs_dim:]
                    obs_buf.append(raw_new.copy())
                    env_step += 1

                    if reward > 0.5:
                        ep_success = True
                        done = True
                    elif terminated or truncated:
                        done = True

                obs_t = torch.from_numpy(obs_new).float().unsqueeze(0).to(device)
                if done:
                    break

            if ep_success:
                n_acts = len(act_buf)
                ep_obs = np.array(obs_buf[:n_acts], dtype=np.float32)
                ep_act = np.array(act_buf, dtype=np.float32)
                successful_trajs.append((ep_obs, ep_act))
                success = True
                break

        if (si + 1) % 10 == 0 or si == n_states - 1:
            elapsed = time.time() - t_start
            print(f"  state {si+1}/{n_states}: collected={len(successful_trajs)}, "
                  f"attempts={total_attempts}, avg_retries={total_attempts/(si+1):.1f}, "
                  f"time={elapsed:.1f}s", flush=True)

        if not success:
            print(f"  WARNING: state {si} failed after {max_retries} retries, skipping",
                  flush=True)

    env.close()
    return successful_trajs, n_states


if __name__ == "__main__":
    main()
