"""Compare eval results across different ManiSkill sim backends.

Must run each backend in a separate process (physx_cpu and physx_cuda
cannot coexist in the same process).
"""
import subprocess
import sys


def run_eval(backend_code, label):
    """Run eval in a subprocess and return (sr, time)."""
    result = subprocess.run(
        [sys.executable, "-c", backend_code],
        capture_output=True, text=True, timeout=300,
    )
    # Parse output
    for line in result.stdout.strip().split("\n"):
        if line.startswith("RESULT:"):
            parts = line.split()
            sr = float(parts[1])
            elapsed = float(parts[2])
            return sr, elapsed
    print(f"  [{label}] STDERR: {result.stderr[-500:]}")
    return None, None


COMMON_SETUP = """
import torch, numpy as np, gymnasium as gym, time
import mani_skill.envs
from DPPO.model.unet_wrapper import DiffusionUNet
from DPPO.model.diffusion import DiffusionModel

device = torch.device("cuda")
ckpt = torch.load("runs/dppo_pretrain/dppo_1000traj_unet_5k/ckpt_1500.pt", map_location=device, weights_only=False)
args = ckpt["args"]
obs_dim, action_dim = ckpt["obs_dim"], ckpt["action_dim"]
cond_steps, horizon_steps, act_steps = args["cond_steps"], args["horizon_steps"], args["act_steps"]
denoising_steps = args["denoising_steps"]
act_offset = cond_steps - 1
network = DiffusionUNet(action_dim=action_dim, horizon_steps=horizon_steps,
    cond_dim=obs_dim*cond_steps, diffusion_step_embed_dim=args.get("diffusion_step_embed_dim",64),
    down_dims=args.get("unet_dims",[64,128,256]), n_groups=args.get("n_groups",8))
model = DiffusionModel(network=network, horizon_steps=horizon_steps, obs_dim=obs_dim,
    action_dim=action_dim, device=device, denoising_steps=denoising_steps,
    denoised_clip_value=1.0, randn_clip_value=10, final_action_clip_value=1.0, predict_epsilon=True)
model.load_state_dict(ckpt["ema"], strict=False)
if hasattr(model,"eta") and hasattr(model.eta,"eta_logit"):
    if torch.isnan(model.eta.eta_logit.data).any() or torch.isinf(model.eta.eta_logit.data).any():
        model.eta.eta_logit.data = torch.atanh(torch.tensor([0.999]))
model.eval()
n_episodes, max_ep_steps, control_mode = 100, 100, "pd_ee_delta_pos"
"""

# --- physx_cpu (SyncVectorEnv + FrameStack, gold standard) ---
CPU_SYNC_CODE = COMMON_SETUP + """
from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack
def make_env(seed):
    def thunk():
        env = gym.make("PickCube-v1", obs_mode="state", control_mode=control_mode,
                      render_mode="rgb_array", max_episode_steps=max_ep_steps, reconfiguration_freq=1)
        env = FrameStack(env, num_stack=cond_steps)
        env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
        return env
    return thunk

envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(10)])
eps_done = 0
success_list = []
t0 = time.time()
with torch.no_grad():
    while eps_done < n_episodes:
        obs, _ = envs.reset()
        obs = torch.from_numpy(obs).float().to(device)
        for step in range(max_ep_steps):
            samples = model({"state": obs}, deterministic=True)
            action_np = samples.trajectories[:, act_offset:act_offset+act_steps].cpu().numpy()
            for a_idx in range(action_np.shape[1]):
                obs_np, rew, terminated, truncated, info = envs.step(action_np[:, a_idx])
                if truncated.any(): break
            obs = torch.from_numpy(obs_np).float().to(device)
            if truncated.any():
                for fi in info.get("final_info", []):
                    if fi and "episode" in fi:
                        success_list.append(fi["episode"]["success_once"])
                eps_done += 10
                break
envs.close()
elapsed = time.time() - t0
sr = np.mean(success_list[:n_episodes])
print(f"RESULT: {sr:.4f} {elapsed:.1f}")
"""

# --- GPU (physx_cuda) via ManiSkillVectorEnv ---
GPU_VEC_CODE = COMMON_SETUP + """
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
num_envs = 100
env = gym.make("PickCube-v1", num_envs=num_envs, obs_mode="state",
               control_mode=control_mode, max_episode_steps=max_ep_steps, sim_backend="gpu")
env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=True, record_metrics=True)

obs, _ = env.reset(seed=0)
obs_hist = obs.float().unsqueeze(1).repeat(1, cond_steps, 1).to(device)
success_list = []
t0 = time.time()
with torch.no_grad():
    for step in range(max_ep_steps):
        samples = model({"state": obs_hist}, deterministic=True)
        action_chunk = samples.trajectories
        for a_idx in range(act_steps):
            act_idx = act_offset + a_idx
            action = action_chunk[:, act_idx] if act_idx < action_chunk.shape[1] else action_chunk[:, -1]
            obs_new, rew, term, trunc, info = env.step(action)
            obs_new = obs_new.float().to(device)
            reset_mask = term | trunc
            if reset_mask.any():
                obs_hist[reset_mask] = obs_new[reset_mask].unsqueeze(1).repeat(1, cond_steps, 1)
            obs_hist[~reset_mask] = torch.cat(
                [obs_hist[~reset_mask, 1:], obs_new[~reset_mask].unsqueeze(1)], dim=1)
            if trunc.any(): break
        if trunc.any():
            if "final_info" in info:
                mask = info["_final_info"]
                vals = info["final_info"]["episode"]["success_once"][mask]
                for v in vals: success_list.append(v.item())
            break
env.close()
elapsed = time.time() - t0
sr = np.mean(success_list) if success_list else 0.0
print(f"RESULT: {sr:.4f} {elapsed:.1f}")
"""

# --- GPU with partial_reset (ignore_terminations=False), same as training ---
GPU_PARTIAL_CODE = COMMON_SETUP + """
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
num_envs = 100
env = gym.make("PickCube-v1", num_envs=num_envs, obs_mode="state",
               control_mode=control_mode, max_episode_steps=max_ep_steps,
               sim_backend="gpu", reward_mode="sparse")
env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=False, record_metrics=True)

obs, _ = env.reset(seed=0)
obs_hist = obs.float().unsqueeze(1).repeat(1, cond_steps, 1).to(device)
n_succ = 0
t0 = time.time()
with torch.no_grad():
    for step in range(25):  # 25 decision steps like finetune rollout
        samples = model({"state": obs_hist}, deterministic=True)
        action_chunk = samples.trajectories
        for a_idx in range(act_steps):
            act_idx = act_offset + a_idx
            action = action_chunk[:, act_idx] if act_idx < action_chunk.shape[1] else action_chunk[:, -1]
            obs_new, rew, term, trunc, info = env.step(action)
            obs_new = obs_new.float().to(device)
            n_succ += (rew > 0.5).sum().item()
            reset_mask = term | trunc
            if reset_mask.any():
                obs_hist[reset_mask] = obs_new[reset_mask].unsqueeze(1).repeat(1, cond_steps, 1)
            obs_hist[~reset_mask] = torch.cat(
                [obs_hist[~reset_mask, 1:], obs_new[~reset_mask].unsqueeze(1)], dim=1)
env.close()
elapsed = time.time() - t0
# Report n_succ as a fraction for comparison
sr = n_succ / (100 * 25 * 8)  # approximate
print(f"RESULT: {n_succ} {elapsed:.1f}")
"""


if __name__ == "__main__":
    print("Backend comparison (ckpt_1500, deterministic eval, 100 episodes):")
    print(f"  Model: UNet T=100 H=16, pd_ee_delta_pos, no_norm")
    print()

    backends = [
        ("physx_cpu (SyncVec, gold std)", CPU_SYNC_CODE),
        ("gpu (ManiSkillVecEnv)", GPU_VEC_CODE),
    ]

    for label, code in backends:
        sr, elapsed = run_eval(code, label)
        if sr is not None:
            print(f"  {label:<35s}  SR={sr:>6.1%}  time={elapsed:>5.1f}s")
        else:
            print(f"  {label:<35s}  FAILED")

    # Also run partial_reset mode (training-style rollout) on GPU
    print()
    print("Training-style rollout (partial_reset, 25 decision steps, 100 envs):")
    sr, elapsed = run_eval(GPU_PARTIAL_CODE, "gpu partial_reset")
    if sr is not None:
        print(f"  gpu (partial_reset)  n_succ={int(float(sr))}  time={elapsed:.1f}s")
