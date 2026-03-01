"""Compare coverage (P(success|s0)) between two checkpoints on the same initial states.

Cross-tabulates: for states that were frac_zero/decisive/frac_one under ckpt_0,
what's their P(success) distribution under ckpt_1?

Usage:
  python dp_coverage_compare.py \
    --ckpt0 runs/dppo_pretrain/dppo_pretrain_PegInsertionSide-v1_T100_H16_1M/best.pt \
    --ckpt1 runs/dppo_finetune/dppo_ft_peg_conservative/best.pt \
    --n-states 200 --n-rollouts 16
"""
import os, sys
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import mani_skill.envs  # noqa
import numpy as np
import torch
import tyro
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack

sys.path.insert(0, os.path.dirname(__file__))
from DPPO.model.unet_wrapper import DiffusionUNet
from DPPO.model.diffusion import DiffusionModel


@dataclass
class Args:
    ckpt0: str = "runs/dppo_pretrain/dppo_pretrain_PegInsertionSide-v1_T100_H16_1M/best.pt"
    ckpt1: str = "runs/dppo_finetune/dppo_ft_peg_conservative/best.pt"
    env_id: str = "PegInsertionSide-v1"
    control_mode: str = "pd_joint_delta_pos"
    max_episode_steps: int = 200

    n_states: int = 200
    n_rollouts: int = 16
    min_sampling_denoising_std: float = 0.01
    ddim_steps: int = 10
    seed_start: int = 0


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    is_finetuned = "model" in ckpt and "ema" not in ckpt

    ckpt_args = ckpt["args"]
    obs_dim, act_dim = ckpt["obs_dim"], ckpt["action_dim"]
    cond_steps = ckpt_args.get("cond_steps", 2)
    horizon_steps = ckpt_args.get("horizon_steps", 16)
    act_steps = ckpt_args.get("act_steps", 8)
    denoising_steps = ckpt_args.get("denoising_steps", 100)

    network = DiffusionUNet(
        action_dim=act_dim, horizon_steps=horizon_steps,
        cond_dim=obs_dim * cond_steps,
        diffusion_step_embed_dim=ckpt_args.get("diffusion_step_embed_dim", 64),
        down_dims=ckpt_args.get("unet_dims", [64, 128, 256]),
        n_groups=ckpt_args.get("n_groups", 8),
    )
    model = DiffusionModel(
        network=network, horizon_steps=horizon_steps,
        obs_dim=obs_dim, action_dim=act_dim, device=device,
        denoising_steps=denoising_steps,
        denoised_clip_value=1.0, randn_clip_value=10,
        final_action_clip_value=1.0, predict_epsilon=True,
    )

    if is_finetuned:
        sd = ckpt["model"]
        remapped = {k.replace("actor_ft.", "network.", 1): v
                    for k, v in sd.items() if k.startswith("actor_ft.")}
        model.load_state_dict(remapped, strict=False)
    else:
        model.load_state_dict(ckpt["ema"], strict=False)
    model.eval()

    return model, cond_steps, act_steps, obs_dim, act_dim


def compute_coverage(model, env, cond_steps, act_steps, seeds, n_rollouts,
                     min_std, ddim_steps, max_steps, device):
    """Compute P(success|s0) for each seed."""
    act_offset = cond_steps - 1
    p_success = np.zeros(len(seeds))

    for i, seed in enumerate(seeds):
        successes = 0
        for r in range(n_rollouts):
            obs, _ = env.reset(seed=int(seed))
            done = False
            success = False
            step = 0
            while step < max_steps and not done:
                obs_t = torch.from_numpy(np.array(obs)).float().unsqueeze(0).to(device)
                cond = {"state": obs_t}
                samples = model(cond, deterministic=False,
                                min_sampling_denoising_std=min_std,
                                ddim_steps=ddim_steps)
                action_np = samples.trajectories[:, act_offset:act_offset + act_steps].cpu().numpy()[0]
                for a_idx in range(action_np.shape[0]):
                    if step >= max_steps or done:
                        break
                    obs, rew, term, trunc, info = env.step(action_np[a_idx])
                    step += 1
                    if rew > 0.5:
                        success = True
                        done = True
                    elif term or trunc:
                        done = True
            if success:
                successes += 1
        p_success[i] = successes / n_rollouts

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(seeds)} done, running SR={p_success[:i+1].mean():.3f}")

    return p_success


def main():
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seeds = np.arange(args.seed_start, args.seed_start + args.n_states)

    # Create env
    env = gym.make(args.env_id, obs_mode="state", render_mode="rgb_array",
                   reward_mode="sparse", control_mode=args.control_mode,
                   max_episode_steps=args.max_episode_steps, reconfiguration_freq=1)
    if isinstance(env.action_space, gym.spaces.Dict):
        from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
        env = FlattenActionSpaceWrapper(env)
    env = FrameStack(env, num_stack=2)
    env = CPUGymWrapper(env, ignore_terminations=False, record_metrics=True)

    # Load and eval ckpt0
    print(f"=== Checkpoint 0 (pretrained): {args.ckpt0} ===")
    model0, cs0, as0, _, _ = load_model(args.ckpt0, device)
    p0 = compute_coverage(model0, env, cs0, as0, seeds, args.n_rollouts,
                          args.min_sampling_denoising_std, args.ddim_steps,
                          args.max_episode_steps, device)
    del model0
    torch.cuda.empty_cache()

    # Load and eval ckpt1
    print(f"\n=== Checkpoint 1 (finetuned): {args.ckpt1} ===")
    model1, cs1, as1, _, _ = load_model(args.ckpt1, device)
    p1 = compute_coverage(model1, env, cs1, as1, seeds, args.n_rollouts,
                          args.min_sampling_denoising_std, args.ddim_steps,
                          args.max_episode_steps, device)
    del model1
    env.close()

    # Analysis
    print("\n" + "=" * 70)
    print("COVERAGE COMPARISON")
    print("=" * 70)

    # Overall stats
    for name, p in [("ckpt0 (pretrained)", p0), ("ckpt1 (finetuned)", p1)]:
        frac_zero = (p < 0.05).mean()
        frac_decisive = ((p >= 0.1) & (p <= 0.9)).mean()
        frac_one = (p > 0.95).mean()
        print(f"\n{name}:")
        print(f"  SR={p.mean():.3f}, frac_zero={frac_zero:.3f}, "
              f"frac_decisive={frac_decisive:.3f}, frac_one={frac_one:.3f}")

    # Cross-tabulation
    print("\n" + "-" * 70)
    print("CROSS-TABULATION: ckpt0 category → ckpt1 P(success)")
    print("-" * 70)

    categories = {
        "frac_zero (P0<0.05)": p0 < 0.05,
        "low (0.05≤P0<0.3)": (p0 >= 0.05) & (p0 < 0.3),
        "decisive (0.3≤P0≤0.7)": (p0 >= 0.3) & (p0 <= 0.7),
        "high (0.7<P0≤0.95)": (p0 > 0.7) & (p0 <= 0.95),
        "frac_one (P0>0.95)": p0 > 0.95,
    }

    for cat_name, mask in categories.items():
        n = mask.sum()
        if n == 0:
            print(f"\n{cat_name}: 0 states")
            continue

        p1_subset = p1[mask]
        p0_subset = p0[mask]

        # How did these states fare under ckpt1?
        still_zero = (p1_subset < 0.05).sum()
        became_decisive = ((p1_subset >= 0.1) & (p1_subset <= 0.9)).sum()
        became_one = (p1_subset > 0.95).sum()

        print(f"\n{cat_name}: {n} states (P0 mean={p0_subset.mean():.3f})")
        print(f"  Under ckpt1: P1 mean={p1_subset.mean():.3f} (Δ={p1_subset.mean()-p0_subset.mean():+.3f})")
        print(f"  → still_zero={still_zero} ({still_zero/n*100:.1f}%), "
              f"decisive={became_decisive} ({became_decisive/n*100:.1f}%), "
              f"frac_one={became_one} ({became_one/n*100:.1f}%)")

        # Distribution of P1 for these states
        pcts = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        quantiles = np.quantile(p1_subset, pcts)
        print(f"  P1 quantiles: " + ", ".join(f"{p:.0%}→{q:.3f}" for p, q in zip(pcts, quantiles)))

    # Correlation
    from scipy.stats import pearsonr, spearmanr
    r_pearson = pearsonr(p0, p1)[0]
    r_spearman = spearmanr(p0, p1).correlation
    print(f"\nCorrelation P0 vs P1: pearson={r_pearson:.3f}, spearman={r_spearman:.3f}")

    # Save data
    np.savez("runs/coverage_compare_peg.npz",
             seeds=seeds, p0=p0, p1=p1,
             ckpt0=args.ckpt0, ckpt1=args.ckpt1)
    print(f"\nData saved to runs/coverage_compare_peg.npz")


if __name__ == "__main__":
    main()
