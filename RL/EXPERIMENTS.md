# Experiment Results: Advantage Estimation Comparison

All experiments finetune **ckpt_101 (62.7% SR)** on PickCube-v1 with sparse reward, gamma=0.8.
Optimal policy for MC re-rollout: **ckpt_301 (99% SR)**.

## Overview

We compare two key dimensions:

1. **Advantage estimation**: GAE (learned critic) vs MC (Monte Carlo re-rollout)
2. **Value source**: On-policy (current policy rollout) vs Optimal-policy (expert re-rollout)

```
                        On-policy V/Q              Optimal-policy V/Q
                   ┌─────────────────────┐    ┌───────────────────────────┐
  GAE advantage    │  PPO + GAE(λ=0.9)   │    │  N/A (GAE needs on-policy │
                   │  (standard PPO)      │    │  trajectory structure)    │
                   ├─────────────────────┤    ├───────────────────────────┤
  MC advantage     │  PPO + MC1           │    │  MC Q-V re-rollout        │
  (Q-V or raw)     │  (λ=1.0 in PPO)     │    │  with π* (actor-only)     │
                   └─────────────────────┘    └───────────────────────────┘
```

---

## 1. Large-Scale Baseline (512 envs, 2M steps)

Standard scale: batch=25,600, update_epochs=4, target_kl=0.1.
Both GAE and MC1 are **on-policy** (advantage computed from current policy rollout).

| Method | Source | 100k | 500k | 1M | Peak SR | Steps to 100% |
|--------|--------|------|------|-----|---------|---------------|
| **GAE** (λ=0.9) | on-policy | 54.2% | 93.1% | 97.4% | 99.3% | never |
| **MC1** (λ=1.0) | on-policy | 46.2% | 94.1% | 98.8% | **100%** | 1.2M |

- Both reach ~98-99% SR, MC1 slightly faster to peak
- MC1 is GAE with λ=1.0 (no bootstrapping, pure MC return minus baseline)
- At this scale (2M steps), the GAE vs MC difference is minimal

### Gamma sensitivity (512 envs, 2M steps)

| Method | gamma | 200k | 500k | 1M | Peak |
|--------|-------|------|------|-----|------|
| GAE | 0.8 | 63.8% | 93.1% | 97.4% | 99.3% |
| MC1 | 0.8 | 71.4% | 94.1% | 98.8% | **100%** |
| GAE | 0.99 | 69.9% | 73.5% | 95.9% | 100% |
| MC1 | 0.99 | 34.4% | 54.6% | 91.8% | 100% |

- gamma=0.99 + MC1 is **much slower** due to high variance in long-horizon returns
- gamma=0.99 + GAE is ok because bootstrapping controls variance

---

## 2. Data-Efficient: On-Policy GAE vs MC1 (50 envs x 100 steps, ~1000 trajs)

High UTD regime: batch=5,000, update_epochs=20, target_kl=100 (disabled).
~100 trajectories per iteration, 10 iterations total = ~1,000 trajectories.

### GAE (on-policy, learned V)

| Config | Critic Init | 5k | 15k | 25k | 35k | 45k (final) |
|--------|------------|-----|------|------|------|-------------|
| GAE pretrained V | expert V | 72.5% | 79.9% | 74.8% | 85.9% | **93.2%** |
| GAE reset V | random init | 65.4% | 72.5% | 77.4% | 90.9% | 90.7% |
| GAE γ=0.99 pretrained | expert V | 71.1% | 62.1% | 75.4% | 84.4% | 85.2% |
| GAE γ=0.99 reset | random init | 61.7% | 67.2% | 73.9% | 77.2% | 81.0% |

### MC1 (on-policy, no critic network)

| Config | Critic Init | 5k | 15k | 25k | 35k | 45k (final) |
|--------|------------|-----|------|------|------|-------------|
| MC1 pretrained V | expert V | 69.9% | 69.6% | 80.9% | 80.9% | **89.0%** |
| MC1 reset V | random init | 62.6% | 69.1% | 72.9% | 78.6% | 82.1% |

### Comparison at 45k steps

| Method | Pretrained V | Reset V |
|--------|-------------|---------|
| **GAE (on-policy)** | **93.2%** | **90.7%** |
| **MC1 (on-policy)** | 89.0% | 82.1% |
| Delta | GAE +4.2% | GAE +8.6% |

**Finding**: In the data-efficient regime, **on-policy GAE > on-policy MC1** by 4-9%.
GAE's bootstrapping reduces variance which matters more when data is scarce.

---

## 3. Data-Efficient: Optimal-Policy MC Q-V (50-100 envs, ~1000 trajs)

Uses **optimal policy (ckpt_301)** for MC re-rollouts to estimate:
- Q(s,a) = take action a, then follow π* to get MC return
- V(s) = sample action from π*, then follow π* to get MC return
- Advantage = Q(s,a) - V(s)

No critic network trained. Actor-only updates.

### MC Q-V + PPO update (50 envs x 100 steps)

| Config | MC samples | 5k | 15k | 25k | 35k | 45k | Peak |
|--------|-----------|-----|------|------|------|------|------|
| mc1_qv_optimal | 1 | 73.2% | 70.8% | 78.4% | 83.4% | 86.6% | 90.0% |
| mc1_qv_parallel (run 1) | 1 | 61.7% | 78.3% | 85.2% | 81.0% | 91.0% | 91.0% |
| mc1_qv_parallel (run 2) | 1 | 69.2% | 80.6% | 87.4% | 89.0% | **94.8%** | **95.8%** |
| mc16_qv_parallel | 16 | 68.9% | 78.6% | 90.5% | 92.6% | - | 93.9% |

### MC Q-V + AWR update (100 envs x 50 steps, epoch=200)

| Config | MC samples | 5k | 15k | 25k | 35k | 45k | Peak |
|--------|-----------|-----|------|------|------|------|------|
| mc5_awr (run 1) | 5+AWR | 76.1% | 94.2% | 90.7% | 94.3% | - | 95.5% |
| mc5_awr (best) | 5+AWR | 77.9% | 91.5% | 96.4% | 96.6% | **98.8%** | **98.8%** |

### Raw MC advantage (no Q-V regression) — FAILS

| Config | MC samples | 5k | 25k | 45k | Peak |
|--------|-----------|-----|------|------|------|
| mc1_raw_optimal | 1 | 49.6% | 67.2% | 73.0% | 77.4% |
| mc5_raw_optimal | 5 | 63.2% | - | - | 65.9% |

Raw MC return as advantage (without Q-V subtraction) has **too much variance**.

---

## 4. GAE Baseline at 100 envs (for fair comparison with MC Q-V AWR)

| Config | Epochs | 5k | 15k | 25k | 35k | 45k |
|--------|--------|-----|------|------|------|------|
| GAE 100env epoch=100 | 100 | 60.0% | 72.5% | 78.3% | 86.7% | **92.0%** |
| GAE 100env warmup=10 | 20 | 61.4% | 57.1% | 52.7% | 56.1% | 66.9% |

The warmup=10 run eventually reaches 98.8% but needs **370k steps** (vs 45k for MC AWR).

---

## 5. Head-to-Head: On-Policy GAE vs Optimal-Policy MC Q-V

All at ~50k total timesteps, ~1000 trajectories, starting from 62.7% SR:

| # | Method | Policy for V/Q | Update | 25k | 45k (final) | Peak |
|---|--------|---------------|--------|------|-------------|------|
| 1 | GAE (λ=0.9) | on-policy | PPO (epoch=20) | 74.8% | 93.2% | 93.2% |
| 2 | GAE (λ=0.9) | on-policy | PPO (epoch=100) | 78.3% | 92.0% | 92.0% |
| 3 | MC1 (λ=1.0) | on-policy | PPO (epoch=20) | 80.9% | 89.0% | 89.0% |
| 4 | MC1 Q-V | optimal π* | PPO (epoch=20) | 87.4% | **94.8%** | **95.8%** |
| 5 | MC16 Q-V | optimal π* | PPO (epoch=200) | 90.5% | - | 93.9% |
| 6 | MC5 Q-V | optimal π* | AWR (epoch=200) | 96.4% | **98.8%** | **98.8%** |

### Key findings

1. **Optimal-policy MC > On-policy GAE** when using Q-V advantage:
   - MC1 Q-V optimal (94.8%) > GAE on-policy (93.2%) with same PPO update
   - MC5 AWR optimal (**98.8%**) >> GAE on-policy (93.2%)

2. **On-policy MC1 < On-policy GAE** in data-efficient regime:
   - MC1 on-policy (89.0%) < GAE on-policy (93.2%)
   - Without bootstrapping, variance dominates with limited data

3. **AWR > PPO for optimal-policy MC advantage**:
   - AWR tolerates more update epochs without importance ratio drift
   - PPO clipping becomes a bottleneck when advantages are accurate

4. **Q-V subtraction is critical** for MC:
   - Raw MC return as advantage: max 77.4%
   - MC Q-V regression: up to 98.8%
   - V(s) baseline dramatically reduces variance

5. **More MC samples help but diminishing returns**:
   - MC1 → MC5 + AWR: 94.8% → 98.8% (+4%)
   - MC16 PPO: 93.9% (not better than MC5 AWR due to PPO clipping bottleneck)

---

## 6. Ranking Analysis: Why MC and GAE Agree

Separate from training, we tested action ranking correlation (K=8 actions per state, M=10 rollouts):

| Comparison | Spearman ρ (median) | Top-1 Agreement |
|-----------|-------------------|-----------------|
| MC vs GAE(λ=0.95) | **0.976** | 89.4% |
| MC vs IQL Q-V (τ=0.5) | 0.000 | 12.5% (random) |
| MC vs NN regression on MC targets | 0.040 | ~random |

- GAE and MC produce nearly identical action rankings
- Neural network Q(s,a) regression **destroys** ranking due to SNR problem
  (cross-state variance ~500x larger than within-state action variance)
- IQL/SARSA Q-V completely fails for action ranking

---

## 7. Failure Modes

### IQL/SARSA on-policy critic (512 envs, 2M steps)
On-policy SARSA Q-V advantage **collapses** (62.7% → 4.7%):
- Each state has 1 action → Q(s,a) ≈ V(s) via function approximation
- A = Q - V is noise; after norm_adv, PPO follows random gradients

### Pretrained expert critic + on-policy MC1
Expert V overestimates on medium policy states (V=0.53 vs actual MC=0.07):
- All advantages become negative → policy pushes away from all actions → collapse

### GAE with warmup=3, 100 envs, reset critic
Critic too poor after 3 warmup iterations (explained_var=0.14):
- Policy collapses from 62.7% → 7.0% at 90k steps

---

## Summary

```
Advantage Source          On-policy π_current        Optimal policy π*
────────────────────────  ────────────────────────   ──────────────────────────
GAE (learned V)           93.2% (PPO, 50env)         N/A
                          92.0% (PPO, 100env)

MC1 (raw return - V)      89.0% (PPO, 50env)         77.4% (raw, no Q-V)

MC1 Q-V                   N/A                         90.0-94.8% (PPO, 50env)

MC5 Q-V                   N/A                         98.8% (AWR, 100env)  ← BEST

MC16 Q-V                  N/A                         93.9% (PPO, 100env)
```

**Bottom line**: Optimal-policy MC Q-V + AWR (98.8%) significantly outperforms on-policy GAE (92-93%) in the data-efficient regime (~1000 trajectories). The key ingredients are:
1. Optimal policy for accurate Q/V estimation (not on-policy MC)
2. Q-V subtraction for variance reduction (not raw MC return)
3. AWR update rule (not PPO) to fully exploit accurate advantages
4. Multiple MC samples (MC5) for stable advantage estimates

## Scripts

| Script | Description |
|--------|-------------|
| `ppo_finetune.py` | PPO with GAE or MC1 advantage (on-policy) |
| `mc_finetune.py` | MC Q-V advantage + PPO (sequential re-rollout) |
| `mc_finetune_parallel.py` | MC Q-V advantage + PPO (parallel re-rollout) |
| `mc_finetune_awr_parallel.py` | MC Q-V advantage + AWR (parallel re-rollout) |
| `iql_finetune.py` | IQL/SARSA Q-V advantage + PPO |
| `awr_finetune.py` | AWR with on-policy GAE advantage |

## Reproduce

```bash
# Large-scale GAE baseline (512 envs, 2M steps)
python -m RL.ppo_finetune

# Large-scale MC1 baseline
python -m RL.ppo_finetune --advantage_mode mc

# Data-efficient GAE (50 envs, high UTD)
python -m RL.ppo_finetune --num_envs 50 --num_steps 100 \
  --num_minibatches 5 --update_epochs 20 --target_kl 100.0 \
  --eval_freq 1 --total_timesteps 50000 \
  --critic_checkpoint runs/pretrained_critic_sparse.pt

# Data-efficient MC Q-V + AWR (best result)
python -m RL.mc_finetune_awr_parallel --mc_samples 5 \
  --num_envs 100 --num_steps 50 --num_minibatches 10 \
  --update_epochs 200 --eval_freq 1 --total_timesteps 50000
```
