# AWR Ablation: Complete Experiment Results

All experiments finetune from `ckpt_101` (62.7% SR) on PickCube-v1 with sparse reward.
Optimal policy for MC re-rollout: `ckpt_301` (99% SR).
Shared: 100 envs, 50 steps, batch=5000, update_epochs=200, actor-only AWR.

---

## 1. Beta Sweep (mc16, gamma=0.8)

| Iter | Step | beta=0.1 | beta=0.3 | beta=0.5 | beta=1.0 | beta=2.0 |
|------|------|----------|----------|----------|----------|----------|
| 1 | 0 | 62.7% | 62.7% | 62.7% | 62.7% | 62.7% |
| 2 | 5,000 | 73.2% | 81.4% | 76.6% | 77.9% | 71.5% |
| 3 | 10,000 | 85.6% | 92.0% | 91.0% | 87.0% | 75.9% |
| 4 | 15,000 | 92.0% | 98.4% | 94.7% | 91.5% | 82.8% |
| 5 | 20,000 | 95.0% | 96.6% | 98.2% | 94.2% | 90.5% |
| 6 | 25,000 | 92.2% | 97.7% | 95.3% | 96.4% | 94.4% |
| 7 | 30,000 | 97.0% | 96.1% | 97.9% | 94.4% | 96.7% |
| 8 | 35,000 | **97.5%** | 98.0% | **99.2%** | 96.6% | **98.0%** |
| 9 | 40,000 | 95.4% | **98.8%** | 96.8% | 95.5% | 97.1% |
| 10 | 45,000 | 96.8% | 98.8% | 97.7% | **98.8%** | 96.5% |

### Summary

| beta | Peak SR | Final SR | Peak Iter |
|------|---------|----------|-----------|
| 0.1 | 97.5% | 96.8% | 8 |
| 0.3 | 98.8% | 98.8% | 9 |
| **0.5** | **99.2%** | **97.7%** | **8** |
| 1.0 | 98.8% | 98.8% | 10 |
| 2.0 | 98.0% | 96.5% | 8 |

**Best: beta=0.5** (99.2% peak). Sweet spot between greedy (0.1) and uniform (2.0). AWR robust across full range (97.5%–99.2%).

---

## 2. MC Samples Sweep (beta=0.5, gamma=0.8)

| Iter | Step | mc=1 | mc=5 | mc=16 |
|------|------|------|------|-------|
| 1 | 0 | 62.7% | 62.7% | 62.7% |
| 2 | 5,000 | 64.7% | 70.5% | 76.6% |
| 3 | 10,000 | 80.0% | 87.4% | 91.0% |
| 4 | 15,000 | 87.6% | 95.1% | 94.7% |
| 5 | 20,000 | 90.9% | 97.5% | 98.2% |
| 6 | 25,000 | 91.7% | 94.4% | 95.3% |
| 7 | 30,000 | **97.5%** | 97.9% | 97.9% |
| 8 | 35,000 | 92.6% | **98.0%** | **99.2%** |
| 9 | 40,000 | 94.3% | 97.5% | 96.8% |
| 10 | 45,000 | 94.9% | 95.2% | 97.7% |

### Summary

| mc_samples | Peak SR | Final SR | Peak Iter |
|------------|---------|----------|-----------|
| 1 | 97.5% | 94.9% | 7 |
| 5 | 98.0% | 95.2% | 8 |
| **16** | **99.2%** | **97.7%** | **8** |

**Best: mc=16**. More MC samples → better advantage estimation → higher peak and more stable final. mc1 has highest variance (97.5% peak but drops to 92.6% next iter).

---

## 3. Gamma Sweep (mc16, beta=0.5)

| Iter | Step | gamma=0.8 | gamma=0.9 | gamma=0.95 | gamma=0.99 |
|------|------|-----------|-----------|------------|------------|
| 1 | 0 | 62.7% | 62.7% | 62.7% | 62.7% |
| 2 | 5,000 | 76.6% | 75.2% | 77.8% | 77.1% |
| 3 | 10,000 | 91.0% | 93.7% | 87.7% | 91.8% |
| 4 | 15,000 | 94.7% | 98.9% | 97.7% | 91.6% |
| 5 | 20,000 | 98.2% | **99.05%** | 96.7% | 96.3% |
| 6 | 25,000 | 95.3% | 96.8% | 97.0% | 96.5% |
| 7 | 30,000 | 97.9% | 98.8% | 97.1% | 95.6% |
| 8 | 35,000 | **99.2%** | 97.5% | **97.95%** | 94.7% |
| 9 | 40,000 | 96.8% | 97.1% | 97.3% | 96.5% |
| 10 | 45,000 | 97.7% | 98.8% | 97.1% | **97.5%** |

### Summary

| gamma | Peak SR | Final SR | Peak Iter |
|-------|---------|----------|-----------|
| **0.8** | **99.2%** | 97.7% | 8 |
| 0.9 | 99.05% | 98.8% | 5 |
| 0.95 | 97.95% | 97.1% | 8 |
| 0.99 | 97.5% | 97.5% | 10 |

**Best peak: gamma=0.8** (99.2%). **Best stability: gamma=0.9** (99.05% peak, 98.8% final). For 50-step episodes, gamma=0.8 is sufficient discount; higher gamma hurts.

---

## 4. Online vs Offline vs On-Policy AWR

### Online AWR (optimal policy re-rollout, iterative)

Already covered in sweeps above. Best config (mc16, beta=0.5, gamma=0.8): **99.2% peak**.

### Offline AWR (optimal policy re-rollout, fixed dataset, 100 iters, eval every 5)

| Iter | mc5 beta=1.0 | mc16 beta=0.5 |
|------|--------------|---------------|
| 1 | 59.4% | 60.9% |
| 5 | 56.1% | 45.5% |
| 10 | 62.2% | 63.7% |
| 15 | 70.4% | 71.1% |
| 20 | 66.9% | 71.6% |
| 25 | 72.3% | 77.0% |
| 30 | 69.7% | 71.9% |
| 35 | 66.4% | 70.9% |
| 40 | 65.0% | 61.2% |
| 45 | 65.2% | 79.0% |
| 50 | 76.3% | 69.4% |
| 55 | 70.4% | 79.3% |
| 60 | 75.4% | 75.0% |
| 65 | 81.2% | 72.5% |
| 70 | 71.9% | 75.4% |
| 75 | 72.4% | **81.0%** |
| 80 | 77.6% | 76.4% |
| 85 | 73.0% | 75.0% |
| 90 | **85.3%** | 77.1% |
| 95 | 73.4% | 75.0% |
| 100 | 77.5% | 79.7% |

### On-Policy AWR (current policy re-rollout, iterative)

| Iter | Step | mc1 beta=0.5 | mc5 beta=0.5 | mc16 beta=0.5 |
|------|------|--------------|--------------|---------------|
| 1 | 0 | 62.7% | 62.7% | 62.7% |
| 2 | 5,000 | 68.7% | 64.0% | 77.9% |
| 3 | 10,000 | 72.7% | 78.6% | 87.1% |
| 4 | 15,000 | 88.4% | 86.6% | 90.4% |
| 5 | 20,000 | 87.9% | 93.1% | 88.3% |
| 6 | 25,000 | 88.3% | 93.8% | 90.2% |
| 7 | 30,000 | 89.2% | 93.5% | 93.2% |
| 8 | 35,000 | 93.7% | 93.6% | **94.9%** |
| 9 | 40,000 | **95.5%** | 93.2% | 94.6% |
| 10 | 45,000 | 94.5% | **95.1%** | 94.6% |

### Full Comparison

| Method | Re-rollout | Data | mc | Peak SR | Final SR |
|--------|-----------|------|-----|---------|----------|
| **Online AWR** | optimal (π*) | iterative | 16 | **99.2%** | 97.7% |
| **Online AWR** | optimal (π*) | iterative | 5 | 98.0% | 95.2% |
| On-policy AWR | current (π) | iterative | 1 | 95.5% | 94.5% |
| On-policy AWR | current (π) | iterative | 5 | 95.1% | 95.1% |
| On-policy AWR | current (π) | iterative | 16 | 94.9% | 94.6% |
| GAE PPO | — | iterative | — | 92.0% | 92.0% |
| Offline MC16 | on-policy (π) | fixed | 16 | 89.1% | 77.2% |
| Offline AWR | optimal (π*) | fixed | 5 | 85.3% | 77.5% |
| Offline MC16 | optimal (π*) | fixed | 16 | 83.9% | 72.9% |
| Offline IQL | IQL Q-V | fixed | — | 83.2% | 79.7% |
| Offline AWR | optimal (π*) | fixed | 16 | 81.0% | 79.7% |

---

## 5. IQL AWR Offline (no MC re-rollout, no optimal policy)

Fully offline: train IQL Q(s,a) and V(s) on offline data, then use A = Q-V for AWR. No simulator access needed.

### Tau Sweep (beta=1.0, 100 iterations, 5-ckpt mixed data)

| Experiment | Iter 1 | Iter 10 | Iter 30 | Iter 50 | Iter 70 | Iter 100 | Peak | Final |
|---|---|---|---|---|---|---|---|---|
| tau0.5 | 61.2% | 56.5% | 68.6% | 67.2% | 61.9% | 75.4% | 75.6% (i35) | 75.4% |
| tau0.7 | 61.2% | 56.4% | 63.8% | 76.4% | 58.5% | 73.2% | 76.4% (i50) | 73.2% |
| tau0.9 | 61.2% | 72.9% | 68.1% | 66.2% | 62.7% | 74.8% | 74.8% (i100) | 74.8% |

### Tuned (tau0.9, nstep=5, 500 ep/ckpt, beta=0.5, lr=1e-4, 200 iters)

Peak: **76.2%** (iter 95) | Final: 70.8% (iter 200)

### Single-Checkpoint (ckpt_101 only, tau0.7, beta=1.0)

Peak: **83.2%** (iter 25) | Final: 79.7% (iter 100)

On-distribution data (ckpt_101 only, 83.2%) >> mixed data (5 ckpts, 76.4%). Fewer but on-policy transitions > more off-policy transitions.

---

## 6. Fair Offline Comparison: MC16 vs IQL vs Optimal (all batch=25,600)

Three advantage sources, all offline AWR on fixed ckpt_101 data:

### Advantage Statistics

| Metric | MC16 on-policy (ckpt_101) | MC16 optimal (ckpt_301) | IQL (ckpt_101) |
|--------|--------------------------|------------------------|-----------------|
| A mean | -0.0002 | **-0.0799** | -0.0104 |
| A std | **0.1113** | 0.1088 | 0.0170 |
| A pos% | ~50% | 22.6% | 28.5% |
| Overhead | 669s | 669s | 35s |

### Results

| Iter | MC16 on-policy | MC16 optimal | IQL |
|------|----------------|--------------|-----|
| 1 | 66.2% | 59.7% | 57.3% |
| 5 | 73.2% | 68.6% | 74.5% |
| 10 | 78.1% | 63.4% | 77.1% |
| 25 | 73.3% | 69.4% | **83.2%** |
| 30 | 83.7% | 73.9% | 69.8% |
| 50 | 80.9% | 67.6% | 80.9% |
| 75 | — | 77.5% | 73.0% |
| 80 | 82.4% | 71.7% | — |
| 95 | **89.1%** | **83.9%** | 78.0% |
| 100 | 77.2% | 72.9% | 79.7% |

### Summary

| Method | Advantage Source | Peak SR | Final SR | A mean |
|--------|----------------|---------|----------|--------|
| **MC16 on-policy** | Q^π_101 - V^π_101 | **89.1%** | 77.2% | -0.0002 |
| MC16 optimal | Q^π* - V^π* | 83.9% | 72.9% | -0.0799 |
| IQL | Q_IQL - V_IQL | 83.2% | 79.7% | -0.0104 |

**Key insight: In offline AWR, on-policy re-rollout > optimal re-rollout.** Optimal policy creates systematic negative advantage bias (A mean = -0.08, only 22.6% positive) because ckpt_101's actions are suboptimal from π\*'s perspective. On-policy has near-zero bias (~50% positive), providing a better learning signal on fixed data.

This is opposite to the iterative setting (optimal 99.2% > on-policy 95.5%), because iterative re-collection corrects the distribution mismatch each iteration.

---

## 7. Best Config

| Parameter | Value |
|-----------|-------|
| mc_samples | 16 |
| awr_beta | 0.5 |
| gamma | 0.8 |
| update_epochs | 200 |
| num_envs | 100 |
| num_steps | 50 |
| **Peak SR** | **99.2%** |

## Key Conclusions

1. **Online >> Offline**: +14% (99.2% vs 85.3%). Iterative re-rollout (adapting data to improving policy) is the critical ingredient.
2. **Iterative: Optimal >> On-policy** (+4%): 99.2% vs 95.5%. Iterative re-collection corrects distribution mismatch each iteration, so optimal policy's better advantage signal wins.
3. **Offline: On-policy >> Optimal** (+5%): 89.1% vs 83.9%. Optimal re-rollout creates systematic negative advantage bias (A mean=-0.08, pos%=22.6%) on fixed data. On-policy has near-zero bias (~50% positive).
4. **On-policy mc1 AWR (95.5%) vs GAE PPO (92.0%)**: Direct comparison — both use on-policy data with 1 sample, but AWR tolerates 200 update epochs vs PPO's 4, explaining the +3.5% gap.
5. **AWR is robust to beta**: 97.5%–99.2% across beta=0.1–2.0. Best at 0.5.
6. **More MC samples help** (with optimal policy): mc1 97.5% → mc16 99.2%. But with suboptimal policy, mc1 ≈ mc5 ≈ mc16 (~95%).
7. **gamma=0.8 optimal** for 50-step sparse-reward episodes. gamma=0.9 close second with best stability.
8. **IQL AWR limited by sparse reward**: Peak 83.2% with on-distribution data (ckpt_101 only). IQL advantage std is 6.5x smaller than MC (0.017 vs 0.111) — weak discriminative signal.
9. **IQL data distribution > quantity**: Single-checkpoint on-policy data (83.2%) >> mixed 5-checkpoint data (76.4%). On-distribution transitions give better Q/V estimates.
10. **All offline methods oscillate heavily** (10%+ between evals) and plateau at 77-89%. The fixed-dataset bottleneck dominates regardless of advantage quality.
