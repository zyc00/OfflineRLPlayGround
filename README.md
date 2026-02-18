# Offline RL Playground

Offline-to-online RL finetuning research platform built on [ManiSkill](https://github.com/haosulab/ManiSkill). Compares advantage estimation methods (GAE, MC, IQL/SARSA) and policy update rules (PPO, AWR, RECAP) for finetuning pretrained manipulation policies.

## Setup

```bash
# Python 3.12+, uv recommended
uv sync

# Or pip
pip install -e .
```

Dependencies: PyTorch 2.10+, ManiSkill 3.0+, Gymnasium 0.29+, TensorBoard.

## Project Structure

```
RL/                          # Training scripts (entry points)
  ppo_finetune.py            # PPO finetuning (GAE or MC advantage)
  mc_finetune.py             # MC Q-V advantage + PPO update
  mc_finetune_parallel.py    # Parallel MC re-rollout + PPO
  mc_finetune_awr_parallel.py  # Parallel MC + AWR update
  mc_finetune_awr_recap.py   # Parallel MC + RECAP-style AWR
  iql_finetune.py            # PPO with IQL/SARSA-learned Q,V
  awr_finetune.py            # AWR finetuning (actor-critic)
  sarsa_finetune.py          # Q-V regression + PPO
  pretrain_critic.py         # Critic pretraining on MC returns
  pretrain_iql.py            # IQL Q,V pretraining

methods/                     # Algorithm implementations
  gae/                       # GAE critic + advantage computation
  iql/                       # IQL Q-network + expectile loss
  mc/                        # Monte Carlo advantage estimation

data/                        # Data utilities
  data_collection/ppo.py     # Agent class (actor-critic MLP)
  data_collection/collect_dataset.py
  offline_dataset.py         # PyTorch Dataset for offline transitions

envs/                        # Custom environments
  pick_cube.py               # PickCube-v2 (custom variant)

stats/                       # Analysis & comparison scripts
```

## Quick Start

All scripts use [tyro](https://github.com/brentyi/tyro) for CLI args. Run `--help` for full options.

### 1. PPO Finetuning (baseline)

```bash
# GAE advantage (default)
python -m RL.ppo_finetune --checkpoint runs/pickcube_ppo/ckpt_101.pt

# MC advantage (1 sample)
python -m RL.ppo_finetune --advantage_mode mc --mc_samples 1
```

### 2. MC Q-V Finetuning (parallel re-rollout)

Uses optimal policy for MC re-rollouts to estimate `Q(s,a) - V(s)`:

```bash
# MC5 + PPO update
python -m RL.mc_finetune_parallel --mc_samples 5

# MC5 + AWR update
python -m RL.mc_finetune_awr_parallel --mc_samples 5 --awr_beta 1.0

# MC5 + RECAP AWR (pushes away bad actions)
python -m RL.mc_finetune_awr_recap --mc_samples 5 --recap_alpha 1.0
```

### 3. IQL / SARSA Finetuning

```bash
# Pretrain IQL Q,V networks
python -m RL.pretrain_iql --expectile_tau 0.7

# Finetune with IQL advantage
python -m RL.iql_finetune --expectile_tau 0.7
```

### 4. Critic Pretraining

```bash
python -m RL.pretrain_critic --gamma 0.8 --reward_mode sparse
```

## Advantage Estimation Methods

| Method | Source | How |
|--------|--------|-----|
| **GAE** | Learned critic V(s) | TD(lambda) with online critic |
| **MC** | Re-rollout with optimal policy | Q = MC return after (s,a); V = MC return from s |
| **IQL** | Pretrained Q(s,a), V(s) | Expectile regression on offline data |
| **SARSA** | Pretrained Q(s,a), V(s) | MSE regression (tau=0.5 IQL) |

## Policy Update Rules

| Method | Positive advantage | Negative advantage |
|--------|-------------------|-------------------|
| **PPO** | Clipped IS ratio increase | Clipped IS ratio decrease |
| **AWR** | `exp(A/beta)` weighted BC | Small weight (near-ignore) |
| **RECAP AWR** | `exp(A/beta)` weighted BC | `exp(|A|/beta)` reverse BC (push away) |

## Key Hyperparameters

| Param | Default | Description |
|-------|---------|-------------|
| `gamma` | 0.8 | Discount factor (0.8 works well for sparse reward, 50-step episodes) |
| `awr_beta` | 1.0 | AWR temperature (lower = more greedy) |
| `awr_max_weight` | 20.0 | Weight clamp to prevent exp explosion |
| `recap_alpha` | 1.0 | Negative loss weight (0 = pure AWR) |
| `mc_samples` | 5 | MC re-rollout samples per (s,a) |
| `num_envs` | 512 | Parallel training environments |
| `reward_scale` | 1.0 | Reward multiplier |

## Environment

Default: **PickCube-v1** (ManiSkill) -- pick up a cube and move it to a goal position.

- 512 parallel envs (GPU-accelerated via PhysX CUDA)
- State observations, `pd_joint_delta_pos` control
- 50-step episodes, sparse reward
- Eval: 128 envs, success rate metric

## Monitoring

```bash
tensorboard --logdir runs/
```

Key metrics: `eval/success_rate`, `charts/advantage_mean`, `losses/policy_loss`, `charts/pos_ratio` (RECAP).

## Experiment Results

All experiments finetune from `ckpt_101` (62.7% SR) on PickCube-v1 with sparse reward. Optimal policy for MC re-rollout: `ckpt_301` (99% SR).

### 1. Large-Scale PPO (512 envs, 2M timesteps)

Standard scale training. Batch=25,600, update_epochs=4, target_kl=0.1.

| Method | gamma | Iter 5 | Iter 10 | Iter 25 | Peak SR | Final SR |
|--------|-------|--------|---------|---------|---------|----------|
| GAE | 0.8 | 55.6% | 90.0% | 99.2% | 99.3% (i50) | 98.6% |
| MC1 | 0.8 | 67.6% | 84.1% | 99.6% | **100%** (i35) | 98.3% |
| GAE | 0.99 | 46.2% | 69.9% | 98.4% | **100%** (i35) | 99.6% |
| MC1 | 0.99 | 25.4% | 34.4% | 73.5% | 100% (i60) | 100% |

- gamma=0.8 converges much faster early; gamma=0.99 is more stable at convergence
- MC1 with gamma=0.99 suffers from high variance in MC returns

### 2. Data-Efficient PPO (50 envs, 50k timesteps, ~10 iterations)

High UTD regime: update_epochs=20, target_kl=100 (disabled), batch=5,000.

| Method | Advantage | Critic | @25k | @45k (final) |
|--------|-----------|--------|------|--------------|
| GAE | GAE(lambda=0.9) | pretrained V^pi | 74.8% | **93.2%** |
| GAE | GAE(lambda=0.9) | reset | 77.4% | 90.7% |
| MC1 | MC(lambda=1.0) | pretrained V^pi | 80.9% | 89.0% |
| GAE | GAE(lambda=0.9) | V* (optimal) | 83.8% | 83.3% |
| MC1 | MC(lambda=1.0) | reset | 72.9% | 82.1% |

- Pretrained critic consistently ~3% better than reset
- GAE > MC1 in data-efficient regime (bias-variance tradeoff helps)
- V* worse than V^pi_expert due to distribution mismatch

### 3. Data-Efficient: Gamma & Regression Tricks (50 envs, 50k timesteps)

| Config | gamma | Critic | Scale | @45k (final) |
|--------|-------|--------|-------|--------------|
| Pretrained | 0.8 | pretrained | 1.0 | **93.2%** |
| Reset | 0.8 | 3x256 reset | 1.0 | 90.7% |
| Pretrained | 0.99 | pretrained | 1.0 | 85.2% |
| Reset | 0.99 | reset | 1.0 | 81.0% |
| Scale 20 | 0.8 | 3x256 reset | 20.0 | 87.7% |
| Big critic | 0.8 | 10x512 reset | 1.0 | 89.7% |
| Both | 0.8 | 10x512 reset | 20.0 | 81.2% |

- gamma=0.99 is 5-8% worse in data-efficient regime
- Regression tricks (reward scaling, big critic) do NOT help online PPO
- Big critic + scale combined is worst (overfits small batch)

### 4. MC Q-V Parallel Re-rollout (100 envs, 50k timesteps)

Actor-only training with oracle advantage from optimal policy re-rollouts.

| Method | MC samples | Update | Epochs | @20k | @45k | Peak SR |
|--------|------------|--------|--------|------|------|---------|
| MC16 AWR | 16 | AWR | 200 | 94.2% | **98.8%** | **98.8%** |
| MC5 PPO | 5 | PPO | 100 | 80.6% | 94.8% | 95.8% |
| GAE PPO | - | PPO | 100 | 74.3% | 92.0% | 92.0% |

- **MC16 AWR is the best data-efficient result: 98.8%** with only 50k timesteps
- AWR converges faster than PPO (no importance ratio drift, tolerates more epochs)
- MC re-rollout advantage > GAE advantage (~3-6% gap)

### Summary: Best Configs by Regime

| Regime | Best Method | Peak SR | Total Timesteps |
|--------|-------------|---------|-----------------|
| Large-scale | MC1 PPO (gamma=0.8) | 100% | 2M |
| Data-efficient (PPO) | GAE + pretrained critic | 93.2% | 50k |
| Data-efficient (AWR) | MC16 AWR (actor-only) | **98.8%** | 50k |

### 5. Action Ranking Quality: Which Advantage Estimator Ranks Actions Correctly?

Offline analysis: for each eval state, sample K=8 actions, estimate advantage with different methods, measure Spearman rho against MC ground truth (M=10 rollouts). Scripts in `methods/gae/rank_*.py`.

**Core comparison (M=10 rollouts per action):**

| Method | Spearman rho | Top-1 Agree | Notes |
|--------|-------------|-------------|-------|
| MC (M=10) | 1.000 | 100% | Ground truth |
| IQL>traj (bypass Q, use IQL's V + GAE) | 0.958 | 90.5% | Best learned method |
| GAE (lambda=0.95, MC-supervised V) | 0.931 | 86.4% | Strong baseline |
| TD(20) | 0.960 | 91.8% | Needs 20 steps |
| TD(5) | 0.343 | 37.6% | Too few steps |
| GAE (lambda=0) / TD(1) | 0.071 | 22.7% | 1-step TD is useless |
| IQL Q-network (tau=0.5) | 0.010 | 12.5% | Random — Q-net fails |

**Why IQL Q-network fails — SNR problem:**

| Metric | Value |
|--------|-------|
| Q cross-state variance | 0.265 |
| Q within-state variance | 0.0001 |
| SNR ratio | 495x (cross-state dominates) |
| IQL V quality (Pearson r) | 0.963 (excellent) |
| IQL Q action ranking | 0.010 (random) |

The Q-network learns V(s) well but cannot resolve the tiny action-dependent residual A(s,a).

**NN regression destroys ranking (rank_nn_regression):**

| Method | rho vs MC | Notes |
|--------|-----------|-------|
| Sample GAE (direct from trajectory) | 0.931 | No NN, direct computation |
| NN(GAE) — NN trained on GAE targets | 0.023 | NN regression destroys it |
| NN(Q_MC) — NN trained on MC Q targets | -0.005 | Q-scale SNR problem |
| NN(A_MC) — NN trained on MC A targets | 0.040 | Even with perfect targets |

**Larger networks don't help (rank_network_size):**

| hidden_dim | GAE(0.95) | IQL Q-net | IQL>traj(0.95) |
|-----------|-----------|-----------|----------------|
| 256 | 0.931 | 0.007 | 0.960 |
| 512 | 0.934 | 0.006 | 0.956 |
| 1024 | 0.934 | 0.010 | 0.954 |

**Effect of rollout count M (rank_nstep_td):**

| M rollouts | MC | TD(50) | GAE(0.95) |
|-----------|-----|--------|-----------|
| 1 | 0.300 | 0.258 | 0.270 |
| 2 | 0.406 | 0.379 | 0.386 |
| 4 | 0.528 | 0.521 | 0.506 |
| 8 | 0.701 | 0.696 | 0.671 |
| 16 | 1.000 | 0.997 | 0.931 |

**Can heavy tuning fix Q regression? (rank_qv_regression):**

MC-supervised Q regression with aggressive tuning (scale_factor=20, action_repeat=8, normalize, 10-layer 512-dim network, 4000 epochs):

| Config | Q-V ranking (mean) | Q-V ranking (med) | GAE ranking (mean) | GAE ranking (med) |
|--------|-------|--------|-------|--------|
| Default (3x256, 200ep) | -0.011 | 0.024 | 0.723 | 0.905 |
| **Tuned** (10x512, norm, repeat=8, scale=20, 4000ep) | **0.726** | **0.898** | **0.862** | **0.952** |

Q ranking improved from random to 0.73/0.90, but still below GAE (0.86/0.95). Required 2.4M params on 3712 samples (645x overparameterized) + MC ground-truth targets.

**Can IQL match MC via tuning? (rank_iql_tune):**

IQL's TD-based training with same tuning knobs — the practical question is whether TD bootstrapping can match MC regression:

| Config | Q_nn-V_nn (mean) | Q_nn-V_nn (med) | GAE w/ IQL V |
|--------|---------|---------|------------|
| Default IQL (3x256, 10ep) | -0.009 | 0.000 | 0.872 |
| +norm, scale=5, repeat=4, nstep=5 | 0.101 | 0.084 | 0.944 |
| +scale=20, 10x512, repeat=8, nstep=1, 4000ep | 0.149 | 0.143 | 0.955 |
| +nstep=10 | 0.154 | 0.156 | 0.958 |
| +nstep=50 | 0.176 | 0.190 | 0.962 |

IQL's Q-network improves with tuning but plateaus at ~0.18 (vs MC regression's 0.73). TD bootstrapping corrupts V during training (V loss goes UP), limiting Q quality. Meanwhile, GAE with IQL's V reaches 0.96 without any Q-network.

**Key conclusions:**
1. **Learned V(s) is excellent; learned Q(s,a) cannot rank actions** — the SNR problem is fundamental, not a capacity issue
2. **GAE(lambda=0.95) with trajectory rollouts** is the best practical method (rho=0.931) — it uses V only for bootstrapping, not action discrimination
3. **NN MSE regression on any target** destroys ranking quality (0.931 -> 0.023) — the regression objective doesn't prioritize within-state ordering
4. **Single rollout (M=1) is unreliable** for all methods (~0.3) — need M>=8 for decent ranking with sparse rewards
5. **Heavy tuning can partially fix Q regression** (0.01 -> 0.73 with MC targets, 0.01 -> 0.18 with TD targets), but **GAE always wins** without needing a Q-network at all

## Experiment Plan

**Established finding**: MC-N (large N) + AWR achieves highly data-efficient online finetuning (MC16 AWR 98.8% @ 50k steps), significantly outperforming PPO.

### Phase 1: How Much Does Online Finetuning Actually Help?

**Core question**: Compare pure offline vs offline-to-online — is the online improvement worth the extra environment interaction cost?

| Experiment | Method | Goal |
|------------|--------|------|
| 1a | Pure offline (AWR/IQL on offline data) | Establish offline-only baseline |
| 1b | Offline + online MC16 AWR (current best) | Quantify incremental gain from online phase |
| 1c | Varying online budget (10k/25k/50k steps) | Plot online budget vs SR curve, find the cost-effectiveness inflection point |

**Expected output**: Online budget vs Success Rate curve, clearly answering "how much online interaction yields how much improvement".

### Phase 2: How to Train an Optimal Critic?

**Core question**: The current MC Q-V method relies on an optimal policy for re-rollout, which is unavailable in practice. Can we train a critic from pure offline data that approximates the optimal critic?

**Evaluation metric**: How closely the trained critic's advantage approximates optimal-policy MC Q-V advantage (the oracle advantage verified to improve efficiency).

| Experiment | Method | Evaluation |
|------------|--------|------------|
| 2a | Q-learning variants (CQL, IQL with aggressive tau) | Spearman ρ vs MC oracle ranking |
| 2b | TD(N) with large N + offline data | Can large N compensate for bootstrapping error? |
| 2c | NN regression on MC targets (larger networks / better losses) | Upper bound exploration beyond current SNR bottleneck |
| 2d | Replace oracle re-rollout with trained critic for AWR finetuning | End-to-end validation: critic quality → finetuning SR |

**Known challenges**:
- Action-dependent signal in Q(s,a) (SNR ~1:500) is dwarfed by state-dependent signal
- NN MSE regression destroys within-state ranking (0.931 → 0.023)
- IQL Q-net after heavy tuning only reaches ρ=0.18, far below GAE (ρ=0.93)

**Potential directions**:
- Contrastive/ranking loss instead of MSE to directly optimize within-state ordering
- Two-stage training: first learn V(s), then learn residual A(s,a) = Q - V
- Leverage action diversity in offline data to construct pairwise comparisons

## Research Questions

1. How do different advantage estimators (GAE vs MC vs IQL) compare for offline-to-online finetuning?
2. Does explicitly pushing away bad actions (RECAP) outperform soft down-weighting (AWR)?
3. What role does the policy (deterministic vs stochastic) play in real-world RL?
4. How does advantage estimation quality scale with MC samples?
