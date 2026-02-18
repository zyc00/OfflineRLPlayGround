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

| Method | Advantage | Critic | Iter 5 | Iter 10 |
|--------|-----------|--------|--------|---------|
| GAE | GAE(lambda=0.9) | pretrained V^pi | 74.8% | **93.2%** |
| GAE | GAE(lambda=0.9) | reset | 73.4% | 90.7% |
| MC1 | MC(lambda=1.0) | pretrained V^pi | 69.9% | 89.0% |
| GAE | GAE(lambda=0.9) | V* (optimal) | 66.2% | 83.3% |
| MC1 | MC(lambda=1.0) | reset | 62.6% | 82.1% |

- Pretrained critic consistently ~3% better than reset
- GAE > MC1 in data-efficient regime (bias-variance tradeoff helps)
- V* worse than V^pi_expert due to distribution mismatch

### 3. Data-Efficient: Gamma & Regression Tricks (50 envs, 50k timesteps)

| Config | gamma | Critic | Scale | Iter 10 |
|--------|-------|--------|-------|---------|
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

| Method | MC samples | Update | Epochs | Iter 5 | Iter 10 | Peak SR |
|--------|------------|--------|--------|--------|---------|---------|
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

## Research Questions

1. How do different advantage estimators (GAE vs MC vs IQL) compare for offline-to-online finetuning?
2. Does explicitly pushing away bad actions (RECAP) outperform soft down-weighting (AWR)?
3. What role does the policy (deterministic vs stochastic) play in real-world RL?
4. How does advantage estimation quality scale with MC samples?
