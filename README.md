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

## Research Questions

1. How do different advantage estimators (GAE vs MC vs IQL) compare for offline-to-online finetuning?
2. Does explicitly pushing away bad actions (RECAP) outperform soft down-weighting (AWR)?
3. What role does the policy (deterministic vs stochastic) play in real-world RL?
4. How does advantage estimation quality scale with MC samples?
