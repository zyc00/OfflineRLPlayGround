# Experiment Log
w

## [PPO Finetuning: Standard 512-env GAE/MC1 with Pretrained Critic] - 2026-02-17 14:00

**Git**: d53c0cc (main)

### Overview
Reproduce GAE and MC1 finetuning at scale (512 envs, default PPO settings) with pretrained critic, comparing gamma=0.8 vs gamma=0.99.

### Shared Settings
| Parameter | Value |
|-----------|-------|
| checkpoint | runs/pickcube_ppo/ckpt_101.pt |
| num_envs | 512 |
| num_eval_envs | 128 |
| num_steps | 50 |
| batch_size | 25,600 |
| num_minibatches | 32 |
| update_epochs | 4 |
| target_kl | 0.1 |
| clip_coef | 0.2 |
| vf_coef | 0.5 |
| reward_scale | 1.0 |
| reset_critic | True |
| total_timesteps | 2,000,000 |

### Commands
```bash
# Pretrain critic gamma=0.99
python -m RL.pretrain_critic --gamma 0.99 --output_path runs/pretrained_critic_sparse_gamma099.pt

# GAE gamma=0.8
python -m RL.ppo_finetune --critic_checkpoint runs/pretrained_critic_sparse.pt --advantage_mode gae --gamma 0.8 --exp_name repro_gae_gamma08

# MC1 gamma=0.8
python -m RL.ppo_finetune --critic_checkpoint runs/pretrained_critic_sparse.pt --advantage_mode mc --gamma 0.8 --exp_name repro_mc1_gamma08

# GAE gamma=0.99
python -m RL.ppo_finetune --critic_checkpoint runs/pretrained_critic_sparse_gamma099.pt --advantage_mode gae --gamma 0.99 --exp_name gae_gamma099

# MC1 gamma=0.99
python -m RL.ppo_finetune --critic_checkpoint runs/pretrained_critic_sparse_gamma099.pt --advantage_mode mc --gamma 0.99 --exp_name mc1_gamma099
```

### Results

| Experiment | Iter 5 | Iter 10 | Iter 15 | Iter 20 | Iter 25 | Iter 35 | Peak | Final (iter 78) |
|---|---|---|---|---|---|---|---|---|
| GAE γ=0.8 | 55.6% | 90.0% | 93.8% | 97.2% | 99.2% | 98.5% | 99.3% (i50) | 98.6% |
| MC1 γ=0.8 | 67.6% | 84.1% | 96.0% | 96.3% | 99.6% | 100% | 100% (i35) | 98.3% |
| GAE γ=0.99 | 46.2% | 69.9% | 87.0% | 97.1% | 98.4% | 100% | 100% (i35,45,65) | 99.6% |
| MC1 γ=0.99 | 25.4% | 34.4% | 42.2% | 54.6% | 73.5% | 86.6% | 100% (i60,78) | 100% |

### Run Dirs
- `runs/repro_gae_gamma08__seed1__1771366667`
- `runs/repro_mc1_gamma08__seed1__1771367181`
- `runs/gae_gamma099__seed1__1771367797`
- `runs/mc1_gamma099__seed1__1771368187`

### Notes
- gamma=0.8 converges faster early (MC1: 84.1% at iter 10 vs 34.4% for γ=0.99)
- gamma=0.99 GAE is surprisingly good — not much slower than γ=0.8, more stable at convergence (~100%)
- MC1 γ=0.99 is much slower to converge (high variance in MC returns with large gamma)
- GAE benefits from bias-variance tradeoff (λ=0.9) which compensates for high-gamma variance
- Pretrained critic for γ=0.99: mean_return=0.74 (vs ~0.24 for γ=0.8), val_MSE=0.103

---

## [Data-Efficient PPO: High-UTD Experiments] - 2026-02-17 15:00

**Git**: d53c0cc (main)

### Overview
Data-efficient RL training using 50 envs, 100 steps, high UTD (update_epochs=20), target_kl=100 (disabled). ~100 trajectories per iteration, 10 iterations ≈ 900 trajectories total. Tests pretrained vs reset critic, gamma=0.8 vs 0.99, and regression tricks (scale, big network).

### Shared Settings
| Parameter | Value |
|-----------|-------|
| checkpoint | runs/pickcube_ppo/ckpt_101.pt |
| num_envs | 50 |
| num_steps | 100 |
| batch_size | 5,000 |
| num_minibatches | 5 |
| update_epochs | 20 |
| target_kl | 100.0 |
| advantage_mode | gae |
| gae_lambda | 0.9 |
| total_timesteps | 50,000 |
| eval_freq | 1 |

### Commands
```bash
# Pretrained critic, gamma=0.8
python -u -m RL.ppo_finetune --num_envs 50 --num_steps 100 --num_minibatches 5 --update_epochs 20 --target_kl 100.0 --critic_checkpoint runs/pretrained_critic_sparse.pt --eval_freq 1 --total_timesteps 50000 --exp_name ppo_gae_highUTD_pretrained

# Reset critic, gamma=0.8
python -u -m RL.ppo_finetune --num_envs 50 --num_steps 100 --num_minibatches 5 --update_epochs 20 --target_kl 100.0 --eval_freq 1 --total_timesteps 50000 --exp_name ppo_gae_highUTD_reset

# Pretrained critic, gamma=0.99
python -u -m RL.ppo_finetune --num_envs 50 --num_steps 100 --num_minibatches 5 --update_epochs 20 --target_kl 100.0 --gamma 0.99 --critic_checkpoint runs/pretrained_critic_sparse_gamma099.pt --eval_freq 1 --total_timesteps 50000 --exp_name ppo_gae_highUTD_pretrained_gamma099

# Reset critic, gamma=0.99
python -u -m RL.ppo_finetune --num_envs 50 --num_steps 100 --num_minibatches 5 --update_epochs 20 --target_kl 100.0 --gamma 0.99 --eval_freq 1 --total_timesteps 50000 --exp_name ppo_gae_highUTD_reset_gamma099

# Ablation: reward_scale=20
python -u -m RL.ppo_finetune --num_envs 50 --num_steps 100 --num_minibatches 5 --update_epochs 20 --target_kl 100.0 --reward_scale 20 --eval_freq 1 --total_timesteps 50000 --exp_name efficient_scale20

# Ablation: big critic (10x512)
python -u -m RL.ppo_finetune --num_envs 50 --num_steps 100 --num_minibatches 5 --update_epochs 20 --target_kl 100.0 --critic_hidden_dim 512 --critic_num_layers 10 --eval_freq 1 --total_timesteps 50000 --exp_name efficient_big_critic

# Ablation: scale20 + big critic
python -u -m RL.ppo_finetune --num_envs 50 --num_steps 100 --num_minibatches 5 --update_epochs 20 --target_kl 100.0 --reward_scale 20 --critic_hidden_dim 512 --critic_num_layers 10 --eval_freq 1 --total_timesteps 50000 --exp_name efficient_scale20_big_critic
```

### Results: Gamma Comparison

| Config | γ | Critic | Iter 5 | Iter 7 | Iter 9 | Iter 10 |
|--------|---|--------|--------|--------|--------|---------|
| Pretrained | 0.8 | pretrained | 74.8% | 81.1% | 86.2% | **93.2%** |
| Reset | 0.8 | reset | 73.4% | 83.9% | 83.9% | **90.7%** |
| Pretrained | 0.99 | pretrained | 70.4% | 79.3% | 84.5% | **85.2%** |
| Reset | 0.99 | reset | 74.2% | 77.0% | 86.2% | **81.0%** |

### Results: Regression Tricks Ablation (all reset critic, γ=0.8)

| Config | Critic | Scale | Iter 5 | Iter 9 | Iter 10 |
|--------|--------|-------|--------|--------|---------|
| Baseline | 3×256 | 1.0 | 73.4% | 83.9% | **90.7%** |
| Scale 20 | 3×256 | 20.0 | 81.2% | 89.0% | **87.7%** |
| Big critic | 10×512 | 1.0 | 69.7% | 81.0% | **89.7%** |
| Both | 10×512 | 20.0 | 63.7% | 72.1% | **81.2%** |

### Results: Earlier UTD Experiments (from previous session, γ=0.8)

| Config | Envs | Steps | Batch | UTD | Critic | Final SR |
|--------|------|-------|-------|-----|--------|----------|
| UTD30 pretrained | 50 | 100 | 5,000 | 30 | pretrained | 93.3% (93.9% peak) |
| UTD20 clip05 | 50 | 100 | 5,000 | 20 | pretrained | 91.6% |
| 250env UTD20 | 250 | 50 | 12,500 | 20 | pretrained | 89.0% (still rising) |
| 10env UTD30 | 10 | 100 | 1,000 | 30 | pretrained | 67.6% (collapsed) |

### Results: Failed Experiments

| Config | Issue |
|--------|-------|
| 100env warmup=3, UTD=20, target_kl=0.1 | Collapsed to 7% — critic not trained enough after 3 warmup iters, target_kl too tight |

### Run Dirs
- `runs/ppo_gae_highUTD_pretrained__seed1__1771316119`
- `runs/ppo_gae_highUTD_reset__seed1__1771316349`
- `runs/ppo_gae_UTD30_pretrained__seed1__1771316496`
- `runs/ppo_gae_UTD20_clip05__seed1__1771316657`
- `runs/ppo_gae_250env_UTD20__seed1__1771316812`
- `runs/ppo_gae_10env_UTD30__seed1__1771316911`
- `runs/ppo_gae_highUTD_pretrained_gamma099__seed1__1771371488`
- `runs/ppo_gae_highUTD_reset_gamma099__seed1__1771371920`
- `runs/efficient_scale20__seed1__1771372089`
- `runs/efficient_big_critic__seed1__1771372229`
- `runs/efficient_scale20_big_critic__seed1__1771372369`
- `runs/efficient_gae_gamma08_warmup__seed1__1771370053`

### Notes
- Best data-efficient config: 50 envs, 100 steps, UTD=20-30, target_kl=100 (disabled), pretrained critic → ~93% with ~900 trajs
- Pretrained critic consistently ~3% better than reset critic
- gamma=0.99 is ~5-8% worse than gamma=0.8 in data-efficient regime (both pretrained and reset)
- Regression tricks (scale_factor, big critic) do NOT help online PPO — they address SNR in Q(s,a) regression, but GAE only needs V(s) with error cancellation via telescoping
- Big critic + scale combined is WORST (81.2%) — overfits the small 5000-sample batch
- 10 envs (batch=1000) collapsed — too little data for stable PPO updates
- target_kl=0.1 (default) with high UTD causes collapse; target_kl=100 (disabled) is needed
- warmup_iters=3 with tight target_kl also causes collapse

---

## [V* Ablation & MC1 in Data-Efficient Setting] - 2026-02-17 16:00

**Git**: d53c0cc (main)

### Overview
Test whether V* (value function of optimal/converged policy) improves data efficiency compared to V^π_expert (expert policy at ckpt_101). Also test MC1 advantage in the data-efficient regime. Hypothesis: V* enables more policy updates per iteration since it's closer to the true value function across policies.

### V* Pretraining
```bash
# Pretrain V* from converged policy (99% SR, ckpt_301)
python -m RL.pretrain_critic --expert_checkpoint runs/pickcube_ppo/ckpt_301.pt --output_path runs/pretrained_critic_optimal.pt
```

### Commands
```bash
# V* critic, 50 envs (same data-efficient config)
python -u -m RL.ppo_finetune --num_envs 50 --num_steps 100 --num_minibatches 5 --update_epochs 20 --target_kl 100.0 --critic_checkpoint runs/pretrained_critic_optimal.pt --eval_freq 1 --total_timesteps 50000 --exp_name vstar_50env

# V* critic, 20 envs (more iterations, same total timesteps)
python -u -m RL.ppo_finetune --num_envs 20 --num_steps 100 --num_minibatches 5 --update_epochs 20 --target_kl 100.0 --critic_checkpoint runs/pretrained_critic_optimal.pt --eval_freq 1 --total_timesteps 50000 --exp_name vstar_20env

# MC1 with reset critic
python -u -m RL.ppo_finetune --num_envs 50 --num_steps 100 --num_minibatches 5 --update_epochs 20 --target_kl 100.0 --advantage_mode mc --eval_freq 1 --total_timesteps 50000 --exp_name efficient_mc1_reset

# MC1 with pretrained critic (V^π_expert)
python -u -m RL.ppo_finetune --num_envs 50 --num_steps 100 --num_minibatches 5 --update_epochs 20 --target_kl 100.0 --advantage_mode mc --critic_checkpoint runs/pretrained_critic_sparse.pt --eval_freq 1 --total_timesteps 50000 --exp_name efficient_mc1_pretrained
```

### Results: Full Data-Efficient Comparison (50 envs, UTD=20, target_kl=100, γ=0.8)

| Config | Advantage | Critic | @5k | @25k | @35k | @45k |
|--------|-----------|--------|-----|------|------|------|
| GAE pretrained | GAE(λ=0.9) | V^π_expert | 72.5% | 74.8% | 85.9% | **93.2%** |
| GAE reset | GAE(λ=0.9) | reset | 65.4% | 77.4% | 90.9% | **90.7%** |
| MC1 pretrained | MC(λ=1.0) | V^π_expert | 69.9% | 80.9% | 80.9% | **89.0%** |
| V* 50env | GAE(λ=0.9) | V* (optimal) | 66.2% | 83.8% | 85.1% | **83.3%** |
| MC1 reset | MC(λ=1.0) | reset | 62.6% | 72.9% | 78.6% | **82.1%** |

### Results: V* with More Iterations (20 envs, 25 iterations)

| Config | Iter 10 | Iter 15 | Iter 20 | Iter 25 |
|--------|---------|---------|---------|---------|
| V* 20env | 68.8% | 74.1% | 77.9% | **92.3%** |

### Run Dirs
- `runs/vstar_50env__seed1__1771373330`
- `runs/vstar_20env__seed1__1771373488`
- `runs/efficient_mc1_reset__seed1__1771373837`
- `runs/efficient_mc1_pretrained__seed1__1771373845`
- `runs/pretrained_critic_optimal.pt`

### Notes
- **V* is WORSE than V^π_expert**: 83.3% vs 93.2% — despite V* being trained on converged policy (99% SR)
- Root cause: **distribution mismatch** — V* was trained on states from π* (converged policy), but during finetuning the initial policy visits different states. V* doesn't generalize to π_0's state distribution
- V* with 20 envs eventually reaches 92.3% but needs 25 iterations (2500 trajectories) — not data-efficient
- MC1 pretrained (89.0%) < GAE pretrained (93.2%) — GAE's bias-variance tradeoff (λ=0.9) helps in data-efficient regime
- MC1 reset (82.1%) < GAE reset (90.7%) — same trend, ~8% gap
- **Conclusion**: V^π_expert > V* for finetuning because the expert (ckpt_101) visits similar states as the initial policy. How to train a V* that generalizes is a separate research question

---

## [MC5 Q-V Parallel Re-rollout PPO] - 2026-02-17 20:52

**Command**: `python -u -m RL.mc_finetune_parallel --mc_samples 5 --num_envs 100 --num_steps 50 --num_minibatches 5 --update_epochs 100 --target_kl 100.0 --eval_freq 1 --total_timesteps 50000 --exp_name mc1_qv_par`
**Git**: d53c0cc (main)
**Run Dir**: `runs/mc1_qv_par__seed1__1771389467`

### Settings
| Parameter | Value |
|-----------|-------|
| checkpoint | runs/pickcube_ppo/ckpt_101.pt |
| optimal_checkpoint | runs/pickcube_ppo/ckpt_301.pt |
| reward | sparse |
| gamma | 0.8 |
| mc_samples | 5 |
| samples_per_env | 10 |
| num_envs | 100 |
| num_mc_envs | 1000 |
| num_steps | 50 |
| batch_size | 5,000 |
| num_minibatches | 5 |
| minibatch_size | 1,000 |
| update_epochs | 100 |
| target_kl | 100.0 (disabled) |
| total_timesteps | 50,000 |
| iterations | 10 |
| actor_params | 144,656 |
| advantage | Q^π*(s,a) - V^π*(s), MC re-rollout |
| training | actor-only PPO (no critic) |

### Results
| Iter | Step | SR | Episodes |
|------|------|----|----------|
| 1 | 0 | 62.7% | 134 |
| 2 | 5,000 | 69.2% | 133 |
| 3 | 10,000 | 72.4% | 145 |
| 4 | 15,000 | 80.6% | 139 |
| 5 | 20,000 | 80.6% | 144 |
| 6 | 25,000 | 87.4% | 151 |
| 7 | 30,000 | 85.0% | 147 |
| 8 | 35,000 | 89.0% | 155 |
| 9 | 40,000 | **95.8%** | 168 |
| 10 | 45,000 | 94.8% | 173 |

| Metric | Value |
|--------|-------|
| Peak SR | **95.8%** (iter 9) |
| Final SR | 94.8% (iter 10) |

### Notes
- Uses `mc_finetune_parallel.py` — parallel Q-V re-rollout with separate `mc_envs`
- Advantage = Q^π*(s,a) - V^π*(s): Q estimated by taking a_t then following π*, V by sampling from π* then following π*
- 5 MC samples each for Q and V estimation (10 total re-rollouts per state)
- **update_epochs=100** (much higher than previous experiments with 20) — likely explains the strong 95.8% peak
- exp_name is `mc1_qv_par` but actually uses mc_samples=5 (mislabeled)
- Compare: serial MC1 Q-V (mc_samples=1, update_epochs=20) peaked at 90.0%; this MC5 with update_epochs=100 peaked at 95.8%
- Note: different hyperparams from previous MC1 parallel run (100 envs / 50 steps / update_epochs=100 vs 50 envs / 100 steps / update_epochs=20), so not directly comparable

---

## [GAE Baseline for MC5 Q-V Comparison] - 2026-02-17 20:58

**Command**: `python -u -m RL.ppo_finetune --num_envs 100 --num_steps 50 --num_minibatches 5 --update_epochs 100 --target_kl 100.0 --eval_freq 1 --total_timesteps 50000 --exp_name gae_100env_epoch100`
**Git**: d53c0cc (main)
**Run Dir**: `runs/gae_100env_epoch100__seed1__1771390535`

### Settings
| Parameter | Value |
|-----------|-------|
| checkpoint | runs/pickcube_ppo/ckpt_101.pt |
| reward | sparse |
| gamma | 0.8 |
| gae_lambda | 0.9 |
| num_envs | 100 |
| num_steps | 50 |
| batch_size | 5,000 |
| num_minibatches | 5 |
| minibatch_size | 1,000 |
| update_epochs | 100 |
| target_kl | 100.0 (disabled) |
| total_timesteps | 50,000 |
| iterations | 10 |
| critic | reset (3×256, 142,849 params) |
| advantage | GAE(λ=0.9) |
| training | actor + critic (on-policy PPO) |

### Results
| Iter | Step | SR | Episodes |
|------|------|----|----------|
| 1 | 0 | 62.7% | 134 |
| 2 | 5,000 | 60.0% | 130 |
| 3 | 10,000 | 65.4% | 133 |
| 4 | 15,000 | 72.5% | 131 |
| 5 | 20,000 | 74.3% | 136 |
| 6 | 25,000 | 78.3% | 138 |
| 7 | 30,000 | 82.3% | 147 |
| 8 | 35,000 | 86.7% | 150 |
| 9 | 40,000 | 91.4% | 152 |
| 10 | 45,000 | 92.0% | 150 |

| Metric | Value |
|--------|-------|
| Peak SR | **92.0%** (iter 10) |
| Final SR | 92.0% (iter 10) |

### Notes
- **Direct comparison with MC5 Q-V** (same setting: 100 envs, 50 steps, update_epochs=100, batch=5000):
  - MC5 Q-V: **95.8%** peak, 94.8% final
  - GAE reset: **92.0%** peak, 92.0% final
- MC5 Q-V is ~3-4% better, and converges faster (80.6% at iter 4 vs 72.5%)
- GAE still rising at iter 10 — might catch up with more iterations
- GAE uses on-policy critic (reset, trained from scratch); MC5 Q-V uses oracle advantage from π* re-rollouts

---

## [MC16 AWR Q-V Parallel Re-rollout] - 2026-02-17 22:43

**Command**: `python -u -m RL.mc_finetune_awr_parallel --mc_samples 16 --num_envs 100 --num_steps 50 --num_minibatches 5 --update_epochs 200 --eval_freq 1 --total_timesteps 50000 --exp_name mc5_awr`
**Git**: d53c0cc (main)
**Run Dir**: `runs/mc5_awr__seed1__1771394434`

### Settings
| Parameter | Value |
|-----------|-------|
| checkpoint | runs/pickcube_ppo/ckpt_101.pt |
| optimal_checkpoint | runs/pickcube_ppo/ckpt_301.pt |
| reward | sparse |
| gamma | 0.8 |
| mc_samples | 16 |
| samples_per_env | 32 |
| num_envs | 100 |
| num_mc_envs | 3,200 |
| num_steps | 50 |
| batch_size | 5,000 |
| num_minibatches | 5 |
| minibatch_size | 1,000 |
| update_epochs | 200 |
| awr_beta | 1.0 |
| awr_max_weight | 20.0 |
| total_timesteps | 50,000 |
| iterations | 10 |
| actor_params | 144,656 |
| advantage | Q^π*(s,a) - V^π*(s), MC re-rollout |
| training | actor-only AWR (no critic) |

### Results
| Iter | Step | SR | Episodes |
|------|------|----|----------|
| 1 | 0 | 62.7% | 134 |
| 2 | 5,000 | 77.9% | 140 |
| 3 | 10,000 | 87.0% | 138 |
| 4 | 15,000 | 91.5% | 165 |
| 5 | 20,000 | 94.2% | 171 |
| 6 | 25,000 | 96.4% | 196 |
| 7 | 30,000 | 94.4% | 215 |
| 8 | 35,000 | 96.6% | 232 |
| 9 | 40,000 | 95.5% | 220 |
| 10 | 45,000 | **98.8%** | 241 |

| Metric | Value |
|--------|-------|
| Peak SR | **98.8%** (iter 10) |
| Final SR | 98.8% (iter 10) |

### Notes
- First AWR experiment — replaces PPO clipped loss with advantage-weighted regression
- **98.8% is the best result so far** in the data-efficient setting (50k timesteps)
- exp_name is `mc5_awr` but actually uses mc_samples=16 (mislabeled)
- Comparison (all 100 envs, 50 steps, batch=5000):
  - MC16 AWR (update_epochs=200): **98.8%** peak
  - MC5 PPO (update_epochs=100): 95.8% peak
  - GAE PPO reset (update_epochs=100): 92.0% peak
- AWR converges faster: 87.0% at iter 3 vs MC5 PPO 72.4% and GAE 65.4%
- Note: update_epochs=200 (2× MC5 PPO, 10× original GAE) — AWR tolerates more epochs since no importance ratio drift
- 3200 MC envs for 16 samples each of Q and V (32 re-rollouts per state)

---

## [AWR Hyperparameter Tuning: Beta, MC Samples, Gamma] - 2026-02-17 23:30

**Git**: d53c0cc (main)

### Overview
Systematic hyperparameter sweep for AWR with Q-V advantage (optimal policy MC re-rollout). Base config: mc16, beta=1.0, update_epochs=200, 100 envs, 50 steps, gamma=0.8. Sweep one parameter at a time.

### Shared Settings
| Parameter | Value |
|-----------|-------|
| checkpoint | runs/pickcube_ppo/ckpt_101.pt |
| optimal_checkpoint | runs/pickcube_ppo/ckpt_301.pt |
| num_envs | 100 |
| num_steps | 50 |
| num_minibatches | 5 |
| update_epochs | 200 |
| total_timesteps | 50,000 |
| awr_max_weight | 20.0 |
| training | actor-only AWR |

### Commands
```bash
# Beta sweep (mc16, gamma=0.8)
python -u -m RL.mc_finetune_awr_parallel --mc_samples 16 --awr_beta 0.1 --num_envs 100 --num_steps 50 --num_minibatches 5 --update_epochs 200 --eval_freq 1 --total_timesteps 50000 --exp_name awr_beta01
python -u -m RL.mc_finetune_awr_parallel --mc_samples 16 --awr_beta 0.3 --num_envs 100 --num_steps 50 --num_minibatches 5 --update_epochs 200 --eval_freq 1 --total_timesteps 50000 --exp_name awr_beta03
python -u -m RL.mc_finetune_awr_parallel --mc_samples 16 --awr_beta 0.5 --num_envs 100 --num_steps 50 --num_minibatches 5 --update_epochs 200 --eval_freq 1 --total_timesteps 50000 --exp_name awr_beta05
# beta=1.0 is the MC16 AWR run above (mc5_awr__seed1__1771394434)
python -u -m RL.mc_finetune_awr_parallel --mc_samples 16 --awr_beta 2.0 --num_envs 100 --num_steps 50 --num_minibatches 5 --update_epochs 200 --eval_freq 1 --total_timesteps 50000 --exp_name awr_beta20

# MC samples sweep (beta=0.5, gamma=0.8)
python -u -m RL.mc_finetune_awr_parallel --mc_samples 1 --awr_beta 0.5 --num_envs 100 --num_steps 50 --num_minibatches 5 --update_epochs 200 --eval_freq 1 --total_timesteps 50000 --exp_name awr_mc1_b05
python -u -m RL.mc_finetune_awr_parallel --mc_samples 5 --awr_beta 0.5 --num_envs 100 --num_steps 50 --num_minibatches 5 --update_epochs 200 --eval_freq 1 --total_timesteps 50000 --exp_name awr_mc5_b05
# mc16 is the beta=0.5 run above (awr_beta05__seed1__1771401922)

# Gamma sweep (mc16, beta=0.5)
# gamma=0.8 is the beta=0.5 run above (awr_beta05__seed1__1771401922)
python -u -m RL.mc_finetune_awr_parallel --mc_samples 16 --awr_beta 0.5 --gamma 0.9 --num_envs 100 --num_steps 50 --num_minibatches 5 --update_epochs 200 --eval_freq 1 --total_timesteps 50000 --exp_name awr_gamma09
python -u -m RL.mc_finetune_awr_parallel --mc_samples 16 --awr_beta 0.5 --gamma 0.95 --num_envs 100 --num_steps 50 --num_minibatches 5 --update_epochs 200 --eval_freq 1 --total_timesteps 50000 --exp_name awr_gamma095
python -u -m RL.mc_finetune_awr_parallel --mc_samples 16 --awr_beta 0.5 --gamma 0.99 --num_envs 100 --num_steps 50 --num_minibatches 5 --update_epochs 200 --eval_freq 1 --total_timesteps 50000 --exp_name awr_gamma099
```

### Per-Iteration Results: Beta Sweep (mc16, gamma=0.8)

#### beta=0.1 — `runs/awr_beta01__seed1__1771398374`

| Iter | Step | SR |
|------|------|----|
| 1 | 0 | 62.7% |
| 2 | 5,000 | 73.2% |
| 3 | 10,000 | 85.6% |
| 4 | 15,000 | 92.0% |
| 5 | 20,000 | 95.0% |
| 6 | 25,000 | 92.2% |
| 7 | 30,000 | 97.0% |
| 8 | 35,000 | **97.5%** |
| 9 | 40,000 | 95.4% |
| 10 | 45,000 | 96.8% |

Peak: **97.5%** (iter 8) | Final: 96.8% (iter 10)

#### beta=0.3 — `runs/awr_beta03__seed1__1771400010`

| Iter | Step | SR |
|------|------|----|
| 1 | 0 | 62.7% |
| 2 | 5,000 | 81.4% |
| 3 | 10,000 | 92.0% |
| 4 | 15,000 | 98.4% |
| 5 | 20,000 | 96.6% |
| 6 | 25,000 | 97.7% |
| 7 | 30,000 | 96.1% |
| 8 | 35,000 | 98.0% |
| 9 | 40,000 | **98.8%** |
| 10 | 45,000 | 98.8% |

Peak: **98.8%** (iter 9) | Final: 98.8% (iter 10)

#### beta=0.5 — `runs/awr_beta05__seed1__1771401922`

| Iter | Step | SR |
|------|------|----|
| 1 | 0 | 62.7% |
| 2 | 5,000 | 76.6% |
| 3 | 10,000 | 91.0% |
| 4 | 15,000 | 94.7% |
| 5 | 20,000 | 98.2% |
| 6 | 25,000 | 95.3% |
| 7 | 30,000 | 97.9% |
| 8 | 35,000 | **99.2%** |
| 9 | 40,000 | 96.8% |
| 10 | 45,000 | 97.7% |

Peak: **99.2%** (iter 8) | Final: 97.7% (iter 10)

#### beta=1.0 — `runs/mc5_awr__seed1__1771394434`

(See [MC16 AWR Q-V Parallel Re-rollout] section above for full per-iteration results.)

Peak: **98.8%** (iter 10) | Final: 98.8% (iter 10)

#### beta=2.0 — `runs/awr_beta20__seed1__1771403702`

| Iter | Step | SR |
|------|------|----|
| 1 | 0 | 62.7% |
| 2 | 5,000 | 71.5% |
| 3 | 10,000 | 75.9% |
| 4 | 15,000 | 82.8% |
| 5 | 20,000 | 90.5% |
| 6 | 25,000 | 94.4% |
| 7 | 30,000 | 96.7% |
| 8 | 35,000 | **98.0%** |
| 9 | 40,000 | 97.1% |
| 10 | 45,000 | 96.5% |

Peak: **98.0%** (iter 8) | Final: 96.5% (iter 10)

### Summary: Beta Sweep

| beta | Peak SR | Final SR |
|------|---------|----------|
| 0.1 | 97.5% | 96.8% |
| 0.3 | 98.8% | 98.8% |
| **0.5** | **99.2%** | 97.7% |
| 1.0 | 98.8% | 98.8% |
| 2.0 | 98.0% | 96.5% |

**Best beta = 0.5** (99.2% peak). Sweet spot between greedy (0.1) and uniform (2.0).

### Per-Iteration Results: MC Samples Sweep (beta=0.5, gamma=0.8)

#### mc_samples=1 — `runs/awr_mc1_b05__seed1__1771405335`

| Iter | Step | SR |
|------|------|----|
| 1 | 0 | 62.7% |
| 2 | 5,000 | 64.7% |
| 3 | 10,000 | 80.0% |
| 4 | 15,000 | 87.6% |
| 5 | 20,000 | 90.9% |
| 6 | 25,000 | 91.7% |
| 7 | 30,000 | **97.5%** |
| 8 | 35,000 | 92.6% |
| 9 | 40,000 | 94.3% |
| 10 | 45,000 | 94.9% |

Peak: **97.5%** (iter 7) | Final: 94.9% (iter 10)

#### mc_samples=5 — `runs/awr_mc5_b05__seed1__1771406160`

| Iter | Step | SR |
|------|------|----|
| 1 | 0 | 62.7% |
| 2 | 5,000 | 70.5% |
| 3 | 10,000 | 87.4% |
| 4 | 15,000 | 95.1% |
| 5 | 20,000 | 97.5% |
| 6 | 25,000 | 94.4% |
| 7 | 30,000 | 97.9% |
| 8 | 35,000 | **98.0%** |
| 9 | 40,000 | 97.5% |
| 10 | 45,000 | 95.2% |

Peak: **98.0%** (iter 8) | Final: 95.2% (iter 10)

#### mc_samples=16 — `runs/awr_beta05__seed1__1771401922`

(Same run as beta=0.5 above.)

Peak: **99.2%** (iter 8) | Final: 97.7% (iter 10)

### Summary: MC Samples Sweep

| mc_samples | Peak SR | Final SR |
|------------|---------|----------|
| 1 | 97.5% | 94.9% |
| 5 | 98.0% | 95.2% |
| **16** | **99.2%** | 97.7% |

**Best mc_samples = 16**. More samples = better advantage estimation = higher peak.

### Per-Iteration Results: Gamma Sweep (mc16, beta=0.5)

#### gamma=0.8 — `runs/awr_beta05__seed1__1771401922`

(Same run as beta=0.5 above.)

Peak: **99.2%** (iter 8) | Final: 97.7% (iter 10)

#### gamma=0.9 — `runs/awr_gamma09__seed1__1771409475`

| Iter | Step | SR |
|------|------|----|
| 1 | 0 | 62.7% |
| 2 | 5,000 | 75.2% |
| 3 | 10,000 | 93.7% |
| 4 | 15,000 | 98.9% |
| 5 | 20,000 | **99.05%** |
| 6 | 25,000 | 96.8% |
| 7 | 30,000 | 98.8% |
| 8 | 35,000 | 97.5% |
| 9 | 40,000 | 97.1% |
| 10 | 45,000 | 98.8% |

Peak: **99.05%** (iter 5) | Final: 98.8% (iter 10)

#### gamma=0.95 — `runs/awr_gamma095__seed1__1771410181`

| Iter | Step | SR |
|------|------|----|
| 1 | 0 | 62.7% |
| 2 | 5,000 | 77.8% |
| 3 | 10,000 | 87.7% |
| 4 | 15,000 | 97.7% |
| 5 | 20,000 | 96.7% |
| 6 | 25,000 | 97.0% |
| 7 | 30,000 | 97.1% |
| 8 | 35,000 | **97.95%** |
| 9 | 40,000 | 97.3% |
| 10 | 45,000 | 97.1% |

Peak: **97.95%** (iter 8) | Final: 97.1% (iter 10)

#### gamma=0.99 — `runs/awr_gamma099__seed1__1771412428`

| Iter | Step | SR |
|------|------|----|
| 1 | 0 | 62.7% |
| 2 | 5,000 | 77.1% |
| 3 | 10,000 | 91.8% |
| 4 | 15,000 | 91.6% |
| 5 | 20,000 | 96.3% |
| 6 | 25,000 | 96.5% |
| 7 | 30,000 | 95.6% |
| 8 | 35,000 | 94.7% |
| 9 | 40,000 | 96.5% |
| 10 | 45,000 | **97.5%** |

Peak: **97.5%** (iter 10) | Final: 97.5% (iter 10)

### Summary: Gamma Sweep

| gamma | Peak SR | Final SR |
|-------|---------|----------|
| **0.8** | **99.2%** | 97.7% |
| 0.9 | 99.05% | 98.8% |
| 0.95 | 97.95% | 97.1% |
| 0.99 | 97.5% | 97.5% |

**Best gamma = 0.8** (highest peak 99.2%). Gamma=0.9 close second with best stability. Higher gamma hurts — for 50-step episodes, gamma=0.8 is sufficient discount.

### Best Config
| Parameter | Value |
|-----------|-------|
| mc_samples | 16 |
| awr_beta | 0.5 |
| gamma | 0.8 |
| update_epochs | 200 |
| **Peak SR** | **99.2%** |

### Notes
- AWR is robust across beta range (97.5%–99.2% for beta 0.1–2.0)
- mc_samples matters: mc1 (97.5%) → mc16 (99.2%), ~2% gap
- gamma=0.8 optimal for 50-step episodes; gamma=0.9 close but 0.95+ degrades
- All experiments ran serially (GPU contention with parallel runs)

---

## [AWR Offline Baseline (Fixed Dataset)] - 2026-02-17 24:10

**Git**: d53c0cc (main)

### Overview
Offline AWR baseline: collect one batch of rollouts with initial policy, compute advantages via one-time MC re-rollout, then train on fixed data for 100 iterations. No iterative re-rollout. Tests marginal benefit of online iterative training.

### Shared Settings
| Parameter | Value |
|-----------|-------|
| checkpoint | runs/pickcube_ppo/ckpt_101.pt |
| optimal_checkpoint | runs/pickcube_ppo/ckpt_301.pt |
| num_envs | 100 |
| num_steps | 50 |
| batch_size | 5,000 (fixed) |
| num_minibatches | 32 |
| update_epochs | 4 |
| gamma | 0.8 |
| num_iterations | 100 |
| eval_freq | 5 |
| training | actor-only AWR on fixed dataset |

### Commands
```bash
python -u -m RL.mc_finetune_awr_offline --mc_samples 5 --awr_beta 1.0 --num_envs 100 --num_steps 50 --num_minibatches 32 --update_epochs 4 --gamma 0.8 --num_iterations 100 --eval_freq 5 --exp_name offline_mc5_b10

python -u -m RL.mc_finetune_awr_offline --mc_samples 16 --awr_beta 0.5 --num_envs 100 --num_steps 50 --num_minibatches 32 --update_epochs 4 --gamma 0.8 --num_iterations 100 --eval_freq 5 --exp_name offline_mc16_b05
```

### Per-Iteration Results

#### Offline mc5 beta=1.0 — `runs/offline_mc5_b10__seed1__1771414074`

| Iter | SR |
|------|----|
| 1 | 59.4% |
| 5 | 56.1% |
| 10 | 62.2% |
| 15 | 70.4% |
| 20 | 66.9% |
| 25 | 72.3% |
| 30 | 69.7% |
| 35 | 66.4% |
| 40 | 65.0% |
| 45 | 65.2% |
| 50 | 76.3% |
| 55 | 70.4% |
| 60 | 75.4% |
| 65 | 81.2% |
| 70 | 71.9% |
| 75 | 72.4% |
| 80 | 77.6% |
| 85 | 73.0% |
| 90 | **85.3%** |
| 95 | 73.4% |
| 100 | 77.5% |

Peak: **85.3%** (iter 90) | Final: 77.5% (iter 100)

#### Offline mc16 beta=0.5 — `runs/offline_mc16_b05__seed1__1771414402`

| Iter | SR |
|------|----|
| 1 | 60.9% |
| 5 | 45.5% |
| 10 | 63.7% |
| 15 | 71.1% |
| 20 | 71.6% |
| 25 | 77.0% |
| 30 | 71.9% |
| 35 | 70.9% |
| 40 | 61.2% |
| 45 | 79.0% |
| 50 | 69.4% |
| 55 | 79.3% |
| 60 | 75.0% |
| 65 | 72.5% |
| 70 | 75.4% |
| 75 | **81.0%** |
| 80 | 76.4% |
| 85 | 75.0% |
| 90 | 77.1% |
| 95 | 75.0% |
| 100 | 79.7% |

Peak: **81.0%** (iter 75) | Final: 79.7% (iter 100)

### Summary: Online vs Offline AWR

| Method | mc_samples | beta | Peak SR | Final SR |
|--------|------------|------|---------|----------|
| **Online AWR** (iterative) | 16 | 0.5 | **99.2%** | 97.7% |
| **Online AWR** (iterative) | 16 | 1.0 | **98.8%** | 98.8% |
| Offline AWR (fixed data) | 5 | 1.0 | 85.3% | 77.5% |
| Offline AWR (fixed data) | 16 | 0.5 | 81.0% | 79.7% |

### Notes
- **Online >> Offline**: 99.2% vs 85.3% — iterative re-rollout provides ~14% improvement
- Offline performance is very noisy (oscillates 10%+ between evals) and plateaus quickly
- Offline mc16 beta=0.5 is actually slightly worse than mc5 beta=1.0 — more greedy weighting on suboptimal fixed data hurts
- Key insight: the iterative aspect (re-computing Q-V advantages with the updated policy's trajectory distribution) is critical, not just the AWR update rule

---

## [On-Policy MC AWR (No Optimal Policy)] - 2026-02-17 24:30

**Git**: d53c0cc (main)

### Overview
Same as `mc_finetune_awr_parallel` but MC re-rollouts use the **current policy** (agent) instead of the **optimal policy** (ckpt_301). Tests whether access to an optimal policy for Q-V estimation is necessary.

Advantage = Q^π(s,a) - V^π(s) where π is the current policy.

### Shared Settings
| Parameter | Value |
|-----------|-------|
| checkpoint | runs/pickcube_ppo/ckpt_101.pt |
| num_envs | 100 |
| num_steps | 50 |
| num_minibatches | 5 |
| update_epochs | 200 |
| awr_beta | 0.5 |
| awr_max_weight | 20.0 |
| gamma | 0.8 |
| total_timesteps | 50,000 |
| training | actor-only AWR (on-policy re-rollout) |

### Commands
```bash
python -u -m RL.mc_finetune_awr_onpolicy --mc_samples 16 --awr_beta 0.5 --num_envs 100 --num_steps 50 --num_minibatches 5 --update_epochs 200 --eval_freq 1 --total_timesteps 50000 --exp_name onpolicy_mc16_b05

python -u -m RL.mc_finetune_awr_onpolicy --mc_samples 5 --awr_beta 0.5 --num_envs 100 --num_steps 50 --num_minibatches 5 --update_epochs 200 --eval_freq 1 --total_timesteps 50000 --exp_name onpolicy_mc5_b05

python -u -m RL.mc_finetune_awr_onpolicy --mc_samples 1 --awr_beta 0.5 --num_envs 100 --num_steps 50 --num_minibatches 5 --update_epochs 200 --eval_freq 1 --total_timesteps 50000 --exp_name onpolicy_mc1_b05
```

### Per-Iteration Results

#### On-policy mc16 beta=0.5 — `runs/onpolicy_mc16_b05__seed1__1771414815`

| Iter | Step | SR |
|------|------|----|
| 1 | 0 | 62.7% |
| 2 | 5,000 | 77.9% |
| 3 | 10,000 | 87.1% |
| 4 | 15,000 | 90.4% |
| 5 | 20,000 | 88.3% |
| 6 | 25,000 | 90.2% |
| 7 | 30,000 | 93.2% |
| 8 | 35,000 | **94.9%** |
| 9 | 40,000 | 94.6% |
| 10 | 45,000 | 94.6% |

Peak: **94.9%** (iter 8) | Final: 94.6% (iter 10)

#### On-policy mc5 beta=0.5 — `runs/onpolicy_mc5_b05__seed1__1771416384`

| Iter | Step | SR |
|------|------|----|
| 1 | 0 | 62.7% |
| 2 | 5,000 | 64.0% |
| 3 | 10,000 | 78.6% |
| 4 | 15,000 | 86.6% |
| 5 | 20,000 | 93.1% |
| 6 | 25,000 | 93.8% |
| 7 | 30,000 | 93.5% |
| 8 | 35,000 | 93.6% |
| 9 | 40,000 | 93.2% |
| 10 | 45,000 | **95.1%** |

Peak: **95.1%** (iter 10) | Final: 95.1% (iter 10)

#### On-policy mc1 beta=0.5 — `runs/onpolicy_mc1_b05__seed1__1771430209`

| Iter | Step | SR |
|------|------|----|
| 1 | 0 | 62.7% |
| 2 | 5,000 | 68.7% |
| 3 | 10,000 | 72.7% |
| 4 | 15,000 | 88.4% |
| 5 | 20,000 | 87.9% |
| 6 | 25,000 | 88.3% |
| 7 | 30,000 | 89.2% |
| 8 | 35,000 | 93.7% |
| 9 | 40,000 | **95.5%** |
| 10 | 45,000 | 94.5% |

Peak: **95.5%** (iter 9) | Final: 94.5% (iter 10)

### Summary: Optimal vs On-Policy MC

| Re-rollout Policy | mc_samples | Peak SR | Final SR |
|-------------------|------------|---------|----------|
| **Optimal (π*)** | 16 | **99.2%** | 97.7% |
| **Optimal (π*)** | 5 | 98.0% | 95.2% |
| On-policy (π) | 1 | 95.5% | 94.5% |
| On-policy (π) | 5 | 95.1% | 95.1% |
| On-policy (π) | 16 | 94.9% | 94.6% |

### Notes
- **Optimal policy re-rollout is ~4% better** than on-policy (99.2% vs 95.5%)
- On-policy mc1 ≈ mc5 ≈ mc16 (~95%) — increasing MC samples doesn't help since the current policy is suboptimal, adding more samples of a weak policy doesn't improve advantage estimation quality
- On-policy mc1 (95.5%) is a **direct comparison to GAE PPO (92.0%)**: both use on-policy data with 1 sample, but AWR tolerates many more update epochs (200 vs 4), explaining the +3.5% gap
- Optimal policy enables more accurate Q-V estimation → better advantage signal → faster convergence
- On-policy AWR (~95%) is still better than standard GAE PPO (92.0%) and comparable to MC5 PPO (95.8%)

---

## [IQL + AWR Offline Finetuning] - 2026-02-18 09:30

**Git**: 57ef663 (main)

### Overview
Replace MC re-rollout with IQL critic for advantage estimation. Train IQL Q(s,a) and V(s) on offline data from multiple policy checkpoints (random → optimal), then use A(s,a) = Q(s,a) - V(s) for AWR updates on fixed finetune data. Fully offline — no optimal policy or MC re-rollout needed.

Script: `RL/iql_awr_offline.py` (new file)

### Shared Settings
| Parameter | Value |
|-----------|-------|
| checkpoint | runs/pickcube_ppo/ckpt_101.pt |
| iql_data_checkpoints | ckpt_1, ckpt_51, ckpt_101, ckpt_201, ckpt_301 |
| reward | sparse |
| gamma | 0.8 |
| num_envs | 512 |
| num_steps | 50 |
| batch_size | 25,600 (fixed finetune data) |
| num_minibatches | 32 |
| update_epochs | 4 |
| awr_max_weight | 20.0 |
| norm_adv | True |
| eval_freq | 5 |
| training | actor-only AWR on fixed dataset, IQL advantage |

### Commands
```bash
# Tau sweep (beta=1.0, 100 iterations, 200 episodes/ckpt, nstep=1)
python -u -m RL.iql_awr_offline --iql_expectile_tau 0.5 --awr_beta 1.0 --exp_name iql_tau0.5_beta1.0
python -u -m RL.iql_awr_offline --iql_expectile_tau 0.7 --awr_beta 1.0 --exp_name iql_tau0.7_beta1.0
python -u -m RL.iql_awr_offline --iql_expectile_tau 0.9 --awr_beta 1.0 --exp_name iql_tau0.9_beta1.0

# Tuned: more data, n-step, lower LR, 200 iterations
python -u -m RL.iql_awr_offline --iql_expectile_tau 0.9 --iql_episodes_per_ckpt 500 --iql_nstep 5 --iql_epochs 300 --iql_patience 80 --awr_beta 0.5 --learning_rate 1e-4 --num_iterations 200 --exp_name iql_tuned
```

### IQL Data & Advantage Statistics

| Experiment | Transitions | Positive Rewards | A mean | A std | pos% | AWR wt mean |
|---|---|---|---|---|---|---|
| tau0.5 (200 ep/ckpt) | 85,504 | 631 (0.7%) | +0.0101 | 0.0208 | 64.6% | 1.78 |
| tau0.7 (200 ep/ckpt) | 85,504 | 631 (0.7%) | -0.0033 | 0.0178 | 44.2% | 1.62 |
| tau0.9 (200 ep/ckpt) | 85,504 | 631 (0.7%) | -0.0188 | 0.0217 | 17.2% | 1.54 |
| tuned tau0.9 n5 (500 ep/ckpt) | 102,912 | 1,358 (1.3%) | -0.0590 | 0.0782 | 19.4% | 2.82 |

### Results: Tau Sweep (beta=1.0, 100 iterations)

| Experiment | Iter 1 | Iter 10 | Iter 20 | Iter 30 | Iter 50 | Iter 70 | Iter 100 | Peak | Final |
|---|---|---|---|---|---|---|---|---|---|
| tau0.5 beta1.0 | 61.2% | 56.5% | 65.2% | 68.6% | 67.2% | 61.9% | 75.4% | 75.6% (i35) | 75.4% |
| tau0.7 beta1.0 | 61.2% | 56.4% | 63.4% | 63.8% | 76.4% | 58.5% | 73.2% | 76.4% (i50) | 73.2% |
| tau0.9 beta1.0 | 61.2% | 72.9% | 63.0% | 68.1% | 66.2% | 62.7% | 74.8% | 74.8% (i100) | 74.8% |

### Results: Tuned (tau0.9, nstep=5, 500 ep/ckpt, beta=0.5, lr=1e-4, 200 iterations)

| Iter | SR |
|------|----|
| 1 | 59.4% |
| 5 | 73.9% |
| 10 | 74.3% |
| 25 | 59.7% |
| 50 | 65.5% |
| 75 | 73.0% |
| 95 | 76.2% |
| 100 | 68.5% |
| 150 | 70.7% |
| 200 | 70.8% |

Peak: **76.2%** (iter 95) | Final: 70.8% (iter 200)

### Summary: IQL AWR vs MC AWR

| Method | Advantage Source | Peak SR | Final SR |
|--------|----------------|---------|----------|
| Online MC16 AWR (iterative) | Q^π* MC re-rollout | **99.2%** | 97.7% |
| Offline MC5 AWR (fixed data) | Q^π* MC re-rollout | 85.3% | 77.5% |
| **IQL AWR (best)** | IQL Q-V (offline) | **76.2%** | 70.8% |
| Baseline (no finetuning) | — | 62.7% | 62.7% |

### Run Dirs
- `runs/iql_tau0.5_beta1.0__seed1__1771433737`
- `runs/iql_tau0.7_beta1.0__seed1__1771434032`
- `runs/iql_tau0.9_beta1.0__seed1__1771434327`
- `runs/iql_tuned__seed1__1771435076`

### Notes
- **IQL AWR plateaus at ~76%** — significantly below MC re-rollout offline (85.3%) and online (99.2%)
- All configurations oscillate heavily (55%~76%) without stable convergence
- Root cause: **sparse reward + IQL = weak signal** — only 0.7%–1.3% of transitions have positive reward, IQL can't learn discriminative Q values from so few reward signals
- Advantage std is tiny (0.02–0.08) compared to MC re-rollout — the IQL Q-V differences are not meaningful enough to guide AWR
- Tau has minimal effect on final SR (74.8%–76.4% peak), despite affecting advantage distribution (pos% ranges 17%–65%)
- Tuned setting (more data, n-step=5, lower LR, 200 iters) did not improve over simple tau sweep — the bottleneck is IQL quality, not AWR tuning
- **Conclusion**: IQL advantage estimation on sparse reward is fundamentally limited. MC re-rollout with optimal policy provides ground-truth returns that IQL cannot match. IQL AWR might work better with dense reward or much larger/diverse offline datasets.

---

## [IQL AWR: Single-Checkpoint Data Ablation (ckpt_101 only)] - 2026-02-18 09:38

**Command**: `python -u -m RL.iql_awr_offline --iql_data_checkpoints "runs/pickcube_ppo/ckpt_101.pt" --iql_episodes_per_ckpt 1000 --iql_expectile_tau 0.7 --awr_beta 1.0 --num_iterations 100 --exp_name iql_ckpt101only`
**Git**: 57ef663 (main)
**Run Dir**: `runs/iql_ckpt101only__seed1__1771435961`

### Settings
| Parameter | Value |
|-----------|-------|
| checkpoint | runs/pickcube_ppo/ckpt_101.pt |
| iql_data_checkpoints | ckpt_101 only (1 checkpoint) |
| iql_episodes_per_ckpt | 1000 |
| iql_expectile_tau | 0.7 |
| iql_epochs | 200 |
| iql_nstep | 1 |
| iql_patience | 50 |
| awr_beta | 1.0 |
| awr_max_weight | 20.0 |
| gamma | 0.8 |
| num_envs | 512 |
| num_steps | 50 |
| batch_size | 25,600 |
| num_minibatches | 32 |
| update_epochs | 4 |
| norm_adv | True |
| num_iterations | 100 |

### IQL Data Statistics
| Metric | Value |
|--------|-------|
| Transitions | 51,200 |
| Trajectories | 1,509 |
| Positive rewards | 662 (1.3%) |
| Episodes collected | 1,090 |
| Traj length (mean) | 33.9 |
| IQL early stop | epoch 68 |
| Q(s,a) mean/std | 0.0774 / 0.0575 |
| V(s) mean/std | 0.0878 / 0.0605 |
| A(s,a) mean/std | -0.0104 / 0.0168 |
| Advantage pos% | 28.5% |
| AWR weight mean | 1.58 |

### Results
| Iter | SR |
|------|----|
| 1 | 57.3% |
| 5 | 74.5% |
| 10 | 77.1% |
| 20 | 77.8% |
| 25 | **83.2%** |
| 30 | 69.8% |
| 40 | 79.9% |
| 50 | 80.9% |
| 55 | 81.0% |
| 70 | 80.7% |
| 100 | 79.7% |

| Metric | Value |
|--------|-------|
| Peak SR | **83.2%** (iter 25) |
| Final SR | 79.7% (iter 100) |

### Comparison: IQL Data Source Ablation

| Data Source | Transitions | Pos Rewards | Peak SR | Final SR |
|---|---|---|---|---|
| 5 ckpts mixed (tau0.7 beta1.0) | 85,504 | 631 (0.7%) | 76.4% | 73.2% |
| **ckpt_101 only** (tau0.7 beta1.0) | 51,200 | 662 (1.3%) | **83.2%** | 79.7% |
| MC5 offline (ground truth adv) | 5,000 | — | 85.3% | 77.5% |

### Notes
- **Single-checkpoint data (83.2%) beats mixed data (76.4%) by ~7%** — on-distribution data is more valuable than diverse but off-distribution data
- Positive reward fraction doubled (1.3% vs 0.7%) despite fewer total transitions (51k vs 85k) — the random/bad checkpoints in mixed data diluted the reward signal
- **83.2% approaches MC5 offline baseline (85.3%)** — IQL advantage quality is competitive with MC re-rollout when trained on on-distribution data
- Still oscillates (70%~83%) — the fixed-dataset AWR limitation remains regardless of IQL data quality
- Key insight: for IQL, **data distribution matters more than data quantity** — fewer but on-policy transitions give better Q/V estimates than more off-policy transitions

---

## [Fair Comparison: MC16 vs IQL, Both Using Only ckpt_101] - 2026-02-18 10:09

**Git**: 57ef663 (main)

### Overview
Fair one-step offline RL comparison: both methods use only ckpt_101 for everything (data collection, advantage estimation). Same finetune batch (25,600), same AWR hyperparameters. Only difference is advantage source: MC16 re-rollout vs IQL Q-V.

### Commands
```bash
# MC16 re-rollout with ckpt_101 as rollout policy (not optimal)
python -u -m RL.mc_finetune_awr_offline --optimal_checkpoint runs/pickcube_ppo/ckpt_101.pt --mc_samples 16 --num_envs 128 --num_steps 200 --awr_beta 1.0 --num_iterations 100 --exp_name mc16_ckpt101_offline

# IQL trained on ckpt_101 data (from previous experiment)
python -u -m RL.iql_awr_offline --iql_data_checkpoints "runs/pickcube_ppo/ckpt_101.pt" --iql_episodes_per_ckpt 1000 --iql_expectile_tau 0.7 --awr_beta 1.0 --num_iterations 100 --exp_name iql_ckpt101only
```

### Settings Comparison
| Parameter | MC16 ckpt_101 | IQL ckpt_101 |
|-----------|---------------|--------------|
| checkpoint | ckpt_101 | ckpt_101 |
| advantage source | MC16 re-rollout (ckpt_101) | IQL Q(s,a)-V(s) |
| num_envs | 128 | 512 |
| num_steps | 200 | 50 |
| batch_size | 25,600 | 25,600 |
| mc_samples | 16 | — |
| num_mc_envs | 4,096 | — |
| iql_episodes_per_ckpt | — | 1,000 |
| iql_expectile_tau | — | 0.7 |
| IQL training data | — | 51,200 transitions |
| awr_beta | 1.0 | 1.0 |
| num_minibatches | 32 | 32 |
| update_epochs | 4 | 4 |
| gamma | 0.8 | 0.8 |
| num_iterations | 100 | 100 |

### Advantage Statistics
| | MC16 ckpt_101 | IQL ckpt_101 |
|---|---|---|
| A mean | -0.0002 | -0.0104 |
| A std | **0.1113** | 0.0170 |
| Overhead | MC re-rollout 669s | IQL training 35s |

### Results
| Iter | MC16 ckpt_101 | IQL ckpt_101 |
|------|---------------|--------------|
| 1 | 66.2% | 57.3% |
| 5 | 73.2% | 74.5% |
| 10 | 78.1% | 77.1% |
| 15 | 79.2% | 72.3% |
| 20 | 79.9% | 77.8% |
| 25 | 73.3% | **83.2%** |
| 30 | 83.7% | 69.8% |
| 50 | 80.9% | 80.9% |
| 80 | 82.4% | 71.5% |
| 95 | **89.1%** | 78.0% |
| 100 | 77.2% | 79.7% |

| Metric | MC16 ckpt_101 | IQL ckpt_101 |
|--------|---------------|--------------|
| Peak SR | **89.1%** (iter 95) | **83.2%** (iter 25) |
| Final SR | 77.2% | 79.7% |

### Run Dirs
- `runs/mc16_ckpt101_offline__seed1__1771437172` (MC16)
- `runs/iql_ckpt101only__seed1__1771435961` (IQL)

### Notes
- **MC16 peak is ~6% higher** (89.1% vs 83.2%) — MC re-rollout provides higher-quality advantage signal (A std = 0.111 vs 0.017, 6.5× stronger signal)
- **IQL is 19× faster** (35s vs 669s for advantage estimation) — no need to create and run thousands of MC envs
- **Both oscillate heavily** in the 70%–89% range — the fixed-dataset AWR bottleneck dominates regardless of advantage quality
- MC16's advantage std (0.111) is 6.5× larger than IQL's (0.017) — IQL struggles to discriminate good vs bad actions with sparse reward; MC directly observes actual returns
- Despite weaker signal, IQL still achieves 83% peak — the advantage *direction* is approximately correct, just the magnitude is compressed
- **Trade-off**: MC16 gives better peak but costs 19× more compute. In settings without simulator access for re-rollout, IQL is the only option

---

## [MC16 Optimal vs On-Policy vs IQL: Advantage Bias in Offline AWR] - 2026-02-18 10:30

**Git**: 57ef663 (main)

### Overview
Test whether using the **optimal policy** (ckpt_301) for MC re-rollout in offline AWR is actually better than using the **on-policy** (ckpt_101) re-rollout. Hypothesis: optimal policy should give better Q-V estimates. Reality: optimal policy creates strongly negative advantage bias on ckpt_101 data.

This experiment uses the same batch size (25,600) as the MC16 ckpt_101 run for a fair comparison.

### Command
```bash
python -u -m RL.mc_finetune_awr_offline --optimal_checkpoint runs/pickcube_ppo/ckpt_301.pt --mc_samples 16 --num_envs 128 --num_steps 200 --awr_beta 1.0 --num_iterations 100 --exp_name mc16_optimal_offline_b25k
```

### Settings
| Parameter | Value |
|-----------|-------|
| checkpoint | runs/pickcube_ppo/ckpt_101.pt |
| optimal_checkpoint | runs/pickcube_ppo/ckpt_301.pt |
| mc_samples | 16 |
| num_envs | 128 |
| num_steps | 200 |
| batch_size | 25,600 |
| num_mc_envs | 4,096 |
| num_minibatches | 32 |
| update_epochs | 4 |
| awr_beta | 1.0 |
| gamma | 0.8 |
| num_iterations | 100 |
| eval_freq | 5 |

### Advantage Statistics
| Metric | MC16 optimal (π*) | MC16 on-policy (π_101) | IQL (ckpt_101) |
|--------|-------------------|------------------------|-----------------|
| A mean | **-0.0799** | -0.0002 | -0.0104 |
| A std | 0.1088 | 0.1113 | 0.0170 |
| A pos% | 22.6% | ~50% | 28.5% |

### Results
| Iter | SR |
|------|----|
| 1 | 59.7% |
| 5 | 68.6% |
| 10 | 63.4% |
| 15 | 67.7% |
| 20 | 66.4% |
| 25 | 69.4% |
| 30 | 73.9% |
| 35 | 75.7% |
| 40 | 72.7% |
| 50 | 67.6% |
| 60 | 75.5% |
| 75 | 77.5% |
| 80 | 71.7% |
| 95 | **83.9%** |
| 100 | 72.9% |

Peak: **83.9%** (iter 95) | Final: 72.9% (iter 100)

### Three-Way Comparison (all offline AWR, batch=25,600)

| Method | Advantage Source | Peak SR | Final SR | A mean |
|--------|----------------|---------|----------|--------|
| MC16 on-policy (ckpt_101) | Q^π_101 - V^π_101 | **89.1%** | 77.2% | -0.0002 |
| MC16 optimal (ckpt_301) | Q^π* - V^π* | 83.9% | 72.9% | -0.0799 |
| IQL (ckpt_101) | Q_IQL - V_IQL | 83.2% | 79.7% | -0.0104 |

### Run Dir
- `runs/mc16_optimal_offline_b25k__seed1__1771438320`

### Notes
- **Optimal MC re-rollout is WORST** in offline setting — 83.9% vs 89.1% for on-policy. Counterintuitive result.
- **Root cause: advantage bias**. Q^π*(s, a_ckpt101) - V^π*(s) is strongly negative (mean=-0.08) because ckpt_101's actions are suboptimal from π*'s perspective. V^π*(s) >> Q^π*(s, a_ckpt101) for most state-action pairs.
- Only 22.6% of advantages are positive → AWR weights are heavily biased toward "least bad" actions rather than truly good ones.
- On-policy MC (ckpt_101 re-rollout) has near-zero mean advantage (-0.0002) because the collected actions *are* from the re-rollout policy, so Q^π ≈ V^π on average.
- **Key insight**: In offline AWR, the re-rollout policy should match the data-collection policy. Using a stronger policy for re-rollout introduces systematic negative bias that hurts learning.
- This explains why online iterative AWR works so well (99.2%): the advantage is recomputed each iteration with fresh on-policy data, keeping the bias near zero.
- IQL (83.2%) ≈ MC16 optimal (83.9%) — both suffer from bias (negative advantage mean), but for different reasons: IQL from sparse reward signal, MC optimal from distribution mismatch.

---

---

## ⚠️ Checkpoint Change Notice

**All experiments above used `ckpt_101` (deterministic SR ~99%, stochastic SR ~62.7%) as finetune starting point.**

Problem: ckpt_101's deterministic SR is too high (~99%), meaning the mean action is already near-optimal. The low stochastic SR (62.7%) is purely due to exploration noise, not poor policy quality. This causes:
1. AWR/PPO improvements mainly come from reducing action noise, not learning better actions
2. High starting point masks true differences between methods
3. Conclusions may not generalize to weaker starting policies

**All subsequent experiments will use a weaker checkpoint (deterministic ~46%, stochastic ~51%, log_std=-1.5) and re-run key experiments.**

---

## [V2: Online AWR & GAE — Deterministic Eval, Weak Checkpoint] - 2026-02-18 16:00

### Overview
Re-run key online finetuning experiments with two critical fixes:
1. **Weaker starting checkpoint**: `ckpt_76_logstd-1.5` (det SR=43.8%, stoch SR≈51%) instead of ckpt_101 (det SR=99%)
2. **Deterministic evaluation**: `deterministic=True` in eval loop, eliminating ~12% eval noise from action sampling

Tests: optimal vs on-policy AWR, MC16 vs MC1, GAE PPO, and gamma sweep (0.8/0.95/0.99).

### Shared Settings
| Parameter | Value |
|-----------|-------|
| checkpoint | runs/pickcube_ppo/ckpt_76_logstd-1.5.pt |
| initial det SR | 43.8% |
| num_envs | 100 |
| num_eval_envs | 128 |
| num_steps | 50 |
| batch_size | 5,000 |
| num_minibatches | 5 |
| eval deterministic | **True** |
| total_timesteps | 50,000 |
| AWR: update_epochs | 200 |
| AWR: awr_beta | 0.5 |
| GAE: update_epochs | 100 |
| GAE: target_kl | 100.0 (disabled) |

### Commands
```bash
# All 9 experiments run via run_v2_all.sh
CKPT="runs/pickcube_ppo/ckpt_76_logstd-1.5.pt"
COMMON_AWR="--awr_beta 0.5 --num_envs 100 --num_steps 50 --num_minibatches 5 --update_epochs 200 --eval_freq 1 --total_timesteps 50000"
COMMON_GAE="--num_envs 100 --num_steps 50 --num_minibatches 5 --update_epochs 100 --target_kl 100.0 --eval_freq 1 --total_timesteps 50000"

# gamma=0.8
python -m RL.mc_finetune_awr_parallel --checkpoint $CKPT --mc_samples 16 $COMMON_AWR --exp_name v2_mc16_optimal_det
python -m RL.mc_finetune_awr_parallel --checkpoint $CKPT --mc_samples 1 $COMMON_AWR --exp_name v2_mc1_optimal_det
python -m RL.mc_finetune_awr_onpolicy --checkpoint $CKPT --mc_samples 16 $COMMON_AWR --exp_name v2_mc16_onpolicy_det
python -m RL.mc_finetune_awr_onpolicy --checkpoint $CKPT --mc_samples 1 $COMMON_AWR --exp_name v2_mc1_onpolicy_det
python -m RL.ppo_finetune --checkpoint $CKPT $COMMON_GAE --exp_name v2_gae_det

# gamma sweep
python -m RL.mc_finetune_awr_parallel --checkpoint $CKPT --mc_samples 16 --gamma 0.95 $COMMON_AWR --exp_name v2_mc16_optimal_g095_det
python -m RL.ppo_finetune --checkpoint $CKPT --gamma 0.95 $COMMON_GAE --exp_name v2_gae_g095_det
python -m RL.mc_finetune_awr_parallel --checkpoint $CKPT --mc_samples 16 --gamma 0.99 $COMMON_AWR --exp_name v2_mc16_optimal_g099_det
python -m RL.ppo_finetune --checkpoint $CKPT --gamma 0.99 $COMMON_GAE --exp_name v2_gae_g099_det
```

### Results — γ=0.8 Main Comparison

| Experiment | Method | MC | i1 | i2 | i3 | i4 | i5 | i6 | i7 | i8 | i9 | i10 | Peak | Final |
|------------|--------|-----|------|------|------|------|------|------|------|------|------|-------|------|-------|
| v2_mc16_optimal_det | optimal AWR | 16 | 43.8 | 81.0 | 87.9 | 96.1 | 98.6 | 96.3 | 97.6 | 96.9 | 97.9 | **99.1** | **99.1%** | **99.1%** |
| v2_mc16_onpolicy_det | on-policy AWR | 16 | 43.8 | 67.7 | 81.0 | 87.9 | 90.6 | 94.8 | 95.1 | 96.6 | **96.7** | 95.0 | 96.7% | 95.0% |
| v2_mc1_optimal_det | optimal AWR | 1 | 43.8 | 51.9 | 55.2 | 71.0 | 69.9 | 76.4 | 77.4 | **78.3** | 76.6 | 70.3 | 78.3% | 70.3% |
| v2_mc1_onpolicy_det | on-policy AWR | 1 | 43.8 | 47.7 | 64.1 | 69.4 | 67.6 | 69.4 | **74.6** | 66.7 | 73.0 | 69.6 | 74.6% | 69.6% |
| v2_gae_det | GAE PPO | - | 43.8 | 52.6 | 52.3 | 64.1 | 80.3 | 76.9 | 73.3 | **82.4** | 79.4 | 71.7 | 82.4% | 71.7% |

### Results — Gamma Sweep (MC16 optimal vs GAE PPO)

| gamma | Method | i1 | i2 | i3 | i4 | i5 | i6 | i7 | i8 | i9 | i10 | Peak | Final |
|-------|--------|------|------|------|------|------|------|------|------|------|-------|------|-------|
| 0.8 | MC16 optimal | 43.8 | 81.0 | 87.9 | 96.1 | 98.6 | 96.3 | 97.6 | 96.9 | 97.9 | 99.1 | **99.1%** | **99.1%** |
| 0.8 | GAE PPO | 43.8 | 52.6 | 52.3 | 64.1 | 80.3 | 76.9 | 73.3 | 82.4 | 79.4 | 71.7 | 82.4% | 71.7% |
| 0.95 | MC16 optimal | 43.8 | 71.0 | 91.0 | 96.5 | 91.5 | 94.9 | 97.1 | 97.3 | 98.7 | 97.7 | 98.7% | 97.7% |
| 0.95 | GAE PPO | 43.8 | 45.4 | 49.6 | 58.1 | 60.3 | 65.6 | 69.6 | 75.2 | 76.7 | 76.1 | 76.7% | 76.1% |
| 0.99 | MC16 optimal | 43.8 | 56.6 | 60.7 | 61.2 | 69.0 | 64.9 | 66.2 | 66.2 | 69.6 | 77.0 | 77.0% | 77.0% |
| 0.99 | GAE PPO | 43.8 | 58.0 | 66.4 | 73.3 | 72.3 | 83.7 | 85.8 | 87.6 | 92.1 | 91.2 | 92.1% | 91.2% |

### Run Dirs
- `runs/v2_mc16_optimal_det__seed1__1771449642`
- `runs/v2_mc1_optimal_det__seed1__1771451334`
- `runs/v2_mc16_onpolicy_det__seed1__1771452128`
- `runs/v2_mc1_onpolicy_det__seed1__1771453853`
- `runs/v2_gae_det__seed1__1771454881`
- `runs/v2_mc16_optimal_g095_det__seed1__1771455206`
- `runs/v2_gae_g095_det__seed1__1771456940`
- `runs/v2_mc16_optimal_g099_det__seed1__1771457068`
- `runs/v2_gae_g099_det__seed1__1771458679`

### Notes
- **MC16 >> MC1** (γ=0.8): mc16 optimal 99.1% vs mc1 optimal 78.3% (+21%). Multi-sample MC is critical for stable advantage estimation.
- **Optimal >> On-policy** (mc16): 99.1% vs 95.0% (+4%). Oracle re-rollout helps but is not essential.
- **AWR >> GAE PPO** (γ=0.8): mc16 on-policy AWR 95.0% vs GAE PPO 71.7% (+23%). AWR's 200 update epochs >> PPO's 100 epochs.
- **GAE PPO is unstable at γ=0.8**: oscillates 52-82%, final=71.7%. Low gamma + sparse reward → noisy TD targets.
- **γ=0.99 reversal**: GAE PPO (92.1%) > MC16 optimal (77.0%). High gamma + sparse reward → MC return variance explodes, AWR advantage is noisy. GAE's bootstrapping provides stability.
- **MC1 is unstable**: both mc1 optimal (78.3%) and mc1 onpolicy (74.6%) oscillate and decay. Single-sample MC variance is too high for reliable learning.
- **vs ckpt_101 experiments**: Results are now more discriminative — starting from 43.8% instead of ~63% (stochastic) reveals true method differences. The 23% gap between AWR and GAE was masked before.

---

## [V2 Offline: MC & IQL AWR — Deterministic Eval, Weak Checkpoint] - 2026-02-18 17:30

### Overview
Offline AWR baselines with weak checkpoint. One-shot data collection + fixed-dataset training. Tests: optimal vs on-policy MC re-rollout (MC16/MC1) and IQL advantage.

### Shared Settings
| Parameter | Value |
|-----------|-------|
| checkpoint | runs/pickcube_ppo/ckpt_76_logstd-1.5.pt |
| initial det SR | 43.8% |
| num_envs | 128 |
| num_steps | 200 |
| batch_size | 25,600 |
| num_minibatches | 32 |
| update_epochs | 4 |
| awr_beta | 1.0 |
| gamma | 0.8 |
| num_iterations | 100 |
| eval_freq | 5 |
| eval deterministic | **True** |

### Commands
```bash
CKPT="runs/pickcube_ppo/ckpt_76_logstd-1.5.pt"
OPTIMAL="runs/pickcube_ppo/ckpt_301.pt"
COMMON="--awr_beta 1.0 --num_envs 128 --num_steps 200 --num_minibatches 32 --update_epochs 4 --num_iterations 100 --eval_freq 5"

python -m RL.mc_finetune_awr_offline --checkpoint $CKPT --optimal_checkpoint $OPTIMAL --mc_samples 16 $COMMON --exp_name v2_offline_mc16_optimal_det
python -m RL.mc_finetune_awr_offline --checkpoint $CKPT --optimal_checkpoint $OPTIMAL --mc_samples 1 $COMMON --exp_name v2_offline_mc1_optimal_det
python -m RL.mc_finetune_awr_offline --checkpoint $CKPT --optimal_checkpoint $CKPT --mc_samples 16 $COMMON --exp_name v2_offline_mc16_onpolicy_det
python -m RL.mc_finetune_awr_offline --checkpoint $CKPT --optimal_checkpoint $CKPT --mc_samples 1 $COMMON --exp_name v2_offline_mc1_onpolicy_det
python -m RL.iql_awr_offline --checkpoint $CKPT --awr_beta 1.0 --num_envs 128 --num_steps 200 --num_minibatches 32 --update_epochs 4 --num_iterations 100 --eval_freq 5 --exp_name v2_offline_iql_det
```

### Results

| Experiment | Method | mc | i1 | i5 | i10 | i15 | i20 | i25 | i50 | i75 | i100 | Peak | Final |
|------------|--------|-----|------|------|------|------|------|------|------|------|-------|------|-------|
| v2_offline_mc16_optimal | optimal AWR | 16 | 42.3 | 60.2 | 67.2 | 71.6 | 66.4 | **88.2** | 61.8 | 80.5 | 84.3 | **88.2%** | 84.3% |
| v2_offline_mc16_onpolicy | on-policy AWR | 16 | 50.4 | 65.4 | 64.4 | 71.6 | 81.8 | 74.1 | 77.9 | 69.9 | 75.0 | 83.3% | 75.0% |
| v2_offline_mc1_optimal | optimal AWR | 1 | 45.0 | 61.1 | 53.0 | 53.5 | 53.4 | 70.1 | 63.4 | 64.7 | 60.6 | 77.7% | 60.6% |
| v2_offline_mc1_onpolicy | on-policy AWR | 1 | 53.7 | 57.7 | 63.4 | 66.7 | 66.2 | 70.7 | 72.5 | 47.7 | 72.0 | 72.5% | 72.0% |
| v2_offline_iql | IQL AWR | - | 49.6 | 82.4 | 83.7 | **91.6** | 76.3 | 78.3 | 77.0 | 83.7 | 67.2 | **91.6%** | 67.2% |

### Run Dirs
- `runs/v2_offline_mc16_optimal_det__seed1__1771461325`
- `runs/v2_offline_mc1_optimal_det__seed1__1771462319`
- `runs/v2_offline_mc16_onpolicy_det__seed1__1771462805`
- `runs/v2_offline_mc1_onpolicy_det__seed1__1771463659`
- `runs/v2_offline_iql_det__seed1__1771464158`

### Notes
- **IQL peaks highest (91.6%)** but collapses to 67.2%. Fast initial learning from diverse IQL training data (5 checkpoint mix), but unstable on fixed finetune data.
- **MC16 optimal has best final (84.3%)** — most stable offline method. Optimal re-rollout advantage is higher quality for sustained learning.
- **Online >> Offline**: online MC16 optimal 99.1% vs offline 88.2% — iterative data refresh adds +11%.
- **MC16 >> MC1 in offline too**: mc16 optimal 88.2% vs mc1 optimal 77.7% (+10.5%). Multi-sample MC advantage is critical regardless of online/offline.
- **All offline methods oscillate 15-25%** — no fresh data to correct policy drift on fixed dataset.
- **Surprise: IQL > MC16 in peak but worse in final** — IQL's advantage from diverse training data (mixed policy levels) gives fast initial boost but doesn't sustain.

---

## [V3 Offline: MC16 Optimal (4x data, 4x eval)] - 2026-02-19 00:08

**Command**: `python -u -m RL.mc_finetune_awr_offline --checkpoint runs/pickcube_ppo/ckpt_76_logstd-1.5.pt --optimal_checkpoint runs/pickcube_ppo/ckpt_301.pt --mc_samples 16 --awr_beta 1.0 --num_envs 128 --num_steps 800 --num_minibatches 128 --update_epochs 4 --num_iterations 100 --eval_freq 5 --num_eval_envs 512 --exp_name v3_offline_mc16_optimal`
**Git**: 98fa4a7 (main)
**Run Dir**: runs/v3_offline_mc16_optimal__seed1__1771475724

### Settings
| Parameter | Value |
|-----------|-------|
| checkpoint | ckpt_76_logstd-1.5.pt (det SR=43.8%) |
| optimal_checkpoint | ckpt_301.pt (~99% SR) |
| mc_samples | 16 |
| awr_beta | 1.0 |
| num_envs | 128 |
| num_steps | 800 |
| batch_size | 102,400 (4x v2) |
| num_minibatches | 128 |
| minibatch_size | 800 |
| update_epochs | 4 |
| num_iterations | 100 |
| eval_freq | 5 |
| num_eval_envs | 512 (4x v2) |
| norm_adv | True |
| awr_max_weight | 20.0 |
| gamma | 0.8 |
| reward | sparse |
| eval | deterministic |

### Results
| Metric | Value |
|--------|-------|
| peak_SR | 93.8% (iter 70) |
| final_SR | 84.2% |
| avg_SR(50-100) | 87.2% |
| std_SR(50-100) | 3.9% |
| advantage_mean | -0.0697 |
| advantage_std | 0.0900 |
| advantage_pos% | 16.2% |
| training_time | 793.3s |

### Notes
- 4x data scaling (102,400 vs 25,600 in v2). Avg(50-100) improved +3.0% over v2 (84.2% → 87.2%).
- Peak reached 93.8% — highest among all offline experiments so far.
- Std actually slightly increased vs v2 (2.4% → 3.9%), suggesting oscillation is training instability not eval noise.
- Still the best offline method overall.

---

## [V3 Offline: MC1 Optimal (4x data, 4x eval)] - 2026-02-19 00:08

**Command**: `python -u -m RL.mc_finetune_awr_offline --checkpoint runs/pickcube_ppo/ckpt_76_logstd-1.5.pt --optimal_checkpoint runs/pickcube_ppo/ckpt_301.pt --mc_samples 1 --awr_beta 1.0 --num_envs 128 --num_steps 800 --num_minibatches 128 --update_epochs 4 --num_iterations 100 --eval_freq 5 --num_eval_envs 512 --exp_name v3_offline_mc1_optimal`
**Git**: 98fa4a7 (main)
**Run Dir**: runs/v3_offline_mc1_optimal__seed1__1771479557

### Settings
| Parameter | Value |
|-----------|-------|
| checkpoint | ckpt_76_logstd-1.5.pt (det SR=43.8%) |
| optimal_checkpoint | ckpt_301.pt (~99% SR) |
| mc_samples | 1 |
| awr_beta | 1.0 |
| num_envs | 128 |
| num_steps | 800 |
| batch_size | 102,400 (4x v2) |
| num_minibatches | 128 |
| update_epochs | 4 |
| num_iterations | 100 |
| eval_freq | 5 |
| num_eval_envs | 512 (4x v2) |
| norm_adv | True |
| gamma | 0.8 |
| reward | sparse |
| eval | deterministic |

### Results
| Metric | Value |
|--------|-------|
| peak_SR | 75.9% (iter 70) |
| final_SR | 64.1% |
| avg_SR(50-100) | 68.6% |
| std_SR(50-100) | 5.6% |
| advantage_mean | -0.0699 |
| advantage_std | 0.1465 |
| advantage_pos% | 17.7% |
| training_time | 792.2s |

### Notes
- MC1 confirms that single-sample MC advantage is much noisier (std=0.1465 vs MC16's 0.0900).
- 4x data improved avg(50-100) by +3.8% over v2 (64.8% → 68.6%).
- Still ~18.6% behind MC16 optimal — multi-sample MC remains critical.

---

## [V3 Offline: MC16 Onpolicy (4x data, 4x eval)] - 2026-02-19 00:08

**Command**: `python -u -m RL.mc_finetune_awr_offline --checkpoint runs/pickcube_ppo/ckpt_76_logstd-1.5.pt --optimal_checkpoint runs/pickcube_ppo/ckpt_76_logstd-1.5.pt --mc_samples 16 --awr_beta 1.0 --num_envs 128 --num_steps 800 --num_minibatches 128 --update_epochs 4 --num_iterations 100 --eval_freq 5 --num_eval_envs 512 --exp_name v3_offline_mc16_onpolicy`
**Git**: 98fa4a7 (main)
**Run Dir**: runs/v3_offline_mc16_onpolicy__seed1__1771481367

### Settings
| Parameter | Value |
|-----------|-------|
| checkpoint | ckpt_76_logstd-1.5.pt (det SR=43.8%) |
| optimal_checkpoint | ckpt_76_logstd-1.5.pt (same as initial — onpolicy) |
| mc_samples | 16 |
| awr_beta | 1.0 |
| num_envs | 128 |
| num_steps | 800 |
| batch_size | 102,400 (4x v2) |
| num_minibatches | 128 |
| update_epochs | 4 |
| num_iterations | 100 |
| eval_freq | 5 |
| num_eval_envs | 512 (4x v2) |
| norm_adv | True |
| gamma | 0.8 |
| reward | sparse |
| eval | deterministic |

### Results
| Metric | Value |
|--------|-------|
| peak_SR | 89.5% (iter 70) |
| final_SR | 84.6% |
| avg_SR(50-100) | 83.8% |
| std_SR(50-100) | 3.9% |
| advantage_mean | 0.0001 |
| advantage_std | 0.0928 |
| advantage_pos% | 40.7% |
| training_time | 794.4s |

### Notes
- Onpolicy MC16 is only 3.4% behind optimal MC16 in avg(50-100) (83.8% vs 87.2%).
- Advantage mean≈0 (centered) with 40.7% positive — well-calibrated since rollout policy = initial policy.
- With 4x data, the gap between optimal and onpolicy narrows. The "optimal critic" advantage is smaller at scale.

---

## [V3 Offline: MC1 Onpolicy (4x data, 4x eval)] - 2026-02-19 00:08

**Command**: `python -u -m RL.mc_finetune_awr_offline --checkpoint runs/pickcube_ppo/ckpt_76_logstd-1.5.pt --optimal_checkpoint runs/pickcube_ppo/ckpt_76_logstd-1.5.pt --mc_samples 1 --awr_beta 1.0 --num_envs 128 --num_steps 800 --num_minibatches 128 --update_epochs 4 --num_iterations 100 --eval_freq 5 --num_eval_envs 512 --exp_name v3_offline_mc1_onpolicy`
**Git**: 98fa4a7 (main)
**Run Dir**: runs/v3_offline_mc1_onpolicy__seed1__1771484628

### Settings
| Parameter | Value |
|-----------|-------|
| checkpoint | ckpt_76_logstd-1.5.pt (det SR=43.8%) |
| optimal_checkpoint | ckpt_76_logstd-1.5.pt (same as initial — onpolicy) |
| mc_samples | 1 |
| awr_beta | 1.0 |
| num_envs | 128 |
| num_steps | 800 |
| batch_size | 102,400 (4x v2) |
| num_minibatches | 128 |
| update_epochs | 4 |
| num_iterations | 100 |
| eval_freq | 5 |
| num_eval_envs | 512 (4x v2) |
| norm_adv | True |
| gamma | 0.8 |
| reward | sparse |
| eval | deterministic |

### Results
| Metric | Value |
|--------|-------|
| peak_SR | 76.0% (iter 35) |
| final_SR | 70.5% |
| avg_SR(50-100) | 67.4% |
| std_SR(50-100) | 4.3% |
| advantage_mean | 0.0001 |
| advantage_std | 0.1976 |
| advantage_pos% | 28.8% |
| training_time | 790.5s |

### Notes
- Weakest MC variant but still improved +2.0% avg over v2 with 4x data.
- High advantage std (0.1976) from single MC sample makes learning noisy.

---

## [V3 Offline: IQL (4x data, 4x eval)] - 2026-02-19 00:08

**Command**: `python -u -m RL.iql_awr_offline --checkpoint runs/pickcube_ppo/ckpt_76_logstd-1.5.pt --awr_beta 1.0 --num_envs 128 --num_steps 800 --num_minibatches 128 --update_epochs 4 --num_iterations 100 --eval_freq 5 --num_eval_envs 512 --exp_name v3_offline_iql`
**Git**: 98fa4a7 (main)
**Run Dir**: runs/v3_offline_iql__seed1__1771486474

### Settings
| Parameter | Value |
|-----------|-------|
| checkpoint | ckpt_76_logstd-1.5.pt (det SR=43.8%) |
| awr_beta | 1.0 |
| iql_expectile_tau | 0.7 |
| iql_epochs | 200 |
| iql_nstep | 1 |
| num_envs | 128 |
| num_steps | 800 |
| batch_size | 102,400 (4x v2) |
| num_minibatches | 128 |
| update_epochs | 4 |
| num_iterations | 100 |
| eval_freq | 5 |
| num_eval_envs | 512 (4x v2) |
| norm_adv | True |
| gamma | 0.8 |
| reward | sparse |
| eval | deterministic |

### Results
| Metric | Value |
|--------|-------|
| peak_SR | 84.6% (iter 30) |
| final_SR | 76.5% |
| avg_SR(50-100) | 78.9% |
| std_SR(50-100) | 3.4% |
| IQL_advantage_mean | -0.0253 |
| IQL_advantage_std | 0.0208 |
| IQL_advantage_pos% | 9.4% |
| IQL_training_time | 75.0s |
| total_training_time | 794.2s |

### Notes
- **Major improvement over v2**: The v2 IQL collapsed from 91.4% peak → 67.2% final (24% drop). V3 IQL is much more stable: 84.6% peak → 76.5% final (8% drop).
- Avg(50-100) jumped from 72.4% → 78.9% (+6.5%) — the biggest gain among all methods from 4x data.
- Std decreased from 4.5% → 3.4% — the only method where oscillation actually reduced.
- 4x data primarily helped by providing enough training signal for the IQL Q/V networks to generalize better.

---

## [V3 Offline: IQL nstep5 (4x data, 4x eval)] - 2026-02-19 00:08

**Command**: `python -u -m RL.iql_awr_offline --checkpoint runs/pickcube_ppo/ckpt_76_logstd-1.5.pt --iql_nstep 5 --awr_beta 1.0 --num_envs 128 --num_steps 800 --num_minibatches 128 --update_epochs 4 --num_iterations 100 --eval_freq 5 --num_eval_envs 512 --exp_name v3_offline_iql_nstep5`
**Git**: 98fa4a7 (main)
**Run Dir**: runs/v3_offline_iql_nstep5__seed1__1771487374

### Settings
| Parameter | Value |
|-----------|-------|
| checkpoint | ckpt_76_logstd-1.5.pt (det SR=43.8%) |
| awr_beta | 1.0 |
| iql_expectile_tau | 0.7 |
| iql_epochs | 200 |
| iql_nstep | 5 |
| num_envs | 128 |
| num_steps | 800 |
| batch_size | 102,400 (4x v2) |
| num_minibatches | 128 |
| update_epochs | 4 |
| num_iterations | 100 |
| eval_freq | 5 |
| num_eval_envs | 512 (4x v2) |
| norm_adv | True |
| gamma | 0.8 |
| reward | sparse |
| eval | deterministic |

### Results
| Metric | Value |
|--------|-------|
| peak_SR | 85.0% (iter 35) |
| final_SR | 78.1% |
| avg_SR(50-100) | 77.7% |
| std_SR(50-100) | 3.2% |
| IQL_advantage_mean | -0.0182 |
| IQL_advantage_std | 0.0299 |
| IQL_advantage_pos% | 17.7% |
| IQL_training_time | 75.4s |
| total_training_time | 800.5s |

### Notes
- Nstep5 has higher advantage diversity (pos%=17.7% vs 9.4% for nstep=1) but didn't outperform base IQL.
- Avg(50-100) actually dropped vs v2 nstep5 (80.7% → 77.7%, -3.0%). This is surprising — nstep5 may have been overfitting less with smaller data.
- With 4x data, base IQL (78.9%) slightly edges out nstep5 IQL (77.7%).

---

## [V3 Offline Batch Summary: 4x Data + 4x Eval] - 2026-02-19 00:08

**Script**: `bash run_v3_offline.sh`
**Git**: 98fa4a7 (main)
**Motivation**: Increase offline dataset from 25,600 to 102,400 samples and eval envs from 128 to 512 to reduce noise and improve training stability.

### Comparison: V3 (102,400 samples, 512 eval) vs V2 (25,600 samples, 128 eval)

| Experiment | V2 Avg(50-100) | V3 Avg(50-100) | Δ Avg | V2 Std | V3 Std | Δ Std |
|---|---|---|---|---|---|---|
| MC16 optimal | 84.2% | **87.2%** | +3.0% | 2.4% | 3.9% | +1.5% |
| MC16 onpolicy | 83.2% | **83.8%** | +0.7% | 2.2% | 3.9% | +1.7% |
| IQL | 72.4% | **78.9%** | **+6.5%** | 4.5% | **3.4%** | **-1.1%** |
| IQL nstep5 | 80.7% | 77.7% | -3.0% | 2.4% | 3.2% | +0.8% |
| MC1 optimal | 64.8% | **68.6%** | +3.8% | 5.3% | 5.6% | +0.3% |
| MC1 onpolicy | 65.4% | **67.4%** | +2.0% | 2.4% | 4.3% | +1.9% |

### V3 Ranking by Avg(50-100)
| Rank | Method | Avg(50-100) | Std | Final |
|---|---|---|---|---|
| 1 | MC16 optimal | 87.2% | 3.9% | 84.2% |
| 2 | MC16 onpolicy | 83.8% | 3.9% | 84.6% |
| 3 | IQL | 78.9% | 3.4% | 76.5% |
| 4 | IQL nstep5 | 77.7% | 3.2% | 78.1% |
| 5 | MC1 optimal | 68.6% | 5.6% | 64.1% |
| 6 | MC1 onpolicy | 67.4% | 4.3% | 70.5% |

### Key Takeaways
- **IQL stability dramatically improved**: V2 IQL collapsed from 91.4% → 67.2% (24% drop). V3 IQL: 84.6% → 76.5% (8% drop). Avg improved +6.5%. This is the biggest win from 4x data.
- **MC16 optimal remains the best**: 87.2% avg, consistent across both v2 and v3.
- **Oscillation is training instability, not eval noise**: Despite 4x eval envs, std did NOT decrease for most methods (some increased). The oscillation comes from overfitting/underfitting cycles on fixed offline data, not from eval sampling noise.
- **MC16 >> MC1**: The gap persists at ~18% — multi-sample MC advantage estimation is the key differentiator.
- **IQL nstep5 didn't benefit from 4x data**: Possible that nstep5 already had enough signal from multi-step TD at smaller scale, or that the larger dataset diluted its advantage.

---

## IQL Debug: Approximation Error Analysis + Reward Scaling + N-step TD - 2026-02-20 07:46

**Goal**: Diagnose IQL bottleneck — is it data quality or method limitation? Test reward scaling and n-step TD as potential fixes.

**Git**: 81d116d (main)
**Settings**: v2 (128 envs × 200 steps = 25,600 batch), ckpt_76_logstd-1.5

### Exp 1a: IQL Fit Analysis (mc_only — best possible data)

**Command**: `python -u -m RL.iql_fit_analysis --iql_data_mode mc_only --num_steps 200 --iql_max_transitions 100000 --cache_path cache/iql_fit_mc_data_v2.pt --output runs/iql_fit_mc_only_v2.png`
**Output**: `runs/iql_fit_mc_only_v2.png`, `runs/iql_fit_mc_only_v2.pt`

| Parameter | Value |
|-----------|-------|
| iql_data_mode | mc_only |
| mc_samples | 16 |
| gamma | 0.8 |
| iql_expectile_tau | 0.7 |
| iql_epochs | 200 (ran all 200) |
| iql_batch_size | 256 |
| iql_max_transitions | 100,000 (subsampled from 6.5M) |
| MC re-rollout time | 809.5s |
| IQL training time | 205.1s |

**IQL outputs**: Q mean=0.4571, std=0.2774 | V mean=0.4779, std=0.2753 | A mean=-0.0208, std=0.0534

| Metric | Pearson r | Spearman rho | RMSE |
|--------|-----------|-------------|------|
| Q(s,a) | 0.7273 | 0.7251 | 0.1842 |
| V(s) | 0.8140 | 0.8164 | 0.1833 |
| **A(s,a)** | **0.2578** | **0.2908** | **0.1177** |

### Exp 1b: IQL Fit Analysis (offline_only — baseline data)

**Command**: `python -u -m RL.iql_fit_analysis --iql_data_mode offline_only --num_steps 200 --cache_path cache/iql_fit_mc_data_v2.pt --output runs/iql_fit_offline_only_v2.png`
**Output**: `runs/iql_fit_offline_only_v2.png`

| Parameter | Value |
|-----------|-------|
| iql_data_mode | offline_only |
| iql_expectile_tau | 0.7 |
| iql_epochs | 200 |
| IQL training time | 56.2s |

**IQL outputs**: Q mean=0.0688, std=0.0619 | V mean=0.0703, std=0.0569 | A mean=-0.0014, std=0.0202

| Metric | Pearson r | Spearman rho | RMSE |
|--------|-----------|-------------|------|
| Q(s,a) | 0.2802 | 0.3263 | 0.3526 |
| V(s) | 0.3321 | 0.3616 | 0.4331 |
| **A(s,a)** | **0.1293** | **0.1451** | **0.1085** |

### Exp 2a: IQL reward_scale=10

**Command**: `python -u -m RL.iql_awr_offline --checkpoint runs/pickcube_ppo/ckpt_76_logstd-1.5.pt --reward_scale 10.0 --awr_beta 1.0 --num_envs 128 --num_steps 200 --num_minibatches 32 --update_epochs 4 --num_iterations 100 --eval_freq 5 --exp_name v2_offline_iql_rs10`
**Run Dir**: `runs/v2_offline_iql_rs10__seed1__1771601108/`

| Parameter | Value |
|-----------|-------|
| reward_scale | 10.0 |
| iql_expectile_tau | 0.7 |
| iql_nstep | 1 |
| awr_beta | 1.0 |
| IQL early stop | epoch 181 |
| Total training time | 243.0s |

**IQL outputs**: Q mean=0.9478, std=1.4489 | A mean=-0.0977, std=0.3666
**IQL Advantage on finetune data**: mean=-0.0701, std=0.3768, pos%=31.3%

| Iter | SR% |
|------|-----|
| 1 | 49.6 |
| 5 | 82.0 |
| 10 | 74.8 |
| 15 | 74.8 |
| 20 | 79.0 |
| 25 | 75.2 |
| 30 | 76.9 |
| **35** | **85.0** |
| 40 | 82.6 |
| 45 | 71.7 |
| 50 | 78.5 |
| 55 | 65.4 |
| 60 | 74.8 |
| 65 | 73.7 |
| 70 | 76.5 |
| 75 | 73.3 |
| 80 | 72.9 |
| 85 | 77.7 |
| 90 | 70.1 |
| 95 | 76.9 |
| 100 | 71.9 |

### Exp 3b: IQL nstep=5 + reward_scale=10

**Command**: `python -u -m RL.iql_awr_offline --checkpoint runs/pickcube_ppo/ckpt_76_logstd-1.5.pt --iql_nstep 5 --reward_scale 10.0 --awr_beta 1.0 --num_envs 128 --num_steps 200 --num_minibatches 32 --update_epochs 4 --num_iterations 100 --eval_freq 5 --exp_name v2_offline_iql_nstep5_rs10`
**Run Dir**: `runs/v2_offline_iql_nstep5_rs10__seed1__1771601458/`

| Parameter | Value |
|-----------|-------|
| reward_scale | 10.0 |
| iql_nstep | 5 |
| iql_expectile_tau | 0.7 |
| awr_beta | 1.0 |
| IQL early stop | epoch 98 |
| Total training time | 243.9s |

**IQL outputs**: Q mean=0.6680, std=1.4366 | A mean=-0.1142, std=0.4809
**IQL Advantage on finetune data**: mean=-0.0300, std=0.4012, pos%=42.6%

| Iter | SR% |
|------|-----|
| 1 | 49.6 |
| 5 | 63.8 |
| 10 | 77.9 |
| 15 | 67.2 |
| 20 | 65.6 |
| 25 | **78.5** |
| 30 | 59.8 |
| 35 | 67.7 |
| 40 | 63.8 |
| 45 | 66.2 |
| 50 | 64.4 |
| 55 | 58.2 |
| 60 | 67.4 |
| 65 | 62.1 |
| 70 | 68.4 |
| 75 | 65.7 |
| 80 | 60.9 |
| 85 | 70.7 |
| 90 | 72.6 |
| 95 | 70.5 |
| 100 | 66.9 |

### Comparison Table

| Experiment | Peak SR | Avg SR (iter 50-100) | A(s,a) rho | pos% | A std |
|------------|---------|---------------------|-----------|------|-------|
| **v2 pure IQL** (baseline) | 91.6% | ~74% | 0.15 (offline) | 9.5% | 0.021 |
| **Exp 2a: rs10** | 85.0% | ~74% | — | 31.3% | 0.377 |
| **Exp 3b: nstep5+rs10** | 78.5% | ~65% | — | 42.6% | 0.401 |
| **Fit: mc_only** | — | — | **0.29** | — | — |
| **Fit: offline_only** | — | — | **0.15** | — | — |

### Notes

- **Core finding: IQL approximation error is the fundamental bottleneck**. Even with perfect MC data (ground truth Q*), IQL advantage A(s,a) ranking is poor (rho=0.29). The Q-V subtraction amplifies errors: RMSE=0.12 >> A std=0.05, making the signal-to-noise ratio very low.
- **Reward scaling (×10) did NOT help SR**: While it improved pos% (9.5% → 31.3%) and advantage magnitude, peak SR was slightly worse (85% vs 91.6% baseline). Larger advantage std = more noise in AWR weights, offsetting any benefit.
- **nstep5 + rs10 combined was WORST**: Peak 78.5%, average ~65%. The combination amplifies instability without fixing the fundamental ranking problem.
- **mc_only vs offline_only fit analysis**: mc_only roughly doubled A rho (0.29 vs 0.15), confirming data quality matters, but 0.29 is still far too low for effective action ranking. The method itself cannot resolve fine-grained within-state action differences.
- **Implication**: Tricks like reward scaling and n-step TD cannot fix IQL's fundamental Q-V subtraction problem. For offline advantage estimation, direct methods (MC rollout, GAE with good V) remain necessary.

---

## [V(s) Correlation Analysis: Optimal MC vs On-policy MC vs IQL V] - 2026-02-20 17:29

**Command**: `python -u -m RL.v_correlation_analysis --cache_path runs/v_corr_cache.pt`
**Git**: 81d116d (main)
**Script**: `RL/v_correlation_analysis.py`
**Output**: `runs/v_correlation.png`, `runs/v_correlation.pt`, `runs/v_corr_cache.pt` (MC cache)

### Overview
Compare 5 value functions evaluated at the **same 102,400 states** (800 steps × 128 envs from ckpt_76 rollout):
- V^{π*}_{MC16}: MC16 estimate using optimal policy (ckpt_301)
- V^{π*}_{MC1}: single-sample MC (optimal)
- V^{π_on}_{MC16}: MC16 estimate using rollout policy (ckpt_76)
- V^{π_on}_{MC1}: single-sample MC (on-policy)
- V_IQL: IQL V trained on offline multi-checkpoint data (5 checkpoints × 200 episodes)

### Settings
| Parameter | Value |
|-----------|-------|
| rollout_policy | ckpt_76_logstd-1.5.pt (det SR=43.8%) |
| optimal_policy | ckpt_301.pt |
| mc_samples | 16 |
| num_envs | 128 |
| num_mc_envs | 4096 (128 × 2 × 16) |
| num_steps | 800 |
| gamma | 0.8 |
| reward_mode | sparse |
| iql_expectile_tau | 0.7 |
| iql_epochs | 200 (early stop at 65) |
| iql_data_checkpoints | ckpt_1, 51, 101, 201, 301 |
| iql_episodes_per_ckpt | 200 |
| iql_offline_num_envs | 512 |
| offline_transitions | 85,504 (3,103 trajectories) |
| MC re-rollout time | 7,593s (~2h7m) |

### V(s) Statistics
| Value Function | Mean | Std | Min | Max |
|----------------|------|-----|-----|-----|
| V^{π*}_{MC16} | 0.4189 | 0.2983 | 0.0000 | 1.0000 |
| V^{π*}_{MC1} | 0.4190 | 0.3114 | 0.0000 | 1.0000 |
| V^{π_on}_{MC16} | 0.0555 | 0.0936 | 0.0000 | 0.6832 |
| V^{π_on}_{MC1} | 0.0559 | 0.1667 | 0.0000 | 1.0000 |
| V_IQL | 0.0893 | 0.0352 | -0.1870 | 0.2171 |

### Pearson r Correlation Matrix
| | V*_MC16 | V*_MC1 | V^on_MC16 | V^on_MC1 | V_IQL |
|---|---|---|---|---|---|
| V*_MC16 | 1.0000 | 0.9579 | 0.7107 | 0.4030 | 0.3137 |
| V*_MC1 | 0.9579 | 1.0000 | 0.6800 | 0.3831 | 0.3011 |
| V^on_MC16 | 0.7107 | 0.6800 | 1.0000 | 0.5634 | 0.0337 |
| V^on_MC1 | 0.4030 | 0.3831 | 0.5634 | 1.0000 | 0.0184 |
| V_IQL | 0.3137 | 0.3011 | 0.0337 | 0.0184 | 1.0000 |

### Spearman ρ Correlation Matrix
| | V*_MC16 | V*_MC1 | V^on_MC16 | V^on_MC1 | V_IQL |
|---|---|---|---|---|---|
| V*_MC16 | 1.0000 | 0.9676 | 0.7150 | 0.2258 | 0.3395 |
| V*_MC1 | 0.9676 | 1.0000 | 0.6892 | 0.2178 | 0.3413 |
| V^on_MC16 | 0.7150 | 0.6892 | 1.0000 | 0.4373 | 0.2093 |
| V^on_MC1 | 0.2258 | 0.2178 | 0.4373 | 1.0000 | 0.0927 |
| V_IQL | 0.3395 | 0.3413 | 0.2093 | 0.0927 | 1.0000 |

### V_IQL Stratified by V* (Does IQL track optimal value?)
| V* range | n | V* mean | V^on mean | V_IQL mean | IQL/V* |
|----------|---|---------|-----------|------------|--------|
| [0.00, 0.05) | 12,243 | 0.022 | 0.000 | 0.048 | 2.23 |
| [0.05, 0.10) | 7,816 | 0.074 | 0.002 | 0.069 | 0.93 |
| [0.10, 0.20) | 11,658 | 0.148 | 0.003 | 0.085 | 0.58 |
| [0.20, 0.40) | 21,549 | 0.308 | 0.011 | 0.101 | 0.33 |
| [0.40, 0.60) | 18,433 | 0.494 | 0.034 | 0.104 | 0.21 |
| [0.60, 0.80) | 14,490 | 0.698 | 0.102 | 0.097 | 0.14 |
| [0.80, 1.00) | 16,123 | 0.893 | 0.202 | 0.094 | 0.11 |

### Key Findings

1. **Optimal MC is self-consistent**: V*_MC16 vs V*_MC1 ρ=0.97. The optimal policy is deterministic enough that single-sample MC captures ranking well.

2. **On-policy MC is noisy**: V^on_MC16 vs V^on_MC1 ρ=0.44. The rollout policy (ckpt_76, SR=43.8%) is stochastic — single MC sample is unreliable. This explains the V2 online result: MC16 on-policy (96.7%) >> MC1 on-policy (74.6%).

3. **IQL V is a near-constant function**: V_IQL std=0.035, IQR=[0.067, 0.113]. When V* ranges from 0.02 to 0.89 (45× range), V_IQL only ranges from 0.048 to 0.104 (2× range). IQL learned essentially a flat value function at out-of-distribution eval states.

4. **TD from offline data learns behavior value, NOT V***: V_IQL/V* = 0.21× (should be ≈1.0 if learning V*). V_IQL/V^on = 1.61× (consistent with learning a behavior-mixture value with τ=0.7 expectile). This confirms: **offline TD (IQL) is fundamentally bounded by the behavior policy's data quality — it cannot recover V* from off-optimal-policy transitions.**

5. **Critic quality is the dominant factor in online RL**: Cross-referencing with V2 online results:
   - MC samples (16 vs 1): **+22%** (on-policy), +21% (optimal)
   - Optimal vs on-policy: +2.4% (MC16), +3.7% (MC1)
   - Conclusion: accurate V estimation (MC16) matters far more than having the optimal policy for re-rollout.

### Code Bug Found
`collect_offline_data` uses `done = (term | trunc).float()` as the terminal flag for IQL TD learning. Truncation (time limit) should NOT disable bootstrapping — only true termination should. Impact is small for γ=0.8 but technically incorrect.

---

## TD-10 Pretrain for Iterative RL (gamma=0.99) — 3 Modes Comparison - 2026-02-21 00:00

**Git**: 81d116d (main)

### Context
Testing whether TD-10 pretrained V improves iterative GAE PPO at gamma=0.99. At gamma=0.99, TD(0) V learning is poor (r=0.21) due to long bootstrap chain, but TD-10 gives much better V (r=0.55). Three modes compared: pretrain first iter only, finetune every iter, retrain (reset) every iter.

### Common Settings
| Parameter | Value |
|-----------|-------|
| checkpoint | ckpt_76_logstd-1.5.pt (det SR=43.8%) |
| gamma | 0.99 |
| gae_lambda | 0.9 |
| td_nstep | 10 |
| td_reward_scale | 1.0 |
| td_epochs | 200 |
| td_batch_size | 256 |
| num_envs | 512 |
| num_steps | 50 |
| update_epochs | 4 |
| target_kl | 0.1 |
| total_timesteps | 2,000,000 |
| eval | deterministic |

### Experiment 1: TD-10 Pretrain (first iter only)

**Command**: `python -m RL.ppo_finetune --checkpoint runs/pickcube_ppo/ckpt_76_logstd-1.5.pt --gamma 0.99 --gae_lambda 0.9 --td_pretrain_v --td_nstep 10 --td_reward_scale 1.0 --td_epochs 200 --td_batch_size 256 --num_envs 512 --num_steps 50 --update_epochs 4 --target_kl 0.1 --total_timesteps 2000000 --eval_freq 5 --exp_name ppo_gae_td10_pretrain_g99 --seed 1`
**Run Dir**: runs/ppo_gae_td10_pretrain_g99__seed1__1771659467

| Iter | SR |
|------|----|
| 1 | 43.8% |
| 5 | 10.2% |
| 10 | 66.4% |
| 15 | 79.1% |
| 20 | 91.5% |
| 25 | 97.0% |
| 30 | **100.0%** |
| 35 | 100.0% |
| 40 | 99.6% |
| 45-60 | 100.0% |
| 78 | 99.6% |

### Experiment 2: TD-10 Finetune (every iter)

**Command**: `python -m RL.ppo_finetune --checkpoint runs/pickcube_ppo/ckpt_76_logstd-1.5.pt --gamma 0.99 --gae_lambda 0.9 --td_pretrain_v --td_mode finetune --td_nstep 10 --td_reward_scale 1.0 --td_epochs 200 --td_batch_size 256 --num_envs 512 --num_steps 50 --update_epochs 4 --target_kl 0.1 --total_timesteps 2000000 --eval_freq 5 --exp_name ppo_gae_td10_finetune_g99 --seed 1`
**Run Dir**: runs/ppo_gae_td10_finetune_g99__seed1__1771659849

| Iter | SR |
|------|----|
| 1 | 43.8% |
| 5-25 | 0.0% |
| 30 | 0.8% |
| 35 | 0.0% |
| 40 | 2.3% |
| 50 | 6.2% |
| 55 | 16.4% |
| 65 | 22.7% |
| 70 | 25.0% |
| 75 | **47.8%** |
| 78 | 43.9% |

### Experiment 3: TD-10 Retrain (reset every iter)

**Command**: `python -m RL.ppo_finetune --checkpoint runs/pickcube_ppo/ckpt_76_logstd-1.5.pt --gamma 0.99 --gae_lambda 0.9 --td_pretrain_v --td_mode retrain --td_nstep 10 --td_reward_scale 1.0 --td_epochs 200 --td_batch_size 256 --num_envs 512 --num_steps 50 --update_epochs 4 --target_kl 0.1 --total_timesteps 2000000 --eval_freq 5 --exp_name ppo_gae_td10_retrain_g99 --seed 1`
**Run Dir**: runs/ppo_gae_td10_retrain_g99__seed1__1771659852

| Iter | SR |
|------|----|
| 1 | 43.8% |
| 5-55 | 0.0% |
| 60 | **0.8%** |
| 65-78 | 0.0% |

### Summary

| Mode | Peak SR | Baseline GAE | Delta |
|------|---------|-------------|-------|
| TD-10 pretrain (first only) | **100.0%** | 92.1% | **+7.9%** |
| TD-10 finetune (every iter) | 47.8% | 92.1% | -44.3% |
| TD-10 retrain (every iter) | 0.8% | 92.1% | -91.3% |

### Notes
- **TD-10 pretrain (first only) is the clear winner**: +8% over baseline GAE. Best result for GAE-based iterative RL so far.
- **Finetune every iter collapses**: 200 epochs of TD each iteration overwrites the critic knowledge PPO accumulates through its own value loss. Policy collapses to 0%, slowly crawls back to 47.8% by iter 75.
- **Retrain every iter completely fails**: Resetting critic from scratch each iteration means PPO can never accumulate any value learning. Permanently stuck at 0%.
- **Key insight**: TD pretrain provides excellent V initialization, but PPO's own online critic training (via value loss in PPO update) is what maintains and improves V over time. Overwriting it every iteration is catastrophically destructive.
- At gamma=0.99, TD-10 is needed (not TD(0)) because the bootstrap chain is 50 steps long. TD(0)+rs=10 pretrain at gamma=0.99 only achieved 91.5% (no benefit over baseline).

---

## [PPO Data Efficiency: Critic Warmstart + Warmup + Clip Tuning] - 2026-02-21 02:10

**Git**: 81d116d (main)

### Objective
Improve online data efficiency of PPO finetuning. Metric: SR vs total env interactions.

### Baseline
```bash
python -m RL.ppo_finetune --checkpoint runs/pickcube_ppo/ckpt_76_logstd-1.5.pt --gamma 0.99 \
  --num_envs 100 --num_steps 50 --num_minibatches 5 --update_epochs 100 --target_kl 100.0 \
  --eval_freq 1 --total_timesteps 50000 --exp_name v2_gae_g099_baseline
```
**Run Dirs**: `runs/v2_gae_g099_det__seed1__1771458679`, `runs/v2_gae_g099_baseline_s2__seed2__1771667985`, `runs/v2_gae_g099_baseline_s3__seed3__1771668119`

| Seed | i1 | i2 | i3 | i4 | i5 | i6 | i7 | i8 | i9 | i10 | Peak |
|------|------|------|------|------|------|------|------|------|------|------|------|
| 1 | 43.8 | 58.0 | 66.4 | 73.3 | 72.3 | 83.7 | 85.8 | 87.6 | **92.1** | 91.2 | 92.1 |
| 2 | 42.2 | 58.0 | 63.8 | 61.4 | 62.8 | 58.8 | 65.9 | **78.7** | 73.5 | 77.4 | 78.7 |
| 3 | 50.8 | 42.6 | 57.7 | 72.1 | 77.4 | 78.9 | 88.2 | 91.1 | **91.5** | 91.2 | 91.5 |
| **Mean** | 45.6 | 52.9 | 62.6 | 68.9 | 70.8 | 73.8 | 80.0 | 85.8 | 85.7 | 86.6 | **87.4** |

### Experiments Run (all share baseline settings unless noted)

#### 1. GAE Pretrain 5 iters (reset critic)
**Command**: `... --gae_pretrain_iters 5 --gae_pretrain_epochs 100 --exp_name v2_gae_g099_pretrain5`
**Run Dir**: `runs/v2_gae_g099_pretrain5__seed1__1771665083`

| i1 | i2 | i3 | i4 | i5 | i6 | i7 | i8 | i9 | i10 | Peak |
|------|------|------|------|------|------|------|------|------|------|------|
| 41.4 | 51.5 | 64.1 | 66.9 | 84.2 | 82.1 | 86.0 | 87.2 | 88.9 | **90.1** | 90.1 |

Pretrain V corr: 0.57→0.72→0.86→0.93→0.97. Peak -2.0% vs baseline. PPO's own V training catches up quickly.

#### 2. update_epochs=200
**Run Dir**: `runs/v2_gae_g099_e200__seed1__1771665278`

| i1 | i2 | i3 | i4 | i5 | i6 | i7 | i8 | i9 | i10 | Peak |
|------|------|------|------|------|------|------|------|------|------|------|
| 43.8 | 56.9 | 63.8 | 68.5 | 55.8 | 69.4 | 72.4 | 71.4 | 67.9 | **73.5** | 73.5 |

Policy overfits to stale advantages. Collapses at i5.

#### 3. num_envs=50, epochs=100 (20 iters)
**Run Dir**: `runs/v2_gae_g099_env50__seed1__1771665536`
Peak: **75.9%**. Oscillates badly — 100 epochs overfits on small batch (2500).

#### 4. num_envs=50, epochs=50 (20 iters)
**Run Dir**: `runs/v2_gae_g099_env50_e50__seed1__1771665810`
Peak: **85.4%** (i20). Still climbing but hasn't converged. Under-trains per iteration.

#### 5. gae_lambda=0.95
**Run Dir**: `runs/v2_gae_g099_lam095__seed1__1771666118`
Peak: **92.1%** (i10). Same as baseline, no improvement.

#### 6. MC1 (lambda=1.0, gamma=0.99)
**Run Dir**: `runs/v2_mc1_g099__seed1__1771666295`
Peak: **69.9%** (i9). MC variance too high at gamma=0.99. Confirms Issue #10.

#### 7. num_envs=200 (5 iters, batch=10K)
**Run Dir**: `runs/v2_gae_g099_env200__seed1__1771666501`
Peak: **81.3%** (i5). Not enough policy updates with only 5 iterations.

#### 8. warmup=1 + clip=0.3 (reset critic)
**Run Dirs**: `runs/v2_gae_g099_reset_warmup1_clip03_s1__seed1__1771679555`, `runs/v2_gae_g099_reset_warmup1_clip03_s2__seed2__1771679688`, `runs/v2_gae_g099_reset_warmup1_clip03_s3__seed3__1771679823`

| Seed | i1 | i2 | i3 | i4 | i5 | i6 | i7 | i8 | i9 | i10 | Peak |
|------|------|------|------|------|------|------|------|------|------|------|------|
| 1 | 43.8 | 36.7 | 73.8 | 72.9 | 69.5 | 76.5 | 76.7 | 79.1 | 81.2 | **85.7** | 85.7 |
| 2 | 42.2 | 42.6 | 52.7 | 43.1 | 56.6 | 62.0 | 70.5 | 82.8 | 82.4 | **88.5** | 88.5 |
| 3 | 50.8 | 40.5 | 73.3 | **73.9** | 71.7 | 68.2 | 61.4 | 66.4 | 76.9 | 73.5 | 73.9 |
| **Mean** | 45.6 | 39.9 | 66.6 | 63.3 | 65.9 | 68.9 | 69.5 | 76.1 | 80.2 | **82.6** | **82.7** |

Worse than baseline (82.7% vs 87.4%). Warmup wastes 1 iteration without policy update, clip=0.3 causes oscillation with bad V (random critic).

### Notes
- **No configuration beat the baseline** (3-seed mean 87.4%) when starting from a reset (random) critic.
- GAE pretrain doesn't help: PPO's own 100-epoch critic training per iteration is sufficient.
- Higher epochs (200) overfits. Smaller batches (50 envs) oscillate. MC1 at gamma=0.99 has too much variance.
- warmup + wider clip only helps when V is already good; with random V it hurts.
- **⚠️ no-reset_critic 不可用**: 曾尝试保留 checkpoint 中 dense reward 训练的 critic（不 reset），配合 warmup=1 + clip=0.3 达到 97.9% mean peak。但此方法依赖 PPO 预训练阶段的 dense reward critic 提供 V 初始化，对 IL policy（无 pretrained critic）不适用，不具备通用性。结论已作废，实验数据不记录。
- **Baseline 的 100 envs / 100 epochs / 10 iters 已经是 reset critic 场景下的最优配置。**

---

