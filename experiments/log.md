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

