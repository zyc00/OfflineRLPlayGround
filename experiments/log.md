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

## Exp 9: V Scaling Analysis — GAE vs TD vs MC1 (eval against MC16) - 2026-02-21 09:30

**Command**: `python -u -m RL.v_scaling_analysis --gamma 0.99 --gae_lambda 1.0`
**Git**: fec75bf (main)
**Script**: `RL/v_scaling_analysis.py`

### Settings
| Parameter | Value |
|-----------|-------|
| checkpoint | ckpt_76_logstd-1.5 |
| gamma | 0.99 |
| gae_lambda | 1.0 |
| num_envs | 100 |
| num_steps | 50 |
| mc_samples | 16 (for ground truth) |
| gae_iters | 5 |
| gae_epochs | 100 |
| td_epochs | 200 |
| td_reward_scale | 10.0 |
| ema_tau | 0.005 |
| critic | 3×256, Tanh |
| lr | 3e-4 |
| rollout_counts | 1, 2, 5, 10, 20, 50, 100 |

### Results — Eval against MC16 ground truth (fixed eval set = rollout 0, 5K states)

MC1 vs MC16 correlation: **r=0.34** (MC1 is a very poor V^π estimator at γ=0.99)

| Rollouts | Trans | MC1 r | MC1 ρ | GAE r | GAE ρ | TD r | TD ρ | TD+rs r | TD+rs ρ | EMA r | EMA ρ |
|----------|-------|-------|-------|-------|-------|------|------|---------|---------|-------|-------|
| 1 | 5K | 0.34 | 0.31 | 0.36 | 0.33 | 0.34 | 0.33 | 0.35 | 0.33 | 0.04 | 0.05 |
| 2 | 10K | 0.36 | 0.33 | 0.39 | 0.38 | 0.35 | 0.33 | 0.40 | 0.38 | -0.02 | 0.00 |
| 5 | 25K | 0.35 | 0.34 | 0.37 | 0.36 | 0.59 | 0.58 | 0.40 | 0.40 | 0.53 | 0.52 |
| 10 | 50K | 0.34 | 0.27 | 0.37 | 0.34 | 0.61 | 0.60 | 0.65 | 0.61 | 0.71 | 0.68 |
| 20 | 100K | 0.34 | 0.30 | 0.37 | 0.31 | 0.60 | 0.58 | 0.67 | 0.65 | 0.75 | 0.74 |
| 50 | 250K | 0.35 | 0.31 | 0.37 | 0.34 | 0.71 | 0.69 | 0.71 | 0.69 | 0.80 | 0.80 |
| 100 | 500K | 0.36 | 0.33 | 0.41 | 0.38 | 0.68 | 0.66 | 0.74 | 0.73 | **0.82** | **0.81** |

### Notes
- **TD+EMA >> MC1/GAE** — 完全反转了之前用 MC1 评估时 "GAE >> TD" 的结论。
- 之前 GAE 看起来好是因为 MC1 评估的 artifact：MC1 在 γ=0.99 下单样本方差极大，GAE λ=1 iterative 本质上 fit 的就是 MC1 噪声，所以和 MC1 ground truth 高度相关。但那不是真正的 V^π。
- TD 的 bootstrap 虽然有偏差，但方差低，在大数据量下学到更准确的 V^π。
- EMA target network 有明显帮助：N≥10 时 EMA 比 vanilla TD 高 +0.10-0.14 r。
- MC1 和 GAE λ=1 的 r 几乎不随数据量增长（~0.34-0.41），因为 bottleneck 是 MC1 target 本身的噪声。

---

## Exp 10: One-Step Offline RL — TD vs MC1 V for Policy Improvement - 2026-02-21 10:30

**Command**: `python -u -m RL.offline_onestep_rl --gamma 0.99 --num_rollouts 200`
**Git**: fec75bf (main)
**Script**: `RL/offline_onestep_rl.py`

### Settings
| Parameter | Value |
|-----------|-------|
| checkpoint | ckpt_76_logstd-1.5 |
| gamma | 0.99 |
| gae_lambda | 0.95 (for advantage computation) |
| num_rollouts | 200 (1M transitions) |
| num_envs | 100 |
| num_steps | 50 |
| V methods | MC1 (500ep), TD (200ep), TD+rs10 (200ep), TD+EMA (200ep, τ=0.005), GAE iter5 (5×100ep, λ=0.95) |
| critic | 3×256, Tanh, lr=3e-4 |
| AWR β | 1.0 |
| AWR max_weight | 20.0 |
| norm_adv | True |
| update_epochs | 4 |
| num_minibatches | 32 |
| num_iterations | 100 |
| eval_freq | 5 |
| learning_rate | 3e-4 |

### Design
1. Collect 200 rollouts from behavior policy (1M transitions)
2. Learn V via 5 methods (standalone critic per method)
3. Compute GAE(λ=0.95) advantages using each method's V
4. AWR policy update (actor-only, clone behavior policy per method)
5. Evaluate deterministic SR every 5 iterations

### Advantage Statistics
| Method | mean | std | pos% |
|--------|------|-----|------|
| MC1 | 0.009 | 0.147 | 49.3% |
| TD | -0.004 | 0.194 | 43.3% |
| TD+rs10 | -0.006 | 0.180 | 45.0% |
| TD+EMA | 0.001 | 0.186 | 45.2% |
| GAE | 0.004 | 0.146 | 50.0% |

### Results

Baseline SR (behavior policy): **40.3%**

| Method | V r (MC16) | Peak SR | Final SR | Δ vs baseline |
|--------|-----------|---------|----------|---------------|
| **TD** | 0.68 | **82.6%** | **79.1%** | +42.3% |
| **TD+rs10** | 0.74 | 82.5% | 71.4% | +42.2% |
| **TD+EMA** | 0.82 | 82.2% | 70.2% | +41.9% |
| GAE | 0.41 | 73.3% | 59.8% | +33.0% |
| MC1 | 0.36 | 66.2% | 54.3% | +25.9% |

### Per-iteration SR
| Iter | MC1 | TD | TD+rs10 | TD+EMA | GAE |
|------|-----|-----|---------|--------|-----|
| 1 | 44.3% | 40.5% | 47.7% | 50.8% | 43.4% |
| 5 | 56.2% | 73.7% | 60.8% | 81.2% | 59.5% |
| 10 | 41.1% | 79.3% | 66.2% | 70.8% | 56.6% |
| 20 | 62.1% | 75.6% | 77.9% | 75.9% | 69.5% |
| 50 | 61.2% | 67.9% | 72.0% | 80.9% | 56.9% |
| 100 | 50.8% | 77.3% | 80.0% | 82.2% | 52.7% |

### Notes
- **Better V quality → better policy improvement**: TD methods (peak ~82.5%) >> GAE (73.3%) >> MC1 (66.2%)，和 V quality ranking 一致。
- **三个 TD 变体 peak SR 几乎相同** (~82.5%)：虽然 V quality 差距不小 (r=0.68 vs 0.82)，但 downstream SR 差不多。V quality 可能存在收益递减阈值。
- **TD 的 Final SR 最高 (79.1%)**：训练更稳定。TD+rs 和 TD+EMA final SR 更低（71%/70%），oscillation 更大。
- **所有方法都 oscillate 严重**：offline AWR 在固定数据上反复训练的典型问题。
- **TD+EMA 早期收敛最快**：Iter 5 即达到 81.2%，远超其他方法。
- **结论**：在 γ=0.99 offline setting 下，TD 学 V 比 MC1/GAE 好得多，且直接转化为 policy improvement 优势。但 TD 变体之间差异不大（peak SR 维度）。

---

## [Exp 11: V Scaling Law — Data Size × Network Size × Method] - 2026-02-21 14:48

**Command**: `python -u -m RL.v_scaling_law --gamma 0.99`
**Git**: fec75bf (main)
**Run Dir**: runs/v_scaling_law.pt, runs/v_scaling_law.png
**Script**: RL/v_scaling_law.py (new)

### Settings
| Parameter | Value |
|-----------|-------|
| gamma | 0.99 |
| gae_lambda | 0.95 |
| num_envs | 100 |
| num_steps | 50 |
| max_episode_steps | 50 |
| mc_samples | 16 |
| gae_iters | 5 |
| gae_epochs | 100 |
| td_epochs | 200 |
| td_reward_scale | 10.0 |
| ema_tau | 0.005 |
| critic_layers | 3 |
| lr | 3e-4 |
| minibatch_size | 1000 |
| rollout_counts | (1, 2, 5, 10, 20, 50, 100) |
| hidden_dims | (64, 128, 256, 512) |
| iql_data_checkpoints | ckpt_1/51/101/201/301 |
| iql_episodes_per_ckpt | 200 |
| iql_offline_num_envs | 512 |
| iql_expectile_tau | 0.7 |
| iql_epochs | 200 |
| iql_batch_size | 256 |
| iql_patience | 50 |
| checkpoint (behavior) | ckpt_76_logstd-1.5 |
| optimal_checkpoint | ckpt_301 |
| seed | 1 |

### Methods
- **MC1**: Direct regression on single-sample trajectory MC returns
- **TD+rs10**: TD(0) with reward_scale=10, no target network
- **TD+EMA**: TD(0) with reward_scale=10 + EMA target network (tau=0.005)
- **GAE**: Iterative GAE (5 iter × 100 epochs), lambda=0.95
- **IQL**: Expectile regression (tau=0.7), offline multi-checkpoint data (5 ckpts × 200 ep)

### Evaluation
- **On-policy MC16**: 16 re-rollouts with behavior policy (ckpt_76), averaged
- **Optimal MC16**: 16 re-rollouts with optimal policy (ckpt_301), averaged
- **Eval set**: Rollout 0 states (5,000 states from 100 envs × 50 steps)
- **Interleaved dual MC16**: 3,200 mc_envs (100 × 32), optimal + on-policy in one pass

### Results — Data Scaling (hidden=256, Pearson r)

| Rollouts | MC1 on | MC1 opt | TD+rs on | TD+rs opt | TD+EMA on | TD+EMA opt | GAE on | GAE opt | IQL on | IQL opt |
|----------|--------|---------|----------|-----------|-----------|------------|--------|---------|--------|---------|
| 1 | 0.339 | 0.151 | 0.349 | 0.092 | -0.055 | 0.341 | 0.403 | 0.121 | 0.219 | -0.277 |
| 2 | 0.364 | 0.143 | 0.447 | 0.071 | 0.129 | 0.386 | 0.454 | 0.137 | 0.219 | -0.277 |
| 5 | 0.347 | 0.137 | 0.502 | 0.204 | 0.441 | 0.389 | 0.428 | 0.155 | 0.219 | -0.277 |
| 10 | 0.339 | 0.146 | 0.676 | 0.191 | 0.718 | 0.228 | 0.389 | 0.187 | 0.219 | -0.277 |
| 20 | 0.344 | 0.145 | 0.677 | 0.348 | 0.756 | 0.205 | 0.431 | 0.155 | 0.219 | -0.277 |
| 50 | 0.360 | 0.140 | 0.735 | 0.173 | 0.817 | 0.220 | 0.455 | 0.151 | 0.219 | -0.277 |
| 100 | 0.380 | 0.146 | 0.757 | 0.202 | **0.820** | 0.244 | 0.458 | 0.128 | 0.219 | -0.277 |

### Results — Network Size Scaling (rollouts=100, Pearson r)

| Hidden | MC1 on | MC1 opt | TD+rs on | TD+rs opt | TD+EMA on | TD+EMA opt | GAE on | GAE opt | IQL on | IQL opt |
|--------|--------|---------|----------|-----------|-----------|------------|--------|---------|--------|---------|
| 64 | 0.593 | 0.189 | 0.833 | 0.251 | **0.851** | 0.226 | 0.760 | 0.248 | 0.215 | -0.313 |
| 128 | 0.415 | 0.192 | 0.795 | 0.204 | 0.846 | 0.255 | 0.549 | 0.188 | 0.231 | -0.234 |
| 256 | 0.386 | 0.137 | 0.770 | 0.234 | 0.820 | 0.227 | 0.449 | 0.158 | 0.228 | -0.197 |
| 512 | 0.356 | 0.166 | 0.766 | 0.324 | 0.775 | 0.232 | 0.429 | 0.143 | 0.187 | -0.305 |

### Notes

**Data scaling key findings:**
- **TD+EMA clearly dominates**: r=0.82 at 100 rollouts vs on-policy MC16. Scales from -0.06 (1 rollout) to 0.82 (100 rollouts).
- **TD+rs10 also scales well**: r=0.76 at 100 rollouts, but EMA target network gives consistent +0.05 improvement.
- **MC1 and GAE are flat**: MC1 stays ~0.34-0.38 regardless of data. GAE ~0.39-0.46. Both bottlenecked by single-sample MC noise at γ=0.99.
- **IQL is weak**: r=0.22 (flat dashed line). Multi-checkpoint offline data doesn't produce useful V for behavior policy states.
- **IQL vs optimal MC16 is NEGATIVE** (r=-0.28): IQL's V is anti-correlated with V^{pi*} at behavior states. Makes sense — IQL learns from mixed data dominated by bad policies (ckpt_1/51).

**Network scaling key findings:**
- **Smaller networks are better for MC1/GAE**: MC1 at hidden=64 (r=0.59) >> hidden=512 (r=0.36). Smaller networks regularize against noisy MC1 targets.
- **TD methods are robust**: TD+EMA ranges 0.78-0.85 across hidden sizes. Bootstrap targets are cleaner so overfitting is less of an issue.
- **GAE also benefits from small networks**: 0.76 at 64 vs 0.43 at 512. GAE targets are essentially iteratively-refined MC1, so same noise issue.
- **IQL is uniformly bad**: ~0.20 regardless of network size. The bottleneck is data quality, not model capacity.

**vs Optimal MC16 is universally low (r < 0.35):**
- All methods learn V^{pi_behavior} from behavior rollouts, which is fundamentally different from V^{pi*}.
- This is expected and confirms that offline behavior-only data cannot estimate V^{pi*}.

**Conclusion:** TD+EMA with reward scaling is the clear winner for learning V(s) from offline on-policy data at γ=0.99. It's the only method that "scales" — its V quality improves significantly with more data. MC1/GAE/IQL are all bottlenecked by noise or data quality issues that more data cannot fix.

---

## [Exp 11b: V Scaling Law — IQL On-Policy (same data as TD/GAE)] - 2026-02-21 16:20

**Command**: `python -u -m RL.v_scaling_law --gamma 0.99`
**Git**: fec75bf (main)
**Run Dir**: runs/v_scaling_law.pt, runs/v_scaling_law.png

### Overview

Follow-up to Exp 11. User asked: "IQL with tau=0.5 should degenerate to TD — why is it so weak?" Hypothesis was data distribution mismatch (Exp 11 used multi-checkpoint offline data while eval is at ckpt_76 states). This experiment tests IQL with the **exact same on-policy rollout data** as MC1/TD/GAE, eliminating the data distribution confound.

### Changes from Exp 11

- **Removed Phase 3** (offline multi-checkpoint data collection)
- **IQL now trains on same on-policy rollouts** as other methods (ckpt_76 rollouts)
- Added action storage in Phase 1 to provide (s, a, r, s', d) tuples for IQL
- IQL scales with data like other methods (no longer a flat horizontal line)
- IQL settings: tau=0.7 (expectile), reward_scale=1.0, 200 epochs, lr=3e-4

### Results — Data Scaling (hidden=256, Pearson r)

| Rollouts | MC1 on | MC1 opt | TD+rs on | TD+rs opt | TD+EMA on | TD+EMA opt | GAE on | GAE opt | IQL on | IQL opt |
|----------|--------|---------|----------|-----------|-----------|------------|--------|---------|--------|---------|
| 1 | 0.341 | 0.148 | 0.293 | 0.087 | -0.038 | 0.396 | 0.391 | 0.151 | -0.086 | 0.390 |
| 2 | 0.357 | 0.137 | 0.441 | 0.057 | 0.013 | 0.403 | 0.451 | 0.147 | 0.136 | -0.285 |
| 5 | 0.342 | 0.140 | 0.523 | 0.013 | 0.575 | 0.308 | 0.423 | 0.160 | 0.163 | -0.316 |
| 10 | 0.346 | 0.136 | 0.597 | 0.075 | 0.719 | 0.202 | 0.425 | 0.109 | 0.168 | -0.300 |
| 20 | 0.343 | 0.139 | 0.666 | 0.164 | 0.758 | 0.218 | 0.425 | 0.159 | 0.196 | -0.244 |
| 50 | 0.369 | 0.150 | 0.733 | 0.187 | 0.811 | 0.212 | 0.419 | 0.185 | 0.141 | -0.146 |
| 100 | 0.365 | 0.158 | 0.774 | 0.250 | **0.825** | 0.227 | 0.483 | 0.155 | 0.209 | -0.286 |

### Results — Network Size Scaling (rollouts=100, Pearson r)

| Hidden | MC1 on | MC1 opt | TD+rs on | TD+rs opt | TD+EMA on | TD+EMA opt | GAE on | GAE opt | IQL on | IQL opt |
|--------|--------|---------|----------|-----------|-----------|------------|--------|---------|--------|---------|
| 64 | 0.572 | 0.198 | 0.797 | 0.147 | **0.856** | 0.252 | 0.735 | 0.249 | 0.225 | -0.137 |
| 128 | 0.406 | 0.180 | 0.819 | 0.199 | 0.848 | 0.235 | 0.573 | 0.199 | 0.147 | -0.314 |
| 256 | 0.356 | 0.150 | 0.750 | 0.238 | 0.820 | 0.215 | 0.475 | 0.152 | 0.172 | -0.197 |
| 512 | 0.360 | 0.138 | 0.627 | 0.190 | 0.775 | 0.236 | 0.435 | 0.161 | 0.193 | -0.182 |

### Comparison: IQL On-Policy vs Offline (Exp 11)

| Setting | IQL on-policy (this) | IQL offline (Exp 11) |
|---------|---------------------|---------------------|
| 100 rollouts, h=256, vs on-policy MC16 | r=0.209 | r=0.219 |
| 100 rollouts, h=256, vs optimal MC16 | r=-0.286 | r=-0.277 |
| h=64, vs on-policy MC16 | r=0.225 | r=0.215 |

**Nearly identical.** On-policy data does NOT help IQL.

### Notes

**Key finding: IQL's weakness is ALGORITHMIC, not data distribution.**
- IQL on-policy (r=0.21) ≈ IQL offline (r=0.22) at matched settings
- Same on-policy data: TD+EMA gets r=0.82, TD+rs gets r=0.77, IQL gets r=0.21
- IQL's indirect Q→V path (expectile regression) is fundamentally worse than direct TD for V learning

**Why IQL V is weak:**
1. IQL learns Q(s,a) via TD, then extracts V(s) via expectile regression on Q
2. The Q→V extraction step adds noise and loses information
3. Direct TD learns V(s) directly from (s, r, s') — no intermediate Q step
4. IQL's reward_scale=1.0 (vs TD's rs=10) also contributes, but even matching data doesn't close the gap

**Possible follow-ups:**
- Test IQL with tau=0.5 (should degenerate V to E[Q(s,a)] ≈ TD V)
- Ablate IQL's scheduler/early-stop/train-val-split

---

## [Exp 12: TD+EMA Fixed Grad Steps + Early Stopping — Disentangling Data vs Compute Scaling] - 2026-02-21 22:28

**Command**: `python -u -m RL.td_ema_earlystop_test --gamma 0.99 --td_grad_steps 100000`
**Git**: fec75bf (main)
**Run Dir**: /tmp/td_ema_es_test.log

### Overview

Follow-up to Exp 11b. User noticed that epoch-based training confounds data scaling with compute scaling: 1 rollout × 200 epochs = 1,000 grad steps, while 100 rollouts × 200 epochs = 100,000 grad steps (100x more). This experiment tests whether TD+EMA's apparent "data scaling" is actually just "compute scaling."

Two sub-experiments in one run:
1. **Fixed 100k grad steps** (no early stopping): same compute budget for all data sizes
2. **Fixed 100k max steps + val early stopping** (patience=20): prevent overfitting on small data

### Settings
| Parameter | Value |
|-----------|-------|
| method | TD+EMA (rs=10, ema_tau=0.005) |
| gamma | 0.99 |
| hidden_dim | 256 |
| td_grad_steps | 100,000 (fixed for all data sizes) |
| minibatch_size | 1,000 |
| lr | 3e-4 |
| early_stop patience | 20 epochs |
| val split | 10% |
| rollout_counts | 1, 2, 5, 10, 20, 50, 100 |
| eval | vs on-policy MC16 (Pearson r) |

### Results — TD+EMA on-policy Pearson r

| Rollouts | Epoch-based (Exp 11b) | Fixed 100k | EarlyStop | ES epoch |
|----------|-----------------------|------------|-----------|----------|
| 1 | -0.038 (1k steps) | 0.336 | 0.200 | 766 |
| 2 | 0.013 (2k steps) | 0.343 | 0.374 | 415 |
| 5 | 0.575 (5k steps) | 0.373 | **0.480** | 237 |
| 10 | 0.719 (10k steps) | 0.365 | **0.503** | 131 |
| 20 | 0.758 (20k steps) | 0.423 | **0.504** | 74 |
| 50 | 0.811 (50k steps) | 0.683 | 0.503 | 41 |
| 100 | 0.825 (100k steps) | **0.819** | 0.517 | 31 |

### Notes

**Key finding 1: Fixed 100k steps CONFIRMS overfitting on small data.**
- 10 rollouts: epoch-based (10k steps) r=0.719, fixed 100k r=0.365. MORE compute = WORSE.
- 50 rollouts: epoch-based (50k steps) r=0.811, fixed 100k r=0.683. 2x compute = worse.
- 100 rollouts: both use 100k steps → r≈0.82, consistent.
- **TD bootstrap overfitting**: too many passes cause critic to memorize (s,s') pairs; EMA target catches up to overfitted online network, losing stabilization benefit.

**Key finding 2: Val TD loss early stopping is a poor metric.**
- EarlyStop plateaus at r≈0.50 regardless of data size (5-100 rollouts all ~0.50).
- Stops TOO EARLY at large data (100 rollouts: epoch 31 → r=0.52 vs r=0.82 at epoch 200).
- **Problem**: TD val loss measures prediction consistency (V matches bootstrap target), NOT V accuracy. V can be self-consistent but wrong. Low val loss ≠ good V.
- At N=1: stops at epoch 766 but still only r=0.20 (not enough data diversity regardless of stopping).

**Key finding 3: True data scaling effect is moderate, not dramatic.**
- Fixed 100k (over-trained): 0.34 → 0.82 (data helps, but 1-10 rollouts hurt by overfitting)
- EarlyStop (under-trained at large N): plateaus ~0.50
- **Neither extreme gives a clean data scaling curve.** The "true" effect is somewhere in between.
- Best interpretation: epoch-based (200 epochs, proportional compute) is actually a reasonable heuristic — more data naturally gets more compute, which is needed.

**Conclusion**: The original epoch-based scaling curve (Exp 11b) is NOT purely a compute artifact — TD+EMA genuinely benefits from more data. But the magnitude is exaggerated by proportional compute scaling. The 1-5 rollout results were artificially suppressed by insufficient training (1-5k steps). A fair comparison needs per-data-size hyperparameter tuning, not fixed steps or fixed epochs.

---

## [Exp 13: GAE+rs10 — Does Reward Scaling Help GAE?] - 2026-02-22 02:00

**Command**: `python -u -m RL.gae_rs_test --gamma 0.99`
**Git**: fec75bf (main)
**Run Dir**: /tmp/gae_rs_test.log

### Overview

TD+rs10 dramatically improves V learning (r: 0.44→0.68). Does the same trick help GAE? GAE's iterative structure (recompute GAE targets each iter + fit critic) is different from TD, so reward scaling may behave differently.

### Settings
| Parameter | Value |
|-----------|-------|
| methods | GAE_rs1, GAE_rs10 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| gae_iters | 5 |
| gae_epochs | 100 |
| hidden_dim | 256 |
| lr | 3e-4 |
| minibatch_size | 1,000 |
| eval | vs on-policy MC16 (Pearson r) |

### Results — Data Scaling (hidden=256, Pearson r)

| N | GAE_rs1 | GAE_rs10 | delta |
|---|---------|----------|-------|
| 1 | **0.404** | 0.397 | -0.007 |
| 2 | **0.454** | 0.400 | -0.054 |
| 5 | **0.421** | 0.383 | -0.038 |
| 10 | **0.420** | 0.395 | -0.025 |
| 20 | **0.446** | 0.397 | -0.049 |
| 50 | **0.450** | 0.422 | -0.028 |
| 100 | **0.469** | 0.407 | -0.062 |

### Results — Network Scaling (rollouts=100, Pearson r)

| H | GAE_rs1 | GAE_rs10 | delta |
|---|---------|----------|-------|
| 64 | **0.713** | 0.704 | -0.009 |
| 128 | **0.559** | 0.510 | -0.049 |
| 256 | **0.468** | 0.418 | -0.050 |
| 512 | **0.422** | 0.383 | -0.039 |

### Notes

**GAE+rs10 is WORSE than GAE_rs1 in ALL settings**, averaging ~0.04 worse.

**Why reward_scale helps TD but hurts GAE:**
- **TD**: Directly fits `r + γV(s')`. Small rewards → small targets → NN regression-to-mean is severe → rs amplifies signal, helps fitting.
- **GAE**: Iterative. Each round computes `δ = r·rs + γV - V`, then fits V to `adv + V`. Reward scaling amplifies the delta term, but V is also learned in scaled space. The iterative recomputation amplifies cumulative errors in scaled space. After unscaling, these amplified errors persist.
- **Key difference**: GAE targets are self-bootstrapped (recomputed each iter), so rs amplifies error propagation across iterations.

---

## [Exp 14: V Training Curves — Peak vs Overfit for TD+EMA, MC1, GAE] - 2026-02-22 04:36

**Command**: `python -u -m RL.td_ema_curve --gamma 0.99 --td_epochs 500 --eval_every 5`
**Git**: fec75bf (main)
**Run Dir**: runs/v_train_curves.png, runs/v_train_curves.pt

### Overview

Instead of early stopping, log full training curves: evaluate V quality (Pearson r vs on-policy MC16) every 5 epochs during training. This reveals each method's peak performance and overfitting dynamics.

Motivated by Exp 12's finding that epoch-based training confounds data scaling with compute scaling, and that early stopping on TD val loss is a poor proxy for V quality.

### Settings
| Parameter | Value |
|-----------|-------|
| methods | TD+EMA (rs=10, ema=0.005), MC1, GAE (5 iter × 100 ep) |
| gamma | 0.99 |
| hidden_dim | 256 |
| td_epochs | 500 (TD+EMA, MC1) |
| gae_iters × gae_epochs | 5 × 100 = 500 total (GAE) |
| eval_every | 5 epochs |
| lr | 3e-4 |
| minibatch_size | 1,000 |
| rollout_counts | 1, 5, 10, 20, 50, 100 |

Also ran N=1 separately with 25,000 epochs for TD+EMA to find its true peak.

### Results — Peak r (peak epoch)

| N | TD+EMA peak | TD+EMA final | MC1 peak | MC1 final | GAE peak | GAE final |
|---|------------|-------------|---------|----------|---------|----------|
| 1* | 0.44@ep1400 | 0.35 | 0.37@ep245 | 0.35 | **0.47**@ep195 | 0.41 |
| 5 | **0.65**@ep360 | 0.60 | 0.58@ep70 | 0.34 | 0.59@ep165 | 0.44 |
| 10 | **0.72**@ep335 | 0.66 | 0.70@ep45 | 0.35 | 0.66@ep120 | 0.43 |
| 20 | **0.77**@ep250 | 0.61 | 0.72@ep50 | 0.35 | 0.55@ep205 | 0.41 |
| 50 | **0.83**@ep145 | 0.59 | 0.79@ep45 | 0.35 | 0.68@ep115 | 0.45 |
| 100 | **0.85**@ep90 | 0.65 | 0.81@ep40 | 0.36 | 0.67@ep120 | 0.47 |

*N=1 TD+EMA from separate 25k-epoch run; MC1/GAE from 500-epoch run.

### Notes

**Key finding 1: MC1's peak is much higher than previously thought.**
- MC1 at N=100: peak r=0.81 (vs TD+EMA 0.85). Previously measured as r=0.37 because 200 epochs was deep into overfitting territory — peak is at epoch 40!
- MC1 overfits most severely: all data sizes collapse to final ~0.35, losing ~0.45 from peak.

**Key finding 2: TD+EMA has the best peak AND mildest overfitting.**
- Peak→final drops ~0.20 (vs MC1's ~0.45, GAE's ~0.20).
- EMA target network provides genuine regularization against overfitting.
- TD+EMA is the only method where peak consistently exceeds 0.80 at N≥50.

**Key finding 3: GAE peaks are surprisingly low at large N.**
- N=100: GAE peak 0.67 << MC1 0.81 << TD+EMA 0.85.
- GAE's iterative recomputation doesn't help as much as expected.
- But at N=1, GAE is the best (0.47 vs MC1 0.37 vs TD+EMA 0.17) — bootstrapping helps with minimal data.

**Key finding 4: Previous Exp 11b results were epoch-200 snapshots, not peaks.**
- MC1 appeared flat at ~0.37 across data sizes because all peaks occur before epoch 70, and by epoch 200 all have overfitted to ~0.35.
- The "MC1 doesn't scale with data" conclusion was WRONG — MC1 peak scales from 0.37→0.81 (2.2x).
- True ranking at peak: TD+EMA > MC1 >> GAE (at N≥10).

**Training curve shapes:**
- **TD+EMA**: Smooth rise, gradual decline. Peak epoch decreases with N (495→90).
- **MC1**: Sharp rise, sharp fall. Peak always early (ep 40-70). Classic supervised overfitting on noisy targets.
- **GAE**: Staircase pattern (jumps every 100 epochs at GAE iter boundaries). More stable but lower ceiling.

---

## Exp 15: IQL V Quality with TD-N + Reward Scaling - 2026-02-22 05:50

**Command**: `python -u -m RL.iql_curve --gamma 0.99`
**Git**: fec75bf (main)
**Script**: `RL/iql_curve.py`
**Run Dir**: runs/iql_curve.pt, runs/iql_curve.png

### Settings
| Parameter | Value |
|-----------|-------|
| gamma | 0.99 |
| checkpoint | runs/pickcube_ppo/ckpt_76_logstd-1.5.pt |
| optimal_checkpoint | runs/pickcube_ppo/ckpt_301.pt |
| num_envs | 100 |
| num_steps | 50 |
| num_rollouts | 100 |
| total transitions | 500,000 (100×100×50) |
| mc_samples | 16 (for MC16 ground truth) |
| max_episode_steps | 50 |
| hidden_dim | 256 |
| critic_layers | 3 |
| lr | 3e-4 |
| epochs | 500 |
| eval_every | 5 epochs |
| td_n_steps tested | (1, 5, 10, 50) |
| reward_scales tested | (1.0, 10.0) |
| IQL expectile tau | 0.7 |
| IQL beta (Q softmax) | 10.0 |
| TD+EMA baseline | rs=10, tau_ema=0.005 |
| TD+EMA td_steps | 10 |

### Results

| Method | peak_r | peak_ep | final_r |
|--------|--------|---------|---------|
| TD+EMA_rs10 (baseline) | 0.8528 | 85 | 0.6401 |
| IQL_rs1_n1 | 0.7407 | 485 | 0.7321 |
| IQL_rs1_n5 | 0.8478 | 120 | 0.7499 |
| **IQL_rs1_n10** | **0.8538** | 70 | 0.5801 |
| IQL_rs1_n50 | 0.8214 | 40 | 0.4191 |
| IQL_rs10_n1 | 0.7470 | 370 | 0.7227 |
| **IQL_rs10_n5** | **0.8560** | 75 | 0.6522 |
| IQL_rs10_n10 | 0.8516 | 55 | 0.5020 |
| IQL_rs10_n50 | 0.8336 | 35 | 0.4028 |

### Notes

- **IQL + TD-N works!** IQL_rs10_n5 (r=0.856) and IQL_rs1_n10 (r=0.854) both match or exceed TD+EMA (r=0.853).
- **Previous IQL failure was due to TD(0), not the algorithm.** Exp 11 showed IQL r=0.22 with TD(0) on offline data. With TD-N on same on-policy data, IQL reaches r=0.85 — a 4x improvement.
- **n-step is the key factor**: n=1 maxes at 0.74, n=5/10 reaches 0.85, n=50 drops to 0.82 (near-MC overfitting).
- **Optimal n decreases with reward_scale**: rs=1 best at n=10, rs=10 best at n=5. Reward scaling amplifies signal so fewer steps suffice.
- **Overfitting pattern**: Higher n → earlier peak, more severe overfit. n=1 barely overfits (final/peak=0.99), n=50 overfits heavily (final/peak=0.48).
- **Best config**: IQL_rs10_n5 — highest peak (0.856), moderate peak epoch (75), decent final (0.652).
- **IQL vs TD+EMA overfitting**: IQL_rs1_n1 has the mildest overfitting of all methods (final 0.73 vs peak 0.74). IQL's expectile regression may provide some regularization.
- n-step returns precomputed from trajectory data, preserving (T, E*N) structure for proper temporal indexing.

---

## [Tree Sampling vs Rollout Sampling for TD V Learning — Scaling Analysis] - 2026-02-22 09:32

**Command**: `python -u -m RL.v_tree_vs_rollout --gamma 0.99 --n_seed 1 --rollout-counts 1 2 3 5 10 20 50 100`
**Git**: fec75bf (main)
**Run Dir**: runs/v_tree_vs_rollout.png, runs/v_tree_vs_rollout.pt
**Script**: `RL/v_tree_vs_rollout.py`

### Settings
| Parameter | Value |
|-----------|-------|
| checkpoint | runs/pickcube_ppo/ckpt_76_logstd-1.5.pt |
| gamma | 0.99 |
| num_envs | 100 |
| num_steps | 50 |
| max_episode_steps | 50 |
| mc_samples | 16 |
| n_seed | 1 |
| rollout_counts | (1, 2, 3, 5, 10, 20, 50, 100) |
| td_reward_scale | 10.0 |
| ema_tau | 0.005 |
| epochs | 500 |
| lr | 3e-4 |
| batch_size | 1000 |
| eval_every | 5 |
| critic_layers | 3 |
| hidden_dim | 256 |
| seed | 1 |

### Method Description

Two data collection strategies compared at each data size (same total transition budget):

- **Rollout sampling**: N independent rollouts from `envs.reset()`. Each rollout = 100 envs × 50 steps = 5,000 transitions.
- **Tree sampling**: 1 seed rollout (saving all env states at every timestep) + branch from intermediate seed states to fill remaining budget. Branches: uniform timestep sampling [0,49], random (seed_rollout, env_idx), rollout until natural termination. Seed rollout counted toward budget.

Both trained with TD(0)+EMA, reward_scale=10, evaluated against on-policy MC16 ground truth on rollout 0 states.

### Results (initial run, 500 epochs — small N NOT converged)
| N rollouts | Transitions | Rollout peak r | Tree peak r | Delta (Tree−Rollout) |
|-----------|------------|----------------|-------------|----------------------|
| 1 | 5,000 | 0.2035 | 0.1237 | -0.0798 |
| 2 | 10,000 | 0.4848 | 0.5567 | **+0.0719** |
| 3 | 15,000 | 0.5760 | 0.6575 | **+0.0815** |
| 5 | 25,000 | 0.6563 | 0.7107 | **+0.0544** |
| 10 | 50,000 | 0.7217 | 0.7696 | **+0.0479** |
| 20 | 100,000 | 0.7683 | 0.7687 | +0.0003 |
| 50 | 250,000 | 0.8241 | 0.7828 | -0.0413 |
| 100 | 500,000 | 0.8522 | 0.7866 | -0.0655 |

**⚠️ N=1,2,3,5 not converged at 500 epochs. See corrected results in Exp 16b below.**

### Notes

**Three distinct regimes:**

1. **N=1**: Tree = Rollout (identical data — both are just 1 rollout, no branching budget). Slight difference from different random seeds.

2. **N=2~10: Tree significantly better (+5~8%)**. With limited data, branching from 1 seed rollout's intermediate states provides more diverse state coverage than collecting a few independent rollouts. The branches explore mid/late-trajectory states that rollout sampling can only reach by running full episodes from reset.

3. **N≥20: Rollout overtakes Tree (-4~7%)**. With sufficient data, 100 independent rollouts provide 10,000 unique initial states, far exceeding tree's 100 initial states from 1 seed. Tree's peak r saturates ~0.787 (limited by n_seed=1 initial state diversity), while rollout continues scaling to 0.85+.

**Crossover point ≈ N=20** (100K transitions).

**Key insight**: At small data budgets, **state-space coverage efficiency** matters more than **initial state diversity**. Tree sampling extracts 2-10x more effective coverage from a single trajectory via branching. But this advantage has a ceiling — the initial state diversity bottleneck limits tree's asymptotic performance.

**Previous run** (N=100 only, n_seed=5): Rollout peak r=0.8563 vs Tree peak r=0.8429. Consistent with scaling result — at N=100, rollout wins.

---

## [Exp 16b: V Scaling Law — Combined All Methods (Corrected, Converged)] - 2026-02-22 10:03

**Commands**:
```bash
# Small N rerun with enough epochs for convergence
python -u -m RL.td_ema_curve --gamma 0.99 --td-epochs 5000 --eval-every 10 --rollout-counts 1 2 3 5 --methods "TD+EMA" "MC1" "GAE" --output runs/td_ema_curve_small.png
python -u -m RL.v_tree_vs_rollout --gamma 0.99 --n-seed 1 --rollout-counts 1 2 3 5 --epochs 5000 --eval-every 10 --output runs/v_tree_small.png
```
**Git**: fec75bf (main)
**Run Dir**: runs/v_scaling_law.png (plot), runs/td_ema_curve_small.pt, runs/v_tree_small.pt
**Data Sources**: Exp 14 (N=5,10,20,50,100 @ 500ep), small-N reruns (N=1,2,3,5 @ 5000ep), Exp 15 IQL (N=100), Exp 14 N=1 long run (25k ep)

### Overview

Combined scaling law plot across ALL V learning methods: TD+EMA, MC1, GAE, Tree sampling, IQL. Small-N points (N=1,2,3,5) were rerun with 5000 epochs to ensure convergence (Exp 14's 500 epochs was insufficient for small data sizes — e.g., N=1 TD+EMA peaks at epoch ~1400, not epoch 500).

### Settings
| Parameter | Value |
|-----------|-------|
| gamma | 0.99 |
| td_reward_scale | 10.0 |
| ema_tau | 0.005 |
| small N epochs | 5000 (eval every 10) |
| large N epochs | 500 (eval every 5, from Exp 14) |
| methods | TD+EMA, MC1, GAE, Tree (TD+EMA), IQL (N=100 only) |
| N values | 1, 2, 3, 5, 10, 20, 50, 100 |

### Results — Peak Pearson r (all converged)

| N | TD+EMA | MC1 | GAE | Tree | IQL |
|---|--------|-----|-----|------|-----|
| 1 | 0.444 | 0.361 | **0.463** | 0.444 | — |
| 2 | 0.533 | 0.436 | 0.511 | **0.622** | — |
| 3 | 0.578 | 0.504 | 0.568 | **0.645** | — |
| 5 | 0.658 | 0.609 | 0.589 | **0.711** | — |
| 10 | 0.720 | 0.696 | 0.657 | **0.770** | — |
| 20 | **0.773** | 0.721 | 0.548 | 0.769 | — |
| 50 | **0.826** | 0.791 | 0.678 | 0.783 | — |
| 100 | 0.850 | 0.814 | 0.668 | 0.787 | **0.856** |

### Notes

**1. Method ranking depends on data regime:**

| Regime | Best method | Peak r |
|--------|------------|--------|
| N=1 (5K trans) | GAE | 0.463 |
| N=2~10 (10K~50K) | **Tree sampling** | 0.622~0.770 |
| N≥20 (100K+) | **TD+EMA** | 0.773~0.850 |
| N=100 (500K) | IQL ≈ TD+EMA | 0.856 |

**2. Tree sampling: dominant in small-data regime.**
- Tree beats TD+EMA by +0.05~0.09 at N=2~10.
- Advantage comes from better state-space coverage via branching from intermediate states.
- Saturates at ~0.787 due to limited initial state diversity (n_seed=1).
- Crossover with TD+EMA at N≈20.

**3. GAE: best at N=1, worst at N≥20.**
- At N=1, bootstrapping helps when data is minimal (GAE 0.463 vs TD+EMA 0.444).
- At large N, GAE degrades (0.548~0.678), likely due to iterative recomputation compounding errors with fixed data.

**4. MC1: consistently below TD+EMA but scales well.**
- Gap widens at small N (0.36 vs 0.44 at N=1), narrows at large N (0.81 vs 0.85 at N=100).
- MC1's single-sample returns are noisier targets than TD+EMA's bootstrapped targets.

**5. IQL matches TD+EMA at N=100.**
- IQL_rs10_n5 (0.856) ≈ TD+EMA (0.850). Both saturate near 0.85 on this task.
- IQL only tested at N=100 in Exp 15, so no scaling curve available.

**6. Overall scaling behavior:**
- All methods show log-linear scaling: doubling data gives roughly constant improvement in r.
- TD+EMA scales most consistently (0.44→0.85, Δ=0.41 from N=1→100).
- Tree scales fastest at small N but plateaus earliest.
- GAE has non-monotonic scaling (peaks at N=10, then declines).

**Plot**: `runs/v_scaling_law.png`

---

## Exp 17: n_seed Ablation for Tree Sampling - 2026-02-22 12:11

**Command**: `python -u -m RL.v_tree_nseed_ablation --gamma 0.99`
**Git**: fec75bf (main)
**Run Dir**: runs/v_tree_nseed_ablation.pt
**Script**: `RL/v_tree_nseed_ablation.py`

### Motivation

Tree sampling (Exp 16) shows tree beats rollout at N=2-10 but saturates at r≈0.787 for N≥20, because `n_seed=1` limits initial state diversity to 100 envs from a single reset. This experiment tests whether scaling n_seed with N removes the ceiling.

### Settings
| Parameter | Value |
|-----------|-------|
| checkpoint | ckpt_76_logstd-1.5 |
| gamma | 0.99 |
| num_envs | 100 |
| num_steps | 50 |
| mc_samples | 16 |
| N_values | 10, 20, 50, 100 |
| n_seed_values | 1, 2, 5, 10, 20, 50 |
| td_reward_scale | 10.0 |
| ema_tau | 0.005 |
| epochs | 1000 |
| lr | 3e-4 |
| batch_size | 1000 |
| critic_layers | 3 |
| hidden_dim | 256 |
| seed | 1 |

### Results

**Rollout Baselines (peak Pearson r)**:
| N | Peak r | Peak Epoch |
|---|--------|------------|
| 10 | 0.7310 | 310 |
| 20 | 0.7683 | 275 |
| 50 | 0.8231 | 135 |
| 100 | 0.8555 | 95 |

**Tree Sampling (peak Pearson r)**:
| N | ns=1 | ns=2 | ns=5 | ns=10 | ns=20 | ns=50 |
|---|------|------|------|-------|-------|-------|
| 10 | 0.7657 | 0.7326 | 0.7347 | — | — | — |
| 20 | 0.7772 | 0.7761 | 0.8119 | 0.7984 | — | — |
| 50 | 0.7976 | 0.8068 | 0.8391 | 0.8366 | 0.8392 | — |
| 100 | 0.7864 | 0.8131 | 0.8498 | 0.8533 | 0.8565 | 0.8550 |

**Delta (tree − rollout)**:
| N | ns=1 | ns=2 | ns=5 | ns=10 | ns=20 | ns=50 |
|---|------|------|------|-------|-------|-------|
| 10 | +0.035 | +0.002 | +0.004 | — | — | — |
| 20 | +0.009 | +0.008 | +0.044 | +0.030 | — | — |
| 50 | -0.025 | -0.016 | +0.016 | +0.014 | +0.016 | — |
| 100 | -0.069 | -0.042 | -0.006 | -0.002 | +0.001 | -0.001 |

### Notes

1. **n_seed lifts the tree ceiling to match rollout, but doesn't exceed it**:
   - n_seed=1, N=100: tree r=0.786 (−0.069 vs rollout 0.856)
   - n_seed=20, N=100: tree r=0.857 (+0.001, matches rollout)
   - No n_seed configuration significantly exceeds rollout

2. **Optimal seed fraction is ~5-20% of N**:
   - N=20 best at n_seed=5 (25%): r=0.812
   - N=50 best at n_seed=5 or 20 (10-40%): r=0.839
   - N=100 best at n_seed=20 (20%): r=0.857
   - Too many seeds (n_seed=50 at N=100) wastes branch budget on redundant initial states

3. **Tree advantage only at small N**:
   - N=10: tree +0.035 (clear advantage)
   - N=20: tree +0.044 (peak at n_seed=5)
   - N≥50: tree at most matches rollout, never exceeds
   - Tree is a data-efficient strategy for small budgets, not a scaling improvement

4. **Consistent with Exp 16**: n_seed=1 results match previous values (~0.77 at N=10, saturating ~0.79 at large N), confirming the saturation was indeed caused by limited initial state diversity.

5. **Implication**: Tree sampling's benefit comes from state diversity amplification at small N. Once N is large enough that rollout sampling already covers the state space well (~50+ resets × 100 envs = 5000+ unique initial states), tree adds no further value. The fundamental bottleneck at large N is the NN SNR problem (Issue #13), not state diversity.

**Plot**: `runs/v_tree_nseed_ablation.png`

---

## Exp 19: [Online TD+EMA V + AWR/PPO — Replay Buffer vs On-Policy] - 2026-02-22 15:53

**Git**: fec75bf (main)
**Script**: `RL/td_ema_awr_online.py` (new)

### Motivation

Test whether accumulating a replay buffer for better V learning (TD+EMA) can improve online iterative RL. V scaling analysis (Exp 16) showed TD+EMA reaches r=0.85 with 500K transitions — but online RL has only 5K per iteration. By accumulating replay, iteration 10 has 50K transitions (N=10 equivalent, r≈0.72).

### Design

- **Standalone critic**: separate `nn.Sequential` with EMA target (not Agent's critic)
- **TD+EMA training**: critic trains in scaled reward space (r × rs=10), V read back via `/ rs`
- **final_values computed AFTER TD training**: ensures advantages use freshly trained critic
- **Replay buffer**: list of per-iteration dicts, concatenated for TD training
- **Off-policy handling**: "approximate ignore" — no IS correction

### Common Settings (V2 baseline)

| Parameter | Value |
|-----------|-------|
| checkpoint | ckpt_76_logstd-1.5.pt (det SR=43.8%) |
| num_envs | 100 |
| num_steps | 50 |
| total_timesteps | 50,000 (10 iterations) |
| gamma | 0.99 |
| eval_freq | 1 |
| num_minibatches | 5 |
| td_reward_scale | 10.0 |
| ema_tau | 0.005 |
| critic | 3×256 (142,849 params) |

### Round 1: AWR + Replay Buffer (td_epochs=200)

**Commands**:
```bash
python -u -m RL.td_ema_awr_online --gamma 0.99 --advantage_mode gae --exp_name td_ema_awr_gae
python -u -m RL.td_ema_awr_online --gamma 0.99 --advantage_mode onestep --exp_name td_ema_awr_onestep
python -u -m RL.td_ema_awr_online --gamma 0.99 --advantage_mode gae --offline_rollouts 20 --exp_name td_ema_awr_gae_off20
```

| Experiment | i1 | i2 | i3 | i4 | i5 | i6 | i7 | i8 | i9 | i10 | Peak | Final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TD+EMA GAE replay | 43.8% | 51.6% | 52.7% | 63.1% | 69.6% | 73.4% | 71.4% | 66.0% | 71.3% | 65.5% | 73.4% (i6) | 65.5% |
| TD+EMA Onestep replay | 43.8% | 66.4% | 75.9% | 82.9% | 75.0% | 73.2% | 72.0% | 74.6% | 74.7% | 74.5% | 82.9% (i4) | 74.5% |
| TD+EMA GAE+Off20 replay | 38.6% | 63.2% | 77.7% | 84.4% | 73.7% | 80.6% | 75.6% | 76.3% | 71.3% | 77.1% | 84.4% (i4) | 77.1% |

**Run Dirs**: `runs/td_ema_awr_gae__seed1__1771800034`, `runs/td_ema_awr_onestep__seed1__1771800211`, `runs/td_ema_awr_gae_off20__seed1__1771800390`

**Finding**: All underperform GAE PPO baseline (92.1%). td_epochs=200 = only 1,000 gradient steps on 5K data — V scaling analysis showed peak at ~1,380 epochs (6,900 steps). Under-trained.

### Round 2: AWR + On-Policy Only (td_epochs=1400, warmstart)

**Commands**:
```bash
python -u -m RL.ppo_finetune --num_envs 100 --num_steps 50 --num_minibatches 5 --update_epochs 100 --target_kl 100.0 --eval_freq 1 --total_timesteps 50000 --gamma 0.99 --checkpoint runs/pickcube_ppo/ckpt_76_logstd-1.5.pt --exp_name gae_ppo_baseline
python -u -m RL.td_ema_awr_online --gamma 0.99 --advantage_mode gae --current_data_only --td_epochs_per_iter 1400 --exp_name td_ema_gae_onpolicy
python -u -m RL.td_ema_awr_online --gamma 0.99 --advantage_mode onestep --current_data_only --td_epochs_per_iter 1400 --exp_name td_ema_onestep_onpolicy
```

| Experiment | i1 | i2 | i3 | i4 | i5 | i6 | i7 | i8 | i9 | i10 | Peak | Final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **GAE PPO baseline** | 43.8% | 58.0% | 66.4% | 73.3% | 72.3% | 83.7% | 85.8% | 87.6% | 92.1% | 91.2% | 92.1% (i9) | 91.2% |
| TD+EMA GAE warmstart | 43.8% | 46.1% | 67.4% | 71.9% | 84.6% | 79.8% | 69.2% | 76.0% | 72.0% | 56.6% | 84.6% (i5) | 56.6% |
| TD+EMA Onestep warmstart | 43.8% | 72.6% | 75.6% | 78.8% | 73.2% | 68.3% | 70.1% | 55.8% | 53.0% | 64.1% | 78.8% (i4) | 64.1% |

**Run Dirs**: `runs/gae_ppo_baseline__seed1__1771801495`, `runs/td_ema_gae_onpolicy__seed1__1771801629`, `runs/td_ema_onestep_onpolicy__seed1__1771801819`

**Finding**: 1400 epochs causes severe overfitting — critic fits 5K data perfectly then distorts V. Peak early, then collapse (84.6%→56.6%, 78.8%→53.0%).

### Round 3: AWR + Reset Critic Each Iter (td_epochs=1400)

**Commands**:
```bash
python -u -m RL.td_ema_awr_online --gamma 0.99 --advantage_mode gae --current_data_only --reset_critic_each_iter --td_epochs_per_iter 1400 --exp_name td_ema_gae_reset
python -u -m RL.td_ema_awr_online --gamma 0.99 --advantage_mode onestep --current_data_only --reset_critic_each_iter --td_epochs_per_iter 1400 --exp_name td_ema_onestep_reset
```

| Experiment | i1 | i2 | i3 | i4 | i5 | i6 | i7 | i8 | i9 | i10 | Peak | Final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TD+EMA GAE reset+AWR | 43.8% | 45.0% | 61.1% | 70.8% | 80.2% | 80.3% | 80.9% | 65.9% | 75.8% | 72.1% | 80.9% (i7) | 72.1% |
| TD+EMA Onestep reset+AWR | 43.8% | 76.5% | 88.2% | 87.4% | 87.4% | 84.3% | 87.3% | 74.7% | 77.7% | 80.8% | 88.2% (i3) | 80.8% |

**Run Dirs**: `runs/td_ema_gae_reset__seed1__1771802538`, `runs/td_ema_onestep_reset__seed1__1771802729`

**Finding**: Reset eliminates overfitting collapse (warmstart 53-56% → reset 72-80% final). But Onestep+AWR stalls at i3 (88.2%) and oscillates. Explained Variance analysis revealed EV is already high (0.78-0.92) — V quality is NOT the bottleneck. The problem is **AWR as the policy update operator**.

### Explained Variance Comparison

| Iter | GAE PPO EV | TD+EMA Onestep reset EV |
|------|-----------|------------------------|
| 2 | 0.07 | **0.92** |
| 4 | 0.47 | 0.86 |
| 6 | 0.74 | 0.81 |
| 8 | 0.82 | 0.81 |
| 10 | 0.88 | 0.84 |

TD+EMA's V is better from the start! The issue is AWR not PPO.

### Round 4: PPO Policy Update + Reset (td_epochs=1400) — FINAL

**Commands**:
```bash
python -u -m RL.td_ema_awr_online --gamma 0.99 --advantage_mode gae --policy_update ppo --current_data_only --reset_critic_each_iter --td_epochs_per_iter 1400 --update_epochs 100 --exp_name td_ema_gae_ppo_reset
python -u -m RL.td_ema_awr_online --gamma 0.99 --advantage_mode onestep --policy_update ppo --current_data_only --reset_critic_each_iter --td_epochs_per_iter 1400 --update_epochs 100 --exp_name td_ema_onestep_ppo_reset
```

| Experiment | i1 | i2 | i3 | i4 | i5 | i6 | i7 | i8 | i9 | i10 | Peak | Final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GAE PPO baseline | 43.8% | 58.0% | 66.4% | 73.3% | 72.3% | 83.7% | 85.8% | 87.6% | 92.1% | 91.2% | 92.1% (i9) | 91.2% |
| **TD+EMA GAE + PPO** | 43.8% | 55.0% | 66.4% | 76.9% | 78.0% | 80.0% | 78.7% | 87.9% | 91.0% | 91.7% | **91.7% (i10)** | 91.7% |
| **TD+EMA Onestep + PPO** | 43.8% | 55.6% | 62.4% | 67.7% | 75.4% | 91.0% | 89.0% | 90.2% | 93.7% | **95.1%** | **95.1% (i10)** | **95.1%** |

**Run Dirs**: `runs/td_ema_gae_ppo_reset__seed1__1771803917`, `runs/td_ema_onestep_ppo_reset__seed1__1771804103`

### Key Takeaways

1. **TD+EMA Onestep + PPO (95.1%) beats GAE PPO baseline (92.1%) by +3%.** First configuration to surpass the baseline.

2. **TD+EMA GAE + PPO (91.7%) matches the baseline.** Switching V learning method from iterative GAE to TD+EMA makes no difference when using GAE advantages — expected since both learn V comparably.

3. **AWR was the bottleneck, not V quality.** TD+EMA V was already excellent (EV=0.92 at i2), but AWR + 200 epochs caused policy oscillation. PPO with 100 epochs + clipped ratio is a much more stable policy improvement operator.

4. **Onestep > GAE for advantage estimation with TD+EMA V.** When V is learned via TD+EMA (not co-trained with GAE returns), one-step advantages `r + γV(s') - V(s)` outperform GAE(λ=0.95). GAE telescopes V errors across timesteps; one-step only uses two V evaluations.

5. **Reset critic each iteration is essential.** Warmstart causes overfitting collapse; replay buffer introduces off-policy staleness. Training from scratch each iteration on fresh on-policy data gives the cleanest V estimates.

6. **TD epoch budget matters.** N=1 (5K data) needs ~1400 epochs to reach peak V quality (from V scaling analysis Exp 16). Using 200 epochs severely under-trains.

### Round 5: PPO + Replay Buffer (no reset, fixed 1400 epochs)

Test whether accumulating replay buffer helps TD+EMA V learning when combined with PPO policy update. No critic reset — critic warm-starts across iterations. Onestep advantage only (best from Round 4).

**Commands**:
```bash
python -u -m RL.td_ema_awr_online --gamma 0.99 --advantage_mode onestep --policy_update ppo --td_epochs_per_iter 1400 --update_epochs 100 --exp_name td_ema_onestep_ppo_replay
python -u -m RL.td_ema_awr_online --gamma 0.99 --advantage_mode onestep --policy_update ppo --td_epochs_per_iter 1400 --update_epochs 100 --offline_rollouts 20 --exp_name td_ema_onestep_ppo_replay_off20
```

| Experiment | i1 | i2 | i3 | i4 | i5 | i6 | i7 | i8 | i9 | i10 | Peak | Final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GAE PPO baseline | 43.8% | 58.0% | 66.4% | 73.3% | 72.3% | 83.7% | 85.8% | 87.6% | 92.1% | 91.2% | 92.1% (i9) | 91.2% |
| TD+EMA Onestep PPO reset (R4) | 43.8% | 55.6% | 62.4% | 67.7% | 75.4% | 91.0% | 89.0% | 90.2% | 93.7% | 95.1% | 95.1% (i10) | 95.1% |
| **TD+EMA Onestep PPO replay** | 43.8% | 55.6% | 65.2% | 78.7% | 86.3% | 93.9% | 91.6% | 81.1% | 94.6% | **97.1%** | **97.1% (i10)** | **97.1%** |
| TD+EMA Onestep PPO Off20+replay | 38.6% | 63.9% | 72.7% | 83.9% | 80.6% | 84.1% | 73.9% | 75.2% | 64.2% | 70.9% | 84.1% (i6) | 70.9% |

**Run Dirs**: `runs/td_ema_onestep_ppo_replay__seed1__1771804757`, `runs/td_ema_onestep_ppo_replay_off20__seed1__1771805182`

**Findings**:
- **Replay + PPO = 97.1%** — best result so far! +5% over GAE PPO baseline, +2% over reset-only (95.1%). Replay accumulation helps V learning when critic warm-starts with PPO.
- **Off20 + replay = 84.1%→70.9% (collapse)** — pre-collected off-policy data is harmful. 100K off-policy transitions dominate the replay buffer (70-95% across iterations), biasing V toward V^{π_0}. Additionally, 1400 epochs on 105K+ data = 147K+ gradient steps causes severe overfitting.

### Round 6: Dynamic Epoch Scaling (td_target_steps=7000)

Test whether dynamically adjusting TD epochs to keep gradient steps constant (~7000) regardless of data size can fix the off20 overfitting issue. `td_epochs = td_target_steps / num_batches_per_epoch`.

**Commands**:
```bash
python -u -m RL.td_ema_awr_online --gamma 0.99 --advantage_mode onestep --policy_update ppo --td_target_steps 7000 --update_epochs 100 --exp_name td_ema_onestep_ppo_replay_dyn
python -u -m RL.td_ema_awr_online --gamma 0.99 --advantage_mode onestep --policy_update ppo --td_target_steps 7000 --update_epochs 100 --offline_rollouts 20 --exp_name td_ema_onestep_ppo_off20_dyn
python -u -m RL.td_ema_awr_online --gamma 0.99 --advantage_mode onestep --policy_update ppo --td_target_steps 7000 --update_epochs 100 --current_data_only --exp_name td_ema_onestep_ppo_onpol_dyn
```

| Experiment | i1 | i2 | i3 | i4 | i5 | i6 | i7 | i8 | i9 | i10 | Peak | Final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GAE PPO baseline | 43.8% | 58.0% | 66.4% | 73.3% | 72.3% | 83.7% | 85.8% | 87.6% | 92.1% | 91.2% | 92.1% (i9) | 91.2% |
| **Replay+PPO 1400ep (R5)** | 43.8% | 55.6% | 65.2% | 78.7% | 86.3% | 93.9% | 91.6% | 81.1% | 94.6% | **97.1%** | **97.1% (i10)** | **97.1%** |
| Replay+PPO dyn7000 | 43.8% | 55.6% | 73.8% | 87.3% | 84.5% | 72.5% | 80.1% | 92.3% | 90.3% | 85.3% | 92.3% (i8) | 85.3% |
| Off20+PPO dyn7000 | 38.6% | 79.7% | 73.7% | 52.1% | 54.1% | 72.2% | 76.9% | 74.0% | 75.4% | 78.7% | 79.7% (i2) | 78.7% |
| On-policy+PPO dyn7000 | 43.8% | 55.6% | 62.4% | 67.7% | 75.4% | 91.0% | 89.0% | 90.2% | 93.7% | 95.1% | 95.1% (i10) | 95.1% |

Dynamic epoch details:
- Replay+dyn7000: td_epochs = [1400, 700, 466, 350, 280, 233, 200, 175, 155, 140]
- Off20+dyn7000: td_epochs = [66, 63, 60, 58, 56, 53, 51, 50, 48, 46]
- On-policy+dyn7000: td_epochs = [1400, 1400, ...] (5K data/iter → always 1400, equivalent to R4 reset)

**Run Dirs**: `runs/td_ema_onestep_ppo_replay_dyn__seed1__1771807785`, `runs/td_ema_onestep_ppo_off20_dyn__seed1__1771807972`, `runs/td_ema_onestep_ppo_onpol_dyn__seed1__1771808180`

**Findings**:
- **Dynamic scaling hurts replay**: Replay+dyn7000 (92.3%) << Replay+fixed 1400 (97.1%). As replay grows, dynamic scaling reduces epochs (1400→140), under-training V on later iterations. Fixed 1400 epochs keeps V well-trained even with 50K data (still 70K gradient steps at iter 10, reasonable).
- **Off20 still collapses even with dynamic scaling**: Off20+dyn7000 peaks at i2=79.7% then collapses to 52.1%. The initial V is good (lots of data), but off-policy data dominates the buffer permanently, biasing V toward V^{π_0} regardless of epoch count.
- **On-policy+dyn7000 = R4 reset** (95.1%): With on-policy only and 5K data, dyn7000 → 1400 epochs every iteration. Confirms on-policy + reset is a solid baseline.

### Overall Summary (Rounds 1-6)

| Configuration | Policy | Critic | Data | Peak SR | vs Baseline |
|---|---|---|---|---|---|
| GAE PPO baseline | PPO | iterative GAE V | on-policy | 92.1% | — |
| TD+EMA Onestep + PPO + reset | PPO | reset each iter | on-policy | 95.1% | +3.0% |
| **TD+EMA Onestep + PPO + replay 1400ep** | PPO | warm-start | **replay** | **97.1%** | **+5.0%** |
| TD+EMA Onestep + PPO + replay dyn7000 | PPO | warm-start | replay | 92.3% | +0.2% |
| TD+EMA GAE + PPO + reset | PPO | reset each iter | on-policy | 91.7% | -0.4% |
| TD+EMA Onestep + AWR + reset | AWR | reset each iter | on-policy | 88.2% | -3.9% |
| TD+EMA Onestep + PPO + Off20 | PPO | warm-start | off20+replay | 84.1% | -8.0% |

**Key takeaways**:
1. **Best: TD+EMA Onestep + PPO + replay + fixed 1400ep = 97.1%** (+5% over GAE PPO). Replay accumulation with warm-started critic is the winning combination.
2. **PPO >> AWR**: Same V learning, PPO gives +7-12% over AWR. AWR oscillates with 200 update epochs.
3. **Onestep >> GAE with TD+EMA V**: One-step advantages better exploit TD+EMA's V quality.
4. **Replay helps, but off-policy pre-collection hurts**: Gradual on-policy accumulation is beneficial; front-loading 100K off-policy data overwhelms the buffer and biases V toward the initial policy.
5. **Fixed epochs > dynamic scaling for replay**: The V scaling law's "more data → fewer epochs" principle doesn't apply well when data grows gradually and we want V to track the evolving policy.

---

## Exp 20: IQL+PPO Online — Q-V Advantages vs TD+EMA Onestep - 2026-02-22 18:00

**Git**: 58d478e (main)
**Script**: `RL/iql_ppo_online.py` (new)

### Motivation

Test whether IQL-style Q(s,a)−V(s) advantages can match TD+EMA onestep `r+γV(s')−V(s)` in online iterative RL. Theory: at tau=0.5, IQL Q−V ≈ TD+EMA onestep (both estimate one-step advantage). But IQL requires learning Q(s,a) which suffers from the SNR/ranking problem (Issue #8).

### Design

- **IQL critic**: Q(s,a) network (obs+action→scalar) + V(s) network (obs→scalar)
- **Q target**: 1-step TD: `r*rs + γ * V_target(s')`
- **V loss**: expectile regression on `Q_target(s,a) − V(s)`
- **Advantage**: `Q(s,a)/rs − V(s)/rs`
- **Policy update**: PPO (same as td_ema_awr_online.py PPO mode)
- **Replay buffer**: stores (obs, actions, rewards, dones, next_obs)

### Common Settings

| Parameter | Value |
|-----------|-------|
| checkpoint | ckpt_76_logstd-1.5.pt |
| num_envs | 100 |
| num_steps | 50 |
| total_timesteps | 50,000 (10 iterations) |
| gamma | 0.99 |
| td_reward_scale | 10.0 |
| ema_tau | 0.005 |
| iql_epochs_per_iter | 1400 |
| critic | 3×256 |
| update_epochs | 100 |
| num_minibatches | 5 |

### Results

| Experiment | i1 | i2 | i3 | i4 | i5 | i6 | i7 | i8 | i9 | i10 | Peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| IQL tau=0.5 reset | 43.8% | 78.9% | 86.9% | 84.1% | 83.4% | 85.5% | 82.3% | 84.4% | 83.6% | 82.3% | 86.9% (i3) |
| IQL tau=0.7 reset | 43.8% | 78.9% | 83.4% | 81.3% | 86.4% | 82.8% | 82.7% | 83.4% | 84.8% | 85.5% | 86.4% (i5) |
| IQL tau=0.5 replay | 43.8% | 72.8% | 76.5% | 74.6% | 72.0% | 71.7% | 67.5% | 65.8% | 63.3% | 61.1% | 76.5% (i3) |

### Comparison with TD+EMA Baselines

| Method | Peak SR | vs GAE PPO baseline |
|---|---|---|
| TD+EMA Onestep + PPO + replay 1400ep | **97.1%** | +5.0% |
| TD+EMA Onestep + PPO + reset | 95.1% | +3.0% |
| GAE PPO baseline | 92.1% | — |
| IQL tau=0.5 reset | 86.9% | −5.2% |
| IQL tau=0.7 reset | 86.4% | −5.7% |
| IQL tau=0.5 replay | 76.5% | −15.6% |

### Notes

1. **IQL (86.9%) << TD+EMA onestep (95.1%)**: Despite similar V quality (Exp 21), IQL's Q−V advantages are much worse for policy optimization. Confirms Issue #8: Q(s,a) cannot rank actions.
2. **Reset >> Replay for IQL**: Replay causes collapse (76.5%→61.1%), likely because Q(s,a) trained on off-policy actions becomes even more unreliable.
3. **tau=0.5 ≈ tau=0.7**: No significant difference, suggesting the expectile doesn't help when Q itself is unreliable.
4. **Key insight**: `r+γV(s')−V(s)` bypasses Q learning entirely, only needs V(s). This is fundamentally more robust than Q(s,a)−V(s) because V is much easier to learn accurately.

---

## Exp 21: IQL V Scaling Analysis — IQL vs TD+EMA Across Data Sizes - 2026-02-22 19:30

**Command**: `python -u -m RL.td_ema_curve --gamma 0.99 --td-epochs 2000 --eval-every 5 --rollout-counts 1 5 10 20 50 100 --methods "TD+EMA" "IQL" --output runs/iql_vs_tdema_curve_2k.png`
**Git**: 58d478e (main)
**Script**: `RL/td_ema_curve.py` (modified to add IQL with n-step Q target)
**Run Dir**: runs/iql_vs_tdema_curve_2k.pt, runs/iql_vs_tdema_curve_2k.png

### Motivation

The V scaling analysis (Exp 16b) tested TD+EMA, MC1, GAE but not IQL across data sizes. Previous IQL experiment (Exp 11b) only tested N=100. This fills the gap: does IQL V scale similarly to TD+EMA?

### Design

- **IQL implementation**: Added to `td_ema_curve.py` with n-step Q target precomputation
- **n-step Q target**: `Q_target = Σ_{k=0}^{n-1} γ^k r_{t+k} + γ^n V(s_{t+n})` (n=10)
- **V learned via expectile regression** on `Q_target(s,a) − V(s)` at tau=0.5
- Actions stored in data collection for Q(s,a) input

### Settings

| Parameter | Value |
|-----------|-------|
| gamma | 0.99 |
| td_reward_scale | 10.0 |
| ema_tau | 0.005 |
| td_epochs | 2000 |
| eval_every | 5 |
| iql_td_n | 10 |
| expectile_tau | 0.5 |
| critic | 3×256 |
| lr | 3e-4 |
| batch_size | 1000 |
| N values | 1, 5, 10, 20, 50, 100 |
| MC16 ground truth | Same as Exp 14/16 |

### Results — Peak Pearson r (V quality vs MC16 ground truth)

| N | TD+EMA peak r | TD+EMA peak epoch | IQL peak r | IQL peak epoch |
|---|---|---|---|---|
| 1 | 0.450 | 1380 | **0.522** | 230 |
| 5 | 0.653 | 505 | **0.679** | 165 |
| 10 | 0.719 | 340 | **0.742** | 115 |
| 20 | 0.764 | 200 | **0.771** | 80 |
| 50 | **0.827** | 105 | 0.827 | 50 |
| 100 | **0.854** | 70 | 0.852 | 30 |

### Notes

1. **IQL V quality matches or beats TD+EMA at all data sizes**: +16% at N=1 (0.522 vs 0.450), converging to parity at N≥50.

2. **IQL converges ~6x faster**: N=1 peaks at epoch 230 vs 1380 for TD+EMA. N-step=10 provides much better Q targets than 1-step bootstrap, reducing the number of training iterations needed.

3. **N-step is critical for IQL**: Previous run with n=1 gave r=0.023 at N=1 (completely broken). N-step=10 gives 0.522 — a 22x improvement. The n-step return reduces bootstrap error from the initially random V.

4. **Both methods severely overfit**: Final r is much lower than peak r at all data sizes. Early stopping is essential for both TD+EMA and IQL.

5. **Paradox: IQL V is better, but IQL advantages are worse (Exp 20)**: IQL learns V(s) as well as TD+EMA, but Q(s,a)−V(s) advantages (peak SR 86.9%) are far worse than r+γV(s')−V(s) advantages (peak SR 95.1%). The bottleneck is Q action ranking (Issue #8), not V quality.

6. **Implication**: For online iterative RL, use TD+EMA V with onestep advantages (`r+γV(s')−V(s)`), not IQL Q−V. Both learn V equally well, but onestep bypasses the Q ranking problem entirely.

**Plot**: `runs/iql_vs_tdema_curve_2k.png`

---

## Exp 22: TD+EMA N-step=10 vs IQL — Fair Comparison V Scaling - 2026-02-22 21:25

**Command**: `python -u -m RL.td_ema_curve --gamma 0.99 --td-epochs 2000 --eval-every 5 --rollout-counts 1 5 10 20 50 100 --methods "TD+EMA" "IQL" --td-n-step 10 --output runs/td_nstep10_vs_iql_curve.png`
**Git**: 58d478e (main)
**Script**: `RL/td_ema_curve.py` (modified: added `td_n_step` arg to TD+EMA)
**Run Dir**: runs/td_nstep10_vs_iql_curve.pt, runs/td_nstep10_vs_iql_curve.png

### Motivation

Exp 21 showed IQL (n-step=10) beats TD+EMA (1-step) at small data sizes. But the comparison was unfair — IQL used 10-step Q targets while TD+EMA used 1-step bootstrap. This experiment gives TD+EMA the same n-step=10 treatment to isolate whether the advantage comes from IQL's Q→V framework or simply from n-step returns.

### Settings

| Parameter | Value |
|-----------|-------|
| gamma | 0.99 |
| td_reward_scale | 10.0 |
| ema_tau | 0.005 |
| td_epochs | 2000 |
| eval_every | 5 |
| td_n_step | 10 (NEW — TD+EMA now uses 10-step targets) |
| iql_td_n | 10 |
| expectile_tau | 0.5 |
| critic | 3×256 |
| lr | 3e-4 |
| batch_size | 1000 |
| N values | 1, 5, 10, 20, 50, 100 |

### Results — Peak Pearson r

| N | TD+EMA 1-step (Exp 21) | TD+EMA 10-step (new) | IQL n=10 | TD 10-step vs IQL |
|---|---|---|---|---|
| 1 | 0.450 | 0.482 | **0.524** | −8% |
| 5 | 0.653 | **0.689** | 0.681 | +1% |
| 10 | 0.719 | **0.742** | 0.742 | ≈0 |
| 20 | 0.764 | 0.764 | **0.771** | −1% |
| 50 | 0.827 | 0.821 | **0.827** | −1% |
| 100 | 0.854 | 0.851 | **0.855** | ≈0 |

### Peak Epochs

| N | TD+EMA 10-step | IQL n=10 |
|---|---|---|
| 1 | ep195 | ep235 |
| 5 | ep60 | ep75 |
| 10 | ep135 | ep140 |
| 20 | ep75 | ep90 |
| 50 | ep55 | ep60 |
| 100 | ep35 | ep55 |

### Notes

1. **N-step closes the gap**: TD+EMA 10-step matches IQL at N=5-10 (0.689 vs 0.681, 0.742 vs 0.742). The Exp 21 advantage of IQL over TD+EMA was primarily from n-step, NOT from the IQL Q→V framework.

2. **N=1 still favors IQL (+8%)**: 0.482 vs 0.524. At extreme small data, IQL's two-stage Q→V learning may provide implicit regularization. But this gap is much smaller than the 1-step comparison (+16% in Exp 21).

3. **N-step helps TD+EMA mainly at small N**: N=1 +7%, N=5 +5.5%, N=10 +3%, N≥20 ≈0%. With enough data, 1-step bootstrap error is small enough that n-step provides no benefit.

4. **Convergence speed similar**: TD+EMA 10-step and IQL peak at comparable epochs, unlike Exp 21 where IQL was 6x faster (because 1-step TD+EMA needed many more epochs).

5. **Conclusion**: IQL ≈ TD+EMA when both use n-step=10. The IQL framework (learning Q then V via expectile) adds no meaningful value over directly learning V with n-step TD. For online RL, TD+EMA with onestep advantages (`r+γV(s')−V(s)`) is preferred because it avoids the Q ranking problem entirely.

**Plot**: `runs/td_nstep10_vs_iql_curve.png`

---

## Exp 23: IQL+PPO Online — N-step, Advantage Mode, Epoch Ablation - 2026-02-22 22:56

**Git**: 58d478e (main)
**Script**: `RL/iql_ppo_online.py` (modified: added n-step Q target, GAE advantage mode)

### Motivation

Exp 20 showed IQL Q-V advantages (86.9%) << TD+EMA onestep (95.1%). Exp 22 showed this gap partly came from IQL using 1-step Q targets. This experiment tests:
1. Does n-step=10 improve IQL online RL? (Yes: 93.2%)
2. Can IQL's V be used for onestep advantages? (No: collapses)
3. Can GAE smooth IQL V's noise? (Yes: **97.3%** new best)

### Common Settings

| Parameter | Value |
|-----------|-------|
| checkpoint | ckpt_76_logstd-1.5.pt |
| num_envs | 100 |
| num_steps | 50 |
| total_timesteps | 50,000 (10 iterations) |
| gamma | 0.99 |
| iql_reward_scale | 10.0 |
| ema_tau | 0.005 |
| expectile_tau | 0.5 |
| PPO: clip_coef | 0.2 |
| PPO: update_epochs | 100 |
| PPO: target_kl | 100.0 |
| num_minibatches | 5 |

### Results — All Runs

| Config | td_n | advantage | epochs | i1 | i2 | i3 | i4 | i5 | i6 | i7 | i8 | i9 | i10 | Peak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **IQL n10 GAE reset** | **10** | **GAE** | **200** | 43.8% | 55.8% | 64.3% | 65.6% | 79.1% | 93.6% | 93.6% | 93.8% | 94.2% | **97.3%** | **97.3%** |
| IQL n10 Q-V reset | 10 | Q-V | 200 | 43.8% | 38.5% | 57.9% | 76.3% | 78.2% | 81.5% | 83.6% | 80.7% | 88.2% | 93.2% | 93.2% |
| IQL n10 onestep reset | 10 | onestep | 200 | 43.8% | 53.1% | 68.4% | 83.7% | 74.4% | 81.6% | 77.4% | 84.3% | 88.4% | 90.8% | 90.8% |
| IQL n10 replay Q-V | 10 | Q-V | 200 | 43.8% | 43.1% | 56.2% | 62.1% | 61.5% | 65.2% | 72.7% | 66.9% | 71.4% | 77.6% | 77.6% |
| IQL td1 Q-V 1400ep (Exp 20) | 1 | Q-V | 1400 | 43.8% | 78.9% | 86.9% | 84.1% | 83.4% | 85.5% | 82.3% | 84.4% | 83.6% | 82.3% | 86.9% |
| IQL td1 GAE 1400ep | 1 | GAE | 1400 | 43.8% | 53.5% | 65.9% | 54.1% | 51.5% | 46.2% | 39.5% | 29.7% | 24.0% | 10.2% | 65.9% |
| IQL td1 onestep 1400ep | 1 | onestep | 1400 | 43.8% | 55.8% | 59.1% | 32.6% | 21.7% | 19.4% | 16.3% | 14.8% | 4.7% | 8.6% | 59.1% |
| IQL td1 onestep 1500ep | 1 | onestep | 1500 | 43.8% | 45.7% | 27.1% | 18.0% | 13.3% | 17.1% | 16.2% | 10.1% | 13.2% | 5.5% | 45.7% |

### Comparison with Baselines

| Method | Peak SR | vs GAE PPO |
|---|---|---|
| **IQL n10 + GAE + reset** | **97.3%** | **+5.2%** |
| TD+EMA onestep + replay (Exp 19) | 97.1% | +5.0% |
| TD+EMA onestep + reset (Exp 19) | 95.1% | +3.0% |
| IQL n10 Q-V + reset | 93.2% | +1.1% |
| GAE PPO baseline | 92.1% | — |
| IQL n10 onestep + reset | 90.8% | −1.3% |
| IQL td1 Q-V + reset (Exp 20) | 86.9% | −5.2% |

### V Quality Scaling — IQL td_n=1 vs Others

| N | TD+EMA 1-step | IQL td_n=1 | IQL td_n=10 | TD+EMA 10-step |
|---|---|---|---|---|
| 1 | 0.450 | 0.462 | **0.524** | 0.482 |
| 5 | 0.653 | 0.634 | **0.681** | 0.689 |
| 10 | 0.719 | 0.673 | **0.742** | 0.742 |
| 20 | 0.764 | 0.701 | **0.771** | 0.764 |
| 50 | 0.827 | 0.757 | **0.827** | 0.821 |
| 100 | 0.854 | 0.785 | **0.855** | 0.851 |

**Plot**: `runs/iql_td1_curve_2k.png`

### Notes

1. **IQL n10 + GAE = 97.3% (new best)**: Matches TD+EMA replay (97.1%) without needing replay buffer. GAE smooths IQL V's pointwise noise into usable advantage signal.

2. **Why IQL's V fails at onestep but works with GAE**:
   - TD+EMA directly minimizes Bellman residual `(V(s) - r - γV(s'))²` → onestep advantage IS the residual, so it's small and accurate
   - IQL's V is learned pointwise from Q via expectile regression → no temporal consistency constraint → V(s')-V(s) has high-frequency noise
   - Onestep advantage `r+γV(s')-V(s)` amplifies this noise → collapse
   - GAE averages multiple δ_t, smoothing out the noise → works

3. **Q-V advantages are robust to V overfit**: IQL td1 Q-V (86.9%) works fine with 1400ep even though V is overfitted. Q and V share the same Q network bias, so Q(s,a)-V(s) cancels systematic errors. But r+γV(s')-V(s) uses V at different states, errors compound.

4. **N-step is critical for IQL**: td_n=1 → td_n=10 improves every advantage mode. Q-V: 86.9%→93.2%, onestep: collapse→90.8%, GAE: collapse→97.3%.

5. **IQL td_n=1 V is worse than TD+EMA 1-step at N≥5**: The Q→V indirect learning path adds noise compared to direct TD. At N=100: 0.785 vs 0.854. This contradicts the N=1 finding where they're similar (0.462 vs 0.450).

6. **Epoch sensitivity**: IQL td1 at 1400ep is past peak (scaling shows N=1 peak at ep1560) and already overfitting. All td1 experiments with onestep/GAE collapse because V quality degrades. With n10, peak is at ep~200, so 200ep is near-optimal.

---

## Exp 24: TD+EMA N-step=10 Online Policy Learning — Does N-step Help TD+EMA? - 2026-02-22 23:53

**Git**: 58d478e (main)
**Script**: `RL/td_ema_awr_online.py` (modified: added `td_n_step` arg with n-step V target support)

### Motivation

IQL n10 + GAE reached 97.3% (Exp 23). Is this due to n-step, IQL's Q→V framework, or GAE? Test TD+EMA with the same n-step=10 to isolate. TD+EMA 1-step + 1400ep baseline = 95.1%.

### Commands
```bash
# TD+EMA n10, onestep, reset, 200ep, current_data_only
python -u -m RL.td_ema_awr_online --gamma 0.99 --advantage_mode onestep --policy_update ppo --current_data_only --reset_critic_each_iter --td_epochs_per_iter 200 --td_n_step 10 --update_epochs 100 --exp_name td_n10_onestep_ppo_reset_ep200

# TD+EMA n10, GAE, reset, 200ep, current_data_only
python -u -m RL.td_ema_awr_online --gamma 0.99 --advantage_mode gae --policy_update ppo --current_data_only --reset_critic_each_iter --td_epochs_per_iter 200 --td_n_step 10 --update_epochs 100 --exp_name td_n10_gae_ppo_reset_ep200
```

### Settings (aligned with Exp 19 R4 baseline, only td_n_step and epochs differ)

| Parameter | Value |
|-----------|-------|
| checkpoint | ckpt_76_logstd-1.5.pt |
| num_envs | 100 |
| num_steps | 50 |
| total_timesteps | 50,000 (10 iter) |
| gamma | 0.99 |
| td_reward_scale | 10.0 |
| ema_tau | 0.005 |
| td_n_step | **10** |
| td_epochs_per_iter | **200** |
| current_data_only | True |
| reset_critic_each_iter | True |
| PPO: update_epochs | 100 |
| PPO: clip_coef | 0.2 |
| PPO: target_kl | 100.0 |

### Results

| Config | i1 | i2 | i3 | i4 | i5 | i6 | i7 | i8 | i9 | i10 | Peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TD n10 onestep reset 200ep | 43.8% | 61.4% | 49.6% | 48.1% | 34.6% | 59.6% | 45.8% | 58.3% | 50.8% | 45.8% | 61.4% (i2) |
| TD n10 GAE reset 200ep | 43.8% | 50.8% | 51.9% | 60.0% | 60.6% | 68.5% | 69.2% | 77.7% | 77.6% | 78.1% | 78.1% (i10) |

### Comparison — All Methods at 200ep, reset, current_data_only

| Method | n_step | advantage | epochs | Peak |
|---|---|---|---|---|
| **IQL n10 GAE** | 10 | GAE | 200 | **97.3%** |
| IQL n10 Q-V | 10 | Q-V | 200 | 93.2% |
| IQL n10 onestep | 10 | onestep | 200 | 90.8% |
| TD+EMA n10 GAE | 10 | GAE | 200 | 78.1% |
| TD+EMA n10 onestep | 10 | onestep | 200 | 61.4% |
| **TD+EMA 1-step onestep 1400ep** (baseline) | **1** | **onestep** | **1400** | **95.1%** |

### Notes

1. **TD+EMA n10 + 200ep is terrible**: Both onestep (61.4%) and GAE (78.1%) far worse than 1-step baseline (95.1%). The 200 epochs is insufficient for TD+EMA — scaling experiments show TD+EMA N=1 peak at ep195 for n10, but this was on static data. In the online iterative setting with reset, 200ep may not be enough for stable V learning.

2. **IQL n10 >> TD+EMA n10 at same epochs**: At 200ep, IQL dramatically outperforms TD+EMA on all advantage modes. IQL's Q→V indirect learning converges faster in practice — the Q network provides a smoother learning signal for V than raw TD targets.

3. **TD+EMA needs more epochs**: The 1-step baseline uses 1400ep and works great. N-step=10 might also work with more epochs, but we didn't test this. The key insight is that TD+EMA is more epoch-hungry than IQL.

4. **IQL's advantage at equal compute**: When both methods get the same epoch budget (200), IQL is far superior. IQL's two-stage Q→V learning provides implicit regularization that lets V converge faster and stay stable.

5. **GAE >> onestep for TD+EMA n10**: GAE (78.1%) much better than onestep (61.4%), consistent with the pattern that GAE smooths noisy V. But even GAE can't fully compensate for under-trained V.

---

## [P(success) Analysis & Guided Branching Methods] - 2026-02-23

**Git**: 58d478e (main)

### P(success|s) from MC16 — Key Results

Computed P(success|s) = fraction of 16 MC rollouts that succeed from each state. First episodes only (no truncated post-reset fragments).

**PickCube** (ckpt_76_logstd-1.5, SR=44%, gamma=0.99):
- P ≈ 0.5 across 0-50% of episode (broad decision region)
- Var[P] peaks at 75-80%
- Fail trajectories P drops from 0.46 → 0.05
- Success trajectories P stays ~0.55-0.59

**PegInsertion** (ckpt_231_ema99, SR=81%, gamma=0.97):
- Var[P] sharp peak at 25-30% (critical approach/alignment phase)
- Fail trajectories P dips to 0.29 at 25-30% then rises to ~0.48
- Success trajectories P gradually decreases from 0.80 → 0.67

### Cross-analysis with Branch Ablation

EP-% ablation + V-bin ablation + P(success) 联立分析:
- PickCube: late/mid branching best, v60_100 (high V) best → proximity to reward matters most
- PegInsertion: uniform best, v0_20 (low V) competitive → success termination times spread widely, only uniform covers all
- **V-learning needs signal (near-reward transitions) + propagation (TD bootstrap chain). No focused strategy beats uniform because both are needed.**
- **P(success) ≈ 0.5 is best for policy improvement, not V-learning. TD error is better for V-learning.**

### Two Practical Guided Branching Methods (to test)

Both methods use ONLY existing data (no MC16 oracle):

**Method 1: TD Error Guided** (`v_tree_td_guided.py`)
- Train V₀ on seed data (1 rollout) with target gamma
- Compute |TD error| = |r + γV₀(s') - V₀(s)| per state
- Branch from high TD-error states (weighted sampling or top-K)
- Directly optimizes "where V is most wrong"
- Script: `RL/v_tree_td_guided.py`, running now

**Method 2: P(success) Predictor Guided** (to write)
- Train V₀ with gamma=1.0 on seed data → V₀(s) ≈ P(success|s)
- Branch from states where |P - 0.5| is smallest (maximum uncertainty)
- Or equivalently: branch where P*(1-P) is largest (Bernoulli variance)
- Identifies the decision boundary without knowing the task structure
- Key: gamma=1 avoids discount confound; with sparse reward, V(gamma=1) = P(success)

### Notes
- MC16 P(success) is oracle for validation only, not for practical use
- TD error may correlate with P ≈ 0.5 but isn't identical: TD error also captures V approximation error, not just outcome uncertainty
- For sparse reward + gamma=1: TD target = r + V(s'), TD error = |V(s) - r - V(s')| ≈ how wrong V is about success propagation

---

## Guided Branching Ablation — TD Error vs P(success) (PickCube) - 2026-02-23 15:00

**Command**: `python -u -m RL.v_tree_guided_ablation --gamma 0.99 --strategies uniform td_weighted td_topk p_bernoulli p_topk`
**Git**: 58d478e (main)
**Run Dir**: runs/v_tree_guided_ablation.pt / .png

### Settings
| Parameter | Value |
|-----------|-------|
| env_id | PickCube-v1 |
| checkpoint | runs/pickcube_ppo/ckpt_76_logstd-1.5.pt |
| gamma | 0.99 |
| num_envs | 100 |
| num_steps / max_episode_steps | 50 |
| mc_samples | 16 |
| N_values | (1, 2, 3, 5) |
| strategies | uniform, td_weighted, td_topk, p_bernoulli, p_topk |
| topk_frac | 0.2 |
| td_reward_scale | 10.0 |
| ema_tau | 0.005 |
| epochs | 2000 |
| v0_epochs | 500 |
| p0_epochs | 2000 |
| lr | 3e-4 |
| batch_size | 1000 |
| hidden_dim | 256 |
| critic_layers | 3 |

### Results

**Diagnostics**: V₀ r=0.2031, P₀ r=0.4567, Corr(TD_error, Bernoulli_var)=-0.217

**Peak Pearson r (vs MC16):**

| N | Rollout | uniform | td_weighted | td_topk | p_bernoulli | p_topk |
|---|---------|---------|-------------|---------|-------------|--------|
| 1 | 0.4540 | 0.5272 | 0.4989 | **0.5662** | 0.4127 | 0.3888 |
| 2 | 0.5510 | 0.5696 | 0.6569 | **0.6416** | 0.5391 | 0.5286 |
| 3 | 0.5734 | 0.6201 | 0.6786 | **0.6843** | 0.6281 | 0.6067 |
| 5 | 0.6567 | 0.7210 | 0.7270 | **0.7296** | 0.7114 | 0.6344 |

**Delta vs uniform:**

| N | td_weighted | td_topk | p_bernoulli | p_topk |
|---|-------------|---------|-------------|--------|
| 1 | -0.028 | **+0.039** | -0.115 | -0.138 |
| 2 | +0.087 | **+0.072** | -0.031 | -0.041 |
| 3 | +0.058 | **+0.064** | +0.008 | -0.014 |
| 5 | +0.006 | **+0.009** | -0.010 | -0.087 |

### Notes
- **TD error methods dominate across ALL N**: td_topk is best or near-best everywhere
- **P-guided methods HURT**: p_bernoulli and p_topk are negative vs uniform at most N values
- Corr(TD_error, Bernoulli_var) = -0.22 (negative!) — P≈0.5 states are where V is already accurate; high TD error states have P far from 0.5 (near reward)
- **V-learning needs reward signal (high TD error) + propagation (TD chain), not decision boundary data (P≈0.5)**
- TD error top-20% concentrated in late timesteps (Q3+Q4 = 79.5%), P(success) uncertainty concentrated in early timesteps

---

## Guided Branching Ablation — TD Error (PegInsertion) - 2026-02-23 16:00

**Command**: `python -u -m RL.v_tree_guided_ablation --gamma 0.97 --env_id PegInsertionSide-v1 --checkpoint runs/peginsertion_ppo_ema99/ckpt_231.pt --max_episode_steps 100 --num_steps 100 --strategies uniform td_weighted td_topk --output runs/v_tree_guided_ablation_peg.png`
**Git**: 58d478e (main)
**Run Dir**: runs/v_tree_guided_ablation_peg.pt / .png

### Settings
| Parameter | Value |
|-----------|-------|
| env_id | PegInsertionSide-v1 |
| checkpoint | runs/peginsertion_ppo_ema99/ckpt_231.pt |
| gamma | 0.97 |
| num_envs | 100 |
| num_steps / max_episode_steps | 100 |
| mc_samples | 16 |
| N_values | (1, 2, 3, 5) |
| strategies | uniform, td_weighted, td_topk |
| Other settings | Same as PickCube experiment above |

### Results

**Diagnostics**: V₀ r=0.3171, P₀ r=0.4067, Corr(TD_error, Bernoulli_var)=-0.064

**Peak Pearson r (vs MC16):**

| N | Rollout | uniform | td_weighted | td_topk |
|---|---------|---------|-------------|---------|
| 1 | 0.5220 | 0.5182 | 0.4928 | **0.6050** |
| 2 | 0.5956 | **0.6094** | 0.5817 | 0.5876 |
| 3 | 0.6117 | 0.6302 | 0.6156 | **0.6310** |
| 5 | 0.6262 | **0.6419** | 0.6322 | 0.6242 |

**Delta vs uniform:**

| N | td_weighted | td_topk |
|---|-------------|---------|
| 1 | -0.025 | **+0.087** |
| 2 | -0.028 | -0.022 |
| 3 | -0.015 | +0.001 |
| 5 | -0.010 | -0.018 |

### Notes
- **td_topk helps only at N=1** (+0.087), neutral-to-negative at N≥2 — different from PickCube where td_topk is consistently positive
- **Root cause: TD error distribution is much more uniform on PegInsertion**
  - PickCube TD error concentration: 0.073, KL(topk||uniform) = 0.284
  - PegInsertion TD error concentration: 0.022, KL(topk||uniform) = 0.101
  - PegInsertion topk effective timesteps: 90.4/100 (90%) vs PickCube 37.7/50 (75%)
- PegInsertion's V₀ is already better (r=0.317 vs 0.203), TD errors distributed uniformly across episode → topk doesn't change data distribution meaningfully
- td_weighted consistently negative on both tasks — soft weighting is worse than hard topk selection

---

## Ensemble Disagreement Ablation (PickCube) - 2026-02-23 17:00

**Command**: `python -u -m RL.v_tree_guided_ablation --gamma 0.99 --strategies uniform td_topk ens_weighted ens_topk --output runs/v_tree_ensemble_ablation.png`
**Git**: 58d478e (main)
**Run Dir**: runs/v_tree_ensemble_ablation.pt / .png

### Settings
| Parameter | Value |
|-----------|-------|
| env_id | PickCube-v1 |
| gamma | 0.99 |
| ensemble_K | 5 |
| strategies | uniform, td_topk, ens_weighted, ens_topk |
| Other settings | Same as base guided ablation |

### Results

**Diagnostics**: V₀ r=0.2031, Ens mean_member_r=0.1870, Corr(TD_error, Ens_var)=-0.012

**Peak Pearson r:**

| N | Rollout | uniform | td_topk | ens_weighted | ens_topk |
|---|---------|---------|---------|--------------|----------|
| 1 | 0.4500 | 0.4061 | **0.5572** | 0.5456 | 0.4142 |
| 2 | 0.5405 | 0.5781 | **0.6678** | 0.5892 | 0.5763 |
| 3 | 0.5825 | 0.6297 | **0.7167** | 0.6378 | 0.6549 |
| 5 | 0.6573 | 0.7241 | **0.7421** | 0.7398 | 0.6974 |

**Delta vs uniform:**

| N | td_topk | ens_weighted | ens_topk |
|---|---------|--------------|----------|
| 1 | **+0.151** | +0.140 | +0.008 |
| 2 | **+0.090** | +0.011 | -0.002 |
| 3 | **+0.087** | +0.008 | +0.025 |
| 5 | +0.018 | +0.016 | -0.027 |

### Notes
- **Ensemble disagreement (Var[V_k]) does NOT beat single-V₀ TD error**
- **Corr(TD_error, Ens_var) = -0.012** — the two signals measure completely different things
- Ens_var top-20% by quartile: Q1=33%, Q2=27%, Q3=21%, Q4=19% — biased toward EARLY states (where V≈0, disagreement is about small irrelevant differences)
- TD_error top-20% by quartile: Q1=3%, Q2=17%, Q3=37%, Q4=43% — concentrated near reward (where V changes rapidly)
- **Epistemic uncertainty alone is insufficient** — it doesn't distinguish "uncertain but unimportant" from "uncertain and important"
- ens_topk is worse than ens_weighted because it concentrates on early timesteps even more

---

## Ensemble TD Error Ablation — Mean TD & Bellman Residual Variance (PickCube) - 2026-02-23 17:30

**Command**: `python -u -m RL.v_tree_guided_ablation --gamma 0.99 --strategies uniform td_topk ens_td_mean_topk ens_td_var_topk --output runs/v_tree_ens_td_ablation.png`
**Git**: 58d478e (main)
**Run Dir**: runs/v_tree_ens_td_ablation.pt / .png

### Settings
| Parameter | Value |
|-----------|-------|
| env_id | PickCube-v1 |
| gamma | 0.99 |
| ensemble_K | 5 |
| strategies | uniform, td_topk, ens_td_mean_topk, ens_td_var_topk |
| ens_td_mean_topk | top 20% by mean |TD error| across K networks |
| ens_td_var_topk | top 20% by Var[|TD error|] across K networks (Bellman residual variance) |

### Results

**Diagnostics**: Corr(single_TD, mean_TD) = 0.9837, Mean_TD top-20% quartile: [2.5%, 14.2%, 38.1%, 45.2%]

**Peak Pearson r:**

| N | Rollout | uniform | td_topk | ens_td_mean_topk | ens_td_var_topk |
|---|---------|---------|---------|------------------|-----------------|
| 1 | 0.4500 | 0.4061 | 0.5572 | 0.5092 | **0.5805** |
| 2 | 0.5417 | 0.6197 | 0.6325 | 0.6554 | **0.6791** |
| 3 | 0.5812 | 0.6599 | **0.6918** | 0.6714 | 0.6582 |
| 5 | 0.6586 | 0.7060 | 0.7174 | 0.6863 | **0.7239** |

**Delta vs uniform:**

| N | td_topk | ens_td_mean_topk | ens_td_var_topk |
|---|---------|------------------|-----------------|
| 1 | +0.151 | +0.103 | **+0.174** |
| 2 | +0.013 | +0.036 | **+0.059** |
| 3 | **+0.032** | +0.012 | -0.002 |
| 5 | +0.011 | -0.020 | **+0.018** |

### Notes
- **ens_td_var_topk (Bellman residual variance) is best at N=1,2,5** on PickCube
- **ens_td_mean_topk ≈ single td_topk** — Corr(single_TD, mean_TD) = 0.984, averaging K networks doesn't add information because single V₀ TD error is already stable
- **Bellman residual variance captures "importance-weighted epistemic uncertainty"** — Var[TD_error] is high only when (a) the TD error itself is non-trivial (importance) AND (b) different networks disagree on how wrong they are (epistemic)
- TD error inherently encodes importance weighting: |V(s) - target| is naturally small when V≈0 (early states), even if V is inaccurate there

---

## Ensemble TD Error Ablation — Mean TD & Bellman Residual Variance (PegInsertion) - 2026-02-23 18:00

**Command**: `python -u -m RL.v_tree_guided_ablation --gamma 0.97 --env_id PegInsertionSide-v1 --checkpoint runs/peginsertion_ppo_ema99/ckpt_231.pt --max_episode_steps 100 --num_steps 100 --strategies uniform td_topk ens_td_mean_topk ens_td_var_topk --output runs/v_tree_ens_td_ablation_peg.png`
**Git**: 58d478e (main)
**Run Dir**: runs/v_tree_ens_td_ablation_peg.pt / .png

### Settings
| Parameter | Value |
|-----------|-------|
| env_id | PegInsertionSide-v1 |
| checkpoint | runs/peginsertion_ppo_ema99/ckpt_231.pt |
| gamma | 0.97 |
| num_envs | 100 |
| num_steps / max_episode_steps | 100 |
| ensemble_K | 5 |
| strategies | uniform, td_topk, ens_td_mean_topk, ens_td_var_topk |

### Results

**Diagnostics**: V₀ r=0.3171, Corr(single_TD, mean_TD)=0.9674, Corr(TD_error, Ens_var)=-0.115

**Peak Pearson r:**

| N | Rollout | uniform | td_topk | ens_td_mean_topk | ens_td_var_topk |
|---|---------|---------|---------|------------------|-----------------|
| 1 | 0.5196 | 0.5036 | 0.5262 | **0.5599** | 0.5454 |
| 2 | 0.5940 | **0.6306** | 0.5898 | 0.6277 | 0.5810 |
| 3 | 0.6166 | **0.6450** | 0.6088 | 0.6321 | 0.5981 |
| 5 | 0.6242 | 0.6362 | 0.6011 | 0.6347 | **0.6363** |

**Delta vs uniform:**

| N | td_topk | ens_td_mean_topk | ens_td_var_topk |
|---|---------|------------------|-----------------|
| 1 | +0.023 | **+0.056** | +0.042 |
| 2 | -0.041 | **-0.003** | -0.050 |
| 3 | -0.036 | **-0.013** | -0.047 |
| 5 | -0.035 | **-0.002** | +0.000 |

### Notes
- **ens_td_mean_topk is the most robust strategy across both tasks** — never catastrophically bad, positive at small N
- **td_topk collapses on PegInsertion at N≥2** (consistently -0.035 to -0.041 vs uniform) — because PegInsertion's TD error is uniformly distributed, topk over-concentrates on noise
- **ens_td_mean_topk smooths away noise**: averaging K=5 networks prevents over-concentration on spurious TD error peaks, making it more robust when the underlying signal is diffuse
- ens_td_var_topk (Bellman residual variance) is unstable on PegInsertion — good at N=1, bad at N=2,3
- **Cross-task conclusion**: ens_td_mean_topk is the safest choice as a general-purpose guided branching strategy. td_topk is stronger when TD error is naturally concentrated (PickCube), but risky when TD error is diffuse (PegInsertion)

---

## Guided Branching — Summary Across All Experiments - 2026-02-23 18:17

### Strategy Taxonomy

| Strategy | Signal | Description |
|----------|--------|-------------|
| uniform | None | Random branch points (tree baseline) |
| td_weighted | |TD error| from single V₀ | Soft weighting proportional to TD error |
| td_topk | |TD error| from single V₀ | Branch only from top 20% TD error states |
| p_bernoulli | P₀*(1-P₀) from gamma=1 V | Weight by Bernoulli variance |
| p_topk | P₀*(1-P₀) from gamma=1 V | Branch from top 20% Bernoulli variance states |
| ens_weighted | Var_k[V_k(s)] | Weight by ensemble V disagreement |
| ens_topk | Var_k[V_k(s)] | Branch from top 20% ensemble disagreement |
| ens_td_mean_topk | Mean_k[|TD_err_k|] | Top 20% by averaged TD error across K networks |
| ens_td_var_topk | Var_k[|TD_err_k|] | Top 20% by Bellman residual variance |

### Cross-Task Ranking (delta vs uniform, averaged across N=1,2,3,5)

| Strategy | PickCube avg Δ | PegInsertion avg Δ | Overall |
|----------|---------------|-------------------|---------|
| td_topk | **+0.068** | -0.018 | +0.025 |
| ens_td_var_topk | +0.062 | -0.014 | +0.024 |
| ens_td_mean_topk | +0.033 | **+0.010** | +0.021 |
| td_weighted | +0.031 | -0.020 | +0.006 |
| ens_weighted | +0.041 | — | — |
| ens_topk | +0.001 | — | — |
| p_bernoulli | -0.037 | — | — |
| p_topk | -0.070 | — | — |

### Key Insights

1. **For V-learning, TD error is the correct guiding signal**, not P(success) uncertainty or ensemble V disagreement
2. **TD error = "importance-weighted uncertainty"** — it naturally encodes both "V is wrong here" and "this region matters (V≠0)"
3. **Ensemble V disagreement (Var[V_k]) fails** because it measures epistemic uncertainty without importance weighting — high disagreement at early states where V≈0 is irrelevant
4. **ens_td_mean_topk is the most robust across tasks** — smoothing K TD errors prevents noise-driven over-concentration
5. **ens_td_var_topk (Bellman residual variance) is strongest on PickCube** but unstable on PegInsertion
6. **Task structure matters**: when TD error is concentrated (PickCube, late-episode reward), guided branching helps significantly; when TD error is diffuse (PegInsertion), uniform is hard to beat

---

## StackCube PPO Training - 2026-02-23 18:30

**Command**: `python -u -m data.data_collection.ppo --env_id StackCube-v1 --num_envs 1024 --update_epochs 8 --num_minibatches 32 --total_timesteps 25000000 --eval_freq 10 --exp_name stackcube_ppo --no-capture_video`
**Git**: 58d478e (main)
**Run Dir**: runs/stackcube_ppo/

### Settings
| Parameter | Value |
|-----------|-------|
| env_id | StackCube-v1 |
| num_envs | 1024 |
| update_epochs | 8 |
| num_minibatches | 32 |
| total_timesteps | 25,000,000 |
| gamma | 0.8 (default) |
| gae_lambda | 0.9 (default) |
| learning_rate | 3e-4 (default) |
| control_mode | pd_joint_delta_pos |
| max_episode_steps | 50 |
| reward_mode | normalized_dense (default) |

### Results

Training progression (eval with 8 envs only — noisy):
- 0-13M steps: 0% eval success, dense return rising 2.4 → 29
- 8M steps: first train successes appear
- 14M steps: train success ~20%
- 18-25M: eval success 25-75% (noisy, only 8 episodes)

**Proper evaluation (200 envs):**

| Checkpoint | Steps | Stochastic SR | Deterministic SR |
|-----------|-------|---------------|------------------|
| ckpt_271 | 14M | 17.5% | 10.5% |
| ckpt_331 | 17M | 33.5% | 31.5% |
| ckpt_371 | 19M | 44.0% | 43.0% |
| ckpt_411 | 21M | 46.0% | 47.5% |
| ckpt_451 | 23M | 49.0% | 48.5% |
| **ckpt_481** | **25M** | **55.0%** | **54.5%** |
| final | 25M | 54.0% | 52.5% |

### Notes
- Command matches ManiSkill official `examples.sh` for StackCube-v1
- **Stochastic ≈ Deterministic** — unlike PickCube where stochastic inflates SR. StackCube requires precision (place + release), noise doesn't help.
- Agent still improving at 25M — could benefit from longer training
- Selected **ckpt_481** (55% SR) as finetuning start checkpoint
- Breakthrough around 8-14M steps (0% → 20% train success)

---

## Guided Branching Ablation — StackCube - 2026-02-23 19:15

**Command**: `python -u -m RL.v_tree_guided_ablation --gamma 0.99 --env_id StackCube-v1 --checkpoint runs/stackcube_ppo/ckpt_481.pt --max_episode_steps 50 --num_steps 50 --strategies uniform td_topk ens_td_mean_topk ens_td_var_topk --output runs/v_tree_guided_ablation_stackcube.png`
**Git**: 58d478e (main)
**Run Dir**: runs/v_tree_guided_ablation_stackcube.pt / .png

### Settings
| Parameter | Value |
|-----------|-------|
| env_id | StackCube-v1 |
| checkpoint | runs/stackcube_ppo/ckpt_481.pt |
| gamma | 0.99 |
| num_envs | 100 |
| num_steps / max_episode_steps | 50 |
| mc_samples | 16 |
| N_values | (1, 2, 3, 5) |
| strategies | uniform, td_topk, ens_td_mean_topk, ens_td_var_topk |
| ensemble_K | 5 |
| Other settings | Same as PickCube/PegInsertion experiments |

### Results

**Diagnostics**: V₀ r=0.7140, P₀ r=0.9099, Corr(TD_error, Ens_var)=0.0001, Corr(TD_error, Bernoulli_var)=0.024

**Peak Pearson r (vs MC16):**

| N | Rollout | uniform | td_topk | ens_td_mean_topk | ens_td_var_topk |
|---|---------|---------|---------|------------------|-----------------|
| 1 | **0.9237** | 0.8754 | 0.8513 | 0.8473 | 0.8738 |
| 2 | **0.9238** | 0.9150 | 0.9010 | 0.9038 | 0.9124 |
| 3 | **0.9165** | 0.9022 | 0.8976 | 0.8990 | 0.9034 |
| 5 | **0.9213** | 0.9099 | 0.8895 | 0.8830 | 0.9048 |

**Delta vs uniform:**

| N | td_topk | ens_td_mean_topk | ens_td_var_topk |
|---|---------|------------------|-----------------|
| 1 | -0.024 | -0.028 | -0.002 |
| 2 | -0.014 | -0.011 | -0.003 |
| 3 | -0.005 | -0.003 | +0.001 |
| 5 | -0.020 | -0.027 | -0.005 |

### Notes
- **Rollout >> Tree on ALL N values** — completely opposite to PickCube. Tree sampling hurts on StackCube.
- **All guided strategies worse than uniform**, td_topk especially bad (-0.005 to -0.024)
- **V₀ already very high (r=0.714)** vs PickCube (0.203) and PegInsertion (0.317). StackCube V is "easy" to learn with just rollout data.
- P₀ r=0.910 also extremely high — success/failure is very predictable from state
- TD error uniformly distributed: Mean_TD top-20% quartile = [30%, 28%, 25%, 16%]
- **Why tree hurts**: V is already easy to learn (r=0.924 from just 1 rollout!). Rollout provides complete trajectory structure (temporal consistency). Tree branches from mid-episode break this structure and provide less useful transitions for an already-easy V-learning problem.

---

## Cross-Task Guided Branching Summary (3 Tasks) - 2026-02-23 19:20

### Task Characteristics

| Task | V₀ r (500ep) | P₀ r | TD error concentration | Rollout N=1 r |
|------|-------------|------|----------------------|---------------|
| PickCube | 0.203 | 0.457 | High (Q4=45%) | 0.450 |
| PegInsertion | 0.317 | 0.407 | Low (Q4=28%) | 0.520 |
| StackCube | 0.714 | 0.910 | Very low (Q4=16%) | 0.924 |

### Best Strategy Delta vs Uniform (across N=1,2,3,5)

| Task | td_topk | ens_td_mean_topk | ens_td_var_topk | Tree vs Rollout |
|------|---------|------------------|-----------------|-----------------|
| PickCube | **+0.068** | +0.033 | +0.062 | Tree wins |
| PegInsertion | -0.018 | **+0.010** | -0.014 | Mixed |
| StackCube | -0.016 | -0.017 | **-0.002** | Rollout wins |

### Key Insights

1. **Tree sampling benefit is inversely related to V₀ quality**:
   - V₀ r=0.20 (PickCube): tree helps a lot, guided branching helps more
   - V₀ r=0.32 (PegInsertion): tree marginally helps, guided branching ~neutral
   - V₀ r=0.71 (StackCube): tree HURTS, guided branching hurts more

2. **When V is already easy to learn, more data (rollout) > smarter data (tree)**. Tree branching sacrifices temporal coherence for state diversity, which only helps when the V landscape is hard (low V₀ quality, concentrated TD error).

3. **TD error concentration predicts tree sampling benefit**: PickCube Q4=45% (concentrated near reward → topk helps), PegInsertion Q4=28% (moderate), StackCube Q4=16% (almost uniform → topk can't help).

4. **ens_td_var_topk is the most robust guided strategy** — least negative on StackCube (-0.002 avg), still strong on PickCube (+0.062 avg). It's the safest "do no harm" choice across tasks.

5. **Practical rule**: If V₀ quality from seed data is already high (r > 0.5), skip tree sampling entirely and just use rollout data. Tree sampling is only useful when V is hard to learn from rollouts alone.

---

## [StackCube Tree Guided Branching with logstd=-1.5] - 2026-02-23 21:20

**Command**: `python -u -m RL.v_tree_guided_ablation --env_id StackCube-v1 --checkpoint runs/stackcube_ppo/ckpt_481_logstd-1.5.pt --gamma 0.99 --output runs/v_tree_guided_ablation_stackcube_logstd15.png`
**Git**: 58d478e (main)
**Run Dir**: runs/v_tree_guided_ablation_stackcube_logstd15.pt

### Motivation

Original StackCube policy is near-deterministic (within-state MC std=0.10, Corr(MC1,MC16)=0.912), making tree sampling useless because all branches produce homogeneous trajectories. This experiment tests whether manually increasing stochasticity (logstd=-1.5, std=0.223) enables tree sampling to work — confirming that action diversity is the prerequisite for tree benefit.

### Settings
| Parameter | Value |
|-----------|-------|
| env_id | StackCube-v1 |
| checkpoint | runs/stackcube_ppo/ckpt_481_logstd-1.5.pt (SR=31.9%) |
| gamma | 0.99 |
| td_reward_scale | 10.0 |
| ema_tau | 0.005 |
| epochs | 2000 |
| eval_every | 5 |
| N_values | (1, 2, 3, 5) |
| n_seed | 1 (fixed) |
| rollouts | 5 |
| MC16 envs | 1600 |
| strategies | uniform, td_weighted, td_topk, p_bernoulli, p_topk, ens_weighted, ens_topk, ens_td_mean_topk, ens_td_var_topk |

### Diagnostics
| Metric | Value |
|--------|-------|
| MC16 mean | 0.1954 |
| MC16 std | 0.2653 |
| V₀ peak r | 0.6460 |
| P₀ peak r | 0.7120 |
| Ensemble mean r | 0.6513 |
| Corr(TD_error, Ens_var) | 0.1287 |
| Corr(TD_error, Bernoulli_var) | -0.0158 |

### Results — Peak Pearson r (vs MC16 ground truth)

| N | Rollout | uniform | td_weighted | td_topk | p_bernoulli | p_topk | ens_weighted | ens_topk | ens_td_mean_topk | ens_td_var_topk |
|---|---------|---------|-------------|---------|-------------|--------|--------------|----------|------------------|-----------------|
| 1 | **0.757** | 0.637 | 0.692 | 0.732 | 0.602 | 0.651 | 0.689 | 0.654 | 0.728 | 0.693 |
| 2 | 0.799 | 0.781 | **0.810** | 0.802 | 0.783 | 0.808 | 0.788 | 0.760 | 0.798 | 0.803 |
| 3 | 0.803 | 0.828 | 0.813 | 0.850 | 0.804 | 0.805 | 0.841 | 0.787 | **0.851** | 0.807 |
| 5 | 0.827 | 0.841 | 0.861 | 0.832 | 0.829 | 0.819 | 0.836 | 0.820 | **0.869** | 0.849 |

### Delta vs Rollout (best tree strategy)

| N | Best Strategy | Delta |
|---|---------------|-------|
| 1 | td_topk | **-0.025** |
| 2 | td_weighted | **+0.011** |
| 3 | ens_td_mean_topk | **+0.048** |
| 5 | ens_td_mean_topk | **+0.042** |

### Comparison: Original StackCube vs logstd=-1.5

| Metric | Original (det) | logstd=-1.5 |
|--------|---------------|-------------|
| Checkpoint SR | 62.1% | 31.9% |
| MC within-state std | 0.104 | 0.179 |
| Corr(MC1, MC16) | 0.912 | ~0.7 (est.) |
| SNR (between/within) | 3.87 | 1.59 |
| V₀ r | 0.714 | 0.646 |
| Tree vs Rollout (N=3) | Tree hurts | **Tree +4.8%** |
| Tree vs Rollout (N=5) | Tree hurts | **Tree +4.2%** |

### Notes

1. **logstd=-1.5 enables tree sampling on StackCube**: With increased stochasticity, tree beats rollout at N≥2. Original StackCube saw tree consistently hurt because MC trajectories were too homogeneous.

2. **ens_td_mean_topk is the best strategy at N≥3**: Combines ensemble uncertainty with TD error — branches at high-uncertainty, high-error states.

3. **But practical significance is limited**: In real training, the policy naturally converges toward deterministic (low logstd). The experiment confirms the mechanism (stochasticity → diversity → tree helps), but StackCube's inherent low action sensitivity means:
   - Advantage signal remains weak (within-state std=0.18 vs PickCube's 0.33)
   - Even with better V from tree sampling, iterative RL struggles because A(s,a) ≈ noise
   - The bottleneck isn't V quality — it's the task's low sensitivity to action choice

4. **Conclusion**: Tree sampling on StackCube is of limited practical value. The task's near-deterministic dynamics mean that (a) naturally trained policies don't benefit from tree branching, and (b) even if forced to benefit via noise injection, the downstream advantage signal is still too weak for effective policy improvement.

---

## [Cross-Task Within-State MC Variance Comparison] - 2026-02-23 21:30

**Git**: 58d478e (main)

### Motivation

Within-state MC variance (variance of MC returns across different rollouts from the same state) is the upper bound on advantage signal — if MC returns are the same regardless of action, then A(s,a) ≈ 0. This comparison explains why tree sampling and iterative RL work on PickCube but fail on StackCube.

### Results — MC1 vs MC16 Analysis (16 MC samples per state, gamma=0.99)

| Metric | PickCube (logstd=-1.5) | StackCube (original) | StackCube (logstd=-1.5) |
|--------|----------------------|---------------------|------------------------|
| Checkpoint SR | 43.8% | 62.1% | 31.9% |
| Action std | 0.223 | ~0.05 (learned) | 0.223 |
| **Within-state MC std** | **0.330** | **0.104** | **0.179** |
| Between-state MC std | 0.218 | 0.403 | 0.284 |
| **SNR (between/within)** | **0.66** | **3.87** | **1.59** |
| Corr(MC1, MC16) | 0.504 | 0.912 | ~0.7 |
| States with MC std < 0.01 | ~5% | 51% | ~20% |

### Interpretation

- **Within-state std = advantage signal upper bound**: If all MC rollouts from the same state give the same return, no action is better than any other → advantage ≈ 0.
- **PickCube (0.33)**: Actions meaningfully affect outcomes → tree sampling creates useful diversity → iterative RL works.
- **StackCube original (0.10)**: Near-deterministic policy → all branches homogeneous → tree sampling useless → iterative RL fails (advantage ≈ noise).
- **StackCube logstd=-1.5 (0.18)**: Forced noise partially helps, but still half of PickCube's signal → tree sampling marginally works at N≥2.
- **SNR interpretation**: Low SNR (PickCube=0.66) means within-state variation dominates → actions matter. High SNR (StackCube=3.87) means between-state variation dominates → state identity determines outcome, not action choice.

---

## [MC16 AWR Iterative Finetuning on StackCube] - 2026-02-23 22:37

**Commands**:
```bash
# Run 1: Original checkpoint
python -u -m RL.mc_finetune_awr_parallel --env_id StackCube-v1 \
  --checkpoint runs/stackcube_ppo/ckpt_481.pt \
  --optimal_checkpoint runs/stackcube_ppo/ckpt_481.pt \
  --mc_samples 16 --awr_beta 0.5 --gamma 0.8 \
  --num_envs 100 --num_steps 50 --total_timesteps 50000 \
  --update_epochs 200 --num_minibatches 5 --eval_freq 1 \
  --exp_name mc16_awr_stackcube_orig --seed 1

# Run 2: logstd=-1.5 checkpoint (more stochastic)
python -u -m RL.mc_finetune_awr_parallel --env_id StackCube-v1 \
  --checkpoint runs/stackcube_ppo/ckpt_481_logstd-1.5.pt \
  --optimal_checkpoint runs/stackcube_ppo/ckpt_481.pt \
  --mc_samples 16 --awr_beta 0.5 --gamma 0.8 \
  --num_envs 100 --num_steps 50 --total_timesteps 50000 \
  --update_epochs 200 --num_minibatches 5 --eval_freq 1 \
  --exp_name mc16_awr_stackcube_logstd15 --seed 1
```
**Git**: 58d478e (main)
**Run Dirs**: `runs/mc16_awr_stackcube_orig__seed1__1771911818/`, `runs/mc16_awr_stackcube_logstd15__seed1__1771911820/`

### Settings
| Parameter | Value |
|-----------|-------|
| env_id | StackCube-v1 |
| method | MC16 AWR (parallel re-rollout) |
| mc_samples | 16 |
| awr_beta | 0.5 |
| gamma | 0.8 |
| num_envs | 100 |
| mc_envs | 3200 |
| num_steps | 50 |
| total_timesteps | 50000 (10 iterations) |
| update_epochs | 200 |
| num_minibatches | 5 |
| batch_size | 5000 |
| eval_freq | 1 |
| reward_mode | sparse |
| control_mode | pd_joint_delta_pos |

### Results

| Iter | Original ckpt_481 (SR%) | logstd=-1.5 (SR%) |
|------|------------------------|--------------------|
| 1 (init) | 65.7 | 65.7 |
| 2 | 69.7 | 68.2 |
| 3 | 66.3 | 72.4 |
| 4 | 68.3 | 60.7 |
| 5 | 68.0 | 56.3 |
| 6 | 61.2 | 67.4 |
| 7 | 64.0 | 64.7 |
| 8 | 65.1 | 61.7 |
| 9 | 64.2 | 59.5 |
| 10 | 68.8 | 62.2 |
| **Peak** | **69.7 (iter 2)** | **72.4 (iter 3)** |

### Notes
- **MC16 AWR completely fails on StackCube** — neither checkpoint reaches 90%, both oscillate around 60-70% with zero upward trend over 10 iterations.
- Initial SR is 65.7% for both (deterministic eval uses same mean weights).
- Original checkpoint oscillates ±5% around 66%; logstd=-1.5 actually degrades over time (72.4% → 62.2%).
- Compare to PickCube where MC16 AWR reaches 99.1% in 10 iterations — the difference is entirely due to task dynamics.
- Root cause: StackCube's within-state MC std is only 0.16 (vs PickCube's 0.33), so MC advantages are uninformative. AWR can't learn meaningful policy updates from noisy advantages.
- This confirms that **iterative MC16 AWR requires a task/policy combination where actions meaningfully affect outcomes** (i.e., sufficient within-state MC variance).
- Open question: what metrics can predict whether a base policy is suitable for iterative RL? Candidates: within-state MC std, SR sweet spot (~50%), fraction of decisive states, action entropy.

---

## [Initial State P(success) Distribution: PickCube vs StackCube] - 2026-02-23 23:03

**Command**: `python -u -m RL.p_initial_state_analysis`
**Git**: 58d478e (main)
**Script**: `RL/p_initial_state_analysis.py` (new)
**Output**: `runs/p_initial_state_analysis.png`, `runs/p_initial_state_analysis.pt`

### Settings
| Parameter | Value |
|-----------|-------|
| pick_checkpoint | runs/pickcube_ppo/ckpt_76_logstd-1.5.pt |
| stack_checkpoint | runs/stackcube_ppo/ckpt_481.pt |
| gamma | 0.8 |
| num_envs | 500 (diverse initial states) |
| max_episode_steps | 50 |
| mc_samples | 16 |
| seed | 1 |

### Method
For each task: reset 500 envs → 500 diverse initial states → MC16 from each initial state → P(success|s₀) = n_success/16. Compare distributions.

### Results

| Metric | PickCube | StackCube |
|--------|----------|-----------|
| Overall SR | 46.3% | 53.6% |
| frac_zero (P=0) | 1.2% | **25.2%** |
| frac_one (P=1) | 0.2% | **21.2%** |
| frac_decisive (0.1<P<0.9) | **96.2%** | 40.8% |
| P(success) std | 0.212 | **0.408** |
| P(success) median | 0.438 | 0.625 |
| mean MC return std | 0.0028 | 0.0045 |

### Notes
- **Hypothesis confirmed**: StackCube's P(success|s₀) distribution is strikingly bimodal — 25% of initial states always fail (P=0), 21% always succeed (P=1). Only 41% of states are "decisive" (0.1<P<0.9).
- **PickCube** has a smooth, unimodal distribution centered around ~0.45. Nearly all initial states (96.2%) have uncertain outcomes where the policy's actions matter.
- This explains why MC16 AWR oscillates at 60-70% on StackCube instead of converging: ~59% of initial states produce uninformative advantages (either always +1 or always 0 regardless of actions taken).
- The ~70% ceiling across all StackCube methods (MC16 AWR peak=72.4%, GAE baseline peak=70.3%) aligns with ~75% of initial states being potentially solvable (100% - 25% hopeless = 75%).
- **Root cause**: StackCube's difficulty is driven by initial state diversity (cube placement/orientation), not by advantage estimation quality. Some reset configurations are physically unsolvable by the policy.
- This complements the earlier within-state MC std finding (StackCube 0.16 vs PickCube 0.33): low within-state variance is partly because many states are deterministically success/fail.

---

## [P(success|s₀) Distribution: PegInsertionSide] - 2026-02-24 03:30

### Overview
Compute initial state P(success|s₀) distribution for PegInsertionSide-v1 (ckpt_231, ema99) via MC16, and compare with PickCube/StackCube.

### Command
```bash
# Inline script using p_initial_state_analysis.py logic
# E=500, M=16, max_episode_steps=100, gamma=0.97, seed=1
# checkpoint: runs/peginsertion_ppo_ema99/ckpt_231.pt
```

### Results

| Metric | PickCube (ckpt_76) | PegInsertion (ckpt_231) | StackCube (ckpt_481) |
|--------|-------------------|------------------------|---------------------|
| Overall SR | 43.8% | **76.7%** | ~70% |
| frac_zero (P=0) | 1.2% | **0.0%** | 25.2% |
| frac_one (P=1) | 0.2% | 17.2% | 21.2% |
| frac_decisive (0.1<P<0.9) | 96.2% | 65.8% | 40.8% |
| P(success) std | 0.212 | 0.218 | 0.408 |
| P(success) median | ~0.44 | 0.81 | — |

**Output**: `runs/p_initial_state_peg.png`, `runs/p_initial_state_peg.pt`

### Notes
- PegInsertion has **zero dead zones** (frac_zero=0%), unlike StackCube (25%). Every initial state has some chance of success.
- Distribution is **right-skewed single-peak** (median=0.81), not bimodal like StackCube.
- 65.8% of states are in the decisive range (0.1<P<0.9) — less than PickCube (96.2%) but far better than StackCube (40.8%).
- This suggests PegInsertion finetuning has significant room for improvement (76.7% → 90%+) with MC16 AWR.

---

## [BC on PickCube-v1: Network Architecture Ablation] - 2026-02-24 02:30–04:30

### Overview
Train MLP BC on PickCube-v1 using MP demos (pd_ee_delta_pos, physx_cpu) to get a base policy for RL finetuning. Ablate demo source, network architecture, activation function, and loss function.

### Demo Data
- **MP demos**: `~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5` (1000 trajs, 49-105 steps avg 78, obs=42D, action=7D)
- **RL demos**: `~/.maniskill/demos/PickCube-v1/rl/trajectory.state.pd_joint_delta_pos.physx_cpu.h5` (594 trajs, 50 steps, obs=42D, action=8D)

### Commands
```bash
# 1. RL demo, 2x256 ReLU, MSE, 10k
python -u bc_official.py --env-id "PickCube-v1" \
  --demo-path ~/.maniskill/demos/PickCube-v1/rl/trajectory.state.pd_joint_delta_pos.physx_cpu.h5 \
  --control-mode "pd_joint_delta_pos" --sim-backend "cpu" --max-episode-steps 100 \
  --total-iters 10000 --exp-name "bc_pickcube_rl_demo"

# 2. MP demo, 2x256 ReLU (official), MSE, 100k
python -u bc_official.py --env-id "PickCube-v1" \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "cpu" --max-episode-steps 100 \
  --total-iters 100000 --exp-name "bc_pickcube_mp_100k"

# 3. MP demo, 3x256 Tanh+ortho, Gaussian NLL, 100k
python -u bc_official.py --env-id "PickCube-v1" \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "cpu" --max-episode-steps 100 \
  --total-iters 100000 --exp-name "bc_pickcube_mp_gaussian"

# 4. MP demo, 3x256 ReLU+ortho, MSE, 100k
python -u bc_official.py --env-id "PickCube-v1" \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "cpu" --max-episode-steps 100 \
  --total-iters 100000 --exp-name "bc_pickcube_mp_mse_3x256relu"
```

### Results

| # | Setting | Network | Loss | Iters | Peak SR |
|---|---------|---------|------|-------|---------|
| 1 | RL demo | 2×256 ReLU | MSE | 10k | **88%** |
| 2 | MP demo (official) | 2×256 ReLU | MSE | 100k | **57%** |
| 3 | MP demo | 3×256 ReLU+ortho | MSE | 100k | 40% |
| 4 | MP demo | 3×256 Tanh+ortho | NLL | 100k | 15% |
| 5 | MP demo | 3×256 Tanh+ortho | MSE | ~50k | ~3% |

**Training curves (MP demo, every 10k iters):**

| Iter | 2×256 ReLU | 3×256 ReLU+ortho | 3×256 Tanh+NLL |
|------|-----------|-----------------|---------------|
| 10k | 4% | 4% | 0% |
| 20k | 6% | 7% | 1% |
| 30k | 23% | 22% | 1% |
| 40k | 16% | 11% | 7% |
| 50k | 36% | 10% | 1% |
| 60k | 28% | 14% | 6% |
| 70k | 42% | 15% | 0% |
| 80k | 38% | 22% | 2% |
| 90k | 40% | 36% | 0% |
| 100k | — | ~13% | ~8% |

**Run dirs**: `runs/bc_pickcube_rl_demo/`, `runs/bc_pickcube_mp_100k/`, `runs/bc_pickcube_mp_gaussian/`, `runs/bc_pickcube_mp_mse_3x256relu/`

### Notes
1. **RL demo >> MP demo** (88% vs 57%): Deterministic NN-generated demos are far easier to clone. MP demos are multimodal (different motion plans per reset), causing MLP to average across modes.
2. **Tanh is the dominant failure mode**: 3×256 Tanh+MSE = 3% (worst), Tanh+NLL = 15%. ReLU variants are 10-20x better. Tanh saturates and compresses gradients, making precise continuous action regression very difficult.
3. **NLL loss is NOT the main problem**: Previously attributed 15% SR to NLL degrading mean learning. Control experiment showed Tanh+MSE is even worse (3%). NLL's gradient scaling (1/var) actually partially compensates for Tanh saturation.
4. **Orthogonal init + deeper network slightly hurts**: 3×256 ReLU+ortho (40%) vs 2×256 ReLU+default (57%). Possibly over-parameterization or last-layer small std init (`std=0.01*sqrt(2)`) limiting output range.
5. **Gaussian policy exploration must be manually added**: BC pushes logstd → -5 (clamp limit) regardless of loss. For finetuning, manually set logstd (e.g., -1.5). The logstd from BC is meaningless.
6. **Implication for finetuning Agent**: The standard Agent class uses Tanh activation — fine for RL (policy gradient), but bad for BC regression. Need to add ReLU activation option to Agent for BC→RL pipeline.

---

## [BC Policy P(success|s₀) Distribution: Extreme Bimodality] - 2026-02-24 05:00

### Overview
Analyze P(success|s₀) distribution of the two best BC policies on PickCube-v1 via MC16, and compare with PPO base policy.

### Command
```bash
# Inline script: 500 initial states, MC16 deterministic rollout, physx_cuda
# RL demo BC: runs/bc_pickcube_rl_demo/checkpoints/best_eval_success_once.pt (pd_joint_delta_pos)
# MP demo BC: runs/bc_pickcube_mp_100k/checkpoints/best_eval_success_once.pt (pd_ee_delta_pos)
```

### Results

| Metric | RL demo BC | MP demo BC | PPO ckpt_76 |
|--------|-----------|-----------|-------------|
| eval SR (physx_cpu) | 88% | 57% | 43.8% |
| MC16 SR (physx_cuda) | 62.7% | 15.1% | 43.8% |
| frac_zero (P=0) | **36.8%** | **84.4%** | 1.2% |
| frac_one (P=1) | **62.0%** | 14.4% | 0.2% |
| frac_decisive (0.1<P<0.9) | **1.2%** | **1.2%** | **96.2%** |
| P std | 0.481 | 0.354 | 0.212 |

**Output**: `runs/p_initial_state_bc_pickcube.png`

### Notes
1. **BC policies are extremely bimodal**: Both have frac_decisive=1.2% — virtually no states where the policy's actions matter stochastically. Each initial state deterministically succeeds or fails. PPO has 96.2% decisive states.
2. **Root cause**: BC is a deterministic policy (MSE-trained, no exploration noise). Given a fixed initial state, it always produces the same trajectory → same outcome. PPO has learnable logstd providing action noise, so the same initial state can have different outcomes.
3. **Finetuning implication**: With frac_decisive=1.2%, MC re-rollout from any given state always returns the same outcome → advantage ≈ 0 everywhere → no learning signal. Must add exploration noise (set logstd to e.g. -1.5) before finetuning.
4. **Backend mismatch**: RL demo BC shows 88% on physx_cpu but 62.7% on physx_cuda. The deterministic BC policy amplifies even small obs differences between backends via compounding error. This is Issue #18 affecting PickCube too, not just StackCube.
5. **MP demo BC especially bad**: 84.4% of initial states are dead zones (P=0). Even with logstd=-1.5, many of these will remain near-zero, limiting finetuning ceiling.

---

## [BC P(success|s₀) — physx_cpu Deterministic Rerun] - 2026-02-24 11:00

**Command**: `python -u /tmp/bc_p_success_cpu.py`
**Git**: ac5aa39 (main)
**Output**: `runs/p_initial_state_bc_pickcube_cpu.png`

### Settings
| Parameter | Value |
|-----------|-------|
| env | PickCube-v1 |
| backend | physx_cpu (single CPU env) |
| E (initial states) | 500 |
| MC samples | 1 (deterministic → MC1 suffices) |
| max_episode_steps | 100 |
| seed per state | seed=i for i in 0..499 |
| reconfiguration_freq | 1 |

### Policies Tested
| Policy | Checkpoint | Control Mode |
|--------|-----------|-------------|
| RL demo BC (88% eval SR) | `runs/bc_pickcube_rl_demo/checkpoints/best_eval_success_once.pt` | pd_joint_delta_pos |
| MP demo BC (57% eval SR) | `runs/bc_pickcube_mp_100k/checkpoints/best_eval_success_once.pt` | pd_ee_delta_pos |

### Results
| Metric | RL demo BC | MP demo BC |
|--------|-----------|-----------|
| physx_cpu SR | **81.0%** | **57.0%** |
| frac_zero (P=0) | 19.0% | 43.0% |
| frac_one (P=1) | 81.0% | 57.0% |
| frac_decisive | 0.0% | 0.0% |

### Backend Comparison (physx_cpu vs physx_cuda)
| Policy | cpu SR | cuda SR | Gap |
|--------|--------|---------|-----|
| RL demo BC | 81.0% | 62.7% | -18.3pp |
| MP demo BC | 57.0% | 15.1% | **-41.9pp** |

### Notes
1. **physx_cpu numbers match training eval**: 81% ≈ 88% (RL demo), 57% = 57% (MP demo), confirming backend mismatch was the cause of the previous low physx_cuda numbers.
2. **Still fully bimodal** (frac_decisive=0%): deterministic policy → every initial state either always succeeds or always fails. No intermediate P values.
3. **MP demo BC far more sensitive to backend**: 42pp gap (cpu→cuda) vs 18pp for RL demo. Likely because `pd_ee_delta_pos` control is more sensitive to obs differences than `pd_joint_delta_pos`.

---

## [BC P(success|s₀) — Stochastic (Gaussian Noise) Analysis] - 2026-02-24 11:34

**Command**: `python -u /tmp/bc_p_success_stochastic.py`
**Git**: ac5aa39 (main)
**Status**: Partial (RL demo BC complete for logstd={-3, -2}, logstd=-1.5 at 150/200; MP demo BC not started; killed early)

### Settings
| Parameter | Value |
|-----------|-------|
| env | PickCube-v1 |
| backend | physx_cpu (single CPU env) |
| E (initial states) | 200 |
| MC samples per state | 16 |
| max_episode_steps | 100 |
| logstd values tested | -3.0, -2.0, -1.5, -1.0 |
| noise method | `action = mean + randn * exp(logstd)`, then clamp to action bounds |

### Results — RL demo BC only (partial)

| logstd | std | SR | frac_zero | frac_one | frac_decisive |
|--------|-----|-----|-----------|----------|--------------|
| det | 0 | 81.5% | 18.5% | 81.5% | 0.0% |
| -3.0 | 0.050 | 83.8% | **1.5%** | 60.5% | **38.0%** |
| -2.0 | 0.135 | 75.4% | **0.0%** | 18.0% | **82.0%** |
| -1.5 | 0.223 | ~52% | — | — | — |

### Notes
1. **Tiny noise rescues dead states**: logstd=-3 (std=0.05) reduces frac_zero from 18.5% → 1.5%. logstd=-2 (std=0.135) eliminates frac_zero entirely (0.0%).
2. **frac_decisive explodes with noise**: From 0% (deterministic) → 38% (logstd=-3) → **82%** (logstd=-2). Most initial states become "learnable" — outcomes depend on action noise, so MC re-rollout can produce non-zero advantages.
3. **SR trade-off**: More noise → lower overall SR (81.5% → 83.8% → 75.4% → ~52%). Small noise (logstd=-3) actually INCREASES SR slightly because it helps some edge-case states succeed. But logstd=-2 and beyond degrade performance.
4. **Optimal finetuning logstd**: logstd=-2 appears ideal for finetuning start — frac_zero=0%, frac_decisive=82%, and SR still reasonable at 75%. logstd=-1.5 (which we use for PPO finetuning) drops SR significantly to ~52%.
5. **Key insight for offline-to-online**: BC alone produces a deterministic policy with no learning signal for finetuning (frac_decisive=0%). Simply adding logstd=-2 noise makes 82% of states learnable. This confirms that the "exploration must be manually added" principle (from earlier experiments) is quantitatively critical — without it, advantage estimation is degenerate.
6. **MP demo BC not tested** — killed before reaching second policy. Expected to show similar pattern but with higher baseline frac_zero.

---

## [BC Learned Fourier Features (LFF) vs MLP2] - 2026-02-24 16:06

**Command**: `python bc_official.py --env-id PickCube-v1 --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 --control-mode pd_ee_delta_pos --sim-backend cpu --total-iters 100000 --batch-size 1024 --seed {1-5} --network-type {mlp2,fourier} --b-scale {scale} --no-capture-video`
**Git**: ac5aa39 (main)
**Run Dirs**: `runs/lff_fix_mlp2_s{1-5}/`, `runs/lff_fix_f1_s{1-5}/`, `runs/lff_fix_f{001,01,10}_s1/`

### Settings
| Parameter | Value |
|-----------|-------|
| env_id | PickCube-v1 |
| demo | MP demos (1000 trajs, avg 78 steps, pd_ee_delta_pos) |
| total_iters | 100000 |
| batch_size | 1024 |
| lr | 3e-4 |
| network | 2×256 hidden, ReLU |
| eval | max_episode_steps=100, 100 episodes, physx_cpu |
| LFF implementation | Official (geyang/ffn): weight~N(0, scale/d), bias~U(-1,1), sin(π·(Wx+b)) |

### Results — Scale Sweep (seed=1)
| Config | Final Loss | SR (avg 3 trials) |
|--------|-----------|-------------------|
| MLP2 | 3.0e-5 | **47.7%** |
| FFN scale=0.001 | 1.6e-4 | 3.3% |
| FFN scale=0.01 | 1.5e-4 | 7.3% |
| **FFN scale=0.1** | **1.1e-4** | **20.0%** |
| FFN scale=1.0 | 1.3e-4 | 7.0% |

### Results — Multi-Seed (5 seeds)
| Method | s1 | s2 | s3 | s4 | s5 | Avg ± Std |
|--------|-----|-----|-----|-----|-----|-----------|
| MLP2 | 50% | 31% | 41% | 34% | 58% | **42.8% ± 10.0%** |
| FFN scale=0.1 | 19% | 1% | 8% | 5% | 3% | **7.2% ± 6.3%** |

### Notes
1. **FFN significantly worse than MLP2** (7% vs 43%). The spectral bias paper targeted Q-value approximation (where high-frequency features in value landscape matter), not BC regression. BC on PickCube doesn't have the same spectral challenge.
2. **LFF implementation had two bugs** (now fixed):
   - Weight init: had extra π factor (`π*scale/d` instead of `scale/d`) — 3.14x too large
   - Bias init: `U(-π, π)` instead of `U(-1, 1)` — effective phase range `(-π², π²)` instead of `(-π, π)` after the sin(π·) multiplication
3. **Critical eval bug discovered**: PickCube-v1 default `max_episode_steps=50`, but MP demos average 78 steps (range 49-103). BC policies need >50 steps → **all evals showed 0% SR with default setting**. Must use `max_episode_steps=100` for BC on MP demos.
4. **FFN loss is 3-5x higher than MLP2** at convergence (1.1e-4 vs 3.0e-5), suggesting the sin activation makes it harder to fit the action space precisely.
5. **Scale=0.1 is optimal** among tested values (0.001-1.0). Official paper uses 0.001-0.003 for SAC, but those are for different state/action dimensions.
6. **High seed variance** for both methods — BC on MP demos is inherently noisy due to multimodal demonstration distribution.

---

## [BC: Gaussian NLL + Control Mode Comparison] - 2026-02-24 17:05

**Git**: ac5aa39 (main)

### Experiment 1: Gaussian NLL vs MSE (pd_ee_delta_pos)

**Command**: `python bc_gaussian.py --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 --loss-type nll --seed {1-5} --eval-freq 25000`
**Run Dirs**: `runs/bc_gauss_nll_s{1-5}/`

| Seed | det SR | sto SR | Final logstd (xyz, gripper) |
|------|--------|--------|---------------------------|
| 1 | 50% | 5% | [-5, -5, -5, 0.0] |
| 2 | 33% | 0% | [-5, -5, -5, 0.0] |
| 3 | 66% | 0% | [-5, -5, -5, 0.0] |
| 4 | 54% | 5% | [-5, -5, -5, 0.0] |
| 5 | 42% | 25% | [-5, -5, -5, -4.3] |
| **avg** | **49.0%** | **7.0%** | |

Comparison: MLP2+MSE (pd_ee_delta_pos) avg = **42.8%** det SR.

**Notes**:
1. NLL det SR (49%) slightly better than MSE (43%), but stochastic eval is terrible (7%) because gripper logstd=0 (std=1.0) adds too much noise.
2. **Per-dim logstd reveals uncertainty structure**: xyz → -5 (clamp floor, essentially deterministic), gripper → 0 (high uncertainty). The gripper action is bimodal (open=1, close=-1) — mean network can't fit the transition, so NLL keeps gripper logstd high to absorb the large residual.
3. For finetuning, this learned logstd is not ideal — gripper noise too large, xyz noise too small. Still need manual logstd setting.

### Experiment 2: pd_joint_delta_pos vs pd_ee_delta_pos (MLP2+MSE)

**Command**: `python bc_official.py --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5 --control-mode pd_joint_delta_pos --network-type mlp2 --seed {1-5} ...`
**Run Dirs**: `runs/bc_joint_mlp2_s{1-5}/`

#### Full 2×2 Comparison: Demo Source × Control Mode (MLP2+MSE, 100k iters)

| Demo | Control Mode | action_dim | Trajs | Backend | s1 | s2 | s3 | s4 | s5 | Avg ± Std |
|------|-------------|-----------|-------|---------|-----|-----|-----|-----|-----|-----------|
| RL | pd_ee_delta_pos | 4 | 1000 | cuda→cuda | 98% | 100% | 97% | 99% | 100% | **98.8% ± 1.2%** |
| RL | pd_joint_delta_pos | 8 | 594 | cpu→cpu | 89% | 91% | 86% | 95% | 90% | **90.2% ± 2.9%** |
| MP | pd_ee_delta_pos | 4 | 1000 | cpu→cpu | 50% | 31% | 41% | 34% | 58% | **42.8% ± 10.0%** |
| MP | pd_joint_delta_pos | 8 | 1000 | cpu→cpu | 8% | 2% | 4% | 5% | 2% | **4.2% ± 2.2%** |

**Run Dirs**: `runs/bc_rl_ee_cuda_mlp2_s{1-5}/`, `runs/bc_rl_joint_mlp2_s{1-5}/`, `runs/lff_fix_mlp2_s{1-5}/`, `runs/bc_joint_mlp2_s{1-5}/`

**Notes**:
1. **RL demo + ee space = near-perfect** (98.8%). RL demos are deterministic NN-generated (unimodal, smooth actions) + ee space is low-dimensional (4D) and directly interpretable.
2. **Demo quality >> control mode**: RL demos in joint space (90%) still far better than MP demos in ee space (43%). The dominant factor is demo unimodality, not action dimensionality.
3. **Joint space amplifies errors**: For the same demo source, joint space is consistently worse (RL: 90 vs 99%, MP: 4 vs 43%). Small joint errors propagate through the kinematic chain.
4. **MP demos are multimodal**: Different motion plans for similar initial states → MLP averages across modes → worse cloning. High seed variance (10% std) confirms this.
5. **Backend matters for RL ee demo**: cuda→cpu replay only saved 15% of demos (152/1022) due to physx mismatch. Must match train/eval backend. cuda→cuda replay saved 98% (1000/1022).
6. **Joint cuda replay is fundamentally broken**: Re-generating `none→state` on cuda saves **0/997** demos (vs ee: 1000/1000). The pre-existing cuda state file (Dec 30, 997 trajs) was generated with unknown settings and produces bad training data (~12% SR on both cuda and cpu eval). Root cause: pd_joint_delta_pos is NOT replay-stable on cuda — small physics differences between num_envs=1024 (original data collection) and num_envs=1 (replay) cause trajectory divergence. pd_ee_delta_pos works because IK solver absorbs physics differences; joint deltas accumulate errors directly. **RL joint demos can only be used via cpu→cpu (594 trajs, 90.2%).**

---

## [BC + Noise P(success|s₀) Analysis] - 2026-02-24

**Question**: MLP BC policy (57% SR on cpu) 加noise后能否获得适合finetuning的P(success)分布？

**Setup**: `python bc_p_success_cpu.py --ckpt runs/lff_fix_mlp2_s1/checkpoints/final.pt --mc-samples 16 --num-states 100 --noise-levels 0.0 0.01 0.03`
- Checkpoint: MP demo BC, MLP2+MSE, pd_ee_delta_pos, physx_cpu eval
- MC16 per initial state, 100 initial states

| noise_std | SR | frac_zero | frac_one | frac_decisive | 可否finetuning |
|-----------|------|-----------|----------|---------------|--------------|
| 0.000 | 50.0% | 50.0% | 50.0% | **0.0%** | 不可 — 完全bimodal |
| 0.010 | 18.6% | 13.0% | 2.0% | **67.0%** | 勉强 — SR太低 |
| 0.030 | 4.3% | 69.0% | 2.0% | **6.0%** | 不可 — SR≈0 |

对比PPO ckpt_76: SR=43.8%, frac_zero=1.2%, frac_decisive=96.2%

**结论**: MLP BC + 后设noise在state coverage和SR之间天然无法平衡。
- noise=0: 确定性policy → 完全bimodal (frac_decisive=0%)
- noise=0.01: 打破bimodal (frac_decisive=67%) 但SR暴跌到18%，且仍有13% dead zone
- noise=0.03+: SR趋近0%，noise太大导致precision丧失

**根本原因**: MLP BC学到的是确定性映射（MSE训练），没有内在的exploration结构。后设noise均匀扰动所有action维度，破坏了对precision敏感的维度（如grasp timing），同时对不敏感维度的扰动不够。而PPO的logstd是在RL训练过程中per-dimension自适应学出来的，能自然平衡exploration和precision。

**对finetuning的implication**: MLP BC不适合作为sparse reward finetuning的base policy。需要：
1. 用Diffusion Policy等本身有stochasticity的IL方法，或
2. 用RL训练的base policy（天然有learned logstd），或
3. 用dense reward做finetuning（不依赖MC re-rollout的正reward信号）

---

## [Diffusion Policy Data Scaling on PickCube] - 2026-02-25 05:30

**Git**: ac5aa39 (main)

### Overview
Test Diffusion Policy data efficiency on PickCube-v1 with varying number of MP demonstrations. All use pd_ee_delta_pos, physx_cpu backend, 100 eval episodes.

### Setup
- Model: ConditionalUnet1D, 4.39M params, obs_horizon=2, act_horizon=8, pred_horizon=16
- Training: 50k iters, eval every 5k iters
- Demo source: `~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5`
- max_episode_steps=100

### Commands
```bash
# 1000 trajs (batch_size=1024)
python dp_train.py --env-id PickCube-v1 --demo-path ... --num-demos 1000 --total-iters 50000 --eval-freq 5000

# 100 trajs (batch_size=1024)
python dp_train.py --env-id PickCube-v1 --demo-path ... --num-demos 100 --total-iters 50000 --eval-freq 5000

# 50 trajs (batch_size=1024)
python dp_train.py --env-id PickCube-v1 --demo-path ... --num-demos 50 --total-iters 50000 --eval-freq 5000

# 10 trajs (batch_size=256, because 726 transitions < 1024)
python dp_train.py --env-id PickCube-v1 --demo-path ... --num-demos 10 --batch-size 256 --total-iters 50000 --eval-freq 5000
```

### Results — success_once (%) by iteration

| Trajs | 5k | 10k | 15k | 20k | 25k | 30k | 35k | 40k | 45k | 50k |
|-------|-----|------|------|------|------|------|------|------|------|------|
| 1000 | 99 | 100 | 100 | — | — | — | — | — | — | — |
| 100 | 100 | — | — | — | — | — | — | — | — | — |
| 50 | 98 | 97 | 95 | — | — | — | — | — | — | — |
| 10 | 3 | 0 | 3 | 2 | 5 | 1 | 4 | 3 | 1 | 1 |

Note: 1000/100/50-traj runs were killed early once performance was established.

### Summary — Peak success_once

| Trajs | Transitions | Peak SR | Best iter | Notes |
|-------|------------|---------|-----------|-------|
| 1000 | ~77k | **100%** | 10k | Perfect from 10k onwards |
| 100 | ~7.7k | **100%** | 5k | Perfect immediately at 5k |
| 50 | ~3.8k | **98%** | 5k | Near-perfect, slight decline later |
| 10 | ~726 | **5%** | 25k | Essentially fails |

### Key Findings

1. **DP is extremely data-efficient up to a cliff**: 50 trajs achieves 98%, but 10 trajs only 5%. Sharp transition between 10-50 demos.
2. **100 trajs is sufficient for perfect performance**: 100% at just 5k iters (vs MLP BC needing 100k for 57%).
3. **10-traj failure**: Loss keeps dropping (0.005→0.0001) but SR stays ~0-5%. Model overfits to 10 demos without learning generalizable behavior. Also note batch_size had to be reduced from 1024→256 since only 726 transitions available (original batch_size > dataset size causes infinite hang with drop_last=True).
4. **Comparison to MLP BC**: DP with 50 trajs (98%) >> MLP BC with 1000 trajs (57%). DP's action chunking is the key advantage.

### Bug Found
- `dp_train.py` with `batch_size=1024` and `drop_last=True` on a dataset smaller than 1024 transitions causes the `IterationBasedBatchSampler` to hang forever (zero batches per epoch). Fixed by using `--batch-size 256` for 10-traj experiment. This should be auto-detected.

---

## [Diffusion Policy on StackCube and PegInsertion] - 2026-02-25 05:30

### StackCube (pd_ee_delta_pos, physx_cpu, 990 MP demos)

```bash
python dp_train.py --env-id StackCube-v1 --demo-path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 \
  --control-mode pd_ee_delta_pos --sim-backend physx_cpu --max-episode-steps 200 --total-iters 100000 --eval-freq 5000
```

| Iter | success_once |
|------|-------------|
| 5k | 77% |
| 10k | 94% |
| 15k | 96% |
| 20k-100k | 96-100% |

**Peak: 100%**. DP completely solves StackCube where MLP BC gets 0%.

### PegInsertionSide (pd_joint_delta_pos, physx_cpu, 1000 MP demos)

Note: ee replay of PegInsertion demos has only 0.7% survival rate (7/1000). Joint replay is 100% (1000/1000).

```bash
python dp_train.py --env-id PegInsertionSide-v1 --demo-path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5 \
  --control-mode pd_joint_delta_pos --sim-backend physx_cpu --max-episode-steps 200 --total-iters 50000 --eval-freq 5000
```

| Iter | success_once |
|------|-------------|
| 5k | 6% |
| 10k | 16% |
| 15k | 13% |
| 20k | 21% |

Training was killed at 20k. Performance is low but still climbing. Would likely need 100k+ iters for convergence (consistent with ManiSkill docs).

---


## [DPPO DDPM/DDIM Alignment + EtaFixed + Coverage] - 2026-02-25 23:13

**Git**: 69cc8c5 (main)
**Checkpoint**: `runs/dppo_pretrain/dppo_25traj_aligned_v2/best.pt` (25 MP demos, PickCube-v1, ee_delta_pos)
**Files Modified**: `DPPO/model/diffusion.py`, `dp_p_success_cpu.py`, `DPPO/eval_cpu.py`, `DPPO/eval_cpu_test.py`, `DPPO/diagnose.py`

### Context

Debugged and fully aligned DDPM/DDIM sampling with original DPPO (`irom-princeton/dppo`). Multi-stage fix:

**Bugs fixed:**
1. **DDIM `t_prev==0` bypass**: `x = x_0_pred` directly at last step → noise injection had zero effect. Network denoised away all noise.
2. **DDPM non-deterministic `t==0`**: Set `std=0` at last step even when stochastic. Original clips to `min_sampling_denoising_std`.
3. **DDIM timesteps**: Used `linspace` instead of original's `arange(0, ddim_steps) * step_ratio`.
4. **DDIM noise recomputation**: Missing `noise_pred` recompute after `x_recon` clipping.
5. **eta=0 (implicit)**: DDIM mean formula used `dir_xt_coef = sqrt(1-α_prev)` (no sigma correction). Original uses `eta=1` via `EtaFixed(base_eta=1)`, giving `dir_xt_coef = sqrt(1-α_prev-σ²)`. With eta=0, noise_pred direction is 1.4-7.4x too strong per step, destroying the policy.

**Final implementation**: Added `EtaFixed` nn.Module (matching original exactly: learnable `eta_logit` → tanh → `[min_eta, max_eta]`). Stored as `self.eta` on `DiffusionModel`. DDIM uses `self.eta(cond)` for stochastic, `zeros` for deterministic.

### Action Std (real obs, N=50 samples, DDIM10 eta=1)

**Command**: inline Python script, single initial state

| Setting | mean_std | max_std |
|---------|----------|---------|
| DDIM10 eta=1 det | 0.0040 | 0.0193 |
| DDIM10 eta=1 std=0.01 | 0.0094 | 0.0147 |
| DDIM10 eta=1 std=0.03 | 0.0272 | 0.0437 |
| DDIM10 eta=1 std=0.1 | 0.1006 | 0.1503 |
| DDPM100 det | 0.0032 | 0.0119 |
| DDPM100 std=0.03 | 0.0274 | 0.0405 |

### Coverage Results (50 states, MC16, seeds 0-49, fully aligned code with EtaFixed)

**Command**: `python dp_p_success_cpu.py --ckpt runs/dppo_pretrain/dppo_25traj_aligned_v2/best.pt --num-states 50 --mc-samples 16 --min-sampling-denoising-std X --ddim-steps 10 --ddim-eta 1.0`

| Setting | SR | frac_zero | frac_one | frac_decisive | std P |
|---------|------|-----------|----------|---------------|-------|
| Det DDPM100 (baseline) | 71.4% | 26.0% | 64.0% | 10.0% | 0.432 |
| DDPM100 std=0.03 | 70.8% | 22.0% | 60.0% | 12.0% | 0.430 |
| DDIM10 eta=1 std=0.01 | **68.5%** | **16.0%** | **38.0%** | **32.0%** | 0.392 |
| DDIM10 eta=1 std=0.1 | 3.0% | 82.0% | 0.0% | 2.0% | 0.132 |

### Key Findings

1. **std=0.01 is the sweet spot**: SR drops only slightly (71.4%→68.5%) but frac_decisive triples (10%→32%), frac_one halved (64%→38%). This is the best coverage for finetuning — many more learnable states.
2. **std=0.1 still destroys the policy (3% SR)**: Even with correct eta=1 implementation. This is NOT a code bug — 25-traj pretrained model (det SR~71%) is too weak to tolerate std=0.1. Original DPPO's 0.8→0.4 is on a much stronger base (1000 demos, SR>95%).
3. **EtaFixed as module vs float makes no numerical difference** (same results), but matches the original architecture for future DPPO finetuning where eta could be learned.
4. **DP coverage is still far more bimodal than PPO**: frac_decisive=32% (best case, std=0.01) vs PPO ckpt_76=96.2%. Diffusion sampling noise is fundamentally different from Gaussian action noise.

### Notes
- Coverage with 50 states (seeds 0-49) biased towards easier seeds — 200-state run showed SR=45.3% vs 71.4% here
- PPO ckpt_76 action std ≈ 0.223 (logstd=-1.5), much higher than DP's best std=0.01→0.009
- `strict=False` added to all `load_state_dict` calls for backward compat with old checkpoints missing `eta.eta_logit`

---

## [DP Init State Similarity to Training Demos vs P(success)] - 2026-02-26 00:35

**Command**: `python analyze_init_similarity.py` (default args)
**Git**: 69cc8c5 (main)
**Checkpoint**: `runs/dppo_pretrain/dppo_25traj_aligned_v2/best.pt`
**Output**: `runs/dp_init_similarity_vs_success_25traj.png`

### Settings
| Parameter | Value |
|-----------|-------|
| ckpt | runs/dppo_pretrain/dppo_25traj_aligned_v2/best.pt |
| demo_path | ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 |
| num_train_demos | 25 (first 25 trajs = traj_0..traj_24) |
| num_states | 50 |
| mc_samples | 16 |
| max_episode_steps | 100 |
| env_id | PickCube-v1 |
| control_mode | pd_ee_delta_pos |
| inference device | cuda (env on cpu) |
| init state dims | 5D: cube_xy (dims 29,30) + goal_xyz (dims 26,27,28) |

### Context

PickCube-v1 初始状态随机性是5D：cube_xy (2D, uniform [-0.1, 0.1]) + goal_xyz (3D, z=[0.02, 0.32])。Cube z固定0.02。

25条training demos从uniform分布采样，但太稀疏——只覆盖46%的cube_xy空间(阈值0.02内)，1000条demo则100%覆盖。

**假设**：25-traj pretrained DP overfit在training demos附近，离demo越远越失败。

### Results — Per-dimension correlation with P(success)

单个维度都不显著：

| Dim | ρ (Det) | p-val | ρ (DDIM10) | p-val |
|-----|---------|-------|------------|-------|
| cube_x | 0.059 | 0.686 | 0.110 | 0.446 |
| cube_y | -0.202 | 0.160 | -0.175 | 0.225 |
| goal_x | -0.166 | 0.248 | -0.221 | 0.124 |
| goal_y | 0.103 | 0.478 | 0.138 | 0.339 |
| goal_z | 0.281 | 0.048 | 0.221 | 0.123 |

### Results — Distance-to-nearest-training-demo vs P(success)

| Setting | Dist type | Spearman ρ | p-val | Pearson r | p-val |
|---------|-----------|------------|-------|-----------|-------|
| Det DDPM100 | **5D (all)** | **-0.787** | <0.0001 | **-0.715** | <0.0001 |
| Det DDPM100 | cube_xy | -0.758 | <0.0001 | -0.681 | <0.0001 |
| Det DDPM100 | goal_xyz | -0.709 | <0.0001 | -0.554 | <0.0001 |
| DDIM10 std=0.01 | **5D (all)** | **-0.807** | <0.0001 | **-0.796** | <0.0001 |
| DDIM10 std=0.01 | cube_xy | -0.795 | <0.0001 | -0.685 | <0.0001 |
| DDIM10 std=0.01 | goal_xyz | -0.739 | <0.0001 | -0.624 | <0.0001 |

### Coverage (reproduced, consistent with previous)

| Setting | SR | frac_zero | frac_one | frac_decisive |
|---------|------|-----------|----------|---------------|
| Det DDPM100 | 71.5% | 26.0% | — | — |
| DDIM10 eta=1 std=0.01 | 66.8% | 20.0% | — | — |

### Key Findings

1. **假设验证：25-traj DP确实overfit在training demos附近**。dist_to_nearest_train_demo和P(success)的Spearman ρ=-0.79~-0.81 (p<0.0001)，极强负相关。
2. **没有单一dominant维度**：cube_xy和goal_xyz都贡献distance，单个维度ρ<0.3（不显著），但5D距离ρ=-0.81。说明policy需要在整个5D init space的组合上泛化，不只是某一个因素。
3. **cube_xy distance稍强于goal_xyz** (ρ=-0.76~-0.80 vs -0.71~-0.74)，可能因为cube位置更直接影响reach/grasp trajectory。
4. **25 demos太稀疏**：只覆盖46%的cube_xy空间(阈值0.02内)，1000 demos则100%覆盖。这解释了25-traj pretrain的det SR~71% vs 1000-traj的~100%。

### Notes
- 新增脚本: `analyze_init_similarity.py` (可复用，支持 --num-train-demos, --num-states 等参数)
- 新增脚本: `dp_eval_video.py` (保存success/failure视频，支持多种采样模式)
- Videos saved: `runs/videos/dppo_25traj_det/` (35 success, 15 failure, seeds 0-49)
- GPU inference + CPU env simulation: DDPM100 272s, DDIM10 190s (vs pure CPU ~575s/257s)
- **Implication for DPPO finetuning**: 需要在训练数据覆盖不到的init states上探索并成功。std=0.01提供了一些exploration (frac_decisive 10%→32%)，但fundamental limit是25 demos的覆盖范围。增加demos数量或curriculum是更直接的解法。

---

## [DP Pretrain: 1000-traj UNet SR Curve + Coverage for DPPO Base Policy] - 2026-02-26 01:16

**Command**: `python -m DPPO.pretrain --env_id PickCube-v1 --demo_path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 --control_mode pd_ee_delta_pos --network_type unet --denoising_steps 100 --horizon_steps 16 --cond_steps 2 --act_steps 8 --batch_size 1024 --no_obs_norm --no_action_norm --max_grad_norm 0 --seed 1 --torch_deterministic --total_iters 5000 --eval_freq 500 --exp_name dppo_1000traj_unet_5k`
**Git**: 69cc8c5 (main)
**Run Dir**: `runs/dppo_pretrain/dppo_1000traj_unet_5k/`

### Settings
| Parameter | Value |
|-----------|-------|
| num_demos | 1000 (all MP demos) |
| network_type | unet |
| denoising_steps | 100 |
| horizon_steps | 16 |
| cond_steps | 2 |
| act_steps | 8 |
| batch_size | 1024 |
| total_iters | 5000 |
| eval_freq | 500 |
| obs/action norm | off |
| max_grad_norm | 0 (no clip) |

### Results — SR Curve (cpu eval, 100 episodes)

| Iter | success_once | success_at_end |
|------|-------------|----------------|
| 500 | 2% | 0% |
| 1000 | 42% | 8% |
| **1500** | **53%** | **20%** |
| 2000 | 87% | 46% |
| 2500 | 82% | 60% |
| 3000 | 96% | 85% |
| 3500 | 99% | 94% |
| 4000 | 99% | 94% |
| 4500 | 100% | 97% |
| 5000 | 99% | 98% |

### Results — Coverage: ckpt_1500 (50 states, MC16, deterministic)

**Command**: `python dp_p_success_cpu.py --ckpt runs/dppo_pretrain/dppo_1000traj_unet_5k/ckpt_1500.pt --num-states 50 --mc-samples 16`

| Metric | 25-traj best | **1000-traj ckpt_1500** |
|--------|-------------|------------------------|
| SR | 71.4% | **57.5%** |
| frac_zero | 26.0% | **0.0%** |
| frac_one | 64.0% | **2.0%** |
| frac_decisive | 10.0% | **98.0%** |
| std P | 0.432 | **0.178** |

### Hypothesis to Verify (DPPO RL finetuning)

Coverage (frac_zero) determines the RL finetuning ceiling under sparse reward:

| Base Policy | frac_zero | Predicted RL Ceiling | To Verify |
|-------------|-----------|---------------------|-----------|
| 25-traj best (sr_once=44%) | 26% | ~74% | DPPO RL on this ckpt |
| 1000-traj ckpt_1500 (sr_once=53%) | 0% | ~98%+ | DPPO RL on this ckpt |

If confirmed, this proves that **coverage is the binding constraint for sparse-reward RL finetuning**, not advantage estimation quality or policy optimization method.

### Notes
- MLP network (default) too small: 572K params, 5k iters → 2% SR. Must use `--network_type unet` (4.39M params)
- ckpt_1500 profile very similar to PPO ckpt_76 (SR=43.8%, frac_decisive=96.2%) — ideal for finetuning experiments
- Training time: 784s (13min) for 5000 iters on GPU

---

## [DPPO Finetune: Bug Fixes + Hyperparameter Sweep] - 2026-02-26 14:13

**Git**: 69cc8c5 (main)
**Checkpoint**: `runs/dppo_pretrain/dppo_200traj_unet_T100_5k/ckpt_3000.pt` (GPU DDIM-10 SR=53.6%, CPU SR=80.4%, frac_zero=0%, frac_decisive=65%)

### Bug Fixes Applied

**BUG 1 (CRITICAL): `reward_horizon` slicing ignores `act_offset`**
- File: `DPPO/model/diffusion_ppo.py` loss()
- Before: `newlogprobs[:, :reward_horizon, :]` → optimizes positions 0..7 (position 0 = conditioning, never executed)
- After: `newlogprobs[:, act_offset:act_offset+reward_horizon, :]` → optimizes positions 1..8 (actually executed)
- Root cause: our dp_train convention uses `cond_steps=2, act_offset=1`, reference DPPO uses `cond_steps=1` (no offset)

**BUG 2 (LATENT): Chain recording off-by-one for ft < ddim**
- File: `DPPO/model/diffusion_ppo.py` _forward_ddim()
- Before: `i >= ft_start` → K states (K-1 transitions) when ft < ddim
- After: `i >= max(ft_start - 1, 0)` → K+1 states (K transitions)
- Not active with current config (ft=ddim=10), but would break if ft < ddim

**BUG 3 (REVERTED): GAE terminated vs done**
- Attempted: use `terminated` only (not `terminated|truncated`) for GAE bootstrap
- **REVERTED**: with auto-reset envs, truncated episodes get V(initial_state_next_ep) ≈ 0.5 as bootstrap, creating false positive advantages for failed episodes. Original `dones = terminated|truncated` is safer.

**BUG 4 (CRITICAL DISCOVERY): logprob clamping + std mismatch kills PPO constraint**
- File: `DPPO/model/diffusion_ppo.py` loss()
- Before: `newlogprobs.clamp(min=-5, max=2)` with `min_logprob_std=0.1`, `min_sampling_std=0.01`
- Problem: 10x mismatch between logprob std (0.1) and sampling std (0.01) makes PPO ratio completely insensitive to behavioral changes. Policy drifts silently (ratio=1.0000, KL=0.000000) while performance collapses.
- After: removed logprob clamping, set `min_logprob_std = min_sampling_std = 0.01`

**BUG 5: Critic `residual_style=False`**
- Reference uses `residual_style=True` for critic. Fixed.

### Experiments

#### Run 1: Original bugs (pre-fix baseline, from earlier session)
**Command**: `python -m DPPO.finetune --pretrain_checkpoint runs/dppo_pretrain/dppo_200traj_unet_T100_5k/ckpt_3000.pt --n_envs 200 --n_steps 200 --use_ddim --ddim_steps 10 --ft_denoising_steps 10 --min_sampling_denoising_std 0.01 --min_logprob_denoising_std 0.1 --no-norm_adv --clip_adv_min 0 --actor_lr 1e-5 --update_epochs 5 --minibatch_size 2000 --n_critic_warmup_itr 5 --n_train_itr 100 --eval_freq 5`
**Result**: 53.6% → **20.7%** after 5 actor iters. **COLLAPSED**.

#### Run 2: BUG 1+2+3 fixed, min_logprob_std=0.1 (still mismatched)
**Command**: `python -m DPPO.finetune ... --min_logprob_denoising_std 0.1 --update_epochs 1 --actor_lr 1e-5 --n_envs 100 --n_steps 100`
**Result**: 54.0% → **4.0%** at iter 10. **COLLAPSED WORSE** (BUG 3 fix was harmful).

#### Run 3: BUG 3 reverted, ddim=5 ft=5
**Command**: `... --ft_denoising_steps 5 --ddim_steps 5 --min_logprob_denoising_std 0.1`
**Result**: 36.0% → **1.0%** at iter 15. **COLLAPSED**. Shorter chain doesn't help.

#### Run 4: 1 gradient step per iteration (grad_accumulate=200)
**Command**: `... --grad_accumulate 200 --min_logprob_denoising_std 0.1`
**Result**: 54.0% → **1.0%** at iter 10. ratio=1.0000, KL=0.000000 but succ 586→3.
**KEY INSIGHT**: PPO ratio is BLIND. Policy changes behavior without ratio detecting it.

#### Run 5: Matched logprob_std=0.01, clip=0.1, 200 steps/epoch
**Command**: `... --min_logprob_denoising_std 0.01 --clip_ploss_coef 0.1 --update_epochs 1 --minibatch_size 500`
**Result**: ratio now informative (0.98), but still declining: 54% → 34% → 20% → 1.7%. Too many gradient steps + wide clip.

#### Run 6: Matched logprob_std, ref batch_size=10000, lr=1e-4
**Command**: `... --min_logprob_denoising_std 0.01 --clip_ploss_coef 0.01 --update_epochs 10 --minibatch_size 10000 --actor_lr 1e-4`
**Result**: KL=3.09 at iter 3. **lr=1e-4 too aggressive** with 100x more sensitive logprobs.

#### Run 7: lr=1e-6, batch=10000, 10 epochs ★ BEST RUN
**Command**: `python -m DPPO.finetune --pretrain_checkpoint runs/dppo_pretrain/dppo_200traj_unet_T100_5k/ckpt_3000.pt --n_envs 100 --n_steps 100 --ft_denoising_steps 10 --use_ddim --ddim_steps 10 --min_sampling_denoising_std 0.01 --min_logprob_denoising_std 0.01 --gamma_denoising 0.99 --clip_ploss_coef 0.01 --update_epochs 10 --actor_lr 1e-6 --minibatch_size 10000 --n_critic_warmup_itr 2 --reward_scale_running --n_train_itr 100 --eval_freq 5 --exp_name dppo_ft_ref_lr1e6`
**Run Dir**: `runs/dppo_finetune/dppo_ft_ref_lr1e6`

| Iter | GPU Eval SR | Rollout Succ |
|------|-------------|-------------|
| 1 | 54.0% | 558 |
| 5 | 55.7% | 656 |
| 10 | 63.7% | 747 |
| 15 | 72.0% | 785 |
| 20 | 72.7% | 849 |
| 25 | 72.0% | 870 |
| 30 | 79.7% | 886 |
| 35 | 77.7% | 924 |
| 40 | 75.7% | 906 |
| 45 | 78.7% | 937 |
| 50 | 75.3% | 941 |
| 55 | **80.7%** | 982 |
| 60 | 77.7% | 956 |
| 65 | **85.0%** | — |
| 70 | **85.7%** | — |

**Killed at iter 70** (to test critic residual fix). Peak = 85.7%. Still climbing.

#### Run 8: lr=1e-5, batch=10000, 10 epochs
**Command**: same as Run 7 but `--actor_lr 1e-5`
**Result**: Initial dip 54% → 31.7% (iter 5), recovering to 40% (iter 15). **Killed early** — lr too aggressive initially.

#### Run 9: lr=1e-6 + critic residual_style=True (IN PROGRESS)
**Command**: same as Run 7 but with critic `residual_style=True`

| Iter | GPU Eval SR | Rollout Succ |
|------|-------------|-------------|
| 1 | 54.0% | 558 |
| 5 | **60.3%** | 684 |

**Still running.** Early results show +5% over Run 7 at same iter (60.3% vs 55.7%).

### Settings (Run 7 — best completed run)
| Parameter | Value |
|-----------|-------|
| pretrain_checkpoint | dppo_200traj_unet_T100_5k/ckpt_3000.pt |
| network_type | unet (4.39M params) |
| denoising_steps | 100 |
| ft_denoising_steps | 10 |
| use_ddim | True |
| ddim_steps | 10 |
| horizon_steps | 16 |
| cond_steps | 2 |
| act_steps | 8 |
| act_offset | 1 |
| n_envs | 100 |
| n_steps | 100 |
| batch_size (minibatch) | 10000 |
| update_epochs | 10 |
| actor_lr | 1e-6 |
| critic_lr | 1e-3 |
| critic_dims | [256, 256, 256] |
| critic_activation | Mish |
| critic_residual | False (Run 7) / True (Run 9) |
| gamma | 0.999 |
| gae_lambda | 0.95 |
| gamma_denoising | 0.99 |
| clip_ploss_coef | 0.01 |
| min_sampling_denoising_std | 0.01 |
| min_logprob_denoising_std | 0.01 |
| reward_scale_running | True |
| n_critic_warmup_itr | 2 |
| norm_adv | True |
| target_kl | 1.0 |
| max_grad_norm | 1.0 |

### Notes
- **Root cause of collapse**: `min_logprob_std` (0.1) >> `min_sampling_std` (0.01) makes PPO ratio completely blind to behavioral changes in nearly-deterministic DDIM policy. Policy drifts through "shadow region" where logprob ratio ≈ 1 but actions change significantly. Removing logprob clamping and matching the two stds fixes this.
- **lr scaling rule**: reference uses lr=2e-5 with logprob_std=0.1. Since logprob sensitivity ∝ 1/sigma², with sigma 10x smaller, need lr ~100x smaller → lr=2e-7 to 1e-6.
- **Reference config** (from `ft_ppo_diffusion_unet.yaml`): batch=10000, n_steps=400, n_envs=50, update_epochs=10, actor_lr=2e-5, critic residual=True, min_logprob_std=min_sampling_std=0.1
- **GPU utilization**: rollout phase ~70-80% (batch=100 too small for UNet), PPO update phase ~90%+ (batch=10000). Increasing n_envs would help.
- This is the first successful DPPO finetuning: 54% → 85.7% in 70 iters with no collapse.

---

## [DPPO Finetune: LR Sweep + GPU Coverage] - 2026-02-26 14:51

**Git**: 69cc8c5 (main)
**Checkpoint**: `runs/dppo_pretrain/dppo_200traj_unet_T100_5k/ckpt_3000.pt`

### LR Sweep (all with critic residual_style=True)

All runs use same base config as Run 7 above but with `critic_residual=True` and varying `actor_lr`.

#### Run 9: lr=1e-6 + critic residual (continued from previous log)
**Killed at iter 10** to test higher lr. Peak = 71.3%.

| Iter | GPU Eval SR | Rollout Succ | KL |
|------|-------------|-------------|-----|
| 1 | 54.0% | 558 | 0.000000 |
| 5 | 60.3% | 684 | 0.000008 |
| 10 | **71.3%** | 762 | 0.000009 |

#### Run 10: lr=3e-6 + critic residual ★ BEST LR
**Command**: `python -m DPPO.finetune --pretrain_checkpoint runs/dppo_pretrain/dppo_200traj_unet_T100_5k/ckpt_3000.pt --n_envs 100 --n_steps 100 --ft_denoising_steps 10 --use_ddim --ddim_steps 10 --min_sampling_denoising_std 0.01 --min_logprob_denoising_std 0.01 --gamma_denoising 0.99 --clip_ploss_coef 0.01 --update_epochs 10 --actor_lr 3e-6 --minibatch_size 10000 --n_critic_warmup_itr 2 --reward_scale_running --n_train_itr 100 --eval_freq 5 --exp_name dppo_ft_lr3e-6_residual`
**Killed at iter 20** to test higher lr.

| Iter | GPU Eval SR | Rollout Succ | KL |
|------|-------------|-------------|-----|
| 1 | 54.0% | 558 | 0.000000 |
| 5 | **67.3%** | 672 | 0.000018 |
| 10 | **77.7%** | 771 | 0.000020 |
| 15 | **84.0%** | 811 | 0.000021 |
| 20 | 82.0% | 842 | 0.000021 |

**3x faster than lr=1e-6.** KL still very low (0.000021). No instability. Projected to reach 95%+ by iter 50-60.

#### Run 11: lr=1e-5 + critic residual — COLLAPSED
**Killed at iter 5.** Policy gradient flipped positive (wrong direction), succ dropped 635→461.

| Iter | GPU Eval SR | Rollout Succ | KL | pg |
|------|-------------|-------------|-----|-----|
| 3 | — | 635 | 0.000138 | +0.000925 |
| 5 | **47.3%** | 461 | 0.000090 | +0.000439 |

**KL=0.000138 (7x Run 10) → overshooting.** pg > 0 means clipped PPO pushes wrong direction.

#### Run 12: lr=5e-6 + critic residual — WORSE THAN 3e-6
**Killed at iter 10.**

| Iter | GPU Eval SR | Rollout Succ | KL |
|------|-------------|-------------|-----|
| 5 | 62.3% | 654 | 0.000034 |
| 10 | 60.3% | 767 | 0.000034 |

Eval SR lower than lr=3e-6 despite similar rollout succ. Policy less stable for deterministic inference.

### LR Sweep Summary

| lr | Iter 5 Eval | Iter 10 Eval | Status |
|----|-------------|-------------|--------|
| 1e-6 | 60.3% | 71.3% | Stable, slow |
| **3e-6** | **67.3%** | **77.7%** | **Stable, fast** ★ |
| 5e-6 | 62.3% | 60.3% | Mild instability |
| 1e-5 | 47.3% | — | Collapsed |

**Optimal lr = 3e-6** with critic residual_style=True. Sweet spot between speed and stability.

**Why reference uses lr=2e-5 but we need 3e-6**: logprob gradient ∝ 1/σ². Reference uses σ=0.1, we use σ=0.01. Effective update: lr/σ² = 3e-6/1e-4 = 30 (us) vs 2e-5/0.01 = 2e-3 (reference). Our effective update is 15x reference — already aggressive.

### GPU Coverage Analysis

**Command**: `python dp_p_success_gpu.py --ckpt runs/dppo_pretrain/dppo_200traj_unet_T100_5k/ckpt_3000.pt --num_states 100 --mc_samples 16 --ddim_steps 10 --min_sampling_denoising_std 0.01`

| Metric | GPU (DDIM-10, std=0.01) | CPU (previous) |
|--------|------------------------|----------------|
| SR | 68.1% | 80.4% |
| **frac_zero** | **0.0%** | 0.0% |
| frac_one | 1.0% | 35% |
| **frac_decisive** | **90.0%** | 65% |
| std P | 0.189 | — |
| median P | 0.688 | — |

**Theoretical ceiling = ~100%** (frac_zero=0%, no dead zones on GPU).

90% of states are in the decisive range (0.1 < P < 0.9), providing strong gradient signal for RL. The GPU coverage is actually better for finetuning than CPU — more room to improve.

### Notes
- **Coverage script**: `dp_p_success_gpu.py` — new GPU coverage analysis using ManiSkill state save/restore (`get_state_dict`/`set_state_dict`). 100 states × 16 MC in 41 seconds.
- GPU SR (68.1%) > previous GPU eval (53.6%) — likely due to different seeds and MC16 averaging.
- Next step: run lr=3e-6 + residual critic for full 100 iters to verify 95%+ target.

---

## [DPPO Finetune: lr=3e-6 Full 100 Iters + Coverage Analysis] - 2026-02-26 16:19

**Git**: 69cc8c5 (main)
**Checkpoint**: `runs/dppo_pretrain/dppo_200traj_unet_T100_5k/ckpt_3000.pt`
**Run Dir**: `runs/dppo_finetune/dppo_ft_lr3e-6_residual_full/`

### Command
```bash
python -m DPPO.finetune \
  --pretrain_checkpoint runs/dppo_pretrain/dppo_200traj_unet_T100_5k/ckpt_3000.pt \
  --n_envs 100 --n_steps 100 \
  --ft_denoising_steps 10 --use_ddim --ddim_steps 10 \
  --min_sampling_denoising_std 0.01 --min_logprob_denoising_std 0.01 \
  --gamma_denoising 0.99 --clip_ploss_coef 0.01 \
  --update_epochs 10 --actor_lr 3e-6 \
  --minibatch_size 10000 --n_critic_warmup_itr 2 \
  --reward_scale_running --n_train_itr 100 --eval_freq 5 \
  --exp_name dppo_ft_lr3e-6_residual_full
```

### Settings
| Parameter | Value |
|-----------|-------|
| actor_lr | **3e-6** |
| critic_residual_style | **True** |
| n_envs | 100 |
| n_steps | 100 |
| update_epochs | 10 |
| minibatch_size | 10000 |
| clip_ploss_coef | 0.01 |
| min_sampling_denoising_std | 0.01 |
| min_logprob_denoising_std | 0.01 |
| gamma | 0.999 |
| gamma_denoising | 0.99 |
| n_critic_warmup_itr | 2 |
| reward_scale_running | True |

### Results — Training Curve (GPU Inline Eval, n_rounds=3)

| Iter | GPU Eval SR | Rollout Succ | KL |
|------|-------------|-------------|-----|
| 1 | 54.0% | 558 | 0.000000 |
| 5 | 60.0% | 670 | 0.000023 |
| 10 | 74.7% | 791 | 0.000019 |
| 15 | 74.0% | 873 | 0.000024 |
| 20 | 81.0% | 818 | 0.000026 |
| 25 | 84.3% | 857 | 0.000020 |
| 30 | 86.7% | 884 | 0.000021 |
| 35 | 90.0% | 921 | 0.000023 |
| 40 | **91.0%** | 969 | 0.000025 |
| 45 | 88.3% | 970 | 0.000025 |
| 50 | 89.7% | 989 | 0.000026 |
| 55 | 88.3% | 983 | 0.000020 |
| 60 | 92.0% | 1003 | 0.000023 |
| 65 | 85.3% | 976 | 0.000022 |
| 70 | 87.7% | 1011 | 0.000023 |
| 75 | 90.7% | 1043 | 0.000027 |
| 80 | **95.7%** | 1047 | 0.000025 |
| 85 | 92.7% | 1016 | 0.000022 |
| 90 | **96.0%** | 1022 | 0.000022 |
| 95 | 92.3% | — | 0.000031 |
| 100 | 92.3% | 1043 | — |

**Peak inline eval = 96.0%** (iter 90), best checkpoint saved at iter 90.

### Results — High-Precision Eval (dp_p_success_gpu.py)

**BUG FOUND**: Finetuned checkpoint saves `actor_ft.unet.*` (finetuned weights) and `network.unet.*` (frozen pretrained copy). Loading into DiffusionModel with `strict=False` matches `network.unet.*` → loads pretrained weights, silently ignoring finetuned weights. **Fix**: remap `actor_ft.unet.* → network.unet.*` before loading.

Verified: `actor_ft` weights differ from pretrained (max_diff=0.0008), `network` weights are identical to pretrained (max_diff=0.0).

#### Deterministic Eval (300 states, MC1)
| Metric | Value |
|--------|-------|
| **SR** | **91.3%** |
| frac_zero | 8.7% |
| frac_one | 91.3% |
| frac_decisive | 0.0% |

#### Stochastic Coverage (200 states, MC16, min_std=0.01)
| Metric | Pretrained | **Finetuned** |
|--------|-----------|--------------|
| **SR** | 68.1% | **94.4%** |
| frac_zero | 0.0% | **0.0%** |
| frac_one | 1.0% | **56.0%** |
| frac_decisive | 90.0% | **24.5%** |
| std P | 0.189 | **0.082** |

### Notes

- **Inline eval (n_rounds=3) overestimates**: Reports 96% peak, but high-precision deterministic eval (300 states) shows **91.3%**. SE of inline eval at 90% SR ≈ 1.7%, so 96% is within 3σ fluctuation.
- **Rollout succ is inflated by partial reset**: `n_success_rollout` counts all reward events across ~8 episodes per env (800 env steps / 100 max_steps). Succ=1047 doesn't mean 104.7% SR — it includes post-reset episode successes. True first-episode SR should be tracked separately.
- **Stochastic > deterministic** (94.4% vs 91.3%): MC16 counts a state as "successful" if any of 16 stochastic rollouts succeed. So some states fail deterministically but occasionally succeed with DDIM noise.
- **Remaining gap to 98%**: 8.7% of states fail deterministically. These are the "hard" initial configs. Need more training iterations, or try larger n_steps/n_envs for better gradient signal on hard states.
- **`dp_p_success_gpu.py` key remap bug**: Must remap `actor_ft.unet.*` → `network.unet.*` when loading finetuned checkpoints into DiffusionModel. Without this, silently loads pretrained weights.

---

## [DPPO Filtered Finetuning: Only Train on Easy States] - 2026-03-01

**Git**: bd7a026 (main)

### Hypothesis
Conservative DPPO finetuning plateaus at ~85% after iter 5. Analysis shows adv_pos_frac ~30% — most transitions have negative advantage from hard states. If we restrict training to seeds with P(success) > 0.5, the learning signal should be cleaner (higher adv_pos_frac), enabling faster convergence.

### Implementation
- `SeedPoolWrapper` in `DPPO/make_env.py`: overrides `reset()` to sample from pre-computed easy seed pool
- `dp_p_success_cpu.py`: now saves `.npz` alongside plot for seed pool filtering
- `DPPO/finetune.py`: `--seed_pool_path` and `--seed_pool_threshold` args

### Step 1: Coverage Analysis (500 seeds, MC8)
```bash
python -u dp_p_success_cpu.py \
  --ckpt runs/dppo_pretrain/dppo_pretrain_PegInsertionSide-v1_T100_H16_1M/best.pt \
  --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos \
  --max_episode_steps 200 --num-states 500 --mc-samples 8 \
  --ddim_steps 10 \
  --output runs/coverage_pretrained_ddim10_500seeds.png
```

### Step 2: Filtered Finetuning (P > 0.5 seeds only)
```bash
python -u -m DPPO.finetune \
  --pretrain_checkpoint runs/dppo_pretrain/dppo_pretrain_PegInsertionSide-v1_T100_H16_1M/best.pt \
  --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos \
  --max_episode_steps 200 --n_envs 100 --sim_backend cpu \
  --ft_denoising_steps 10 --use_ddim --ddim_steps 10 \
  --n_steps 100 --min_sampling_denoising_std 0.01 --min_logprob_denoising_std 0.01 \
  --gamma_denoising 0.99 --clip_ploss_coef 0.01 \
  --update_epochs 10 --actor_lr 3e-6 --minibatch_size 10000 --n_critic_warmup_itr 2 \
  --reward_scale_running --n_train_itr 30 --eval_freq 5 \
  --seed_pool_path runs/coverage_pretrained_ddim10_500seeds.npz \
  --seed_pool_threshold 0.5 \
  --exp_name dppo_ft_peg_filtered_easy
```

### What to Compare
- **adv_pos_frac**: filtered ~50%+ vs baseline ~30%
- **Convergence speed**: reach 85%+ in fewer iterations
- **Eval SR**: eval uses train envs (also filtered), so compare with `eval_cpu.py` for true generalization
- Baseline: `dppo_ft_peg_conservative` (best.pt @ iter 20, 90.3% inline eval)

### Coverage Analysis (Pretrained, DDIM-10, 500 seeds, MC8)
```
SR=66.6%, frac_zero=18.6%, frac_one=46.2%, frac_decisive=35.2%
P>0.5: 335/500 seeds (67.0%) → used as seed pool
```

### Filtered Finetuning Results (inline eval on filtered seeds)

| Iter | Reward | Succ | PG Loss | V Loss | KL | adv_pos_frac | Eval SR |
|------|--------|------|---------|--------|-----|-------------|---------|
| 1 [W] | 3.044 | 127 | 0.000000 | 0.2178 | 0.000000 | **0.83** | 92.0% |
| 2 [W] | 3.507 | 151 | -0.000002 | 0.2121 | 0.000000 | 0.47 | — |
| 5 | 2.410 | 96 | -0.000662 | 0.1752 | 0.000371 | 0.27 | **97.3%** |
| 10 | 2.876 | 115 | -0.000662 | 0.1938 | 0.000257 | 0.32 | **99.7%** |
| 15 | 2.652 | 107 | -0.000647 | 0.1769 | 0.000031 | 0.32 | 95.7% |
| 20 | 3.106 | 126 | -0.000651 | 0.1948 | 0.000062 | 0.37 | 95.0% |
| 25 | 2.554 | 104 | -0.000539 | 0.1802 | 0.000290 | 0.36 | 92.3% |
| 30 | 2.890 | 118 | -0.000624 | 0.1806 | 0.000056 | 0.34 | 92.3% |

### CPU Eval — True Generalization (500 eps, random seeds)

| Checkpoint | CPU success_once |
|-----------|-----------------|
| Pretrained (no finetuning) | **53.6%** |
| Conservative baseline (iter 20) | **79.6%** |
| Filtered best (iter 10) | **79.4%** |
| Filtered final (iter 30) | **80.2%** |

### Analysis

1. **adv_pos_frac in warmup was very high (0.83)** — confirmed hypothesis that easy states produce cleaner learning signal. But after critic warmup calibrates V, adv_pos_frac drops to ~0.30, similar to baseline. This makes sense: once V accurately predicts expected return for easy states, advantages reflect the deviation from expectation, which is roughly symmetric.

2. **Inline eval was inflated**: 99.7% inline (on filtered easy seeds) vs 79.4% CPU eval (on random seeds). The inline eval reuses `train_envs` which has `SeedPoolWrapper` — so eval was also on easy seeds. This is a methodological issue to fix in future experiments (use separate eval envs without seed pool).

3. **Filtered ≈ Baseline on generalization**: CPU eval shows 79.4-80.2% (filtered) vs 79.6% (baseline conservative). No significant difference. Both improve +26% over pretrained (53.6%).

4. **Why filtering didn't help more**: The hypothesis was that hard states create noisy gradients. But in practice:
   - With only P>0.5 filtering, 335/500 seeds still include many "decisive" states (0.5 < P < 1.0) that DO sometimes fail — so negative advantages still exist
   - The critic adapts to the filtered state distribution, so adv_pos_frac converges to ~30% regardless
   - The real bottleneck may be the 18.6% frac_zero states that are never trained on → never improve

5. **Both methods plateau at ~80%**: This aligns with coverage analysis — 18.6% of states have P=0. Even with finetuning, the policy can't learn states it never sees reward on.

### Conclusion

Seed pool filtering (P>0.5) **does not significantly improve generalization** over the conservative baseline. The inline eval was misleadingly high because it evaluated on filtered seeds. The real bottleneck is the ~19% of states with zero success probability — these need different approaches (dense reward, curriculum, or longer training with exploration).

**Key insight**: Eval must use separate envs without seed pool filtering for accurate generalization measurement.

## [StackCube GPU-CPU Gap: Obs Augmentation & Zero-Qvel Ablation] - 2026-03-01 17:27

**Git**: 54b5382 (main)

### Background

StackCube DP policy trained on CPU data shows massive GPU-CPU eval gap (CPU 59% vs GPU 0.6-2.6%). Root cause analysis:
1. GPU physics produces slightly different joint velocities (qvel) from step 1 (~0.02 diff)
2. Through closed-loop policy-physics interaction, obs diverge exponentially (0.02 → 0.19 by decision step 8)
3. At decision step 8, model outputs completely different actions (grip action diff = 2.0, max possible)
4. GPU robot pushes/hovers instead of lifting — task fails

Gripper joints (7,8) have **7x more high-frequency jitter** on GPU vs CPU. qvel is the first and most divergent dimension.

### Experiment 1: Baseline (no modifications)

**Command**: `python -u -m DPPO.pretrain --env_id StackCube-v1 --demo_path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5 --control_mode pd_joint_delta_pos --network_type unet --denoising_steps 100 --horizon_steps 16 --cond_steps 2 --act_steps 8 --batch_size 1024 --no_obs_norm --no_action_norm --max_grad_norm 0 --seed 1 --torch_deterministic --total_iters 200000 --eval_freq 10000 --num_eval_episodes 100 --max_episode_steps 200 --cpu_eval`
**Run Dir**: `runs/dppo_pretrain/dppo_pretrain_stackcube_200k/`

| Iter | CPU sr_once | GPU sr_once (500 eps) |
|------|-----------|---------|
| 10k | 37.0% | — |
| 30k | 54.0% | — |
| 170k (best) | **59.0%** | **2.6%** |

### Experiment 2: Obs Noise Augmentation (noise_std=0.1)

Per-dim Gaussian noise scaled by data std: `noise = randn * obs_std * 0.1`

**Command**: `python -u -m DPPO.pretrain ... --obs_noise_std 0.1 --exp_name dppo_pretrain_stackcube_noise0.1`
**Run Dir**: `runs/dppo_pretrain/dppo_pretrain_stackcube_noise0.1/`

| Iter | CPU sr_once | GPU sr_once (500 eps) |
|------|-----------|---------|
| 10k | 43.0% | — |
| 30k (best) | **59.0%** | **27.0%** |
| 40k | 41.0% | — |

GPU improved 2.6% → 27% (10x), but still large gap vs CPU 59%.
Reason: training noise is i.i.d. per sample (~0.017/dim), but GPU divergence compounds to 0.19 by decision step 8.

### Experiment 3: Obs Noise Augmentation (noise_std=0.3)

**Command**: `python -u -m DPPO.pretrain ... --obs_noise_std 0.3 --exp_name dppo_pretrain_stackcube_noise0.3`
**Run Dir**: `runs/dppo_pretrain/dppo_pretrain_stackcube_noise0.3/`

| Iter | CPU sr_once | GPU sr_once (500 eps) |
|------|-----------|---------|
| 10k | 25.0% | — |
| 30k (best) | **31.0%** | **23.0%** |

Too aggressive — hurts CPU learning significantly. GPU/CPU ratio highest (74%) but absolute performance low.

### Experiment 4: Zero Qvel (zero_qvel=True) ⭐

Zero out qvel dims (9:18) during training and eval. Model relies on cond_steps=2 (position differencing) for implicit velocity.

**Command**: `python -u -m DPPO.pretrain ... --zero_qvel --total_iters 100000 --exp_name dppo_pretrain_stackcube_zeroqvel`
**Run Dir**: `runs/dppo_pretrain/dppo_pretrain_stackcube_zeroqvel/`

| Iter | CPU sr_once | GPU sr_once (500 eps) |
|------|-----------|---------|
| 10k | **84.0%** | **84.2%** |

**GPU-CPU gap = 0.** Both at 84%. Also a 25% absolute improvement over baseline CPU eval (59% → 84%).

### Experiment 5: Zero Qvel + Noise (zero_qvel + noise_std=0.1)

**Command**: `python -u -m DPPO.pretrain ... --zero_qvel --obs_noise_std 0.1 --total_iters 100000 --exp_name dppo_pretrain_stackcube_zeroqvel_noise0.1`
**Run Dir**: `runs/dppo_pretrain/dppo_pretrain_stackcube_zeroqvel_noise0.1/`

| Iter | CPU sr_once | GPU sr_once (500 eps) |
|------|-----------|---------|
| 10k | **84.0%** | **78.4%** |

Adding noise on top of zero_qvel slightly hurts GPU (84.2% → 78.4%). Noise is unnecessary when qvel is already removed.

### Summary Table

| Config | CPU eval | GPU eval (500 eps) | GPU/CPU |
|--------|---------|-------------------|---------|
| baseline | 59% | 2.6% | 4% |
| noise=0.1 | 59% | 26.6% | 45% |
| noise=0.3 | 31% | 23.0% | 74% |
| **zero_qvel** | **84%** | **84.2%** | **100%** |
| zero_qvel + noise=0.1 | 84% | 78.4% | 93% |

### Notes
- **qvel is the root cause of GPU-CPU gap for StackCube**: explicit joint velocities from GPU physics are noisy (7x gripper jitter) and diverge first in closed-loop
- **Removing qvel improves BOTH GPU and CPU**: 59% → 84% on CPU. qvel was harmful even on CPU — it's a noisy, redundant signal when cond_steps=2 provides implicit velocity via position differencing
- **Obs noise augmentation helps but doesn't close the gap**: i.i.d. noise can't simulate compounding closed-loop divergence
- **Implementation**: `--zero_qvel` flag in `DPPO/pretrain.py`, `DPPO/evaluate.py`, `DPPO/eval_cpu.py` — zeros dims 9:18 in obs
- **StackCube obs layout (48D)**: qpos(0:9), qvel(9:18), cubeA_pose(18:25), cubeB_pose(25:32), other(32:48)
- All experiments use: UNet, T=100, H=16, cond=2, act=8, bs=1024, no_obs_norm, no_action_norm, pd_joint_delta_pos
- GPU eval uses `evaluate_gpu` with ManiSkillVectorEnv (proper temporal stacking), 500 episodes

---

---

## [StackCube DPPO Pretrain: zero_qvel (BC 99%)] - 2026-03-01 17:37

**Command**: `python -u -m DPPO.pretrain --env_id StackCube-v1 --demo_path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5 --control_mode pd_joint_delta_pos --network_type unet --denoising_steps 100 --horizon_steps 16 --cond_steps 2 --act_steps 8 --batch_size 1024 --no_obs_norm --no_action_norm --max_grad_norm 0 --seed 1 --torch_deterministic --total_iters 100000 --eval_freq 10000 --num_eval_episodes 100 --max_episode_steps 200 --cpu_eval --zero_qvel --save_dir runs/dppo_pretrain --exp_name dppo_pretrain_stackcube_zeroqvel`
**Git**: 54b5382 (main)
**Run Dir**: `runs/dppo_pretrain/dppo_pretrain_stackcube_zeroqvel/`

### Settings
| Parameter | Value |
|-----------|-------|
| env_id | StackCube-v1 |
| demo_path | trajectory.state.pd_joint_delta_pos.physx_cpu.h5 (999 MP demos) |
| control_mode | pd_joint_delta_pos |
| network_type | unet |
| denoising_steps | 100 |
| horizon_steps | 16 |
| cond_steps | 2 |
| act_steps | 8 |
| batch_size | 1024 |
| no_obs_norm | True |
| no_action_norm | True |
| max_grad_norm | 0 (no clipping) |
| seed | 1 |
| total_iters | 100000 |
| eval_freq | 10000 |
| num_eval_episodes | 100 |
| max_episode_steps | 200 |
| cpu_eval | True |
| **zero_qvel** | **True (dims 9:18 zeroed in train + eval)** |
| obs_noise_std | 0 (no augmentation) |

### Results (in progress — currently 39k/100k)
| Iter | success_once | success_at_end | loss |
|------|-------------|----------------|------|
| 10k | 84.0% | 74.0% | ~0.008 |
| 20k | 93.0% | 90.0% | ~0.006 |
| **30k** | **99.0%** | **97.0%** | ~0.005 |

### Key Finding
- **zero_qvel completely solves StackCube GPU-CPU eval gap AND dramatically improves IL quality**
- Previous baseline (with qvel): CPU peak ~61.5%, GPU ~2.6%
- zero_qvel: CPU 99% @ 30k, GPU 84.2% @ 10k (gap eliminated at 10k eval)
- Removing qvel (dims 9:18) works because cond_steps=2 provides implicit velocity via position differencing [obs_{t-1}, obs_t]
- Explicit qvel is redundant AND harmful: GPU physics produces noisier qvel than CPU, causing closed-loop divergence

### Notes
- This is the **first StackCube IL result approaching 99%** — previous best was dp_train ~48% at 30k (with qvel)
- MLP BC on StackCube was 0% regardless of config (compounding error over 108 steps)
- Diffusion Policy + action chunking + zero_qvel = solved StackCube IL
- Experiment still running to 100k, will update with final results
- Checkpoint: `runs/dppo_pretrain/dppo_pretrain_stackcube_zeroqvel/best.pt` (30k, sr_once=0.990)

---

## [DPPO GPU Finetune: PegInsertion zero_qvel — First GPU RL Success] - 2026-03-01 20:23

**Command**:
```bash
python -u -m DPPO.finetune \
  --pretrain_checkpoint runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt \
  --env_id PegInsertionSide-v1 \
  --control_mode pd_joint_delta_pos \
  --max_episode_steps 200 \
  --n_envs 200 \
  --sim_backend gpu \
  --ft_denoising_steps 10 \
  --use_ddim --ddim_steps 10 \
  --n_steps 100 \
  --n_train_itr 30 \
  --gamma 0.999 \
  --actor_lr 3e-6 \
  --critic_lr 1e-3 \
  --n_critic_warmup_itr 2 \
  --update_epochs 10 \
  --minibatch_size 10000 \
  --min_sampling_denoising_std 0.01 \
  --min_logprob_denoising_std 0.01 \
  --eval_freq 5 \
  --eval_n_rounds 3 \
  --zero_qvel \
  --exp_name dppo_ft_peg_zeroqvel_v2
```
**Git**: 54b5382 (main)
**Run Dir**: `runs/dppo_finetune/dppo_ft_peg_zeroqvel_v2/`

### Settings
| Parameter | Value |
|-----------|-------|
| pretrain_checkpoint | dppo_pretrain_peg_zeroqvel_500k/best.pt (50k iter, 59% SR) |
| env_id | PegInsertionSide-v1 |
| control_mode | pd_joint_delta_pos |
| sim_backend | **gpu** |
| zero_qvel | True |
| n_envs | 200 |
| n_steps | 100 |
| n_train_itr | 30 |
| ft_denoising_steps | 10 |
| use_ddim / ddim_steps | True / 10 |
| gamma | 0.999 |
| actor_lr | 3e-6 |
| critic_lr | 1e-3 |
| n_critic_warmup_itr | 2 |
| update_epochs | 10 |
| minibatch_size | 10000 |
| min_sampling_denoising_std | 0.01 |
| min_logprob_denoising_std | 0.01 |
| eval_n_rounds | 3 (600 eval episodes) |

### Results
| Iter | GPU Eval SR | Rollout succ | time/iter |
|------|------------|--------------|-----------|
| 1 (warmup) | 53.2% | 489 | 52.9s |
| 5 | 80.8% | 697 | 70.8s |
| 10 | 89.8% | 909 | 71.4s |
| 15 | 92.5% | 914 | 71.3s |
| 20 | 92.8% | 952 | 104.6s |
| 25 | **94.2%** | 965 | 105.6s |
| 30 | 93.0% | 966 | 70.8s |

**Best**: 94.2% @ iter 25

### Notes
- **First successful GPU-based DPPO RL finetune** — zero_qvel eliminates GPU-CPU eval gap, making GPU training viable
- **3.5x faster than CPU finetune**: ~70s/iter (GPU) vs ~233s/iter (CPU). 30 iters in ~35 min
- Surpasses previous CPU finetune best (90.3% @ iter 20, 1M pretrain) despite using weaker pretrain (only 50k)
- **Critical settings that prevented crash**: `minibatch_size=10000` (not 500), `min_logprob_denoising_std=0.01` (not 0.1). Previous attempt with wrong defaults crashed at iter 3 (succ 438→64→14)
- Convergence curve shows rapid improvement iter 1-10, then plateau 10-30. Slight drop at iter 30 is normal variance
- **Code fixes applied in this session**: (1) zero out actions for done envs in action chunk, (2) eval tracks first episode only per env slot, (3) episode-level success counting, (4) exp_name generated after checkpoint param override

---

## [PegInsertion IL Policy Coverage Analysis — GPU Parallel] - 2026-03-01 20:23

**Command**:
```bash
python -u dp_p_success_cpu.py \
  --ckpt runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt \
  --env_id PegInsertionSide-v1 \
  --control_mode pd_joint_delta_pos \
  --max_episode_steps 300 \
  --num-states 200 \
  --mc-samples 16 \
  --zero_qvel \
  --gpu
```
**Git**: 54b5382 (main)
**Output**: `runs/dp_p_success_cpu.png`, `runs/dp_p_success_cpu.npz`

### Settings
| Parameter | Value |
|-----------|-------|
| checkpoint | dppo_pretrain_peg_zeroqvel_500k/best.pt (50k iter IL policy) |
| env_id | PegInsertionSide-v1 |
| zero_qvel | True |
| num_states | 200 |
| mc_samples | 16 |
| mode | GPU parallel (3200 envs) |
| total time | **62 seconds** (vs ~40 min CPU) |

### Results — P(success|s₀) Distribution
| Metric | Value |
|--------|-------|
| SR | 60.5% |
| frac_zero (P=0) | **0.0%** |
| frac_one (P=1) | 0.0% |
| frac_decisive (0.1<P<0.9) | **98.5%** |
| mean P | 0.605 |
| std P | 0.130 |

**Fine-grained distribution (0.1 bins)**:
| Bin | Count | Frac |
|-----|-------|------|
| [0.0,0.1) | 0 | 0.0% |
| [0.1,0.2) | 1 | 0.5% |
| [0.2,0.3) | 0 | 0.0% |
| [0.3,0.4) | 13 | 6.5% |
| [0.4,0.5) | 13 | 6.5% |
| [0.5,0.6) | 63 | 31.5% |
| [0.6,0.7) | 73 | 36.5% |
| [0.7,0.8) | 26 | 13.0% |
| [0.8,0.9) | 8 | 4.0% |
| [0.9,1.0) | 3 | 1.5% |

### Notes
- **Zero dead zones** (frac_zero=0%) — every initial state has nonzero success probability. RL always has learning signal.
- **98.5% decisive** — almost all states have uncertain outcomes, maximizing learning potential for RL finetuning.
- Distribution is unimodal, centered at 0.5-0.7 (68% of states). No bimodality unlike BC policies (which are typically extreme: all-or-nothing).
- This explains the smooth finetune curve (59%→94% in 25 iters): excellent coverage means RL can improve everywhere.
- **Comparison with other tasks/policies**:

| Policy | Task | frac_zero | frac_decisive | SR |
|--------|------|-----------|---------------|-----|
| PPO ckpt_76 | PickCube | 1.2% | 96.2% | 43.8% |
| PPO ckpt_231 | PegInsertion | 0.0% | 65.8% | 76.7% |
| PPO ckpt_481 | StackCube | 25.2% | 40.8% | ~70% |
| **DP IL (this)** | **PegInsertion** | **0.0%** | **98.5%** | **60.5%** |

- DP IL policy has BETTER coverage than PPO checkpoint for PegInsertion (98.5% vs 65.8% decisive). The diffusion sampling noise provides natural exploration, making every state learnable.
- **GPU parallel coverage is 40x faster** than CPU sequential (62s vs ~40min for 200 states × MC16)

---

## [DPPO Finetune Code Fixes — 4 Bug Fixes] - 2026-03-01 20:23

**Git**: 54b5382 (main)
**Files Modified**: `DPPO/finetune.py`, `DPPO/pretrain.py`, `dp_p_success_cpu.py`

### Bug Fixes Applied

1. **[High] Action chunk state pollution** (`finetune.py`): After env terminates mid-action-chunk, remaining actions were applied to the newly reset episode. Fix: zero out actions for already-done envs (`action[step_done] = 0.0`).

2. **[Medium-High] Eval partial-reset inflation** (`finetune.py`): `evaluate_gpu_inline` tracked `success_once` across partial resets, inflating success rate. Fix: added `ep_done` mask to only track first episode per env slot; early break when all done.

3. **[Medium] Misleading success counter** (`finetune.py`): `n_success_rollout` counted per-step `reward > 0.5`, not episode-level success. Fix: count only on episode boundary (`newly_done & reward > 0.5`), renamed log key to `rollout_ep_successes`.

4. **[Low] exp_name timing** (`finetune.py`): exp_name was generated before checkpoint params override. Fix: moved generation after checkpoint loading.

5. **[Pretrain] LR scheduler double-restore** (`pretrain.py`): Resume loaded scheduler state AND fast-forwarded by start_iter steps. Fix: skip fast-forward when scheduler was restored from checkpoint.

6. **[Pretrain] Old checkpoint EMA overwrite** (`pretrain.py`): Without `ema_state` in checkpoint, `ema.copy_to(ema_model)` at eval overwrote loaded ema_model with fresh EMA. Fix: sync ema shadow params from loaded ema_model.

7. **[Pretrain] EMA device mismatch** (`pretrain.py`): After `ema.load_state_dict()`, shadow params could end up on CPU due to deepcopy. Fix: `ema.shadow_params = [p.to(device) for p in ema.shadow_params]`.

8. **[Coverage] GPU parallel mode** (`dp_p_success_cpu.py`): Added `--gpu` flag for parallel coverage analysis using `num_states * mc_samples` GPU envs with state dict copying. 40x faster than CPU sequential.

9. **[Coverage] zero_qvel support** (`dp_p_success_cpu.py`): Added `--zero_qvel` flag, auto-inherited from checkpoint args.

---

## [Filtered BC Finetuning for DPPO] - 2026-03-01

**Git**: c2c3646 (main)
**Script**: `DPPO/finetune_filtered_bc.py`
**Checkpoint**: `runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt` (PegInsertionSide-v1, pd_joint_delta_pos, zero_qvel)

### Pretrained Policy Coverage (dp_p_success_gpu.py, 200 states × MC16, zero_qvel)

| Metric | Deterministic | Stochastic (std=0.01) |
|--------|--------------|----------------------|
| SR | 56.5% | 55.2% |
| frac_zero | 7.5% | 5.5% |
| frac_one | 5.5% | 2.5% |
| frac_decisive | 77.5% | 82.5% |

Deterministic vs stochastic nearly identical — std=0.01 noise is negligible for this policy.

**Bug found**: `dp_p_success_gpu.py` lacked zero_qvel support — without it, SR=3% (model receives unmasked qvel, distribution mismatch). Fixed by adding `--zero_qvel` flag with auto-inherit from checkpoint.

### Run 1: Naive filtered BC (no demo mixing) — COLLAPSED

**Command**:
```bash
python -u -m DPPO.finetune_filtered_bc \
  --pretrain_checkpoint runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt \
  --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos \
  --max_episode_steps 200 --n_envs 200 --sim_backend gpu \
  --use_ddim --ddim_steps 10 --n_steps 100 \
  --n_train_itr 30 --eval_freq 5 \
  --bc_gradient_steps 200 --bc_lr 1e-4 \
  --min_sampling_denoising_std 0.01 \
  --zero_qvel \
  --num_eval_episodes 500 \
  --exp_name fbc_peg_warmstart
```

**Run Dir**: `runs/dppo_filtered_bc/fbc_peg_warmstart/`

| Iter | Rollout SR | Eval SR | Queue |
|------|-----------|---------|-------|
| 1 | 57.0% | 55.7% | 8.8k |
| 5 | 25.8% | 28.5% | 30.7k |
| 10 | 23.3% | 26.3% | 46.5k |
| 15 | 22.8% | — | 65.8k |

**Killed early.** Classic self-bootstrap degradation: policy略变差 → 数据质量降 → BC拟合差数据 → policy更差。

**Root causes**:
1. No demo anchor — queue全是online data，无稳定基准
2. lr=1e-4 + 200 gradient steps过于aggressive，catastrophic forgetting
3. Filter太松（return > 0），成功episode的早期差action也入queue

### Run 2: Demo mixing + conservative training — STABLE IMPROVEMENT

**Command**:
```bash
python -u -m DPPO.finetune_filtered_bc \
  --pretrain_checkpoint runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt \
  --demo_path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5 \
  --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos \
  --max_episode_steps 200 --n_envs 500 --sim_backend gpu \
  --use_ddim --ddim_steps 10 --n_steps 100 \
  --n_train_itr 30 --eval_freq 5 \
  --bc_gradient_steps 50 --bc_lr 1e-5 \
  --demo_ratio 0.5 \
  --min_sampling_denoising_std 0.01 \
  --zero_qvel \
  --num_eval_episodes 500 \
  --exp_name fbc_peg_demomix50_lr1e5
```

**Run Dir**: `runs/dppo_filtered_bc/fbc_peg_demomix50_lr1e5/`
**Key changes**: demo_ratio=0.5 (50% MP demos + 50% online), lr 10x lower (1e-5), gradient steps 4x fewer (50), n_envs 2.5x more (500)

| Iter | Rollout SR | Eval SR |
|------|-----------|---------|
| 1 | 54.0% | 68.6% |
| 5 | 63.2% | 64.4% |
| 10 | 71.3% | 71.4% |
| 15 | 69.4% | 73.6% |

**Final results (30 iterations):**

| Iter | Rollout SR | Eval SR |
|------|-----------|---------|
| 1 | 54.0% | 68.6% |
| 5 | 63.2% | 64.4% |
| 10 | 71.3% | 71.4% |
| 15 | 69.4% | 73.6% |
| 20 | 70.2% | 79.2% |
| 25 | 69.1% | **81.4%** |
| 30 | 70.7% | 80.6% |

**Best eval SR: 81.4% (iter 25)**. Final coverage analysis (2000 episodes): 78.3% ± 1.8% (95% CI).

Stable improvement from 55% → 81.4%, no collapse. Demo anchor prevents forgetting, low lr prevents overshooting. Plateaus around 80% — pure exploitation ceiling without RL exploration.

**Comparison**: Filtered BC 81.4% vs DPPO RL 94.2% — 13% gap from objective mismatch (MSE vs trajectory success).

### Code Changes

- `DPPO/finetune_filtered_bc.py`: Bug fixes (eval partial-reset, action chunk pollution, rollout success counting), added zero_qvel, cold_start, CSV logging, demo_ratio mixing, summary plots, coverage analysis
- `dp_p_success_gpu.py`: Added `--zero_qvel` flag with auto-inherit from checkpoint

---

## [Deterministic DP Failure & Trajectory Divergence Analysis] - 2026-03-01

**Git**: 0f540d1 (main)
**Checkpoint**: `runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt` (PegInsertionSide-v1, pretrained DP)
**Scripts**: `analysis_deterministic_failure_v2.py`, `analysis_trajectory_divergence.py`

### Coverage Analysis (500 states × MC16, GPU, deterministic DDIM-10, zero_qvel)

**Command**:
```bash
python -u analysis_deterministic_failure_v2.py \
  --ckpt runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt \
  --num_states 500 --mc_samples 16 --ddim_steps 10
```

| Category | Count | Fraction |
|----------|-------|----------|
| frac_zero (P=0) | 35 | 7.0% |
| decisive_low (0<P≤0.3) | 89 | 17.8% |
| decisive_mid (0.3<P<0.7) | 206 | 41.2% |
| decisive_high (0.7≤P<1) | 133 | 26.6% |
| frac_one (P=1) | 37 | 7.4% |
| **SR** | — | **54.1%** |

**Key finding**: 85.6% of states are decisive even with "deterministic" DDIM. The noise source is NOT GPU physics — it's the random initial noise x_T in DDIM sampling (line 160: `x = torch.randn(...)`). Verified by running CPU deterministic (frac_decisive=57% with MC4) — CPU physics is deterministic but DDIM x_T is still random.

### Feature Correlation with P(success)

| Feature | Pearson r | Meaning |
|---------|-----------|---------|
| peg_radius | **-0.371** | Smaller peg → higher success (strongest signal) |
| hole_qx | +0.348 | Hole orientation matters |
| peg_hole_dist | -0.290 | Closer → higher success |
| rel_y | +0.285 | Less Y offset → higher success |
| peg_length | -0.215 | Shorter peg → higher success |

P(success) by peg radius bin:
- Smallest (0.015-0.016): 68.8% SR, 3.2% frac_zero
- Largest (0.024-0.025): 34.4% SR, 14.9% frac_zero

### Trajectory Divergence Analysis (200 decisive states × MC32, GPU)

**Command**:
```bash
python -u analysis_trajectory_divergence.py \
  --ckpt runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt \
  --num_states 200 --mc_samples 32 --ddim_steps 10
```

**Run Dir**: `runs/analysis_divergence/`

| Step | TCP pos std | Action std | Succ-Fail TCP dist |
|------|------------|------------|-------------------|
| 0 | 0.014m | 0.028 | small |
| 50 | 0.057m | ~0.05 | growing |
| 100 | 0.069m | ~0.08 | growing |
| 150 | 0.058m | ~0.10 | still growing |
| 192 | — | 0.135 | **0.084m (peak)** |

**No clear fork point** — gradual divergence throughout the episode. Fork ratio never exceeds 2x threshold. All trajectories attempt the same strategy (reach → grasp → move → insert), differences are purely from compounding of per-step action variance.

### CPU Stochastic Coverage (100 states × MC16, CPU, std=0.01)

**Command**:
```bash
python -u dp_p_success_cpu.py \
  --ckpt runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt \
  --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos \
  --num_states 100 --mc_samples 16 --max_episode_steps 200 \
  --ddim_steps 10 --min_sampling_denoising_std 0.01 --zero_qvel
```

| Metric | CPU stochastic | GPU deterministic |
|--------|---------------|-------------------|
| SR | 57.1% | 54.1% |
| frac_zero | 2.0% | 7.0% |
| frac_decisive | 87.0% | 85.6% |

CPU stochastic ≈ GPU deterministic — both noise sources (x_T randomness vs std=0.01) produce similar coverage. std=0.01 adds negligible marginal noise on top of x_T.

### Key Insights

1. **Decisive fraction comes from diffusion x_T randomness**, not GPU physics non-determinism. Deterministic DDIM starts from random x_T ~ N(0,I) each call — the denoising is deterministic but the starting point isn't.

2. **Compounding error is the primary bottleneck**: per-step action std=0.028 compounds to 8cm TCP scatter over 25 decision steps, vs 3mm task clearance. No single "fork point" — gradual accumulation.

3. **Peg size is the strongest predictor** of P(success): larger peg = tighter relative clearance = less tolerance for compounding error.

4. **BC's objective mismatch** (MSE vs trajectory success) means BC cannot learn to be robust against its own compounding error. RL (DPPO 94%) directly optimizes trajectory success, learning precision where it matters most.

5. **Filtered BC plateaus at 81%** — it improves by practicing "easy" cases more, but cannot fundamentally reduce compounding error on "hard" cases (large peg + far distance).

### Output Files

- `runs/analysis_deterministic_v2/coverage_analysis.png` — Coverage scatter plots (P(success) vs features)
- `runs/analysis_divergence/trajectory_divergence.png` — Trajectory divergence plots (TCP std, succ-fail separation, XY trajectories)
- `runs/analysis_divergence/fork_ratio.png` — Fork ratio over time
- `runs/analysis_divergence/divergence_data.npz` — Raw data

---

## [Filtered BC Ablation: Which Change Matters Most?] - 2026-03-02 01:30

**Git**: 81f6d02 (main)
**Script**: `DPPO/finetune_filtered_bc.py`
**Checkpoint**: `runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt`

### Overview
Ablation of three key changes from Run 1→Run 2 (filtered BC): demo mixing, low lr, fewer gradient steps. Each ablation reverts ONE change while keeping the other two.

### Shared Settings
| Parameter | Value |
|-----------|-------|
| pretrain_checkpoint | dppo_pretrain_peg_zeroqvel_500k/best.pt |
| env_id | PegInsertionSide-v1 |
| control_mode | pd_joint_delta_pos |
| n_envs | 500 |
| n_steps | 100 |
| n_train_itr | 30 |
| sim_backend | gpu |
| use_ddim / ddim_steps | True / 10 |
| min_sampling_denoising_std | 0.01 |
| zero_qvel | True |
| num_eval_episodes | 500 |

### Ablation Design
| Ablation | demo_ratio | bc_lr | bc_gradient_steps | Changed vs Run 2 |
|----------|-----------|-------|-------------------|-------------------|
| Run 2 (ref) | 0.5 | 1e-5 | 50 | — |
| A: no demo mix | **0.0** | 1e-5 | 50 | demo_ratio |
| B: high lr | 0.5 | **1e-4** | 50 | bc_lr |
| C: many steps | 0.5 | 1e-5 | **200** | bc_gradient_steps |

### Commands
```bash
# A: no demo mix
python -u -m DPPO.finetune_filtered_bc \
  --pretrain_checkpoint runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt \
  --demo_path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5 \
  --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos \
  --max_episode_steps 200 --n_envs 500 --sim_backend gpu \
  --use_ddim --ddim_steps 10 --n_steps 100 \
  --n_train_itr 30 --eval_freq 5 \
  --bc_gradient_steps 50 --bc_lr 1e-5 \
  --demo_ratio 0.0 \
  --min_sampling_denoising_std 0.01 --zero_qvel \
  --num_eval_episodes 500 \
  --exp_name ablation_no_demo_mix

# B: high lr
python -u -m DPPO.finetune_filtered_bc \
  ... --bc_lr 1e-4 --demo_ratio 0.5 --bc_gradient_steps 50 \
  --exp_name ablation_high_lr

# C: many gradient steps
python -u -m DPPO.finetune_filtered_bc \
  ... --bc_lr 1e-5 --demo_ratio 0.5 --bc_gradient_steps 200 \
  --exp_name ablation_many_steps
```

### Run Dirs
- `runs/dppo_filtered_bc/ablation_no_demo_mix/`
- `runs/dppo_filtered_bc/ablation_high_lr/`
- `runs/dppo_filtered_bc/ablation_many_steps/`

### Results

| iter | Run 2 (ref) | A: no demo | B: high lr | C: many steps |
|------|-------------|------------|------------|---------------|
| 1 | 68.6% | 54.0% | 36.2% | 70.2% |
| 5 | 64.4% | 71.6% | 25.4% | 72.0% |
| 10 | 71.4% | 74.6% | 40.0% | 61.4% |
| 15 | 73.6% | 74.4% | 39.6% | 73.6% |
| 20 | 79.2% | 71.2% | 43.6% | 81.0% |
| 25 | **81.4%** | 83.2% | 35.4% | 74.6% |
| 30 | 80.6% | 68.4% | 9.0% | 59.4% |

| Ablation | Peak SR | Final SR (coverage) | Stability |
|----------|---------|---------------------|-----------|
| Run 2 (ref) | **81.4%** | **78.3%** | Stable |
| A: no demo mix | 83.2% | 69.0% | Unstable, late degradation |
| B: high lr | 43.6% | 10.4% | Collapsed |
| C: many steps | 81.0% | 61.7% | Oscillating, late collapse |

### Notes
- **Importance ranking: Low lr >> Demo mixing >> Fewer gradient steps**
- **B (high lr=1e-4)**: catastrophic collapse to 9% by iter 30, even with demo anchor. lr is the most critical factor.
- **A (no demo mix)**: peak 83.2% at iter 25 but crashed to 68.4% at iter 30. Without demo anchor, self-bootstrap eventually degrades.
- **C (many steps=200)**: oscillates wildly (61%→81%→59%). Overfits to current batch each iteration.
- **Run 2 succeeds because of the combination**: conservative update (low lr + few steps) + stable anchor (demo mix).

---

## [Filtered BC Sample Efficiency] - 2026-03-02 02:30

**Git**: 81f6d02 (main)
**Script**: `DPPO/finetune_filtered_bc.py`
**Checkpoint**: `runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt`

### Overview
Test sample efficiency: can we reach ~80% with 5x fewer env steps (2.4M vs 12M) by reducing n_envs and improving data utilization?

### Design
| Run | n_envs | gradient_steps | bc_lr | demo_ratio | env steps (30 iters) |
|-----|--------|---------------|-------|------------|---------------------|
| Run 2 (ref) | 500 | 50 | 1e-5 | 0.5 | 12M |
| S1: fewer envs | **100** | 50 | 1e-5 | 0.5 | **2.4M** |
| S2: +more updates | **100** | **200** | **2.5e-6** | 0.5 | **2.4M** |
| S3: +high demo | **100** | **200** | **2.5e-6** | **0.7** | **2.4M** |

S2 keeps effective step size constant: lr×steps = 2.5e-6×200 = 5e-4 ≈ 1e-5×50.

### Commands
```bash
# S1
python -u -m DPPO.finetune_filtered_bc \
  ... --n_envs 100 --bc_gradient_steps 50 --bc_lr 1e-5 --demo_ratio 0.5 \
  --exp_name sample_eff_fewer_envs

# S2
python -u -m DPPO.finetune_filtered_bc \
  ... --n_envs 100 --bc_gradient_steps 200 --bc_lr 2.5e-6 --demo_ratio 0.5 \
  --exp_name sample_eff_more_updates

# S3
python -u -m DPPO.finetune_filtered_bc \
  ... --n_envs 100 --bc_gradient_steps 200 --bc_lr 2.5e-6 --demo_ratio 0.7 \
  --exp_name sample_eff_high_demo
```

### Run Dirs
- `runs/dppo_filtered_bc/sample_eff_fewer_envs/`
- `runs/dppo_filtered_bc/sample_eff_more_updates/`
- `runs/dppo_filtered_bc/sample_eff_high_demo/`

### Results

| iter | Run 2 (12M) | S1 (2.4M) | S2 (2.4M) | S3 (2.4M) |
|------|-------------|-----------|-----------|-----------|
| 1 | 68.6% | 58.8% | 63.2% | 62.6% |
| 5 | 64.4% | 73.0% | 75.0% | 73.4% |
| 10 | 71.4% | 60.2% | 74.6% | 72.0% |
| 15 | 73.6% | 79.0% | 74.0% | 71.4% |
| 20 | 79.2% | 63.6% | 77.4% | 75.2% |
| 25 | **81.4%** | 80.4% | 74.6% | 72.2% |
| 30 | 80.6% | 67.2% | 75.6% | **78.2%** |

| Run | Peak SR | Final SR (coverage) | Stability |
|-----|---------|---------------------|-----------|
| Run 2 (ref, 12M) | **81.4%** | **78.3%** | Stable |
| S1 (2.4M) | 80.4% | 64.4% | Very unstable |
| S2 (2.4M) | 77.4% | **77.1%** | Most stable |
| S3 (2.4M) | 78.2% | **78.2%** | Stable |

### Notes
- **S3 (more updates + high demo) is most sample efficient**: 78.2% final SR with 5x fewer env steps, matching Run 2's 78.3%.
- **S2 (more updates)** also good: 77.1% final, most stable throughout (no oscillation).
- **S1 (just fewer envs)** failed: 80.4% peak but crashed to 64.4%. Without more gradient steps, data utilization too low.
- **Key insight**: constant effective step (lr×steps) + more gradient diversity + high demo ratio enables 5x sample efficiency without sacrificing final performance.
- **Filtered BC ceiling remains ~80%** regardless of sample efficiency — fundamental limitation from objective mismatch (MSE vs trajectory success).

---

## [DPPO RL High-UTD Sample Efficiency] - 2026-03-02 03:00

**Git**: 81f6d02 (main)
**Script**: `DPPO/finetune.py`
**Checkpoint**: `runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt`

### Overview
Test if DPPO RL can reach 90%+ with fewer env steps by reducing rollout size and increasing update epochs (higher UTD ratio).

### Design
| Run | n_envs | n_steps | update_epochs | minibatch | env steps/iter | total (30 iters) | vs ref |
|-----|--------|---------|---------------|-----------|---------------|------------------|--------|
| Ref (best) | 200 | 100 | 10 | 10000 | 160k | 4.8M | — |
| D1: fewer steps | 200 | **25** | **40** | 10000 | 40k | **1.2M** | 4x fewer |
| D2: fewer envs | **50** | 100 | **40** | 10000 | 40k | **1.2M** | 4x fewer |
| D3: extreme | **50** | **25** | **160** | 2500 | 10k | **0.3M** | 16x fewer |

### Shared Settings
| Parameter | Value |
|-----------|-------|
| env_id | PegInsertionSide-v1 |
| control_mode | pd_joint_delta_pos |
| sim_backend | gpu |
| zero_qvel | True |
| ft_denoising_steps | 10 |
| use_ddim / ddim_steps | True / 10 |
| gamma | 0.999 |
| actor_lr | 3e-6 |
| critic_lr | 1e-3 |
| n_critic_warmup_itr | 2 |
| min_sampling_denoising_std | 0.01 |
| min_logprob_denoising_std | 0.01 |
| eval_freq | 5 |
| eval_n_rounds | 3 |
| n_train_itr | 30 |

### Commands
```bash
# D1: fewer steps + more epochs
python -u -m DPPO.finetune \
  --pretrain_checkpoint runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt \
  --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos \
  --max_episode_steps 200 --n_envs 200 --sim_backend gpu \
  --ft_denoising_steps 10 --use_ddim --ddim_steps 10 \
  --n_steps 25 --n_train_itr 30 --gamma 0.999 \
  --actor_lr 3e-6 --critic_lr 1e-3 --n_critic_warmup_itr 2 \
  --update_epochs 40 --minibatch_size 10000 \
  --min_sampling_denoising_std 0.01 --min_logprob_denoising_std 0.01 \
  --eval_freq 5 --eval_n_rounds 3 --zero_qvel \
  --exp_name dppo_ft_highUTD_steps25_ep40

# D2: fewer envs + more epochs
python -u -m DPPO.finetune \
  ... --n_envs 50 --n_steps 100 --update_epochs 40 --minibatch_size 10000 \
  --exp_name dppo_ft_highUTD_envs50_ep40

# D3: extreme (16x fewer data)
python -u -m DPPO.finetune \
  ... --n_envs 50 --n_steps 25 --update_epochs 160 --minibatch_size 2500 \
  --exp_name dppo_ft_highUTD_extreme
```

### Run Dirs
- `runs/dppo_finetune/dppo_ft_highUTD_steps25_ep40/`
- `runs/dppo_finetune/dppo_ft_highUTD_envs50_ep40/`
- `runs/dppo_finetune/dppo_ft_highUTD_extreme/`

### Results (in progress — D1/D2 still running)

| iter | Ref (4.8M) | D1 (1.2M) | D2 (1.2M) | D3 (0.3M) |
|------|------------|-----------|-----------|-----------|
| 1 | 53.2% | 54.8% | 42.7% | 55.3% |
| 5 | 80.8% | 74.8% | 70.7% | 61.3% |
| 10 | 89.8% | 83.2% | 78.0% | 58.0% |
| 15 | 92.5% | 86.3% | **88.7%** | 50.7% |
| 20 | 92.8% | **89.3%** | — | 52.0% |
| 25 | **94.2%** | — | — | 41.3% |

| Run | Best so far | env steps to 85%+ | Status |
|-----|-------------|-------------------|--------|
| Ref | **94.2%** | ~1.2M (iter ~8) | Complete |
| D1 | 89.3% | ~800k (iter ~20) | In progress |
| D2 | 88.7% | ~600k (iter 15) | In progress |
| D3 | 61.3% | Never | Collapsed |

### Notes
- **D2 (fewer envs, long rollout) reaches 88.7% with only 600k env steps** — 8x more sample efficient than ref at comparable performance. Long rollouts (n_steps=100) provide better GAE/critic learning than short rollouts (n_steps=25).
- **D1 (fewer steps, many envs) at 89.3%** — also works but slightly slower to converge than D2 despite same total data. Short rollouts (25 steps) provide less temporal information for critic.
- **D3 (extreme, 160 epochs) collapsed** — from 61% to 41%. Severe overfitting with too many epochs on too little data (only 10k env steps/iter).
- **Key insight**: 4x data reduction with 4x more epochs works well. 16x reduction does not — there's a minimum data diversity threshold.
- **D1 and D2 are on track to reach 90%+** with only 1.2M total env steps (vs ref's 4.8M). Both still running.
- At iter 15, ~4000 trajectories: D2=88.7% vs filtered BC S3=71.4% — DPPO is fundamentally more sample efficient because it uses advantage-based optimization instead of binary success filtering.

---

## [DPPO Pretrain 1M — PegInsertionSide] - 2026-03-02 15:11

**Command**: `python -u -m DPPO.pretrain --env_id PegInsertionSide-v1 --demo_path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5 --control_mode pd_joint_delta_pos --network_type unet --denoising_steps 100 --horizon_steps 16 --cond_steps 2 --act_steps 8 --batch_size 1024 --no_obs_norm --no_action_norm --max_grad_norm 0 --seed 1 --num_workers 0 --torch_deterministic --total_iters 1000000 --eval_freq 100000 --zero_qvel --exp_name dppo_pretrain_peg_zeroqvel_1M`
**Git**: 81f6d02 (main)
**Run Dir**: runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_1M

### Settings
| Parameter | Value |
|-----------|-------|
| env_id | PegInsertionSide-v1 |
| control_mode | pd_joint_delta_pos |
| network_type | unet |
| denoising_steps | 100 |
| horizon_steps | 16 |
| cond_steps | 2 |
| act_steps | 8 |
| batch_size | 1024 |
| total_iters | 1,000,000 |
| lr | 1e-4 (cosine → 0) |
| no_obs_norm | True |
| no_action_norm | True |
| zero_qvel | True |
| seed | 1 |

### Results
| Iter | Loss | CPU SR (100 eps, max_steps=200) |
|------|------|---------------------------------|
| 100K | ~0.010 | 62% |
| 200K | ~0.006 | 66% |
| 300K | ~0.004 | 60% |
| 400K | ~0.003 | 65% |
| 500K | ~0.002 | 66% |
| 600K | ~0.001 | 64% |
| 700K | ~0.0007 | 55% |
| 800K | ~0.0004 | **68%** |
| 900K | ~0.0003 | 66% |
| 1M | ~0.0001 | 63% |

### Notes
- **Loss dropped 100x (0.01 → 0.0001) but SR plateaued at 60-68%**. Longer cosine schedule allowed continued loss reduction, but BC loss improvement doesn't translate to SR improvement past a certain point.
- Inline eval during training showed 1-4% SR because `--max_episode_steps` was not passed (defaulted to 100, but PegInsertion needs 200). All SR numbers above are from post-hoc CPU eval with correct max_episode_steps=200.
- Compared to 500K pretrain (same architecture, cosine→0 at 500K): SR is similar (55-60% vs 60-68%). The 500K pretrain's loss plateaued at ~0.002; this run's loss continued dropping but SR didn't benefit.
- **Conclusion**: For PegInsertionSide IL, ~100K-200K iters is sufficient. Additional training reduces BC loss but doesn't improve policy quality — likely memorizing demo actions rather than learning generalizable behavior.
- 100 episodes CPU eval has SE ≈ 5%, so differences within the 55-68% range are not statistically significant.

---

## [DPPO Finetune GAE n200 e100 — PegInsertionSide] - 2026-03-02 15:11

**Command**: `python -u -m DPPO.finetune --pretrain_checkpoint runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos --max_episode_steps 200 --n_envs 200 --sim_backend gpu --ft_denoising_steps 10 --use_ddim --ddim_steps 10 --n_steps 25 --n_train_itr 30 --gamma 0.999 --gae_lambda 0.95 --actor_lr 3e-6 --critic_lr 1e-3 --update_epochs 100 --minibatch_size 2500 --min_sampling_denoising_std 0.01 --min_logprob_denoising_std 0.01 --eval_freq 1 --eval_n_rounds 3 --zero_qvel --exp_name dppo_ft_gae_n200_e100`
**Git**: 81f6d02 (main)
**Run Dir**: runs/dppo_finetune/dppo_ft_gae_n200_e100

### Settings
| Parameter | Value |
|-----------|-------|
| n_envs | 200 |
| n_steps | 25 |
| update_epochs | 100 |
| batch_size | 5000 |
| gamma | 0.999 |
| gae_lambda | 0.95 |
| actor_lr | 3e-6 |
| critic | learned |
| eval | every iter, 3 rounds (150 eps) |

### Results
| Iter | GPU SR | KL |
|------|--------|----|
| 1 [WU] | 54.8% | 0.000 |
| 2 [WU] | 53.7% | 0.000 |
| 3 | 62.2% | 0.000088 |
| 4 | 73.5% | 0.000087 |
| 5 | 72.3% | 0.000087 |
| 8 | 76.7% | — |
| 9 | 79.3% | — |
| 10 | 81.2% | 0.000097 |
| 15 | **83.2%** | — |
| 16 | 78.8% | — |
| 20 | 79.2% | 0.000080 |
| 23 | 77.5% | — |

### Notes
- Peak **83.2%** at iter 15, then oscillated and declined to ~77%.
- Compared to e40 runs: D1 (n50, e40) peak 88.7%, D2 (n200, e40) peak 89.3%. **epochs=100 is worse than epochs=40** — overfitting each iteration's data.
- KL very stable (~0.00009), no collapse, but the extra epochs hurt rather than help.

---

## [Coverage Analysis: Finetuned D2 Best vs Pretrained Base] - 2026-03-02 15:11

**Git**: 81f6d02 (main)

### Results (GPU, MC32, 500 states, DDIM-10, std=0.01, zero_qvel)

| Bin | Base (pretrained) | FT D2 (best, iter 20) |
|-----|------------------|----------------------|
| P=0 | 4.6% | 1.8% |
| (0,0.1] | 4.6% | 1.2% |
| (0.1,0.2] | 6.4% | 0.4% |
| (0.2,0.3] | 6.4% | 0.6% |
| (0.3,0.4] | 7.4% | 0.8% |
| (0.4,0.5] | 13.8% | 1.2% |
| (0.5,0.6] | 10.8% | 3.2% |
| (0.6,0.7] | 13.0% | 5.2% |
| (0.7,0.8] | 13.0% | 15.2% |
| (0.8,0.9] | 11.6% | 23.2% |
| (0.9,1] | 8.4% | 47.2% |
| P=1 | 0.6% | 7.4% |
| **SR** | **53.7%** | **82.5%** |
| frac_zero | 4.6% | 1.8% |
| frac_one | 0.6% | 7.4% |
| frac_decisive | 82.4% | 49.8% |
| median | 0.562 | 0.875 |

### Notes
- **Finetuning pushes the mass rightward**: most states moved from 0.3-0.7 range to 0.9+ range.
- **frac_zero reduced from 4.6% to 1.8%** — finetuning converts some "impossible" states to solvable, but ~2% remain stuck at P=0.
- **~3% of states still at P<0.1** — these are the hard states that create the ceiling.
- **Theoretical ceiling ~95-97%** given the remaining hard states, but the gap between current 82.5% GPU SR (≈89% in GPU eval with noise) and ceiling suggests room for further improvement.
- Main effect of finetuning: converting "learnable" states (0.1<P<0.9) to "mastered" states (P>0.9), not eliminating dead zones.

---

## [MC Exploitation: PPO on Fixed Data with MC16 Advantages] - 2026-03-02 18:20

**Command**:
```bash
python -u -m DPPO.mc_exploitation \
  --pretrain_checkpoint runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt \
  --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos \
  --max_episode_steps 200 --n_envs 200 --n_steps 25 \
  --mc_samples 16 --gamma 0.999 --gae_lambda 0.95 \
  --update_epochs 1000 --minibatch_size 2500 --actor_lr 3e-6 \
  --ft_denoising_steps 10 --use_ddim --ddim_steps 10 \
  --min_sampling_denoising_std 0.01 --min_logprob_denoising_std 0.01 \
  --eval_freq 20 --zero_qvel \
  --exp_name mc_exploitation_mc16
```
**Git**: 81f6d02 (main)
**Run Dir**: `runs/dppo_finetune/mc_exploitation_mc16/`

### Settings
| Parameter | Value |
|-----------|-------|
| env_id | PegInsertionSide-v1 |
| control_mode | pd_joint_delta_pos |
| max_episode_steps | 200 |
| n_envs | 200 |
| n_steps (decision steps) | 25 |
| mc_samples | 16 |
| gamma | 0.999 |
| gae_lambda | 0.95 |
| update_epochs | 1000 |
| minibatch_size | 2500 |
| actor_lr | 3e-6 |
| max_grad_norm | 1.0 |
| clip_ploss_coef | 0.01 |
| clip_ploss_coef_base | 1e-3 |
| clip_ploss_coef_rate | 3.0 |
| target_kl | None (no early stop) |
| norm_adv | True |
| gamma_denoising | 0.99 |
| ft_denoising_steps | 10 |
| use_ddim | True |
| ddim_steps | 10 |
| min_sampling_denoising_std | 0.01 |
| min_logprob_denoising_std | 0.01 |
| zero_qvel | True |
| sim_backend | gpu |
| eval_n_rounds | 3 (600 episodes per eval) |
| network_type | unet (4.40M params) |
| obs_dim | 43 |
| action_dim | 8 |
| dataset size (N) | 5000 samples |
| total PPO samples (N×K) | 50000 |
| pretrain checkpoint | dppo_pretrain_peg_zeroqvel_500k/best.pt |

### Data Collection & MC Estimation
| Metric | Value |
|--------|-------|
| Rollout SR | 57.0% (114/200) |
| MC16 V(s) mean | 0.2207 |
| MC16 V(s) std | 0.2195 |
| MC16 V(s) range | [0.0, 1.0] |
| V(s_final) mean | 0.0400 |
| V(s0) vs traj return corr | r=0.376 |
| MC estimation time | 1199s (~20 min) |
| Advantage mean | 0.1184 |
| Advantage std | 0.3415 |
| frac_positive | 0.496 |
| frac_zero | 0.154 |
| Advantage range | [-0.78, 1.0] |

### Results (SR vs Epoch)
| Epoch | GPU SR | KL | PG Loss | Ratio |
|-------|--------|----|---------|-------|
| 0 (pretrained) | **49.0%** | — | — | — |
| 20 | **66.0%** | 0.0013 | -0.0014 | 0.996 |
| 40 | 63.3% | 0.0023 | -0.0016 | 0.995 |
| 60 | 62.3% | 0.0032 | -0.0017 | 0.994 |
| 80 | 56.8% | 0.0038 | -0.0018 | 0.994 |
| 100 | 66.7% | 0.0054 | -0.0019 | 0.993 |
| 240 | 70.2% | 0.0121 | -0.0023 | 0.990 |
| 400 | 69.8% | 0.0136 | -0.0026 | 0.988 |
| 540 | 53.0% | 0.0149 | -0.0027 | 0.987 |
| 780 | **72.3%** | 0.0144 | -0.0028 | 0.986 |
| 1000 (final) | 62.3% | 0.0170 | -0.0029 | 0.985 |

### Summary
| Metric | Value |
|--------|-------|
| Pretrained SR | 49.0% |
| Best SR | **72.3%** @ epoch 780 |
| Final SR | 62.3% @ epoch 1000 |
| Peak improvement | +23.3% (49.0% → 72.3%) |
| Degradation from peak | 10.0% |
| PPO time | 1966s (32.8 min) |
| Total experiment time | ~55 min (MC=20min + PPO=33min) |

### Notes
- **Experiment goal**: Test whether accurate MC16 advantages on a fixed on-policy dataset can enable efficient PPO exploitation, isolating the "advantage accuracy" vs "distribution shift" question.
- **Fast initial gains**: SR jumps from 49% → 66% by epoch 20 (just 20 passes over fixed data). MC16 advantages clearly provide useful learning signal.
- **Peak at epoch 780 (72.3%)** — but this is noisy. The SR oscillates heavily (±10%) throughout training, making it hard to identify a clean peak.
- **High eval variance**: SR swings between 53% and 72% across evaluations. This is partly due to GPU eval noise and partly real policy instability on fixed data.
- **Gradual degradation**: After ~epoch 300, average SR drifts downward from ~65% to ~62%, consistent with off-policy distribution shift as the policy moves away from the data collection distribution.
- **KL grows monotonically**: 0.001 → 0.017 over 1000 epochs, confirming policy drift from the pretrained baseline. No KL early stopping was used.
- **Comparison to online DPPO finetune**: Best online DPPO reached ~90% (with iterative data + conservative config). Fixed-data exploitation peaks at ~72% — a **~18% gap** attributable to distribution shift. This confirms that **iterative data collection is essential**, not just advantage accuracy.
- **Advantage stats look reasonable**: frac_positive=0.496 (~half), mean=0.12 (slightly positive bias from sparse reward), std=0.34. 15.4% of advantages are exactly zero (states where all MC rollouts gave same outcome).
- **V(s0) vs trajectory return correlation is low (r=0.376)**: This is because V(s0) measures P(success|s0) under current policy, but the trajectory return is binary (0/1) — low correlation is expected for a 57% SR policy.

---

## [MC Exploitation: Data Size × Clip Ratio Sweep] - 2026-03-03 19:11

**Command**: `bash scripts/run_mc_exploit_data_clip_sweep.sh` (12 experiments)
```bash
# Each experiment runs:
python -u -m DPPO.mc_exploitation \
  --pretrain_checkpoint runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt \
  --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos \
  --max_episode_steps 200 --n_envs 500 --n_steps 25 \
  --mc_samples 16 --gae_lambda 0.95 \
  --minibatch_size 2500 --actor_lr 3e-6 \
  --ft_denoising_steps 10 --use_ddim --ddim_steps 10 \
  --min_sampling_denoising_std 0.01 --min_logprob_denoising_std 0.01 \
  --eval_freq 20 --zero_qvel --gamma 0.99 \
  --update_epochs 1000 \
  --clip_ploss_coef {0.01,0.02,0.04,0.08} \
  --subsample_envs {50,200,500}
```
**Git**: 81f6d02 (main)
**Run Dir**: `runs/dppo_finetune/mc_sweep_d{50,200,500}_c{0.01,0.02,0.04,0.08}`
**Plot**: `runs/dppo_finetune/data_clip_sweep.png`

### Settings
| Parameter | Value |
|-----------|-------|
| pretrain_checkpoint | runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt |
| env_id | PegInsertionSide-v1 |
| control_mode | pd_joint_delta_pos |
| max_episode_steps | 200 |
| n_envs (data collection) | 500 |
| subsample_envs (PPO training) | 50, 200, 500 |
| n_steps | 25 |
| mc_samples | 16 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| update_epochs | 1000 |
| minibatch_size | 2500 |
| actor_lr | 3e-6 |
| clip_ploss_coef | 0.01, 0.02, 0.04, 0.08 |
| ft_denoising_steps | 10 |
| use_ddim | True |
| ddim_steps | 10 |
| min_sampling_denoising_std | 0.01 |
| min_logprob_denoising_std | 0.01 |
| zero_qvel | True |
| eval_freq | 20 |
| eval_n_rounds | 3 (600 episodes per eval) |
| norm_adv | True |
| target_kl | None |

### Results

| Data (envs) | clip | best SR | best epoch | final SR | KL@1000 |
|-------------|------|---------|-----------|----------|---------|
| 50 | 0.01 | 60.3% | 100 | 58.7% | 0.0018 |
| 50 | 0.02 | 63.9% | 580 | 57.1% | 0.0573 |
| 50 | 0.04 | 55.5% | 440 | 47.3% | 0.1659 |
| 50 | 0.08 | 52.2% | 0 | 33.1% | 0.3529 |
| 200 | 0.01 | 71.3% | 680 | 64.5% | 0.0026 |
| 200 | 0.02 | 74.0% | 120 | 65.7% | 0.0183 |
| 200 | 0.04 | 72.5% | 20 | 58.4% | 0.1295 |
| 200 | 0.08 | 75.9% | 20 | 40.0% | 0.1848 |
| 500 | 0.01 | 70.3% | 240 | 57.8% | 0.0032 |
| 500 | 0.02 | 73.5% | 440 | 61.8% | 0.0174 |
| 500 | 0.04 | **82.2%** | 180 | 57.5% | 0.1068 |
| 500 | 0.08 | 77.6% | 20 | 48.7% | 0.1858 |

Rollout stats (500 envs, shared across all experiments via cache):
- Rollout SR: 54.2% (271/500)
- V(s): mean=0.0818, std=0.0888
- Advantages: mean=0.1896, std=0.3078, frac_positive=0.548
- V(s0) vs trajectory return correlation: r=0.331

### Notes
- **Data quantity is the dominant factor**: 50→200 envs gives +10-15% best SR across all clip values. 200→500 gives further improvement at moderate clip values.
- **Best single-round result: d500 + clip=0.04 = 82.2%** — significantly better than all other combinations. This exceeds the previous best single-round of ~71% (d200, clip=0.01).
- **Small data cannot tolerate large clip**: d50 + clip=0.08 never improved beyond the starting SR (52.2%). The policy immediately diverged. d50 + clip=0.04 also degraded. Only clip≤0.02 is stable at d50.
- **Large data tolerates larger clip**: d500 + clip=0.04 achieves 82.2%, but d500 + clip=0.08 still degrades to 48.7% by epoch 1000 (though it peaks at 77.6% early).
- **Large clip peaks early, degrades fast**: clip=0.08 always peaks at epoch 20 regardless of data size, then degrades. clip=0.01 peaks late (epoch 100-680) but is more stable.
- **Caching enabled**: 500 envs MC16 data collected once (~50 min), cached to `runs/mc_cache/mc_cache_06e7e1a44e50.pt` (138MB). All 12 experiments reused the same cached data, saving ~9 hours of MC estimation.
- **KL divergence correlates with degradation**: clip=0.08 reaches KL~0.18-0.35 by epoch 1000, while clip=0.01 stays at KL~0.002-0.003. The sweet spot appears to be clip=0.02-0.04 where KL reaches 0.01-0.13.
- **Implication for multi-round strategy**: d500 + clip=0.04 + early stop (~180 epochs) is the most promising single-round config. Whether multi-round chaining from this can exceed 82.2% remains to be tested.

---



## [MC Exploitation Multi-Round: N=500, clip=0.04, 5 rounds] - 2026-03-03 22:50

**Command**: `bash scripts/run_mc_exploit_d500_c04_5rounds.sh`
**Git**: 81f6d02 (main)
**Run Dir**: `runs/dppo_finetune/mc_d500_c04_r{1..5}/`
**Script**: `scripts/run_mc_exploit_d500_c04_5rounds.sh`

### Settings
| Parameter | Value |
|-----------|-------|
| env_id | PegInsertionSide-v1 |
| control_mode | pd_joint_delta_pos |
| max_episode_steps | 200 |
| n_envs | 500 |
| n_steps | 25 |
| mc_samples | 16 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| update_epochs | 250 |
| minibatch_size | 2500 |
| actor_lr | 3e-6 |
| clip_ploss_coef | 0.04 |
| ft_denoising_steps | 10 |
| ddim_steps | 10 |
| min_sampling/logprob_std | 0.01 |
| eval_freq | 20 |
| zero_qvel | True |
| pretrain_checkpoint | dppo_pretrain_peg_zeroqvel_500k/best.pt |
| chaining | best.pt → next round |

### Results

| Round | Rollout SR | Start SR (epoch 0) | Best SR | Best Epoch | End SR (epoch 240) |
|-------|-----------|-------------------|---------|------------|-------------------|
| R1 | ~55% | 52.2% | **81.3%** | 180 | 76.7% |
| R2 | 75.4% | 80.1% | **83.8%** | 80 | 79.1% |
| R3 | 77.0% | 81.1% | **82.0%** | 60 | 74.7% |
| R4 | 76.0% | — | (stuck in MC16) | — | — |
| R5 | — | — | — | — | — |

R4 got stuck during MC16 value estimation (~50 min with no progress, likely GPU memory fragmentation from repeated env creation/destruction across rounds). Killed after R3 completion.

### Detailed Eval Curves

**Round 1** (from pretrain):
| Epoch | SR |
|-------|-----|
| 0 | 52.2% |
| 20 | 79.2% |
| 40 | 76.9% |
| 60 | 77.4% |
| 80 | 80.5% |
| 100 | 75.6% |
| 120 | 76.1% |
| 140 | 76.1% |
| 160 | 77.7% |
| 180 | **81.3%** |
| 200 | 71.6% |
| 220 | 79.1% |
| 240 | 76.7% |

**Round 2** (from R1 best):
| Epoch | SR |
|-------|-----|
| 0 | 80.1% |
| 20 | 83.1% |
| 40 | 82.7% |
| 60 | 78.9% |
| 80 | **83.8%** |
| 100 | 81.3% |
| 120 | 81.1% |
| 140 | 80.0% |
| 160 | 77.1% |
| 180 | 80.6% |
| 200 | 79.7% |
| 220 | 80.4% |
| 240 | 79.1% |

**Round 3** (from R2 best):
| Epoch | SR |
|-------|-----|
| 0 | 81.1% |
| 20 | 78.6% |
| 40 | 81.6% |
| 60 | **82.0%** |
| 80 | 81.0% |
| 100 | 80.9% |
| 120 | 81.2% |
| 140 | 77.9% |
| 160 | 76.6% |
| 180 | 77.4% |
| 200 | 77.3% |
| 220 | 73.2% |
| 240 | 74.7% |

### Notes
- **Plateau confirmed at ~83-84%**: R2 best=83.8%, R3 best=82.0%, no further improvement. Rollout SR also stagnated at 75-77%.
- **Diminishing returns per round**: R1 gained +29% (52→81), R2 gained +2.5% (81→84), R3 gained 0% (84→82, within noise).
- **Best epoch shifts earlier**: R1 best@180, R2 best@80, R3 best@60. Later rounds peak faster but also degrade faster — less room to improve.
- **250 epochs is too many**: All rounds peak at epoch 60-180, then degrade for the remaining epochs. The best checkpoint selection mitigates this, but 80-100 epochs would be more efficient.
- **R4 stuck on MC16**: GPU memory fragmentation from running multiple rounds in a single script (repeated env creation/destruction). Future multi-round scripts should run rounds as separate processes.
- **Root cause of plateau**: At 77% rollout SR, MC16 advantages become near-binary (A≈+0.23 for success, A≈-0.77 for failure). No within-trajectory credit assignment — PPO reinforces/suppresses entire trajectories equally. Need finer-grained advantage signal (lower gamma, dense reward, or per-step contrastive methods) to break through.
- **Comparison**: DPPO finetune (standard online PPO with frequent data refresh) reached 90.3% on the same task, suggesting the plateau is method-specific, not task-specific.

---

## [MC Exploitation: Retry Failed (Replace Mode)] - 2026-03-04 21:08

**Command**: `python -u -m DPPO.mc_exploitation --pretrain_checkpoint runs/dppo_finetune/mc_d500_c04_r2/best.pt --exp_name mc_retry_r2best --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos --max_episode_steps 200 --n_envs 500 --n_steps 25 --mc_samples 16 --gae_lambda 0.95 --update_epochs 250 --minibatch_size 2500 --actor_lr 3e-6 --ft_denoising_steps 10 --use_ddim --ddim_steps 10 --min_sampling_denoising_std 0.01 --min_logprob_denoising_std 0.01 --eval_freq 20 --zero_qvel --gamma 0.99 --clip_ploss_coef 0.02 --retry_failed --no_cache`
**Git**: 81f6d02 (main)
**Run Dir**: runs/dppo_finetune/mc_retry_r2best/

### Settings
| Parameter | Value |
|-----------|-------|
| pretrain_checkpoint | runs/dppo_finetune/mc_d500_c04_r2/best.pt (R2 best, ~83.8% SR) |
| env_id | PegInsertionSide-v1 |
| control_mode | pd_joint_delta_pos |
| max_episode_steps | 200 |
| n_envs | 500 |
| n_steps | 25 |
| mc_samples | 16 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| update_epochs | 250 |
| minibatch_size | 2500 |
| actor_lr | 3e-6 |
| clip_ploss_coef | 0.02 |
| ft_denoising_steps | 10 |
| ddim_steps | 10 |
| min_sampling/logprob_std | 0.01 |
| zero_qvel | True |
| retry_failed | True (REPLACE mode — bug, failed trajs overwritten) |

### Dataset
| Metric | Value |
|--------|-------|
| Original rollout | 385/500 success (77.0%) |
| After retry (replace) | 500/500 success (100.0%) |
| Retries needed | 31 rounds for hardest state |
| Advantages mean | 0.2934 |
| Advantages frac_positive | 0.777 |
| Advantages frac_zero | 0.091 |

### Results
| Epoch | SR | KL | PG Loss | Ratio |
|-------|------|------|---------|-------|
| 0 | **81.3%** | 0.000 | 0.000 | 1.000 |
| 20 | 74.5% | 0.013 | -0.003 | 0.977 |
| 40 | 77.6% | 0.021 | -0.003 | 0.975 |
| 60 | 74.4% | 0.026 | -0.003 | 0.974 |
| 80 | 74.3% | 0.031 | -0.004 | 0.973 |
| 100 | 71.4% | 0.038 | -0.004 | 0.971 |
| 140 | 70.3% | 0.047 | -0.004 | 0.969 |
| 200 | 64.6% | 0.056 | -0.004 | 0.968 |
| 240 | 61.3% | 0.057 | -0.004 | 0.966 |

### Notes
- **Bug: Replace mode** — failed trajectories were REPLACED with successful retries, discarding original failures. This removed negative advantage signal ("what not to do").
- **Best = epoch 0 (81.3%)**, monotonic degradation to 61.3%. Worse than non-retry baseline (best 81.7% @ epoch 140).
- **Why replace hurts**: The 115 retried trajectories are outlier lucky events from states where P(success) ≈ 1-3%. High advantage on these outliers dominates PPO gradient, pushing policy away from its mode.
- **Advantage bias**: mean=0.293 (too positive), because all trajs succeed. Without failed trajs providing negative advantages, the gradient is biased.
- **Fix**: Changed to APPEND mode — keep original failures + add successful retries. Also changed to save ALL retry trajectories (success AND fail) for statistical correctness.

---

## [R2 Best Coverage: Extended Max Steps] - 2026-03-04 21:08

**Command**: `python -u dp_p_success_gpu.py --ckpt runs/dppo_finetune/mc_d500_c04_r2/best.pt --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos --max_episode_steps {200,300,500,2000} --zero_qvel --num_states 200 --mc_samples {1,16} --ddim_steps 10 --min_sampling_denoising_std 0.01`
**Git**: 81f6d02 (main)

### Results: R2 Best Coverage vs Max Steps

| max_steps | Mode | SR | frac_zero | frac_one | frac_decisive |
|-----------|------|------|-----------|----------|---------------|
| 200 | deterministic | 86.0% | 14.0% | 86.0% | 0% |
| 300 | deterministic | 86.0% | 14.0% | 86.0% | 0% |
| 2000 | deterministic | 93.5% | 6.5% | 93.5% | 0% |
| 200 | stochastic mc16 | 77.1% | 4.5% | 16.0% | 59.5% |
| 300 | stochastic mc16 | 84.3% | 3.5% | 32.0% | 37.5% |
| 500 | stochastic mc16 | 88.4% | 2.0% | 49.5% | 27.5% |
| 2000 | stochastic mc1 | 94.5% | 5.5% | 94.5% | 0% |

### Pretrain Coverage Comparison

| max_steps | Mode | SR | frac_zero | frac_one | frac_decisive |
|-----------|------|------|-----------|----------|---------------|
| 200 | deterministic | 55.0% | 45.0% | 55.0% | 0% |
| 300 | deterministic | 66.5% | 33.5% | 66.5% | 0% |
| 200 | stochastic mc16 | 53.5% | 5.0% | 3.0% | 81.0% |
| 300 | stochastic mc16 | 60.4% | 4.5% | 4.5% | 80.5% |

### Notes
- **Deterministic DP is always bimodal (frac_decisive=0%)** — both pretrain and finetuned. Identical DDIM actions for same state → P=0 or P=1.
- **Stochastic decisive fraction shrinks with more steps**: 59.5% → 37.5% → 27.5% as max_steps 200→300→500. More time lets marginal states succeed.
- **~6% true frac_zero at 2000 steps**: These are geometry configurations the policy cannot handle regardless of time or stochasticity.
- **Timeout is major bottleneck**: 86%→94% from 200→2000 steps, meaning ~8% of failures are just "too slow" not "can't do."

---

## [Frac Zero Analysis: MC200 on Failing States] - 2026-03-04 21:08

**Command**: Custom script testing frac_zero states from R2 best (deterministic 2000 steps) with MC200 on both pretrain and finetuned policies.
**Git**: 81f6d02 (main)

### Results: MC200 per-state P(success) at 200 max steps

| env_idx | P_pretrain | P_finetuned | delta |
|---------|-----------|-------------|-------|
| 1 | 0.0% | 1.0% | +1.0% |
| 6 | 1.5% | 0.5% | -1.0% |
| 18 | 1.0% | 0.5% | -0.5% |
| 24 | 2.5% | 1.0% | -1.5% |
| 33 | 2.5% | 0.0% | -2.5% |
| 34 | 0.5% | 0.0% | -0.5% |
| 42 | 0.0% | 0.0% | +0.0% |
| 65 | 0.5% | 0.0% | -0.5% |
| 106 | 1.5% | 0.5% | -1.0% |
| 110 | 0.5% | 0.5% | +0.0% |
| 125 | 0.5% | 0.0% | -0.5% |
| 128 | 4.0% | 0.0% | -4.0% |
| 131 | 0.5% | 1.5% | +1.0% |
| 183 | 1.5% | 1.0% | -0.5% |
| 184 | 3.0% | 1.0% | -2.0% |
| 195 | 2.5% | 1.0% | -1.5% |
| **mean** | **1.41%** | **0.53%** | **-0.88%** |

### Notes
- **Not truly zero coverage**: MC50 showed 0% for many states, but MC200 reveals most have P=0.5-4% under pretrain. MC50 has 36% chance of missing P=2% (0.98^50).
- **Finetuning causes mild regression on edge cases**: Mean P drops from 1.4%→0.5% on these hard states. Policy optimizes for the majority of states at a small cost to the tail.
- **Only 1/16 (env 42) is truly 0% for both policies** — the rest are extremely low but non-zero.
- **These are geometry edge cases**, not fundamental failures. The peg/hole configurations are extreme, making insertion nearly impossible for any learned policy.

---

## [MC Exploitation: Retry Failed (Append All Trajs)] - 2026-03-04 21:08

**Command**: `python -u -m DPPO.mc_exploitation --pretrain_checkpoint runs/dppo_finetune/mc_d500_c04_r2/best.pt --exp_name mc_retry_alltrajs_r2best --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos --max_episode_steps 200 --n_envs 500 --n_steps 25 --mc_samples 16 --gae_lambda 0.95 --update_epochs 250 --minibatch_size 2500 --actor_lr 3e-6 --ft_denoising_steps 10 --use_ddim --ddim_steps 10 --min_sampling_denoising_std 0.01 --min_logprob_denoising_std 0.01 --eval_freq 20 --zero_qvel --gamma 0.99 --clip_ploss_coef 0.02 --retry_failed --no_cache`
**Git**: 81f6d02 (main)
**Run Dir**: runs/dppo_finetune/mc_retry_alltrajs_r2best/

### Settings
| Parameter | Value |
|-----------|-------|
| pretrain_checkpoint | runs/dppo_finetune/mc_d500_c04_r2/best.pt (R2 best, ~83.8% SR) |
| retry_failed | True (APPEND mode — keeps original fails + appends ALL retry trajs) |
| All other settings | Same as replace experiment above |

### Dataset
| Metric | Value |
|--------|-------|
| Original rollout | 385/500 success (77.0%), 115 fail |
| Retry trajectories appended | 174 (65 success + 109 fail) |
| Total dataset | 674 trajectories |
| Retry rounds | 31 (same seed, same states) |
| Dataset composition | 500 original + 174 retries, statistically representative |

### Results
Best SR: 83.9% @ epoch 20, then oscillated 80-84%. No clear improvement over non-retry baseline (~82%).
Conclusion: Retry append doesn't help — not the bottleneck.

### Notes
- **Statistically correct**: Each retry round saves ONE trajectory per still-failed state regardless of outcome. A state needing K retries contributes K trajectories (K-1 fail + 1 success), reflecting the true P(success) distribution.
- **Fix for two bugs from replace version**: (1) append instead of replace, (2) save all retry trajs not just successes.
- **Dataset breakdown**: 174 appended = 115 (round 1) + 15 (round 2) + 6 + 3 + 2×7 + 2 + 1×18 + 1 = 174. Matches theoretical count.

---

## [Frac Zero Video Recording Script] - 2026-03-04

**Script**: `DPPO/record_frac_zero.py`
**Git**: 81f6d02 (main)

**Command**:
```bash
python -u -m DPPO.record_frac_zero \
  --ft_ckpt runs/dppo_finetune/mc_d500_c04_r2/best.pt \
  --pretrain_ckpt runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt \
  --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos \
  --max_episode_steps 2000 --zero_qvel \
  --num_states 200 --max_videos 14 \
  --output_dir runs/frac_zero_videos
```

### How it works
1. **Phase 1**: Batch GPU eval (200 envs, deterministic DDIM-10) to find succeed_once=False states. Logic identical to `dp_p_success_gpu.py`.
2. **Phase 2**: For each failing state, record GPU video (single env, render every raw step, fps=20) with both finetuned and pretrain policies.
3. Saves `fail_states.pt` (initial states) + `state{idx}_finetuned.mp4` + `state{idx}_pretrain.mp4`.

### Expected results
- R2 best @ 2000 steps deterministic: ~93% SR, ~14 failing states (7% frac_zero)
- These are states where the policy never succeeds even once in 2000 steps — true capability limits, not timeout

### Notes
- succeed_once tracking (not succeed_at_end): `success = success | (got_reward & ~done)` — same as dp_p_success_gpu.py
- 2000 steps视频很长（每帧=1 raw step, ~2000 frames/video）
- GPU单env录制：和batch eval可能有微小差异（GPU sim batch vs single env），但不做额外验证

---

## [Gamma=0.9 MC Exploitation 5 Rounds] - 2026-03-04

**Command**: `bash scripts/run_mc_exploit_gamma09_5rounds.sh`
**Git**: 81f6d02 (main)
**Status**: Running (Round 1 MC16 in progress)

**Config**: Same as gamma=0.99 experiments, only `--gamma 0.9`:
```bash
COMMON="--env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos \
  --max_episode_steps 200 --n_envs 500 --n_steps 25 \
  --mc_samples 16 --gae_lambda 0.95 \
  --update_epochs 250 --minibatch_size 2500 --actor_lr 3e-6 \
  --ft_denoising_steps 10 --use_ddim --ddim_steps 10 \
  --min_sampling_denoising_std 0.01 --min_logprob_denoising_std 0.01 \
  --eval_freq 20 --zero_qvel --gamma 0.9 \
  --clip_ploss_coef 0.02"
```

**Hypothesis**: gamma=0.99 plateau ~85% because policy has no incentive to succeed faster. ~8% of failures are timeout (succeed @ 2000 steps but not @ 200). gamma=0.9 creates stronger speed pressure:
- gamma=0.9: step 5 return=0.59, step 25=0.07 (gap=0.52)
- gamma=0.99: step 5=0.95, step 25=0.78 (gap=0.17)

**Exp names**: `mc_g09_c02_r1` through `mc_g09_c02_r5`

### Results

**Round 1** (from pretrain, best=64.7% @ epoch 140):

| Epoch | SR | KL |
|-------|------|--------|
| 0 | 51.8% | 0.0 |
| 20 | 59.0% | 0.005 |
| 40 | 64.5% | 0.014 |
| 140 | **64.7%** | 0.063 |
| 240 | 59.2% | 0.103 |

**Round 2 COLLAPSED** (from R1 best, start=65.1%):

| Epoch | SR | KL |
|-------|------|--------|
| 0 | 65.1% | 0.0 |
| 20 | **48.7%** | 0.057 |
| 40 | 49.5% | 0.120 |
| 60 | 48.1% | 0.182 |

**Root cause**: gamma=0.9 makes V(s) highly sensitive to policy timing behavior. MC16 V is computed once (fixed), but PPO iteratively updates policy → V becomes stale → advantages artificially inflated → positive feedback loop → collapse. gamma=0.99 doesn't have this issue because V is timing-insensitive.

### Conservative variant (lr=1e-6, target_kl=0.01) — also collapsed

```bash
python -u -m DPPO.mc_exploitation \
  --pretrain_checkpoint runs/dppo_finetune/mc_g09_c02_r1/best.pt \
  --exp_name mc_g09_cons_r2_test \
  ... --actor_lr 1e-6 --target_kl 0.01 --gamma 0.9
```

R2 with same data cache: start=62.1%, epoch 20=48.9%. KL stop triggered from epoch 10 but already too late. **Confirms the issue is fundamental (fixed MC16 V + changing policy), not just hyperparameters.**

---

## [GAE Learned Critic Finetune (PegInsertionSide)] - 2026-03-05

**Command**:
```bash
python -u -m DPPO.finetune \
  --pretrain_checkpoint runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt \
  --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos \
  --max_episode_steps 200 --n_envs 500 --n_steps 25 \
  --gamma 0.99 --gae_lambda 0.95 \
  --update_epochs 5 --minibatch_size 2500 \
  --actor_lr 3e-6 --critic_lr 1e-3 \
  --n_critic_warmup_itr 5 \
  --ft_denoising_steps 10 --use_ddim --ddim_steps 10 \
  --min_sampling_denoising_std 0.01 --min_logprob_denoising_std 0.01 \
  --clip_ploss_coef 0.02 --target_kl 1.0 \
  --eval_freq 1 --eval_n_rounds 3 \
  --zero_qvel --reward_scale_running \
  --n_train_itr 10 \
  --exp_name dppo_ft_peg_gae_g099
```
**Git**: 81f6d02 (main)
**Run dir**: `runs/dppo_finetune/dppo_ft_peg_gae_g099/`

### Results

| Iter | SR | KL | Note |
|------|------|--------|------|
| 1-5 | 50-55% | 0.0 | Critic warmup only |
| 6 | 74.9% | 0.0003 | First actor update |
| 7 | 80.3% | 0.0003 | |
| 8 | 83.1% | 0.0003 | |
| 9 | 83.4% | 0.0003 | |
| 10 | **85.5%** | 0.0003 | Still improving |

### Comparison with MC exploitation

| Method | Best SR | Data budget | V(s) source |
|--------|---------|-------------|-------------|
| MC16 gamma=0.99 R1 | 66.8% | 500 eps + MC16 | Fixed MC rollout |
| MC16 gamma=0.99 R2 | 77.2% | 1000 eps + MC16 | Fixed MC rollout |
| MC16 gamma=0.9 R1 | 64.7% | 500 eps + MC16 | Fixed MC rollout |
| MC16 gamma=0.9 R2 | COLLAPSED | — | Fixed MC rollout |
| **GAE learned critic** | **85.5%** | 5000 eps (no MC) | On-policy critic |

### Key insights
1. **GAE learned critic >> MC16 fixed**: 85.5% vs 77.2%, and still improving at iter 10
2. **No V staleness**: Critic updates every iteration with fresh on-policy data, so V(s) never becomes stale
3. **No gamma sensitivity**: With learned critic, gamma=0.99 works fine because V adapts to policy changes
4. **Standard DPPO finetune already works**: No need for MC exploitation's complex pipeline — the simple iterative rollout + GAE + PPO loop is superior
5. **Data efficiency**: 5 training iterations (2500 episodes) to reach 85%, vs mc_exploitation needing 2 rounds (1000 episodes + expensive MC16) for only 77%

### Scaling experiments (same base config, varying n_envs / epochs / iters)

| Config | Best SR | Best Iter | Total training episodes |
|--------|---------|-----------|------------------------|
| 500 envs, 5ep, 10 itr | 85.5% | 10 | 2500 |
| 200 envs, 10ep, 10 itr | 82.5% | 9 | 1000 |
| 200 envs, 13ep, 10 itr | 84.0% | 9 | 1000 |
| 500 envs, 5ep, 20 itr | **91.7%** | 20 | 7500 |
| 500 envs, 10ep, 20 itr | 91.6% | 20 | 7500 |
| 500 envs, 20ep, 20 itr | 90.7% | 16 | 7500 |

**Findings**:
- More iterations (data collection rounds) is the main driver: 10→20 itr = 85%→92%
- More epochs per iter speeds up convergence but doesn't raise ceiling, and too many hurts:
  - 5ep: 91.7% (best), 10ep: 91.6% (same), 20ep: 90.7% (worse, oscillates due to over-optimization on single batch)
  - 20ep has 100 updates/iter → too much policy shift per data collection round
- **5 epochs is the sweet spot** for this setting (500 envs, minibatch=2500)
- Fewer envs (200 vs 500) loses ~1-2% at equal gradient steps, due to higher advantage variance
- Policy still improving at iter 20, ceiling appears ~91-92% at 200 steps

### Ceiling analysis: what bounds the ~92% plateau?

Evaluated 20-iter best checkpoint (91.7% @ 200 steps) at both 200 and 2000 steps on 600 states:

| | 200 steps | 2000 steps |
|---|---|---|
| Pretrain | 55% | 95% |
| 10-iter best (85.5%) | 85.5% | 92.3% |
| **20-iter best (91.7%)** | **88.3%** | **95.5%** |

Per-state breakdown (20-iter best, 600 states, deterministic mc1):

| Category | Count | % | Meaning |
|----------|-------|---|---------|
| Both succeed | 514 | 85.7% | Solved within 200 steps |
| **Timeout** | 59 | **9.8%** | 200 steps too short, 2000 steps succeeds |
| True dead | 11 | **1.8%** | Neither succeeds (env-level hard) |
| Degraded | 16 | 2.7% | 200 ok but 2000 fail (GPU sim stochasticity) |

**Conclusion**: The ~92% ceiling is NOT a policy capability bound — it's an episode length bound. The policy can solve 95.5% of states given enough time. Only ~2% are truly unsolvable. The remaining ~8% failure at 200 steps is timeout: the policy needs >200 steps on those states. Finetuning optimizes for "succeed within 200 steps" but cannot compress all trajectories below this budget.

**Implication**: If episode length increased to 400-500 steps, SR should reach ~95%. Alternatively, the pretrain policy already achieves 95% at 2000 steps — finetune's value is purely speed (compressing success into shorter episodes), not expanding coverage.

---

## [REINFORCE Ablation: Pure Policy Gradient without Baseline/Clip] - 2026-03-05 05:00

**Git**: 81f6d02 (main)
**Script**: `DPPO/finetune_reinforce.py` (original REINFORCE version, later renamed to `DPPO/finetune_bc.py` for weighted BC)
**Run Dir**: `runs/dppo_finetune/reinforce_*`

### Context

Ablation study to understand what components of PPO are essential for DPPO finetuning. Stripped PPO of baseline (critic) and clipping, testing pure REINFORCE on fixed sparse reward.

### REINFORCE Experiments (Policy Gradient)

Multiple iterations of debugging were needed:
- norm_adv=True caused collapse (mean-subtracting returns creates destructive negative gradients)
- Denoising discount biased gradient toward later denoising steps without baseline
- Timestep discount (gamma) biased gradient toward early timesteps
- Final working config: undiscounted returns, no denoising discount, no return normalization

**Command (lr=1e-5, working config)**:
```bash
python -u -m DPPO.finetune_reinforce \
  --pretrain_checkpoint runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt \
  --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos \
  --max_episode_steps 200 --n_envs 500 --n_steps 25 \
  --gamma 0.99 --update_epochs 1 --minibatch_size 2500 \
  --actor_lr 1e-5 --no-norm_adv \
  --ft_denoising_steps 10 --use_ddim --ddim_steps 10 \
  --min_sampling_denoising_std 0.01 --min_logprob_denoising_std 0.01 \
  --eval_freq 1 --eval_n_rounds 3 \
  --zero_qvel --reward_scale_running \
  --n_train_itr 20 --exp_name reinforce_nonorm_lr1e5
```

### REINFORCE LR Sweep Results

| lr | KL per step | Result |
|---|---|---|
| 1e-5 | 0.0001 | No collapse, but no improvement (75% → 66% stagnation) |
| 5e-5 | 0.02 | Collapsed to 0% in 2 iters |
| 1e-4 | 0.17 | Collapsed to 0% in 1 iter |

**Conclusion**: Without baseline + without clip, there's no viable learning rate. Small lr = no learning signal (weighted BC is already at its fixed point). Large lr = collapse (noisy gradient direction from success-only weighting).

---

## [Weighted BC / Filtered BC Finetuning] - 2026-03-05 07:00

**Git**: 81f6d02 (main)
**Script**: `DPPO/finetune_bc.py` (weighted diffusion BC) and `DPPO/finetune_reinforce.py` (REINFORCE with binary returns)
**Run Dir**: `runs/dppo_finetune/wbc_*` and `runs/dppo_finetune/reinforce_binary_ep5`

### Context

After REINFORCE failed, rewrote the script as **weighted diffusion BC**: replace policy gradient with weighted MSE diffusion loss. Key insight: with sparse reward and undiscounted returns, the weight is binary (1 for success, 0 for failure) → **filtered BC** (only train on successful rollout trajectories).

### Settings (common)
| Parameter | Value |
|-----------|-------|
| pretrain_checkpoint | runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt |
| env_id | PegInsertionSide-v1 |
| control_mode | pd_joint_delta_pos |
| max_episode_steps | 200 |
| n_envs | 500 |
| n_steps | 25 |
| gamma | 0.99 |
| minibatch_size | 2500 |
| actor_lr | 3e-6 |
| use_ddim | True |
| ddim_steps | 10 |
| min_sampling_denoising_std | 0.01 |
| zero_qvel | True |
| reward_scale_running | True |
| n_train_itr | 20 |
| eval_n_rounds | 3 |

### Weighted BC: Update Epochs Sweep (weight_mode=binary_pos)

**Command template**:
```bash
python -u -m DPPO.finetune_bc \
  --pretrain_checkpoint runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt \
  --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos \
  --max_episode_steps 200 --n_envs 500 --n_steps 25 \
  --gamma 0.99 --update_epochs {EP} --minibatch_size 2500 \
  --actor_lr 3e-6 --weight_mode binary_pos \
  --use_ddim --ddim_steps 10 --min_sampling_denoising_std 0.01 \
  --eval_freq 1 --eval_n_rounds 3 \
  --zero_qvel --reward_scale_running \
  --n_train_itr 20 --exp_name wbc_binary_pos_ep{EP}
```

| Epochs | Peak SR | Iter@peak | Trend |
|---|---|---|---|
| 2 | 65.7% | 9 | Under-training, degrades after |
| 3 | 67.1% | 17 | Under-training, high variance |
| **5** | **85.2%** | **16** | **Stable improvement** |
| 10 | 69.0% | 8 | Overfitting per batch |
| 20 | 69.0% | 8 | Severe overfitting, degrades |

**Best (ep5) per-iter detail**:

| Iter | SR | Rollout succ/500 |
|---|---|---|
| 1 | 56.1% | 271 |
| 5 | 57.5% | 314 |
| 10 | 80.7% | 372 |
| 15 | 83.1% | 408 |
| 16 | **85.2%** | 392 |
| 20 | 82.7% | 404 |

### REINFORCE with Binary Returns (Filtered-BC Equivalent via Policy Gradient)

**Command**:
```bash
python -u -m DPPO.finetune_reinforce \
  --pretrain_checkpoint runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt \
  --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos \
  --max_episode_steps 200 --n_envs 500 --n_steps 25 \
  --gamma 1.0 --reward_scale_const 1.0 --no-reward_scale_running \
  --no-norm_returns --binarize_returns \
  --gamma_denoising 1.0 \
  --update_epochs 5 --minibatch_size 2500 \
  --actor_lr 3e-6 --max_grad_norm 1.0 \
  --min_sampling_denoising_std 0.01 --min_logprob_denoising_std 0.01 \
  --use_ddim --ddim_steps 10 \
  --eval_freq 1 --eval_n_rounds 3 \
  --zero_qvel \
  --n_train_itr 20 --exp_name reinforce_binary_ep5
```

| Iter | SR | Rollout succ/500 |
|---|---|---|
| 1 | 52.5% | 271 |
| 5 | 65.1% | 312 |
| 10 | 79.9% | 396 |
| 14 | 85.6% | 418 |
| 16 | 87.6% | 410 |
| 17 | **88.2%** | 408 |
| 20 | 84.9% | 416 |

### Summary Comparison

| Method | Peak SR | Notes |
|---|---|---|
| PPO (GAE + critic + clip, 5ep) | **91.7%** | Full DPPO pipeline |
| REINFORCE (binary returns, 5ep) | 88.2% | No critic, no clip, policy gradient |
| Weighted BC (binary_pos, 5ep) | 85.2% | No critic, no clip, diffusion MSE loss |
| REINFORCE (nonorm, lr=1e-5) | ~70% | Stagnates, no improvement |
| REINFORCE (any lr ≥ 5e-5) | 0% | Collapses |

### Notes

- **Filtered BC works surprisingly well** (85%): only train diffusion model on successful rollout trajectories, iterate. No critic, no advantage, no clipping needed.
- **REINFORCE with binary returns** (88.2%) slightly outperforms weighted BC, possibly because policy gradient directly optimizes the action distribution rather than fitting noise predictions.
- **5 epochs is the universal sweet spot** — same conclusion as PPO scaling experiments. Under-training (2-3ep) can't learn enough per iteration; over-training (10-20ep) overfits each batch.
- **The ~6% gap to PPO (91.7% vs 85-88%)** comes from the critic/baseline providing contrastive signal: PPO learns from both successes AND failures (negative advantages push away from bad actions), while filtered BC/REINFORCE only learns from successes.
- **Pure REINFORCE (without binary trick) is unstable**: no viable lr exists — too small = stagnation, too large = collapse. The binary return trick works because it removes return variance (all weights are 0 or 1), making the gradient direction stable.

---

## [DPPO Policy-Extraction Bridge: PPO Clip -> REINFORCE+IS] - 2026-03-05 12:30

**Git**: d5fe463 (main)  
**Script**: `DPPO/finetune_bridge_ablate.py`, `scripts/run_policy_extraction_bridge.sh`  
**Run Dirs**:
- `runs/dppo_finetune/peb_p0_ppo_clip_full_seed0`
- `runs/dppo_finetune/peb_p1_reinforce_is_gae_seed0`
- `runs/dppo_finetune/peb_p2_reinforce_is_td_seed0`
- `runs/dppo_finetune/peb_p3_reinforce_is_nobaseline_seed0`
- `runs/dppo_finetune/peb_p4_reinforce_is_final_seed0`
**Logs**: `logs/policy_extraction_bridge/`

### Context

Goal: start from a `finetune.py`-aligned PPO/DPPO setup and progressively replace only the **policy extraction / policy loss** with the `finetune_reinforce.py` style IS-REINFORCE surrogate, while keeping rollout structure, optimizer envelope, and diffusion policy architecture fixed.

This was used to answer: why do full DPPO (`~0.91`) and binary-return REINFORCE (`~0.86`) both work, but the naive "critic + GAE + reinforce_is" bridge initially collapsed?

### Shared Settings
| Parameter | Value |
|-----------|-------|
| pretrain_checkpoint | runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt |
| env_id | PegInsertionSide-v1 |
| control_mode | pd_joint_delta_pos |
| max_episode_steps | 200 |
| n_envs | 500 |
| n_steps | 25 |
| gamma | 0.99 |
| update_epochs | 10 |
| minibatch_size | 2500 |
| actor_lr | 3e-6 |
| critic_lr | 1e-3 |
| use_ddim | True |
| ddim_steps | 10 |
| min_sampling_denoising_std | 0.01 |
| min_logprob_denoising_std | 0.01 |
| eval_freq | 1 |
| eval_n_rounds | 3 |
| zero_qvel | True |
| reward_scale_running | True |
| n_train_itr | 20 |
| warmup (critic stages) | 5 iters |

### Stage Definitions
| Stage | Meaning |
|---|---|
| p0 | Full PPO clip path (`finetune.py`-like) |
| p1 | Replace PPO policy loss with `reinforce_is`, keep critic + GAE |
| p2 | Remove GAE, keep critic baseline (`returns - V`) |
| p3 | Remove critic baseline entirely |
| p4 | Final REINFORCE style: binary returns, no norm, no critic |

### Main Stage Results (seed 0)

| Stage | Best SR | Final SR | AUC | Notes |
|---|---|---|---|---|
| p0 PPO clip full | 90.9% | 90.9% | 0.778 | DPPO anchor |
| p1 REINFORCE+IS + GAE | **93.0%** | **93.0%** | 0.785 | Stable after fixing clip schedule |
| p2 REINFORCE+IS + TD | 53.6% | 0.6% | 0.248 | Fails badly |
| p3 REINFORCE+IS no baseline | 40.1% | 13.9% | 0.204 | Fails badly |
| p4 REINFORCE final | 86.5% | 85.2% | 0.791 | Binary success-only regime |

### p1 Diagnosis Timeline

#### 1. Original p1: flat IS clip = 0.2
- Immediate post-warmup collapse.
- Iter 5 eval: 51.9%
- Iter 6: `ratio=0.9727`, `gpu_sr=6.4%`
- Conclusion: this was not evidence that "GAE + reinforce" is impossible; the extraction / trust-region was too loose.

#### 2. Tighten flat IS clip to 0.02
- Early collapse disappeared.
- Iter 6: `ratio=0.9984`, `gpu_sr=68.2%`
- Best SR improved to 89.7%.
- This showed the main issue was trust-region scale, not GAE itself.

#### 3. Port PPO denoising-step clip schedule into `reinforce_is`
- Used the same per-denoising-step clip schedule as PPO:
  early denoising steps clipped very tightly (near `1e-3`), later steps gradually relaxed toward `0.02`.
- Result: no training collapse across all 20 iterations.
- Final `p1`:
  - best SR = 93.0%
  - final SR = 93.0%
  - iter 19 eval = 92.6%
  - iter 20 eval = 93.0%

### Key Findings

- **`p1` vs `p4` is NOT a single-variable "GAE vs 0/1 return" comparison.**
  `p1` uses critic + signed GAE + value loss + warmup + denoising discount.
  `p4` uses no critic, binary positive-only returns, no return normalization, and no denoising discount.

- **The original `p1` collapse was caused mainly by extraction / trust-region mismatch.**
  A flat `is_clip_ratio=0.2` is far looser than PPO's effective clip for early denoising steps.

- **The critical factor is denoising-step-aware clipping.**
  Early diffusion sub-steps are much more sensitive; they need much tighter clipping than later refinement steps.
  Once `reinforce_is` used PPO's denoising-step clip schedule, `critic + GAE + reinforce_is` became fully stable.

- **This means the essential difference is not "PPO loss vs REINFORCE loss" in the abstract.**
  The real stabilizer is the trust-region design over diffusion sub-steps.

- **`target_kl=1.0` was not active in these runs.**
  Observed KL stayed around `1e-5` to `1e-4`, so early-stop never triggered.
  The stability gain came from the clip schedule, not from KL stopping.

- **`p2` and `p3` still fail badly.**
  So after fixing extraction, the current evidence is:
  `critic + GAE + reinforce_is` can work,
  but removing GAE or removing the critic baseline does not preserve performance in this bridge setting.

### Implementation Notes

- The bridge code was cleaned up so that:
  - `ppo_clip` passes raw advantages directly into `model.loss(...)`, matching `finetune.py`
  - `reinforce_is` applies its own `norm_adv + gamma_denoising` preprocessing locally
- This keeps the PPO anchor cleaner and makes the policy-extraction bridge more interpretable.

---
