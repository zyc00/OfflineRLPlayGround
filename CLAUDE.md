# Offline RL Playground

Offline-to-online RL finetuning research platform. Compares advantage estimation methods (GAE, MC, IQL) and policy update rules (PPO, AWR).

## Common Commands

```bash
# Online AWR (best method) - mc16 optimal, gamma=0.8, beta=0.5
python -m RL.mc_finetune_awr_parallel --mc_samples 16 --awr_beta 0.5 --gamma 0.8 \
  --num_envs 100 --num_steps 50 --update_epochs 200 --total_timesteps 50000

# On-policy AWR (no optimal policy needed)
python -m RL.mc_finetune_awr_onpolicy --mc_samples 16 --awr_beta 0.5

# GAE PPO baseline
python -m RL.ppo_finetune --num_envs 100 --num_steps 50 --update_epochs 100 --target_kl 100.0

# Offline AWR (fixed dataset)
python -m RL.mc_finetune_awr_offline --mc_samples 16 --awr_beta 1.0 --num_iterations 100

# IQL offline
python -m RL.iql_awr_offline --iql_expectile_tau 0.7 --awr_beta 1.0
```

## Project Structure

- `RL/` - Training scripts (entry points)
- `methods/gae/` - GAE and ranking analysis scripts
- `methods/iql/` - IQL implementation
- `methods/mc/` - MC advantage estimation
- `experiments/log.md` - Detailed experiment results
- `runs/` - TensorBoard logs and checkpoints

## Best Configurations

| Regime | Method | Peak SR | Config |
|--------|--------|---------|--------|
| Online (oracle) | MC16 AWR | 99.1% | mc=16, beta=0.5, gamma=0.8, epochs=200 |
| Online (no oracle) | MC16 on-policy AWR | 96.7% | mc=16, beta=0.5, gamma=0.8 |
| GAE PPO | GAE + pretrained critic | 93.2% | gamma=0.8, pretrained V |
| Offline | MC16 optimal AWR | 88.2% | Large batch, epochs=4 |

---

## KNOWN ISSUES & BUGS - DO NOT REPEAT

### 1. Checkpoint Selection

**WRONG**: Using `ckpt_101` for finetuning experiments
- Deterministic SR ~99%, stochastic ~62.7%
- Too strong - improvements come from reducing noise, not learning
- Masks true method differences

**CORRECT**: Use `ckpt_76_logstd-1.5` (det SR=43.8%)
```bash
--checkpoint runs/pickcube_ppo/ckpt_76_logstd-1.5.pt
```

### 2. Evaluation Mode

**WRONG**: Stochastic evaluation (default)
- ~12% eval noise from action sampling
- Results oscillate wildly

**CORRECT**: Always use deterministic eval
- Set `deterministic=True` in eval loop
- Consistent, reproducible results

### 3. High UTD PPO Collapse

**WRONG**: High update_epochs with tight target_kl
```bash
--update_epochs 20 --target_kl 0.1  # COLLAPSES
```

**CORRECT**: Disable KL constraint for high UTD
```bash
--update_epochs 20 --target_kl 100.0  # Works
```

### 4. Warmup + Tight KL Collapse

**WRONG**: Short warmup with restrictive KL
```bash
--warmup_iters 3 --target_kl 0.1  # Collapsed to 7%
```

**CORRECT**: Either longer warmup or disable KL constraint

### 5. Experiment Naming Mislabeling

**HISTORICAL BUG**: Several experiments have wrong exp_name:
- `mc5_awr` actually used mc_samples=16
- `mc1_qv_par` actually used mc_samples=5

**CORRECT**: Always double-check mc_samples matches exp_name

### 6. V* Distribution Mismatch

**WRONG**: Using V* (optimal value function) for finetuning
- V* trained on pi* states doesn't generalize to pi_0 states
- Result: 83.3% vs 93.2% with V^pi_expert

**CORRECT**: Use V^pi_expert (expert's value function)
```bash
--critic_checkpoint runs/pretrained_critic_sparse.pt  # V^pi_expert, not V*
```

### 7. Optimal MC in Offline Setting

**WRONG**: Using optimal policy for MC re-rollout in OFFLINE AWR
- Creates strong negative advantage bias (mean=-0.08)
- Only 22.6% positive advantages
- Result: 83.9% vs 89.1% with on-policy

**CORRECT**: For OFFLINE AWR, use on-policy (ckpt_101) re-rollout
```bash
# Offline: use same checkpoint for data and rollout
--optimal_checkpoint runs/pickcube_ppo/ckpt_101.pt

# Online: optimal policy is fine (iterative recomputation fixes bias)
--optimal_checkpoint runs/pickcube_ppo/ckpt_301.pt
```

### 8. IQL Q-Network Cannot Rank Actions

**FUNDAMENTAL LIMITATION**: IQL's Q-network learns V(s) well but cannot rank actions
- SNR problem: cross-state variance 500x > within-state variance
- Q ranking rho=0.01 (random), V quality r=0.96 (excellent)
- Larger networks don't help (256/512/1024 all ~0.01)

**WORKAROUND**: Use GAE with IQL's V for bootstrapping (rho=0.93)
```python
# Don't use: Q(s,a) - V(s) from IQL
# Do use: GAE(lambda=0.95) with IQL's V(s) for value bootstrap
```

### 9. NN Regression Destroys Ranking

**WRONG**: Training NN to regress advantage/Q targets
- GAE direct: rho=0.93
- NN(GAE targets): rho=0.02 (destroyed)
- Even with MC ground truth: 0.93 -> 0.02

**CORRECT**: Use GAE computed directly from trajectories, not through NN regression

### 10. Gamma Selection

**FOR SPARSE REWARD (50-step episodes)**:
- gamma=0.8 optimal for MC methods (99.1%)
- gamma=0.99 causes high variance in MC returns
- GAE prefers higher gamma (0.99 -> 92.1%, 0.8 -> 71.7%)

**RULE**: MC AWR use gamma=0.8, GAE PPO use gamma=0.95-0.99

### 11. Reward Scale: Doesn't Help Online PPO, But Helps Offline V Learning

**Online PPO**: reward_scale doesn't help
- reward_scale=20: doesn't help
- big critic (10x512): doesn't help
- both combined: WORST (81.2% vs 90.7% baseline)
- REASON: Online PPO iteratively improves V, so V is already good enough for GAE

**Offline/single-batch V learning**: reward_scale DOES help
- TD(0)+rs10 learns V with r=0.68 vs r=0.44 without scaling (Issue #13)
- GAE advantage ranking with different V sources (vs MC16 ground truth):

| V source for GAE | A r | A ρ (ranking) |
|-------------------|-----|---------------|
| V_mc1 (≈traditional) | 0.357 | 0.023 |
| V_td0 (rs=1) | 0.470 | 0.069 |
| V_td0 (rs=10) | 0.491 | 0.100 |
| V_mc_target (rs=10, NN upper bound) | 0.528 | 0.130 |
| V_mc16 (oracle) | 0.627 | 0.357 |

- TD+rs V gives 4-5x better advantage ranking than MC1-level V (ρ=0.10 vs 0.02)
- But absolute ranking quality is still low (ρ<0.15) due to fundamental SNR problem (Issue #8)

**RULE**: For online iterative RL, reward_scale is unnecessary. For offline/single-batch V learning, reward_scale improves V quality and downstream advantage estimation.

### 12. Batch Size for Offline AWR

**WRONG**: Small batch with many epochs
- Overfits, oscillates heavily

**CORRECT**: Large batch (25,600+) with few epochs (4)
```bash
--num_envs 128 --num_steps 200 --num_minibatches 32 --update_epochs 4
```

### 14. Experiment Setting Alignment

**WRONG**: Running experiments with different settings from the baseline, then comparing results
```bash
# Baseline: num_envs=100, total_timesteps=50000 (10 iter, batch=5000)
# New experiment: num_envs=512, total_timesteps=2000000 (78 iter, batch=25600)
# 40x more training data → meaningless comparison
```

**CORRECT**: Always use the same settings as `run_v2_all.sh` for fair comparison
```bash
COMMON_GAE="--num_envs 100 --num_steps 50 --num_minibatches 5 --update_epochs 100 --target_kl 100.0 --eval_freq 1 --total_timesteps 50000"
```
When testing a new method, run the baseline with identical settings in the same batch to avoid misalignment.

### 15. TD Pretrain Does Not Help Online GAE PPO

**WRONG**: Assuming TD-pretrained V initialization improves iterative GAE PPO
- TD-10 pretrain (first iter only): peak 78.5% vs baseline 92.1% (WORSE)
- TD-10 finetune (every iter, critic TD only): peak 79.6% (WORSE)
- TD-10 retrain (every iter, reset critic): peak 78.4% (WORSE)

**All TD-10 variants are ~12-14% worse than baseline at gamma=0.99.**

**WHY**: PPO with update_epochs=100 already trains the critic extensively each iteration (100 epochs of value loss on on-policy data). This is much more effective than TD-10's 200 epochs because:
1. PPO's value targets (GAE returns) are computed from on-policy data with telescoping error cancellation
2. TD-10's targets use bootstrap from a bad initial V, compounding errors
3. TD-10 pretrain at iter 1 uses a random critic for bootstrapping → bad targets → bad initialization → hurts early iterations

| Mode | Peak SR | vs Baseline |
|------|---------|-------------|
| Baseline GAE (gamma=0.99) | 92.1% | — |
| TD-10 pretrain (first only) | 78.5% | -13.6% |
| TD-10 finetune (every iter) | 79.6% | -12.5% |
| TD-10 retrain (every iter) | 78.4% | -13.7% |

**RULE**: For online iterative RL with sufficient update epochs, do NOT use TD pretrain. PPO's own critic learning is already strong enough. TD pretrain only appeared to help when using misaligned settings (78 iter vs 10 iter baseline).

### 13. TD/NN V Learning: Small Signal SNR is the Bottleneck

**PROBLEM**: TD(0) learns V with only r=0.44 correlation to MC16 V^π (gamma=0.8, sparse reward).
The V range gets systematically compressed (pred_std/MC_std ≈ 33%).

**ROOT CAUSE**: NN small-signal SNR, NOT TD bootstrap error.

Evidence:
- **n-step TD (1→50) doesn't help**: r stays at 0.44 regardless of n. 50-step ≈ MC, still 0.44
- **MC target supervised regression also only r=0.49**: NN can't fit V even with perfect targets
- **MC1 vs MC16 correlation is r=0.48**: MC16 target itself is noisy at this sample size
- **reward_scale=10 dramatically helps TD**: r jumps 0.44→0.68 (V values become larger, NN regression-to-mean effect shrinks relatively)
- **MC target + reward_scale=10 gives r=0.88**: NN CAN learn V when signal is large enough

| Method | V r | V ρ | pred_std/MC_std |
|--------|-----|-----|-----------------|
| TD(0) rs=1 | 0.435 | 0.377 | 33% |
| MC target rs=1 | 0.488 | 0.416 | 47% |
| TD(0) rs=10 | 0.677 | 0.632 | 80% |
| TD(0) rs=100 | 0.724 | 0.702 | 70% |
| MC target rs=10 | 0.880 | 0.816 | 85% |

**IMPLICATION**: For sparse reward with small V magnitudes, scale up rewards before TD/regression. The bottleneck is NN expressiveness at small signal levels, not the learning algorithm (TD vs MC vs n-step).

---

## Key Research Findings

1. **Online >> Offline**: +14% (99.2% vs 85.3%) - iterative re-rollout is critical
2. **AWR >> PPO for data efficiency**: AWR tolerates 200 epochs vs PPO's 4
3. **MC16 >> MC1**: +21% (99.1% vs 78.3%) - multi-sample MC reduces variance
4. **Optimal >> On-policy**: +4% (99.1% vs 96.7%) - helpful but not essential
5. **Learned Q cannot rank actions**: SNR problem is fundamental, use GAE instead

### 6. Role of Data in Iterative RL — Data Provides States, Not Learning Signal

In online iterative RL (AWR/PPO), each iteration: rollout → MC re-rollout → advantages → policy update.

**Transitions (s, a, r, s') only determine WHERE to evaluate (which states). The actual learning signal (advantages) comes entirely from MC re-rollout, NOT from the transitions' rewards.** The rewards in the collected transitions are never used — advantages are computed purely from re-rollout returns.

This explains the relative importance of each factor:

| Factor | Impact | Why |
|--------|--------|-----|
| MC samples (16 vs 1) | **+22%** | Re-rollout quality = advantage quality = learning signal quality |
| Optimal vs on-policy | +3% | Re-rollout policy quality, but iterative recomputation compensates |
| Online vs offline | +14% | Online refreshes states (on-policy); offline states become OOD |

In offline RL, data is forced to serve BOTH roles: (1) providing states, and (2) providing the learning signal (via TD → V estimate). But **TD from off-policy transitions can only learn V^{behavior}, not V\***. V correlation analysis confirmed: V_IQL(s) ≈ 0.09 (constant, ≈ behavior value), while V*(s) = 0.42 at the same states. IQL V doesn't track V* at all — when V* ranges 0.02→0.89, V_IQL only varies 0.05→0.10.

**Conclusion: offline data cannot provide accurate value estimates for states outside the behavior distribution. For effective RL, the critic must come from re-rollout (MC) or bootstrapping (GAE with good V), not from offline TD learning.**

## Recording Experiments

Always log to `experiments/log.md` with:
- Git commit hash
- Full command
- Per-iteration results table
- Run directory
- Notes on any issues encountered
