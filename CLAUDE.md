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

### 11. Regression Tricks Don't Help Online PPO

**WRONG**: Applying offline regression tricks to online PPO
- reward_scale=20: doesn't help
- big critic (10x512): doesn't help
- both combined: WORST (81.2% vs 90.7% baseline)

**REASON**: GAE only needs V(s), not Q(s,a) - error cancellation via telescoping makes SNR tricks unnecessary

### 12. Batch Size for Offline AWR

**WRONG**: Small batch with many epochs
- Overfits, oscillates heavily

**CORRECT**: Large batch (25,600+) with few epochs (4)
```bash
--num_envs 128 --num_steps 200 --num_minibatches 32 --update_epochs 4
```

---

## Key Research Findings

1. **Online >> Offline**: +14% (99.2% vs 85.3%) - iterative re-rollout is critical
2. **AWR >> PPO for data efficiency**: AWR tolerates 200 epochs vs PPO's 4
3. **MC16 >> MC1**: +21% (99.1% vs 78.3%) - multi-sample MC reduces variance
4. **Optimal >> On-policy**: +4% (99.1% vs 96.7%) - helpful but not essential
5. **Learned Q cannot rank actions**: SNR problem is fundamental, use GAE instead

## Recording Experiments

Always log to `experiments/log.md` with:
- Git commit hash
- Full command
- Per-iteration results table
- Run directory
- Notes on any issues encountered
