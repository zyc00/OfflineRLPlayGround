#!/bin/bash
# PegInsertion Offline Experiments
# Weak checkpoint: ckpt_231 (det 68.4%, stoch 86.6%)
# Optimal checkpoint: ckpt_611 (det 91.9%, stoch 95.9%)
# From EMA 0.99 training run

set -eo pipefail

CKPT="runs/peginsertion_ppo_ema99/ckpt_231.pt"
OPTIMAL="runs/peginsertion_ppo_ema99/ckpt_611.pt"
ENV="PegInsertionSide-v1"
COMMON="--env_id $ENV --max_episode_steps 100 --awr_beta 1.0 --gamma 0.97 --num_envs 128 --num_steps 200 --num_minibatches 32 --update_epochs 4 --num_iterations 100 --eval_freq 5 --num_eval_envs 512 --no-capture-video"

mkdir -p logs cache

# [1/3] MC16 Optimal re-rollout
echo "=== [1/3] MC16 Optimal ==="
python -u -m RL.mc_finetune_awr_offline \
    --checkpoint $CKPT \
    --optimal_checkpoint $OPTIMAL \
    --mc_samples 16 \
    $COMMON \
    --cache_path cache/peg_mc16_optimal.pt \
    --exp_name peg_offline_mc16_optimal \
    2>&1 | tee logs/peg_offline_mc16_optimal.log

# [2/3] MC16 On-policy re-rollout
echo "=== [2/3] MC16 On-policy ==="
python -u -m RL.mc_finetune_awr_offline \
    --checkpoint $CKPT \
    --optimal_checkpoint $CKPT \
    --mc_samples 16 \
    $COMMON \
    --cache_path cache/peg_mc16_onpolicy.pt \
    --exp_name peg_offline_mc16_onpolicy \
    2>&1 | tee logs/peg_offline_mc16_onpolicy.log

# [3/3] IQL
# IQL data checkpoints: various quality levels from EMA 0.99 run
echo "=== [3/3] IQL ==="
python -u -m RL.iql_awr_offline \
    --checkpoint $CKPT \
    --iql_data_checkpoints \
        runs/peginsertion_ppo_ema99/ckpt_1.pt \
        runs/peginsertion_ppo_ema99/ckpt_41.pt \
        runs/peginsertion_ppo_ema99/ckpt_231.pt \
        runs/peginsertion_ppo_ema99/ckpt_401.pt \
        runs/peginsertion_ppo_ema99/ckpt_611.pt \
    --env_id $ENV --max_episode_steps 100 \
    --awr_beta 1.0 --gamma 0.97 \
    --num_envs 128 --num_steps 200 \
    --num_minibatches 32 --update_epochs 4 \
    --num_iterations 100 --eval_freq 5 --num_eval_envs 512 \
    --no-capture-video \
    --exp_name peg_offline_iql \
    2>&1 | tee logs/peg_offline_iql.log

echo "=== All offline experiments done ==="
