#!/bin/bash
# PegInsertion Online Data-Efficient Experiments
# Adapted from run_v2_all.sh (PickCube) for PegInsertion
# PickCube: 100 envs, 50 steps, batch=5000, minibatch=1000
# PegInsertion: 100 envs, 100 steps (2x ep), batch=10000, minibatch=1000
#
# Weak checkpoint: ckpt_231 (det 68.4%)
# Optimal checkpoint: ckpt_611 (det 91.9%)

set -eo pipefail

CKPT="runs/peginsertion_ppo_ema99/ckpt_231.pt"
OPTIMAL="runs/peginsertion_ppo_ema99/ckpt_611.pt"
ENV="PegInsertionSide-v1"
COMMON_AWR="--env_id $ENV --max_episode_steps 100 --gamma 0.97 --awr_beta 0.5 --num_envs 100 --num_steps 100 --num_minibatches 10 --update_epochs 200 --num_eval_envs 512 --eval_freq 1 --total_timesteps 100000 --no-capture-video"
COMMON_GAE="--env_id $ENV --max_episode_steps 100 --gamma 0.97 --num_envs 100 --num_steps 100 --num_minibatches 10 --update_epochs 100 --target_kl 100.0 --num_eval_envs 512 --eval_freq 1 --total_timesteps 100000 --no-capture-video"

mkdir -p logs

echo "===== Starting PegInsertion online experiments ====="
echo "Start time: $(date)"

echo ""; echo "===== [1/5] MC16 optimal re-rollout AWR ====="
python -u -m RL.mc_finetune_awr_parallel \
    --checkpoint $CKPT --optimal_checkpoint $OPTIMAL \
    --mc_samples 16 $COMMON_AWR \
    --exp_name peg_mc16_optimal \
    2>&1 | tee logs/peg_mc16_optimal.log

echo ""; echo "===== [2/5] MC1 optimal re-rollout AWR ====="
python -u -m RL.mc_finetune_awr_parallel \
    --checkpoint $CKPT --optimal_checkpoint $OPTIMAL \
    --mc_samples 1 $COMMON_AWR \
    --exp_name peg_mc1_optimal \
    2>&1 | tee logs/peg_mc1_optimal.log

echo ""; echo "===== [3/5] MC16 on-policy AWR ====="
python -u -m RL.mc_finetune_awr_onpolicy \
    --checkpoint $CKPT \
    --mc_samples 16 $COMMON_AWR \
    --exp_name peg_mc16_onpolicy \
    2>&1 | tee logs/peg_mc16_onpolicy.log

echo ""; echo "===== [4/5] MC1 on-policy AWR ====="
python -u -m RL.mc_finetune_awr_onpolicy \
    --checkpoint $CKPT \
    --mc_samples 1 $COMMON_AWR \
    --exp_name peg_mc1_onpolicy \
    2>&1 | tee logs/peg_mc1_onpolicy.log

echo ""; echo "===== [5/5] GAE PPO baseline ====="
python -u -m RL.ppo_finetune \
    --checkpoint $CKPT \
    $COMMON_GAE \
    --exp_name peg_gae_baseline \
    2>&1 | tee logs/peg_gae_baseline.log

echo ""
echo "===== All PegInsertion online experiments complete ====="
echo "End time: $(date)"
