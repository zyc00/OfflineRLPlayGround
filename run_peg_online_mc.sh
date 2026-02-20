#!/bin/bash
# PegInsertion Online: On-policy MC + PPO with gamma=0.99
# MC advantage = GAE with lambda=1.0 (pure on-policy MC returns)
# Weak checkpoint: ckpt_231 (det 68.4%)

set -eo pipefail

CKPT="runs/peginsertion_ppo_ema99/ckpt_231.pt"
ENV="PegInsertionSide-v1"

mkdir -p logs

echo "=== On-policy MC + PPO (gamma=0.99) ==="
python -u -m RL.ppo_finetune \
    --checkpoint $CKPT \
    --env_id $ENV \
    --max_episode_steps 100 \
    --advantage_mode mc \
    --mc_samples 1 \
    --gamma 0.99 \
    --num_envs 512 \
    --num_eval_envs 512 \
    --num_steps 100 \
    --num_minibatches 32 \
    --update_epochs 4 \
    --total_timesteps 5000000 \
    --eval_freq 5 \
    --reward_mode sparse \
    --reset_critic \
    --no-capture-video \
    --exp_name peg_online_mc1_gamma99 \
    2>&1 | tee logs/peg_online_mc1_gamma99.log

echo "=== Done ==="
