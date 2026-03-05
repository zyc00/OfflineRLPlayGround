#!/bin/bash
# Filtered BC finetuning (binary_pos: only train on successful trajectories)
# Best setting so far: ep5, lr=3e-6, 20 iters → 85% peak
set -e

PRETRAIN="runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt"

# Tunable parameters
EPOCHS=10              # sweet spot=5, tested 2/3/5/10/20
LR=3e-6              # default
ITERS=20             # number of rollout iterations
WEIGHT_MODE=binary_pos  # binary_pos | return_pos | uniform | posneg_return
EXP_NAME="wbc_${WEIGHT_MODE}_ep${EPOCHS}"

python -u -m DPPO.finetune_bc \
  --pretrain_checkpoint $PRETRAIN \
  --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos \
  --max_episode_steps 200 --n_envs 500 --n_steps 25 \
  --gamma 0.99 --update_epochs $EPOCHS --minibatch_size 2500 \
  --actor_lr $LR --weight_mode $WEIGHT_MODE \
  --use_ddim --ddim_steps 10 \
  --min_sampling_denoising_std 0.01 \
  --eval_freq 1 --eval_n_rounds 3 \
  --zero_qvel --reward_scale_running \
  --n_train_itr $ITERS --exp_name $EXP_NAME --no-norm_returns
