#!/bin/bash
# REINFORCE finetuning with optional IS/clipping (PPO-like ratio update)
set -e

PRETRAIN="runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt"

# Tunable parameters
EPOCHS=5
LR=1e-5
ITERS=40
MINIBATCH=2500
IS_CLIP=0.2
BIN_RET=true
NORM_RET=false
USE_IS=true
EXP_NAME="reinforce_is_clip${IS_CLIP}_ep${EPOCHS}_itr${ITERS}"

CMD=(
  python -u -m DPPO.finetune_reinforce
  --pretrain_checkpoint "$PRETRAIN"
  --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos
  --max_episode_steps 200 --n_envs 500 --n_steps 25
  --gamma 0.99 --update_epochs "$EPOCHS" --minibatch_size "$MINIBATCH"
  --actor_lr "$LR"
  --use_ddim --ddim_steps 10
  --min_sampling_denoising_std 0.01
  --min_logprob_denoising_std 0.01
  --eval_freq 1 --eval_n_rounds 3
  --zero_qvel --reward_scale_running
  --n_train_itr "$ITERS" --exp_name "$EXP_NAME"
)

if [ "$BIN_RET" = true ]; then
  CMD+=(--binarize_returns)
else
  CMD+=(--no-binarize_returns)
fi

if [ "$NORM_RET" = true ]; then
  CMD+=(--norm_returns)
else
  CMD+=(--no-norm_returns)
fi

if [ "$USE_IS" = true ]; then
  CMD+=(--use_is --is_clip_ratio "$IS_CLIP")
else
  CMD+=(--no-use_is)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
