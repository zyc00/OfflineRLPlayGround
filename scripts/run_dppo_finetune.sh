#!/bin/bash
# DPPO finetune (GAE + learned critic + PPO)
# 500 envs, 10ep, 20 itr → 91.6% peak SR
set -e

python -u -m DPPO.finetune \
  --pretrain_checkpoint runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt \
  --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos \
  --max_episode_steps 200 --n_envs 500 --n_steps 25 \
  --gamma 0.99 --gae_lambda 0.95 \
  --update_epochs 10 --minibatch_size 2500 \
  --actor_lr 3e-6 --critic_lr 1e-3 \
  --n_critic_warmup_itr 5 \
  --ft_denoising_steps 10 --use_ddim --ddim_steps 10 \
  --min_sampling_denoising_std 0.01 --min_logprob_denoising_std 0.01 \
  --clip_ploss_coef 0.02 --target_kl 1.0 \
  --eval_freq 1 --eval_n_rounds 3 \
  --zero_qvel --reward_scale_running \
  --n_train_itr 20 \
  --exp_name dppo_ft_peg_gae_g099
