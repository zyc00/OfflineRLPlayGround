#!/bin/bash
# Standard p4d run on PegInsertion with ~4k trajectories.
set -e

python -u -m DPPO.finetune_bridge_ablate \
  --pretrain_checkpoint runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt \
  --bridge_stage p1_reinforce_is_gae \
  --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos \
  --max_episode_steps 200 --n_envs 250 --n_steps 25 \
  --gamma 0.99 \
  --update_epochs 15 --minibatch_size 10000 \
  --gamma_denoising 1.0 --is_clip_ratio 0.02 \
  --use_ddim --ddim_steps 10 \
  --min_sampling_denoising_std 0.01 \
  --min_logprob_denoising_std 0.01 \
  --eval_freq 1 --eval_n_rounds 3 \
  --zero_qvel --reward_scale_running \
  --n_train_itr 20 \
  --seed 0 \
  --exp_name p4d_peg_ne400_ns25_ep10_gd1p0_c0p02_itr10_seed0 \
  --offpolicy_history_iters 3 \
  --offpolicy_mix_ratio 0.75 \
  --actor-return-warmup-itr 5 \
