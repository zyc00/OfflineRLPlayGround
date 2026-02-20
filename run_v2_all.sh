#!/bin/bash
set -e
CKPT="runs/pickcube_ppo/ckpt_76_logstd-1.5.pt"
COMMON_AWR="--awr_beta 0.5 --num_envs 100 --num_steps 50 --num_minibatches 5 --update_epochs 200 --eval_freq 1 --total_timesteps 50000"
COMMON_GAE="--num_envs 100 --num_steps 50 --num_minibatches 5 --update_epochs 100 --target_kl 100.0 --eval_freq 1 --total_timesteps 50000"

echo "===== Starting all v2 experiments ====="
echo "Start time: $(date)"

echo ""; echo "===== [1/9] v2_mc16_optimal_det (gamma=0.8) ====="
python -u -m RL.mc_finetune_awr_parallel --checkpoint $CKPT --mc_samples 16 $COMMON_AWR --exp_name v2_mc16_optimal_det 2>&1 | tee logs/v2_mc16_optimal_det.log

echo ""; echo "===== [2/9] v2_mc1_optimal_det (gamma=0.8) ====="
python -u -m RL.mc_finetune_awr_parallel --checkpoint $CKPT --mc_samples 1 $COMMON_AWR --exp_name v2_mc1_optimal_det 2>&1 | tee logs/v2_mc1_optimal_det.log

echo ""; echo "===== [3/9] v2_mc16_onpolicy_det (gamma=0.8) ====="
python -u -m RL.mc_finetune_awr_onpolicy --checkpoint $CKPT --mc_samples 16 $COMMON_AWR --exp_name v2_mc16_onpolicy_det 2>&1 | tee logs/v2_mc16_onpolicy_det.log

echo ""; echo "===== [4/9] v2_mc1_onpolicy_det (gamma=0.8) ====="
python -u -m RL.mc_finetune_awr_onpolicy --checkpoint $CKPT --mc_samples 1 $COMMON_AWR --exp_name v2_mc1_onpolicy_det 2>&1 | tee logs/v2_mc1_onpolicy_det.log

echo ""; echo "===== [5/9] v2_gae_det (gamma=0.8) ====="
python -u -m RL.ppo_finetune --checkpoint $CKPT $COMMON_GAE --exp_name v2_gae_det 2>&1 | tee logs/v2_gae_det.log

echo ""; echo "===== [6/9] v2_mc16_optimal_g095_det ====="
python -u -m RL.mc_finetune_awr_parallel --checkpoint $CKPT --mc_samples 16 --gamma 0.95 $COMMON_AWR --exp_name v2_mc16_optimal_g095_det 2>&1 | tee logs/v2_mc16_optimal_g095_det.log

echo ""; echo "===== [7/9] v2_gae_g095_det ====="
python -u -m RL.ppo_finetune --checkpoint $CKPT --gamma 0.95 $COMMON_GAE --exp_name v2_gae_g095_det 2>&1 | tee logs/v2_gae_g095_det.log

echo ""; echo "===== [8/9] v2_mc16_optimal_g099_det ====="
python -u -m RL.mc_finetune_awr_parallel --checkpoint $CKPT --mc_samples 16 --gamma 0.99 $COMMON_AWR --exp_name v2_mc16_optimal_g099_det 2>&1 | tee logs/v2_mc16_optimal_g099_det.log

echo ""; echo "===== [9/9] v2_gae_g099_det ====="
python -u -m RL.ppo_finetune --checkpoint $CKPT --gamma 0.99 $COMMON_GAE --exp_name v2_gae_g099_det 2>&1 | tee logs/v2_gae_g099_det.log

echo ""
echo "===== All v2 experiments complete ====="
echo "End time: $(date)"
