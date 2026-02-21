#!/bin/bash
set -e
CKPT="runs/pickcube_ppo/ckpt_76_logstd-1.5.pt"
OPTIMAL="runs/pickcube_ppo/ckpt_301.pt"
# Match previous offline settings: large batch (25600), few epochs (4), beta=1.0
COMMON="--awr_beta 1.0 --num_envs 128 --num_steps 200 --num_minibatches 32 --update_epochs 4 --num_iterations 100 --eval_freq 5"

echo "===== Starting all v2 offline experiments ====="
echo "Start time: $(date)"

echo ""; echo "===== [1/5] v2_offline_mc16_optimal ====="
python -u -m RL.mc_finetune_awr_offline --checkpoint $CKPT --optimal_checkpoint $OPTIMAL --mc_samples 16 $COMMON --exp_name v2_offline_mc16_optimal_det 2>&1 | tee logs/v2_offline_mc16_optimal_det.log

echo ""; echo "===== [2/5] v2_offline_mc1_optimal ====="
python -u -m RL.mc_finetune_awr_offline --checkpoint $CKPT --optimal_checkpoint $OPTIMAL --mc_samples 1 $COMMON --exp_name v2_offline_mc1_optimal_det 2>&1 | tee logs/v2_offline_mc1_optimal_det.log

echo ""; echo "===== [3/5] v2_offline_mc16_onpolicy ====="
python -u -m RL.mc_finetune_awr_offline --checkpoint $CKPT --optimal_checkpoint $CKPT --mc_samples 16 $COMMON --exp_name v2_offline_mc16_onpolicy_det 2>&1 | tee logs/v2_offline_mc16_onpolicy_det.log

echo ""; echo "===== [4/5] v2_offline_mc1_onpolicy ====="
python -u -m RL.mc_finetune_awr_offline --checkpoint $CKPT --optimal_checkpoint $CKPT --mc_samples 1 $COMMON --exp_name v2_offline_mc1_onpolicy_det 2>&1 | tee logs/v2_offline_mc1_onpolicy_det.log

echo ""; echo "===== [5/6] v2_offline_iql ====="
python -u -m RL.iql_awr_offline --checkpoint $CKPT --awr_beta 1.0 --num_envs 128 --num_steps 200 --num_minibatches 32 --update_epochs 4 --num_iterations 100 --eval_freq 5 --exp_name v2_offline_iql_det 2>&1 | tee logs/v2_offline_iql_det.log

echo ""; echo "===== [6/6] v2_offline_iql_mc_aug ====="
python -u -m RL.iql_awr_offline_augmented --checkpoint $CKPT --optimal_checkpoint $OPTIMAL --augment_with_mc --mc_samples 16 --mc_num_envs 100 --mc_num_steps 50 --awr_beta 1.0 --num_envs 128 --num_steps 200 --num_minibatches 32 --update_epochs 4 --num_iterations 100 --eval_freq 5 --exp_name v2_offline_iql_mc_aug_det 2>&1 | tee logs/v2_offline_iql_mc_aug_det.log

echo ""
echo "===== All v2 offline experiments complete ====="
echo "End time: $(date)"
