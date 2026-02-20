#!/bin/bash
set -e
CKPT="runs/pickcube_ppo/ckpt_76_logstd-1.5.pt"
OPTIMAL="runs/pickcube_ppo/ckpt_301.pt"
# v3: 4x data (128 envs × 800 steps = 102,400), 4x eval (512 eval envs)
# Keep minibatch_size=800 same as v2 → num_minibatches=128
COMMON="--awr_beta 1.0 --num_envs 128 --num_steps 800 --num_minibatches 128 --update_epochs 4 --num_iterations 100 --eval_freq 5 --num_eval_envs 512"

echo "===== Starting v3 offline experiments (4x data, 4x eval) ====="
echo "Start time: $(date)"

echo ""; echo "===== [1/6] v3_offline_mc16_optimal ====="
python -u -m RL.mc_finetune_awr_offline --checkpoint $CKPT --optimal_checkpoint $OPTIMAL --mc_samples 16 $COMMON --exp_name v3_offline_mc16_optimal 2>&1 | tee logs/v3_offline_mc16_optimal.log

echo ""; echo "===== [2/6] v3_offline_mc1_optimal ====="
python -u -m RL.mc_finetune_awr_offline --checkpoint $CKPT --optimal_checkpoint $OPTIMAL --mc_samples 1 $COMMON --exp_name v3_offline_mc1_optimal 2>&1 | tee logs/v3_offline_mc1_optimal.log

echo ""; echo "===== [3/6] v3_offline_mc16_onpolicy ====="
python -u -m RL.mc_finetune_awr_offline --checkpoint $CKPT --optimal_checkpoint $CKPT --mc_samples 16 $COMMON --exp_name v3_offline_mc16_onpolicy 2>&1 | tee logs/v3_offline_mc16_onpolicy.log

echo ""; echo "===== [4/6] v3_offline_mc1_onpolicy ====="
python -u -m RL.mc_finetune_awr_offline --checkpoint $CKPT --optimal_checkpoint $CKPT --mc_samples 1 $COMMON --exp_name v3_offline_mc1_onpolicy 2>&1 | tee logs/v3_offline_mc1_onpolicy.log

echo ""; echo "===== [5/6] v3_offline_iql ====="
python -u -m RL.iql_awr_offline --checkpoint $CKPT $COMMON --exp_name v3_offline_iql 2>&1 | tee logs/v3_offline_iql.log

echo ""; echo "===== [6/6] v3_offline_iql_nstep5 ====="
python -u -m RL.iql_awr_offline --checkpoint $CKPT --iql_nstep 5 $COMMON --exp_name v3_offline_iql_nstep5 2>&1 | tee logs/v3_offline_iql_nstep5.log

echo ""
echo "===== All v3 offline experiments complete ====="
echo "End time: $(date)"
