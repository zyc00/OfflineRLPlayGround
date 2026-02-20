#!/bin/bash
set -e
CKPT="runs/pickcube_ppo/ckpt_76_logstd-1.5.pt"
# Base settings match v2_offline experiments
COMMON="--awr_beta 1.0 --num_envs 128 --num_steps 200 --num_minibatches 32 --update_epochs 4 --num_iterations 100 --eval_freq 5"

echo "===== Starting IQL tuning experiments ====="
echo "Start time: $(date)"

# Baseline: tau=0.7, nstep=1 (already have results from v2_offline_iql_det)
# Peak=91.6% (i15), Final=67.2%

echo ""; echo "===== [1/4] iql_tau09 (tau=0.9, nstep=1) ====="
python -u -m RL.iql_awr_offline --checkpoint $CKPT --iql_expectile_tau 0.9 $COMMON --exp_name iql_tau09_det 2>&1 | tee logs/iql_tau09_det.log

echo ""; echo "===== [2/4] iql_nstep5 (tau=0.7, nstep=5) ====="
python -u -m RL.iql_awr_offline --checkpoint $CKPT --iql_nstep 5 $COMMON --exp_name iql_nstep5_det 2>&1 | tee logs/iql_nstep5_det.log

echo ""; echo "===== [3/4] iql_tau09_nstep5 (tau=0.9, nstep=5) ====="
python -u -m RL.iql_awr_offline --checkpoint $CKPT --iql_expectile_tau 0.9 --iql_nstep 5 $COMMON --exp_name iql_tau09_nstep5_det 2>&1 | tee logs/iql_tau09_nstep5_det.log

echo ""; echo "===== [4/4] iql_tau09_nstep5_lr1e4 (tau=0.9, nstep=5, lr=1e-4) ====="
python -u -m RL.iql_awr_offline --checkpoint $CKPT --iql_expectile_tau 0.9 --iql_nstep 5 --learning_rate 1e-4 $COMMON --exp_name iql_tau09_nstep5_lr1e4_det 2>&1 | tee logs/iql_tau09_nstep5_lr1e4_det.log

echo ""
echo "===== All IQL tuning experiments complete ====="
echo "End time: $(date)"
