#!/bin/bash
# IQL debug experiments (v2 settings: 128 envs × 200 steps = 25600 batch)
set -e
cd /home/jigu/projects/OfflineRLPlayGround

CKPT="runs/pickcube_ppo/ckpt_76_logstd-1.5.pt"
V2="--awr_beta 1.0 --num_envs 128 --num_steps 200 --num_minibatches 32 --update_epochs 4 --num_iterations 100 --eval_freq 5"

mkdir -p logs cache

echo "===== IQL Debug Experiments (v2 settings) ====="
echo "Start: $(date)"

# --- Exp 1a: mc_only fit analysis (caches MC data, subsamples for IQL training) ---
echo ""; echo "===== [1/4] Exp 1a: IQL fit analysis (mc_only) ====="
python -u -m RL.iql_fit_analysis \
    --iql_data_mode mc_only \
    --num_steps 200 \
    --iql_max_transitions 100000 \
    --cache_path cache/iql_fit_mc_data_v2.pt \
    --output runs/iql_fit_mc_only_v2.png \
    2>&1 | tee logs/iql_fit_mc_only_v2.log

# --- Exp 2a: IQL with reward_scale=10 ---
echo ""; echo "===== [2/4] Exp 2a: IQL reward_scale=10 ====="
python -u -m RL.iql_awr_offline \
    --checkpoint $CKPT \
    --reward_scale 10.0 \
    $V2 \
    --exp_name v2_offline_iql_rs10 \
    2>&1 | tee logs/v2_offline_iql_rs10.log

# --- Exp 3b: nstep=5 + reward_scale=10 ---
echo ""; echo "===== [3/4] Exp 3b: IQL nstep=5 + reward_scale=10 ====="
python -u -m RL.iql_awr_offline \
    --checkpoint $CKPT \
    --iql_nstep 5 --reward_scale 10.0 \
    $V2 \
    --exp_name v2_offline_iql_nstep5_rs10 \
    2>&1 | tee logs/v2_offline_iql_nstep5_rs10.log

# --- Exp 1b: offline_only fit analysis (reuses cached MC ground truth) ---
echo ""; echo "===== [4/4] Exp 1b: IQL fit analysis (offline_only, reuses cache) ====="
python -u -m RL.iql_fit_analysis \
    --iql_data_mode offline_only \
    --num_steps 200 \
    --cache_path cache/iql_fit_mc_data_v2.pt \
    --output runs/iql_fit_offline_only_v2.png \
    2>&1 | tee logs/iql_fit_offline_only_v2.log

echo ""
echo "===== All debug experiments complete: $(date) ====="
