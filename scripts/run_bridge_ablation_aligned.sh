#!/bin/bash
set -euo pipefail

# Aligned bridge ablation:
# - Start from reinforce script config (for reference)
# - Bridge with DPPO-aligned budget/optimizer settings
# - End with exact DPPO finetune reference (anchor)

PRETRAIN=${PRETRAIN:-"runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt"}
SEEDS=${SEEDS:-"0 100 200"}
STAGES=${STAGES:-"ref_reinforce_script bridge_reinforce_dppo_budget bridge_td_warmup bridge_ppo_full ref_dppo_script"}
SIM_BACKEND=${SIM_BACKEND:-gpu}

mkdir -p logs/bridge_ablation_aligned

run_ref_reinforce() {
  local seed="$1"
  local exp="aligned_ref_reinforce_seed${seed}"
  local log="logs/bridge_ablation_aligned/${exp}.log"
  python -u -m DPPO.finetune_reinforce \
    --pretrain_checkpoint "$PRETRAIN" \
    --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos \
    --max_episode_steps 200 --n_envs 500 --n_steps 25 --sim_backend "$SIM_BACKEND" \
    --gamma 0.99 --update_epochs 5 --minibatch_size 2500 \
    --actor_lr 1e-5 \
    --use_ddim --ddim_steps 10 \
    --min_sampling_denoising_std 0.01 --min_logprob_denoising_std 0.01 \
    --eval_freq 1 --eval_n_rounds 3 \
    --zero_qvel --reward_scale_running \
    --n_train_itr 40 --seed "$seed" --exp_name "$exp" \
    --binarize_returns --no-norm_returns --use_is --is_clip_ratio 0.2 \
    2>&1 | tee "$log"
}

run_bridge_stage() {
  local stage="$1"
  local seed="$2"
  local exp="aligned_${stage}_seed${seed}"
  local log="logs/bridge_ablation_aligned/${exp}.log"
  python -u -m DPPO.finetune_bridge_ablate \
    --pretrain_checkpoint "$PRETRAIN" \
    --bridge_stage "$stage" \
    --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos \
    --max_episode_steps 200 --n_envs 500 --n_steps 25 --sim_backend "$SIM_BACKEND" \
    --gamma 0.99 --update_epochs 10 --minibatch_size 2500 \
    --actor_lr 3e-6 --critic_lr 1e-3 \
    --use_ddim --ddim_steps 10 \
    --min_sampling_denoising_std 0.01 --min_logprob_denoising_std 0.01 \
    --eval_freq 1 --eval_n_rounds 3 \
    --zero_qvel --reward_scale_running \
    --n_train_itr 20 --seed "$seed" --exp_name "$exp" \
    2>&1 | tee "$log"
}

run_ref_dppo() {
  local seed="$1"
  local exp="aligned_ref_dppo_seed${seed}"
  local log="logs/bridge_ablation_aligned/${exp}.log"
  python -u -m DPPO.finetune \
    --pretrain_checkpoint "$PRETRAIN" \
    --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos \
    --max_episode_steps 200 --n_envs 500 --n_steps 25 --sim_backend "$SIM_BACKEND" \
    --gamma 0.99 --gae_lambda 0.95 \
    --update_epochs 10 --minibatch_size 2500 \
    --actor_lr 3e-6 --critic_lr 1e-3 \
    --n_critic_warmup_itr 5 \
    --ft_denoising_steps 10 --use_ddim --ddim_steps 10 \
    --min_sampling_denoising_std 0.01 --min_logprob_denoising_std 0.01 \
    --clip_ploss_coef 0.02 --target_kl 1.0 \
    --eval_freq 1 --eval_n_rounds 3 \
    --zero_qvel --reward_scale_running \
    --n_train_itr 20 --seed "$seed" --exp_name "$exp" \
    2>&1 | tee "$log"
}

for stage in $STAGES; do
  for seed in $SEEDS; do
    echo "===================================================="
    echo "Running stage=${stage} seed=${seed}"
    case "$stage" in
      ref_reinforce_script)
        run_ref_reinforce "$seed"
        ;;
      bridge_reinforce_dppo_budget)
        # No critic, but move to DPPO budget and return processing.
        run_bridge_stage "s2_gamma_denoising" "$seed"
        ;;
      bridge_td_warmup)
        # Add value baseline with mandatory warmup=5.
        run_bridge_stage "s3_value_baseline_td1" "$seed"
        ;;
      bridge_ppo_full)
        # Full PPO-like bridge objective with GAE/norm_adv/warmup.
        run_bridge_stage "s7_full_dppo_equiv" "$seed"
        ;;
      ref_dppo_script)
        run_ref_dppo "$seed"
        ;;
      *)
        echo "Unknown stage: $stage" >&2
        exit 1
        ;;
    esac
  done
done

echo "All aligned bridge runs finished."
