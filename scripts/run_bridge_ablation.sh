#!/bin/bash
set -euo pipefail

PRETRAIN=${PRETRAIN:-"runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt"}
SEEDS=${SEEDS:-"0 100 200"}
STAGES=${STAGES:-"s0_reinforce_is_matched s1_continuous_return s2_gamma_denoising s3_value_baseline_td1 s3b_value_baseline_td1_warmup s4_gae s4b_gae_warmup s5_norm_adv s6_critic_warmup s7_full_dppo_equiv"}

N_ENVS=${N_ENVS:-500}
N_STEPS=${N_STEPS:-25}
ITERS=${ITERS:-20}
EPOCHS=${EPOCHS:-10}
MB=${MB:-2500}
ACTOR_LR=${ACTOR_LR:-3e-6}
CRITIC_LR=${CRITIC_LR:-1e-3}
SIM_BACKEND=${SIM_BACKEND:-gpu}

mkdir -p logs/bridge_ablation

for stage in $STAGES; do
  for seed in $SEEDS; do
    exp="bridge_${stage}_seed${seed}"
    log="logs/bridge_ablation/${exp}.log"

    echo "===================================================="
    echo "Running stage=${stage} seed=${seed}"
    echo "log=${log}"

    python -u -m DPPO.finetune_bridge_ablate \
      --pretrain_checkpoint "$PRETRAIN" \
      --bridge_stage "$stage" \
      --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos \
      --max_episode_steps 200 --n_envs "$N_ENVS" --n_steps "$N_STEPS" --sim_backend "$SIM_BACKEND" \
      --gamma 0.99 --update_epochs "$EPOCHS" --minibatch_size "$MB" \
      --actor_lr "$ACTOR_LR" --critic_lr "$CRITIC_LR" \
      --use_ddim --ddim_steps 10 \
      --min_sampling_denoising_std 0.01 --min_logprob_denoising_std 0.01 \
      --eval_freq 1 --eval_n_rounds 3 \
      --zero_qvel --reward_scale_running \
      --n_train_itr "$ITERS" --seed "$seed" --exp_name "$exp" \
      2>&1 | tee "$log"
  done
done

echo "All runs finished."
