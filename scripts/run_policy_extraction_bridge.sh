#!/bin/bash
set -euo pipefail

# Policy-extraction bridge:
# p0 (PPO-clip full, finetune.py-like) -> p4 (IS-REINFORCE final style)
# Keep rollout and optimizer envelope fixed to run_dppo_finetune.sh unless overridden.

PRETRAIN=${PRETRAIN:-"runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt"}
SEEDS=${SEEDS:-"0 100 200"}
STAGES=${STAGES:-"p0_ppo_clip_full p1_reinforce_is_gae p2_reinforce_is_td p3_reinforce_is_nobaseline p4_reinforce_is_final"}
SIM_BACKEND=${SIM_BACKEND:-gpu}

N_ENVS=${N_ENVS:-500}
N_STEPS=${N_STEPS:-25}
ITERS=${ITERS:-20}
EPOCHS=${EPOCHS:-10}
MB=${MB:-2500}
ACTOR_LR=${ACTOR_LR:-3e-6}
CRITIC_LR=${CRITIC_LR:-1e-3}

mkdir -p logs/policy_extraction_bridge

for stage in $STAGES; do
  for seed in $SEEDS; do
    exp="peb_${stage}_seed${seed}"
    log="logs/policy_extraction_bridge/${exp}.log"

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
      --clip_ploss_coef 0.02 --clip_ploss_coef_base 1e-3 --clip_ploss_coef_rate 3.0 --target_kl 1.0 \
      --use_ddim --ddim_steps 10 \
      --min_sampling_denoising_std 0.01 --min_logprob_denoising_std 0.01 \
      --eval_freq 1 --eval_n_rounds 3 \
      --zero_qvel --reward_scale_running \
      --n_train_itr "$ITERS" --seed "$seed" --exp_name "$exp" \
      2>&1 | tee "$log"
  done
done

echo "Policy-extraction bridge runs finished."
