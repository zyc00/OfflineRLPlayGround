#!/bin/bash
set -euo pipefail

# Network-structure ablation under fixed DPPO finetune settings.
# Keeps finetune hyperparams aligned with scripts/run_dppo_finetune.sh.

SEEDS=${SEEDS:-"0 100 200"}
MODELS=${MODELS:-"unet_zeroqvel_500k mlp_il_base_200k_v2"}
SIM_BACKEND=${SIM_BACKEND:-gpu}

N_ENVS=${N_ENVS:-500}
N_STEPS=${N_STEPS:-25}
ITERS=${ITERS:-20}

mkdir -p logs/net_struct_ablation

ckpt_for() {
  case "$1" in
    unet_zeroqvel_500k)
      echo "runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt"
      ;;
    unet_policy_200k)
      echo "runs/dppo_pretrain/dppo_pretrain_peg_policy_demos_200k/best.pt"
      ;;
    mlp_il_base_200k_v2)
      echo "runs/dppo_pretrain/dppo_pretrain_peg_il_base_demos_200k_v2/best.pt"
      ;;
    mlp_uniform100)
      echo "runs/dppo_pretrain/dppo_pretrain_peg_uniform100/best.pt"
      ;;
    *)
      echo "Unknown model key: $1" >&2
      exit 1
      ;;
  esac
}

for model in $MODELS; do
  ckpt=$(ckpt_for "$model")
  for seed in $SEEDS; do
    exp="netab_${model}_seed${seed}"
    log="logs/net_struct_ablation/${exp}.log"

    echo "===================================================="
    echo "Running model=${model} seed=${seed}"
    echo "ckpt=${ckpt}"
    echo "log=${log}"

    python -u -m DPPO.finetune \
      --pretrain_checkpoint "$ckpt" \
      --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos \
      --max_episode_steps 200 --n_envs "$N_ENVS" --n_steps "$N_STEPS" --sim_backend "$SIM_BACKEND" \
      --gamma 0.99 --gae_lambda 0.95 \
      --update_epochs 10 --minibatch_size 2500 \
      --actor_lr 3e-6 --critic_lr 1e-3 \
      --n_critic_warmup_itr 5 \
      --ft_denoising_steps 10 --use_ddim --ddim_steps 10 \
      --min_sampling_denoising_std 0.01 --min_logprob_denoising_std 0.01 \
      --clip_ploss_coef 0.02 --target_kl 1.0 \
      --eval_freq 1 --eval_n_rounds 3 \
      --reward_scale_running \
      --n_train_itr "$ITERS" --seed "$seed" --exp_name "$exp" \
      2>&1 | tee "$log"
  done
done

echo "All network-structure ablation runs finished."
