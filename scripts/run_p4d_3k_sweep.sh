#!/bin/bash
set -euo pipefail

PRETRAIN=${PRETRAIN:-"runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt"}
SIM_BACKEND=${SIM_BACKEND:-gpu}
SEEDS=${SEEDS:-"0"}
TARGET_TRAJ=${TARGET_TRAJ:-3000}
OUT_DIR=${OUT_DIR:-"logs/p4d_3k_sweep"}

# Columns:
# n_envs n_steps gamma update_epochs minibatch_size gamma_denoising is_clip_ratio
DEFAULT_CANDIDATES=$(cat <<'EOF'
300 25 0.99 10 2500 1.0 0.02
350 25 0.99 10 2500 1.0 0.02
400 25 0.99 10 2500 1.0 0.02
500 25 0.99 10 2500 1.0 0.02
500 25 0.995 10 2500 1.0 0.02
500 25 0.999 10 2500 1.0 0.02
500 25 0.99 8 2500 1.0 0.02
500 25 0.99 10 5000 1.0 0.02
500 25 0.99 10 10000 1.0 0.02
500 25 0.99 10 2500 0.99 0.02
500 25 0.99 10 2500 1.0 0.015
500 25 0.99 10 2500 1.0 0.03
300 50 0.99 10 2500 1.0 0.02
250 25 0.99 10 2500 1.0 0.02
EOF
)

mkdir -p "$OUT_DIR"

manifest="$OUT_DIR/manifest.tsv"
echo -e "exp_name\tseed\tn_envs\tn_steps\tgamma\tupdate_epochs\tminibatch_size\tgamma_denoising\tis_clip_ratio\ttraj_per_iter\tn_train_itr\ttotal_traj" > "$manifest"

if [[ -n "${CANDIDATES_FILE:-}" ]]; then
  CANDIDATE_SOURCE=$(grep -v '^\s*#' "$CANDIDATES_FILE" | sed '/^\s*$/d')
else
  CANDIDATE_SOURCE=$(printf "%s\n" "$DEFAULT_CANDIDATES")
fi

sanitize_float() {
  printf "%s" "$1" | tr '.' 'p'
}

calc_iters() {
  python - "$1" "$2" "$3" <<'PY'
import sys
n_envs = int(sys.argv[1])
n_steps = int(sys.argv[2])
target = int(sys.argv[3])
traj_per_iter = max(1, round(n_envs * n_steps / 25.0))
iters = max(1, round(target / traj_per_iter))
print(traj_per_iter, iters, traj_per_iter * iters)
PY
}

while read -r N_ENVS N_STEPS GAMMA EPOCHS MB GD CLIP; do
  [[ -z "${N_ENVS:-}" ]] && continue

  read -r TRAJ_PER_ITER ITERS TOTAL_TRAJ <<<"$(calc_iters "$N_ENVS" "$N_STEPS" "$TARGET_TRAJ")"

  for SEED in $SEEDS; do
    EXP="p4d3k_ne${N_ENVS}_ns${N_STEPS}_g$(sanitize_float "$GAMMA")_ep${EPOCHS}_mb${MB}_gd$(sanitize_float "$GD")_clip$(sanitize_float "$CLIP")_itr${ITERS}_seed${SEED}"
    LOG="$OUT_DIR/${EXP}.log"

    echo "===================================================="
    echo "Running $EXP"
    echo "  approx traj/iter=$TRAJ_PER_ITER total_traj=$TOTAL_TRAJ"
    echo "  log=$LOG"

    echo -e "${EXP}\t${SEED}\t${N_ENVS}\t${N_STEPS}\t${GAMMA}\t${EPOCHS}\t${MB}\t${GD}\t${CLIP}\t${TRAJ_PER_ITER}\t${ITERS}\t${TOTAL_TRAJ}" >> "$manifest"

    python -u -m DPPO.finetune_bridge_ablate \
      --pretrain_checkpoint "$PRETRAIN" \
      --bridge_stage p4d_reinforce_pm1_schedclip \
      --env_id PegInsertionSide-v1 --control_mode pd_joint_delta_pos \
      --max_episode_steps 200 --n_envs "$N_ENVS" --n_steps "$N_STEPS" --sim_backend "$SIM_BACKEND" \
      --gamma "$GAMMA" --update_epochs "$EPOCHS" --minibatch_size "$MB" \
      --gamma_denoising "$GD" --is_clip_ratio "$CLIP" \
      --actor_lr 3e-6 --critic_lr 1e-3 \
      --clip_ploss_coef 0.02 --clip_ploss_coef_base 1e-3 --clip_ploss_coef_rate 3.0 --target_kl 1.0 \
      --use_ddim --ddim_steps 10 \
      --min_sampling_denoising_std 0.01 --min_logprob_denoising_std 0.01 \
      --eval_freq 1 --eval_n_rounds 3 \
      --zero_qvel --reward_scale_running \
      --n_train_itr "$ITERS" --seed "$SEED" --exp_name "$EXP" \
      2>&1 | tee "$LOG"
  done
done <<< "$CANDIDATE_SOURCE"

echo "Sweep complete."
