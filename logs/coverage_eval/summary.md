# PegInsertion GPU Coverage Summary

Config shared by all runs:
- env_id: `PegInsertionSide-v1`
- control_mode: `pd_joint_delta_pos`
- max_episode_steps: `200`
- num_states: `1000`
- ddim_steps: `10`
- min_sampling_denoising_std: `0.01`
- seed: `0`
- deterministic: `mc_samples=1`
- stochastic: `mc_samples=16`

## Checkpoints
- `pretrain`: `runs/dppo_pretrain/dppo_pretrain_peg_zeroqvel_500k/best.pt`
- `dppo_g099`: `runs/dppo_finetune/dppo_ft_peg_gae_g099/best.pt`
- `dppo_zeroqvel_v2`: `runs/dppo_finetune/dppo_ft_peg_zeroqvel_v2/best.pt`
- `p4d_full`: `runs/dppo_finetune/peb_p4d_reinforce_pm1_schedclip_seed0/best.pt`
- `p4d_3k`: `runs/dppo_finetune/peb_p4d_budget3k_env300_step25_ep10_itr10_seed0/best.pt`

## Summary Table

| Policy | Mode | SR | frac_zero | frac_one | frac_decisive | mean P | std P | median P |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pretrain | deterministic | 52.2% | 47.8% | 52.2% | 0.0% | 0.522 | 0.500 | 1.000 |
| pretrain | stochastic | 53.6% | 5.1% | 3.7% | 82.2% | 0.536 | 0.283 | 0.562 |
| dppo_g099 | deterministic | 89.8% | 10.2% | 89.8% | 0.0% | 0.898 | 0.303 | 1.000 |
| dppo_g099 | stochastic | 86.4% | 2.0% | 30.2% | 39.2% | 0.864 | 0.199 | 0.938 |
| dppo_zeroqvel_v2 | deterministic | 93.6% | 6.4% | 93.6% | 0.0% | 0.936 | 0.245 | 1.000 |
| dppo_zeroqvel_v2 | stochastic | 91.2% | 2.0% | 48.5% | 22.5% | 0.912 | 0.176 | 0.938 |
| p4d_full | deterministic | 91.6% | 8.4% | 91.6% | 0.0% | 0.916 | 0.277 | 1.000 |
| p4d_full | stochastic | 88.3% | 2.1% | 37.2% | 34.4% | 0.883 | 0.190 | 0.938 |
| p4d_3k | deterministic | 89.2% | 10.8% | 89.2% | 0.0% | 0.892 | 0.310 | 1.000 |
| p4d_3k | stochastic | 84.6% | 2.3% | 30.5% | 41.6% | 0.846 | 0.216 | 0.938 |

## Training Eval vs Fixed-State Coverage

These numbers should not be conflated:
- training `best_sr_once` is the inline finetune eval metric (fresh eval episodes during training)
- coverage SR here is computed on a fixed set of `1000` initial states with `seed=0`

| Policy | Training best `sr_once` | Fixed-state deterministic coverage SR |
|---|---:|---:|
| dppo_g099 | 91.6% | 89.8% |
| dppo_zeroqvel_v2 | 94.2% | 93.6% |
| p4d_full | 93.4% | 91.6% |
| p4d_3k | 90.6% | 89.2% |

## Main Takeaways
- `dppo_zeroqvel_v2` is the true 93+ DPPO checkpoint: `93.6%` deterministic, `91.2%` stochastic.
- `p4d_full` reaches `93.4%` in training eval, but under the fixed-state coverage protocol it is `91.6%` deterministic and `88.3%` stochastic.
- `dppo_g099` is a different, weaker DPPO run: `89.8%` deterministic, `86.4%` stochastic.
- `p4d_3k` is close to `dppo_g099` under deterministic eval (`89.2%` vs `89.8%`), but clearly worse under stochastic coverage (`84.6%` vs `86.4%`).
- `pretrain` has low mean performance but strong per-state stochasticity: deterministic `frac_zero=47.8%`, stochastic `frac_zero=5.1%`, and `frac_decisive=82.2%`.
- For all finetuned policies, deterministic evaluation collapses coverage into only `0` or `1`, while stochastic evaluation exposes the amount of intermediate-probability states.
