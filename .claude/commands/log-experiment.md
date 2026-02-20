Record the experiment that was just run in this conversation to the experiment log.

## Instructions

1. **Extract experiment info from the conversation context above.** Look for:
   - The Python command that was executed (e.g. `python -m RL.ppo_finetune ...`)
   - All hyperparameters and settings printed in stdout or passed as CLI args
   - Key result metrics: success_rate, mean_reward, episode_length, loss values, or any final evaluation numbers
   - The run directory path (e.g. `runs/ppo_gae__seed1__...`)
   - Any errors or notable observations

2. **Gather git info** by running:
   - `git rev-parse --short HEAD` for commit hash
   - `git branch --show-current` for branch name

3. **Get current date/time** by running `date '+%Y-%m-%d %H:%M'`

4. **Append an entry to `experiments/log.md`** using the format below. If the file doesn't exist or is empty, add the title header first.

## Entry Format

```
## [Experiment Name] - YYYY-MM-DD HH:MM

**Command**: `the full python command`
**Git**: abc1234 (branch-name)
**Run Dir**: runs/...

### Settings
| Parameter | Value |
|-----------|-------|
| param1 | value1 |
| param2 | value2 |
| ... | ... |

### Results
| Metric | Value |
|--------|-------|
| success_rate | 0.95 |
| mean_reward | 123.4 |
| ... | ... |

### Notes
- Any notable observations, errors, or takeaways from this run.

---
```

## Rules
- The experiment name should be concise and descriptive (e.g. "PPO + GAE finetuning", "IQL pretrain tau=0.7")
- Include ALL hyperparameters that were explicitly set or printed, not just the non-default ones
- If results are not yet available (experiment still running), write "In progress" in the Results section
- If the user provides additional notes or context, include them in the Notes section
- Always append to the end of the file, never overwrite existing entries
- Keep the log in reverse chronological order is NOT required — just append to the end
