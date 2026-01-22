#!/bin/bash
#SBATCH --job-name=smm_extensive
#SBATCH --output=logs/smm_%A_%a.out
#SBATCH --error=logs/smm_%A_%a.err
#SBATCH --array=0-9                 # (10 Repeats) - 1
#SBATCH --time=5:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

mkdir -p logs

# 1. Define parameter arrays
NUM_REPEATS=10

# 2. Map SLURM_ARRAY_TASK_ID to indices
# Index for repeat (0-9)
i_repeat=$(( SLURM_ARRAY_TASK_ID % NUM_REPEATS ))

VAL="empirical_expectation" # OR explicit_regulariser

# 3. Execution
echo "Task: $SLURM_ARRAY_TASK_ID | Run: $((i_repeat + 1))/$NUM_REPEATS"

python cleanrl/sac_continuous_action.py \
    --no-torch_deterministic \
    --value_est="$VAL" \
    --exp_name="SAC_AUTOTUNE_EMPIR_EXPEC"
