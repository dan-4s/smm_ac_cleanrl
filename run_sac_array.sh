#!/bin/bash
#SBATCH --job-name=smm_extensive
#SBATCH --output=logs/smm_%A_%a.out
#SBATCH --error=logs/smm_%A_%a.err
#SBATCH --array=0-29                 # (10 Repeats) - 1
#SBATCH --time=5:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

mkdir -p logs

# Create the results directory if it doesn't exist
RESULTS_DIR=results_january_23
mkdir -p $RESULTS_DIR

# 1. Define parameter arrays
num_val_est_samples=(1 2 5) # Length: 3
NUM_REPEATS=10

# 2. Map SLURM_ARRAY_TASK_ID to indices
# Index for repeat (0-9)
N=${num_val_est_samples[SLURM_ARRAY_TASK_ID / NUM_REPEATS]}
i_repeat=$(( SLURM_ARRAY_TASK_ID % NUM_REPEATS ))

VAL_EST="explicit_regulariser" # empirical_expectation OR explicit_regulariser

# 3. Execution
echo "Task: $SLURM_ARRAY_TASK_ID | Run: $((i_repeat + 1))/$NUM_REPEATS"
RUN_NAME="sac_N=$N"
RESULTS_SUB_DIR="SAC_N=$N"
mkdir -p "$RESULTS_DIR/$RESULTS_SUB_DIR"

python cleanrl/sac_continuous_action.py \
    --no-torch_deterministic \
    --value_est="$VAL_EST" \
    --num_val_est_samples="$N" \
    --wandb_run_name="$RUN_NAME" \
    --output_filename="$RESULTS_DIR/$RESULTS_SUB_DIR/RUN_NAME_$i_repeat"
