#!/bin/bash
#SBATCH --job-name=smm_extensive
#SBATCH --output=logs/smm_%A_%a.out
#SBATCH --error=logs/smm_%A_%a.err
#SBATCH --array=0-29                 # (10 Repeats) - 1
#SBATCH --time=11:00:00
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
N=${num_val_est_samples[$((SLURM_ARRAY_TASK_ID / NUM_REPEATS))]}
i_repeat=$(( SLURM_ARRAY_TASK_ID % NUM_REPEATS ))

ENV="Hopper-v4" # "Ant-v4", "HalfCheetah-v4", "Hopper-v4", "Humanoid-v4", "Pusher-v4", "Swimmer-v4", "Walker2d-v4"
VAL_EST="explicit_regulariser" # empirical_expectation OR explicit_regulariser

# 3. Execution
echo "Task: $SLURM_ARRAY_TASK_ID | Run: $((i_repeat + 1))/$NUM_REPEATS"
RUN_NAME="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
RESULTS_SUB_DIR="${ENV}__SAC_N=${N}"
mkdir -p "$RESULTS_DIR/$RESULTS_SUB_DIR"

python cleanrl/sac_continuous_action.py \
    --no-torch_deterministic \
    --value_est="$VAL_EST" \
    --num_val_est_samples="$N" \
    --wandb_project_name="SMM-AC-$ENV" \
    --wandb_run_name="${RESULTS_SUB_DIR}" \
    --output_filename="${RESULTS_DIR}/${RESULTS_SUB_DIR}/${RUN_NAME}_${i_repeat}"
