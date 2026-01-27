#!/bin/bash
#SBATCH --job-name=sac_parallel
#SBATCH --output=logs/sac_%j.out   # %j is the Job ID
#SBATCH --error=logs/sac_%j.err
#SBATCH --ntasks=1                 # One primary task
#SBATCH --cpus-per-task=10         # 1 CPU per agent repeat
#SBATCH --gres=gpu:1               # All 10 repeats share this 1 GPU
#SBATCH --mem=32G                  # Increased memory for 10 agents
#SBATCH --time=10:00:00

mkdir -p logs

# Create the results directory if it doesn't exist
RESULTS_DIR=results_january_25_N_1
mkdir -p $RESULTS_DIR

# 1. Configuration
N=1
NUM_REPEATS=10
ENV="Ant-v4" # "Ant-v4", "HalfCheetah-v4", "Hopper-v4", "Humanoid-v4", "Pusher-v4", "Swimmer-v4", "Walker2d-v4"
VAL_EST="explicit_regulariser" # empirical_expectation OR explicit_regulariser
RESULTS_SUB_DIR="${ENV}__SAC_N=${N}"
mkdir -p "$RESULTS_DIR/$RESULTS_SUB_DIR"

# 2. Prevent library conflicts
# This stops each process from trying to use all 10 CPUs for math
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Launching $NUM_REPEATS repeats of $ENV in parallel on 1 GPU..."

# 3. Execution Loop
for i_repeat in $(seq 0 $((NUM_REPEATS - 1)))
do
    # Unique identifier for each repeat
    RUN_ID="${SLURM_JOB_ID}_r${i_repeat}"
    
    # Launch in background (&)
    python cleanrl/sac_continuous_action.py \
        --value_est="$VAL_EST" \
        --num_val_est_samples="$N" \
        --wandb_project_name="SMM-AC-$ENV" \
        --wandb_group="${RESULTS_SUB_DIR}" \
        --wandb_run_name="${RESULTS_SUB_DIR}_seed${i_repeat}" \
        --output_filename="${RESULTS_DIR}/${RESULTS_SUB_DIR}/${RUN_ID}" &
    
    # Short sleep to prevent simultaneous database/file access conflicts
    sleep 2
done

# 4. Critical: Wait for all background processes to finish
wait
echo "All experiments completed."
# ===================
