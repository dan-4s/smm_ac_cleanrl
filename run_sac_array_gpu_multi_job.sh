#!/bin/bash
#SBATCH --job-name=sac_parallel
#SBATCH --output=logs/sac_%A_%a.out   # %j is the Job ID
#SBATCH --error=logs/sac_%A_%a.err
#SBATCH --ntasks=1                 # One primary task
#SBATCH --cpus-per-task=5          # 1 CPU per agent repeat
#SBATCH --gres=gpu:1               # All 5 repeats share this 1 GPU
#SBATCH --mem=32G                  # Increased memory for 5 agents
#SBATCH --time=15:00:00
#SBATCH --requeue                  # <--- 1. Tell Slurm to allow requeuing
#SBATCH --signal=B:SIGUSR1@120     # <--- 2. Send SIGUSR1 120 seconds before time limit
#SBATCH --array=0-4

# 0. Define the cleanup/resubmission handler
cleanup_handler() {
    echo "Preemption or Timeout signal received at $(date)"
    
    # Kill all child processes (the python runs)
    # This triggers the internal python signal handlers to save checkpoints
    trap - SIGTERM # prevent recursion
    child_pids=$(jobs -p)
    if [ -n "$child_pids" ]; then
        echo "Stopping child processes: $child_pids"
        kill -SIGTERM $child_pids
        wait $child_pids
    fi

    echo "Requeuing job $SLURM_JOB_ID"
    scontrol requeue $SLURM_JOB_ID
    exit 0
}

# Register the handler for both SIGUSR1 (timeout) and SIGTERM (preemption)
trap 'cleanup_handler' SIGUSR1 SIGTERM

# Create logs directory.
mkdir -p logs

# Create the results directory if it doesn't exist.
RESULTS_DIR=results_january_27_sac_multi_run
mkdir -p $RESULTS_DIR

# 1. Configuration
N=1
NUM_REPEATS=5 # 10 seems to choke the GPU, we'll see what 5 does.
ENV_LIST=("HalfCheetah-v4" "Humanoid-v4" "Pusher-v4" "Swimmer-v4" "Walker2d-v4")
# ENV_LIST=("Ant-v4" "HalfCheetah-v4" "Hopper-v4" "Humanoid-v4" "Pusher-v4" "Swimmer-v4" "Walker2d-v4")
ENV=${ENV_LIST[$SLURM_ARRAY_TASK_ID]}
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
    RUN_ID="${ENV}_r${i_repeat}"
    
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

# 4. Wait for background tasks
# We use a loop to wait so the script stays alive to catch signals
while pgrep -P $$ > /dev/null; do 
    sleep 5
done
echo "All experiments completed."
# ===================


