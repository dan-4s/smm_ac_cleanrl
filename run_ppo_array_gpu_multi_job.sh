#!/bin/bash
#SBATCH --job-name=ppo_parallel
#SBATCH --output=logs/ppo_%A_%a.out   # %A is Job ID, %a is Array Index
#SBATCH --error=logs/ppo_%A_%a.err
#SBATCH --ntasks=1                 # One primary task
#SBATCH --cpus-per-task=5          # 1 CPU per agent repeat
#SBATCH --gres=gpu:1               # All 5 repeats share this 1 GPU
#SBATCH --mem=32G                  # Memory for 5 agents
#SBATCH --time=15:00:00
#SBATCH --requeue                  # <--- Allow requeuing on preemption
#SBATCH --signal=B:SIGUSR1@120     # <--- Send SIGUSR1 120s before time limit
#SBATCH --array=0-6                # Array for 5 environments

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

    echo "Requeuing job $SLURM_ARRAY_JOB_ID index $SLURM_ARRAY_TASK_ID"
    scontrol requeue $SLURM_JOB_ID
    exit 0
}

# Register the handler for both SIGUSR1 (timeout) and SIGTERM (preemption)
trap 'cleanup_handler' SIGUSR1 SIGTERM

# Create logs directory
mkdir -p logs

# Create the results directory
RESULTS_DIR=results_january_28_ppo_multi_run
mkdir -p $RESULTS_DIR

# 1. Configuration
NUM_REPEATS=5
ENV_LIST=("Ant-v4" "HalfCheetah-v4" "Hopper-v4" "Humanoid-v4" "Pusher-v4" "Swimmer-v4" "Walker2d-v4")
ENV=${ENV_LIST[$SLURM_ARRAY_TASK_ID]}

# Note: PPO script doesn't use VAL_EST or N usually, 
# but we keep naming conventions consistent.
RESULTS_SUB_DIR="${ENV}__PPO"
mkdir -p "$RESULTS_DIR/$RESULTS_SUB_DIR"

# 2. Prevent library conflicts
# This stops each process from trying to use all 10 CPUs for math
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Launching $NUM_REPEATS repeats of $ENV in parallel on 1 GPU (Array Index $SLURM_ARRAY_TASK_ID)..."

# 3. Execution Loop
for i_repeat in $(seq 0 $((NUM_REPEATS - 1)))
do
    # Unique identifier for each repeat (using environment name + repeat index)
    RUN_ID="${RESULTS_SUB_DIR}_r${i_repeat}"
    
    # Launch in background (&)
    # Ensure the path to ppo_continuous_action.py is correct
    python cleanrl/ppo_continuous_action.py \
        --env_id="$ENV" \
        --wandb_project_name="SMM-AC-$ENV" \
        --wandb_group="${RESULTS_SUB_DIR}" \
        --wandb_run_name="${RESULTS_SUB_DIR}_seed${i_repeat}" \
        --output_filename="${RESULTS_DIR}/${RESULTS_SUB_DIR}/${RUN_ID}" &
    
    # Short sleep to prevent simultaneous file access conflicts
    sleep 2
done

# 4. Wait for background tasks
# We use pgrep in a loop so the shell remains "active" to catch signals 
# trapped in step 0.
while pgrep -P $$ > /dev/null; do 
    sleep 5
done

echo "All PPO experiments for $ENV completed."
# ===================
