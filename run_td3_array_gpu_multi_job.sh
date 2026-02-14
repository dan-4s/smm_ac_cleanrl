#!/bin/bash
#SBATCH --job-name=td3_parallel
#SBATCH --output=logs/td3_%A_%a.out   # %A is Job ID, %a is Array Index
#SBATCH --error=logs/td3_%A_%a.err
#SBATCH --ntasks=1                 # One primary task
#SBATCH --cpus-per-task=5          # 1 CPU per agent repeat
#SBATCH --gres=gpu:1               # All 5 repeats share this 1 GPU
#SBATCH --mem=32G                  # Memory for 5 agents
#SBATCH --time=40:00:00
#SBATCH --requeue                  # <--- Allow requeuing on preemption
#SBATCH --signal=B:SIGUSR1@120     # <--- Send SIGUSR1 120s before time limit
#SBATCH --array=0-269              # <--- MODIFIED: 5 Envs * 6 LRs * 3 Freqs * 3 Noises = 270 jobs

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

source ~/.bashrc
source ~/setup_env.sh
module load OpenSSL/1.1 # Just in case, since we have seen errors from this.

# Create logs directory
mkdir -p logs

# Create the results directory
RESULTS_DIR="results_feb_13_td3_multi_run"
mkdir -p "$RESULTS_DIR"

# 1. Configuration
NUM_REPEATS=5
ENV_LIST=("Ant-v4" "HalfCheetah-v4" "Hopper-v4" "Humanoid-v4" "Walker2d-v4" "InvertedDoublePendulum-v4")
LR_LIST=(1e-5 3e-5 8e-5 1e-4 3e-4 8e-4)
UP_FREQ_LIST=(2 4 6)
POL_NOISE_LIST=(0.1 0.2 0.3)

# --- INDEXING LOGIC (GRID SEARCH) ---
# We decode the SLURM_ARRAY_TASK_ID into indices for each list
# Logic: Inner-most loop (Noise) -> Freq -> LR -> Outer (Env)
IDX=$SLURM_ARRAY_TASK_ID

# 1. Policy Noise
NUM_NOISE=${#POL_NOISE_LIST[@]}
NOISE_IDX=$(( IDX % NUM_NOISE ))
POL_NOISE=${POL_NOISE_LIST[$NOISE_IDX]}
IDX=$(( IDX / NUM_NOISE ))

# 2. Update Frequency
NUM_FREQ=${#UP_FREQ_LIST[@]}
FREQ_IDX=$(( IDX % NUM_FREQ ))
UP_FREQ=${UP_FREQ_LIST[$FREQ_IDX]}
IDX=$(( IDX / NUM_FREQ ))

# 3. Learning Rate
NUM_LR=${#LR_LIST[@]}
LR_IDX=$(( IDX % NUM_LR ))
LR=${LR_LIST[$LR_IDX]}
IDX=$(( IDX / NUM_LR ))

# 4. Environment
NUM_ENV=${#ENV_LIST[@]}
ENV_IDX=$(( IDX % NUM_ENV ))
ENV=${ENV_LIST[$ENV_IDX]}

echo "Task ID $SLURM_ARRAY_TASK_ID processing: Env=$ENV | LR=$LR | Freq=$UP_FREQ | Noise=$POL_NOISE"

# Note: MODIFIED directory name to include hyperparams so jobs don't overwrite each other
RESULTS_SUB_DIR="${ENV}__TD3_LR=${LR}_Freq=${UP_FREQ}_Noise=${POL_NOISE}"
mkdir -p "$RESULTS_DIR/$RESULTS_SUB_DIR"

# 2. Prevent library conflicts
# This stops each process from trying to use all 10 CPUs for math
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Launching $NUM_REPEATS repeats of $ENV in parallel on 1 GPU..."

# 3. Execution Loop
for i_repeat in $(seq 0 $((NUM_REPEATS - 1)))
do
    # Unique identifier for each repeat (using environment name + repeat index)
    RUN_ID="${RESULTS_SUB_DIR}_r${i_repeat}"
    
    # Launch in background (&)
    python cleanrl/td3_continuous_action.py \
        --env_id="$ENV" \
        --policy_lr=$LR \
        --policy_frequency=$UP_FREQ \
        --policy_noise=$POL_NOISE \
        --wandb_project_name="SMM-AC-$ENV" \
        --wandb_group="${RESULTS_SUB_DIR}" \
        --wandb_run_name="${RESULTS_SUB_DIR}_seed${i_repeat}" \
        --output_filename="${RESULTS_DIR}/${RESULTS_SUB_DIR}/${RUN_ID}" &
    
    # Short sleep to prevent simultaneous file access conflicts
    sleep 2
done

# 4. Wait for background tasks
wait

echo "All TD3 experiments for $ENV completed."
