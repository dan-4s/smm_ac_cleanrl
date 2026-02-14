#!/bin/bash
#SBATCH --job-name=ppo_parallel
#SBATCH --output=logs/ppo_%A_%a.out
#SBATCH --error=logs/ppo_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=40:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --array=0-107               # <--- MODIFIED: 6 Envs * 3 Ents * 6 LRs = 108 jobs (0-62)

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

# --- CONFIGURATION ---
ENV_LIST=("Hopper-v4" "Walker2d-v4" "HalfCheetah-v4" "Ant-v4" "Humanoid-v4" "InvertedDoublePendulum-v4")
ENT_COEF_LIST=(0.0 0.01 0.1)     # Example values
LR_LIST=(1e-5 3e-5 8e-5 1e-4 3e-4 8e-4)         # Example values
NUM_REPEATS=5

# --- INDEXING LOGIC ---
IDX=$SLURM_ARRAY_TASK_ID

# 1. Learning Rate
NUM_LRS=${#LR_LIST[@]}
LR_IDX=$(( IDX % NUM_LRS ))
LR=${LR_LIST[$LR_IDX]}
IDX=$(( IDX / NUM_LRS ))

# 2. Entropy Coefficient
NUM_ENTS=${#ENT_COEF_LIST[@]}
ENT_IDX=$(( IDX % NUM_ENTS ))
ENT_COEF=${ENT_COEF_LIST[$ENT_IDX]}
IDX=$(( IDX / NUM_ENTS ))

# 3. Environment
NUM_ENVS=${#ENV_LIST[@]}
ENV_IDX=$(( IDX % NUM_ENVS ))
ENV=${ENV_LIST[$ENV_IDX]}

echo "Task ID $SLURM_ARRAY_TASK_ID processing: Env=$ENV | LR=$LR | Ent=$ENT_COEF"

RESULTS_SUB_DIR="${ENV}__PPO_Ent=${ENT_COEF}_LR=${LR}"
RESULTS_DIR="results_feb_13_ppo_multi_run"
mkdir -p "$RESULTS_DIR/$RESULTS_SUB_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for i_repeat in $(seq 0 $((NUM_REPEATS - 1)))
do
    RUN_ID="${RESULTS_SUB_DIR}_r${i_repeat}"
    
    python cleanrl/ppo_continuous_action.py \
        --env_id="$ENV" \
        --ent_coef=$ENT_COEF \
        --learning_rate=$LR \
        --wandb_project_name="SMM-AC-$ENV" \
        --wandb_group="$RESULTS_SUB_DIR" \
        --wandb_run_name="${RESULTS_SUB_DIR}_seed${i_repeat}" \
        --output_filename="${RESULTS_DIR}/${RESULTS_SUB_DIR}/${RUN_ID}" \
        --seed $i_repeat \
        --track &
        
    sleep 2
done

wait
