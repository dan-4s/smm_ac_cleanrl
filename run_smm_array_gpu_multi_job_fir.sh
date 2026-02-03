#!/bin/bash
#SBATCH --job-name=smm_fir_h100
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH --output=logs/smm_%A_%a.out
#SBATCH --error=logs/smm_%A_%a.err
#SBATCH --array=0-20               
#SBATCH --time=40:00:00
#SBATCH --mem=64G                  
#SBATCH --cpus-per-task=12         # Optimized for Fir's AMD EPYC (6 cores per CCD)
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:7  # <--- Specific H100 MIG name
#SBATCH --requeue                  
#SBATCH --signal=B:SIGUSR1@120   

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
RESULTS_DIR=results_feb_02_smm_multi_run
mkdir -p $RESULTS_DIR

# Capture the H100 MIG UUIDs
IFS=',' read -ra MIG_IDS <<< "$CUDA_VISIBLE_DEVICES"

# 1. Define parameter arrays
# pi_ref_learning_rates=(1e-6 5e-6 1e-5 5e-5) # Length: 4
pi_ref_freq=(2 4 6)                         # Length: 3
ENV_LIST=("Ant-v4" "HalfCheetah-v4" "Hopper-v4" "Humanoid-v4" "Pusher-v4" "Swimmer-v4" "Walker2d-v4")

# 2. Map SLURM_ARRAY_TASK_ID to indices
# Think of this like nested loops: LR -> Freq -> N -> Repeat
i_freq=$(( SLURM_ARRAY_TASK_ID % 3 ))
# i_lr=$(( (SLURM_ARRAY_TASK_ID / 3) % 4 ))
i_env=$(( SLURM_ARRAY_TASK_ID / 3 ))
# i_env=$(( SLURM_ARRAY_TASK_ID / 12 ))

# 3. Select values
ENV=${ENV_LIST[$i_env]}
PI_REF_LR=1e-5
# PI_REF_LR=${pi_ref_learning_rates[$i_lr]}
REF_FREQ=${pi_ref_freq[$i_freq]}

# Fixed values
SMM_VAL="explicit_regulariser" # explicit_regulariser OR empirical_expectation
ALPHA=1
OMEGA=20
N=1
NUM_REPEATS=7

# 4. Optimization: Thread management
# Prevents processes from competing for the same CPU cores
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 5. Execution
echo "Task: $SLURM_ARRAY_TASK_ID | LR: $PI_REF_LR | Freq: $REF_FREQ | N: $N"
RESULTS_SUB_DIR="${ENV}__SMM__OMEGA=${OMEGA}_LIN_SCHED_ref_freq=${REF_FREQ}"
mkdir -p "${RESULTS_DIR}/${RESULTS_SUB_DIR}"

for i_repeat in $(seq 0 $((NUM_REPEATS - 1)))
do
   export CUDA_VISIBLE_DEVICES=${MIG_IDS[$i_repeat]}

   # Unique identifier for each repeat
   RUN_ID="${RESULTS_SUB_DIR}_r${i_repeat}"
   
   # Launch in background with '&'
   python cleanrl/smm_ac_continuous_action.py \
        --env_id="$ENV" \
        --value_est="$SMM_VAL" \
        --num_val_est_samples="$N" \
        --wandb_project_name="SMM-AC-$ENV" \
        --wandb_run_name="${RESULTS_SUB_DIR}_seed${i_repeat}" \
        --wandb_group="${RESULTS_SUB_DIR}" \
        --ref_policy_frequency="$REF_FREQ" \
        --alpha="$ALPHA" \
        --omega="$OMEGA" \
        --pi_ref_lr="$PI_REF_LR" \
        --output_filename="${RESULTS_DIR}/${RESULTS_SUB_DIR}/${RUN_ID}" &
   
   sleep 2 # Small stagger to prevent W&B login/file-lock spikes
done

# 4. Wait for background tasks
# We use a loop to wait so the script stays alive to catch signals
while pgrep -P $$ > /dev/null; do 
    sleep 5
done
echo "All experiments for $ENV completed."
# ===================

