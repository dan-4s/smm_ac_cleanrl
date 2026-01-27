#!/bin/bash
#SBATCH --job-name=smm_parallel
#SBATCH --output=logs/smm_%A_%a.out
#SBATCH --error=logs/smm_%A_%a.err
#SBATCH --array=0-11               # FIXED: 4 LRs * 3 Freqs = 12 combinations
#SBATCH --time=15:00:00
#SBATCH --mem=32G                  # INCREASED: To support 10 parallel processes
#SBATCH --cpus-per-task=10         # FIXED: Requesting 10 CPUs (1 per background process)
#SBATCH --gres=gpu:1               # Request 1 GPU for all 10 processes to share

mkdir -p logs
RESULTS_DIR=results_january_27_N_1
mkdir -p $RESULTS_DIR

# 1. Define parameter arrays
pi_ref_learning_rates=(1e-6 5e-6 1e-5 5e-5) # Length: 4
pi_ref_freq=(2 4 6)                         # Length: 3
NUM_REPEATS=10

# 2. Map SLURM_ARRAY_TASK_ID to indices
# Think of this like nested loops: LR -> Freq -> N -> Repeat
i_freq=$(( SLURM_ARRAY_TASK_ID % 3 ))
i_lr=$(( SLURM_ARRAY_TASK_ID / 3 ))

# 3. Select values
PI_REF_LR=${pi_ref_learning_rates[$i_lr]}
REF_FREQ=${pi_ref_freq[$i_freq]}

# Fixed values
ENV="Ant-v4" # "Ant-v4", "HalfCheetah-v4", "Hopper-v4", "Humanoid-v4", "Pusher-v4", "Swimmer-v4", "Walker2d-v4"
SMM_VAL="explicit_regulariser" # explicit_regulariser OR empirical_expectation
ALPHA=1
OMEGA=5
N=1

# 4. Optimization: Thread management
# Prevents processes from competing for the same CPU cores
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 5. Execution
echo "Task: $SLURM_ARRAY_TASK_ID | LR: $PI_REF_LR | Freq: $REF_FREQ | N: $N"
RESULTS_SUB_DIR="${ENV}__SMM_lr=${PI_REF_LR}_ref_freq=${REF_FREQ}_N=${N}"
mkdir -p "${RESULTS_DIR}/${RESULTS_SUB_DIR}"

for i_repeat in $(seq 0 $((NUM_REPEATS - 1)))
do
   # We use a unique RUN_NAME per repeat for file tracking
   RUN_NAME="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_r${i_repeat}"
   
   # Launch in background with '&'
   python cleanrl/smm_ac_continuous_action.py \
        --value_est="$SMM_VAL" \
        --num_val_est_samples="$N" \
        --wandb_project_name="SMM-AC-$ENV" \
        --wandb_run_name="${RESULTS_SUB_DIR}_seed${i_repeat}" \
        --wandb_group="${RESULTS_SUB_DIR}" \
        --ref_policy_frequency="$REF_FREQ" \
        --alpha="$ALPHA" \
        --omega="$OMEGA" \
        --pi_ref_lr="$PI_REF_LR" \
        --output_filename="${RESULTS_DIR}/${RESULTS_SUB_DIR}/${RUN_NAME}" &
   
   sleep 2 # Small stagger to prevent W&B login/file-lock spikes
done

# 5. Wait for all 10 background processes to finish
wait

