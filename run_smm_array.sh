#!/bin/bash
#SBATCH --job-name=smm_extensive
#SBATCH --output=logs/smm_%A_%a.out
#SBATCH --error=logs/smm_%A_%a.err
#SBATCH --array=0-359            # (4 LRs * 3 Freqs * 3 Ns * 10 Repeats) - 1
#SBATCH --time=11:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

mkdir -p logs

RESULTS_DIR=results_january_23
mkdir -p $RESULTS_DIR

# 1. Define parameter arrays
pi_ref_learning_rates=(1e-6 5e-6 1e-5 5e-5) # Length: 4
pi_ref_freq=(2 4 6)                         # Length: 3
num_val_est_samples=(1 2 5)                 # Length: 3
NUM_REPEATS=10

# 2. Map SLURM_ARRAY_TASK_ID to indices
# Think of this like nested loops: LR -> Freq -> N -> Repeat
i_repeat=$(( SLURM_ARRAY_TASK_ID % NUM_REPEATS ))
i_n=$(( (SLURM_ARRAY_TASK_ID / NUM_REPEATS) % 3 ))
i_freq=$(( (SLURM_ARRAY_TASK_ID / (NUM_REPEATS * 3)) % 3 ))
i_lr=$(( SLURM_ARRAY_TASK_ID / (NUM_REPEATS * 3 * 3) ))

# 3. Select values
PI_REF_LR=${pi_ref_learning_rates[$i_lr]}
REF_FREQ=${pi_ref_freq[$i_freq]}
N=${num_val_est_samples[$i_n]}

# Fixed values
ENV="Hopper-v4" # "Ant-v4", "HalfCheetah-v4", "Hopper-v4", "Humanoid-v4", "Pusher-v4", "Swimmer-v4", "Walker2d-v4"
SMM_VAL="explicit_regulariser" # explicit_regulariser OR empirical_expectation
ALPHA=1
OMEGA=5

# 4. Execution
echo "Task: $SLURM_ARRAY_TASK_ID | LR: $PI_REF_LR | Freq: $REF_FREQ | N: $N | Run: $((i_repeat + 1))"
RUN_NAME="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
RESULTS_SUB_DIR="${ENV}__SMM_lr=${PI_REF_LR}_ref_freq=${REF_FREQ}_N=${N}"
mkdir -p "${RESULTS_DIR}/${RESULTS_SUB_DIR}"

python cleanrl/smm_ac_continuous_action.py \
    --value_est="$SMM_VAL" \
    --num_val_est_samples="$N" \
    --wandb_project_name="SMM-AC-$ENV" \
    --wandb_run_name="${RESULTS_SUB_DIR}" \
    --ref_policy_frequency="$REF_FREQ" \
    --alpha="$ALPHA" \
    --omega="$OMEGA" \
    --pi_ref_lr="$PI_REF_LR" \
    --no-torch_deterministic \
    --output_filename="${RESULTS_DIR}/${RESULTS_SUB_DIR}/${RUN_NAME}_${i_repeat}"
