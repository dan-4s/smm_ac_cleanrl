#!/bin/bash
#SBATCH --job-name=smm_extensive
#SBATCH --output=logs/smm_%A_%a.out
#SBATCH --error=logs/smm_%A_%a.err
#SBATCH --array=0-149            # (5 LRs * 3 Freqs * 10 Repeats) - 1
#SBATCH --time=5:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

mkdir -p logs

# Create the results directory if it doesn't exist
RESULTS_DIR=results_january_23
mkdir -p $RESULTS_DIR

# 1. Define parameter arrays
pi_ref_learning_rates=(1e-6 5e-6 1e-5 5e-5) # Length: 4
pi_ref_freq=(2 4 6)                         # Length: 3
num_val_est_samples=(1 2 5)                 # Length: 3
NUM_REPEATS=10

# 2. Map SLURM_ARRAY_TASK_ID to indices
# Index for repeat (0-9)
i_repeat=$(( SLURM_ARRAY_TASK_ID % NUM_REPEATS ))

# Index for frequency (0-2)
# Divide by 10 to move past the repeat block
i_freq=$(( (SLURM_ARRAY_TASK_ID / NUM_REPEATS) % 3 ))

# Index for learning rate (0-4)
# Divide by 30 (10 repeats * 3 frequencies) to move past the frequency blocks
i_lr=$(( SLURM_ARRAY_TASK_ID / (NUM_REPEATS * 3) ))

# 3. Select values
PI_REF_LR=${pi_ref_learning_rates[$i_lr]}
REF_FREQ=${pi_ref_freq[$i_freq]}

# Fixed values
SMM_VAL="explicit_regulariser" # explicit_regulariser OR empirical_expectation
ALPHA=1
OMEGA=5

# 4. Execution
echo "Task: $SLURM_ARRAY_TASK_ID | LR: $PI_REF_LR | Freq: $REF_FREQ | Run: $((i_repeat + 1))/$NUM_REPEATS"

python cleanrl/smm_ac_continuous_action.py \
    --smm_value_est="$SMM_VAL" \
    --ref_policy_frequency="$REF_FREQ" \
    --alpha="$ALPHA" \
    --omega="$OMEGA" \
    --pi_ref_lr="$PI_REF_LR" \
    --no-torch_deterministic \
    --exp_name="SMM_lr=${PI_REF_LR}_ref_freq=${REF_FREQ}"
