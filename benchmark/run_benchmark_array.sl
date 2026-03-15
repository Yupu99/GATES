#!/bin/bash -e
#SBATCH --job-name=GES_bench
#SBATCH --account=your_account
#SBATCH --time=120:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --output=slurm_out_bench/%A_%a.out
#SBATCH --array=0-59   # 12 variants x 5 seeds

mkdir -p slurm_out_bench

# ============== Define experiment grid ==============
SEEDS=(42 123 456 789 1024)

# Variant definitions: name,alpha,k,surr,optim_name
VARIANTS=(
  "openai_es,0.5,3,2,es_openai"
  "guided_default,0.5,3,2,guided_es"
  "guided_surr1,0.5,3,1,guided_es"
  "guided_surr5,0.5,3,5,guided_es"
  "guided_surr10,0.5,3,10,guided_es"
  "guided_k1,0.5,1,2,guided_es"
  "guided_k5,0.5,5,2,guided_es"
  "guided_k10,0.5,10,2,guided_es"
  "guided_a01,0.1,3,2,guided_es"
  "guided_a03,0.3,3,2,guided_es"
  "guided_a07,0.7,3,2,guided_es"
  "guided_a09,0.9,3,2,guided_es"
)

NUM_VARIANTS=${#VARIANTS[@]}
NUM_SEEDS=${#SEEDS[@]}

# Map SLURM_ARRAY_TASK_ID to (variant_idx, seed_idx)
VARIANT_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS ))

IFS=',' read -r VNAME ALPHA K SURR OPTIM_NAME <<< "${VARIANTS[$VARIANT_IDX]}"
SEED=${SEEDS[$SEED_IDX]}

echo "=========================================="
echo "Variant: $VNAME | Alpha: $ALPHA | K: $K | Surr: $SURR | Seed: $SEED"
echo "Optim:   $OPTIM_NAME"
echo "=========================================="

# Export environment variables for parameter override
export GATES_SEED=$SEED
export GATES_ALPHA=$ALPHA
export GATES_SUBSPACE_K=$K
export GATES_SURR_EPISODES=$SURR
export GATES_GENERATION_NUM=2000
export GATES_EVAL_EP_NUM=10
export GATES_PROCESSOR_NUM=1

# Tag for log directory organization
export RUN_TAG="${VNAME}_seed${SEED}"

if [ "$OPTIM_NAME" == "guided_es" ]; then
  python3 main_guided_es.py
else
  python3 main.py -r 1
fi
