#!/bin/bash -e

#SBATCH --job-name=DWS    	    # job name (shows up in the queue)
#SBATCH --account=./...
#SBATCH --time=120:00:00         	# Wall time (HH:MM:SS)  Max: 3 Weeks on one genoa node
#SBATCH --mem=16G                   # Amount of memory per node
#SBATCH --cpus-per-task=8         	# Number of CPUs (align with --processor_num below)
#SBATCH --mail-type=END
#SBATCH --output=slurm_out_train/%A_%a.out  # %A=array job ID, %a=task index

out_dir="slurm_out_train"  # Directory to store .out files
mkdir -p "$out_dir"

# $1: config yaml path (e.g. config/workflow_scheduling_guided_es_ppo.yaml)
# $SLURM_ARRAY_TASK_ID: run index (1-10) set by --array when submitting

CONFIG=${1:-config/workflow_scheduling_es_openai.yaml}
RUN=${SLURM_ARRAY_TASK_ID:-1}

echo "======================="
echo " Slurm Job Information "
echo "======================="
scontrol show job "$SLURM_JOB_ID"
echo "======================="
echo "Config : $CONFIG"
echo "Run    : $RUN"
echo ""

python3 main.py -r $RUN --config $CONFIG --processor_num 8

