#!/bin/bash -e

#SBATCH --job-name=DWS    	    # job name (shows up in the queue)
#SBATCH --account=./...
#SBATCH --time=120:00:00         	# Wall time (HH:MM:SS)  Max: 3 Weeks on one genoa node
#SBATCH --mem=8G                    # Amount of memory per node
#SBATCH --cpus-per-task=1         	# Number of CPUs per task
#SBATCH --mail-type=END
#SBATCH --output=slurm_out_train/%j.out

out_dir="slurm_out_train"  # Directory to store .out files
mkdir -p "$out_dir"

echo "======================="
echo " Slurm Job Information "
echo "======================="
scontrol show job "$SLURM_JOB_ID"
echo "======================="
echo ""
# display information about the available GPUs
# nvidia-smi
# check the value of the CUDA_VISIBLE_DEVICES variable
# echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

python3 main.py -r $1  # for multiple independent runs

