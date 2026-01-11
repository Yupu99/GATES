#!/bin/bash -e

#SBATCH --job-name=DWS_test    		    # job name (shows up in the queue)
#SBATCH --account=./...
#SBATCH --time=168:00:00         	# Wall time (HH:MM:SS)
#SBATCH --mem=4G                  	# Amount of memory per node
#SBATCH --cpus-per-task=1         	# Number of CPUs per task
#SBATCH --mail-type=END
#SBATCH --output=slurm_out_test/%j.out

out_dir="slurm_out_test"  # Directory to store .out files
mkdir -p "$out_dir"

python3 eval_rl.py -g $1 -w $2 -log $3 -m $4  # test on the specific model