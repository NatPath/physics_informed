#!/bin/bash

##Resource Request

#SBATCH --job-name print_result
#SBATCH --mail-user=dor-hay.sha@campus.technion.ac.il
#SBATCH --mail-type=ALL           # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output /home/dor-hay.sha/project/physics_informed/slurm_script/output/print-result-%j.out   ## filename of the output; the %j is equivalent to jobID; default is slurm-[jobID].out
#SBATCH --ntasks=1  ## number of tasks (analyses) to run
#SBATCH --gres=gpu:A4000:1          # Request 1 gpu type A40
#SBATCH --time=0-0:20:00  ## time for analysis (day-hour:min:sec)

##Load the CUDA module
module load cuda

eval "$(conda shell.bash hook)"
conda activate pino-env

## Run the script
nvidia-smi
python draw_spdc.py --config_path configs/test/SPDC-test-tmp_pure.yaml --emd_off
echo Done
