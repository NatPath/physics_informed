#!/bin/bash

##Resource Request

#SBATCH --job-name pino-100
#SBATCH --mail-user=nativ.maor@campus.technion.ac.il
#SBATCH --mail-type=ALL           # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output /home/nativ.maor/ML4Physics/physics_informed/slurm_script/output/pino100-result-%j.out   ## filename of the output; the %j is equivalent to jobID; default is slurm-[jobID].out
#SBATCH --ntasks=1  ## number of tasks (analyses) to run
#SBATCH --gres=gpu:A40:1          # Request 1 gpu type A40
#SBATCH --exclude=nlp-a40-1
#SBATCH --time=0-10:10:00  ## time for analysis (day-hour:min:sec)

##Load the CUDA module
module load cuda

eval "$(conda shell.bash hook)"
conda activate pino-env

## Run the script
nvidia-smi
python train_spdc.py --config_path configs/ngc/SPDC-pino-100-loss_ratio-4-pump-random.yaml --mode train --log --validate
echo Done