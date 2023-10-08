#!/usr/bin/env bash
#SBATCH --job-name=model_t5
#SBATCH --output=test%j.log
#SBATCH --error=test%j.err
#SBATCH --mail-user=correa@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:2

echo "Before running model_t5.py"
srun /home/correa/miniconda3/envs/model_t5/bin/python model_t5.py       # python jobs require the srun command to work