#!/usr/bin/env bash
#SBATCH --job-name=model_longformer_training
#SBATCH --output=test%j.log
#SBATCH --error=test%j.err
#SBATCH --mail-user=correa@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

echo "Before running model_longformer_training.py"
srun /home/correa/miniconda3/envs/model_longformer_training/bin/python model_longformer_training.py --train True       # python jobs require the srun command to work