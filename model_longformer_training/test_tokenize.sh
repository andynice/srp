#!/usr/bin/env bash
#SBATCH --job-name=tok_lf
#SBATCH --output=test%j.log
#SBATCH --error=test%j.err
#SBATCH --mail-user=correa@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

echo "Before running tokenize_longformer.py"
srun /home/correa/miniconda3/envs/model_longformer_training/bin/python tokenize_longformer.py -s 2021-01-01 -e 2021-04-01       # python jobs require the srun command to work