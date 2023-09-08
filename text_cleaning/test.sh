#!/usr/bin/env bash
#SBATCH --job-name=text_cleaning
#SBATCH --output=test%j.log
#SBATCH --error=test%j.err
#SBATCH --mail-user=correa@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:2

echo "Before running text_cleaner.py"
srun /home/correa/miniconda3/envs/clean_text/bin/python text_cleaner.py        # python jobs require the srun command to work