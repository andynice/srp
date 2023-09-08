#!/usr/bin/env bash
#SBATCH --job-name=word2vec
#SBATCH --output=test%j.log
#SBATCH --error=test%j.err
#SBATCH --mail-user=correa@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:2

echo "This is a test echo"
srun /home/correa/miniconda3/envs/word2vec/bin/python word_embedding.py        # python jobs require the srun command to work