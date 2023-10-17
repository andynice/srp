#!/usr/bin/env bash
#SBATCH --job-name=model_bert_training
#SBATCH --output=test%j.log
#SBATCH --error=test%j.err
#SBATCH --mail-user=correa@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

echo "Before running model_covid_twitter_bert_acc_training.py"
srun /home/correa/miniconda3/envs/model_bert_training/bin/python model_covid_twitter_bert_acc_training.py --train True -s 2021-01-01 -e 2021-04-01      # python jobs require the srun command to work