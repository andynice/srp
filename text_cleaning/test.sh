#!/usr/bin/env bash
#SBATCH --job-name=text_cleaning
#SBATCH --output=test%j.log
#SBATCH --error=test%j.err
#SBATCH --mail-user=correa@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:2

echo "Before running text_cleaner.py"
srun /home/correa/miniconda3/envs/text_cleaning/bin/python text_cleaner.py --clean False --merge True --startDate 2021-01-01 --endDate 2021-01-02        # python jobs require the srun command to work