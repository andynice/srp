import sys, getopt

arguments = sys.argv[1:]
short_options = "s:e:"
long_options = ["startDate=", "endDate="]

options, values = getopt.getopt(arguments, short_options, long_options)

for o, v in options:
    print(f"Option is {o}. Value is {v}.")
    if o == "-s" or o == "--startDate":
        startDate = v
    if o == "-e" or o == "--endDate":
        endDate = v 

import torch
import pandas as pd
import datetime
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split

# from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, AdamW, get_scheduler
from torch.utils.data import DataLoader

from time import time

# MODEL DEFINITION
BASE_MODEL = "./longformer-base-4096"

MAX_LENGTH = 4096
# BATCH_SIZE = 16
BATCH_SIZE = 2

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# DATA PREPARATION

# date_ranges = [['2021-01-01', '2021-04-01']]
# date_ranges = [['2020-01-01', '2020-01-12']]
date_ranges = [[startDate, endDate]]
for date_range in date_ranges:
    start = datetime.datetime.strptime(date_range[0], "%Y-%m-%d")
    end = datetime.datetime.strptime(date_range[1], "%Y-%m-%d")
    date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]

    for date in date_generated:
        date_str = date.strftime("%Y-%m-%d")

        filename = f"./data/en_{date_str}_output.csv"
        df_data = pd.read_csv(filename)
        df_data = df_data.replace(r'^\s*$', np.nan, regex=True)
        df_data["tokenized_tweets"] = str(tokenizer(df_data["clean_tweets"][0], truncation=True, padding="max_length", max_length=MAX_LENGTH))
        df_data = df_data.drop(["clean_tweets"], axis=1, errors="ignore")
        print(f"df_data.head(): {df_data.head()}")
        
        # Write processed data to output file
        output_file = f"./tokenize_longformer_output/tok_{date_str}_output.csv"
        df_data.to_csv(output_file, mode='a', index=False, header=True)
