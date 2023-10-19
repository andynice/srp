import sys, getopt
google_colab = False

if google_colab:
    root_path = "/content"
    train_arg = True
    startDate = '2021-01-01'
    endDate = '2021-04-01'
else:
    root_path = "."
    arguments = sys.argv[1:]
    short_options = "t:s:e:"
    long_options = ["train=", "startDate=", "endDate="]

    options, values = getopt.getopt(arguments, short_options, long_options)

    for o, v in options:
        print(f"Option is {o}. Value is {v}.")
        if o == "-t" or o == "--train":
            if v == "True":
                train_arg = True
            else:
                train_arg = False

        if o == "-s" or o == "--startDate":
            startDate = v
        if o == "-e" or o == "--endDate":
            endDate = v 

import json
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

# TRAIN OR EVAL
train = train_arg
trained_model_name = "model_led.model"

# MODEL DEFINITION
if train:
    if google_colab:
        BASE_MODEL = "allenai/led-base-16384"
    else:
        BASE_MODEL = f"{root_path}/led-base-16384"
else:
    BASE_MODEL = "./" + trained_model_name
LEARNING_RATE = 2e-5
# BATCH_SIZE = 16
BATCH_SIZE = 1
EPOCHS = 20

# accelerator = Accelerator()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)
optimizer = AdamW(model.parameters(), lr=3e-5)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

print(torch.cuda.is_available())

print(torch.cuda.device_count())

print(torch.cuda.current_device())

print(torch.cuda.device(0))

print(torch.cuda.get_device_name(0))

# DATA PREPARATION

g_cases_filename = f"{root_path}/data/g_cases_2021.csv"
y = pd.read_csv(g_cases_filename)

X = pd.DataFrame()
# date_ranges = [['2021-01-01', '2021-04-01']]
# date_ranges = [['2020-01-01', '2020-01-12']]
date_ranges = [[startDate, endDate]]
for date_range in date_ranges:
    start = datetime.datetime.strptime(date_range[0], "%Y-%m-%d")
    end = datetime.datetime.strptime(date_range[1], "%Y-%m-%d")
    date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]

    for date in date_generated:
        date_str = date.strftime("%Y-%m-%d")

        # filename = f"./data/en_{date_str}_output.csv"
        filename = f"{root_path}/tokenize_led_output/tok_{date_str}_output.csv"
        df_data = pd.read_csv(filename)
        X = pd.concat([X, df_data], ignore_index=True)

print(f"X.shape: {X.shape}")
print(f"X.head(): {X.head()}")

train_ratio = 0.70
validation_ratio = 0.15
test_ratio = 0.15

# train is now 70% of the entire data set, test size 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=42, shuffle=False)
#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=42, shuffle=False)

# test is now 15% of the initial data set
# validation is now 15% of the initial data set
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=42, shuffle=False)
train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

print("train_df")
print(f"train_df.shape: {train_df.shape}")
print(train_df)

print("val_df")
print(f"val_df.shape: {val_df.shape}")
print(val_df)

print("test_df")
print(f"test_df.shape: {test_df.shape}")
print(test_df)

raw_train_ds = Dataset.from_pandas(train_df)
raw_val_ds = Dataset.from_pandas(val_df)
raw_test_ds = Dataset.from_pandas(test_df)

ds = {"train": raw_train_ds, "validation": raw_val_ds, "test": raw_test_ds}

print("ds")
print(ds)

def preprocess_function(examples):
    label = examples["g_values"] 
    print("examples['tokenized_tweets']")
    print(examples["tokenized_tweets"])
    s = examples["tokenized_tweets"].replace("\'", "\"")
    examples = json.loads(s)
    
    examples["label"] = float(label)
    return examples

from torch.utils.data import DataLoader

if train:
    ds["train"] = ds["train"].map(preprocess_function, remove_columns=["__index_level_0__", "date", "total_cases", "g_values", "created_at", "tokenized_tweets"])
    ds["validation"] = ds["validation"].map(preprocess_function, remove_columns=["__index_level_0__", "date", "total_cases", "g_values", "created_at", "tokenized_tweets"])

    train_dataloader = DataLoader(
        ds["train"], shuffle=False, batch_size=BATCH_SIZE, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        ds["validation"], batch_size=BATCH_SIZE, collate_fn=data_collator
    )
else:
    ds["test"] = ds["test"].map(preprocess_function, remove_columns=["__index_level_0__", "date", "total_cases", "g_values", "created_at", "tokenized_tweets"])

    test_dataloader = DataLoader(
        ds["test"], batch_size=BATCH_SIZE, collate_fn=data_collator
    )

# for split in ds:
    # ds[split] = ds[split].map(preprocess_function, remove_columns=["__index_level_0__", "date", "total_cases", "g_values", "created_at", "clean_tweets"])

print("ds")
print(ds)

# for batch in train_dataloader:
#     batch = {k: v.to(device) for k, v in batch.items()}
#     break
# print({k: v.shape for k, v in batch.items()})
    
# outputs = model(**batch)
# print(outputs.loss, outputs.logits.shape)

def training():
    num_training_steps = EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    print(f"num_training_steps: {num_training_steps}")

    from tqdm.auto import tqdm

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(EPOCHS):
        for batch_idx, batch in enumerate(train_dataloader):
            print(f"epoch: {epoch}, batch: {batch_idx}")
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            # accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

def evaluation(dataloader):
    eval_preds = []
    eval_labels = []
    model.eval()
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        print(f"outputs['logits'].shape: {outputs['logits'].shape}")
        eval_preds.append(outputs['logits'])
        
        print(f"batch['labels'].shape: {batch['labels'].shape}")
        eval_labels.append(batch['labels'])

    eval_preds = torch.cat(eval_preds, dim=0)
    print(f"eval_preds: {eval_preds}")
    eval_labels = torch.cat(eval_labels, dim=0)
    print(f"eval_labels: {eval_labels}")
    metrics = compute_metrics_for_regression(eval_preds, eval_labels)
    print(metrics)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def compute_metrics_for_regression(eval_pred, eval_labels):
    print(f"eval_pred: {eval_pred}")
    print(f"eval_labels: {eval_labels}")
    logits = eval_pred
    labels = eval_labels
    labels = torch.reshape(labels, (-1, 1))
    print(f"labels: {labels}")
    
    mse = mean_squared_error(labels.cpu(), logits.cpu())
    mae = mean_absolute_error(labels.cpu(), logits.cpu())
    r2 = r2_score(labels.cpu(), logits.cpu())
    single_squared_errors = ((logits.cpu() - labels.cpu()).flatten()**2).tolist()
    
    # Compute accuracy 
    # Based on the fact that the rounded score = true score only if |single_squared_errors| < 0.5
    accuracy = sum([1 for e in single_squared_errors if e < 0.25]) / len(single_squared_errors)
    
    return {"mse": mse, "mae": mae, "r2": r2, "accuracy": accuracy}

if train:
    ## TRAINING
    t = time()

    training()
    evaluation(eval_dataloader)
    model.save_pretrained(trained_model_name, from_pt=True)
    tokenizer.save_pretrained(trained_model_name)

    print('Time to train model: {} mins'.format(round((time() - t) / 60, 2)))

else:
    ## EVALUATION
    t = time()
    evaluation(test_dataloader)

    print('Time to eval model: {} mins'.format(round((time() - t) / 60, 2)))