import sys, getopt

arguments = sys.argv[1:]
short_options = "t:"
long_options = ["train="]

options, values = getopt.getopt(arguments, short_options, long_options)

for o, v in options:
    print(f"Option is {o}. Value for 'train' is {v}.")
    if o == "-t" or o == "--train":
        if v == "True":
            train_arg = True
        else:
            train_arg = False

import pandas as pd
import datetime
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader

from time import time

# TRAIN OR EVAL
train = train_arg
trained_model_name = "model_covid_twitter_bert_v2.model"

# MODEL DEFINITION
if train:
    # BASE_MODEL = "digitalepidemiologylab/covid-twitter-bert"
    # BASE_MODEL = "./covid-twitter-bert"
    # BASE_MODEL = "digitalepidemiologylab/covid-twitter-bert-v2"
    BASE_MODEL = "./covid-twitter-bert-v2"
else:
    BASE_MODEL = "./" + trained_model_name
LEARNING_RATE = 2e-5
# MAX_LENGTH = 256
# BATCH_SIZE = 16
BATCH_SIZE = 8
EPOCHS = 20

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)

# DATA PREPARATION

g_cases_filename = f"./data/g_cases_2021.csv"
y = pd.read_csv(g_cases_filename)

X = pd.DataFrame()
date_ranges = [['2021-01-01', '2021-04-01']]
# date_ranges = [['2020-01-01', '2020-01-12']]
for date_range in date_ranges:
    start = datetime.datetime.strptime(date_range[0], "%Y-%m-%d")
    end = datetime.datetime.strptime(date_range[1], "%Y-%m-%d")
    date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]

    for date in date_generated:
        date_str = date.strftime("%Y-%m-%d")

        filename = f"./data/en_{date_str}_output.csv"
        df_data = pd.read_csv(filename)
        df_data = df_data.replace(r'^\s*$', np.nan, regex=True)
        #X = total_dataframe.append(df_data, ignore_index=True)
        X = pd.concat([X, df_data], ignore_index=True)

print(f"X.shape: {X.shape}")

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

max_length = tokenizer.model_max_length
print("Maximum input sequence length:", max_length)
# Maximum input sequence length: 1000000000000000019884624838656

ds = {"train": raw_train_ds, "validation": raw_val_ds, "test": raw_test_ds}

print("ds")
print(ds)

# CALCULATE MAX_LENGTH
# Tokenize all tweets to find the maximum tokenized length
# max_length = 0

# for index, row in X.iterrows():
#     tweet = row['clean_tweets']

#     # Tokenize the tweet
#     tokens = tokenizer(tweet, return_tensors="pt")

#     # Get the length of the tokenized sequence
#     length = tokens['input_ids'].shape[1]
#     print(f"current length: {length}")

#     # Update max_length if needed
#     if length > max_length:
#         max_length = length

# print(f"Calculated max_length: {max_length}")
# Model's max length
max_length = 512
# RuntimeError: The size of tensor a (600) must match the size of tensor b (512) at non-singleton dimension 1

def preprocess_function(examples):
    label = examples["g_values"] 
    # examples = tokenizer(examples["clean_tweets"], truncation=True, padding="max_length", max_length=256)
    # examples = tokenizer(examples["clean_tweets"], truncation=False, padding=True, return_tensors="pt")
    examples = tokenizer(examples["clean_tweets"], truncation=True, padding="max_length", max_length=max_length)
    # examples = tokenizer(examples["clean_tweets"], truncation=True, padding="max_length")
    
    # examples = tokenizer(examples["clean_tweets"])

    # examples["label"] = float(label)
    examples["label"] = [float(i) for i in label]
    print(examples)
    return examples

for split in ds:
    # ds[split] = ds[split].map(preprocess_function, remove_columns=["date", "total_cases", "g_values", "created_at", "clean_tweets"])
    ds[split] = ds[split].map(preprocess_function, remove_columns=["date", "total_cases", "g_values", "created_at", "clean_tweets"], batched=True)

print("ds")
print(ds)


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)
    
    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    single_squared_errors = ((logits - labels).flatten()**2).tolist()
    
    # Compute accuracy 
    # Based on the fact that the rounded score = true score only if |single_squared_errors| < 0.5
    accuracy = sum([1 for e in single_squared_errors if e < 0.25]) / len(single_squared_errors)
    
    return {"mse": mse, "mae": mae, "r2": r2, "accuracy": accuracy}



from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./models/covid-twitter-bert-v2-fine-tuned-regression",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="accuracy",
    load_best_model_at_end=True,
    weight_decay=0.01,
)


import torch
from transformers import Trainer

class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0][:, 0]
        loss = torch.nn.functional.mse_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss


trainer = RegressionTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    compute_metrics=compute_metrics_for_regression,
)


if train:
    ## TRAINING
    t = time()

    # trainer.train()
    # trainer.save_model(trained_model_name)
    # tokenizer.save_pretrained(trained_model_name)

    print('Time to train model: {} mins'.format(round((time() - t) / 60, 2)))

else:
    ## EVALUATION
    t = time()
    
    trainer.eval_dataset=ds["test"]
    metrics = trainer.evaluate()
    print(metrics)

    print('Time to eval model: {} mins'.format(round((time() - t) / 60, 2)))

# # from local folder
# # model = AutoModelForSequenceClassification.from_pretrained("./saved_model/")

# # from transformers import AutoTokenizer

# # tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# # sequence = "mf"
# # tokens = tokenizer.tokenize(sequence)

# # print(tokens)

# # ids = tokenizer.convert_tokens_to_ids(tokens)

# # print(ids)