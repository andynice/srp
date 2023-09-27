import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

g_cases_filename = f"./data/g_cases_2021.csv"
y = pd.read_csv(g_cases_filename)

X = pd.DataFrame()
date_ranges = [['2021-01-01', '2021-01-04']]
for date_range in date_ranges:
    start = datetime.datetime.strptime(date_range[0], "%Y-%m-%d")
    end = datetime.datetime.strptime(date_range[1], "%Y-%m-%d")
    date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]

    for date in date_generated:
        date_str = date.strftime("%Y-%m-%d")

        filename = f"./data/en_{date_str}_output.csv"
        df_data = pd.read_csv(filename)
        print(f"df_data.shape: {df_data.shape}")
        df_data = df_data.replace(r'^\s*$', np.nan, regex=True)
        #X = total_dataframe.append(df_data, ignore_index=True)
        X = pd.concat([X, df_data], ignore_index=True)
        print(X.isnull().sum())

tweets_filename = f"./data/en_2021-01-01_output.csv"
X = pd.read_csv(tweets_filename)

X, X_test, y, y_test = train_test_split(X, y, test_size=0.3,train_size=0.7)

dataset = Dataset.from_pandas(X)

Dataset({
   features: ['g_values'],
   num_rows: 2
})

raw_train_ds = Dataset.from_json("data/sentiments.train.jsonlines")
raw_val_ds = Dataset.from_json("data/sentiments.test.jsonlines")
raw_test_ds = Dataset.from_json("data/sentiments.test.jsonlines")

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader

BASE_MODEL = "camembert-base"
LEARNING_RATE = 2e-5
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 20

# Let's name the classes 0, 1, 2, 3, 4 like their indices
id2label = {k:k for k in range(5)}
label2id = {k:k for k in range(5)}

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, id2label=id2label, label2id=label2id)

## {'id': 457, 'text': 'Trop désagréable au téléphone 😡! ! !', 'uuid': '91c4efaaada14a1b9b050268185b6ae5', 'score': 1}

ds = {"train": raw_train_ds, "validation": raw_val_ds, "test": raw_test_ds}

def preprocess_function(examples):
    label = examples["score"] 
    examples = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
    examples["label"] = label
    return examples

for split in ds:
    ds[split] = ds[split].map(preprocess_function, remove_columns=["id", "uuid", "text", "score"])


import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)



from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="../models/camembert-fine-tuned-regression",
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


from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("model_bert_classification.model")