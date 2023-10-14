import pandas as pd
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorWithPadding, Trainer, TrainingArguments
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Specify the file path for the CSV file
csv_file_path = 'aggregated_tweets.csv'

# Load the data from the CSV file into a DataFrame
aggregated_df = pd.read_csv(csv_file_path)

# Define constants
BASE_MODEL = "gpt2"
LEARNING_RATE = 2e-5
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3   #20

# Initialize GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL)
model = GPT2LMHeadModel.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Define a custom dataset
def preprocess_function(examples):
    text = examples["clean_tweets"]
    label = examples["g_value"]
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
    inputs["labels"] = torch.tensor(label, dtype=torch.float32)
    return inputs

# Split the data into train, validation, and test sets
train_df, test_df = train_test_split(aggregated_df, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Preprocess the datasets
train_dataset = train_dataset.map(preprocess_function)
test_dataset = test_dataset.map(preprocess_function)

# Define the data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, max_length=MAX_LENGTH)

# Training arguments
training_args = TrainingArguments(
    output_dir="../models/gpt2-fine-tuned-regression",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Define a custom trainer for regression
class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs).logits
        
        # Reshape the outputs tensor
        outputs = outputs.view(-1, 1)
        
        loss = torch.nn.functional.mse_loss(outputs, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics_for_regression(eval_pred):
    predictions = eval_pred.predictions  # Extract the generated text
    labels = eval_pred.label_ids.tolist()
    # You can process predictions here if needed, e.g., convert text to numbers
    # For simplicity, let's assume predictions are already numbers
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
    }

# Initialize the trainer
trainer = RegressionTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics_for_regression,
)


# Train the model
trainer.train()
trainer.save_model("model_gpt2_regression")

# Evaluate the model
results = trainer.evaluate()
print("Evaluation results:", results)
