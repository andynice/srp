# -*- coding: utf-8 -*-
"""t5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1163XigIOHlVlg4JDgss0QcTJpAQhcbQ2
"""

import os
import pandas as pd
import numpy as np
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, Dataset

# Directory containing CSV files with cleaned tweets
data_dir = 'your_data_directory'

# Initialize empty lists to store tweets and G values
tweets = []
growth_rate = []

# Loop through CSV files in the directory
for filename in os.listdir(data_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_dir, filename)
        data = pd.read_csv(file_path)

        # Extract tweets and G values from each CSV file
        tweets.extend(data['clean_tweets'].astype(str).values)
        growth_rate.extend(data['G value'].astype(float).values)

# Convert 'G value' to log-scale (optional but can help with regression)
growth_rate = np.log1p(growth_rate)

# Split data into train, validation, and test sets
train_tweets, test_tweets, train_growth_rate, test_growth_rate = train_test_split(
    tweets, growth_rate, test_size=0.2, random_state=42
)
val_tweets, test_tweets, val_growth_rate, test_growth_rate = train_test_split(
    test_tweets, test_growth_rate, test_size=0.5, random_state=42
)

# Define a custom dataset for T5
class CustomDataset(Dataset):
    def __init__(self, tweets, targets, tokenizer, max_length):
        self.tweets = tweets
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        input_text = "On [Date], the following tweets were posted about COVID-19: " + self.tweets[idx]
        target_text = "The global growth rate in confirmed COVID-19 cases on [Date] is " + str(self.targets[idx])
        input_ids = self.tokenizer.encode(input_text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        target_ids = self.tokenizer.encode(target_text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        return {
            'input_ids': input_ids.view(-1),
            'attention_mask': (input_ids != 0).view(-1),
            'labels': target_ids.view(-1),
        }

# Initialize the T5 tokenizer and model
model_name = 't5-small'  # You can use other T5 variants like 't5-base' for more capacity
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Define training parameters
batch_size = 8
max_length = 128
num_epochs = 5
learning_rate = 1e-4

# Create custom datasets and data loaders
train_dataset = CustomDataset(train_tweets, train_growth_rate, tokenizer, max_length)
val_dataset = CustomDataset(val_tweets, val_growth_rate, tokenizer, max_length)
test_dataset = CustomDataset(test_tweets, test_growth_rate, tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * num_epochs)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    # Calculate validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch + 1}/{num_epochs} - Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")

# Evaluation on test data
model.eval()
predictions = []
true_values = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)
        predictions.extend(outputs.tolist())
        true_values.extend(labels.tolist())

# Convert predictions back to linear scale if you used log-scale earlier
predictions = np.expm1(predictions)

# Calculate regression metrics
mse = mean_squared_error(true_values, predictions)
mae = mean_absolute_error(true_values, predictions)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")