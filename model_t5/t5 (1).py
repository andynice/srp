# -*- coding: utf-8 -*-
"""t5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18hz80wWbrUDb_5lvVCa6IXk5iJx9tpvW
"""

!pip install transformers

!pip install sentencepiece

import os
import pandas as pd
import numpy as np
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset

# Directory containing CSV files with cleaned tweets
data_dir = '/content/drive/MyDrive/cleaned tweets'

# Initialize empty lists to store data for each date
data_for_dates = []

# Loop through CSV files in the directory
for filename in os.listdir(data_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_dir, filename)
        data = pd.read_csv(file_path)

        # Extract tweets and G values for each date and store as a dictionary
        date_data = {
            'date': filename[:-4],  # Remove ".csv" to get the date
            'tweets': data['clean_tweets'].astype(str).values,
            'growth_rate': data['G value'].astype(float).values
        }
        data_for_dates.append(date_data)

# Define a custom dataset for T5
class CustomDataset(Dataset):
    def __init__(self, data_for_dates, tokenizer, max_length):
        self.data_for_dates = data_for_dates
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_for_dates)

    def __getitem__(self, idx):
        date_data = self.data_for_dates[idx]
        input_text = f"On {date_data['date']}, the following tweets were posted about COVID-19: " + ' '.join(date_data['tweets'])
        target_text = f"The global growth rate in confirmed COVID-19 cases on {date_data['date']} is " + str(date_data['growth_rate'])
        input_ids = self.tokenizer.encode(input_text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        target_ids = self.tokenizer.encode(target_text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        return {
            'input_ids': input_ids.view(-1),
            'attention_mask': (input_ids != 0).view(-1),
            'labels': target_ids.view(-1),
        }

# Initialize the T5 tokenizer and model
model_name = 't5-large'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Training parameters
batch_size = 1
max_length = 512
num_epochs = 5
learning_rate = 1e-4

# Create custom datasets and data loaders
dataset = CustomDataset(data_for_dates, tokenizer, max_length)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(data_loader) * num_epochs)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in data_loader:
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

    avg_train_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch + 1}/{num_epochs} - Avg Train Loss: {avg_train_loss:.4f}")

# Save the trained model
model.save_pretrained('your_model_directory')