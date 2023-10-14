from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

# MODEL DEFINITION
BASE_MODEL = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)

tokenizer.save_pretrained("./gpt2")
model.save_pretrained("./gpt2")
