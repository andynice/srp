from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

# MODEL DEFINITION
BASE_MODEL = "allenai/led-base-16384"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)

tokenizer.save_pretrained("./led-base-16384")
model.save_pretrained("./led-base-16384")