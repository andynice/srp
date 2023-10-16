from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

# MODEL DEFINITION
BASE_MODEL = "allenai/longformer-base-4096"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)

tokenizer.save_pretrained("./longformer-base-4096")
model.save_pretrained("./longformer-base-4096")