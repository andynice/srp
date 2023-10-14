from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

# MODEL DEFINITION
# BASE_MODEL = "digitalepidemiologylab/covid-twitter-bert"
BASE_MODEL = "digitalepidemiologylab/covid-twitter-bert-v2"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)

# tokenizer.save_pretrained("./covid-twitter-bert")
# model.save_pretrained("./covid-twitter-bert")

tokenizer.save_pretrained("./covid-twitter-bert-v2")
model.save_pretrained("./covid-twitter-bert-v2")