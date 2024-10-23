from transformers import AutoTokenizer
from datasets import load_dataset
print("Starting")

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

raw_datasets = load_dataset("glue", "mrpc");
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
#tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])

tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)

print(tokenized_dataset)
