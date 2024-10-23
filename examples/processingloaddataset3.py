from transformers import AutoTokenizer
from datasets import load_dataset
print("Starting")

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

raw_datasets = load_dataset("glue", "mrpc");
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

tokenized_dataset= tokenize_function(raw_datasets["train"])

print(tokenized_dataset)
