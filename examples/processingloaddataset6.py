from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorWithPadding

print("Starting")

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

raw_datasets = load_dataset("glue", "mrpc");
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
samples = tokenized_datasets["train"][:8]
batch = data_collator(samples)
print(batch)
{print(k): v.shape for k, v in batch.items()}



#print(tokenized_datasets)
