from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding


from torch.utils.data import DataLoader

from transformers import AutoModelForSequenceClassification

from transformers import AdamW

from transformers import get_scheduler

import torch









raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names

print("Columns names:")
print(tokenized_datasets["train"].column_names);

# fortsätt här


train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)
print("Length train_dataloader: ")
print(len(train_dataloader))

for batch in train_dataloader:
    break

# Dictionary comprehension
dictionary = {k: v.shape for k, v in batch.items()}

print("Length:",len(dictionary))
print("Shapes",dictionary)


model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
outputs = model(**batch)
print("loss:",outputs.loss, " shape: ", outputs.logits.shape)

optimizer = AdamW(model.parameters(), lr=5e-5)


num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print("Steps:",num_training_steps)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print("device",device);
