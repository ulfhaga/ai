from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorWithPadding

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

raw_datasets = load_dataset("glue", "mrpc");
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

samples = tokenized_datasets["train"][:8]
#print ("\n\n")
#print ("samples")
#print (samples)
#print ("\n\n")
# Removes the columns idx, sentence1, and sentence2
samplesNew = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
print ("samplesNew")
#print (samplesNew)
#print ("\n\n")

print([len(x) for x in samplesNew["input_ids"]])

# Dynamic padding means the samples in this batch should all be padded to a length of 67, the maximum length inside the batch. 
print ("samplesNew with pad 67")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
batch = data_collator(samplesNew)
print ({k: v.shape for k, v in batch.items()})

