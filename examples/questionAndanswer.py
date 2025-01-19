# Description: This script is used to train a BERT model on a given dataset. https://huggingface.co/learn/nlp-course/chapter7/7

# Import necessary libraries
from datasets import load_dataset
raw_datasets = load_dataset("squad")
import torch
from transformers import AutoTokenizer

yellow = "\033[0;33m"
color_off = "\033[0m" 
red = "\033[0;31m"
green = "\033[0;32m"


print(raw_datasets);

print("Context: ", raw_datasets["train"][0]["context"])
print("Question: ", raw_datasets["train"][0]["question"])
print("Answer: ", raw_datasets["train"][0]["answers"])


print(raw_datasets["train"].filter(lambda x: len(x["answers"]["text"]) != 1))

print(raw_datasets["validation"][0]["answers"])
print("\nValidation:\n")
print(raw_datasets["validation"][2]["answers"])
print(raw_datasets["validation"][2]["context"])
print(raw_datasets["validation"][2]["question"])



model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
print("tokenizer.is_fast:",tokenizer.is_fast)

context = raw_datasets["train"][0]["context"]
question = raw_datasets["train"][0]["question"]

#inputs = tokenizer(question, context)
#inputs = tokenizer(question, context, return_tensors="pt")


inputs = tokenizer(
    question,
    context,
    max_length=100,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True
)

#print (yellow,"\ninput_ids:\n",color_off,tokenizer.decode(inputs["input_ids"]))

print (yellow,"\nDecode ds:\n",color_off)
for ids in inputs["input_ids"]:
    print("\n",tokenizer.decode(ids))


inputs = tokenizer(
    question,
    context,
    max_length=100,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)

print(yellow,"\ninputs.keys:\n",color_off,inputs.keys())

print(yellow,"overflow_to_sample_mapping:\n",inputs["overflow_to_sample_mapping"],color_off)

inputs = tokenizer(
    raw_datasets["train"][2:6]["question"],
    raw_datasets["train"][2:6]["context"],
    max_length=100,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)

print(f"The 4 examples gave {len(inputs['input_ids'])} features.")
print(f"Here is where each comes from overflow_to_sample_mapping: {inputs['overflow_to_sample_mapping']}.")

print( "6 context:\n", raw_datasets["train"][5:6]["context"])