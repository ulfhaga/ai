'''
wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip"
unzip drugsCom_raw.zip
'''

from datasets import load_dataset
import html
from transformers import AutoTokenizer
from datasets import Dataset
from datasets import load_from_disk

yellow = "\033[0;33m"
color_off = "\033[0m" 
red = "\033[0;31m"
green = "\033[0;32m"


data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
# \t is the tab character in Python
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
# Peek at the first few examples
# print(drug_sample[:3])




for split in drug_dataset.keys():
    assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0")) , "Fel"



drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0", new_column_name="patient_id"
)
print(drug_dataset)

def filter_nones(x):
    return x["condition"] is not None

def lowercase_condition(example):
    return {"condition": example["condition"].lower()}


#drug_dataset.filter(filter_nones)
#drug_dataset.map(lowercase_condition)

drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)
drug_dataset = drug_dataset.map(lowercase_condition)
# Check that lowercasing worked
print(drug_dataset)
print(drug_dataset["train"]["condition"][:3])


def compute_review_length(example):
    return {"review_length": len(example["review"].split())}


drug_dataset = drug_dataset.map(compute_review_length)
# Inspect the first training example
print("\ncompute_review_length:\n ",drug_dataset["train"][0])


print("\nSort\n",drug_dataset["train"].sort("review_length")[:3])


drug_dataset = drug_dataset.filter(lambda x: x["review_length"] > 30)
print("\nFilter review_length:\n",drug_dataset.num_rows)


drug_dataset = drug_dataset.map(lambda x: {"review": html.unescape(x["review"])})



new_drug_dataset = drug_dataset.map(
    lambda x: {"review": [html.unescape(o) for o in x["review"]]}, batched=True
)

print("\nnew_drug_dataset:\n", new_drug_dataset)





tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["review"], truncation=True)


def tokenize_and_split(examples):
    return tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True
    )

result = tokenize_and_split(drug_dataset["train"][0])
len_array = [len(inp) for inp in result["input_ids"]]
print (len_array)

# error tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)

tokenized_dataset = drug_dataset.map(
    tokenize_and_split, batched=True, remove_columns=drug_dataset["train"].column_names
)
    
# print(drug_dataset["train"].column_names)

print(len(tokenized_dataset["train"]), len(drug_dataset["train"]))



def tokenize_and_split_2(examples):
    result = tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
    # Extract mapping between new and old indices
    sample_map = result.pop("overflow_to_sample_mapping")
    for key, values in examples.items():
        result[key] = [values[i] for i in sample_map]
    return result


tokenized_dataset = drug_dataset.map(tokenize_and_split_2, batched=True)

print(yellow,"\ntokenized_dataset:\n",color_off ,tokenized_dataset)


drug_dataset.set_format("pandas")

print (yellow,"pandas:",color_off,drug_dataset["train"][:3])


train_df = drug_dataset["train"][:]

frequencies = (
    train_df["condition"]
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={"index": "condition", "condition": "frequency"})
)
print (yellow,"frequencies.head:", color_off, frequencies.head())


freq_dataset = Dataset.from_pandas(frequencies)
print(freq_dataset)


drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
# Rename the default "test" split to "validation"
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
# Add the "test" set to our `DatasetDict`
drug_dataset_clean["test"] = drug_dataset["test"]
print (drug_dataset_clean)

drug_dataset_clean.save_to_disk("/tmp/drug-reviews")




drug_dataset_reloaded = load_from_disk("/tmp/drug-reviews")
print (drug_dataset_reloaded)


for split, dataset in drug_dataset_clean.items():
    dataset.to_json(f"/tmp/drug/drug-reviews-{split}.jsonl")


data_files = {
    "train": "/tmp/drug/drug-reviews-train.jsonl",
    "validation": "/tmp/drug/drug-reviews-validation.jsonl",
    "test": "/tmp/drug/drug-reviews-test.jsonl",
}

print(yellow,"drug_dataset_reloaded:\n")

drug_dataset_reloaded = load_dataset("json", data_files=data_files)    