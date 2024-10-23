from datasets import load_dataset
print("Starting")
raw_datasets = load_dataset("glue", "mrpc");
print(raw_datasets)
raw_train_dataset = raw_datasets["train"]
print("\n---0---\n")
print(raw_train_dataset[0])
print("\n---1---\n")
print(raw_train_dataset[1])
print("\n---2---\n")
print(raw_train_dataset[2])
print("\n---3---\n")
print(raw_train_dataset[3])

print("\n---features---\n")
print(raw_train_dataset.features)
