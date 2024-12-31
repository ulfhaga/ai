'''
wget https://github.com/crux82/squad-it/raw/master/SQuAD_it-train.json.gz;
wget https://github.com/crux82/squad-it/raw/master/SQuAD_it-test.json.gz;
gzip -dkv SQuAD_it-*.json.gz;
'''

from datasets import load_dataset

# squad_it_dataset = load_dataset("json", data_files="SQuAD_it-train.json", field="data")

# print(squad_it_dataset)

# print(squad_it_dataset["train"][0])


data_files = {"train": "SQuAD_it-train.json", "test": "SQuAD_it-test.json"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
print(squad_it_dataset)