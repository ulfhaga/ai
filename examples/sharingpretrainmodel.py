from transformers import TrainingArguments, AutoModelForMaskedLM, AutoTokenizer

import os
# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'




'''
training_args = TrainingArguments(
    "bert-finetuned-mrpc", save_strategy="epoch", push_to_hub=True
)
'''



checkpoint = "camembert-base"

model = AutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model.save_pretrained("<path_to_dummy_folder>")
tokenizer.save_pretrained("<path_to_dummy_folder>")

'''
model.push_to_hub("dummy-model")
tokenizer.push_to_hub("dummy-model")
'''