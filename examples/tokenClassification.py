from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from datasets import load_dataset
import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"




# Load the CoNLL-2003 dataset
dataset = load_dataset("conll2003",trust_remote_code=True)

# Load pre-trained model and tokenizer
model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

# Get the list of labels
label_list = dataset["train"].features["ner_tags"].feature.names

# Tokenize the dataset
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Metrics calculation
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "accuracy": accuracy_score(true_labels, true_predictions),
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    no_cuda=True
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Fine-tune the model
trainer.train()

# Evaluate the model
evaluation_results = trainer.evaluate()
print("Evaluation Results:", evaluation_results)

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_ner_model")
tokenizer.save_pretrained("./fine_tuned_ner_model")

# Test the model on a custom sentence
def test_ner(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    print(f"\nSentence: {sentence}")
    print("Named Entities:")
    for token, prediction in zip(tokens, predictions[0]):
        if prediction != 0:  # 0 is usually the 'O' (Outside) tag
            print(f"{token}: {label_list[prediction]}")

# Test the fine-tuned model
test_ner("John Smith works at Microsoft in Seattle.")
test_ner("The Eiffel Tower is located in Paris, France.")