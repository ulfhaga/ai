
from datasets import Dataset
from transformers import TFAutoModelForSequenceClassification

model = TFAutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")


def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])  # doctest: +SKIP

dataset = dataset.map(tokenize_dataset)  

tf_dataset = model.prepare_tf_dataset(
    dataset["train"], batch_size=16, shuffle=True, tokenizer=tokenizer
)  # doctest: +SKIP

from tensorflow.keras.optimizers import Adam

model.compile(optimizer='adam')  # No loss argument!
model.fit(tf_dataset)  # doctest: +SKIP