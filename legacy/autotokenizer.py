'''
A tokenizer is responsible for preprocessing text into an array of numbers as inputs to a model. 
There are multiple rules that govern the tokenization process, including how to split a word and at what level words 
should be split (learn more about tokenization in the tokenizer summary). 
The most important thing to remember is you need to instantiate a tokenizer with the same model name to ensure you’re using the same tokenization rules a model was pretrained with.
'''
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoding = tokenizer("We are very happy to show you the 🤗 Transformers library.")
print(encoding)


print('---------- tf_batch  ------------')

tf_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="tf",
)

print(tf_batch)

print('---------------------- TFAutoModelForSequenceClassification ------------------')



model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tf_model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

tf_outputs = tf_model(tf_batch)

print(tf_outputs)



print('---------------------   Apply the softmax function to the logits to retrieve the probabilities ----------')
import tensorflow as tf

tf_predictions = tf.nn.softmax(tf_outputs.logits, axis=-1)
tf_predictions


print('---------------------   save it with its tokenizer ----------')
tf_save_directory = "./tf_save_pretrained"
tokenizer.save_pretrained(tf_save_directory)
tf_model.save_pretrained(tf_save_directory)

print('-------------  reload trained model  -------------------------');

tf_model = TFAutoModelForSequenceClassification.from_pretrained("./tf_save_pretrained")


