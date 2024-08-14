from transformers import TFAutoModelForSequenceClassification

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tf_model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

tf_outputs = tf_model(tf_batch)