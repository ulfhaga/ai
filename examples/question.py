from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model_name = "NousResearch/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=64, do_sample=True, temperature=0.7, device='cpu')

question = "Question: What is the difference between a car and a truck?"
answer = pipe(question)
print(question)
print(answer[0]['generated_text'])
