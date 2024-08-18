from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "NousResearch/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=64, do_sample=True, temperature=0.7)

question = "Question: What is the difference between a car and a truck?"
answer = pipe(question)
print(question)
print(answer[0]['generated_text'])
