from transformers import pipeline # type: ignore

generator = pipeline("text2text-generation",device="cuda")
text = "translate from English to German: I'm very happy"
print(text)
result = generator(text)
print(result[0]['generated_text'])

print('')

print('What is 42 ?')
result = generator("question: What is 42 ? context: 42 is just a number")
print(result[0]['generated_text'])

print('What is 66 ?')
result = generator("question: What is 42 ? context: Route 66 is a highway in USA")
print(result[0]['generated_text'])


