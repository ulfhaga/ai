from transformers import pipeline # type: ignore


generator = pipeline("text-generation",device="cuda")
result = generator("In this course, we will teach you how to")
print(result)


