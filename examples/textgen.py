from transformers import pipeline # type: ignore


generator = pipeline("text-generation",device="cuda")
generator("In this course, we will teach you how to")


