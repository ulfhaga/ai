from transformers import pipeline

generator = pipeline("text-generation",device="cuda")
generator("In this course, we will teach you how to")