from transformers import pipeline

transcriber = pipeline(task="automatic-speech-recognition")

transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")





