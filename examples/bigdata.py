from transformers import AutoTokenizer
from datasets import load_dataset
import psutil
import timeit

yellow = "\033[0;33m"
color_off = "\033[0m" 
red = "\033[0;31m"
green = "\033[0;32m"



# This takes a few minutes to run, so go grab a tea or coffee while you wait :)
#data_files = "https://the-eye.eu/public/AI/pile_preliminary_components/PUBMED_title_abstracts_2019_baseline.jsonl.zst"
data_files =  "https://storage.googleapis.com/kaggle-data-sets/944776/1600797/compressed/PUBMED_title_abstracts_2019_baseline.jsonl.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250101%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250101T150923Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=7c774aabdecd3d865d415dbd4c3d9394045273b59391514fbcda53bb39d46002f655e49c77866696e5a773462d5ca31391cb1b2692b2b0f70b126a49aab6d32dddd6591a90eb497f501a7f8a479251ffac7d72223e108659ba63e430d7d615a2decc510ecb702c08c3a12bb30f811fcb06229d4779a8e7c19d3c1c7c662ab23550a207bf774d57e581f60b75bfdaf64802a52c0e328c1b56122c31b0009f0b2d4e4f95d773e344e36fec97456c0160d9fad4ab7da500a414838efa214374071d2a53fb95ccc568c7f79d7bc6b1929c09a7f028093834347f106067afe90e600153ccbadc56bc24f75faf33b2a57c7c4eca96ac7f1bf48fff04bf95f7727a2afd"
#data_files = "PUBMED_title_abstracts_2019_baseline.jsonl"
pubmed_dataset = load_dataset("json", data_files=data_files, split="train")
print(pubmed_dataset)
print(pubmed_dataset[0])

# Process.memory_info is expressed in bytes, so convert to megabytes
print(green,f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

print(f"Number of files in dataset : {pubmed_dataset.dataset_size}")
size_gb = pubmed_dataset.dataset_size / (1024**3)
print(f"Dataset size (cache file) : {size_gb:.2f} GB",color_off)

code_snippet = """batch_size = 1000

for idx in range(0, len(pubmed_dataset), batch_size):
    _ = pubmed_dataset[idx:idx + batch_size]
"""

time = timeit.timeit(stmt=code_snippet, number=1, globals=globals())
print(
    f"Iterated over {len(pubmed_dataset)} examples (about {size_gb:.1f} GB) in "
    f"{time:.1f}s, i.e. {size_gb/time:.3f} GB/s"
)


pubmed_dataset_streamed = load_dataset(
    "json", data_files=data_files, split="train", streaming=True
)

# access the first element 
element = next(iter(pubmed_dataset_streamed))
print(element)


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenized_dataset = pubmed_dataset_streamed.map(lambda x: tokenizer(x["text"]))
first_tokenized = next(iter(tokenized_dataset))
print("\nfirst_tokenized:\n",first_tokenized)

shuffled_dataset = pubmed_dataset_streamed.shuffle(buffer_size=10_000, seed=42)
shuffled = next(iter(shuffled_dataset))
print("shuffled:\n",shuffled)

dataset_head = pubmed_dataset_streamed.take(5)
list = list(dataset_head)
print(yellow,"List 5:\n", color_off, list)


# Skip the first 1,000 examples and include the rest in the training set
train_dataset = shuffled_dataset.skip(1000)
# Take the first 1,000 examples for the validation set
first_1000 = validation_dataset = shuffled_dataset.take(1000)
print(yellow,"first_1000:\n",first_1000, color_off)

