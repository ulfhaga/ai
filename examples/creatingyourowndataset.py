import requests
import json
import jq
url = "https://api.github.com/repos/huggingface/datasets/issues?page=1&per_page=1"
response = requests.get(url)
print(json.dumps(response.json()))


