import requests
import json
import time
import math
from pathlib import Path
import pandas as pd
from tqdm.notebook import tqdm
from datasets import load_dataset

url = "https://api.github.com/repos/huggingface/datasets/issues?page=1&per_page=1"

GITHUB_TOKEN = "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" 

headers = {"Authorization": f"token {GITHUB_TOKEN}"}


def fetch_issues(
    owner="huggingface",
    repo="datasets",
    num_issues=10_000,
    rate_limit=5_000,
    issues_path=Path("."),
):
    if not issues_path.is_dir():
        issues_path.mkdir(exist_ok=True)

    batch = []
    all_issues = []
    per_page = 100  # Number of issues to return per page
    num_pages = math.ceil(num_issues / per_page)
    base_url = "https://api.github.com/repos"

    for page in tqdm(range(num_pages)):
        # Query with state=all to get both open and closed issues
        query = f"issues?page={page}&per_page={per_page}&state=all"
        issues = requests.get(f"{base_url}/{owner}/{repo}/{query}", headers=headers)
        batch.extend(issues.json())

        if len(batch) > rate_limit and len(all_issues) < num_issues:
            all_issues.extend(batch)
            batch = []  # Flush batch for next time period
            print(f"Reached GitHub rate limit. Sleeping for one hour ...")
            time.sleep(60 * 60 + 1)

    all_issues.extend(batch)
    df = pd.DataFrame.from_records(all_issues)
    df.to_json(f"{issues_path}/{repo}-issues.jsonl", orient="records", lines=True)
    print(
        f"Downloaded all the issues for {repo}! Dataset stored at {issues_path}/{repo}-issues.jsonl"
    )

# Depending on your internet connection, this can take several minutes to run...
print("\n")

if False :
  fetch_issues()
  issues_dataset = load_dataset("json", data_files="datasets-issues.jsonl", split="train")
else:
  response = requests.get(url)
  print(json.dumps(response.json()))



print("\n")

issue_number = 2792
url = f"https://api.github.com/repos/huggingface/datasets/issues/{issue_number}/comments"
response = requests.get(url, headers=headers)

#print(json.dumps(response.json()))


print("\n")

def get_comments(issue_number):
    url = f"https://api.github.com/repos/huggingface/datasets/issues/{issue_number}/comments"
    response = requests.get(url, headers=headers)
    return [r["body"] for r in response.json()]


# Test our function works as expected
print(json.dumps(get_comments(2792)))


# Depending on your internet connection, this can take a few minutes...
issues_with_comments_dataset = issues_dataset.map(
    lambda x: {"comments": get_comments(x["number"])}
)
