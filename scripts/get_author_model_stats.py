import requests
import pandas as pd
from datetime import datetime
import os
import argparse
import time

# Create a persistent session for all requests
session = requests.Session()

def get_json_with_retries(url, retries=3, backoff_factor=10):
    """
    Attempt to GET a URL and return its JSON content.
    If a 429 (rate limit) response is encountered, wait with exponential backoff.
    """
    for attempt in range(retries):
        response = session.get(url)
        if response.status_code == 429:
            wait_time = backoff_factor * (attempt + 1)
            print(f"Rate limit encountered at {url}. Sleeping for {wait_time} seconds (attempt {attempt+1}/{retries}).")
            time.sleep(wait_time)
            continue
        elif not response.ok:
            print(f"Request to {url} failed with status code {response.status_code}. Response text:")
            print(response.text)
            response.raise_for_status()

        try:
            return response.json()
        except Exception as e:
            print(f"Error parsing JSON from {url}: {e}")
            print("Response text:")
            print(response.text)
            # Optionally, back off before retrying JSON parsing failures.
            time.sleep(backoff_factor * (attempt + 1))
    raise Exception(f"Failed to retrieve valid JSON from {url} after {retries} attempts")

def get_author_model_stats(author: str):
    models_url = f"https://huggingface.co/api/models?author={author}"
    print(f"Fetching models list from: {models_url}")
    data = get_json_with_retries(models_url)

    models = []
    for model in data:
        model_id = model["id"]
        likes = model["likes"]
        downloads = model["downloads"]

        details_url = f"https://huggingface.co/api/models/{model_id}?expand[]=downloadsAllTime"
        print(f"\nFetching details for model: {model_id}")
        print("Details URL:", details_url)
        details_data = get_json_with_retries(details_url)
        downloads_all_time = details_data.get("downloadsAllTime", 0)

        models.append(
            {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "model_id": model_id,
                "likes": likes,
                "downloads": downloads,
                "downloads_all_time": downloads_all_time,
            }
        )

    return pd.DataFrame(models)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get model stats for a Hugging Face author")
    parser.add_argument("--author", type=str, default="neuralmagic", 
                        help="Author name on Hugging Face (default: neuralmagic)")
    args = parser.parse_args()
    
    author = args.author
    output_file = f"model_stats_{author}.csv"
    
    print(f"\nFetching model stats for author: {author}")
    new_data = get_author_model_stats(author)

    if os.path.exists(output_file):
        print(f"\nUpdating existing file: {output_file}")
        existing_data = pd.read_csv(output_file)
        updated_data = pd.concat([existing_data, new_data]).drop_duplicates(
            subset=["date", "model_id"], keep="last"
        )
    else:
        print(f"\nCreating new file: {output_file}")
        updated_data = new_data

    updated_data.to_csv(output_file, index=False)
    print(f"\nNew Stats for {author}:")
    print(new_data.to_markdown())
