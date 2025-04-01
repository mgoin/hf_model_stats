import requests
import pandas as pd
from datetime import datetime
import os
import argparse

def safe_json(response):
    try:
        return response.json()
    except Exception as e:
        print(f"Error parsing JSON from request URL: {response.url}")
        print("Status code:", response.status_code)
        print("Response text:", response.text)
        raise e

def get_author_model_stats(author: str):
    url = f"https://huggingface.co/api/models?author={author}"
    print(f"Fetching models list from: {url}")
    response = requests.get(url)
    print("Status code for models list:", response.status_code)
    data = safe_json(response)

    models = []
    for model in data:
        model_id = model["id"]
        likes = model["likes"]
        downloads = model["downloads"]

        details_url = f"https://huggingface.co/api/models/{model_id}?expand[]=downloadsAllTime"
        print(f"\nFetching details for model: {model_id}")
        print("Details URL:", details_url)
        details_response = requests.get(details_url)
        print("Status code for model details:", details_response.status_code)
        details_data = safe_json(details_response)
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
