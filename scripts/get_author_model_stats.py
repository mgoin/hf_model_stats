import requests
import pandas as pd
from datetime import datetime
import os
import argparse


def get_author_model_stats(author: str):
    url = f"https://huggingface.co/api/models?author={author}"
    response = requests.get(url)
    data = response.json()

    models = []
    for model in data:
        model_id = model["id"]
        likes = model["likes"]
        downloads = model["downloads"]

        # Get downloadsAllTime
        details_url = (
            f"https://huggingface.co/api/models/{model_id}?expand[]=downloadsAllTime"
        )
        details_response = requests.get(details_url)
        details_data = details_response.json()
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
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Get model stats for a Hugging Face author")
    parser.add_argument("--author", type=str, default="neuralmagic", 
                       help="Author name on Hugging Face (default: neuralmagic)")
    args = parser.parse_args()
    
    author = args.author
    output_file = f"model_stats_{author}.csv"
    
    print(f"Fetching model stats for author: {author}")
    new_data = get_author_model_stats(author)

    if os.path.exists(output_file):
        print(f"Updating existing file: {output_file}")
        existing_data = pd.read_csv(output_file)
        updated_data = pd.concat([existing_data, new_data]).drop_duplicates(
            subset=["date", "model_id"], keep="last"
        )
    else:
        print(f"Creating new file: {output_file}")
        updated_data = new_data

    updated_data.to_csv(output_file, index=False)
    print(f"\nNew Stats for {author}:")
    print(new_data.to_markdown())