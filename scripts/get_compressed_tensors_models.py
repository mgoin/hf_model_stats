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
    Attempt to GET a URL and return its JSON content and response headers.
    If a 429 (rate limit) response is encountered, wait with exponential backoff.
    """
    for attempt in range(retries):
        response = session.get(url)
        if response.status_code == 429:
            wait_time = backoff_factor * (attempt + 1)
            print(
                f"Rate limit encountered at {url}. Sleeping for {wait_time} seconds (attempt {attempt+1}/{retries})."
            )
            time.sleep(wait_time)
            continue
        elif not response.ok:
            print(
                f"Request to {url} failed with status code {response.status_code}. Response text:"
            )
            print(response.text)
            response.raise_for_status()

        try:
            return response.json(), response.headers
        except Exception as e:
            print(f"Error parsing JSON from {url}: {e}")
            print("Response text:")
            print(response.text)
            # Optionally, back off before retrying JSON parsing failures.
            time.sleep(backoff_factor * (attempt + 1))
    raise Exception(
        f"Failed to retrieve valid JSON from {url} after {retries} attempts"
    )


def parse_link_header(link_header):
    """
    Parse the Link header to extract the next page URL.
    Link header format: '<url>; rel="next", <url>; rel="last"'
    """
    if not link_header:
        return None

    links = {}
    for link in link_header.split(","):
        parts = link.strip().split(";")
        if len(parts) == 2:
            url = parts[0].strip("<>")
            rel = parts[1].strip().split("=")[1].strip('"')
            links[rel] = url

    return links.get("next")


def get_compressed_tensors_models():
    """
    Get all models with compressed-tensors tag using pagination to fetch all results.
    """
    all_models = []

    # Start with the first page
    base_url = "https://huggingface.co/api/models"
    params = {
        "filter": "compressed-tensors",
        "expand": ["downloads", "downloadsAllTime"],
        "limit": 500,  # Use a reasonable page size
    }

    # Build initial URL
    current_url = f"{base_url}?{'&'.join([f'{k}={v}' if not isinstance(v, list) else '&'.join([f'{k}={item}' for item in v]) for k, v in params.items()])}"
    page = 1

    while current_url:
        print(f"Fetching page {page}: {current_url}")

        try:
            data, headers = get_json_with_retries(current_url)
            print(f"Found {len(data)} models on page {page}")
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            break

        # Process models on this page
        for model in data:
            model_id = model.get("id", "")
            likes = model.get("likes", 0)
            downloads = model.get("downloads", 0)
            downloads_all_time = model.get("downloadsAllTime", 0)
            trending_score = model.get("trendingScore", 0)

            # Extract author from model_id (format is typically "author/model-name")
            author = model_id.split("/")[0] if "/" in model_id else ""

            all_models.append(
                {
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "model_id": model_id,
                    "author": author,
                    "likes": likes,
                    "downloads": downloads,
                    "downloads_all_time": downloads_all_time,
                    "trending_score": trending_score,
                }
            )

        # Check for next page using Link header
        link_header = headers.get("Link") or headers.get("link")
        next_url = parse_link_header(link_header)

        if next_url:
            current_url = next_url
            page += 1
        else:
            print(f"No more pages found. Finished at page {page}")
            break

        # Safety break to avoid infinite loops
        if page > 50:  # Adjust this limit as needed
            print(f"Reached maximum page limit (50)")
            break

    print(f"Total models collected: {len(all_models)}")
    return pd.DataFrame(all_models)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get all models with compressed-tensors tag from Hugging Face"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="compressed_tensors_models.csv",
        help="Output CSV filename",
    )
    args = parser.parse_args()

    output_file = args.output

    print("Fetching all models with compressed-tensors tag...")
    new_data = get_compressed_tensors_models()

    if new_data.empty:
        print("No models found.")
        exit(1)

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

    print(f"\nFound {len(new_data)} compressed-tensors models:")
    print(f"Total unique models in dataset: {len(updated_data)}")

    # Show summary stats
    print(f"\nSummary of found models:")
    print(f"- Total models: {len(new_data)}")
    print(f"- Unique authors: {new_data['author'].nunique()}")
    print(f"- Total likes: {new_data['likes'].sum()}")
    print(f"- Total downloads (30 day): {new_data['downloads'].sum()}")
    print(f"- Total downloads (all time): {new_data['downloads_all_time'].sum()}")

    print(f"\nTop 10 models by all-time downloads:")
    top_models = new_data.nlargest(10, "downloads_all_time")[
        ["model_id", "author", "downloads_all_time", "likes"]
    ]
    print(top_models.to_string(index=False))

    print(f"\nTop 10 authors by total downloads:")
    author_stats = (
        new_data.groupby("author")
        .agg({"downloads_all_time": "sum", "model_id": "count"})
        .rename(columns={"model_id": "model_count"})
        .sort_values("downloads_all_time", ascending=False)
    )
    print(author_stats.head(10).to_string())
