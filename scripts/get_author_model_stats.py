import requests
import pandas as pd
from datetime import datetime
import os


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
    new_data = get_author_model_stats("neuralmagic")

    if os.path.exists("model_stats.csv"):
        existing_data = pd.read_csv("model_stats.csv")
        updated_data = pd.concat([existing_data, new_data]).drop_duplicates(
            subset=["date", "model_id"], keep="last"
        )
    else:
        updated_data = new_data

    updated_data.to_csv("model_stats.csv", index=False)
    print(updated_data.to_markdown())
