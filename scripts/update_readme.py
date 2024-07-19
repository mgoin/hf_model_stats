import pandas as pd


def update_readme():
    # Read the existing README content
    with open("README.md", "r") as f:
        readme_content = f.read()

    # Read the latest stats
    df = pd.read_csv("model_stats.csv")

    # Create a markdown table with the latest stats
    table = df.to_markdown(index=False)

    # Update the table in the README
    if "## Latest Stats" in readme_content:
        updated_content = f"{readme_content}\n\n## Latest Stats\n\n{table}"
    else:
        raise Exception("Could not find '## Latest Stats' in README!")

    # Write the updated content back to README.md
    with open("README.md", "w") as f:
        f.write(updated_content)


if __name__ == "__main__":
    update_readme()
