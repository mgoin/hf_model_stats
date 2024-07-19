import pandas as pd
from datetime import datetime

def update_readme():
    # Read the existing README content
    with open('README.md', 'r') as f:
        readme_content = f.read()

    # Read the latest stats
    df = pd.read_csv('model_stats.csv')

    # Create a markdown table with the latest stats
    table = df.to_markdown(index=False)

    # Update or append the table in the README
    if '## Latest Stats' in readme_content:
        # Update existing table
        start = readme_content.index('## Latest Stats')
        end = readme_content.index('\n\n', start)
        updated_content = f"{readme_content[:start]}## Latest Stats\n\n{table}\n\n{readme_content[end:]}"
    else:
        # Append new table
        updated_content = f"{readme_content}\n\n## Latest Stats\n\n{table}"

    # Write the updated content back to README.md
    with open('README.md', 'w') as f:
        f.write(updated_content)

if __name__ == "__main__":
    update_readme()
