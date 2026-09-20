import requests
import os

def download_file(url, save_path):
    """Download a file if it doesn't exist."""
    if os.path.exists(save_path):
        print(f"File already exists at {save_path}. Skipping download.")
        return 

    print(f"Downloading file from {url} to {save_path}...")
    response = requests.get(url, stream=True)

    if response.status_code != 200:
        print(f"Failed to download file. HTTP Status Code: {response.status_code}")
        return

    # Written in chunks to a .part file, then renamed: the SAM checkpoint is 2.4 GB, which
    # response.content would hold in memory all at once, and an interrupted download must not
    # leave a truncated file behind for the "already exists" check above to accept.
    partial_path = f"{save_path}.part"

    with open(partial_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            file.write(chunk)

    os.replace(partial_path, save_path)

    print(f"File downloaded and saved as {save_path}")


