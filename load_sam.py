import requests
import os

def download_file(url, save_path):
    """Download a file if it doesn't exist."""
    if os.path.exists(save_path):
        print(f"File already exists at {save_path}. Skipping download.")
        return 

    print(f"Downloading file from {url} to {save_path}...")
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded and saved as {save_path}")
    else:
        print(f"Failed to download file. HTTP Status Code: {response.status_code}")


