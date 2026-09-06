import requests
import os
from pathlib import Path

# Tạo folders
base_dir = Path("../data/datasets")
folders = ["face", "speech", "gait", "pose"]
for folder in folders:
    (base_dir / folder).mkdir(parents=True, exist_ok=True)

# Dataset URLs (cần tìm direct links)
datasets = {
    "face/sunnybrook.zip": "https://DIRECT_LINK_HERE",
    "speech/torgo.zip": "https://DIRECT_LINK_HERE",
    "gait/kth.zip": "https://DIRECT_LINK_HERE",
    "pose/coco.zip": "https://DIRECT_LINK_HERE"
}

def download_file(url, dest_path):
    """Download file với progress bar"""
    print(f"Downloading {dest_path}...")

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            percent = (downloaded / total_size) * 100 if total_size > 0 else 0
            print(f"\rProgress: {percent:.1f}%", end='')

    print(f"\n✓ Downloaded: {dest_path}")

# Download tất cả
for filename, url in datasets.items():
    if url != "https://DIRECT_LINK_HERE":
        dest_path = base_dir / filename
        download_file(url, dest_path)
    else:
        print(f"⚠ Skipping {filename} - need direct link")

print("\n✅ Download complete!")
