#!/usr/bin/env python3
"""
Download face parsing weights that are stored in Git LFS.
Run this script if you get errors about invalid pickle files.
"""

import os
import requests
from pathlib import Path

# Weight files from Git LFS
WEIGHTS = {
    'rtnet50-fcn-14.torch': 'https://media.githubusercontent.com/media/hhj1897/face_parsing/master/ibug/face_parsing/rtnet/weights/rtnet50-fcn-14.torch',
    'rtnet50-fcn-11.torch': 'https://media.githubusercontent.com/media/hhj1897/face_parsing/master/ibug/face_parsing/rtnet/weights/rtnet50-fcn-11.torch',
    'rtnet101-fcn-14.torch': 'https://media.githubusercontent.com/media/hhj1897/face_parsing/master/ibug/face_parsing/rtnet/weights/rtnet101-fcn-14.torch'
}

def download_file(url, dest_path):
    """Download a file from URL to destination path."""
    print(f"Downloading {dest_path.name}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Get total file size
    total_size = int(response.headers.get('content-length', 0))

    # Download with progress
    with open(dest_path, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"  Progress: {percent:.1f}%", end='\r')

    print(f"  Downloaded {dest_path.name} ({os.path.getsize(dest_path) / 1024 / 1024:.1f} MB)")

def main():
    # Define the weights directory
    weights_dir = Path('nemo/losses/face_parsing/ibug/face_parsing/rtnet/weights')

    if not weights_dir.exists():
        print(f"Error: Weights directory not found at {weights_dir}")
        print("Please run this script from the VASA-1-hack root directory")
        return

    print("Downloading face parsing weights...")
    print("-" * 50)

    for filename, url in WEIGHTS.items():
        dest_path = weights_dir / filename

        # Check if file already exists and is valid (not a Git LFS pointer)
        if dest_path.exists():
            with open(dest_path, 'rb') as f:
                header = f.read(100)
                if b'git-lfs' in header:
                    print(f"{filename} is a Git LFS pointer, downloading actual file...")
                    download_file(url, dest_path)
                else:
                    file_size = os.path.getsize(dest_path)
                    if file_size < 1000:  # Less than 1KB, probably a pointer
                        print(f"{filename} appears to be invalid, re-downloading...")
                        download_file(url, dest_path)
                    else:
                        print(f"{filename} already exists ({file_size / 1024 / 1024:.1f} MB), skipping...")
        else:
            download_file(url, dest_path)

    print("-" * 50)
    print("✓ All weights downloaded successfully!")
    print("\nYou can now run train_overfit.py without Git LFS errors.")

if __name__ == "__main__":
    main()