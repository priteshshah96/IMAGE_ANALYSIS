#!/usr/bin/env python3
"""
Download Pitt Image Advertisement Dataset
"""

import requests
from pathlib import Path
from tqdm import tqdm

def download_dataset():
    """Download Pitt Image dataset"""
    # You'll need to replace with actual dataset URL
    url = "https://example.com/pitt-image-dataset.zip"  # Replace with real URL
    filename = "pitt_image_dataset.zip"
    
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    
    if filepath.exists():
        print(f"Dataset already exists: {filepath}")
        return filepath
    
    print(f"Downloading Pitt Image dataset from {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"Downloaded to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please download manually and place in data/raw/")

if __name__ == "__main__":
    download_dataset() 