"""
Download Foggy Zurich (Foggy Driving) dataset.

Dataset: Foggy Driving
Source: ETH Zurich
URL: https://data.vision.ee.ethz.ch/csakarid/shared/SFSU_synthetic/Downloads/Foggy_Driving.zip
Paper: Semantic Foggy Scene Understanding with Synthetic Data

This dataset contains synthetically fogged driving scenes.
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile

def download_file(url, output_path):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")

def main():
    # Setup paths
    script_dir = Path(__file__).parent
    data_root = script_dir.parent / "Data"
    foggy_dir = data_root / "FoggyZurich"
    
    foggy_dir.mkdir(parents=True, exist_ok=True)
    
    # Download URL
    url = "https://data.vision.ee.ethz.ch/csakarid/shared/SFSU_synthetic/Downloads/Foggy_Driving.zip"
    zip_path = foggy_dir / "Foggy_Driving.zip"
    
    print("="*60)
    print("Downloading Foggy Zurich (Foggy Driving) Dataset")
    print("="*60)
    print(f"URL: {url}")
    print(f"Destination: {foggy_dir}")
    print()
    
    # Download if not exists
    if zip_path.exists():
        print(f"[INFO] Archive already exists: {zip_path}")
    else:
        print(f"Downloading to {zip_path}...")
        try:
            download_file(url, zip_path)
            print(f"✓ Download complete: {zip_path}")
        except Exception as e:
            print(f"[ERROR] Download failed: {e}")
            return
    
    # Extract
    if (foggy_dir / "Foggy_Driving").exists():
        print(f"[INFO] Dataset already extracted")
    else:
        extract_zip(zip_path, foggy_dir)
        print(f"✓ Extraction complete")
    
    print("\n" + "="*60)
    print("✓ Foggy Zurich dataset ready!")
    print("="*60)
    print(f"Location: {foggy_dir}")
    print("\nNext steps:")
    print("  1. Run preprocessing: python dataset_utils/preprocess_foggy_zurich.py")
    print("  2. Enable in config: Set foggy_zurich.enabled = true in configs/base.yaml")

if __name__ == "__main__":
    main()
