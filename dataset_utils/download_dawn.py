import os
import requests
import zipfile
from tqdm import tqdm
import sys

def download_file(url, filename):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024 # 1MB
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(filename, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR: Download incomplete")
            return False
        return True
    except Exception as e:
        print(f"Error downloading: {e}")
        return False

def main():
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Target Data/DAWN (relative to dataset_utils)
    data_root = os.path.abspath(os.path.join(script_dir, "../Data"))
    dawn_dir = os.path.join(data_root, "DAWN")
    
    if not os.path.exists(dawn_dir):
        os.makedirs(dawn_dir)
        
    # Mendeley public-api zip endpoint: redirects to a freshly-signed S3 URL.
    # (The old hardcoded prod-dcd-datasets-cache-zipfiles.s3 URL now 403s because
    # Mendeley rotates those signatures.)
    url = "https://data.mendeley.com/public-api/zip/766ygrbt8y/download/3"
    zip_path = os.path.join(dawn_dir, "dawn.zip")

    print(f"Downloading DAWN dataset to {zip_path}...")
    if download_file(url, zip_path):
        print("Extracting...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dawn_dir)

            # The archive nests one zip per weather (Fog.zip, Rain.zip, Snow.zip,
            # Sand.zip) under DAWN/DAWN/. Extract those in place so the converter
            # sees DAWN/DAWN/{Fog,Rain,Snow}/ with images + {Weather}_PASCAL_VOC/.
            nested_zips = []
            for root, _dirs, files in os.walk(dawn_dir):
                for name in files:
                    if name.lower().endswith(".zip") and os.path.join(root, name) != zip_path:
                        nested_zips.append(os.path.join(root, name))
            for nz in nested_zips:
                with zipfile.ZipFile(nz, 'r') as zf:
                    zf.extractall(os.path.dirname(nz))
                print(f"  extracted nested archive: {os.path.relpath(nz, dawn_dir)}")

            print(f"Successfully downloaded and extracted DAWN dataset to {dawn_dir}")
            print("Next: python dataset_utils/convert_dawn_to_coco.py")

            # Optional: Remove zip file after extraction
            # os.remove(zip_path)
        except zipfile.BadZipFile:
            print("Error: The downloaded file is not a valid zip file.")
    else:
        print("Download failed.")

if __name__ == "__main__":
    main()
