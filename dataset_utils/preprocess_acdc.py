"""
Preprocess ACDC dataset for adverse weather validation.

ACDC (Adverse Conditions Dataset with Correspondences) contains:
- Weather types: fog, rain, snow, night
- Splits: train, val, test
- Original labels use Cityscapes categories (IDs 24-33)

This script:
1. Extracts adverse weather images (fog, rain, snow) from the zip archive
2. Combines train+val+test splits into a single validation set
3. Converts Cityscapes labels to BDD100K format
4. Creates COCO-format JSON annotations

BDD100K categories (COCO category_id, 1-indexed to match convert_bdd100k_to_coco.py):
1 bike, 2 bus, 3 car, 4 motor, 5 person, 6 rider, 7 traffic light, 8 traffic sign, 9 train, 10 truck

ACDC/Cityscapes categories (IDs 24-33):
person(24), rider(25), car(26), truck(27), bus(28), train(31), motorcycle(32), bicycle(33)

Label Mapping (Cityscapes id -> BDD100K category_id):
- person (24) -> person (5)
- rider (25) -> rider (6)
- car (26) -> car (3)
- truck (27) -> truck (10)
- bus (28) -> bus (2)
- train (31) -> train (9)
- motorcycle (32) -> motor (4)
- bicycle (33) -> bike (1)
- traffic light/sign -> Not present in ACDC, skip
"""

import os
import json
import zipfile
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import sys

# Import YOLO conversion utility
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir.parent))  # repo root, for weather_utils
from coco_to_yolo_labels import convert_coco_to_yolo
from weather_utils import canonical_weather_or_other

# Label mapping from ACDC/Cityscapes to BDD100K
LABEL_MAPPING = {
    24: 5,  # person -> person
    25: 6,  # rider -> rider
    26: 3,  # car -> car
    27: 10,  # truck -> truck
    28: 2,  # bus -> bus
    31: 9,  # train -> train
    32: 4,  # motorcycle -> motor
    33: 1,  # bicycle -> bike
}

BDD100K_CATEGORIES = [
    {"id": 1, "name": "bike"},
    {"id": 2, "name": "bus"},
    {"id": 3, "name": "car"},
    {"id": 4, "name": "motor"},
    {"id": 5, "name": "person"},
    {"id": 6, "name": "rider"},
    {"id": 7, "name": "traffic light"},  # Not in ACDC
    {"id": 8, "name": "traffic sign"},    # Not in ACDC
    {"id": 9, "name": "train"},
    {"id": 10, "name": "truck"}
]

# id -> name lookup. category_id is 1-indexed (matching BDD100K), NOT a list
# position, so it must be looked up by id (BDD100K_CATEGORIES[10] would be out of
# range for the truck id).
BDD100K_ID_TO_NAME = {cat["id"]: cat["name"] for cat in BDD100K_CATEGORIES}

def extract_acdc_images(zip_path, output_dir, weather_types=["fog", "rain", "snow"]):
    """
    Extract adverse weather images from ACDC zip archive.
    
    Args:
        zip_path: Path to rgb_anon_trainvaltest.zip
        output_dir: Output directory for extracted images
        weather_types: List of weather conditions to extract
    """
    print(f"Extracting ACDC images to {output_dir}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get all image files
        all_files = zip_ref.namelist()
        
        # Filter for adverse weather images
        image_files = []
        for f in all_files:
            # Check if file is an image in one of the target weather conditions
            if f.endswith('_rgb_anon.png'):
                for weather in weather_types:
                    if f'rgb_anon/{weather}/' in f:
                        image_files.append(f)
                        break
        
        print(f"Found {len(image_files)} adverse weather images")
        print(f"Weather types: {weather_types}")
        
        # Extract with progress bar
        for file in tqdm(image_files, desc="Extracting images"):
            zip_ref.extract(file, output_dir)
    
    print(f"✓ Extraction complete: {output_dir}")

def convert_acdc_to_coco(gt_dir, images_dir, output_dir, weather_types=["fog", "rain", "snow"]):
    """
    Convert ACDC annotations to COCO format with BDD100K label mapping.
    Creates separate JSON files for train, val, test splits.
    
    Args:
        gt_dir: Path to gt_detection/ directory
        images_dir: Path to extracted rgb_anon/ directory  
        output_dir: Directory to save COCO JSON files
        weather_types: List of weather types to include
    """
    print(f"\nConverting ACDC annotations to COCO format...")
    
    # ACDC has combined files for train/val/test across all weather types
    split_files = {
        "train": "instancesonly_train_gt_detection.json",
        "val": "instancesonly_val_gt_detection.json",
        "test": "instancesonly_test_image_info.json"  # Test has no GT annotations
    }
    
    # Create one COCO dict per split
    coco_dicts = {
        "train": {
            "images": [],
            "annotations": [],
            "categories": BDD100K_CATEGORIES,
            "info": {"description": "ACDC - Train Split", "split": "train"},
            "licenses": []
        },
        "val": {
            "images": [],
            "annotations": [],
            "categories": BDD100K_CATEGORIES,
            "info": {"description": "ACDC - Val Split", "split": "val"},
            "licenses": []
        }
        # Note: Test split excluded - no ground truth labels available
    }
    
    # Track IDs per split
    image_ids = {"train": 0, "val": 0}
    # Annotation ids must start at 1: pycocotools' COCOeval stores the matched
    # GT id in its match matrix and treats 0 as "unmatched", so a GT annotation
    # with id 0 can never count as a true positive.
    annotation_ids = {"train": 1, "val": 1}
    
    for split, filename in split_files.items():
        # Skip test split - no ground truth available
        if split == 'test':
            print(f"[INFO] Skipping {split} split - no ground truth labels")
            continue
            
        gt_file = os.path.join(gt_dir, filename)
        
        if not os.path.exists(gt_file):
            print(f"[WARNING] Ground truth not found: {gt_file}")
            continue
        
        # Load ACDC COCO annotations
        with open(gt_file, 'r') as f:
            acdc_data = json.load(f)
        
        # Map old image IDs to new ones
        image_id_map = {}
        
        # Process images - filter by weather type
        for img in tqdm(acdc_data.get('images', []), desc=f"Processing {split}"):
            # Extract weather from file_name (e.g., "fog/train/GP010475/...")
            file_name = img['file_name']
            weather = file_name.split('/')[0]

            # Skip if not in desired weather types
            if weather not in weather_types:
                continue

            # Store the canonical condition (fog -> foggy, rain -> rainy, ...)
            canonical_weather = canonical_weather_or_other(weather)
            
            old_image_id = img['id']
            new_image_id = image_ids[split]
            image_id_map[old_image_id] = new_image_id
            
            coco_dicts[split]['images'].append({
                "id": new_image_id,
                "file_name": os.path.join("rgb_anon", file_name),  # Prepend rgb_anon/
                "width": img['width'],
                "height": img['height'],
                "weather": canonical_weather,
                "split": split
            })
            
            image_ids[split] += 1
        
        # Process annotations (test split has no annotations)
        if split != "test":
            for ann in acdc_data.get('annotations', []):
                old_image_id = ann['image_id']
                
                # Skip if image wasn't included (filtered by weather)
                if old_image_id not in image_id_map:
                    continue
                
                # Map ACDC category to BDD100K
                old_category_id = ann['category_id']
                if old_category_id not in LABEL_MAPPING:
                    continue
                
                new_category_id = LABEL_MAPPING[old_category_id]
                
                coco_dicts[split]['annotations'].append({
                    "id": annotation_ids[split],
                    "image_id": image_id_map[old_image_id],
                    "category_id": new_category_id,
                    "bbox": ann['bbox'],
                    "area": ann['area'],
                    "iscrowd": ann.get('iscrowd', 0)
                })
                
                annotation_ids[split] += 1
    
    # Save separate COCO JSONs for each split (only train and val have GT)
    output_files = {}
    for split in ["train", "val"]:
        output_json = os.path.join(output_dir, f"acdc_{split}_coco.json")
        with open(output_json, 'w') as f:
            json.dump(coco_dicts[split], f, indent=2)
        output_files[split] = output_json
        
        print(f"\n✓ Created {split} split: {output_json}")
        print(f"  Images: {len(coco_dicts[split]['images'])}")
        print(f"  Annotations: {len(coco_dicts[split]['annotations'])}")
        
        # Print weather distribution for this split
        weather_counts = defaultdict(int)
        for img in coco_dicts[split]['images']:
            weather_counts[img['weather']] += 1
        
        print(f"  Weather: {dict(weather_counts)}")
    
    # Print combined label distribution
    print(f"\n  Combined Label Distribution:")
    all_label_counts = defaultdict(int)
    for split in ["train", "val"]:  # Only count splits with GT
        for ann in coco_dicts[split]['annotations']:
            cat_id = ann['category_id']
            cat_name = BDD100K_ID_TO_NAME[cat_id]
            all_label_counts[cat_name] += 1
    
    for label in sorted(all_label_counts.keys()):
        print(f"    {label}: {all_label_counts[label]}")
    
    
    # Save combined statistics file
    stats_file = os.path.join(output_dir, 'acdc_dataset_stats.txt')
    with open(stats_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("ACDC DATASET STATISTICS\n")
        f.write("="*60 + "\n\n")
        
        for split in ["train", "val"]:
            f.write(f"{split.upper()} Split:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Total images: {len(coco_dicts[split]['images'])}\n")
            f.write(f"  Total annotations: {len(coco_dicts[split]['annotations'])}\n")
            
            weather_counts_split = defaultdict(int)
            for img in coco_dicts[split]['images']:
                weather_counts_split[img['weather']] += 1
            
            f.write(f"  Weather breakdown:\n")
            for weather, count in sorted(weather_counts_split.items()):
                f.write(f"    {weather}: {count} images\n")
            f.write("\n")
        
        f.write("Combined Label Distribution (train + val):\n")
        f.write("-" * 40 + "\n")
        for label, count in sorted(all_label_counts.items()):
            f.write(f"  {label:15} {count:5} annotations\n")
        
        f.write("\nMissing BDD100K Categories:\n")
        f.write("-" * 40 + "\n")
        all_cats = {c['name'] for c in BDD100K_CATEGORIES}
        present_cats = set(all_label_counts.keys())
        missing_cats = all_cats - present_cats
        for cat in sorted(missing_cats):
            f.write(f"  {cat}\n")
        
        f.write("\nNote: Test split excluded (no ground truth labels)\n")
    
    print(f"\nStatistics saved to: {stats_file}")
    
    return output_files
    """
    Convert ACDC annotations to COCO format with BDD100K label mapping.
    Creates separate JSON files for train, val, test splits.
    
    Args:
        gt_dir: Path to gt_detection/ directory
        images_dir: Path to extracted rgb_anon/ directory  
        output_dir: Directory to save COCO JSON files
        weather_types: List of weather types to include
    """
    print(f"\nConverting ACDC annotations to COCO format...")
    
    # Create one COCO dict per split
    coco_dicts = {
        "train": {
            "images": [],
            "annotations": [],
            "categories": BDD100K_CATEGORIES,
            "info": {"description": "ACDC - Train Split", "split": "train"},
            "licenses": []
        },
        "val": {
            "images": [],
            "annotations": [],
            "categories": BDD100K_CATEGORIES,
            "info": {"description": "ACDC - Val Split", "split": "val"},
            "licenses": []
        },
        "test": {
            "images": [],
            "annotations": [],
            "categories": BDD100K_CATEGORIES,
            "info": {"description": "ACDC - Test Split", "split": "test"},
            "licenses": []
        }
    }
    
    # Track IDs per split
    image_ids = {"train": 0, "val": 0, "test": 0}
    annotation_ids = {"train": 0, "val": 0, "test": 0}
    
    for weather in weather_types:
        for split in ["train", "val", "test"]:
            gt_file = os.path.join(gt_dir, f"{weather}_{split}.json")
            
            if not os.path.exists(gt_file):
                print(f"[WARNING] Ground truth not found: {gt_file}")
                continue
            
            # Load ACDC COCO annotations
            with open(gt_file, 'r') as f:
                acdc_data = json.load(f)
            
            # Map old image IDs to new ones
            image_id_map = {}
            
            # Process images
            for img in tqdm(acdc_data.get('images', []), desc=f"{weather} {split}"):
                old_image_id = img['id']
                new_image_id = image_ids[split]
                image_id_map[old_image_id] = new_image_id
                
                # Construct file path (ACDC stores relative path in file_name)
                file_name = img['file_name']
                
                coco_dicts[split]['images'].append({
                    "id": new_image_id,
                    "file_name": file_name,
                    "width": img['width'],
                    "height": img['height'],
                    "weather": weather,
                    "split": split
                })
                
                image_ids[split] += 1
            
            # Process annotations (skip for test split - no annotations)
            if split != "test":
                for ann in acdc_data.get('annotations', []):
                    old_image_id = ann['image_id']
                    
                    # Skip if image wasn't included
                    if old_image_id not in image_id_map:
                        continue
                    
                    # Map ACDC category to BDD100K
                    old_category_id = ann['category_id']
                    if old_category_id not in LABEL_MAPPING:
                        continue
                    
                    new_category_id = LABEL_MAPPING[old_category_id]
                    
                    coco_dicts[split]['annotations'].append({
                        "id": annotation_ids[split],
                        "image_id": image_id_map[old_image_id],
                        "category_id": new_category_id,
                        "bbox": ann['bbox'],
                        "area": ann['area'],
                        "iscrowd": ann.get('iscrowd', 0)
                    })
                    
                    annotation_ids[split] += 1
    
    # Save separate COCO JSONs for each split
    output_files = {}
    for split in ["train", "val", "test"]:
        output_json = os.path.join(output_dir, f"acdc_{split}_coco.json")
        with open(output_json, 'w') as f:
            json.dump(coco_dicts[split], f, indent=2)
        output_files[split] = output_json
        
        print(f"\n✓ Created {split} split: {output_json}")
        print(f"  Images: {len(coco_dicts[split]['images'])}")
        print(f"  Annotations: {len(coco_dicts[split]['annotations'])}")
        
        # Print weather distribution for this split
        weather_counts = defaultdict(int)
        for img in coco_dicts[split]['images']:
            weather_counts[img['weather']] += 1
        
        print(f"  Weather: {dict(weather_counts)}")
    
    # Print combined label distribution
    print(f"\n  Combined Label Distribution:")
    all_label_counts = defaultdict(int)
    for split in ["train", "val"]:  # Only count splits with GT
        for ann in coco_dicts[split]['annotations']:
            cat_id = ann['category_id']
            cat_name = BDD100K_ID_TO_NAME[cat_id]
            all_label_counts[cat_name] += 1
    
    for label in sorted(all_label_counts.keys()):
        print(f"    {label}: {all_label_counts[label]}")
    
    return output_files
    """
    Convert ACDC annotations to COCO format with BDD100K label mapping.
    
    Args:
        gt_dir: Directory containing ACDC ground truth JSON files
        images_dir: Directory containing extracted images
        output_json: Output path for COCO JSON file
        weather_types: List of weather conditions to include
    """
    print(f"\nConverting ACDC annotations to COCO format...")
    
    # Initialize COCO structure
    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": BDD100K_CATEGORIES,
        "info": {
            "description": "ACDC - Adverse Conditions Dataset (fog, rain, snow)",
            "url": "https://acdc.vision.ee.ethz.ch/",
            "version": "1.0",
            "year": 2021
        },
        "licenses": []
    }
    
    image_id = 0
    annotation_id = 1  # 0 is reserved as "unmatched" in COCOeval match matrices
    
    # Process each weather type and split
    splits = ["train", "val", "test"]
    
    for weather in weather_types:
        for split in splits:
            # Construct GT JSON path
            gt_file = os.path.join(gt_dir, weather, f"instancesonly_{weather}_{split}_gt_detection.json")
            
            # Test split only has image_info, not full annotations
            if split == "test":
                gt_file = os.path.join(gt_dir, weather, f"instancesonly_{weather}_{split}_image_info.json")
            
            if not os.path.exists(gt_file):
                print(f"[WARNING] GT file not found: {gt_file}")
                continue
            
            print(f"Processing {weather}/{split}...")
            
            # Load ACDC annotations
            with open(gt_file, 'r') as f:
                acdc_data = json.load(f)
            
            # Create mapping from original image IDs to new sequential IDs
            image_id_map = {}
            
            # Process images
            for img in acdc_data.get('images', []):
                old_id = img['id']
                new_id = image_id
                image_id_map[old_id] = new_id
                
                # Extract relative path from file_name
                # Format from JSON: {weather}/{split}/{scene}/{filename}
                # Extracted path: rgb_anon/{weather}/{split}/{scene}/{filename}
                file_path_from_json = img['file_name']
                file_path = os.path.join('rgb_anon', file_path_from_json)
                
                # Verify image exists
                full_path = os.path.join(images_dir, file_path)
                if not os.path.exists(full_path):
                    # Try without rgb_anon prefix (in case extraction was different)
                    full_path_alt = os.path.join(images_dir, file_path_from_json)
                    if os.path.exists(full_path_alt):
                        file_path = file_path_from_json
                    else:
                        continue  # Skip this image silently
                
                coco_dict['images'].append({
                    "id": new_id,
                    "file_name": file_path,
                    "width": img['width'],
                    "height": img['height'],
                    "weather": weather,
                    "split": split  # track original split
                })
                
                image_id += 1
            
            # Process annotations (skip for test split)
            if split != "test":
                for ann in acdc_data.get('annotations', []):
                    old_image_id = ann['image_id']
                    
                    # Skip if image wasn't included
                    if old_image_id not in image_id_map:
                        continue
                    
                    # Map ACDC category to BDD100K
                    old_category_id = ann['category_id']
                    if old_category_id not in LABEL_MAPPING:
                        # Skip categories not in BDD100K
                        continue
                    
                    new_category_id = LABEL_MAPPING[old_category_id]
                    
                    coco_dict['annotations'].append({
                        "id": annotation_id,
                        "image_id": image_id_map[old_image_id],
                        "category_id": new_category_id,
                        "bbox": ann['bbox'],
                        "area": ann['area'],
                        "iscrowd": ann.get('iscrowd', 0)
                    })
                    
                    annotation_id += 1
    
    # Save COCO JSON
    with open(output_json, 'w') as f:
        json.dump(coco_dict, f, indent=2)
    
    print(f"\n✓ Created COCO JSON: {output_json}")
    print(f"  Total images: {len(coco_dict['images'])}")
    print(f"  Total annotations: {len(coco_dict['annotations'])}")
    
    # Print statistics
    weather_counts = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for img in coco_dict['images']:
        weather_counts[img['weather']][img['split']] += 1
    
    print(f"\n  Weather distribution:")
    for weather in sorted(weather_counts.keys()):
        counts = weather_counts[weather]
        total = sum(counts.values())
        print(f"    {weather}: {total} ({counts['train']} train, {counts['val']} val, {counts['test']} test)")
    
    # Print label distribution
    label_counts = defaultdict(int)
    for ann in coco_dict['annotations']:
        cat_id = ann['category_id']
        cat_name = BDD100K_ID_TO_NAME[cat_id]
        label_counts[cat_name] += 1
    
    print(f"\n  Label distribution:")
    for label in sorted(label_counts.keys()):
        print(f"    {label}: {label_counts[label]}")

def main():
    # Setup paths
    script_dir = Path(__file__).parent
    data_root = script_dir.parent / "Data"
    acdc_zip_dir = script_dir / "acdc_dataset_zip_files"
    
    rgb_zip = acdc_zip_dir / "rgb_anon_trainvaltest.zip"
    gt_dir = acdc_zip_dir / "gt_detection"
    
    # Output paths
    output_dir = data_root / "ACDC"
    output_images_dir = output_dir / "rgb_anon"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if ground truth is extracted
    if not gt_dir.exists():
        print("[ERROR] Ground truth not extracted yet!")
        print(f"Please extract gt_detection_trainval.zip in {acdc_zip_dir}")
        return
    
    # Extract images (only adverse weather: fog, rain, snow)
    if not output_images_dir.exists():
        extract_acdc_images(
            zip_path=str(rgb_zip),
            output_dir=str(output_dir),
            weather_types=["fog", "rain", "snow"]
        )
    else:
        print(f"Images already extracted: {output_images_dir}")
    
    # Convert to COCO format (creates separate JSONs for train/val/test)
    output_files = convert_acdc_to_coco(
        gt_dir=str(gt_dir),
        images_dir=str(output_dir),
        output_dir=str(output_dir),
        weather_types=["fog", "rain", "snow"]
    )
    
    print("\n✓ ACDC preprocessing complete!")
    print(f"  Images: {output_images_dir}")
    print(f"  Annotations:")
    for split, path in output_files.items():
        print(f"    {split}: {path}")
    
    # Convert to YOLO format
    print("\n" + "=" * 80)
    print("Converting to YOLO format...")
    print("=" * 80)
    
    yolo_labels_base = output_dir / "labels"
    
    # Convert train split
    if 'train' in output_files:
        print("\nConverting train split...")
        train_yolo_dir = yolo_labels_base / "train"
        convert_coco_to_yolo(
            coco_json_path=str(output_files['train']),
            output_labels_dir=str(train_yolo_dir),
            images_dir=str(output_dir)
        )
    
    # Convert val split
    if 'val' in output_files:
        print("\nConverting val split...")
        val_yolo_dir = yolo_labels_base / "val"
        convert_coco_to_yolo(
            coco_json_path=str(output_files['val']),
            output_labels_dir=str(val_yolo_dir),
            images_dir=str(output_dir)
        )
    
    print(f"\nYOLO labels saved to: {yolo_labels_base}")

if __name__ == "__main__":
    main()
