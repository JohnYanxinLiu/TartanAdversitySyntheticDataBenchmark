"""
Preprocess Foggy Zurich dataset and convert to COCO format with BDD100K label mapping.

The Foggy Zurich (Foggy Driving) dataset contains 101 real-world foggy road scenes.
This script converts the bounding box annotations to COCO format.

Foggy Zurich classes (8):
- 0: car
- 1: person
- 2: bicycle
- 3: bus
- 4: truck
- 5: train
- 6: motorcycle
- 7: rider

BDD100K classes (10, 1-indexed COCO category_id to match convert_bdd100k_to_coco.py):
- 1: bike (bicycle)
- 2: bus
- 3: car
- 4: motor (motorcycle)
- 5: person
- 6: rider
- 7: traffic light
- 8: traffic sign
- 9: train
- 10: truck
"""

import os
import json
from pathlib import Path
from PIL import Image
from datetime import datetime
import sys

# Import YOLO conversion utility
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
from coco_to_yolo_labels import convert_coco_to_yolo

# Mapping from Foggy Zurich class IDs to BDD100K class IDs
FOGGY_TO_BDD = {
    0: 3,  # car -> car
    1: 5,  # person -> person
    2: 1,  # bicycle -> bike
    3: 2,  # bus -> bus
    4: 10,  # truck -> truck
    5: 9,  # train -> train
    6: 4,  # motorcycle -> motor
    7: 6,  # rider -> rider
}

# 1-indexed to match BDD100K/ACDC/synthetic COCO (convert_bdd100k_to_coco.py) and
# the FOGGY_TO_BDD targets above; keeps the declared categories consistent with the
# category_id values written into the annotations.
BDD100K_CATEGORIES = [
    {"id": 1, "name": "bike"},
    {"id": 2, "name": "bus"},
    {"id": 3, "name": "car"},
    {"id": 4, "name": "motor"},
    {"id": 5, "name": "person"},
    {"id": 6, "name": "rider"},
    {"id": 7, "name": "traffic light"},
    {"id": 8, "name": "traffic sign"},
    {"id": 9, "name": "train"},
    {"id": 10, "name": "truck"},
]

# id -> name lookup (ids are not list positions, so index by id explicitly)
BDD100K_ID_TO_NAME = {cat["id"]: cat["name"] for cat in BDD100K_CATEGORIES}


def parse_bbox_file(bbox_path):
    """
    Parse a Foggy Zurich bbox annotation file.
    
    Format: class x1 y1 x2 y2 (1-based pixel coordinates)
    
    Returns:
        List of tuples: [(class_id, x1, y1, x2, y2), ...]
    """
    boxes = []
    with open(bbox_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x1, y1, x2, y2 = map(int, parts[1:])
                boxes.append((class_id, x1, y1, x2, y2))
    return boxes


def convert_to_coco(data_root, output_path, stats_path):
    """
    Convert Foggy Zurich dataset to COCO format.
    
    Args:
        data_root: Path to Foggy_Driving directory
        output_path: Path to output COCO JSON file
        stats_path: Path to output statistics file
    """
    data_root = Path(data_root)
    
    # Process all splits: test (fine annotations) and test_extra (coarse annotations)
    splits_to_process = [
        ("test", "public"),
        ("test", "pedestrian"),
        ("test_extra", "web"),
        ("test_extra", "pedestrian"),
    ]
    # Process all splits: test (fine annotations) and test_extra (coarse annotations)
    splits_to_process = [
        ("test", "public"),
        ("test", "pedestrian"),
        ("test_extra", "web"),
        ("test_extra", "pedestrian"),
    ]
    
    coco_data = {
        "info": {
            "description": "Foggy Zurich (Foggy Driving) Dataset - Real-world foggy road scenes",
            "url": "https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/",
            "version": "1.0",
            "year": 2018,
            "contributor": "ETH Zurich",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": BDD100K_CATEGORIES
    }
    
    annotation_id = 1
    image_id = 1
    
    # Statistics tracking
    stats = {
        'total_images': 0,
        'total_annotations': 0,
        'images_with_annotations': 0,
        'images_without_annotations': 0,
        'class_counts': {cat['name']: 0 for cat in BDD100K_CATEGORIES},
        'foggy_zurich_class_counts': {
            'car': 0, 'person': 0, 'bicycle': 0, 'bus': 0,
            'truck': 0, 'train': 0, 'motorcycle': 0, 'rider': 0
        },
        'split_counts': {
            'test_public': 0,
            'test_pedestrian': 0,
            'test_extra_web': 0,
            'test_extra_pedestrian': 0
        }
    }
    
    # Process each split
    for split, record in splits_to_process:
        bbox_dir = data_root / "bboxGt" / split / record
        img_dir = data_root / "leftImg8bit" / split / record
        
        if not bbox_dir.exists():
            print(f"Warning: Bbox directory not found: {bbox_dir}")
            continue
        if not img_dir.exists():
            print(f"Warning: Image directory not found: {img_dir}")
            continue
        
        # Get all bbox files
        bbox_files = sorted(bbox_dir.glob("*.txt"))
        split_key = f"{split}_{record}"
        stats['split_counts'][split_key] = len(bbox_files)
        
        print(f"Processing {split}/{record}: {len(bbox_files)} images")
        
        for bbox_file in bbox_files:
            # Construct corresponding image filename
            # public_20161213_081721.txt -> public_20161213_081721_leftImg8bit.png
            base_name = bbox_file.stem
            img_name = f"{base_name}_leftImg8bit.png"
            img_path = img_dir / img_name
            
            if not img_path.exists():
                print(f"Warning: Image not found for {bbox_file.name}: {img_path}")
                continue
            
            # Get image dimensions
            with Image.open(img_path) as img:
                width, height = img.size
            
            # Add image to COCO
            image_info = {
                "id": image_id,
                "file_name": f"{split}/{record}/{img_name}",  # Include split/record in path
                "width": width,
                "height": height,
                "weather": "foggy",  # All Foggy Zurich images are foggy (canonical label)
                "split": split,
                "record": record
            }
            coco_data["images"].append(image_info)
            stats['total_images'] += 1
            
            # Parse bounding boxes
            boxes = parse_bbox_file(bbox_file)
            
            if not boxes:
                stats['images_without_annotations'] += 1
            else:
                stats['images_with_annotations'] += 1
            
            for class_id, x1, y1, x2, y2 in boxes:
                # Convert from 1-based to 0-based coordinates
                x1 -= 1
                y1 -= 1
                x2 -= 1
                y2 -= 1
                
                # Convert to COCO format (x, y, width, height)
                bbox_x = x1
                bbox_y = y1
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                
                # Skip invalid boxes
                if bbox_width <= 0 or bbox_height <= 0:
                    continue
                
                # Map to BDD100K category
                if class_id in FOGGY_TO_BDD:
                    bdd_category_id = FOGGY_TO_BDD[class_id]
                    
                    # Track Foggy Zurich original classes
                    foggy_class_names = {
                        0: 'car', 1: 'person', 2: 'bicycle', 3: 'bus',
                        4: 'truck', 5: 'train', 6: 'motorcycle', 7: 'rider'
                    }
                    if class_id in foggy_class_names:
                        stats['foggy_zurich_class_counts'][foggy_class_names[class_id]] += 1
                    
                    # Track BDD100K mapped classes (look up by id, not list position)
                    bdd_category_name = BDD100K_ID_TO_NAME[bdd_category_id]
                    stats['class_counts'][bdd_category_name] += 1
                    
                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": bdd_category_id,
                        "bbox": [bbox_x, bbox_y, bbox_width, bbox_height],
                        "area": bbox_width * bbox_height,
                        "iscrowd": 0
                    }
                    coco_data["annotations"].append(annotation)
                    stats['total_annotations'] += 1
                    annotation_id += 1
            
            image_id += 1
    
    # Save COCO JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\nCOCO annotations saved to: {output_path}")
    print(f"Total images: {stats['total_images']}")
    print(f"Total annotations: {stats['total_annotations']}")
    print(f"Images with annotations: {stats['images_with_annotations']}")
    print(f"Images without annotations: {stats['images_without_annotations']}")
    
    # Save statistics file
    save_statistics(stats, stats_path)
    
    return coco_data, stats


def save_statistics(stats, stats_path):
    """Save dataset statistics to a text file."""
    stats_path = Path(stats_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(stats_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FOGGY ZURICH (FOGGY DRIVING) DATASET STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Dataset: Foggy Zurich (Foggy Driving)\n")
        f.write(f"Source: ETH Zurich - Real-world foggy road scenes\n")
        f.write(f"Weather: All images contain fog\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total images: {stats['total_images']}\n")
        f.write(f"Total annotations: {stats['total_annotations']}\n")
        f.write(f"Images with annotations: {stats['images_with_annotations']}\n")
        f.write(f"Images without annotations: {stats['images_without_annotations']}\n")
        f.write(f"Average annotations per image: {stats['total_annotations'] / stats['total_images']:.2f}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("SPLIT DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        for split_name, count in sorted(stats['split_counts'].items()):
            f.write(f"{split_name:25s}: {count:3d} images\n")
        f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("BDD100K CLASS DISTRIBUTION (MAPPED)\n")
        f.write("-" * 80 + "\n")
        for class_name, count in sorted(stats['class_counts'].items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                percentage = (count / stats['total_annotations']) * 100
                f.write(f"{class_name:15s}: {count:5d} ({percentage:5.2f}%)\n")
        
        f.write("\n")
        f.write("-" * 80 + "\n")
        f.write("ORIGINAL FOGGY ZURICH CLASS DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        for class_name, count in sorted(stats['foggy_zurich_class_counts'].items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                percentage = (count / stats['total_annotations']) * 100
                f.write(f"{class_name:15s}: {count:5d} ({percentage:5.2f}%)\n")
        
        f.write("\n")
        f.write("-" * 80 + "\n")
        f.write("MISSING BDD100K CLASSES (NOT IN FOGGY ZURICH)\n")
        f.write("-" * 80 + "\n")
        missing_classes = [name for name, count in stats['class_counts'].items() if count == 0]
        if missing_classes:
            for class_name in sorted(missing_classes):
                f.write(f"  - {class_name}\n")
        else:
            f.write("  None (all BDD100K classes present)\n")
        
        f.write("\n")
        f.write("-" * 80 + "\n")
        f.write("DATASET NOTES\n")
        f.write("-" * 80 + "\n")
        f.write("- Includes all 101 images from Foggy Zurich dataset\n")
        f.write("- test split: 33 images with fine pixel-level semantic annotations\n")
        f.write("  - test/public: 27 images (recorded from public transportation)\n")
        f.write("  - test/pedestrian: 6 images (recorded from pedestrian viewpoint)\n")
        f.write("- test_extra split: 68 images with coarse semantic annotations\n")
        f.write("  - test_extra/web: 50 images (collected from web)\n")
        f.write("  - test_extra/pedestrian: 18 images (pedestrian viewpoint)\n")
        f.write("- All images contain real-world fog (not synthetic)\n")
        f.write("- Bounding boxes use 8 object classes from Cityscapes format\n")
        f.write("- Original coordinates are 1-based, converted to 0-based for COCO\n")
        f.write("\n")
        f.write("=" * 80 + "\n")
    
    print(f"Statistics saved to: {stats_path}")


def main():
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_root = project_root / "Data" / "FoggyZurich" / "Foggy_Driving"
    output_dir = project_root / "Data" / "FoggyZurich"
    
    output_json = output_dir / "foggy_zurich_coco.json"
    stats_file = output_dir / "foggy_zurich_stats.txt"
    
    print("=" * 80)
    print("Foggy Zurich Dataset Preprocessing")
    print("=" * 80)
    print(f"Data root: {data_root}")
    print(f"Output JSON: {output_json}")
    print(f"Output stats: {stats_file}")
    print("=" * 80)
    
    if not data_root.exists():
        print(f"\nError: Data directory not found: {data_root}")
        print("Please run download_foggy_zurich.py first to download the dataset.")
        return
    
    # Convert to COCO format
    convert_to_coco(data_root, output_json, stats_file)
    
    print("\n" + "=" * 80)
    print("Preprocessing complete!")
    print("=" * 80)
    
    # Convert to YOLO format
    print("\n" + "=" * 80)
    print("Converting to YOLO format...")
    print("=" * 80)
    yolo_labels_dir = output_dir / "labels"
    convert_coco_to_yolo(
        coco_json_path=str(output_json),
        output_labels_dir=str(yolo_labels_dir),
        images_dir=str(data_root / "leftImg8bit")
    )
    print(f"\nYOLO labels saved to: {yolo_labels_dir}")


if __name__ == "__main__":
    main()
