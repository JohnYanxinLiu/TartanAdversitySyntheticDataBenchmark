#!/usr/bin/env python3
"""
Convert BDD100K training data to COCO format.

This script converts BDD100K annotations from the custom Supervisely format
to COCO format compatible with YOLO and other object detection frameworks.
"""

import json
import os
from pathlib import Path
from typing import Dict, List


# BDD100K category name mapping to COCO format
# Based on the categories found in the reference COCO file
CATEGORY_MAPPING = {
    'bike': 'bike',
    'bicycle': 'bike',
    'bus': 'bus',
    'car': 'car',
    'motor': 'motor',
    'motorcycle': 'motor',
    'person': 'person',
    'pedestrian': 'person',
    'rider': 'rider',
    'traffic light': 'traffic light',
    'traffic sign': 'traffic sign',
    'train': 'train',
    'truck': 'truck'
}

# COCO categories (from the reference file)
COCO_CATEGORIES = [
    {'id': 1, 'name': 'bike'},
    {'id': 2, 'name': 'bus'},
    {'id': 3, 'name': 'car'},
    {'id': 4, 'name': 'motor'},
    {'id': 5, 'name': 'person'},
    {'id': 6, 'name': 'rider'},
    {'id': 7, 'name': 'traffic light'},
    {'id': 8, 'name': 'traffic sign'},
    {'id': 9, 'name': 'train'},
    {'id': 10, 'name': 'truck'}
]

# Create category name to ID mapping
CATEGORY_NAME_TO_ID = {cat['name']: cat['id'] for cat in COCO_CATEGORIES}


def convert_bbox_to_coco(points: Dict) -> List[float]:
    """
    Convert BDD100K bbox format to COCO format [x, y, width, height].
    
    Args:
        points: Dictionary with 'exterior' key containing [[x1, y1], [x2, y2]]
    
    Returns:
        List of [x, y, width, height] in COCO format
    """
    exterior = points['exterior']
    x1, y1 = exterior[0]
    x2, y2 = exterior[1]
    
    # Ensure x1, y1 is top-left and x2, y2 is bottom-right
    x_min = min(x1, x2)
    y_min = min(y1, y2)
    x_max = max(x1, x2)
    y_max = max(y1, y2)
    
    width = x_max - x_min
    height = y_max - y_min
    
    return [x_min, y_min, width, height]


def extract_weather_from_tags(tags: List[Dict]) -> str:
    """
    Extract weather information from BDD100K tags.
    
    Args:
        tags: List of tag dictionaries
    
    Returns:
        Weather value or 'unknown'
    """
    for tag in tags:
        if tag.get('name') == 'weather':
            return tag.get('value', 'unknown')
    return 'unknown'


def convert_bdd100k_to_coco(bdd100k_train_path: str, output_path: str):
    """
    Convert BDD100K training data to COCO format.
    
    Args:
        bdd100k_train_path: Path to BDD100K/train directory
        output_path: Path where the output COCO JSON file will be saved
    """
    train_path = Path(bdd100k_train_path)
    ann_dir = train_path / 'ann'
    img_dir = train_path / 'img'
    
    if not ann_dir.exists():
        raise ValueError(f"Annotation directory not found: {ann_dir}")
    if not img_dir.exists():
        raise ValueError(f"Image directory not found: {img_dir}")
    
    # Initialize COCO structure
    coco_data = {
        'images': [],
        'annotations': [],
        'categories': COCO_CATEGORIES
    }
    
    image_id = 0
    annotation_id = 1  # 0 is reserved as "unmatched" in COCOeval match matrices
    
    # Get all annotation files
    ann_files = sorted(ann_dir.glob('*.json'))
    
    print(f"Found {len(ann_files)} annotation files")
    
    for ann_file in ann_files:
        # Corresponding image file (remove .json extension)
        img_filename = ann_file.stem  # e.g., '0000f77c-6257be58.jpg'
        img_path = img_dir / img_filename
        
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}, skipping...")
            continue
        
        # Load BDD100K annotation
        with open(ann_file, 'r') as f:
            bdd_ann = json.load(f)
        
        # Get image dimensions
        size = bdd_ann.get('size', {})
        width = size.get('width', 1280)
        height = size.get('height', 720)
        
        # Extract weather information
        weather = extract_weather_from_tags(bdd_ann.get('tags', []))
        
        # Add image entry
        image_entry = {
            'id': image_id,
            'file_name': img_filename,
            'width': width,
            'height': height,
            'weather': weather
        }
        coco_data['images'].append(image_entry)
        
        # Process objects/annotations
        objects = bdd_ann.get('objects', [])
        for obj in objects:
            class_title = obj.get('classTitle', '').lower()
            
            # Map class title to COCO category
            mapped_category = CATEGORY_MAPPING.get(class_title)
            if not mapped_category:
                # Skip unknown categories
                continue
            
            category_id = CATEGORY_NAME_TO_ID.get(mapped_category)
            if category_id is None:
                continue
            
            # Convert bbox to COCO format
            points = obj.get('points', {})
            if not points or 'exterior' not in points:
                continue
            
            try:
                bbox = convert_bbox_to_coco(points)
            except Exception as e:
                print(f"Error converting bbox for {img_filename}: {e}")
                continue
            
            # Skip invalid boxes with zero or negative width/height
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
            
            # Calculate area
            area = bbox[2] * bbox[3]  # width * height
            
            # Create annotation entry
            annotation_entry = {
                'id': annotation_id,
                'image_id': image_id,
                'category_id': category_id,
                'bbox': bbox,
                'area': area,
                'iscrowd': 0
            }
            coco_data['annotations'].append(annotation_entry)
            annotation_id += 1
        
        image_id += 1
        
        # Progress indicator
        if image_id % 1000 == 0:
            print(f"Processed {image_id} images...")
    
    # Save COCO JSON
    print(f"\nSaving COCO format to: {output_path}")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print("Conversion complete!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert BDD100K training data to COCO format'
    )
    parser.add_argument(
        '--bdd100k_train_path',
        type=str,
        default='BDD100K/train',
        help='Path to BDD100K/train directory (default: BDD100K/train)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='bdd100k_train_coco.json',
        help='Output COCO JSON file path (default: bdd100k_train_coco.json)'
    )
    
    args = parser.parse_args()
    
    convert_bdd100k_to_coco(args.bdd100k_train_path, args.output)
