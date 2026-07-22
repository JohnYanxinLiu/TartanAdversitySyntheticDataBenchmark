#!/usr/bin/env python3
"""
Convert COCO JSON annotations to YOLO format labels.

YOLO format: <class_id> <x_center> <y_center> <width> <height>
All values are normalized [0, 1] relative to image dimensions.
"""

import json
import argparse
import os
from pathlib import Path
from collections import defaultdict


def coco_to_yolo_bbox(bbox, img_width, img_height):
    """
    Convert COCO bbox format to YOLO format.
    
    COCO: [x_min, y_min, width, height] (absolute pixels)
    YOLO: [x_center, y_center, width, height] (normalized 0-1)
    
    Args:
        bbox: COCO bbox [x, y, w, h]
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        YOLO bbox [x_center, y_center, w, h] (normalized)
    """
    x_min, y_min, w, h = bbox
    
    # Calculate center
    x_center = x_min + w / 2
    y_center = y_min + h / 2
    
    # Normalize by image dimensions
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    
    # Clamp to [0, 1]
    x_center_norm = max(0, min(1, x_center_norm))
    y_center_norm = max(0, min(1, y_center_norm))
    w_norm = max(0, min(1, w_norm))
    h_norm = max(0, min(1, h_norm))
    
    return [x_center_norm, y_center_norm, w_norm, h_norm]


def convert_coco_to_yolo(coco_json_path, output_labels_dir, images_dir=None):
    """
    Convert COCO JSON annotations to YOLO format label files.
    
    Args:
        coco_json_path: Path to COCO JSON file
        output_labels_dir: Directory to save YOLO label .txt files
        images_dir: Optional directory containing images (for validation)
    """
    print(f"Loading COCO JSON: {coco_json_path}")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directory
    os.makedirs(output_labels_dir, exist_ok=True)
    print(f"Output labels directory: {output_labels_dir}")
    
    # Build image ID to info mapping
    image_info = {}
    for img in coco_data['images']:
        image_info[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }
    
    # Group annotations by image ID
    annotations_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)
    
    print(f"\nProcessing {len(image_info)} images with {len(coco_data['annotations'])} annotations...")
    
    # Convert each image's annotations
    converted_count = 0
    
    for img_id, info in image_info.items():
        file_name = info['file_name']
        img_width = info['width']
        img_height = info['height']
        
        # Get base filename without extension
        base_name = Path(file_name).stem
        output_label_path = os.path.join(output_labels_dir, f"{base_name}.txt")
        
        # Get annotations for this image
        anns = annotations_by_image[img_id]
        
        # Convert annotations to YOLO format
        yolo_lines = []
        for ann in anns:
            category_id = ann['category_id'] - 1  # Assuming category IDs start at 1 in COCO
            bbox = ann['bbox']
            
            # Convert to YOLO format
            yolo_bbox = coco_to_yolo_bbox(bbox, img_width, img_height)
            
            # YOLO format: class x_center y_center width height
            yolo_line = f"{category_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
            yolo_lines.append(yolo_line)
        
        # Write to label file
        with open(output_label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
            if yolo_lines:  # Add final newline if file is not empty
                f.write('\n')
        
        converted_count += 1
        if converted_count % 1000 == 0:
            print(f"  Converted {converted_count}/{len(image_info)} images...")
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Converted: {converted_count} images")
    print(f"  Total annotations: {len(coco_data['annotations'])}")
    print(f"  Output directory: {output_labels_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Convert COCO JSON annotations to YOLO format labels')
    parser.add_argument('--coco_json', type=str, required=True,
                       help='Path to COCO JSON file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save YOLO label .txt files')
    parser.add_argument('--images_dir', type=str,
                       help='Optional: Directory containing images (for validation)')
    
    args = parser.parse_args()
    
    convert_coco_to_yolo(args.coco_json, args.output_dir, args.images_dir)


if __name__ == '__main__':
    main()
