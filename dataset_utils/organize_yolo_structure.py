#!/usr/bin/env python3
"""
Utility to reorganize dataset structures to be YOLO-friendly.

Problem: 
Many datasets (ACDC, DAWN, Foggy Zurich) maintain complex folder structures for images 
(e.g. by weather, city, or split) but provide labels in a single flat directory or inconsistent format.
YOLO training requires the label directory structure to mirror the image directory structure exactly.

Solution:
This script scans the image directory structure and reorganizes the label directory 
to match it perfectly, copying/moving the corresponding flat label files into the 
correct nested subdirectories.

Usage:
    python organize_yolo_structure.py --dataset acdc --root ../Data/ACDC
    python organize_yolo_structure.py --dataset dawn --root ../Data/DAWN
"""

import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

def organize_dataset(dataset_name, root_path, images_rel, labels_source_rel, labels_dest_rel="labels_yolo", dry_run=False):
    """
    Organize labels to match image structure.
    
    Args:
        dataset_name: Name of dataset
        root_path: Root of the dataset (e.g. Data/ACDC)
        images_rel: Relative path to images root (e.g. rgb_anon)
        labels_source_rel: Relative path to source flat labels (e.g. labels)
        labels_dest_rel: Relative path to destination nested labels (e.g. labels_yolo)
        dry_run: If True, only print what would happen
    """
    root = Path(root_path).resolve()
    images_root = root / images_rel
    labels_source = root / labels_source_rel
    labels_dest = root / labels_dest_rel

    if not images_root.exists():
        print(f"[ERROR] Images root not found: {images_root}")
        return

    if not labels_source.exists():
        print(f"[ERROR] Source labels root not found: {labels_source}")
        return

    print(f"\n=== Organizing {dataset_name} ===")
    print(f"Root: {root}")
    print(f"Images: {images_rel} (Scanning nested structure)")
    print(f"Source Labels: {labels_source_rel} (Flat/Partial Structure)")
    print(f"Dest Labels: {labels_dest_rel} (Target Nested Structure)")

    # 1. Index available labels
    label_index = {}
    print("Indexing source labels...")
    
    for p in labels_source.rglob("*.txt"):
        if p.name == "classes.txt" or p.name == "dataset_stats.txt": continue
        label_index[p.stem] = p

    print(f"Found {len(label_index)} source label files.")

    # 2. Process image tree
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    count_copied = 0
    count_missing = 0
    
    print("Scanning images and creating label structure...")
    
    for image_path in tqdm(list(images_root.rglob("*"))):
        if not image_path.is_file() or image_path.suffix.lower() not in valid_extensions:
            continue
            
        rel_path = image_path.relative_to(images_root)
        stem = image_path.stem
        
        if stem in label_index:
            src_label = label_index[stem]
            
            # Determine target label path
            # It should be at labels_dest / rel_path.parent / (stem + .txt)
            target_dir = labels_dest / rel_path.parent
            target_label = target_dir / (stem + ".txt")
            
            if not dry_run:
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_label, target_label)
                
            count_copied += 1
        else:
            # print(f"Missing label for {rel_path}")
            count_missing += 1

    print(f"Processed {dataset_name}:")
    print(f"  - Copied {count_copied} labels to {labels_dest_rel}")
    print(f"  - {count_missing} images have no labels (e.g. test set).")
    
    if dry_run:
        print("[DRY RUN] No changes were made.")


def main():
    parser = argparse.ArgumentParser(description="Restructure labels to match nested image folders.")
    parser.add_argument("--root", type=str, required=True, help="Path to dataset root (e.g. Data/ACDC)")
    parser.add_argument("--dataset", type=str, required=True, choices=['acdc', 'dawn', 'foggy', 'custom'], help="Dataset preset")
    parser.add_argument("--images", type=str, help="Name of images folder (overrides preset)")
    parser.add_argument("--src_labels", type=str, help="Name of source labels folder (overrides preset)")
    parser.add_argument("--dest_labels", type=str, default="labels_yolo", help="Name of destination labels folder")
    parser.add_argument("--dry_run", action="store_true", help="Don't actually move/copy files")
    
    args = parser.parse_args()
    
    # Presets
    if args.dataset == 'acdc':
        # ACDC: Images in 'rgb_anon', Labels in 'labels'
        img_dir = args.images or "rgb_anon"
        src_lbl = args.src_labels or "labels"
        
    elif args.dataset == 'dawn':
        # DAWN: Images in 'DAWN', Labels in 'labels'
        img_dir = args.images or "DAWN"
        src_lbl = args.src_labels or "labels"
        
    elif args.dataset == 'foggy':
        # Foggy Zurich: Images in 'Foggy_Driving/leftImg8bit', Labels in 'labels'
        img_dir = args.images or "Foggy_Driving/leftImg8bit"
        src_lbl = args.src_labels or "labels"
        
    else: # Custom
        if not args.images or not args.src_labels:
            print("[ERROR] For custom dataset, provide --images and --src_labels.")
            return
        img_dir = args.images
        src_lbl = args.src_labels

    organize_dataset(args.dataset, args.root, img_dir, src_lbl, args.dest_labels, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
