"""
Convert DAWN dataset to COCO format for validation.
DAWN contains images with PASCAL VOC annotations, organized by weather type.

DAWN Categories (from PASCAL VOC annotations):
- bicycle, bus, car, motorcycle, person, truck

BDD100K categories (IDs 0-9):
bike, bus, car, motor, person, rider, traffic light, traffic sign, train, truck

Label Mapping:
- bicycle -> bike (0)
- bus -> bus (1)
- car -> car (2)
- motorcycle -> motor (3)
- person -> person (4)
- truck -> truck (9)
- rider, traffic light, traffic sign, train -> Not in DAWN
"""
import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
import sys
from pathlib import Path

# Import YOLO conversion utility
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir.parent))  # repo root, for weather_utils
from coco_to_yolo_labels import convert_coco_to_yolo
from weather_utils import canonical_weather_or_other

# Label mapping from DAWN to BDD100K
LABEL_MAPPING = {
    "bicycle": 1,     # bike
    "bus": 2,         # bus
    "car": 3,         # car
    "motorcycle": 4,  # motor
    "person": 5,      # person
    "truck": 10,       # truck
}

def parse_pascal_voc(xml_file):
    """
    Parse PASCAL VOC XML annotation file.
    
    Returns:
        list of dicts with keys: name, xmin, ymin, xmax, ymax
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        objects.append({
            'name': name,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        })
    
    return objects

def create_dawn_coco_json(dawn_root, output_json):
    """
    Create COCO-format JSON for DAWN dataset with PASCAL VOC annotations.
    
    Args:
        dawn_root: Path to DAWN root directory containing Fog/, Rain/, Snow/
        output_json: Path to output COCO JSON file
    """
    # BDD100K categories (1-indexed COCO ids, matching convert_bdd100k_to_coco.py,
    # ACDC, the synthetic data, and the LABEL_MAPPING targets below).
    categories = [
        {"id": 1, "name": "bike"},
        {"id": 2, "name": "bus"},
        {"id": 3, "name": "car"},
        {"id": 4, "name": "motor"},
        {"id": 5, "name": "person"},
        {"id": 6, "name": "rider"},
        {"id": 7, "name": "traffic light"},
        {"id": 8, "name": "traffic sign"},
        {"id": 9, "name": "train"},
        {"id": 10, "name": "truck"}
    ]
    
    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": categories,
        "info": {
            "description": "DAWN Dataset - Detection in Adverse Weather and Night",
            "url": "https://data.mendeley.com/datasets/766ygrbt8y/3",
            "version": "3",
            "year": 2020
        },
        "licenses": []
    }
    
    # Exclude 'Sand' as it's not relevant for typical weather conditions
    weather_types = ["Fog", "Rain", "Snow"]
    image_id = 0
    annotation_id = 1  # 0 is reserved as "unmatched" in COCOeval match matrices
    
    for weather in weather_types:
        weather_dir = os.path.join(dawn_root, weather)
        if not os.path.exists(weather_dir):
            print(f"[WARNING] Weather directory not found: {weather_dir}")
            continue
        
        # Find PASCAL VOC annotation directory
        voc_dir = os.path.join(weather_dir, f"{weather}_PASCAL_VOC")
        if not os.path.exists(voc_dir):
            print(f"[WARNING] PASCAL VOC directory not found: {voc_dir}")
            continue
            
        image_files = sorted([f for f in os.listdir(weather_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"Processing {weather}: {len(image_files)} images")
        
        for img_file in tqdm(image_files, desc=f"{weather}"):
            img_path = os.path.join(weather_dir, img_file)
            
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    
                # Determine weather from filename prefix, falling back to the
                # directory name. The Fog directory mixes foggy/haze/mist, which
                # all collapse to the canonical "foggy" condition.
                img_prefix = os.path.splitext(img_file)[0].rsplit('-', 1)[0]  # e.g., "foggy", "haze", "mist", "rain_storm"
                weather_type = canonical_weather_or_other(img_prefix)
                if weather_type == "other":
                    weather_type = canonical_weather_or_other(weather)

                image_info = {
                    "id": image_id,
                    "file_name": os.path.join(weather, img_file),
                    "width": width,
                    "height": height,
                    "weather": weather_type  # canonical: clear, rainy, foggy, snowy
                }
                
                coco_dict["images"].append(image_info)
                
                # Parse PASCAL VOC annotations for this image
                xml_file = os.path.join(voc_dir, os.path.splitext(img_file)[0] + '.xml')
                if os.path.exists(xml_file):
                    objects = parse_pascal_voc(xml_file)
                    
                    for obj in objects:
                        # Map DAWN label to BDD100K category ID
                        dawn_label = obj['name']
                        if dawn_label not in LABEL_MAPPING:
                            continue  # Skip unmapped labels
                        
                        category_id = LABEL_MAPPING[dawn_label]
                        
                        # Convert to COCO bbox format [x, y, width, height]
                        x = obj['xmin']
                        y = obj['ymin']
                        w = obj['xmax'] - obj['xmin']
                        h = obj['ymax'] - obj['ymin']
                        
                        annotation = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "iscrowd": 0
                        }
                        
                        coco_dict["annotations"].append(annotation)
                        annotation_id += 1
                
                image_id += 1
                
            except Exception as e:
                print(f"[ERROR] Failed to process {img_path}: {e}")
    
    # Save COCO JSON
    with open(output_json, 'w') as f:
        json.dump(coco_dict, f, indent=2)
    
    # Collect statistics
    weather_counts = {}
    for img in coco_dict["images"]:
        weather = img["weather"]
        weather_counts[weather] = weather_counts.get(weather, 0) + 1
    
    # Count labels
    from collections import Counter
    label_counts = Counter()
    images_with_anns = set()
    for ann in coco_dict['annotations']:
        cat_id = ann['category_id']
        cat_name = [c['name'] for c in coco_dict['categories'] if c['id'] == cat_id][0]
        label_counts[cat_name] += 1
        images_with_anns.add(ann['image_id'])
    
    images_without_anns = len(coco_dict['images']) - len(images_with_anns)
    
    # Save statistics file
    stats_file = output_json.replace('.json', '_stats.txt')
    with open(stats_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("DAWN DATASET STATISTICS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total images: {len(coco_dict['images'])}\n")
        f.write(f"Images with annotations: {len(images_with_anns)}\n")
        f.write(f"Images without annotations: {images_without_anns}\n")
        f.write(f"Total annotations: {len(coco_dict['annotations'])}\n\n")
        
        f.write("Weather Distribution (fine-grained):\n")
        f.write("-" * 40 + "\n")
        for weather, count in sorted(weather_counts.items()):
            f.write(f"  {weather:10} {count:4} images\n")
        
        f.write("\nLabel Distribution:\n")
        f.write("-" * 40 + "\n")
        for label, count in sorted(label_counts.items()):
            f.write(f"  {label:15} {count:5} annotations\n")
        
        f.write("\nMissing BDD100K Categories:\n")
        f.write("-" * 40 + "\n")
        all_cats = {c['name'] for c in coco_dict['categories']}
        present_cats = set(label_counts.keys())
        missing_cats = all_cats - present_cats
        for cat in sorted(missing_cats):
            f.write(f"  {cat}\n")
    
    # Print statistics
    print(f"\nSuccessfully created COCO JSON with {len(coco_dict['images'])} images")
    print(f"Saved to: {output_json}")
    print(f"Statistics saved to: {stats_file}")
    
    print("\nWeather distribution (fine-grained):")
    for weather, count in sorted(weather_counts.items()):
        print(f"  {weather}: {count}")
    
    print(f"\nTotal annotations: {len(coco_dict['annotations'])}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.abspath(os.path.join(script_dir, "../Data"))
    
    dawn_root = os.path.join(data_root, "DAWN", "DAWN")
    output_json = os.path.join(data_root, "DAWN", "dawn_coco.json")
    
    if not os.path.exists(dawn_root):
        print(f"DAWN directory not found: {dawn_root}")
        print("Please run download_dawn.py first")
        return
    
    create_dawn_coco_json(dawn_root, output_json)
    
    # Convert to YOLO format
    print("\n" + "=" * 80)
    print("Converting to YOLO format...")
    print("=" * 80)
    yolo_labels_dir = os.path.join(data_root, "DAWN", "labels")
    convert_coco_to_yolo(
        coco_json_path=output_json,
        output_labels_dir=yolo_labels_dir,
        images_dir=dawn_root
    )
    print(f"\nYOLO labels saved to: {yolo_labels_dir}")

if __name__ == "__main__":
    main()
