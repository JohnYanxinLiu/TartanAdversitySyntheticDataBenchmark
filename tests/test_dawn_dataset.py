"""
Quick test to verify DAWN dataset integration
"""
import os
import sys
import json

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from dataset import MixedGenDataSet
import torch

def test_dawn_dataset():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.abspath(os.path.join(script_dir, "..", "Data"))
    
    annotations_path = os.path.join(data_root, "DAWN", "dawn_coco.json")
    images_path = os.path.join(data_root, "DAWN", "DAWN")
    
    print(f"Testing DAWN dataset loading...")
    print(f"Annotations: {annotations_path}")
    print(f"Images: {images_path}")
    
    # Load COCO JSON
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    print(f"\nCOCO JSON stats:")
    print(f"  Images: {len(coco_data['images'])}")
    print(f"  Annotations: {len(coco_data['annotations'])}")
    print(f"  Categories: {len(coco_data['categories'])}")
    
    # Create dataset
    training_config = {
        'use_augmentation': False,
        'image_size': [720, 1280]
    }
    
    dataset = MixedGenDataSet(
        real_annotations_json=annotations_path,
        real_img_dir=images_path,
        augmentation_names=None,
        synthetic_annotations_jsons=None,
        synthetic_img_dir=None,
        aug_percentage_cutoffs=None,
        mixing_method=None,
        mix_rate=None,
        is_train=False,
        training_config=training_config
    )
    
    # Apply normalization
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
    imagenet_std = torch.tensor([0.229, 0.224, 0.225])
    dataset.update_normalization(imagenet_mean, imagenet_std)
    
    print(f"\nDataset created successfully!")
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading first image
    img, target = dataset[0]
    print(f"\nFirst sample:")
    print(f"  Image shape: {img.shape}")
    print(f"  Image ID: {target['image_id']}")
    print(f"  Weather: {target.get('weather', 'N/A')}")
    print(f"  Filename: {target['filename']}")
    print(f"  Boxes: {target['boxes'].shape}")
    print(f"  Labels: {target['labels'].shape}")
    
    # Test samples from different weather
    weather_samples = {}
    for i in range(min(len(dataset), 100)):
        _, target = dataset[i]
        weather = target.get('weather', 'unknown')
        if weather not in weather_samples:
            weather_samples[weather] = i
    
    print(f"\nWeather type samples found:")
    for weather, idx in sorted(weather_samples.items()):
        print(f"  {weather}: index {idx}")
    
    print(f"\n✓ DAWN dataset test passed!")

if __name__ == "__main__":
    test_dawn_dataset()
