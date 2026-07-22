"""
Test loading ACDC and DAWN datasets with MixedGenDataSet to verify integration.
"""
import sys
import os
# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from dataset import MixedGenDataSet
from config_manager import ConfigManager
import torch

def test_dataset_loading(dataset_name, annotations_path, images_path):
    """Test loading a dataset"""
    print(f"\n{'='*60}")
    print(f"Testing {dataset_name.upper()} Dataset")
    print(f"{'='*60}")
    
    try:
        # Create dataset
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
            training_config={
                "image_size": [720, 1280],
                "resize": [720, 1280],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        )
        
        print(f"✓ Dataset created successfully")
        print(f"  Total samples: {len(dataset)}")
        
        if len(dataset) == 0:
            print(f"  ✗ WARNING: Dataset is empty!")
            return False
        
        # Test loading first sample
        img, target = dataset[0]
        
        print(f"\n  First sample:")
        print(f"    Image shape: {img.shape}")
        print(f"    Image ID: {target.get('image_id', 'N/A')}")
        print(f"    Weather: {target.get('weather', 'unknown')}")
        
        if 'boxes' in target:
            print(f"    Boxes: {target['boxes'].shape}")
            print(f"    Labels: {target['labels'].shape}")
            
            if len(target['labels']) > 0:
                # Check label range
                min_label = target['labels'].min().item()
                max_label = target['labels'].max().item()
                print(f"    Label range: {min_label} to {max_label}")
                
                if min_label < 0 or max_label > 9:
                    print(f"    ✗ WARNING: Labels outside BDD100K range [0-9]!")
                else:
                    print(f"    ✓ Labels are in valid range")
        
        # Test a few more samples to check diversity
        weather_types = set()
        for i in range(min(10, len(dataset))):
            _, target = dataset[i]
            weather_types.add(target.get('weather', 'unknown'))
        
        print(f"\n  Weather types found (first 10 samples): {sorted(weather_types)}")
        
        # Count total annotations
        total_boxes = 0
        for i in range(min(100, len(dataset))):
            _, target = dataset[i]
            if 'boxes' in target:
                total_boxes += len(target['boxes'])
        
        print(f"  Total boxes in first {min(100, len(dataset))} samples: {total_boxes}")
        
        print(f"\n✓ {dataset_name.upper()} dataset test PASSED!")
        return True
        
    except Exception as e:
        print(f"✗ {dataset_name.upper()} dataset test FAILED!")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("VALIDATION DATASETS INTEGRATION TEST")
    print("="*60)
    
    # Load config
    config = ConfigManager(config_dir="configs", model_type="resnet")
    paths = config.base_config["paths"]
    data_dir = paths.get("data_dir", "Data")
    val_datasets = paths.get("val_datasets", {})
    
    results = {}
    
    # Test each enabled dataset
    for dataset_name, dataset_config in val_datasets.items():
        if not dataset_config.get("enabled", False):
            print(f"\n[SKIP] {dataset_name.upper()} is disabled in config")
            continue
        
        annotations = os.path.join(data_dir, dataset_config["annotations"])
        images = os.path.join(data_dir, dataset_config["images"])
        
        if not os.path.exists(annotations):
            print(f"\n[SKIP] {dataset_name.upper()} annotations not found: {annotations}")
            continue
        
        if not os.path.exists(images):
            print(f"\n[SKIP] {dataset_name.upper()} images not found: {images}")
            continue
        
        results[dataset_name] = test_dataset_loading(dataset_name, annotations, images)
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    if not results:
        print("No enabled datasets found to test")
        return
    
    for dataset_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {dataset_name.upper()}")
    
    all_passed = all(results.values())
    print(f"\n{'='*60}")
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
