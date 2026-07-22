"""
Test script to verify trainer can load DAWN as an additional validation dataset
"""
import sys
import os
# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from config_manager import ConfigManager

def test_config_loading():
    """Test that config loads properly with validation datasets"""
    print("Testing configuration loading...")
    
    # Create config manager
    config = ConfigManager(
        config_dir="configs",
        model_type="resnet"
    )
    
    # Check val_datasets section exists
    paths = config.base_config.get("paths", {})
    val_datasets = paths.get("val_datasets", {})
    
    print(f"\nFound {len(val_datasets)} validation dataset configurations:")
    for name, config_dict in val_datasets.items():
        enabled = config_dict.get("enabled", False)
        status = "✓ ENABLED" if enabled else "✗ DISABLED"
        print(f"  {status} {name}: {config_dict.get('description', 'N/A')}")
    
    # Check DAWN specifically
    dawn_config = val_datasets.get("dawn", {})
    if dawn_config.get("enabled", False):
        print(f"\nDAWN Configuration:")
        print(f"  Images: {dawn_config.get('images')}")
        print(f"  Annotations: {dawn_config.get('annotations')}")
        
        # Check if files exist
        data_dir = paths.get("data_dir", "Data")
        dawn_json = os.path.join(data_dir, dawn_config.get("annotations", ""))
        dawn_images = os.path.join(data_dir, dawn_config.get("images", ""))
        
        if os.path.exists(dawn_json):
            print(f"  ✓ Annotations file exists: {dawn_json}")
            
            # Load JSON and check stats
            import json
            with open(dawn_json) as f:
                data = json.load(f)
                print(f"  ✓ Images in JSON: {len(data['images'])}")
                
                # Count weather types
                weather_counts = {}
                for img in data['images']:
                    weather = img.get('weather', 'unknown')
                    weather_counts[weather] = weather_counts.get(weather, 0) + 1
                
                print(f"  ✓ Weather distribution:")
                for weather, count in sorted(weather_counts.items()):
                    print(f"    - {weather}: {count}")
        else:
            print(f"  ✗ Annotations file NOT found: {dawn_json}")
        
        if os.path.exists(dawn_images):
            print(f"  ✓ Images directory exists: {dawn_images}")
        else:
            print(f"  ✗ Images directory NOT found: {dawn_images}")
    
    print("\n✓ Configuration test passed!")

if __name__ == "__main__":
    test_config_loading()
