import os
import shutil
from pathlib import Path
from tqdm import tqdm
from PIL import Image

"""
Data preparation utilities for synthetic data replacement experiments
"""
import json
import pandas as pd
import os
import shutil
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
from config_manager import ConfigManager


def load_and_rank_synthetic_data(weather: str, rankings_path: str, annotations_path: str, cutoff: float = 0.8) -> List[Dict]:
    """
    Load synthetic data and return top percentage based on rankings
    
    Args:
        rankings_path: Path to CSV file with rankings
        annotations_path: Path to COCO annotations JSON
        cutoff: Percentage of top data to keep (0.8 = top 80%)
    
    Returns:
        List of annotation dictionaries sorted by quality
    """
    # Load rankings
    rankings_df = pd.read_csv(rankings_path)
    rankings_df = rankings_df.sort_values('metric_score')  # Lower scores are better
    
    # Calculate cutoff index
    cutoff_idx = int(len(rankings_df) * cutoff)
    top_images = rankings_df.head(cutoff_idx)['base_image_name'].tolist()
    
    # Load annotations
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create mapping from filename to image data
    image_map = {}
    for img in coco_data['images']:
        filename = os.path.basename(img['file_name'])
        image_map[filename] = img
    
    # Create mapping from image_id to annotations
    annot_map = {}
    for annot in coco_data['annotations']:
        img_id = annot['image_id']
        if img_id not in annot_map:
            annot_map[img_id] = []
        annot_map[img_id].append(annot)
    
    # Filter and sort data by rankings
    filtered_data = []
    for filename in top_images:
        filename = f"{weather}-{filename}"
        if filename in image_map:
            img_data = image_map[filename]
            img_id = img_data['id']
            annotations = annot_map.get(img_id, [])
            
            filtered_data.append({
                'image_id': img_id,
                'name': filename,
                'image_data': img_data,
                'annotations': annotations
            })
    return filtered_data


def create_bdd_subset(bdd_annotations_path: str, bdd_images_dir: str, target_size: int, output_dir: str) -> str:
    """
    Create a subset of BDD100K dataset with specified size
    
    Args:
        bdd_annotations_path: Path to BDD100K COCO annotations
        bdd_images_dir: Directory containing BDD100K images
        target_size: Number of images to include
        output_dir: Output directory for subset
    
    Returns:
        Path to new annotations file
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    # Load original annotations
    with open(bdd_annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    # Randomly sample images
    np.random.seed(42)  # For reproducibility
    selected_indices = np.random.choice(len(coco_data['images']), target_size, replace=False)
    selected_images = [coco_data['images'][i] for i in selected_indices]
    selected_image_ids = {img['id'] for img in selected_images}
    
    # Filter annotations for selected images
    filtered_annotations = [
        annot for annot in coco_data['annotations'] 
        if annot['image_id'] in selected_image_ids
    ]
    
    # Copy selected images
    print(f"Copying {len(selected_images)} images...")
    for img in tqdm(selected_images):
        src_path = os.path.join(bdd_images_dir, img['file_name'])
        dst_path = os.path.join(output_dir, "images", img['file_name'])
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
    
    # Create new annotations file
    new_coco_data = {
        'images': selected_images,
        'annotations': filtered_annotations,
        'categories': coco_data['categories']
    }
    
    annotations_output_path = os.path.join(output_dir, "annotations.json")
    with open(annotations_output_path, 'w') as f:
        json.dump(new_coco_data, f)
    
    return annotations_output_path


def prepare_synthetic_datasets(config_manager: ConfigManager, augmentation_type: str):
    """
    Prepare synthetic datasets for a specific augmentation type
    
    Args:
        config_manager: ConfigManager instance
        augmentation_type: "gemini" or "automold"
    
    Returns:
        Dict with weather types as keys and filtered data as values
    """
    print(f"Preparing {augmentation_type} synthetic datasets...")
    paths_config = config_manager.get_paths(augmentation_type)
    experiment_config = config_manager.get_experiment_settings()
    dataset_cutoff = experiment_config["dataset_cutoff"]
    weather_types = experiment_config["weather_types"]
    
    prepared_data = {}
    for weather in weather_types:
        print(f"Processing {augmentation_type} {weather} data...")
        
        filtered_data = load_and_rank_synthetic_data(
            weather,
            paths_config["synthetic"]["rankings"][weather], 
            paths_config["synthetic"]["annotations"][weather], 
            dataset_cutoff
        )
        
        prepared_data[weather] = filtered_data
        print(f"  Loaded {len(filtered_data)} top-quality {weather} images")
    
    return prepared_data


def create_experiment_datasets(prepared_synthetic_data: Dict, mix_rate: float, 
                             augmentation_type: str, 
                             config_manager: ConfigManager) -> Tuple[str, Dict, str]:
    """
    Create dataset configuration for a specific mix rate
    
    Args:
        prepared_synthetic_data: Pre-processed synthetic data for the specified augmentation type
        mix_rate: Fraction of data to replace/add (0.0 to 1.0)
        augmentation_type: "gemini" or "automold"
        config_manager: Configuration manager instance
    
    Returns:
        Tuple of (real_annotations_path, synthetic_annotations_dict, synthetic_img_dir)
    """
    paths_config = config_manager.get_paths(augmentation_type)
    experiment_config = config_manager.get_experiment_settings()
    
    # Calculate how many images to replace or add
    dataset_size = experiment_config["dataset_size"]
    
    mixing_method = getattr(config_manager, "mixing_method", "replacement")
    
    if mixing_method == "addition":
        if mix_rate >= 1.0:
            raise ValueError("For addition, percentage must be < 1.0 (it represents final synthetic ratio)")
        # S / (R + S) = rate  =>  S = rate * R + rate * S  =>  S(1 - rate) = rate * R  => S = R * rate / (1 - rate)
        num_target = int(dataset_size * (mix_rate / (1 - mix_rate)))
    else:
        # Replacement
        num_target = int(dataset_size * mix_rate)
    
    if num_target == 0:
        # No replacement - return original dataset
        return paths_config["bdd"]["train_annotations"], {}, ""
    
    # Distribute replacements evenly across weather types
    weather_types = experiment_config["weather_types"]
    num_per_weather = num_target // len(weather_types)
    remainder = num_target % len(weather_types)
    
    synthetic_annotations = {}
    
    # Randomly sample from top 80% for each weather type
    np.random.seed(42)  # For reproducibility
    for i, weather in enumerate(weather_types):
        # Add remainder to first weather types
        count = num_per_weather + (1 if i < remainder else 0)
        
        if count > 0:
            weather_data = prepared_synthetic_data[weather]
            
            # Randomly sample 'count' images from the top 80%
            if count <= len(weather_data):
                sampled_indices = np.random.choice(len(weather_data), size=count, replace=False)
                sampled_data = [weather_data[idx] for idx in sampled_indices]
            else:
                print(f"[WARNING] Requested {count} {weather} images but only {len(weather_data)} available. Using all.")
                sampled_data = weather_data
            
            # Create annotations in expected format
            synthetic_annotations[weather] = []
            for item in sampled_data:
                synthetic_annotations[weather].append({
                    'image_id': item['image_id'],
                    'name': item['name'],
                    'annotations': item['annotations']
                })
    
    # Get the synthetic image directory
    synthetic_img_dir = paths_config["synthetic"]["images"]
    
    return paths_config["bdd"]["train_annotations"], synthetic_annotations, synthetic_img_dir


def create_bdd_train_subset(config_manager: ConfigManager = None):
    """
    Create a training subset of BDD100K if it doesn't exist
    """
    if config_manager is None:
        config_manager = ConfigManager()
    
    paths_config = config_manager.get_paths()
    experiment_config = config_manager.get_experiment_settings()
    
    train_dir = os.path.join(paths_config["data_dir"], "bdd100k_train")
    if not os.path.exists(os.path.join(train_dir, "annotations.json")):
        print("Creating BDD100K training subset...")
        
        # Use validation set as source (since we have it)
        # In practice, you'd use the actual training set
        create_bdd_subset(
            paths_config["bdd"]["val_annotations"],
            paths_config["bdd"]["val_images"],
            experiment_config["dataset_size"],
            train_dir
        )
        
        # Update paths in config
        paths_config["bdd"]["train_annotations"] = os.path.join(train_dir, "annotations.json")
        paths_config["bdd"]["train_images"] = os.path.join(train_dir, "images")

if __name__ == "__main__":
    # Test the simplified config manager
    config_manager = ConfigManager(model_type="resnet")
    
    # Prepare synthetic datasets for gemini
    synthetic_data = prepare_synthetic_datasets(config_manager, "gemini")
    
    print("Data preparation complete!")
    print(f"Gemini data loaded: {sum(len(v) for v in synthetic_data.values())} images")

