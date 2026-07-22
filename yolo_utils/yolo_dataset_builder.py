"""
YOLO Dataset Builder

Functions for preparing temporary YOLO-format datasets:
- Creating/clearing temporary directories
- Copying real data
- Adding synthetic data with ranking-based selection
- Generating data.yaml configuration
"""
import os
import shutil
import numpy as np
import yaml
from tqdm import tqdm
from config_manager import ConfigManager
from data_utils import load_and_rank_synthetic_data


def clear_temp_dataset_dir(temp_dataset_dir: str):
    """Clear and create temporary directory."""

    # Clears contents of temp_dataset_dir if it exists
    for entry in os.scandir(temp_dataset_dir):
        try:
            if entry.is_file() or entry.is_symlink():
                os.unlink(entry.path)  # remove files and symlinks
            elif entry.is_dir():
                shutil.rmtree(entry.path)  # remove directories recursively
        except Exception as e:
            print(f"Failed to remove {entry.path}: {e}")


def copy_data_to_temp_yolo_dataset(config_manager: ConfigManager, split: str,
                                    images_dir: str, labels_dir: str):
    """Copy data for a specific split (train/val/test) to temporary dataset directory."""
    
    data_dir = config_manager.base_config["paths"]["data_dir"]
    paths = config_manager.base_config["paths"]

    shutil.copytree(
        os.path.join(data_dir, paths[split]["images"]),
        images_dir,
        dirs_exist_ok=True
    )
    shutil.copytree(
        os.path.join(data_dir, paths[split]["labels"]),
        labels_dir,
        dirs_exist_ok=True
    )
    print(f"Copied original {split} data.")


def calculate_replacement_counts(real_count: int, synthetic_dict: dict, mix_rate: float, is_addition: bool = False):
    """Calculate how many synthetic images to use per augmentation type."""
    if is_addition:
        # Addition Formula: synth = (mix_rate / (1 - mix_rate)) * num_real
        num_replace_total = int((mix_rate / (1.0 - mix_rate)) * real_count)
    else:
        # Replacement Formula: synth = mix_rate * num_real
        num_replace_total = int(real_count * mix_rate)

    aug_names = list(synthetic_dict.keys())
    num_augs = len(aug_names)
    if num_augs == 0 or num_replace_total == 0:
        return {aug: 0 for aug in aug_names}

    num_per_aug = num_replace_total // num_augs
    remainder = num_replace_total % num_augs
    replacement_counts = {}
    for i, aug in enumerate(aug_names):
        count = num_per_aug + (1 if i < remainder else 0)
        replacement_counts[aug] = min(count, len(synthetic_dict[aug]))
    return replacement_counts


def add_training_data(config_manager: ConfigManager, train_images_dir: str, train_labels_dir: str):
    """Copy real training data and add synthetic data to temporary dataset directory."""
    data_dir = config_manager.base_config["paths"]["data_dir"]
    paths = config_manager.base_config["paths"]
    
    shutil.copytree(
        os.path.join(data_dir, paths["train"]["images"]),
        train_images_dir,
        dirs_exist_ok=True
    )
    shutil.copytree(
        os.path.join(data_dir, paths["train"]["labels"]),
        train_labels_dir,
        dirs_exist_ok=True
    )
    initial_count = len(os.listdir(train_images_dir))
    print(f"Copied {initial_count} original training images.")

    # Determine mix rate: replacement OR addition
    # Handle both new 'mixing_method' style and old property style
    if hasattr(config_manager, "mixing_method"):
        if config_manager.mixing_method == "replacement":
            replacement_p = config_manager.mix_rate if config_manager.mix_rate is not None else 0.0
            addition_p = 0.0
        else: # addition
            replacement_p = 0.0
            addition_p = config_manager.mix_rate if config_manager.mix_rate is not None else 0.0
    else:
        # Fallback for old style logic (if any traces remain)
        replacement_p = getattr(config_manager, "replacement_percentage", 0.0)
        addition_p = getattr(config_manager, "addition_percentage", 0.0)
        
        if replacement_p is None: replacement_p = 0.0
        if addition_p is None: addition_p = 0.0

    
    # Use whichever is set (only one should be > 0 ideally, handled by ConfigManager logic)
    # If replacement > 0, we delete real images.
    # If addition > 0, we delete nothing but use addition_p to calculate count.
    
    # 1. Replacement Logic
    if replacement_p > 0 and addition_p == 0:
        # Strict replacement mode: Delete real images
        num_to_delete = int(len(os.listdir(train_images_dir)) * replacement_p)
        if num_to_delete > 0:
            print(f"Deleting {num_to_delete} real training images for replacement (Rate={replacement_p})...")
            all_train_images = [entry.name for entry in os.scandir(train_images_dir) if entry.is_file()]
            np.random.seed(42)
            images_to_delete = np.random.choice(all_train_images, num_to_delete, replace=False)
            for fname in tqdm(images_to_delete, desc="Deleting real images"):
                os.remove(os.path.join(train_images_dir, fname))
                base, _ = os.path.splitext(fname)
                os.remove(os.path.join(train_labels_dir, base + ".txt"))
            print("Deleted replaced real training images.")

    # Add synthetic images
    weather_augs = config_manager.base_config["experiment"]["weather_types"]
    real_train_images = os.listdir(os.path.join(data_dir, paths["train"]["images"]))
    real_count = len(real_train_images)
    
    # Target mix rate is either replacement_p or addition_p
    target_mix_rate = max(replacement_p, addition_p)
    
    synthetic_config = paths["synthetic"][config_manager.augmentation_type]
    synthetic_images_path = os.path.join(data_dir, synthetic_config["images"])

    synthetic_dict = {}
    for weather in weather_augs:
        synthetic_dict[weather] = [
            entry.path for entry in os.scandir(synthetic_images_path)
            if entry.is_file() and entry.name.startswith(weather)
        ]

    # Calculate how many synthetic to ADD
    is_addition = (addition_p > 0)
    replacement_counts = calculate_replacement_counts(real_count, synthetic_dict, target_mix_rate, is_addition=is_addition)
    synthetic_images_dir = os.path.join(data_dir, synthetic_config["images"])
    synthetic_labels_dir = os.path.join(data_dir, synthetic_config["labels"])
    synthetic_rankings_dir = os.path.join(data_dir, synthetic_config["rankings"])
    dataset_cutoff = config_manager.base_config["experiment"]["dataset_cutoff"]

    for weather, count in replacement_counts.items():
        if count == 0:
            continue
        
        # Load and rank synthetic data for this weather type
        ranking_file = os.path.join(synthetic_rankings_dir, f"{weather}.csv")
        # annotations not used for YOLO, but do need file names
        annotations_path = os.path.join(data_dir, synthetic_config["annotations"][weather])
        
        # Get top-X%-ranked synthetic data
        ranked_data = load_and_rank_synthetic_data(
            weather=weather,
            rankings_path=ranking_file,
            annotations_path=annotations_path,
            cutoff=dataset_cutoff
        )
        print(f"Loaded and ranked {len(ranked_data)} synthetic {weather} images.")
        
        # Randomly sample from the top X% for which ones to actually use
        np.random.seed(42)
        if count <= len(ranked_data):
            selected_indices = np.random.choice(len(ranked_data), count, replace=False)
            selected_data = [ranked_data[i] for i in selected_indices]
        else:
            # If we need more than available, just use all
            selected_data = ranked_data
        
        # Copy synthetic images and labels
        for item in tqdm(selected_data, desc=f"Adding synthetic {weather} images"):
            img_filename = item['name']  # Already has weather prefix
            
            # 2. Check for Name Clashes in Addition Mode
            # If we are adding (not replacing), we must ensure we don't accidentally overwrite a real image
            # that happens to share a filename (unlikely if synth has prefix, but safe to check).
            dst_img = os.path.join(train_images_dir, img_filename)
            
            # If file exists and we are in strict addition mode, we skip it to preserve real data
            if os.path.exists(dst_img) and replacement_p == 0 and addition_p > 0:
                print(f"[WARNING] Skipping synthetic image {img_filename} - file already exists (protecting real data)")
                continue

            # Copy image
            src_img = os.path.join(synthetic_images_dir, img_filename)
            # dst_img already defined above
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
            
            # Copy label (change extension to .txt)
            base_without_ext = os.path.splitext(img_filename)[0]
            src_label = os.path.join(synthetic_labels_dir, f"{base_without_ext}.txt")
            dst_label = os.path.join(train_labels_dir, f"{base_without_ext}.txt")
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
        
        print(f"Added {len(selected_data)} synthetic {weather} images (sampled from top {int(dataset_cutoff*100)}%)")
    
    print("Added synthetic images.")


def prepare_temp_dataset(config_manager: ConfigManager, temp_dataset_dir: str, keep_cached_data: bool):
    """
    Prepare a temporary dataset directory with YOLO format:
    temp_dataset/
      data.yaml
      train/
        images/
        labels/
      val/
        images/
        labels/
      test/
        images/
        labels/
    """
    print("Preparing temporary YOLO dataset...")
    # Create base directory first
    os.makedirs(temp_dataset_dir, exist_ok=True)
    
    data_yaml_path = os.path.join(temp_dataset_dir, "data.yaml")

    train_dir = os.path.join(temp_dataset_dir, "train")
    val_dir = os.path.join(temp_dataset_dir, "val")

    train_images_dir = os.path.join(train_dir, "images")
    train_labels_dir = os.path.join(train_dir, "labels")
    val_images_dir = os.path.join(val_dir, "images")
    val_labels_dir = os.path.join(val_dir, "labels")

    path = os.path.abspath(temp_dataset_dir)
    nc = config_manager.model_config["model"]["num_classes"]
    names = config_manager.model_config["model"]["names"]
    data_yaml = {"train": "train", "val": "val", "nc": nc, "names": names, "path": path}
    
    # Check if using cached data BEFORE writing data.yaml
    if keep_cached_data and os.path.exists(train_images_dir) and len(os.listdir(train_images_dir)) > 0:
        print(f"[INFO] Using cached temporary dataset directory: {temp_dataset_dir}")
        print(f"       Found {len(os.listdir(train_images_dir))} training images")
        # Still write data.yaml to ensure it's up to date
        with open(data_yaml_path, "w") as f:
            yaml.dump(data_yaml, f)
        return data_yaml_path
    
    # Not using cache - proceed with creation
    print("[INFO] Creating new temporary dataset directory...")

    # Clear and recreate subdirectories
    clear_temp_dataset_dir(temp_dataset_dir)
    
    # Always write data.yaml
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_yaml, f)
        
    print("Cleared temporary dataset directory.")
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    
    # Validation data: Use symlinks instead of copying (faster)
    # Ensure parent 'val' directory exists
    os.makedirs(val_dir, exist_ok=True)
    
    print("[INFO] Symlinking validation data...")
    data_dir = config_manager.base_config["paths"]["data_dir"]
    src_val_images = os.path.abspath(os.path.join(data_dir, config_manager.base_config["paths"]["val"]["images"]))
    src_val_labels = os.path.abspath(os.path.join(data_dir, config_manager.base_config["paths"]["val"]["labels"]))
    
    # Create symlinks
    if os.path.exists(src_val_images):
        # Linking the folder directly
        os.symlink(src_val_images, val_images_dir)
    else:
        print(f"[WARNING] Validation images not found at {src_val_images}")
        
    if os.path.exists(src_val_labels):
        # Linking the folder directly
        os.symlink(src_val_labels, val_labels_dir)
    else:
        print(f"[WARNING] Validation labels not found at {src_val_labels}")

    print("Adding training data...")
    add_training_data(config_manager, train_images_dir, train_labels_dir)

    # Count final number of training images
    final_train_count = len([f for f in os.listdir(train_images_dir) if os.path.isfile(os.path.join(train_images_dir, f))])
    print(f"[INFO] Final training dataset size: {final_train_count} images")

    print(f"Prepared dataset at: {temp_dataset_dir}")
    
    print(f"[DEBUG] data_yaml path: {data_yaml}")
    print(f"[DEBUG] data_yaml exists: {os.path.exists(data_yaml_path)}")
    return data_yaml_path


def create_temp_validation_dataset(dataset_name, images_path, labels_path, temp_base_dir, config_manager):
    """
    Creates a temporary standard YOLO structure for an additional validation dataset.
    Uses symlinks to avoid data duplication.
    structure:
       temp_base_dir/
          dataset_name/
             images/  -> symlink to images_path
             labels/  -> symlink to labels_path
             data.yaml
             
    Args:
        dataset_name: Name of dataset (e.g. "dawn")
        images_path: Absolute path to source images directory
        labels_path: Absolute path to source labels directory
        temp_base_dir: Directory to create structure in
        config_manager: ConfigManager for class names/counts
        
    Returns:
        Path to the generated data.yaml file
    """
    # Ensure source paths are absolute so symlinks resolve correctly
    images_path = os.path.abspath(images_path)
    if labels_path:
        labels_path = os.path.abspath(labels_path)

    dataset_dir = os.path.join(temp_base_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Define target paths
    target_images = os.path.join(dataset_dir, "images")
    target_labels = os.path.join(dataset_dir, "labels")
    
    # Remove existing symlinks/dirs if they exist
    if os.path.lexists(target_images):
        if os.path.islink(target_images): os.unlink(target_images)
        elif os.path.isdir(target_images): shutil.rmtree(target_images)
            
    if os.path.lexists(target_labels):
        if os.path.islink(target_labels): os.unlink(target_labels)
        elif os.path.isdir(target_labels): shutil.rmtree(target_labels)
    
    # Logic to handle Structure Mismatch (e.g. ACDC: nested images, flat labels)
    use_simple_symlink = True
    
    if os.path.exists(labels_path) and os.path.isdir(labels_path):
        # 1. Probe for mismatch by checking multiple images
        detected_mismatch = False
        detected_match = False
        
        # Check first 100 images to find ANY valid pair
        checked_count = 0
        MAX_CHECKS = 100
        
        for root, dirs, files in os.walk(images_path):
            valid_imgs = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))]
            for img_file in valid_imgs:
                checked_count += 1
                
                # Calculate paths
                rel_dir = os.path.relpath(root, images_path)
                if rel_dir == ".": rel_dir = ""
                sample_image_rel = os.path.join(rel_dir, img_file)
                base_name = os.path.splitext(img_file)[0]
                txt_name = base_name + ".txt"
                
                # Check Standard YOLO (Nested/Matching structure)
                nested_label_path = os.path.join(labels_path, os.path.dirname(sample_image_rel), txt_name)
                
                # Check Flat Label (ACDC/Dawn style)
                flat_label_path = os.path.join(labels_path, txt_name)
                
                if os.path.exists(nested_label_path):
                    detected_match = True # Found a standard pair, assume simple symlink is safe-ish
                    break
                elif os.path.exists(flat_label_path):
                    # Found a flat pair while image is nested (or not).
                    # If image is nested, this is a mismatch.
                    if os.path.dirname(sample_image_rel) and os.path.dirname(sample_image_rel) != ".":
                         detected_mismatch = True
                         break
            
            if detected_match or detected_mismatch or checked_count >= MAX_CHECKS:
                break
        
        # Decision Logic:
        # If we explicitly found a mismatch -> Use Complex Strategy
        # If we found matches -> Use Simple Strategy
        # If we found NOTHING (e.g. superset images, subset labels) -> Use Complex Strategy to filter
        if detected_mismatch:
             print(f"[INFO] Detected structure mismatch for {dataset_name}. Images are nested, labels are flat. Remapping...")
             use_simple_symlink = False
        elif not detected_match and checked_count > 0:
             print(f"[INFO] No direct nested matches found for {dataset_name} (checked {checked_count} files). Defaulting to intersection remapping to filter subset.")
             use_simple_symlink = False

    # Create symlinks
    try:
        if use_simple_symlink:
            # Standard case: direct symlinks for the whole folders
            os.symlink(images_path, target_images)
            print(f"[INFO] Created symlink for {dataset_name} images: {target_images} -> {images_path}")
            
            if os.path.exists(labels_path):
                os.symlink(labels_path, target_labels)
                print(f"[INFO] Created symlink for {dataset_name} labels: {target_labels} -> {labels_path}")
            else:
                print(f"[WARNING] Labels path does not exist: {labels_path}. Validation will likely fail.")
        else:
            # Structure Mismatch / Intersection Strategy
            # When images are nested but labels are flat (ACDC, DAWN, Foggy), 
            # and potentially the image folder contains unwanted splits (ACDC).
            # We recreate the structure by symlinking ONLY the images that have corresponding labels.
            # This effectively filters the dataset (e.g. keeps only 'train' subset) and fixes the nesting mismatch.
            
            print(f"[INFO] Starting complex intersection remapping for {dataset_name}...")
            os.makedirs(target_images, exist_ok=True)
            os.makedirs(target_labels, exist_ok=True)
            
            count_linked = 0
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
            
            # Walk image directory to find candidates
            for root, dirs, files in os.walk(images_path):
                for file in files:
                    if os.path.splitext(file)[1].lower() in image_extensions:
                        # Relative path of this image from images root (preserves nesting structure)
                        rel_dir = os.path.relpath(root, images_path)
                        base_name = os.path.splitext(file)[0]
                        
                        # 1. Identify Source Label (assuming flat structure in labels_path)
                        src_label_file = os.path.join(labels_path, base_name + ".txt")
                        
                        # 2. If valid pair exists, link both
                        if os.path.exists(src_label_file):
                            # Ensure subdirectories exist in target
                            target_img_subdir = os.path.join(target_images, rel_dir)
                            target_lbl_subdir = os.path.join(target_labels, rel_dir)
                            os.makedirs(target_img_subdir, exist_ok=True)
                            os.makedirs(target_lbl_subdir, exist_ok=True)
                            
                            # Link Image
                            src_img_file = os.path.abspath(os.path.join(root, file))
                            dst_img_link = os.path.join(target_img_subdir, file)
                            if not os.path.exists(dst_img_link):
                                os.symlink(src_img_file, dst_img_link)
                            
                            # Link Label (renaming flat label to match nested structure)
                            dst_lbl_link = os.path.join(target_lbl_subdir, base_name + ".txt")
                            src_label_file = os.path.abspath(src_label_file)
                            if not os.path.exists(dst_lbl_link):
                                os.symlink(src_label_file, dst_lbl_link)
                                
                            count_linked += 1
            
            print(f"[INFO] Successfully linked {count_linked} image/label pairs for {dataset_name}.")
            if count_linked == 0:
                print(f"[WARNING] No matching pairs found for {dataset_name}! Check filenames.")

    except Exception as e:
        print(f"[ERROR] Failed to create datasets for {dataset_name}: {e}")
    
    # Create data.yaml
    path = os.path.abspath(dataset_dir)
    nc = config_manager.model_config["model"]["num_classes"]
    names = config_manager.model_config["model"]["names"]
    
    # In the yaml, we point to the local 'images' symlink
    # YOLO requires 'path' to be absolute or relative to cwd. 
    # Providing absolute 'path' and relative 'val' usually works best.
    data_yaml = {
        "path": path,
        "train": "images", # dummy
        "val": "images",
        "nc": nc,
        "names": names
    }
    
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)
        
    return yaml_path
