from torch.utils.data import Dataset
import pandas as pd
from enum import Enum
import json
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch
from tqdm import tqdm
from torchvision.transforms import functional as F
import torchvision.transforms.v2 as T2
from torchvision import tv_tensors
import albumentations as A

from weather_utils import canonical_weather_or_other


# Augmentation method for synthetic data
class AugmentationType(Enum):
    AUTOMOLD="automold" # Automold-generated synthetic data
    GEMINI="gemini" # Gemini-generated synthetic data

# The method to mix synthetic data into real.
class MixingMethod(Enum):
    REPLACEMENT="replacement" # Takes the set of real world data, and replaces real data points with synthetic
    ADDITION="addition" # Simply adds synthetic data points to the real data set



# TODO: Might need to coalesce all the synthetic data into one big json file.
class MixedGenDataSet(Dataset):
    
    """
    A dataset class for mixed real and synthetic data.
    real_annotations is expected to be the COCO-labeled annotations of the real images being used for training (remove any undesired images from training).
    real_img_dir can be the directory containing the real images
    synthetic_annotations is expected to be the COCO-labeled annotations of the synthetic images
    synthetic_snow/rain/fog_rankings_file
    """
    
    def __init__(self, real_annotations_json : str, 
                real_img_dir : str,
                augmentation_names : list, # ["snow", "rain", "fog"] if you are making snowy/rainy/foggy images. Used as keys in dictionary for 
                synthetic_annotations_jsons : dict, # Dictionary of synthetic annotations JSON files, e.g. {"snow": "synthetic_snow_annotations.json", "rain": "synthetic_rain_annotations.json", "fog": "synthetic_fog_annotations.json"}
                                                    # Note, annotations in json files should be sorted from "best" to "worst" to keep for training.
                synthetic_img_dir : str, # Single directory containing all synthetic images
                aug_percentage_cutoffs : dict, # Dictionary of augmentation percentage cutoffs, e.g. {"snow": 0.8, "rain": 0.7, "fog": 0.6}
                mixing_method : MixingMethod,
                mix_rate : float,
                is_train : bool,
                dataset_size : int = None, # Maximum number of real images to load (None for no limit)
                training_config : dict = None
                ):

        self._validate_inputs(
            real_annotations_json=real_annotations_json,
            real_img_dir=real_img_dir,
            augmentation_names=augmentation_names,
            synthetic_annotations_jsons=synthetic_annotations_jsons,
            synthetic_img_dir=synthetic_img_dir,
            aug_percentage_cutoffs=aug_percentage_cutoffs,
            mixing_method=mixing_method,
            mix_rate=mix_rate,
            is_train=is_train,
            dataset_size=dataset_size
        )

        real_annotations = json.load(open(real_annotations_json, 'r'))
        
        # Build real image file paths from COCO format
        real_img_file_paths = []
        real_image_ids = []
        real_annotations_list = []
        real_weather_metadata = []  # Track weather from COCO images
        
        # Create mapping from image_id to annotations
        annot_map = {}
        for annot in real_annotations['annotations']:
            img_id = annot['image_id']
            if img_id not in annot_map:
                annot_map[img_id] = []
            annot_map[img_id].append(annot)
        
        # Process images (limit to dataset_size if specified)
        images_to_process = real_annotations['images']
        if dataset_size is not None and len(images_to_process) > dataset_size:
            images_to_process = images_to_process[:dataset_size]
            print(f"Limiting dataset to {dataset_size} images (from {len(real_annotations['images'])} total)")
        
        for img in images_to_process:
            real_img_file_paths.append(os.path.join(real_img_dir, img['file_name']))
            real_image_ids.append(img['id'])
            real_annotations_list.append(annot_map.get(img['id'], []))
            # Extract weather from COCO JSON (should be pre-populated using add_weather_to_coco.py)
            real_weather_metadata.append(img.get('weather', 'unknown'))

        synthetic_annotations = {}
        synthetic_img_file_paths = {}

        if augmentation_names is not None:
            for aug in augmentation_names:
                if aug in synthetic_annotations_jsons and synthetic_annotations_jsons[aug] is not None:
                    # Handle both file paths (string) and pre-loaded data (list)
                    if isinstance(synthetic_annotations_jsons[aug], str):
                        # Load from JSON file
                        with open(synthetic_annotations_jsons[aug], 'r') as f:
                            synthetic_data_raw = json.load(f)
                        # Convert COCO format to expected format
                        synthetic_data = []
                        # This would need implementation based on your JSON structure
                        # For now, assuming the JSON already has the right format
                    else:
                        # Use pre-loaded data directly
                        synthetic_data = synthetic_annotations_jsons[aug]
                    
                    synthetic_annotations[aug] = []
                    synthetic_img_file_paths[aug] = []
                    
                    for item in synthetic_data:
                        synthetic_annotations[aug].append(item['annotations'])
                        synthetic_img_file_paths[aug].append(
                            os.path.join(synthetic_img_dir, item['name'])
                        )

        # For validation/test datasets, skip mixing and just use real data
        if not is_train:
            self.mixed_file_paths = real_img_file_paths
            self.mixed_annotations = real_annotations_list 
            self.mixed_image_ids = real_image_ids
            self.mixed_weather = real_weather_metadata  # Preserve weather for val/test
        else:
            # For training datasets, apply mixing
            self.mixed_file_paths, self.mixed_annotations, self.mixed_image_ids, self.mixed_weather = self._generate_mixed_file_paths(
                real_img_file_paths=real_img_file_paths, 
                real_annotations=real_annotations_list,
                real_image_ids=real_image_ids,
                real_weather_metadata=real_weather_metadata,
                synthetic_img_file_paths=synthetic_img_file_paths, 
                synthetic_annotations=synthetic_annotations, 
                mixing_method=mixing_method, 
                mix_rate=mix_rate
            )

        # Store target image size for resizing in __getitem__
        image_size = training_config["image_size"]
        # Get normalization stats from config (use ImageNet defaults if not specified)
        normalize_mean = training_config.get("normalize_mean", [0.485, 0.456, 0.406])
        normalize_std = training_config.get("normalize_std", [0.229, 0.224, 0.225])
        
        # Store normalization parameters (but don't apply - FasterRCNN will handle it)
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        
        self.use_augmentation = training_config.get("use_augmentation", False) and is_train
        self.augmentation_params = training_config.get("augmentation_params", {})
        
        if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
            self.image_size = (image_size[0], image_size[1])
            target_h, target_w = self.image_size
            
            # Build transform pipeline WITHOUT normalization (FasterRCNN handles it)
            transforms_list = [
                T2.ToImage(),
                T2.ToDtype(torch.float32, scale=True),  # Convert to float [0, 1]
                T2.Resize((target_h, target_w), antialias=True)
            ]
            
            self.transforms = T2.Compose(transforms_list)
        else:
            self.image_size = None
            # Initialize transform without resize or normalization
            transforms_list = [
                T2.ToImage(),
                T2.ToDtype(torch.float32, scale=True)
            ]
            
            self.transforms = T2.Compose(transforms_list)
        
        # Optional: cache all images in memory if specified
        self.cache_images = training_config.get("cache_images_in_memory", False)
        self.image_cache = {}
        if self.cache_images:
            print(f"[INFO] Caching {len(self.mixed_file_paths)} images in memory...")
            for img_path in tqdm(self.mixed_file_paths, desc="Loading images into memory"):
                self.image_cache[img_path] = Image.open(img_path).convert("RGB")

    def _validate_inputs(self, real_annotations_json : str, 
            real_img_dir : str,
            augmentation_names : list, 
            synthetic_annotations_jsons : dict, 
            synthetic_img_dir : str, 
            aug_percentage_cutoffs : dict,
            mixing_method : MixingMethod,
            mix_rate : float,
            is_train : bool,
            dataset_size : int = None
        ):
        
        assert os.path.exists(real_annotations_json), f"Real annotations JSON {real_annotations_json} does not exist"
        assert os.path.exists(real_img_dir), f"Real image directory {real_img_dir} does not exist"

        # Load real categories
        with open(real_annotations_json, "r") as f:
            real_data = json.load(f)
        real_categories = sorted(real_data["categories"], key=lambda x: x["id"])

        if not is_train:
            # For validation/test datasets, we don't need synthetic data parameters
            return

        if augmentation_names is None or len(augmentation_names) == 0:
            assert mix_rate == 0, f"If no augmentation_names provided, mix_rate must be 0, got {mix_rate}"
            return

        assert mixing_method is not None, "mixing_method must be specified"
        assert set(augmentation_names) == set(synthetic_annotations_jsons.keys()), "Augmentation names must match keys in synthetic_annotations_jsons"
        assert set(augmentation_names) == set(aug_percentage_cutoffs.keys()), "Augmentation names must match keys in aug_percentage_cutoffs"
        assert 0 <= mix_rate <= 1, f"mix_rate must be between 0 and 1"
        assert os.path.exists(synthetic_img_dir), f"Synthetic image directory {synthetic_img_dir} does not exist"

        for aug in augmentation_names:
            syn_ann = synthetic_annotations_jsons[aug]
            if isinstance(syn_ann, str):
                assert os.path.exists(syn_ann), f"Synthetic annotations JSON {syn_ann} does not exist"
                # Load categories from synthetic JSON and compare
                with open(syn_ann, "r") as f:
                    syn_data = json.load(f)
                syn_categories = sorted(syn_data["categories"], key=lambda x: x["id"])
                assert syn_categories == real_categories, (
                    f"Category mismatch in augmentation '{aug}'.\n"
                    f"Real categories: {real_categories}\n"
                    f"Synthetic categories: {syn_categories}"
                )
            elif isinstance(syn_ann, list):
                # If pre-loaded, assume category check must be handled upstream
                pass

        for name in augmentation_names:
            assert 0 <= aug_percentage_cutoffs[name] <= 1, f"{name} aug_percentage_cutoff must be between 0 and 1"

    def _build_augmentation_transforms(self):
        """Build Albumentations augmentation transforms from config parameters.
        
        Uses Albumentations to match YOLO's augmentation pipeline exactly:
        - brightness, contrast (RandomBrightnessContrast)
        - gamma (RandomGamma)
        - hue, saturation (HueSaturationValue)
        - gaussian_blur_kernel, gaussian_blur_sigma (GaussianBlur)
        - motion_blur_kernel (MotionBlur)
        - gaussian_noise_std (GaussNoise)
        """
        params = self.augmentation_params
        
        # Extract parameters (fail loudly if missing)
        brightness = params["brightness"]
        contrast = params["contrast"]
        gamma = params["gamma"]
        hue = params["hue"]
        saturation = params["saturation"]
        gaussian_blur_kernel = params["gaussian_blur_kernel"]
        gaussian_blur_sigma = params["gaussian_blur_sigma"]
        motion_blur_kernel = params["motion_blur_kernel"]
        gaussian_noise_std = params["gaussian_noise_std"]
        
        print(f"[INFO] Building Albumentations pipeline:")
        print(f"  brightness={brightness}, contrast={contrast}")
        print(f"  gamma={gamma}")
        print(f"  hue={hue}, saturation={saturation}")
        print(f"  gaussian_blur_kernel={gaussian_blur_kernel}, gaussian_blur_sigma={gaussian_blur_sigma}")
        print(f"  motion_blur_kernel={motion_blur_kernel}")
        print(f"  gaussian_noise_std={gaussian_noise_std}")
        
        # Build Albumentations transforms (matching YOLO exactly)
        transforms = []
        
        # 1. RandomBrightnessContrast
        if brightness > 0 or contrast > 0:
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=brightness,
                contrast_limit=contrast,
                p=0.5,
            ))
        
        # 2. RandomGamma
        if gamma[0] != 1.0 or gamma[1] != 1.0:
            gamma_min = int(gamma[0] * 100)
            gamma_max = int(gamma[1] * 100)
            transforms.append(A.RandomGamma(
                gamma_limit=(gamma_min, gamma_max),
                p=0.5,
            ))
        
        # 3. HueSaturationValue
        if hue > 0 or saturation > 0:
            hue_shift = int(hue * 360)
            sat_shift = int(saturation * 100)
            transforms.append(A.HueSaturationValue(
                hue_shift_limit=hue_shift,
                sat_shift_limit=sat_shift,
                val_shift_limit=0,
                p=0.5,
            ))
        
        # 4. GaussianBlur
        if gaussian_blur_kernel[0] > 1:
            transforms.append(A.GaussianBlur(
                blur_limit=tuple(gaussian_blur_kernel),
                sigma_limit=tuple(gaussian_blur_sigma),
                p=0.3,
            ))
        
        # 5. MotionBlur
        if motion_blur_kernel > 1:
            transforms.append(A.MotionBlur(
                blur_limit=motion_blur_kernel,
                p=0.2,
            ))
        
        # 6. GaussNoise
        if gaussian_noise_std > 0:
            var_limit = (0, int((gaussian_noise_std * 255) ** 2))
            transforms.append(A.GaussNoise(
                var_limit=var_limit,
                p=0.3,
            ))
        
        # Create Albumentations compose (NO bbox params for detection)
        self.albumentations_transform = A.Compose(transforms)
        
        print(f"[INFO] Albumentations pipeline initialized with {len(transforms)} transforms")
        print(f"[INFO] Transform types: {[type(t).__name__ for t in transforms]}")
        
        # Return empty list since we'll apply Albumentations separately
        return []

    def _resize_all_targets_to_image_size(self):
        """Resize all bounding boxes in self.mixed_annotations to the configured image size."""
        target_h, target_w = self.image_size

        resized_annots = []
        for i, anns in tqdm(enumerate(self.mixed_annotations), desc="Resizing annotations"):
            img_path = self.mixed_file_paths[i]

            try:
                with Image.open(img_path) as img:
                    orig_w, orig_h = img.size
            except Exception as e:
                print(f"[WARN] Could not open {img_path}: {e}")
                resized_annots.append(anns)
                continue

            scale_x = target_w / orig_w
            scale_y = target_h / orig_h

            new_anns = []
            for annot in anns:
                # COCO bbox format: [x, y, width, height]
                if "bbox" in annot:
                    x, y, w, h = annot["bbox"]
                    x1, y1 = x * scale_x, y * scale_y
                    w_scaled, h_scaled = w * scale_x, h * scale_y
                    annot["bbox"] = [x1, y1, w_scaled, h_scaled]

                new_anns.append(annot)

            resized_annots.append(new_anns)

        self.mixed_annotations = resized_annots

        
    def _generate_mixed_file_paths(
            self, 
            real_img_file_paths, 
            real_annotations, 
            real_image_ids,
            real_weather_metadata,
            synthetic_img_file_paths, 
            synthetic_annotations, 
            mixing_method, 
            mix_rate
        ):
        
        if mixing_method == MixingMethod.REPLACEMENT:
            return self._generate_replacement_mixed_file_paths(
                    real_img_file_paths=real_img_file_paths,
                    real_annotations=real_annotations,
                    real_image_ids=real_image_ids,
                    real_weather_metadata=real_weather_metadata,
                    synthetic_img_file_paths=synthetic_img_file_paths,
                    synthetic_annotations=synthetic_annotations,
                    mix_rate=mix_rate
                )
        
        elif mixing_method == MixingMethod.ADDITION:
            return self._generate_addition_mixed_file_paths(
                real_img_file_paths=real_img_file_paths,
                real_annotations=real_annotations,
                real_image_ids=real_image_ids,
                real_weather_metadata=real_weather_metadata,
                synthetic_img_file_paths=synthetic_img_file_paths,
                synthetic_annotations=synthetic_annotations,
                mix_rate=mix_rate
            )


    """
    _generate_replacement_mixed_file_paths builds a mixed dataset by:
    1. Starting with all real image file paths.
    2. Randomly selecting a fixed subset (up to mix_rate fraction) of real images to remove.
    3. Evenly splitting the replacement budget across augmentation types (e.g., snow, rain, fog).
    4. Appending the top-ranked synthetic images for each augmentation to replace the removed real ones.

    The result is a dataset containing (1 - mix_rate) fraction of the original real images
    plus mix_rate fraction synthetic images, with the replacement pattern deterministic
    when using the same random seed.
    """
    def _generate_replacement_mixed_file_paths(self, real_img_file_paths, real_annotations, 
                                                real_image_ids, real_weather_metadata, synthetic_img_file_paths, 
                                                synthetic_annotations, mix_rate, seed=42):
        rng = np.random.default_rng(seed)
        num_real = len(real_img_file_paths)

        # total number of real images to replace
        num_replace_total = int(mix_rate * num_real)

        # randomly choose which real indices to remove
        replace_indices = rng.choice(num_real, size=num_replace_total, replace=False)

        # Drop the chosen real images and annotations
        mixed_img_file_paths = []
        mixed_annotations = []
        mixed_image_ids = []
        mixed_weather = []
        for i, (fp, annot, img_id, weather) in tqdm(enumerate(zip(real_img_file_paths, real_annotations, real_image_ids, real_weather_metadata)), desc="Replacing real images"):
            if i not in replace_indices:
                mixed_img_file_paths.append(fp)
                mixed_annotations.append(annot)
                mixed_image_ids.append(img_id)
                mixed_weather.append(weather)

        # Use all provided synthetic images (already sampled in create_experiment_datasets)
        aug_names = list(synthetic_img_file_paths.keys())
        
        if len(aug_names) > 0:
            synthetic_id_counter = max(mixed_image_ids) + 1 if mixed_image_ids else 0
            
            for aug in aug_names:
                # Use all synthetic images provided for this augmentation type
                count = len(synthetic_img_file_paths[aug])
                if count > 0:
                    for idx in range(count):
                        mixed_img_file_paths.append(synthetic_img_file_paths[aug][idx])
                        mixed_annotations.append(synthetic_annotations[aug][idx])
                        mixed_image_ids.append(synthetic_id_counter)
                        mixed_weather.append(aug)
                        synthetic_id_counter += 1

        return mixed_img_file_paths, mixed_annotations, mixed_image_ids, mixed_weather


    """
    _generate_addition_mixed_file_paths builds a mixed dataset by:
    1. Starting with all real image file paths.
    2. Finding the number of additional synthetic images to add to ensure the mixed dataset has mix_rate proportion of synthetic images.
    3. Evenly splitting the replacement budget across augmentation types (e.g., snow, rain, fog).
    4. Appending the top-ranked synthetic images to real images.

    The result is a dataset containing all original real images plus the additional synthetic images.
    """
    def _generate_addition_mixed_file_paths(self, real_img_file_paths, real_annotations, 
                                                real_image_ids, real_weather_metadata, synthetic_img_file_paths, 
                                                synthetic_annotations, mix_rate):
        num_real = len(real_img_file_paths)

        # total dataset size = real + synthetic
        # we want synthetic / total = mix_rate
        # so, synthetic = (mix_rate / (1 - mix_rate)) * num_real
        num_synth = int((mix_rate / (1 - mix_rate)) * num_real)

        mixed_img_file_paths = real_img_file_paths.copy()
        mixed_annotations = real_annotations.copy()
        mixed_image_ids = real_image_ids.copy()
        mixed_weather = real_weather_metadata.copy()
    
        # split synthetic quota equally among augmentations
        aug_names = list(synthetic_img_file_paths.keys())
        num_augs = len(aug_names)
        
        if num_augs > 0:
            num_per_aug = num_synth // num_augs
            remainder = num_synth % num_augs

            synthetic_id_counter = max(mixed_image_ids) + 1 if mixed_image_ids else 0

            for i, aug in enumerate(aug_names):
                count = num_per_aug + (1 if i < remainder else 0)
                if count > 0:
                    mixed_img_file_paths.extend(synthetic_img_file_paths[aug][:count])
                    mixed_annotations.extend(synthetic_annotations[aug][:count])
                    # Generate unique IDs for synthetic images
                    mixed_image_ids.extend(range(synthetic_id_counter, synthetic_id_counter + count))
                    synthetic_id_counter += count
                    # Tag synthetic images with their augmentation type (fog, rain, snow)
                    mixed_weather.extend([aug] * count)

        return mixed_img_file_paths, mixed_annotations, mixed_image_ids, mixed_weather


    def __len__(self):
        return len(self.mixed_file_paths)
    

    def __getitem__(self, idx):
        # Load image (from cache if available)
        img_path = self.mixed_file_paths[idx]
        if self.cache_images:
            img = self.image_cache[img_path].copy()
        else:
            img = Image.open(img_path).convert("RGB")

        # Get original image size for bounding box conversion
        orig_w, orig_h = img.size

        # Load annotations
        image_id = self.mixed_image_ids[idx]
        annots = self.mixed_annotations[idx]

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for obj in annots:
            # COCO bbox format: [x, y, width, height] -> convert to XYXY format [x_min, y_min, x_max, y_max]
            x, y, w, h = obj["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(obj["category_id"])
            areas.append(obj.get("area", w * h))
            iscrowd.append(obj.get("iscrowd", 0))

        # Apply Albumentations augmentation BEFORE torchvision transforms
        if self.use_augmentation and hasattr(self, 'albumentations_transform'):
            # Convert PIL to numpy for Albumentations
            img_np = np.array(img)
            augmented = self.albumentations_transform(image=img_np)
            img = Image.fromarray(augmented['image'])

        # Prepare target dictionary with proper tv_tensors for transform
        if len(boxes) == 0:
            target = {
                "boxes": tv_tensors.BoundingBoxes(
                    torch.zeros((0, 4), dtype=torch.float32),
                    format="XYXY",
                    canvas_size=(orig_h, orig_w)
                ),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
            }
        else:
            target = {
                "boxes": tv_tensors.BoundingBoxes(
                    torch.as_tensor(boxes, dtype=torch.float32),
                    format="XYXY",
                    canvas_size=(orig_h, orig_w)
                ),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "area": torch.as_tensor(areas, dtype=torch.float32),
                "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
            }

        # Apply transform to both image and target (boxes will be resized automatically)
        img, target = self.transforms(img, target)
        
        # Add metadata fields after transform (these don't need to be transformed)
        target["image_id"] = image_id
        target["filename"] = img_path
        raw_weather = self.mixed_weather[idx] if hasattr(self, 'mixed_weather') else None
        # Collapse dataset-specific labels (e.g. DAWN haze/mist -> foggy) to the
        # canonical set so all models log identical per-weather metrics.
        target["weather"] = canonical_weather_or_other(raw_weather)

        return img, target


    def compute_normalization_stats(self, cache_path=None):
        """
        Compute exact global mean and standard deviation across all pixels
        in the dataset. Stores the result as self.normalization_transform = T.Normalize(mean, std).

        This is a full serial pass over every training image, so it costs many
        minutes. The mixture is seeded and deterministic, which makes the result
        a pure function of the run configuration -- pass ``cache_path`` to read
        it back on subsequent calls instead of recomputing. The cache records the
        image count it was built from and is ignored if that no longer matches.
        """
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                if cached.get("num_images") == len(self.mixed_file_paths):
                    means = torch.tensor(cached["mean"], dtype=torch.float32)
                    stds = torch.tensor(cached["std"], dtype=torch.float32)
                    print(f"[INFO] Loaded normalization stats from {cache_path}")
                    print(f"Dataset mean: {means.tolist()}")
                    print(f"Dataset std:  {stds.tolist()}")
                    return means, stds
                print(f"[WARNING] Ignoring {cache_path}: built from "
                      f"{cached.get('num_images')} images, dataset now has "
                      f"{len(self.mixed_file_paths)}")
            except (ValueError, KeyError, OSError) as e:
                print(f"[WARNING] Could not read normalization cache {cache_path}: {e}")

        # For debugging just print the number of images
        print(f"Computing normalization stats for {len(self.mixed_file_paths)} images...")
        total_sum = torch.zeros(3)
        total_sq_sum = torch.zeros(3)
        total_pixels = 0

        for img_path in tqdm(self.mixed_file_paths, desc="Computing dataset stats"):
            img = Image.open(img_path).convert("RGB")
            # Use v2 transforms for consistency
            img_tensor = T2.ToImage()(img)
            img_tensor = T2.ToDtype(torch.float32, scale=True)(img_tensor)  # [3, H, W], values in [0,1]
            c, h, w = img_tensor.shape
            num_pixels = h * w

            total_sum += img_tensor.reshape(c, -1).sum(dim=1)
            total_sq_sum += (img_tensor.reshape(c, -1) ** 2).sum(dim=1)
            total_pixels += num_pixels

        means = total_sum / total_pixels
        stds = torch.sqrt((total_sq_sum / total_pixels) - (means ** 2))

        print(f"Dataset mean: {means.tolist()}")
        print(f"Dataset std:  {stds.tolist()}")

        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            tmp = cache_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"mean": means.tolist(), "std": stds.tolist(),
                           "num_images": len(self.mixed_file_paths)}, f, indent=2)
            os.replace(tmp, cache_path)  # atomic: concurrent writers can't tear it
            print(f"[INFO] Cached normalization stats to {cache_path}")

        return means, stds
    
    def update_normalization(self, mean, std):
        """Update normalization values in the transforms."""
        if isinstance(mean, torch.Tensor):
            mean = mean.tolist()
        if isinstance(std, torch.Tensor):
            std = std.tolist()
        
        self.normalize_mean = mean
        self.normalize_std = std
        
        print(f"[INFO] Stored normalization stats (mean={mean}, std={std}) - FasterRCNN will apply them")
        
        # Rebuild transforms with new normalization
        # if self.image_size is not None:
        #     target_h, target_w = self.image_size
        #     self.transforms = T2.Compose([
        #         T2.ToImage(),
        #         T2.ToDtype(torch.float32, scale=True),
        #         T2.Resize((target_h, target_w), antialias=True),
        #         T2.Normalize(mean=mean, std=std)
        #     ])
        # else:
        #     self.transforms = T2.Compose([
        #         T2.ToImage(),
        #         T2.ToDtype(torch.float32, scale=True),
        #         T2.Normalize(mean=mean, std=std)
        #     ])
        
        print(f"Updated normalization to mean={mean}, std={std}")
