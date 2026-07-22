"""
Shared evaluation utilities for COCO-based per-weather metrics.

This module contains evaluation logic that's shared between:
- trainer.py (ResNet/ConvNeXt - PyTorch native)
- train_yolo.py (YOLO - uses YOLO's API)

Both models produce predictions in different ways, but once we have
COCO-format predictions, the evaluation logic is identical.
"""

import os
import json
import sys
import io
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from weather_utils import OTHER_WEATHER, canonical_weather_or_other


def create_weather_mapping_from_annotations(annotations_path):
    """
    Create weather mapping from COCO annotations file.
    
    Args:
        annotations_path: Path to COCO JSON annotations
        
    Returns:
        Dictionary with weather mappings:
        {
            "image_id_to_weather": {image_id: weather_str},
            "filename_to_weather": {filename: weather_str},
            "weather_counts": {weather_str: count}
        }
    """
    if not os.path.exists(annotations_path):
        print(f"[WARNING] Annotations not found at {annotations_path}")
        return {"image_id_to_weather": {}, "filename_to_weather": {}, "weather_counts": {}}
    
    print(f"[INFO] Reading weather metadata from: {annotations_path}")
    
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    image_id_to_weather = {}
    filename_to_weather = {}
    weather_counts = {}
    
    for img in data['images']:
        img_id = img['id']
        filename = img['file_name']
        # Collapse dataset-specific labels (e.g. DAWN haze/mist -> foggy) to the
        # canonical set so all models log identical per-weather metrics.
        weather = canonical_weather_or_other(img.get('weather'))

        image_id_to_weather[img_id] = weather
        filename_to_weather[filename] = weather
        weather_counts[weather] = weather_counts.get(weather, 0) + 1
    
    return {
        "image_id_to_weather": image_id_to_weather,
        "filename_to_weather": filename_to_weather,
        "weather_counts": weather_counts
    }


def compute_per_weather_coco_metrics(coco_gt_all, all_predictions, weather_mapping, split_name="val"):
    """
    Compute COCO metrics broken down by weather condition.
    
    Args:
        coco_gt_all: pycocotools COCO object with all ground truth
        all_predictions: List of COCO-format predictions (all images)
        weather_mapping: Dictionary from create_weather_mapping_from_annotations()
        split_name: Name prefix for metrics (e.g., "val", "train", "dawn")
        
    Returns:
        Dictionary of per-weather metrics with keys like:
        "{split_name}/{weather}/mAP", "{split_name}/{weather}/mAP_50", etc.
    """
    # Get image ID to filename mapping
    img_id_to_filename = {img['id']: img['file_name'] for img in coco_gt_all.dataset['images']}
    
    # Create per-weather ground truth and prediction sets
    print(f"\n=== Per-Weather COCO Evaluation for {split_name} ===")
    weather_gt_dicts = {}
    weather_predictions = {}
    
    image_id_to_weather = weather_mapping.get("image_id_to_weather", {})
    filename_to_weather = weather_mapping.get("filename_to_weather", {})
    
    # Initialize per-weather dicts
    for weather in weather_mapping.get('weather_counts', {}).keys():
        weather_gt_dicts[weather] = {
            "images": [],
            "annotations": [],
            "categories": coco_gt_all.dataset['categories'],
            "info": {},
            "licenses": []
        }
        weather_predictions[weather] = []
    
    # Populate weather-specific GT dicts
    for img in coco_gt_all.dataset['images']:
        img_id = img['id']
        weather = image_id_to_weather.get(img_id, filename_to_weather.get(img['file_name'], "unknown"))
        if weather in weather_gt_dicts:
            weather_gt_dicts[weather]["images"].append(img)
    
    for ann in coco_gt_all.dataset['annotations']:
        img_id = ann['image_id']
        img_filename = img_id_to_filename.get(img_id, "")
        weather = image_id_to_weather.get(img_id, filename_to_weather.get(img_filename, "unknown"))
        if weather in weather_gt_dicts:
            weather_gt_dicts[weather]["annotations"].append(ann)
    
    # Populate weather-specific predictions
    for pred in all_predictions:
        img_id = pred['image_id']
        img_filename = img_id_to_filename.get(img_id, "")
        weather = image_id_to_weather.get(img_id, filename_to_weather.get(img_filename, "unknown"))
        if weather in weather_predictions:
            weather_predictions[weather].append(pred)
    
    # Compute per-weather COCO metrics
    per_weather_metrics = {}
    
    for weather in sorted(weather_gt_dicts.keys()):
        # Skip non-canonical conditions (already folded into overall metrics).
        if weather == OTHER_WEATHER:
            continue

        w_gt_dict = weather_gt_dicts[weather]
        w_preds = weather_predictions[weather]

        if len(w_gt_dict['images']) == 0:
            continue
        
        if not w_preds:
            print(f"  {weather} ({len(w_gt_dict['images'])} images): No predictions, skipping")
            continue
        
        # Build COCO objects for this weather
        w_coco_gt = COCO()
        w_coco_gt.dataset = w_gt_dict
        w_coco_gt.createIndex()
        
        print(f"  {weather}: {len(w_preds)} predictions for {len(w_gt_dict['images'])} images")
        
        try:
            w_coco_dt = w_coco_gt.loadRes(w_preds)
        except Exception as e:
            print(f"[ERROR] Failed to load predictions for {weather}: {e}")
            continue
        
        # Run COCO evaluation (suppress output)
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            w_coco_eval = COCOeval(w_coco_gt, w_coco_dt, iouType='bbox')
            w_coco_eval.evaluate()
            w_coco_eval.accumulate()
            w_coco_eval.summarize()
        except Exception as e:
            sys.stdout = old_stdout
            print(f"[ERROR] COCO evaluation failed for {weather}: {e}")
            continue
        finally:
            sys.stdout = old_stdout
        
        # Extract metrics
        weather_metrics = {
            f"{split_name}/{weather}/mAP": float(w_coco_eval.stats[0]),
            f"{split_name}/{weather}/mAP_50": float(w_coco_eval.stats[1]),
            f"{split_name}/{weather}/mAP_75": float(w_coco_eval.stats[2]),
            f"{split_name}/{weather}/AR": float(w_coco_eval.stats[8]),
        }
        
        per_weather_metrics.update(weather_metrics)
        
        print(f"  {weather}: mAP={weather_metrics[f'{split_name}/{weather}/mAP']:.4f}, "
              f"mAP@50={weather_metrics[f'{split_name}/{weather}/mAP_50']:.4f}")
    
    return per_weather_metrics


def compute_overall_coco_metrics(coco_gt, all_predictions, split_name="val"):
    """
    Compute overall COCO metrics for a dataset.
    
    Args:
        coco_gt: pycocotools COCO object with ground truth
        all_predictions: List of COCO-format predictions
        split_name: Name prefix for metrics (e.g., "val", "dawn")
        
    Returns:
        Dictionary with keys: "{split_name}/mAP", "{split_name}/mAP_50", etc.
    """
    if not all_predictions:
        print(f"[WARNING] No predictions for {split_name}, returning zero metrics")
        return {
            f"{split_name}/mAP": 0.0,
            f"{split_name}/mAP_50": 0.0,
            f"{split_name}/mAP_75": 0.0,
            f"{split_name}/AR": 0.0,
        }
    
    try:
        coco_dt = coco_gt.loadRes(all_predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        
        # Suppress output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        finally:
            sys.stdout = old_stdout
        
        overall_metrics = {
            f"{split_name}/mAP": float(coco_eval.stats[0]),
            f"{split_name}/mAP_50": float(coco_eval.stats[1]),
            f"{split_name}/mAP_75": float(coco_eval.stats[2]),
            f"{split_name}/AR": float(coco_eval.stats[8]),
        }
        
        print(f"\n{split_name} overall metrics:")
        print(f"  mAP: {overall_metrics[f'{split_name}/mAP']:.4f}")
        print(f"  mAP@50: {overall_metrics[f'{split_name}/mAP_50']:.4f}")
        
        return overall_metrics
        
    except Exception as e:
        print(f"[WARNING] Failed to compute overall metrics for {split_name}: {e}")
        return {}


def convert_coco_categories_to_0_indexed(coco_gt):
    """
    Convert COCO categories from 1-indexed to 0-indexed for YOLO compatibility.
    
    COCO format (for Faster R-CNN): 1=bike, 2=bus, ..., 10=truck (0 reserved for background)
    YOLO format: 0=bike, 1=bus, ..., 9=truck (no background class)
    
    Modifies the COCO object in-place.
    
    Args:
        coco_gt: pycocotools COCO object to modify
    """
    print("[INFO] Converting COCO categories from 1-indexed to 0-indexed for YOLO compatibility")
    
    for ann in coco_gt.dataset['annotations']:
        ann['category_id'] = ann['category_id'] - 1
    
    for cat in coco_gt.dataset['categories']:
        cat['id'] = cat['id'] - 1
    
    # Recreate index after modifying category IDs
    coco_gt.createIndex()
    
    # Ensure required fields exist
    if 'info' not in coco_gt.dataset:
        coco_gt.dataset['info'] = {}
    if 'licenses' not in coco_gt.dataset:
        coco_gt.dataset['licenses'] = []


def verify_label_indices(label_path, dataset_name, num_classes=10):
    """
    Check a few label files to ensure indices are within range [0, num_classes-1].
    Helpful to catch 0-indexed vs 1-indexed issues.
    """
    print(f"[INFO] Verifying label indices for {dataset_name} in {label_path}...")
    count = 0
    max_check = 100
    
    # 0 vs 1 indexing problem:
    # If we see '10' (and num_classes=10), it's likely 1-indexed (1-10) or background issue.
    # YOLO requires 0-9.
    
    warnings = []
    
    for root, dirs, files in os.walk(label_path):
        for f in files:
            if not f.endswith('.txt'): continue
            
            p = os.path.join(root, f)
            try:
                with open(p, 'r') as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if not parts: continue
                        cls_id = int(float(parts[0]))
                        
                        if cls_id < 0 or cls_id >= num_classes:
                            warnings.append(f"{f}: Found class {cls_id} (max expected {num_classes-1})")
                        
                        # Just check first label of first 100 files
                        break
            except Exception as e:
                pass
            
            count += 1
            if count >= max_check: break
        if count >= max_check: break
        
    if warnings:
        print(f"[WARNING] Potential label index issues in {dataset_name}:")
        for w in warnings[:5]:
            print(f"  - {w}")
        if len(warnings) > 5: print(f"  ... and {len(warnings)-5} more")
    else:
        print(f"[INFO] Label indices look correct for {dataset_name} (checked {count} files)")


def evaluate_additional_datasets(config_manager, model, verify_labels=True):
    """
    Evaluate on additional validation datasets (DAWN, ACDC, etc.).
    
    Args:
        config_manager: ConfigManager instance with configuration
        model: YOLO model (can be ultralytics.YOLO or similar wrapper)
        verify_labels: Whether to perform safety checks on label indices
    
    Returns:
        Dictionary with keys like "{dataset_name}/mAP", "{dataset_name}/mAP_50"
    """
    # Import locally to avoid circular imports if any
    from yolo_utils.yolo_dataset_builder import create_temp_validation_dataset
    
    val_datasets = config_manager.base_config["paths"].get("val_datasets", {})
    additional_metrics = {}
    
    data_dir = config_manager.base_config["paths"]["data_dir"]
    temp_dir = config_manager.model_config.get("yolo_temp_dataset_path", "temp_yolo_dataset")
    
    # Isolate validation datasets per run to prevent collisions during concurrent experiments
    if hasattr(config_manager, "run_name") and config_manager.run_name:
        temp_dir = os.path.join(temp_dir, config_manager.run_name, "validation_sets")

    for name, cfg in val_datasets.items():
        if not cfg.get("enabled", False):
            continue
            
        print(f"\n[EVALUATE] Additional Dataset: {name}")
        
        # 1. Resolve paths
        img_path = os.path.join(data_dir, cfg["images"])
        lbl_path = os.path.join(data_dir, cfg["labels"])
        
        if not os.path.exists(img_path):
            print(f"[WARNING] Image path not found for {name}: {img_path}")
            continue
            
        # 2. Safety Check: Verify Label Indices (0-indexed vs 1-indexed)
        if verify_labels:
            if os.path.exists(lbl_path):
                verify_label_indices(lbl_path, name, config_manager.model_config["model"]["num_classes"])
            
        # 3. Create JIT temporary dataset
        # We use a subdirectory of the main temp folder to avoid conflicts
        try:
            dataset_yaml_path = create_temp_validation_dataset(
                dataset_name=name,
                images_path=img_path,
                labels_path=lbl_path,
                temp_base_dir=temp_dir,
                config_manager=config_manager
            )
            
            # 4. Run Validation
            print(f"Running validation on {name}...")
            # Check for COCO annotations for Per-Weather metrics
            has_coco_annotations = "annotations" in cfg and os.path.exists(os.path.join(data_dir, cfg["annotations"]))
            
            results = model.val(data=dataset_yaml_path, save_json=has_coco_annotations, verbose=False)
            
            # Extract metrics
            map50_95 = results.box.map    # mAP 50-95
            map50 = results.box.map50     # mAP 50
            
            additional_metrics[f"{name}/mAP"] = map50_95
            additional_metrics[f"{name}/mAP_50"] = map50
            
            print(f"--> {name}: mAP={map50_95:.4f}, mAP_50={map50:.4f}")

            # 5. Per-Weather breakdown (if annotations available)
            if has_coco_annotations:
                print(f"Computing per-weather metrics for {name}...")
                
                # Load GT
                ann_path = os.path.join(data_dir, cfg["annotations"])
                coco_gt = COCO(ann_path)
                convert_coco_categories_to_0_indexed(coco_gt)
                
                # Load Predictions
                pred_json_path = results.save_dir / "predictions.json"
                if pred_json_path.exists():
                    with open(pred_json_path, 'r') as f:
                        preds = json.load(f)
                    
                    # Create mapping from filename to ID (needed to link preds to GT)
                    filename_to_id = {}
                    basename_to_id = {} # Fallback map for when YOLO strips directory paths
                    
                    for img in coco_gt.dataset['images']:
                        # Full path without extension (e.g. "Fog/foggy-001")
                        full_stem = os.path.splitext(img['file_name'])[0]
                        filename_to_id[full_stem] = img['id']
                        
                        # Basename without extension (e.g. "foggy-001")
                        base_stem = os.path.basename(full_stem)
                        basename_to_id[base_stem] = img['id']
                    
                    # Remap prediction image_ids
                    # YOLO outputs plain filenames or IDs often mismatched with original COCO IDs
                    remapped_preds = []
                    
                    if preds:
                        print(f"[DEBUG] Sample YOLO prediction IDs: {[p.get('image_id') for p in preds[:3]]}")
                        print(f"[DEBUG] Sample filename_to_id keys: {list(filename_to_id.keys())[:3]}")
                        print(f"[DEBUG] Sample basename_to_id keys: {list(basename_to_id.keys())[:3]}")

                    for p in preds:
                        p_new = dict(p)
                        yolo_id = p.get('image_id')
                        matched_id = None
                        
                        # Strategy 1: Exact match (full path stem)
                        if yolo_id in filename_to_id:
                            matched_id = filename_to_id[yolo_id]
                        
                        # Strategy 2: Basename match (e.g. "foggy-001" -> "Fog/foggy-001")
                        elif isinstance(yolo_id, str):
                            yolo_base = os.path.basename(yolo_id)
                            if yolo_base in basename_to_id:
                                matched_id = basename_to_id[yolo_base]
                        
                        if matched_id is not None:
                            p_new['image_id'] = matched_id
                            # CRITICAL FIX: Shift category_id from 1-indexed (YOLO export) to 0-indexed (our internal GT)
                            p_new['category_id'] -= 1
                            remapped_preds.append(p_new)
                            
                    print(f"[INFO] Rematched {len(remapped_preds)}/{len(preds)} predictions for {name}")
                            
                    # Get weather mapping for this specific dataset
                    weather_mapping = create_weather_mapping_from_annotations(ann_path)
                    
                    # Compute per-weather metrics
                    # Pass split_name=name to prefix metrics like "dawn/rain/mAP"
                    pw_metrics = compute_per_weather_coco_metrics(
                        coco_gt, remapped_preds, weather_mapping, split_name=name
                    )
                    additional_metrics.update(pw_metrics)
                else:
                    print(f"[WARNING] Predictions file not found at {pred_json_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to evaluate {name}: {e}")
            import traceback
            traceback.print_exc()

    return additional_metrics
