"""
YOLO Training Script with Synthetic Data Replacement

YOLO-specific workflow:
1. Prepare temporary dataset directory (YOLO format: images/ + labels/ + data.yaml)
2. Copy/mix real + synthetic data based on replacement parameters
3. Train using YOLO's built-in .train() API
4. Evaluate using shared YOLO evaluation utilities (yolo_evaluation_utils.py)

Uses shared evaluation logic with trainer.py via yolo_evaluation_utils module.
"""
import torch
import argparse
import os
import sys
import json
import io
from ultralytics import YOLO
from pycocotools.coco import COCO

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config_manager import ConfigManager, resolve_wandb_target
from yolo_utils.yolo_utils import set_seed, build_yolo_model
from yolo_utils.yolo_dataset_builder import prepare_temp_dataset
from yolo_utils.yolo_evaluation_utils import (
    create_weather_mapping_from_annotations,
    compute_per_weather_coco_metrics,
    convert_coco_categories_to_0_indexed,
    evaluate_additional_datasets
)
import wandb


class YOLOExperimentRunner:
    """Runner for YOLO experiments."""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager

    def run_experiment(self, wandb_key=None, keep_cached_data=False):
        print(f"Running YOLO experiment: {self.config_manager.run_name}")
        print(f"Mixing: {self.config_manager.mixing_method} ({self.config_manager.mix_rate*100:.1f}%), Aug: {self.config_manager.augmentation_type}")

        # Get evaluation frequency from config
        eval_freq = self.config_manager.model_config['evaluation']['frequency']
        print(f"Evaluation frequency: every {eval_freq} epoch(s)")

        # Construct temp dataset directory path
        temp_base = self.config_manager.model_config["yolo_temp_dataset_path"]
        temp_dataset_dir = os.path.join(temp_base, self.config_manager.run_name)
        
        data_yaml = prepare_temp_dataset(self.config_manager, temp_dataset_dir, keep_cached_data=keep_cached_data)

        # Build YOLO model deterministically (YOLO handles device placement internally)
        # Check config for pretrained weights setting (defaults to False if not specified)
        use_pretrained = self.config_manager.model_config["model"].get("pretrained", False)
        print(f"[INFO] Building model: {self.config_manager.model_config['model']['name']} (Pretrained: {use_pretrained})")
        
        model = build_yolo_model(
            self.config_manager.model_config["model"]["name"],
            num_classes=self.config_manager.model_config["model"]["num_classes"],
            use_pretrained_weights=use_pretrained
        )

        # --- W&B setup (new run each time) ---
        if wandb_key:
            os.environ["WANDB_API_KEY"] = wandb_key
            wandb.login(key=wandb_key)

            # Ensure a clean init in this process
            if wandb.run is not None:
                wandb.finish()

            # Create a new run every time (no persistent run ID)
            wandb_entity, wandb_project = resolve_wandb_target(self.config_manager.base_config.get("wandb", {}))
            wandb.init(
                entity=wandb_entity,
                project=wandb_project,
                name=self.config_manager.run_name,
                reinit=True,
            )

            # Define 'epoch' as the custom step metric (consistent with train.py evaluation)
            # This decouples the x-axis from the internal wandb _step counter
            wandb.define_metric("epoch")
            wandb.define_metric("*", step_metric="epoch")
            
            # Explicitly define per-weather metric patterns to ensure they map to epoch correctly
            wandb.define_metric("val/*", step_metric="epoch")
            wandb.define_metric("dawn/*", step_metric="epoch")
            wandb.define_metric("acdc_val/*", step_metric="epoch")
            wandb.define_metric("acdc_train/*", step_metric="epoch")
            wandb.define_metric("foggy_zurich/*", step_metric="epoch")

            
            # Create weather mapping once for use in callbacks
            print("\n[Pre-training] Creating weather type mapping from validation set...")
            weather_mapping = self._create_weather_mapping()
            print(f"Weather distribution: {dict(sorted(weather_mapping['weather_counts'].items()))}")

            # Enhanced callbacks with per-weather evaluation
            def _on_fit_epoch_end(trainer):
                raw_metrics = trainer.metrics or {}
                safe_metrics = {}
                for k, v in raw_metrics.items():
                    if torch.is_tensor(v):
                        # convert to float safely (does not mutate original tensor)
                        safe_metrics[k] = float(v.detach().cpu())
                    else:
                        safe_metrics[k] = v
                safe_metrics.setdefault("epoch", trainer.epoch)
                wandb.log(safe_metrics, step=trainer.epoch)
                
                # Note: Per-weather evaluation is deferred to post-training
                # to avoid interfering with the training loop state and pickling issues.

            def _on_train_end(trainer):
                print("\n[Training End] flushing final metrics to W&B...")
                # Do NOT finish wandb here, so we can log post-training metrics to the same run
                # wandb.finish()

            model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
            model.add_callback("on_train_end", _on_train_end)
        else:
            # No W&B - just create weather mapping for potential post-training eval
            weather_mapping = self._create_weather_mapping()

        # --- Training args (fresh run each time) ---
        train_cfg = self.config_manager.model_config["training"]
        train_args = {
            "data": data_yaml,
            "epochs": train_cfg["epochs"],
            "batch": train_cfg["batch_size"],
            "imgsz": train_cfg["image_size"],
            "lr0": train_cfg["learning_rate"],
            "optimizer": train_cfg["optimizer"],
            "momentum": train_cfg["momentum"],
            "weight_decay": train_cfg["weight_decay"],
            "warmup_epochs": train_cfg["warmup_epochs"],
            "warmup_bias_lr": train_cfg["warmup_bias_lr"],
            "workers": self.config_manager.model_config["hardware"]["num_workers"],
            "resume": False,      # Start fresh each time
            "save_period": 1,     # checkpoint every epoch
            "exist_ok": True,     # reuse run dir if it exists
            "project": self.config_manager.model_config["yolo_outputs_dir"],
            "name": self.config_manager.run_name,
        }

        results = model.train(**train_args)
        
        # --- Post-training evaluation of all checkpoints ---
        if wandb_key:
            print("\n[Post-Training] Starting evaluation of all checkpoints...")
            
            # Use minimal tagging update if needed, but primarily just continue the run
            print(f"[INFO] Continuing existing W&B run for per-weather metrics...")

            # Tag convention
            mix_tag = f"r{int(self.config_manager.mix_rate*100)}" if self.config_manager.mixing_method == "replacement" else f"a{int(self.config_manager.mix_rate*100)}"
            
            # Find all checkpoints. Use the directory Ultralytics actually saved
            # to (results.save_dir / model.trainer.save_dir) rather than
            # reconstructing it.
            save_dir = getattr(results, "save_dir", None) or getattr(
                getattr(model, "trainer", None), "save_dir", None)
            if save_dir is None:
                save_dir = os.path.join(
                    self.config_manager.model_config["yolo_outputs_dir"],
                    self.config_manager.run_name,
                )
            run_dir = str(save_dir)
            weights_dir = os.path.join(run_dir, "weights")
            print(f"[INFO] Looking for checkpoints in: {weights_dir}")
            
            if os.path.exists(weights_dir):
                checkpoints = sorted([f for f in os.listdir(weights_dir) if f.endswith(".pt") and "epoch" in f])
                
                # Sort by epoch n[umber
                def get_epoch(fname):
                    try:
                        return int(''.join(filter(str.isdigit, fname)))
                    except:
                        return 999999

                checkpoints.sort(key=get_epoch)
                
                print(f"Found checkpoints: {checkpoints}")
                
                for ckpt_file in checkpoints:
                    ckpt_path = os.path.join(weights_dir, ckpt_file)
                    ckpt_epoch_num = get_epoch(ckpt_file)  # 0-indexed from filename
                    
                    if ckpt_epoch_num == 999999: 
                        print(f"[SKIP] {ckpt_file} - invalid epoch number")
                        continue
                    
                    # YOLO saves epoch0.pt after completing epoch 1 (0-indexed internally as epoch 0)
                    # We need to use the same 0-indexed numbering to match training logs
                    # Only evaluate every eval_freq epochs
                    actual_epoch = ckpt_epoch_num + 1  # Human-readable (1-indexed)
                    
                    print(f"[CHECK] {ckpt_file}: ckpt_epoch_num={ckpt_epoch_num}, actual_epoch={actual_epoch}, eval_freq={eval_freq}, {actual_epoch}%{eval_freq}={actual_epoch % eval_freq}")
                    
                    if actual_epoch % eval_freq != 0:
                        print(f"[SKIP] {ckpt_file} - not a multiple of eval_freq")
                        continue
                        
                    print(f"\n[EVALUATE] Checkpoint: {ckpt_file} (Training Epoch {actual_epoch}, Step {ckpt_epoch_num})")
                    
                    try:
                        # Load model from checkpoint - fresh instance
                        eval_model = YOLO(ckpt_path)
                        
                        # Run per-weather evaluation
                        weather_metrics = self._evaluate_per_weather_metrics(eval_model, weather_mapping, data_yaml)
                        
                        # Run additional dataset evaluation (DAWN, ACDC, etc.)
                        additional_metrics = evaluate_additional_datasets(self.config_manager, eval_model)
                        
                        # Merge all metrics
                        all_metrics = {}
                        if weather_metrics:
                            all_metrics.update(weather_metrics)
                        if additional_metrics:
                            all_metrics.update(additional_metrics)
                        
                        if all_metrics:
                            # Add epoch to metrics for the custom step
                            all_metrics["epoch"] = ckpt_epoch_num

                            # Log using the 0-indexed step to match YOLO's internal logging during training
                            print(f"Logging {len(all_metrics)} metrics for training epoch {actual_epoch} (step={ckpt_epoch_num})")
                            print(f"[DEBUG] Sample metrics being logged: {list(all_metrics.items())[:3]}")
                            
                            # Log to the dedicated weather metrics run (using 'epoch' as x-axis)
                            wandb.log(all_metrics)
                            
                            print(f"Successfully logged {len(all_metrics)} metrics to W&B at epoch {ckpt_epoch_num}")
                            print(f"[INFO] View metrics at: {wandb.run.url}")
                            
                        # Clean up
                        del eval_model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        print(f"Failed to evaluate {ckpt_file}: {e}")
            else:
                print(f"[WARNING] Weights directory not found at {weights_dir}")
            
            # Summary of what was logged
            if wandb.run is not None:
                print(f"\n[SUMMARY] Per-weather metrics logged to: {wandb.run.url}")
                print(f"[SUMMARY] Metrics should appear as time series with multiple data points")
                print(f"[SUMMARY] Look for metrics like: val/clear/mAP, val/rainy/mAP_50, etc.")
        
        return results
    
    def _create_weather_mapping(self):
        """
        Read COCO validation annotations and create a mapping of image_id/filename to weather type.
        Uses shared utility from yolo_evaluation_utils.
        """
        # Get validation annotations path from config
        yolo_paths = self.config_manager.model_config.get("paths", {})
        data_dir = self.config_manager.base_config["paths"]["data_dir"]
        
        if "val" in yolo_paths and "annotations" in yolo_paths["val"]:
            val_annotations_path = os.path.join(data_dir, yolo_paths["val"]["annotations"])
        else:
            val_annotations_path = os.path.join(data_dir, "bdd100k_val/bdd100k_val_coco.json")
        
        return create_weather_mapping_from_annotations(val_annotations_path)
    
    def _evaluate_per_weather_metrics(self, model, weather_mapping, data_yaml):
        """
        Evaluate YOLO model on validation set with per-weather breakdown.
        Uses YOLO's built-in validation for speed, then recomputes per-weather metrics.
        """
        print("\n=== Running YOLO Validation ===")

        # Get validation annotations path
        yolo_paths = self.config_manager.model_config.get("paths", {})
        data_dir = self.config_manager.base_config["paths"]["data_dir"]

        if "val" in yolo_paths and "annotations" in yolo_paths["val"]:
            val_annotations_path = os.path.join(data_dir, yolo_paths["val"]["annotations"])
        else:
            val_annotations_path = os.path.join(data_dir, "bdd100k_val/bdd100k_val_coco.json")

        if not os.path.exists(val_annotations_path):
            print(f"[WARNING] Cannot compute metrics: {val_annotations_path} not found")
            return None

        # Load COCO ground truth
        print(f"Loading COCO annotations from: {val_annotations_path}")
        coco_gt_all = COCO(val_annotations_path)

        # Convert COCO categories from 1-indexed to 0-indexed for YOLO compatibility
        convert_coco_categories_to_0_indexed(coco_gt_all)

        # Run YOLO's built-in validation to get predictions quickly
        print("Running YOLO validation (this will generate predictions)...")

        # Isolate YOLO's val output. By default model.val() writes predictions.json
        # under runs/detect/valN, which here is a symlink into the old checkout and
        # whose N-autoincrement is NOT concurrency-safe: parallel array tasks pick
        # the same valN and clobber each other's predictions.json, producing a
        # truncated file that json.load then chokes on. Pin an explicit, per-process
        # directory in this repo instead.
        val_project = os.path.join(
            self.config_manager.model_config["yolo_temp_dataset_path"],
            self.config_manager.run_name, "val_out")
        val_name = f"bdd_{os.getpid()}"

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        val_results = None
        try:
            with torch.no_grad():
                # Run validation directly on the model passed in (which is now a fresh loaded model)
                val_results = model.val(data=data_yaml, save_json=True, save_hybrid=False,
                                        verbose=False, project=val_project, name=val_name,
                                        exist_ok=True)
        finally:
            sys.stdout = old_stdout
            
        # BREAKPOINT FOR EVALUATION HERE
        # import pdb; pdb.set_trace()

        if val_results is None:
            print("[WARNING] YOLO validation did not return results")
            return None

        # Load predictions from YOLO's saved JSON
        # YOLO saves to runs/{name}/predictions.json
        pred_json_path = getattr(val_results, "save_dir", None)
        if pred_json_path is None:
            print("[WARNING] val_results.save_dir missing")
            return None

        pred_json_path = pred_json_path / "predictions.json"
        if not pred_json_path.exists():
            print(f"[WARNING] Predictions JSON not found at {pred_json_path}")
            return None

        print(f"Loading predictions from: {pred_json_path}")
        with open(pred_json_path, 'r') as f:
            all_predictions = json.load(f)

        print(f"Loaded {len(all_predictions)} predictions from YOLO")

        # Create mapping from filename (without extension) to COCO image ID
        filename_to_coco_id = {}
        for img in coco_gt_all.dataset['images']:
            filename_stem = os.path.splitext(img['file_name'])[0]
            filename_to_coco_id[filename_stem] = img['id'] # this is a mapping in a dict

        print(f"Created filename-to-ID mapping for {len(filename_to_coco_id)} images")
        print(f"Sample mapping: {list(filename_to_coco_id.items())[:3]}")
        
        # After loading predictions, add this debug block:
        # Debug: Check what YOLO's image_id actually looks like
        if all_predictions:
            sample_pred_ids = list(set([p.get('image_id') for p in all_predictions[:100]]))[:5]
            print(f"Sample YOLO prediction IDs: {sample_pred_ids}")
            print(f"Sample filename_to_coco_id keys: {list(filename_to_coco_id.keys())[:5]}")


        # Remap YOLO prediction IDs to COCO IDs
        remapped_predictions = []
        unmatched_count = 0

        for pred in all_predictions:
            yolo_img_id = pred.get('image_id')
            if yolo_img_id in filename_to_coco_id:
                # create a shallow copy so we don't mutate the original list items in place
                p = dict(pred)
                p['image_id'] = filename_to_coco_id[yolo_img_id]
                remapped_predictions.append(p)
            else:
                unmatched_count += 1

        # CRITICAL FIX FOR PER-WEATHER METRICS TO WORK
        for pred in remapped_predictions: pred['category_id'] -= 1

        all_predictions = remapped_predictions
        
        print(f"Remapped {len(all_predictions)} predictions to COCO IDs ({unmatched_count} unmatched)")

        # Use shared utility for per-weather COCO evaluation
        per_weather_metrics = compute_per_weather_coco_metrics(
            coco_gt_all=coco_gt_all,
            all_predictions=remapped_predictions,
            weather_mapping=weather_mapping,
            split_name="val"
        )
        
        # Merge overall standard metrics from YOLO validation results
        if val_results and hasattr(val_results, 'results_dict'):
            print(f"[INFO] Merging {len(val_results.results_dict)} standard YOLO metrics")
            # Prefix with val/ to match typical convention if not already present, 
            # or just include as-is. YOLO usually gives 'metrics/mAP50-95(B)' etc.
            # We will clean them up to match user expectation of 'val/mAP' if possible, 
            # but getting the raw dict is safer.
            per_weather_metrics.update(val_results.results_dict)
            
            # Map specific keys to simpler names if desired (optional)
            # e.g. metrics/mAP50-95(B) -> val/mAP
            if 'metrics/mAP50-95(B)' in val_results.results_dict:
                per_weather_metrics['val/mAP'] = val_results.results_dict['metrics/mAP50-95(B)']
            if 'metrics/mAP50(B)' in val_results.results_dict:
                per_weather_metrics['val/mAP_50'] = val_results.results_dict['metrics/mAP50(B)']

        # Return only per-weather metrics (YOLO already logs overall metrics)
        print(f"\n[INFO] Returning {len(per_weather_metrics)} per-weather metrics for logging")
        if per_weather_metrics:
            print(f"[INFO] Sample per-weather metrics: {list(per_weather_metrics.items())[:3]}")

        return per_weather_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replacement_percentage", type=float, required=False, default=None,
                        help="Synthetic data replacement percentage (0.0 to 1.0)")
    parser.add_argument("--addition_percentage", type=float, required=False, default=None,
                        help="Synthetic data addition percentage (0.0 to <1.0)")
    parser.add_argument("--augmentation_type", type=str, required=True,
                        choices=["gemini", "automold"],
                        help="Synthetic augmentation type")
    parser.add_argument("--wandb_key", type=str, default=os.getenv("WANDB_API_KEY"),
                        help="Weights & Biases API key (defaults to WANDB_API_KEY env var; see .env.example)")
    parser.add_argument("--keep_cached_data", action="store_true",
                        help="Keep cached temporary dataset directory")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and evaluate a specific checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to weights for eval_only mode")
    parser.add_argument("--eval_epoch", type=int, default=None,
                        help="Epoch number to record alongside --eval_only metrics in W&B")
    args = parser.parse_args()
    
    # Validate mixing arguments
    if args.replacement_percentage is not None and args.addition_percentage is not None:
        parser.error("Cannot specify both --replacement_percentage and --addition_percentage")
    
    if args.replacement_percentage is None and args.addition_percentage is None:
        parser.error("Must specify either --replacement_percentage or --addition_percentage")

    # Determine mixing parameters
    if args.replacement_percentage is not None:
        if not 0.0 <= args.replacement_percentage <= 1.0:
            parser.error("replacement_percentage must be between 0.0 and 1.0")
        mixing_method = "replacement"
        rate = args.replacement_percentage
    else:
        if not 0.0 <= args.addition_percentage < 1.0:
            parser.error("addition_percentage must be between 0.0 and <1.0")
        mixing_method = "addition"
        rate = args.addition_percentage

    print("=== Running YOLO Experiment ===")
    set_seed(42)
    
    config_manager = ConfigManager("configs", model_type="yolo",
                                   mix_rate=rate,
                                   augmentation_type=args.augmentation_type,
                                   mixing_method=mixing_method)
    print("Loaded YOLO config.")
    runner = YOLOExperimentRunner(config_manager)
    print(f"\nRunning {args.augmentation_type} with {rate*100:.1f}% {mixing_method}")
    
    if args.eval_only:
        if args.checkpoint_path is None:
            parser.error("--checkpoint_path is required when using --eval_only")

        print(f"=== Running Eval Only Mode: {args.checkpoint_path} ===")
        # 1. Build (or reuse) the temp dataset. The BDD100K per-weather pass runs
        #    model.val(data=data_yaml), so the mixed dataset this run was trained
        #    on has to exist -- reconstructing the path alone is not enough.
        temp_base = config_manager.model_config["yolo_temp_dataset_path"]
        temp_dataset_dir = os.path.join(temp_base, config_manager.run_name)
        data_yaml = prepare_temp_dataset(config_manager, temp_dataset_dir,
                                         keep_cached_data=True)

        # 2. Load the model and mapping
        model = YOLO(args.checkpoint_path)
        weather_mapping = runner._create_weather_mapping()

        # 3. Run evaluation
        print("Running BDD100K Per-Weather Evaluation...")
        results = runner._evaluate_per_weather_metrics(model, weather_mapping, data_yaml) or {}

        print("\nRunning Additional Datasets Evaluation (DAWN, ACDC, etc)...")
        results.update(evaluate_additional_datasets(config_manager, model) or {})

        # 4. Log the metrics we just computed. Without this the whole evaluation
        #    is discarded on exit.
        if args.wandb_key and results:
            os.environ["WANDB_API_KEY"] = args.wandb_key
            wandb.login(key=args.wandb_key)
            if wandb.run is not None:
                wandb.finish()
            wandb_entity, wandb_project = resolve_wandb_target(
                config_manager.base_config.get("wandb", {}))
            wandb.init(entity=wandb_entity, project=wandb_project,
                       name=config_manager.run_name, reinit=True)
            wandb.define_metric("epoch")
            wandb.define_metric("*", step_metric="epoch")
            if args.eval_epoch is not None:
                results["epoch"] = args.eval_epoch
            wandb.log(results)
            print(f"[INFO] Logged {len(results)} metrics to {wandb.run.url}")
            wandb.finish()
        elif results:
            print("[INFO] No W&B key provided; metrics were computed but not logged.")

        for k in sorted(results):
            print(f"  {k}: {results[k]}")
    else:
        # Standard overnight training
        results = runner.run_experiment(args.wandb_key, keep_cached_data=args.keep_cached_data)
        
        # Ensure W&B run is properly closed and synced
        if args.wandb_key and wandb.run is not None:
            print("[INFO] Finishing W&B run...")
            wandb.finish()
        print(results)

    print("Experiment finished")

if __name__ == "__main__":
    main()