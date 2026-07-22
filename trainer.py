import os
import time
import json
from collections import defaultdict
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import numpy as np
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from dataset import MixedGenDataSet, MixingMethod
from torchvision.ops import box_iou
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from data_utils import create_experiment_datasets
from config_manager import resolve_wandb_target
from weather_utils import OTHER_WEATHER
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class BaseTrainer:
    """Generic Trainer for PyTorch object detection models (ResNet, ConvNeXt)"""

    def __init__(self, config_manager, wandb_key=None, resume_wandb_id=None):
        self.config_manager = config_manager
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if wandb_key:
            wandb.login(key=wandb_key)

            wandb_entity, wandb_project = resolve_wandb_target(config_manager.base_config.get("wandb", {}))
            # Append augmentation status to run name
            use_augmentation = getattr(config_manager, 'use_augmentation', False)
            aug_suffix = "_aug" if use_augmentation else "_noaug"
            run_name_with_aug = f"{config_manager.run_name}{aug_suffix}"

            # Resume existing run or create new one
            if resume_wandb_id:
                print(f"[INFO] Resuming wandb run: {resume_wandb_id}")
                wandb.init(
                    entity=wandb_entity,
                    project=wandb_project,
                    id=resume_wandb_id,
                    resume="allow",
                    config={**config_manager.model_config, "model_type": config_manager.model_type, "use_augmentation": use_augmentation}
                )
            else:
                wandb.init(
                    entity=wandb_entity,
                    project=wandb_project,
                    name=run_name_with_aug,
                    config={**config_manager.model_config, "model_type": config_manager.model_type, "use_augmentation": use_augmentation}
                )

        self.model = None
        self.optimizer = None
        self.scheduler = None

        self.train_dataset = None
        self.val_dataset = None
        self.additional_val_datasets = {}
        self.train_loader = None
        self.val_loader = None
        self.additional_val_loaders = {}

        self.current_epoch = 0
        self.best_map = 0.0
        self.best_map_50 = 0.0
        self.best_iou = 0.0
        self.best_weather_map = {}
        self.best_weather_map_50 = {}
        self.step_count = 0
        
        # Store normalization stats for visualization
        self.norm_mean = None
        self.norm_std = None

        self.checkpoint_dir = os.path.join(
            config_manager.base_config["paths"]["checkpoints_dir"], config_manager.run_name
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.runs_dir = os.path.join("runs", config_manager.run_name)
        os.makedirs(self.runs_dir, exist_ok=True)
        
        self.viz_dir = os.path.join(self.runs_dir, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Save wandb run ID if this is a new run (not resuming)
        if wandb_key and wandb.run is not None and not resume_wandb_id:
            wandb_run_id_file = os.path.join(self.runs_dir, "wandb_run_id.txt")
            with open(wandb_run_id_file, "w") as f:
                f.write(wandb.run.id)
            print(f"[INFO] Saved wandb run ID to: {wandb_run_id_file}")
            print(f"[INFO] Wandb run ID: {wandb.run.id}")

    def set_model(self, model):
        self.model = model.to(self.device)

    def build_optimizer(self):
        training_config = self.config_manager.model_config["training"]
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer_type = training_config["optimizer"]
        lr = training_config["learning_rate"]
        weight_decay = training_config["weight_decay"]

        if optimizer_type == "sgd":
            print("[INFO] Using SGD optimizer.")
            momentum = training_config["momentum"]
            self.optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_type == "adam":
            print("[INFO] Using Adam optimizer.")
            self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == "adamw":
            print("[INFO] Using AdamW optimizer.")
            self.optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

    def build_scheduler(self):
        training_config = self.config_manager.model_config["training"]
        lr_schedule = training_config["lr_schedule"]

        if lr_schedule == "reduce_lr_on_plateau":
            print("[INFO] Using ReduceLROnPlateau scheduler.")
            gamma = training_config["lr_gamma"]
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=gamma, patience=10)
        elif lr_schedule == "cosine":
            print("[INFO] Using CosineAnnealingLR scheduler.")
            epochs = training_config["epochs"]
            warmup_epochs = training_config.get("warmup_epochs", 0)
            
            if warmup_epochs > 0:
                print(f"[INFO] Using LinearLR warmup ({warmup_epochs} epochs) + CosineAnnealingLR ({epochs - warmup_epochs} epochs).")
                warmup_start_factor = training_config.get("warmup_start_factor", 0.1)
                
                warmup_scheduler = optim.lr_scheduler.LinearLR(
                    self.optimizer, 
                    start_factor=warmup_start_factor, 
                    total_iters=warmup_epochs
                )
                
                cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, 
                    T_max=epochs - warmup_epochs
                )
                
                self.scheduler = optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs]
                )
            else:
                print(f"[INFO] Using CosineAnnealingLR scheduler ({epochs} epochs).")
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        elif lr_schedule == "linear_warmup":
            print("[INFO] Using LinearLR warmup scheduler only.")
            warmup_epochs = training_config.get("warmup_epochs", 3)
            warmup_start_factor = training_config.get("warmup_start_factor", 0.1)
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer, 
                start_factor=warmup_start_factor, 
                total_iters=warmup_epochs
            )
        elif lr_schedule == "step":
            print("[INFO] Using MultiStepLR scheduler.")
            lr_steps = training_config.get("lr_steps", [30, 40])
            gamma = training_config.get("lr_gamma", 0.1)
            warmup_epochs = training_config.get("warmup_epochs", 0)
            
            if warmup_epochs > 0:
                print(f"[INFO] Using LinearLR warmup ({warmup_epochs} epochs) + MultiStepLR (steps at {lr_steps}).")
                warmup_start_factor = training_config.get("warmup_start_factor", 0.1)
                
                warmup_scheduler = optim.lr_scheduler.LinearLR(
                    self.optimizer, 
                    start_factor=warmup_start_factor, 
                    total_iters=warmup_epochs
                )
                
                adjusted_steps = [step - warmup_epochs for step in lr_steps if step > warmup_epochs]
                
                step_scheduler = optim.lr_scheduler.MultiStepLR(
                    self.optimizer, 
                    milestones=adjusted_steps,
                    gamma=gamma
                )
                
                self.scheduler = optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, step_scheduler],
                    milestones=[warmup_epochs]
                )
            else:
                print(f"[INFO] Using MultiStepLR scheduler (steps at {lr_steps}, gamma={gamma}).")
                self.scheduler = optim.lr_scheduler.MultiStepLR(
                    self.optimizer, 
                    milestones=lr_steps,
                    gamma=gamma
                )
        else:
            print("[INFO] No learning rate scheduler will be used.")
            self.scheduler = None

    def build_datasets(self, prepared_synthetic_data):
        
        paths_config = self.config_manager.base_config["paths"]
        training_config = self.config_manager.model_config["training"].copy()
        data_dir = paths_config["data_dir"]
        
        # Add augmentation parameters to training config
        use_augmentation = getattr(self.config_manager, 'use_augmentation', False)
        training_config['use_augmentation'] = use_augmentation
        training_config['augmentation_params'] = self.config_manager.base_config.get('augmentation', {})

        real_annotations, synthetic_annotations, synthetic_img_dir = create_experiment_datasets(
            prepared_synthetic_data,
            self.config_manager.mix_rate,
            self.config_manager.augmentation_type,
            self.config_manager
        )

        augmentation_names = list(synthetic_annotations.keys()) if synthetic_annotations else None
        dataset_cutoff = self.config_manager.base_config["experiment"]["dataset_cutoff"]
        dataset_size = self.config_manager.base_config["experiment"]["dataset_size"]

        train_images = os.path.join(data_dir, paths_config["train"]["images"])
        val_images = os.path.join(data_dir, paths_config["val"]["images"])
        
        mixing_method_str = getattr(self.config_manager, "mixing_method", "replacement")
        mixing_method = MixingMethod.ADDITION if mixing_method_str == "addition" else MixingMethod.REPLACEMENT

        self.train_dataset = MixedGenDataSet(
            real_annotations_json=real_annotations,
            real_img_dir=train_images,
            augmentation_names=augmentation_names,
            synthetic_annotations_jsons=synthetic_annotations,
            synthetic_img_dir=synthetic_img_dir,
            aug_percentage_cutoffs={aug: dataset_cutoff for aug in (augmentation_names or [])},
            mixing_method=mixing_method,
            mix_rate=self.config_manager.mix_rate,
            is_train=True,
            dataset_size=dataset_size,
            training_config=training_config
        )

        # Check dataset size
        if mixing_method == MixingMethod.ADDITION:
             # For addition, size should be greater than base dataset size if mix_rate > 0
             assert len(self.train_dataset) >= self.config_manager.base_config["experiment"]["dataset_size"]
        else:
             # For replacement, size should be exactly maintained
             assert len(self.train_dataset) == self.config_manager.base_config["experiment"]["dataset_size"], f"Expected dataset size {self.config_manager.base_config['experiment']['dataset_size']}, got {len(self.train_dataset)}"
        
        # Compute stats from the dataset
        train_mean, train_std = self.train_dataset.compute_normalization_stats()
        
        # Store for use in model initialization AND visualization
        self.norm_mean = train_mean
        self.norm_std = train_std
        
        print(f"[INFO] Computed normalization stats - will pass to FasterRCNN model")
        print(f"       Mean: {train_mean.tolist()}")
        print(f"       Std:  {train_std.tolist()}")

        # Update datasets to store the stats (though they won't apply normalization)
        self.train_dataset.update_normalization(train_mean, train_std)

        # Primary validation dataset (BDD100K val)
        self.val_dataset = MixedGenDataSet(
            real_annotations_json=os.path.join(data_dir, paths_config["val"]["annotations"]),
            real_img_dir=val_images,
            augmentation_names=None,
            synthetic_annotations_jsons=None,
            synthetic_img_dir=None,
            aug_percentage_cutoffs=None,
            mixing_method=None,
            mix_rate=None,
            is_train=False,
            training_config=training_config
        )
        
        # Apply same normalization to validation dataset
        self.val_dataset.update_normalization(train_mean, train_std)
        
        # Load additional validation datasets (DAWN, STF, ACDC, etc.)
        self.additional_val_datasets = {}
        val_datasets_config = paths_config.get("val_datasets", {})
        for dataset_name, dataset_config in val_datasets_config.items():
            if dataset_config.get("enabled", False):
                print(f"[INFO] Loading additional validation dataset: {dataset_name}")
                print(f"       {dataset_config.get('description', '')}")
                try:
                    val_ds = MixedGenDataSet(
                        real_annotations_json=os.path.join(data_dir, dataset_config["annotations"]),
                        real_img_dir=os.path.join(data_dir, dataset_config["images"]),
                        augmentation_names=None,
                        synthetic_annotations_jsons=None,
                        synthetic_img_dir=None,
                        aug_percentage_cutoffs=None,
                        mixing_method=None,
                        mix_rate=None,
                        is_train=False,
                        training_config=training_config
                    )
                    val_ds.update_normalization(train_mean, train_std)
                    self.additional_val_datasets[dataset_name] = val_ds
                    print(f"       Loaded {len(val_ds)} images")
                except Exception as e:
                    print(f"[WARNING] Failed to load {dataset_name}: {e}")

    def build_dataloaders(self):
        training_config = self.config_manager.model_config["training"]
        hardware_config = self.config_manager.model_config["hardware"]

        batch_size = training_config["batch_size"]
        num_workers = hardware_config["num_workers"]
        pin_memory = hardware_config["pin_memory"]

        def collate_fn(batch):
            return tuple(zip(*batch))

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn
        )
        
        self.additional_val_loaders = {}
        for dataset_name, dataset in self.additional_val_datasets.items():
            self.additional_val_loaders[dataset_name] = DataLoader(
                dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn
            )

    def train_epoch(self):
        self.model.train()
        epoch_losses = defaultdict(list)
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        scaler = GradScaler()

        for images, targets in pbar:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            self.optimizer.zero_grad()

            losses = 0
            
            with autocast(device_type="cuda" if self.device.type == "cuda" else "cpu"):
                loss_dict = self.model(images, targets)
                
                loss_weights = self.config_manager.model_config["training"].get("loss_weights", {})
                for k, v in loss_dict.items():
                    losses += v * loss_weights[k]

            scaler.scale(losses).backward()
            
            scaler.unscale_(self.optimizer)
            
            clip_value = self.config_manager.model_config["training"].get("gradient_clipping", None)
            if clip_value:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
            
            scaler.step(self.optimizer)
            scaler.update()

            for k, v in loss_dict.items():
                epoch_losses[k].append(v.item())
            epoch_losses["total_loss"].append(losses.item())
            pbar.set_postfix({"loss": f"{losses.item():.4f}"})
            self.step_count += 1

        return {k: np.mean(v) for k, v in epoch_losses.items()}
    
    @torch.no_grad()
    def evaluate(self, dataloader, split_name="val"):
        """
        Evaluate the model on a dataloader using COCO evaluation metrics.
        """
        self.model.eval()
        all_predictions = []
        per_detection_ious = []
        per_image_ious = []
        
        weather_predictions = {}
        weather_gt_dicts = {}

        if "categories" in self.config_manager.base_config.get("experiment", {}):
            categories = self.config_manager.base_config["experiment"]["categories"]
        else:
            categories = [{"id": i, "name": str(i)} for i in range(1, 91)]

        ann_id_counter = 1
        coco_gt_dict = {
            "images": [],
            "annotations": [],
            "categories": categories,
            "info": {},
            "licenses": []
        }

        for images, targets in tqdm(dataloader, desc=f"Evaluating: {split_name}"):
            images = [img.to(self.device) for img in images]

            preds = self.model(images)
            preds = [{k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in p.items()} for p in preds]
            targets = [{k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]

            for t in targets:
                image_id = int(t["image_id"])
                weather = t.get("weather", "unknown")
                coco_gt_dict["images"].append({"id": image_id})
                
                if weather not in weather_gt_dicts:
                    weather_gt_dicts[weather] = {
                        "images": [],
                        "annotations": [],
                        "categories": categories,
                        "info": {},
                        "licenses": []
                    }
                    weather_predictions[weather] = []
                
                weather_gt_dicts[weather]["images"].append({"id": image_id})

                if "boxes" in t and t["boxes"].numel() > 0:
                    for box, label in zip(t["boxes"], t["labels"]):
                        x1, y1, x2, y2 = box.tolist()
                        w, h = x2 - x1, y2 - y1
                        ann_dict = {
                            "id": ann_id_counter,
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": [x1, y1, w, h],
                            "area": w*h,
                            "iscrowd": 0
                        }
                        coco_gt_dict["annotations"].append(ann_dict)
                        weather_gt_dicts[weather]["annotations"].append(ann_dict.copy())
                        ann_id_counter += 1

            eval_config = self.config_manager.base_config.get("experiment", {}).get("evaluation", {})
            conf_threshold = eval_config.get("conf_threshold", 0.0)

            for pred, gt in zip(preds, targets):
                pred_boxes = pred.get("boxes", torch.empty((0, 4)))
                pred_scores = pred.get("scores", torch.empty((0,)))
                gt_boxes = gt.get("boxes", torch.empty((0, 4)))
                
                if conf_threshold > 0:
                    mask = pred_scores >= conf_threshold
                    pred_boxes = pred_boxes[mask]

                if len(pred_boxes) > 0:
                    if len(gt_boxes) > 0:
                        ious_matrix = box_iou(pred_boxes, gt_boxes)
                        max_ious, _ = ious_matrix.max(dim=1)
                        ious_vals = max_ious.cpu().numpy()
                        per_detection_ious.extend(ious_vals)
                        per_image_ious.append(ious_vals.mean().item())
                    else:
                        per_detection_ious.extend([0.0] * len(pred_boxes))
                        per_image_ious.append(0.0)

            for p, t in zip(preds, targets):
                image_id = int(t["image_id"])
                weather = t.get("weather", "unknown")
                for box, score, label in zip(p.get("boxes", []), p.get("scores", []), p.get("labels", [])):
                    x1, y1, x2, y2 = box.tolist()
                    w, h = x2 - x1, y2 - y1
                    pred_dict = {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, w, h],
                        "score": float(score)
                    }
                    all_predictions.append(pred_dict)
                    weather_predictions[weather].append(pred_dict.copy())

        coco_gt = COCO()
        coco_gt.dataset = coco_gt_dict
        coco_gt.createIndex()
        
        gt_category_ids = set()
        for ann in coco_gt_dict["annotations"]:
            gt_category_ids.add(ann["category_id"])
        
        if gt_category_ids:
            gt_category_names = [cat["name"] for cat in categories if cat["id"] in gt_category_ids]
            print(f"[INFO] {split_name} dataset has {len(gt_category_ids)} categories: {', '.join(sorted(gt_category_names))}")
        else:
            print(f"[WARNING] {split_name} dataset has no annotations (zero-shot evaluation)")

        if all_predictions:
            coco_dt = coco_gt.loadRes(all_predictions)
        else:
            print("[WARNING] No predictions available, skipping COCO evaluation.")
            coco_dt = COCO()
            coco_dt.dataset = {"images": coco_gt_dict["images"], "annotations": [], "categories": coco_gt_dict["categories"]}
            coco_dt.createIndex()

        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metrics = {
            "mAP": coco_eval.stats[0] if len(all_predictions) > 0 else 0.0,
            "mAP_50": coco_eval.stats[1] if len(all_predictions) > 0 else 0.0,
            "mAP_75": coco_eval.stats[2] if len(all_predictions) > 0 else 0.0,
            "AR": coco_eval.stats[8] if len(all_predictions) > 0 else 0.0,
            "mean_IoU": float(np.mean(per_detection_ious)) if per_detection_ious else float("nan"),
            "mean_IoU_per_image": float(np.mean(per_image_ious)) if per_image_ious else float("nan"),
        }
        
        if len(all_predictions) > 0 and hasattr(coco_eval, 'eval') and coco_eval.eval is not None:
            precision = coco_eval.eval['precision']
            
            iou_idx = 0 
            area_idx = 0 
            maxdet_idx = 2 
            
            pr_curve = precision[iou_idx, :, :, area_idx, maxdet_idx]
            pr_curve_mean = np.mean(pr_curve[pr_curve > -1]) if pr_curve.size > 0 else 0.0
            metrics["precision_at_recall_50"] = float(pr_curve_mean) if not np.isnan(pr_curve_mean) else 0.0
        
        metrics["_coco_eval"] = coco_eval
        
        if len(all_predictions) > 0 and hasattr(coco_eval, 'eval') and coco_eval.eval is not None:
            precision = coco_eval.eval['precision']
            for iou_idx, metric_name in [(8, "mAP_90"), (9, "mAP_95")]:
                iou_precision = precision[iou_idx, :, :, 0, 2]
                valid_precision = iou_precision[iou_precision > -1]
                metrics[metric_name] = float(np.mean(valid_precision)) if valid_precision.size > 0 else 0.0
        else:
            metrics["mAP_90"] = 0.0
            metrics["mAP_95"] = 0.0
        
        for weather, w_preds in weather_predictions.items():
            if weather not in weather_gt_dicts:
                continue
            # Skip non-canonical conditions (already folded into overall metrics).
            if weather == OTHER_WEATHER:
                continue

            w_gt_dict = weather_gt_dicts[weather]
            w_coco_gt = COCO()
            w_coco_gt.dataset = w_gt_dict
            w_coco_gt.createIndex()
            
            if len(w_preds) > 0:
                w_coco_dt = w_coco_gt.loadRes(w_preds)
            else:
                w_coco_dt = COCO()
                w_coco_dt.dataset = {"images": w_gt_dict["images"], "annotations": [], "categories": w_gt_dict["categories"]}
                w_coco_dt.createIndex()
            
            w_coco_eval = COCOeval(w_coco_gt, w_coco_dt, iouType='bbox')
            w_coco_eval.evaluate()
            w_coco_eval.accumulate()
            w_coco_eval.summarize()
            
            metrics[f"{weather}/mAP"] = w_coco_eval.stats[0] if len(w_preds) > 0 else 0.0
            metrics[f"{weather}/mAP_50"] = w_coco_eval.stats[1] if len(w_preds) > 0 else 0.0
            metrics[f"{weather}/mAP_75"] = w_coco_eval.stats[2] if len(w_preds) > 0 else 0.0
            metrics[f"{weather}/AR"] = w_coco_eval.stats[8] if len(w_preds) > 0 else 0.0

        return metrics

    def save_checkpoint(self, filename=None, is_best=False, is_last=False, is_eval=False):
        if filename is None:
            filename = "best_model.pth" if is_best else "last_model.pth" if is_last else f"epoch_{self.current_epoch}.pth"
        
        if is_best or is_last:
            path = os.path.join(self.checkpoint_dir, filename)
        elif is_eval:
            path = os.path.join(self.runs_dir, filename)
        else:
            path = os.path.join(self.checkpoint_dir, filename)

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_map": self.best_map,
            "best_map_50": self.best_map_50,
            "best_iou": self.best_iou,
            "normalization_mean": self.norm_mean.tolist() if self.norm_mean is not None else None,
            "normalization_std": self.norm_std.tolist() if self.norm_std is not None else None
        }
        torch.save(checkpoint, path)
        meta_path = path.replace(".pth", "_metadata.json")
        with open(meta_path, "w") as f:
            json.dump({"epoch": self.current_epoch, "best_map": self.best_map}, f, indent=2)
    
    def train(self, prepared_synthetic_data=None, skip_dataset_build=False, train_only=False):
        """
        Full training loop for PyTorch models (ResNet, ConvNeXt).
        
        Args:
            prepared_synthetic_data: Prepared synthetic dataset paths (optional if skip_dataset_build=True)
            skip_dataset_build: If True, assumes datasets/loaders/optimizer/scheduler already built (default: False)
            train_only: If True, skip evaluation during training (default: False)
        """
        # Build datasets and loaders (unless already done)
        if not skip_dataset_build:
            if prepared_synthetic_data is None:
                raise ValueError("prepared_synthetic_data required when skip_dataset_build=False")
            
            self.build_datasets(prepared_synthetic_data)
            self.build_dataloaders()

        # Build optimizer and scheduler
        self.build_optimizer()
        self.build_scheduler()

        training_config = self.config_manager.model_config["training"]
        epochs = training_config["epochs"]

        print(f"Starting training for {epochs} epochs...")
        if train_only:
            print("[INFO] Train-only mode: Evaluation disabled, checkpoints will be saved for later evaluation")
        overall_start = time.time()

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train for one epoch
            train_metrics = self.train_epoch()

            # Debug: log gradient norms if enabled
            log_gradients = training_config["log_gradients"]
            if log_gradients:
                grad_norms = []
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norms.append(p.grad.norm().item())
                avg_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
            else:
                avg_grad_norm = None

            # --- Evaluation Logic (skip if train_only=True) ---
            eval_config = self.config_manager.model_config["evaluation"]
            epochs_to_eval = compute_evaluation_epochs(eval_config, epochs)

            if not train_only and epoch in epochs_to_eval:
                # Evaluate on training data for debugging (if enabled in config)
                log_training_metrics = self.config_manager.model_config["evaluation"].get("log_training_metrics", False)
                if log_training_metrics:
                    train_eval_metrics = self.evaluate(self.train_loader, split_name="train")
                else:
                    train_eval_metrics = None
                
                # Evaluate on validation data
                val_metrics = self.evaluate(self.val_loader, split_name="val")
                
                # Evaluate on additional validation datasets (DAWN, STF, ACDC, etc.)
                additional_val_metrics = {}
                
                if self.additional_val_loaders:
                    # Evaluate each dataset individually
                    for dataset_name, dataloader in self.additional_val_loaders.items():
                        print(f"\n[INFO] Evaluating on {dataset_name}...")
                        metrics = self.evaluate(dataloader, split_name=dataset_name)
                        additional_val_metrics[dataset_name] = metrics
                
                # Track best overall metrics
                if val_metrics["mAP"] > self.best_map:
                    self.best_map = val_metrics["mAP"]
                    self.save_checkpoint(is_best=True)
                if val_metrics["mAP_50"] > self.best_map_50:
                    self.best_map_50 = val_metrics["mAP_50"]
                if val_metrics["mean_IoU"] > self.best_iou:
                    self.best_iou = val_metrics["mean_IoU"]
                
                # Track best per-weather metrics
                for key, value in val_metrics.items():
                    if '/' in key:  # Weather-specific metric (e.g., "clear/mAP")
                        weather, metric = key.split('/', 1)
                        if metric == "mAP":
                            if weather not in self.best_weather_map or value > self.best_weather_map[weather]:
                                self.best_weather_map[weather] = value
                        elif metric == "mAP_50":
                            if weather not in self.best_weather_map_50 or value > self.best_weather_map_50[weather]:
                                self.best_weather_map_50[weather] = value
            else:
                train_eval_metrics = None
                val_metrics = None
                additional_val_metrics = {}

            # Scheduler step
            if self.scheduler:
                self.scheduler.step()

            # --- Checkpointing Logic (Independent of Evaluation) ---
            save_frequency = self.config_manager.model_config["logging"].get("save_frequency", 1)
            
            # Save checkpoint if frequency is met OR if it's the last epoch
            if epoch % save_frequency == 0 or epoch == epochs:
                # We save it as an "eval" checkpoint (normal epoch checkpoint)
                self.save_checkpoint(filename=f"epoch_{epoch}.pth", is_eval=True)
                print(f"[INFO] Saved checkpoint: epoch_{epoch}.pth")

            # Always save last checkpoint (overwriting previous last)
            self.save_checkpoint(is_last=True)

            # --- W&B Logging ---
            log_dict = {
                "epoch": epoch,
                "best_val_map": self.best_map,
                "best_val_map_50": self.best_map_50,
                "best_val_iou": self.best_iou,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            
            # Log best metrics under val_best/ namespace
            log_dict["val_best/mAP"] = self.best_map
            log_dict["val_best/mAP_50"] = self.best_map_50
            log_dict["val_best/mean_IoU"] = self.best_iou
            
            # Log best per-weather metrics
            for weather, best_map in self.best_weather_map.items():
                log_dict[f"val_best/{weather}/mAP"] = best_map
            for weather, best_map_50 in self.best_weather_map_50.items():
                log_dict[f"val_best/{weather}/mAP_50"] = best_map_50
            if avg_grad_norm is not None:
                log_dict["avg_grad_norm"] = avg_grad_norm
            
            # Flatten train loss metrics
            for key, value in train_metrics.items():
                log_dict[f"train/{key}"] = value
            
            # Flatten train evaluation metrics (only if evaluated)
            if train_eval_metrics is not None:
                for key, value in train_eval_metrics.items():
                    if key.startswith('_'): continue
                    if '/' in key:
                        log_dict[f"train/{key}"] = value
                    else:
                        log_dict[f"train/eval_{key}"] = value
            
            # Flatten validation metrics (only if evaluated)
            if val_metrics is not None:
                for key, value in val_metrics.items():
                    if key.startswith('_'): continue
                    if '/' in key:
                        log_dict[f"val/{key}"] = value
                    else:
                        log_dict[f"val/{key}"] = value
            
            # Log all additional validation datasets
            if additional_val_metrics:
                for dataset_name, metrics in additional_val_metrics.items():
                    for key, value in metrics.items():
                        if key.startswith('_'): continue
                        log_dict[f"{dataset_name}/{key}"] = value

            wandb.log(log_dict, step=epoch)
            
            torch.cuda.empty_cache()

            # Console print
            grad_norm_str = f"{avg_grad_norm:.4f}" if avg_grad_norm is not None else "N/A"
            if val_metrics is not None:
                train_map_str = f" | Train mAP={train_eval_metrics['mAP']:.4f}" if train_eval_metrics else ""
                additional_val_str = ""
                if additional_val_metrics:
                    for dataset_name, metrics in additional_val_metrics.items():
                        additional_val_str += f" | {dataset_name.upper()} mAP={metrics['mAP']:.4f}"
                
                print(f"[Epoch {epoch}/{epochs}] "
                    f"Train loss={train_metrics['total_loss']:.4f}{train_map_str} | "
                    f"Val mAP={val_metrics['mAP']:.4f} | "
                    f"Best mAP={self.best_map:.4f}{additional_val_str} | "
                    f"Avg grad norm={grad_norm_str} | "
                    f"Time={time.time()-epoch_start:.1f}s")
            else:
                print(f"[Epoch {epoch}/{epochs}] "
                    f"Train loss={train_metrics['total_loss']:.4f} | "
                    f"Best mAP={self.best_map:.4f} | "
                    f"Avg grad norm={grad_norm_str} | "
                    f"Time={time.time()-epoch_start:.1f}s (no eval)")

        training_time = time.time() - overall_start

        return {
            "best_val_map": self.best_map,
            "best_val_map_50": self.best_map_50,
            "best_val_iou": self.best_iou,
            "training_time": training_time,
            "checkpoint_dir": self.checkpoint_dir,
        }
    
def compute_evaluation_epochs(eval_config, total_epochs):
    """
    Compute which epochs to evaluate based on staged frequency configuration.
    
    Args:
        eval_config: Evaluation config dict with either 'frequency' (legacy) or 'stages' (new)
        total_epochs: Total number of training epochs
    
    Returns:
        Set of epoch numbers to evaluate
    """
    epochs_to_eval = set()
    
    # Legacy support: single frequency for all epochs
    if "frequency" in eval_config and "stages" not in eval_config:
        eval_frequency = eval_config["frequency"]
        for epoch in range(1, total_epochs + 1):
            if epoch % eval_frequency == 0 or epoch == total_epochs:
                epochs_to_eval.add(epoch)
        return sorted(epochs_to_eval)
    
    # New staged frequency system
    if "stages" not in eval_config:
        raise ValueError("evaluation config must have either 'frequency' or 'stages'")
    
    stages = eval_config["stages"]
    
    # Sort stages by epoch
    sorted_stages = sorted(stages.items(), key=lambda x: int(x[0]))
    
    prev_end = 0
    for stage_end_epoch, frequency in sorted_stages:
        stage_end_epoch = int(stage_end_epoch)
        frequency = int(frequency)
        
        # Evaluate from (prev_end + frequency) to stage_end_epoch at 'frequency' intervals
        for epoch in range(prev_end + frequency, stage_end_epoch + 1, frequency):
            epochs_to_eval.add(epoch)
        
        prev_end = stage_end_epoch
    
    # Always evaluate last epoch
    epochs_to_eval.add(total_epochs)
    
    return sorted(epochs_to_eval)