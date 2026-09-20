"""
Unified Faster R-CNN trainer supporting ResNet and ConvNeXt backbones
Based on Wilson et al. "Predictive Inequity in Object Detection" settings

Supports:
- --train_only: Train without evaluation (faster training)
- --evaluate_only: Evaluate existing checkpoints without training
"""
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models import resnet50
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import argparse
import os
import sys
import numpy as np
import random
import glob
import json
import wandb

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_manager import ConfigManager
from trainer import BaseTrainer as Trainer, compute_evaluation_epochs
from data_utils import prepare_synthetic_datasets


def build_resnet_model(num_classes=11, use_pretrained_weights=False, image_mean=None, image_std=None):
    """
    Build ResNet-50 Faster R-CNN model.
    
    Args:
        num_classes: Number of classes (including background)
        use_pretrained_weights: If True, load pretrained weights
        image_mean: Mean for normalization (computed from training data)
        image_std: Std for normalization (computed from training data)
    
    Returns:
        PyTorch Faster R-CNN model with ResNet-50 backbone
    """
    # Default to ImageNet stats if not provided
    if image_mean is None:
        image_mean = [0.485, 0.456, 0.406]
    if image_std is None:
        image_std = [0.229, 0.224, 0.225]
    
    # Create custom transform with computed normalization
    transform = GeneralizedRCNNTransform(
        min_size=800,
        max_size=1333,
        image_mean=image_mean,
        image_std=image_std
    )
    
    if use_pretrained_weights:
        # Use standard helper for pretrained weights (FrozenBN by default)
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        # Replace classifier head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        # Replace transform with custom one
        model.transform = transform
    else:
        # Manual build for scratch training with GroupNorm (more stable than BatchNorm)
        print("Building ResNet from scratch with GroupNorm...")
        
        # Define callable that creates GroupNorm with 32 groups
        def norm_layer(channels):
            return nn.GroupNorm(num_groups=32, num_channels=channels)
        
        backbone = resnet_fpn_backbone(
            backbone_name='resnet50',
            weights=None,
            norm_layer=norm_layer
        )

        for m in backbone.modules():
            if isinstance(m, nn.GroupNorm):
                m.weight.requires_grad = True
                m.bias.requires_grad = True

        model = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            transform=transform
        )
    
    
    return model


def build_convnext_model(num_classes=11, use_pretrained_weights=False, image_mean=None, image_std=None):
    from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
    from torchvision.models.feature_extraction import create_feature_extractor
    from torchvision.models import convnext_base, ConvNeXt_Base_Weights
    import torch.nn as nn

    # Default to ImageNet stats if not provided
    if image_mean is None:
        raise ValueError("image_mean must be provided for ConvNeXt backbone")
    if image_std is None:
        raise ValueError("image_std must be provided for ConvNeXt backbone")
    
    # Create custom transform with computed normalization
    transform = GeneralizedRCNNTransform(
        min_size=800,
        max_size=1333,
        image_mean=image_mean,
        image_std=image_std
    )

    weights = ConvNeXt_Base_Weights.DEFAULT if use_pretrained_weights else None
    backbone_model = convnext_base(weights=weights)

    # Remove classifier head
    backbone_model.classifier = nn.Identity()

    # ConvNeXt stage outputs
    return_nodes = {
        "features.1": "0",  # C=128
        "features.3": "1",  # C=256
        "features.5": "2",  # C=512
        "features.7": "3",  # C=1024
    }

    feature_extractor = create_feature_extractor(
        backbone_model,
        return_nodes=return_nodes
    )

    in_channels_list = [128, 256, 512, 1024]
    out_channels = 256

    class ConvNeXtBackboneWithFPN(nn.Module):
        def __init__(self, body):
            super().__init__()
            self.body = body
            self.fpn = FeaturePyramidNetwork(
                in_channels_list=in_channels_list,
                out_channels=out_channels,
                extra_blocks=LastLevelMaxPool(),
            )
            self.out_channels = out_channels
        
        def forward(self, x):
            x = self.body(x)
            return self.fpn(x)

    backbone = ConvNeXtBackboneWithFPN(feature_extractor)

    model = FasterRCNN(
        backbone, 
        num_classes=num_classes,
        transform=transform
    )
    return model


def build_model(model_type, num_classes=11, use_pretrained_weights=False, image_mean=None, image_std=None):
    """
    Build Faster R-CNN model with specified backbone.
    
    Args:
        model_type: "resnet" or "convnext"
        num_classes: Number of classes (including background)
        use_pretrained_weights: If True, load pretrained weights
        image_mean: Mean for normalization (computed from training data)
        image_std: Std for normalization (computed from training data)
    
    Returns:
        PyTorch Faster R-CNN model
    """
    # Deterministic seeding
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Build model based on type
    if model_type == "resnet":
        model = build_resnet_model(num_classes, use_pretrained_weights, image_mean, image_std)
    elif model_type == "convnext":
        model = build_convnext_model(num_classes, use_pretrained_weights, image_mean, image_std)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'resnet' or 'convnext'")

    # Weight initialization
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            init.kaiming_uniform_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    if not use_pretrained_weights:
        print("Applying Kaiming initialization to all layers for scratch training...")
        model.apply(init_weights)
    else:
        for m in model.roi_heads.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    for layer in [model.roi_heads.box_predictor.cls_score, model.roi_heads.box_predictor.bbox_pred]:
        nn.init.normal_(layer.weight, std=0.01)
        nn.init.zeros_(layer.bias)

    print(f"{model_type.upper()} Faster R-CNN model built with {num_classes} classes. Pretrained weights: {use_pretrained_weights}. Parameter count: {sum(p.numel() for p in model.parameters())}")

    return model


def find_checkpoints(run_name):
    """
    Find all checkpoint files for a given run.
    
    Args:
        run_name: Name of the run (e.g., "resnet_0.5_gemini")
    
    Returns:
        List of tuples (epoch_num, checkpoint_path, metadata_path)
    """
    runs_dir = os.path.join("runs", run_name)
    
    if not os.path.exists(runs_dir):
        raise FileNotFoundError(f"Run directory not found: {runs_dir}")
    
    # Find all epoch_*.pth files
    checkpoint_pattern = os.path.join(runs_dir, "epoch_*.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {runs_dir}")
    
    checkpoints = []
    for ckpt_path in sorted(checkpoint_files):
        # Extract epoch number from filename
        basename = os.path.basename(ckpt_path)  # e.g., "epoch_10.pth"
        try:
            epoch_num = int(basename.replace("epoch_", "").replace(".pth", ""))
            metadata_path = ckpt_path.replace(".pth", "_metadata.json")
            checkpoints.append((epoch_num, ckpt_path, metadata_path))
        except ValueError:
            print(f"[WARNING] Skipping invalid checkpoint filename: {basename}")
            continue
    
    return sorted(checkpoints, key=lambda x: x[0])


def evaluate_checkpoints(model_type, mix_rate, augmentation_type, mixing_method="replacement", wandb_key=None, use_augmentation=False):
    """
    Evaluate all checkpoints for a run based on the evaluation frequency in config.
    
    Args:
        model_type: "resnet" or "convnext"
        mix_rate: Float between 0.0 and 1.0 (replacement rate or addition rate)
        augmentation_type: "gemini" or "automold"
        mixing_method: "replacement" or "addition"
        wandb_key: Optional W&B API key
        use_augmentation: Whether training used augmentations
        
    Returns:
        Dictionary with evaluation results
    """
    # Create config manager to get run name and eval frequency
    config_manager = ConfigManager(
        config_dir="configs",
        model_type=model_type,
        mix_rate=mix_rate,
        augmentation_type=augmentation_type,
        mixing_method=mixing_method
    )
    
    config_manager.use_augmentation = use_augmentation
    
    # Get evaluation configuration
    eval_config = config_manager.model_config["evaluation"]
    total_epochs = config_manager.model_config["training"]["epochs"]

    run_name = config_manager.run_name
    print(f"\n{'='*80}")
    print(f"EVALUATION MODE: {run_name}")
    print(f"Total Epochs: {total_epochs}")
    print(f"{'='*80}\n")

    # Find all checkpoints
    try:
        checkpoints = find_checkpoints(run_name)
        print(f"Found {len(checkpoints)} checkpoint(s)")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return None

    # Compute which epochs to evaluate based on config
    epochs_to_eval = compute_evaluation_epochs(eval_config, total_epochs)

    print(f"Epochs to evaluate: {epochs_to_eval}")
    
    # Load wandb run ID if it exists and resume the run
    wandb_run_id_file = os.path.join("runs", run_name, "wandb_run_id.txt")
    wandb_run_id = None
    if os.path.exists(wandb_run_id_file):
        with open(wandb_run_id_file, "r") as f:
            wandb_run_id = f.read().strip()
        print(f"[INFO] Found existing wandb run ID: {wandb_run_id}")
        print(f"[INFO] Will resume wandb logging to the same run")
    else:
        print(f"[WARNING] No wandb run ID file found at {wandb_run_id_file}")
        if wandb_key:
            print(f"[WARNING] Will create a new wandb run for evaluation")
    
    # Initialize trainer with wandb resumption
    trainer = Trainer(config_manager, wandb_key, resume_wandb_id=wandb_run_id)

    # Define metrics for wandb to ensure they get charted
    if wandb.run is not None:
        # Set epoch as the primary x-axis for all metrics
        wandb.define_metric("epoch")
        # Link ALL metrics to use 'epoch' as the step instead of internal '_step'
        wandb.define_metric("*", step_metric="epoch")
    
    # Prepare synthetic data and build datasets
    prepared_data = prepare_synthetic_datasets(config_manager, augmentation_type)
    trainer.build_datasets(prepared_data)
    trainer.build_dataloaders()
    
    # Build model with normalization stats
    num_classes = config_manager.model_config["model"]["num_classes"]
    assert trainer.norm_mean is not None and trainer.norm_std is not None, "Normalization stats not computed!"
    image_mean = trainer.norm_mean.tolist()
    image_std = trainer.norm_std.tolist()
    
    model = build_model(
        model_type, 
        num_classes, 
        use_pretrained_weights=config_manager.model_config["model"]["pretrained"],
        image_mean=image_mean,
        image_std=image_std
    )
    
    trainer.set_model(model)
    
    # Evaluate each checkpoint
    results = {}
    for epoch_num, ckpt_path, metadata_path in checkpoints:
        if epoch_num not in epochs_to_eval:
            print(f"[INFO] Skipping epoch {epoch_num} (not in evaluation schedule)")
            continue
        
        print(f"\n{'='*80}")
        print(f"Evaluating checkpoint: epoch_{epoch_num}.pth")
        print(f"{'='*80}")
        
        # Load checkpoint. weights_only=False because these checkpoints store
        # numpy scalars (best_map etc.) beside the state dict; torch>=2.6 would
        # otherwise refuse to unpickle them. They are our own trained files.
        checkpoint = torch.load(ckpt_path, map_location=trainer.device, weights_only=False)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.current_epoch = checkpoint['epoch']  # Set epoch for wandb logging
        print(f"Loaded model state from epoch {checkpoint['epoch']}")
        
        # Evaluate on validation set
        val_metrics = trainer.evaluate(trainer.val_loader, split_name="val")
        
        # Evaluate on additional validation datasets
        additional_val_metrics = {}
        if trainer.additional_val_loaders:
            for dataset_name, dataloader in trainer.additional_val_loaders.items():
                print(f"\n[INFO] Evaluating on {dataset_name}...")
                metrics = trainer.evaluate(dataloader, split_name=dataset_name)
                additional_val_metrics[dataset_name] = metrics
        
        # Store results
        results[epoch_num] = {
            "val": val_metrics,
            "additional_val": additional_val_metrics
        }
        
        # Log to wandb if active
        if wandb.run is not None:
            log_dict = {
                "epoch": epoch_num,
            }
            
            # Log validation metrics
            for key, value in val_metrics.items():
                if key.startswith('_'): continue
                if '/' in key:
                    log_dict[f"val/{key}"] = value
                else:
                    log_dict[f"val/{key}"] = value
            
            # Log all additional validation datasets
            for dataset_name, metrics in additional_val_metrics.items():
                for key, value in metrics.items():
                    if key.startswith('_'): continue
                    log_dict[f"{dataset_name}/{key}"] = value
            
            wandb.log(log_dict)  # Remove step=epoch_num
            print(f"[INFO] Logged evaluation metrics to wandb for epoch {epoch_num}")
        
        # Print summary
        print(f"\n[Epoch {epoch_num}] Val mAP: {val_metrics['mAP']:.4f} | Val mAP@50: {val_metrics['mAP_50']:.4f}")
        for dataset_name, metrics in additional_val_metrics.items():
            print(f"[Epoch {epoch_num}] {dataset_name.upper()} mAP: {metrics['mAP']:.4f} | mAP@50: {metrics['mAP_50']:.4f}")
        
        # Save evaluation results to file
        eval_results_path = os.path.join("runs", run_name, f"epoch_{epoch_num}_eval_results.json")
        with open(eval_results_path, "w") as f:
            # Convert any non-serializable objects (like COCO eval)
            serializable_results = {
                "epoch": epoch_num,
                "val": {k: v for k, v in val_metrics.items() if not k.startswith('_')},
                "additional_val": {
                    ds_name: {k: v for k, v in metrics.items() if not k.startswith('_')}
                    for ds_name, metrics in additional_val_metrics.items()
                }
            }
            json.dump(serializable_results, f, indent=2)
        print(f"[INFO] Saved evaluation results to: {eval_results_path}")
    
    # Save summary of all evaluations
    summary_path = os.path.join("runs", run_name, "evaluation_summary.json")
    summary = {
        "run_name": run_name,
        "model_type": model_type,
        "mix_rate": mix_rate,
        "augmentation_type": augmentation_type,
        "mixing_method": mixing_method,
        "use_augmentation": use_augmentation,
        "epochs_evaluated": list(results.keys()),
        "results": {
            str(epoch): {
                "val_mAP": res["val"]["mAP"],
                "val_mAP_50": res["val"]["mAP_50"],
                **{f"{ds_name}_mAP": res["additional_val"][ds_name]["mAP"] 
                   for ds_name in res["additional_val"]}
            }
            for epoch, res in results.items()
        }
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[INFO] Saved evaluation summary to: {summary_path}")
    
    return results


def run_experiment(model_type, mix_rate, augmentation_type, mixing_method, wandb_key=None, use_augmentation=False, train_only=False):
    """
    Run a single experiment with given parameters.
    
    Args:
        model_type: "resnet" or "convnext"
        mix_rate: Float between 0.0 and 1.0
        augmentation_type: "gemini" or "automold"
        mixing_method: "replacement" or "addition"
        wandb_key: Optional W&B API key
        use_augmentation: Whether to apply training augmentations
        train_only: If True, skip evaluation during training (faster)
        
    Returns:
        Dictionary with experiment results
    """
    # Create config manager for this experiment
    config_manager = ConfigManager(
        config_dir="configs",
        model_type=model_type,
        mix_rate=mix_rate,
        augmentation_type=augmentation_type,
        mixing_method=mixing_method
    )
    
    # Store use_augmentation for dataset access
    config_manager.use_augmentation = use_augmentation
    
    print(f"Starting {model_type.upper()} experiment: {config_manager.run_name}")
    print(f"Mixing: {mixing_method}, Rate: {mix_rate*100:.1f}%, Aug: {augmentation_type}")
    if train_only:
        print("[INFO] TRAIN-ONLY MODE: Evaluation will be skipped during training")
    
    # Initialize trainer
    trainer = Trainer(config_manager, wandb_key)
    
    # Prepare synthetic data
    prepared_data = prepare_synthetic_datasets(config_manager, augmentation_type)
    
    # Build datasets first to compute normalization stats
    trainer.build_datasets(prepared_data)
    trainer.build_dataloaders()
    
    # Now build model with the computed normalization stats
    num_classes = config_manager.model_config["model"]["num_classes"]
    
    # Get normalization stats from trainer (computed from training data)
    assert trainer.norm_mean is not None and trainer.norm_std is not None, "Normalization stats not computed!"
    image_mean = trainer.norm_mean.tolist()
    image_std = trainer.norm_std.tolist()
    
    model = build_model(
        model_type, 
        num_classes, 
        use_pretrained_weights=config_manager.model_config["model"]["pretrained"],
        image_mean=image_mean,
        image_std=image_std
    )
    
    # Sanity check: Ensure normalization layers are trainable (only for models trained from scratch)
    if not config_manager.model_config["model"]["pretrained"]:
        norm_layer_count = 0
        for m in model.modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm, torch.nn.LayerNorm)):
                assert m.weight.requires_grad, f"{type(m).__name__} weight should be trainable when training from scratch"
                assert m.bias.requires_grad, f"{type(m).__name__} bias should be trainable when training from scratch"
                norm_layer_count += 1
        print(f"[VERIFIED] All {norm_layer_count} normalization layers are trainable (ResNet: GroupNorm, ConvNeXt: LayerNorm)")
    
    trainer.set_model(model)
    
    # Run training (skip dataset/loader building since we already did it)
    results = trainer.train(skip_dataset_build=True, train_only=train_only)
    
    print(f"{model_type.upper()} experiment completed: {config_manager.run_name}")
    if not train_only:
        print(f"Best val mAP: {results['best_val_map']:.4f}")
        print(f"Best val mAP@50: {results['best_val_map_50']:.4f}")
        print(f"Best val IoU: {results['best_val_iou']:.4f}")
    else:
        print(f"[INFO] Training completed. Run with --evaluate_only to evaluate checkpoints.")
    print(f"Models saved in: {results['checkpoint_dir']}")
    
    return results


def main():
    """
    Main function to run experiments from command line
    """
    parser = argparse.ArgumentParser(
        description="Train Faster R-CNN with ResNet or ConvNeXt backbone on BDD100K with synthetic data replacement"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["resnet", "convnext"],
        help="Model backbone type"
    )
    parser.add_argument(
        "--replacement_percentage",
        type=float,
        required=False,
        default=None,
        help="Replacement percentage (0.0 to 1.0). Cannot be used with --addition_percentage."
    )
    parser.add_argument(
        "--addition_percentage",
        type=float,
        required=False,
        default=None,
        help="Addition percentage (0.0 to <1.0). Target synthetic ratio in final dataset. Cannot be used with --replacement_percentage."
    )
    parser.add_argument(
        "--augmentation_type",
        type=str,
        required=True,
        choices=["gemini", "automold"],
        help="Augmentation type"
    )
    parser.add_argument(
        "--wandb_key",
        type=str,
        default=os.getenv("WANDB_API_KEY"),
        help="Weights & Biases API key for logging (defaults to WANDB_API_KEY env var; see .env.example)"
    )
    parser.add_argument(
        "--use_augmentation",
        action="store_true",
        help="Enable training augmentations (default: False)"
    )
    parser.add_argument(
        "--train_only",
        action="store_true",
        help="Train without evaluation (faster training, evaluate later with --evaluate_only)"
    )
    parser.add_argument(
        "--evaluate_only",
        action="store_true",
        help="Only evaluate existing checkpoints (no training)"
    )
    
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

    # Check for conflicting flags
    if args.train_only and args.evaluate_only:
        parser.error("Cannot specify both --train_only and --evaluate_only")
    
    # Run evaluation only
    if args.evaluate_only:
        results = evaluate_checkpoints(
            model_type=args.model_type,
            mix_rate=rate,
            mixing_method=mixing_method,
            augmentation_type=args.augmentation_type,
            wandb_key=args.wandb_key,
            use_augmentation=args.use_augmentation
        )
        
        if results is not None:
            print("\n" + "="*80)
            print("Evaluation completed!")
            print(f"Evaluated {len(results)} epoch(s)")
            print("="*80)
    else:
        # Run training (with or without evaluation)
        results = run_experiment(
            model_type=args.model_type,
            mix_rate=rate,
            mixing_method=mixing_method,
            augmentation_type=args.augmentation_type,
            wandb_key=args.wandb_key,
            use_augmentation=args.use_augmentation,
            train_only=args.train_only
        )
        
        print("\n" + "="*80)
        print("Experiment completed!")
        if not args.train_only:
            print(f"Best validation mAP: {results['best_val_map']:.4f}")
        print("="*80)


if __name__ == "__main__":
    main()
