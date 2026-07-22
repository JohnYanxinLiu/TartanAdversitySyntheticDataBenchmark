"""
Configuration manager for synthetic data replacement experiments
Simplified for single experiment execution
"""
import os
import yaml
from typing import Any, Dict, Optional, Tuple


def resolve_wandb_target(wandb_config: Optional[Dict[str, Any]] = None) -> Tuple[Optional[str], str]:
    """Resolve the Weights & Biases entity and project to log to.

    Environment variables (typically set from a sourced ``.env`` file) take
    precedence over values in ``configs/base.yaml`` so that no personal
    account details need to be committed:

        WANDB_ENTITY   -> wandb entity (team/user); optional, falls back to the
                          account default if unset in both env and config.
        WANDB_PROJECT  -> wandb project; required (env or config).

    See ``.env.example`` for the expected variables.
    """
    wandb_config = wandb_config or {}
    entity = os.getenv("WANDB_ENTITY") or wandb_config.get("entity")
    project = os.getenv("WANDB_PROJECT") or wandb_config.get("project")
    if not project:
        raise ValueError(
            "W&B project is not configured. Set WANDB_PROJECT in your .env "
            "(see .env.example) or 'project' under 'wandb' in configs/base.yaml."
        )
    return entity, project


class ConfigManager:
    """Manages configuration for single experiments"""

    def __init__(self, config_dir: str = "configs", model_type: Optional[str] = None,
                 mix_rate: Optional[float] = None, 
                 augmentation_type: Optional[str] = None,
                 mixing_method: str = "replacement"):
        self.config_dir = config_dir
        self.base_config = None
        self.model_config = None
        self.model_type = model_type
        self.mix_rate = mix_rate
        self.augmentation_type = augmentation_type
        self.mixing_method = mixing_method
        self.run_name = None
        self._load_configs()
        
        if model_type and mix_rate is not None and augmentation_type:
            # naming convention: rXX for replacement (e.g. r05 = 5% replacement)
            #                    aXX for addition (e.g. a05 = 5% addition)
            method_char = "a" if mixing_method == "addition" else "r"
            
            self.run_name = (
                f"{model_type}_{method_char}{int(mix_rate*100):02d}_{augmentation_type}"
            )

    def _load_configs(self):
        """Load base and model configuration files"""
        # Base config
        base_config_path = os.path.join(self.config_dir, "base.yaml")
        if not os.path.exists(base_config_path):
            raise FileNotFoundError(f"Base config not found: {base_config_path}")

        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)

        # Model config (if specified)
        if self.model_type:
            config_path = os.path.join(self.config_dir, f"{self.model_type}.yaml")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Model config not found: {config_path}")
            with open(config_path, 'r') as f:
                self.model_config = yaml.safe_load(f)
            print(f"Loaded {self.model_type} config")

    def get_paths(self, augmentation_type: str) -> Dict[str, Any]:
        """
        Build paths using YAML configuration.
        
        Returns paths for training, validation, and synthetic data based on base.yaml
        """
        # Get base paths from YAML
        paths_config = self.base_config["paths"]
        data_dir = paths_config["data_dir"]
        
        # Build full paths for training data (from YAML train section)
        train_images = os.path.join(data_dir, paths_config["train"]["images"])
        train_annotations = os.path.join(data_dir, paths_config["train"]["annotations"])
        
        # Build full paths for validation data (from YAML val section)
        val_images = os.path.join(data_dir, paths_config["val"]["images"])
        val_annotations = os.path.join(data_dir, paths_config["val"]["annotations"])
        
        # Build full paths for synthetic data based on augmentation type
        if augmentation_type not in ["gemini", "automold"]:
            raise ValueError(f"Unknown augmentation_type: {augmentation_type}")
        
        synthetic_config = paths_config["synthetic"][augmentation_type]
        synthetic_base_dir = os.path.join(data_dir, synthetic_config["base_dir"])
        synthetic_images = os.path.join(data_dir, synthetic_config["images"])
        
        paths = {
            "data_dir": data_dir,
            "bdd": {
                "train_annotations": train_annotations,
                "train_images": train_images,
                "val_annotations": val_annotations,
                "val_images": val_images,
            },
            "synthetic": {
                "base_dir": synthetic_base_dir,
                "images": synthetic_images,
                "annotations": {
                    "fog": os.path.join(data_dir, synthetic_config["annotations"]["fog"]),
                    "rain": os.path.join(data_dir, synthetic_config["annotations"]["rain"]),
                    "snow": os.path.join(data_dir, synthetic_config["annotations"]["snow"]),
                },
                "rankings": {
                    "fog": os.path.join(data_dir, synthetic_config["rankings"], "fog.csv"),
                    "rain": os.path.join(data_dir, synthetic_config["rankings"], "rain.csv"),
                    "snow": os.path.join(data_dir, synthetic_config["rankings"], "snow.csv"),
                }
            },
            "checkpoints_dir": paths_config["checkpoints_dir"],
            "results_dir": paths_config["results_dir"],
        }
        
        return paths

    def get_experiment_settings(self) -> Dict[str, Any]:
        """Return experiment settings"""
        return self.base_config["experiment"]

    def get_wandb_config(self) -> Dict[str, Any]:
        """Return wandb configuration"""
        return self.base_config["wandb"]

    def get_training_config(self) -> Dict[str, Any]:
        """Return training configuration for the current model"""
        if not self.model_config:
            raise ValueError("No model config loaded")
        return self.model_config["training"]

    def get_augmentation_config(self) -> Dict[str, Any]:
        """Return augmentation configuration from base config"""
        return self.base_config.get("augmentation", {})


if __name__ == "__main__":
    config_manager = ConfigManager(model_type="resnet", mix_rate=0.2,
                                   augmentation_type="gemini")
    print("Dataset size:", config_manager.get_experiment_settings()["dataset_size"])
    print(f"Run name: {config_manager.run_name}")
    
    paths = config_manager.get_paths("gemini")
    print(f"Train annotations: {paths['bdd']['train_annotations']}")
