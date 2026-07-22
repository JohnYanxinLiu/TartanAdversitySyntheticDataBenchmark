"""
YOLO Utility Functions

General helper functions for YOLO training including:
- Seeding for reproducibility
- Model building
"""
import torch
import numpy as np
import random
from ultralytics import YOLO


def set_seed(seed: int = 42):
    """Deterministic seeding for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_yolo_model(model_name="yolov8s", num_classes=10, use_pretrained_weights=False):
    """Build YOLOv8 model with deterministic seeding"""
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_path = f"{model_name}.yaml" if not use_pretrained_weights else f"{model_name}.pt"
    model = YOLO(model_path)
    model.model.nc = num_classes
    return model
