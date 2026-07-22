# Additional Validation Datasets

This directory contains scripts to download and integrate additional datasets for validation of object detection models trained on BDD100K.

## Available Datasets

### 1. DAWN (Detection in Adverse Weather and Night)
- **Source**: https://data.mendeley.com/datasets/766ygrbt8y/3
- **Size**: 1,027 images
- **Weather types**: Fog (300), Rain (200), Sand (323), Snow (204)
- **Format**: Images only (no ground truth annotations)
- **Use case**: Zero-shot generalization testing on diverse adverse weather

### 2. SeeingThroughFog (Planned)
- **Source**: Dense foggy driving scenes
- **Format**: Multi-part archives with annotations
- **Use case**: Dense fog scenarios

## Setup Instructions

### Download Datasets

```bash
# Base dataset — BDD100K (Images 100K) via DatasetNinja.
# Downloads to Data/BDD100K and auto-converts train + val to COCO.
pip install --upgrade dataset-tools
python dataset_utils/download_bdd100k.py

# Download DAWN
python dataset_utils/download_dawn.py

# Download Foggy Zurich
python dataset_utils/download_foggy_zurich.py
```

The synthetically augmented set (**TartanAdversity** — Gemini/Automold data) is
hosted on Hugging Face and loaded separately:

```python
from datasets import load_dataset
ds = load_dataset("JohnYanxinLiu/TartanAdversity", split="train")
```

### Convert to COCO Format

```bash
# BDD100K is converted automatically by download_bdd100k.py, or run manually:
python dataset_utils/convert_bdd100k_to_coco.py \
    --bdd100k_train_path Data/BDD100K/train \
    --output Data/BDD100K/bdd100k_train_coco.json

# Convert DAWN to COCO format
python dataset_utils/convert_dawn_to_coco.py
```

## Validation Workflow

### Validate trained model on additional datasets

```bash
# Validate on DAWN
python validate_additional_datasets.py \
    --checkpoint checkpoints/resnet_r20_gemini/best_model.pth \
    --model_type resnet \
    --datasets dawn

# Validate on multiple datasets
python validate_additional_datasets.py \
    --checkpoint checkpoints/resnet_r20_gemini/best_model.pth \
    --model_type resnet \
    --datasets dawn stf
```

### Output

The validation script provides:
- Overall mAP, mAP@50, mAP@75, AR
- Per-weather breakdown (fog, rain, sand, snow)
- Class-agnostic IoU metrics

## Configuration

Additional validation datasets are configured in `configs/base.yaml`:

```yaml
paths:
  val_datasets:
    dawn:
      images: DAWN/DAWN
      annotations: DAWN/dawn_coco.json
      description: "DAWN - Detection in Adverse Weather and Night"
    stf:
      images: SeeingThroughFog/cam_stereo_left_extracted
      annotations: SeeingThroughFog/stf_coco.json
      description: "SeeingThroughFog - Dense foggy scenes"
```

## Dataset Directory Structure

```
Data/
├── DAWN/
│   ├── DAWN/
│   │   ├── Fog/
│   │   ├── Rain/
│   │   ├── Sand/
│   │   └── Snow/
│   └── dawn_coco.json
├── SeeingThroughFog/
│   ├── cam_stereo_left/
│   └── stf_coco.json (when converted)
└── bdd100k_val/  # Original validation set
```

## Notes

- **No Ground Truth**: DAWN dataset has images only, so mAP will be 0.0. The dataset is useful for:
  - Qualitative evaluation
  - Checking if model produces reasonable detections
  - Testing weather-specific generalization
  
- **Same Categories**: All datasets use BDD100K's 10 categories (bike, bus, car, motor, person, rider, traffic light, traffic sign, train, truck)

- **Normalization**: Uses the same normalization statistics as training (computed from BDD100K training set)

## Future Work

1. Complete SeeingThroughFog integration (multi-part archive extraction + annotation conversion)
2. Add ACDC (Adverse Conditions Dataset with Correspondences) validation
3. Add weather-conditioned metrics
4. Create visualization tools for qualitative comparison
