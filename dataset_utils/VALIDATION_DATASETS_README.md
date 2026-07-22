# Additional Validation Datasets Integration

This document describes how to configure and use additional validation datasets (DAWN, SeeingThroughFog, ACDC, etc.) during model training.

## Overview

The training pipeline now supports **multiple additional validation datasets** that can be enabled/disabled through configuration. These datasets are evaluated alongside the primary BDD100K validation set, providing insights into model generalization across different weather conditions and datasets.

## Supported Datasets

### 1. DAWN (Detection in Adverse Weather and Night)
- **Source**: https://data.mendeley.com/datasets/766ygrbt8y/3
- **Size**: 704 images (after excluding sand)
- **Weather Types**: Fog (300), Rain (200), Snow (204)
- **Annotations**: None (zero-shot evaluation only)
- **Use Case**: Test generalization to adverse weather without fine-tuning

**Note**: Sand weather was excluded as it's not relevant for typical autonomous driving scenarios.

### 2. SeeingThroughFog (STF)
- **Source**: AWS S3 bucket
- **Size**: 472 multi-part archives
- **Weather Types**: Dense fog scenes
- **Annotations**: TBD (requires conversion script)
- **Use Case**: Dense fog scenario testing

### 3. ACDC (Adverse Conditions Dataset with Correspondences)
- **Source**: User upload in progress
- **Location**: `dataset_utils/acdc_dataset_zip_files/`
- **Weather Types**: TBD
- **Annotations**: TBD
- **Status**: Not yet integrated (waiting for upload completion)

## Configuration

### Enabling/Disabling Datasets

Edit `configs/base.yaml` to control which validation datasets are used:

```yaml
paths:
  val_datasets:
    dawn:
      enabled: true  # Set to false to disable
      images: DAWN/DAWN
      annotations: DAWN/dawn_coco.json
      description: "DAWN - Detection in Adverse Weather and Night (fog, rain, snow)"
    
    stf:
      enabled: false  # Currently disabled
      images: SeeingThroughFog/cam_stereo_left_extracted
      annotations: SeeingThroughFog/stf_coco.json
      description: "SeeingThroughFog - Dense foggy scenes"
    
    acdc:
      enabled: false  # Not yet ready
      images: ACDC/rgb_anon
      annotations: ACDC/acdc_coco.json
      description: "ACDC - Adverse Conditions Dataset with Correspondences"
```

### Adding New Validation Datasets

1. **Download and prepare the dataset** (images + COCO JSON annotations)
2. **Add configuration to `base.yaml`**:
   ```yaml
   my_dataset:
     enabled: true
     images: MyDataset/images
     annotations: MyDataset/annotations.json
     description: "Description of the dataset"
   ```
3. **Run training** - the dataset will be automatically loaded and evaluated

## Training Workflow

### During Training

When validation is triggered (every `eval_frequency` epochs):

1. **Primary validation** on BDD100K val set
2. **Additional validation** on all enabled datasets (e.g., DAWN)
3. **Metrics logged to W&B** with dataset-specific namespaces:
   - `val/mAP` - BDD100K validation mAP
   - `dawn/mAP` - DAWN overall mAP
   - `dawn/fog/mAP` - DAWN fog-specific mAP
   - `dawn/rain/mAP` - DAWN rain-specific mAP
   - `dawn/snow/mAP` - DAWN snow-specific mAP

### Console Output Example

```
[Epoch 10/100] Train loss=0.4523 | Val mAP=0.3654 | Best mAP=0.3654 | DAWN mAP=0.1234 | Avg grad norm=0.0523 | Time=45.2s
```

## Dataset Utilities

### DAWN Dataset

**Download**:
```bash
python dataset_utils/download_dawn.py
```

**Convert to COCO format** (excludes sand):
```bash
python dataset_utils/convert_dawn_to_coco.py
```

**Test loading**:
```bash
python dataset_utils/test_dawn_dataset.py
```

### SeeingThroughFog Dataset

**Download** (472 multi-part archives):
```bash
python dataset_utils/download_stf.py
```
*Note: Download is in progress and takes several hours*

### ACDC Dataset

*Scripts will be created once upload is complete*

## Validation-Only vs Training

**Important**: All additional datasets (DAWN, STF, ACDC) are **validation-only**. They are:
- ✅ Evaluated at each validation epoch
- ✅ Logged to W&B with per-weather breakdown
- ❌ NOT used for training
- ❌ NOT mixed with training data

This ensures clean zero-shot generalization testing without data leakage.

## Weather-Specific Metrics

The evaluation pipeline automatically computes per-weather metrics:

- **Overall metrics**: `mAP`, `mAP_50`, `mAP_75`, `AR`, `mean_IoU`
- **Per-weather metrics**: `{weather}/mAP`, `{weather}/mAP_50`, etc.
- **Precision-Recall curves**: Per-weather PR curves for visualization

Example W&B metrics:
```
dawn/mAP = 0.234
dawn/fog/mAP = 0.256
dawn/rain/mAP = 0.198
dawn/snow/mAP = 0.248
```

## Implementation Details

### Code Changes

1. **`trainer.py`**:
   - Added `self.additional_val_datasets` dict to store enabled datasets
   - Added `self.additional_val_loaders` dict for dataloaders
   - Modified `build_datasets()` to load enabled validation datasets
   - Modified `build_dataloaders()` to create dataloaders for additional datasets
   - Modified training loop to evaluate on all additional datasets
   - Added W&B logging for additional dataset metrics

2. **`configs/base.yaml`**:
   - Added `paths.val_datasets` section with `enabled` flags
   - Added DAWN, STF, ACDC configurations

3. **`dataset_utils/convert_dawn_to_coco.py`**:
   - Removed "Sand" from weather types (only Fog, Rain, Snow)
   - Updated from 1027 → 704 images

## Troubleshooting

### Dataset not loading
- Check `enabled: true` in `base.yaml`
- Verify paths are correct relative to `Data/` directory
- Ensure COCO JSON exists and is valid

### Missing metrics in W&B
- Check that dataset was successfully loaded (see console output during dataset building)
- Verify dataloader was created in `build_dataloaders()`

### Out of memory
- Reduce batch size for validation
- Disable some validation datasets if memory is limited

## Future Work

- [ ] Complete SeeingThroughFog integration (convert to COCO, test)
- [ ] Integrate ACDC dataset once upload completes
- [ ] Add support for filtering validation datasets by weather type
- [ ] Add cross-dataset evaluation comparison plots
