# Dataset Preprocessing Guide

This directory contains scripts to preprocess additional validation datasets (DAWN, SeeingThroughFog, ACDC) for adverse weather testing.

## Datasets

### 1. DAWN ✅ READY
- **Status**: Fully preprocessed
- **Weather**: Fog (104), Haze (114), Mist (82), Rain (200), Snow (204)
- **Images**: 704
- **Annotations**: 5,610 (BDD100K format)
- **Labels**: bike (17), bus (106), car (4706), motor (28), person (303), truck (450)
- **Missing**: rider, traffic light, traffic sign, train
- **Script**: `convert_dawn_to_coco.py`

**Weather Subcategories**:
- Fog directory contains 3 types: fog (foggy-*.jpg), haze (haze-*.jpg), mist (mist-*.jpg)
- Rain/Snow each have single type: rain, snow

**Usage**:
```bash
python dataset_utils/convert_dawn_to_coco.py
```

Already integrated! Enable in `configs/base.yaml`:
```yaml
val_datasets:
  dawn:
    enabled: true
```

### 2. ACDC ✅ READY
- **Status**: Fully preprocessed
- **Weather**: Fog, Rain, Snow (Night excluded)
- **Images**: 3,800 (400 fog train, 1200 rain train, 400 snow train + val/test)
- **Annotations**: 17,491 (BDD100K format)
- **Labels**: bike, bus, car, motor, person, rider, train, truck
- **Script**: `preprocess_acdc.py`

**Usage**:
```bash
# Already run! ACDC is ready to use.
# If you need to re-run:
python dataset_utils/preprocess_acdc.py
```

**Paths**:
- Images: `Data/ACDC/rgb_anon/`
- Annotations: `Data/ACDC/acdc_coco.json`

Enable in `configs/base.yaml`:
```yaml
val_datasets:
  acdc:
    enabled: true
    images: ACDC
    annotations: ACDC/acdc_coco.json
```

### 3. SeeingThroughFog ⚠️ MANUAL EXTRACTION REQUIRED
- **Status**: Downloaded, needs manual extraction
- **Weather**: Dense fog, Light fog, Rain, Snow
- **Images**: ~12,000 (estimated)
- **Annotations**: Available in gt_labels (not downloaded yet)
- **Script**: `preprocess_stf.py`

**Manual Steps Required**:

1. **Extract multi-volume archive** (requires 7z):
   ```bash
   cd Data/SeeingThroughFog/cam_stereo_left
   # Multi-part archives already combined into cam_stereo_left.zip
   
   # Install 7z if not available
   module load p7zip  # or: sudo apt install p7zip-full
   
   # Extract (this will take a while - 21GB archive)
   7z x cam_stereo_left.zip
   ```

2. **Download gt_labels** from [STF Dataset Website](https://light.princeton.edu/datasets/automated_driving_dataset/)
   - Download `gt_labels.zip`
   - Extract to `Data/SeeingThroughFog/gt_labels/`

3. **Run preprocessing**:
   ```bash
   python dataset_utils/preprocess_stf.py
   ```

4. **Enable in config**:
   ```yaml
   val_datasets:
     stf:
       enabled: true
       images: SeeingThroughFog/cam_stereo_left_lut
       annotations: SeeingThroughFog/stf_coco.json
   ```

**Why Manual?**:
- STF uses multi-volume archives requiring 7z (not available in standard Python)
- GT labels are separate download from main dataset
- Dataset is very large (21GB compressed)

## Label Mapping

All datasets are converted to BDD100K label format:

| BDD100K ID | Name | ACDC/Cityscapes | DAWN | STF |
|------------|------|----------------|------|-----|
| 0 | bike | bicycle (33) ✓ | ✓ | Cyclist ✓ |
| 1 | bus | bus (28) ✓ | ✓ | - |
| 2 | car | car (26) ✓ | ✓ | Car ✓ |
| 3 | motor | motorcycle (32) ✓ | ✓ | - |
| 4 | person | person (24) ✓ | ✓ | Pedestrian ✓ |
| 5 | rider | rider (25) ✓ | - | - |
| 6 | traffic light | - | - | - |
| 7 | traffic sign | - | - | - |
| 8 | train | train (31) ✓ | - | - |
| 9 | truck | truck (27) ✓ | ✓ | - |

**Notes**:
- DAWN has 6/10 BDD100K classes (bike, bus, car, motor, person, truck)
- ACDC has 8/10 BDD100K classes (missing traffic light/sign)
- STF has 3/10 BDD100K classes (bike, car, person)
- COCO evaluation only evaluates categories present in ground truth
- Model is NOT penalized for predicting categories missing from dataset

## Preprocessing Scripts

### `download_dawn.py`
Downloads and extracts DAWN dataset from Mendeley.

### `convert_dawn_to_coco.py`
Converts DAWN to COCO format (images only, no annotations).
- Excludes sand weather
- 704 images: fog (300), rain (200), snow (204)

### `preprocess_acdc.py`
Extracts and converts ACDC dataset.
- Extracts images from `rgb_anon_trainvaltest.zip`
- Converts labels from Cityscapes to BDD100K format
- Combines train+val+test into single validation set
- 3,800 images: fog (1000), rain (1800), snow (1000)

### `preprocess_stf.py`
Converts STF dataset (requires manual extraction).
- Reads weather splits from STF repo in `dataset_utils/SeeingThroughFog_repo/`
- Maps STF labels (Pedestrian, Car, Cyclist) to BDD100K
- Creates COCO JSON with adverse weather subsets

**Adding STF repo** (if not present):
```bash
cd JohnWorkspace/GenDataTraining
git submodule add https://github.com/princeton-computational-imaging/SeeingThroughFog.git dataset_utils/SeeingThroughFog_repo
```

## Testing

All test scripts are now located in the `tests/` directory:

### `tests/test_dawn_dataset.py`
Tests DAWN dataset loading with MixedGenDataSet.

### `tests/test_additional_datasets.py`
Tests loading DAWN and ACDC datasets with MixedGenDataSet.

### `tests/test_val_datasets_config.py`
Tests configuration loading for validation datasets.

**To run tests**:
```bash
cd JohnWorkspace/GenDataTraining
python tests/test_additional_datasets.py
python tests/test_val_datasets_config.py
python tests/test_dawn_dataset.py
```

## Quick Reference

**Datasets Ready to Use**:
- ✅ DAWN (704 images, no annotations)
- ✅ ACDC (3,800 images, 17,491 annotations)

**Datasets Pending**:
- ⚠️ STF (needs manual extraction + gt_labels download)

**To Enable a Dataset**:
1. Edit `configs/base.yaml`
2. Set `enabled: true` for the dataset
3. Run training - dataset will be automatically loaded and evaluated

**To Test Loading**:
```bash
python dataset_utils/test_dawn_dataset.py
# Or create similar test for ACDC/STF
```

## File Structure

```
Data/
├── DAWN/
│   ├── DAWN/
│   │   ├── Fog/
│   │   ├── Rain/
│   │   └── Snow/
│   └── dawn_coco.json
├── ACDC/
│   ├── rgb_anon/
│   │   ├── fog/
│   │   ├── rain/
│   │   └── snow/
│   └── acdc_coco.json
└── SeeingThroughFog/
    ├── cam_stereo_left/        (needs extraction)
    ├── cam_stereo_left_lut/    (optional, processed images)
    ├── gt_labels/              (needs download)
    ├── calib_*.json
    └── stf_coco.json           (created by preprocessing)
```

## Troubleshooting

**"Images not found" warnings**:
- Check that zip files are fully extracted
- Verify paths in config match actual directory structure

**"Ground truth not found"**:
- For ACDC: Extract `gt_detection_trainval.zip` in `dataset_utils/acdc_dataset_zip_files/`
- For STF: Download gt_labels from STF website

**Out of memory during extraction**:
- Extract on a machine with sufficient disk space (ACDC: ~17GB, STF: ~21GB)
- For STF, extraction can be done incrementally

**Dataset not loading in trainer**:
- Verify `enabled: true` in config
- Check paths are relative to `Data/` directory
- Run test scripts to verify COCO JSON is valid

## Next Steps

1. ✅ DAWN and ACDC are ready - enable in config
2. ⏳ For STF: Extract images + download gt_labels
3. 🧪 Test with training pipeline
4. 📊 Compare performance across all datasets
