# TartanAdversity — Synthetic Data Learnability Benchmark

Code for benchmarking the **learnability of synthetically produced data** for object
detection under adverse weather. We train three detectors — a Faster R-CNN with a
**ResNet** backbone, a Faster R-CNN with a **ConvNeXt** backbone, and **YOLO** — on
mixtures of real ([BDD100K](https://bdd-data.berkeley.edu/)) and synthetically
augmented images, then evaluate generalization on real adverse-weather datasets.

Synthetic augmentations come from two sources:

- **Automold** — classical, procedurally generated weather effects.
- **Gemini / TartanAdversity** — the fully synthetically produced augmented data
  (released separately on Hugging Face).

## Weather conditions

All three models report **the same canonical per-weather metrics** so results are
directly comparable. Every dataset's raw weather labels are collapsed to a single
set (see [`weather_utils.py`](weather_utils.py)):

| Canonical | Raw labels folded in                     |
|-----------|------------------------------------------|
| `clear`   | clear, sunny                             |
| `rainy`   | rain, rainy, rain_storm                  |
| `foggy`   | fog, foggy, **haze**, **mist**           |
| `snowy`   | snow, snowy, snow_storm                  |

Everything else (overcast, partly cloudy, undefined, night, sand, ...) is excluded
from the per-weather breakdown but still contributes to the **overall** metric.
Metrics are logged as `{split}/mAP` (overall) and `{split}/{condition}/mAP` for each
canonical condition, e.g. `dawn/foggy/mAP`, `val/rainy/mAP`.

## Setup

Requires Python 3.10+ and a CUDA-capable GPU for training. From the repo root:

```bash
python -m venv .venv && source .venv/bin/activate   # or conda
pip install -r requirements.txt

# Configure credentials (Weights & Biases) — never committed
cp .env.example .env      # then edit .env with your values
source .env
```

Weights & Biases entity/project are read from `WANDB_ENTITY` / `WANDB_PROJECT`
(falling back to `configs/base.yaml`), and the API key from `WANDB_API_KEY`.
Training runs without W&B if no key is provided.

## Reproduction

Datasets are **not** distributed with the code — download and preprocess them
into a local `Data/` directory (git-ignored). All converters emit COCO-format
annotations with a canonical `weather` field and a shared 10-class, 1-indexed
taxonomy (`bike…truck`), plus 0-indexed YOLO `.txt` labels. Paths are wired in
`configs/base.yaml`. See `dataset_utils/PREPROCESSING_GUIDE.md` for details.

### Quick start

```bash
./setup.sh          # creates .venv, installs requirements, builds every dataset
```

`setup.sh` creates the environment, checks for ACDC first (it's the one dataset
that needs a manual, registration-gated download — see step 5), then downloads
and preprocesses everything else. It's idempotent, so after dropping the ACDC
archives in place you can just re-run it. Use `./setup.sh --max-train 800` for a
smaller quick-run subset, or `--skip-env` to reuse the current environment.

The manual per-dataset steps below are equivalent, if you'd rather run them
individually (each writes into `Data/` under the name `configs/base.yaml`
expects).

**1. BDD100K — real training + in-distribution validation**

```bash
pip install dataset-tools                          # DatasetNinja downloader
python dataset_utils/download_bdd100k.py           # ~5.4 GB; downloads + converts train/val to COCO
python dataset_utils/build_clear_real_subset.py --max_train 28800
#   -> Data/ClearRealSubset/ (clear-weather train) and Data/bdd100k_val/ (full-weather val)
#   Omit --max_train for all clear images, or pass a smaller value for a quick run.
```

**2. TartanAdversity — synthetic training data (Gemini + Automold)**

```bash
python dataset_utils/download_tartanadversity.py
#   -> Data/GeminiAugmented/ and Data/AutomoldAugmented/ (images, labels, per-weather COCO)
#   Quick check without pulling everything: --limit 20
```

This pulls [`JohnYanxinLiu/TartanAdversity`](https://huggingface.co/datasets/JohnYanxinLiu/TartanAdversity)
from Hugging Face and writes the layout `configs/base.yaml` expects — images,
labels, per-weather COCO JSONs, and the per-image fidelity **ranking CSVs**
(`{Gemini,Automold}Rankings/`) used for the top-X% quality pruning. The rankings
live as loose files in the dataset repo (they are not part of the parquet) and
are fetched automatically.

**3. DAWN — real adverse weather (foggy/rainy/snowy)**

```bash
python dataset_utils/download_dawn.py              # downloads + unpacks nested per-weather zips
python dataset_utils/convert_dawn_to_coco.py
```

**4. Foggy Zurich — real fog**

```bash
python dataset_utils/download_foggy_zurich.py
python dataset_utils/preprocess_foggy_zurich.py
```

**5. ACDC — real adverse weather (registration required)**

ACDC is license-gated, so it can't be scripted. Create an account at
<https://acdc.vision.ee.ethz.ch/>, download **`rgb_anon_trainvaltest.zip`** and
**`gt_detection_trainval.zip`**, then place them in
`dataset_utils/acdc_dataset_zip_files/` (leave the RGB archive zipped; extract
the GT archive so a `gt_detection/` folder sits alongside it):

```bash
dataset_utils/acdc_dataset_zip_files/
├── rgb_anon_trainvaltest.zip
└── gt_detection/            # from gt_detection_trainval.zip
python dataset_utils/preprocess_acdc.py            # extracts fog/rain/snow, builds train + val COCO
```

Validation datasets, once prepared: **BDD100K val** (in-distribution), **DAWN**,
**ACDC** (train + val), and **Foggy Zurich**. Enable/disable any of them via
`paths.val_datasets.*.enabled` in `configs/base.yaml`.

## Running experiments

ResNet / ConvNeXt (Faster R-CNN):

```bash
python train.py --model_type resnet   --replacement_percentage 0.2 --augmentation_type gemini
python train.py --model_type convnext --addition_percentage    0.2 --augmentation_type automold
```

YOLO:

```bash
python train_yolo.py --replacement_percentage 0.2 --augmentation_type gemini
```

`--replacement_percentage` and `--addition_percentage` are mutually exclusive and
select the mixing method (replace real images vs. add synthetic ones). Model
hyperparameters live in `configs/{resnet,convnext,yolo}.yaml`.

## Tests

The scripts in [`tests/`](tests/) are integration checks that exercise dataset
loading and per-weather evaluation against a populated `Data/` directory. Run them
individually, e.g.:

```bash
python tests/test_val_datasets_config.py
python tests/test_dawn_dataset.py
```

## Repository layout

```
train.py                 # ResNet/ConvNeXt entry point
train_yolo.py            # YOLO entry point
trainer.py               # Faster R-CNN training/eval loop
dataset.py               # Mixed real+synthetic dataset
data_utils.py            # Synthetic data ranking/mixing
config_manager.py        # Config loading + W&B target resolution
weather_utils.py         # Canonical weather normalization (shared)
configs/                 # base + per-model YAML configs
yolo_utils/              # YOLO dataset building + shared COCO eval
dataset_utils/           # Dataset download + COCO/YOLO conversion
tests/                   # Integration tests
```

## License

Code in this repository is released under the **Apache License 2.0** (see
[`LICENSE`](LICENSE)). The **TartanAdversity dataset** is released separately
under **CC-BY-4.0** (see `TartanAdversity_HF_datacard.md`). Third-party datasets
used for evaluation (BDD100K, DAWN, ACDC, Foggy Zurich) are **not** redistributed
here and remain under their own respective licenses — download them from their
original sources and observe their terms.
