"""Download the TartanAdversity synthetic dataset from the Hugging Face Hub and
materialize it into the on-disk layout the training pipeline expects.

The HF dataset (https://huggingface.co/datasets/JohnYanxinLiu/TartanAdversity) is
a single parquet `train` split where each row carries the augmented image, its
`source` (gemini/automold), `weather`, and both YOLO and COCO annotations. This
script reverses `TartanAdversity/upload_dataset.py`, writing:

    Data/GeminiAugmented/
      images/                      <weather>-<bdd_name>.jpg
      labels/                      <weather>-<bdd_name>.txt   (0-indexed YOLO)
      GeminiFogCoco.json           per-weather COCO (1-indexed, canonical weather)
      GeminiRainCoco.json
      GeminiSnowCoco.json
    Data/AutomoldAugmented/  ... (same, "Automold" prefix)

matching the `paths.synthetic` entries in `configs/base.yaml`.

RANKINGS: the quality/fidelity scores used for the top-X% pruning
(`{Gemini,Automold}Rankings/{fog,rain,snow}.csv`, columns
`base_image_name,metric_score`) are NOT part of the HF parquet. They are required
for any experiment with a mix rate > 0. This script downloads them from the HF
repo if the author uploaded them there as loose files; otherwise it prints
instructions. Without rankings, only the 0%-synthetic baseline can run.

Usage:
    pip install datasets huggingface_hub pillow
    python dataset_utils/download_tartanadversity.py
    # quick check without pulling everything (streams N rows):
    python dataset_utils/download_tartanadversity.py --limit 20

If `load_dataset` stalls on the Xet transfer client, set HF_HUB_DISABLE_XET=1.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from itertools import islice
from pathlib import Path

# source label (in HF `source` field) -> (Data/ base dir, file prefix for json/rankings)
SOURCES = {
    "gemini": ("GeminiAugmented", "Gemini"),
    "automold": ("AutomoldAugmented", "Automold"),
}

# canonical weather -> (COCO json title, ranking csv stem)
WEATHER = {
    "foggy": ("Fog", "fog"),
    "rainy": ("Rain", "rain"),
    "snowy": ("Snow", "snow"),
}

CATEGORIES = [
    {"id": 1, "name": "bike"}, {"id": 2, "name": "bus"}, {"id": 3, "name": "car"},
    {"id": 4, "name": "motor"}, {"id": 5, "name": "person"}, {"id": 6, "name": "rider"},
    {"id": 7, "name": "traffic light"}, {"id": 8, "name": "traffic sign"},
    {"id": 9, "name": "train"}, {"id": 10, "name": "truck"},
]


def weather_key(raw):
    """Map an HF `weather` value (canonical or raw) to a WEATHER key, or None."""
    w = (raw or "").strip().lower()
    if w in ("foggy", "fog"):
        return "foggy"
    if w in ("rainy", "rain"):
        return "rainy"
    if w in ("snowy", "snow"):
        return "snowy"
    return None


def write_yolo_label(path, yolo_annotations):
    """Write a 0-indexed YOLO .txt from the row's parallel-list yolo_annotations."""
    class_ids = yolo_annotations.get("class_id", [])
    bboxes = yolo_annotations.get("bbox", [])
    lines = []
    for cls, bbox in zip(class_ids, bboxes):
        xc, yc, w, h = bbox
        lines.append(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def fetch_rankings(repo_id, base_dir, prefix, data_dir):
    """Ensure {prefix}Rankings/{fog,rain,snow}.csv exist under Data/{base_dir}/.

    Uses local copies if present; otherwise tries to pull them from the HF repo
    as loose files. Returns the list of still-missing csv stems.
    """
    rank_dir = data_dir / base_dir / f"{prefix}Rankings"
    rank_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for _title, stem in WEATHER.values():
        dst = rank_dir / f"{stem}.csv"
        if dst.exists():
            continue
        try:
            from huggingface_hub import hf_hub_download
            src = hf_hub_download(
                repo_id=repo_id, repo_type="dataset",
                filename=f"{prefix}Rankings/{stem}.csv",
            )
            dst.write_bytes(Path(src).read_bytes())
            print(f"  [rankings] fetched {prefix}Rankings/{stem}.csv from HF")
        except Exception:
            missing.append(stem)
    return missing


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--repo-id", default="JohnYanxinLiu/TartanAdversity")
    parser.add_argument("--data-dir", default=str(repo_root / "Data"),
                        help="Output Data/ directory (default: <repo>/Data)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only process the first N rows (streams them); 0 = full dataset")
    args = parser.parse_args()

    from datasets import load_dataset

    data_dir = Path(args.data_dir)
    for base_dir, _prefix in SOURCES.values():
        (data_dir / base_dir / "images").mkdir(parents=True, exist_ok=True)
        (data_dir / base_dir / "labels").mkdir(parents=True, exist_ok=True)

    streaming = args.limit > 0
    print(f"Loading {args.repo_id} (streaming={streaming})...")
    ds = load_dataset(args.repo_id, split="train", streaming=streaming)
    rows = islice(ds, args.limit) if streaming else ds

    # Accumulate COCO metadata per (source, weather); images are written as we go.
    coco = defaultdict(lambda: {"images": [], "annotations": [],
                                "categories": CATEGORIES,
                                "info": {"description": "TartanAdversity synthetic"},
                                "licenses": []})
    next_img_id = defaultdict(lambda: 1)
    next_ann_id = defaultdict(lambda: 1)  # ann ids start at 1 (COCOeval treats 0 as unmatched)
    counts = defaultdict(int)

    for i, row in enumerate(rows):
        src = row["source"]
        if src not in SOURCES:
            continue
        base_dir, prefix = SOURCES[src]
        fn = row["file_name"]                      # already weather-prefixed, e.g. "fog-<bdd>.jpg"
        img = row["image"]                         # PIL.Image
        width, height = img.size

        img.save(data_dir / base_dir / "images" / fn)
        write_yolo_label(data_dir / base_dir / "labels" / f"{Path(fn).stem}.txt",
                         row["yolo_annotations"])

        wk = weather_key(row["weather"])
        if wk is not None:
            key = (src, wk)
            img_id = next_img_id[key]; next_img_id[key] += 1
            coco[key]["images"].append({
                "id": img_id, "file_name": fn, "width": width, "height": height,
                "weather": wk,
            })
            ca = row["coco_annotations"]
            for cat, bbox, area, iscrowd in zip(
                    ca.get("category_id", []), ca.get("bbox", []),
                    ca.get("area", []), ca.get("iscrowd", [])):
                ann_id = next_ann_id[key]; next_ann_id[key] += 1
                coco[key]["annotations"].append({
                    "id": ann_id, "image_id": img_id, "category_id": int(cat),
                    "bbox": [float(v) for v in bbox], "area": float(area),
                    "iscrowd": int(iscrowd), "segmentation": [],
                })
        counts[src] += 1
        if (i + 1) % 2000 == 0:
            print(f"  ...{i + 1} rows")

    # Write per-weather COCO JSONs.
    for (src, wk), payload in coco.items():
        base_dir, prefix = SOURCES[src]
        title, _stem = WEATHER[wk]
        out = data_dir / base_dir / f"{prefix}{title}Coco.json"
        out.write_text(json.dumps(payload))
        print(f"  wrote {out.relative_to(data_dir)}: "
              f"{len(payload['images'])} images, {len(payload['annotations'])} anns")

    print(f"\nMaterialized: " + ", ".join(f"{src}={n}" for src, n in counts.items()))

    # Rankings (required for mix rate > 0; not part of the HF parquet).
    all_missing = {}
    for src, (base_dir, prefix) in SOURCES.items():
        missing = fetch_rankings(args.repo_id, base_dir, prefix, data_dir)
        if missing:
            all_missing[prefix] = missing
    if all_missing:
        print("\n" + "=" * 72)
        print("[WARNING] Ranking CSVs are missing and could not be fetched from the")
        print("HF repo. They hold the fidelity scores used for top-X% pruning and are")
        print("NOT included in the HF parquet. Without them, only a 0%-synthetic")
        print("(pure-real) baseline can run; any --replacement/--addition > 0 needs them.")
        for prefix, stems in all_missing.items():
            for stem in stems:
                print(f"    place: Data/{prefix}Augmented/{prefix}Rankings/{stem}.csv"
                      f"  (columns: base_image_name,metric_score)")
        print("=" * 72)
    else:
        print("Rankings present for all sources.")

    print("\nDone. Synthetic data is under", data_dir)


if __name__ == "__main__":
    main()
    # All output files are written and flushed above. Hard-exit to sidestep a
    # benign segfault in the HF `datasets` streaming worker's interpreter-teardown
    # (only triggers on the --limit streaming path; harmless here since work is done).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
