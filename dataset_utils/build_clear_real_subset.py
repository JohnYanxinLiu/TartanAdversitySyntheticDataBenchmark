"""Build the real-data training/validation directories from a BDD100K download.

Produces the two directory layouts configs/base.yaml expects:

  Data/ClearRealSubset/
    images/                     clear-weather training images
    combined_dataset_coco.json  COCO annotations (weather tags preserved)
    labels/                     YOLO labels (0-indexed classes)
  Data/bdd100k_val/
    images/ + bdd100k_val_coco.json + labels/   full-weather validation set

Expects the layout produced by download_bdd100k.py: Supervisely split dirs
(with ann/ + img/) under --bdd_dir plus the converted COCO JSONs
(bdd100k_train_coco.json / bdd100k_val_coco.json) alongside them.

The full experiment used a 28,800-image clear subset; pass --max_train /
--max_val to build a smaller one for a quick run (0 = keep everything):

    python dataset_utils/build_clear_real_subset.py --max_train 28800
"""
import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
from coco_to_yolo_labels import convert_coco_to_yolo
from download_bdd100k import find_split_dirs


def subset_coco(coco, keep_weathers=None, max_images=0, seed=42):
    """Filter a COCO dict by weather and cap the image count (0 = no cap)."""
    images = coco["images"]
    if keep_weathers is not None:
        images = [i for i in images if i.get("weather") in keep_weathers]
    if max_images and len(images) > max_images:
        rng = random.Random(seed)
        images = rng.sample(images, max_images)
    kept_ids = {i["id"] for i in images}
    annotations = [a for a in coco["annotations"] if a["image_id"] in kept_ids]
    return {"images": images, "annotations": annotations,
            "categories": coco["categories"]}


def build_split(split_img_dir, coco_json_path, out_dir, out_json_name,
                keep_weathers, max_images, seed):
    with open(coco_json_path) as f:
        coco = json.load(f)
    sub = subset_coco(coco, keep_weathers, max_images, seed)
    weather_counts = {}
    for img in sub["images"]:
        weather_counts[img.get("weather")] = weather_counts.get(img.get("weather"), 0) + 1
    print(f"\n{out_dir}: {len(sub['images'])} images, "
          f"{len(sub['annotations'])} annotations")
    print(f"  weather: {dict(sorted(weather_counts.items()))}")

    images_out = os.path.join(out_dir, "images")
    os.makedirs(images_out, exist_ok=True)
    missing = 0
    for img in sub["images"]:
        src = os.path.join(split_img_dir, img["file_name"])
        dst = os.path.join(images_out, img["file_name"])
        if os.path.exists(src):
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        else:
            missing += 1
    if missing:
        print(f"  [WARNING] {missing} images listed in COCO were missing on disk")

    out_json = os.path.join(out_dir, out_json_name)
    with open(out_json, "w") as f:
        json.dump(sub, f)
    convert_coco_to_yolo(out_json, os.path.join(out_dir, "labels"))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo_root = script_dir.parent
    parser.add_argument("--bdd_dir", default=str(repo_root / "Data" / "BDD100K"),
                        help="BDD100K download dir (from download_bdd100k.py)")
    parser.add_argument("--data_dir", default=str(repo_root / "Data"),
                        help="Output Data directory")
    parser.add_argument("--max_train", type=int, default=0,
                        help="Cap on clear training images (0 = all; experiment used 28800)")
    parser.add_argument("--max_val", type=int, default=0,
                        help="Cap on validation images (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    splits = find_split_dirs(args.bdd_dir)
    for split in ("train", "val"):
        if split not in splits:
            sys.exit(f"Split dir '{split}' (with ann/ + img/) not found under {args.bdd_dir}")
        coco_json = os.path.join(args.bdd_dir, f"bdd100k_{split}_coco.json")
        if not os.path.exists(coco_json):
            sys.exit(f"{coco_json} not found - run download_bdd100k.py's conversion first")

    build_split(os.path.join(splits["train"], "img"),
                os.path.join(args.bdd_dir, "bdd100k_train_coco.json"),
                os.path.join(args.data_dir, "ClearRealSubset"),
                "combined_dataset_coco.json",
                keep_weathers={"clear"}, max_images=args.max_train, seed=args.seed)

    build_split(os.path.join(splits["val"], "img"),
                os.path.join(args.bdd_dir, "bdd100k_val_coco.json"),
                os.path.join(args.data_dir, "bdd100k_val"),
                "bdd100k_val_coco.json",
                keep_weathers=None, max_images=args.max_val, seed=args.seed)

    print("\nDone. configs/base.yaml paths now resolve against", args.data_dir)


if __name__ == "__main__":
    main()
