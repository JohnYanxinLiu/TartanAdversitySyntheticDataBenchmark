"""
Download the BDD100K "Images 100K" dataset via DatasetNinja (dataset_tools).

Reference: https://datasetninja.com/bdd100k#download

DatasetNinja delivers BDD100K in **Supervisely format** (per-split ``ann/`` +
``img/`` folders, image-level ``weather`` tags, and ``points.exterior`` boxes),
which is exactly what ``convert_bdd100k_to_coco.py`` consumes. So after
downloading, this script can optionally run that conversion to produce the
COCO-format JSONs (``bdd100k_train_coco.json`` / ``bdd100k_val_coco.json``) that
the training/eval pipeline expects.

Usage:
    pip install --upgrade dataset-tools

    # Download to Data/BDD100K and convert train + val to COCO:
    python dataset_utils/download_bdd100k.py

    # Download only (skip COCO conversion):
    python dataset_utils/download_bdd100k.py --no_convert

    # Convert an already-downloaded copy without re-downloading:
    python dataset_utils/download_bdd100k.py --skip_download
"""
import argparse
import os
import sys
from pathlib import Path

# DatasetNinja's exact dataset name for the 100K image split.
DATASET_NAME = "BDD100K: Images 100K"

# Splits the pipeline uses; the rest (e.g. "test") are downloaded but not converted.
SPLITS_TO_CONVERT = ("train", "val")


def download_bdd100k(dst_dir):
    """Download BDD100K (Images 100K) into ``dst_dir`` via dataset_tools."""
    try:
        import dataset_tools as dtools
    except ImportError:
        sys.exit(
            "The 'dataset_tools' package is required for the download.\n"
            "Install it with:\n"
            "    pip install --upgrade dataset-tools\n"
            "See https://datasetninja.com/bdd100k#download"
        )

    os.makedirs(dst_dir, exist_ok=True)
    print("=" * 60)
    print(f"Downloading '{DATASET_NAME}' (~5.4 GB) via DatasetNinja")
    print(f"Destination: {dst_dir}")
    print("=" * 60)
    dtools.download(dataset=DATASET_NAME, dst_dir=str(dst_dir))


def find_split_dirs(root):
    """Find Supervisely split dirs under ``root`` (those with both ann/ and img/).

    Returns a dict mapping split name (the folder's basename, e.g. "train") to
    its absolute path.
    """
    splits = {}
    for dirpath, dirnames, _ in os.walk(root):
        if "ann" in dirnames and "img" in dirnames:
            splits[os.path.basename(dirpath)] = dirpath
    return splits


def print_next_steps(dst_dir, splits, converted):
    print("\n" + "=" * 60)
    print("BDD100K download complete.")
    print("=" * 60)
    print(f"Location: {dst_dir}")
    print(f"Detected splits: {', '.join(sorted(splits)) or 'none'}")
    if converted:
        print("\nGenerated COCO annotations:")
        for split, path in converted.items():
            print(f"  {split}: {path}")
    print("\nWire it into configs/base.yaml, e.g.:")
    print("  paths:")
    print("    val:")
    print(f"      images: {os.path.relpath(os.path.join(splits.get('val', 'BDD100K/val'), 'img'), dst_dir.parent)}")
    print("      annotations: BDD100K/bdd100k_val_coco.json")
    print("\nNote: the pipeline trains on a curated *clear* subset (ClearRealSubset).")
    print("Filter the converted train COCO to weather == 'clear' to build it.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    script_dir = Path(__file__).parent
    default_dst = (script_dir.parent / "Data" / "BDD100K").resolve()
    parser.add_argument(
        "--dst_dir", type=str, default=str(default_dst),
        help=f"Download destination directory (default: {default_dst})"
    )
    parser.add_argument(
        "--no_convert", action="store_true",
        help="Download only; skip COCO conversion of the train/val splits"
    )
    parser.add_argument(
        "--skip_download", action="store_true",
        help="Skip the download and only convert an existing copy in --dst_dir"
    )
    args = parser.parse_args()

    dst_dir = Path(args.dst_dir)

    if not args.skip_download:
        download_bdd100k(dst_dir)
    else:
        print(f"[INFO] Skipping download; using existing data in {dst_dir}")

    splits = find_split_dirs(dst_dir)
    if not splits:
        print(f"\n[WARNING] No Supervisely split dirs (with ann/ + img/) found under {dst_dir}.")
        print("          Inspect the download layout and run convert_bdd100k_to_coco.py manually.")
        return

    converted = {}
    if not args.no_convert:
        # The converter lives alongside this script.
        sys.path.insert(0, str(script_dir))
        from convert_bdd100k_to_coco import convert_bdd100k_to_coco

        for split in SPLITS_TO_CONVERT:
            if split not in splits:
                print(f"[INFO] Split '{split}' not found in download; skipping conversion.")
                continue
            out_path = dst_dir / f"bdd100k_{split}_coco.json"
            print(f"\nConverting '{split}' split -> {out_path}")
            convert_bdd100k_to_coco(splits[split], str(out_path))
            converted[split] = str(out_path)
    else:
        print("[INFO] --no_convert set; skipping COCO conversion.")

    print_next_steps(dst_dir, splits, converted)


if __name__ == "__main__":
    main()
