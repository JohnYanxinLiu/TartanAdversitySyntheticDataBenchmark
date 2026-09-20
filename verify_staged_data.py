"""Verify the staged Data/ against the old repo's Data/.

Confirms the regenerated COCO annotations differ from the originals only in the
ways the converter fixes intend (1-indexed categories where they were wrong,
annotation_id starting at 1, canonical weather strings) and that the box
geometry, image sets, and class assignments are otherwise untouched.

Usage:  python verify_staged_data.py [/path/to/GenDataTraining]
"""
import json
import os
import sys
from collections import Counter

REPO = os.path.dirname(os.path.abspath(__file__))
OLD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(REPO), "GenDataTraining")

DATASETS = {
    "dawn": "DAWN/dawn_coco.json",
    "foggy_zurich": "FoggyZurich/foggy_zurich_coco.json",
    "acdc_train": "ACDC/acdc_train_coco.json",
    "acdc_val": "ACDC/acdc_val_coco.json",
    "bdd100k_val": "bdd100k_val/bdd100k_val_coco.json",
}

CANONICAL = {"clear", "rainy", "foggy", "snowy"}
ok = True


def box_key(a, imgs):
    """Identify an annotation by image file + geometry, independent of its id."""
    return (imgs[a["image_id"]], tuple(round(float(v), 3) for v in a["bbox"]), a["category_id"])


for name, rel in DATASETS.items():
    new_p = os.path.join(REPO, "Data", rel)
    old_p = os.path.join(OLD, "Data", rel)
    print(f"\n=== {name} ===")
    if not os.path.exists(new_p):
        print(f"  MISSING staged file: {new_p}")
        ok = False
        continue
    new = json.load(open(new_p))
    old = json.load(open(old_p)) if os.path.exists(old_p) else None

    cats = {c["id"]: c["name"] for c in new["categories"]}
    ann_cats = sorted({a["category_id"] for a in new["annotations"]})
    min_id = min(a["id"] for a in new["annotations"])
    weather = Counter(i.get("weather") for i in new["images"])

    print(f"  images={len(new['images'])}  annotations={len(new['annotations'])}  min_ann_id={min_id}")
    print(f"  category ids={sorted(cats)}  used by annotations={ann_cats}")
    print(f"  weather={dict(weather)}")

    # Gate 1: declared categories must cover every category actually used.
    missing = [c for c in ann_cats if c not in cats]
    if missing:
        print(f"  FAIL: annotations use undeclared categories {missing}")
        ok = False
    else:
        print("  PASS: every used category is declared")

    # Gate 2: categories 1-indexed, matching configs/base.yaml.
    if sorted(cats) != list(range(1, 11)):
        print(f"  FAIL: categories are not 1..10")
        ok = False
    else:
        print("  PASS: categories are 1-indexed 1..10")

    # Gate 3: no annotation id 0 (COCOeval treats 0 as 'unmatched').
    if min_id < 1:
        print(f"  FAIL: annotation ids start at {min_id}")
        ok = False
    else:
        print("  PASS: annotation ids start at >= 1")

    # Gate 4: weather labels are canonical (or absent for non-weather sets).
    noncanon = {w for w in weather if w not in CANONICAL and w is not None}
    if noncanon and name != "bdd100k_val":
        print(f"  FAIL: non-canonical weather labels {noncanon}")
        ok = False
    elif name != "bdd100k_val":
        print("  PASS: weather labels are canonical")

    # Gate 5: the actual boxes must be unchanged vs the old file.
    if old is not None:
        new_imgs = {i["id"]: i["file_name"] for i in new["images"]}
        old_imgs = {i["id"]: i["file_name"] for i in old["images"]}
        if {i["file_name"] for i in new["images"]} != {i["file_name"] for i in old["images"]}:
            print("  FAIL: image set changed")
            ok = False
        else:
            print("  PASS: identical image set")
        new_boxes = Counter(box_key(a, new_imgs) for a in new["annotations"])
        old_boxes = Counter(box_key(a, old_imgs) for a in old["annotations"])
        if new_boxes == old_boxes:
            print("  PASS: identical boxes + class assignments")
        else:
            only_new = sum((new_boxes - old_boxes).values())
            only_old = sum((old_boxes - new_boxes).values())
            print(f"  FAIL: box sets differ (+{only_new} / -{only_old})")
            ok = False

        # Report the intended weather remapping for visibility.
        ow = Counter(i.get("weather") for i in old["images"])
        if dict(ow) != dict(weather):
            print(f"  note: weather relabelled  {dict(ow)} -> {dict(weather)}")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
