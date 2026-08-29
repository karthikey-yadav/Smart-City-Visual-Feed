"""
organize_kadid.py

Reads KADID-10k's dmos.csv, maps its 25 distortion types onto our
5 target categories (blur, noise, underexposure, overexposure, corruption),
assigns a quality_label from the DMOS score, and splits everything into
train/val/test sets.

Expected input layout:
    kadid10k/kadid10k/images/*.png
    kadid10k/kadid10k/dmos.csv        (columns: dist_img,dmos,var)

Filename format: Ixx_yy_zz.png
    xx = reference image id
    yy = distortion type (1-25)
    zz = distortion level (1-5)

Usage:
    python organize_kadid.py --root kadid10k/kadid10k --output data/kadid_split
"""

import os
import csv
import json
import random
import shutil
import argparse

random.seed(42)

# KADID-10k distortion type -> our 5 categories
# (types not listed below are grouped into "other" and excluded from
#  the 5-class training set, but kept in the manifest for reference)
DISTORTION_MAP = {
    1: "blur",           # Gaussian blur
    2: "blur",           # Lens blur
    3: "blur",           # Motion blur
    11: "noise",         # White noise
    12: "noise",         # White noise in color component
    13: "noise",         # Impulse noise
    14: "noise",         # Multiplicative noise
    16: "overexposure",  # Brighten
    17: "underexposure", # Darken
    9: "corruption",     # JPEG2000 compression
    10: "corruption",    # JPEG compression
    21: "corruption",    # Pixelate
    23: "corruption",    # Color block
}

TARGET_CATEGORIES = {"blur", "noise", "underexposure", "overexposure", "corruption"}


def dmos_to_label(dmos):
    # Per KADID docs: higher DMOS = higher visual quality (scale ~1-5)
    if dmos >= 4.0:
        return "ACCEPTABLE"
    elif dmos >= 2.5:
        return "DEGRADED"
    else:
        return "DEFECTIVE"


def parse_filename(fname):
    # e.g. I01_01_01.png -> ref=1, dist_type=1, level=1
    name = os.path.splitext(fname)[0]
    parts = name.split("_")
    if len(parts) != 3:
        return None
    try:
        ref_id = int(parts[0].lstrip("Ii"))
        dist_type = int(parts[1])
        level = int(parts[2])
        return ref_id, dist_type, level
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Path to kadid10k/kadid10k folder")
    parser.add_argument("--output", required=True, help="Where to write split folders + manifest")
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--keep_other", action="store_true",
                         help="Also keep distortion types outside our 5 categories")
    args = parser.parse_args()

    images_dir = os.path.join(args.root, "images")
    dmos_path = os.path.join(args.root, "dmos.csv")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Could not find images folder at {images_dir}")
    if not os.path.isfile(dmos_path):
        raise FileNotFoundError(f"Could not find dmos.csv at {dmos_path}")

    records = []
    with open(dmos_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("dist_img") or row.get("dist_img ") or list(row.values())[0]
            fname = fname.strip()
            try:
                dmos = float(row.get("dmos", list(row.values())[1]))
            except (ValueError, IndexError):
                continue

            parsed = parse_filename(fname)
            if parsed is None:
                continue
            ref_id, dist_type, level = parsed
            category = DISTORTION_MAP.get(dist_type, "other")

            if category == "other" and not args.keep_other:
                continue

            records.append({
                "filename": fname,
                "ref_id": ref_id,
                "distortion_type_id": dist_type,
                "category": category,
                "level": level,
                "dmos": dmos,
                "quality_label": dmos_to_label(dmos),
            })

    if not records:
        raise RuntimeError("No records parsed — check dmos.csv column names / image naming.")

    print(f"Parsed {len(records)} usable records "
          f"(categories: {sorted(TARGET_CATEGORIES)})")

    # Split by reference image id so the same scene doesn't leak across splits
    ref_ids = sorted(set(r["ref_id"] for r in records))
    random.shuffle(ref_ids)
    n = len(ref_ids)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)

    train_refs = set(ref_ids[:n_train])
    val_refs = set(ref_ids[n_train:n_train + n_val])
    test_refs = set(ref_ids[n_train + n_val:])

    def split_of(ref_id):
        if ref_id in train_refs:
            return "train"
        elif ref_id in val_refs:
            return "val"
        return "test"

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(args.output, split), exist_ok=True)

    manifest = {"train": [], "val": [], "test": []}

    for r in records:
        split = split_of(r["ref_id"])
        src = os.path.join(images_dir, r["filename"])
        if not os.path.isfile(src):
            continue
        dst = os.path.join(args.output, split, r["filename"])
        shutil.copyfile(src, dst)
        manifest[split].append(r)

    for split in manifest:
        out_path = os.path.join(args.output, f"{split}_labels.json")
        with open(out_path, "w") as f:
            json.dump(manifest[split], f, indent=2)
        print(f"{split}: {len(manifest[split])} images -> {out_path}")


if __name__ == "__main__":
    main()
