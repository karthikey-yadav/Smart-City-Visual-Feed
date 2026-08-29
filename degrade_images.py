"""
degrade_images.py

Takes clean images from an input folder and generates degraded versions
(blur, underexposure, overexposure, noise, corruption) with known
ground-truth labels. Used to build the smart-city demo/sample-image set
required by the assessment submission.

Usage:
    python degrade_images.py --input data/demo_clean --output data/demo_degraded
"""

import os
import cv2
import numpy as np
import argparse
import json
import random

random.seed(42)
np.random.seed(42)


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def apply_blur(img, severity):
    # severity: 1 (mild) - 3 (severe)
    kernel_sizes = {1: 5, 2: 11, 3: 21}
    k = kernel_sizes[severity]
    return cv2.GaussianBlur(img, (k, k), 0)


def apply_underexposure(img, severity):
    factors = {1: 0.6, 2: 0.35, 3: 0.15}
    return np.clip(img.astype(np.float32) * factors[severity], 0, 255).astype(np.uint8)


def apply_overexposure(img, severity):
    factors = {1: 1.5, 2: 2.2, 3: 3.0}
    return np.clip(img.astype(np.float32) * factors[severity], 0, 255).astype(np.uint8)


def apply_noise(img, severity):
    sigma_map = {1: 10, 2: 25, 3: 45}
    sigma = sigma_map[severity]
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def apply_corruption(img, severity):
    # Simulate corruption via aggressive JPEG re-compression + random block dropout
    quality_map = {1: 25, 2: 10, 3: 3}
    quality = quality_map[severity]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    decoded = cv2.imdecode(encimg, cv2.IMREAD_COLOR)

    if severity >= 2:
        # add random dropped-out blocks to simulate sensor/transmission corruption
        h, w = decoded.shape[:2]
        n_blocks = 3 if severity == 2 else 8
        for _ in range(n_blocks):
            bw, bh = random.randint(w // 20, w // 8), random.randint(h // 20, h // 8)
            x, y = random.randint(0, w - bw), random.randint(0, h - bh)
            decoded[y:y + bh, x:x + bw] = np.random.randint(0, 255, (bh, bw, 3), dtype=np.uint8)
    return decoded


DEGRADATIONS = {
    "blur": apply_blur,
    "underexposure": apply_underexposure,
    "overexposure": apply_overexposure,
    "noise": apply_noise,
    "corruption": apply_corruption,
}

# Maps degradation type -> which quality_label it should produce at each severity
LABEL_MAP = {
    1: "DEGRADED",
    2: "DEGRADED",
    3: "DEFECTIVE",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder of clean images")
    parser.add_argument("--output", required=True, help="Folder to save degraded images + labels")
    parser.add_argument("--severities", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    files = [f for f in os.listdir(args.input) if f.lower().endswith(valid_ext)]

    if not files:
        print(f"No images found in {args.input}")
        return

    manifest = []

    # Also copy a few originals through untouched, labeled ACCEPTABLE
    for i, fname in enumerate(files):
        in_path = os.path.join(args.input, fname)
        img = load_image(in_path)
        base_name = os.path.splitext(fname)[0]

        # 1 in 4 images kept clean -> ACCEPTABLE ground truth
        if i % 4 == 0:
            out_name = f"{base_name}_clean.jpg"
            out_path = os.path.join(args.output, out_name)
            cv2.imwrite(out_path, img)
            manifest.append({
                "filename": out_name,
                "source": fname,
                "degradation_type": "none",
                "severity": 0,
                "quality_label": "ACCEPTABLE"
            })
            continue

        # otherwise apply one random degradation type at a random severity
        deg_type = random.choice(list(DEGRADATIONS.keys()))
        severity = random.choice(args.severities)
        degraded = DEGRADATIONS[deg_type](img, severity)

        out_name = f"{base_name}_{deg_type}_sev{severity}.jpg"
        out_path = os.path.join(args.output, out_name)
        cv2.imwrite(out_path, degraded)

        manifest.append({
            "filename": out_name,
            "source": fname,
            "degradation_type": deg_type,
            "severity": severity,
            "quality_label": LABEL_MAP[severity]
        })

    manifest_path = os.path.join(args.output, "labels.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated {len(manifest)} images -> {args.output}")
    print(f"Labels written to {manifest_path}")


if __name__ == "__main__":
    main()
