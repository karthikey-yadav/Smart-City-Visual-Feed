"""
feature_extractor.py

Classical, explainable computer-vision features for image quality
assessment. No ML/DL here — pure signal-processing measures that
directly correspond to the required detection capabilities:
blur, under/over-exposure, noise, and general degradation (contrast).

Can be used as:
  1. An importable module (call `extract_features(image)` from the
     backend or the CNN training pipeline for a hybrid approach).
  2. A CLI tool to batch-extract features from a folder of images
     into a CSV, useful for sanity-checking your dataset or feeding
     a classical-ML baseline model.

Usage (CLI):
    python feature_extractor.py --input data/demo_degraded --output features.csv
"""

import os
import cv2
import numpy as np
import argparse
import csv


def blur_score(gray):
    """
    Variance of the Laplacian. Sharp images have high variance
    (lots of edge energy); blurry images have low variance.
    Typical rule of thumb: < 100 is noticeably blurry (varies by image size).
    """
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def brightness_score(gray):
    """
    Mean pixel intensity (0-255). Used to flag under/over-exposure.
    < ~60 tends to look underexposed, > ~200 tends to look overexposed,
    but these thresholds should be tuned against your dataset.
    """
    return float(np.mean(gray))


def exposure_flags(gray, dark_thresh=60, bright_thresh=200,
                    clip_frac_thresh=0.05):
    """
    Returns (is_underexposed, is_overexposed, dark_clip_frac, bright_clip_frac).
    Clip fractions = proportion of near-black / near-white pixels,
    which is a more robust exposure signal than mean brightness alone
    (a photo can have a normal mean but blown-out highlights, etc).
    """
    mean_brightness = brightness_score(gray)
    dark_clip_frac = float(np.mean(gray < 10))
    bright_clip_frac = float(np.mean(gray > 245))

    is_under = mean_brightness < dark_thresh or dark_clip_frac > clip_frac_thresh
    is_over = mean_brightness > bright_thresh or bright_clip_frac > clip_frac_thresh

    return is_under, is_over, dark_clip_frac, bright_clip_frac


def noise_score(gray):
    """
    Estimates noise using the median absolute deviation of the
    high-frequency residual (image minus a heavily blurred version).
    Higher = noisier. This is more robust to genuine texture than
    a flat high-pass energy sum.
    """
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    residual = gray.astype(np.float32) - blurred.astype(np.float32)
    mad = np.median(np.abs(residual - np.median(residual)))
    # Convert MAD to an approximate noise sigma (standard scaling factor)
    sigma = 1.4826 * mad
    return float(sigma)


def contrast_score(gray):
    """
    Michelson-style contrast using 1st/99th percentile intensities
    (robust to a few extreme outlier pixels vs. true min/max).
    Range 0 (flat/no contrast) to 1 (full range).
    """
    p1, p99 = np.percentile(gray, [1, 99])
    if p1 + p99 == 0:
        return 0.0
    return float((p99 - p1) / (p99 + p1 + 1e-6))


def saturation_score(bgr):
    """
    Mean saturation from HSV space. Very low saturation across a
    color image can indicate washed-out / corrupted color channels.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1]))


def blockiness_score(gray, block_size=8):
    """
    Classic JPEG-blockiness detector: compares pixel discontinuity
    strength AT 8x8 block boundaries vs. WITHIN blocks. Heavy JPEG
    recompression / block-level corruption creates a periodic grid
    of hard edges at block boundaries that isn't present in natural
    image gradients. Returns the ratio of boundary-edge energy to
    interior-edge energy (>1 means boundaries are artificially sharper
    than the image's natural texture would suggest).
    """
    gray = gray.astype(np.float32)
    h, w = gray.shape

    # Horizontal boundary differences (across vertical block edges)
    boundary_cols = list(range(block_size, w - 1, block_size))
    interior_cols = [c for c in range(1, w - 1) if c % block_size != 0]

    if not boundary_cols or not interior_cols:
        return 0.0

    boundary_diff = np.mean([np.mean(np.abs(gray[:, c] - gray[:, c - 1])) for c in boundary_cols])
    interior_sample = interior_cols[::max(1, len(interior_cols) // len(boundary_cols))]
    interior_diff = np.mean([np.mean(np.abs(gray[:, c] - gray[:, c - 1])) for c in interior_sample])

    if interior_diff < 1e-3:
        return 0.0
    return float(boundary_diff / interior_diff)


def flat_block_fraction(gray, block_size=16, std_thresh=2.0):
    """
    Fraction of the image made up of near-uniform blocks (very low
    local standard deviation) — flags dead-pixel regions, dropped-out
    sensor blocks, or heavily quantized/color-blocked corruption.
    """
    h, w = gray.shape
    flat_count, total = 0, 0
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = gray[y:y + block_size, x:x + block_size]
            total += 1
            if np.std(block) < std_thresh:
                flat_count += 1
    return flat_count / total if total else 0.0


def corruption_flag(bgr, gray, blockiness_thresh=1.2, flat_block_thresh=0.4):
    """
    Flags likely corruption primarily via blockiness (periodic hard
    edges at 8x8 JPEG block boundaries) — this is the reliable signal
    in practice. flat_block_fraction is kept as a secondary signal
    with a much higher threshold, since clipped exposure regions
    (pure black/white from over/under-exposure) also register as
    locally flat and would otherwise cause false positives.
    """
    blockiness = blockiness_score(gray)
    flat_frac = flat_block_fraction(gray)
    likely_corrupt = blockiness > blockiness_thresh or flat_frac > flat_block_thresh
    return likely_corrupt, blockiness, flat_frac


def extract_features(bgr_image):
    """
    Main entry point. Takes a BGR image (as loaded by cv2.imread)
    and returns a dict of all quality-relevant features plus
    heuristic issue flags. This dict is what feeds into the
    quality_score / issues JSON response in the backend, and can
    also be concatenated with CNN embeddings for a hybrid model.
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    blur = blur_score(gray)
    is_under, is_over, dark_clip, bright_clip = exposure_flags(gray)
    noise = noise_score(gray)
    contrast = contrast_score(gray)
    saturation = saturation_score(bgr_image)
    is_corrupt, blockiness, flat_frac = corruption_flag(bgr_image, gray)

    return {
        "blur_score": round(blur, 3),
        "is_blurry": bool(blur < 100),
        "brightness_mean": round(brightness_score(gray), 3),
        "is_underexposed": bool(is_under),
        "is_overexposed": bool(is_over),
        "dark_clip_fraction": round(dark_clip, 4),
        "bright_clip_fraction": round(bright_clip, 4),
        "noise_sigma": round(noise, 3),
        "is_noisy": bool(noise > 11),
        "contrast": round(contrast, 4),
        "saturation_mean": round(saturation, 3),
        "is_likely_corrupt": bool(is_corrupt),
        "blockiness_score": round(blockiness, 3),
        "flat_block_fraction": round(flat_frac, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder of images")
    parser.add_argument("--output", required=True, help="CSV output path")
    args = parser.parse_args()

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    files = [f for f in os.listdir(args.input) if f.lower().endswith(valid_ext)]

    if not files:
        print(f"No images found in {args.input}")
        return

    rows = []
    for fname in files:
        path = os.path.join(args.input, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"Skipping unreadable file: {fname}")
            continue
        feats = extract_features(img)
        feats["filename"] = fname
        rows.append(feats)

    if not rows:
        print("No features extracted.")
        return

    fieldnames = ["filename"] + [k for k in rows[0].keys() if k != "filename"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Extracted features for {len(rows)} images -> {args.output}")


if __name__ == "__main__":
    main()
