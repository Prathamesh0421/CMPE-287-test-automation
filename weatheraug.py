#!/usr/bin/env python3
"""Weather Image Augmentation Tool - Fast, deterministic weather effects for ML test data."""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

# Import from modular components
from image_utils import load_image
from augmentation import augment_image, apply_preset
from presets import PRESETS, ALL_OPS


def main():
    """Main CLI interface for weather augmentation tool."""
    parser = argparse.ArgumentParser(description="Weather Image Augmentation Tool")
    parser.add_argument("--input", "-i", required=True, help="Input image or directory")
    parser.add_argument("--output", "-o", default="artifacts", help="Output directory")
    parser.add_argument("--preset", "-p", help="Preset name (weather_basic, weather_showcase)")
    parser.add_argument("--ops", nargs="+", choices=ALL_OPS, help="Operations to apply")
    parser.add_argument("--per-image", type=int, default=5, help="Augmentations per image")
    parser.add_argument("--intensity", type=float, default=0.6, help="Effect intensity 0-1")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    start = time.time()
    path = Path(args.input)
    if path.is_dir():
        images = list(path.glob("*.jpg")) + list(path.glob("*.png")) + list(path.glob("*.jpeg"))
    else:
        images = [path]

    if args.preset:
        ops = PRESETS.get(args.preset, PRESETS["weather_basic"])
    elif args.ops:
        ops = args.ops
    else:
        ops = ALL_OPS

    timings = {op: 0.0 for op in ops}
    counts = {op: 0 for op in ops}
    out_paths = []
    rng = np.random.default_rng(args.seed)

    for img_path in images:
        basename = img_path.stem
        img_out_dir = Path(args.output) / basename
        img_out_dir.mkdir(parents=True, exist_ok=True)
        img = load_image(img_path)
        for k in range(args.per_image):
            op = ops[k % len(ops)]
            op_start = time.time()
            img_seed = int(rng.integers(0, 2**31))
            aug = augment_image(img, ops=[op], intensity=args.intensity, seed=img_seed)
            timings[op] += time.time() - op_start
            counts[op] += 1
            out_path = img_out_dir / f"{op}_{k}.jpg"
            cv2.imwrite(str(out_path), aug, [cv2.IMWRITE_JPEG_QUALITY, 95])
            out_paths.append(str(out_path))

    total = time.time() - start
    print(f"\n{'='*50}")
    print(f"Weather Augmentation Complete | Seed: {args.seed}")
    print(f"{'='*50}")
    print(f"{'Effect':<15} {'Count':>8} {'Time (s)':>10} {'Avg (ms)':>10}")
    print(f"{'-'*50}")
    for op in ops:
        if counts[op] > 0:
            avg_ms = (timings[op] / counts[op]) * 1000
            print(f"{op:<15} {counts[op]:>8} {timings[op]:>10.3f} {avg_ms:>10.1f}")
    print(f"{'-'*50}")
    print(f"{'TOTAL':<15} {len(out_paths):>8} {total:>10.3f}")
    print(f"\nOutput: {args.output}/ ({len(out_paths)} files)")


if __name__ == "__main__":
    main()
