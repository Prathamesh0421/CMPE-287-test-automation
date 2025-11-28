#!/usr/bin/env python3
"""Weather Image Augmentation Tool - Fast, deterministic weather effects for ML test data."""

import argparse
import os
import time
from pathlib import Path
from typing import Union, List, Optional

import cv2
import numpy as np
from PIL import Image

PRESETS = {
    "weather_basic": ["rain", "fog", "low_light"],
    "weather_showcase": ["sun_glare", "snow", "fog"],
}
ALL_OPS = ["rain", "snow", "fog", "sun_glare", "low_light"]


def _load_image(img_or_path: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Load image from path or pass through numpy array. Returns BGR."""
    if isinstance(img_or_path, np.ndarray):
        return img_or_path.copy()
    img = cv2.imread(str(img_or_path))
    if img is None:
        raise ValueError(f"Cannot load image: {img_or_path}")
    return img


def _apply_rain(img: np.ndarray, intensity: float, rng: np.random.Generator) -> np.ndarray:
    """Add diagonal rain streaks."""
    out = img.copy()
    h, w = img.shape[:2]
    num_drops = int(intensity * 800 * (h * w) / (512 * 512))
    for _ in range(num_drops):
        x = rng.integers(0, w)
        y = rng.integers(0, h)
        length = rng.integers(10, 30)
        thickness = 1 if rng.random() > 0.3 else 2
        alpha = int(80 + intensity * 120)
        x2, y2 = x + length // 3, y + length
        cv2.line(out, (x, y), (x2, y2), (200, 200, 200, alpha), thickness, cv2.LINE_AA)
    blur_k = max(1, int(intensity * 2)) * 2 + 1
    out = cv2.GaussianBlur(out, (blur_k, blur_k), 0)
    return cv2.addWeighted(img, 0.7, out, 0.3, 0)


def _apply_snow(img: np.ndarray, intensity: float, rng: np.random.Generator) -> np.ndarray:
    """Add snowflake particles."""
    out = img.copy()
    h, w = img.shape[:2]
    num_flakes = int(intensity * 600 * (h * w) / (512 * 512))
    for _ in range(num_flakes):
        x, y = rng.integers(0, w), rng.integers(0, h)
        radius = rng.integers(1, 4)
        brightness = int(200 + rng.random() * 55)
        cv2.circle(out, (x, y), radius, (brightness, brightness, brightness), -1, cv2.LINE_AA)
    k = max(3, int(intensity * 4)) | 1
    out = cv2.GaussianBlur(out, (k, k), 0)
    brightness_boost = np.full_like(img, int(intensity * 30), dtype=np.uint8)
    out = cv2.add(out, brightness_boost)
    return out


def _apply_fog(img: np.ndarray, intensity: float, rng: np.random.Generator) -> np.ndarray:
    """Add fog/mist overlay."""
    h, w = img.shape[:2]
    fog = np.ones((h, w, 3), dtype=np.float32) * 255
    noise = rng.random((h // 8, w // 8)).astype(np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
    noise = cv2.GaussianBlur(noise, (51, 51), 0)
    fog = fog * (0.6 + 0.4 * noise[:, :, np.newaxis])
    alpha = 0.3 + intensity * 0.45
    out = cv2.addWeighted(img.astype(np.float32), 1 - alpha, fog, alpha, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_sun_glare(img: np.ndarray, intensity: float, rng: np.random.Generator) -> np.ndarray:
    """Add sun glare / lens flare effect."""
    out = img.astype(np.float32)
    h, w = img.shape[:2]
    cx = int(w * (0.2 + rng.random() * 0.6))
    cy = int(h * (0.1 + rng.random() * 0.3))
    # Main glare
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    radius = int(min(h, w) * (0.2 + intensity * 0.3))
    glare = np.clip(1 - dist / radius, 0, 1) ** 2
    glare_color = np.array([180, 220, 255]) * intensity
    out += glare[:, :, np.newaxis] * glare_color * 1.5
    # Secondary flares
    for i in range(int(2 + intensity * 3)):
        fx = cx + int((rng.random() - 0.5) * w * 0.6)
        fy = cy + int(rng.random() * h * 0.5)
        fr = int(20 + rng.random() * 40 * intensity)
        fdist = np.sqrt((X - fx) ** 2 + (Y - fy) ** 2)
        flare = np.clip(1 - fdist / fr, 0, 1) ** 1.5
        fc = np.array([100 + rng.random() * 100, 150 + rng.random() * 80, 200])
        out += flare[:, :, np.newaxis] * fc * intensity * 0.5
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_low_light(img: np.ndarray, intensity: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate low light / night conditions."""
    out = img.astype(np.float32)
    darkness = 0.2 + (1 - intensity) * 0.4
    out *= darkness
    blue_tint = np.array([1.15, 1.0, 0.85])
    out *= blue_tint
    noise_level = intensity * 15
    noise = rng.normal(0, noise_level, out.shape).astype(np.float32)
    out += noise
    return np.clip(out, 0, 255).astype(np.uint8)


EFFECT_FNS = {
    "rain": _apply_rain,
    "snow": _apply_snow,
    "fog": _apply_fog,
    "sun_glare": _apply_sun_glare,
    "low_light": _apply_low_light,
}


def augment_image(
    img_or_path: Union[str, Path, np.ndarray],
    ops: Optional[List[str]] = None,
    intensity: float = 0.6,
    seed: int = 42,
) -> np.ndarray:
    """Apply weather augmentation(s) to an image.
    
    Args:
        img_or_path: Image path or numpy array (BGR or RGB).
        ops: List of operations. Default: all effects.
        intensity: Effect strength 0.0-1.0.
        seed: Random seed for reproducibility.
    
    Returns:
        Augmented image as numpy array (BGR).
    """
    if ops is None:
        ops = ALL_OPS
    img = _load_image(img_or_path)
    rng = np.random.default_rng(seed)
    for op in ops:
        if op not in EFFECT_FNS:
            raise ValueError(f"Unknown op: {op}. Available: {ALL_OPS}")
        op_seed = rng.integers(0, 2**31)
        op_rng = np.random.default_rng(op_seed)
        img = EFFECT_FNS[op](img, intensity, op_rng)
    return img


def apply_preset(
    path_or_dir: Union[str, Path],
    preset: str = "weather_basic",
    out_dir: str = "artifacts",
    per_image: int = 5,
    seed: int = 7,
) -> List[str]:
    """Apply preset augmentations to image(s).
    
    Args:
        path_or_dir: Single image path or directory.
        preset: Preset name (weather_basic, weather_showcase).
        out_dir: Output directory.
        per_image: Number of augmented versions per input.
        seed: Base seed.
    
    Returns:
        List of output file paths.
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}")
    ops = PRESETS[preset]
    path = Path(path_or_dir)
    if path.is_dir():
        images = list(path.glob("*.jpg")) + list(path.glob("*.png")) + list(path.glob("*.jpeg"))
    else:
        images = [path]
    out_paths = []
    rng = np.random.default_rng(seed)
    for img_path in images:
        basename = img_path.stem
        img_out_dir = Path(out_dir) / basename
        img_out_dir.mkdir(parents=True, exist_ok=True)
        img = _load_image(img_path)
        for k in range(per_image):
            op = ops[k % len(ops)]
            img_seed = int(rng.integers(0, 2**31))
            aug = augment_image(img, ops=[op], intensity=0.6, seed=img_seed)
            out_path = img_out_dir / f"{op}_{k}.jpg"
            cv2.imwrite(str(out_path), aug, [cv2.IMWRITE_JPEG_QUALITY, 95])
            out_paths.append(str(out_path))
    return out_paths


def main():
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
        img = _load_image(img_path)
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
