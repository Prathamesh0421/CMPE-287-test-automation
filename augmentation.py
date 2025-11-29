"""Core augmentation logic and batch processing."""

from pathlib import Path
from typing import Union, List, Optional

import cv2
import numpy as np

from image_utils import load_image
from weather_effects import EFFECT_FNS
from presets import PRESETS, ALL_OPS


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
        
    Raises:
        ValueError: If unknown operation is specified.
    """
    if ops is None:
        ops = ALL_OPS
    img = load_image(img_or_path)
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
        
    Raises:
        ValueError: If unknown preset is specified.
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
        img = load_image(img_path)
        for k in range(per_image):
            op = ops[k % len(ops)]
            img_seed = int(rng.integers(0, 2**31))
            aug = augment_image(img, ops=[op], intensity=0.6, seed=img_seed)
            out_path = img_out_dir / f"{op}_{k}.jpg"
            cv2.imwrite(str(out_path), aug, [cv2.IMWRITE_JPEG_QUALITY, 95])
            out_paths.append(str(out_path))
    return out_paths
