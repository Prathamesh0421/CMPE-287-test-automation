"""Image loading and processing utilities."""

from pathlib import Path
from typing import Union

import cv2
import numpy as np


def load_image(img_or_path: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Load image from path or pass through numpy array. Returns BGR.
    
    Args:
        img_or_path: Image path or numpy array (BGR or RGB).
        
    Returns:
        Image as numpy array in BGR format.
        
    Raises:
        ValueError: If image cannot be loaded from path.
    """
    if isinstance(img_or_path, np.ndarray):
        return img_or_path.copy()
    img = cv2.imread(str(img_or_path))
    if img is None:
        raise ValueError(f"Cannot load image: {img_or_path}")
    return img
