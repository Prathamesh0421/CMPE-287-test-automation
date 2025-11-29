"""Individual weather effect implementations."""

import cv2
import numpy as np


def apply_rain(img: np.ndarray, intensity: float, rng: np.random.Generator) -> np.ndarray:
    """Add diagonal rain streaks.
    
    Args:
        img: Input image (BGR).
        intensity: Effect strength 0.0-1.0.
        rng: Random number generator for reproducibility.
        
    Returns:
        Image with rain effect applied.
    """
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


def apply_snow(img: np.ndarray, intensity: float, rng: np.random.Generator) -> np.ndarray:
    """Add snowflake particles.
    
    Args:
        img: Input image (BGR).
        intensity: Effect strength 0.0-1.0.
        rng: Random number generator for reproducibility.
        
    Returns:
        Image with snow effect applied.
    """
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


def apply_fog(img: np.ndarray, intensity: float, rng: np.random.Generator) -> np.ndarray:
    """Add fog/mist overlay.
    
    Args:
        img: Input image (BGR).
        intensity: Effect strength 0.0-1.0.
        rng: Random number generator for reproducibility.
        
    Returns:
        Image with fog effect applied.
    """
    h, w = img.shape[:2]
    fog = np.ones((h, w, 3), dtype=np.float32) * 255
    noise = rng.random((h // 8, w // 8)).astype(np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
    noise = cv2.GaussianBlur(noise, (51, 51), 0)
    fog = fog * (0.6 + 0.4 * noise[:, :, np.newaxis])
    alpha = 0.3 + intensity * 0.45
    out = cv2.addWeighted(img.astype(np.float32), 1 - alpha, fog, alpha, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_sun_glare(img: np.ndarray, intensity: float, rng: np.random.Generator) -> np.ndarray:
    """Add sun glare / lens flare effect.
    
    Args:
        img: Input image (BGR).
        intensity: Effect strength 0.0-1.0.
        rng: Random number generator for reproducibility.
        
    Returns:
        Image with sun glare effect applied.
    """
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


def apply_low_light(img: np.ndarray, intensity: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate low light / night conditions.
    
    Args:
        img: Input image (BGR).
        intensity: Effect strength 0.0-1.0.
        rng: Random number generator for reproducibility.
        
    Returns:
        Image with low light effect applied.
    """
    out = img.astype(np.float32)
    darkness = 0.2 + (1 - intensity) * 0.4
    out *= darkness
    blue_tint = np.array([1.15, 1.0, 0.85])
    out *= blue_tint
    noise_level = intensity * 15
    noise = rng.normal(0, noise_level, out.shape).astype(np.float32)
    out += noise
    return np.clip(out, 0, 255).astype(np.uint8)


# Effect function registry
EFFECT_FNS = {
    "rain": apply_rain,
    "snow": apply_snow,
    "fog": apply_fog,
    "sun_glare": apply_sun_glare,
    "low_light": apply_low_light,
}
