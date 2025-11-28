# Weather Image Augmentation Tool

Fast, deterministic weather effects for ML test data augmentation.

## Installation

```bash
pip install opencv-python numpy Pillow
```

## Quick Start

### CLI Usage

Apply preset with 5 augmented images per input:
```bash
python weatheraug.py --input ./images --output ./artifacts --preset weather_basic --per-image 5 --seed 42
```

Apply specific effects:
```bash
python weatheraug.py --input ./images --output ./artifacts --ops rain snow fog sun_glare low_light --seed 1
```

Single image:
```bash
python weatheraug.py --input photo.jpg --output ./out --ops fog rain --intensity 0.7 --seed 123
```

### Python API

```python
from weatheraug import augment_image, apply_preset

# Single image with specific effects
result = augment_image("photo.jpg", ops=["rain", "fog"], intensity=0.6, seed=42)

# Or pass numpy array
import cv2
img = cv2.imread("photo.jpg")
result = augment_image(img, ops=["snow"], intensity=0.8, seed=7)

# Apply preset to directory
paths = apply_preset("./images", preset="weather_basic", out_dir="artifacts", per_image=5, seed=7)
```

## Effects

| Effect | Description |
|--------|-------------|
| `rain` | Diagonal rain streaks with blur |
| `snow` | White particle snowflakes |
| `fog` | Haze overlay with noise texture |
| `sun_glare` | Lens flare with secondary artifacts |
| `low_light` | Dark, blue-tinted night simulation |

## Presets

- `weather_basic`: rain, fog, low_light
- `weather_showcase`: sun_glare, snow, fog

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--intensity` | 0.6 | Effect strength (0.0-1.0) |
| `--seed` | 42 | Random seed for reproducibility |
| `--per-image` | 5 | Augmentations per input image |

## Output Structure

```
artifacts/
├── image1/
│   ├── rain_0.jpg
│   ├── fog_1.jpg
│   └── low_light_2.jpg
└── image2/
    └── ...
```

## Determinism

All outputs are reproducible. Same seed = same results:
```bash
python weatheraug.py -i img.jpg -o out1 --seed 42
python weatheraug.py -i img.jpg -o out2 --seed 42
# out1 and out2 are identical
```

## Requirements

- Python 3.10+
- opencv-python
- numpy  
- Pillow
