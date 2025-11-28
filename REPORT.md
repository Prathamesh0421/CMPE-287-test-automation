# Weather Augmentation Tool - Technical Report

## Features

- **5 Weather Effects**: Rain, snow, fog, sun glare, and low-light/night simulation
- **Deterministic**: Global seed + per-operation seeded RNG ensures reproducibility
- **Flexible Input**: Accepts file paths or NumPy arrays; preserves aspect ratio
- **Fast**: Processes 3-5 images in seconds using optimized NumPy/OpenCV operations
- **Dual Interface**: Python API and full-featured CLI

## Effect Implementations

### Rain
Draws randomized diagonal line streaks (simulating falling rain at an angle). Applies Gaussian blur to soften edges, then alpha-blends with original to preserve detail. Streak count scales with image size and intensity.

### Snow
Renders small white circles at random positions with varying radii and brightness. Applies light blur for softness and adds global brightness boost to simulate snow reflection.

### Fog
Creates a white overlay modulated by low-frequency Perlin-like noise (generated via resized random array + blur). Alpha-blends overlay with source image; intensity controls opacity.

### Sun Glare
Places a radial gradient "sun" at a random upper position. Computes distance-based falloff with quadratic curve. Adds 2-5 secondary lens flare spots with randomized colors (warm tones). All flares are additively composited.

### Low Light
Multiplies image by darkness factor (intensity-dependent). Applies blue color tint to simulate moonlight. Adds Gaussian noise to simulate sensor noise in dark conditions.

## API Summary

```python
augment_image(img_or_path, ops=["rain"], intensity=0.6, seed=42) -> np.ndarray
apply_preset(path_or_dir, preset="weather_basic", out_dir="artifacts", per_image=5, seed=7) -> List[str]
```

## CLI Summary

```bash
python weatheraug.py --input DIR --output OUT --preset weather_basic --per-image 5 --seed 42
python weatheraug.py --input DIR --output OUT --ops rain snow fog --intensity 0.6 --seed 1
```

Output: `OUT/<basename>/<op>_<k>.jpg` with timing table printed to stdout.

## Test Data Augmentation Use Case

This tool supports ML testing by generating diverse weather conditions from clean images:

1. **Robustness Testing**: Verify model performance under adverse weather
2. **Edge Case Generation**: Create rare conditions (heavy fog, night) without collection
3. **Reproducibility**: Deterministic seeds enable regression testing
4. **Scalability**: Batch process datasets with presets

## Demo Video Instructions

To create a demo video:
1. Prepare 3-5 sample images in `./sample_images/`
2. Run: `bash make_samples.sh`
3. Screen-record terminal output and browse `./artifacts/` folder
4. Show before/after comparisons for each effect
