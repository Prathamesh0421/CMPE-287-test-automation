# Weather Image Augmentation Framework
## Technical Report

---

**Project:** ML Test Data Augmentation Tool  
**Version:** 1.0  
**Date:** November 2025  
**Author:** CMPE-287 Test Automation Project

---

## Executive Summary

The Weather Image Augmentation Framework is a Python-based tool designed to generate synthetic weather conditions for machine learning test data augmentation. The framework provides fast, deterministic, and reproducible weather effects that enable robust testing of computer vision models under adverse environmental conditions.

**Key Highlights:**
- 5 built-in weather effects with customizable intensity
- Deterministic and reproducible outputs via seeded random number generation
- Dual interface: Python API and command-line interface (CLI)
- Modular architecture for maintainability and extensibility
- High performance: processes images in milliseconds using optimized NumPy/OpenCV operations

---

## 1. Framework Architecture

### 1.1 Modular Design

The framework follows a modular architecture with clear separation of concerns:

```
weatheraug/
├── presets.py           # Configuration and preset definitions
├── image_utils.py       # Image loading utilities
├── weather_effects.py   # Individual weather effect implementations
├── augmentation.py      # Core augmentation logic
└── weatheraug.py        # CLI interface and main entry point
```

**Design Principles:**
- **Single Responsibility:** Each module handles one specific aspect
- **Separation of Concerns:** Effects, utilities, and orchestration are isolated
- **Extensibility:** New effects can be added without modifying existing code
- **Reusability:** Modules can be imported independently for custom workflows

### 1.2 Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| `presets.py` | Stores preset configurations and available operations |
| `image_utils.py` | Handles image loading from paths or NumPy arrays |
| `weather_effects.py` | Implements individual weather transformation algorithms |
| `augmentation.py` | Orchestrates effect application and batch processing |
| `weatheraug.py` | Provides CLI interface and user interaction |

---

## 2. Built-in Weather Effects

The framework includes five scientifically-inspired weather effects, each implemented using computer vision techniques:

### 2.1 Rain Effect

**Description:** Simulates rainfall with diagonal streaks and atmospheric blur.

**Implementation:**
- Generates random diagonal line segments scaled to image dimensions
- Varies streak length (10-30 pixels) and thickness (1-2 pixels)
- Applies Gaussian blur to simulate motion and atmospheric diffusion
- Alpha-blends with original image (70% original, 30% rain layer)

**Parameters:**
- Intensity controls: number of raindrops, blur kernel size, alpha values
- Deterministic placement via seeded RNG

**Use Cases:** Testing autonomous vehicle perception in rainy conditions, outdoor surveillance systems

---

### 2.2 Snow Effect

**Description:** Adds snowflake particles with brightness variation and global illumination boost.

**Implementation:**
- Renders white circular particles at random positions
- Varies particle radius (1-4 pixels) and brightness (200-255)
- Applies Gaussian blur for soft, natural appearance
- Adds global brightness boost to simulate snow reflection

**Parameters:**
- Intensity controls: particle count, blur strength, brightness boost magnitude

**Use Cases:** Winter weather testing for robotics, outdoor camera systems

---

### 2.3 Fog Effect

**Description:** Creates realistic fog/mist overlay using procedural noise.

**Implementation:**
- Generates white overlay modulated by low-frequency Perlin-like noise
- Creates noise via downsampled random array with heavy Gaussian blur
- Alpha-blends fog layer with original image
- Intensity controls fog opacity (30-75% alpha)

**Parameters:**
- Noise frequency: 1/8 of image resolution
- Blur kernel: 51x51 pixels for smooth gradients

**Use Cases:** Testing visibility-dependent algorithms, maritime/aviation systems

---

### 2.4 Sun Glare Effect

**Description:** Simulates lens flare and sun glare artifacts.

**Implementation:**
- Places main radial gradient "sun" at random upper position
- Computes distance-based falloff with quadratic attenuation
- Adds 2-5 secondary lens flare spots with warm color tones
- Additively composites all flare elements

**Parameters:**
- Main glare radius: 20-50% of image dimension
- Secondary flares: randomized position, size, and color
- Intensity controls: brightness multiplier and flare count

**Use Cases:** Testing camera systems under direct sunlight, glare-resistant algorithms

---

### 2.5 Low Light Effect

**Description:** Simulates night/low-light conditions with sensor noise.

**Implementation:**
- Applies darkness multiplication (20-60% brightness reduction)
- Adds blue color tint to simulate moonlight (B: 1.15, G: 1.0, R: 0.85)
- Injects Gaussian noise to simulate sensor noise in dark conditions
- Noise level scales with intensity

**Parameters:**
- Darkness factor: intensity-dependent (0.2-0.6)
- Noise standard deviation: 0-15 (intensity-scaled)

**Use Cases:** Night vision testing, low-light object detection, security cameras

---

## 3. API Reference

### 3.1 Python API

The framework provides two primary functions for programmatic use:

#### `augment_image()`

Apply one or more weather effects to a single image.

```python
def augment_image(
    img_or_path: Union[str, Path, np.ndarray],
    ops: Optional[List[str]] = None,
    intensity: float = 0.6,
    seed: int = 42,
) -> np.ndarray
```

**Parameters:**
- `img_or_path`: Image file path or NumPy array (BGR/RGB)
- `ops`: List of effects to apply (default: all effects)
- `intensity`: Effect strength, range 0.0-1.0 (default: 0.6)
- `seed`: Random seed for reproducibility (default: 42)

**Returns:** Augmented image as NumPy array (BGR format)

**Example:**
```python
from augmentation import augment_image

# Apply rain and fog to an image
result = augment_image("photo.jpg", ops=["rain", "fog"], intensity=0.7, seed=123)

# Or use NumPy array
import cv2
img = cv2.imread("photo.jpg")
result = augment_image(img, ops=["snow"], intensity=0.8, seed=7)
```

---

#### `apply_preset()`

Batch process images with predefined effect combinations.

```python
def apply_preset(
    path_or_dir: Union[str, Path],
    preset: str = "weather_basic",
    out_dir: str = "artifacts",
    per_image: int = 5,
    seed: int = 7,
) -> List[str]
```

**Parameters:**
- `path_or_dir`: Single image path or directory
- `preset`: Preset name ("weather_basic" or "weather_showcase")
- `out_dir`: Output directory for augmented images
- `per_image`: Number of augmented versions per input
- `seed`: Base random seed

**Returns:** List of output file paths

**Example:**
```python
from augmentation import apply_preset

# Process entire directory with preset
paths = apply_preset(
    "./images", 
    preset="weather_basic", 
    out_dir="artifacts", 
    per_image=5, 
    seed=42
)
print(f"Generated {len(paths)} augmented images")
```

---

### 3.2 Command-Line Interface

The CLI provides full access to framework features via terminal:

#### Basic Usage

```bash
# Apply preset to directory
python weatheraug.py --input ./images --output ./artifacts --preset weather_basic --per-image 5 --seed 42

# Apply specific effects
python weatheraug.py --input ./images --output ./artifacts --ops rain snow fog --intensity 0.6 --seed 1

# Process single image
python weatheraug.py --input photo.jpg --output ./out --ops fog rain --intensity 0.7 --seed 123
```

#### CLI Parameters

| Parameter | Short | Default | Description |
|-----------|-------|---------|-------------|
| `--input` | `-i` | Required | Input image or directory path |
| `--output` | `-o` | `artifacts` | Output directory |
| `--preset` | `-p` | None | Preset name (weather_basic, weather_showcase) |
| `--ops` | - | All effects | Space-separated list of effects |
| `--per-image` | - | 5 | Augmentations per input image |
| `--intensity` | - | 0.6 | Effect strength (0.0-1.0) |
| `--seed` | - | 42 | Random seed for reproducibility |

#### Output Format

The CLI generates organized output with performance metrics:

```
artifacts/
├── image1/
│   ├── rain_0.jpg
│   ├── fog_1.jpg
│   └── low_light_2.jpg
└── image2/
    └── ...

==================================================
Weather Augmentation Complete | Seed: 42
==================================================
Effect             Count   Time (s)   Avg (ms)
--------------------------------------------------
rain                   6      0.017        2.8
fog                    6      0.022        3.6
low_light              3      0.018        5.9
--------------------------------------------------
TOTAL                 15      0.086

Output: ./artifacts/ (15 files)
```

---

## 4. Presets and Configuration

### 4.1 Built-in Presets

The framework includes two predefined presets for common use cases:

| Preset | Effects | Use Case |
|--------|---------|----------|
| `weather_basic` | rain, fog, low_light | General robustness testing |
| `weather_showcase` | sun_glare, snow, fog | Demonstration and edge cases |

### 4.2 Available Operations

All five effects can be used individually or combined:

```python
ALL_OPS = ["rain", "snow", "fog", "sun_glare", "low_light"]
```

### 4.3 Custom Presets

Users can define custom presets by modifying `presets.py`:

```python
PRESETS = {
    "weather_basic": ["rain", "fog", "low_light"],
    "weather_showcase": ["sun_glare", "snow", "fog"],
    "custom_harsh": ["rain", "fog", "low_light", "sun_glare"],  # Add custom
}
```

---

## 5. Key Features

### 5.1 Deterministic and Reproducible

**Problem:** Non-deterministic augmentation makes regression testing impossible.

**Solution:** The framework uses seeded random number generators at two levels:
1. **Global seed:** Controls overall randomization sequence
2. **Per-operation seed:** Derived from global seed, ensures effect-level reproducibility

**Benefit:** Same seed always produces identical outputs, enabling:
- Regression testing of ML models
- Reproducible research results
- Debugging and validation

**Example:**
```bash
# These commands produce identical outputs
python weatheraug.py -i img.jpg -o out1 --seed 42
python weatheraug.py -i img.jpg -o out2 --seed 42
# out1 and out2 are pixel-perfect identical
```

---

### 5.2 High Performance

**Optimization Strategies:**
- **Vectorized Operations:** Uses NumPy array operations instead of loops where possible
- **Efficient Blur:** Leverages OpenCV's optimized Gaussian blur implementation
- **In-place Operations:** Minimizes memory allocations
- **Batch Processing:** Amortizes I/O overhead across multiple images

**Performance Metrics:**
- Rain: ~2.8ms per image (512×512)
- Snow: ~1.3ms per image
- Fog: ~3.6ms per image
- Sun Glare: ~14.0ms per image
- Low Light: ~5.9ms per image

**Scalability:** Can process hundreds of images in seconds on standard hardware.

---

### 5.3 Flexible Input/Output

**Input Flexibility:**
- File paths (JPEG, PNG)
- NumPy arrays (BGR or RGB)
- Single images or entire directories

**Output Options:**
- Organized directory structure by image basename
- High-quality JPEG output (95% quality)
- Customizable output directory

---

### 5.4 Extensibility

**Adding New Effects:**

1. Implement effect function in `weather_effects.py`:
```python
def apply_custom_effect(img: np.ndarray, intensity: float, rng: np.random.Generator) -> np.ndarray:
    # Implementation here
    return modified_img
```

2. Register in effect registry:
```python
EFFECT_FNS = {
    # ... existing effects
    "custom_effect": apply_custom_effect,
}
```

3. Add to available operations in `presets.py`:
```python
ALL_OPS = ["rain", "snow", "fog", "sun_glare", "low_light", "custom_effect"]
```

---

## 6. Machine Learning Integration

### 6.1 Test Data Augmentation Use Cases

The framework supports ML testing workflows in several ways:

#### Robustness Testing
- **Objective:** Verify model performance under adverse weather
- **Approach:** Generate augmented test sets with varying intensity levels
- **Metrics:** Measure accuracy degradation across weather conditions

#### Edge Case Generation
- **Objective:** Create rare conditions without expensive data collection
- **Approach:** Combine multiple effects (e.g., rain + fog + low_light)
- **Benefit:** Test corner cases that are difficult to capture in real-world datasets

#### Regression Testing
- **Objective:** Ensure model updates don't degrade weather robustness
- **Approach:** Use deterministic seeds to create fixed test sets
- **Benefit:** Compare model versions on identical augmented data

#### Dataset Expansion
- **Objective:** Increase training data diversity
- **Approach:** Apply effects to existing datasets with varied intensities
- **Benefit:** Improve model generalization without additional data collection

---

### 6.2 Integration Examples

#### PyTorch Integration

```python
import torch
from torch.utils.data import Dataset
from augmentation import augment_image

class WeatherAugmentedDataset(Dataset):
    def __init__(self, image_paths, effects, intensity=0.6):
        self.image_paths = image_paths
        self.effects = effects
        self.intensity = intensity
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Apply weather augmentation
        img = augment_image(img_path, ops=self.effects, intensity=self.intensity, seed=idx)
        # Convert to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img_tensor
    
    def __len__(self):
        return len(self.image_paths)
```

#### TensorFlow Integration

```python
import tensorflow as tf
from augmentation import augment_image

def weather_augment_fn(image_path, effect="rain", intensity=0.6):
    def _augment(path):
        # Load and augment
        img = augment_image(path.numpy().decode(), ops=[effect], intensity=intensity)
        return img.astype('float32') / 255.0
    
    return tf.py_function(_augment, [image_path], tf.float32)

# Use in tf.data pipeline
dataset = tf.data.Dataset.from_tensor_slices(image_paths)
dataset = dataset.map(lambda x: weather_augment_fn(x, effect="fog"))
```

---

### 6.3 Model Testing Workflow

**Recommended Testing Pipeline:**

1. **Baseline Evaluation:** Test model on clean images
2. **Single Effect Testing:** Evaluate each weather effect independently
3. **Intensity Sweep:** Test multiple intensity levels (0.3, 0.5, 0.7, 0.9)
4. **Combined Effects:** Test realistic combinations (rain + fog, snow + low_light)
5. **Regression Testing:** Use fixed seeds for version-to-version comparison

**Example Test Script:**

```python
from augmentation import augment_image
import cv2

def evaluate_model_robustness(model, test_images, effects, intensities):
    results = {}
    for effect in effects:
        for intensity in intensities:
            accuracies = []
            for img_path in test_images:
                # Apply augmentation
                aug_img = augment_image(img_path, ops=[effect], intensity=intensity, seed=42)
                # Run inference
                prediction = model.predict(aug_img)
                accuracy = compute_accuracy(prediction, ground_truth)
                accuracies.append(accuracy)
            results[f"{effect}_{intensity}"] = np.mean(accuracies)
    return results
```

---

## 7. Technical Specifications

### 7.1 System Requirements

**Software Requirements:**
- Python 3.10 or higher
- OpenCV (opencv-python) 4.5+
- NumPy 1.20+
- Pillow 8.0+ (optional, for additional format support)

**Hardware Requirements:**
- Minimum: 2GB RAM, dual-core CPU
- Recommended: 4GB+ RAM, quad-core CPU for batch processing
- GPU: Not required (CPU-optimized implementation)

### 7.2 Dependencies

```
opencv-python>=4.5.0
numpy>=1.20.0
Pillow>=8.0.0
```

### 7.3 Installation

```bash
# Clone repository
git clone https://github.com/username/weather-augmentation.git
cd weather-augmentation

# Install dependencies
pip install opencv-python numpy Pillow

# Verify installation
python weatheraug.py --help
```

---

## 8. Performance Benchmarks

### 8.1 Processing Speed

Benchmarks performed on MacBook Pro (M1, 8GB RAM) with 512×512 images:

| Effect | Avg Time (ms) | Images/sec |
|--------|---------------|------------|
| Rain | 2.8 | 357 |
| Snow | 1.3 | 769 |
| Fog | 3.6 | 278 |
| Sun Glare | 14.0 | 71 |
| Low Light | 5.9 | 169 |

**Batch Processing:** 15 images (5 effects × 3 images) processed in ~86ms total.

### 8.2 Memory Usage

- **Single Image:** ~10-20MB peak memory (for 512×512 image)
- **Batch Processing:** Scales linearly with image count
- **Memory Efficiency:** Images processed sequentially to minimize memory footprint

---

## 9. Limitations and Future Work

### 9.1 Current Limitations

1. **Effect Realism:** Simplified physics models (not photorealistic)
2. **Fixed Effect Library:** Limited to 5 predefined effects
3. **No Depth Awareness:** Effects applied uniformly without depth information
4. **JPEG Output Only:** No support for PNG or other lossless formats in CLI
5. **No GPU Acceleration:** CPU-only implementation

### 9.2 Future Enhancements

**Planned Features:**
- Additional effects: hail, dust, smoke, motion blur
- Depth-aware effects using depth maps or segmentation
- GPU acceleration via CUDA/OpenCL
- Real-time video processing
- Integration with popular ML frameworks (PyTorch, TensorFlow)
- Web-based UI for interactive augmentation

**Research Directions:**
- GAN-based photorealistic weather synthesis
- Physics-based rendering for accurate light transport
- Learned augmentation policies via AutoAugment
- Multi-modal augmentation (weather + time-of-day + season)

---

## 10. Conclusion

The Weather Image Augmentation Framework provides a robust, efficient, and extensible solution for generating synthetic weather conditions in ML test data. With its modular architecture, deterministic outputs, and dual API/CLI interface, the framework enables comprehensive robustness testing of computer vision models under adverse environmental conditions.

**Key Strengths:**
- ✅ Fast and efficient (processes images in milliseconds)
- ✅ Deterministic and reproducible (seeded RNG)
- ✅ Modular and extensible (easy to add new effects)
- ✅ Dual interface (Python API + CLI)
- ✅ Production-ready (comprehensive documentation and examples)

**Ideal For:**
- Autonomous vehicle testing
- Outdoor surveillance systems
- Robotics perception
- Drone/aerial imaging
- Any ML system requiring weather robustness

---

## Appendix A: Code Examples

### Example 1: Batch Processing with Custom Intensity

```python
from augmentation import augment_image
import cv2
from pathlib import Path

def batch_augment_with_intensity_sweep(input_dir, output_dir):
    intensities = [0.3, 0.5, 0.7, 0.9]
    effects = ["rain", "fog", "snow"]
    
    for img_path in Path(input_dir).glob("*.jpg"):
        for effect in effects:
            for intensity in intensities:
                aug = augment_image(img_path, ops=[effect], intensity=intensity, seed=42)
                out_path = f"{output_dir}/{img_path.stem}_{effect}_{intensity}.jpg"
                cv2.imwrite(out_path, aug, [cv2.IMWRITE_JPEG_QUALITY, 95])
```

### Example 2: Combined Effects

```python
from augmentation import augment_image

# Simulate heavy storm (rain + fog + low_light)
storm_img = augment_image("photo.jpg", ops=["rain", "fog", "low_light"], intensity=0.8, seed=123)

# Simulate winter night (snow + low_light)
winter_night = augment_image("photo.jpg", ops=["snow", "low_light"], intensity=0.7, seed=456)
```

---

## Appendix B: References

1. **OpenCV Documentation:** https://docs.opencv.org/
2. **NumPy Documentation:** https://numpy.org/doc/
3. **Image Augmentation for Deep Learning:** https://arxiv.org/abs/1906.11172
4. **Albumentations Library:** https://albumentations.ai/ (inspiration for API design)

---

**Document Version:** 1.0  
**Last Updated:** November 2025  
**Contact:** CMPE-287 Test Automation Team
