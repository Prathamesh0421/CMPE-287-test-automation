#!/bin/bash
# Generate weather-augmented samples using both presets

set -e

INPUT_DIR="${1:-./sample_images}"
OUTPUT_DIR="${2:-./artifacts}"

echo "=================================================="
echo "Weather Augmentation - Sample Generation"
echo "=================================================="
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

if [ ! -d "$INPUT_DIR" ]; then
    echo "Creating sample input directory..."
    mkdir -p "$INPUT_DIR"
    echo "Please add images to $INPUT_DIR and re-run."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Running preset: weather_basic"
echo "--------------------------------------------------"
python weatheraug.py \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR/weather_basic" \
    --preset weather_basic \
    --per-image 5 \
    --intensity 0.6 \
    --seed 42

echo ""
echo "Running preset: weather_showcase"
echo "--------------------------------------------------"
python weatheraug.py \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR/weather_showcase" \
    --preset weather_showcase \
    --per-image 5 \
    --intensity 0.6 \
    --seed 42

echo ""
echo "=================================================="
echo "Done! Samples saved to $OUTPUT_DIR"
echo "=================================================="
find "$OUTPUT_DIR" -name "*.jpg" | wc -l | xargs echo "Total images generated:"
