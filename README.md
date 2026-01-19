# Arrow Detection Pipeline

A geometric signature-based arrow detection system for biomedical images.

## Overview

This pipeline detects overlaid arrows in medical images (CT, MRI, X-ray, etc.) and extracts their position, direction, and tip location. It's designed to help create training data for medical image understanding models.

**Based on:** "Overlaid Arrow Detection for Labeling Regions of Interest in Biomedical Images" by K.C. Santosh et al., IEEE Intelligent Systems 2016.

## Features

- Template-free detection (no pre-defined arrow templates needed)
- Works with arrows of any size, orientation, or color
- Handles partially occluded or degraded arrows
- Multi-level binarization for robust detection
- Geometric signature matching for accurate recognition

## Installation

```bash
# Dependencies
pip install opencv-python numpy scipy
```

## Quick Start

```bash
# Detect arrows in a single image
python detect-arrows.py image.jpg

# Process all images in a directory
python detect-arrows.py ./images/

# Adjust detection threshold (lower = more detections, higher = fewer but more confident)
python detect-arrows.py image.jpg -t 0.65
```

## Usage

```
python detect-arrows.py <input> [options]

Arguments:
  input                 Path to image file or directory

Options:
  -t, --threshold       Similarity threshold (default: 0.7, range: 0.5-0.9)
  -o, --output          Output directory (default: ./results/)
  --max-dimension       Max image size for processing (default: 720)
  --save-binarization   Save binarization debug images
  -q, --quiet           Suppress detailed output
```

### Examples

```bash
# Basic usage
python detect-arrows.py scan.jpg

# Lower threshold to detect more arrows
python detect-arrows.py scan.jpg -t 0.6

# Process a folder of images
python detect-arrows.py ./medical_images/ -o ./output/

# Save debug images to analyze binarization
python detect-arrows.py scan.jpg --save-binarization
```

## Output

For each image, the pipeline outputs:
- **Visualization**: Image with detected arrows highlighted (`<name>_detected.jpg`)
- **Arrow data**: Tip position, base points, direction angle, confidence score

Example output:
```
Processing: scan.jpg
  Detected: 2 arrow(s)
    Arrow 1: tip=(195, 153), score=0.78, direction=30.3°
    Arrow 2: tip=(433, 411), score=0.72, direction=-150.4°
```

## How It Works

The pipeline follows an 8-step process:

1. **Fuzzy Binarization** - Convert image to binary at multiple threshold levels
2. **Connected Components** - Extract candidate regions from each level
3. **Key Points Selection** - Find arrow vertices (tip C, base points A & B) using orthogonal scanning
4. **Candidate Validation** - Check geometric properties (isosceles triangle)
5. **Discrete Signature** - Compute distance signature from each vertex
6. **Theoretical Signature** - Generate expected signature for ideal arrow model
7. **Similarity Matching** - Compare signatures using Tanimoto index
8. **Redundancy Elimination** - Merge duplicate detections across levels

## Programmatic Usage

```python
from pipeline import ArrowDetector, visualize_results
import cv2

# Load image
image = cv2.imread('scan.jpg')

# Create detector
detector = ArrowDetector(similarity_threshold=0.7)

# Detect arrows
arrows = detector.detect_arrows(image)

# Process results
for arrow in arrows:
    print(f"Tip: {arrow.tip}")
    print(f"Direction: {arrow.get_direction_degrees()}°")
    print(f"Score: {arrow.recognition_score}")

# Save visualization
visualize_results(image, arrows, 'output.jpg')
```

## Tuning the Threshold

| Threshold | Use Case |
|-----------|----------|
| 0.6 - 0.65 | Detect degraded/occluded arrows, may have false positives |
| 0.7 (default) | Balanced detection for typical medical images |
| 0.75 - 0.8 | High confidence detections only |
| 0.85+ | Very strict, only clear arrows |

## Project Context

This is part of a data pipeline for the RAIVN Lab at UW. The goal is to train models that can perform position reasoning over medical images - detecting, pointing to, and counting objects of interest.

**Workflow:**
1. Detect arrows in medical images (this pipeline)
2. Extract arrow positions and directions
3. Remove arrows using image editing
4. Use cleaned images + arrow locations as training data

## Files

- `detect-arrows.py` - Command-line interface
- `pipeline.py` - Core detection algorithm
- `debug_signatures.py` - Visualization tools for debugging

## License

Research use only.
