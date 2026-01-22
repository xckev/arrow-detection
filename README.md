# Arrow Detection

Detects overlaid arrows in medical images and extracts their direction and tip location.


## Status

This geometric signature-based approach (based on "Overlaid Arrow Detection for Labeling Regions of Interest in Biomedical Images" by Santosh et al., IEEE 2016) works on clean arrows but **does not perform consistently enough** on real-world medical images with varying arrow styles, occlusion, and background complexity.

The **YOLO-based detection method** is implemented as a more robust alternative. We use fine-tune Yolo11 pose estimation models to identify arrows, their tips, and their tails.

## Yolo-Based Usage

Code is available to run in the arrow-detection iPython notebook.

## Signature-Based Usage

```bash
# Single image
python detect-arrows.py image.jpg

# Directory of images
python detect-arrows.py ./images/

# Adjust threshold (lower = more detections, higher = stricter)
python detect-arrows.py image.jpg -t 0.65
```

## Signature-Based Options

| Option | Description |
|--------|-------------|
| `-t, --threshold` | Detection threshold (default: 0.7) |
| `-o, --output` | Output directory (default: ./results/) |
| `-q, --quiet` | Less verbose output |

## How Signature-Based Method works

1. Multi-level binarization extracts arrow candidates
2. Geometric analysis finds arrow vertices (tip + base points)
3. Signature matching compares shape to theoretical arrow model
4. Candidates above threshold are returned as detections

## Installation

```bash
pip install opencv-python numpy scipy matplotlib
```

## Files

- `detect-arrows.py` - CLI tool
- `pipeline.py` - Detection algorithm
- `debug_signatures.py` - Debugging visualization
