#!/usr/bin/env python3
"""
Arrow Detection CLI Tool

Detect arrows in biomedical images using geometric signature-based approach.
Based on: "Overlaid Arrow Detection for Labeling Regions of Interest in Biomedical Images"
by K.C. Santosh et al., IEEE Intelligent Systems 2016

Usage:
    python detect-arrows.py <image_or_directory> [options]

Examples:
    python detect-arrows.py image.jpg
    python detect-arrows.py image.jpg --threshold 0.65
    python detect-arrows.py ./images/ --threshold 0.7 --output ./results/
"""

import argparse
import cv2
import sys
from pathlib import Path

from pipeline import ArrowDetector, visualize_results


def process_image(image_path: Path, detector: ArrowDetector, output_dir: Path, 
                  save_binarization: bool = False, verbose: bool = True) -> int:
    """
    Process a single image and save results.
    
    Returns:
        Number of arrows detected
    """
    if verbose:
        print(f"Processing: {image_path.name}")
    
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  Error: Could not load image")
        return 0
    
    if verbose:
        print(f"  Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Detect arrows
    arrows = detector.detect_arrows(image)
    
    if verbose:
        print(f"  Detected: {len(arrows)} arrow(s)")
    
    # Save visualization
    output_path = output_dir / f"{image_path.stem}_detected.jpg"
    visualize_results(image, arrows, str(output_path))
    
    if verbose:
        print(f"  Saved: {output_path}")
    
    # Save binarization levels if requested
    if save_binarization:
        bin_dir = output_dir / f"{image_path.stem}_binarization"
        bin_dir.mkdir(exist_ok=True)
        binary_images = detector.binarizer.binarize(image)
        for level, binary_img in enumerate(binary_images, 1):
            cv2.imwrite(str(bin_dir / f"level_{level}.jpg"), binary_img)
    
    # Print arrow details
    if verbose and len(arrows) > 0:
        for i, arrow in enumerate(arrows, 1):
            print(f"    Arrow {i}: tip={arrow.tip}, score={arrow.recognition_score:.3f}, "
                  f"direction={arrow.get_direction_degrees():.1f}°")
    
    return len(arrows)


def process_directory(dir_path: Path, detector: ArrowDetector, output_dir: Path,
                      save_binarization: bool = False) -> None:
    """Process all images in a directory."""
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}
    image_files = [f for f in dir_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in {dir_path}")
        return
    
    print(f"Found {len(image_files)} images in {dir_path}")
    print()
    
    total_arrows = 0
    
    for i, img_path in enumerate(sorted(image_files), 1):
        print(f"[{i}/{len(image_files)}] ", end="")
        arrows = process_image(img_path, detector, output_dir, save_binarization)
        total_arrows += arrows
        print()
    
    print("=" * 50)
    print(f"Complete: {len(image_files)} images, {total_arrows} arrows detected")
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Detect arrows in biomedical images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.jpg                     Detect arrows in a single image
  %(prog)s image.jpg -t 0.65             Use custom threshold
  %(prog)s ./images/                     Process all images in directory
  %(prog)s image.jpg -o ./output/        Specify output directory
  %(prog)s image.jpg --save-binarization Save binarization debug images
        """
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Path to image file or directory"
    )
    
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for detection (default: 0.7, range: 0.5-0.9)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory (default: ./results/)"
    )
    
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=720,
        help="Max image dimension for processing (default: 720, use 0 to disable)"
    )
    
    parser.add_argument(
        "--save-binarization",
        action="store_true",
        help="Save binarization level images for debugging"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )
    
    args = parser.parse_args()
    
    # Validate input path
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {args.input} does not exist")
        sys.exit(1)
    
    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        print(f"Error: Threshold must be between 0 and 1, got {args.threshold}")
        sys.exit(1)
    
    # Create detector (verbose=False to suppress internal pipeline messages)
    max_dim = args.max_dimension if args.max_dimension > 0 else None
    detector = ArrowDetector(
        similarity_threshold=args.threshold,
        max_dimension=max_dim,
        verbose=False
    )
    
    if not args.quiet:
        print("=" * 50)
        print("Arrow Detection")
        print("=" * 50)
        print(f"  Threshold: {args.threshold}")
        print(f"  Max dimension: {max_dim or 'disabled'}")
        print(f"  Output: {output_dir}")
        print()
    
    # Process input
    if input_path.is_file():
        process_image(input_path, detector, output_dir, 
                     args.save_binarization, verbose=not args.quiet)
    elif input_path.is_dir():
        process_directory(input_path, detector, output_dir, args.save_binarization)
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
