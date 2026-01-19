#!/usr/bin/env python3
"""
Debug tool for signature analysis.

Visualizes discrete vs theoretical signatures to help understand
why certain arrows are or aren't being detected.

Usage:
    python debug_signatures.py <image_path> [candidate_index]
    
Examples:
    python debug_signatures.py pics/pic3.png          # Auto-select best candidate
    python debug_signatures.py pics/pic3.png 50       # Analyze candidate #50
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pipeline import ArrowDetector
import sys


def find_best_candidate(detector, all_candidates):
    """Find the candidate with highest recognition score."""
    best_score = -1
    best_idx = 0
    best_candidate = None
    best_keypoints = None
    
    for i, candidate in enumerate(all_candidates):
        key_points = detector.keypoints_selector.select_key_points(candidate)
        if key_points is None:
            continue
            
        # Quick validation
        A, B, C = key_points.A, key_points.B, key_points.C
        a = detector._euclidean_distance(B, C)
        b = detector._euclidean_distance(A, C)
        if a <= 0 or b <= 0:
            continue
        isosceles_ratio = abs(a - b) / max(a, b)
        if isosceles_ratio > 0.5:
            continue
            
        score = detector._compute_recognition_score(key_points, candidate)
        if score > best_score:
            best_score = score
            best_idx = i
            best_candidate = candidate
            best_keypoints = key_points
    
    return best_idx, best_candidate, best_keypoints, best_score


def debug_candidate_signatures(image_path: str, candidate_index: int = None):
    """
    Debug signatures for a specific candidate or auto-select the best one.

    Args:
        image_path: Path to image
        candidate_index: Which candidate to debug (None = auto-select best)
    """
    print("=" * 70)
    print("SIGNATURE DEBUGGING")
    print("=" * 70)
    print()

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return

    print(f"Image: {image_path}")
    print(f"Size: {image.shape[1]}x{image.shape[0]}")
    print()

    # Create detector (verbose=False to suppress internal messages)
    detector = ArrowDetector(similarity_threshold=0.85, verbose=False)

    # Get binarization levels
    binary_images = detector.binarizer.binarize(image)

    # Extract candidates
    all_candidates = []
    for level, binary_img in enumerate(binary_images, start=1):
        candidates = detector.cc_extractor.extract(binary_img, level)
        all_candidates.extend(candidates)

    print(f"Total candidates: {len(all_candidates)}")

    if len(all_candidates) == 0:
        print("No candidates found in image.")
        return

    # Auto-select best candidate if not specified
    if candidate_index is None:
        print("Auto-selecting best candidate...")
        candidate_index, candidate, key_points, auto_score = find_best_candidate(detector, all_candidates)
        if candidate is None:
            print("No valid candidates found.")
            return
        print(f"Selected candidate {candidate_index} (score: {auto_score:.3f})")
    else:
        if candidate_index >= len(all_candidates):
            print(f"Error: Candidate index {candidate_index} out of range (max {len(all_candidates)-1})")
            return
        candidate = all_candidates[candidate_index]
        key_points = detector.keypoints_selector.select_key_points(candidate)

    print()
    print(f"Candidate {candidate_index}:")
    print(f"  Level: {candidate.binarization_level}")
    print(f"  Points: {len(candidate.points)}")
    print()

    if key_points is None:
        print("ERROR: Could not find key points A, B, C")
        return

    A, B, C = key_points.A, key_points.B, key_points.C

    print(f"Key Points:")
    print(f"  A (base): {A}")
    print(f"  B (base): {B}")
    print(f"  C (tip):  {C}")
    print()

    # Compute triangle properties
    a = detector._euclidean_distance(B, C)
    b = detector._euclidean_distance(A, C)
    c = detector._euclidean_distance(A, B)

    print(f"Triangle sides:")
    print(f"  BC (a): {a:.1f}")
    print(f"  AC (b): {b:.1f}")
    print(f"  AB (c): {c:.1f}")
    print(f"  Isosceles ratio: {abs(a-b)/max(a,b):.3f}")
    print()

    # Compute signatures
    print("Computing signatures...")
    V_A = detector._compute_discrete_signature(key_points.A, candidate.points, num_bins=360)
    V_B = detector._compute_discrete_signature(key_points.B, candidate.points, num_bins=360)
    V_C = detector._compute_discrete_signature(key_points.C, candidate.points, num_bins=360)

    S_A = detector._compute_theoretical_signature(key_points, 'A', num_bins=360)
    S_B = detector._compute_theoretical_signature(key_points, 'B', num_bins=360)
    S_C = detector._compute_theoretical_signature(key_points, 'C', num_bins=360)

    # Compute statistics
    print("\nDiscrete signature coverage:")
    print(f"  V_A: {np.sum(V_A > 0)}/360 bins ({100*np.sum(V_A > 0)/360:.1f}%)")
    print(f"  V_B: {np.sum(V_B > 0)}/360 bins ({100*np.sum(V_B > 0)/360:.1f}%)")
    print(f"  V_C: {np.sum(V_C > 0)}/360 bins ({100*np.sum(V_C > 0)/360:.1f}%)")

    print("\nTheoretical signature coverage:")
    print(f"  S_A: {np.sum(S_A > 0)}/360 bins ({100*np.sum(S_A > 0)/360:.1f}%)")
    print(f"  S_B: {np.sum(S_B > 0)}/360 bins ({100*np.sum(S_B > 0)/360:.1f}%)")
    print(f"  S_C: {np.sum(S_C > 0)}/360 bins ({100*np.sum(S_C > 0)/360:.1f}%)")
    print()

    # Compute similarity for each point
    sim_A = detector._tanimoto_similarity(V_A, S_A)
    sim_B = detector._tanimoto_similarity(V_B, S_B)
    sim_C = detector._tanimoto_similarity(V_C, S_C)

    print(f"Individual Tanimoto similarities:")
    print(f"  A: {sim_A:.3f}")
    print(f"  B: {sim_B:.3f}")
    print(f"  C: {sim_C:.3f}")
    
    recognition_score = (sim_A + sim_B + sim_C) / 3.0
    print(f"  Average (pipeline score): {recognition_score:.3f}")
    print()

    # Create output directory
    output_dir = Path("debug_output")
    output_dir.mkdir(exist_ok=True)

    # Visualize signatures
    print("Creating visualization...")

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Signature Analysis - Candidate {candidate_index} (Score: {recognition_score:.3f})', fontsize=16)

    angles = np.arange(360)

    # Point A signatures
    axes[0, 0].plot(angles, V_A, 'b-', label='Discrete V_A', linewidth=2)
    axes[0, 0].plot(angles, S_A, 'r--', label='Theoretical S_A', linewidth=2)
    axes[0, 0].set_title(f'Point A (base) - Similarity: {sim_A:.3f}')
    axes[0, 0].set_xlabel('Angle (degrees)')
    axes[0, 0].set_ylabel('Normalized distance')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Point B signatures
    axes[1, 0].plot(angles, V_B, 'b-', label='Discrete V_B', linewidth=2)
    axes[1, 0].plot(angles, S_B, 'r--', label='Theoretical S_B', linewidth=2)
    axes[1, 0].set_title(f'Point B (base) - Similarity: {sim_B:.3f}')
    axes[1, 0].set_xlabel('Angle (degrees)')
    axes[1, 0].set_ylabel('Normalized distance')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Point C signatures
    axes[2, 0].plot(angles, V_C, 'b-', label='Discrete V_C', linewidth=2)
    axes[2, 0].plot(angles, S_C, 'r--', label='Theoretical S_C', linewidth=2)
    axes[2, 0].set_title(f'Point C (tip) - Similarity: {sim_C:.3f}')
    axes[2, 0].set_xlabel('Angle (degrees)')
    axes[2, 0].set_ylabel('Normalized distance')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Superimposed signatures
    V_combined = np.maximum.reduce([V_A, V_B, V_C])
    S_combined = np.maximum.reduce([S_A, S_B, S_C])

    axes[0, 1].plot(angles, V_combined, 'b-', label='Discrete (superimposed)', linewidth=2)
    axes[0, 1].plot(angles, S_combined, 'r--', label='Theoretical (superimposed)', linewidth=2)
    axes[0, 1].set_title('Superimposed Signatures')
    axes[0, 1].set_xlabel('Angle (degrees)')
    axes[0, 1].set_ylabel('Normalized distance')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Individual similarities bar chart
    similarities = [sim_A, sim_B, sim_C]
    labels = ['Point A', 'Point B', 'Point C']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    bars = axes[1, 1].bar(labels, similarities, color=colors, edgecolor='black', linewidth=1.5)
    axes[1, 1].axhline(y=recognition_score, color='purple', linestyle='--', linewidth=2, 
                       label=f'Average: {recognition_score:.3f}')
    axes[1, 1].axhline(y=0.7, color='orange', linestyle=':', linewidth=2, label='Threshold: 0.70')
    axes[1, 1].set_title('Individual Point Similarities')
    axes[1, 1].set_ylabel('Tanimoto Similarity')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    for bar, sim in zip(bars, similarities):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        f'{sim:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Visualize the arrow shape
    min_y, min_x = candidate.points.min(axis=0)
    max_y, max_x = candidate.points.max(axis=0)

    pad = 20
    min_y = max(0, min_y - pad)
    min_x = max(0, min_x - pad)
    max_y = min(image.shape[0], max_y + pad)
    max_x = min(image.shape[1], max_x + pad)

    region = image[min_y:max_y, min_x:max_x].copy()
    if len(region.shape) == 3:
        region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)

    A_local = (A[1] - min_x, A[0] - min_y)
    B_local = (B[1] - min_x, B[0] - min_y)
    C_local = (C[1] - min_x, C[0] - min_y)

    region_vis = region.copy()
    if len(region_vis.shape) == 2:
        region_vis = cv2.cvtColor(region_vis, cv2.COLOR_GRAY2RGB)

    pts = np.array([C_local, A_local, B_local], np.int32)
    cv2.polylines(region_vis, [pts], True, (0, 255, 0), 2)

    cv2.circle(region_vis, C_local, 5, (255, 0, 0), -1)
    cv2.circle(region_vis, A_local, 5, (0, 255, 0), -1)
    cv2.circle(region_vis, B_local, 5, (0, 255, 0), -1)

    cv2.putText(region_vis, 'C', (C_local[0]+8, C_local[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    cv2.putText(region_vis, 'A', (A_local[0]+8, A_local[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(region_vis, 'B', (B_local[0]+8, B_local[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    axes[2, 1].imshow(region_vis)
    axes[2, 1].set_title('Candidate with Key Points')
    axes[2, 1].axis('off')

    plt.tight_layout()
    output_path = output_dir / "signature_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.show()

    # Diagnostic checks
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    if recognition_score >= 0.7:
        print(f"\n✓ Score {recognition_score:.3f} >= 0.7 threshold - would be DETECTED")
    else:
        print(f"\n✗ Score {recognition_score:.3f} < 0.7 threshold - would be REJECTED")
        
    # Find weakest point
    min_sim = min(sim_A, sim_B, sim_C)
    if min_sim == sim_A:
        weak_point = "A (base)"
    elif min_sim == sim_B:
        weak_point = "B (base)"
    else:
        weak_point = "C (tip)"
    
    print(f"  Weakest point: {weak_point} ({min_sim:.3f})")
    
    # Coverage analysis
    discrete_coverage = (np.sum(V_A > 0) + np.sum(V_B > 0) + np.sum(V_C > 0)) / 3
    theoretical_coverage = (np.sum(S_A > 0) + np.sum(S_B > 0) + np.sum(S_C > 0)) / 3
    print(f"  Avg discrete coverage: {discrete_coverage:.0f}/360 bins")
    print(f"  Avg theoretical coverage: {theoretical_coverage:.0f}/360 bins")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_signatures.py <image_path> [candidate_index]")
        print()
        print("Examples:")
        print("  python debug_signatures.py pics/pic3.png          # Auto-select best")
        print("  python debug_signatures.py pics/pic3.png 50       # Specific candidate")
        sys.exit(1)

    image_path = sys.argv[1]
    candidate_idx = int(sys.argv[2]) if len(sys.argv) > 2 else None

    debug_candidate_signatures(image_path, candidate_idx)
