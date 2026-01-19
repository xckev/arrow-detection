"""
Arrow Detection Pipeline
Based on: "Overlaid Arrow Detection for Labeling Regions of Interest in Biomedical Images"
by K.C. Santosh et al., IEEE Intelligent Systems 2016

This implements a template-free, geometric signature-based technique for detecting
arrow annotations on biomedical images.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy.ndimage import label
import math


@dataclass
class ArrowCandidate:
    """Represents a potential arrow candidate"""
    points: np.ndarray  # All points in the connected component
    binarization_level: int  # Which binarization level it came from
    label_id: int  # Connected component label


@dataclass
class ArrowKeyPoints:
    """Key points representing an arrowhead (triangle ABC)"""
    A: Tuple[int, int]  # First base point
    B: Tuple[int, int]  # Second base point
    C: Tuple[int, int]  # Tip point
    symmetry_angle: float  # Angle of symmetry axis


@dataclass
class DetectedArrow:
    """A detected arrow with all its properties"""
    tip: Tuple[int, int]  # Arrow tip location (point C)
    base_points: Tuple[Tuple[int, int], Tuple[int, int]]  # Points A and B
    direction: float  # Direction angle in radians
    recognition_score: float  # Similarity score
    binarization_level: int  # Which level it was detected at

    def get_position(self) -> Tuple[int, int]:
        """Returns the arrow tip position"""
        return self.tip

    def get_direction_degrees(self) -> float:
        """Returns direction in degrees"""
        return math.degrees(self.direction)


class FuzzyBinarizer:
    """
    Fuzzy binarization using Z-function on 2D histogram.
    Produces multiple levels including image inversion, percentile-based, and color-based detection.
    """

    def __init__(self):
        self.num_levels = 12  # 4 Z-function + 4 percentile + 2 adaptive + 2 color-based

    def binarize(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Apply fuzzy binarization at multiple levels.

        Args:
            image: Input image (color or grayscale)

        Returns:
            List of binary images (12 levels)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            is_color = True
        else:
            gray = image.copy()
            is_color = False

        # Compute 2D histogram considering intensity and local variation
        # Local variation approximated using gradient magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        if gradient_mag.max() > 0:
            gradient_mag = np.uint8(255 * gradient_mag / gradient_mag.max())
        else:
            gradient_mag = np.zeros_like(gray)

        # Compute 2D histogram (intensity vs local variation)
        hist_2d, _, _ = np.histogram2d(
            gray.ravel(),
            gradient_mag.ravel(),
            bins=[256, 256]
        )

        # Apply fuzzy entropy-based thresholding with Z-function
        binary_images = []

        # Compute thresholds once (reused for normal and inverted)
        threshold_low = self._compute_threshold_zfunction(hist_2d, level='low')
        threshold_high = self._compute_threshold_zfunction(hist_2d, level='high')
        gray_inv = 255 - gray

        # Level 1-2: Z-function thresholds (original approach)
        binary_images.append(self._apply_threshold(gray, threshold_low))
        binary_images.append(self._apply_threshold(gray, threshold_high))
        
        # Level 3-4: Z-function on inverted image
        binary_images.append(self._apply_threshold(gray_inv, threshold_low))
        binary_images.append(self._apply_threshold(gray_inv, threshold_high))
        
        # Level 5-6: Percentile-based thresholds for bright objects
        # These help isolate bright arrows that are brighter than surroundings
        # but may not be captured by Z-function thresholds
        pct_85 = np.percentile(gray, 85)
        pct_92 = np.percentile(gray, 92)
        binary_images.append((gray > pct_85).astype(np.uint8) * 255)
        binary_images.append((gray > pct_92).astype(np.uint8) * 255)
        
        # Level 7-8: Percentile-based on inverted (for dark objects)
        pct_inv_85 = np.percentile(gray_inv, 85)
        pct_inv_92 = np.percentile(gray_inv, 92)
        binary_images.append((gray_inv > pct_inv_85).astype(np.uint8) * 255)
        binary_images.append((gray_inv > pct_inv_92).astype(np.uint8) * 255)
        
        # Level 9-10: Adaptive thresholding - detects locally bright/dark regions
        # This helps with arrows that are bright relative to local surroundings
        # but not globally bright (like thin white arrows on variable backgrounds)
        adaptive_mean = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, -10
        )
        adaptive_gaussian = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -10
        )
        binary_images.append(adaptive_mean)
        binary_images.append(adaptive_gaussian)
        
        # Level 11-12: Color-based binarization for colored arrows
        if is_color:
            color_binaries = self._color_based_binarization(image)
            binary_images.extend(color_binaries)
        else:
            # Add empty levels for grayscale images to maintain consistent indexing
            binary_images.append(np.zeros_like(gray))
            binary_images.append(np.zeros_like(gray))

        return binary_images
    
    def _color_based_binarization(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect colored regions (arrows) using saturation and color channels.
        
        Colored arrows (green, red, yellow, etc.) have high saturation compared
        to grayscale medical image backgrounds.
        """
        binary_images = []
        
        # Convert to HSV for saturation-based detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        
        # High saturation regions (colored objects)
        sat_threshold = max(30, np.percentile(saturation, 90))
        binary_sat = (saturation > sat_threshold).astype(np.uint8) * 255
        binary_images.append(binary_sat)
        
        # Color channel difference - colored objects have high difference between channels
        b, g, r = cv2.split(image)
        # Max difference between any two channels
        diff_rg = np.abs(r.astype(np.int16) - g.astype(np.int16))
        diff_rb = np.abs(r.astype(np.int16) - b.astype(np.int16))
        diff_gb = np.abs(g.astype(np.int16) - b.astype(np.int16))
        max_diff = np.maximum(np.maximum(diff_rg, diff_rb), diff_gb).astype(np.uint8)
        
        diff_threshold = max(30, np.percentile(max_diff, 85))
        binary_diff = (max_diff > diff_threshold).astype(np.uint8) * 255
        binary_images.append(binary_diff)
        
        return binary_images

    def _compute_threshold_zfunction(self, hist_2d: np.ndarray, level: str) -> int:
        """
        Compute adaptive threshold using Z-function with fuzzy entropy.

        Args:
            hist_2d: 2D histogram
            level: 'low' or 'high' cut

        Returns:
            Threshold value
        """
        # Simplified fuzzy entropy optimization
        # In practice, this would involve more sophisticated fuzzy partition

        # Compute 1D marginal histogram
        hist_1d = hist_2d.sum(axis=1)
        hist_1d = hist_1d / hist_1d.sum()  # Normalize

        # Find threshold maximizing fuzzy entropy
        best_threshold = 0
        max_entropy = -np.inf

        for t in range(1, 255):
            # Z-function parameters
            if level == 'low':
                a, b = t - 30, t
            else:  # high
                a, b = t, t + 30

            # Compute fuzzy membership
            entropy = self._fuzzy_entropy(hist_1d, a, b)

            if entropy > max_entropy:
                max_entropy = entropy
                best_threshold = t

        return best_threshold

    def _fuzzy_entropy(self, hist: np.ndarray, a: int, b: int) -> float:
        """Compute fuzzy entropy for Z-function with parameters a, b"""
        if a >= b or a < 0 or b >= 256:
            return -np.inf

        entropy = 0.0
        for i in range(256):
            if hist[i] > 0:
                # Z-function membership
                if i <= a:
                    mu = 1.0
                elif i >= b:
                    mu = 0.0
                else:
                    mu = 1.0 - (i - a) / (b - a)

                # Shannon entropy
                if 0 < mu < 1:
                    entropy -= mu * np.log(mu) + (1 - mu) * np.log(1 - mu)

        return entropy

    def _apply_threshold(self, image: np.ndarray, threshold: int) -> np.ndarray:
        """Apply threshold to create binary image"""
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return binary


class ConnectedComponentsExtractor:
    """Extract connected components from binary images"""

    def extract(self, binary_image: np.ndarray, level: int) -> List[ArrowCandidate]:
        """
        Extract connected components from binary image.

        Args:
            binary_image: Binary image
            level: Binarization level

        Returns:
            List of arrow candidates
        """
        # Invert the binary image to find both black and white arrows
        # Try both the original and inverted to catch arrows of either color
        binary_inverted = 255 - binary_image

        # Label connected components on both versions
        labeled_array, num_features = label(binary_image)
        labeled_array_inv, num_features_inv = label(binary_inverted)

        candidates = []

        # Process both original and inverted to find arrows of any color
        for labeled, num_feat in [(labeled_array, num_features), (labeled_array_inv, num_features_inv)]:
            for label_id in range(1, num_feat + 1):
                # Get points for this component
                points = np.argwhere(labeled == label_id)

                # Filter by size (ignore very small and very large components)
                if len(points) < 20:  # Minimum arrow size
                    continue

                # Reject huge regions (likely background, not arrows)
                max_arrow_size = binary_image.shape[0] * binary_image.shape[1] * 0.2
                if len(points) > max_arrow_size:
                    continue

                candidates.append(ArrowCandidate(
                    points=points,
                    binarization_level=level,
                    label_id=label_id
                ))

        return candidates


class KeyPointsSelector:
    """
    Select key points (A, B, C) representing arrowhead using orthogonal scanning.
    """

    def select_key_points(self, candidate: ArrowCandidate) -> Optional[ArrowKeyPoints]:
        """
        Perform orthogonal scanning to find key points.

        Args:
            candidate: Arrow candidate

        Returns:
            ArrowKeyPoints if found, None otherwise
        """
        points = candidate.points

        # Convert to image coordinates for easier processing
        min_y, min_x = points.min(axis=0)
        max_y, max_x = points.max(axis=0)

        # Create binary mask
        mask = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.uint8)
        for y, x in points:
            mask[y - min_y, x - min_x] = 1

        # Perform 4 orthogonal scans
        scan_lists = []

        # Scan 1: Top-left corner, column-wise
        scan_lists.append(self._scan_columnwise(mask, from_top=True))

        # Scan 2: Top-left corner, row-wise
        scan_lists.append(self._scan_rowwise(mask, from_left=True))

        # Scan 3: Bottom-right corner, column-wise
        scan_lists.append(self._scan_columnwise(mask, from_top=False))

        # Scan 4: Bottom-right corner, row-wise
        scan_lists.append(self._scan_rowwise(mask, from_left=False))

        # Combine all scanned points
        all_scan_points = []
        for scan_list in scan_lists:
            all_scan_points.extend(scan_list)

        if len(all_scan_points) < 3:
            return None

        all_scan_points = np.array(all_scan_points)

        # Find point C using symmetry
        C, symmetry_angle = self._find_tip_point(all_scan_points, mask)

        if C is None:
            return None

        # Create local coordinate points for the component
        local_points = points - np.array([min_y, min_x])
        
        # Find points A and B (base of arrow, far from tip)
        A, B = self._find_base_points(all_scan_points, C, symmetry_angle, local_points)

        if A is None or B is None:
            return None

        # Convert back to original image coordinates
        C_orig = (C[0] + min_y, C[1] + min_x)
        A_orig = (A[0] + min_y, A[1] + min_x)
        B_orig = (B[0] + min_y, B[1] + min_x)

        return ArrowKeyPoints(
            A=A_orig,
            B=B_orig,
            C=C_orig,
            symmetry_angle=symmetry_angle
        )

    def _scan_columnwise(self, mask: np.ndarray, from_top: bool) -> List[Tuple[int, int]]:
        """Scan column-wise to collect boundary points"""
        points = []
        h, w = mask.shape

        for x in range(w):
            if from_top:
                for y in range(h):
                    if mask[y, x] == 1:
                        points.append((y, x))
                        break
            else:
                for y in range(h - 1, -1, -1):
                    if mask[y, x] == 1:
                        points.append((y, x))
                        break

        return points

    def _scan_rowwise(self, mask: np.ndarray, from_left: bool) -> List[Tuple[int, int]]:
        """Scan row-wise to collect boundary points"""
        points = []
        h, w = mask.shape

        for y in range(h):
            if from_left:
                for x in range(w):
                    if mask[y, x] == 1:
                        points.append((y, x))
                        break
            else:
                for x in range(w - 1, -1, -1):
                    if mask[y, x] == 1:
                        points.append((y, x))
                        break

        return points

    def _find_tip_point(self, points: np.ndarray, mask: np.ndarray) -> Tuple[Optional[Tuple[int, int]], float]:
        """
        Find tip point C using PCA to find arrow direction.

        Returns:
            Tuple of (point C, symmetry angle)
        """
        if len(points) == 0:
            return None, 0.0

        # Use PCA to find the principal axis (arrow direction)
        centroid = points.mean(axis=0)
        centered = points - centroid

        # Compute covariance matrix
        cov = np.cov(centered.T)

        # Get principal components
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Principal axis is the eigenvector with largest eigenvalue
        principal_idx = np.argmax(eigenvalues)
        principal_axis = eigenvectors[:, principal_idx]

        # Project all points onto principal axis
        projections = np.dot(centered, principal_axis)

        # Find point farthest along principal axis (positive direction)
        # and farthest in negative direction
        max_idx = np.argmax(projections)
        min_idx = np.argmin(projections)

        pos_extreme = points[max_idx]
        neg_extreme = points[min_idx]

        # Also get convex hull for sharp corner detection
        try:
            hull = cv2.convexHull(points.astype(np.float32))
            hull_points = hull.squeeze()

            if len(hull_points.shape) == 1:
                hull_points = hull_points.reshape(1, -1)
        except (cv2.error, ValueError):
            # If hull fails, use extreme point
            tip_point = tuple(pos_extreme.astype(int))
            angle = np.arctan2(principal_axis[0], principal_axis[1])
            return tip_point, angle

        # Check both extremes to find which is the arrowhead
        # Arrowhead should be the sharpest point among hull vertices
        def get_corner_angle(point_idx, hull_pts):
            """Compute angle at a hull vertex"""
            n = len(hull_pts)
            prev = np.array(hull_pts[(point_idx - 1) % n])
            curr = np.array(hull_pts[point_idx])
            next_pt = np.array(hull_pts[(point_idx + 1) % n])

            v1 = prev - curr
            v2 = next_pt - curr

            cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            cos_ang = np.clip(cos_ang, -1, 1)
            return np.degrees(np.arccos(cos_ang))

        # Find hull indices closest to extreme points
        hull_points_list = hull_points.tolist() if len(hull_points) > 1 else [hull_points.tolist()]

        def find_closest_hull_idx(target_point, hull_pts_list):
            """Find index in hull closest to target point"""
            min_dist = np.inf
            best_idx = 0
            for idx, hpt in enumerate(hull_pts_list):
                dist = np.linalg.norm(np.array(hpt) - target_point)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx
            return best_idx

        pos_hull_idx = find_closest_hull_idx(pos_extreme, hull_points_list)
        neg_hull_idx = find_closest_hull_idx(neg_extreme, hull_points_list)

        pos_angle = get_corner_angle(pos_hull_idx, hull_points_list)
        neg_angle = get_corner_angle(neg_hull_idx, hull_points_list)

        # Count nearby points to estimate local "width" at each end
        # Arrow tips are narrow, bases are wider
        def count_nearby_points(extreme_pt, radius=20):
            dists = np.linalg.norm(points - extreme_pt, axis=1)
            return np.sum(dists < radius)

        pos_width = count_nearby_points(pos_extreme)
        neg_width = count_nearby_points(neg_extreme)

        # Choose the sharper AND narrower endpoint as the tip
        # Sharp angle < 60° is strong indicator of tip
        # Otherwise use width as tiebreaker
        pos_score = (60 - pos_angle) / 60.0 if pos_angle < 60 else 0  # Normalized sharpness
        neg_score = (60 - neg_angle) / 60.0 if neg_angle < 60 else 0

        # Add width penalty (narrower is better for tip)
        pos_score += 0.5 * (1 - pos_width / (pos_width + neg_width + 1))
        neg_score += 0.5 * (1 - neg_width / (pos_width + neg_width + 1))

        if pos_score > neg_score:
            tip_point = tuple(pos_extreme.astype(int))
            tip_angle = np.arctan2(principal_axis[0], principal_axis[1])
        else:
            tip_point = tuple(neg_extreme.astype(int))
            tip_angle = np.arctan2(-principal_axis[0], -principal_axis[1])

        return tip_point, tip_angle

    def _find_base_points(self, points: np.ndarray, C: Tuple[int, int],
                         symmetry_angle: float, all_component_points: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """
        Find base points A and B at the arrowhead's base (the "barbs").
        
        According to the paper, A and B are at the base of the isosceles triangle
        that forms the arrowhead - the widest points of the head where it meets
        the tail. These are the "barb" points.

        Args:
            points: Scanned boundary points
            C: Tip point  
            symmetry_angle: Angle of symmetry axis (pointing from base toward tip)
            all_component_points: All points in the connected component

        Returns:
            Tuple of (point A, point B)
        """
        C_array = np.array([C[0], C[1]], dtype=float)
        
        # Direction along arrow axis (from tip toward tail)
        axis_dir = np.array([np.sin(symmetry_angle + np.pi), np.cos(symmetry_angle + np.pi)])
        
        # Direction perpendicular to arrow axis
        perp_dir = np.array([np.sin(symmetry_angle + np.pi/2), np.cos(symmetry_angle + np.pi/2)])
        
        # Project all points along the arrow axis and perpendicular to it
        vectors = all_component_points - C_array
        axis_projections = np.dot(vectors, axis_dir)
        perp_projections = np.dot(vectors, perp_dir)
        
        # Find the widest point perpendicular to the axis for each position along axis
        # The arrowhead barbs are where width is maximum
        
        # Bin points by their axis position
        max_axis = np.max(axis_projections)
        if max_axis < 1:
            max_axis = 1
            
        num_bins = 20
        bin_indices = np.clip((axis_projections / max_axis * num_bins).astype(int), 0, num_bins - 1)
        
        # Find max perpendicular extent (width) for each bin
        widths = np.zeros(num_bins)
        for i in range(num_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                widths[i] = np.max(np.abs(perp_projections[mask]))
        
        # The arrowhead barbs are at the bin with maximum width
        # This is typically near the front of the arrow (close to tip)
        # Find the first major width peak (the barbs)
        max_width_bin = np.argmax(widths)
        
        # Find points near this axis position with extreme perpendicular positions
        axis_pos_at_max_width = (max_width_bin + 0.5) / num_bins * max_axis
        axis_tolerance = max_axis / num_bins * 2  # Allow some tolerance
        
        near_barb_mask = np.abs(axis_projections - axis_pos_at_max_width) < axis_tolerance
        
        if np.sum(near_barb_mask) < 2:
            # Fallback: use widest points overall
            near_barb_mask = np.ones(len(all_component_points), dtype=bool)
        
        # Find the two extreme points perpendicular to axis
        masked_perp = perp_projections.copy()
        masked_perp[~near_barb_mask] = 0
        max_perp_idx = np.argmax(masked_perp)
        
        masked_perp_neg = perp_projections.copy()
        masked_perp_neg[~near_barb_mask] = 0
        min_perp_idx = np.argmin(masked_perp_neg)
        
        A = tuple(all_component_points[max_perp_idx].astype(int))
        B = tuple(all_component_points[min_perp_idx].astype(int))
        
        return A, B


class ArrowDetector:
    """
    Main arrow detection class implementing the complete pipeline.
    """

    def __init__(self, similarity_threshold: float = 0.85, max_dimension: int = 720,
                 verbose: bool = True):
        """
        Initialize arrow detector.

        Args:
            similarity_threshold: Threshold for arrow detection (λ in paper)
            max_dimension: Maximum dimension (width or height) for processing.
                          Images larger than this will be resized to speed up detection.
                          Set to None to disable resizing.
            verbose: Print progress messages during detection
        """
        self.threshold = similarity_threshold
        self.max_dimension = max_dimension
        self.verbose = verbose
        self.binarizer = FuzzyBinarizer()
        self.cc_extractor = ConnectedComponentsExtractor()
        self.keypoints_selector = KeyPointsSelector()

    def _resize_if_needed(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Resize image if it exceeds max_dimension.
        
        Returns:
            Tuple of (resized_image, scale_factor)
            scale_factor is used to convert coordinates back to original size
        """
        if self.max_dimension is None:
            return image, 1.0
            
        h, w = image.shape[:2]
        max_dim = max(h, w)
        
        if max_dim <= self.max_dimension:
            return image, 1.0
        
        scale = self.max_dimension / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale

    def detect_arrows(self, image: np.ndarray) -> List[DetectedArrow]:
        """
        Detect arrows in biomedical image.

        Args:
            image: Input image (grayscale or color)

        Returns:
            List of detected arrows with positions, directions, and scores
        """
        # Resize image if too large (for speed)
        processed_image, scale = self._resize_if_needed(image)
        
        # Step 1: Fuzzy binarization at multiple levels
        binary_images = self.binarizer.binarize(processed_image)

        # Step 2: Extract connected components from all levels
        all_candidates = []
        for level, binary_img in enumerate(binary_images, start=1):
            candidates = self.cc_extractor.extract(binary_img, level)
            all_candidates.extend(candidates)

        if self.verbose:
            print(f"Found {len(all_candidates)} candidates across all binarization levels")

        # Get image shape for validation (use processed/resized image)
        if len(processed_image.shape) == 3:
            img_height, img_width = processed_image.shape[:2]
        else:
            img_height, img_width = processed_image.shape
        image_shape = (img_height, img_width)

        # Step 3-7: Process each candidate
        detected_arrows = []

        for candidate in all_candidates:
            # Step 3: Key points selection
            key_points = self.keypoints_selector.select_key_points(candidate)

            if key_points is None:
                continue

            # Step 4: Candidate selection (symmetry and overlap checks)
            if not self._validate_candidate(key_points, candidate, image_shape):
                continue

            # Step 5-6: Signature comparison
            recognition_score = self._compute_recognition_score(key_points, candidate)

            # Step 7: Decision based on threshold
            if recognition_score >= self.threshold:
                # Compute arrow direction
                direction = self._compute_direction(key_points)
                
                # Scale coordinates back to original image size
                if scale != 1.0:
                    inv_scale = 1.0 / scale
                    tip = (int(key_points.C[0] * inv_scale), int(key_points.C[1] * inv_scale))
                    A = (int(key_points.A[0] * inv_scale), int(key_points.A[1] * inv_scale))
                    B = (int(key_points.B[0] * inv_scale), int(key_points.B[1] * inv_scale))
                else:
                    tip = key_points.C
                    A = key_points.A
                    B = key_points.B

                detected_arrows.append(DetectedArrow(
                    tip=tip,
                    base_points=(A, B),
                    direction=direction,
                    recognition_score=recognition_score,
                    binarization_level=candidate.binarization_level
                ))

        # Step 8: Eliminate redundant detections
        final_arrows = self._eliminate_redundancy(detected_arrows)

        if self.verbose:
            print(f"Detected {len(final_arrows)} arrows after redundancy elimination")

        return final_arrows

    def _validate_candidate(self, key_points: ArrowKeyPoints, candidate: ArrowCandidate, 
                            image_shape: Tuple[int, int] = None) -> bool:
        """
        Validate candidate based on the paper's criteria.
        
        The paper uses two main geometric criteria before signature comparison:
        1. Isosceles check: d(A,C) ≈ d(B,C) - arrow modeled as isosceles triangle
        2. Valid triangle: non-degenerate triangle with positive area

        Args:
            key_points: Key points A, B, C
            candidate: Arrow candidate
            image_shape: Not used (kept for API compatibility)

        Returns:
            True if valid, False otherwise
        """
        A, B, C = key_points.A, key_points.B, key_points.C

        # Compute distances (paper notation: a = d(B,C), b = d(A,C), c = d(A,B))
        a = self._euclidean_distance(B, C)
        b = self._euclidean_distance(A, C)
        c = self._euclidean_distance(A, B)

        # Check if it forms a valid triangle (non-degenerate)
        if a <= 0 or b <= 0 or c <= 0:
            return False

        # ===== Paper criterion: Isosceles check =====
        # The paper models arrow as isosceles triangle: d(A,C) ≈ d(B,C)
        isosceles_ratio = abs(a - b) / max(a, b)
        if isosceles_ratio > 0.5:  # 50% tolerance for isosceles
            return False

        # Verify triangle inequality (valid triangle)
        l = (a + b + c) / 2  # Semi-perimeter
        if l <= a or l <= b or l <= c:
            return False

        return True

    def _compute_recognition_score(self, key_points: ArrowKeyPoints,
                                   candidate: ArrowCandidate) -> float:
        """
        Compute recognition score by comparing discrete and theoretical signatures.

        Uses Tanimoto index (min/max) for each point, then averages.
        The paper compares discrete V_X with theoretical S_X for X in {A, B, C}.

        Args:
            key_points: Key points A, B, C
            candidate: Arrow candidate

        Returns:
            Recognition score in [0, 1]
        """
        # Compute discrete signatures for A, B, C using ALL points in the component
        V_A = self._compute_discrete_signature(key_points.A, candidate.points)
        V_B = self._compute_discrete_signature(key_points.B, candidate.points)
        V_C = self._compute_discrete_signature(key_points.C, candidate.points)

        # Compute theoretical signatures
        S_A = self._compute_theoretical_signature(key_points, 'A')
        S_B = self._compute_theoretical_signature(key_points, 'B')
        S_C = self._compute_theoretical_signature(key_points, 'C')

        # Compute Tanimoto similarity for each point separately
        sim_A = self._tanimoto_similarity(V_A, S_A)
        sim_B = self._tanimoto_similarity(V_B, S_B)
        sim_C = self._tanimoto_similarity(V_C, S_C)
        
        # Average similarity across all three points
        avg_similarity = (sim_A + sim_B + sim_C) / 3.0

        return max(0.0, min(1.0, avg_similarity))
    
    def _tanimoto_similarity(self, discrete: np.ndarray, theoretical: np.ndarray) -> float:
        """
        Compute Tanimoto similarity between discrete and theoretical signatures.
        
        Uses the standard Tanimoto index across all angular bins where either
        signature is active.
        
        Args:
            discrete: Discrete signature array
            theoretical: Theoretical signature array
            
        Returns:
            Similarity score in [0, 1]
        """
        min_sum = 0.0
        max_sum = 0.0
        
        for i in range(len(discrete)):
            d, t = discrete[i], theoretical[i]
            if d > 0.01 or t > 0.01:  # Either signature is active
                min_sum += min(d, t)
                max_sum += max(d, t)
        
        if max_sum < 1e-10:
            return 0.0
        
        return min_sum / max_sum

    def _compute_discrete_signature(self, point: Tuple[int, int],
                                   all_points: np.ndarray, num_bins: int = 360) -> np.ndarray:
        """
        Compute discrete signature from a point using pencil of lines.

        Args:
            point: Origin point for signature
            all_points: All points in the shape
            num_bins: Number of angular bins

        Returns:
            Signature array
        """
        signature = np.zeros(num_bins)

        # Compute vectors from point to all other points
        point_array = np.array([point[0], point[1]])
        vectors = all_points - point_array

        # Compute distances and angles
        distances = np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2)
        angles = np.arctan2(vectors[:, 0], vectors[:, 1])
        angles = (angles + np.pi) % (2 * np.pi)  # Normalize to [0, 2π)

        # Bin angles and record max distance in each bin
        angle_bins = (angles / (2 * np.pi) * num_bins).astype(int)
        angle_bins = np.clip(angle_bins, 0, num_bins - 1)

        for i, bin_idx in enumerate(angle_bins):
            signature[bin_idx] = max(signature[bin_idx], distances[i])

        # Interpolate gaps in signature (fill zeros between non-zero values)
        # This handles sparse angular coverage in thin shapes like arrows
        signature = self._interpolate_signature_gaps(signature)

        # Normalize
        if signature.max() > 0:
            signature = signature / signature.max()

        return signature
    
    def _interpolate_signature_gaps(self, signature: np.ndarray) -> np.ndarray:
        """
        Interpolate gaps (zeros) in a circular signature array.
        
        For thin shapes like arrows, some angular bins have no points.
        This interpolates between nearby non-zero values to fill small gaps.
        """
        n = len(signature)
        result = signature.copy()
        
        # Find non-zero indices
        non_zero_idx = np.where(signature > 0)[0]
        if len(non_zero_idx) < 2:
            return result
        
        # For each zero bin, interpolate from nearest non-zero neighbors
        for i in range(n):
            if signature[i] == 0:
                # Find nearest non-zero bins (circular)
                left_dist = right_dist = 0
                left_val = right_val = 0
                
                # Search left (decreasing index, wrapping)
                for d in range(1, n):
                    idx = (i - d) % n
                    if signature[idx] > 0:
                        left_dist = d
                        left_val = signature[idx]
                        break
                
                # Search right (increasing index, wrapping)
                for d in range(1, n):
                    idx = (i + d) % n
                    if signature[idx] > 0:
                        right_dist = d
                        right_val = signature[idx]
                        break
                
                # Only interpolate if gap is small (< 30 degrees)
                if left_dist > 0 and right_dist > 0 and (left_dist + right_dist) < 30:
                    # Linear interpolation
                    total_dist = left_dist + right_dist
                    result[i] = (right_dist * left_val + left_dist * right_val) / total_dist
        
        return result

    def _compute_theoretical_signature(self, key_points: ArrowKeyPoints,
                                      point_name: str, num_bins: int = 360,
                                      tail_length_ratio: float = 2.0,
                                      tail_width_ratio: float = 0.2) -> np.ndarray:
        """
        Compute theoretical signature based on geometric arrow model.
        
        The paper models arrows as: Triangle ABC (arrowhead) + Rectangle (tail).
        The signature from each vertex represents the distance to the ENTIRE 
        arrow boundary at each angle θ, not just the triangle.

        Args:
            key_points: Key points A, B, C
            point_name: Which point ('A', 'B', or 'C')
            num_bins: Number of angular bins
            tail_length_ratio: Tail length as multiple of head height
            tail_width_ratio: Tail width as fraction of head base width

        Returns:
            Theoretical signature array (normalized)
        """
        A = np.array([key_points.A[0], key_points.A[1]], dtype=float)
        B = np.array([key_points.B[0], key_points.B[1]], dtype=float)
        C = np.array([key_points.C[0], key_points.C[1]], dtype=float)
        
        # Get the vertex we're computing the signature from
        if point_name == 'A':
            V = A
        elif point_name == 'B':
            V = B
        else:  # C (tip)
            V = C
        
        # Build the complete arrow model: Triangle ABC + Rectangular tail
        # The tail extends from the base AB in the direction opposite to C
        
        # Midpoint of base AB
        M = (A + B) / 2
        
        # Arrow axis direction (from tip C toward tail)
        axis_vec = M - C
        axis_len = np.linalg.norm(axis_vec)
        if axis_len < 1e-6:
            return np.zeros(num_bins)
        axis_unit = axis_vec / axis_len
        
        # Perpendicular direction (from A toward B, normalized)
        perp_vec = B - A
        base_width = np.linalg.norm(perp_vec)
        if base_width < 1e-6:
            return np.zeros(num_bins)
        perp_unit = perp_vec / base_width
        
        # Tail dimensions: much narrower than head (typically 10-20% of head width)
        tail_width = base_width * tail_width_ratio
        tail_length = axis_len * tail_length_ratio
        
        # Define the 4 corners of the rectangular tail
        # Tail starts at base AB and extends away from C
        # D1, D2 are at the base (between A and B), E1, E2 are at the tail end
        D1 = M - perp_unit * (tail_width / 2)  # Base of tail, one side
        D2 = M + perp_unit * (tail_width / 2)  # Base of tail, other side
        E1 = D1 + axis_unit * tail_length      # End of tail, one side
        E2 = D2 + axis_unit * tail_length      # End of tail, other side
        
        # All edges of the arrow model:
        # Triangle edges: CA, CB, (AB is shared with tail base)
        # Tail edges: D1-E1 (left side), E1-E2 (back), D2-E2 (right side)
        # Connection edges: A-D1 and B-D2 (from barbs to tail)
        edges = [
            (C, A),      # Triangle edge CA
            (C, B),      # Triangle edge CB  
            (A, D1),     # Barb to tail connection (left)
            (B, D2),     # Barb to tail connection (right)
            (D1, E1),    # Tail left side
            (E1, E2),    # Tail back
            (D2, E2),    # Tail right side
        ]
        
        # Generate signature: for each angle, find distance to nearest edge
        signature = np.zeros(num_bins)
        
        for i in range(num_bins):
            theta = i * 2 * np.pi / num_bins
            
            # Find minimum positive distance to any edge at this angle
            min_dist = np.inf
            for edge_start, edge_end in edges:
                dist = self._distance_to_edge_at_angle(V, edge_start, edge_end, theta)
                if dist > 1e-6 and dist < min_dist:
                    min_dist = dist
            
            if min_dist < np.inf:
                signature[i] = min_dist
        
        # Apply smoothing to reduce sharp discontinuities at edge transitions
        # This better matches the discrete signature from actual filled shapes
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        # Circular convolution (wrap around for angular data)
        signature_padded = np.concatenate([signature[-kernel_size//2:], signature, signature[:kernel_size//2]])
        signature_smooth = np.convolve(signature_padded, kernel, mode='valid')
        signature = signature_smooth[:num_bins]
        
        # Normalize
        if signature.max() > 0:
            signature = signature / signature.max()
        
        return signature
    
    def _distance_to_edge_at_angle(self, V: np.ndarray, P1: np.ndarray, 
                                   P2: np.ndarray, theta: float) -> float:
        """
        Compute distance from vertex V to edge P1-P2 along direction theta.
        
        This implements ray-line intersection: shoot a ray from V at angle theta
        and find where it intersects the line segment P1-P2.
        
        Args:
            V: Origin vertex (in y, x image coordinates)
            P1, P2: The two vertices defining the opposite edge (in y, x)
            theta: Direction angle in normalized form: (atan2(dy,dx) + π) % 2π
            
        Returns:
            Distance to intersection, or 0 if no valid intersection
        """
        # Convert from normalized theta back to original atan2 angle
        # If normalized = (atan2(dy,dx) + π) % 2π, then original = normalized - π
        # The ray direction for atan2(dy, dx) is [sin(angle), cos(angle)] = [dy/r, dx/r]
        # Using sin(θ-π) = -sin(θ), cos(θ-π) = -cos(θ)
        ray_dir = np.array([-np.sin(theta), -np.cos(theta)])
        
        # Edge direction
        edge_dir = P2 - P1
        edge_len = np.linalg.norm(edge_dir)
        if edge_len < 1e-6:
            return 0.0
        
        # Solve for intersection: V + t * ray_dir = P1 + s * edge_dir
        # This is a 2x2 linear system
        # [ray_dir_x, -edge_dir_x] [t]   [P1_x - V_x]
        # [ray_dir_y, -edge_dir_y] [s] = [P1_y - V_y]
        
        denom = ray_dir[0] * (-edge_dir[1]) - ray_dir[1] * (-edge_dir[0])
        if abs(denom) < 1e-10:
            return 0.0  # Parallel lines
        
        diff = P1 - V
        t = (diff[0] * (-edge_dir[1]) - diff[1] * (-edge_dir[0])) / denom
        s = (ray_dir[0] * diff[1] - ray_dir[1] * diff[0]) / denom
        
        # Check if intersection is valid:
        # t > 0: intersection is in front of the ray origin
        # 0 <= s <= 1: intersection is on the edge segment
        if t > 1e-6 and 0 <= s <= 1:
            return t
        
        return 0.0

    def _compute_direction(self, key_points: ArrowKeyPoints) -> float:
        """
        Compute arrow direction from base midpoint to tip.

        Returns:
            Direction angle in radians
        """
        A, B, C = key_points.A, key_points.B, key_points.C

        # Midpoint of base
        mid_y = (A[0] + B[0]) / 2
        mid_x = (A[1] + B[1]) / 2

        # Direction vector from midpoint to tip
        dy = C[0] - mid_y
        dx = C[1] - mid_x

        # Angle in radians
        angle = math.atan2(dy, dx)

        return angle

    def _eliminate_redundancy(self, arrows: List[DetectedArrow]) -> List[DetectedArrow]:
        """
        Eliminate redundant arrows detected at multiple binarization levels.

        Select best recognition rate for each location.

        Args:
            arrows: List of detected arrows

        Returns:
            Filtered list with redundancy removed
        """
        if len(arrows) == 0:
            return []

        # Group arrows by location (using proximity threshold)
        location_threshold = 20  # pixels

        groups = []
        used = [False] * len(arrows)

        for i, arrow1 in enumerate(arrows):
            if used[i]:
                continue

            # Start new group
            group = [arrow1]
            used[i] = True

            # Find nearby arrows
            for j, arrow2 in enumerate(arrows):
                if used[j]:
                    continue

                dist = self._euclidean_distance(arrow1.tip, arrow2.tip)
                if dist < location_threshold:
                    group.append(arrow2)
                    used[j] = True

            groups.append(group)

        # Select best arrow from each group
        final_arrows = []
        for group in groups:
            best_arrow = max(group, key=lambda a: a.recognition_score)
            final_arrows.append(best_arrow)

        return final_arrows

    @staticmethod
    def _euclidean_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Compute Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def visualize_results(image: np.ndarray, arrows: List[DetectedArrow],
                     output_path: Optional[str] = None) -> np.ndarray:
    """
    Visualize detected arrows on the image.

    Args:
        image: Input image
        arrows: List of detected arrows
        output_path: Optional path to save visualization

    Returns:
        Annotated image
    """
    # Create color image for visualization
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()

    for i, arrow in enumerate(arrows):
        # Draw arrow tip
        cv2.circle(vis_image, (arrow.tip[1], arrow.tip[0]), 5, (0, 0, 255), -1)

        # Draw base points
        A, B = arrow.base_points
        cv2.circle(vis_image, (A[1], A[0]), 3, (0, 255, 0), -1)
        cv2.circle(vis_image, (B[1], B[0]), 3, (0, 255, 0), -1)

        # Draw triangle
        pts = np.array([[arrow.tip[1], arrow.tip[0]], [A[1], A[0]], [B[1], B[0]]], np.int32)
        cv2.polylines(vis_image, [pts], True, (255, 0, 0), 2)

        # Draw direction arrow
        arrow_len = 30
        end_x = int(arrow.tip[1] + arrow_len * math.cos(arrow.direction))
        end_y = int(arrow.tip[0] + arrow_len * math.sin(arrow.direction))
        cv2.arrowedLine(vis_image, (arrow.tip[1], arrow.tip[0]), (end_x, end_y),
                       (255, 255, 0), 2, tipLength=0.3)

        # Add label with score
        label = f"{i+1}: {arrow.recognition_score:.2f}"
        cv2.putText(vis_image, label, (arrow.tip[1] + 10, arrow.tip[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if output_path:
        cv2.imwrite(output_path, vis_image)

    return vis_image


if __name__ == "__main__":
    print("Arrow Detection Pipeline")
    print("Based on Santosh et al., IEEE Intelligent Systems 2016")
    print("\nImplementation complete. Ready to process images.")
