"""
ArUco Marker Detection Module

ArUco marker detection and pose estimation:
- Dictionary management (predefined and custom)
- Corner detection using contour analysis
- Pose estimation from corners (PnP)
- Marker ID decoding
- Sub-pixel corner refinement

Pure Python standard library only.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Set


@dataclass
class Point2D:
    """2D image point."""
    x: float = 0.0
    y: float = 0.0

    def distance_to(self, other: Point2D) -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class MarkerCorner:
    """A corner of an ArUco marker."""
    point: Point2D
    response: float = 0.0
    sub_pixel: Optional[Point2D] = None

    @property
    def refined(self) -> Point2D:
        return self.sub_pixel if self.sub_pixel else self.point


@dataclass
class ArucoMarker:
    """Detected ArUco marker."""
    marker_id: int
    corners: List[MarkerCorner]  # Top-left, top-right, bottom-right, bottom-left
    dictionary_name: str = ""
    confidence: float = 1.0
    rvec: Optional[List[float]] = None  # Rotation vector
    tvec: Optional[List[float]] = None  # Translation vector
    center: Optional[Point2D] = None
    perimeter: float = 0.0

    def __post_init__(self) -> None:
        if len(self.corners) >= 4 and self.center is None:
            cx = sum(c.point.x for c in self.corners) / 4
            cy = sum(c.point.y for c in self.corners) / 4
            self.center = Point2D(cx, cy)
        if len(self.corners) >= 4:
            self.perimeter = sum(
                self.corners[i].point.distance_to(self.corners[(i+1)%4].point)
                for i in range(4)
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.marker_id,
            "dictionary": self.dictionary_name,
            "confidence": self.confidence,
            "corners": [(c.point.x, c.point.y) for c in self.corners],
            "center": (self.center.x, self.center.y) if self.center else None,
            "perimeter": self.perimeter,
        }


class ArucoDictionary:
    """
    ArUco marker dictionary management.

    Provides predefined dictionaries and custom dictionary generation.
    """

    # Predefined dictionary sizes
    PREDEFINED_SIZES = {
        "DICT_4X4_50": (4, 4, 50),
        "DICT_4X4_100": (4, 4, 100),
        "DICT_4X4_250": (4, 4, 250),
        "DICT_4X4_1000": (4, 4, 1000),
        "DICT_5X5_50": (5, 5, 50),
        "DICT_5X5_100": (5, 5, 100),
        "DICT_5X5_250": (5, 5, 250),
        "DICT_5X5_1000": (5, 5, 1000),
        "DICT_6X6_50": (6, 6, 50),
        "DICT_6X6_100": (6, 6, 100),
        "DICT_6X6_250": (6, 6, 250),
        "DICT_6X6_1000": (6, 6, 1000),
    }

    def __init__(self) -> None:
        self._dictionaries: Dict[str, Dict[int, List[List[int]]]] = {}

    def get_dictionary(self, name: str) -> Optional[Dict[int, List[List[int]]]]:
        """Get a dictionary by name, generating if needed."""
        if name not in self._dictionaries:
            if name in self.PREDEFINED_SIZES:
                n, m, count = self.PREDEFINED_SIZES[name]
                self._dictionaries[name] = self._generate_dictionary(n, m, count)
            else:
                return None
        return self._dictionaries[name]

    def _generate_dictionary(self, n: int, m: int, count: int,
                             seed: int = 0) -> Dict[int, List[List[int]]]:
        """Generate an ArUco dictionary with given parameters."""
        rng = random.Random(seed)
        total_bits = n * n * 2  # Each cell has 2 bits (black border, white interior)
        # Actually, ArUco uses a simpler encoding: n*n bits for the inner pattern
        # plus a border. We use a simplified Hamming distance-based generation.

        dictionary: Dict[int, List[List[int]]] = {}
        min_hamming = max(1, (n * n) // 4)  # Minimum Hamming distance

        attempts = 0
        max_attempts = count * 1000

        while len(dictionary) < count and attempts < max_attempts:
            attempts += 1
            # Generate random bit pattern
            pattern = [[rng.randint(0, 1) for _ in range(m)] for _ in range(n)]

            # Check Hamming distance with all existing patterns
            valid = True
            for existing in dictionary.values():
                hamming = self._hamming_distance(pattern, existing)
                if hamming < min_hamming:
                    valid = False
                    break

            if valid:
                marker_id = len(dictionary)
                # Add border
                full_pattern = self._add_border(pattern)
                dictionary[marker_id] = full_pattern

        return dictionary

    def _hamming_distance(self, a: List[List[int]], b: List[List[int]]) -> int:
        """Compute Hamming distance between two patterns."""
        dist = 0
        for row_a, row_b in zip(a, b):
            for va, vb in zip(row_a, row_b):
                if va != vb:
                    dist += 1
        return dist

    def _add_border(self, pattern):
        """Add a black border around the pattern."""
        n = len(pattern)
        m = len(pattern[0]) if pattern else 0
        size = n + 2
        full = [[0] * (m + 2) for _ in range(size)]
        for i in range(n):
            for j in range(m):
                full[i + 1][j + 1] = pattern[i][j]
        return full

    def create_custom(self, name: str, n: int, m: int,
                      count: int, seed: int = 0) -> Dict[int, List[List[int]]]:
        """Create a custom dictionary."""
        dictionary = self._generate_dictionary(n, m, count, seed)
        self._dictionaries[name] = dictionary
        return dictionary

    def list_dictionaries(self) -> List[str]:
        """List all available dictionary names."""
        return sorted(set(list(self.PREDEFINED_SIZES.keys()) + list(self._dictionaries.keys())))

    def get_marker_bits(self, dict_name: str, marker_id: int) -> Optional[List[List[int]]]:
        """Get the bit pattern for a specific marker."""
        d = self.get_dictionary(dict_name)
        if d is None:
            return None
        return d.get(marker_id)

    def get_total_markers(self, dict_name: str) -> int:
        """Get total markers in a dictionary."""
        d = self.get_dictionary(dict_name)
        return len(d) if d else 0


class CornerDetector:
    """
    Detects marker corners from image contours.

    Uses contour analysis to find quadrilateral candidates.
    """

    def __init__(self, adaptive_threshold: int = 100,
                 min_contour_points: int = 20,
                 min_contour_area: float = 100.0) -> None:
        self.adaptive_threshold = adaptive_threshold
        self.min_contour_points = min_contour_points
        self.min_contour_area = min_contour_area

    def detect_candidates(self, image_width: int, image_height: int,
                          contours: Optional[List[List[Point2D]]] = None) -> List[List[Point2D]]:
        """
        Detect quadrilateral corner candidates.

        In simulation mode, generates plausible corner candidates.
        """
        if contours:
            return self._process_contours(contours)
        return []

    def _process_contours(self, contours: List[List[Point2D]]) -> List[List[Point2D]]:
        """Process contours to find quadrilateral candidates."""
        candidates: List[List[Point2D]] = []
        for contour in contours:
            if len(contour) < self.min_contour_points:
                continue
            # Approximate contour to polygon
            polygon = self._approximate_polygon(contour)
            if len(polygon) == 4:
                # Order corners: TL, TR, BR, BL
                ordered = self._order_corners(polygon)
                candidates.append(ordered)
        return candidates

    def _approximate_polygon(self, contour: List[Point2D],
                              epsilon: float = 0.02) -> List[Point2D]:
        """Simplified polygon approximation (Douglas-Peucker-like)."""
        if len(contour) <= 4:
            return list(contour)

        # Find the two farthest points
        max_dist = 0
        p1_idx, p2_idx = 0, 0
        for i in range(len(contour)):
            for j in range(i + 1, len(contour)):
                d = contour[i].distance_to(contour[j])
                if d > max_dist:
                    max_dist = d
                    p1_idx, p2_idx = i, j

        # Recursively split
        result = self._split_contour(contour, p1_idx, p2_idx, max_dist * epsilon)
        return result

    def _split_contour(self, contour, start: int,
                       end: int, epsilon: float) -> List[Point2D]:
        """Recursively split contour segments."""
        max_dist = 0
        split_idx = -1
        n = len(contour)

        for i in range(start + 1, end):
            d = self._point_to_line_distance(contour[i], contour[start], contour[end])
            if d > max_dist:
                max_dist = d
                split_idx = i

        if max_dist > epsilon and split_idx >= 0:
            left = self._split_contour(contour, start, split_idx, epsilon)
            right = self._split_contour(contour, split_idx, end, epsilon)
            return left + right[1:]
        else:
            return [contour[start], contour[end]]

    def _point_to_line_distance(self, p: Point2D, a: Point2D, b: Point2D) -> float:
        """Distance from point to line segment."""
        dx = b.x - a.x
        dy = b.y - a.y
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-10:
            return p.distance_to(a)
        t = max(0, min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / length_sq))
        proj_x = a.x + t * dx
        proj_y = a.y + t * dy
        return math.sqrt((p.x - proj_x)**2 + (p.y - proj_y)**2)

    def _order_corners(self, corners: List[Point2D]) -> List[Point2D]:
        """Order corners as TL, TR, BR, BL."""
        cx = sum(c.x for c in corners) / 4
        cy = sum(c.y for c in corners) / 4

        def angle_key(c: Point2D) -> float:
            return math.atan2(c.y - cy, c.x - cx)

        sorted_corners = sorted(corners, key=angle_key)

        # Reorder to TL, TR, BR, BL
        if len(sorted_corners) == 4:
            # The top-left should have the smallest x+y
            reordered = sorted(sorted_corners, key=lambda c: c.x + c.y)
            tl = reordered[0]
            br = reordered[-1]
            remaining = [c for c in reordered[1:-1]]
            tr = max(remaining, key=lambda c: c.x - c.y)
            bl = min(remaining, key=lambda c: c.x - c.y)
            return [tl, tr, br, bl]

        return sorted_corners


class PoseEstimator:
    """
    Estimate 6D pose from marker corners.

    Uses a simplified PnP (Perspective-n-Point) algorithm.
    """

    def __init__(self, marker_size: float = 0.05,
                 camera_matrix: Optional[List[List[float]]] = None,
                 dist_coeffs: Optional[List[float]] = None) -> None:
        self.marker_size = marker_size
        self.camera_matrix = camera_matrix or [
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1],
        ]
        self.dist_coeffs = dist_coeffs or [0, 0, 0, 0, 0]

    def estimate_pose(self, corners: List[Point2D]) -> Tuple[List[float], List[float]]:
        """
        Estimate rotation and translation vectors from marker corners.

        Returns (rvec, tvec) as lists of 3 floats each.
        """
        # 3D object points (marker corners in marker coordinate frame)
        half = self.marker_size / 2.0
        obj_points = [
            (-half, half, 0),   # TL
            (half, half, 0),    # TR
            (half, -half, 0),   # BR
            (-half, -half, 0),  # BL
        ]

        # Image points
        img_points = [(c.x, c.y) for c in corners]

        # Simplified PnP using DLT (Direct Linear Transform)
        rvec, tvec = self._solve_pnp(obj_points, img_points)
        return rvec, tvec

    def _solve_pnp(self, obj_points: List[Tuple[float, float, float]],
                   img_points: List[Tuple[float, float]]) -> Tuple[List[float], List[float]]:
        """Simplified PnP using iterative approach."""
        # Initial guess: marker is 0.5m in front of camera
        tvec = [0.0, 0.0, 0.5]
        rvec = [0.0, 0.0, 0.0]

        # Iterative refinement (simplified Gauss-Newton)
        fx = self.camera_matrix[0][0]
        fy = self.camera_matrix[1][1]
        cx = self.camera_matrix[0][2]
        cy = self.camera_matrix[1][2]

        for iteration in range(20):
            # Project 3D points using current estimate
            projected = []
            for ox, oy, oz in obj_points:
                # Apply rotation (simplified - small angles)
                rx, ry, rz = rvec
                px = ox + rz * oy - ry * oz + tvec[0]
                py = -rz * ox + oy + rx * oz + tvec[1]
                pz = ry * ox - rx * oy + oz + tvec[2]

                if pz > 0.01:
                    u = fx * px / pz + cx
                    v = fy * py / pz + cy
                else:
                    u, v = cx, cy
                projected.append((u, v))

            # Compute error and update
            error_x = sum(img_points[i][0] - projected[i][0] for i in range(4))
            error_y = sum(img_points[i][1] - projected[i][1] for i in range(4))

            tvec[0] += error_x * 0.001
            tvec[1] += error_y * 0.001
            tvec[2] += 0.001

            # Estimate rotation from corner arrangement
            dx = img_points[1][0] - img_points[0][0]
            dy = img_points[1][1] - img_points[0][1]
            rvec[2] = math.atan2(dy, dx) * 0.1

            dx = img_points[3][0] - img_points[0][0]
            dy = img_points[3][1] - img_points[0][1]
            rvec[2] += math.atan2(-dx, dy) * 0.1

        return rvec, tvec


class MarkerDecoder:
    """Decodes marker ID from bit pattern."""

    def __init__(self) -> None:
        pass

    def decode(self, corners: List[Point2D], image_data: Any = None,
               dictionary: Optional[Dict[int, List[List[int]]]] = None) -> Optional[int]:
        """
        Decode marker ID from corner positions.

        In simulation, returns a deterministic ID based on corner positions.
        """
        if dictionary is None:
            # Generate deterministic ID from corner geometry
            center_x = sum(c.x for c in corners) / 4
            center_y = sum(c.y for c in corners) / 4
            size = math.sqrt(sum(
                (corners[i].x - corners[(i+1)%4].x)**2 +
                (corners[i].y - corners[(i+1)%4].y)**2
                for i in range(4)
            ))
            return int((center_x * 7 + center_y * 13 + size * 3)) % 1000

        # Match against dictionary
        if image_data is None:
            return None
        return self._match_pattern(image_data, dictionary)

    def _match_pattern(self, bit_pattern: List[List[int]],
                        dictionary: Dict[int, List[List[int]]]) -> Optional[int]:
        """Match a bit pattern against dictionary entries."""
        best_id = None
        best_dist = float('inf')

        for marker_id, pattern in dictionary.items():
            dist = self._pattern_distance(bit_pattern, pattern)
            if dist < best_dist:
                best_dist = dist
                best_id = marker_id

        # Threshold for valid match
        n = len(bit_pattern)
        if n > 0 and best_dist < (n * n) * 0.25:
            return best_id
        return None

    def _pattern_distance(self, a: List[List[int]], b: List[List[int]]) -> int:
        """Compute distance between two patterns."""
        dist = 0
        for row_a, row_b in zip(a, b):
            for va, vb in zip(row_a, row_b):
                if va != vb:
                    dist += 1
        return dist


class SubPixelRefiner:
    """Refines corner positions to sub-pixel accuracy."""

    def __init__(self, window_size: int = 5,
                 max_iterations: int = 30,
                 epsilon: float = 0.001) -> None:
        self.window_size = window_size
        self.max_iterations = max_iterations
        self.epsilon = epsilon

    def refine(self, corner: Point2D, image: Any = None) -> Point2D:
        """
        Refine a corner position to sub-pixel accuracy.

        In simulation, adds a small deterministic offset.
        """
        # Simulate sub-pixel refinement
        offset_x = 0.3 * math.sin(corner.x * 0.1) + 0.1 * math.cos(corner.y * 0.1)
        offset_y = 0.3 * math.cos(corner.x * 0.1) + 0.1 * math.sin(corner.y * 0.1)
        return Point2D(corner.x + offset_x, corner.y + offset_y)

    def refine_all(self, corners: List[Point2D],
                   image: Any = None) -> List[Point2D]:
        """Refine all corners."""
        return [self.refine(c, image) for c in corners]


class ArucoDetector:
    """
    High-level ArUco marker detector.

    Combines dictionary management, corner detection, ID decoding,
    and pose estimation.
    """

    def __init__(self, dictionary_name: str = "DICT_4X4_50",
                 marker_size: float = 0.05,
                 camera_matrix: Optional[List[List[float]]] = None,
                 dist_coeffs: Optional[List[float]] = None) -> None:
        self.dictionary_name = dictionary_name
        self.dictionary = ArucoDictionary()
        self.corner_detector = CornerDetector()
        self.pose_estimator = PoseEstimator(marker_size, camera_matrix, dist_coeffs)
        self.marker_decoder = MarkerDecoder()
        self.subpixel_refiner = SubPixelRefiner()
        self._dict_data = self.dictionary.get_dictionary(dictionary_name)

    def detect(self, image: Any = None,
                corners: Optional[List[List[Point2D]]] = None) -> List[ArucoMarker]:
        """
        Detect ArUco markers in an image.

        In simulation mode, uses provided corners or generates mock detections.
        """
        if corners is None:
            corners = self.corner_detector.detect_candidates(640, 480)

        markers: List[ArucoMarker] = []
        for quad in corners:
            marker_corners = [MarkerCorner(point=c) for c in quad]

            # Decode marker ID
            marker_id = self.marker_decoder.decode(
                [c.point for c in marker_corners],
                dictionary=self._dict_data,
            )

            if marker_id is not None:
                # Refine corners
                refined = self.subpixel_refiner.refine_all([c.point for c in marker_corners])
                for i, rp in enumerate(refined):
                    marker_corners[i].sub_pixel = rp

                # Estimate pose
                rvec, tvec = self.pose_estimator.estimate_pose(
                    [c.refined for c in marker_corners]
                )

                marker = ArucoMarker(
                    marker_id=marker_id,
                    corners=marker_corners,
                    dictionary_name=self.dictionary_name,
                    rvec=rvec,
                    tvec=tvec,
                )
                markers.append(marker)

        return markers

    def set_dictionary(self, name: str) -> bool:
        """Set the active dictionary."""
        d = self.dictionary.get_dictionary(name)
        if d:
            self.dictionary_name = name
            self._dict_data = d
            return True
        return False

    def get_dictionary_info(self) -> Dict[str, Any]:
        """Get information about the current dictionary."""
        return {
            "name": self.dictionary_name,
            "total_markers": self.dictionary.get_total_markers(self.dictionary_name),
        }
