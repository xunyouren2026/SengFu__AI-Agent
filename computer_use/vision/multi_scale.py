"""
Multi-Scale Template Matching Module

Provides scale-invariant and rotation-invariant template matching:
- Image pyramid construction (Gaussian and Laplacian)
- Scale-invariant matching
- Rotation-invariant search
- Multi-resolution strategy
- Adaptive search strategy

Pure Python standard library only.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any, Callable


# Type aliases
Pixel = Tuple[int, int, int]
Image = List[List[Pixel]]
GrayImage = List[List[int]]


@dataclass
class MatchResult:
    """Result of a template matching operation."""
    x: int
    y: int
    scale: float
    angle: float
    score: float
    template_width: int
    template_height: int

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y,
                self.x + self.template_width,
                self.y + self.template_height)

    def __repr__(self) -> str:
        return (f"MatchResult(x={self.x}, y={self.y}, scale={self.scale:.3f}, "
                f"angle={self.angle:.1f}, score={self.score:.4f})")


class PyramidType(Enum):
    """Types of image pyramids."""
    GAUSSIAN = "gaussian"
    LAPLACIAN = "laplacian"


@dataclass
class PyramidLevel:
    """A single level in an image pyramid."""
    image: GrayImage
    scale: float
    level: int
    width: int
    height: int


def rgb_to_gray(image: Image) -> GrayImage:
    """Convert RGB image to grayscale."""
    gray: GrayImage = []
    for row in image:
        gray_row: List[int] = []
        for pixel in row:
            r, g, b = pixel[0], pixel[1], pixel[2]
            gray_row.append(max(0, min(255, int(0.299 * r + 0.587 * g + 0.114 * b))))
        gray.append(gray_row)
    return gray


def gray_to_rgb(image: GrayImage) -> Image:
    """Convert grayscale image to RGB."""
    return [[(v, v, v) for v in row] for row in image]


def resize_image(image: GrayImage, new_width: int, new_height: int) -> GrayImage:
    """Resize a grayscale image using bilinear interpolation."""
    if not image or not image[0]:
        return []
    old_height = len(image)
    old_width = len(image[0])

    if old_width == 0 or old_height == 0:
        return [[0] * new_width for _ in range(new_height)]

    result: GrayImage = []
    for y in range(new_height):
        row: List[int] = []
        src_y = y * (old_height - 1) / max(1, new_height - 1)
        y0 = int(math.floor(src_y))
        y1 = min(y0 + 1, old_height - 1)
        fy = src_y - y0

        for x in range(new_width):
            src_x = x * (old_width - 1) / max(1, new_width - 1)
            x0 = int(math.floor(src_x))
            x1 = min(x0 + 1, old_width - 1)
            fx = src_x - x0

            val = (image[y0][x0] * (1 - fx) * (1 - fy)
                   + image[y0][x1] * fx * (1 - fy)
                   + image[y1][x0] * (1 - fx) * fy
                   + image[y1][x1] * fx * fy)
            row.append(max(0, min(255, int(round(val)))))
        result.append(row)
    return result


def rotate_image(image: GrayImage, angle_deg: float) -> GrayImage:
    """Rotate a grayscale image by the given angle."""
    if not image or not image[0]:
        return []
    height = len(image)
    width = len(image[0])
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    new_w = int(abs(width * cos_a) + abs(height * sin_a)) + 1
    new_h = int(abs(width * sin_a) + abs(height * cos_a)) + 1

    cx_old = width / 2.0
    cy_old = height / 2.0
    cx_new = new_w / 2.0
    cy_new = new_h / 2.0

    result: GrayImage = [[0] * new_w for _ in range(new_h)]
    for y in range(new_h):
        for x in range(new_w):
            src_x = (x - cx_new) * cos_a + (y - cy_new) * sin_a + cx_old
            src_y = -(x - cx_new) * sin_a + (y - cy_new) * cos_a + cy_old
            x0 = int(math.floor(src_x))
            y0 = int(math.floor(src_y))
            x1 = x0 + 1
            y1 = y0 + 1
            fx = src_x - x0
            fy = src_y - y0

            if 0 <= x0 < width and 0 <= y0 < height:
                val = image[y0][x0] * (1 - fx) * (1 - fy)
                if 0 <= x1 < width:
                    val += image[y0][x1] * fx * (1 - fy)
                if 0 <= y1 < height:
                    val += image[y1][x0] * (1 - fx) * fy
                if 0 <= x1 < width and 0 <= y1 < height:
                    val += image[y1][x1] * fx * fy
                result[y][x] = max(0, min(255, int(round(val))))
    return result


def compute_ssim(img1: GrayImage, img2: GrayImage) -> float:
    """Compute structural similarity between two images."""
    if not img1 or not img2:
        return 0.0
    h1, w1 = len(img1), len(img1[0])
    h2, w2 = len(img2), len(img2[0])
    if h1 != h2 or w1 != w2:
        return 0.0

    n = h1 * w1
    mean1 = sum(img1[y][x] for y in range(h1) for x in range(w1)) / n
    mean2 = sum(img2[y][x] for y in range(h1) for x in range(w1)) / n

    var1 = sum((img1[y][x] - mean1) ** 2 for y in range(h1) for x in range(w1)) / n
    var2 = sum((img2[y][x] - mean2) ** 2 for y in range(h1) for x in range(w1)) / n
    cov = sum((img1[y][x] - mean1) * (img2[y][x] - mean2)
              for y in range(h1) for x in range(w1)) / n

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    num = (2 * mean1 * mean2 + c1) * (2 * cov + c2)
    den = (mean1 ** 2 + mean2 ** 2 + c1) * (var1 + var2 + c2)
    return num / den if den > 0 else 0.0


def compute_ncc(img1: GrayImage, img2: GrayImage) -> float:
    """Compute normalized cross-correlation between two images."""
    if not img1 or not img2:
        return 0.0
    h1, w1 = len(img1), len(img1[0])
    h2, w2 = len(img2), len(img2[0])
    if h1 != h2 or w1 != w2:
        return 0.0

    n = h1 * w1
    mean1 = sum(img1[y][x] for y in range(h1) for x in range(w1)) / n
    mean2 = sum(img2[y][x] for y in range(h1) for x in range(w1)) / n

    num = sum((img1[y][x] - mean1) * (img2[y][x] - mean2)
              for y in range(h1) for x in range(w1))
    den1 = math.sqrt(sum((img1[y][x] - mean1) ** 2 for y in range(h1) for x in range(w1)))
    den2 = math.sqrt(sum((img2[y][x] - mean2) ** 2 for y in range(h1) for x in range(w1)))

    if den1 == 0 or den2 == 0:
        return 0.0
    return num / (den1 * den2)


def compute_sad(img1: GrayImage, img2: GrayImage) -> float:
    """Compute sum of absolute differences between two images."""
    if not img1 or not img2:
        return float('inf')
    h1, w1 = len(img1), len(img1[0])
    h2, w2 = len(img2), len(img2[0])
    if h1 != h2 or w1 != w2:
        return float('inf')
    return sum(abs(img1[y][x] - img2[y][x])
               for y in range(h1) for x in range(w1))


class ImagePyramid:
    """
    Image pyramid construction for multi-scale analysis.

    Supports Gaussian pyramids (progressive blur + downsample) and
    Laplacian pyramids (difference of Gaussians for edge preservation).
    """

    def __init__(self, min_size: int = 32, scale_factor: float = 0.75,
                 pyramid_type: PyramidType = PyramidType.GAUSSIAN,
                 sigma: float = 1.0) -> None:
        self.min_size = min_size
        self.scale_factor = scale_factor
        self.pyramid_type = pyramid_type
        self.sigma = sigma
        self._levels: List[PyramidLevel] = []

    def build(self, image: GrayImage) -> List[PyramidLevel]:
        """
        Build the image pyramid.

        Returns a list of PyramidLevel from finest (original) to coarsest.
        """
        self._levels = []
        current = image
        level = 0
        scale = 1.0

        while True:
            h = len(current) if current else 0
            w = len(current[0]) if current and current[0] else 0

            if h < self.min_size or w < self.min_size:
                break

            self._levels.append(PyramidLevel(
                image=current, scale=scale, level=level,
                width=w, height=h,
            ))

            if self.pyramid_type == PyramidType.GAUSSIAN:
                blurred = self._gaussian_blur(current)
                new_w = max(self.min_size, int(w * self.scale_factor))
                new_h = max(self.min_size, int(h * self.scale_factor))
                current = resize_image(blurred, new_w, new_h)
            else:
                blurred = self._gaussian_blur(current)
                new_w = max(self.min_size, int(w * self.scale_factor))
                new_h = max(self.min_size, int(h * self.scale_factor))
                downsampled = resize_image(blurred, new_w, new_h)
                # Laplacian = original - upsampled(blurred)
                upsampled = resize_image(blurred, w, h)
                laplacian: GrayImage = []
                for y in range(h):
                    row: List[int] = []
                    for x in range(w):
                        val = current[y][x] - upsampled[y][x] + 128
                        row.append(max(0, min(255, val)))
                    laplacian.append(row)
                current = downsampled

            scale *= self.scale_factor
            level += 1

        return self._levels

    def _gaussian_blur(self, image: GrayImage) -> GrayImage:
        """Apply a simple 3x3 Gaussian blur."""
        if not image or not image[0]:
            return image
        h = len(image)
        w = len(image[0])
        kernel = [
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1],
        ]
        ksum = 16
        result: GrayImage = []
        for y in range(h):
            row: List[int] = []
            for x in range(w):
                val = 0.0
                for ky in range(-1, 2):
                    for kx in range(-1, 2):
                        ny, nx = y + ky, x + kx
                        if 0 <= ny < h and 0 <= nx < w:
                            val += image[ny][nx] * kernel[ky + 1][kx + 1]
                row.append(max(0, min(255, int(round(val / ksum)))))
            result.append(row)
        return result

    def get_levels(self) -> List[PyramidLevel]:
        """Get the pyramid levels."""
        return self._levels

    def get_level(self, level_idx: int) -> Optional[PyramidLevel]:
        """Get a specific pyramid level."""
        if 0 <= level_idx < len(self._levels):
            return self._levels[level_idx]
        return None

    def get_scale_for_size(self, target_width: int, target_height: int) -> float:
        """Get the pyramid scale that best matches the target size."""
        if not self._levels:
            return 1.0
        best_level = self._levels[0]
        best_diff = float('inf')
        for level in self._levels:
            diff = abs(level.width - target_width) + abs(level.height - target_height)
            if diff < best_diff:
                best_diff = diff
                best_level = level
        return best_level.scale


class ScaleEstimator:
    """
    Estimate the scale of a template in a target image.

    Uses coarse-to-fine search across the image pyramid.
    """

    def __init__(self, pyramid: ImagePyramid,
                 match_threshold: float = 0.7) -> None:
        self.pyramid = pyramid
        self.match_threshold = match_threshold

    def estimate(self, source: GrayImage, template: GrayImage) -> Tuple[float, MatchResult]:
        """
        Estimate the scale of the template in the source image.

        Returns (best_scale, match_result).
        """
        levels = self.pyramid.build(source)
        if not levels:
            return 1.0, MatchResult(0, 0, 1.0, 0.0, 0.0,
                                    len(template[0]) if template else 0,
                                    len(template) if template else 0)

        best_result: Optional[MatchResult] = None
        best_score = -1.0

        # Coarse-to-fine: start from smallest level
        for level in reversed(levels):
            scaled_template = self._scale_template(template, level.scale)
            if not scaled_template or not scaled_template[0]:
                continue

            th = len(scaled_template)
            tw = len(scaled_template[0])
            lh = level.height
            lw = level.width

            if tw > lw or th > lh:
                continue

            # Slide template across the level
            for y in range(lh - th + 1):
                for x in range(lw - tw + 1):
                    patch = [level.image[y + dy][x:x + tw] for dy in range(th)]
                    score = compute_ncc(patch, scaled_template)

                    if score > best_score:
                        best_score = score
                        best_result = MatchResult(
                            x=int(x / level.scale),
                            y=int(y / level.scale),
                            scale=level.scale,
                            angle=0.0,
                            score=score,
                            template_width=int(tw / level.scale),
                            template_height=int(th / level.scale),
                        )

        if best_result and best_score >= self.match_threshold:
            return best_result.scale, best_result

        return 1.0, MatchResult(0, 0, 1.0, 0.0, 0.0,
                                len(template[0]) if template else 0,
                                len(template) if template else 0)

    def _scale_template(self, template: GrayImage, scale: float) -> GrayImage:
        """Scale the template to match a pyramid level."""
        if not template or not template[0]:
            return template
        th = len(template)
        tw = len(template[0])
        new_w = max(1, int(tw * scale))
        new_h = max(1, int(th * scale))
        return resize_image(template, new_w, new_h)


class RotationSearch:
    """
    Rotation-invariant template matching.

    Searches for a template at multiple rotation angles.
    """

    def __init__(self, angle_range: Tuple[float, float] = (-180.0, 180.0),
                 angle_step: float = 15.0,
                 refinement_step: float = 1.0,
                 match_threshold: float = 0.7) -> None:
        self.angle_range = angle_range
        self.angle_step = angle_step
        self.refinement_step = refinement_step
        self.match_threshold = match_threshold
        self._rotated_templates: Dict[float, GrayImage] = {}

    def search(self, source: GrayImage, template: GrayImage) -> MatchResult:
        """
        Search for the template in the source at various rotations.

        Returns the best match found.
        """
        if not template or not template[0] or not source or not source[0]:
            return MatchResult(0, 0, 1.0, 0.0, 0.0, 0, 0)

        # Coarse search
        best_result = self._coarse_search(source, template)
        if best_result.score < self.match_threshold:
            return best_result

        # Fine refinement around best angle
        refined = self._fine_search(source, template, best_result.angle)
        return refined if refined.score > best_result.score else best_result

    def _coarse_search(self, source: GrayImage, template: GrayImage) -> MatchResult:
        """Coarse rotation search with large angle steps."""
        best = MatchResult(0, 0, 1.0, 0.0, 0.0,
                           len(template[0]), len(template))

        angle = self.angle_range[0]
        while angle <= self.angle_range[1]:
            rotated = rotate_image(template, angle)
            if not rotated or not rotated[0]:
                angle += self.angle_step
                continue

            match = self._match_template(source, rotated, angle)
            if match.score > best.score:
                best = match
            angle += self.angle_step

        return best

    def _fine_search(self, source: GrayImage, template: GrayImage,
                     center_angle: float) -> MatchResult:
        """Fine rotation search around the best coarse angle."""
        best = MatchResult(0, 0, 1.0, 0.0, 0.0,
                           len(template[0]), len(template))

        search_range = self.angle_step * 2
        angle = center_angle - search_range
        while angle <= center_angle + search_range:
            rotated = rotate_image(template, angle)
            if not rotated or not rotated[0]:
                angle += self.refinement_step
                continue

            match = self._match_template(source, rotated, angle)
            if match.score > best.score:
                best = match
            angle += self.refinement_step

        return best

    def _match_template(self, source: GrayImage, template: GrayImage,
                        angle: float) -> MatchResult:
        """Slide template across source and find best match."""
        if not template or not template[0] or not source or not source[0]:
            return MatchResult(0, 0, 1.0, angle, 0.0, 0, 0)

        th = len(template)
        tw = len(template[0])
        sh = len(source)
        sw = len(source[0])

        if tw > sw or th > sh:
            return MatchResult(0, 0, 1.0, angle, 0.0, tw, th)

        best_score = -1.0
        best_x, best_y = 0, 0

        # Subsample for speed
        step = max(1, min(sw, sh) // 100)

        for y in range(0, sh - th + 1, step):
            for x in range(0, sw - tw + 1, step):
                patch = [source[y + dy][x:x + tw] for dy in range(th)]
                score = compute_ncc(patch, template)
                if score > best_score:
                    best_score = score
                    best_x, best_y = x, y

        return MatchResult(
            x=best_x, y=best_y, scale=1.0, angle=angle,
            score=best_score, template_width=tw, template_height=th,
        )


class AdaptiveSearchStrategy:
    """
    Adaptive multi-scale, multi-rotation search strategy.

    Dynamically adjusts search parameters based on intermediate results
    to balance accuracy and speed.
    """

    def __init__(self, initial_scale_range: Tuple[float, float] = (0.5, 2.0),
                 initial_angle_range: Tuple[float, float] = (-45.0, 45.0),
                 match_threshold: float = 0.7,
                 max_iterations: int = 5) -> None:
        self.initial_scale_range = initial_scale_range
        self.initial_angle_range = initial_angle_range
        self.match_threshold = match_threshold
        self.max_iterations = max_iterations

    def search(self, source: GrayImage, template: GrayImage) -> List[MatchResult]:
        """
        Adaptive search combining scale and rotation.

        Uses iterative refinement to narrow down search space.
        """
        results: List[MatchResult] = []

        # Phase 1: Coarse scale estimation
        pyramid = ImagePyramid(min_size=32, scale_factor=0.8)
        scale_estimator = ScaleEstimator(pyramid, self.match_threshold * 0.8)
        best_scale, scale_match = scale_estimator.estimate(source, template)

        # Phase 2: Rotation search at estimated scale
        scaled_template = resize_image(
            template,
            max(1, int(len(template[0]) * best_scale)),
            max(1, int(len(template) * best_scale)),
        )
        rotation_search = RotationSearch(
            angle_range=self.initial_angle_range,
            angle_step=15.0,
            refinement_step=2.0,
            match_threshold=self.match_threshold * 0.8,
        )
        rot_match = rotation_search.search(source, scaled_template)

        # Phase 3: Iterative refinement
        current_scale = best_scale
        current_angle = rot_match.angle
        best_score = rot_match.score

        for iteration in range(self.max_iterations):
            # Narrow search ranges
            scale_range = (
                max(0.1, current_scale - 0.2 * current_scale),
                current_scale + 0.2 * current_scale,
            )
            angle_range = (
                current_angle - 10.0,
                current_angle + 10.0,
            )

            # Test at refined parameters
            test_scales = [
                scale_range[0],
                (scale_range[0] + scale_range[1]) / 2,
                scale_range[1],
            ]
            test_angles = [
                angle_range[0],
                (angle_range[0] + angle_range[1]) / 2,
                angle_range[1],
            ]

            improved = False
            for scale in test_scales:
                for angle in test_angles:
                    st = resize_image(
                        template,
                        max(1, int(len(template[0]) * scale)),
                        max(1, int(len(template) * scale)),
                    )
                    rotated = rotate_image(st, angle)
                    if not rotated or not rotated[0]:
                        continue

                    match = self._quick_match(source, rotated, scale, angle)
                    if match.score > best_score:
                        best_score = match.score
                        current_scale = scale
                        current_angle = angle
                        improved = True

            if not improved:
                break

        if best_score >= self.match_threshold:
            results.append(MatchResult(
                x=rot_match.x, y=rot_match.y,
                scale=current_scale, angle=current_angle,
                score=best_score,
                template_width=len(template[0]) if template else 0,
                template_height=len(template) if template else 0,
            ))

        return results

    def _quick_match(self, source: GrayImage, template: GrayImage,
                     scale: float, angle: float) -> MatchResult:
        """Quick template match with subsampling."""
        if not template or not template[0] or not source or not source[0]:
            return MatchResult(0, 0, scale, angle, 0.0, 0, 0)

        th = len(template)
        tw = len(template[0])
        sh = len(source)
        sw = len(source[0])

        if tw > sw or th > sh:
            return MatchResult(0, 0, scale, angle, 0.0, tw, th)

        best_score = -1.0
        best_x, best_y = 0, 0
        step = max(1, min(sw, sh) // 50)

        for y in range(0, sh - th + 1, step):
            for x in range(0, sw - tw + 1, step):
                patch = [source[y + dy][x:x + tw] for dy in range(th)]
                score = compute_ncc(patch, template)
                if score > best_score:
                    best_score = score
                    best_x, best_y = x, y

        return MatchResult(
            x=best_x, y=best_y, scale=scale, angle=angle,
            score=best_score, template_width=tw, template_height=th,
        )


class MultiScaleMatcher:
    """
    Multi-scale template matching combining all components.

    Provides a high-level API for scale-invariant and rotation-invariant
    template matching.
    """

    def __init__(self, match_threshold: float = 0.7,
                 enable_rotation: bool = True,
                 enable_scale: bool = True,
                 max_results: int = 5) -> None:
        self.match_threshold = match_threshold
        self.enable_rotation = enable_rotation
        self.enable_scale = enable_scale
        self.max_results = max_results
        self.pyramid = ImagePyramid(min_size=32, scale_factor=0.8)
        self.scale_estimator = ScaleEstimator(self.pyramid, match_threshold * 0.8)
        self.rotation_search = RotationSearch(
            angle_range=(-180.0, 180.0),
            angle_step=15.0,
            refinement_step=2.0,
            match_threshold=match_threshold * 0.8,
        )
        self.adaptive_strategy = AdaptiveSearchStrategy(
            match_threshold=match_threshold,
        )

    def match(self, source: Image, template: Image) -> List[MatchResult]:
        """
        Match template in source image.

        Handles both RGB and grayscale inputs.
        """
        source_gray = source if (source and isinstance(source[0][0], int)) else rgb_to_gray(source)
        template_gray = template if (template and isinstance(template[0][0], int)) else rgb_to_gray(template)

        if self.enable_scale and self.enable_rotation:
            return self.adaptive_strategy.search(source_gray, template_gray)
        elif self.enable_scale:
            _, result = self.scale_estimator.estimate(source_gray, template_gray)
            return [result] if result.score >= self.match_threshold else []
        elif self.enable_rotation:
            result = self.rotation_search.search(source_gray, template_gray)
            return [result] if result.score >= self.match_threshold else []
        else:
            result = self._simple_match(source_gray, template_gray)
            return [result] if result.score >= self.match_threshold else []

    def match_grayscale(self, source: GrayImage, template: GrayImage) -> List[MatchResult]:
        """Match template in grayscale source image."""
        if self.enable_scale and self.enable_rotation:
            return self.adaptive_strategy.search(source, template)
        elif self.enable_scale:
            _, result = self.scale_estimator.estimate(source, template)
            return [result] if result.score >= self.match_threshold else []
        elif self.enable_rotation:
            result = self.rotation_search.search(source, template)
            return [result] if result.score >= self.match_threshold else []
        else:
            result = self._simple_match(source, template)
            return [result] if result.score >= self.match_threshold else []

    def _simple_match(self, source: GrayImage, template: GrayImage) -> MatchResult:
        """Simple sliding window template match."""
        if not template or not template[0] or not source or not source[0]:
            return MatchResult(0, 0, 1.0, 0.0, 0.0, 0, 0)

        th = len(template)
        tw = len(template[0])
        sh = len(source)
        sw = len(source[0])

        if tw > sw or th > sh:
            return MatchResult(0, 0, 1.0, 0.0, 0.0, tw, th)

        best_score = -1.0
        best_x, best_y = 0, 0

        for y in range(sh - th + 1):
            for x in range(sw - tw + 1):
                patch = [source[y + dy][x:x + tw] for dy in range(th)]
                score = compute_ncc(patch, template)
                if score > best_score:
                    best_score = score
                    best_x, best_y = x, y

        return MatchResult(
            x=best_x, y=best_y, scale=1.0, angle=0.0,
            score=best_score, template_width=tw, template_height=th,
        )

    def match_all_scales(self, source: GrayImage, template: GrayImage,
                         scales: Optional[List[float]] = None) -> List[MatchResult]:
        """Match at specific scales."""
        if scales is None:
            scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

        results: List[MatchResult] = []
        for scale in scales:
            scaled = resize_image(
                template,
                max(1, int(len(template[0]) * scale)),
                max(1, int(len(template) * scale)),
            )
            result = self._simple_match(source, scaled)
            result.scale = scale
            if result.score >= self.match_threshold:
                results.append(result)

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:self.max_results]
