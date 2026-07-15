"""
OCR Image Preprocessing Module

Provides image preprocessing pipeline for OCR:
- Binarization (Otsu, adaptive)
- Denoising (median, Gaussian)
- Rotation correction (Hough transform deskew)
- Contrast enhancement (CLAHE)
- Border removal
- Line segmentation

All implemented with pure math on pixel arrays (list of lists of tuples).
Pure Python standard library only.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any, Callable


# Type aliases
Pixel = Tuple[int, int, int]
GrayPixel = int
Image = List[List[Pixel]]
GrayImage = List[List[GrayPixel]]
BinaryImage = List[List[int]]


class BinarizationMethod(Enum):
    """Supported binarization methods."""
    OTSU = "otsu"
    ADAPTIVE_MEAN = "adaptive_mean"
    ADAPTIVE_GAUSSIAN = "adaptive_gaussian"
    FIXED_THRESHOLD = "fixed_threshold"


class DenoisingMethod(Enum):
    """Supported denoising methods."""
    MEDIAN = "median"
    GAUSSIAN = "gaussian"
    MEAN = "mean"
    BILATERAL = "bilateral"


@dataclass
class PreprocessingConfig:
    """Configuration for the preprocessing pipeline."""
    binarize: bool = True
    binarization_method: BinarizationMethod = BinarizationMethod.OTSU
    fixed_threshold: int = 128
    denoise: bool = True
    denoising_method: DenoisingMethod = DenoisingMethod.MEDIAN
    denoise_kernel_size: int = 3
    deskew: bool = True
    enhance_contrast: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    remove_borders: bool = True
    border_threshold: int = 30
    segment_lines: bool = False
    min_line_height: int = 5


def rgb_to_gray(image: Image) -> GrayImage:
    """Convert RGB image to grayscale using luminance formula."""
    gray: GrayImage = []
    for row in image:
        gray_row: List[GrayPixel] = []
        for pixel in row:
            r, g, b = pixel[0], pixel[1], pixel[2]
            gray_val = int(0.299 * r + 0.587 * g + 0.114 * b)
            gray_row.append(max(0, min(255, gray_val)))
        gray.append(gray_row)
    return gray


def gray_to_binary(image: GrayImage, threshold: int) -> BinaryImage:
    """Convert grayscale image to binary using a threshold."""
    binary: BinaryImage = []
    for row in image:
        binary_row: List[int] = []
        for val in row:
            binary_row.append(1 if val >= threshold else 0)
        binary.append(binary_row)
    return binary


def binary_to_gray(image: BinaryImage) -> GrayImage:
    """Convert binary image back to grayscale (0 or 255)."""
    gray: GrayImage = []
    for row in image:
        gray_row: List[GrayPixel] = [255 if v else 0 for v in row]
        gray.append(gray_row)
    return gray


def gray_to_rgb(image: GrayImage) -> Image:
    """Convert grayscale image to RGB."""
    rgb: Image = []
    for row in image:
        rgb_row: List[Pixel] = [(v, v, v) for v in row]
        rgb.append(rgb_row)
    return rgb


def compute_histogram(image: GrayImage) -> List[int]:
    """Compute 256-bin histogram of a grayscale image."""
    histogram = [0] * 256
    for row in image:
        for val in row:
            histogram[val] += 1
    return histogram


def compute_otsu_threshold(image: GrayImage) -> int:
    """
    Compute optimal threshold using Otsu's method.

    Maximizes between-class variance by iterating over all possible
    thresholds and selecting the one that best separates foreground
    from background.
    """
    histogram = compute_histogram(image)
    total_pixels = sum(histogram)
    if total_pixels == 0:
        return 128

    sum_total = sum(i * histogram[i] for i in range(256))
    sum_bg: float = 0.0
    weight_bg: int = 0
    max_variance: float = 0.0
    best_threshold: int = 0

    for t in range(256):
        weight_bg += histogram[t]
        if weight_bg == 0:
            continue
        weight_fg = total_pixels - weight_bg
        if weight_fg == 0:
            break

        sum_bg += t * histogram[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        between_variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between_variance > max_variance:
            max_variance = between_variance
            best_threshold = t

    return best_threshold


class Binarizer:
    """Image binarization using various methods."""

    def __init__(self, method: BinarizationMethod = BinarizationMethod.OTSU,
                 fixed_threshold: int = 128, block_size: int = 11,
                 constant: int = 2) -> None:
        self.method = method
        self.fixed_threshold = fixed_threshold
        self.block_size = block_size
        self.constant = constant

    def binarize(self, image: GrayImage) -> BinaryImage:
        """Binarize a grayscale image."""
        if self.method == BinarizationMethod.OTSU:
            threshold = compute_otsu_threshold(image)
            return gray_to_binary(image, threshold)
        elif self.method == BinarizationMethod.FIXED_THRESHOLD:
            return gray_to_binary(image, self.fixed_threshold)
        elif self.method == BinarizationMethod.ADAPTIVE_MEAN:
            return self._adaptive_threshold(image, use_gaussian=False)
        elif self.method == BinarizationMethod.ADAPTIVE_GAUSSIAN:
            return self._adaptive_threshold(image, use_gaussian=True)
        else:
            return gray_to_binary(image, 128)

    def _adaptive_threshold(self, image: GrayImage, use_gaussian: bool) -> BinaryImage:
        """
        Adaptive thresholding using local mean or Gaussian-weighted mean.

        For each pixel, compute the local mean in a block_size x block_size
        neighborhood, then threshold the pixel against that local mean.
        """
        height = len(image)
        if height == 0:
            return []
        width = len(image[0])
        half = self.block_size // 2
        binary: BinaryImage = []

        # Build integral image for fast local mean computation
        integral = self._compute_integral(image)

        for y in range(height):
            binary_row: List[int] = []
            for x in range(width):
                y1 = max(0, y - half)
                y2 = min(height - 1, y + half)
                x1 = max(0, x - half)
                x2 = min(width - 1, x + half)

                count = (y2 - y1 + 1) * (x2 - x1 + 1)
                local_sum = (integral[y2 + 1][x2 + 1]
                             - integral[y1][x2 + 1]
                             - integral[y2 + 1][x1]
                             + integral[y1][x1])
                local_mean = local_sum / count

                if use_gaussian:
                    local_mean = self._gaussian_weighted_mean(image, x, y, half)

                binary_row.append(1 if image[y][x] >= local_mean - self.constant else 0)
            binary.append(binary_row)
        return binary

    def _compute_integral(self, image: GrayImage) -> List[List[int]]:
        """Compute the integral (summed area) table of an image."""
        height = len(image)
        if height == 0:
            return []
        width = len(image[0])
        integral: List[List[int]] = [[0] * (width + 1) for _ in range(height + 1)]
        for y in range(height):
            row_sum = 0
            for x in range(width):
                row_sum += image[y][x]
                integral[y + 1][x + 1] = integral[y][x + 1] + row_sum
        return integral

    def _gaussian_weighted_mean(self, image: GrayImage, cx: int, cy: int,
                                 radius: int) -> float:
        """Compute Gaussian-weighted local mean around a pixel."""
        height = len(image)
        width = len(image[0])
        sigma = radius / 2.0
        total_weight = 0.0
        weighted_sum = 0.0

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < height and 0 <= nx < width:
                    dist_sq = dx * dx + dy * dy
                    weight = math.exp(-dist_sq / (2.0 * sigma * sigma))
                    weighted_sum += image[ny][nx] * weight
                    total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 128.0


class Denoiser:
    """Image denoising using various filters."""

    def __init__(self, method: DenoisingMethod = DenoisingMethod.MEDIAN,
                 kernel_size: int = 3, sigma: float = 1.0) -> None:
        self.method = method
        self.kernel_size = kernel_size
        self.sigma = sigma
        self._gaussian_kernel = self._build_gaussian_kernel()

    def denoise(self, image: GrayImage) -> GrayImage:
        """Apply denoising filter to a grayscale image."""
        if self.method == DenoisingMethod.MEDIAN:
            return self._median_filter(image)
        elif self.method == DenoisingMethod.GAUSSIAN:
            return self._gaussian_filter(image)
        elif self.method == DenoisingMethod.MEAN:
            return self._mean_filter(image)
        elif self.method == DenoisingMethod.BILATERAL:
            return self._bilateral_filter(image)
        return image

    def _get_neighbors(self, image: GrayImage, y: int, x: int) -> List[int]:
        """Get pixel values in the kernel neighborhood."""
        height = len(image)
        width = len(image[0])
        half = self.kernel_size // 2
        neighbors: List[int] = []
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    neighbors.append(image[ny][nx])
                else:
                    neighbors.append(0)
        return neighbors

    def _median_filter(self, image: GrayImage) -> GrayImage:
        """Apply median filter for salt-and-pepper noise removal."""
        height = len(image)
        if height == 0:
            return []
        width = len(image[0])
        result: GrayImage = []
        for y in range(height):
            row: List[GrayPixel] = []
            for x in range(width):
                neighbors = self._get_neighbors(image, y, x)
                neighbors.sort()
                row.append(neighbors[len(neighbors) // 2])
            result.append(row)
        return result

    def _build_gaussian_kernel(self) -> List[List[float]]:
        """Build a Gaussian kernel for convolution."""
        size = self.kernel_size
        half = size // 2
        kernel: List[List[float]] = []
        total = 0.0
        for y in range(-half, half + 1):
            row: List[float] = []
            for x in range(-half, half + 1):
                val = math.exp(-(x * x + y * y) / (2.0 * self.sigma * self.sigma))
                row.append(val)
                total += val
            kernel.append(row)
        # Normalize
        for y in range(size):
            for x in range(size):
                kernel[y][x] /= total
        return kernel

    def _gaussian_filter(self, image: GrayImage) -> GrayImage:
        """Apply Gaussian blur filter."""
        height = len(image)
        if height == 0:
            return []
        width = len(image[0])
        half = self.kernel_size // 2
        result: GrayImage = []
        for y in range(height):
            row: List[GrayPixel] = []
            for x in range(width):
                val = 0.0
                for ky in range(-half, half + 1):
                    for kx in range(-half, half + 1):
                        ny, nx = y + ky, x + kx
                        if 0 <= ny < height and 0 <= nx < width:
                            val += image[ny][nx] * self._gaussian_kernel[ky + half][kx + half]
                row.append(max(0, min(255, int(round(val)))))
            result.append(row)
        return result

    def _mean_filter(self, image: GrayImage) -> GrayImage:
        """Apply mean (averaging) filter."""
        height = len(image)
        if height == 0:
            return []
        width = len(image[0])
        half = self.kernel_size // 2
        result: GrayImage = []
        for y in range(height):
            row: List[GrayPixel] = []
            for x in range(width):
                neighbors = self._get_neighbors(image, y, x)
                row.append(int(sum(neighbors) / len(neighbors)))
            result.append(row)
        return result

    def _bilateral_filter(self, image: GrayImage) -> GrayImage:
        """
        Bilateral filter: edge-preserving smoothing.

        Combines spatial Gaussian weighting with intensity-based weighting
        to smooth while preserving edges.
        """
        height = len(image)
        if height == 0:
            return []
        width = len(image[0])
        half = self.kernel_size // 2
        sigma_intensity = 30.0
        result: GrayImage = []
        for y in range(height):
            row: List[GrayPixel] = []
            for x in range(width):
                center_val = image[y][x]
                weighted_sum = 0.0
                total_weight = 0.0
                for ky in range(-half, half + 1):
                    for kx in range(-half, half + 1):
                        ny, nx = y + ky, x + kx
                        if 0 <= ny < height and 0 <= nx < width:
                            spatial_dist = kx * kx + ky * ky
                            spatial_weight = math.exp(-spatial_dist / (2.0 * self.sigma * self.sigma))
                            intensity_diff = image[ny][nx] - center_val
                            intensity_weight = math.exp(-intensity_diff * intensity_diff / (2.0 * sigma_intensity * sigma_intensity))
                            w = spatial_weight * intensity_weight
                            weighted_sum += image[ny][nx] * w
                            total_weight += w
                val = weighted_sum / total_weight if total_weight > 0 else center_val
                row.append(max(0, min(255, int(round(val)))))
            result.append(row)
        return result


class Deskewer:
    """
    Rotation correction using Hough transform-based deskew detection.

    Detects the dominant text line angle and rotates the image to correct it.
    """

    def __init__(self, angle_range: Tuple[float, float] = (-45.0, 45.0),
                 angle_step: float = 0.5, min_line_length: int = 50,
                 hough_threshold: int = 50) -> None:
        self.angle_range = angle_range
        self.angle_step = angle_step
        self.min_line_length = min_line_length
        self.hough_threshold = hough_threshold

    def detect_skew_angle(self, image: GrayImage) -> float:
        """
        Detect the skew angle of a document image using the Hough transform.

        Returns the detected angle in degrees (positive = counter-clockwise).
        """
        binary = gray_to_binary(image, compute_otsu_threshold(image))
        edges = self._detect_edges(binary)
        angle = self._hough_detect_angle(edges)
        return angle

    def _detect_edges(self, image: BinaryImage) -> BinaryImage:
        """Simple edge detection using horizontal gradient."""
        height = len(image)
        if height == 0:
            return []
        width = len(image[0])
        edges: BinaryImage = [[0] * width for _ in range(height)]
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                gx = abs(int(image[y][x + 1]) - int(image[y][x - 1]))
                gy = abs(int(image[y + 1][x]) - int(image[y - 1][x]))
                edges[y][x] = 1 if (gx + gy) > 0 else 0
        return edges

    def _hough_detect_angle(self, edges: BinaryImage) -> float:
        """
        Use Hough transform to detect the dominant line angle.

        For each edge pixel, vote for lines at various angles.
        The angle with the most votes (after peak detection) is the skew angle.
        """
        height = len(edges)
        if height == 0:
            return 0.0
        width = len(edges[0])

        # Collect edge pixel coordinates
        edge_pixels: List[Tuple[int, int]] = []
        for y in range(height):
            for x in range(width):
                if edges[y][x]:
                    edge_pixels.append((x, y))

        if len(edge_pixels) < self.min_line_length:
            return 0.0

        # Hough accumulator for angles
        angle_min, angle_max = self.angle_range
        angles: List[float] = []
        a = angle_min
        while a <= angle_max:
            angles.append(a)
            a += self.angle_step

        accumulator: Dict[int, int] = {}
        diag = math.sqrt(width * width + height * height)
        max_rho = int(math.ceil(diag))

        for angle_deg in angles:
            angle_rad = math.radians(angle_deg)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            votes = 0
            for px, py in edge_pixels:
                rho = int(round(px * cos_a + py * sin_a))
                votes += 1
            bucket = int(round(angle_deg / self.angle_step))
            accumulator[bucket] = accumulator.get(bucket, 0) + votes

        if not accumulator:
            return 0.0

        # Find the angle with maximum votes
        best_bucket = max(accumulator, key=accumulator.get)
        best_angle = best_bucket * self.angle_step

        # Refine: look for near-horizontal lines (text lines)
        # Text skew is typically small, so prefer angles near 0
        if abs(best_angle) > 30:
            # Try to find a secondary peak closer to 0
            sorted_buckets = sorted(accumulator.items(), key=lambda x: x[1], reverse=True)
            for bucket, _ in sorted_buckets[:5]:
                angle = bucket * self.angle_step
                if abs(angle) < 15:
                    best_angle = angle
                    break

        return best_angle

    def deskew(self, image: GrayImage) -> GrayImage:
        """Detect and correct the skew angle of the image."""
        angle = self.detect_skew_angle(image)
        if abs(angle) < 0.1:
            return image
        return self._rotate_image(image, -angle)

    def _rotate_image(self, image: GrayImage, angle_deg: float) -> GrayImage:
        """
        Rotate a grayscale image by the given angle using bilinear interpolation.

        Positive angle = counter-clockwise rotation.
        """
        height = len(image)
        if height == 0:
            return []
        width = len(image[0])
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Compute new image dimensions
        new_width = int(abs(width * cos_a) + abs(height * sin_a)) + 1
        new_height = int(abs(width * sin_a) + abs(height * cos_a)) + 1

        cx_old = width / 2.0
        cy_old = height / 2.0
        cx_new = new_width / 2.0
        cy_new = new_height / 2.0

        result: GrayImage = [[0] * new_width for _ in range(new_height)]

        for y in range(new_height):
            for x in range(new_width):
                # Map new pixel back to old coordinates
                src_x = (x - cx_new) * cos_a + (y - cy_new) * sin_a + cx_old
                src_y = -(x - cx_new) * sin_a + (y - cy_new) * cos_a + cy_old

                # Bilinear interpolation
                x0 = int(math.floor(src_x))
                y0 = int(math.floor(src_y))
                x1 = x0 + 1
                y1 = y0 + 1

                fx = src_x - x0
                fy = src_y - y0

                if 0 <= x0 < width and 0 <= y0 < height:
                    val = (image[y0][x0] * (1 - fx) * (1 - fy))
                    if 0 <= x1 < width:
                        val += image[y0][x1] * fx * (1 - fy)
                    if 0 <= y1 < height:
                        val += image[y1][x0] * (1 - fx) * fy
                    if 0 <= x1 < width and 0 <= y1 < height:
                        val += image[y1][x1] * fx * fy
                    result[y][x] = max(0, min(255, int(round(val))))

        return result


class ContrastEnhancer:
    """
    Contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Divides the image into tiles and performs histogram equalization on each tile
    with a clip limit to prevent noise amplification.
    """

    def __init__(self, clip_limit: float = 2.0, grid_size: Tuple[int, int] = (8, 8)) -> None:
        self.clip_limit = clip_limit
        self.grid_size = grid_size

    def enhance(self, image: GrayImage) -> GrayImage:
        """Apply CLAHE contrast enhancement."""
        height = len(image)
        if height == 0:
            return []
        width = len(image[0])
        grid_rows, grid_cols = self.grid_size

        tile_h = height // grid_rows
        tile_w = width // grid_cols

        if tile_h < 2 or tile_w < 2:
            return self._global_histogram_equalization(image)

        # Compute CDFs for each tile
        cdfs: List[List[List[int]]] = []
        for gr in range(grid_rows):
            cdf_row: List[List[int]] = []
            for gc in range(grid_cols):
                y_start = gr * tile_h
                y_end = min((gr + 1) * tile_h, height)
                x_start = gc * tile_w
                x_end = min((gc + 1) * tile_w, width)

                tile = [image[y][x_start:x_end] for y in range(y_start, y_end)]
                cdf = self._compute_clahe_cdf(tile)
                cdf_row.append(cdf)
            cdfs.append(cdf_row)

        # Interpolate CDFs for each pixel
        result: GrayImage = [[0] * width for _ in range(height)]
        for y in range(height):
            for x in range(width):
                # Determine which tile this pixel belongs to
                gr = min(int(y / tile_h), grid_rows - 1)
                gc = min(int(x / tile_w), grid_cols - 1)

                # Bilinear interpolation between neighboring CDFs
                fy = (y - gr * tile_h) / tile_h
                fx = (x - gc * tile_w) / tile_w

                # Get the four surrounding CDFs
                cdf00 = cdfs[gr][gc]
                cdf10 = cdfs[min(gr + 1, grid_rows - 1)][gc]
                cdf01 = cdfs[gr][min(gc + 1, grid_cols - 1)]
                cdf11 = cdfs[min(gr + 1, grid_rows - 1)][min(gc + 1, grid_cols - 1)]

                val = image[y][x]
                v00 = cdf00[val]
                v10 = cdf10[val]
                v01 = cdf01[val]
                v11 = cdf11[val]

                interp = (v00 * (1 - fx) * (1 - fy)
                          + v01 * fx * (1 - fy)
                          + v10 * (1 - fx) * fy
                          + v11 * fx * fy)
                result[y][x] = max(0, min(255, int(round(interp))))

        return result

    def _compute_clahe_cdf(self, tile: GrayImage) -> List[int]:
        """Compute clipped and equalized CDF for a tile."""
        histogram = compute_histogram(tile)
        total_pixels = sum(histogram)
        if total_pixels == 0:
            return list(range(256))

        # Clip the histogram
        clip_limit = int(self.clip_limit * total_pixels / 256)
        excess = 0
        for i in range(256):
            if histogram[i] > clip_limit:
                excess += histogram[i] - clip_limit
                histogram[i] = clip_limit

        # Redistribute excess evenly
        avg_increment = excess // 256
        remainder = excess % 256
        for i in range(256):
            histogram[i] += avg_increment
            if i < remainder:
                histogram[i] += 1

        # Compute CDF
        cdf: List[int] = [0] * 256
        cdf[0] = histogram[0]
        for i in range(1, 256):
            cdf[i] = cdf[i - 1] + histogram[i]

        # Normalize CDF to [0, 255]
        cdf_min = cdf[0]
        cdf_max = cdf[255]
        denom = cdf_max - cdf_min
        if denom == 0:
            return list(range(256))

        normalized: List[int] = []
        for i in range(256):
            normalized.append(int(round(255.0 * (cdf[i] - cdf_min) / denom)))
        return normalized

    def _global_histogram_equalization(self, image: GrayImage) -> GrayImage:
        """Apply global histogram equalization as fallback."""
        histogram = compute_histogram(image)
        total_pixels = sum(histogram)
        if total_pixels == 0:
            return image

        cdf: List[int] = [0] * 256
        cdf[0] = histogram[0]
        for i in range(1, 256):
            cdf[i] = cdf[i - 1] + histogram[i]

        cdf_min = cdf[0]
        lut: List[int] = []
        for i in range(256):
            lut.append(int(round(255.0 * (cdf[i] - cdf_min) / (total_pixels - cdf_min))))

        result: GrayImage = []
        for row in image:
            result.append([lut[v] for v in row])
        return result


class BorderRemover:
    """
    Remove borders and margins from document images.

    Detects and removes uniform borders by analyzing the variance
    of pixel values in rows and columns from the edges.
    """

    def __init__(self, threshold: int = 30, min_border_ratio: float = 0.02) -> None:
        self.threshold = threshold
        self.min_border_ratio = min_border_ratio

    def remove_borders(self, image: GrayImage) -> GrayImage:
        """Detect and remove borders from the image."""
        top = self._find_top_border(image)
        bottom = self._find_bottom_border(image)
        left = self._find_left_border(image)
        right = self._find_right_border(image)

        height = len(image)
        if height == 0:
            return []
        width = len(image[0])

        # Ensure minimum border
        min_border = int(min(height, width) * self.min_border_ratio)
        top = max(0, top - min_border)
        bottom = min(height, bottom + min_border)
        left = max(0, left - min_border)
        right = min(width, right + min_border)

        if top >= bottom or left >= right:
            return image

        return [row[left:right] for row in image[top:bottom]]

    def _find_top_border(self, image: GrayImage) -> int:
        """Find the top border by scanning rows from the top."""
        height = len(image)
        if height == 0:
            return 0
        width = len(image[0])
        threshold_sq = self.threshold * self.threshold

        for y in range(height):
            variance = self._row_variance(image[y])
            if variance > threshold_sq:
                return y
        return 0

    def _find_bottom_border(self, image: GrayImage) -> int:
        """Find the bottom border by scanning rows from the bottom."""
        height = len(image)
        if height == 0:
            return height
        threshold_sq = self.threshold * self.threshold

        for y in range(height - 1, -1, -1):
            variance = self._row_variance(image[y])
            if variance > threshold_sq:
                return y + 1
        return height

    def _find_left_border(self, image: GrayImage) -> int:
        """Find the left border by scanning columns from the left."""
        height = len(image)
        if height == 0:
            return 0
        width = len(image[0])
        threshold_sq = self.threshold * self.threshold

        for x in range(width):
            col = [image[y][x] for y in range(height)]
            variance = self._column_variance(col)
            if variance > threshold_sq:
                return x
        return 0

    def _find_right_border(self, image: GrayImage) -> int:
        """Find the right border by scanning columns from the right."""
        height = len(image)
        if height == 0:
            return 0
        width = len(image[0])
        threshold_sq = self.threshold * self.threshold

        for x in range(width - 1, -1, -1):
            col = [image[y][x] for y in range(height)]
            variance = self._column_variance(col)
            if variance > threshold_sq:
                return x + 1
        return width

    def _row_variance(self, row: List[int]) -> float:
        """Compute the variance of pixel values in a row."""
        if len(row) < 2:
            return 0.0
        mean = sum(row) / len(row)
        return sum((v - mean) ** 2 for v in row) / len(row)

    def _column_variance(self, col: List[int]) -> float:
        """Compute the variance of pixel values in a column."""
        if len(col) < 2:
            return 0.0
        mean = sum(col) / len(col)
        return sum((v - mean) ** 2 for v in col) / len(col)


class LineSegmenter:
    """
    Text line segmentation from document images.

    Uses horizontal projection profile analysis to identify
    text lines and segment the image accordingly.
    """

    def __init__(self, min_line_height: int = 5, min_gap_ratio: float = 0.3,
                 merge_threshold: float = 0.5) -> None:
        self.min_line_height = min_line_height
        self.min_gap_ratio = min_gap_ratio
        self.merge_threshold = merge_threshold

    def segment(self, image: GrayImage) -> List[GrayImage]:
        """
        Segment the image into horizontal text lines.

        Returns a list of grayscale image strips, one per text line.
        """
        if not image:
            return []

        projection = self._horizontal_projection(image)
        lines = self._find_line_boundaries(projection)
        return self._extract_lines(image, lines)

    def _horizontal_projection(self, image: GrayImage) -> List[int]:
        """Compute horizontal projection profile (sum of dark pixels per row)."""
        projection: List[int] = []
        for row in image:
            dark_count = sum(1 for v in row if v < 128)
            projection.append(dark_count)
        return projection

    def _find_line_boundaries(self, projection: List[int]) -> List[Tuple[int, int]]:
        """
        Find text line boundaries from the projection profile.

        Uses thresholding on the projection to identify text regions
        (high density) and gaps (low density).
        """
        if not projection:
            return []

        max_val = max(projection) if projection else 0
        if max_val == 0:
            return []

        threshold = max_val * self.min_gap_ratio

        in_line = False
        line_start = 0
        boundaries: List[Tuple[int, int]] = []

        for i, val in enumerate(projection):
            if not in_line and val > threshold:
                in_line = True
                line_start = i
            elif in_line and val <= threshold:
                in_line = False
                if i - line_start >= self.min_line_height:
                    boundaries.append((line_start, i))

        if in_line:
            if len(projection) - line_start >= self.min_line_height:
                boundaries.append((line_start, len(projection)))

        # Merge lines that are very close together
        if len(boundaries) > 1:
            merged: List[Tuple[int, int]] = [boundaries[0]]
            for start, end in boundaries[1:]:
                prev_start, prev_end = merged[-1]
                gap = start - prev_end
                avg_height = (prev_end - prev_start + end - start) / 2
                if gap < avg_height * self.merge_threshold:
                    merged[-1] = (prev_start, end)
                else:
                    merged.append((start, end))
            boundaries = merged

        return boundaries

    def _extract_lines(self, image: GrayImage,
                       boundaries: List[Tuple[int, int]]) -> List[GrayImage]:
        """Extract image strips for each detected line."""
        width = len(image[0]) if image else 0
        lines: List[GrayImage] = []
        for start, end in boundaries:
            line_image = [row[:] for row in image[start:end]]
            lines.append(line_image)
        return lines

    def get_word_segments(self, line_image: GrayImage) -> List[GrayImage]:
        """
        Segment a text line image into individual words.

        Uses vertical projection profile to find word boundaries.
        """
        if not line_image:
            return []

        projection = self._vertical_projection(line_image)
        word_boundaries = self._find_word_boundaries(projection)
        return self._extract_words(line_image, word_boundaries)

    def _vertical_projection(self, image: GrayImage) -> List[int]:
        """Compute vertical projection profile."""
        height = len(image)
        if height == 0:
            return []
        width = len(image[0])
        projection: List[int] = [0] * width
        for y in range(height):
            for x in range(width):
                if image[y][x] < 128:
                    projection[x] += 1
        return projection

    def _find_word_boundaries(self, projection: List[int]) -> List[Tuple[int, int]]:
        """Find word boundaries from vertical projection."""
        if not projection:
            return []

        max_val = max(projection) if projection else 0
        if max_val == 0:
            return []

        threshold = max_val * 0.1
        in_word = False
        word_start = 0
        boundaries: List[Tuple[int, int]] = []

        for i, val in enumerate(projection):
            if not in_word and val > threshold:
                in_word = True
                word_start = i
            elif in_word and val <= threshold:
                in_word = False
                if i - word_start >= 2:
                    boundaries.append((word_start, i))

        if in_word and len(projection) - word_start >= 2:
            boundaries.append((word_start, len(projection)))

        return boundaries

    def _extract_words(self, image: GrayImage,
                       boundaries: List[Tuple[int, int]]) -> List[GrayImage]:
        """Extract word image strips."""
        words: List[GrayImage] = []
        for start, end in boundaries:
            word = [row[start:end] for row in image]
            words.append(word)
        return words


class PreprocessingPipeline:
    """
    Configurable preprocessing pipeline for OCR.

    Chains preprocessing steps in order: contrast enhancement -> denoising ->
    binarization -> deskew -> border removal -> line segmentation.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        self.config = config or PreprocessingConfig()
        self.binarizer = Binarizer(
            method=self.config.binarization_method,
            fixed_threshold=self.config.fixed_threshold
        )
        self.denoiser = Denoiser(
            method=self.config.denoising_method,
            kernel_size=self.config.denoise_kernel_size
        )
        self.deskewer = Deskewer()
        self.enhancer = ContrastEnhancer(
            clip_limit=self.config.clahe_clip_limit,
            grid_size=self.config.clahe_grid_size
        )
        self.border_remover = BorderRemover(
            threshold=self.config.border_threshold
        )
        self.line_segmenter = LineSegmenter(
            min_line_height=self.config.min_line_height
        )
        self._steps: List[Tuple[str, Callable]] = []
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """Build the pipeline step sequence."""
        self._steps = []
        if self.config.enhance_contrast:
            self._steps.append(("contrast_enhancement", self._step_enhance))
        if self.config.denoise:
            self._steps.append(("denoising", self._step_denoise))
        if self.config.binarize:
            self._steps.append(("binarization", self._step_binarize))
        if self.config.deskew:
            self._steps.append(("deskew", self._step_deskew))
        if self.config.remove_borders:
            self._steps.append(("border_removal", self._step_remove_borders))
        if self.config.segment_lines:
            self._steps.append(("line_segmentation", self._step_segment_lines))

    def process(self, image: Image) -> Dict[str, Any]:
        """
        Process an RGB image through the full preprocessing pipeline.

        Returns a dictionary with intermediate results and final output.
        """
        results: Dict[str, Any] = {"original_size": (len(image), len(image[0]) if image else 0)}

        # Convert to grayscale
        gray = rgb_to_gray(image)
        results["grayscale"] = gray
        results["grayscale_size"] = (len(gray), len(gray[0]) if gray else 0)

        current: GrayImage = gray

        for step_name, step_func in self._steps:
            current = step_func(current)
            results[step_name] = current

        results["final"] = current
        if isinstance(current, list) and current and isinstance(current[0], list):
            if isinstance(current[0][0], list):
                results["final_type"] = "line_segments"
                results["num_lines"] = len(current)
            else:
                results["final_type"] = "image"
                results["final_size"] = (len(current), len(current[0]) if current else 0)

        return results

    def process_grayscale(self, image: GrayImage) -> Dict[str, Any]:
        """Process a grayscale image through the pipeline."""
        results: Dict[str, Any] = {}
        current = image

        for step_name, step_func in self._steps:
            current = step_func(current)
            results[step_name] = current

        results["final"] = current
        return results

    def _step_enhance(self, image: GrayImage) -> GrayImage:
        """Apply contrast enhancement."""
        return self.enhancer.enhance(image)

    def _step_denoise(self, image: GrayImage) -> GrayImage:
        """Apply denoising."""
        return self.denoiser.denoise(image)

    def _step_binarize(self, image: GrayImage) -> GrayImage:
        """Apply binarization (returns as grayscale 0/255)."""
        binary = self.binarizer.binarize(image)
        return binary_to_gray(binary)

    def _step_deskew(self, image: GrayImage) -> GrayImage:
        """Apply deskew correction."""
        return self.deskewer.deskew(image)

    def _step_remove_borders(self, image: GrayImage) -> GrayImage:
        """Remove borders."""
        return self.border_remover.remove_borders(image)

    def _step_segment_lines(self, image: GrayImage) -> List[GrayImage]:
        """Segment into text lines."""
        return self.line_segmenter.segment(image)

    def add_step(self, name: str, func: Callable[[GrayImage], Any]) -> None:
        """Add a custom preprocessing step."""
        self._steps.append((name, func))

    def remove_step(self, name: str) -> bool:
        """Remove a step by name."""
        for i, (step_name, _) in enumerate(self._steps):
            if step_name == name:
                self._steps.pop(i)
                return True
        return False

    def get_step_names(self) -> List[str]:
        """Get the list of step names in order."""
        return [name for name, _ in self._steps]

    def set_config(self, config: PreprocessingConfig) -> None:
        """Update the pipeline configuration and rebuild."""
        self.config = config
        self.binarizer = Binarizer(
            method=config.binarization_method,
            fixed_threshold=config.fixed_threshold
        )
        self.denoiser = Denoiser(
            method=config.denoising_method,
            kernel_size=config.denoise_kernel_size
        )
        self.enhancer = ContrastEnhancer(
            clip_limit=config.clahe_clip_limit,
            grid_size=config.clahe_grid_size
        )
        self.border_remover = BorderRemover(threshold=config.border_threshold)
        self.line_segmenter = LineSegmenter(min_line_height=config.min_line_height)
        self._build_pipeline()
