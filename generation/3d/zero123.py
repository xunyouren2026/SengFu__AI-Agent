"""
Zero123++ Multi-View Generation - 多视角3D生成管线

本模块实现了Zero123++多视角生成系统，包含单图到多视角生成、
相机位姿生成、视角一致性检查、新视角合成和深度估计功能。
仅使用标准库，不依赖外部库。
"""

import math
import random
import time
import threading
import hashlib
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


# ============================================================================
# 辅助函数
# ============================================================================

def _generate_id() -> str:
    """生成唯一ID"""
    raw = f"{time.time()}-{random.random()}-{threading.get_ident()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _smooth_step(t: float) -> float:
    t = _clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _mat_mul_3x3(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """3x3矩阵乘法"""
    result = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                result[i][j] += a[i][k] * b[k][j]
    return result


def _mat_vec_mul_3(mat: List[List[float]], vec: List[float]) -> List[float]:
    """3x3矩阵乘3向量"""
    return [
        mat[0][0] * vec[0] + mat[0][1] * vec[1] + mat[0][2] * vec[2],
        mat[1][0] * vec[0] + mat[1][1] * vec[1] + mat[1][2] * vec[2],
        mat[2][0] * vec[0] + mat[2][1] * vec[1] + mat[2][2] * vec[2],
    ]


def _identity_3x3() -> List[List[float]]:
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _rotation_x(angle: float) -> List[List[float]]:
    c, s = math.cos(angle), math.sin(angle)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def _rotation_y(angle: float) -> List[List[float]]:
    c, s = math.cos(angle), math.sin(angle)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def _rotation_z(angle: float) -> List[List[float]]:
    c, s = math.cos(angle), math.sin(angle)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def _normalize_vec(v: List[float]) -> List[float]:
    length = math.sqrt(sum(x * x for x in v))
    if length < 1e-10:
        return [0.0, 0.0, 0.0]
    return [x / length for x in v]


def _cross_product(a: List[float], b: List[float]) -> List[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _dot_product(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _vec_length(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _vec_sub(a: List[float], b: List[float]) -> List[float]:
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _vec_add(a: List[float], b: List[float]) -> List[float]:
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def _vec_scale(v: List[float], s: float) -> List[float]:
    return [x * s for x in v]


def _gaussian_noise(mean: float, std: float) -> float:
    u1 = random.random()
    u2 = random.random()
    while u1 == 0:
        u1 = random.random()
    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mean + std * z0


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class CameraPose:
    """相机位姿（外参）"""
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    rotation: List[List[float]] = field(default_factory=_identity_3x3)
    fov: float = 60.0
    near: float = 0.1
    far: float = 100.0
    elevation: float = 0.0
    azimuth: float = 0.0
    distance: float = 1.0

    def get_view_matrix(self) -> List[List[float]]:
        """获取视图矩阵"""
        forward = _normalize_vec(_vec_scale(self.position, -1.0))
        world_up = [0.0, 1.0, 0.0]
        right = _normalize_vec(_cross_product(forward, world_up))
        up = _cross_product(right, forward)

        return [
            [right[0], right[1], right[2], -_dot_product(right, self.position)],
            [up[0], up[1], up[2], -_dot_product(up, self.position)],
            [-forward[0], -forward[1], -forward[2], _dot_product(forward, self.position)],
            [0.0, 0.0, 0.0, 1.0],
        ]

    def look_at(self, target: List[float]) -> None:
        """设置相机朝向目标"""
        direction = _normalize_vec(_vec_sub(target, self.position))
        right = _normalize_vec(_cross_product(direction, [0.0, 1.0, 0.0]))
        if _vec_length(right) < 1e-6:
            right = _normalize_vec(_cross_product(direction, [0.0, 0.0, 1.0]))
        up = _normalize_vec(_cross_product(right, direction))

        self.rotation = [
            [right[0], up[0], direction[0]],
            [right[1], up[1], direction[1]],
            [right[2], up[2], direction[2]],
        ]


@dataclass
class DepthMap:
    """深度图"""
    width: int = 0
    height: int = 0
    data: List[List[float]] = field(default_factory=list)
    min_depth: float = 0.0
    max_depth: float = 1.0

    def get_depth(self, x: int, y: int) -> float:
        if 0 <= y < self.height and 0 <= x < self.width and self.data:
            return self.data[y][x]
        return self.max_depth

    def normalize(self) -> List[List[float]]:
        """归一化深度图到[0, 1]"""
        if not self.data or self.max_depth == self.min_depth:
            return self.data
        range_val = self.max_depth - self.min_depth
        return [
            [(d - self.min_depth) / range_val for d in row]
            for row in self.data
        ]


@dataclass
class ViewImage:
    """视角图像"""
    pixels: List[List[Tuple[float, float, float]]] = field(default_factory=list)
    width: int = 0
    height: int = 0
    camera_pose: Optional[CameraPose] = None
    depth_map: Optional[DepthMap] = None
    confidence: float = 1.0


@dataclass
class MultiViewOutput:
    """多视角输出"""
    views: List[ViewImage] = field(default_factory=list)
    source_image: Optional[ViewImage] = None
    camera_poses: List[CameraPose] = field(default_factory=list)
    depth_maps: List[DepthMap] = field(default_factory=list)
    consistency_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# CameraPoseGenerator - 相机位姿生成器
# ============================================================================

class CameraPoseGenerator:
    """
    相机位姿生成器：生成围绕物体的多视角相机位姿。

    支持球面分布、均匀分布和自定义分布策略。
    """

    def __init__(
        self,
        default_distance: float = 1.5,
        default_fov: float = 60.0,
    ):
        self._default_distance = default_distance
        self._default_fov = default_fov

    def generate_orbit_poses(
        self,
        num_views: int = 6,
        elevation_range: Tuple[float, float] = (-30.0, 30.0),
        azimuth_start: float = 0.0,
        distance: Optional[float] = None,
    ) -> List[CameraPose]:
        """
        生成球面轨道相机位姿。

        Args:
            num_views: 视角数量
            elevation_range: 仰角范围（度）
            azimuth_start: 起始方位角（度）
            distance: 相机距离
        """
        dist = distance or self._default_distance
        poses: List[CameraPose] = []

        for i in range(num_views):
            azimuth = azimuth_start + (360.0 / num_views) * i
            # 仰角在范围内交替变化
            if num_views > 1:
                t = i / (num_views - 1)
                elevation = _lerp(elevation_range[0], elevation_range[1], t)
            else:
                elevation = 0.0

            pose = self._create_pose_from_spherical(
                azimuth, elevation, dist
            )
            poses.append(pose)

        return poses

    def generate_uniform_sphere_poses(
        self,
        num_views: int = 8,
        distance: Optional[float] = None,
    ) -> List[CameraPose]:
        """使用Fibonacci球面分布生成均匀位姿"""
        dist = distance or self._default_distance
        poses: List[CameraPose] = []

        golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
        for i in range(num_views):
            theta = math.acos(1.0 - 2.0 * (i + 0.5) / num_views)
            phi = 2.0 * math.pi * i / golden_ratio

            x = dist * math.sin(theta) * math.cos(phi)
            y = dist * math.cos(theta)
            z = dist * math.sin(theta) * math.sin(phi)

            pose = CameraPose(
                position=[x, y, z],
                fov=self._default_fov,
                distance=dist,
                elevation=math.degrees(theta) - 90.0,
                azimuth=math.degrees(phi),
            )
            pose.look_at([0.0, 0.0, 0.0])
            poses.append(pose)

        return poses

    def generate_frontal_poses(
        self,
        num_views: int = 4,
        horizontal_range: float = 60.0,
        distance: Optional[float] = None,
    ) -> List[CameraPose]:
        """生成正面视角位姿（适合人脸/物体正面）"""
        dist = distance or self._default_distance
        poses: List[CameraPose] = []

        for i in range(num_views):
            if num_views > 1:
                t = i / (num_views - 1)
                azimuth = _lerp(-horizontal_range / 2, horizontal_range / 2, t)
            else:
                azimuth = 0.0
            elevation = 0.0

            pose = self._create_pose_from_spherical(
                azimuth, elevation, dist
            )
            poses.append(pose)

        return poses

    def _create_pose_from_spherical(
        self,
        azimuth_deg: float,
        elevation_deg: float,
        distance: float,
    ) -> CameraPose:
        """从球面坐标创建相机位姿"""
        az = math.radians(azimuth_deg)
        el = math.radians(elevation_deg)

        x = distance * math.cos(el) * math.sin(az)
        y = distance * math.sin(el)
        z = distance * math.cos(el) * math.cos(az)

        pose = CameraPose(
            position=[x, y, z],
            fov=self._default_fov,
            distance=distance,
            elevation=elevation_deg,
            azimuth=azimuth_deg,
        )
        pose.look_at([0.0, 0.0, 0.0])
        return pose


# ============================================================================
# ViewConsistency - 视角一致性检查器
# ============================================================================

class ViewConsistency:
    """
    视角一致性检查器：确保多视角生成结果之间的视觉一致性。

    检查维度:
    - 颜色一致性
    - 边缘一致性
    - 几何一致性
    - 光照一致性
    """

    def __init__(self, tolerance: float = 0.15):
        self._tolerance = tolerance

    def check_color_consistency(
        self, views: List[ViewImage]
    ) -> Dict[str, float]:
        """检查颜色一致性"""
        if len(views) < 2:
            return {"mean_consistency": 1.0, "min_consistency": 1.0}

        histograms = [self._compute_color_histogram(v) for v in views]
        consistencies: List[float] = []

        for i in range(len(histograms)):
            for j in range(i + 1, len(histograms)):
                sim = self._histogram_similarity(histograms[i], histograms[j])
                consistencies.append(sim)

        return {
            "mean_consistency": sum(consistencies) / len(consistencies),
            "min_consistency": min(consistencies),
            "max_consistency": max(consistencies),
        }

    def check_edge_consistency(
        self, views: List[ViewImage]
    ) -> Dict[str, float]:
        """检查边缘一致性"""
        if len(views) < 2:
            return {"mean_consistency": 1.0}

        edge_maps = [self._compute_edge_map(v) for v in views]
        consistencies: List[float] = []

        for i in range(len(edge_maps)):
            for j in range(i + 1, len(edge_maps)):
                sim = self._edge_map_similarity(edge_maps[i], edge_maps[j])
                consistencies.append(sim)

        return {
            "mean_consistency": sum(consistencies) / len(consistencies),
        }

    def check_geometric_consistency(
        self,
        views: List[ViewImage],
        depth_maps: List[DepthMap],
    ) -> Dict[str, float]:
        """检查几何一致性（基于深度图）"""
        if len(depth_maps) < 2:
            return {"mean_consistency": 1.0}

        consistencies: List[float] = []
        for i in range(len(depth_maps)):
            for j in range(i + 1, len(depth_maps)):
                sim = self._depth_consistency(depth_maps[i], depth_maps[j])
                consistencies.append(sim)

        return {
            "mean_consistency": sum(consistencies) / len(consistencies),
        }

    def compute_overall_consistency(
        self,
        views: List[ViewImage],
        depth_maps: Optional[List[DepthMap]] = None,
    ) -> float:
        """计算总体一致性分数"""
        color_score = self.check_color_consistency(views)["mean_consistency"]
        edge_score = self.check_edge_consistency(views)["mean_consistency"]

        scores = [color_score * 0.4, edge_score * 0.3]

        if depth_maps and len(depth_maps) >= 2:
            geo_score = self.check_geometric_consistency(views, depth_maps)["mean_consistency"]
            scores.append(geo_score * 0.3)

        return sum(scores)

    def _compute_color_histogram(
        self, view: ViewImage, bins: int = 32
    ) -> Dict[str, List[float]]:
        """计算颜色直方图"""
        hist_r = [0.0] * bins
        hist_g = [0.0] * bins
        hist_b = [0.0] * bins
        total_pixels = 0

        for row in view.pixels:
            for r, g, b in row:
                ri = min(int(r / 256.0 * bins), bins - 1)
                gi = min(int(g / 256.0 * bins), bins - 1)
                bi = min(int(b / 256.0 * bins), bins - 1)
                hist_r[ri] += 1.0
                hist_g[gi] += 1.0
                hist_b[bi] += 1.0
                total_pixels += 1

        if total_pixels > 0:
            hist_r = [h / total_pixels for h in hist_r]
            hist_g = [h / total_pixels for h in hist_g]
            hist_b = [h / total_pixels for h in hist_b]

        return {"r": hist_r, "g": hist_g, "b": hist_b}

    def _histogram_similarity(
        self, h1: Dict[str, List[float]], h2: Dict[str, List[float]]
    ) -> float:
        """计算直方图相似度（Bhattacharyya系数）"""
        scores: List[float] = []
        for channel in ("r", "g", "b"):
            a = h1.get(channel, [])
            b = h2.get(channel, [])
            bc = sum(math.sqrt(x * y) for x, y in zip(a, b))
            scores.append(bc)
        return sum(scores) / len(scores) if scores else 0.0

    def _compute_edge_map(
        self, view: ViewImage
    ) -> List[List[float]]:
        """计算边缘图（Sobel算子）"""
        if not view.pixels:
            return []

        h = len(view.pixels)
        w = len(view.pixels[0]) if h > 0 else 0

        # 转灰度
        gray: List[List[float]] = []
        for row in view.pixels:
            gray_row = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in row]
            gray.append(gray_row)

        # Sobel
        edges: List[List[float]] = [[0.0] * w for _ in range(h)]
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                gx = (
                    -gray[y-1][x-1] + gray[y-1][x+1]
                    - 2*gray[y][x-1] + 2*gray[y][x+1]
                    - gray[y+1][x-1] + gray[y+1][x+1]
                )
                gy = (
                    -gray[y-1][x-1] - 2*gray[y-1][x] - gray[y-1][x+1]
                    + gray[y+1][x-1] + 2*gray[y+1][x] + gray[y+1][x+1]
                )
                edges[y][x] = math.sqrt(gx * gx + gy * gy)

        return edges

    def _edge_map_similarity(
        self, e1: List[List[float]], e2: List[List[float]]
    ) -> float:
        """计算边缘图相似度"""
        if not e1 or not e2:
            return 0.0

        h = min(len(e1), len(e2))
        w = min(len(e1[0]) if e1 else 0, len(e2[0]) if e2 else 0)
        if w == 0 or h == 0:
            return 0.0

        # 归一化
        max_e1 = max(max(row) for row in e1) or 1.0
        max_e2 = max(max(row) for row in e2) or 1.0

        total_sim = 0.0
        count = 0
        for y in range(h):
            for x in range(w):
                v1 = e1[y][x] / max_e1
                v2 = e2[y][x] / max_e2
                total_sim += 1.0 - abs(v1 - v2)
                count += 1

        return total_sim / max(count, 1)

    def _depth_consistency(
        self, d1: DepthMap, d2: DepthMap
    ) -> float:
        """计算深度图一致性"""
        if not d1.data or not d2.data:
            return 0.0

        h = min(d1.height, d2.height)
        w = min(d1.width, d2.width)
        if w == 0 or h == 0:
            return 0.0

        total_diff = 0.0
        count = 0
        for y in range(h):
            for x in range(w):
                diff = abs(d1.get_depth(x, y) - d2.get_depth(x, y))
                max_d = max(d1.get_depth(x, y), d2.get_depth(x, y), 0.001)
                total_diff += diff / max_d
                count += 1

        avg_diff = total_diff / max(count, 1)
        return max(0.0, 1.0 - avg_diff)


# ============================================================================
# DepthEstimator - 深度估计器
# ============================================================================

class DepthEstimator:
    """
    深度估计器：从单张图像估计深度图。

    使用简化的深度估计算法，基于图像特征（边缘、颜色梯度等）。
    """

    def __init__(self, base_depth: float = 1.0, depth_scale: float = 0.5):
        self._base_depth = base_depth
        self._depth_scale = depth_scale

    def estimate_depth(self, view: ViewImage) -> DepthMap:
        """估计深度图"""
        if not view.pixels:
            return DepthMap()

        h = len(view.pixels)
        w = len(view.pixels[0]) if h > 0 else 0

        # 转灰度
        gray: List[List[float]] = []
        for row in view.pixels:
            gray.append([0.299 * r + 0.587 * g + 0.114 * b for r, g, b in row])

        # 计算梯度幅值作为深度线索
        gradients: List[List[float]] = [[0.0] * w for _ in range(h)]
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                dx = gray[y][x + 1] - gray[y][x - 1]
                dy = gray[y + 1][x] - gray[y - 1][x]
                gradients[y][x] = math.sqrt(dx * dx + dy * dy)

        # 梯度越大 -> 深度变化越大（边缘处）
        # 中心区域深度较浅，边缘区域深度较深
        cx, cy = w / 2.0, h / 2.0
        max_dist = math.sqrt(cx * cx + cy * cy)

        depth_data: List[List[float]] = []
        min_d = float('inf')
        max_d = float('-inf')

        for y in range(h):
            row: List[float] = []
            for x in range(w):
                # 距离中心的距离
                dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                normalized_dist = dist / max(max_dist, 1.0)

                # 梯度影响
                grad = gradients[y][x] / 255.0

                # 深度 = 基础深度 + 距离因子 + 梯度因子
                depth = (
                    self._base_depth
                    + normalized_dist * self._depth_scale
                    + grad * 0.3
                )
                depth = _clamp(depth, 0.01, 10.0)
                row.append(depth)
                min_d = min(min_d, depth)
                max_d = max(max_d, depth)
            depth_data.append(row)

        return DepthMap(
            width=w, height=h, data=depth_data,
            min_depth=min_d, max_depth=max_d,
        )

    def refine_depth(
        self,
        depth: DepthMap,
        iterations: int = 3,
    ) -> DepthMap:
        """深度图精炼（双边滤波近似）"""
        current = [row[:] for row in depth.data]
        h, w = depth.height, depth.width

        for _ in range(iterations):
            new_data: List[List[float]] = [[0.0] * w for _ in range(h)]
            for y in range(h):
                for x in range(w):
                    total = 0.0
                    weight_sum = 0.0
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                spatial_w = math.exp(-(dx*dx + dy*dy) / 8.0)
                                intensity_diff = abs(current[ny][nx] - current[y][x])
                                range_w = math.exp(-intensity_diff * intensity_diff / 0.5)
                                w_val = spatial_w * range_w
                                total += current[ny][nx] * w_val
                                weight_sum += w_val
                    new_data[y][x] = total / max(weight_sum, 1e-8)
            current = new_data

        min_d = min(min(row) for row in current)
        max_d = max(max(row) for row in current)
        return DepthMap(
            width=w, height=h, data=current,
            min_depth=min_d, max_depth=max_d,
        )


# ============================================================================
# NovelViewSynthesizer - 新视角合成器
# ============================================================================

class NovelViewSynthesizer:
    """
    新视角合成器：从已有视角合成新视角图像。

    使用基于深度的图像变形（DBWR）方法。
    """

    def __init__(self, output_size: Tuple[int, int] = (256, 256)):
        self._output_w, self._output_h = output_size

    def synthesize_view(
        self,
        source: ViewImage,
        source_pose: CameraPose,
        target_pose: CameraPose,
        depth_map: Optional[DepthMap] = None,
    ) -> ViewImage:
        """
        从源视角合成目标视角。

        使用深度图进行反向映射。
        """
        if not source.pixels:
            return ViewImage(width=self._output_w, height=self._output_h)

        src_h = len(source.pixels)
        src_w = len(source.pixels[0]) if src_h > 0 else 0

        # 如果没有深度图，创建默认深度图
        if depth_map is None:
            depth_map = DepthMap(
                width=src_w, height=src_h,
                data=[[1.0] * src_w for _ in range(src_h)],
                min_depth=1.0, max_depth=1.0,
            )

        # 计算相对相机变换
        rel_transform = self._compute_relative_transform(
            source_pose, target_pose
        )

        # 逐像素反向映射
        result_pixels: List[List[Tuple[float, float, float]]] = []
        for y in range(self._output_h):
            row: List[Tuple[float, float, float]] = []
            for x in range(self._output_w):
                # 归一化坐标
                nx = (2.0 * x / self._output_w - 1.0)
                ny = (2.0 * y / self._output_h - 1.0)

                # 获取深度
                src_x = int(x * src_w / self._output_w)
                src_y = int(y * src_h / self._output_h)
                src_x = _clamp(src_x, 0, src_w - 1)
                src_y = _clamp(src_y, 0, src_h - 1)
                depth = depth_map.get_depth(src_x, src_y)

                # 应用变换
                warped_x, warped_y = self._apply_transform(
                    nx, ny, depth, rel_transform
                )

                # 映射回源图像坐标
                sample_x = int((warped_x + 1.0) / 2.0 * src_w)
                sample_y = int((warped_y + 1.0) / 2.0 * src_h)

                if 0 <= sample_x < src_w and 0 <= sample_y < src_h:
                    row.append(source.pixels[sample_y][sample_x])
                else:
                    row.append((0.0, 0.0, 0.0))
            result_pixels.append(row)

        return ViewImage(
            pixels=result_pixels,
            width=self._output_w,
            height=self._output_h,
            camera_pose=target_pose,
            confidence=0.8,
        )

    def _compute_relative_transform(
        self,
        src_pose: CameraPose,
        tgt_pose: CameraPose,
    ) -> Dict[str, float]:
        """计算相对变换参数"""
        # 方位角差
        azimuth_diff = tgt_pose.azimuth - src_pose.azimuth
        elevation_diff = tgt_pose.elevation - src_pose.elevation
        distance_ratio = tgt_pose.distance / max(src_pose.distance, 0.01)

        return {
            "azimuth_diff": azimuth_diff,
            "elevation_diff": elevation_diff,
            "distance_ratio": distance_ratio,
        }

    def _apply_transform(
        self,
        x: float, y: float,
        depth: float,
        transform: Dict[str, float],
    ) -> Tuple[float, float]:
        """应用视角变换"""
        az = math.radians(transform["azimuth_diff"])
        el = math.radians(transform["elevation_diff"])
        scale = transform["distance_ratio"]

        # 简化的透视变换
        parallax = depth * 0.3
        new_x = x * math.cos(az) - y * math.sin(az) * math.sin(el) + parallax * math.sin(az)
        new_y = y * math.cos(el) + parallax * math.sin(el)
        new_x *= scale

        return _clamp(new_x, -1.0, 1.0), _clamp(new_y, -1.0, 1.0)

    def synthesize_multi_view(
        self,
        source: ViewImage,
        source_pose: CameraPose,
        target_poses: List[CameraPose],
        depth_map: Optional[DepthMap] = None,
    ) -> List[ViewImage]:
        """合成多个目标视角"""
        views: List[ViewImage] = []
        for pose in target_poses:
            view = self.synthesize_view(source, source_pose, pose, depth_map)
            views.append(view)
        return views


# ============================================================================
# Zero123Pipeline - Zero123++主管线
# ============================================================================

class Zero123Pipeline:
    """
    Zero123++多视角生成管线。

    流程:
    1. 编码输入图像
    2. 生成目标相机位姿
    3. 估计深度图
    4. 合成新视角
    5. 一致性检查和优化
    6. 输出多视角结果

    使用方法:
        pipeline = Zero123Pipeline()
        result = pipeline.generate(image, num_views=6)
    """

    def __init__(
        self,
        output_size: Tuple[int, int] = (256, 256),
        default_distance: float = 1.5,
    ):
        self._output_size = output_size
        self._pose_generator = CameraPoseGenerator(
            default_distance=default_distance
        )
        self._consistency_checker = ViewConsistency()
        self._depth_estimator = DepthEstimator()
        self._synthesizer = NovelViewSynthesizer(output_size)

    def generate(
        self,
        image: ViewImage,
        num_views: int = 6,
        elevation_range: Tuple[float, float] = (-30.0, 30.0),
        seed: Optional[int] = None,
    ) -> MultiViewOutput:
        """
        从单张图像生成多视角。

        Args:
            image: 输入图像
            num_views: 生成视角数量
            elevation_range: 仰角范围
            seed: 随机种子

        Returns:
            MultiViewOutput: 多视角输出
        """
        if seed is not None:
            random.seed(seed)

        # 步骤1: 创建源视角
        source_pose = CameraPose(
            position=[0.0, 0.0, self._pose_generator._default_distance],
            fov=60.0,
            distance=self._pose_generator._default_distance,
            elevation=0.0,
            azimuth=0.0,
        )
        source_pose.look_at([0.0, 0.0, 0.0])

        source_view = ViewImage(
            pixels=image.pixels,
            width=image.width,
            height=image.height,
            camera_pose=source_pose,
            confidence=1.0,
        )

        # 步骤2: 生成目标相机位姿
        target_poses = self._pose_generator.generate_orbit_poses(
            num_views=num_views,
            elevation_range=elevation_range,
            azimuth_start=45.0,
        )

        # 步骤3: 估计深度图
        depth_map = self._depth_estimator.estimate_depth(source_view)
        refined_depth = self._depth_estimator.refine_depth(depth_map, iterations=2)

        # 步骤4: 合成新视角
        generated_views = self._synthesizer.synthesize_multi_view(
            source_view, source_pose, target_poses, refined_depth
        )

        # 步骤5: 为每个生成视角估计深度
        depth_maps: List[DepthMap] = [refined_depth]
        for view in generated_views:
            view_depth = self._depth_estimator.estimate_depth(view)
            view_depth = self._depth_estimator.refine_depth(view_depth, iterations=1)
            depth_maps.append(view_depth)

        # 步骤6: 一致性检查
        all_views = [source_view] + generated_views
        consistency = self._consistency_checker.compute_overall_consistency(
            all_views, depth_maps
        )

        # 如果一致性低，进行优化
        if consistency < 0.5:
            generated_views = self._optimize_consistency(
                generated_views, target_poses, refined_depth
            )

        return MultiViewOutput(
            views=generated_views,
            source_image=source_view,
            camera_poses=target_poses,
            depth_maps=depth_maps,
            consistency_score=consistency,
            metadata={
                "num_views": num_views,
                "generation_id": _generate_id(),
                "output_size": self._output_size,
            },
        )

    def _optimize_consistency(
        self,
        views: List[ViewImage],
        poses: List[CameraPose],
        depth_map: DepthMap,
    ) -> List[ViewImage]:
        """优化视角一致性"""
        if not views:
            return views

        # 使用源视角的颜色分布来调整生成视角
        if not views:
            return views

        ref_hist = self._consistency_checker._compute_color_histogram(views[0])
        optimized: List[ViewImage] = []

        for i, view in enumerate(views):
            pixels: List[List[Tuple[float, float, float]]] = []
            for row in view.pixels:
                new_row: List[Tuple[float, float, float]] = []
                for r, g, b in row:
                    # 颜色校正
                    r = _clamp(r * 0.95 + 128 * 0.05, 0, 255)
                    g = _clamp(g * 0.95 + 128 * 0.05, 0, 255)
                    b = _clamp(b * 0.95 + 128 * 0.05, 0, 255)
                    new_row.append((r, g, b))
                pixels.append(new_row)

            pose = poses[i] if i < len(poses) else None
            optimized.append(ViewImage(
                pixels=pixels,
                width=view.width,
                height=view.height,
                camera_pose=pose,
                confidence=view.confidence * 0.9,
            ))

        return optimized

    def generate_with_custom_poses(
        self,
        image: ViewImage,
        target_poses: List[CameraPose],
        seed: Optional[int] = None,
    ) -> MultiViewOutput:
        """使用自定义相机位姿生成多视角"""
        if seed is not None:
            random.seed(seed)

        source_pose = CameraPose(
            position=[0.0, 0.0, self._pose_generator._default_distance],
            fov=60.0,
        )
        source_pose.look_at([0.0, 0.0, 0.0])

        source_view = ViewImage(
            pixels=image.pixels,
            width=image.width,
            height=image.height,
            camera_pose=source_pose,
        )

        depth_map = self._depth_estimator.estimate_depth(source_view)
        refined_depth = self._depth_estimator.refine_depth(depth_map)

        generated_views = self._synthesizer.synthesize_multi_view(
            source_view, source_pose, target_poses, refined_depth
        )

        depth_maps: List[DepthMap] = [refined_depth]
        for view in generated_views:
            d = self._depth_estimator.estimate_depth(view)
            depth_maps.append(self._depth_estimator.refine_depth(d, iterations=1))

        all_views = [source_view] + generated_views
        consistency = self._consistency_checker.compute_overall_consistency(
            all_views, depth_maps
        )

        return MultiViewOutput(
            views=generated_views,
            source_image=source_view,
            camera_poses=target_poses,
            depth_maps=depth_maps,
            consistency_score=consistency,
            metadata={"custom_poses": True, "generation_id": _generate_id()},
        )
