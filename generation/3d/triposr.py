"""
TripoSR 快速单图到3D网格生成管线
================================

本模块实现了 TripoSR 单图到3D网格生成系统，包含图像编码器（特征金字塔）、
3D体积重建（神经隐式表面）、Marching Cubes 网格提取、纹理生成、
后处理（网格平滑、法线估计）等功能。

仅使用 Python 标准库，不依赖外部库。所有神经网络操作通过数组上的数学运算模拟。
"""

from __future__ import annotations

import math
import random
import time
import hashlib
import threading
from dataclasses import dataclass, field
from typing import (
    List, Tuple, Optional, Dict, Any, Sequence, Callable
)


# ============================================================================
# 辅助函数
# ============================================================================

def _generate_id() -> str:
    """生成唯一标识符。"""
    raw = f"{time.time()}-{random.random()}-{threading.get_ident()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _smooth_step(t: float) -> float:
    t = _clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


def _relu(x: float) -> float:
    return max(0.0, x)


def _tanh(x: float) -> float:
    return math.tanh(x)


def _vec_add(a: List[float], b: List[float]) -> List[float]:
    return [a[i] + b[i] for i in range(len(a))]


def _vec_sub(a: List[float], b: List[float]) -> List[float]:
    return [a[i] - b[i] for i in range(len(a))]


def _vec_scale(a: List[float], s: float) -> List[float]:
    return [v * s for v in a]


def _vec_dot(a: List[float], b: List[float]) -> float:
    return sum(a[i] * b[i] for i in range(len(a)))


def _vec_cross(a: List[float], b: List[float]) -> List[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _vec_length(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _vec_normalize(v: List[float]) -> List[float]:
    length = _vec_length(v)
    if length < 1e-12:
        return [0.0, 0.0, 0.0]
    return [x / length for x in v]


def _mat3_identity() -> List[List[float]]:
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _mat3_mul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    result = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                result[i][j] += a[i][k] * b[k][j]
    return result


def _mat3_vec(m: List[List[float]], v: List[float]) -> List[float]:
    return [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]


def _rotation_y(angle: float) -> List[List[float]]:
    c, s = math.cos(angle), math.sin(angle)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def _rotation_x(angle: float) -> List[List[float]]:
    c, s = math.cos(angle), math.sin(angle)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


# ============================================================================
# TripoSRConfig - 配置
# ============================================================================

@dataclass
class TripoSRConfig:
    """TripoSR 管线配置参数。"""

    # 图像编码器
    encoder_channels: int = 64
    encoder_depth: int = 4
    pyramid_levels: int = 4
    image_size: int = 256

    # 体积重建
    volume_resolution: int = 64
    volume_range: float = 1.0
    num_samples_per_ray: int = 32
    sdf_threshold: float = 0.01

    # Marching Cubes
    mc_isovalue: float = 0.0
    mc_adaptive: bool = True

    # 纹理
    texture_resolution: int = 1024
    texture_samples: int = 4

    # 后处理
    smoothing_iterations: int = 3
    smoothing_lambda: float = 0.5
    normal_estimation_radius: int = 2

    # 通用
    seed: Optional[int] = None
    device: str = "cpu"
    precision: str = "fp32"

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TripoSRConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid}
        return cls(**filtered)


# ============================================================================
# ImageEncoder - 图像编码器
# ============================================================================

class ImageEncoder:
    """图像编码器：将输入图像编码为多尺度特征图。

    使用模拟的卷积神经网络，通过数学运算在二维数组上实现
    特征提取。包含卷积、激活函数和池化操作。
    """

    def __init__(self, config: TripoSRConfig) -> None:
        self.config = config
        self.channels = config.encoder_channels
        self.depth = config.encoder_depth
        self._weights: List[List[List[List[float]]]] = []
        self._biases: List[float] = []
        self._init_weights()

    def _init_weights(self) -> None:
        """初始化模拟卷积核权重。"""
        rng = random.Random(42)
        for layer in range(self.depth):
            kernel_size = 3
            in_ch = self.channels if layer > 0 else 3
            out_ch = self.channels
            layer_weights: List[List[List[float]]] = []
            for oc in range(out_ch):
                kernel: List[List[float]] = []
                for ic in range(in_ch):
                    k = [rng.gauss(0, math.sqrt(2.0 / (in_ch * kernel_size * kernel_size)))
                         for _ in range(kernel_size * kernel_size)]
                    kernel.append(k)
                layer_weights.append(kernel)
            self._weights.append(layer_weights)
            self._biases.append(rng.gauss(0, 0.01))

    def _conv2d(self, feature_map: List[List[List[float]]],
                weights: List[List[List[float]]], bias: float,
                stride: int = 1, padding: int = 1) -> List[List[List[float]]]:
        """模拟二维卷积运算。"""
        in_channels = len(feature_map)
        h = len(feature_map[0])
        w = len(feature_map[0][0])
        out_channels = len(weights)
        kernel_size = int(math.sqrt(len(weights[0][0])))

        out_h = (h + 2 * padding - kernel_size) // stride + 1
        out_w = (w + 2 * padding - kernel_size) // stride + 1
        output: List[List[List[float]]] = [
            [[0.0] * out_w for _ in range(out_h)] for _ in range(out_channels)
        ]

        for oc in range(out_channels):
            for oh in range(out_h):
                for ow in range(out_w):
                    val = bias
                    for ic in range(in_channels):
                        for kh in range(kernel_size):
                            for kw in range(kernel_size):
                                ih = oh * stride + kh - padding
                                iw = ow * stride + kw - padding
                                if 0 <= ih < h and 0 <= iw < w:
                                    val += (feature_map[ic][ih][iw]
                                            * weights[oc][ic][kh * kernel_size + kw])
                    output[oc][oh][ow] = _relu(val)
        return output

    def _max_pool2d(self, feature_map: List[List[List[float]]],
                    pool_size: int = 2) -> List[List[List[float]]]:
        """模拟最大池化。"""
        channels = len(feature_map)
        h = len(feature_map[0])
        w = len(feature_map[0][0])
        out_h = h // pool_size
        out_w = w // pool_size
        output: List[List[List[float]]] = [
            [[0.0] * out_w for _ in range(out_h)] for _ in range(channels)
        ]
        for c in range(channels):
            for oh in range(out_h):
                for ow in range(out_w):
                    max_val = -1e30
                    for ph in range(pool_size):
                        for pw in range(pool_size):
                            ih = oh * pool_size + ph
                            iw = ow * pool_size + pw
                            if ih < h and iw < w:
                                max_val = max(max_val, feature_map[c][ih][iw])
                    output[c][oh][ow] = max_val
        return output

    def encode(self, image: List[List[List[float]]]) -> List[List[List[List[float]]]]:
        """编码图像为多尺度特征图。

        Args:
            image: 输入图像，形状为 [C, H, W]，C=3 (RGB)。

        Returns:
            多尺度特征图列表，每个元素形状为 [channels, h, w]。
        """
        features: List[List[List[List[float]]]] = []
        x = [list(channel) for channel in image]

        for layer_idx in range(self.depth):
            x = self._conv2d(x, self._weights[layer_idx], self._biases[layer_idx])
            features.append([list(ch) for ch in x])
            if layer_idx < self.depth - 1:
                x = self._max_pool2d(x)

        return features


# ============================================================================
# FeaturePyramid - 特征金字塔
# ============================================================================

class FeaturePyramid:
    """特征金字塔网络：将多尺度编码器特征融合为统一表示。

    通过上采样和逐元素相加，将深层语义信息与浅层空间信息融合。
    """

    def __init__(self, config: TripoSRConfig) -> None:
        self.config = config
        self.num_levels = config.pyramid_levels
        self._lateral_weights: List[List[List[List[float]]]] = []
        self._smooth_weights: List[List[List[List[float]]]] = []
        self._init_weights()

    def _init_weights(self) -> None:
        """初始化侧连接和平滑层权重。"""
        rng = random.Random(123)
        for level in range(self.num_levels):
            lateral: List[List[float]] = []
            for _ in range(self.config.encoder_channels):
                k = [rng.gauss(0, 0.01) for _ in range(self.config.encoder_channels)]
                lateral.append(k)
            self._lateral_weights.append(lateral)

            smooth: List[List[float]] = []
            for _ in range(self.config.encoder_channels):
                k = [rng.gauss(0, 0.01) for _ in range(self.config.encoder_channels)]
                smooth.append(k)
            self._smooth_weights.append(smooth)

    def _upsample_nearest(self, feature_map: List[List[List[float]]],
                          target_h: int, target_w: int) -> List[List[List[float]]]:
        """最近邻上采样。"""
        channels = len(feature_map)
        src_h = len(feature_map[0])
        src_w = len(feature_map[0][0])
        output: List[List[List[float]]] = [
            [[0.0] * target_w for _ in range(target_h)] for _ in range(channels)
        ]
        for c in range(channels):
            for oh in range(target_h):
                for ow in range(target_w):
                    ih = min(oh * src_h // target_h, src_h - 1)
                    iw = min(ow * src_w // target_w, src_w - 1)
                    output[c][oh][ow] = feature_map[c][ih][iw]
        return output

    def _apply_1x1_conv(self, feature_map: List[List[List[float]]],
                         weights: List[List[float]]) -> List[List[List[float]]]:
        """模拟 1x1 卷积（逐通道线性变换）。"""
        in_ch = len(feature_map)
        out_ch = len(weights)
        h = len(feature_map[0])
        w = len(feature_map[0][0])
        output: List[List[List[float]]] = [
            [[0.0] * w for _ in range(h)] for _ in range(out_ch)
        ]
        for oc in range(out_ch):
            for oh in range(h):
                for ow in range(w):
                    val = 0.0
                    for ic in range(in_ch):
                        val += feature_map[ic][oh][ow] * weights[oc][ic]
                    output[oc][oh][ow] = _relu(val)
        return output

    def build(self, encoder_features: List[List[List[List[float]]]]
              ) -> List[List[List[List[float]]]]:
        """构建特征金字塔。

        Args:
            encoder_features: 编码器输出的多尺度特征列表，
                              从底层到高层排列。

        Returns:
            融合后的金字塔特征列表，从高层到底层排列。
        """
        num_features = len(encoder_features)
        pyramid: List[List[List[List[float]]]] = []

        # 从最高层开始，逐层上采样并融合
        current = self._apply_1x1_conv(
            encoder_features[-1], self._lateral_weights[min(num_features - 1, self.num_levels - 1)]
        )
        pyramid.append(current)

        for i in range(num_features - 2, -1, -1):
            level_idx = min(i, self.num_levels - 1)
            target_h = len(encoder_features[i][0])
            target_w = len(encoder_features[i][0][0])
            upsampled = self._upsample_nearest(current, target_h, target_w)
            lateral = self._apply_1x1_conv(encoder_features[i], self._lateral_weights[level_idx])

            # 逐元素相加
            channels = len(upsampled)
            fused: List[List[List[float]]] = [
                [[0.0] * target_w for _ in range(target_h)] for _ in range(channels)
            ]
            for c in range(channels):
                for h in range(target_h):
                    for w in range(target_w):
                        fused[c][h][w] = upsampled[c][h][w] + lateral[c][h][w]

            # 平滑层
            smoothed = self._apply_1x1_conv(fused, self._smooth_weights[level_idx])
            pyramid.append(smoothed)
            current = smoothed

        pyramid.reverse()
        return pyramid


# ============================================================================
# ImplicitSurface - 神经隐式表面
# ============================================================================

class ImplicitSurface:
    """神经隐式表面表示：使用多层感知机 (MLP) 对 SDF 进行建模。

    给定一个3D点坐标，输出该点的有符号距离场值 (SDF)。
    """

    def __init__(self, config: TripoSRConfig) -> None:
        self.config = config
        self.hidden_dim = 128
        self.num_layers = 6
        self._weights: List[List[List[float]]] = []
        self._biases: List[List[float]] = []
        self._init_weights()

    def _init_weights(self) -> None:
        """使用 Xavier 初始化 MLP 权重。"""
        rng = random.Random(777)
        dims = [3] + [self.hidden_dim] * (self.num_layers - 1) + [1]
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            fan_out = dims[i + 1]
            std = math.sqrt(2.0 / (fan_in + fan_out))
            w: List[List[float]] = []
            for _ in range(fan_out):
                row = [rng.gauss(0, std) for _ in range(fan_in)]
                w.append(row)
            self._weights.append(w)
            self._biases.append([rng.gauss(0, 0.001) for _ in range(fan_out)])

    def _mlp_forward(self, x: List[float]) -> float:
        """MLP 前向传播。"""
        h = list(x)
        for layer_idx in range(self.num_layers):
            w = self._weights[layer_idx]
            b = self._biases[layer_idx]
            out_dim = len(w)
            new_h: List[float] = [0.0] * out_dim
            for j in range(out_dim):
                val = b[j]
                for k in range(len(h)):
                    val += w[j][k] * h[k]
                if layer_idx < self.num_layers - 1:
                    new_h[j] = _relu(val)
                else:
                    new_h[j] = _tanh(val)
            h = new_h
        return h[0]

    def query_sdf(self, point: Tuple[float, float, float]) -> float:
        """查询给定3D点的 SDF 值。

        Args:
            point: 3D坐标 (x, y, z)。

        Returns:
            该点的有符号距离场值，正值表示在表面外部，
            负值表示在内部。
        """
        return self._mlp_forward([point[0], point[1], point[2]])

    def query_gradient(self, point: Tuple[float, float, float],
                       eps: float = 1e-4) -> Tuple[float, float, float]:
        """通过有限差分计算 SDF 梯度（即表面法线方向）。

        Args:
            point: 3D坐标。
            eps: 有限差分步长。

        Returns:
            梯度向量 (dx, dy, dz)。
        """
        px, py, pz = point
        dx = (self.query_sdf((px + eps, py, pz)) - self.query_sdf((px - eps, py, pz))) / (2.0 * eps)
        dy = (self.query_sdf((px, py + eps, pz)) - self.query_sdf((px, py - eps, pz))) / (2.0 * eps)
        dz = (self.query_sdf((px, py, pz + eps)) - self.query_sdf((px, py, pz - eps))) / (2.0 * eps)
        return (dx, dy, dz)


# ============================================================================
# VolumeReconstructor - 3D体积重建
# ============================================================================

class VolumeReconstructor:
    """3D体积重建器：从图像特征和隐式表面模型重建3D体积。

    通过在规则网格上采样 SDF 值，构建用于 Marching Cubes 的
    标量场。
    """

    def __init__(self, config: TripoSRConfig) -> None:
        self.config = config
        self.surface = ImplicitSurface(config)

    def _world_to_grid(self, point: Tuple[float, float, float],
                       resolution: int, vol_range: float) -> Tuple[int, int, int]:
        """将世界坐标转换为体素网格坐标。"""
        x = int((point[0] / vol_range + 1.0) * 0.5 * (resolution - 1))
        y = int((point[1] / vol_range + 1.0) * 0.5 * (resolution - 1))
        z = int((point[2] / vol_range + 1.0) * 0.5 * (resolution - 1))
        return (
            _clamp(x, 0, resolution - 1),
            _clamp(y, 0, resolution - 1),
            _clamp(z, 0, resolution - 1),
        )

    def _grid_to_world(self, ix: int, iy: int, iz: int,
                       resolution: int, vol_range: float) -> Tuple[float, float, float]:
        """将体素网格坐标转换为世界坐标。"""
        x = (ix / (resolution - 1)) * 2.0 * vol_range - vol_range
        y = (iy / (resolution - 1)) * 2.0 * vol_range - vol_range
        z = (iz / (resolution - 1)) * 2.0 * vol_range - vol_range
        return (x, y, z)

    def reconstruct_volume(
        self,
        pyramid_features: List[List[List[List[float]]]],
    ) -> List[List[List[float]]]:
        """从特征金字塔重建3D SDF 体积。

        Args:
            pyramid_features: 特征金字塔输出。

        Returns:
            3D SDF 标量场，形状为 [res, res, res]。
        """
        res = self.config.volume_resolution
        vol_range = self.config.volume_range
        volume: List[List[List[float]]] = [
            [[0.0] * res for _ in range(res)] for _ in range(res)
        ]

        # 计算图像特征的统计量，用于调制 SDF
        feature_stats: List[float] = []
        for level_features in pyramid_features:
            total = 0.0
            count = 0
            for ch in level_features:
                for row in ch:
                    for val in row:
                        total += val
                        count += 1
            feature_stats.append(total / max(count, 1))

        # 在规则网格上采样 SDF
        for iz in range(res):
            for iy in range(res):
                for ix in range(res):
                    point = self._grid_to_world(ix, iy, iz, res, vol_range)
                    sdf_val = self.surface.query_sdf(point)

                    # 使用图像特征调制 SDF
                    modulation = 1.0
                    if feature_stats:
                        avg_feat = sum(feature_stats) / len(feature_stats)
                        modulation = _sigmoid(avg_feat * 0.1)

                    volume[ix][iy][iz] = sdf_val * modulation

        return volume


# ============================================================================
# MarchingCubes - Marching Cubes 网格提取
# ============================================================================

class MarchingCubes:
    """Marching Cubes 算法：从标量场中提取等值面网格。

    使用查找表方法，在体素网格的每个单元内确定三角面片。
    """

    # 256 种边配置的三角化表（简化版，每个配置最多5个三角形）
    _EDGE_TABLE: List[int] = []
    _TRI_TABLE: List[List[int]] = []

    def __init__(self, config: TripoSRConfig) -> None:
        self.config = config
        self.isovalue = config.mc_isovalue
        self._build_tables()

    def _build_tables(self) -> None:
        """构建 Marching Cubes 边和三角化查找表。"""
        self._EDGE_TABLE = [0] * 256
        self._TRI_TABLE = [[] for _ in range(256)]

        # 12 条边的顶点对索引
        edge_pairs = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
            (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
            (0, 4), (1, 5), (2, 6), (3, 7),  # 竖直边
        ]

        for case in range(256):
            edges = 0
            for e_idx, (v0, v1) in enumerate(edge_pairs):
                inside0 = bool(case & (1 << v0))
                inside1 = bool(case & (1 << v1))
                if inside0 != inside1:
                    edges |= (1 << e_idx)
            self._EDGE_TABLE[case] = edges

            # 简化三角化：对每条交叉边生成顶点
            tri_edges: List[int] = []
            for e_idx in range(12):
                if edges & (1 << e_idx):
                    tri_edges.append(e_idx)
            # 将边索引组成三角形（每3条边一个三角形）
            tris: List[int] = []
            for i in range(0, len(tri_edges) - 2, 3):
                tris.extend([tri_edges[i], tri_edges[i + 1], tri_edges[i + 2]])
            self._TRI_TABLE[case] = tris

    def _vertex_interp(
        self,
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
        v1: float,
        v2: float,
    ) -> Tuple[float, float, float]:
        """在两个顶点之间线性插值等值面交点。"""
        if abs(v1 - v2) < 1e-12:
            return p1
        t = (self.isovalue - v1) / (v2 - v1)
        t = _clamp(t, 0.0, 1.0)
        return (
            _lerp(p1[0], p2[0], t),
            _lerp(p1[1], p2[1], t),
            _lerp(p1[2], p2[2], t),
        )

    def extract(
        self,
        volume: List[List[List[float]]],
    ) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
        """从3D标量场提取三角网格。

        Args:
            volume: 3D SDF 标量场 [res, res, res]。

        Returns:
            (vertices, faces) 顶点列表和三角面片列表。
        """
        res = len(volume)
        vol_range = self.config.volume_range
        vertices: List[Tuple[float, float, float]] = []
        faces: List[Tuple[int, int, int]] = []
        vertex_map: Dict[Tuple[int, int, int], int] = {}

        def get_vertex(edge_key: Tuple[int, int, int],
                       p1: Tuple[float, float, float],
                       p2: Tuple[float, float, float],
                       v1: float, v2: float) -> int:
            if edge_key in vertex_map:
                return vertex_map[edge_key]
            pt = self._vertex_interp(p1, p2, v1, v2)
            idx = len(vertices)
            vertices.append(pt)
            vertex_map[edge_key] = idx
            return idx

        for iz in range(res - 1):
            for iy in range(res - 1):
                for ix in range(res - 1):
                    # 8 个角点的值
                    vals = [
                        volume[ix][iy][iz],
                        volume[ix + 1][iy][iz],
                        volume[ix + 1][iy + 1][iz],
                        volume[ix][iy + 1][iz],
                        volume[ix][iy][iz + 1],
                        volume[ix + 1][iy][iz + 1],
                        volume[ix + 1][iy + 1][iz + 1],
                        volume[ix][iy + 1][iz + 1],
                    ]

                    # 计算查找表索引
                    case_idx = 0
                    for i in range(8):
                        if vals[i] < self.isovalue:
                            case_idx |= (1 << i)

                    if case_idx == 0 or case_idx == 255:
                        continue

                    # 8 个角点的3D坐标
                    corners = [
                        self._grid_to_world(ix, iy, iz, res, vol_range),
                        self._grid_to_world(ix + 1, iy, iz, res, vol_range),
                        self._grid_to_world(ix + 1, iy + 1, iz, res, vol_range),
                        self._grid_to_world(ix, iy + 1, iz, res, vol_range),
                        self._grid_to_world(ix, iy, iz + 1, res, vol_range),
                        self._grid_to_world(ix + 1, iy, iz + 1, res, vol_range),
                        self._grid_to_world(ix + 1, iy + 1, iz + 1, res, vol_range),
                        self._grid_to_world(ix, iy + 1, iz + 1, res, vol_range),
                    ]

                    edge_vertex_pairs = [
                        (0, 1), (1, 2), (2, 3), (3, 0),
                        (4, 5), (5, 6), (6, 7), (7, 4),
                        (0, 4), (1, 5), (2, 6), (3, 7),
                    ]

                    # 计算每条边的交叉点顶点
                    edge_vertices: List[int] = [0] * 12
                    for e_idx, (v0, v1) in enumerate(edge_vertex_pairs):
                        if self._EDGE_TABLE[case_idx] & (1 << e_idx):
                            edge_key = (ix, iy, iz, e_idx)
                            edge_vertices[e_idx] = get_vertex(
                                edge_key,
                                corners[v0], corners[v1],
                                vals[v0], vals[v1],
                            )

                    # 生成三角面片
                    tri_edges = self._TRI_TABLE[case_idx]
                    for t in range(0, len(tri_edges) - 2, 3):
                        e0 = tri_edges[t]
                        e1 = tri_edges[t + 1]
                        e2 = tri_edges[t + 2]
                        faces.append((
                            edge_vertices[e0],
                            edge_vertices[e1],
                            edge_vertices[e2],
                        ))

        return vertices, faces

    def _grid_to_world(self, ix: int, iy: int, iz: int,
                       resolution: int, vol_range: float) -> Tuple[float, float, float]:
        x = (ix / (resolution - 1)) * 2.0 * vol_range - vol_range
        y = (iy / (resolution - 1)) * 2.0 * vol_range - vol_range
        z = (iz / (resolution - 1)) * 2.0 * vol_range - vol_range
        return (x, y, z)


# ============================================================================
# TextureGenerator - 纹理生成
# ============================================================================

class TextureGenerator:
    """纹理生成器：为3D网格生成 UV 纹理贴图。

    通过球面投影将3D顶点映射到2D UV 空间，然后基于原始图像
    和视角信息生成纹理颜色。
    """

    def __init__(self, config: TripoSRConfig) -> None:
        self.config = config
        self.tex_res = config.texture_resolution

    def _spherical_projection(self, point: Tuple[float, float, float]
                              ) -> Tuple[float, float]:
        """球面投影：将3D点映射到UV坐标。

        Args:
            point: 归一化的3D坐标。

        Returns:
            (u, v) UV 坐标，范围 [0, 1]。
        """
        x, y, z = point
        r = math.sqrt(x * x + y * y + z * z)
        if r < 1e-12:
            return (0.5, 0.5)
        x /= r
        y /= r
        z /= r

        theta = math.atan2(z, x)
        phi = math.asin(_clamp(y, -1.0, 1.0))

        u = (theta / math.pi + 1.0) * 0.5
        v = phi / math.pi + 0.5
        return (_clamp(u, 0.0, 1.0), _clamp(v, 0.0, 1.0))

    def generate_uvs(
        self,
        vertices: List[Tuple[float, float, float]],
    ) -> List[Tuple[float, float]]:
        """为所有顶点生成 UV 坐标。

        Args:
            vertices: 3D顶点列表。

        Returns:
            UV 坐标列表。
        """
        uvs: List[Tuple[float, float]] = []
        for v in vertices:
            uv = self._spherical_projection(v)
            uvs.append(uv)
        return uvs

    def generate_texture_map(
        self,
        vertices: List[Tuple[float, float, float]],
        normals: List[Tuple[float, float, float]],
        source_image: List[List[List[float]]],
    ) -> List[List[List[float]]]:
        """生成纹理贴图。

        Args:
            vertices: 3D顶点列表。
            normals: 法线列表。
            source_image: 源图像 [C, H, W]。

        Returns:
            纹理贴图 [C, tex_res, tex_res]。
        """
        res = self.tex_res
        img_h = len(source_image[0])
        img_w = len(source_image[0][0])
        channels = len(source_image)

        texture: List[List[List[float]]] = [
            [[0.0] * res for _ in range(res)] for _ in range(channels)
        ]

        # 为每个纹素采样颜色
        for ty in range(res):
            for tx in range(res):
                u = tx / res
                v = ty / res

                # 将 UV 转回3D方向
                theta = (u * 2.0 - 1.0) * math.pi
                phi = (v - 0.5) * math.pi
                dx = math.cos(phi) * math.cos(theta)
                dy = math.sin(phi)
                dz = math.cos(phi) * math.sin(theta)

                # 找到最近的顶点获取颜色信息
                direction = [dx, dy, dz]
                best_dist = float('inf')
                best_color: List[float] = [0.5, 0.5, 0.5]

                for vi, vert in enumerate(vertices):
                    vert_norm = _vec_normalize(list(vert))
                    dist = sum(abs(vert_norm[k] - direction[k]) for k in range(3))
                    if dist < best_dist:
                        best_dist = dist
                        # 从源图像采样
                        img_x = int(_clamp(u * img_w, 0, img_w - 1))
                        img_y = int(_clamp(v * img_h, 0, img_h - 1))
                        best_color = [source_image[c][img_y][img_x] for c in range(channels)]

                for c in range(channels):
                    texture[c][ty][tx] = _clamp(best_color[c], 0.0, 1.0)

        return texture


# ============================================================================
# NormalEstimator - 法线估计
# ============================================================================

class NormalEstimator:
    """法线估计器：基于邻域点云估计表面法线。

    使用主成分分析 (PCA) 方法，通过协方差矩阵的最小特征值
    对应的特征向量作为法线方向。
    """

    def __init__(self, config: TripoSRConfig) -> None:
        self.config = config
        self.radius = config.normal_estimation_radius

    def _compute_covariance(
        self,
        points: List[Tuple[float, float, float]],
        centroid: Tuple[float, float, float],
    ) -> List[List[float]]:
        """计算点集的 3x3 协方差矩阵。"""
        cov = [[0.0] * 3 for _ in range(3)]
        for p in points:
            d = [p[k] - centroid[k] for k in range(3)]
            for i in range(3):
                for j in range(3):
                    cov[i][j] += d[i] * d[j]
        n = max(len(points), 1)
        for i in range(3):
            for j in range(3):
                cov[i][j] /= n
        return cov

    def _power_iteration(
        self,
        matrix: List[List[float]],
        num_iterations: int = 50,
    ) -> Tuple[float, List[float]]:
        """幂迭代法求矩阵的最大特征值和特征向量。"""
        n = len(matrix)
        rng = random.Random(0)
        vec = [rng.gauss(0, 1) for _ in range(n)]
        norm = math.sqrt(sum(v * v for v in vec))
        vec = [v / norm for v in vec]

        eigenvalue = 0.0
        for _ in range(num_iterations):
            new_vec = [sum(matrix[i][j] * vec[j] for j in range(n)) for i in range(n)]
            norm = math.sqrt(sum(v * v for v in new_vec))
            if norm < 1e-12:
                break
            vec = [v / norm for v in new_vec]
            eigenvalue = sum(vec[i] * sum(matrix[i][j] * vec[j] for j in range(n))
                             for i in range(n))
        return eigenvalue, vec

    def _min_eigen_vector(
        self,
        cov: List[List[float]],
    ) -> List[float]:
        """通过迭代求最小特征向量（法线方向）。"""
        # 使用逆幂迭代
        n = 3
        vec = [1.0, 0.0, 0.0]
        for _ in range(100):
            # 解 cov * new_vec = vec (简化为迭代)
            new_vec = [sum(cov[i][j] * vec[j] for j in range(n)) for i in range(n)]
            norm = math.sqrt(sum(v * v for v in new_vec))
            if norm < 1e-12:
                break
            new_vec = [v / norm for v in new_vec]
            vec = new_vec

        # 最小特征向量对应法线
        # 简化：使用叉积方法
        if abs(vec[0]) < abs(vec[1]):
            ref = [1.0, 0.0, 0.0]
        else:
            ref = [0.0, 1.0, 0.0]
        normal = _vec_cross(vec, ref)
        norm = _vec_length(normal)
        if norm > 1e-12:
            normal = [n / norm for n in normal]
        return normal

    def estimate(
        self,
        vertices: List[Tuple[float, float, float]],
        faces: List[Tuple[int, int, int]],
    ) -> List[Tuple[float, float, float]]:
        """估计所有顶点的法线。

        Args:
            vertices: 顶点列表。
            faces: 三角面片列表。

        Returns:
            法线列表，与顶点一一对应。
        """
        normals: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * len(vertices)
        vertex_face_count: List[int] = [0] * len(vertices)

        # 首先基于面法线计算
        for face in faces:
            i0, i1, i2 = face
            if i0 >= len(vertices) or i1 >= len(vertices) or i2 >= len(vertices):
                continue
            v0 = vertices[i0]
            v1 = vertices[i1]
            v2 = vertices[i2]
            e1 = _vec_sub(list(v1), list(v0))
            e2 = _vec_sub(list(v2), list(v0))
            face_normal = _vec_cross(e1, e2)
            norm = _vec_length(face_normal)
            if norm > 1e-12:
                face_normal = [fn / norm for fn in face_normal]

            for idx in (i0, i1, i2):
                old = list(normals[idx])
                new_n = [old[k] + face_normal[k] for k in range(3)]
                normals[idx] = (new_n[0], new_n[1], new_n[2])
                vertex_face_count[idx] += 1

        # 归一化
        for i in range(len(normals)):
            if vertex_face_count[i] > 0:
                n = list(normals[i])
                norm = _vec_length(n)
                if norm > 1e-12:
                    normals[i] = (n[0] / norm, n[1] / norm, n[2] / norm)

        # 对法线为零的顶点使用 PCA 估计
        for i in range(len(vertices)):
            if vertex_face_count[i] == 0:
                # 收集邻域点
                neighbors: List[Tuple[float, float, float]] = []
                center = vertices[i]
                for j, v in enumerate(vertices):
                    if i != j:
                        dist = _vec_length(_vec_sub(list(v), list(center)))
                        if dist < self.radius * 0.1:
                            neighbors.append(v)
                if len(neighbors) >= 3:
                    centroid = (
                        sum(p[0] for p in neighbors) / len(neighbors),
                        sum(p[1] for p in neighbors) / len(neighbors),
                        sum(p[2] for p in neighbors) / len(neighbors),
                    )
                    cov = self._compute_covariance(neighbors, centroid)
                    normal = self._min_eigen_vector(cov)
                    normals[i] = (normal[0], normal[1], normal[2])

        return normals


# ============================================================================
# MeshPostProcessor - 网格后处理
# ============================================================================

class MeshPostProcessor:
    """网格后处理器：对提取的网格进行平滑和优化。

    包含拉普拉斯平滑、网格简化和拓扑修复功能。
    """

    def __init__(self, config: TripoSRConfig) -> None:
        self.config = config
        self.iterations = config.smoothing_iterations
        self.lam = config.smoothing_lambda

    def _build_adjacency(
        self,
        vertices: List[Tuple[float, float, float]],
        faces: List[Tuple[int, int, int]],
    ) -> Dict[int, List[int]]:
        """构建顶点邻接表。"""
        adj: Dict[int, List[int]] = {i: [] for i in range(len(vertices))}
        for face in faces:
            i0, i1, i2 = face
            for a, b in [(i0, i1), (i1, i2), (i2, i0)]:
                if a < len(vertices) and b < len(vertices):
                    if b not in adj[a]:
                        adj[a].append(b)
                    if a not in adj[b]:
                        adj[b].append(a)
        return adj

    def laplacian_smooth(
        self,
        vertices: List[Tuple[float, float, float]],
        faces: List[Tuple[int, int, int]],
    ) -> List[Tuple[float, float, float]]:
        """拉普拉斯平滑：将每个顶点移动到其邻域的质心。

        Args:
            vertices: 原始顶点列表。
            faces: 三角面片列表。

        Returns:
            平滑后的顶点列表。
        """
        adj = self._build_adjacency(vertices, faces)
        current = [list(v) for v in vertices]

        for iteration in range(self.iterations):
            new_vertices = [list(v) for v in current]
            for i in range(len(current)):
                neighbors = adj.get(i, [])
                if not neighbors:
                    continue
                # 计算邻域质心
                centroid = [0.0, 0.0, 0.0]
                for n in neighbors:
                    for k in range(3):
                        centroid[k] += current[n][k]
                for k in range(3):
                    centroid[k] /= len(neighbors)

                # 混合原始位置和平滑位置
                alpha = self.lam * (1.0 - iteration / max(self.iterations, 1) * 0.5)
                for k in range(3):
                    new_vertices[i][k] = _lerp(current[i][k], centroid[k], alpha)

            current = new_vertices

        return [tuple(v) for v in current]

    def remove_duplicate_vertices(
        self,
        vertices: List[Tuple[float, float, float]],
        faces: List[Tuple[int, int, int]],
        tolerance: float = 1e-6,
    ) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
        """移除重复顶点并更新面片索引。

        Args:
            vertices: 顶点列表。
            faces: 面片列表。
            tolerance: 距离容差。

        Returns:
            (去重后的顶点, 更新后的面片)。
        """
        unique_vertices: List[Tuple[float, float, float]] = []
        index_map: Dict[int, int] = {}

        for i, v in enumerate(vertices):
            found = -1
            for j, uv in enumerate(unique_vertices):
                dist = math.sqrt(sum((v[k] - uv[k]) ** 2 for k in range(3)))
                if dist < tolerance:
                    found = j
                    break
            if found >= 0:
                index_map[i] = found
            else:
                index_map[i] = len(unique_vertices)
                unique_vertices.append(v)

        new_faces = [
            (index_map[f[0]], index_map[f[1]], index_map[f[2]])
            for f in faces
        ]

        # 移除退化面片
        valid_faces = [
            f for f in new_faces
            if f[0] != f[1] and f[1] != f[2] and f[0] != f[2]
        ]

        return unique_vertices, valid_faces

    def process(
        self,
        vertices: List[Tuple[float, float, float]],
        faces: List[Tuple[int, int, int]],
    ) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
        """执行完整的后处理流程。

        Args:
            vertices: 原始顶点。
            faces: 原始面片。

        Returns:
            处理后的 (顶点, 面片)。
        """
        # 拉普拉斯平滑
        vertices = self.laplacian_smooth(vertices, faces)

        # 去除重复顶点
        vertices, faces = self.remove_duplicate_vertices(vertices, faces)

        return vertices, faces


# ============================================================================
# TripoSRPipeline - 主管线
# ============================================================================

class TripoSRPipeline:
    """TripoSR 单图到3D网格生成主管线。

    将图像编码器、特征金字塔、体积重建、Marching Cubes、
    纹理生成和后处理组件串联为完整的生成流程。
    """

    def __init__(self, config: Optional[TripoSRConfig] = None) -> None:
        self.config = config or TripoSRConfig()
        if self.config.seed is not None:
            random.seed(self.config.seed)

        # 初始化各组件
        self.image_encoder = ImageEncoder(self.config)
        self.feature_pyramid = FeaturePyramid(self.config)
        self.volume_reconstructor = VolumeReconstructor(self.config)
        self.marching_cubes = MarchingCubes(self.config)
        self.texture_generator = TextureGenerator(self.config)
        self.normal_estimator = NormalEstimator(self.config)
        self.mesh_post_processor = MeshPostProcessor(self.config)

        # 管线状态
        self._pipeline_id: str = _generate_id()
        self._is_loaded: bool = True

    @property
    def pipeline_id(self) -> str:
        return self._pipeline_id

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def preprocess_image(
        self,
        image: List[List[List[float]]],
        target_size: Optional[int] = None,
    ) -> List[List[List[float]]]:
        """预处理输入图像：调整大小并归一化。

        Args:
            image: 输入图像 [C, H, W]。
            target_size: 目标尺寸，默认使用配置值。

        Returns:
            预处理后的图像。
        """
        size = target_size or self.config.image_size
        channels = len(image)
        src_h = len(image[0])
        src_w = len(image[0][0])

        output: List[List[List[float]]] = [
            [[0.0] * size for _ in range(size)] for _ in range(channels)
        ]

        for c in range(channels):
            for oh in range(size):
                for ow in range(size):
                    # 双线性插值
                    src_y = oh * (src_h - 1) / max(size - 1, 1)
                    src_x = ow * (src_w - 1) / max(size - 1, 1)
                    y0 = int(src_y)
                    x0 = int(src_x)
                    y1 = min(y0 + 1, src_h - 1)
                    x1 = min(x0 + 1, src_w - 1)
                    fy = src_y - y0
                    fx = src_x - x0
                    val = (image[c][y0][x0] * (1 - fy) * (1 - fx)
                           + image[c][y1][x0] * fy * (1 - fx)
                           + image[c][y0][x1] * (1 - fy) * fx
                           + image[c][y1][x1] * fy * fx)
                    output[c][oh][ow] = _clamp(val, 0.0, 1.0)

        return output

    def generate(
        self,
        image: List[List[List[float]]],
        return_texture: bool = True,
    ) -> Dict[str, Any]:
        """从单张图像生成3D网格。

        Args:
            image: 输入图像 [C, H, W]，C=3 (RGB)，值范围 [0, 1]。
            return_texture: 是否生成纹理。

        Returns:
            包含顶点、面片、法线、UV 和纹理的字典。
        """
        start_time = time.time()

        # 1. 预处理
        processed_image = self.preprocess_image(image)

        # 2. 图像编码
        encoder_features = self.image_encoder.encode(processed_image)

        # 3. 特征金字塔
        pyramid_features = self.feature_pyramid.build(encoder_features)

        # 4. 体积重建
        volume = self.volume_reconstructor.reconstruct_volume(pyramid_features)

        # 5. Marching Cubes 提取网格
        vertices, faces = self.marching_cubes.extract(volume)

        if not vertices:
            return {
                "vertices": [],
                "faces": [],
                "normals": [],
                "uvs": [],
                "texture": None,
                "metadata": {
                    "pipeline_id": self._pipeline_id,
                    "generation_time": time.time() - start_time,
                    "num_vertices": 0,
                    "num_faces": 0,
                    "status": "empty",
                },
            }

        # 6. 后处理
        vertices, faces = self.mesh_post_processor.process(vertices, faces)

        # 7. 法线估计
        normals = self.normal_estimator.estimate(vertices, faces)

        # 8. 纹理生成
        uvs: List[Tuple[float, float]] = []
        texture: Optional[List[List[List[float]]]] = None
        if return_texture:
            uvs = self.texture_generator.generate_uvs(vertices)
            texture = self.texture_generator.generate_texture_map(
                vertices, normals, processed_image
            )

        elapsed = time.time() - start_time

        return {
            "vertices": vertices,
            "faces": faces,
            "normals": normals,
            "uvs": uvs,
            "texture": texture,
            "metadata": {
                "pipeline_id": self._pipeline_id,
                "generation_time": elapsed,
                "num_vertices": len(vertices),
                "num_faces": len(faces),
                "volume_resolution": self.config.volume_resolution,
                "status": "success",
            },
        }

    def generate_multi_view(
        self,
        image: List[List[List[float]]],
        num_views: int = 6,
    ) -> List[Dict[str, Any]]:
        """生成多视角3D网格变体。

        Args:
            image: 输入图像。
            num_views: 视角数量。

        Returns:
            多个视角的网格结果列表。
        """
        results: List[Dict[str, Any]] = []
        for view_idx in range(num_views):
            angle = 2.0 * math.pi * view_idx / num_views
            # 旋转图像特征模拟不同视角
            rotated_image = self._rotate_image_features(image, angle)
            result = self.generate(rotated_image, return_texture=True)
            result["metadata"]["view_index"] = view_idx
            result["metadata"]["view_angle"] = angle
            results.append(result)
        return results

    def _rotate_image_features(
        self,
        image: List[List[List[float]]],
        angle: float,
    ) -> List[List[List[float]]]:
        """通过旋转模拟不同视角的图像特征。"""
        channels = len(image)
        h = len(image[0])
        w = len(image[0][0])
        cx, cy = w / 2.0, h / 2.0
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        output: List[List[List[float]]] = [
            [[0.0] * w for _ in range(h)] for _ in range(channels)
        ]

        for c in range(channels):
            for oy in range(h):
                for ox in range(w):
                    dx = ox - cx
                    dy = oy - cy
                    sx = cos_a * dx - sin_a * dy + cx
                    sy = sin_a * dx + cos_a * dy + cy
                    ix = int(_clamp(sx, 0, w - 1))
                    iy = int(_clamp(sy, 0, h - 1))
                    output[c][oy][ox] = image[c][iy][ix]

        return output

    def export_obj(
        self,
        vertices: List[Tuple[float, float, float]],
        faces: List[Tuple[int, int, int]],
        normals: Optional[List[Tuple[float, float, float]]] = None,
        uvs: Optional[List[Tuple[float, float]]] = None,
    ) -> str:
        """将网格导出为 OBJ 格式字符串。

        Args:
            vertices: 顶点列表。
            faces: 面片列表。
            normals: 法线列表（可选）。
            uvs: UV 坐标列表（可选）。

        Returns:
            OBJ 格式字符串。
        """
        lines: List[str] = ["# TripoSR Generated Mesh", f"# Vertices: {len(vertices)}",
                            f"# Faces: {len(faces)}", ""]

        for v in vertices:
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

        lines.append("")

        if uvs:
            for uv in uvs:
                lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}")
            lines.append("")

        if normals:
            for n in normals:
                lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
            lines.append("")

        for f in faces:
            if uvs and normals:
                i0, i1, i2 = f[0] + 1, f[1] + 1, f[2] + 1
                lines.append(f"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}")
            elif normals:
                i0, i1, i2 = f[0] + 1, f[1] + 1, f[2] + 1
                lines.append(f"f {i0}//{i0} {i1}//{i1} {i2}//{i2}")
            else:
                i0, i1, i2 = f[0] + 1, f[1] + 1, f[2] + 1
                lines.append(f"f {i0} {i1} {i2}")

        return "\n".join(lines)

    def get_pipeline_info(self) -> Dict[str, Any]:
        """获取管线信息。"""
        return {
            "pipeline_id": self._pipeline_id,
            "is_loaded": self._is_loaded,
            "config": self.config.to_dict(),
            "components": [
                "ImageEncoder",
                "FeaturePyramid",
                "VolumeReconstructor",
                "ImplicitSurface",
                "MarchingCubes",
                "TextureGenerator",
                "NormalEstimator",
                "MeshPostProcessor",
            ],
        }
