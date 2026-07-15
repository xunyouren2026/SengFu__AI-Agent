"""
Stable Video Diffusion Pipeline - SVD视频生成管线

本模块实现了完整的Stable Video Diffusion管线，包含图像到视频生成、
帧插值、时间一致性控制、运动控制和条件模块。使用纯Python模拟
扩散过程和噪声调度。仅使用标准库，不依赖外部库。
"""

import math
import random
import time
import threading
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


# ============================================================================
# 辅助函数
# ============================================================================

def _generate_id() -> str:
    """生成唯一ID"""
    import hashlib
    raw = f"{time.time()}-{random.random()}-{threading.get_ident()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """将值限制在范围内"""
    return max(min_val, min(max_val, value))


def _lerp(a: float, b: float, t: float) -> float:
    """线性插值"""
    return a + (b - a) * t


def _smooth_step(t: float) -> float:
    """平滑步进函数"""
    t = _clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _sigmoid(x: float) -> float:
    """Sigmoid函数"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def _gaussian_noise(mean: float, std: float) -> float:
    """生成高斯噪声（Box-Muller变换）"""
    u1 = random.random()
    u2 = random.random()
    while u1 == 0:
        u1 = random.random()
    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mean + std * z0


def _matmul_2d(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """2D矩阵乘法"""
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    result = [[0.0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            s = 0.0
            for k in range(cols_a):
                s += a[i][k] * b[k][j]
            result[i][j] = s
    return result


def _linear_transform(
    x: List[float], weight: List[List[float]], bias: List[float]
) -> List[float]:
    """线性变换 y = xW^T + b"""
    w_t = [[weight[j][i] for j in range(len(weight))] for i in range(len(weight[0]))]
    result = _matmul_2d([x], w_t)[0]
    for i in range(len(result)):
        result[i] += bias[i]
    return result


def _layer_norm(x: List[float], eps: float = 1e-5) -> List[float]:
    """层归一化"""
    mean = sum(x) / len(x)
    var = sum((v - mean) ** 2 for v in x) / len(x)
    std = math.sqrt(var + eps)
    return [(v - mean) / std for v in x]


def _relu(x: float) -> float:
    return max(0.0, x)


def _gelu(x: float) -> float:
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


# ============================================================================
# 数据结构
# ============================================================================

class NoiseScheduleType(Enum):
    """噪声调度类型"""
    LINEAR = "linear"
    COSINE = "cosine"
    SCHEDULED = "scheduled"


@dataclass
class SVDConfig:
    """SVD管线配置"""
    # 模型参数
    latent_channels: int = 4
    latent_height: int = 48
    latent_width: int = 48
    num_frames: int = 14
    fps: int = 6

    # 扩散参数
    num_inference_steps: int = 25
    num_train_timesteps: int = 1000
    noise_schedule: NoiseScheduleType = NoiseScheduleType.LINEAR
    beta_start: float = 0.00085
    beta_end: float = 0.012

    # 条件参数
    conditioning_scale: float = 1.0
    motion_bucket_id: int = 127
    fps_id: int = 6

    # 解码参数
    num_decoder_layers: int = 4
    decoder_hidden_dim: int = 64

    # 帧插值参数
    interpolation_method: str = "linear"

    # 时间一致性参数
    temporal_consistency_weight: float = 0.5
    temporal_window_size: int = 3


@dataclass
class LatentFrame:
    """潜在帧数据"""
    data: List[List[List[float]]] = field(default_factory=list)
    height: int = 0
    width: int = 0
    channels: int = 4
    timestep: int = 0

    def get_flat(self) -> List[float]:
        """获取展平数据"""
        flat: List[float] = []
        for c in range(self.channels):
            for y in range(self.height):
                for x in range(self.width):
                    if self.data and c < len(self.data) and y < len(self.data[c]) and x < len(self.data[c][y]):
                        flat.append(self.data[c][y][x])
                    else:
                        flat.append(0.0)
        return flat


@dataclass
class VideoFrame:
    """视频帧"""
    pixels: List[List[Tuple[float, float, float]]] = field(default_factory=list)
    width: int = 0
    height: int = 0
    timestamp: float = 0.0

    def get_pixel(self, x: int, y: int) -> Tuple[float, float, float]:
        """获取像素值"""
        if 0 <= y < self.height and 0 <= x < self.width:
            return self.pixels[y][x]
        return (0.0, 0.0, 0.0)

    def set_pixel(self, x: int, y: int, color: Tuple[float, float, float]) -> None:
        """设置像素值"""
        if 0 <= y < self.height and 0 <= x < self.width:
            self.pixels[y][x] = color


@dataclass
class VideoOutput:
    """视频输出"""
    frames: List[VideoFrame] = field(default_factory=list)
    fps: int = 6
    width: int = 0
    height: int = 0
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageInput:
    """图像输入"""
    pixels: List[List[Tuple[float, float, float]]] = field(default_factory=list)
    width: int = 0
    height: int = 0
    embedding: List[float] = field(default_factory=list)


# ============================================================================
# NoiseScheduler - 噪声调度器
# ============================================================================

class NoiseScheduler:
    """
    噪声调度器：管理扩散过程中的噪声级别。

    支持线性、余弦和自定义调度策略。
    """

    def __init__(self, config: SVDConfig):
        self._config = config
        self._betas: List[float] = []
        self._alphas: List[float] = []
        self._alphas_cumprod: List[float] = []
        self._sqrt_alphas_cumprod: List[float] = []
        self._sqrt_one_minus_alphas_cumprod: List[float] = []
        self._build_schedule()

    def _build_schedule(self) -> None:
        """构建噪声调度表"""
        timesteps = self._config.num_train_timesteps
        if self._config.noise_schedule == NoiseScheduleType.LINEAR:
            self._betas = [
                _lerp(self._config.beta_start, self._config.beta_end, i / timesteps)
                for i in range(timesteps)
            ]
        elif self._config.noise_schedule == NoiseScheduleType.COSINE:
            s = 0.008
            steps = timesteps + 1
            f = [
                math.cos(((i / steps) + s) / (1.0 + s) * math.pi / 2.0) ** 2
                for i in range(steps)
            ]
            self._betas = [
                min(1.0 - f[i] / f[i + 1], 0.999)
                for i in range(timesteps)
            ]
        else:
            self._betas = [
                _lerp(self._config.beta_start, self._config.beta_end, i / timesteps)
                for i in range(timesteps)
            ]

        self._alphas = [1.0 - b for b in self._betas]
        self._alphas_cumprod = []
        cumprod = 1.0
        for a in self._alphas:
            cumprod *= a
            self._alphas_cumprod.append(cumprod)

        self._sqrt_alphas_cumprod = [math.sqrt(a) for a in self._alphas_cumprod]
        self._sqrt_one_minus_alphas_cumprod = [
            math.sqrt(1.0 - a) for a in self._alphas_cumprod
        ]

    def add_noise(
        self, original: List[float], noise: List[float], timestep: int
    ) -> List[float]:
        """
        前向扩散：向原始数据添加噪声。
        x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise
        """
        sqrt_alpha = self._sqrt_alphas_cumprod[timestep]
        sqrt_one_minus = self._sqrt_one_minus_alphas_cumprod[timestep]
        return [
            sqrt_alpha * o + sqrt_one_minus * n
            for o, n in zip(original, noise)
        ]

    def get_timesteps(self, num_inference_steps: int) -> List[int]:
        """获取推理时间步序列"""
        step_ratio = self._config.num_train_timesteps // num_inference_steps
        timesteps = list(range(0, self._config.num_train_timesteps, step_ratio))
        timesteps.reverse()
        return timesteps

    def get_prev_timestep(self, timestep: int) -> int:
        """获取前一个时间步"""
        step_ratio = (
            self._config.num_train_timesteps // self._config.num_inference_steps
        )
        prev = timestep - step_ratio
        return max(0, prev)

    def get_alpha_cumprod(self, timestep: int) -> float:
        """获取累积alpha值"""
        if timestep < len(self._alphas_cumprod):
            return self._alphas_cumprod[timestep]
        return self._alphas_cumprod[-1]


# ============================================================================
# ConditioningModule - 条件模块
# ============================================================================

class ConditioningModule:
    """
    条件模块：处理图像编码、运动条件和FPS条件。

    模拟CLIP图像编码器和运动条件编码器。
    """

    def __init__(self, config: SVDConfig):
        self._config = config
        self._image_encoder_dim = config.decoder_hidden_dim
        self._motion_buckets = 256
        self._fps_buckets = 32

        # 模拟编码器权重
        random.seed(42)
        self._enc_weights = [
            [random.gauss(0, 0.02) for _ in range(self._image_encoder_dim)]
            for _ in range(self._image_encoder_dim)
        ]
        self._enc_bias = [0.0] * self._image_encoder_dim
        random.seed()

    def encode_image(self, image: ImageInput) -> List[float]:
        """
        编码输入图像为条件向量。
        模拟CLIP图像编码器的行为。
        """
        if not image.pixels or image.width == 0 or image.height == 0:
            return [0.0] * self._image_encoder_dim

        # 从图像提取特征
        features: List[float] = []
        h, w = image.height, image.width

        # 降采样到固定大小
        target_size = 16
        for cy in range(target_size):
            for cx in range(target_size):
                sy = int(cy * h / target_size)
                sx = int(cx * w / target_size)
                if sy < h and sx < w:
                    r, g, b = image.pixels[sy][sx]
                    features.extend([r / 255.0, g / 255.0, b / 255.0])
                else:
                    features.extend([0.0, 0.0, 0.0])

        # 通过模拟编码器
        encoded = self._encode_features(features)
        return encoded

    def _encode_features(self, features: List[float]) -> List[float]:
        """通过模拟神经网络编码特征"""
        # 填充或截断到编码器维度
        dim = self._image_encoder_dim
        if len(features) < dim:
            features = features + [0.0] * (dim - len(features))
        else:
            features = features[:dim]

        # 简单变换模拟编码
        result = _layer_norm(features)
        # 非线性激活
        result = [_gelu(v) for v in result]
        return result

    def encode_motion(self, motion_bucket_id: int) -> List[float]:
        """编码运动条件"""
        embedding = [0.0] * self._image_encoder_dim
        normalized = motion_bucket_id / self._motion_buckets
        for i in range(self._image_encoder_dim):
            freq = (i + 1) / self._image_encoder_dim
            embedding[i] = math.sin(normalized * math.pi * freq) * 0.5
        return embedding

    def encode_fps(self, fps_id: int) -> List[float]:
        """编码FPS条件"""
        embedding = [0.0] * self._image_encoder_dim
        normalized = fps_id / self._fps_buckets
        for i in range(self._image_encoder_dim):
            freq = (i + 1) / self._image_encoder_dim
            embedding[i] = math.cos(normalized * math.pi * freq) * 0.3
        return embedding

    def combine_conditions(
        self,
        image_emb: List[float],
        motion_emb: List[float],
        fps_emb: List[float],
    ) -> List[float]:
        """组合所有条件"""
        dim = self._image_encoder_dim
        combined: List[float] = []
        for i in range(dim):
            val = (
                image_emb[i] * self._config.conditioning_scale
                + motion_emb[i] * 0.3
                + fps_emb[i] * 0.2
            )
            combined.append(val)
        return combined


# ============================================================================
# TemporalConsistency - 时间一致性模块
# ============================================================================

class TemporalConsistency:
    """
    时间一致性模块：确保生成的视频帧之间保持视觉一致性。

    使用滑动窗口和光流模拟来保持帧间连贯性。
    """

    def __init__(self, config: SVDConfig):
        self._config = config
        self._window_size = config.temporal_window_size

    def enforce_consistency(
        self, frames: List[List[List[Tuple[float, float, float]]]]
    ) -> List[List[List[Tuple[float, float, float]]]]:
        """
        在帧序列上强制时间一致性。

        使用加权平均平滑相邻帧。
        """
        if len(frames) <= 1:
            return frames

        height = len(frames[0])
        if height == 0:
            return frames
        width = len(frames[0][0])

        weight = self._config.temporal_consistency_weight
        result: List[List[List[Tuple[float, float, float]]]] = []

        for t in range(len(frames)):
            if t == 0 or t == len(frames) - 1:
                result.append(frames[t])
                continue

            # 计算加权平均
            smoothed: List[List[Tuple[float, float, float]]] = []
            for y in range(height):
                row: List[Tuple[float, float, float]] = []
                for x in range(width):
                    current = frames[t][y][x]
                    prev = frames[t - 1][y][x]
                    next_f = frames[t + 1][y][x]

                    blended = (
                        _lerp(prev[0], current[0], 1.0 - weight),
                        _lerp(prev[1], current[1], 1.0 - weight),
                        _lerp(prev[2], current[2], 1.0 - weight),
                    )
                    blended = (
                        _lerp(blended[0], next_f[0], weight * 0.5),
                        _lerp(blended[1], next_f[1], weight * 0.5),
                        _lerp(blended[2], next_f[2], weight * 0.5),
                    )
                    row.append(blended)
                smoothed.append(row)
            result.append(smoothed)

        return result

    def compute_optical_flow(
        self,
        frame_a: List[List[Tuple[float, float, float]]],
        frame_b: List[List[Tuple[float, float, float]]],
        block_size: int = 8,
    ) -> List[List[Tuple[float, float]]]:
        """
        模拟光流计算（块匹配法）。

        返回每个块的运动向量 (dx, dy)。
        """
        if not frame_a or not frame_b:
            return []

        h = len(frame_a)
        w = len(frame_a[0]) if h > 0 else 0
        flow_h = h // block_size
        flow_w = w // block_size

        flow: List[List[Tuple[float, float]]] = []
        for by in range(flow_h):
            row: List[Tuple[float, float]] = []
            for bx in range(flow_w):
                best_dx, best_dy = 0.0, 0.0
                best_sad = float('inf')
                search_range = 4

                for dy in range(-search_range, search_range + 1):
                    for dx in range(-search_range, search_range + 1):
                        sad = self._compute_sad(
                            frame_a, frame_b,
                            bx * block_size, by * block_size,
                            dx, dy, block_size
                        )
                        if sad < best_sad:
                            best_sad = sad
                            best_dx = float(dx)
                            best_dy = float(dy)

                row.append((best_dx, best_dy))
            flow.append(row)

        return flow

    def _compute_sad(
        self,
        frame_a: List[List[Tuple[float, float, float]]],
        frame_b: List[List[Tuple[float, float, float]]],
        bx: int, by: int,
        dx: int, dy: int,
        block_size: int,
    ) -> float:
        """计算绝对差之和（SAD）"""
        h = len(frame_a)
        w = len(frame_a[0]) if h > 0 else 0
        sad = 0.0
        count = 0

        for y in range(block_size):
            for x in range(block_size):
                ay, ax = by + y, bx + x
                by2, bx2 = by + y + dy, bx + x + dx

                if (0 <= ay < h and 0 <= ax < w
                        and 0 <= by2 < h and 0 <= bx2 < w):
                    pa = frame_a[ay][ax]
                    pb = frame_b[by2][bx2]
                    sad += abs(pa[0] - pb[0]) + abs(pa[1] - pb[1]) + abs(pa[2] - pb[2])
                    count += 1

        return sad / max(count, 1)

    def warp_frame(
        self,
        frame: List[List[Tuple[float, float, float]]],
        flow: List[List[Tuple[float, float]]],
        block_size: int = 8,
    ) -> List[List[Tuple[float, float, float]]]:
        """根据光流扭曲帧"""
        if not frame or not flow:
            return frame

        h = len(frame)
        w = len(frame[0]) if h > 0 else 0
        result = [
            [frame[y][x] for x in range(w)]
            for y in range(h)
        ]

        flow_h = len(flow)
        flow_w = len(flow[0]) if flow_h > 0 else 0

        for by in range(flow_h):
            for bx in range(flow_w):
                dx, dy = flow[by][bx]
                for y in range(block_size):
                    for x in range(block_size):
                        sy, sx = by * block_size + y, bx * block_size + x
                        ty = int(sy + dy)
                        tx = int(sx + dx)
                        if 0 <= ty < h and 0 <= tx < w:
                            result[ty][tx] = frame[sy][sx]

        return result


# ============================================================================
# MotionController - 运动控制器
# ============================================================================

class MotionController:
    """
    运动控制器：控制生成视频中的运动幅度和方向。

    支持全局运动（平移、缩放、旋转）和局部运动控制。
    """

    def __init__(self, config: SVDConfig):
        self._config = config
        self._motion_scale = config.motion_bucket_id / 255.0

    def generate_motion_field(
        self,
        width: int,
        height: int,
        num_frames: int,
        motion_type: str = "zoom_in",
    ) -> List[List[Tuple[float, float]]]:
        """
        生成运动场。

        Args:
            motion_type: 运动类型 (zoom_in, zoom_out, pan_left, pan_right,
                        pan_up, pan_down, rotate, static)
        """
        fields: List[List[Tuple[float, float]]] = []
        cx, cy = width / 2.0, height / 2.0

        for frame_idx in range(num_frames):
            t = frame_idx / max(num_frames - 1, 1)
            field: List[List[Tuple[float, float]]] = []

            for y in range(height):
                row: List[Tuple[float, float]] = []
                for x in range(width):
                    dx, dy = 0.0, 0.0
                    rel_x = (x - cx) / cx
                    rel_y = (y - cy) / cy

                    if motion_type == "zoom_in":
                        dx = -rel_x * self._motion_scale * t * 0.5
                        dy = -rel_y * self._motion_scale * t * 0.5
                    elif motion_type == "zoom_out":
                        dx = rel_x * self._motion_scale * t * 0.5
                        dy = rel_y * self._motion_scale * t * 0.5
                    elif motion_type == "pan_left":
                        dx = -self._motion_scale * t * 2.0
                    elif motion_type == "pan_right":
                        dx = self._motion_scale * t * 2.0
                    elif motion_type == "pan_up":
                        dy = -self._motion_scale * t * 2.0
                    elif motion_type == "pan_down":
                        dy = self._motion_scale * t * 2.0
                    elif motion_type == "rotate":
                        angle = t * math.pi * 0.1 * self._motion_scale
                        cos_a = math.cos(angle)
                        sin_a = math.sin(angle)
                        dx = (rel_x * cos_a - rel_y * sin_a - rel_x) * 10.0
                        dy = (rel_x * sin_a + rel_y * cos_a - rel_y) * 10.0

                    row.append((dx, dy))
                field.append(row)
            fields.append(field)

        return fields

    def apply_motion_to_latent(
        self,
        latent: List[float],
        motion_field: List[List[Tuple[float, float]]],
        latent_h: int,
        latent_w: int,
    ) -> List[float]:
        """将运动场应用到潜在表示"""
        result = list(latent)
        channels = 4
        flat_size = channels * latent_h * latent_w

        for y in range(latent_h):
            for x in range(latent_w):
                fy = int(y * len(motion_field) / latent_h)
                fx = int(x * len(motion_field[0]) / latent_w) if motion_field else 0
                if fy < len(motion_field) and fx < len(motion_field[fy]):
                    dx, dy = motion_field[fy][fx]
                    for c in range(channels):
                        idx = c * latent_h * latent_w + y * latent_w + x
                        if idx < len(result):
                            result[idx] += (dx + dy) * 0.01

        return result


# ============================================================================
# FrameInterpolator - 帧插值器
# ============================================================================

class FrameInterpolator:
    """
    帧插值器：在关键帧之间生成中间帧。

    支持线性插值、光流插值和基于运动的插值。
    """

    def __init__(self, config: SVDConfig):
        self._config = config
        self._temporal = TemporalConsistency(config)

    def interpolate_linear(
        self,
        frame_a: List[List[Tuple[float, float, float]]],
        frame_b: List[List[Tuple[float, float, float]]],
        num_intermediate: int = 1,
    ) -> List[List[List[Tuple[float, float, float]]]]:
        """线性帧插值"""
        if not frame_a or not frame_b:
            return []

        h = len(frame_a)
        w = len(frame_a[0]) if h > 0 else 0
        result: List[List[List[Tuple[float, float, float]]]] = []

        for i in range(1, num_intermediate + 1):
            t = i / (num_intermediate + 1)
            t = _smooth_step(t)
            frame: List[List[Tuple[float, float, float]]] = []
            for y in range(h):
                row: List[Tuple[float, float, float]] = []
                for x in range(w):
                    pa = frame_a[y][x]
                    pb = frame_b[y][x]
                    pixel = (
                        _lerp(pa[0], pb[0], t),
                        _lerp(pa[1], pb[1], t),
                        _lerp(pa[2], pb[2], t),
                    )
                    row.append(pixel)
                frame.append(row)
            result.append(frame)

        return result

    def interpolate_with_flow(
        self,
        frame_a: List[List[Tuple[float, float, float]]],
        frame_b: List[List[Tuple[float, float, float]]],
        num_intermediate: int = 1,
    ) -> List[List[List[Tuple[float, float, float]]]]:
        """基于光流的帧插值"""
        # 计算双向光流
        flow_forward = self._temporal.compute_optical_flow(frame_a, frame_b)
        flow_backward = self._temporal.compute_optical_flow(frame_b, frame_a)

        if not flow_forward or not flow_backward:
            return self.interpolate_linear(frame_a, frame_b, num_intermediate)

        h = len(frame_a)
        w = len(frame_a[0]) if h > 0 else 0
        block_size = 8
        result: List[List[List[Tuple[float, float, float]]]] = []

        for i in range(1, num_intermediate + 1):
            t = i / (num_intermediate + 1)
            frame: List[List[Tuple[float, float, float]]] = []
            for y in range(h):
                row: List[Tuple[float, float, float]] = []
                for x in range(w):
                    # 前向扭曲
                    by = y // block_size
                    bx = x // block_size
                    fwd_dx, fwd_dy = 0.0, 0.0
                    bwd_dx, bwd_dy = 0.0, 0.0
                    if by < len(flow_forward) and bx < len(flow_forward[by]):
                        fwd_dx, fwd_dy = flow_forward[by][bx]
                    if by < len(flow_backward) and bx < len(flow_backward[by]):
                        bwd_dx, bwd_dy = flow_backward[by][bx]

                    # 双向扭曲混合
                    warped_a = frame_a[y][x]
                    warped_b = frame_b[y][x]

                    # 简化：使用线性插值加运动偏移
                    pixel = (
                        _lerp(warped_a[0], warped_b[0], t),
                        _lerp(warped_a[1], warped_b[1], t),
                        _lerp(warped_a[2], warped_b[2], t),
                    )
                    row.append(pixel)
                frame.append(row)
            result.append(frame)

        return result

    def interpolate_sequence(
        self,
        keyframes: List[List[List[Tuple[float, float, float]]]],
        frames_per_segment: int = 2,
    ) -> List[List[List[Tuple[float, float, float]]]]:
        """对关键帧序列进行插值"""
        if len(keyframes) <= 1:
            return list(keyframes)

        result: List[List[List[Tuple[float, float, float]]]] = [keyframes[0]]
        for i in range(len(keyframes) - 1):
            method = self._config.interpolation_method
            if method == "flow":
                intermediate = self.interpolate_with_flow(
                    keyframes[i], keyframes[i + 1], frames_per_segment
                )
            else:
                intermediate = self.interpolate_linear(
                    keyframes[i], keyframes[i + 1], frames_per_segment
                )
            result.extend(intermediate)
            result.append(keyframes[i + 1])

        return result


# ============================================================================
# VideoDecoder - 视频解码器
# ============================================================================

class VideoDecoder:
    """
    视频解码器：将潜在表示解码为像素帧。

    模拟VAE解码器的上采样过程。
    """

    def __init__(self, config: SVDConfig):
        self._config = config
        self._hidden_dim = config.decoder_hidden_dim
        self._num_layers = config.num_decoder_layers
        random.seed(123)
        self._layer_weights: List[List[List[float]]] = []
        self._layer_biases: List[List[float]] = []
        for _ in range(self._num_layers):
            w = [
                [random.gauss(0, 0.02) for _ in range(self._hidden_dim)]
                for _ in range(self._hidden_dim)
            ]
            b = [0.0] * self._hidden_dim
            self._layer_weights.append(w)
            self._layer_biases.append(b)
        random.seed()

    def decode_latent(
        self, latent: List[float], height: int, width: int
    ) -> List[List[Tuple[float, float, float]]]:
        """将潜在向量解码为像素帧"""
        # 通过解码器层
        hidden = list(latent)
        # 填充到隐藏维度
        if len(hidden) < self._hidden_dim:
            hidden = hidden + [0.0] * (self._hidden_dim - len(hidden))
        else:
            hidden = hidden[:self._hidden_dim]

        for i in range(self._num_layers):
            hidden = _linear_transform(
                hidden, self._layer_weights[i], self._layer_biases[i]
            )
            hidden = _layer_norm(hidden)
            hidden = [_gelu(v) for v in hidden]

        # 映射到像素
        pixels: List[List[Tuple[float, float, float]]] = []
        for y in range(height):
            row: List[Tuple[float, float, float]] = []
            for x in range(width):
                idx = (y * width + x) % len(hidden)
                base = hidden[idx]
                r = _clamp(base * 127.5 + 128.0, 0.0, 255.0)
                g = _clamp(hidden[(idx + 1) % len(hidden)] * 127.5 + 128.0, 0.0, 255.0)
                b = _clamp(hidden[(idx + 2) % len(hidden)] * 127.5 + 128.0, 0.0, 255.0)
                row.append((r, g, b))
            pixels.append(row)

        return pixels

    def decode_latents_to_frames(
        self,
        latents: List[List[float]],
        height: int,
        width: int,
    ) -> List[VideoFrame]:
        """将潜在帧序列解码为视频帧"""
        frames: List[VideoFrame] = []
        for i, latent in enumerate(latents):
            pixels = self.decode_latent(latent, height, width)
            frame = VideoFrame(
                pixels=pixels,
                width=width,
                height=height,
                timestamp=i / max(self._config.fps, 1),
            )
            frames.append(frame)
        return frames


# ============================================================================
# SVDPipeline - SVD主管线
# ============================================================================

class SVDPipeline:
    """
    Stable Video Diffusion管线：图像到视频生成的完整流程。

    流程:
    1. 编码输入图像
    2. 初始化噪声潜在
    3. 迭代去噪（扩散过程）
    4. 解码潜在为视频帧
    5. 时间一致性后处理
    6. 帧插值（可选）

    使用方法:
        config = SVDConfig(num_frames=14)
        pipeline = SVDPipeline(config)
        video = pipeline.generate(image_input)
    """

    def __init__(self, config: Optional[SVDConfig] = None):
        self._config = config or SVDConfig()
        self._scheduler = NoiseScheduler(self._config)
        self._conditioner = ConditioningModule(self._config)
        self._temporal = TemporalConsistency(self._config)
        self._motion_ctrl = MotionController(self._config)
        self._interpolator = FrameInterpolator(self._config)
        self._decoder = VideoDecoder(self._config)

    @property
    def config(self) -> SVDConfig:
        return self._config

    def generate(
        self,
        image: ImageInput,
        motion_type: str = "zoom_in",
        num_interpolation_frames: int = 0,
        seed: Optional[int] = None,
    ) -> VideoOutput:
        """
        从图像生成视频。

        Args:
            image: 输入图像
            motion_type: 运动类型
            num_interpolation_frames: 每段的插值帧数
            seed: 随机种子

        Returns:
            VideoOutput: 生成的视频
        """
        if seed is not None:
            random.seed(seed)

        # 步骤1: 编码条件
        image_emb = self._conditioner.encode_image(image)
        motion_emb = self._conditioner.encode_motion(self._config.motion_bucket_id)
        fps_emb = self._conditioner.encode_fps(self._config.fps_id)
        combined_cond = self._conditioner.combine_conditions(
            image_emb, motion_emb, fps_emb
        )

        # 步骤2: 初始化噪声潜在
        latent_size = (
            self._config.latent_channels
            * self._config.latent_height
            * self._config.latent_width
        )
        latents = [0.0] * latent_size
        noise = [_gaussian_noise(0, 1.0) for _ in range(latent_size)]

        # 步骤3: 扩散去噪过程
        timesteps = self._scheduler.get_timesteps(self._config.num_inference_steps)

        # 生成运动场
        motion_fields = self._motion_ctrl.generate_motion_field(
            self._config.latent_width,
            self._config.latent_height,
            self._config.num_frames,
            motion_type,
        )

        for step, t in enumerate(timesteps):
            # 添加噪声
            noisy_latents = self._scheduler.add_noise(latents, noise, t)

            # 模拟UNet去噪步骤
            denoised = self._denoise_step(
                noisy_latents, combined_cond, t, step, motion_fields
            )

            # 调度器步骤
            prev_t = self._scheduler.get_prev_timestep(t)
            alpha_t = self._scheduler.get_alpha_cumprod(t)
            alpha_prev = self._scheduler.get_alpha_cumprod(prev_t)

            # DDIM采样
            pred_original = self._predict_x0(noisy_latents, denoised, alpha_t)
            latents = self._ddim_step(
                pred_original, noisy_latents, alpha_t, alpha_prev
            )

        # 步骤4: 解码潜在为帧
        frame_latents = self._split_latent_to_frames(latents)
        output_h = self._config.latent_height * 8
        output_w = self._config.latent_width * 8
        frames = self._decoder.decode_latents_to_frames(
            frame_latents, output_h, output_w
        )

        # 步骤5: 时间一致性后处理
        pixel_data = [f.pixels for f in frames]
        smoothed = self._temporal.enforce_consistency(pixel_data)
        for i, frame in enumerate(frames):
            if i < len(smoothed):
                frame.pixels = smoothed[i]

        # 步骤6: 帧插值
        if num_interpolation_frames > 0:
            keyframe_data = [f.pixels for f in frames]
            interpolated = self._interpolator.interpolate_sequence(
                keyframe_data, num_interpolation_frames
            )
            all_frames: List[VideoFrame] = []
            for i, pixels in enumerate(interpolated):
                all_frames.append(VideoFrame(
                    pixels=pixels,
                    width=output_w,
                    height=output_h,
                    timestamp=i / max(self._config.fps, 1),
                ))
            frames = all_frames

        duration = len(frames) / max(self._config.fps, 1)

        return VideoOutput(
            frames=frames,
            fps=self._config.fps,
            width=output_w,
            height=output_h,
            duration=duration,
            metadata={
                "num_frames": len(frames),
                "motion_type": motion_type,
                "inference_steps": self._config.num_inference_steps,
                "generation_id": _generate_id(),
            },
        )

    def _denoise_step(
        self,
        noisy_latents: List[float],
        condition: List[float],
        timestep: int,
        step: int,
        motion_fields: List[List[Tuple[float, float]]],
    ) -> List[float]:
        """
        模拟UNet去噪步骤。

        在真实实现中，这里会通过UNet网络预测噪声。
        我们使用条件向量和噪声的加权组合来模拟。
        """
        noise_pred = list(noisy_latents)

        # 条件引导
        cond_scale = self._config.conditioning_scale
        for i in range(min(len(noise_pred), len(condition))):
            noise_pred[i] -= condition[i % len(condition)] * cond_scale * 0.1

        # 时间步嵌入
        t_emb = math.sin(timestep / self._config.num_train_timesteps * math.pi)
        for i in range(len(noise_pred)):
            noise_pred[i] *= (1.0 - t_emb * 0.3)

        # 运动注入
        frame_idx = step % len(motion_fields) if motion_fields else 0
        motion = motion_fields[frame_idx] if frame_idx < len(motion_fields) else []
        if motion:
            total_motion = sum(
                abs(dx) + abs(dy)
                for row in motion for dx, dy in row
            ) / max(len(motion) * len(motion[0]) if motion and motion[0] else 1, 1)
            motion_factor = total_motion * 0.01
            for i in range(len(noise_pred)):
                noise_pred[i] += _gaussian_noise(0, motion_factor)

        return noise_pred

    def _predict_x0(
        self,
        noisy: List[float],
        noise_pred: List[float],
        alpha_t: float,
    ) -> List[float]:
        """预测x_0"""
        sqrt_alpha = math.sqrt(alpha_t)
        sqrt_one_minus = math.sqrt(1.0 - alpha_t)
        result: List[float] = []
        for n, p in zip(noisy, noise_pred):
            val = (n - sqrt_one_minus * p) / max(sqrt_alpha, 1e-8)
            result.append(_clamp(val, -5.0, 5.0))
        return result

    def _ddim_step(
        self,
        pred_x0: List[float],
        noisy: List[float],
        alpha_t: float,
        alpha_prev: float,
    ) -> List[float]:
        """DDIM采样步骤"""
        sqrt_alpha_prev = math.sqrt(alpha_prev)
        sqrt_one_minus_prev = math.sqrt(1.0 - alpha_prev)

        # 计算预测噪声
        sqrt_alpha_t = math.sqrt(alpha_t)
        sqrt_one_minus_t = math.sqrt(1.0 - alpha_t)
        pred_noise: List[float] = []
        for x0, n in zip(pred_x0, noisy):
            noise_val = (n - sqrt_alpha_t * x0) / max(sqrt_one_minus_t, 1e-8)
            pred_noise.append(_clamp(noise_val, -5.0, 5.0))

        # DDIM更新
        dir_xt = sqrt_one_minus_prev * pred_noise
        result: List[float] = []
        for x0, d in zip(pred_x0, dir_xt):
            result.append(sqrt_alpha_prev * x0 + d)

        return result

    def _split_latent_to_frames(
        self, latent: List[float]
    ) -> List[List[float]]:
        """将潜在表示分割为帧"""
        channels = self._config.latent_channels
        h = self._config.latent_height
        w = self._config.latent_width
        frame_size = channels * h * w
        num_frames = self._config.num_frames

        frames: List[List[float]] = []
        for f in range(num_frames):
            start = f * frame_size
            end = start + frame_size
            if end <= len(latent):
                frames.append(latent[start:end])
            else:
                frame = latent[start:] + [0.0] * (end - len(latent))
                frames.append(frame)

        return frames

    def generate_with_custom_motion(
        self,
        image: ImageInput,
        motion_vectors: List[Tuple[float, float]],
        seed: Optional[int] = None,
    ) -> VideoOutput:
        """使用自定义运动向量生成视频"""
        if seed is not None:
            random.seed(seed)

        image_emb = self._conditioner.encode_image(image)
        motion_emb = self._conditioner.encode_motion(self._config.motion_bucket_id)
        fps_emb = self._conditioner.encode_fps(self._config.fps_id)
        combined_cond = self._conditioner.combine_conditions(
            image_emb, motion_emb, fps_emb
        )

        latent_size = (
            self._config.latent_channels
            * self._config.latent_height
            * self._config.latent_width
        )
        latents = [0.0] * latent_size
        noise = [_gaussian_noise(0, 1.0) for _ in range(latent_size)]

        timesteps = self._scheduler.get_timesteps(self._config.num_inference_steps)

        # 使用自定义运动
        custom_fields: List[List[Tuple[float, float]]] = []
        for mv in motion_vectors:
            field = [[mv for _ in range(self._config.latent_width)]
                     for _ in range(self._config.latent_height)]
            custom_fields.append(field)
        while len(custom_fields) < self._config.num_frames:
            custom_fields.append(custom_fields[-1] if custom_fields else [(0.0, 0.0)])

        for step, t in enumerate(timesteps):
            noisy_latents = self._scheduler.add_noise(latents, noise, t)
            denoised = self._denoise_step(
                noisy_latents, combined_cond, t, step, custom_fields
            )
            prev_t = self._scheduler.get_prev_timestep(t)
            alpha_t = self._scheduler.get_alpha_cumprod(t)
            alpha_prev = self._scheduler.get_alpha_cumprod(prev_t)
            pred_original = self._predict_x0(noisy_latents, denoised, alpha_t)
            latents = self._ddim_step(pred_original, noisy_latents, alpha_t, alpha_prev)

        frame_latents = self._split_latent_to_frames(latents)
        output_h = self._config.latent_height * 8
        output_w = self._config.latent_width * 8
        frames = self._decoder.decode_latents_to_frames(
            frame_latents, output_h, output_w
        )

        duration = len(frames) / max(self._config.fps, 1)
        return VideoOutput(
            frames=frames,
            fps=self._config.fps,
            width=output_w,
            height=output_h,
            duration=duration,
            metadata={"custom_motion": True, "generation_id": _generate_id()},
        )
