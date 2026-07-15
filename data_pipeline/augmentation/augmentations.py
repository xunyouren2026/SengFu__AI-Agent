"""
数据增强模块 - 综合数据增强框架
包含: 图像增强、文本增强、混合增强、AutoAugment、音频增强、增强管道、高级增强
纯 Python 实现，无外部依赖（仅使用标准库 math/random/copy）。
"""

import math
import random
import copy
from typing import Optional, Tuple, List, Union, Callable, Dict, Any
from abc import ABC, abstractmethod
from enum import Enum


# ============================================================================
# 辅助数据结构
# ============================================================================

class Image:
    """轻量级图像容器，data 形状 [H][W][C] 或 [H][W]（灰度）。"""

    def __init__(self, data: List[List[List[float]]], channels: int = 3):
        self.data = data
        self.channels = channels
        self.height = len(data)
        self.width = len(data[0]) if data else 0

    def get_pixel(self, h: int, w: int, c: int = 0) -> float:
        if 0 <= h < self.height and 0 <= w < self.width:
            if self.channels > 1:
                v = self.data[h][w]
                return v[c] if isinstance(v, list) else v
            return self.data[h][w]
        return 0.0

    def set_pixel(self, h: int, w: int, value: Union[float, List[float]]):
        if 0 <= h < self.height and 0 <= w < self.width:
            self.data[h][w] = value

    def copy(self) -> 'Image':
        return Image(copy.deepcopy(self.data), self.channels)


class AudioSignal:
    """轻量级音频信号容器，samples 为一维采样序列。"""

    def __init__(self, samples: List[float], sample_rate: int = 16000):
        self.samples = samples
        self.sample_rate = sample_rate
        self.duration = len(samples) / sample_rate

    def copy(self) -> 'AudioSignal':
        return AudioSignal(list(self.samples), self.sample_rate)


class Spectrogram:
    """轻量级频谱图容器，data 形状 [freq_bins][time_frames]。"""

    def __init__(self, data: List[List[float]], sample_rate: int = 16000, hop_length: int = 160):
        self.data = data
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_freq = len(data)
        self.n_frames = len(data[0]) if data else 0

    def copy(self) -> 'Spectrogram':
        return Spectrogram(copy.deepcopy(self.data), self.sample_rate, self.hop_length)


# ============================================================================
# 1. ImageAugmentation - 图像变换
# ============================================================================

class Augmentation(ABC):
    """数据增强基类。"""

    def __init__(self, p: float = 1.0):
        self.p = p

    @abstractmethod
    def apply(self, image: Image, target: Any = None) -> Tuple[Image, Any]:
        pass

    def __call__(self, image: Image, target: Any = None) -> Tuple[Image, Any]:
        if random.random() < self.p:
            return self.apply(image, target)
        return image, target


class RandomHorizontalFlip(Augmentation):
    """随机水平翻转。"""

    def __init__(self, p: float = 0.5):
        super().__init__(p)

    def apply(self, image: Image, target: Any = None) -> Tuple[Image, Any]:
        flipped = Image([row[::-1] for row in image.data], image.channels)
        if target is not None and isinstance(target, dict) and 'boxes' in target:
            new_boxes = []
            for box in target['boxes']:
                x1, y1, x2, y2 = box[:4]
                new_boxes.append([image.width - x2, y1, image.width - x1, y2] + list(box[4:]))
            target = {**target, 'boxes': new_boxes}
        return flipped, target


class RandomVerticalFlip(Augmentation):
    """随机垂直翻转。"""

    def __init__(self, p: float = 0.5):
        super().__init__(p)

    def apply(self, image: Image, target: Any = None) -> Tuple[Image, Any]:
        flipped = Image(image.data[::-1], image.channels)
        if target is not None and isinstance(target, dict) and 'boxes' in target:
            new_boxes = []
            for box in target['boxes']:
                x1, y1, x2, y2 = box[:4]
                new_boxes.append([x1, image.height - y2, x2, image.height - y1] + list(box[4:]))
            target = {**target, 'boxes': new_boxes}
        return flipped, target


class RandomRotation(Augmentation):
    """随机旋转（双线性插值）。"""

    def __init__(self, degrees: float = 10.0, p: float = 0.5):
        super().__init__(p)
        self.degrees = degrees

    def apply(self, image: Image, target: Any = None) -> Tuple[Image, Any]:
        angle = random.uniform(-self.degrees, self.degrees)
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        new_h = int(abs(image.height * cos_a) + abs(image.width * sin_a))
        new_w = int(abs(image.height * sin_a) + abs(image.width * cos_a))
        new_data = [[[0.0] * image.channels for _ in range(new_w)] for _ in range(new_h)]
        cx, cy = image.width / 2, image.height / 2
        ncx, ncy = new_w / 2, new_h / 2
        for y in range(new_h):
            for x in range(new_w):
                dx, dy = x - ncx, y - ncy
                sx = dx * cos_a + dy * sin_a + cx
                sy = -dx * sin_a + dy * cos_a + cy
                x0, y0 = int(sx), int(sy)
                x1, y1 = x0 + 1, y0 + 1
                if 0 <= x0 < image.width and 0 <= y0 < image.height:
                    wx, wy = sx - x0, sy - y0
                    for c in range(image.channels):
                        v00 = image.get_pixel(y0, x0, c)
                        v01 = image.get_pixel(y0, min(x1, image.width - 1), c)
                        v10 = image.get_pixel(min(y1, image.height - 1), x0, c)
                        v11 = image.get_pixel(min(y1, image.height - 1), min(x1, image.width - 1), c)
                        new_data[y][x][c] = (v00 * (1 - wx) * (1 - wy) + v01 * wx * (1 - wy)
                                             + v10 * (1 - wx) * wy + v11 * wx * wy)
        return Image(new_data, image.channels), target


class RandomCrop(Augmentation):
    """随机裁剪。"""

    def __init__(self, size: Tuple[int, int], padding: int = 0, pad_value: float = 0.0, p: float = 1.0):
        super().__init__(p)
        self.size = size
        self.padding = padding
        self.pad_value = pad_value

    def apply(self, image: Image, target: Any = None) -> Tuple[Image, Any]:
        src = image
        if self.padding > 0:
            pad = self.padding
            padded = []
            for i in range(image.height + 2 * pad):
                row = []
                for j in range(image.width + 2 * pad):
                    if pad <= i < image.height + pad and pad <= j < image.width + pad:
                        row.append(image.data[i - pad][j - pad])
                    else:
                        row.append([self.pad_value] * image.channels if image.channels > 1 else self.pad_value)
                padded.append(row)
            src = Image(padded, image.channels)
        h, w = self.size
        h, w = min(h, src.height), min(w, src.width)
        top = random.randint(0, src.height - h)
        left = random.randint(0, src.width - w)
        cropped = Image([row[left:left + w] for row in src.data[top:top + h]], image.channels)
        if target is not None and isinstance(target, dict) and 'boxes' in target:
            new_boxes = []
            for box in target['boxes']:
                bx1, by1, bx2, by2 = box[:4]
                nx1, ny1 = bx1 - left + self.padding, by1 - top + self.padding
                nx2, ny2 = bx2 - left + self.padding, by2 - top + self.padding
                if nx2 > 0 and ny2 > 0 and nx1 < w and ny1 < h:
                    new_boxes.append([max(0, nx1), max(0, ny1), min(w, nx2), min(h, ny2)] + list(box[4:]))
            target = {**target, 'boxes': new_boxes}
        return cropped, target


class CenterCrop(Augmentation):
    """中心裁剪。"""

    def __init__(self, size: Tuple[int, int]):
        super().__init__(1.0)
        self.size = size

    def apply(self, image: Image, target: Any = None) -> Tuple[Image, Any]:
        h, w = self.size
        top = max(0, (image.height - h) // 2)
        left = max(0, (image.width - w) // 2)
        h, w = min(h, image.height - top), min(w, image.width - left)
        cropped = Image([row[left:left + w] for row in image.data[top:top + h]], image.channels)
        return cropped, target


class Resize(Augmentation):
    """双线性插值缩放。"""

    def __init__(self, size: Tuple[int, int], p: float = 1.0):
        super().__init__(p)
        self.size = size

    def apply(self, image: Image, target: Any = None) -> Tuple[Image, Any]:
        new_h, new_w = self.size
        new_data = [[[0.0] * image.channels for _ in range(new_w)] for _ in range(new_h)]
        sy, sx = image.height / new_h, image.width / new_w
        for y in range(new_h):
            for x in range(new_w):
                src_y, src_x = y * sy, x * sx
                y0, x0 = int(src_y), int(src_x)
                y1, x1 = min(y0 + 1, image.height - 1), min(x0 + 1, image.width - 1)
                wy, wx = src_y - y0, src_x - x0
                for c in range(image.channels):
                    v00 = image.get_pixel(y0, x0, c)
                    v01 = image.get_pixel(y0, x1, c)
                    v10 = image.get_pixel(y1, x0, c)
                    v11 = image.get_pixel(y1, x1, c)
                    new_data[y][x][c] = (v00 * (1 - wx) * (1 - wy) + v01 * wx * (1 - wy)
                                         + v10 * (1 - wx) * wy + v11 * wx * wy)
        return Image(new_data, image.channels), target


class ColorJitter(Augmentation):
    """颜色抖动（亮度/对比度/饱和度/色相）。"""

    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0, p=1.0):
        super().__init__(p)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def apply(self, image: Image, target: Any = None) -> Tuple[Image, Any]:
        ops = []
        if self.brightness > 0:
            ops.append(self._brightness)
        if self.contrast > 0:
            ops.append(self._contrast)
        if self.saturation > 0:
            ops.append(self._saturation)
        if self.hue > 0:
            ops.append(self._hue)
        random.shuffle(ops)
        result = image.copy()
        for op in ops:
            result = op(result)
        return result, target

    def _brightness(self, img: Image) -> Image:
        f = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        return Image([[[min(255, max(0, img.get_pixel(h, w, c) * f))
                        for c in range(img.channels)] for w in range(img.width)] for h in range(img.height)],
                     img.channels)

    def _contrast(self, img: Image) -> Image:
        f = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        mean = sum(img.get_pixel(h, w, c) for h in range(img.height) for w in range(img.width)
                   for c in range(img.channels)) / (img.height * img.width * img.channels)
        return Image([[[min(255, max(0, mean + (img.get_pixel(h, w, c) - mean) * f))
                        for c in range(img.channels)] for w in range(img.width)] for h in range(img.height)],
                     img.channels)

    def _saturation(self, img: Image) -> Image:
        if img.channels < 3:
            return img
        f = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
        new_data = []
        for h in range(img.height):
            row = []
            for w in range(img.width):
                r, g, b = img.data[h][w][:3]
                gray = 0.299 * r + 0.587 * g + 0.114 * b
                px = [min(255, max(0, gray + (ch - gray) * f)) for ch in (r, g, b)]
                if img.channels > 3:
                    px.extend(img.data[h][w][3:])
                row.append(px)
            new_data.append(row)
        return Image(new_data, img.channels)

    def _hue(self, img: Image) -> Image:
        if img.channels < 3:
            return img
        factor = random.uniform(-self.hue, self.hue)
        new_data = []
        for h in range(img.height):
            row = []
            for w in range(img.width):
                r, g, b = img.data[h][w][:3]
                mx, mn = max(r, g, b), min(r, g, b)
                delta = mx - mn
                if delta == 0:
                    hue_val = 0
                elif mx == r:
                    hue_val = 60 * (((g - b) / delta) % 6)
                elif mx == g:
                    hue_val = 60 * ((b - r) / delta + 2)
                else:
                    hue_val = 60 * ((r - g) / delta + 4)
                sat = delta / mx if mx > 0 else 0
                val = mx
                hue_val = (hue_val + factor * 360) % 360
                c = val * sat
                x = c * (1 - abs((hue_val / 60) % 2 - 1))
                m = val - c
                if hue_val < 60:
                    rv, gv, bv = c, x, 0
                elif hue_val < 120:
                    rv, gv, bv = x, c, 0
                elif hue_val < 180:
                    rv, gv, bv = 0, c, x
                elif hue_val < 240:
                    rv, gv, bv = 0, x, c
                elif hue_val < 300:
                    rv, gv, bv = x, 0, c
                else:
                    rv, gv, bv = c, 0, x
                px = [rv + m, gv + m, bv + m]
                if img.channels > 3:
                    px.extend(img.data[h][w][3:])
                row.append(px)
            new_data.append(row)
        return Image(new_data, img.channels)


class RandomGrayscale(Augmentation):
    """随机转灰度。"""

    def __init__(self, p: float = 0.1):
        super().__init__(p)

    def apply(self, image: Image, target: Any = None) -> Tuple[Image, Any]:
        new_data = []
        for h in range(image.height):
            row = []
            for w in range(image.width):
                gray = 0.299 * image.get_pixel(h, w, 0) + 0.587 * image.get_pixel(h, w, 1) + 0.114 * image.get_pixel(h, w, 2)
                row.append([gray] * image.channels)
            new_data.append(row)
        return Image(new_data, image.channels), target


class GaussianBlur(Augmentation):
    """高斯模糊（可分离卷积）。"""

    def __init__(self, kernel_size: int = 3, sigma: Tuple[float, float] = (0.1, 2.0), p: float = 0.5):
        super().__init__(p)
        self.kernel_size = kernel_size
        self.sigma = sigma

    def apply(self, image: Image, target: Any = None) -> Tuple[Image, Any]:
        sigma = random.uniform(*self.sigma)
        k = self._make_kernel(self.kernel_size, sigma)
        pad = self.kernel_size // 2
        # 水平卷积
        temp = [[[0.0] * image.channels for _ in range(image.width)] for _ in range(image.height)]
        for h in range(image.height):
            for w in range(image.width):
                for c in range(image.channels):
                    val = 0.0
                    for kw in range(self.kernel_size):
                        nw = w + kw - pad
                        if 0 <= nw < image.width:
                            val += k[kw] * image.get_pixel(h, nw, c)
                    temp[h][w][c] = val
        # 垂直卷积
        new_data = [[[0.0] * image.channels for _ in range(image.width)] for _ in range(image.height)]
        for h in range(image.height):
            for w in range(image.width):
                for c in range(image.channels):
                    val = 0.0
                    for kh in range(self.kernel_size):
                        nh = h + kh - pad
                        if 0 <= nh < image.height:
                            val += k[kh] * temp[nh][w][c]
                    new_data[h][w][c] = val
        return Image(new_data, image.channels), target

    @staticmethod
    def _make_kernel(size: int, sigma: float) -> List[float]:
        center = size // 2
        kernel = [math.exp(-((i - center) ** 2) / (2 * sigma * sigma)) for i in range(size)]
        total = sum(kernel)
        return [k / total for k in kernel]


class GaussianNoise(Augmentation):
    """高斯噪声。"""

    def __init__(self, mean: float = 0.0, std: float = 25.0, p: float = 0.5):
        super().__init__(p)
        self.mean = mean
        self.std = std

    def apply(self, image: Image, target: Any = None) -> Tuple[Image, Any]:
        new_data = [[[min(255, max(0, image.get_pixel(h, w, c) + random.gauss(self.mean, self.std)))
                       for c in range(image.channels)] for w in range(image.width)] for h in range(image.height)]
        return Image(new_data, image.channels), target


class RandomErasing(Augmentation):
    """随机擦除。"""

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.0):
        super().__init__(p)
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def apply(self, image: Image, target: Any = None) -> Tuple[Image, Any]:
        area = image.height * image.width
        for _ in range(10):
            ta = random.uniform(*self.scale) * area
            ar = random.uniform(*self.ratio)
            eh = int(round(math.sqrt(ta * ar)))
            ew = int(round(math.sqrt(ta / ar)))
            if ew < image.width and eh < image.height:
                ex = random.randint(0, image.width - ew)
                ey = random.randint(0, image.height - eh)
                new_data = []
                for i in range(image.height):
                    row = []
                    for j in range(image.width):
                        if ey <= i < ey + eh and ex <= j < ex + ew:
                            row.append([self.value] * image.channels if image.channels > 1 else self.value)
                        else:
                            row.append(image.data[i][j])
                    new_data.append(row)
                return Image(new_data, image.channels), target
        return image, target


class Cutout(Augmentation):
    """Cutout - 在图像中裁剪正方形区域并填零。"""

    def __init__(self, n_holes: int = 1, length: int = 16, p: float = 0.5):
        super().__init__(p)
        self.n_holes = n_holes
        self.length = length

    def apply(self, image: Image, target: Any = None) -> Tuple[Image, Any]:
        new_data = copy.deepcopy(image.data)
        for _ in range(self.n_holes):
            y = random.randint(0, image.height - self.length)
            x = random.randint(0, image.width - self.length)
            for i in range(y, y + self.length):
                for j in range(x, x + self.length):
                    new_data[i][j] = [0.0] * image.channels if image.channels > 1 else 0.0
        return Image(new_data, image.channels), target


class Solarize(Augmentation):
    """Solarize - 超过阈值的像素取反。"""

    def __init__(self, threshold: float = 128.0, p: float = 0.5):
        super().__init__(p)
        self.threshold = threshold

    def apply(self, image: Image, target: Any = None) -> Tuple[Image, Any]:
        new_data = [[[255 - image.get_pixel(h, w, c) if image.get_pixel(h, w, c) > self.threshold
                       else image.get_pixel(h, w, c) for c in range(image.channels)]
                      for w in range(image.width)] for h in range(image.height)]
        return Image(new_data, image.channels), target


class Equalize(Augmentation):
    """直方图均衡化。"""

    def __init__(self, p: float = 0.5):
        super().__init__(p)

    def apply(self, image: Image, target: Any = None) -> Tuple[Image, Any]:
        new_data = []
        for c in range(image.channels):
            hist = [0] * 256
            for h in range(image.height):
                for w in range(image.width):
                    hist[min(255, max(0, int(image.get_pixel(h, w, c))))] += 1
            cdf = [0] * 256
            cdf[0] = hist[0]
            for i in range(1, 256):
                cdf[i] = cdf[i - 1] + hist[i]
            cdf_min = min(v for v in cdf if v > 0)
            total = image.height * image.width
            lut = [int((cdf[i] - cdf_min) / (total - cdf_min) * 255) if cdf[i] > cdf_min else 0 for i in range(256)]
            for h in range(image.height):
                if len(new_data) <= h:
                    new_data.append([])
                for w in range(image.width):
                    if len(new_data[h]) <= w:
                        new_data[h].append([0.0] * image.channels)
                    new_data[h][w][c] = lut[min(255, max(0, int(image.get_pixel(h, w, c))))]
        return Image(new_data, image.channels), target


class Posterize(Augmentation):
    """色调分离。"""

    def __init__(self, bits: int = 4, p: float = 0.5):
        super().__init__(p)
        self.bits = max(1, min(8, bits))

    def apply(self, image: Image, target: Any = None) -> Tuple[Image, Any]:
        levels = 2 ** self.bits
        factor = 256.0 / levels
        new_data = [[[int(image.get_pixel(h, w, c) / factor) * factor
                       for c in range(image.channels)] for w in range(image.width)] for h in range(image.height)]
        return Image(new_data, image.channels), target


class Invert(Augmentation):
    """颜色反转。"""

    def __init__(self, p: float = 0.5):
        super().__init__(p)

    def apply(self, image: Image, target: Any = None) -> Tuple[Image, Any]:
        new_data = [[[255 - image.get_pixel(h, w, c) for c in range(image.channels)]
                      for w in range(image.width)] for h in range(image.height)]
        return Image(new_data, image.channels), target


class Pad(Augmentation):
    """填充。"""

    def __init__(self, padding: Union[int, Tuple[int, int, int, int]], fill: float = 0.0, p: float = 1.0):
        super().__init__(p)
        if isinstance(padding, int):
            self.pad = (padding, padding, padding, padding)
        else:
            self.pad = padding
        self.fill = fill

    def apply(self, image: Image, target: Any = None) -> Tuple[Image, Any]:
        pt, pb, pl, pr = self.pad
        new_h = image.height + pt + pb
        new_w = image.width + pl + pr
        fill_px = [self.fill] * image.channels if image.channels > 1 else self.fill
        new_data = [[fill_px for _ in range(new_w)] for _ in range(new_h)]
        for h in range(image.height):
            for w in range(image.width):
                new_data[h + pt][w + pl] = image.data[h][w]
        return Image(new_data, image.channels), target


class RandomAffine(Augmentation):
    """随机仿射变换（平移+旋转+缩放+剪切）。"""

    def __init__(self, degrees: float = 0, translate: Tuple[float, float] = (0.0, 0.0),
                 scale: Tuple[float, float] = (1.0, 1.0), shear: float = 0, p: float = 0.5):
        super().__init__(p)
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def apply(self, image: Image, target: Any = None) -> Tuple[Image, Any]:
        angle = random.uniform(-self.degrees, self.degrees)
        rad = math.radians(angle)
        tx = random.uniform(-self.translate[0], self.translate[0]) * image.width
        ty = random.uniform(-self.translate[1], self.translate[1]) * image.height
        sc = random.uniform(*self.scale)
        sh = random.uniform(-self.shear, self.shear)
        sh_rad = math.radians(sh)
        cx, cy = image.width / 2, image.height / 2
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        cos_s, sin_s = math.cos(sh_rad), math.sin(sh_rad)
        new_data = [[[0.0] * image.channels for _ in range(image.width)] for _ in range(image.height)]
        for y in range(image.height):
            for x in range(image.width):
                dx, dy = x - cx, y - cy
                sx = (cos_a * dx - sin_a * dy) / sc - cos_s * dy / sc + cx + tx
                sy = (sin_a * dx + cos_a * dy) / sc - sin_s * dx / sc + cy + ty
                x0, y0 = int(sx), int(sy)
                if 0 <= x0 < image.width and 0 <= y0 < image.height:
                    for c in range(image.channels):
                        new_data[y][x][c] = image.get_pixel(y0, x0, c)
        return Image(new_data, image.channels), target


class RandomPerspective(Augmentation):
    """随机透视变换。"""

    def __init__(self, distortion_scale: float = 0.5, p: float = 0.5):
        super().__init__(p)
        self.distortion_scale = distortion_scale

    def apply(self, image: Image, target: Any = None) -> Tuple[Image, Any]:
        h, w = image.height, image.width
        ds = self.distortion_scale
        half_w, half_h = w / 2, h / 2
        src_pts = [(0, 0), (w, 0), (w, h), (0, h)]
        dst_pts = [
            (random.uniform(0, ds * half_w), random.uniform(0, ds * half_h)),
            (w - random.uniform(0, ds * half_w), random.uniform(0, ds * half_h)),
            (w - random.uniform(0, ds * half_w), h - random.uniform(0, ds * half_h)),
            (random.uniform(0, ds * half_w), h - random.uniform(0, ds * half_h)),
        ]
        coeffs = self._get_perspective_coeffs(src_pts, dst_pts)
        new_data = [[[0.0] * image.channels for _ in range(w)] for _ in range(h)]
        for y in range(h):
            for x in range(w):
                denom = coeffs[6] * x + coeffs[7] * y + 1
                if abs(denom) < 1e-8:
                    continue
                sx = (coeffs[0] * x + coeffs[1] * y + coeffs[2]) / denom
                sy = (coeffs[3] * x + coeffs[4] * y + coeffs[5]) / denom
                x0, y0 = int(sx), int(sy)
                if 0 <= x0 < w and 0 <= y0 < h:
                    for c in range(image.channels):
                        new_data[y][x][c] = image.get_pixel(y0, x0, c)
        return Image(new_data, image.channels), target

    @staticmethod
    def _get_perspective_coeffs(src, dst):
        """求解 3x3 透视变换矩阵（8 自由度），使用高斯消元法。"""
        A = []
        B = []
        for (xs, ys), (xd, yd) in zip(src, dst):
            A.append([xs, ys, 1, 0, 0, 0, -xd * xs, -xd * ys])
            A.append([0, 0, 0, xs, ys, 1, -yd * xs, -yd * ys])
            B.extend([xd, yd])
        # 高斯消元法求解
        n = len(A)
        for i in range(n):
            A[i].append(B[i])
        for col in range(n):
            max_row = max(range(col, n), key=lambda r: abs(A[r][col]))
            A[col], A[max_row] = A[max_row], A[col]
            pivot = A[col][col]
            if abs(pivot) < 1e-12:
                continue
            for j in range(col, n + 1):
                A[col][j] /= pivot
            for row in range(n):
                if row != col:
                    factor = A[row][col]
                    for j in range(col, n + 1):
                        A[row][j] -= factor * A[col][j]
        return [A[i][n] for i in range(n)]


class Normalize:
    """归一化。"""

    def __init__(self, mean: List[float], std: List[float]):
        self.mean = mean
        self.std = std

    def __call__(self, image: Image, target: Any = None) -> Tuple[Image, Any]:
        new_data = [[[(image.get_pixel(h, w, c) - self.mean[c]) / self.std[c]
                       for c in range(image.channels)] for w in range(image.width)] for h in range(image.height)]
        return Image(new_data, image.channels), target


class ToTensor:
    """转换为 [C, H, W] 张量格式。"""

    def __call__(self, image: Image, target: Any = None) -> Tuple[List, Any]:
        tensor = [[[image.get_pixel(h, w, c) for w in range(image.width)]
                    for h in range(image.height)] for c in range(image.channels)]
        return tensor, target


# ============================================================================
# 2. TextAugmentation - NLP 数据增强
# ============================================================================

class TextAugmentation(ABC):
    """文本增强基类。"""

    def __init__(self, p: float = 1.0):
        self.p = p

    @abstractmethod
    def apply(self, text: str) -> str:
        pass

    def __call__(self, text: str) -> str:
        if random.random() < self.p:
            return self.apply(text)
        return text


class SynonymReplacement(TextAugmentation):
    """同义词替换。使用内置词典进行替换。"""

    def __init__(self, n: int = 1, p: float = 0.5):
        super().__init__(p)
        self.n = n
        self._synonyms = {
            'good': ['great', 'fine', 'excellent', 'nice'],
            'bad': ['poor', 'terrible', 'awful', 'dreadful'],
            'big': ['large', 'huge', 'enormous', 'massive'],
            'small': ['tiny', 'little', 'miniature', 'minute'],
            'happy': ['joyful', 'glad', 'pleased', 'cheerful'],
            'sad': ['unhappy', 'sorrowful', 'gloomy', 'melancholy'],
            'fast': ['quick', 'rapid', 'swift', 'speedy'],
            'slow': ['sluggish', 'gradual', 'unhurried', 'leisurely'],
            'beautiful': ['gorgeous', 'stunning', 'lovely', 'attractive'],
            'ugly': ['hideous', 'unsightly', 'grotesque', 'repulsive'],
            'smart': ['intelligent', 'clever', 'brilliant', 'wise'],
            'strong': ['powerful', 'mighty', 'sturdy', 'robust'],
            'weak': ['feeble', 'frail', 'fragile', 'delicate'],
            'old': ['ancient', 'aged', 'elderly', 'vintage'],
            'new': ['fresh', 'novel', 'recent', 'modern'],
            'hot': ['warm', 'scorching', 'boiling', 'sizzling'],
            'cold': ['chilly', 'freezing', 'frigid', 'icy'],
            'walk': ['stroll', 'stride', 'march', 'saunter'],
            'run': ['sprint', 'dash', 'rush', 'bolt'],
            'say': ['state', 'declare', 'mention', 'remark'],
            'think': ['believe', 'consider', 'ponder', 'reflect'],
            'make': ['create', 'build', 'construct', 'produce'],
            'go': ['proceed', 'advance', 'move', 'travel'],
            'see': ['observe', 'notice', 'perceive', 'spot'],
            'know': ['understand', 'realize', 'recognize', 'comprehend'],
            'want': ['desire', 'wish', 'crave', 'yearn'],
            'use': ['utilize', 'employ', 'apply', 'operate'],
            'find': ['discover', 'locate', 'uncover', 'detect'],
            'give': ['provide', 'offer', 'present', 'donate'],
            'tell': ['inform', 'notify', 'advise', 'explain'],
            'work': ['function', 'operate', 'perform', 'labor'],
            'help': ['assist', 'aid', 'support', 'facilitate'],
        }

    def apply(self, text: str) -> str:
        words = text.split()
        candidates = [i for i, w in enumerate(words) if w.lower() in self._synonyms]
        random.shuffle(candidates)
        n_replace = min(self.n, len(candidates))
        for idx in candidates[:n_replace]:
            word = words[idx]
            syns = self._synonyms.get(word.lower(), self._synonyms.get(word, []))
            if syns:
                replacement = random.choice(syns)
                words[idx] = replacement if word.islower() else replacement.capitalize()
        return ' '.join(words)


class RandomInsertion(TextAugmentation):
    """随机插入同义词。"""

    def __init__(self, n: int = 1, p: float = 0.5):
        super().__init__(p)
        self.n = n
        self._extra_words = [
            'very', 'really', 'quite', 'rather', 'somewhat', 'extremely',
            'absolutely', 'definitely', 'certainly', 'clearly', 'obviously',
            'perhaps', 'maybe', 'indeed', 'actually', 'generally', 'usually',
        ]

    def apply(self, text: str) -> str:
        words = text.split()
        for _ in range(min(self.n, len(words))):
            pos = random.randint(0, len(words))
            word = random.choice(self._extra_words)
            words.insert(pos, word)
        return ' '.join(words)


class RandomSwap(TextAugmentation):
    """随机交换相邻词对。"""

    def __init__(self, n: int = 1, p: float = 0.5):
        super().__init__(p)
        self.n = n

    def apply(self, text: str) -> str:
        words = text.split()
        if len(words) < 2:
            return text
        for _ in range(min(self.n, len(words) - 1)):
            i = random.randint(0, len(words) - 2)
            words[i], words[i + 1] = words[i + 1], words[i]
        return ' '.join(words)


class RandomDeletion(TextAugmentation):
    """随机删除词。"""

    def __init__(self, p: float = 0.1):
        super().__init__(1.0)
        self.word_p = p

    def apply(self, text: str) -> str:
        words = text.split()
        if len(words) <= 1:
            return text
        remaining = [w for w in words if random.random() > self.word_p]
        return ' '.join(remaining) if remaining else random.choice(words)


class BackTranslation(TextAugmentation):
    """模拟回译增强：通过同义改写 + 句式变换模拟翻译-回译。"""

    def __init__(self, p: float = 0.5):
        super().__init__(p)
        self._patterns = [
            lambda s: s.replace(' is ', ' was ').replace(' are ', ' were '),
            lambda s: s.replace(' was ', ' is ').replace(' were ', ' are '),
            lambda s: 'It is true that ' + s if not s.lower().startswith('it ') else s,
            lambda s: s.replace(' will ', ' is going to ').replace(' can ', ' is able to '),
            lambda s: s.replace(' not ', ' n\'t ').replace(' n\'t ', ' not '),
            lambda s: 'Indeed, ' + s[0].lower() + s[1:] if s else s,
        ]

    def apply(self, text: str) -> str:
        pattern = random.choice(self._patterns)
        return pattern(text)


class ContextualWordSubstitution(TextAugmentation):
    """BERT 风格的上下文词替换模拟：用同义词 + 词性保持替换。"""

    def __init__(self, n: int = 1, p: float = 0.5):
        super().__init__(p)
        self.n = n
        self._context_syns = {
            'the': ['this', 'that', 'a', 'an'],
            'a': ['the', 'this', 'one', 'some'],
            'is': ['was', 'has been', 'remains', 'appears'],
            'are': ['were', 'have been', 'remain', 'appear'],
            'was': ['is', 'had been', 'seemed', 'looked'],
            'were': ['are', 'had been', 'seemed', 'looked'],
            'and': ['as well as', 'along with', 'plus', 'also'],
            'but': ['however', 'yet', 'although', 'nevertheless'],
            'or': ['alternatively', 'otherwise', 'or else'],
            'in': ['within', 'inside', 'during', 'throughout'],
            'on': ['upon', 'atop', 'during', 'regarding'],
            'at': ['by', 'near', 'around', 'close to'],
            'to': ['toward', 'unto', 'for', 'aiming at'],
            'for': ['intended for', 'meant for', 'serving', 'supporting'],
            'with': ['alongside', 'accompanied by', 'using', 'having'],
            'from': ['originating at', 'out of', 'since', 'derived from'],
            'by': ['via', 'through', 'using', 'means of'],
            'as': ['like', 'being', 'functioning as', 'serving as'],
            'that': ['which', 'who', 'whom', 'where'],
            'it': ['this', 'that', 'the thing', 'such'],
            'very': ['extremely', 'remarkably', 'exceptionally', 'incredibly'],
            'not': ['never', 'no longer', 'hardly', 'scarcely'],
            'can': ['could', 'might', 'may', 'is able to'],
            'will': ['shall', 'would', 'is going to', 'may'],
        }

    def apply(self, text: str) -> str:
        words = text.split()
        candidates = [i for i, w in enumerate(words) if w.lower() in self._context_syns]
        random.shuffle(candidates)
        for idx in candidates[:self.n]:
            word = words[idx]
            syns = self._context_syns.get(word.lower(), [])
            if syns:
                replacement = random.choice(syns)
                words[idx] = replacement if word.islower() else replacement.capitalize()
        return ' '.join(words)


# ============================================================================
# 3. MixAugmentation - 样本混合策略
# ============================================================================

class _DistributionMixin:
    """分布采样工具。"""

    @staticmethod
    def _beta_sample(alpha: float, beta: float) -> float:
        x = _DistributionMixin._gamma_sample(alpha)
        y = _DistributionMixin._gamma_sample(beta)
        return x / (x + y) if (x + y) > 0 else 0.5

    @staticmethod
    def _gamma_sample(shape: float) -> float:
        if shape >= 1:
            d = shape - 1.0 / 3.0
            c = 1.0 / math.sqrt(9.0 * d)
            while True:
                x = random.gauss(0, 1)
                v = (1 + c * x) ** 3
                if v > 0:
                    u = random.random()
                    if u < 1 - 0.0331 * (x ** 2) ** 2:
                        return d * v
                    if math.log(u) < 0.5 * x * x + d * (1 - v + math.log(v)):
                        return d * v
        else:
            return _DistributionMixin._gamma_sample(shape + 1) * (random.random() ** (1.0 / shape))

    @staticmethod
    def _dirichlet_sample(alpha: float, n: int) -> List[float]:
        samples = [_DistributionMixin._gamma_sample(alpha) for _ in range(n)]
        total = sum(samples)
        return [s / total for s in samples] if total > 0 else [1.0 / n] * n


class Mixup(_DistributionMixin):
    """Mixup: lambda * x_i + (1-lambda) * x_j。"""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(self, images: List[Image], targets: List[Any]) -> Tuple[List[Image], List[Any], List[float]]:
        bs = len(images)
        indices = list(range(bs))
        random.shuffle(indices)
        mixed_imgs, mixed_targets, lambdas = [], [], []
        for i in range(bs):
            lam = self._beta_sample(self.alpha, self.alpha) if self.alpha > 0 else 1.0
            j = indices[i]
            mixed_data = [[[lam * images[i].get_pixel(h, w, c) + (1 - lam) * images[j].get_pixel(h, w, c)
                            for c in range(images[i].channels)] for w in range(images[i].width)]
                           for h in range(images[i].height)]
            mixed_imgs.append(Image(mixed_data, images[i].channels))
            mixed_targets.append({'labels': [targets[i], targets[j]], 'weights': [lam, 1 - lam]})
            lambdas.append(lam)
        return mixed_imgs, mixed_targets, lambdas


class CutMix(_DistributionMixin):
    """CutMix: 用一张图的矩形区域替换另一张图。"""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, images: List[Image], targets: List[Any]) -> Tuple[List[Image], List[Any], List[float]]:
        bs = len(images)
        indices = list(range(bs))
        random.shuffle(indices)
        mixed_imgs, mixed_targets, lambdas = [], [], []
        for i in range(bs):
            j = indices[i]
            lam = self._beta_sample(self.alpha, self.alpha) if self.alpha > 0 else 1.0
            img = images[i]
            cw = int(img.width * math.sqrt(1 - lam))
            ch = int(img.height * math.sqrt(1 - lam))
            cx, cy = random.randint(0, img.width), random.randint(0, img.height)
            x1, y1 = max(0, cx - cw // 2), max(0, cy - ch // 2)
            x2, y2 = min(img.width, cx + cw // 2), min(img.height, cy + ch // 2)
            actual_lam = 1 - (x2 - x1) * (y2 - y1) / (img.width * img.height)
            mixed_data = []
            for h in range(img.height):
                row = []
                for w in range(img.width):
                    if y1 <= h < y2 and x1 <= w < x2:
                        row.append(images[j].data[h][w])
                    else:
                        row.append(img.data[h][w])
                mixed_data.append(row)
            mixed_imgs.append(Image(mixed_data, img.channels))
            mixed_targets.append({'labels': [targets[i], targets[j]], 'weights': [actual_lam, 1 - actual_lam]})
            lambdas.append(actual_lam)
        return mixed_imgs, mixed_targets, lambdas


class FMix(_DistributionMixin):
    """FMix: 傅里叶域混合 - 使用低频二值掩码混合两张图。"""

    def __init__(self, alpha: float = 1.0, decay_power: float = 3.0, size: Tuple[int, int] = (32, 32)):
        self.alpha = alpha
        self.decay_power = decay_power
        self.size = size

    def __call__(self, images: List[Image], targets: List[Any]) -> Tuple[List[Image], List[Any], List[float]]:
        bs = len(images)
        indices = list(range(bs))
        random.shuffle(indices)
        mask = self._generate_mask()
        mixed_imgs, mixed_targets, lambdas = [], [], []
        lam = sum(sum(row) for row in mask) / (self.size[0] * self.size[1])
        for i in range(bs):
            j = indices[i]
            img = images[i]
            mixed_data = []
            for h in range(img.height):
                row = []
                for w in range(img.width):
                    mh = int(h / img.height * self.size[0])
                    mw = int(w / img.width * self.size[1])
                    mh, mw = min(mh, self.size[0] - 1), min(mw, self.size[1] - 1)
                    m = mask[mh][mw]
                    row.append([m * images[i].get_pixel(h, w, c) + (1 - m) * images[j].get_pixel(h, w, c)
                                for c in range(img.channels)])
                mixed_data.append(row)
            mixed_imgs.append(Image(mixed_data, img.channels))
            mixed_targets.append({'labels': [targets[i], targets[j]], 'weights': [lam, 1 - lam]})
            lambdas.append(lam)
        return mixed_imgs, mixed_targets, lambdas

    def _generate_mask(self) -> List[List[float]]:
        """生成低频二值掩码（简化 DCT）。"""
        h, w = self.size
        # 随机频域系数
        coeffs = [[[random.gauss(0, 1) for _ in range(w)] for _ in range(h)] for _ in range(2)]
        # 简化逆 DCT
        mask = [[0.0] * w for _ in range(h)]
        for y in range(h):
            for x in range(w):
                val = 0.0
                for u in range(h):
                    for v in range(w):
                        freq = math.sqrt(u * u + v * v) + 1e-8
                        magnitude = 1.0 / (freq ** self.decay_power)
                        val += (coeffs[0][u][v] * math.cos(2 * math.pi * (u * y / h + v * x / w))
                                + coeffs[1][u][v] * math.sin(2 * math.pi * (u * y / h + v * x / w))) * magnitude
                mask[y][x] = val
        # 归一化并二值化
        flat = [mask[y][x] for y in range(h) for x in range(w)]
        mn, mx = min(flat), max(flat)
        rng = mx - mn if mx > mn else 1.0
        threshold = random.random()
        return [[1.0 if (mask[y][x] - mn) / rng > threshold else 0.0 for x in range(w)] for y in range(h)]


class SnapMix(_DistributionMixin):
    """SnapMix: 注意力引导的混合。使用类激活图 (CAM) 模拟。"""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, images: List[Image], targets: List[Any]) -> Tuple[List[Image], List[Any], List[float]]:
        bs = len(images)
        indices = list(range(bs))
        random.shuffle(indices)
        mixed_imgs, mixed_targets, lambdas = [], [], []
        for i in range(bs):
            j = indices[i]
            lam = self._beta_sample(self.alpha, self.alpha) if self.alpha > 0 else 1.0
            img = images[i]
            # 模拟 CAM 注意力图：中心区域权重高
            cam = self._simulate_cam(img.height, img.width)
            # 按阈值二值化
            flat = sorted([cam[y][x] for y in range(img.height) for x in range(img.width)], reverse=True)
            k = int(lam * len(flat))
            threshold = flat[min(k, len(flat) - 1)] if flat else 0.5
            mixed_data = []
            count = 0
            for h in range(img.height):
                row = []
                for w in range(img.width):
                    if cam[h][w] >= threshold:
                        row.append(images[i].data[h][w])
                        count += 1
                    else:
                        row.append(images[j].data[h][w])
                mixed_data.append(row)
            actual_lam = count / (img.height * img.width)
            mixed_imgs.append(Image(mixed_data, img.channels))
            mixed_targets.append({'labels': [targets[i], targets[j]], 'weights': [actual_lam, 1 - actual_lam]})
            lambdas.append(actual_lam)
        return mixed_imgs, mixed_targets, lambdas

    @staticmethod
    def _simulate_cam(h: int, w: int) -> List[List[float]]:
        """模拟类激活图（高斯分布）。"""
        cy, cx = h / 2, w / 2
        sigma = min(h, w) / 4
        cam = [[math.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma * sigma)) for x in range(w)] for y in range(h)]
        # 添加随机偏移
        oy, ox = random.randint(-h // 4, h // 4), random.randint(-w // 4, w // 4)
        cam2 = [[math.exp(-((y - cy - oy) ** 2 + (x - cx - ox) ** 2) / (2 * sigma * sigma)) for x in range(w)] for y in range(h)]
        mx = max(max(row) for row in cam2) or 1.0
        cam = [[max(cam[y][x], cam2[y][x] / mx) for x in range(w)] for y in range(h)]
        return cam


class GridMix(_DistributionMixin):
    """GridMix: 基于网格的混合。将图像划分为网格，随机选择网格区域替换。"""

    def __init__(self, alpha: float = 1.0, n_grid: int = 2):
        self.alpha = alpha
        self.n_grid = n_grid

    def __call__(self, images: List[Image], targets: List[Any]) -> Tuple[List[Image], List[Any], List[float]]:
        bs = len(images)
        indices = list(range(bs))
        random.shuffle(indices)
        mixed_imgs, mixed_targets, lambdas = [], [], []
        for i in range(bs):
            j = indices[i]
            img = images[i]
            gh, gw = img.height // self.n_grid, img.width // self.n_grid
            # 随机选择网格
            selected = set()
            for _ in range(random.randint(1, self.n_grid * self.n_grid)):
                selected.add((random.randint(0, self.n_grid - 1), random.randint(0, self.n_grid - 1)))
            mixed_data = []
            count = 0
            total = img.height * img.width
            for h in range(img.height):
                row = []
                for w in range(img.width):
                    gi, gj = h // gh, w // gw
                    if (gi, gj) in selected:
                        row.append(images[j].data[h][w])
                        count += 1
                    else:
                        row.append(img.data[h][w])
                mixed_data.append(row)
            lam = 1 - count / total
            mixed_imgs.append(Image(mixed_data, img.channels))
            mixed_targets.append({'labels': [targets[i], targets[j]], 'weights': [lam, 1 - lam]})
            lambdas.append(lam)
        return mixed_imgs, mixed_targets, lambdas


class SaliencyMix(_DistributionMixin):
    """SaliencyMix: 显著图引导的混合。"""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, images: List[Image], targets: List[Any]) -> Tuple[List[Image], List[Any], List[float]]:
        bs = len(images)
        indices = list(range(bs))
        random.shuffle(indices)
        mixed_imgs, mixed_targets, lambdas = [], [], []
        for i in range(bs):
            j = indices[i]
            lam = self._beta_sample(self.alpha, self.alpha) if self.alpha > 0 else 1.0
            img_i, img_j = images[i], images[j]
            saliency = self._compute_saliency(img_i)
            flat = sorted([saliency[y][x] for y in range(img_i.height) for x in range(img_i.width)], reverse=True)
            k = int(lam * len(flat))
            threshold = flat[min(k, len(flat) - 1)] if flat else 0.5
            mixed_data = []
            count = 0
            for h in range(img_i.height):
                row = []
                for w in range(img_i.width):
                    if saliency[h][w] >= threshold:
                        row.append(img_i.data[h][w])
                        count += 1
                    else:
                        row.append(img_j.data[h][w])
                mixed_data.append(row)
            actual_lam = count / (img_i.height * img_i.width)
            mixed_imgs.append(Image(mixed_data, img_i.channels))
            mixed_targets.append({'labels': [targets[i], targets[j]], 'weights': [actual_lam, 1 - actual_lam]})
            lambdas.append(actual_lam)
        return mixed_imgs, mixed_targets, lambdas

    @staticmethod
    def _compute_saliency(img: Image) -> List[List[float]]:
        """基于中心-周围对比度的简化显著图。"""
        h, w = img.height, img.width
        # 计算灰度
        gray = [[0.299 * img.get_pixel(y, x, 0) + 0.587 * img.get_pixel(y, x, 1) + 0.114 * img.get_pixel(y, x, 2)
                 for x in range(w)] for y in range(h)]
        # 多尺度中心-周围差
        saliency = [[0.0] * w for _ in range(h)]
        for radius in [min(h, w) // 8, min(h, w) // 4, min(h, w) // 2]:
            for y in range(h):
                for x in range(w):
                    center_sum, center_cnt = 0.0, 0
                    surround_sum, surround_cnt = 0.0, 0
                    r2 = radius * radius
                    for dy in range(-radius, radius + 1, max(1, radius // 2)):
                        for dx in range(-radius, radius + 1, max(1, radius // 2)):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                d2 = dy * dy + dx * dx
                                if d2 <= r2 * 0.25:
                                    center_sum += gray[ny][nx]
                                    center_cnt += 1
                                else:
                                    surround_sum += gray[ny][nx]
                                    surround_cnt += 1
                    if center_cnt > 0 and surround_cnt > 0:
                        saliency[y][x] += abs(center_sum / center_cnt - surround_sum / surround_cnt)
        # 归一化
        flat = [saliency[y][x] for y in range(h) for x in range(w)]
        mx = max(flat) if flat else 1.0
        if mx > 0:
            for y in range(h):
                for x in range(w):
                    saliency[y][x] /= mx
        return saliency


# ============================================================================
# 4. AutoAugment - 可学习增强策略
# ============================================================================

class AutoAugment:
    """AutoAugment: 使用预定义策略序列。"""

    def __init__(self, policy: str = 'imagenet'):
        self.policies = self._get_policies(policy)

    @staticmethod
    def _get_policies(policy: str) -> List[List[Tuple[str, float, float]]]:
        if policy == 'imagenet':
            return [
                [('Posterize', 0.4, 8), ('Rotate', 0.6, 9)],
                [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 0)],
                [('Equalize', 0.8, 0), ('Equalize', 0.6, 0)],
                [('Posterize', 0.6, 7), ('Posterize', 0.6, 6)],
                [('Equalize', 0.4, 0), ('Solarize', 0.2, 4)],
                [('Equalize', 0.4, 0), ('Rotate', 0.8, 8)],
                [('Solarize', 0.6, 3), ('Equalize', 0.6, 0)],
                [('Posterize', 0.8, 5), ('Equalize', 1.0, 0)],
                [('Rotate', 0.2, 3), ('Solarize', 0.6, 8)],
                [('AutoContrast', 0.6, 0), ('Equalize', 0.8, 0)],
                [('Equalize', 0.6, 0), ('Posterize', 0.4, 6)],
                [('Posterize', 0.6, 7), ('Solarize', 0.6, 6)],
                [('Posterize', 0.6, 6), ('AutoContrast', 0.8, 0)],
                [('Solarize', 0.7, 3), ('Posterize', 0.6, 7)],
                [('Invert', 0.4, 0), ('Equalize', 0.6, 0)],
                [('Color', 0.7, 7), ('Rotate', 0.3, 2)],
                [('Invert', 0.9, 0), ('Color', 0.7, 9)],
                [('Equalize', 0.6, 0), ('Color', 0.4, 5)],
                [('Color', 0.3, 9), ('Equalize', 0.6, 0)],
                [('Color', 0.7, 7), ('Solarize', 0.1, 2)],
            ]
        elif policy == 'cifar10':
            return [
                [('Invert', 0.1, 0), ('Contrast', 0.2, 6)],
                [('Rotate', 0.7, 2), ('TranslateX', 0.3, 9)],
                [('ShearY', 0.1, 6), ('Solarize', 0.6, 8)],
                [('ShearX', 0.6, 5), ('Invert', 0.2, 0)],
                [('Color', 0.3, 7), ('Cutout', 0.1, 8)],
                [('Posterize', 0.4, 3), ('Brightness', 0.4, 1)],
                [('Contrast', 0.9, 4), ('Cutout', 0.1, 10)],
                [('Equalize', 0.1, 0), ('Cutout', 0.2, 8)],
                [('ShearY', 0.1, 2), ('Solarize', 0.1, 6)],
                [('TranslateY', 0.7, 6), ('AutoContrast', 0.1, 0)],
            ]
        return [[('Identity', 1.0, 0)]]

    def __call__(self, image: Image, target: Any = None) -> Tuple[Image, Any]:
        policy = random.choice(self.policies)
        for op_name, prob, mag in policy:
            if random.random() < prob:
                image = self._apply_op(image, op_name, mag)
        return image, target

    def _apply_op(self, image: Image, name: str, mag: int) -> Image:
        if name == 'Identity':
            return image
        elif name == 'AutoContrast':
            mn = min(image.get_pixel(h, w, c) for h in range(image.height) for w in range(image.width) for c in range(image.channels))
            mx = max(image.get_pixel(h, w, c) for h in range(image.height) for w in range(image.width) for c in range(image.channels))
            if mx == mn:
                return image
            return Image([[[255 * (image.get_pixel(h, w, c) - mn) / (mx - mn) for c in range(image.channels)]
                           for w in range(image.width)] for h in range(image.height)], image.channels)
        elif name == 'Equalize':
            aug = Equalize(p=1.0)
            r, _ = aug.apply(image)
            return r
        elif name == 'Invert':
            aug = Invert(p=1.0)
            r, _ = aug.apply(image)
            return r
        elif name == 'Rotate':
            aug = RandomRotation(degrees=(mag / 10.0) * 30, p=1.0)
            r, _ = aug.apply(image)
            return r
        elif name == 'Posterize':
            aug = Posterize(bits=max(1, 8 - mag), p=1.0)
            r, _ = aug.apply(image)
            return r
        elif name == 'Solarize':
            aug = Solarize(threshold=256 * (1 - mag / 10.0), p=1.0)
            r, _ = aug.apply(image)
            return r
        elif name == 'Color':
            aug = ColorJitter(saturation=(mag / 10.0) * 0.9, p=1.0)
            r, _ = aug.apply(image)
            return r
        elif name == 'Contrast':
            aug = ColorJitter(contrast=(mag / 10.0) * 0.9, p=1.0)
            r, _ = aug.apply(image)
            return r
        elif name == 'Brightness':
            aug = ColorJitter(brightness=(mag / 10.0) * 0.9, p=1.0)
            r, _ = aug.apply(image)
            return r
        elif name == 'Sharpness':
            return image  # 简化
        elif name == 'Cutout':
            size = int((mag / 10.0) * min(image.height, image.width) / 2)
            aug = Cutout(n_holes=1, length=max(1, size), p=1.0)
            r, _ = aug.apply(image)
            return r
        elif name == 'ShearX':
            aug = RandomAffine(shear=(mag / 10.0) * 0.3, p=1.0)
            r, _ = aug.apply(image)
            return r
        elif name == 'ShearY':
            aug = RandomAffine(shear=(mag / 10.0) * 0.3, p=1.0)
            r, _ = aug.apply(image)
            return r
        elif name == 'TranslateX':
            offset = int((mag / 10.0) * image.width * 0.45)
            new_data = [[[0.0] * image.channels for _ in range(image.width)] for _ in range(image.height)]
            for h in range(image.height):
                for w in range(image.width):
                    nw = w + offset
                    if 0 <= nw < image.width:
                        new_data[h][nw] = image.data[h][w]
            return Image(new_data, image.channels)
        elif name == 'TranslateY':
            offset = int((mag / 10.0) * image.height * 0.45)
            new_data = [[[0.0] * image.channels for _ in range(image.width)] for _ in range(image.height)]
            for h in range(image.height):
                for w in range(image.width):
                    nh = h + offset
                    if 0 <= nh < image.height:
                        new_data[nh][w] = image.data[h][w]
            return Image(new_data, image.channels)
        return image


class RandAugment:
    """RandAugment: 随机选择 N 个操作，统一幅度 M。"""

    def __init__(self, n: int = 2, m: int = 10):
        self.n = n
        self.m = m
        self._ops = [
            'AutoContrast', 'Equalize', 'Invert', 'Rotate', 'Posterize',
            'Solarize', 'Color', 'Contrast', 'Brightness', 'Sharpness',
            'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Cutout'
        ]
        self._aa = AutoAugment()

    def __call__(self, image: Image, target: Any = None) -> Tuple[Image, Any]:
        ops = random.choices(self._ops, k=self.n)
        for op in ops:
            image = self._aa._apply_op(image, op, self.m)
        return image, target


class TrivialAugment:
    """TrivialAugment: 每次只随机选一个操作，幅度随机。"""

    def __init__(self):
        self._ops = [
            'AutoContrast', 'Equalize', 'Invert', 'Rotate', 'Posterize',
            'Solarize', 'Color', 'Contrast', 'Brightness', 'Sharpness',
            'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Cutout'
        ]
        self._aa = AutoAugment()

    def __call__(self, image: Image, target: Any = None) -> Tuple[Image, Any]:
        op = random.choice(self._ops)
        mag = random.randint(0, 10)
        image = self._aa._apply_op(image, op, mag)
        return image, target


class AugMix(_DistributionMixin):
    """AugMix: 混合多个增强路径的结果。"""

    def __init__(self, severity: int = 3, width: int = 3, depth: int = -1, alpha: float = 1.0):
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha
        self._ops = ['AutoContrast', 'Equalize', 'Rotate', 'Posterize',
                     'Solarize', 'Color', 'Contrast', 'Brightness', 'Sharpness']
        self._aa = AutoAugment()

    def __call__(self, image: Image, target: Any = None) -> Tuple[Image, Any]:
        weights = self._dirichlet_sample(self.alpha, self.width + 1)
        mixed = None
        for i in range(self.width):
            aug_img = image.copy()
            d = self.depth if self.depth > 0 else random.randint(1, 3)
            for _ in range(d):
                op = random.choice(self._ops)
                aug_img = self._aa._apply_op(aug_img, op, self.severity)
            scaled = Image([[[aug_img.get_pixel(h, w, c) * weights[i] for c in range(aug_img.channels)]
                             for w in range(aug_img.width)] for h in range(aug_img.height)], aug_img.channels)
            if mixed is None:
                mixed = scaled
            else:
                mixed = Image([[[mixed.get_pixel(h, w, c) + scaled.get_pixel(h, w, c)
                                  for c in range(mixed.channels)] for w in range(mixed.width)]
                               for h in range(mixed.height)], mixed.channels)
        orig_scaled = Image([[[image.get_pixel(h, w, c) * weights[-1] for c in range(image.channels)]
                               for w in range(image.width)] for h in range(image.height)], image.channels)
        result = Image([[[mixed.get_pixel(h, w, c) + orig_scaled.get_pixel(h, w, c)
                          for c in range(mixed.channels)] for w in range(mixed.width)]
                        for h in range(mixed.height)], mixed.channels)
        return result, target


# ============================================================================
# 5. AudioAugmentation - 音频变换
# ============================================================================

class AudioAugmentation(ABC):
    """音频增强基类。"""

    def __init__(self, p: float = 1.0):
        self.p = p

    @abstractmethod
    def apply(self, audio: AudioSignal) -> AudioSignal:
        pass

    def __call__(self, audio: AudioSignal) -> AudioSignal:
        if random.random() < self.p:
            return self.apply(audio)
        return audio


class TimeStretch(AudioAugmentation):
    """时间拉伸（重采样插值）。"""

    def __init__(self, rate_range: Tuple[float, float] = (0.8, 1.2), p: float = 0.5):
        super().__init__(p)
        self.rate_range = rate_range

    def apply(self, audio: AudioSignal) -> AudioSignal:
        rate = random.uniform(*self.rate_range)
        new_len = int(len(audio.samples) / rate)
        new_samples = []
        for i in range(new_len):
            src_pos = i * rate
            idx = int(src_pos)
            frac = src_pos - idx
            if idx + 1 < len(audio.samples):
                new_samples.append(audio.samples[idx] * (1 - frac) + audio.samples[idx + 1] * frac)
            elif idx < len(audio.samples):
                new_samples.append(audio.samples[idx])
        return AudioSignal(new_samples, audio.sample_rate)


class PitchShift(AudioAugmentation):
    """音高偏移（通过重采样模拟）。"""

    def __init__(self, n_steps_range: Tuple[int, int] = (-2, 2), p: float = 0.5):
        super().__init__(p)
        self.n_steps_range = n_steps_range

    def apply(self, audio: AudioSignal) -> AudioSignal:
        steps = random.randint(*self.n_steps_range)
        rate = 2.0 ** (steps / 12.0)
        new_len = int(len(audio.samples) * rate)
        new_samples = []
        for i in range(new_len):
            src_pos = i / rate
            idx = int(src_pos)
            frac = src_pos - idx
            if idx + 1 < len(audio.samples):
                new_samples.append(audio.samples[idx] * (1 - frac) + audio.samples[idx + 1] * frac)
            elif idx < len(audio.samples):
                new_samples.append(audio.samples[idx])
        return AudioSignal(new_samples, audio.sample_rate)


class AddNoise(AudioAugmentation):
    """添加高斯噪声。"""

    def __init__(self, snr_db: float = 20.0, p: float = 0.5):
        super().__init__(p)
        self.snr_db = snr_db

    def apply(self, audio: AudioSignal) -> AudioSignal:
        signal_power = sum(s * s for s in audio.samples) / len(audio.samples) if audio.samples else 1e-10
        snr_linear = 10 ** (self.snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise_std = math.sqrt(noise_power)
        new_samples = [s + random.gauss(0, noise_std) for s in audio.samples]
        return AudioSignal(new_samples, audio.sample_rate)


class AddBackgroundNoise(AudioAugmentation):
    """添加背景噪声（有色噪声模拟）。"""

    def __init__(self, noise_type: str = 'pink', snr_db: float = 15.0, p: float = 0.5):
        super().__init__(p)
        self.noise_type = noise_type
        self.snr_db = snr_db

    def apply(self, audio: AudioSignal) -> AudioSignal:
        n = len(audio.samples)
        if self.noise_type == 'white':
            noise = [random.gauss(0, 1) for _ in range(n)]
        elif self.noise_type == 'pink':
            noise = [0.0] * n
            b0, b1, b2, b3, b4, b5, b6 = 0.99886, 0.99332, 0.96900, 0.86650, 0.55000, 0.7616, 0.0
            for i in range(n):
                white = random.gauss(0, 1)
                b0 = 0.99886 * b0 + white * 0.0555179
                b1 = 0.99332 * b1 + white * 0.0750759
                b2 = 0.96900 * b2 + white * 0.1538520
                b3 = 0.86650 * b3 + white * 0.3104856
                b4 = 0.55000 * b4 + white * 0.5329522
                b5 = -0.7616 * b5 - white * 0.0168980
                noise[i] = b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362
        else:  # brown
            noise = [0.0] * n
            last = 0.0
            for i in range(n):
                white = random.gauss(0, 1)
                last = (last + (0.02 * white)) / 1.02
                noise[i] = last * 3.5
        # 调整 SNR
        signal_power = sum(s * s for s in audio.samples) / n if audio.samples else 1e-10
        noise_power = sum(s * s for s in noise) / n if noise else 1e-10
        if noise_power > 0:
            scale = math.sqrt(signal_power / (noise_power * (10 ** (self.snr_db / 10.0))))
            noise = [s * scale for s in noise]
        new_samples = [audio.samples[i] + noise[i] for i in range(n)]
        return AudioSignal(new_samples, audio.sample_rate)


class TimeMasking(AudioAugmentation):
    """SpecAugment 时间遮蔽（在波形上应用）。"""

    def __init__(self, max_mask_pct: float = 0.1, p: float = 0.5):
        super().__init__(p)
        self.max_mask_pct = max_mask_pct

    def apply(self, audio: AudioSignal) -> AudioSignal:
        n = len(audio.samples)
        mask_len = random.randint(1, int(n * self.max_mask_pct))
        start = random.randint(0, n - mask_len)
        new_samples = list(audio.samples)
        for i in range(start, start + mask_len):
            new_samples[i] = 0.0
        return AudioSignal(new_samples, audio.sample_rate)


class FrequencyMasking:
    """SpecAugment 频率遮蔽（在频谱图上应用）。"""

    def __init__(self, max_mask_bins: int = 10, p: float = 0.5):
        self.p = p
        self.max_mask_bins = max_mask_bins

    def __call__(self, spec: Spectrogram) -> Spectrogram:
        if random.random() >= self.p:
            return spec
        mask_bins = random.randint(1, min(self.max_mask_bins, spec.n_freq))
        start_bin = random.randint(0, spec.n_freq - mask_bins)
        new_data = copy.deepcopy(spec.data)
        for f in range(start_bin, start_bin + mask_bins):
            for t in range(spec.n_frames):
                new_data[f][t] = 0.0
        return Spectrogram(new_data, spec.sample_rate, spec.hop_length)


class Gain(AudioAugmentation):
    """增益调整。"""

    def __init__(self, min_gain_db: float = -10.0, max_gain_db: float = 10.0, p: float = 0.5):
        super().__init__(p)
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

    def apply(self, audio: AudioSignal) -> AudioSignal:
        gain_db = random.uniform(self.min_gain_db, self.max_gain_db)
        gain = 10 ** (gain_db / 20.0)
        return AudioSignal([s * gain for s in audio.samples], audio.sample_rate)


class LowPassFilter(AudioAugmentation):
    """低通滤波器（一阶 IIR）。"""

    def __init__(self, cutoff_hz: float = 4000.0, p: float = 0.5):
        super().__init__(p)
        self.cutoff_hz = cutoff_hz

    def apply(self, audio: AudioSignal) -> AudioSignal:
        rc = 1.0 / (2.0 * math.pi * self.cutoff_hz)
        dt = 1.0 / audio.sample_rate
        alpha = dt / (rc + dt)
        filtered = [0.0] * len(audio.samples)
        filtered[0] = audio.samples[0] * alpha
        for i in range(1, len(audio.samples)):
            filtered[i] = filtered[i - 1] + alpha * (audio.samples[i] - filtered[i - 1])
        return AudioSignal(filtered, audio.sample_rate)


class HighPassFilter(AudioAugmentation):
    """高通滤波器（一阶 IIR）。"""

    def __init__(self, cutoff_hz: float = 200.0, p: float = 0.5):
        super().__init__(p)
        self.cutoff_hz = cutoff_hz

    def apply(self, audio: AudioSignal) -> AudioSignal:
        rc = 1.0 / (2.0 * math.pi * self.cutoff_hz)
        dt = 1.0 / audio.sample_rate
        alpha = rc / (rc + dt)
        filtered = [0.0] * len(audio.samples)
        for i in range(1, len(audio.samples)):
            filtered[i] = alpha * (filtered[i - 1] + audio.samples[i] - audio.samples[i - 1])
        return AudioSignal(filtered, audio.sample_rate)


# ============================================================================
# 6. AugmentationPipeline - 可组合增强管道
# ============================================================================

class Sequential:
    """顺序执行所有变换。"""

    def __init__(self, transforms: List[Any]):
        self.transforms = transforms

    def __call__(self, data: Any, target: Any = None) -> Tuple[Any, Any]:
        for t in self.transforms:
            if hasattr(t, '__call__'):
                result = t(data, target)
                if isinstance(result, tuple) and len(result) == 2:
                    data, target = result
                else:
                    data = result
        return data, target


class RandomChoice:
    """随机选择一个变换执行。"""

    def __init__(self, transforms: List[Any], p: List[float] = None):
        self.transforms = transforms
        self.p = p

    def __call__(self, data: Any, target: Any = None) -> Tuple[Any, Any]:
        t = random.choices(self.transforms, weights=self.p, k=1)[0]
        result = t(data, target)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, target


class RandomApply:
    """以给定概率应用变换。"""

    def __init__(self, transform: Any, p: float = 0.5):
        self.transform = transform
        self.p = p

    def __call__(self, data: Any, target: Any = None) -> Tuple[Any, Any]:
        if random.random() < self.p:
            result = self.transform(data, target)
            if isinstance(result, tuple) and len(result) == 2:
                return result
            return result, target
        return data, target


class OneOf:
    """从多个变换中选择一个执行（等概率或加权）。"""

    def __init__(self, transforms: List[Any], weights: List[float] = None):
        self.transforms = transforms
        self.weights = weights

    def __call__(self, data: Any, target: Any = None) -> Tuple[Any, Any]:
        t = random.choices(self.transforms, weights=self.weights, k=1)[0]
        result = t(data, target)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, target


class SomeOf:
    """从多个变换中随机选择 k 个执行。"""

    def __init__(self, transforms: List[Any], k: int = 2, weights: List[float] = None):
        self.transforms = transforms
        self.k = min(k, len(transforms))
        self.weights = weights

    def __call__(self, data: Any, target: Any = None) -> Tuple[Any, Any]:
        chosen = random.choices(self.transforms, weights=self.weights, k=self.k)
        # 去重
        seen = set()
        unique = []
        for t in chosen:
            tid = id(t)
            if tid not in seen:
                seen.add(tid)
                unique.append(t)
        for t in unique:
            result = t(data, target)
            if isinstance(result, tuple) and len(result) == 2:
                data, target = result
            else:
                data = result
        return data, target


# ============================================================================
# 7. AdvancedAugmentation - 研究级增强
# ============================================================================

class MosaicAugmentation:
    """Mosaic 增强 (YOLOv4 风格): 将 4 张图像拼成一张。"""

    def __init__(self, output_size: Tuple[int, int] = (640, 640)):
        self.output_size = output_size

    def __call__(self, images: List[Image], targets: List[Any] = None) -> Tuple[Image, List[Any]]:
        assert len(images) == 4, "Mosaic 需要 4 张图像"
        oh, ow = self.output_size
        hc, wc = oh // 2, ow // 2
        # 随机偏移中心
        cx = random.randint(wc // 2, wc + wc // 2)
        cy = random.randint(hc // 2, hc + hc // 2)
        mosaic_data = [[[0.0] * images[0].channels for _ in range(ow)] for _ in range(oh)]
        new_targets = [{} if t is None else t for t in (targets or [None] * 4)]
        placements = [
            (0, 0, cx, cy),           # 左上
            (cx, 0, ow, cy),          # 右上
            (0, cy, cx, oh),          # 左下
            (cx, cy, ow, oh),         # 右下
        ]
        for idx, (x1, y1, x2, y2) in enumerate(placements):
            img = images[idx]
            pw, ph = x2 - x1, y2 - y1
            # 缩放图像以适应
            scale = min(pw / img.width, ph / img.height)
            sw, sh = int(img.width * scale), int(img.height * scale)
            # 随机偏移
            ox = random.randint(0, max(0, pw - sw))
            oy = random.randint(0, max(0, ph - sh))
            # 放置
            for sy in range(sh):
                for sx in range(sw):
                    dy, dx = y1 + oy + sy, x1 + ox + sx
                    if 0 <= dy < oh and 0 <= dx < ow:
                        src_y = int(sy / scale)
                        src_x = int(sx / scale)
                        if 0 <= src_y < img.height and 0 <= src_x < img.width:
                            mosaic_data[dy][dx] = img.data[src_y][src_x]
            # 调整边界框
            if new_targets[idx] and isinstance(new_targets[idx], dict) and 'boxes' in new_targets[idx]:
                adjusted = []
                for box in new_targets[idx]['boxes']:
                    bx1, by1, bx2, by2 = box[:4]
                    nbx1 = bx1 * scale + x1 + ox
                    nby1 = by1 * scale + y1 + oy
                    nbx2 = bx2 * scale + x1 + ox
                    nby2 = by2 * scale + y1 + oy
                    # 裁剪到有效区域
                    nbx1 = max(x1, min(nbx1, x2))
                    nby1 = max(y1, min(nby1, y2))
                    nbx2 = max(x1, min(nbx2, x2))
                    nby2 = max(y1, min(nby2, y2))
                    if nbx2 > nbx1 and nby2 > nby1:
                        adjusted.append([nbx1, nby1, nbx2, nby2] + list(box[4:]))
                new_targets[idx] = {**new_targets[idx], 'boxes': adjusted}
        # 合并目标
        merged_boxes = []
        for t in new_targets:
            if isinstance(t, dict) and 'boxes' in t:
                merged_boxes.extend(t['boxes'])
        merged_target = {'boxes': merged_boxes} if merged_boxes else {}
        return Image(mosaic_data, images[0].channels), merged_target


class CopyPaste:
    """CopyPaste: 实例级复制粘贴增强。"""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, source_image: Image, source_target: Dict,
                 target_image: Image, target_target: Dict) -> Tuple[Image, Dict]:
        if random.random() > self.p:
            return target_image, target_target
        if 'boxes' not in source_target or not source_target['boxes']:
            return target_image, target_target
        # 随机选择一个实例
        box = random.choice(source_target['boxes'])
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(source_image.width, x2), min(source_image.height, y2)
        if x2 <= x1 or y2 <= y1:
            return target_image, target_target
        # 随机放置位置
        bh, bw = y2 - y1, x2 - x1
        px = random.randint(0, max(0, target_image.width - bw))
        py = random.randint(0, max(0, target_image.height - bh))
        # 复制粘贴
        new_data = copy.deepcopy(target_image.data)
        for sy in range(bh):
            for sx in range(bw):
                dy, dx = py + sy, px + sx
                if 0 <= dy < target_image.height and 0 <= dx < target_image.width:
                    new_data[dy][dx] = source_image.data[y1 + sy][x1 + sx]
        # 添加新边界框
        new_boxes = list(target_target.get('boxes', []))
        new_boxes.append([px, py, px + bw, py + bh] + list(box[4:]))
        new_target = {**target_target, 'boxes': new_boxes}
        return Image(new_data, target_image.channels), new_target


class StyleTransferAugmentation:
    """风格迁移增强模拟：通过颜色空间变换模拟风格效果。"""

    def __init__(self, style: str = 'random', p: float = 0.5):
        super().__init__(p) if hasattr(super(), '__init__') else None
        self.p = p
        self.style = style
        self._styles = {
            'warm': {'r_boost': 1.15, 'g_boost': 1.0, 'b_boost': 0.85, 'contrast': 1.1},
            'cool': {'r_boost': 0.85, 'g_boost': 1.0, 'b_boost': 1.15, 'contrast': 1.1},
            'vintage': {'r_boost': 1.1, 'g_boost': 0.95, 'b_boost': 0.8, 'contrast': 0.9},
            'dramatic': {'r_boost': 1.0, 'g_boost': 1.0, 'b_boost': 1.0, 'contrast': 1.5},
            'faded': {'r_boost': 1.05, 'g_boost': 1.05, 'b_boost': 1.0, 'contrast': 0.7},
            'noir': {'r_boost': 0.3, 'g_boost': 0.3, 'b_boost': 0.3, 'contrast': 1.4},
        }

    def __call__(self, image: Image, target: Any = None) -> Tuple[Image, Any]:
        if random.random() > self.p:
            return image, target
        if image.channels < 3:
            return image, target
        if self.style == 'random':
            params = random.choice(list(self._styles.values()))
        else:
            params = self._styles.get(self.style, self._styles['warm'])
        # 计算均值
        mean_r = sum(image.get_pixel(h, w, 0) for h in range(image.height) for w in range(image.width)) / (image.height * image.width)
        mean_g = sum(image.get_pixel(h, w, 1) for h in range(image.height) for w in range(image.width)) / (image.height * image.width)
        mean_b = sum(image.get_pixel(h, w, 2) for h in range(image.height) for w in range(image.width)) / (image.height * image.width)
        contrast = params['contrast']
        new_data = []
        for h in range(image.height):
            row = []
            for w in range(image.width):
                r = image.get_pixel(h, w, 0)
                g = image.get_pixel(h, w, 1)
                b = image.get_pixel(h, w, 2)
                r = min(255, max(0, mean_r + (r - mean_r) * contrast) * params['r_boost'])
                g = min(255, max(0, mean_g + (g - mean_g) * contrast) * params['g_boost'])
                b = min(255, max(0, mean_b + (b - mean_b) * contrast) * params['b_boost'])
                px = [r, g, b]
                if image.channels > 3:
                    px.extend(image.data[h][w][3:])
                row.append(px)
            new_data.append(row)
        return Image(new_data, image.channels), target


class TestTimeAugmentation:
    """测试时增强 (TTA): 多裁剪 + 翻转，返回多个增强版本用于集成预测。"""

    def __init__(self, crop_size: Tuple[int, int] = (224, 224), scales: List[float] = None,
                 flips: bool = True):
        self.crop_size = crop_size
        self.scales = scales or [0.8, 0.9, 1.0]
        self.flips = flips

    def __call__(self, image: Image) -> List[Image]:
        augmented = []
        for scale in self.scales:
            new_h = int(image.height * scale)
            new_w = int(image.width * scale)
            # 先缩放
            resize = Resize((new_h, new_w))
            resized, _ = resize.apply(image)
            # 中心裁剪
            ch, cw = self.crop_size
            top = max(0, (new_h - ch) // 2)
            left = max(0, (new_w - cw) // 2)
            cropped_data = [row[left:left + cw] for row in resized.data[top:top + ch]]
            cropped = Image(cropped_data, image.channels)
            augmented.append(cropped)
            # 水平翻转
            if self.flips:
                flipped = Image([row[::-1] for row in cropped_data], image.channels)
                augmented.append(flipped)
        return augmented

    @staticmethod
    def merge_predictions(predictions: List[Any], method: str = 'mean') -> Any:
        """合并多个预测结果。"""
        if not predictions:
            return None
        if isinstance(predictions[0], (int, float)):
            if method == 'mean':
                return sum(predictions) / len(predictions)
            elif method == 'max':
                return max(predictions)
            elif method == 'min':
                return min(predictions)
        elif isinstance(predictions[0], list):
            n = len(predictions[0])
            if method == 'mean':
                return [sum(p[i] for p in predictions) / len(predictions) for i in range(n)]
            elif method == 'max':
                return [max(p[i] for p in predictions) for i in range(n)]
        elif isinstance(predictions[0], dict):
            result = {}
            for key in predictions[0]:
                vals = [p[key] for p in predictions]
                if isinstance(vals[0], (int, float)):
                    result[key] = sum(vals) / len(vals) if method == 'mean' else vals
                else:
                    result[key] = vals
            return result
        return predictions[0]


# ============================================================================
# 工厂函数与便捷接口
# ============================================================================

def get_augmentation(name: str, **kwargs) -> Any:
    """根据名称获取增强实例。"""
    registry = {
        # 图像增强
        'random_horizontal_flip': RandomHorizontalFlip,
        'random_vertical_flip': RandomVerticalFlip,
        'random_rotation': RandomRotation,
        'random_crop': RandomCrop,
        'center_crop': CenterCrop,
        'resize': Resize,
        'color_jitter': ColorJitter,
        'random_grayscale': RandomGrayscale,
        'gaussian_blur': GaussianBlur,
        'gaussian_noise': GaussianNoise,
        'random_erasing': RandomErasing,
        'cutout': Cutout,
        'solarize': Solarize,
        'equalize': Equalize,
        'posterize': Posterize,
        'invert': Invert,
        'pad': Pad,
        'random_affine': RandomAffine,
        'random_perspective': RandomPerspective,
        'normalize': Normalize,
        'to_tensor': ToTensor,
        # 混合增强
        'mixup': Mixup,
        'cutmix': CutMix,
        'fmix': FMix,
        'snapmix': SnapMix,
        'gridmix': GridMix,
        'saliencymix': SaliencyMix,
        # AutoAugment
        'auto_augment': AutoAugment,
        'rand_augment': RandAugment,
        'trivial_augment': TrivialAugment,
        'augmix': AugMix,
        # 音频增强
        'time_stretch': TimeStretch,
        'pitch_shift': PitchShift,
        'add_noise': AddNoise,
        'add_background_noise': AddBackgroundNoise,
        'time_masking': TimeMasking,
        'frequency_masking': FrequencyMasking,
        'gain': Gain,
        'low_pass_filter': LowPassFilter,
        'high_pass_filter': HighPassFilter,
        # 文本增强
        'synonym_replacement': SynonymReplacement,
        'random_insertion': RandomInsertion,
        'random_swap': RandomSwap,
        'random_deletion': RandomDeletion,
        'back_translation': BackTranslation,
        'contextual_word_substitution': ContextualWordSubstitution,
        # 高级增强
        'mosaic': MosaicAugmentation,
        'copy_paste': CopyPaste,
        'style_transfer': StyleTransferAugmentation,
        'test_time_augmentation': TestTimeAugmentation,
    }
    key = name.lower().replace('-', '_')
    if key not in registry:
        raise ValueError(f"未知增强: {name}. 可用: {list(registry.keys())}")
    return registry[key](**kwargs)


def create_training_pipeline(
    image_size: Tuple[int, int] = (224, 224),
    mean: List[float] = None,
    std: List[float] = None,
    auto_augment: str = None,
    rand_augment: bool = False,
    hflip: bool = True,
    color_jitter: bool = True,
) -> Sequential:
    """创建标准训练增强管道。"""
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]
    transforms = [RandomResizedCrop(image_size) if hasattr(Resize, '__call__') else
                  Sequential([Resize(image_size), RandomCrop(image_size)])]
    if hflip:
        transforms.append(RandomHorizontalFlip())
    if auto_augment:
        transforms.append(AutoAugment(policy=auto_augment))
    elif rand_augment:
        transforms.append(RandAugment())
    if color_jitter:
        transforms.append(ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8))
    transforms.append(Normalize(mean, std))
    transforms.append(ToTensor())
    return Sequential(transforms)


def create_validation_pipeline(
    image_size: Tuple[int, int] = (224, 224),
    mean: List[float] = None,
    std: List[float] = None,
) -> Sequential:
    """创建验证增强管道。"""
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]
    return Sequential([Resize(image_size), CenterCrop(image_size), Normalize(mean, std), ToTensor()])


class RandomResizedCrop(Augmentation):
    """随机大小裁剪并缩放到指定尺寸。"""

    def __init__(self, size: Tuple[int, int], scale=(0.08, 1.0), ratio=(0.75, 1.333), p=1.0):
        super().__init__(p)
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def apply(self, image: Image, target: Any = None) -> Tuple[Image, Any]:
        area = image.height * image.width
        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            ar = random.uniform(*self.ratio)
            w = int(round(math.sqrt(target_area * ar)))
            h = int(round(math.sqrt(target_area / ar)))
            if 0 < w <= image.width and 0 < h <= image.height:
                top = random.randint(0, image.height - h)
                left = random.randint(0, image.width - w)
                cropped = Image([row[left:left + w] for row in image.data[top:top + h]], image.channels)
                resize = Resize(self.size)
                return resize.apply(cropped, target)
        # 回退到中心裁剪
        cc = CenterCrop(self.size)
        return cc.apply(image, target)
