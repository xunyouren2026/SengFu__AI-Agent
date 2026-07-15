"""
ImageDataset - 测试图像数据集生成器

模块路径: testing/database/image_dataset.py

提供测试用图像数据集的生成、管理和查询功能。
"""

import os
import sys
import json
import time
import random
import hashlib
import struct
import io
import math
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Generator
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

import pytest
import numpy as np


class ImageFormat(Enum):
    """图像格式枚举"""
    JPEG = "jpeg"
    PNG = "png"
    BMP = "bmp"
    WEBP = "webp"
    RAW = "raw"


class ImageCategory(Enum):
    """图像类别枚举"""
    LANDSCAPE = "landscape"
    PORTRAIT = "portrait"
    DOCUMENT = "document"
    ICON = "icon"
    CHART = "chart"
    PHOTOGRAPH = "photograph"
    SCREENSHOT = "screenshot"
    QR_CODE = "qr_code"


@dataclass
class ImageMetadata:
    """图像元数据"""
    image_id: str
    filename: str
    format: str = "png"
    width: int = 224
    height: int = 224
    channels: int = 3
    file_size: int = 0
    category: str = "photograph"
    label: str = ""
    description: str = ""
    checksum_md5: str = ""
    created_at: float = 0.0
    tags: List[str] = field(default_factory=list)
    annotations: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "image_id": self.image_id,
            "filename": self.filename,
            "format": self.format,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "file_size": self.file_size,
            "category": self.category,
            "label": self.label,
            "description": self.description,
            "checksum_md5": self.checksum_md5,
            "created_at": self.created_at,
            "tags": self.tags,
            "annotations": self.annotations,
        }


@dataclass
class BoundingBox:
    """边界框标注"""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: str = ""
    confidence: float = 1.0


class ImageDataset:
    """测试图像数据集生成器

    提供测试用图像数据集的生成和管理功能:
        - 生成各种尺寸和格式的合成图像数据
        - 支持RGB、灰度、多通道图像
        - 生成带标注的图像（边界框、分类标签）
        - 支持数据增强变换（翻转、旋转、裁剪等）
        - 提供数据集划分（训练/验证/测试）
        - 生成图像元数据和统计信息
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialized = False
        self._images: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, ImageMetadata] = {}
        self._annotations: Dict[str, List[BoundingBox]] = {}
        self._categories: Dict[str, List[str]] = defaultdict(list)
        self._default_size: Tuple[int, int] = tuple(self.config.get("default_size", (224, 224)))
        self._default_channels: int = self.config.get("default_channels", 3)
        self._seed: int = self.config.get("seed", 42)

    def initialize(self) -> None:
        """初始化数据集生成器"""
        random.seed(self._seed)
        np.random.seed(self._seed)
        self._initialized = True

    def generate_solid_image(self, width: int = 224, height: int = 224,
                              channels: int = 3,
                              color: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """生成纯色图像

        Args:
            width: 图像宽度
            height: 图像高度
            channels: 通道数
            color: RGB颜色值，默认随机

        Returns:
            numpy数组形式的图像数据
        """
        if color is None:
            color = tuple(random.randint(0, 255) for _ in range(channels))
        img = np.full((height, width, channels), color, dtype=np.uint8)
        return img

    def generate_gradient_image(self, width: int = 224, height: int = 224,
                                 channels: int = 3,
                                 direction: str = "horizontal") -> np.ndarray:
        """生成渐变图像

        Args:
            width: 图像宽度
            height: 图像高度
            channels: 通道数
            direction: 渐变方向 (horizontal/vertical/diagonal)

        Returns:
            numpy数组形式的图像数据
        """
        img = np.zeros((height, width, channels), dtype=np.uint8)
        for c in range(channels):
            start_val = random.randint(0, 128)
            end_val = random.randint(128, 255)
            if direction == "horizontal":
                gradient = np.linspace(start_val, end_val, width, dtype=np.uint8)
                img[:, :, c] = gradient[np.newaxis, :]
            elif direction == "vertical":
                gradient = np.linspace(start_val, end_val, height, dtype=np.uint8)
                img[:, :, c] = gradient[:, np.newaxis]
            else:
                for y in range(height):
                    for x in range(width):
                        t = (x + y) / (width + height)
                        img[y, x, c] = int(start_val + (end_val - start_val) * t)
        return img

    def generate_noise_image(self, width: int = 224, height: int = 224,
                              channels: int = 3,
                              noise_type: str = "gaussian") -> np.ndarray:
        """生成噪声图像

        Args:
            width: 图像宽度
            height: 图像高度
            channels: 通道数
            noise_type: 噪声类型 (gaussian/uniform/salt_pepper)

        Returns:
            numpy数组形式的图像数据
        """
        if noise_type == "gaussian":
            img = np.random.normal(128, 30, (height, width, channels)).clip(0, 255).astype(np.uint8)
        elif noise_type == "uniform":
            img = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        elif noise_type == "salt_pepper":
            img = np.random.randint(100, 156, (height, width, channels), dtype=np.uint8)
            num_salt = int(0.01 * width * height)
            salt_coords = (random.randint(0, height - 1, num_salt),
                          random.randint(0, width - 1, num_salt))
            img[salt_coords] = 255
        else:
            img = np.zeros((height, width, channels), dtype=np.uint8)
        return img

    def generate_pattern_image(self, width: int = 224, height: int = 224,
                                channels: int = 3,
                                pattern: str = "checkerboard") -> np.ndarray:
        """生成图案图像

        Args:
            width: 图像宽度
            height: 图像高度
            channels: 通道数
            pattern: 图案类型 (checkerboard/stripes/grid/dots)

        Returns:
            numpy数组形式的图像数据
        """
        img = np.zeros((height, width, channels), dtype=np.uint8)
        color_a = np.array([random.randint(0, 128) for _ in range(channels)], dtype=np.uint8)
        color_b = np.array([random.randint(128, 255) for _ in range(channels)], dtype=np.uint8)

        if pattern == "checkerboard":
            block_size = max(8, min(width, height) // 8)
            for y in range(0, height, block_size):
                for x in range(0, width, block_size):
                    color = color_a if ((x // block_size) + (y // block_size)) % 2 == 0 else color_b
                    y_end = min(y + block_size, height)
                    x_end = min(x + block_size, width)
                    img[y:y_end, x:x_end] = color
        elif pattern == "stripes":
            stripe_width = max(4, width // 16)
            for x in range(0, width, stripe_width * 2):
                x_end = min(x + stripe_width, width)
                img[:, x:x_end] = color_a
                x_end2 = min(x + stripe_width * 2, width)
                img[:, x + stripe_width:x_end2] = color_b
        elif pattern == "grid":
            line_width = 2
            img[:, ::line_width * 10] = color_a
            img[::line_width * 10, :] = color_a
        elif pattern == "dots":
            for _ in range(50):
                cx, cy = random.randint(0, width - 1), random.randint(0, height - 1)
                radius = random.randint(5, 20)
                y_start, y_end = max(0, cy - radius), min(height, cy + radius)
                x_start, x_end = max(0, cx - radius), min(width, cx + radius)
                img[y_start:y_end, x_start:x_end] = color_a
        return img

    def generate_batch(self, count: int, width: int = 224, height: int = 224,
                        channels: int = 3, category: str = "photograph",
                        image_type: str = "mixed") -> List[ImageMetadata]:
        """批量生成图像数据

        Args:
            count: 生成数量
            width: 图像宽度
            height: 图像高度
            channels: 通道数
            category: 图像类别
            image_type: 图像类型 (solid/gradient/noise/pattern/mixed)

        Returns:
            生成的图像元数据列表
        """
        if not self._initialized:
            self.initialize()
        generators = {
            "solid": self.generate_solid_image,
            "gradient": self.generate_gradient_image,
            "noise": self.generate_noise_image,
            "pattern": self.generate_pattern_image,
        }
        batch_metadata = []
        for i in range(count):
            image_id = f"img_{hashlib.md5(f'{category}_{i}_{time.time()}'.encode()).hexdigest()[:12]}"
            if image_type == "mixed":
                gen = random.choice(list(generators.values()))
            else:
                gen = generators.get(image_type, self.generate_solid_image)

            img = gen(width, height, channels)
            img_bytes = img.tobytes()
            filename = f"{image_id}.png"

            metadata = ImageMetadata(
                image_id=image_id,
                filename=filename,
                format="png",
                width=width,
                height=height,
                channels=channels,
                file_size=len(img_bytes),
                category=category,
                label=f"{category}_{i}",
                description=f"Generated test image {i} of category {category}",
                checksum_md5=hashlib.md5(img_bytes).hexdigest(),
                created_at=time.time(),
                tags=[category, image_type, f"{width}x{height}"],
            )
            self._images[image_id] = img
            self._metadata[image_id] = metadata
            self._categories[category].append(image_id)
            batch_metadata.append(metadata)
        return batch_metadata

    def add_annotation(self, image_id: str, bbox: BoundingBox) -> None:
        """为图像添加边界框标注

        Args:
            image_id: 图像ID
            bbox: 边界框
        """
        if image_id not in self._annotations:
            self._annotations[image_id] = []
        self._annotations[image_id].append(bbox)

    def get_annotations(self, image_id: str) -> List[BoundingBox]:
        """获取图像的标注

        Args:
            image_id: 图像ID

        Returns:
            边界框列表
        """
        return self._annotations.get(image_id, [])

    def apply_augmentation(self, image: np.ndarray,
                            augmentations: Optional[List[str]] = None) -> np.ndarray:
        """对图像应用数据增强

        Args:
            image: 输入图像
            augmentations: 增强操作列表，默认随机选择

        Returns:
            增强后的图像
        """
        if augmentations is None:
            augmentations = random.sample(
                ["flip_h", "flip_v", "rotate_90", "rotate_180", "brightness", "contrast"],
                k=random.randint(1, 3),
            )
        img = image.copy()
        for aug in augmentations:
            if aug == "flip_h":
                img = np.fliplr(img)
            elif aug == "flip_v":
                img = np.flipud(img)
            elif aug == "rotate_90":
                img = np.rot90(img, k=1)
            elif aug == "rotate_180":
                img = np.rot90(img, k=2)
            elif aug == "brightness":
                factor = random.uniform(0.7, 1.3)
                img = (img.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)
            elif aug == "contrast":
                mean_val = img.mean()
                factor = random.uniform(0.7, 1.3)
                img = ((img.astype(np.float32) - mean_val) * factor + mean_val).clip(0, 255).astype(np.uint8)
        return img

    def split_dataset(self, test_ratio: float = 0.2, val_ratio: float = 0.1,
                       seed: Optional[int] = None) -> Dict[str, List[str]]:
        """划分数据集为训练/验证/测试集

        Args:
            test_ratio: 测试集比例
            val_ratio: 验证集比例
            seed: 随机种子

        Returns:
            包含train/val/test划分的字典
        """
        rng = random.Random(seed or self._seed)
        all_ids = list(self._images.keys())
        rng.shuffle(all_ids)
        n = len(all_ids)
        test_end = int(n * test_ratio)
        val_end = test_end + int(n * val_ratio)
        return {
            "train": all_ids[val_end:],
            "val": all_ids[test_end:val_end],
            "test": all_ids[:test_end],
        }

    def get_image(self, image_id: str) -> Optional[np.ndarray]:
        """获取图像数据

        Args:
            image_id: 图像ID

        Returns:
            numpy数组或None
        """
        return self._images.get(image_id)

    def get_metadata(self, image_id: str) -> Optional[ImageMetadata]:
        """获取图像元数据

        Args:
            image_id: 图像ID

        Returns:
            ImageMetadata或None
        """
        return self._metadata.get(image_id)

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息

        Returns:
            统计信息字典
        """
        if not self._metadata:
            return {"total_images": 0}
        widths = [m.width for m in self._metadata.values()]
        heights = [m.height for m in self._metadata.values()]
        sizes = [m.file_size for m in self._metadata.values()]
        categories = defaultdict(int)
        for m in self._metadata.values():
            categories[m.category] += 1
        return {
            "total_images": len(self._metadata),
            "width_range": (min(widths), max(widths)),
            "height_range": (min(heights), max(heights)),
            "avg_file_size": sum(sizes) / len(sizes) if sizes else 0,
            "total_file_size": sum(sizes),
            "categories": dict(categories),
            "images_with_annotations": len(self._annotations),
        }

    def get_images_by_category(self, category: str) -> List[str]:
        """按类别获取图像ID列表

        Args:
            category: 图像类别

        Returns:
            图像ID列表
        """
        return list(self._categories.get(category, []))

    def save_metadata_json(self, filepath: str) -> None:
        """保存元数据为JSON文件

        Args:
            filepath: 输出文件路径
        """
        data = {
            "statistics": self.get_statistics(),
            "images": {k: v.to_dict() for k, v in self._metadata.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def reset(self) -> None:
        """重置数据集"""
        self._images.clear()
        self._metadata.clear()
        self._annotations.clear()
        self._categories.clear()
