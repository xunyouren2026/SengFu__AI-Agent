"""
AGI Unified Framework - 数据预处理器 (Data Preprocessor)

本模块提供通用的数据预处理功能，支持文本、图像、数值等多种数据类型的
清洗、转换、归一化和增强操作。基于纯Python标准库实现。

核心功能:
    - 文本预处理: 清洗、分词、归一化、编码
    - 图像预处理: 缩放、裁剪、归一化、颜色空间转换
    - 数值预处理: 标准化、归一化、缺失值填充、异常值处理
    - 数据增强: 翻转、旋转、噪声注入、混合增强

使用示例:
    >>> from agi_unified_framework.data_pipeline.preprocessor import DataPreprocessor
    >>> preprocessor = DataPreprocessor()
    >>> # 文本清洗
    >>> clean_text = preprocessor.clean_text("Hello\\nWorld!!  ")
    >>> # 数值标准化
    >>> normalized = preprocessor.standardize([1.0, 2.0, 3.0, 4.0, 5.0])
"""

from __future__ import annotations

import math
import re
import random
import unicodedata
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from collections import Counter

__all__ = ["DataPreprocessor"]
__version__ = "1.0.0"

logger = logging.getLogger(__name__)


# ==================== 文本预处理配置 ====================

@dataclass
class TextPreprocessConfig:
    """文本预处理配置"""
    lowercase: bool = True               # 是否转为小写
    remove_html: bool = True             # 是否去除HTML标签
    remove_urls: bool = True             # 是否去除URL
    remove_emails: bool = True           # 是否去除邮箱地址
    remove_numbers: bool = False         # 是否去除数字
    remove_punctuation: bool = False     # 是否去除标点符号
    remove_extra_whitespace: bool = True # 是否去除多余空白
    normalize_unicode: bool = True       # 是否标准化Unicode
    strip_accents: bool = False          # 是否去除重音符号
    max_length: int = 0                  # 最大文本长度（0表示不限制）
    padding_token: str = "[PAD]"         # 填充标记
    unknown_token: str = "[UNK]"         # 未知标记


# ==================== 图像预处理配置 ====================

@dataclass
class ImagePreprocessConfig:
    """图像预处理配置"""
    target_width: int = 224              # 目标宽度
    target_height: int = 224             # 目标高度
    resize_mode: str = "bilinear"        # 缩放模式: nearest, bilinear
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    channels_first: bool = True          # 是否将通道维度放在前面 (C, H, W)


# ==================== 数值预处理配置 ====================

@dataclass
class NumericPreprocessConfig:
    """数值预处理配置"""
    standardize: bool = True             # 是否标准化 (z-score)
    min_max_scale: bool = False          # 是否使用MinMax缩放
    feature_range: Tuple[float, float] = (0.0, 1.0)  # MinMax目标范围
    handle_missing: str = "mean"         # 缺失值处理: mean, median, zero, drop
    clip_outliers: bool = False          # 是否裁剪异常值
    outlier_std_threshold: float = 3.0   # 异常值标准差阈值


# ==================== 数据增强配置 ====================

@dataclass
class AugmentConfig:
    """数据增强配置"""
    horizontal_flip_prob: float = 0.5    # 水平翻转概率
    vertical_flip_prob: float = 0.0      # 垂直翻转概率
    rotation_range: float = 0.0          # 旋转角度范围（度），0表示不旋转
    noise_std: float = 0.0               # 高斯噪声标准差
    brightness_range: Tuple[float, float] = (1.0, 1.0)  # 亮度调整范围
    contrast_range: Tuple[float, float] = (1.0, 1.0)    # 对比度调整范围
    mixup_alpha: float = 0.0             # MixUp混合系数（0表示不使用）
    cutmix_alpha: float = 0.0            # CutMix混合系数（0表示不使用）


# ==================== 数据预处理器 ====================

class DataPreprocessor:
    """
    数据预处理器

    提供统一的数据预处理接口，支持文本、图像、数值等多种数据类型。
    所有操作基于纯Python标准库实现，不依赖第三方库。

    功能概览:
        文本处理:
            - clean_text(): 文本清洗（去HTML、URL、特殊字符等）
            - tokenize(): 文本分词（支持多种分词策略）
            - build_vocab(): 构建词汇表
            - encode_text(): 文本编码为ID序列
            - normalize_text(): 文本归一化

        图像处理:
            - resize_image(): 图像缩放（最近邻/双线性插值）
            - crop_image(): 图像裁剪（中心/随机）
            - normalize_image(): 图像归一化
            - flip_image(): 图像翻转

        数值处理:
            - standardize(): Z-score标准化
            - minmax_scale(): MinMax缩放
            - fill_missing(): 缺失值填充
            - clip_outliers(): 异常值裁剪

        数据增强:
            - augment_text(): 文本增强（同义词替换等）
            - augment_image(): 图像增强（翻转、旋转、噪声等）
            - augment_numeric(): 数值增强（噪声注入）

    使用示例:
        >>> preprocessor = DataPreprocessor()
        >>> # 文本预处理流水线
        >>> text = "  Check out https://example.com for <b>info</b>!  "
        >>> cleaned = preprocessor.clean_text(text)
        >>> tokens = preprocessor.tokenize(cleaned)
        >>> # 数值预处理
        >>> data = [1.0, 2.0, float('nan'), 4.0, 5.0]
        >>> filled = preprocessor.fill_missing(data)
        >>> normalized = preprocessor.standardize(filled)
    """

    def __init__(
        self,
        text_config: Optional[TextPreprocessConfig] = None,
        image_config: Optional[ImagePreprocessConfig] = None,
        numeric_config: Optional[NumericPreprocessConfig] = None,
        augment_config: Optional[AugmentConfig] = None,
        seed: Optional[int] = None,
    ):
        """
        初始化数据预处理器

        Args:
            text_config: 文本预处理配置
            image_config: 图像预处理配置
            numeric_config: 数值预处理配置
            augment_config: 数据增强配置
            seed: 随机种子
        """
        self.text_config = text_config or TextPreprocessConfig()
        self.image_config = image_config or ImagePreprocessConfig()
        self.numeric_config = numeric_config or NumericPreprocessConfig()
        self.augment_config = augment_config or AugmentConfig()

        if seed is not None:
            random.seed(seed)

        # 词汇表（用于文本编码）
        self._vocab: Dict[str, int] = {}
        self._inverse_vocab: Dict[int, str] = {}

        # 数值预处理统计量（拟合后存储）
        self._mean: Optional[float] = None
        self._std: Optional[float] = None
        self._min_val: Optional[float] = None
        self._max_val: Optional[float] = None
        self._median: Optional[float] = None

        # 正则表达式预编译（提升性能）
        self._html_pattern = re.compile(r"<[^>]+>")
        self._url_pattern = re.compile(
            r"https?://\S+|www\.\S+"
        )
        self._email_pattern = re.compile(r"\S+@\S+\.\S+")
        self._number_pattern = re.compile(r"\d+")
        self._punctuation_pattern = re.compile(
            r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]'
        )
        self._whitespace_pattern = re.compile(r"\s+")
        self._non_printable_pattern = re.compile(
            r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
        )

        logger.info("DataPreprocessor 初始化完成")

    # ==================== 文本预处理 ====================

    def clean_text(self, text: str) -> str:
        """
        文本清洗

        执行以下清洗步骤（按配置）:
        1. 去除不可打印字符
        2. 标准化Unicode
        3. 去除重音符号
        4. 去除HTML标签
        5. 去除URL和邮箱
        6. 去除数字（可选）
        7. 去除标点符号（可选）
        8. 转为小写（可选）
        9. 去除多余空白

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        if not text:
            return ""

        cfg = self.text_config

        # 去除不可打印字符
        text = self._non_printable_pattern.sub("", text)

        # Unicode标准化
        if cfg.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        # 去除重音符号
        if cfg.strip_accents:
            text = unicodedata.normalize("NFD", text)
            text = "".join(
                c for c in text
                if unicodedata.category(c) != "Mn"
            )

        # 去除HTML标签
        if cfg.remove_html:
            text = self._html_pattern.sub(" ", text)

        # 去除URL
        if cfg.remove_urls:
            text = self._url_pattern.sub(" ", text)

        # 去除邮箱
        if cfg.remove_emails:
            text = self._email_pattern.sub(" ", text)

        # 去除数字
        if cfg.remove_numbers:
            text = self._number_pattern.sub(" ", text)

        # 去除标点符号
        if cfg.remove_punctuation:
            text = self._punctuation_pattern.sub(" ", text)

        # 转为小写
        if cfg.lowercase:
            text = text.lower()

        # 去除多余空白
        if cfg.remove_extra_whitespace:
            text = self._whitespace_pattern.sub(" ", text).strip()

        # 截断到最大长度
        if cfg.max_length > 0 and len(text) > cfg.max_length:
            text = text[:cfg.max_length]

        return text

    def tokenize(
        self,
        text: str,
        method: str = "whitespace",
    ) -> List[str]:
        """
        文本分词

        Args:
            text: 输入文本
            method: 分词方法
                - "whitespace": 按空白字符分割（默认）
                - "character": 按字符分割
                - "ngram": N-gram分词

        Returns:
            分词结果列表
        """
        if not text:
            return []

        if method == "whitespace":
            return text.split()
        elif method == "character":
            return list(text)
        elif method == "ngram":
            # 默认使用bigram
            tokens = text.split()
            ngrams = []
            for i in range(len(tokens) - 1):
                ngrams.append(f"{tokens[i]}_{tokens[i+1]}")
            return ngrams
        else:
            raise ValueError(f"未知的分词方法: {method}")

    def build_vocab(
        self,
        texts: List[str],
        min_freq: int = 1,
        max_vocab_size: int = 50000,
    ) -> Dict[str, int]:
        """
        构建词汇表

        Args:
            texts: 文本列表
            min_freq: 最小词频阈值
            max_vocab_size: 最大词汇表大小

        Returns:
            词汇表字典 {token: id}
        """
        # 统计词频
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)

        # 按词频排序，过滤低频词
        special_tokens = [
            self.text_config.padding_token,
            self.text_config.unknown_token,
        ]

        self._vocab = {}
        idx = 0
        for token in special_tokens:
            self._vocab[token] = idx
            idx += 1

        for token, freq in counter.most_common(max_vocab_size):
            if freq >= min_freq and token not in self._vocab:
                self._vocab[token] = idx
                idx += 1
                if idx >= max_vocab_size:
                    break

        # 构建反向词汇表
        self._inverse_vocab = {v: k for k, v in self._vocab.items()}

        logger.info(
            f"词汇表构建完成: 大小={len(self._vocab)}, "
            f"特殊标记={special_tokens}"
        )
        return self._vocab

    def encode_text(
        self,
        text: str,
        max_length: int = 128,
        add_padding: bool = True,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """
        将文本编码为ID序列

        Args:
            text: 输入文本
            max_length: 最大序列长度
            add_padding: 是否添加填充
            add_bos: 是否添加起始标记
            add_eos: 是否添加结束标记

        Returns:
            编码后的ID列表
        """
        tokens = self.tokenize(text)

        # 添加特殊标记
        ids = []
        if add_bos:
            ids.append(self._vocab.get("<bos>", 0))
        ids.extend(
            self._vocab.get(t, self._vocab.get(
                self.text_config.unknown_token, 1
            ))
            for t in tokens
        )
        if add_eos:
            ids.append(self._vocab.get("<eos>", 0))

        # 截断
        if max_length > 0 and len(ids) > max_length:
            ids = ids[:max_length]

        # 填充
        if add_padding and max_length > 0:
            pad_id = self._vocab.get(self.text_config.padding_token, 0)
            while len(ids) < max_length:
                ids.append(pad_id)

        return ids

    def decode_text(self, ids: List[int]) -> str:
        """
        将ID序列解码为文本

        Args:
            ids: ID列表

        Returns:
            解码后的文本
        """
        tokens = [
            self._inverse_vocab.get(idx, self.text_config.unknown_token)
            for idx in ids
        ]
        return " ".join(tokens)

    def normalize_text(self, text: str) -> str:
        """
        文本归一化

        执行Unicode标准化、空白归一化等操作。

        Args:
            text: 输入文本

        Returns:
            归一化后的文本
        """
        if not text:
            return ""

        # Unicode标准化
        text = unicodedata.normalize("NFKC", text)

        # 全角转半角
        result = []
        for char in text:
            code = ord(char)
            # 全角空格
            if code == 0x3000:
                result.append(" ")
            # 全角字符范围 (！到～)
            elif 0xFF01 <= code <= 0xFF5E:
                result.append(chr(code - 0xFEE0))
            else:
                result.append(char)

        text = "".join(result)

        # 空白归一化
        text = self._whitespace_pattern.sub(" ", text).strip()

        return text

    # ==================== 图像预处理 ====================

    def resize_image(
        self,
        image: List[List[List[float]]],
        target_width: int = 0,
        target_height: int = 0,
        mode: str = "bilinear",
    ) -> List[List[List[float]]]:
        """
        图像缩放

        支持最近邻插值和双线性插值两种缩放模式。
        输入图像格式: [height][width][channels] (HWC)

        Args:
            image: 输入图像（三维列表，HWC格式）
            target_width: 目标宽度（0表示使用配置默认值）
            target_height: 目标高度（0表示使用配置默认值）
            mode: 缩放模式 ("nearest" 或 "bilinear")

        Returns:
            缩放后的图像
        """
        if not image or not image[0]:
            return image

        tw = target_width or self.image_config.target_width
        th = target_height or self.image_config.target_height

        h = len(image)
        w = len(image[0])
        c = len(image[0][0])

        if mode == "nearest":
            # 最近邻插值
            resized = []
            for y in range(th):
                row = []
                src_y = min(int(y * h / th), h - 1)
                for x in range(tw):
                    src_x = min(int(x * w / tw), w - 1)
                    row.append(list(image[src_y][src_x]))
                resized.append(row)
            return resized

        elif mode == "bilinear":
            # 双线性插值
            resized = []
            for y in range(th):
                row = []
                # 源坐标（浮点）
                src_y_f = y * (h - 1) / max(1, th - 1)
                src_y0 = int(src_y_f)
                src_y1 = min(src_y0 + 1, h - 1)
                fy = src_y_f - src_y0

                for x in range(tw):
                    src_x_f = x * (w - 1) / max(1, tw - 1)
                    src_x0 = int(src_x_f)
                    src_x1 = min(src_x0 + 1, w - 1)
                    fx = src_x_f - src_x0

                    # 四个邻近像素的双线性插值
                    pixel = []
                    for ch in range(c):
                        val = (
                            image[src_y0][src_x0][ch] * (1 - fx) * (1 - fy)
                            + image[src_y0][src_x1][ch] * fx * (1 - fy)
                            + image[src_y1][src_x0][ch] * (1 - fx) * fy
                            + image[src_y1][src_x1][ch] * fx * fy
                        )
                        pixel.append(val)
                    row.append(pixel)
                resized.append(row)
            return resized

        else:
            raise ValueError(f"未知的缩放模式: {mode}")

    def crop_image(
        self,
        image: List[List[List[float]]],
        crop_height: int,
        crop_width: int,
        mode: str = "center",
    ) -> List[List[List[float]]]:
        """
        图像裁剪

        Args:
            image: 输入图像（HWC格式）
            crop_height: 裁剪高度
            crop_width: 裁剪宽度
            mode: 裁剪模式 ("center" 中心裁剪, "random" 随机裁剪)

        Returns:
            裁剪后的图像
        """
        if not image or not image[0]:
            return image

        h = len(image)
        w = len(image[0])

        # 确保裁剪尺寸不超过原图
        crop_height = min(crop_height, h)
        crop_width = min(crop_width, w)

        if mode == "center":
            # 中心裁剪
            start_y = (h - crop_height) // 2
            start_x = (w - crop_width) // 2
        elif mode == "random":
            # 随机裁剪
            start_y = random.randint(0, h - crop_height)
            start_x = random.randint(0, w - crop_width)
        else:
            raise ValueError(f"未知的裁剪模式: {mode}")

        cropped = []
        for y in range(start_y, start_y + crop_height):
            row = []
            for x in range(start_x, start_x + crop_width):
                row.append(list(image[y][x]))
            cropped.append(row)

        return cropped

    def normalize_image(
        self,
        image: List[List[List[float]]],
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ) -> List[List[List[float]]]:
        """
        图像归一化

        对每个通道执行: normalized = (pixel - mean) / std

        Args:
            image: 输入图像（HWC格式），像素值范围通常为 [0, 255] 或 [0, 1]
            mean: 各通道均值列表
            std: 各通道标准差列表

        Returns:
            归一化后的图像
        """
        if not image or not image[0]:
            return image

        mean = mean or self.image_config.normalize_mean
        std = std or self.image_config.normalize_std

        num_channels = len(image[0][0])

        # 确保mean和std的长度匹配
        if len(mean) < num_channels:
            mean = mean + [0.0] * (num_channels - len(mean))
        if len(std) < num_channels:
            std = std + [1.0] * (num_channels - len(std))

        normalized = []
        for row in image:
            new_row = []
            for pixel in row:
                new_pixel = []
                for ch in range(num_channels):
                    val = (pixel[ch] - mean[ch]) / (std[ch] + 1e-8)
                    new_pixel.append(val)
                new_row.append(new_pixel)
            normalized.append(new_row)

        return normalized

    def flip_image(
        self,
        image: List[List[List[float]]],
        direction: str = "horizontal",
    ) -> List[List[List[float]]]:
        """
        图像翻转

        Args:
            image: 输入图像（HWC格式）
            direction: 翻转方向 ("horizontal" 水平, "vertical" 垂直)

        Returns:
            翻转后的图像
        """
        if not image:
            return image

        if direction == "horizontal":
            # 水平翻转（每行逆序）
            return [row[::-1] for row in image]
        elif direction == "vertical":
            # 垂直翻转（行逆序）
            return image[::-1]
        else:
            raise ValueError(f"未知的翻转方向: {direction}")

    def rotate_image(
        self,
        image: List[List[List[float]]],
        angle: float,
        fill_value: float = 0.0,
    ) -> List[List[List[float]]]:
        """
        图像旋转（90度倍数旋转，使用矩阵转置实现）

        Args:
            image: 输入图像（HWC格式）
            angle: 旋转角度（仅支持90, 180, 270）
            fill_value: 填充值

        Returns:
            旋转后的图像
        """
        if not image:
            return image

        angle = angle % 360

        if angle == 90:
            # 顺时针90度 = 转置 + 水平翻转
            h = len(image)
            w = len(image[0])
            rotated = []
            for x in range(w):
                row = []
                for y in range(h - 1, -1, -1):
                    row.append(list(image[y][x]))
                rotated.append(row)
            return rotated
        elif angle == 180:
            # 180度 = 垂直翻转 + 水平翻转
            return [row[::-1] for row in image[::-1]]
        elif angle == 270:
            # 顺时针270度 = 转置 + 垂直翻转
            h = len(image)
            w = len(image[0])
            rotated = []
            for x in range(w - 1, -1, -1):
                row = []
                for y in range(h):
                    row.append(list(image[y][x]))
                rotated.append(row)
            return rotated
        else:
            raise ValueError(
                f"仅支持90度倍数旋转，收到: {angle}度"
            )

    # ==================== 数值预处理 ====================

    def standardize(
        self,
        data: List[float],
        return_stats: bool = False,
    ) -> Union[List[float], Tuple[List[float], Dict[str, float]]]:
        """
        Z-score标准化

        转换公式: z = (x - mean) / std

        Args:
            data: 数值列表
            return_stats: 是否返回统计量

        Returns:
            标准化后的列表，或 (标准化列表, 统计量字典)
        """
        if not data:
            return ([], {"mean": 0.0, "std": 1.0}) if return_stats else []

        # 计算均值和标准差
        n = len(data)
        mean = sum(data) / n
        variance = sum((x - mean) ** 2 for x in data) / n
        std = math.sqrt(variance) if variance > 0 else 1e-8

        # 缓存统计量
        self._mean = mean
        self._std = std

        # 标准化
        standardized = [(x - mean) / std for x in data]

        if return_stats:
            return standardized, {"mean": mean, "std": std}
        return standardized

    def minmax_scale(
        self,
        data: List[float],
        feature_range: Tuple[float, float] = (0.0, 1.0),
    ) -> List[float]:
        """
        MinMax缩放

        转换公式: x_scaled = (x - min) / (max - min) * (range_max - range_min) + range_min

        Args:
            data: 数值列表
            feature_range: 目标范围

        Returns:
            缩放后的列表
        """
        if not data:
            return []

        min_val = min(data)
        max_val = max(data)
        data_range = max_val - min_val

        # 缓存统计量
        self._min_val = min_val
        self._max_val = max_val

        if data_range < 1e-8:
            # 所有值相同，映射到范围中点
            mid = (feature_range[0] + feature_range[1]) / 2
            return [mid] * len(data)

        range_span = feature_range[1] - feature_range[0]
        scaled = [
            (x - min_val) / data_range * range_span + feature_range[0]
            for x in data
        ]
        return scaled

    def fill_missing(
        self,
        data: List[Optional[float]],
        strategy: str = "",
    ) -> List[float]:
        """
        缺失值填充

        Args:
            data: 可能包含None的数值列表
            strategy: 填充策略
                - "mean": 用均值填充（默认）
                - "median": 用中位数填充
                - "zero": 用0填充
                - "forward": 前向填充
                - "drop": 删除缺失值

        Returns:
            填充后的列表
        """
        strategy = strategy or self.numeric_config.handle_missing

        if not data:
            return []

        # 分离有效值和缺失位置
        valid_values = [x for x in data if x is not None]

        if not valid_values:
            return [0.0] * len(data)

        if strategy == "mean":
            fill_value = sum(valid_values) / len(valid_values)
            self._mean = fill_value
        elif strategy == "median":
            sorted_vals = sorted(valid_values)
            n = len(sorted_vals)
            if n % 2 == 0:
                fill_value = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
            else:
                fill_value = sorted_vals[n // 2]
            self._median = fill_value
        elif strategy == "zero":
            fill_value = 0.0
        elif strategy == "forward":
            # 前向填充
            result = []
            last_valid = valid_values[0]
            for x in data:
                if x is not None:
                    last_valid = x
                    result.append(x)
                else:
                    result.append(last_valid)
            return result
        elif strategy == "drop":
            return valid_values
        else:
            raise ValueError(f"未知的缺失值填充策略: {strategy}")

        return [
            x if x is not None else fill_value for x in data
        ]

    def clip_outliers(
        self,
        data: List[float],
        threshold: float = 0.0,
    ) -> List[float]:
        """
        异常值裁剪

        将超出 mean +/- threshold * std 范围的值裁剪到边界。

        Args:
            data: 数值列表
            threshold: 标准差倍数阈值（0表示使用配置默认值）

        Returns:
            裁剪后的列表
        """
        if not data:
            return []

        threshold = threshold or self.numeric_config.outlier_std_threshold

        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        std = math.sqrt(variance) if variance > 0 else 1e-8

        lower = mean - threshold * std
        upper = mean + threshold * std

        return [max(lower, min(upper, x)) for x in data]

    # ==================== 数据增强 ====================

    def augment_text(
        self,
        text: str,
        synonym_replace_prob: float = 0.1,
        random_delete_prob: float = 0.0,
        random_swap_prob: float = 0.0,
    ) -> str:
        """
        文本数据增强

        Args:
            text: 输入文本
            synonym_replace_prob: 同义词替换概率（简化实现为随机删除再插入）
            random_delete_prob: 随机删除词的概率
            random_swap_prob: 随机交换相邻词的概率

        Returns:
            增强后的文本
        """
        tokens = self.tokenize(text)
        if len(tokens) <= 1:
            return text

        # 随机删除
        if random_delete_prob > 0:
            tokens = [
                t for t in tokens
                if random.random() > random_delete_prob
            ]
            if not tokens:
                tokens = [text.split()[0]] if text.split() else [""]

        # 随机交换
        if random_swap_prob > 0 and len(tokens) >= 2:
            num_swaps = max(1, int(len(tokens) * random_swap_prob))
            for _ in range(num_swaps):
                i = random.randint(0, len(tokens) - 2)
                tokens[i], tokens[i + 1] = tokens[i + 1], tokens[i]

        return " ".join(tokens)

    def augment_image(
        self,
        image: List[List[List[float]]],
        config: Optional[AugmentConfig] = None,
    ) -> List[List[List[float]]]:
        """
        图像数据增强

        按配置依次应用: 水平翻转、垂直翻转、旋转、亮度调整、噪声注入。

        Args:
            image: 输入图像（HWC格式）
            config: 增强配置（为None时使用默认配置）

        Returns:
            增强后的图像
        """
        if not image:
            return image

        cfg = config or self.augment_config

        # 水平翻转
        if cfg.horizontal_flip_prob > 0 and random.random() < cfg.horizontal_flip_prob:
            image = self.flip_image(image, "horizontal")

        # 垂直翻转
        if cfg.vertical_flip_prob > 0 and random.random() < cfg.vertical_flip_prob:
            image = self.flip_image(image, "vertical")

        # 旋转（90度倍数）
        if cfg.rotation_range > 0:
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                image = self.rotate_image(image, angle)

        # 亮度调整
        if cfg.brightness_range != (1.0, 1.0):
            factor = random.uniform(*cfg.brightness_range)
            image = [
                [
                    [min(1.0, max(0.0, ch * factor)) for ch in pixel]
                    for pixel in row
                ]
                for row in image
            ]

        # 对比度调整
        if cfg.contrast_range != (1.0, 1.0):
            factor = random.uniform(*cfg.contrast_range)
            # 计算均值
            h = len(image)
            w = len(image[0])
            c = len(image[0][0])
            mean_val = sum(
                image[y][x][ch]
                for y in range(h) for x in range(w) for ch in range(c)
            ) / (h * w * c)
            image = [
                [
                    [
                        min(1.0, max(0.0, mean_val + (ch - mean_val) * factor))
                        for ch in pixel
                    ]
                    for pixel in row
                ]
                for row in image
            ]

        # 高斯噪声
        if cfg.noise_std > 0:
            image = [
                [
                    [
                        ch + random.gauss(0, cfg.noise_std)
                        for ch in pixel
                    ]
                    for pixel in row
                ]
                for row in image
            ]

        return image

    def augment_numeric(
        self,
        data: List[float],
        noise_std: float = 0.0,
        scale_range: Tuple[float, float] = (1.0, 1.0),
    ) -> List[float]:
        """
        数值数据增强

        Args:
            data: 数值列表
            noise_std: 高斯噪声标准差
            scale_range: 随机缩放范围

        Returns:
            增强后的列表
        """
        result = list(data)

        # 随机缩放
        if scale_range != (1.0, 1.0):
            factor = random.uniform(*scale_range)
            result = [x * factor for x in result]

        # 高斯噪声
        if noise_std > 0:
            result = [x + random.gauss(0, noise_std) for x in result]

        return result

    # ==================== 批量处理 ====================

    def preprocess_batch(
        self,
        texts: Optional[List[str]] = None,
        images: Optional[List[List[List[List[float]]]]] = None,
        numerical: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        """
        批量预处理

        对多种数据类型进行批量预处理，返回统一的结果字典。

        Args:
            texts: 文本列表
            images: 图像列表
            numerical: 数值数据列表

        Returns:
            包含所有预处理结果的字典
        """
        result = {}

        if texts is not None:
            result["cleaned_texts"] = [self.clean_text(t) for t in texts]
            result["tokens"] = [
                self.tokenize(t) for t in result["cleaned_texts"]
            ]
            if self._vocab:
                result["encoded"] = [
                    self.encode_text(t) for t in result["cleaned_texts"]
                ]

        if images is not None:
            result["resized_images"] = [
                self.resize_image(img) for img in images
            ]
            result["normalized_images"] = [
                self.normalize_image(img)
                for img in result["resized_images"]
            ]

        if numerical is not None:
            # 填充缺失值
            filled = [self.fill_missing(d) for d in numerical]
            result["filled_numerical"] = filled

            # 标准化
            if self.numeric_config.standardize:
                result["standardized"] = [self.standardize(d) for d in filled]

            # MinMax缩放
            if self.numeric_config.min_max_scale:
                result["scaled"] = [
                    self.minmax_scale(d, self.numeric_config.feature_range)
                    for d in filled
                ]

        return result

    def get_stats(self) -> Dict[str, Any]:
        """
        获取预处理器统计信息

        Returns:
            包含词汇表大小、数值统计量等的字典
        """
        return {
            "vocab_size": len(self._vocab),
            "mean": self._mean,
            "std": self._std,
            "min_val": self._min_val,
            "max_val": self._max_val,
            "median": self._median,
        }

    def __repr__(self) -> str:
        return (
            f"DataPreprocessor("
            f"vocab_size={len(self._vocab)}, "
            f"text_config={self.text_config}, "
            f"image_config={self.image_config})"
        )
