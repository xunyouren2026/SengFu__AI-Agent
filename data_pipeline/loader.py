"""
AGI Unified Framework - 统一数据加载器 (Unified Data Loader)

本模块提供通用的数据加载功能，支持多种数据格式、批量加载、数据打乱、
多进程预取和数据集分割等核心能力。基于纯Python标准库实现。

核心功能:
    - 多格式支持: JSON, CSV, 文本文件, 图像文件
    - 批量加载: 自动分批次、支持可变批次大小
    - 数据打乱: 随机打乱数据顺序
    - 多进程预取: 使用多进程并行预加载数据
    - 数据集分割: 支持 train/val/test 三分割

使用示例:
    >>> from agi_unified_framework.data_pipeline.loader import AGIDataLoader
    >>> loader = AGIDataLoader(data_dir="./data")
    >>> # 加载JSON数据
    >>> data = loader.load_json("annotations.json")
    >>> # 加载CSV数据
    >>> data = loader.load_csv("data.csv")
    >>> # 数据集分割
    >>> train, val, test = loader.split_dataset(data, ratios=[0.7, 0.15, 0.15])
"""

from __future__ import annotations

import os
import csv
import json
import random
import logging
import threading
import queue
import hashlib
import pickle
import copy
from typing import Dict, List, Optional, Tuple, Any, Callable, Union, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Manager, Queue as MPQueue
from collections import defaultdict

__all__ = ["AGIDataLoader"]
__version__ = "1.0.0"

logger = logging.getLogger(__name__)


# ==================== 数据集配置 ====================

@dataclass
class DatasetConfig:
    """数据集配置"""
    batch_size: int = 32                # 批次大小
    shuffle: bool = True                # 是否打乱数据
    num_workers: int = 0                # 预取工作进程数（0表示不使用多进程）
    prefetch_factor: int = 2            # 每个工作进程预取的批次数
    drop_last: bool = False             # 是否丢弃最后不完整的批次
    pin_memory: bool = False            # 是否将数据复制到固定内存区域
    max_queue_size: int = 10            # 预取队列最大大小
    seed: int = 42                      # 随机种子


# ==================== 数据样本 ====================

@dataclass
class DataSample:
    """
    数据样本

    统一的数据样本封装，支持多种数据类型。
    每个样本包含原始数据、标签和元信息。

    Attributes:
        data: 样本数据（可以是任意类型）
        label: 样本标签
        index: 样本在数据集中的索引
        metadata: 附加元信息
        source_path: 数据来源路径
    """
    data: Any = None
    label: Any = None
    index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "data": self.data,
            "label": self.label,
            "index": self.index,
            "metadata": self.metadata,
            "source_path": self.source_path,
        }


# ==================== 数据批次 ====================

@dataclass
class DataBatch:
    """
    数据批次

    将多个数据样本组合为一个批次，用于批量训练/推理。

    Attributes:
        samples: 样本列表
        batch_data: 批次数据列表
        batch_labels: 批次标签列表
        batch_indices: 批次索引列表
        batch_size: 实际批次大小
    """
    samples: List[DataSample] = field(default_factory=list)
    batch_data: List[Any] = field(default_factory=list)
    batch_labels: List[Any] = field(default_factory=list)
    batch_indices: List[int] = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        """实际批次大小"""
        return len(self.samples)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "data": self.batch_data,
            "labels": self.batch_labels,
            "indices": self.batch_indices,
            "batch_size": self.batch_size,
        }


# ==================== 统一数据加载器 ====================

class AGIDataLoader:
    """
    统一数据加载器

    提供企业级的数据加载功能，支持多种数据格式和高效的数据管线。

    支持的数据格式:
        - JSON (.json): 结构化数据加载
        - CSV (.csv): 表格数据加载
        - 文本 (.txt): 纯文本数据加载
        - 图像 (.jpg, .png, .bmp): 图像元数据加载
        - JSONL (.jsonl): 逐行JSON数据加载
        - Pickle (.pkl, .pickle): Python对象序列化数据

    核心特性:
        - 自动格式检测（根据文件扩展名）
        - 批量加载和迭代
        - 数据打乱和采样
        - 多进程/多线程预取
        - 数据集分割（train/val/test）
        - 数据变换管道

    使用示例:
        >>> loader = AGIDataLoader(data_dir="./data", batch_size=64)
        >>> # 加载数据
        >>> samples = loader.load("train.json")
        >>> # 创建数据迭代器
        >>> for batch in loader.iter_batches(samples):
        ...     data, labels = batch.batch_data, batch.batch_labels
        ...     # 训练步骤...
        >>> # 数据集分割
        >>> train, val, test = loader.split(samples, [0.8, 0.1, 0.1])
    """

    # 支持的文件格式映射
    SUPPORTED_FORMATS = {
        ".json": "json",
        ".jsonl": "jsonl",
        ".csv": "csv",
        ".txt": "text",
        ".jpg": "image",
        ".jpeg": "image",
        ".png": "image",
        ".bmp": "image",
        ".gif": "image",
        ".pkl": "pickle",
        ".pickle": "pickle",
    }

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        drop_last: bool = False,
        seed: int = 42,
        transform: Optional[Callable] = None,
        collate_fn: Optional[Callable] = None,
    ):
        """
        初始化数据加载器

        Args:
            data_dir: 数据根目录
            batch_size: 默认批次大小
            shuffle: 是否默认打乱数据
            num_workers: 预取工作进程数
            prefetch_factor: 每个工作进程预取的批次数
            drop_last: 是否丢弃最后不完整的批次
            seed: 随机种子
            transform: 数据变换函数
            collate_fn: 批次整理函数
        """
        self.data_dir = Path(data_dir)
        self.config = DatasetConfig(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            drop_last=drop_last,
            seed=seed,
        )
        self.transform = transform
        self.collate_fn = collate_fn

        # 随机数生成器
        self._rng = random.Random(seed)

        # 数据缓存
        self._data_cache: Dict[str, List[DataSample]] = {}

        # 预取线程/进程控制
        self._prefetch_queue: Optional[queue.Queue] = None
        self._prefetch_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # 统计信息
        self._stats = {
            "total_loaded": 0,
            "total_batches": 0,
            "load_time": 0.0,
        }

        # 确保数据目录存在
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"创建数据目录: {self.data_dir}")

        logger.info(
            f"AGIDataLoader 初始化完成: data_dir={self.data_dir}, "
            f"batch_size={batch_size}, shuffle={shuffle}, "
            f"num_workers={num_workers}"
        )

    # ==================== 数据加载 ====================

    def load(
        self,
        filepath: str,
        format: Optional[str] = None,
        **kwargs,
    ) -> List[DataSample]:
        """
        加载数据文件（自动检测格式）

        Args:
            filepath: 文件路径（相对路径基于data_dir）
            format: 强制指定格式（为None时自动检测）
            **kwargs: 传递给具体加载函数的额外参数

        Returns:
            DataSample列表
        """
        filepath = self._resolve_path(filepath)

        # 自动检测格式
        if format is None:
            format = self._detect_format(filepath)

        logger.info(f"加载数据: {filepath}, 格式={format}")

        if format == "json":
            return self.load_json(filepath, **kwargs)
        elif format == "jsonl":
            return self.load_jsonl(filepath, **kwargs)
        elif format == "csv":
            return self.load_csv(filepath, **kwargs)
        elif format == "text":
            return self.load_text(filepath, **kwargs)
        elif format == "image":
            return self.load_image_index(filepath, **kwargs)
        elif format == "pickle":
            return self.load_pickle(filepath, **kwargs)
        else:
            raise ValueError(
                f"不支持的文件格式: {format}，"
                f"支持的格式: {list(self.SUPPORTED_FORMATS.values())}"
            )

    def load_json(
        self,
        filepath: str,
        data_key: Optional[str] = None,
        label_key: Optional[str] = None,
        encoding: str = "utf-8",
    ) -> List[DataSample]:
        """
        加载JSON文件

        Args:
            filepath: JSON文件路径
            data_key: 数据字段名（为None时整个JSON作为数据）
            label_key: 标签字段名
            encoding: 文件编码

        Returns:
            DataSample列表
        """
        filepath = self._resolve_path(filepath)

        with open(filepath, "r", encoding=encoding) as f:
            data = json.load(f)

        samples = []

        # 处理列表格式
        if isinstance(data, list):
            for idx, item in enumerate(data):
                if isinstance(item, dict) and data_key:
                    sample_data = item.get(data_key, item)
                    sample_label = item.get(label_key) if label_key else None
                else:
                    sample_data = item
                    sample_label = None

                sample = DataSample(
                    data=sample_data,
                    label=sample_label,
                    index=idx,
                    source_path=str(filepath),
                )
                samples.append(sample)

        # 处理字典格式
        elif isinstance(data, dict):
            if data_key and data_key in data:
                items = data[data_key]
                if isinstance(items, list):
                    for idx, item in enumerate(items):
                        if isinstance(item, dict) and label_key:
                            sample_label = item.get(label_key)
                        else:
                            sample_label = None
                        sample = DataSample(
                            data=item,
                            label=sample_label,
                            index=idx,
                            source_path=str(filepath),
                        )
                        samples.append(sample)
            else:
                # 整个字典作为单个样本
                sample = DataSample(
                    data=data,
                    label=data.get(label_key) if label_key else None,
                    index=0,
                    source_path=str(filepath),
                )
                samples.append(sample)

        self._update_stats(len(samples))
        logger.info(f"JSON加载完成: {len(samples)} 个样本")
        return samples

    def load_jsonl(
        self,
        filepath: str,
        data_key: Optional[str] = None,
        label_key: Optional[str] = None,
        encoding: str = "utf-8",
        skip_errors: bool = True,
    ) -> List[DataSample]:
        """
        加载JSONL（逐行JSON）文件

        Args:
            filepath: JSONL文件路径
            data_key: 数据字段名
            label_key: 标签字段名
            encoding: 文件编码
            skip_errors: 是否跳过解析错误的行

        Returns:
            DataSample列表
        """
        filepath = self._resolve_path(filepath)
        samples = []

        with open(filepath, "r", encoding=encoding) as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)

                    if data_key:
                        sample_data = item.get(data_key, item)
                    else:
                        sample_data = item

                    sample_label = item.get(label_key) if label_key else None

                    sample = DataSample(
                        data=sample_data,
                        label=sample_label,
                        index=idx,
                        source_path=str(filepath),
                    )
                    samples.append(sample)

                except json.JSONDecodeError as e:
                    if skip_errors:
                        logger.warning(f"跳过解析错误行 {idx}: {e}")
                    else:
                        raise

        self._update_stats(len(samples))
        logger.info(f"JSONL加载完成: {len(samples)} 个样本")
        return samples

    def load_csv(
        self,
        filepath: str,
        delimiter: str = ",",
        has_header: bool = True,
        label_column: Optional[Union[str, int]] = None,
        encoding: str = "utf-8",
        columns: Optional[List[str]] = None,
    ) -> List[DataSample]:
        """
        加载CSV文件

        Args:
            filepath: CSV文件路径
            delimiter: 分隔符
            has_header: 是否有表头
            label_column: 标签列（列名或列索引）
            encoding: 文件编码
            columns: 要加载的列名列表（为None时加载所有列）

        Returns:
            DataSample列表
        """
        filepath = self._resolve_path(filepath)
        samples = []

        with open(filepath, "r", encoding=encoding, newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)

            headers = []
            if has_header:
                try:
                    headers = next(reader)
                except StopIteration:
                    return samples

            # 确定标签列索引
            label_idx = None
            if label_column is not None:
                if isinstance(label_column, str) and headers:
                    label_idx = headers.index(label_column) if label_column in headers else None
                elif isinstance(label_column, int):
                    label_idx = label_column

            # 确定数据列索引
            data_indices = list(range(len(headers))) if headers else None
            if columns and headers:
                data_indices = [
                    headers.index(col) for col in columns if col in headers
                ]
                if label_idx is not None and label_idx in data_indices:
                    data_indices.remove(label_idx)

            for idx, row in enumerate(reader):
                if not row:
                    continue

                # 提取标签
                sample_label = None
                if label_idx is not None and label_idx < len(row):
                    label_str = row[label_idx]
                    # 尝试转换为数值
                    try:
                        sample_label = int(label_str)
                    except ValueError:
                        try:
                            sample_label = float(label_str)
                        except ValueError:
                            sample_label = label_str

                # 提取数据
                if data_indices is not None:
                    sample_data = {headers[i]: row[i] for i in data_indices if i < len(row)}
                else:
                    sample_data = row

                sample = DataSample(
                    data=sample_data,
                    label=sample_label,
                    index=idx,
                    source_path=str(filepath),
                )
                samples.append(sample)

        self._update_stats(len(samples))
        logger.info(f"CSV加载完成: {len(samples)} 个样本, 列={headers}")
        return samples

    def load_text(
        self,
        filepath: str,
        encoding: str = "utf-8",
        skip_empty: bool = True,
        max_lines: int = 0,
    ) -> List[DataSample]:
        """
        加载文本文件

        Args:
            filepath: 文本文件路径
            encoding: 文件编码
            skip_empty: 是否跳过空行
            max_lines: 最大加载行数（0表示不限制）

        Returns:
            DataSample列表
        """
        filepath = self._resolve_path(filepath)
        samples = []

        with open(filepath, "r", encoding=encoding) as f:
            for idx, line in enumerate(f):
                if max_lines > 0 and idx >= max_lines:
                    break

                line = line.rstrip("\n\r")

                if skip_empty and not line.strip():
                    continue

                sample = DataSample(
                    data=line,
                    label=idx,
                    index=idx,
                    source_path=str(filepath),
                )
                samples.append(sample)

        self._update_stats(len(samples))
        logger.info(f"文本加载完成: {len(samples)} 个样本")
        return samples

    def load_image_index(
        self,
        filepath: str,
        encoding: str = "utf-8",
    ) -> List[DataSample]:
        """
        加载图像索引文件

        索引文件格式（每行一个）:
            image_path,label
            或
            image_path

        Args:
            filepath: 索引文件路径
            encoding: 文件编码

        Returns:
            DataSample列表（data为图像路径，非图像数据本身）
        """
        filepath = self._resolve_path(filepath)
        samples = []

        with open(filepath, "r", encoding=encoding) as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                parts = line.split(",")
                img_path = parts[0]
                label = parts[1] if len(parts) > 1 else None

                # 尝试转换标签
                if label is not None:
                    try:
                        label = int(label)
                    except ValueError:
                        try:
                            label = float(label)
                        except ValueError:
                            pass

                sample = DataSample(
                    data=img_path,
                    label=label,
                    index=idx,
                    source_path=str(filepath),
                    metadata={"image_path": img_path},
                )
                samples.append(sample)

        self._update_stats(len(samples))
        logger.info(f"图像索引加载完成: {len(samples)} 个样本")
        return samples

    def load_pickle(
        self,
        filepath: str,
    ) -> List[DataSample]:
        """
        加载Pickle文件

        Args:
            filepath: Pickle文件路径

        Returns:
            DataSample列表
        """
        filepath = self._resolve_path(filepath)

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        samples = []

        if isinstance(data, list):
            for idx, item in enumerate(data):
                sample = DataSample(
                    data=item,
                    index=idx,
                    source_path=str(filepath),
                )
                samples.append(sample)
        elif isinstance(data, dict):
            for idx, (key, value) in enumerate(data.items()):
                sample = DataSample(
                    data=value,
                    label=key,
                    index=idx,
                    source_path=str(filepath),
                )
                samples.append(sample)
        else:
            sample = DataSample(
                data=data,
                index=0,
                source_path=str(filepath),
            )
            samples.append(sample)

        self._update_stats(len(samples))
        logger.info(f"Pickle加载完成: {len(samples)} 个样本")
        return samples

    def load_directory(
        self,
        dirpath: str,
        pattern: str = "*",
        recursive: bool = False,
    ) -> List[DataSample]:
        """
        加载目录中的所有数据文件

        Args:
            dirpath: 目录路径
            pattern: 文件匹配模式（glob格式）
            recursive: 是否递归搜索子目录

        Returns:
            DataSample列表
        """
        dirpath = self._resolve_path(dirpath)
        samples = []

        if recursive:
            files = sorted(dirpath.rglob(pattern))
        else:
            files = sorted(dirpath.glob(pattern))

        for idx, filepath in enumerate(files):
            if filepath.is_file():
                try:
                    format = self._detect_format(filepath)
                    if format:
                        sample = DataSample(
                            data=str(filepath),
                            label=format,
                            index=idx,
                            source_path=str(filepath),
                            metadata={"format": format, "size": filepath.stat().st_size},
                        )
                        samples.append(sample)
                except Exception as e:
                    logger.warning(f"跳过文件 {filepath}: {e}")

        self._update_stats(len(samples))
        logger.info(f"目录加载完成: {len(samples)} 个文件")
        return samples

    # ==================== 批量处理 ====================

    def iter_batches(
        self,
        samples: List[DataSample],
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
        drop_last: Optional[bool] = None,
    ) -> Iterator[DataBatch]:
        """
        创建批次迭代器

        Args:
            samples: 数据样本列表
            batch_size: 批次大小（为None时使用配置默认值）
            shuffle: 是否打乱（为None时使用配置默认值）
            drop_last: 是否丢弃最后不完整批次

        Yields:
            DataBatch对象
        """
        if not samples:
            return

        batch_size = batch_size or self.config.batch_size
        shuffle = shuffle if shuffle is not None else self.config.shuffle
        drop_last = drop_last if drop_last is not None else self.config.drop_last

        # 打乱数据
        indices = list(range(len(samples)))
        if shuffle:
            self._rng.shuffle(indices)

        # 生成批次
        num_batches = len(indices) // batch_size
        if not drop_last and len(indices) % batch_size != 0:
            num_batches += 1

        self._stats["total_batches"] = num_batches

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(indices))
            batch_indices = indices[start:end]

            batch_samples = [samples[i] for i in batch_indices]

            # 应用变换
            if self.transform:
                batch_samples = [
                    self._apply_transform(s) for s in batch_samples
                ]

            # 创建批次
            batch = DataBatch(
                samples=batch_samples,
                batch_data=[s.data for s in batch_samples],
                batch_labels=[s.label for s in batch_samples],
                batch_indices=[s.index for s in batch_samples],
            )

            # 应用自定义collate函数
            if self.collate_fn:
                batch = self.collate_fn(batch)

            yield batch

    def get_batch(
        self,
        samples: List[DataSample],
        batch_idx: int,
        batch_size: Optional[int] = None,
    ) -> DataBatch:
        """
        获取指定批次

        Args:
            samples: 数据样本列表
            batch_idx: 批次索引
            batch_size: 批次大小

        Returns:
            DataBatch对象
        """
        batch_size = batch_size or self.config.batch_size
        start = batch_idx * batch_size
        end = min(start + batch_size, len(samples))

        if start >= len(samples):
            raise IndexError(f"批次索引越界: {batch_idx}")

        batch_samples = samples[start:end]

        return DataBatch(
            samples=batch_samples,
            batch_data=[s.data for s in batch_samples],
            batch_labels=[s.label for s in batch_samples],
            batch_indices=[s.index for s in batch_samples],
        )

    # ==================== 数据集分割 ====================

    def split_dataset(
        self,
        samples: List[DataSample],
        ratios: List[float] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> List[List[DataSample]]:
        """
        分割数据集

        Args:
            samples: 数据样本列表
            ratios: 分割比例列表，如 [0.7, 0.15, 0.15]
            shuffle: 是否在分割前打乱
            seed: 随机种子

        Returns:
            分割后的数据集列表

        Raises:
            ValueError: 比例之和不等于1
        """
        if ratios is None:
            ratios = [0.8, 0.1, 0.1]

        # 验证比例
        total_ratio = sum(ratios)
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"分割比例之和必须为1.0，当前为: {total_ratio}"
            )

        if not samples:
            return [[] for _ in ratios]

        # 复制并打乱
        indices = list(range(len(samples)))
        if shuffle:
            rng = random.Random(seed or self.config.seed)
            rng.shuffle(indices)

        # 分割
        splits = []
        start = 0
        total = len(indices)

        for ratio in ratios:
            size = int(total * ratio)
            split_indices = indices[start:start + size]
            split_samples = [samples[i] for i in split_indices]
            splits.append(split_samples)
            start += size

        # 将剩余样本分配到最后一个分割
        if start < total:
            remaining_indices = indices[start:]
            remaining_samples = [samples[i] for i in remaining_indices]
            splits[-1].extend(remaining_samples)

        split_names = ["train", "val", "test"]
        for i, split in enumerate(splits):
            name = split_names[i] if i < len(split_names) else f"split_{i}"
            logger.info(f"数据集分割 [{name}]: {len(split)} 个样本")

        return splits

    # ==================== 多进程预取 ====================

    def prefetch(
        self,
        samples: List[DataSample],
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> Iterator[DataBatch]:
        """
        多线程预取数据加载

        使用线程池预加载批次数据，减少I/O等待时间。

        Args:
            samples: 数据样本列表
            batch_size: 批次大小
            num_workers: 工作线程数

        Yields:
            DataBatch对象
        """
        batch_size = batch_size or self.config.batch_size
        num_workers = num_workers or self.config.num_workers

        if num_workers <= 0:
            # 不使用多线程，直接迭代
            yield from self.iter_batches(samples, batch_size)
            return

        # 创建预取队列
        prefetch_queue = queue.Queue(
            maxsize=num_workers * self.config.prefetch_factor
        )
        stop_event = threading.Event()

        # 预取工作函数
        def prefetch_worker(batch_indices_list):
            for batch_indices in batch_indices_list:
                if stop_event.is_set():
                    break
                batch_samples = [samples[i] for i in batch_indices]
                if self.transform:
                    batch_samples = [
                        self._apply_transform(s) for s in batch_samples
                    ]
                batch = DataBatch(
                    samples=batch_samples,
                    batch_data=[s.data for s in batch_samples],
                    batch_labels=[s.label for s in batch_samples],
                    batch_indices=[s.index for s in batch_samples],
                )
                prefetch_queue.put(batch)
            prefetch_queue.put(None)  # 结束标记

        # 准备批次索引
        indices = list(range(len(samples)))
        if self.config.shuffle:
            self._rng.shuffle(indices)

        batch_indices_list = []
        for i in range(0, len(indices), batch_size):
            batch_indices_list.append(indices[i:i + batch_size])

        # 启动工作线程
        threads = []
        chunk_size = max(1, len(batch_indices_list) // num_workers)
        for w in range(num_workers):
            start = w * chunk_size
            end = min(start + chunk_size, len(batch_indices_list))
            if start >= len(batch_indices_list):
                break
            chunk = batch_indices_list[start:end]
            t = threading.Thread(target=prefetch_worker, args=(chunk,))
            t.daemon = True
            t.start()
            threads.append(t)

        # 消费预取队列
        active_workers = len(threads)
        while active_workers > 0:
            batch = prefetch_queue.get()
            if batch is None:
                active_workers -= 1
            else:
                if self.collate_fn:
                    batch = self.collate_fn(batch)
                yield batch

        # 等待所有线程结束
        stop_event.set()
        for t in threads:
            t.join(timeout=1.0)

    # ==================== 工具方法 ====================

    def _resolve_path(self, filepath: str) -> Path:
        """
        解析文件路径

        如果是相对路径，则基于data_dir解析为绝对路径。

        Args:
            filepath: 文件路径

        Returns:
            解析后的Path对象
        """
        path = Path(filepath)
        if not path.is_absolute():
            path = self.data_dir / path
        return path

    def _detect_format(self, filepath: Union[str, Path]) -> str:
        """
        根据文件扩展名检测数据格式

        Args:
            filepath: 文件路径

        Returns:
            格式字符串

        Raises:
            ValueError: 无法识别的文件格式
        """
        filepath = Path(filepath)
        ext = filepath.suffix.lower()

        if ext in self.SUPPORTED_FORMATS:
            return self.SUPPORTED_FORMATS[ext]

        raise ValueError(
            f"无法识别的文件格式: {ext}，"
            f"支持的扩展名: {list(self.SUPPORTED_FORMATS.keys())}"
        )

    def _apply_transform(self, sample: DataSample) -> DataSample:
        """
        对单个样本应用变换

        Args:
            sample: 数据样本

        Returns:
            变换后的样本
        """
        if self.transform is None:
            return sample

        try:
            transformed = self.transform(sample)
            if isinstance(transformed, DataSample):
                return transformed
            elif isinstance(transformed, tuple) and len(transformed) == 2:
                return DataSample(
                    data=transformed[0],
                    label=transformed[1],
                    index=sample.index,
                    source_path=sample.source_path,
                    metadata=sample.metadata,
                )
            else:
                return DataSample(
                    data=transformed,
                    label=sample.label,
                    index=sample.index,
                    source_path=sample.source_path,
                    metadata=sample.metadata,
                )
        except Exception as e:
            logger.warning(f"变换失败 (样本 {sample.index}): {e}")
            return sample

    def _update_stats(self, num_samples: int) -> None:
        """更新加载统计"""
        self._stats["total_loaded"] += num_samples

    def get_stats(self) -> Dict[str, Any]:
        """
        获取加载统计信息

        Returns:
            统计信息字典
        """
        return {
            **self._stats,
            "data_dir": str(self.data_dir),
            "config": {
                "batch_size": self.config.batch_size,
                "shuffle": self.config.shuffle,
                "num_workers": self.config.num_workers,
            },
        }

    def list_files(
        self,
        pattern: str = "*",
        recursive: bool = False,
    ) -> List[str]:
        """
        列出数据目录中的文件

        Args:
            pattern: 文件匹配模式
            recursive: 是否递归搜索

        Returns:
            文件路径列表
        """
        if recursive:
            files = sorted(self.data_dir.rglob(pattern))
        else:
            files = sorted(self.data_dir.glob(pattern))

        return [str(f) for f in files if f.is_file()]

    def __len__(self) -> int:
        """返回已加载的总样本数"""
        return self._stats["total_loaded"]

    def __repr__(self) -> str:
        return (
            f"AGIDataLoader(data_dir='{self.data_dir}', "
            f"batch_size={self.config.batch_size}, "
            f"total_loaded={self._stats['total_loaded']})"
        )
