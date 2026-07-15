"""
AGI Unified Framework - 企业级数据管道模块 (Enterprise Data Pipeline Module)

本模块提供完整的数据管道解决方案，涵盖数据下载、物理仿真数据生成、
数据预处理、数据加载和缓存管理等核心功能。

模块组成:
    - downloader: 数据下载器，支持断点续传、并行下载、文件校验
    - physics_sim: 物理仿真数据生成，支持刚体模拟、点云生成、场景序列
    - preprocessor: 数据预处理，支持点云、视频、文本等多种数据类型
    - loader: 数据加载器，支持多种格式、动态batching、分布式采样
    - cache: 缓存管理，支持磁盘LRU、内存缓存、压缩存储

使用示例:
    >>> from agi_unified_framework.data_pipeline import (
    ...     DataDownloader,
    ...     PhysicsDataGenerator,
    ...     DataPreprocessor,
    ...     AGIDataLoader,
    ...     DataCache,
    ... )
    >>> downloader = DataDownloader(output_dir="./downloads")
    >>> downloader.download("https://example.com/data.zip")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# 配置模块级别日志
logger = logging.getLogger(__name__)

__all__ = [
    "DataDownloader",
    "PhysicsDataGenerator",
    "DataPreprocessor",
    "AGIDataLoader",
    "DataCache",
    "DownloadTask",
    "DownloadStatus",
    "RigidBody",
    "PhysicsScene",
    "PointCloudData",
    "CacheEntry",
    "CacheStats",
]

# 延迟导入，避免循环依赖和启动开销
def __getattr__(name: str):
    """延迟导入模块组件，减少启动时的导入开销。"""
    if name == "DataDownloader":
        from .downloader import DataDownloader
        return DataDownloader
    if name == "DownloadTask":
        from .downloader import DownloadTask
        return DownloadTask
    if name == "DownloadStatus":
        from .downloader import DownloadStatus
        return DownloadStatus
    if name == "PhysicsDataGenerator":
        from .physics_sim import PhysicsDataGenerator
        return PhysicsDataGenerator
    if name == "RigidBody":
        from .physics_sim import RigidBody
        return RigidBody
    if name == "PhysicsScene":
        from .physics_sim import PhysicsScene
        return PhysicsScene
    if name == "PointCloudData":
        from .physics_sim import PointCloudData
        return PointCloudData
    if name == "DataPreprocessor":
        from .preprocessor import DataPreprocessor
        return DataPreprocessor
    if name == "AGIDataLoader":
        from .loader import AGIDataLoader
        return AGIDataLoader
    if name == "DataCache":
        from .cache import DataCache
        return DataCache
    if name == "CacheEntry":
        from .cache import CacheEntry
        return CacheEntry
    if name == "CacheStats":
        from .cache import CacheStats
        return CacheStats
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "1.0.0"
__author__ = "AGI Unified Framework Team"
