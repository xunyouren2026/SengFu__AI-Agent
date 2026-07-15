"""
AGI Unified Framework - 数据缓存管理器 (Data Cache Manager)

本模块提供企业级的数据缓存解决方案，支持LRU淘汰策略、磁盘持久化、
内存管理和缓存统计等功能。基于纯Python标准库实现。

核心组件:
    - CacheEntry: 缓存条目，封装缓存数据的元信息
    - CacheStats: 缓存统计，追踪命中率、内存使用等指标
    - DataCache: LRU缓存管理器，支持内存缓存和磁盘持久化

使用示例:
    >>> from agi_unified_framework.data_pipeline.cache import DataCache
    >>> cache = DataCache(max_size=1000, max_memory_mb=512)
    >>> cache.put("key1", {"data": [1, 2, 3]})
    >>> value = cache.get("key1")
    >>> stats = cache.get_stats()
    >>> print(f"命中率: {stats.hit_rate:.1%}")
"""

from __future__ import annotations

import os
import json
import time
import struct
import logging
import threading
import hashlib
import pickle
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
from collections import OrderedDict

__all__ = ["DataCache", "CacheEntry", "CacheStats"]
__version__ = "1.0.0"

logger = logging.getLogger(__name__)


# ==================== 缓存条目 ====================

@dataclass
class CacheEntry:
    """
    缓存条目

    封装缓存中单个数据项的完整信息，包括键、值、时间戳、
    大小和访问计数等元信息。缓存管理器基于这些信息执行
    LRU淘汰策略和容量管理。

    Attributes:
        key: 缓存键（唯一标识符）
        value: 缓存值（任意Python对象）
        timestamp: 创建时间戳（Unix时间，秒）
        last_access_time: 最后访问时间戳
        access_count: 累计访问次数
        size_bytes: 缓存值占用的字节大小（估算值）
        ttl: 生存时间（秒），0表示永不过期
        metadata: 附加元信息字典
        is_persistent: 是否已持久化到磁盘

    使用示例:
        >>> entry = CacheEntry(key="model_output", value=[1, 2, 3])
        >>> print(f"大小: {entry.size_bytes} 字节, 访问次数: {entry.access_count}")
    """
    key: str
    value: Any = None
    timestamp: float = 0.0
    last_access_time: float = 0.0
    access_count: int = 0
    size_bytes: int = 0
    ttl: int = 0  # 秒，0表示永不过期
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_persistent: bool = False

    def __post_init__(self):
        """初始化后自动设置时间戳和大小"""
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.last_access_time == 0.0:
            self.last_access_time = self.timestamp
        if self.size_bytes == 0 and self.value is not None:
            self.size_bytes = self._estimate_size(self.value)

    @property
    def is_expired(self) -> bool:
        """判断缓存条目是否已过期"""
        if self.ttl <= 0:
            return False
        return (time.time() - self.timestamp) > self.ttl

    @property
    def age_seconds(self) -> float:
        """缓存条目的年龄（秒）"""
        return time.time() - self.timestamp

    @property
    def idle_seconds(self) -> float:
        """缓存条目的空闲时间（秒），即距上次访问的时间"""
        return time.time() - self.last_access_time

    @property
    def human_size(self) -> str:
        """人类可读的大小"""
        size = self.size_bytes
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size / 1024 / 1024:.2f} MB"
        else:
            return f"{size / 1024 / 1024 / 1024:.2f} GB"

    def touch(self) -> None:
        """更新访问时间和计数（LRU策略的核心操作）"""
        self.last_access_time = time.time()
        self.access_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（不包含value，仅元信息）"""
        return {
            "key": self.key,
            "timestamp": self.timestamp,
            "last_access_time": self.last_access_time,
            "access_count": self.access_count,
            "size_bytes": self.size_bytes,
            "ttl": self.ttl,
            "metadata": self.metadata,
            "is_persistent": self.is_persistent,
        }

    @staticmethod
    def _estimate_size(obj: Any) -> int:
        """
        估算Python对象的内存占用（字节）

        使用递归方式遍历对象结构，累加各部分的估算大小。
        注意: 这是近似值，不反映实际Python内存分配。

        Args:
            obj: 要估算大小的Python对象

        Returns:
            估算的字节大小
        """
        # 基础类型大小
        if obj is None:
            return 0
        elif isinstance(obj, bool):
            return 28
        elif isinstance(obj, int):
            return 28
        elif isinstance(obj, float):
            return 24
        elif isinstance(obj, str):
            return len(obj.encode("utf-8")) + 49
        elif isinstance(obj, bytes):
            return len(obj) + 33
        elif isinstance(obj, (list, tuple)):
            total = 56  # 列表/元组对象头
            for item in obj:
                total += 8 + CacheEntry._estimate_size(item)  # 指针 + 元素
            return total
        elif isinstance(obj, dict):
            total = 232  # 字典对象头
            for k, v in obj.items():
                total += 8 + CacheEntry._estimate_size(k)  # 键
                total += 8 + CacheEntry._estimate_size(v)  # 值
            return total
        elif isinstance(obj, set):
            total = 216
            for item in obj:
                total += 8 + CacheEntry._estimate_size(item)
            return total
        else:
            # 其他类型使用pickle估算
            try:
                return len(pickle.dumps(obj))
            except Exception:
                return 64  # 默认估算


# ==================== 缓存统计 ====================

@dataclass
class CacheStats:
    """
    缓存统计信息

    追踪缓存系统的运行状态和性能指标，用于监控和优化缓存策略。

    Attributes:
        total_hits: 缓存命中次数
        total_misses: 缓存未命中次数
        total_evictions: 缓存驱逐次数
        total_expirations: 缓存过期次数
        total_puts: 缓存写入次数
        total_deletes: 缓存删除次数
        current_entries: 当前缓存条目数
        current_memory_bytes: 当前内存占用（字节）
        max_memory_bytes: 最大内存限制（字节）
        peak_memory_bytes: 峰值内存占用（字节）
        start_time: 统计开始时间戳

    使用示例:
        >>> stats = CacheStats()
        >>> stats.record_hit()
        >>> stats.record_miss()
        >>> print(f"命中率: {stats.hit_rate:.1%}")
    """
    total_hits: int = 0
    total_misses: int = 0
    total_evictions: int = 0
    total_expirations: int = 0
    total_puts: int = 0
    total_deletes: int = 0
    current_entries: int = 0
    current_memory_bytes: int = 0
    max_memory_bytes: int = 0
    peak_memory_bytes: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def total_requests(self) -> int:
        """总请求数（命中 + 未命中）"""
        return self.total_hits + self.total_misses

    @property
    def hit_rate(self) -> float:
        """缓存命中率（0.0 ~ 1.0）"""
        if self.total_requests == 0:
            return 0.0
        return self.total_hits / self.total_requests

    @property
    def miss_rate(self) -> float:
        """缓存未命中率"""
        return 1.0 - self.hit_rate

    @property
    def memory_usage_mb(self) -> float:
        """当前内存使用量（MB）"""
        return self.current_memory_bytes / (1024 * 1024)

    @property
    def peak_memory_mb(self) -> float:
        """峰值内存使用量（MB）"""
        return self.peak_memory_bytes / (1024 * 1024)

    @property
    def max_memory_mb(self) -> float:
        """最大内存限制（MB）"""
        return self.max_memory_bytes / (1024 * 1024)

    @property
    def memory_utilization(self) -> float:
        """内存利用率（0.0 ~ 1.0）"""
        if self.max_memory_bytes <= 0:
            return 0.0
        return min(1.0, self.current_memory_bytes / self.max_memory_bytes)

    @property
    def uptime_seconds(self) -> float:
        """缓存运行时间（秒）"""
        return time.time() - self.start_time

    @property
    def requests_per_second(self) -> float:
        """平均每秒请求数"""
        uptime = self.uptime_seconds
        if uptime <= 0:
            return 0.0
        return self.total_requests / uptime

    def record_hit(self) -> None:
        """记录一次缓存命中"""
        self.total_hits += 1

    def record_miss(self) -> None:
        """记录一次缓存未命中"""
        self.total_misses += 1

    def record_eviction(self) -> None:
        """记录一次缓存驱逐"""
        self.total_evictions += 1

    def record_expiration(self) -> None:
        """记录一次缓存过期"""
        self.total_expirations += 1

    def record_put(self, size_bytes: int = 0) -> None:
        """记录一次缓存写入"""
        self.total_puts += 1
        self.current_entries += 1
        self.current_memory_bytes += size_bytes
        if self.current_memory_bytes > self.peak_memory_bytes:
            self.peak_memory_bytes = self.current_memory_bytes

    def record_delete(self, size_bytes: int = 0) -> None:
        """记录一次缓存删除"""
        self.total_deletes += 1
        self.current_entries = max(0, self.current_entries - 1)
        self.current_memory_bytes = max(0, self.current_memory_bytes - size_bytes)

    def update_memory(self, current_bytes: int) -> None:
        """更新当前内存使用量"""
        self.current_memory_bytes = current_bytes
        if current_bytes > self.peak_memory_bytes:
            self.peak_memory_bytes = current_bytes

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "total_requests": self.total_requests,
            "hit_rate": round(self.hit_rate, 4),
            "miss_rate": round(self.miss_rate, 4),
            "total_evictions": self.total_evictions,
            "total_expirations": self.total_expirations,
            "total_puts": self.total_puts,
            "total_deletes": self.total_deletes,
            "current_entries": self.current_entries,
            "memory_usage_mb": round(self.memory_usage_mb, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "max_memory_mb": round(self.max_memory_mb, 2),
            "memory_utilization": round(self.memory_utilization, 4),
            "uptime_seconds": round(self.uptime_seconds, 1),
            "requests_per_second": round(self.requests_per_second, 2),
        }

    def reset(self) -> None:
        """重置所有统计"""
        self.total_hits = 0
        self.total_misses = 0
        self.total_evictions = 0
        self.total_expirations = 0
        self.total_puts = 0
        self.total_deletes = 0
        self.peak_memory_bytes = 0
        self.start_time = time.time()

    def __repr__(self) -> str:
        return (
            f"CacheStats("
            f"entries={self.current_entries}, "
            f"hit_rate={self.hit_rate:.1%}, "
            f"memory={self.memory_usage_mb:.1f}MB"
            f")"
        )


# ==================== LRU缓存管理器 ====================

class DataCache:
    """
    LRU缓存管理器

    提供企业级的缓存管理功能，核心特性包括:
    - LRU (Least Recently Used) 淘汰策略
    - 可配置的容量限制（条目数和内存大小）
    - TTL (Time To Live) 过期机制
    - 磁盘持久化（自动将缓存数据写入磁盘）
    - 线程安全（支持多线程并发访问）
    - 缓存统计和监控

    使用示例:
        >>> # 创建缓存
        >>> cache = DataCache(
        ...     max_size=1000,
        ...     max_memory_mb=512,
        ...     persistent_dir="./cache",
        ... )
        >>> # 写入缓存
        >>> cache.put("user_123", {"name": "Alice", "age": 30})
        >>> cache.put("model_output", [1, 2, 3, 4, 5], ttl=3600)
        >>> # 读取缓存
        >>> data = cache.get("user_123")
        >>> # 批量操作
        >>> cache.put_many({"k1": "v1", "k2": "v2", "k3": "v3"})
        >>> results = cache.get_many(["k1", "k2", "k3"])
        >>> # 查看统计
        >>> stats = cache.get_stats()
        >>> print(f"命中率: {stats.hit_rate:.1%}")
    """

    def __init__(
        self,
        max_size: int = 10000,
        max_memory_mb: float = 512.0,
        default_ttl: int = 0,
        persistent_dir: Optional[str] = None,
        auto_persist: bool = False,
        persist_interval: int = 60,
        cleanup_interval: int = 300,
        key_prefix: str = "agi_cache",
    ):
        """
        初始化缓存管理器

        Args:
            max_size: 最大缓存条目数
            max_memory_mb: 最大内存使用量（MB）
            default_ttl: 默认生存时间（秒），0表示永不过期
            persistent_dir: 磁盘持久化目录（为None时不启用持久化）
            auto_persist: 是否自动持久化到磁盘
            persist_interval: 自动持久化间隔（秒）
            cleanup_interval: 自动清理间隔（秒）
            key_prefix: 缓存键前缀
        """
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.default_ttl = default_ttl
        self.persistent_dir = Path(persistent_dir) if persistent_dir else None
        self.auto_persist = auto_persist
        self.persist_interval = persist_interval
        self.cleanup_interval = cleanup_interval
        self.key_prefix = key_prefix

        # LRU缓存存储（使用OrderedDict实现O(1)的LRU操作）
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # 线程锁
        self._lock = threading.RLock()

        # 统计
        self._stats = CacheStats(
            max_memory_bytes=self.max_memory_bytes,
        )

        # 持久化目录
        if self.persistent_dir:
            self.persistent_dir.mkdir(parents=True, exist_ok=True)
            self._load_persistent()

        # 自动清理线程
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        if self.cleanup_interval > 0:
            self._start_cleanup_thread()

        logger.info(
            f"DataCache 初始化完成: max_size={max_size}, "
            f"max_memory={max_memory_mb}MB, "
            f"persistent={self.persistent_dir is not None}"
        )

    def put(
        self,
        key: str,
        value: Any,
        ttl: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CacheEntry:
        """
        写入缓存

        如果缓存已满，自动驱逐最久未使用的条目。

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 生存时间（秒），0表示使用默认值
            metadata: 附加元信息

        Returns:
            创建的CacheEntry对象
        """
        # 构建完整键
        full_key = self._build_key(key)

        with self._lock:
            # 如果键已存在，先删除旧条目
            if full_key in self._cache:
                old_entry = self._cache.pop(full_key)
                self._stats.record_delete(old_entry.size_bytes)

            # 创建新条目
            entry = CacheEntry(
                key=full_key,
                value=value,
                ttl=ttl if ttl > 0 else self.default_ttl,
                metadata=metadata or {},
            )

            # 检查容量，必要时驱逐
            self._evict_if_needed(entry.size_bytes)

            # 写入缓存
            self._cache[full_key] = entry
            self._cache.move_to_end(full_key)  # 标记为最近使用
            self._stats.record_put(entry.size_bytes)

            # 持久化
            if self.auto_persist and self.persistent_dir:
                self._persist_entry(entry)

        return entry

    def get(
        self,
        key: str,
        default: Any = None,
        update_access: bool = True,
    ) -> Any:
        """
        读取缓存

        Args:
            key: 缓存键
            default: 键不存在时的默认返回值
            update_access: 是否更新访问时间（LRU）

        Returns:
            缓存值，不存在则返回default
        """
        full_key = self._build_key(key)

        with self._lock:
            if full_key not in self._cache:
                self._stats.record_miss()

                # 尝试从磁盘加载
                if self.persistent_dir:
                    entry = self._load_entry_from_disk(full_key)
                    if entry is not None and not entry.is_expired:
                        self._cache[full_key] = entry
                        self._cache.move_to_end(full_key)
                        self._stats.record_put(entry.size_bytes)
                        self._stats.record_hit()
                        return entry.value

                return default

            entry = self._cache[full_key]

            # 检查过期
            if entry.is_expired:
                self._remove_entry(full_key)
                self._stats.record_expiration()
                self._stats.record_miss()
                return default

            # 更新访问信息
            if update_access:
                entry.touch()
                self._cache.move_to_end(full_key)

            self._stats.record_hit()
            return entry.value

    def get_entry(self, key: str) -> Optional[CacheEntry]:
        """
        获取缓存条目（包含完整元信息）

        Args:
            key: 缓存键

        Returns:
            CacheEntry对象，不存在则返回None
        """
        full_key = self._build_key(key)

        with self._lock:
            if full_key not in self._cache:
                return None

            entry = self._cache[full_key]
            if entry.is_expired:
                return None

            entry.touch()
            self._cache.move_to_end(full_key)
            return entry

    def delete(self, key: str) -> bool:
        """
        删除缓存条目

        Args:
            key: 缓存键

        Returns:
            是否成功删除
        """
        full_key = self._build_key(key)

        with self._lock:
            if full_key in self._cache:
                self._remove_entry(full_key)
                return True
            return False

    def has(self, key: str) -> bool:
        """
        检查缓存键是否存在（且未过期）

        Args:
            key: 缓存键

        Returns:
            是否存在
        """
        full_key = self._build_key(key)

        with self._lock:
            if full_key not in self._cache:
                return False
            entry = self._cache[full_key]
            return not entry.is_expired

    def put_many(
        self,
        data: Dict[str, Any],
        ttl: int = 0,
    ) -> List[CacheEntry]:
        """
        批量写入缓存

        Args:
            data: 键值对字典
            ttl: 生存时间（秒）

        Returns:
            CacheEntry列表
        """
        entries = []
        for key, value in data.items():
            entry = self.put(key, value, ttl=ttl)
            entries.append(entry)
        return entries

    def get_many(
        self,
        keys: List[str],
        default: Any = None,
    ) -> Dict[str, Any]:
        """
        批量读取缓存

        Args:
            keys: 缓存键列表
            default: 默认值

        Returns:
            键值对字典（仅包含命中的键）
        """
        results = {}
        for key in keys:
            value = self.get(key, default=default)
            if value is not default:
                results[key] = value
        return results

    def clear(self) -> int:
        """
        清空所有缓存

        Returns:
            清除的条目数
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats.current_entries = 0
            self._stats.current_memory_bytes = 0

            # 清除磁盘缓存
            if self.persistent_dir and self.persistent_dir.exists():
                for cache_file in self.persistent_dir.glob("*.cache"):
                    try:
                        cache_file.unlink()
                    except OSError as e:
                        logger.warning(f"删除缓存文件失败: {e}")

            logger.info(f"缓存已清空: {count} 个条目")
            return count

    def cleanup(self) -> int:
        """
        清理过期条目

        Returns:
            清理的条目数
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]

            for key in expired_keys:
                self._remove_entry(key)
                self._stats.record_expiration()

            if expired_keys:
                logger.debug(f"清理过期条目: {len(expired_keys)} 个")

            return len(expired_keys)

    def get_stats(self) -> CacheStats:
        """
        获取缓存统计信息

        Returns:
            CacheStats对象
        """
        with self._lock:
            # 更新当前状态
            self._stats.current_entries = len(self._cache)
            total_memory = sum(
                entry.size_bytes for entry in self._cache.values()
            )
            self._stats.update_memory(total_memory)

            # 返回副本
            stats = CacheStats(
                total_hits=self._stats.total_hits,
                total_misses=self._stats.total_misses,
                total_evictions=self._stats.total_evictions,
                total_expirations=self._stats.total_expirations,
                total_puts=self._stats.total_puts,
                total_deletes=self._stats.total_deletes,
                current_entries=self._stats.current_entries,
                current_memory_bytes=self._stats.current_memory_bytes,
                max_memory_bytes=self._stats.max_memory_bytes,
                peak_memory_bytes=self._stats.peak_memory_bytes,
                start_time=self._stats.start_time,
            )
            return stats

    def keys(self) -> List[str]:
        """
        获取所有缓存键

        Returns:
            键列表（按LRU顺序，最近使用的在最后）
        """
        with self._lock:
            return list(self._cache.keys())

    def values(self) -> List[Any]:
        """
        获取所有缓存值

        Returns:
            值列表
        """
        with self._lock:
            return [entry.value for entry in self._cache.values()]

    def items(self) -> List[Tuple[str, Any]]:
        """
        获取所有缓存键值对

        Returns:
            (key, value) 元组列表
        """
        with self._lock:
            return [
                (entry.key, entry.value)
                for entry in self._cache.values()
            ]

    def resize(self, max_size: int, max_memory_mb: float) -> None:
        """
        动态调整缓存容量

        如果新容量小于当前使用量，自动驱逐多余条目。

        Args:
            max_size: 新的最大条目数
            max_memory_mb: 新的最大内存限制（MB）
        """
        with self._lock:
            self.max_size = max_size
            self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
            self._stats.max_memory_bytes = self.max_memory_bytes

            # 驱逐多余条目
            while len(self._cache) > self.max_size:
                self._evict_lru()

            # 驱逐超出内存限制的条目
            total_memory = sum(
                e.size_bytes for e in self._cache.values()
            )
            while total_memory > self.max_memory_bytes and self._cache:
                evicted = self._evict_lru()
                if evicted:
                    total_memory -= evicted.size_bytes
                else:
                    break

        logger.info(
            f"缓存容量调整: max_size={max_size}, "
            f"max_memory={max_memory_mb}MB"
        )

    def persist(self) -> int:
        """
        手动将所有缓存持久化到磁盘

        Returns:
            持久化的条目数
        """
        if not self.persistent_dir:
            logger.warning("未配置持久化目录")
            return 0

        count = 0
        with self._lock:
            for entry in self._cache.values():
                try:
                    self._persist_entry(entry)
                    entry.is_persistent = True
                    count += 1
                except Exception as e:
                    logger.warning(f"持久化失败 [{entry.key}]: {e}")

        logger.info(f"持久化完成: {count} 个条目")
        return count

    def __contains__(self, key: str) -> bool:
        """支持 in 操作符"""
        return self.has(key)

    def __getitem__(self, key: str) -> Any:
        """支持 [] 取值语法"""
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        """支持 [] 赋值语法"""
        self.put(key, value)

    def __delitem__(self, key: str) -> None:
        """支持 del 语法"""
        if not self.delete(key):
            raise KeyError(key)

    def __len__(self) -> int:
        """返回当前缓存条目数"""
        with self._lock:
            return len(self._cache)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"DataCache("
            f"entries={stats.current_entries}, "
            f"hit_rate={stats.hit_rate:.1%}, "
            f"memory={stats.memory_usage_mb:.1f}MB"
            f")"
        )

    # ==================== 内部方法 ====================

    def _build_key(self, key: str) -> str:
        """
        构建完整的缓存键

        Args:
            key: 原始键

        Returns:
            带前缀的完整键
        """
        return f"{self.key_prefix}:{key}"

    def _evict_if_needed(self, new_entry_size: int) -> None:
        """
        检查并在必要时驱逐条目

        驱逐策略:
        1. 先驱逐过期条目
        2. 如果仍超出条目数限制，驱逐最久未使用的条目
        3. 如果仍超出内存限制，继续驱逐最久未使用的条目

        Args:
            new_entry_size: 即将写入的条目大小
        """
        # 驱逐过期条目
        self.cleanup()

        # 检查条目数限制
        while len(self._cache) >= self.max_size:
            self._evict_lru()

        # 检查内存限制
        current_memory = sum(
            e.size_bytes for e in self._cache.values()
        )
        while (current_memory + new_entry_size > self.max_memory_bytes
               and self._cache):
            evicted = self._evict_lru()
            if evicted:
                current_memory -= evicted.size_bytes
            else:
                break

    def _evict_lru(self) -> Optional[CacheEntry]:
        """
        驱逐最久未使用的条目（LRU策略）

        Returns:
            被驱逐的CacheEntry，缓存为空时返回None
        """
        if not self._cache:
            return None

        # OrderedDict的第一个元素就是最久未使用的
        key, entry = self._cache.popitem(last=False)
        self._stats.record_delete(entry.size_bytes)
        self._stats.record_eviction()

        logger.debug(f"LRU驱逐: key={key}, size={entry.human_size}")

        # 删除磁盘文件
        if self.persistent_dir:
            disk_path = self._get_disk_path(key)
            if disk_path.exists():
                try:
                    disk_path.unlink()
                except OSError:
                    pass

        return entry

    def _remove_entry(self, key: str) -> None:
        """
        移除指定缓存条目

        Args:
            key: 完整缓存键
        """
        if key in self._cache:
            entry = self._cache.pop(key)
            self._stats.record_delete(entry.size_bytes)

    def _get_disk_path(self, key: str) -> Path:
        """
        获取缓存条目的磁盘文件路径

        使用键的MD5哈希作为文件名，避免文件名过长和特殊字符问题。

        Args:
            key: 完整缓存键

        Returns:
            磁盘文件路径
        """
        key_hash = hashlib.md5(key.encode("utf-8")).hexdigest()
        return self.persistent_dir / f"{key_hash}.cache"

    def _persist_entry(self, entry: CacheEntry) -> None:
        """
        将缓存条目持久化到磁盘

        使用pickle序列化缓存值，同时保存元信息为JSON。

        Args:
            entry: 缓存条目
        """
        if not self.persistent_dir:
            return

        disk_path = self._get_disk_path(entry.key)

        try:
            # 序列化值
            serialized = pickle.dumps(entry.value, protocol=pickle.HIGHEST_PROTOCOL)

            # 写入文件: [元信息长度(4字节)] [元信息JSON] [序列化值]
            meta = entry.to_dict()
            meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")

            with open(disk_path, "wb") as f:
                f.write(struct.pack("!I", len(meta_bytes)))
                f.write(meta_bytes)
                f.write(serialized)

            entry.is_persistent = True

        except Exception as e:
            logger.warning(f"持久化失败 [{entry.key}]: {e}")

    def _load_entry_from_disk(self, key: str) -> Optional[CacheEntry]:
        """
        从磁盘加载缓存条目

        Args:
            key: 完整缓存键

        Returns:
            CacheEntry对象，加载失败返回None
        """
        if not self.persistent_dir:
            return None

        disk_path = self._get_disk_path(key)

        if not disk_path.exists():
            return None

        try:
            with open(disk_path, "rb") as f:
                # 读取元信息长度
                meta_len_bytes = f.read(4)
                if len(meta_len_bytes) < 4:
                    return None
                meta_len = struct.unpack("!I", meta_len_bytes)[0]

                # 读取元信息
                meta_bytes = f.read(meta_len)
                meta = json.loads(meta_bytes.decode("utf-8"))

                # 读取序列化值
                serialized = f.read()
                value = pickle.loads(serialized)

            # 重建缓存条目
            entry = CacheEntry(
                key=meta["key"],
                value=value,
                timestamp=meta["timestamp"],
                last_access_time=meta["last_access_time"],
                access_count=meta["access_count"],
                size_bytes=meta["size_bytes"],
                ttl=meta["ttl"],
                metadata=meta.get("metadata", {}),
                is_persistent=True,
            )

            return entry

        except Exception as e:
            logger.warning(f"从磁盘加载失败 [{key}]: {e}")
            # 删除损坏的文件
            try:
                disk_path.unlink()
            except OSError:
                pass
            return None

    def _load_persistent(self) -> int:
        """
        从磁盘加载所有持久化的缓存条目

        Returns:
            成功加载的条目数
        """
        if not self.persistent_dir or not self.persistent_dir.exists():
            return 0

        count = 0
        for cache_file in self.persistent_dir.glob("*.cache"):
            try:
                with open(cache_file, "rb") as f:
                    meta_len_bytes = f.read(4)
                    if len(meta_len_bytes) < 4:
                        continue
                    meta_len = struct.unpack("!I", meta_len_bytes)[0]

                    meta_bytes = f.read(meta_len)
                    meta = json.loads(meta_bytes.decode("utf-8"))

                    # 跳过值数据（延迟加载）
                    serialized = f.read()
                    value = pickle.loads(serialized)

                key = meta["key"]
                entry = CacheEntry(
                    key=key,
                    value=value,
                    timestamp=meta["timestamp"],
                    last_access_time=meta["last_access_time"],
                    access_count=meta["access_count"],
                    size_bytes=meta["size_bytes"],
                    ttl=meta["ttl"],
                    metadata=meta.get("metadata", {}),
                    is_persistent=True,
                )

                # 跳过已过期条目
                if not entry.is_expired:
                    self._cache[key] = entry
                    count += 1

            except Exception as e:
                logger.warning(f"加载缓存文件失败 [{cache_file}]: {e}")

        if count > 0:
            logger.info(f"从磁盘加载缓存: {count} 个条目")

        return count

    def _start_cleanup_thread(self) -> None:
        """启动自动清理线程"""
        def cleanup_loop():
            while not self._stop_event.wait(self.cleanup_interval):
                try:
                    cleaned = self.cleanup()
                    if cleaned > 0:
                        logger.debug(f"自动清理: {cleaned} 个过期条目")

                    # 自动持久化
                    if self.auto_persist and self.persistent_dir:
                        self.persist()

                except Exception as e:
                    logger.error(f"自动清理异常: {e}")

        self._cleanup_thread = threading.Thread(
            target=cleanup_loop,
            daemon=True,
            name="DataCache-Cleanup",
        )
        self._cleanup_thread.start()

    def shutdown(self) -> None:
        """
        关闭缓存管理器

        执行: 停止清理线程、持久化数据、释放资源。
        """
        # 停止清理线程
        self._stop_event.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)

        # 最终持久化
        if self.auto_persist and self.persistent_dir:
            self.persist()

        stats = self.get_stats()
        logger.info(
            f"DataCache 关闭: "
            f"总请求={stats.total_requests}, "
            f"命中率={stats.hit_rate:.1%}, "
            f"驱逐={stats.total_evictions}"
        )

    def __enter__(self) -> DataCache:
        """支持上下文管理器协议"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """退出上下文时关闭缓存"""
        self.shutdown()
