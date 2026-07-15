"""
工具结果缓存模块

提供基于 LRU 淘汰和 TTL 过期的工具结果缓存。
仅使用 Python 标准库。
"""

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# CacheStats - 缓存统计
# ---------------------------------------------------------------------------
@dataclass
class CacheStats:
    """缓存统计信息"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.hits / self.total

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "evictions": self.evictions,
            "size": self.size,
        }


# ---------------------------------------------------------------------------
# CacheEntry - 缓存条目
# ---------------------------------------------------------------------------
@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    result: Any
    created_at: float
    ttl: float
    access_count: int = 0

    @property
    def is_expired(self) -> bool:
        if self.ttl <= 0:
            return False
        return time.time() - self.created_at > self.ttl


# ---------------------------------------------------------------------------
# ToolResultCache - 工具结果缓存
# ---------------------------------------------------------------------------
class ToolResultCache:
    """工具结果缓存

    特性:
    - 基于 OrderedDict 的 LRU 淘汰策略
    - 支持 TTL 过期
    - 线程安全
    - 可配置最大容量
    """

    def __init__(
        self,
        max_size: int = 1024,
        default_ttl: float = 300.0,  # 默认 5 分钟
    ):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = threading.RLock()
        self._stats = CacheStats()

    # ----- 缓存操作 -----

    def get(
        self, tool_name: str, params: dict
    ) -> Optional[Any]:
        """获取缓存结果

        Args:
            tool_name: 工具名称
            params: 参数字典

        Returns:
            缓存的结果，如果未命中或已过期则返回 None
        """
        key = self.compute_key(tool_name, params)

        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats.misses += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._stats.misses += 1
                self._stats.size = len(self._cache)
                return None

            # LRU: 移动到末尾
            self._cache.move_to_end(key)
            entry.access_count += 1
            self._stats.hits += 1
            return entry.result

    def set(
        self,
        tool_name: str,
        params: dict,
        result: Any,
        ttl: Optional[float] = None,
    ) -> None:
        """设置缓存

        Args:
            tool_name: 工具名称
            params: 参数字典
            result: 缓存结果
            ttl: 过期时间（秒），None 使用默认值，0 表示永不过期
        """
        key = self.compute_key(tool_name, params)
        effective_ttl = self._default_ttl if ttl is None else ttl

        with self._lock:
            # 如果已存在，先删除
            if key in self._cache:
                del self._cache[key]

            # 淘汰
            self._evict_if_needed()

            self._cache[key] = CacheEntry(
                key=key,
                result=result,
                created_at=time.time(),
                ttl=effective_ttl,
            )
            self._stats.size = len(self._cache)

    def invalidate(self, tool_name: str) -> int:
        """失效指定工具的所有缓存

        Returns:
            失效的条目数量
        """
        prefix = self._tool_prefix(tool_name)
        count = 0

        with self._lock:
            keys_to_remove = [
                k for k in self._cache.keys() if k.startswith(prefix)
            ]
            for key in keys_to_remove:
                del self._cache[key]
                count += 1
            self._stats.size = len(self._cache)

        return count

    def invalidate_by_key(self, tool_name: str, params: dict) -> bool:
        """失效指定工具和参数的缓存"""
        key = self.compute_key(tool_name, params)
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.size = len(self._cache)
                return True
            return False

    def clear(self) -> int:
        """清空所有缓存

        Returns:
            清空的条目数量
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats.size = 0
            return count

    def cleanup_expired(self) -> int:
        """清理所有过期条目

        Returns:
            清理的条目数量
        """
        count = 0
        with self._lock:
            expired_keys = [
                k for k, v in self._cache.items() if v.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]
                count += 1
            self._stats.size = len(self._cache)
        return count

    # ----- 缓存键计算 -----

    def compute_key(self, tool_name: str, params: dict) -> str:
        """基于参数哈希计算缓存键

        对参数进行排序后 JSON 序列化，再取 MD5 哈希，确保相同参数
        生成相同的键。
        """
        normalized = self._normalize_params(params)
        raw = f"{tool_name}:{json.dumps(normalized, sort_keys=True, default=str)}"
        hash_val = hashlib.md5(raw.encode("utf-8")).hexdigest()
        return f"{self._tool_prefix(tool_name)}{hash_val}"

    @staticmethod
    def _normalize_params(params: dict) -> dict:
        """规范化参数：排序键、处理特殊类型"""
        if not isinstance(params, dict):
            return {"_value": params}

        result = {}
        for k, v in sorted(params.items()):
            if isinstance(v, dict):
                result[k] = ToolResultCache._normalize_params(v)
            elif isinstance(v, (list, tuple)):
                result[k] = [
                    ToolResultCache._normalize_params(i)
                    if isinstance(i, dict) else i
                    for i in v
                ]
            elif isinstance(v, set):
                result[k] = sorted(v)
            else:
                result[k] = v
        return result

    @staticmethod
    def _tool_prefix(tool_name: str) -> str:
        return f"tool:{tool_name}:"

    # ----- 淘汰策略 -----

    def _evict_if_needed(self) -> None:
        """LRU 淘汰：当缓存满时，移除最久未访问的条目"""
        while len(self._cache) >= self._max_size:
            # 移除最前面的（最久未访问的）
            self._cache.popitem(last=False)
            self._stats.evictions += 1

    # ----- 统计 -----

    def get_stats(self) -> CacheStats:
        """获取缓存统计"""
        with self._lock:
            self._stats.size = len(self._cache)
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                size=self._stats.size,
            )

    def reset_stats(self) -> None:
        """重置统计"""
        with self._lock:
            self._stats = CacheStats(size=len(self._cache))

    # ----- 配置 -----

    @property
    def max_size(self) -> int:
        return self._max_size

    @max_size.setter
    def max_size(self, value: int) -> None:
        if value < 1:
            raise ValueError("max_size 必须 >= 1")
        with self._lock:
            self._max_size = value
            self._evict_if_needed()
            self._stats.size = len(self._cache)

    @property
    def default_ttl(self) -> float:
        return self._default_ttl

    @default_ttl.setter
    def default_ttl(self, value: float) -> None:
        self._default_ttl = value
