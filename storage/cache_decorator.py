"""
缓存装饰器

为Repository添加LRU缓存层，支持TTL过期、写穿透和缓存失效。
"""

import collections
import threading
import time
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Callable, Deque, Dict, List, Optional, TypeVar

from .base import (
    Entity,
    Pagination,
    QueryFilter,
    QueryResult,
    Repository,
    SortOrder,
)

T = TypeVar("T", bound="Entity")


@dataclass
class CacheStats:
    """
    缓存统计信息

    Attributes:
        hits: 缓存命中次数
        misses: 缓存未命中次数
        evictions: 缓存驱逐次数
        invalidations: 缓存失效次数
        size: 当前缓存大小
        max_size: 最大缓存大小
    """
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    invalidations: int = 0
    size: int = 0
    max_size: int = 1000

    @property
    def hit_rate(self) -> float:
        """缓存命中率"""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "invalidations": self.invalidations,
            "size": self.size,
            "max_size": self.max_size,
            "hit_rate": round(self.hit_rate, 4),
        }

    def __repr__(self) -> str:
        return (
            f"CacheStats(hits={self.hits}, misses={self.misses}, "
            f"hit_rate={self.hit_rate:.2%}, size={self.size})"
        )


class _CacheEntry:
    """缓存条目"""

    __slots__ = ("key", "value", "expires_at", "created_at")

    def __init__(self, key: str, value: Any, ttl: Optional[float] = None):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.expires_at = self.created_at + ttl if ttl else None

    @property
    def is_expired(self) -> bool:
        """是否已过期"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @property
    def age(self) -> float:
        """缓存条目年龄"""
        return time.time() - self.created_at


class _LRUCache:
    """LRU缓存实现"""

    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = None):
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache: Dict[str, _CacheEntry] = {}
        self._order: Deque[str] = collections.deque()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None

            if entry.is_expired:
                self._remove_entry(key)
                return None

            # 移到队尾（最近使用）
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)

            return entry.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """放入缓存"""
        with self._lock:
            # 如果已存在，先移除
            if key in self._cache:
                self._remove_entry(key)

            # 检查容量
            while len(self._cache) >= self._max_size:
                self._evict_one()

            effective_ttl = ttl if ttl is not None else self._default_ttl
            entry = _CacheEntry(key, value, effective_ttl)
            self._cache[key] = entry
            self._order.append(key)

    def delete(self, key: str) -> bool:
        """删除缓存条目"""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._order.clear()

    def invalidate_pattern(self, prefix: str) -> int:
        """按前缀失效缓存"""
        with self._lock:
            to_remove = [
                key for key in self._cache
                if key.startswith(prefix)
            ]
            for key in to_remove:
                self._remove_entry(key)
            return len(to_remove)

    def size(self) -> int:
        """当前缓存大小"""
        with self._lock:
            return len(self._cache)

    def _remove_entry(self, key: str) -> None:
        """移除缓存条目"""
        self._cache.pop(key, None)
        try:
            self._order.remove(key)
        except ValueError:
            pass

    def _evict_one(self) -> None:
        """驱逐最久未使用的条目"""
        if self._order:
            oldest_key = self._order.popleft()
            self._cache.pop(oldest_key, None)


class CachedRepository(Repository[T]):
    """
    缓存装饰器

    包装任意Repository实现，添加LRU缓存层。

    Features:
        - read() 缓存：先查缓存，miss时查存储并回填
        - write() 写穿透：同时更新缓存和存储
        - query() 结果缓存
        - LRU驱逐策略
        - TTL过期
        - 手动缓存失效
        - 缓存统计

    Args:
        repository: 被包装的存储实例
        max_size: 最大缓存条目数
        default_ttl: 默认TTL（秒），None表示永不过期
        cache_reads: 是否缓存读取
        cache_queries: 是否缓存查询结果
    """

    def __init__(
        self,
        repository: Repository[T],
        max_size: int = 1000,
        default_ttl: Optional[float] = 300.0,
        cache_reads: bool = True,
        cache_queries: bool = True,
    ):
        self._repo = repository
        self._cache = _LRUCache(max_size=max_size, default_ttl=default_ttl)
        self._stats = CacheStats(max_size=max_size)
        self._stats_lock = threading.Lock()
        self._cache_reads = cache_reads
        self._cache_queries = cache_queries

    def create(self, entity: T) -> T:
        """创建实体（写穿透）"""
        result = self._repo.create(entity)
        # 更新缓存
        cache_key = self._entity_key(result.id)
        self._cache.put(cache_key, result)
        # 失效相关查询缓存
        self._cache.invalidate_pattern("query:")
        return result

    def read(self, id: str) -> Optional[T]:
        """读取实体（先查缓存）"""
        if not self._cache_reads:
            return self._repo.read(id)

        cache_key = self._entity_key(id)
        cached = self._cache.get(cache_key)

        if cached is not None:
            self._increment_hits()
            return cached

        self._increment_misses()
        entity = self._repo.read(id)
        if entity is not None:
            self._cache.put(cache_key, entity)
        return entity

    def update(self, id: str, data: Dict[str, Any]) -> Optional[T]:
        """更新实体（写穿透）"""
        result = self._repo.update(id, data)
        if result is not None:
            cache_key = self._entity_key(id)
            self._cache.put(cache_key, result)
        # 失效查询缓存
        self._cache.invalidate_pattern("query:")
        return result

    def delete(self, id: str) -> bool:
        """删除实体（同时删除缓存）"""
        cache_key = self._entity_key(id)
        self._cache.delete(cache_key)
        self._cache.invalidate_pattern("query:")
        return self._repo.delete(id)

    def query(
        self,
        filters: Optional[List[QueryFilter]] = None,
        sort: Optional[List[SortOrder]] = None,
        pagination: Optional[Pagination] = None,
    ) -> QueryResult:
        """查询实体（结果缓存）"""
        if not self._cache_queries:
            return self._repo.query(filters, sort, pagination)

        cache_key = self._query_key(filters, sort, pagination)
        cached = self._cache.get(cache_key)

        if cached is not None:
            self._increment_hits()
            return cached

        self._increment_misses()
        result = self._repo.query(filters, sort, pagination)
        self._cache.put(cache_key, result)
        return result

    def count(self, filters: Optional[List[QueryFilter]] = None) -> int:
        """计数"""
        return self._repo.count(filters)

    def exists(self, id: str) -> bool:
        """检查存在"""
        return self._repo.exists(id)

    # ============================================================
    # 缓存管理
    # ============================================================

    def invalidate(self, id: str) -> bool:
        """
        使指定实体的缓存失效

        Args:
            id: 实体ID

        Returns:
            是否成功失效
        """
        cache_key = self._entity_key(id)
        result = self._cache.delete(cache_key)
        self._cache.invalidate_pattern("query:")
        if result:
            with self._stats_lock:
                self._stats.invalidations += 1
        return result

    def invalidate_all(self) -> None:
        """使所有缓存失效"""
        self._cache.clear()
        with self._stats_lock:
            self._stats.invalidations += 1

    def get_stats(self) -> CacheStats:
        """获取缓存统计"""
        with self._stats_lock:
            self._stats.size = self._cache.size()
            stats = CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                invalidations=self._stats.invalidations,
                size=self._cache.size(),
                max_size=self._stats.max_size,
            )
            return stats

    def reset_stats(self) -> None:
        """重置统计信息"""
        with self._stats_lock:
            self._stats.hits = 0
            self._stats.misses = 0
            self._stats.evictions = 0
            self._stats.invalidations = 0

    @property
    def underlying_repository(self) -> Repository[T]:
        """获取底层存储实例"""
        return self._repo

    # ============================================================
    # 内部方法
    # ============================================================

    def _entity_key(self, entity_id: str) -> str:
        """生成实体缓存键"""
        return f"entity:{entity_id}"

    def _query_key(
        self,
        filters: Optional[List[QueryFilter]],
        sort: Optional[List[SortOrder]],
        pagination: Optional[Pagination],
    ) -> str:
        """生成查询缓存键"""
        parts = ["query:"]
        if filters:
            filter_strs = []
            for f in filters:
                filter_strs.append(f"{f.field}:{f.operator.value}:{f.value}")
            parts.append("f=" + "&".join(filter_strs))
        if sort:
            sort_strs = [f"{s.field}:{s.direction.value}" for s in sort]
            parts.append("s=" + ",".join(sort_strs))
        if pagination:
            parts.append(f"p={pagination.page}:{pagination.page_size}")
        return "|".join(parts)

    def _increment_hits(self) -> None:
        """增加命中计数"""
        with self._stats_lock:
            self._stats.hits += 1

    def _increment_misses(self) -> None:
        """增加未命中计数"""
        with self._stats_lock:
            self._stats.misses += 1

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"<CachedRepository hit_rate={stats.hit_rate:.2%} size={stats.size}>"
