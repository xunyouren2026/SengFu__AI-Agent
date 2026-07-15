"""
Memory Cache Implementation - 内存缓存实现

提供基于内存的本地缓存功能，支持:
- LRU淘汰策略
- TTL过期
- 最大容量限制
- 线程安全
"""

import time
import threading
from typing import Any, Optional, Dict, List, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from collections import OrderedDict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


@dataclass
class CacheEntry(Generic[V]):
    """缓存条目"""
    value: V
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    hits: int = 0
    size: int = 0  # 字节数估计
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def touch(self) -> None:
        """更新访问"""
        self.hits += 1


class LRUCache(Generic[K, V]):
    """LRU缓存实现"""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory: Optional[int] = None,  # 最大内存字节数
        default_ttl: Optional[int] = None,
    ):
        self.max_size = max_size
        self.max_memory = max_memory
        self.default_ttl = default_ttl
        
        self._cache: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self._lock = threading.RLock()
        self._current_memory = 0
        
        # 统计
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def get(self, key: K) -> Optional[V]:
        """获取缓存值"""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # 检查过期
            if entry.is_expired():
                self._remove_entry(key)
                self._misses += 1
                return None
            
            # 移到末尾（最近使用）
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1
            
            return entry.value
    
    def set(
        self,
        key: K,
        value: V,
        ttl: Optional[int] = None,
    ) -> bool:
        """设置缓存值"""
        with self._lock:
            # 计算大小估计
            size = self._estimate_size(value)
            
            # 检查是否需要淘汰
            self._evict_if_needed(size)
            
            # 计算过期时间
            ttl = ttl or self.default_ttl
            expires_at = time.time() + ttl if ttl else None
            
            # 如果键已存在，先移除
            if key in self._cache:
                self._remove_entry(key)
            
            # 创建条目
            entry = CacheEntry(
                value=value,
                expires_at=expires_at,
                size=size,
            )
            
            # 添加到缓存
            self._cache[key] = entry
            self._current_memory += size
            
            return True
    
    def delete(self, key: K) -> bool:
        """删除缓存"""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def exists(self, key: K) -> bool:
        """检查键是否存在"""
        with self._lock:
            if key not in self._cache:
                return False
            
            entry = self._cache[key]
            if entry.is_expired():
                self._remove_entry(key)
                return False
            
            return True
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0
    
    def _remove_entry(self, key: K) -> None:
        """移除条目"""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._current_memory -= entry.size
    
    def _evict_if_needed(self, new_size: int) -> None:
        """按需淘汰"""
        # 按数量淘汰
        while len(self._cache) >= self.max_size:
            self._evict_one()
        
        # 按内存淘汰
        if self.max_memory:
            while self._current_memory + new_size > self.max_memory:
                if not self._cache:
                    break
                self._evict_one()
    
    def _evict_one(self) -> None:
        """淘汰一个条目"""
        if self._cache:
            key, entry = self._cache.popitem(last=False)
            self._current_memory -= entry.size
            self._evictions += 1
    
    def _estimate_size(self, value: Any) -> int:
        """估计值的大小"""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (int, float, bool)):
                return 8
            elif isinstance(value, (list, tuple, set)):
                return sum(self._estimate_size(v) for v in value)
            elif isinstance(value, dict):
                return sum(
                    self._estimate_size(k) + self._estimate_size(v)
                    for k, v in value.items()
                )
            else:
                return 100  # 默认估计
        except Exception:
            return 100
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "memory": self._current_memory,
                "max_memory": self.max_memory,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": hit_rate,
            }


class TTLCache(Generic[K, V]):
    """TTL缓存实现（基于过期时间）"""
    
    def __init__(
        self,
        default_ttl: int = 3600,
        cleanup_interval: int = 60,
    ):
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        
        self._cache: Dict[K, CacheEntry[V]] = {}
        self._lock = threading.RLock()
        self._last_cleanup = time.time()
        
        # 统计
        self._hits = 0
        self._misses = 0
    
    def get(self, key: K) -> Optional[V]:
        """获取缓存值"""
        with self._lock:
            self._maybe_cleanup()
            
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None
            
            entry.touch()
            self._hits += 1
            return entry.value
    
    def set(
        self,
        key: K,
        value: V,
        ttl: Optional[int] = None,
    ) -> bool:
        """设置缓存值"""
        with self._lock:
            ttl = ttl or self.default_ttl
            expires_at = time.time() + ttl
            
            self._cache[key] = CacheEntry(
                value=value,
                expires_at=expires_at,
            )
            
            return True
    
    def delete(self, key: K) -> bool:
        """删除缓存"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def _maybe_cleanup(self) -> None:
        """可能执行清理"""
        now = time.time()
        if now - self._last_cleanup > self.cleanup_interval:
            self._cleanup()
            self._last_cleanup = now
    
    def _cleanup(self) -> None:
        """清理过期条目"""
        expired_keys = [
            k for k, v in self._cache.items()
            if v.is_expired()
        ]
        
        for key in expired_keys:
            del self._cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }


class MemoryCache(Generic[K, V]):
    """
    内存缓存 - 统一的内存缓存接口
    
    支持多种淘汰策略和TTL过期。
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory: Optional[int] = None,
        default_ttl: Optional[int] = None,
        eviction_policy: str = "lru",  # lru, lfu, fifo
    ):
        self.max_size = max_size
        self.max_memory = max_memory
        self.default_ttl = default_ttl
        self.eviction_policy = eviction_policy
        
        self._lru_cache = LRUCache(
            max_size=max_size,
            max_memory=max_memory,
            default_ttl=default_ttl,
        )
        
        self._ttl_cache = TTLCache(
            default_ttl=default_ttl or 3600,
        )
        
        self._lock = threading.RLock()
    
    def get(self, key: K) -> Optional[V]:
        """获取缓存值"""
        with self._lock:
            if self.eviction_policy == "lru":
                return self._lru_cache.get(key)
            else:
                return self._ttl_cache.get(key)
    
    def set(
        self,
        key: K,
        value: V,
        ttl: Optional[int] = None,
    ) -> bool:
        """设置缓存值"""
        with self._lock:
            if self.eviction_policy == "lru":
                return self._lru_cache.set(key, value, ttl)
            else:
                return self._ttl_cache.set(key, value, ttl)
    
    def delete(self, key: K) -> bool:
        """删除缓存"""
        with self._lock:
            if self.eviction_policy == "lru":
                return self._lru_cache.delete(key)
            else:
                return self._ttl_cache.delete(key)
    
    def exists(self, key: K) -> bool:
        """检查键是否存在"""
        with self._lock:
            if self.eviction_policy == "lru":
                return self._lru_cache.exists(key)
            else:
                return self._ttl_cache.get(key) is not None
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._lru_cache.clear()
            self._ttl_cache._cache.clear()
    
    def get_many(self, keys: List[K]) -> Dict[K, V]:
        """批量获取"""
        result = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                result[key] = value
        return result
    
    def set_many(
        self,
        mapping: Dict[K, V],
        ttl: Optional[int] = None,
    ) -> bool:
        """批量设置"""
        for key, value in mapping.items():
            self.set(key, value, ttl)
        return True
    
    def delete_many(self, keys: List[K]) -> int:
        """批量删除"""
        count = 0
        for key in keys:
            if self.delete(key):
                count += 1
        return count
    
    def get_or_set(
        self,
        key: K,
        factory: Callable[[], V],
        ttl: Optional[int] = None,
    ) -> V:
        """获取或设置"""
        value = self.get(key)
        if value is not None:
            return value
        
        value = factory()
        self.set(key, value, ttl)
        return value
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if self.eviction_policy == "lru":
            return self._lru_cache.get_stats()
        else:
            return self._ttl_cache.get_stats()
    
    def __len__(self) -> int:
        return self.get_stats().get("size", 0)
    
    def __contains__(self, key: K) -> bool:
        return self.exists(key)
    
    def __getitem__(self, key: K) -> V:
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value
    
    def __setitem__(self, key: K, value: V) -> None:
        self.set(key, value)
    
    def __delitem__(self, key: K) -> None:
        if not self.delete(key):
            raise KeyError(key)
