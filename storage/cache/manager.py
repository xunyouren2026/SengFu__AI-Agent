"""
Cache Manager - 缓存管理器

提供统一的缓存管理接口，支持:
- 多级缓存
- 缓存策略
- 统计监控
- 装饰器模式
"""

import time
import functools
import hashlib
import json
from typing import (
    Any,
    Optional,
    Dict,
    List,
    Callable,
    TypeVar,
    Union,
    Awaitable,
)
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio

from .redis_cache import RedisCache, RedisCacheConfig
from .memory_cache import MemoryCache, LRUCache

logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable)


class CachePolicy(Enum):
    """缓存策略"""
    CACHE_ASIDE = "cache_aside"  # 旁路缓存
    READ_THROUGH = "read_through"  # 读穿透
    WRITE_THROUGH = "write_through"  # 写穿透
    WRITE_BEHIND = "write_behind"  # 写回
    REFRESH_AHEAD = "refresh_ahead"  # 预刷新


@dataclass
class CacheStats:
    """缓存统计"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    evictions: int = 0
    
    @property
    def total_requests(self) -> int:
        return self.hits + self.misses
    
    @property
    def hit_rate(self) -> float:
        total = self.total_requests
        return self.hits / total if total > 0 else 0.0


class CacheManager:
    """
    缓存管理器
    
    提供多级缓存和统一管理接口。
    """
    
    def __init__(
        self,
        redis_config: Optional[RedisCacheConfig] = None,
        memory_cache_size: int = 1000,
        default_ttl: int = 3600,
        policy: CachePolicy = CachePolicy.CACHE_ASIDE,
    ):
        self.default_ttl = default_ttl
        self.policy = policy
        
        # L1缓存 - 内存
        self._memory_cache = MemoryCache(
            max_size=memory_cache_size,
            default_ttl=default_ttl,
        )
        
        # L2缓存 - Redis
        self._redis_cache: Optional[RedisCache] = None
        if redis_config:
            self._redis_cache = RedisCache(redis_config)
        
        # 统计
        self._stats = CacheStats()
        
        # 命名空间
        self._namespace = "default"
    
    def set_namespace(self, namespace: str) -> None:
        """设置命名空间"""
        self._namespace = namespace
    
    def _make_key(self, key: str) -> str:
        """构建完整键"""
        return f"{self._namespace}:{key}"
    
    # ==================== 基本操作 ====================
    
    def get(
        self,
        key: str,
        use_memory: bool = True,
        use_redis: bool = True,
    ) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            use_memory: 是否使用内存缓存
            use_redis: 是否使用Redis缓存
        
        Returns:
            缓存值或None
        """
        full_key = self._make_key(key)
        
        # L1 - 内存缓存
        if use_memory:
            value = self._memory_cache.get(full_key)
            if value is not None:
                self._stats.hits += 1
                return value
        
        # L2 - Redis缓存
        if use_redis and self._redis_cache:
            value = self._redis_cache.get(full_key)
            if value is not None:
                self._stats.hits += 1
                # 回填L1
                if use_memory:
                    self._memory_cache.set(full_key, value)
                return value
        
        self._stats.misses += 1
        return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        use_memory: bool = True,
        use_redis: bool = True,
    ) -> bool:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）
            use_memory: 是否使用内存缓存
            use_redis: 是否使用Redis缓存
        
        Returns:
            是否成功
        """
        full_key = self._make_key(key)
        ttl = ttl or self.default_ttl
        
        success = True
        
        # L1 - 内存缓存
        if use_memory:
            if not self._memory_cache.set(full_key, value, ttl):
                success = False
        
        # L2 - Redis缓存
        if use_redis and self._redis_cache:
            if not self._redis_cache.set(full_key, value, ttl):
                success = False
        
        self._stats.sets += 1
        return success
    
    def delete(
        self,
        key: str,
        use_memory: bool = True,
        use_redis: bool = True,
    ) -> bool:
        """
        删除缓存
        
        Args:
            key: 缓存键
            use_memory: 是否从内存缓存删除
            use_redis: 是否从Redis缓存删除
        
        Returns:
            是否成功
        """
        full_key = self._make_key(key)
        
        success = True
        
        if use_memory:
            if not self._memory_cache.delete(full_key):
                success = False
        
        if use_redis and self._redis_cache:
            if not self._redis_cache.delete(full_key):
                success = False
        
        self._stats.deletes += 1
        return success
    
    def exists(
        self,
        key: str,
        use_memory: bool = True,
        use_redis: bool = True,
    ) -> bool:
        """检查键是否存在"""
        full_key = self._make_key(key)
        
        if use_memory and self._memory_cache.exists(full_key):
            return True
        
        if use_redis and self._redis_cache and self._redis_cache.exists(full_key):
            return True
        
        return False
    
    # ==================== 批量操作 ====================
    
    def get_many(
        self,
        keys: List[str],
    ) -> Dict[str, Any]:
        """批量获取"""
        result = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                result[key] = value
        return result
    
    def set_many(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """批量设置"""
        for key, value in mapping.items():
            self.set(key, value, ttl)
        return True
    
    def delete_many(
        self,
        keys: List[str],
    ) -> int:
        """批量删除"""
        count = 0
        for key in keys:
            if self.delete(key):
                count += 1
        return count
    
    # ==================== 高级操作 ====================
    
    def get_or_set(
        self,
        key: str,
        factory: Callable[[], T],
        ttl: Optional[int] = None,
    ) -> T:
        """
        获取或设置
        
        如果缓存不存在，调用factory生成值并缓存。
        """
        value = self.get(key)
        if value is not None:
            return value
        
        value = factory()
        self.set(key, value, ttl)
        return value
    
    async def async_get_or_set(
        self,
        key: str,
        factory: Callable[[], Awaitable[T]],
        ttl: Optional[int] = None,
    ) -> T:
        """
        异步获取或设置
        """
        value = self.get(key)
        if value is not None:
            return value
        
        value = await factory()
        self.set(key, value, ttl)
        return value
    
    def invalidate(self, key: str) -> bool:
        """使缓存失效"""
        return self.delete(key)
    
    def invalidate_pattern(self, pattern: str) -> int:
        """使匹配模式的缓存失效"""
        if self._redis_cache:
            return self._redis_cache.delete_pattern(self._make_key(pattern))
        return 0
    
    def refresh(
        self,
        key: str,
        factory: Callable[[], T],
        ttl: Optional[int] = None,
    ) -> T:
        """刷新缓存"""
        value = factory()
        self.set(key, value, ttl)
        return value
    
    # ==================== 装饰器 ====================
    
    def cached(
        self,
        key: Optional[str] = None,
        key_builder: Optional[Callable[..., str]] = None,
        ttl: Optional[int] = None,
        skip_args: Optional[List[str]] = None,
    ) -> Callable[[F], F]:
        """
        缓存装饰器
        
        Args:
            key: 固定缓存键
            key_builder: 动态构建键的函数
            ttl: 过期时间
            skip_args: 忽略的参数名
        
        Returns:
            装饰后的函数
        """
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # 构建缓存键
                if key:
                    cache_key = key
                elif key_builder:
                    cache_key = key_builder(*args, **kwargs)
                else:
                    # 默认基于函数名和参数构建
                    func_name = f"{func.__module__}.{func.__name__}"
                    
                    # 过滤参数
                    filtered_kwargs = kwargs.copy()
                    if skip_args:
                        for arg in skip_args:
                            filtered_kwargs.pop(arg, None)
                    
                    # 生成哈希
                    key_data = json.dumps({
                        "func": func_name,
                        "args": args,
                        "kwargs": filtered_kwargs,
                    }, sort_keys=True, default=str)
                    key_hash = hashlib.sha256(key_data.encode()).hexdigest()[:16]
                    cache_key = f"{func_name}:{key_hash}"
                
                # 尝试从缓存获取
                result = self.get(cache_key)
                if result is not None:
                    return result
                
                # 调用函数
                result = func(*args, **kwargs)
                
                # 存入缓存
                self.set(cache_key, result, ttl)
                
                return result
            
            return wrapper
        
        return decorator
    
    def async_cached(
        self,
        key: Optional[str] = None,
        key_builder: Optional[Callable[..., str]] = None,
        ttl: Optional[int] = None,
    ) -> Callable[[F], F]:
        """异步缓存装饰器"""
        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # 构建缓存键
                if key:
                    cache_key = key
                elif key_builder:
                    cache_key = key_builder(*args, **kwargs)
                else:
                    func_name = f"{func.__module__}.{func.__name__}"
                    key_data = json.dumps({
                        "func": func_name,
                        "args": args,
                        "kwargs": kwargs,
                    }, sort_keys=True, default=str)
                    key_hash = hashlib.sha256(key_data.encode()).hexdigest()[:16]
                    cache_key = f"{func_name}:{key_hash}"
                
                # 尝试从缓存获取
                result = self.get(cache_key)
                if result is not None:
                    return result
                
                # 调用函数
                result = await func(*args, **kwargs)
                
                # 存入缓存
                self.set(cache_key, result, ttl)
                
                return result
            
            return wrapper
        
        return decorator
    
    # ==================== 统计 ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "sets": self._stats.sets,
            "deletes": self._stats.deletes,
            "errors": self._stats.errors,
            "hit_rate": self._stats.hit_rate,
            "memory_cache": self._memory_cache.get_stats(),
        }
        
        if self._redis_cache:
            stats["redis_cache"] = self._redis_cache.get_stats()
        
        return stats
    
    def reset_stats(self) -> None:
        """重置统计"""
        self._stats = CacheStats()
    
    # ==================== 管理 ====================
    
    def clear_memory_cache(self) -> None:
        """清空内存缓存"""
        self._memory_cache.clear()
    
    def clear_all(self) -> None:
        """清空所有缓存"""
        self._memory_cache.clear()
        
        if self._redis_cache:
            self._redis_cache.flush_namespace(self._namespace)
    
    def close(self) -> None:
        """关闭缓存管理器"""
        if self._redis_cache:
            self._redis_cache.close()
    
    async def async_close(self) -> None:
        """异步关闭"""
        if self._redis_cache:
            await self._redis_cache.async_close()
    
    def __enter__(self) -> "CacheManager":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    async def __aenter__(self) -> "CacheManager":
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.async_close()


# 全局缓存管理器实例
_global_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """获取全局缓存管理器"""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager


def set_cache_manager(manager: CacheManager) -> None:
    """设置全局缓存管理器"""
    global _global_cache_manager
    _global_cache_manager = manager
