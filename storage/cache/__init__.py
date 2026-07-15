"""
Storage Cache Module - 缓存层

提供Redis缓存、内存缓存、分布式缓存等缓存实现。

主要组件:
- RedisCache: Redis分布式缓存
- MemoryCache: 本地内存缓存
- CacheManager: 缓存管理器
- CacheDecorator: 缓存装饰器
"""

from .redis_cache import (
    RedisCache,
    RedisCacheConfig,
    CacheKeyBuilder,
    CacheSerializer,
)

from .memory_cache import (
    MemoryCache,
    LRUCache,
    TTLCache,
)

from .manager import (
    CacheManager,
    CacheStats,
    CachePolicy,
)

__all__ = [
    # Redis缓存
    "RedisCache",
    "RedisCacheConfig",
    "CacheKeyBuilder",
    "CacheSerializer",
    # 内存缓存
    "MemoryCache",
    "LRUCache",
    "TTLCache",
    # 缓存管理
    "CacheManager",
    "CacheStats",
    "CachePolicy",
]

__version__ = "1.0.0"
