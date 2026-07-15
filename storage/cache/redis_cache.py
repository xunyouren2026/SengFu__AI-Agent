"""
Redis Cache Implementation - Redis缓存实现

提供基于Redis的分布式缓存功能，支持:
- 键值存储
- TTL过期
- 批量操作
- 分布式锁
- 发布订阅
"""

import asyncio
import json
import pickle
import hashlib
import time
from typing import (
    Any,
    Optional,
    Dict,
    List,
    Union,
    Callable,
    TypeVar,
    Generic,
    Awaitable,
)
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

try:
    import redis
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SerializationFormat(Enum):
    """序列化格式"""
    JSON = "json"
    PICKLE = "pickle"
    STRING = "string"


@dataclass
class RedisCacheConfig:
    """Redis缓存配置"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    username: Optional[str] = None
    
    # 连接池配置
    max_connections: int = 50
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    
    # 缓存配置
    default_ttl: int = 3600  # 默认TTL 1小时
    key_prefix: str = "agi:cache:"
    serialization: SerializationFormat = SerializationFormat.JSON
    
    # 压缩配置
    compress_threshold: int = 1024  # 超过1KB压缩
    compression: str = "gzip"  # gzip, lz4, none
    
    # 集群配置
    cluster_mode: bool = False
    cluster_nodes: List[str] = field(default_factory=list)
    
    # 哨兵配置
    sentinel_mode: bool = False
    sentinel_hosts: List[str] = field(default_factory=list)
    sentinel_master: str = "mymaster"
    
    def get_connection_url(self) -> str:
        """获取连接URL"""
        auth = ""
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        elif self.password:
            auth = f":{self.password}@"
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class CacheSerializer:
    """缓存序列化器"""
    
    def __init__(
        self,
        format: SerializationFormat = SerializationFormat.JSON,
        compress_threshold: int = 1024,
        compression: str = "gzip",
    ):
        self.format = format
        self.compress_threshold = compress_threshold
        self.compression = compression
    
    def serialize(self, value: Any) -> bytes:
        """序列化值"""
        # 先序列化
        if self.format == SerializationFormat.JSON:
            data = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
        elif self.format == SerializationFormat.PICKLE:
            data = pickle.dumps(value)
        else:
            data = str(value).encode("utf-8")
        
        # 压缩
        if self.compression != "none" and len(data) > self.compress_threshold:
            data = self._compress(data)
        
        return data
    
    def deserialize(self, data: bytes) -> Any:
        """反序列化值"""
        if not data:
            return None
        
        # 检查是否压缩
        data = self._decompress(data)
        
        # 反序列化
        if self.format == SerializationFormat.JSON:
            return json.loads(data.decode("utf-8"))
        elif self.format == SerializationFormat.PICKLE:
            return pickle.loads(data)
        else:
            return data.decode("utf-8")
    
    def _compress(self, data: bytes) -> bytes:
        """压缩数据"""
        if self.compression == "gzip":
            import gzip
            return gzip.compress(data)
        elif self.compression == "lz4":
            try:
                import lz4.frame
                return lz4.frame.compress(data)
            except ImportError:
                return data
        return data
    
    def _decompress(self, data: bytes) -> bytes:
        """解压数据"""
        # 检测gzip
        if len(data) >= 2 and data[:2] == b'\x1f\x8b':
            import gzip
            return gzip.decompress(data)
        
        # 检测lz4
        if len(data) >= 4 and data[:4] == b'\x04\x22\x4d\x18':
            try:
                import lz4.frame
                return lz4.frame.decompress(data)
            except ImportError:
                pass
        
        return data


class CacheKeyBuilder:
    """缓存键构建器"""
    
    def __init__(self, prefix: str = "agi:cache:"):
        self.prefix = prefix
    
    def build(
        self,
        *parts: str,
        namespace: Optional[str] = None,
        version: Optional[str] = None,
    ) -> str:
        """构建缓存键"""
        key_parts = [self.prefix]
        
        if namespace:
            key_parts.append(namespace)
        
        if version:
            key_parts.append(f"v{version}")
        
        key_parts.extend(str(p) for p in parts)
        
        return ":".join(key_parts)
    
    def build_hash(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """基于参数构建哈希键"""
        data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        hash_value = hashlib.sha256(data.encode()).hexdigest()[:16]
        return f"{self.prefix}hash:{hash_value}"
    
    def parse(self, key: str) -> Dict[str, str]:
        """解析缓存键"""
        parts = key.split(":")
        result = {"full_key": key}
        
        if len(parts) >= 2:
            result["prefix"] = parts[0]
        
        if len(parts) >= 3:
            result["namespace"] = parts[1]
        
        return result


class RedisCache:
    """Redis缓存实现"""
    
    def __init__(self, config: Optional[RedisCacheConfig] = None):
        self.config = config or RedisCacheConfig()
        self.serializer = CacheSerializer(
            format=self.config.serialization,
            compress_threshold=self.config.compress_threshold,
            compression=self.config.compression,
        )
        self.key_builder = CacheKeyBuilder(self.config.key_prefix)
        
        self._sync_client: Optional[redis.Redis] = None
        self._async_client: Optional[aioredis.Redis] = None
        self._connected = False
    
    def _get_sync_client(self) -> redis.Redis:
        """获取同步客户端"""
        if not REDIS_AVAILABLE:
            raise RuntimeError("Redis library not installed. Run: pip install redis")
        
        if self._sync_client is None:
            self._sync_client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                username=self.config.username,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                decode_responses=False,
            )
            self._connected = True
        
        return self._sync_client
    
    async def _get_async_client(self) -> aioredis.Redis:
        """获取异步客户端"""
        if not REDIS_AVAILABLE:
            raise RuntimeError("Redis library not installed. Run: pip install redis")
        
        if self._async_client is None:
            self._async_client = aioredis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                username=self.config.username,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                decode_responses=False,
            )
            self._connected = True
        
        return self._async_client
    
    # ==================== 同步方法 ====================
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        try:
            client = self._get_sync_client()
            full_key = self.key_builder.build(key)
            data = client.get(full_key)
            
            if data is None:
                return None
            
            return self.serializer.deserialize(data)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """设置缓存值"""
        try:
            client = self._get_sync_client()
            full_key = self.key_builder.build(key)
            data = self.serializer.serialize(value)
            
            ttl = ttl or self.config.default_ttl
            
            result = client.set(
                full_key,
                data,
                ex=ttl,
                nx=nx,
                xx=xx,
            )
            return result is not None or result is True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """删除缓存"""
        try:
            client = self._get_sync_client()
            full_key = self.key_builder.build(key)
            return client.delete(full_key) > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        try:
            client = self._get_sync_client()
            full_key = self.key_builder.build(key)
            return client.exists(full_key) > 0
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    def expire(self, key: str, ttl: int) -> bool:
        """设置过期时间"""
        try:
            client = self._get_sync_client()
            full_key = self.key_builder.build(key)
            return client.expire(full_key, ttl)
        except Exception as e:
            logger.error(f"Redis expire error: {e}")
            return False
    
    def ttl(self, key: str) -> int:
        """获取剩余TTL"""
        try:
            client = self._get_sync_client()
            full_key = self.key_builder.build(key)
            return client.ttl(full_key)
        except Exception as e:
            logger.error(f"Redis ttl error: {e}")
            return -1
    
    def incr(self, key: str, amount: int = 1) -> int:
        """递增"""
        try:
            client = self._get_sync_client()
            full_key = self.key_builder.build(key)
            return client.incrby(full_key, amount)
        except Exception as e:
            logger.error(f"Redis incr error: {e}")
            return 0
    
    def decr(self, key: str, amount: int = 1) -> int:
        """递减"""
        return self.incr(key, -amount)
    
    # ==================== 批量操作 ====================
    
    def mget(self, keys: List[str]) -> Dict[str, Any]:
        """批量获取"""
        try:
            client = self._get_sync_client()
            full_keys = [self.key_builder.build(k) for k in keys]
            values = client.mget(full_keys)
            
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = self.serializer.deserialize(value)
            
            return result
        except Exception as e:
            logger.error(f"Redis mget error: {e}")
            return {}
    
    def mset(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """批量设置"""
        try:
            client = self._get_sync_client()
            pipe = client.pipeline()
            
            for key, value in mapping.items():
                full_key = self.key_builder.build(key)
                data = self.serializer.serialize(value)
                pipe.set(full_key, data, ex=ttl or self.config.default_ttl)
            
            pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Redis mset error: {e}")
            return False
    
    def delete_many(self, keys: List[str]) -> int:
        """批量删除"""
        try:
            client = self._get_sync_client()
            full_keys = [self.key_builder.build(k) for k in keys]
            return client.delete(*full_keys)
        except Exception as e:
            logger.error(f"Redis delete_many error: {e}")
            return 0
    
    def delete_pattern(self, pattern: str) -> int:
        """删除匹配模式的所有键"""
        try:
            client = self._get_sync_client()
            full_pattern = self.key_builder.build(pattern)
            keys = list(client.scan_iter(match=full_pattern))
            
            if keys:
                return client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Redis delete_pattern error: {e}")
            return 0
    
    # ==================== 异步方法 ====================
    
    async def async_get(self, key: str) -> Optional[Any]:
        """异步获取缓存值"""
        try:
            client = await self._get_async_client()
            full_key = self.key_builder.build(key)
            data = await client.get(full_key)
            
            if data is None:
                return None
            
            return self.serializer.deserialize(data)
        except Exception as e:
            logger.error(f"Redis async_get error: {e}")
            return None
    
    async def async_set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """异步设置缓存值"""
        try:
            client = await self._get_async_client()
            full_key = self.key_builder.build(key)
            data = self.serializer.serialize(value)
            
            ttl = ttl or self.config.default_ttl
            await client.set(full_key, data, ex=ttl)
            return True
        except Exception as e:
            logger.error(f"Redis async_set error: {e}")
            return False
    
    async def async_delete(self, key: str) -> bool:
        """异步删除缓存"""
        try:
            client = await self._get_async_client()
            full_key = self.key_builder.build(key)
            return await client.delete(full_key) > 0
        except Exception as e:
            logger.error(f"Redis async_delete error: {e}")
            return False
    
    async def async_mget(self, keys: List[str]) -> Dict[str, Any]:
        """异步批量获取"""
        try:
            client = await self._get_async_client()
            full_keys = [self.key_builder.build(k) for k in keys]
            values = await client.mget(full_keys)
            
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = self.serializer.deserialize(value)
            
            return result
        except Exception as e:
            logger.error(f"Redis async_mget error: {e}")
            return {}
    
    async def async_mset(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """异步批量设置"""
        try:
            client = await self._get_async_client()
            pipe = client.pipeline()
            
            for key, value in mapping.items():
                full_key = self.key_builder.build(key)
                data = self.serializer.serialize(value)
                pipe.set(full_key, data, ex=ttl or self.config.default_ttl)
            
            await pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Redis async_mset error: {e}")
            return False
    
    # ==================== 分布式锁 ====================
    
    def acquire_lock(
        self,
        lock_name: str,
        timeout: int = 10,
        blocking: bool = True,
        blocking_timeout: Optional[float] = None,
    ) -> bool:
        """获取分布式锁"""
        try:
            client = self._get_sync_client()
            lock_key = self.key_builder.build("lock", lock_name)
            
            start_time = time.time()
            
            while True:
                acquired = client.set(lock_key, "1", nx=True, ex=timeout)
                
                if acquired:
                    return True
                
                if not blocking:
                    return False
                
                if blocking_timeout is not None:
                    if time.time() - start_time > blocking_timeout:
                        return False
                
                time.sleep(0.1)
        except Exception as e:
            logger.error(f"Redis acquire_lock error: {e}")
            return False
    
    def release_lock(self, lock_name: str) -> bool:
        """释放分布式锁"""
        try:
            client = self._get_sync_client()
            lock_key = self.key_builder.build("lock", lock_name)
            return client.delete(lock_key) > 0
        except Exception as e:
            logger.error(f"Redis release_lock error: {e}")
            return False
    
    # ==================== 统计信息 ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        try:
            client = self._get_sync_client()
            info = client.info()
            
            return {
                "connected": True,
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calc_hit_rate(info),
            }
        except Exception as e:
            logger.error(f"Redis get_stats error: {e}")
            return {"connected": False, "error": str(e)}
    
    def _calc_hit_rate(self, info: Dict) -> float:
        """计算命中率"""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        
        if total == 0:
            return 0.0
        
        return hits / total
    
    def count_keys(self, pattern: str = "*") -> int:
        """统计键数量"""
        try:
            client = self._get_sync_client()
            full_pattern = self.key_builder.build(pattern)
            return len(list(client.scan_iter(match=full_pattern)))
        except Exception as e:
            logger.error(f"Redis count_keys error: {e}")
            return 0
    
    def flush_namespace(self, namespace: str) -> int:
        """清空命名空间"""
        return self.delete_pattern(f"{namespace}:*")
    
    # ==================== 上下文管理 ====================
    
    def close(self) -> None:
        """关闭连接"""
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None
        
        if self._async_client:
            asyncio.create_task(self._async_client.close())
            self._async_client = None
        
        self._connected = False
    
    async def async_close(self) -> None:
        """异步关闭连接"""
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None
        
        if self._async_client:
            await self._async_client.close()
            self._async_client = None
        
        self._connected = False
    
    def __enter__(self) -> "RedisCache":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    async def __aenter__(self) -> "RedisCache":
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.async_close()
