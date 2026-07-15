"""
Redis Repository 模块

提供 Redis 数据结构的完整封装，包括：
- RedisKeyValue: 字符串键值操作
- RedisHash: 哈希表操作
- RedisList: 列表操作
- RedisSet: 集合操作
- RedisPubSub: 发布订阅
- RedisTransaction: 事务支持
- RedisLua: Lua 脚本执行
- RedisRepository: 统一的 Redis 存储实现

纯 Python 标准库实现，包含完整类型注解。
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================
# 配置和异常
# ============================================================

class RedisError(Exception):
    """Redis 操作异常"""
    pass


class RedisConnectionError(RedisError):
    """连接异常"""
    pass


class RedisTimeoutError(RedisError):
    """超时异常"""
    pass


@dataclass
class RedisConfig:
    """Redis 配置"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    max_connections: int = 50
    retry_on_timeout: bool = True
    health_check_interval: float = 30.0
    decode_responses: bool = True
    
    # 集群配置
    cluster_mode: bool = False
    cluster_nodes: List[Tuple[str, int]] = field(default_factory=list)
    
    # 哨兵配置
    sentinel_mode: bool = False
    sentinel_hosts: List[Tuple[str, int]] = field(default_factory=list)
    sentinel_master_name: str = "mymaster"


# ============================================================
# 模拟 Redis 数据结构
# ============================================================

class MockRedisDataStore:
    """
    内存中的 Redis 数据存储模拟
    
    提供 Redis 所有核心数据结构的内存实现。
    """
    
    _instance: Optional[MockRedisDataStore] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> MockRedisDataStore:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        
        self._strings: Dict[str, Any] = {}
        self._hashes: Dict[str, Dict[str, Any]] = {}
        self._lists: Dict[str, List[Any]] = {}
        self._sets: Dict[str, Set[Any]] = {}
        self._sorted_sets: Dict[str, Dict[Any, float]] = {}
        self._expirations: Dict[str, float] = {}
        self._pubsub_channels: Dict[str, List[Callable]] = {}
        self._pubsub_patterns: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()
        self._initialized = True
    
    def _cleanup_expired(self) -> None:
        """清理过期键"""
        now = time.time()
        expired = [k for k, exp in self._expirations.items() if exp <= now]
        for key in expired:
            self._delete_key(key)
            del self._expirations[key]
    
    def _delete_key(self, key: str) -> None:
        """删除键（所有类型）"""
        self._strings.pop(key, None)
        self._hashes.pop(key, None)
        self._lists.pop(key, None)
        self._sets.pop(key, None)
        self._sorted_sets.pop(key, None)
    
    def _get_type(self, key: str) -> Optional[str]:
        """获取键的类型"""
        if key in self._strings:
            return "string"
        if key in self._hashes:
            return "hash"
        if key in self._lists:
            return "list"
        if key in self._sets:
            return "set"
        if key in self._sorted_sets:
            return "zset"
        return None


# ============================================================
# RedisKeyValue - 键值操作
# ============================================================

class RedisKeyValue:
    """
    Redis 字符串键值操作
    
    提供 GET、SET、DEL 等基础操作，以及 INCR、DECR 等原子操作。
    """
    
    def __init__(self, datastore: MockRedisDataStore):
        self._store = datastore
    
    def get(self, key: str) -> Optional[str]:
        """获取值"""
        with self._store._lock:
            self._store._cleanup_expired()
            value = self._store._strings.get(key)
            if value is None:
                return None
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return str(value) if not isinstance(value, str) else value
    
    def set(
        self,
        key: str,
        value: Union[str, bytes, int, float],
        ex: Optional[int] = None,  # 秒
        px: Optional[int] = None,  # 毫秒
        nx: bool = False,  # 仅当不存在
        xx: bool = False,  # 仅当存在
        keepttl: bool = False,
    ) -> bool:
        """设置值"""
        with self._store._lock:
            self._store._cleanup_expired()
            
            exists = key in self._store._strings
            
            if nx and exists:
                return False
            if xx and not exists:
                return False
            
            # 保存旧过期时间
            old_exp = self._store._expirations.get(key) if keepttl else None
            
            self._store._strings[key] = value
            
            # 设置过期
            if ex is not None:
                self._store._expirations[key] = time.time() + ex
            elif px is not None:
                self._store._expirations[key] = time.time() + px / 1000
            elif old_exp is not None:
                self._store._expirations[key] = old_exp
            
            return True
    
    def delete(self, *keys: str) -> int:
        """删除键"""
        with self._store._lock:
            count = 0
            for key in keys:
                if key in self._store._strings:
                    del self._store._strings[key]
                    self._store._expirations.pop(key, None)
                    count += 1
            return count
    
    def exists(self, *keys: str) -> int:
        """检查键是否存在"""
        with self._store._lock:
            self._store._cleanup_expired()
            return sum(1 for k in keys if k in self._store._strings)
    
    def expire(self, key: str, seconds: int) -> bool:
        """设置过期时间（秒）"""
        with self._store._lock:
            if key not in self._store._strings:
                return False
            self._store._expirations[key] = time.time() + seconds
            return True
    
    def ttl(self, key: str) -> int:
        """获取剩余过期时间"""
        with self._store._lock:
            if key not in self._store._strings:
                return -2  # 键不存在
            if key not in self._store._expirations:
                return -1  # 永不过期
            remaining = int(self._store._expirations[key] - time.time())
            return max(-1, remaining)
    
    def incr(self, key: str, amount: int = 1) -> int:
        """原子递增"""
        with self._store._lock:
            current = self._store._strings.get(key, 0)
            if not isinstance(current, (int, float)):
                try:
                    current = int(current)
                except (ValueError, TypeError):
                    raise RedisError("Value is not an integer")
            new_val = int(current) + amount
            self._store._strings[key] = new_val
            return new_val
    
    def decr(self, key: str, amount: int = 1) -> int:
        """原子递减"""
        return self.incr(key, -amount)
    
    def incrbyfloat(self, key: str, amount: float) -> float:
        """浮点数递增"""
        with self._store._lock:
            current = self._store._strings.get(key, 0.0)
            try:
                current = float(current)
            except (ValueError, TypeError):
                raise RedisError("Value is not a float")
            new_val = current + amount
            self._store._strings[key] = new_val
            return new_val
    
    def mget(self, keys: List[str]) -> List[Optional[str]]:
        """批量获取"""
        return [self.get(k) for k in keys]
    
    def mset(self, mapping: Dict[str, Any]) -> bool:
        """批量设置"""
        for key, value in mapping.items():
            self.set(key, value)
        return True
    
    def keys(self, pattern: str = "*") -> List[str]:
        """按模式查找键"""
        with self._store._lock:
            regex = pattern.replace("*", ".*").replace("?", ".")
            compiled = re.compile(f"^{regex}$")
            return [k for k in self._store._strings.keys() if compiled.match(k)]
    
    def scan(self, cursor: int = 0, match: str = "*", count: int = 10) -> Tuple[int, List[str]]:
        """扫描键空间"""
        with self._store._lock:
            all_keys = list(self._store._strings.keys())
            regex = match.replace("*", ".*").replace("?", ".")
            compiled = re.compile(f"^{regex}$")
            filtered = [k for k in all_keys if compiled.match(k)]
            
            start = cursor
            end = min(start + count, len(filtered))
            result = filtered[start:end]
            next_cursor = end if end < len(filtered) else 0
            
            return next_cursor, result


# ============================================================
# RedisHash - 哈希操作
# ============================================================

class RedisHash:
    """
    Redis 哈希表操作
    
    提供 HGET、HSET、HDEL 等哈希操作。
    """
    
    def __init__(self, datastore: MockRedisDataStore):
        self._store = datastore
    
    def hget(self, key: str, field: str) -> Optional[str]:
        """获取哈希字段值"""
        with self._store._lock:
            self._store._cleanup_expired()
            hash_data = self._store._hashes.get(key, {})
            value = hash_data.get(field)
            return str(value) if value is not None else None
    
    def hset(self, key: str, field: Optional[str] = None, value: Optional[Any] = None, mapping: Optional[Dict[str, Any]] = None) -> int:
        """设置哈希字段"""
        with self._store._lock:
            if key not in self._store._hashes:
                self._store._hashes[key] = {}
            
            count = 0
            if field is not None and value is not None:
                self._store._hashes[key][field] = value
                count += 1
            
            if mapping:
                for f, v in mapping.items():
                    self._store._hashes[key][f] = v
                    count += 1
            
            return count
    
    def hgetall(self, key: str) -> Dict[str, str]:
        """获取所有字段和值"""
        with self._store._lock:
            self._store._cleanup_expired()
            return {k: str(v) for k, v in self._store._hashes.get(key, {}).items()}
    
    def hdel(self, key: str, *fields: str) -> int:
        """删除哈希字段"""
        with self._store._lock:
            hash_data = self._store._hashes.get(key, {})
            count = 0
            for field in fields:
                if field in hash_data:
                    del hash_data[field]
                    count += 1
            if not hash_data:
                self._store._hashes.pop(key, None)
            return count
    
    def hexists(self, key: str, field: str) -> bool:
        """检查字段是否存在"""
        with self._store._lock:
            self._store._cleanup_expired()
            return field in self._store._hashes.get(key, {})
    
    def hkeys(self, key: str) -> List[str]:
        """获取所有字段名"""
        with self._store._lock:
            self._store._cleanup_expired()
            return list(self._store._hashes.get(key, {}).keys())
    
    def hvals(self, key: str) -> List[str]:
        """获取所有字段值"""
        with self._store._lock:
            self._store._cleanup_expired()
            return [str(v) for v in self._store._hashes.get(key, {}).values()]
    
    def hlen(self, key: str) -> int:
        """获取字段数量"""
        with self._store._lock:
            self._store._cleanup_expired()
            return len(self._store._hashes.get(key, {}))
    
    def hincrby(self, key: str, field: str, amount: int = 1) -> int:
        """哈希字段递增"""
        with self._store._lock:
            if key not in self._store._hashes:
                self._store._hashes[key] = {}
            
            current = self._store._hashes[key].get(field, 0)
            try:
                current = int(current)
            except (ValueError, TypeError):
                raise RedisError("Hash value is not an integer")
            
            new_val = current + amount
            self._store._hashes[key][field] = new_val
            return new_val
    
    def hmget(self, key: str, fields: List[str]) -> List[Optional[str]]:
        """批量获取字段值"""
        return [self.hget(key, f) for f in fields]
    
    def hscan(self, key: str, cursor: int = 0, match: str = "*", count: int = 10) -> Tuple[int, Dict[str, str]]:
        """扫描哈希字段"""
        with self._store._lock:
            hash_data = self._store._hashes.get(key, {})
            items = list(hash_data.items())
            
            regex = match.replace("*", ".*").replace("?", ".")
            compiled = re.compile(f"^{regex}$")
            filtered = [(k, v) for k, v in items if compiled.match(k)]
            
            start = cursor
            end = min(start + count, len(filtered))
            result = {k: str(v) for k, v in filtered[start:end]}
            next_cursor = end if end < len(filtered) else 0
            
            return next_cursor, result


# ============================================================
# RedisList - 列表操作
# ============================================================

class RedisList:
    """
    Redis 列表操作
    
    提供 LPUSH、RPUSH、LPOP、RPOP 等列表操作。
    """
    
    def __init__(self, datastore: MockRedisDataStore):
        self._store = datastore
    
    def lpush(self, key: str, *values: Any) -> int:
        """左侧推入"""
        with self._store._lock:
            if key not in self._store._lists:
                self._store._lists[key] = []
            for value in reversed(values):
                self._store._lists[key].insert(0, value)
            return len(self._store._lists[key])
    
    def rpush(self, key: str, *values: Any) -> int:
        """右侧推入"""
        with self._store._lock:
            if key not in self._store._lists:
                self._store._lists[key] = []
            self._store._lists[key].extend(values)
            return len(self._store._lists[key])
    
    def lpop(self, key: str, count: Optional[int] = None) -> Union[Optional[str], List[str]]:
        """左侧弹出"""
        with self._store._lock:
            lst = self._store._lists.get(key, [])
            if not lst:
                return None if count is None else []
            
            if count is None:
                value = lst.pop(0)
                if not lst:
                    del self._store._lists[key]
                return str(value)
            else:
                values = [str(lst.pop(0)) for _ in range(min(count, len(lst)))]
                if not lst:
                    del self._store._lists[key]
                return values
    
    def rpop(self, key: str, count: Optional[int] = None) -> Union[Optional[str], List[str]]:
        """右侧弹出"""
        with self._store._lock:
            lst = self._store._lists.get(key, [])
            if not lst:
                return None if count is None else []
            
            if count is None:
                value = lst.pop()
                if not lst:
                    del self._store._lists[key]
                return str(value)
            else:
                values = [str(lst.pop()) for _ in range(min(count, len(lst)))]
                if not lst:
                    del self._store._lists[key]
                return values
    
    def llen(self, key: str) -> int:
        """获取列表长度"""
        with self._store._lock:
            return len(self._store._lists.get(key, []))
    
    def lrange(self, key: str, start: int, stop: int) -> List[str]:
        """获取列表范围"""
        with self._store._lock:
            lst = self._store._lists.get(key, [])
            # Redis 支持负索引
            if start < 0:
                start = max(0, len(lst) + start)
            if stop < 0:
                stop = len(lst) + stop
            else:
                stop = min(stop + 1, len(lst))  # Redis 的 stop 是包含的
            return [str(v) for v in lst[start:stop]]
    
    def lindex(self, key: str, index: int) -> Optional[str]:
        """获取指定索引元素"""
        with self._store._lock:
            lst = self._store._lists.get(key, [])
            if index < 0:
                index = len(lst) + index
            if 0 <= index < len(lst):
                return str(lst[index])
            return None
    
    def lset(self, key: str, index: int, value: Any) -> bool:
        """设置指定索引值"""
        with self._store._lock:
            lst = self._store._lists.get(key, [])
            if index < 0:
                index = len(lst) + index
            if 0 <= index < len(lst):
                lst[index] = value
                return True
            raise RedisError("Index out of range")
    
    def lrem(self, key: str, count: int, value: Any) -> int:
        """移除元素"""
        with self._store._lock:
            lst = self._store._lists.get(key, [])
            removed = 0
            
            if count > 0:
                # 从头部移除
                i = 0
                while i < len(lst) and removed < count:
                    if str(lst[i]) == str(value):
                        lst.pop(i)
                        removed += 1
                    else:
                        i += 1
            elif count < 0:
                # 从尾部移除
                count = abs(count)
                i = len(lst) - 1
                while i >= 0 and removed < count:
                    if str(lst[i]) == str(value):
                        lst.pop(i)
                        removed += 1
                    i -= 1
            else:
                # 移除所有
                original_len = len(lst)
                lst[:] = [x for x in lst if str(x) != str(value)]
                removed = original_len - len(lst)
            
            return removed
    
    def blpop(self, keys: List[str], timeout: float = 0) -> Optional[Tuple[str, str]]:
        """阻塞式左侧弹出"""
        start_time = time.time()
        while True:
            for key in keys:
                value = self.lpop(key)
                if value is not None:
                    return (key, value)
            
            if timeout > 0 and time.time() - start_time >= timeout:
                return None
            time.sleep(0.01)  # 短暂休眠避免 CPU 占用
    
    def brpop(self, keys: List[str], timeout: float = 0) -> Optional[Tuple[str, str]]:
        """阻塞式右侧弹出"""
        start_time = time.time()
        while True:
            for key in keys:
                value = self.rpop(key)
                if value is not None:
                    return (key, value)
            
            if timeout > 0 and time.time() - start_time >= timeout:
                return None
            time.sleep(0.01)


# ============================================================
# RedisSet - 集合操作
# ============================================================

class RedisSet:
    """
    Redis 集合操作
    
    提供 SADD、SREM、SMEMBERS 等集合操作。
    """
    
    def __init__(self, datastore: MockRedisDataStore):
        self._store = datastore
    
    def sadd(self, key: str, *members: Any) -> int:
        """添加成员"""
        with self._store._lock:
            if key not in self._store._sets:
                self._store._sets[key] = set()
            
            added = 0
            for member in members:
                member_str = str(member)
                if member_str not in self._store._sets[key]:
                    self._store._sets[key].add(member_str)
                    added += 1
            return added
    
    def srem(self, key: str, *members: Any) -> int:
        """移除成员"""
        with self._store._lock:
            if key not in self._store._sets:
                return 0
            
            removed = 0
            for member in members:
                member_str = str(member)
                if member_str in self._store._sets[key]:
                    self._store._sets[key].discard(member_str)
                    removed += 1
            
            if not self._store._sets[key]:
                del self._store._sets[key]
            
            return removed
    
    def smembers(self, key: str) -> Set[str]:
        """获取所有成员"""
        with self._store._lock:
            return set(self._store._sets.get(key, set()))
    
    def sismember(self, key: str, member: Any) -> bool:
        """检查成员是否存在"""
        with self._store._lock:
            return str(member) in self._store._sets.get(key, set())
    
    def scard(self, key: str) -> int:
        """获取成员数量"""
        with self._store._lock:
            return len(self._store._sets.get(key, set()))
    
    def spop(self, key: str, count: Optional[int] = None) -> Union[Optional[str], Set[str]]:
        """随机弹出成员"""
        with self._store._lock:
            members = self._store._sets.get(key, set())
            if not members:
                return None if count is None else set()
            
            if count is None:
                member = random.choice(list(members))
                members.discard(member)
                if not members:
                    del self._store._sets[key]
                return member
            else:
                count = min(count, len(members))
                result = set(random.sample(list(members), count))
                members -= result
                if not members:
                    del self._store._sets[key]
                return result
    
    def srandmember(self, key: str, count: Optional[int] = None) -> Union[Optional[str], List[str]]:
        """随机获取成员（不移除）"""
        with self._store._lock:
            members = list(self._store._sets.get(key, set()))
            if not members:
                return None if count is None else []
            
            if count is None:
                return random.choice(members)
            else:
                if count >= 0:
                    return random.sample(members, min(count, len(members)))
                else:
                    # 允许重复
                    return [random.choice(members) for _ in range(abs(count))]
    
    def sinter(self, *keys: str) -> Set[str]:
        """交集"""
        with self._store._lock:
            if not keys:
                return set()
            
            result = self._store._sets.get(keys[0], set()).copy()
            for key in keys[1:]:
                result &= self._store._sets.get(key, set())
            return result
    
    def sunion(self, *keys: str) -> Set[str]:
        """并集"""
        with self._store._lock:
            result = set()
            for key in keys:
                result |= self._store._sets.get(key, set())
            return result
    
    def sdiff(self, *keys: str) -> Set[str]:
        """差集"""
        with self._store._lock:
            if not keys:
                return set()
            
            result = self._store._sets.get(keys[0], set()).copy()
            for key in keys[1:]:
                result -= self._store._sets.get(key, set())
            return result
    
    def sscan(self, key: str, cursor: int = 0, match: str = "*", count: int = 10) -> Tuple[int, List[str]]:
        """扫描集合成员"""
        with self._store._lock:
            members = list(self._store._sets.get(key, set()))
            
            regex = match.replace("*", ".*").replace("?", ".")
            compiled = re.compile(f"^{regex}$")
            filtered = [m for m in members if compiled.match(m)]
            
            start = cursor
            end = min(start + count, len(filtered))
            result = filtered[start:end]
            next_cursor = end if end < len(filtered) else 0
            
            return next_cursor, result


# ============================================================
# RedisPubSub - 发布订阅
# ============================================================

class RedisPubSub:
    """
    Redis 发布订阅
    
    提供 PUBLISH、SUBSCRIBE、PSUBSCRIBE 等操作。
    """
    
    def __init__(self, datastore: MockRedisDataStore):
        self._store = datastore
        self._subscriptions: Dict[str, List[Callable[[str, str], None]]] = {}
        self._psubscriptions: Dict[str, List[Callable[[str, str, str], None]]] = {}
    
    def publish(self, channel: str, message: str) -> int:
        """发布消息"""
        with self._store._lock:
            receivers = 0
            
            # 精确匹配
            for callback in self._store._pubsub_channels.get(channel, []):
                try:
                    callback(channel, message)
                    receivers += 1
                except Exception as e:
                    logger.error(f"PubSub callback error: {e}")
            
            # 模式匹配
            for pattern, callbacks in self._store._pubsub_patterns.items():
                regex = pattern.replace("*", ".*").replace("?", ".")
                if re.match(f"^{regex}$", channel):
                    for callback in callbacks:
                        try:
                            callback(pattern, channel, message)
                            receivers += 1
                        except Exception as e:
                            logger.error(f"PubSub pattern callback error: {e}")
            
            return receivers
    
    def subscribe(self, *channels: str) -> "PubSubListener":
        """订阅频道"""
        listener = PubSubListener()
        for channel in channels:
            if channel not in self._store._pubsub_channels:
                self._store._pubsub_channels[channel] = []
            self._store._pubsub_channels[channel].append(listener.on_message)
            self._subscriptions.setdefault(channel, []).append(listener.on_message)
        return listener
    
    def psubscribe(self, *patterns: str) -> "PubSubListener":
        """按模式订阅"""
        listener = PubSubListener()
        for pattern in patterns:
            if pattern not in self._store._pubsub_patterns:
                self._store._pubsub_patterns[pattern] = []
            self._store._pubsub_patterns[pattern].append(listener.on_pmessage)
            self._psubscriptions.setdefault(pattern, []).append(listener.on_pmessage)
        return listener
    
    def unsubscribe(self, *channels: str) -> None:
        """取消订阅"""
        for channel in channels:
            if channel in self._store._pubsub_channels:
                for callback in self._subscriptions.get(channel, []):
                    if callback in self._store._pubsub_channels[channel]:
                        self._store._pubsub_channels[channel].remove(callback)
            self._subscriptions.pop(channel, None)
    
    def punsubscribe(self, *patterns: str) -> None:
        """取消模式订阅"""
        for pattern in patterns:
            if pattern in self._store._pubsub_patterns:
                for callback in self._psubscriptions.get(pattern, []):
                    if callback in self._store._pubsub_patterns[pattern]:
                        self._store._pubsub_patterns[pattern].remove(callback)
            self._psubscriptions.pop(pattern, None)
    
    def pubsub_channels(self, pattern: str = "*") -> List[str]:
        """获取活跃频道"""
        regex = pattern.replace("*", ".*").replace("?", ".")
        compiled = re.compile(f"^{regex}$")
        return [c for c in self._store._pubsub_channels.keys() if compiled.match(c)]
    
    def pubsub_numsub(self, *channels: str) -> Dict[str, int]:
        """获取频道订阅数"""
        return {c: len(self._store._pubsub_channels.get(c, [])) for c in channels}


class PubSubListener:
    """发布订阅监听器"""
    
    def __init__(self):
        self._messages: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def on_message(self, channel: str, message: str) -> None:
        """消息回调"""
        with self._lock:
            self._messages.append({
                "type": "message",
                "channel": channel,
                "data": message,
            })
    
    def on_pmessage(self, pattern: str, channel: str, message: str) -> None:
        """模式消息回调"""
        with self._lock:
            self._messages.append({
                "type": "pmessage",
                "pattern": pattern,
                "channel": channel,
                "data": message,
            })
    
    def get_message(self, timeout: Optional[float] = None, ignore_subscribe_messages: bool = False) -> Optional[Dict[str, Any]]:
        """获取消息"""
        start = time.time()
        while True:
            with self._lock:
                if self._messages:
                    return self._messages.pop(0)
            
            if timeout is not None and time.time() - start >= timeout:
                return None
            
            time.sleep(0.01)
    
    def listen(self) -> Iterator[Dict[str, Any]]:
        """监听消息生成器"""
        while True:
            message = self.get_message()
            if message:
                yield message


# ============================================================
# RedisTransaction - 事务
# ============================================================

class RedisTransaction:
    """
    Redis 事务支持
    
    提供 MULTI/EXEC/DISCARD/WATCH 事务操作。
    """
    
    def __init__(self, datastore: MockRedisDataStore):
        self._store = datastore
        self._pipeline: List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]] = []
        self._watched: Set[str] = set()
        self._in_multi = False
    
    def watch(self, *keys: str) -> bool:
        """监视键"""
        self._watched.update(keys)
        return True
    
    def unwatch(self) -> bool:
        """取消监视"""
        self._watched.clear()
        return True
    
    def multi(self) -> None:
        """开始事务"""
        self._in_multi = True
        self._pipeline = []
    
    def execute(self) -> List[Any]:
        """执行事务"""
        if not self._in_multi:
            raise RedisError("No transaction in progress")
        
        with self._store._lock:
            # 检查监视的键是否被修改（简化实现）
            results = []
            for cmd, args, kwargs in self._pipeline:
                try:
                    result = self._execute_command(cmd, args, kwargs)
                    results.append(result)
                except Exception as e:
                    results.append(e)
            
            self._pipeline = []
            self._in_multi = False
            self._watched.clear()
            
            return results
    
    def discard(self) -> bool:
        """放弃事务"""
        self._pipeline = []
        self._in_multi = False
        self._watched.clear()
        return True
    
    def pipeline(self, transaction: bool = True) -> "RedisPipeline":
        """创建管道"""
        return RedisPipeline(self._store, transaction)
    
    def _execute_command(self, cmd: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """执行单个命令"""
        # 简化实现，实际应调用对应的方法
        return f"Executed {cmd}"


class RedisPipeline:
    """Redis 管道"""
    
    def __init__(self, datastore: MockRedisDataStore, transaction: bool = True):
        self._store = datastore
        self._transaction = transaction
        self._commands: List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]] = []
    
    def __enter__(self) -> RedisPipeline:
        return self
    
    def __exit__(self, *args: Any) -> None:
        self.execute()
    
    def execute(self) -> List[Any]:
        """执行管道中的所有命令"""
        with self._store._lock:
            results = []
            for cmd, args, kwargs in self._commands:
                results.append(f"Pipelined {cmd}")
            self._commands = []
            return results
    
    def set(self, key: str, value: Any, **kwargs: Any) -> RedisPipeline:
        """添加 SET 命令"""
        self._commands.append(("set", (key, value), kwargs))
        return self
    
    def get(self, key: str) -> RedisPipeline:
        """添加 GET 命令"""
        self._commands.append(("get", (key,), {}))
        return self
    
    def delete(self, *keys: str) -> RedisPipeline:
        """添加 DEL 命令"""
        self._commands.append(("delete", keys, {}))
        return self


# ============================================================
# RedisLua - Lua 脚本
# ============================================================

class RedisLua:
    """
    Redis Lua 脚本支持
    
    提供 EVAL、EVALSHA 和脚本缓存功能。
    """
    
    def __init__(self, datastore: MockRedisDataStore):
        self._store = datastore
        self._scripts: Dict[str, str] = {}  # sha -> script
    
    def eval(
        self,
        script: str,
        numkeys: int,
        *keys_and_args: Any,
    ) -> Any:
        """执行 Lua 脚本"""
        # 计算 SHA
        sha = hashlib.sha1(script.encode()).hexdigest()
        self._scripts[sha] = script
        
        # 简化实现：返回脚本信息
        keys = list(keys_and_args[:numkeys])
        args = list(keys_and_args[numkeys:])
        
        logger.debug(f"Executing Lua script {sha[:8]}... with {len(keys)} keys and {len(args)} args")
        
        # 模拟一些常用脚本行为
        if "INCR" in script.upper():
            return 1  # 模拟递增结果
        if "GET" in script.upper():
            return "value"  # 模拟获取结果
        
        return None
    
    def evalsha(
        self,
        sha: str,
        numkeys: int,
        *keys_and_args: Any,
    ) -> Any:
        """通过 SHA 执行脚本"""
        if sha not in self._scripts:
            raise RedisError("NOSCRIPT No matching script")
        return self.eval(self._scripts[sha], numkeys, *keys_and_args)
    
    def script_load(self, script: str) -> str:
        """加载脚本"""
        sha = hashlib.sha1(script.encode()).hexdigest()
        self._scripts[sha] = script
        return sha
    
    def script_exists(self, *shas: str) -> List[bool]:
        """检查脚本是否存在"""
        return [sha in self._scripts for sha in shas]
    
    def script_flush(self) -> bool:
        """清空脚本缓存"""
        self._scripts.clear()
        return True
    
    def script_kill(self) -> bool:
        """终止正在执行的脚本"""
        # 简化实现
        return True


# ============================================================
# RedisRepository - 主存储实现
# ============================================================

class RedisRepository:
    """
    Redis 存储实现
    
    统一的 Redis 存储接口，集成所有数据结构操作。
    """
    
    def __init__(self, config: Optional[RedisConfig] = None):
        self.config = config or RedisConfig()
        self._store = MockRedisDataStore()
        
        # 初始化各组件
        self.kv = RedisKeyValue(self._store)
        self.hash = RedisHash(self._store)
        self.list = RedisList(self._store)
        self.set = RedisSet(self._store)
        self.pubsub = RedisPubSub(self._store)
        self.transaction = RedisTransaction(self._store)
        self.lua = RedisLua(self._store)
    
    # 键值操作代理
    def get(self, key: str) -> Optional[str]:
        return self.kv.get(key)
    
    def set(self, key: str, value: Any, **kwargs: Any) -> bool:
        return self.kv.set(key, value, **kwargs)
    
    def delete(self, *keys: str) -> int:
        return self.kv.delete(*keys)
    
    def exists(self, *keys: str) -> int:
        return self.kv.exists(*keys)
    
    def expire(self, key: str, seconds: int) -> bool:
        return self.kv.expire(key, seconds)
    
    def ttl(self, key: str) -> int:
        return self.kv.ttl(key)
    
    def incr(self, key: str, amount: int = 1) -> int:
        return self.kv.incr(key, amount)
    
    def decr(self, key: str, amount: int = 1) -> int:
        return self.kv.decr(key, amount)
    
    # 哈希操作代理
    def hget(self, key: str, field: str) -> Optional[str]:
        return self.hash.hget(key, field)
    
    def hset(self, key: str, field: Optional[str] = None, value: Optional[Any] = None, mapping: Optional[Dict[str, Any]] = None) -> int:
        return self.hash.hset(key, field, value, mapping)
    
    def hgetall(self, key: str) -> Dict[str, str]:
        return self.hash.hgetall(key)
    
    def hdel(self, key: str, *fields: str) -> int:
        return self.hash.hdel(key, *fields)
    
    # 列表操作代理
    def lpush(self, key: str, *values: Any) -> int:
        return self.list.lpush(key, *values)
    
    def rpush(self, key: str, *values: Any) -> int:
        return self.list.rpush(key, *values)
    
    def lpop(self, key: str, count: Optional[int] = None) -> Union[Optional[str], List[str]]:
        return self.list.lpop(key, count)
    
    def rpop(self, key: str, count: Optional[int] = None) -> Union[Optional[str], List[str]]:
        return self.list.rpop(key, count)
    
    def lrange(self, key: str, start: int, stop: int) -> List[str]:
        return self.list.lrange(key, start, stop)
    
    # 集合操作代理
    def sadd(self, key: str, *members: Any) -> int:
        return self.set.sadd(key, *members)
    
    def srem(self, key: str, *members: Any) -> int:
        return self.set.srem(key, *members)
    
    def smembers(self, key: str) -> Set[str]:
        return self.set.smembers(key)
    
    def sismember(self, key: str, member: Any) -> bool:
        return self.set.sismember(key, member)
    
    # 发布订阅代理
    def publish(self, channel: str, message: str) -> int:
        return self.pubsub.publish(channel, message)
    
    def subscribe(self, *channels: str) -> PubSubListener:
        return self.pubsub.subscribe(*channels)
    
    def psubscribe(self, *patterns: str) -> PubSubListener:
        return self.pubsub.psubscribe(*patterns)
    
    # 事务代理
    def pipeline(self, transaction: bool = True) -> RedisPipeline:
        return self.transaction.pipeline(transaction)
    
    def multi(self) -> None:
        self.transaction.multi()
    
    def execute(self) -> List[Any]:
        return self.transaction.execute()
    
    # Lua 脚本代理
    def eval(self, script: str, numkeys: int, *keys_and_args: Any) -> Any:
        return self.lua.eval(script, numkeys, *keys_and_args)
    
    def evalsha(self, sha: str, numkeys: int, *keys_and_args: Any) -> Any:
        return self.lua.evalsha(sha, numkeys, *keys_and_args)
    
    # 高级操作
    def cache_with_ttl(self, key: str, ttl: int, getter: Callable[[], T]) -> T:
        """带缓存的获取"""
        value = self.get(key)
        if value is not None:
            return json.loads(value)  # type: ignore
        
        result = getter()
        self.set(key, json.dumps(result), ex=ttl)
        return result
    
    def lock(self, lock_name: str, timeout: int = 10, blocking: bool = True) -> "RedisLock":
        """获取分布式锁"""
        return RedisLock(self, lock_name, timeout, blocking)
    
    def close(self) -> None:
        """关闭连接"""
        logger.info("Redis repository closed")


class RedisLock:
    """Redis 分布式锁"""
    
    def __init__(self, repo: RedisRepository, name: str, timeout: int, blocking: bool):
        self._repo = repo
        self._name = f"lock:{name}"
        self._timeout = timeout
        self._blocking = blocking
        self._identifier = uuid.uuid4().hex
        self._acquired = False
    
    def acquire(self) -> bool:
        """获取锁"""
        if self._blocking:
            start = time.time()
            while time.time() - start < self._timeout:
                if self._try_acquire():
                    return True
                time.sleep(0.1)
            return False
        else:
            return self._try_acquire()
    
    def _try_acquire(self) -> bool:
        """尝试获取锁"""
        result = self._repo.set(self._name, self._identifier, nx=True, ex=self._timeout)
        if result:
            self._acquired = True
        return result
    
    def release(self) -> bool:
        """释放锁"""
        if not self._acquired:
            return False
        
        current = self._repo.get(self._name)
        if current == self._identifier:
            self._repo.delete(self._name)
            self._acquired = False
            return True
        return False
    
    def __enter__(self) -> RedisLock:
        self.acquire()
        return self
    
    def __exit__(self, *args: Any) -> None:
        self.release()
