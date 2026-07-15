"""
消息去重模块

基于滑动窗口和哈希集合实现消息去重，支持TTL过期。
"""

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class MessageDeduplicator:
    """
    消息去重器

    使用滑动窗口 + 哈希集合实现高效的消息去重。
    基于消息ID计算哈希值，在窗口期内拒绝重复消息。

    Usage:
        dedup = MessageDeduplicator(window_size=10000, ttl_seconds=300)

        # 检查消息是否重复
        if dedup.is_duplicate(message_id="msg-001", payload={"key": "value"}):
            print("重复消息，跳过")

        # 使用消息内容自动生成ID
        if dedup.is_duplicate(payload={"key": "value"}):
            print("重复消息，跳过")

        # 获取统计
        stats = dedup.get_stats()
    """

    def __init__(
        self,
        window_size: int = 10000,
        ttl_seconds: float = 300.0,
        hash_algorithm: str = "sha256",
    ):
        """
        初始化消息去重器

        Args:
            window_size: 滑动窗口大小（最大去重记录数）
            ttl_seconds: 记录存活时间（秒）
            hash_algorithm: 哈希算法（md5/sha256/sha1）
        """
        if window_size <= 0:
            raise ValueError("window_size 必须大于0")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds 必须大于0")
        if hash_algorithm not in ("md5", "sha256", "sha1"):
            raise ValueError(
                f"不支持的哈希算法: {hash_algorithm}，"
                f"可选值: md5, sha256, sha1"
            )

        self._window_size = window_size
        self._ttl_seconds = ttl_seconds
        self._hash_algorithm = hash_algorithm

        # OrderedDict维护插入顺序，用于滑动窗口淘汰
        self._seen: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.RLock()

        # 统计
        self._total_checked: int = 0
        self._total_duplicates: int = 0
        self._total_evicted: int = 0
        self._total_expired: int = 0

    def is_duplicate(
        self,
        message_id: Optional[str] = None,
        payload: Optional[Any] = None,
        topic: Optional[str] = None,
    ) -> bool:
        """
        检查消息是否重复

        优先使用message_id进行去重。如果未提供message_id，
        则基于topic + payload生成哈希值进行去重。

        Args:
            message_id: 消息唯一标识
            payload: 消息负载（用于生成哈希）
            topic: 消息主题（用于生成哈希）

        Returns:
            True表示消息是重复的
        """
        with self._lock:
            self._total_checked += 1

            # 生成去重键
            dedup_key = self._compute_key(message_id, payload, topic)

            # 清理过期记录
            self._cleanup_expired()

            # 检查是否已存在
            if dedup_key in self._seen:
                self._total_duplicates += 1
                # 更新时间戳（续期）
                self._seen[dedup_key] = time.monotonic()
                # 移到末尾（表示最近访问）
                self._seen.move_to_end(dedup_key)
                logger.debug("检测到重复消息: key=%s", dedup_key[:16])
                return True

            # 添加新记录
            self._seen[dedup_key] = time.monotonic()

            # 检查窗口大小，淘汰最旧的记录
            while len(self._seen) > self._window_size:
                self._seen.popitem(last=False)
                self._total_evicted += 1

            return False

    def _compute_key(
        self,
        message_id: Optional[str],
        payload: Optional[Any],
        topic: Optional[str],
    ) -> str:
        """
        计算去重键

        Args:
            message_id: 消息ID
            payload: 消息负载
            topic: 消息主题

        Returns:
            哈希值字符串
        """
        if message_id:
            # 直接使用message_id的哈希
            data = message_id.encode("utf-8")
        elif payload is not None:
            # 基于topic + payload生成哈希
            prefix = (topic or "").encode("utf-8")
            payload_str = self._serialize_for_hash(payload)
            data = prefix + b"||" + payload_str.encode("utf-8")
        else:
            # 无法生成有意义的键
            raise ValueError(
                "必须提供 message_id 或 payload 参数"
            )

        hasher = hashlib.new(self._hash_algorithm)
        hasher.update(data)
        return hasher.hexdigest()

    @staticmethod
    def _serialize_for_hash(value: Any) -> str:
        """
        将值序列化为用于哈希的字符串表示

        对于字典和列表，按键排序以确保一致性。

        Args:
            value: 要序列化的值

        Returns:
            字符串表示
        """
        if isinstance(value, dict):
            sorted_items = sorted(value.items(), key=lambda x: str(x[0]))
            parts = []
            for k, v in sorted_items:
                parts.append(f"{k}={MessageDeduplicator._serialize_for_hash(v)}")
            return "{" + ",".join(parts) + "}"
        elif isinstance(value, (list, tuple)):
            parts = [
                MessageDeduplicator._serialize_for_hash(item)
                for item in value
            ]
            return "[" + ",".join(parts) + "]"
        elif isinstance(value, (str, int, float, bool)):
            return str(value)
        elif value is None:
            return "null"
        else:
            return str(value)

    def _cleanup_expired(self) -> int:
        """
        清理过期的去重记录

        Returns:
            清理的记录数量
        """
        now = time.monotonic()
        expired_keys: list = []

        for key, timestamp in self._seen.items():
            if now - timestamp > self._ttl_seconds:
                expired_keys.append(key)
            else:
                # OrderedDict按插入顺序，遇到未过期的就可以停止
                break

        for key in expired_keys:
            del self._seen[key]
            self._total_expired += 1

        return len(expired_keys)

    def remove(self, message_id: Optional[str] = None, payload: Optional[Any] = None, topic: Optional[str] = None) -> bool:
        """
        手动移除去重记录

        Args:
            message_id: 消息ID
            payload: 消息负载
            topic: 消息主题

        Returns:
            是否成功移除
        """
        with self._lock:
            try:
                dedup_key = self._compute_key(message_id, payload, topic)
                if dedup_key in self._seen:
                    del self._seen[dedup_key]
                    return True
            except ValueError:
                pass
        return False

    def clear(self) -> int:
        """
        清空所有去重记录

        Returns:
            清除的记录数量
        """
        with self._lock:
            count = len(self._seen)
            self._seen.clear()
            return count

    def get_stats(self) -> Dict[str, Any]:
        """
        获取去重统计信息

        Returns:
            统计信息字典
        """
        with self._lock:
            self._cleanup_expired()
            return {
                "window_size": self._window_size,
                "current_size": len(self._seen),
                "utilization": len(self._seen) / self._window_size if self._window_size > 0 else 0,
                "ttl_seconds": self._ttl_seconds,
                "hash_algorithm": self._hash_algorithm,
                "total_checked": self._total_checked,
                "total_duplicates": self._total_duplicates,
                "total_evicted": self._total_evicted,
                "total_expired": self._total_expired,
                "duplicate_rate": (
                    self._total_duplicates / self._total_checked
                    if self._total_checked > 0 else 0.0
                ),
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._seen)

    def __repr__(self) -> str:
        return (
            f"MessageDeduplicator(size={len(self._seen)}/{self._window_size}, "
            f"duplicates={self._total_duplicates})"
        )
