"""
死信队列模块

处理无法正常消费的消息，提供消息暂存、重试和清理功能。
"""

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DLQEntry:
    """
    死信条目

    记录失败消息的完整信息，包括原始消息、错误详情和重试历史。

    Attributes:
        entry_id: 条目唯一标识
        original_message: 原始消息内容
        error: 导致失败的错误信息
        retry_count: 已重试次数
        first_failed_at: 首次失败时间
        last_retry_at: 最近一次重试时间
        metadata: 附带元数据
        topic: 消息主题
        max_retries: 最大重试次数
    """

    entry_id: str = ""
    original_message: Any = None
    error: str = ""
    retry_count: int = 0
    first_failed_at: Optional[datetime] = None
    last_retry_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    topic: str = ""
    max_retries: int = 5

    def __post_init__(self):
        if not self.entry_id:
            self.entry_id = str(uuid.uuid4())
        if self.first_failed_at is None:
            self.first_failed_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "entry_id": self.entry_id,
            "topic": self.topic,
            "original_message": self.original_message,
            "error": self.error,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "first_failed_at": (
                self.first_failed_at.isoformat()
                if self.first_failed_at else None
            ),
            "last_retry_at": (
                self.last_retry_at.isoformat()
                if self.last_retry_at else None
            ),
            "metadata": self.metadata,
        }


class DeadLetterQueue:
    """
    死信队列

    暂存处理失败的消息，支持重试和自动清理过期条目。

    Usage:
        dlq = DeadLetterQueue(max_size=1000, default_max_retries=3)

        # 添加失败消息
        dlq.add(
            message={"key": "value"},
            topic="orders",
            error="处理超时",
            metadata={"order_id": "12345"}
        )

        # 重试所有消息
        results = dlq.retry_all(handler=my_handler)

        # 清理过期消息
        purged = dlq.purge(max_age_seconds=86400)
    """

    def __init__(
        self,
        max_size: int = 10000,
        default_max_retries: int = 5,
        max_age_seconds: float = 86400.0 * 7,  # 默认7天
    ):
        """
        初始化死信队列

        Args:
            max_size: 队列最大容量
            default_max_retries: 默认最大重试次数
            max_age_seconds: 消息最大存活时间（秒）
        """
        if max_size <= 0:
            raise ValueError("max_size 必须大于0")
        if default_max_retries < 0:
            raise ValueError("default_max_retries 不能为负数")
        if max_age_seconds <= 0:
            raise ValueError("max_age_seconds 必须大于0")

        self._queue: Deque[DLQEntry] = deque()
        self._max_size = max_size
        self._default_max_retries = default_max_retries
        self._max_age_seconds = max_age_seconds
        self._lock = threading.RLock()
        self._total_added: int = 0
        self._total_retried: int = 0
        self._total_purged: int = 0
        self._total_successful_retries: int = 0

    def add(
        self,
        message: Any,
        topic: str = "",
        error: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        max_retries: Optional[int] = None,
    ) -> DLQEntry:
        """
        添加失败消息到死信队列

        Args:
            message: 原始消息内容
            topic: 消息主题
            error: 错误描述
            metadata: 附带元数据
            max_retries: 最大重试次数（覆盖默认值）

        Returns:
            创建的DLQEntry实例
        """
        with self._lock:
            # 如果队列已满，移除最旧的条目
            while len(self._queue) >= self._max_size:
                removed = self._queue.popleft()
                self._total_purged += 1
                logger.debug(
                    "死信队列已满，移除最旧条目: %s", removed.entry_id
                )

            entry = DLQEntry(
                original_message=message,
                error=error,
                metadata=metadata or {},
                topic=topic,
                max_retries=max_retries or self._default_max_retries,
            )
            self._queue.append(entry)
            self._total_added += 1

            logger.info(
                "消息已添加到死信队列: entry_id=%s, topic=%s, error=%s",
                entry.entry_id,
                topic,
                error,
            )
            return entry

    def retry_all(
        self,
        handler: Callable[[Any, Dict[str, Any]], bool],
        batch_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        重试所有死信消息

        Args:
            handler: 处理函数，接收 (message, metadata) 参数，
                     返回 True 表示成功
            batch_size: 每批处理的数量

        Returns:
            重试结果列表，每个元素包含 entry_id, success, error 信息
        """
        if not callable(handler):
            raise TypeError("handler 必须是可调用对象")

        results: List[Dict[str, Any]] = []
        entries_to_remove: List[str] = []

        with self._lock:
            entries = list(self._queue)

        for entry in entries:
            if entry.retry_count >= entry.max_retries:
                continue

            entry.retry_count += 1
            entry.last_retry_at = datetime.now(timezone.utc)
            self._total_retried += 1

            try:
                success = handler(entry.original_message, entry.metadata)
                if success:
                    entries_to_remove.append(entry.entry_id)
                    self._total_successful_retries += 1
                    results.append({
                        "entry_id": entry.entry_id,
                        "success": True,
                        "retry_count": entry.retry_count,
                    })
                    logger.info(
                        "死信消息重试成功: entry_id=%s, retry_count=%d",
                        entry.entry_id,
                        entry.retry_count,
                    )
                else:
                    results.append({
                        "entry_id": entry.entry_id,
                        "success": False,
                        "error": "处理函数返回False",
                        "retry_count": entry.retry_count,
                    })
            except Exception as exc:
                entry.error = f"{entry.error} | 重试错误: {str(exc)}"
                results.append({
                    "entry_id": entry.entry_id,
                    "success": False,
                    "error": str(exc),
                    "retry_count": entry.retry_count,
                })
                logger.warning(
                    "死信消息重试失败: entry_id=%s, error=%s",
                    entry.entry_id,
                    exc,
                )

        # 移除成功的条目
        if entries_to_remove:
            with self._lock:
                entry_ids_set = set(entries_to_remove)
                self._queue = deque(
                    e for e in self._queue if e.entry_id not in entry_ids_set
                )

        return results

    def retry_one(
        self,
        entry_id: str,
        handler: Callable[[Any, Dict[str, Any]], bool],
    ) -> Optional[Dict[str, Any]]:
        """
        重试指定的死信消息

        Args:
            entry_id: 条目ID
            handler: 处理函数

        Returns:
            重试结果字典，如果未找到条目返回None
        """
        entry = self.get_entry(entry_id)
        if entry is None:
            return None

        if entry.retry_count >= entry.max_retries:
            return {
                "entry_id": entry_id,
                "success": False,
                "error": "已达到最大重试次数",
                "retry_count": entry.retry_count,
            }

        entry.retry_count += 1
        entry.last_retry_at = datetime.now(timezone.utc)
        self._total_retried += 1

        try:
            success = handler(entry.original_message, entry.metadata)
            if success:
                with self._lock:
                    self._queue = deque(
                        e for e in self._queue if e.entry_id != entry_id
                    )
                self._total_successful_retries += 1
                return {
                    "entry_id": entry_id,
                    "success": True,
                    "retry_count": entry.retry_count,
                }
            else:
                return {
                    "entry_id": entry_id,
                    "success": False,
                    "error": "处理函数返回False",
                    "retry_count": entry.retry_count,
                }
        except Exception as exc:
            entry.error = f"{entry.error} | 重试错误: {str(exc)}"
            return {
                "entry_id": entry_id,
                "success": False,
                "error": str(exc),
                "retry_count": entry.retry_count,
            }

    def purge(
        self,
        max_age_seconds: Optional[float] = None,
        max_retries_exceeded: bool = True,
    ) -> int:
        """
        清理过期或超过重试次数的消息

        Args:
            max_age_seconds: 最大存活时间（秒），None使用默认值
            max_retries_exceeded: 是否清理超过最大重试次数的条目

        Returns:
            清理的条目数量
        """
        max_age = max_age_seconds if max_age_seconds is not None else self._max_age_seconds
        now = datetime.now(timezone.utc)
        purged_count = 0

        with self._lock:
            new_queue: Deque[DLQEntry] = deque()
            for entry in self._queue:
                should_remove = False

                # 检查是否过期
                if entry.first_failed_at is not None:
                    age = (now - entry.first_failed_at).total_seconds()
                    if age > max_age:
                        should_remove = True

                # 检查是否超过最大重试次数
                if not should_remove and max_retries_exceeded:
                    if entry.retry_count >= entry.max_retries:
                        should_remove = True

                if should_remove:
                    purged_count += 1
                else:
                    new_queue.append(entry)

            self._queue = new_queue
            self._total_purged += purged_count

        if purged_count > 0:
            logger.info("死信队列清理完成，移除 %d 条过期消息", purged_count)

        return purged_count

    def get_entry(self, entry_id: str) -> Optional[DLQEntry]:
        """
        根据ID获取死信条目

        Args:
            entry_id: 条目ID

        Returns:
            DLQEntry实例或None
        """
        with self._lock:
            for entry in self._queue:
                if entry.entry_id == entry_id:
                    return entry
        return None

    def get_stats(self) -> Dict[str, Any]:
        """
        获取死信队列统计信息

        Returns:
            包含详细统计的字典
        """
        with self._lock:
            topic_counts: Dict[str, int] = {}
            exhausted_count = 0

            for entry in self._queue:
                if entry.topic:
                    topic_counts[entry.topic] = (
                        topic_counts.get(entry.topic, 0) + 1
                    )
                if entry.retry_count >= entry.max_retries:
                    exhausted_count += 1

            # 按数量排序
            sorted_topics = sorted(
                topic_counts.items(), key=lambda x: x[1], reverse=True
            )

            return {
                "queue_size": len(self._queue),
                "max_size": self._max_size,
                "utilization": len(self._queue) / self._max_size if self._max_size > 0 else 0,
                "total_added": self._total_added,
                "total_retried": self._total_retried,
                "total_successful_retries": self._total_successful_retries,
                "total_purged": self._total_purged,
                "exhausted_entries": exhausted_count,
                "topic_distribution": dict(sorted_topics),
                "max_age_seconds": self._max_age_seconds,
                "default_max_retries": self._default_max_retries,
            }

    def get_entries(
        self,
        topic: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[DLQEntry]:
        """
        获取死信条目列表

        Args:
            topic: 按主题过滤
            limit: 最大返回数量
            offset: 偏移量

        Returns:
            DLQEntry列表
        """
        with self._lock:
            entries = list(self._queue)

        if topic:
            entries = [e for e in entries if e.topic == topic]

        return entries[offset: offset + limit]

    def clear(self) -> int:
        """
        清空死信队列

        Returns:
            清除的条目数量
        """
        with self._lock:
            count = len(self._queue)
            self._total_purged += count
            self._queue.clear()
            logger.info("死信队列已清空，移除 %d 条消息", count)
            return count

    def size(self) -> int:
        """获取当前队列大小"""
        with self._lock:
            return len(self._queue)

    def __len__(self) -> int:
        return self.size()

    def __repr__(self) -> str:
        return (
            f"DeadLetterQueue(size={self.size()}/{self._max_size}, "
            f"total_added={self._total_added})"
        )
