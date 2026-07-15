"""
消息总线监控模块

提供总线运行时的监控和统计功能，包括吞吐量、延迟和队列深度等指标。
"""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BusStats:
    """
    总线统计数据类

    汇总消息总线的运行时指标。

    Attributes:
        timestamp: 统计时间
        total_published: 总发布消息数
        total_received: 总接收消息数
        total_errors: 总错误数
        throughput_per_second: 当前吞吐量（消息/秒）
        avg_latency_ms: 平均延迟（毫秒）
        p50_latency_ms: P50延迟
        p95_latency_ms: P95延迟
        p99_latency_ms: P99延迟
        queue_depths: 各主题队列深度
        topic_stats: 各主题统计
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_published: int = 0
    total_received: int = 0
    total_errors: int = 0
    throughput_per_second: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    queue_depths: Dict[str, int] = field(default_factory=dict)
    topic_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_published": self.total_published,
            "total_received": self.total_received,
            "total_errors": self.total_errors,
            "throughput_per_second": round(self.throughput_per_second, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 3),
            "p50_latency_ms": round(self.p50_latency_ms, 3),
            "p95_latency_ms": round(self.p95_latency_ms, 3),
            "p99_latency_ms": round(self.p99_latency_ms, 3),
            "queue_depths": self.queue_depths,
            "topic_stats": self.topic_stats,
        }


class BusMonitor:
    """
    总线监控

    跟踪消息总线的运行时指标，包括发布/接收计数、错误率、
    吞吐量和延迟统计。

    Usage:
        monitor = BusMonitor(window_size=1000, window_seconds=60)

        # 记录事件
        monitor.record_publish(topic="events.user", message_id="msg-001")
        monitor.record_receive(topic="events.user", message_id="msg-001", latency_ms=2.5)
        monitor.record_error(topic="events.user", error="处理超时")

        # 获取统计
        stats = monitor.get_bus_stats()
        throughput = monitor.get_throughput()
        latency = monitor.get_latency_stats()
    """

    def __init__(
        self,
        window_size: int = 10000,
        window_seconds: float = 60.0,
    ):
        """
        初始化总线监控

        Args:
            window_size: 滑动窗口大小（最大记录数）
            window_seconds: 统计窗口时间（秒）
        """
        self._window_size = window_size
        self._window_seconds = window_seconds

        self._lock = threading.RLock()

        # 计数器
        self._total_published: int = 0
        self._total_received: int = 0
        self._total_errors: int = 0

        # 按主题统计
        self._topic_published: Dict[str, int] = defaultdict(int)
        self._topic_received: Dict[str, int] = defaultdict(int)
        self._topic_errors: Dict[str, int] = defaultdict(int)

        # 时间序列数据（用于计算吞吐量）
        self._publish_times: Deque[float] = deque(maxlen=window_size)
        self._receive_times: Deque[float] = deque(maxlen=window_size)

        # 延迟数据
        self._latencies: Deque[float] = deque(maxlen=window_size)

        # 队列深度跟踪
        self._queue_depths: Dict[str, int] = defaultdict(int)

        # 错误记录
        self._error_records: Deque[Dict[str, Any]] = deque(maxlen=1000)

    def record_publish(
        self,
        topic: str,
        message_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        记录消息发布事件

        Args:
            topic: 消息主题
            message_id: 消息ID
            metadata: 附带元数据
        """
        now = time.monotonic()

        with self._lock:
            self._total_published += 1
            self._topic_published[topic] += 1
            self._publish_times.append(now)

    def record_receive(
        self,
        topic: str,
        message_id: str = "",
        latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        记录消息接收事件

        Args:
            topic: 消息主题
            message_id: 消息ID
            latency_ms: 处理延迟（毫秒）
            metadata: 附带元数据
        """
        now = time.monotonic()

        with self._lock:
            self._total_received += 1
            self._topic_received[topic] += 1
            self._receive_times.append(now)

            if latency_ms > 0:
                self._latencies.append(latency_ms)

    def record_error(
        self,
        topic: str,
        error: str = "",
        message_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        记录错误事件

        Args:
            topic: 消息主题
            error: 错误描述
            message_id: 消息ID
            metadata: 附带元数据
        """
        with self._lock:
            self._total_errors += 1
            self._topic_errors[topic] += 1
            self._error_records.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "topic": topic,
                "error": error,
                "message_id": message_id,
                "metadata": metadata or {},
            })

    def update_queue_depth(self, topic: str, depth: int) -> None:
        """
        更新主题队列深度

        Args:
            topic: 消息主题
            depth: 当前队列深度
        """
        with self._lock:
            self._queue_depths[topic] = depth

    def get_queue_depth(self, topic: Optional[str] = None) -> Dict[str, int]:
        """
        获取队列深度

        Args:
            topic: 指定主题，None返回所有

        Returns:
            主题到队列深度的映射
        """
        with self._lock:
            if topic:
                return {topic: self._queue_depths.get(topic, 0)}
            return dict(self._queue_depths)

    def get_throughput(self, window_seconds: Optional[float] = None) -> Dict[str, float]:
        """
        获取吞吐量统计

        计算指定时间窗口内的消息吞吐量。

        Args:
            window_seconds: 统计窗口（秒），None使用默认值

        Returns:
            包含发布和接收吞吐量的字典
        """
        window = window_seconds or self._window_seconds
        now = time.monotonic()
        cutoff = now - window

        with self._lock:
            # 统计窗口内的发布数
            publish_count = sum(
                1 for t in self._publish_times if t >= cutoff
            )
            # 统计窗口内的接收数
            receive_count = sum(
                1 for t in self._receive_times if t >= cutoff
            )

        return {
            "publish_per_second": publish_count / window if window > 0 else 0.0,
            "receive_per_second": receive_count / window if window > 0 else 0.0,
            "window_seconds": window,
        }

    def get_latency_stats(self) -> Dict[str, float]:
        """
        获取延迟统计

        计算延迟的平均值、P50、P95和P99。

        Returns:
            延迟统计字典（单位：毫秒）
        """
        with self._lock:
            if not self._latencies:
                return {
                    "avg_ms": 0.0,
                    "min_ms": 0.0,
                    "max_ms": 0.0,
                    "p50_ms": 0.0,
                    "p95_ms": 0.0,
                    "p99_ms": 0.0,
                    "sample_count": 0,
                }

            sorted_latencies = sorted(self._latencies)
            count = len(sorted_latencies)

            avg = sum(sorted_latencies) / count
            min_val = sorted_latencies[0]
            max_val = sorted_latencies[-1]

            def percentile(data: List[float], p: float) -> float:
                """计算百分位数"""
                if not data:
                    return 0.0
                idx = int(len(data) * p / 100)
                idx = min(idx, len(data) - 1)
                return data[idx]

            return {
                "avg_ms": round(avg, 3),
                "min_ms": round(min_val, 3),
                "max_ms": round(max_val, 3),
                "p50_ms": round(percentile(sorted_latencies, 50), 3),
                "p95_ms": round(percentile(sorted_latencies, 95), 3),
                "p99_ms": round(percentile(sorted_latencies, 99), 3),
                "sample_count": count,
            }

    def get_bus_stats(self) -> BusStats:
        """
        获取完整的总线统计

        Returns:
            BusStats实例
        """
        with self._lock:
            throughput = self.get_throughput()
            latency = self.get_latency_stats()

            # 构建主题统计
            topic_stats: Dict[str, Dict[str, Any]] = {}
            all_topics = set(
                list(self._topic_published.keys())
                + list(self._topic_received.keys())
                + list(self._topic_errors.keys())
            )
            for topic in all_topics:
                topic_stats[topic] = {
                    "published": self._topic_published.get(topic, 0),
                    "received": self._topic_received.get(topic, 0),
                    "errors": self._topic_errors.get(topic, 0),
                    "queue_depth": self._queue_depths.get(topic, 0),
                }

            return BusStats(
                total_published=self._total_published,
                total_received=self._total_received,
                total_errors=self._total_errors,
                throughput_per_second=throughput["publish_per_second"],
                avg_latency_ms=latency["avg_ms"],
                p50_latency_ms=latency["p50_ms"],
                p95_latency_ms=latency["p95_ms"],
                p99_latency_ms=latency["p99_ms"],
                queue_depths=dict(self._queue_depths),
                topic_stats=topic_stats,
            )

    def get_error_summary(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近的错误摘要

        Args:
            limit: 最大返回数量

        Returns:
            错误记录列表
        """
        with self._lock:
            errors = list(self._error_records)
            return errors[-limit:]

    def get_topic_stats(self, topic: str) -> Dict[str, Any]:
        """
        获取指定主题的统计

        Args:
            topic: 主题名称

        Returns:
            主题统计字典
        """
        with self._lock:
            published = self._topic_published.get(topic, 0)
            received = self._topic_received.get(topic, 0)
            errors = self._topic_errors.get(topic, 0)

            return {
                "topic": topic,
                "published": published,
                "received": received,
                "errors": errors,
                "error_rate": (
                    errors / published if published > 0 else 0.0
                ),
                "queue_depth": self._queue_depths.get(topic, 0),
            }

    def reset(self) -> None:
        """重置所有统计数据"""
        with self._lock:
            self._total_published = 0
            self._total_received = 0
            self._total_errors = 0
            self._topic_published.clear()
            self._topic_received.clear()
            self._topic_errors.clear()
            self._publish_times.clear()
            self._receive_times.clear()
            self._latencies.clear()
            self._queue_depths.clear()
            self._error_records.clear()
            logger.info("总线监控统计已重置")

    def __repr__(self) -> str:
        return (
            f"BusMonitor("
            f"published={self._total_published}, "
            f"received={self._total_received}, "
            f"errors={self._total_errors})"
        )
