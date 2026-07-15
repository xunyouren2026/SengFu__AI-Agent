"""
内存消息总线实现

基于内存的消息总线，使用字典和线程实现发布订阅模式。
支持消费者组、消息确认、请求-响应模式和通配符匹配。
"""

import fnmatch
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

from .interface import Message, MessageBus, MessageHandler

logger = logging.getLogger(__name__)


class _Subscription:
    """订阅信息"""

    def __init__(
        self,
        subscription_id: str,
        topic: str,
        handler: MessageHandler,
        group: str = "",
    ):
        self.subscription_id = subscription_id
        self.topic = topic
        self.handler = handler
        self.group = group
        self.created_at = datetime.now(timezone.utc)
        self.message_count = 0


class _PendingReply:
    """待处理回复"""

    def __init__(self, correlation_id: str, event: threading.Event):
        self.correlation_id = correlation_id
        self.event = event
        self.response: Any = None
        self.error: Optional[str] = None
        self.created_at = time.monotonic()


class MemoryBackend(MessageBus):
    """
    内存消息总线实现

    特性:
    - 基于dict + threading的发布订阅
    - 消费者组支持（同组内负载均衡）
    - 消息确认机制
    - 请求-响应模式（使用threading.Event等待回复）
    - topic通配符匹配（支持 * 和 ? 通配符）
    - 消息持久化到内存队列

    Usage:
        bus = MemoryBackend()

        # 发布订阅
        bus.subscribe("events.*", handler)
        bus.publish("events.user_created", {"user_id": 123})

        # 请求响应
        response = bus.request("rpc.get_user", {"user_id": 123}, timeout=5.0)

        # 消费者组
        bus.subscribe("tasks", handler, group="worker-group")
    """

    def __init__(self, max_queue_size: int = 10000):
        """
        初始化内存消息总线

        Args:
            max_queue_size: 每个主题的最大消息队列大小
        """
        self._subscriptions: Dict[str, List[_Subscription]] = defaultdict(list)
        self._consumer_groups: Dict[str, List[_Subscription]] = defaultdict(list)
        self._group_index: Dict[str, int] = defaultdict(int)
        self._message_queue: Dict[str, Deque[Message]] = defaultdict(lambda: deque(maxlen=max_queue_size))
        self._pending_replies: Dict[str, _PendingReply] = {}
        self._reply_topic_prefix = "_reply_"
        self._lock = threading.RLock()
        self._max_queue_size = max_queue_size
        self._closed = False
        self._total_published = 0
        self._total_received = 0
        self._total_errors = 0

        # 订阅回复主题
        self.subscribe(
            f"{self._reply_topic_prefix}*", self._handle_reply
        )

    def publish(
        self,
        topic: str,
        message: Any,
        headers: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        发布消息到指定主题

        将消息投递到所有匹配的订阅者。如果订阅者属于消费者组，
        则在同组内进行负载均衡。

        Args:
            topic: 目标主题
            message: 消息内容
            headers: 消息头部

        Returns:
            消息ID
        """
        if self._closed:
            raise RuntimeError("消息总线已关闭")

        msg = Message(
            topic=topic,
            payload=message,
            headers=headers or {},
        )

        # 持久化到内存队列
        self._message_queue[topic].append(msg)

        with self._lock:
            self._total_published += 1
            matched_subscriptions = self._find_matching_subscriptions(topic)

        if not matched_subscriptions:
            logger.debug("主题 %s 没有匹配的订阅者", topic)
            return msg.id

        # 投递消息到匹配的订阅者
        for sub in matched_subscriptions:
            try:
                sub.handler(msg)
                sub.message_count += 1
                with self._lock:
                    self._total_received += 1
            except Exception as exc:
                with self._lock:
                    self._total_errors += 1
                logger.error(
                    "消息处理错误: subscription=%s, topic=%s, error=%s",
                    sub.subscription_id,
                    topic,
                    exc,
                )

        return msg.id

    def subscribe(
        self,
        topic: str,
        handler: MessageHandler,
        group: str = "",
    ) -> str:
        """
        订阅主题

        支持通配符匹配:
        - * 匹配任意字符（不含分隔符.）
        - ? 匹配单个字符
        - 支持 fnmatch 风格的模式

        Args:
            topic: 订阅主题（支持通配符）
            handler: 消息处理函数
            group: 消费者组名称

        Returns:
            订阅ID
        """
        if self._closed:
            raise RuntimeError("消息总线已关闭")
        if not callable(handler):
            raise TypeError("handler 必须是可调用对象")

        subscription_id = str(uuid.uuid4())
        sub = _Subscription(
            subscription_id=subscription_id,
            topic=topic,
            handler=handler,
            group=group,
        )

        with self._lock:
            self._subscriptions[topic].append(sub)
            if group:
                self._consumer_groups[group].append(sub)

        logger.debug(
            "已订阅: topic=%s, group=%s, subscription_id=%s",
            topic,
            group,
            subscription_id,
        )
        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        取消订阅

        Args:
            subscription_id: 订阅ID

        Returns:
            是否成功取消
        """
        with self._lock:
            for topic, subs in self._subscriptions.items():
                for i, sub in enumerate(subs):
                    if sub.subscription_id == subscription_id:
                        subs.pop(i)
                        # 从消费者组中也移除
                        for group, group_subs in self._consumer_groups.items():
                            self._consumer_groups[group] = [
                                s for s in group_subs
                                if s.subscription_id != subscription_id
                            ]
                        logger.debug(
                            "已取消订阅: subscription_id=%s, topic=%s",
                            subscription_id,
                            topic,
                        )
                        return True
        return False

    def request(
        self,
        topic: str,
        message: Any,
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        请求-响应模式

        发送请求消息并阻塞等待响应。使用唯一的correlation_id
        关联请求和响应，通过threading.Event实现等待。

        Args:
            topic: 目标主题
            message: 请求内容
            timeout: 超时时间（秒）
            headers: 消息头部

        Returns:
            响应内容

        Raises:
            TimeoutError: 等待响应超时
            RuntimeError: 响应包含错误
        """
        if self._closed:
            raise RuntimeError("消息总线已关闭")

        correlation_id = str(uuid.uuid4())
        reply_topic = f"{self._reply_topic_prefix}{correlation_id}"

        # 注册等待回复
        event = threading.Event()
        pending = _PendingReply(correlation_id=correlation_id, event=event)

        with self._lock:
            self._pending_replies[correlation_id] = pending

        # 订阅回复主题
        self.subscribe(reply_topic, self._handle_reply)

        try:
            # 发布请求消息
            request_headers = headers or {}
            request_headers["reply_to"] = reply_topic
            request_headers["correlation_id"] = correlation_id
            request_headers["message_type"] = "request"

            self.publish(topic, message, headers=request_headers)

            # 等待响应
            if not event.wait(timeout=timeout):
                with self._lock:
                    self._pending_replies.pop(correlation_id, None)
                raise TimeoutError(
                    f"请求超时: topic={topic}, timeout={timeout}s, "
                    f"correlation_id={correlation_id}"
                )

            if pending.error:
                raise RuntimeError(
                    f"RPC请求失败: {pending.error}"
                )

            return pending.response

        finally:
            # 清理
            with self._lock:
                self._pending_replies.pop(correlation_id, None)
            self._unsubscribe_by_topic(reply_topic)

    def _handle_reply(self, message: Message) -> None:
        """处理回复消息"""
        # 优先检查Message的correlation_id字段，其次检查headers
        correlation_id = message.correlation_id or message.headers.get("correlation_id", "")
        if not correlation_id:
            return

        with self._lock:
            pending = self._pending_replies.get(correlation_id)

        if pending is None:
            logger.debug("收到无匹配请求的回复: correlation_id=%s", correlation_id)
            return

        # 检查是否是错误响应
        if message.headers.get("message_type") == "error":
            pending.error = str(message.payload)
        else:
            pending.response = message.payload

        pending.event.set()

    def _find_matching_subscriptions(self, topic: str) -> List[_Subscription]:
        """
        查找匹配指定主题的所有订阅

        支持通配符匹配。对于消费者组，只返回组内一个订阅者（轮询）。

        Args:
            topic: 目标主题

        Returns:
            匹配的订阅列表
        """
        matched: List[_Subscription] = []
        seen_groups: Set[str] = set()

        for pattern, subs in self._subscriptions.items():
            if fnmatch.fnmatch(topic, pattern):
                for sub in subs:
                    if sub.group:
                        # 消费者组内负载均衡
                        if sub.group in seen_groups:
                            continue
                        seen_groups.add(sub.group)

                        group_subs = self._consumer_groups.get(sub.group, [])
                        if group_subs:
                            index = self._group_index[sub.group] % len(group_subs)
                            selected = group_subs[index]
                            self._group_index[sub.group] = index + 1
                            matched.append(selected)
                    else:
                        matched.append(sub)

        return matched

    def _unsubscribe_by_topic(self, topic: str) -> None:
        """取消指定主题的所有订阅"""
        with self._lock:
            if topic in self._subscriptions:
                self._subscriptions[topic].clear()

    def get_topics(self) -> list:
        """获取所有活跃主题"""
        with self._lock:
            return list(self._subscriptions.keys())

    def get_subscriptions(self, topic: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取订阅信息

        Args:
            topic: 按主题过滤

        Returns:
            订阅信息列表
        """
        with self._lock:
            result = []
            for pattern, subs in self._subscriptions.items():
                if topic and not fnmatch.fnmatch(topic, pattern):
                    continue
                for sub in subs:
                    result.append({
                        "subscription_id": sub.subscription_id,
                        "topic": sub.topic,
                        "group": sub.group,
                        "message_count": sub.message_count,
                        "created_at": sub.created_at.isoformat(),
                    })
            return result

    def get_queue_depth(self, topic: Optional[str] = None) -> Dict[str, int]:
        """
        获取消息队列深度

        Args:
            topic: 指定主题，None返回所有

        Returns:
            主题到队列深度的映射
        """
        with self._lock:
            if topic:
                return {topic: len(self._message_queue.get(topic, deque()))}
            return {
                t: len(q) for t, q in self._message_queue.items() if q
            }

    def get_stats(self) -> Dict[str, Any]:
        """获取总线统计信息"""
        with self._lock:
            total_subscriptions = sum(
                len(subs) for subs in self._subscriptions.values()
            )
            total_groups = len(self._consumer_groups)
            return {
                "type": "MemoryBackend",
                "closed": self._closed,
                "total_published": self._total_published,
                "total_received": self._total_received,
                "total_errors": self._total_errors,
                "total_subscriptions": total_subscriptions,
                "total_consumer_groups": total_groups,
                "topics": len(self._subscriptions),
                "pending_replies": len(self._pending_replies),
                "max_queue_size": self._max_queue_size,
            }

    def close(self) -> None:
        """关闭消息总线"""
        with self._lock:
            self._closed = True
            self._subscriptions.clear()
            self._consumer_groups.clear()
            self._message_queue.clear()
            self._pending_replies.clear()
            logger.info("内存消息总线已关闭")

    def __repr__(self) -> str:
        return (
            f"MemoryBackend(topics={len(self._subscriptions)}, "
            f"published={self._total_published})"
        )
