"""
消息总线抽象接口

定义消息总线的核心抽象接口和数据结构。
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

MessageHandler = Callable[["Message"], Any]


@dataclass
class Message:
    """
    消息数据类

    消息总线中传递的基本消息单元。

    Attributes:
        id: 消息唯一标识
        topic: 消息主题
        payload: 消息负载
        headers: 消息头部元数据
        timestamp: 消息创建时间
        reply_to: 回复主题（用于请求-响应模式）
        correlation_id: 关联ID（用于关联请求和响应）
        source: 消息来源
        priority: 消息优先级（0-9，数值越大优先级越高）
    """

    id: str = ""
    topic: str = ""
    payload: Any = None
    headers: Dict[str, str] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    reply_to: str = ""
    correlation_id: str = ""
    source: str = ""
    priority: int = 5

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """将消息转换为字典格式"""
        return {
            "id": self.id,
            "topic": self.topic,
            "payload": self.payload,
            "headers": self.headers,
            "timestamp": (
                self.timestamp.isoformat() if self.timestamp else None
            ),
            "reply_to": self.reply_to,
            "correlation_id": self.correlation_id,
            "source": self.source,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """从字典创建消息"""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return cls(
            id=data.get("id", ""),
            topic=data.get("topic", ""),
            payload=data.get("payload"),
            headers=data.get("headers", {}),
            timestamp=timestamp,
            reply_to=data.get("reply_to", ""),
            correlation_id=data.get("correlation_id", ""),
            source=data.get("source", ""),
            priority=data.get("priority", 5),
        )

    def create_reply(self, payload: Any, **kwargs: Any) -> "Message":
        """
        创建回复消息

        Args:
            payload: 回复负载
            **kwargs: 其他消息属性覆盖

        Returns:
            新的回复消息
        """
        return Message(
            topic=self.reply_to,
            payload=payload,
            correlation_id=self.correlation_id or self.id,
            headers=kwargs.pop("headers", {}),
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"Message(id={self.id!r}, topic={self.topic!r}, "
            f"priority={self.priority})"
        )


class MessageBus(ABC):
    """
    消息总线抽象接口

    定义消息发布/订阅、请求-响应等核心操作的标准接口。

    所有消息总线实现（内存、Redis、Kafka等）都必须实现此接口。
    """

    @abstractmethod
    def publish(self, topic: str, message: Any, headers: Optional[Dict[str, str]] = None) -> str:
        """
        发布消息到指定主题

        Args:
            topic: 目标主题
            message: 消息内容
            headers: 消息头部

        Returns:
            消息ID
        """
        ...

    @abstractmethod
    def subscribe(self, topic: str, handler: MessageHandler, group: str = "") -> str:
        """
        订阅主题

        Args:
            topic: 订阅主题
            handler: 消息处理函数
            group: 消费者组名称

        Returns:
            订阅ID
        """
        ...

    @abstractmethod
    def unsubscribe(self, subscription_id: str) -> bool:
        """
        取消订阅

        Args:
            subscription_id: 订阅ID

        Returns:
            是否成功取消
        """
        ...

    @abstractmethod
    def request(
        self,
        topic: str,
        message: Any,
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        请求-响应模式

        发送请求并等待响应。

        Args:
            topic: 目标主题
            message: 请求内容
            timeout: 超时时间（秒）
            headers: 消息头部

        Returns:
            响应内容

        Raises:
            TimeoutError: 等待响应超时
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """关闭消息总线，释放资源"""
        ...

    @abstractmethod
    def get_topics(self) -> list:
        """获取所有活跃主题"""
        ...
