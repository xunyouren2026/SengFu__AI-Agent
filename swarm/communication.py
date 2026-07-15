"""
智能体通信总线模块

提供智能体间的消息传递机制，支持点对点、广播、多播通信，
消息优先级排序、请求-响应模式和消息持久化。线程安全实现。
"""

import threading
import time
import uuid
import copy
import json
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import deque


class MessageType(Enum):
    """消息类型枚举"""
    TASK = "task"
    RESULT = "result"
    QUERY = "query"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    NEGOTIATE = "negotiate"
    INFORM = "inform"


class MessagePriority(IntEnum):
    """消息优先级（数值越大优先级越高）"""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    URGENT = 10


class AgentMessage:
    """智能体消息

    描述智能体间通信的消息对象。

    Attributes:
        id: 消息唯一标识
        from_id: 发送者ID
        to_id: 接收者ID
        type: 消息类型
        content: 消息内容
        timestamp: 发送时间戳
        reply_to: 回复的消息ID
        priority: 消息优先级
        metadata: 附加元数据
    """

    def __init__(
        self,
        from_id: str,
        to_id: str,
        type: MessageType = MessageType.INFORM,
        content: Any = None,
        reply_to: Optional[str] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
        message_id: Optional[str] = None,
    ):
        self.id = message_id or str(uuid.uuid4())
        self.from_id = from_id
        self.to_id = to_id
        self.type = type
        self.content = content
        self.timestamp = time.time()
        self.reply_to = reply_to
        self.priority = priority
        self.metadata = metadata or {}
        self.delivered = False
        self.read = False

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "id": self.id,
            "from_id": self.from_id,
            "to_id": self.to_id,
            "type": self.type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "reply_to": self.reply_to,
            "priority": int(self.priority),
            "metadata": self.metadata,
            "delivered": self.delivered,
            "read": self.read,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """从字典反序列化"""
        msg = cls(
            from_id=data["from_id"],
            to_id=data["to_id"],
            type=MessageType(data.get("type", "inform")),
            content=data.get("content"),
            reply_to=data.get("reply_to"),
            priority=MessagePriority(data.get("priority", 5)),
            metadata=data.get("metadata", {}),
            message_id=data.get("id"),
        )
        msg.timestamp = data.get("timestamp", time.time())
        msg.delivered = data.get("delivered", False)
        msg.read = data.get("read", False)
        return msg

    def __repr__(self) -> str:
        short_id = repr(self.id)[:12]
        return (
            f"AgentMessage(id={short_id}, from={self.from_id!r}, "
            f"to={self.to_id!r}, type={self.type.value})"
        )

    def __lt__(self, other: "AgentMessage") -> bool:
        """用于优先级排序（优先级高的排前面）"""
        if not isinstance(other, AgentMessage):
            return NotImplemented
        # 优先级高的先出队，时间早的先出队
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.timestamp < other.timestamp


class CommunicationBus:
    """智能体通信总线

    管理智能体间的消息路由和传递。支持：
    - 点对点消息
    - 广播消息
    - 多播消息
    - 请求-响应模式
    - 消息优先级
    - 消息持久化
    - 消息订阅

    线程安全实现。
    """

    def __init__(self, persist: bool = False):
        """
        Args:
            persist: 是否持久化消息
        """
        self._lock = threading.RLock()
        # 每个智能体一个消息队列
        self._queues: Dict[str, deque] = {}
        # 消息订阅处理器
        self._subscribers: Dict[str, List[Callable]] = {}
        # 等待响应的请求 {request_id: {"event": Event, "response": None}}
        self._pending_requests: Dict[str, Dict[str, Any]] = {}
        # 消息历史（持久化）
        self._persist = persist
        self._message_history: List[AgentMessage] = []
        self._max_history = 10000
        # 消息统计
        self._stats = {
            "sent": 0,
            "received": 0,
            "broadcast": 0,
            "dropped": 0,
        }

    def _ensure_queue(self, agent_id: str) -> deque:
        """确保智能体消息队列存在"""
        if agent_id not in self._queues:
            self._queues[agent_id] = deque()
        return self._queues[agent_id]

    def send(self, message: AgentMessage) -> bool:
        """发送消息（点对点）

        Args:
            message: 消息对象

        Returns:
            是否成功发送
        """
        if not isinstance(message, AgentMessage):
            raise TypeError("参数必须是AgentMessage实例")

        with self._lock:
            queue = self._ensure_queue(message.to_id)
            queue.append(message)
            message.delivered = True
            self._stats["sent"] += 1

            # 持久化
            if self._persist:
                self._add_to_history(message)

            # 通知订阅者
            self._notify_subscribers(message.to_id, message)

            # 检查是否是对某个请求的响应
            if message.reply_to and message.reply_to in self._pending_requests:
                pending = self._pending_requests[message.reply_to]
                pending["response"] = message
                pending["event"].set()

            return True

    def receive(self, agent_id: str, block: bool = False, timeout: float = 0.0) -> Optional[AgentMessage]:
        """接收消息

        按优先级从队列中取出消息。

        Args:
            agent_id: 接收者ID
            block: 是否阻塞等待
            timeout: 阻塞超时时间（秒）

        Returns:
            消息对象，无消息时返回None
        """
        deadline = time.time() + timeout if timeout > 0 else None

        while True:
            with self._lock:
                queue = self._queues.get(agent_id)
                if queue and queue:
                    # 按优先级排序取出
                    messages = sorted(queue, key=lambda m: (-m.priority, m.timestamp))
                    message = messages[0]
                    queue.remove(message)
                    message.read = True
                    self._stats["received"] += 1
                    return message

            if not block:
                return None

            if deadline and time.time() >= deadline:
                return None

            time.sleep(0.05)  # 避免忙等待

    def receive_all(self, agent_id: str) -> List[AgentMessage]:
        """接收所有待处理消息

        Args:
            agent_id: 接收者ID

        Returns:
            按优先级排序的消息列表
        """
        with self._lock:
            queue = self._queues.get(agent_id)
            if not queue:
                return []

            messages = sorted(queue, key=lambda m: (-m.priority, m.timestamp))
            queue.clear()
            for msg in messages:
                msg.read = True
                self._stats["received"] += 1
            return messages

    def broadcast(self, sender_id: str, content: Any, exclude_self: bool = True,
                  priority: MessagePriority = MessagePriority.NORMAL,
                  msg_type: MessageType = MessageType.BROADCAST) -> int:
        """广播消息

        向所有已注册的智能体发送广播消息。

        Args:
            sender_id: 发送者ID
            content: 消息内容
            exclude_self: 是否排除发送者自己
            priority: 消息优先级
            msg_type: 消息类型

        Returns:
            成功发送的消息数量
        """
        with self._lock:
            targets = list(self._queues.keys())
            if exclude_self:
                targets = [t for t in targets if t != sender_id]

            count = 0
            for target_id in targets:
                message = AgentMessage(
                    from_id=sender_id,
                    to_id=target_id,
                    type=msg_type,
                    content=content,
                    priority=priority,
                )
                self._ensure_queue(target_id).append(message)
                message.delivered = True
                self._notify_subscribers(target_id, message)
                count += 1

            self._stats["broadcast"] += count

            if self._persist:
                for target_id in targets:
                    msg = AgentMessage(
                        from_id=sender_id,
                        to_id=target_id,
                        type=msg_type,
                        content=content,
                        priority=priority,
                    )
                    self._add_to_history(msg)

            return count

    def multicast(self, sender_id: str, target_ids: List[str], content: Any,
                  priority: MessagePriority = MessagePriority.NORMAL,
                  msg_type: MessageType = MessageType.INFORM) -> int:
        """多播消息

        向指定智能体列表发送消息。

        Args:
            sender_id: 发送者ID
            target_ids: 目标智能体ID列表
            content: 消息内容
            priority: 消息优先级
            msg_type: 消息类型

        Returns:
            成功发送的消息数量
        """
        count = 0
        for target_id in target_ids:
            message = AgentMessage(
                from_id=sender_id,
                to_id=target_id,
                type=msg_type,
                content=content,
                priority=priority,
            )
            if self.send(message):
                count += 1
        return count

    def request(self, from_id: str, to_id: str, content: Any,
                timeout: float = 30.0, priority: MessagePriority = MessagePriority.HIGH) -> Optional[AgentMessage]:
        """请求-响应模式

        发送请求并等待响应。

        Args:
            from_id: 请求者ID
            to_id: 被请求者ID
            content: 请求内容
            timeout: 超时时间（秒）
            priority: 请求优先级

        Returns:
            响应消息，超时返回None
        """
        # 创建请求消息
        request_msg = AgentMessage(
            from_id=from_id,
            to_id=to_id,
            type=MessageType.QUERY,
            content=content,
            priority=priority,
        )

        # 注册等待
        event = threading.Event()
        with self._lock:
            self._pending_requests[request_msg.id] = {
                "event": event,
                "response": None,
                "from_id": from_id,
                "to_id": to_id,
            }

        # 发送请求
        self.send(request_msg)

        # 等待响应
        event.wait(timeout=timeout)

        with self._lock:
            pending = self._pending_requests.pop(request_msg.id, None)
            if pending and pending["response"] is not None:
                return pending["response"]

        return None

    def subscribe(self, agent_id: str, handler: Callable[[AgentMessage], None]) -> bool:
        """订阅消息

        为指定智能体注册消息处理回调。当该智能体收到新消息时，
        所有注册的处理器会被依次调用。

        Args:
            agent_id: 智能体ID
            handler: 消息处理函数 (message) -> None

        Returns:
            是否成功订阅
        """
        if not callable(handler):
            raise TypeError("处理器必须是可调用对象")

        with self._lock:
            if agent_id not in self._subscribers:
                self._subscribers[agent_id] = []
            self._subscribers[agent_id].append(handler)
            return True

    def unsubscribe(self, agent_id: str, handler: Callable) -> bool:
        """取消订阅"""
        with self._lock:
            handlers = self._subscribers.get(agent_id, [])
            try:
                handlers.remove(handler)
                return True
            except ValueError:
                return False

    def _notify_subscribers(self, agent_id: str, message: AgentMessage) -> None:
        """通知订阅者"""
        handlers = self._subscribers.get(agent_id, [])
        for handler in handlers:
            try:
                handler(message)
            except Exception:
                pass

    def _add_to_history(self, message: AgentMessage) -> None:
        """添加消息到历史记录"""
        self._message_history.append(copy.deepcopy(message))
        # 限制历史记录大小
        if len(self._message_history) > self._max_history:
            self._message_history = self._message_history[-self._max_history:]

    def get_pending_count(self, agent_id: str) -> int:
        """获取智能体待处理消息数"""
        with self._lock:
            queue = self._queues.get(agent_id)
            return len(queue) if queue else 0

    def get_history(
        self,
        agent_id: Optional[str] = None,
        msg_type: Optional[MessageType] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """获取消息历史

        Args:
            agent_id: 按智能体过滤
            msg_type: 按消息类型过滤
            limit: 返回数量限制

        Returns:
            消息字典列表
        """
        with self._lock:
            messages = self._message_history
            if agent_id:
                messages = [m for m in messages if m.from_id == agent_id or m.to_id == agent_id]
            if msg_type:
                messages = [m for m in messages if m.type == msg_type]

            return [m.to_dict() for m in messages[-limit:]]

    def get_statistics(self) -> Dict[str, Any]:
        """获取通信统计信息"""
        with self._lock:
            queue_sizes = {aid: len(q) for aid, q in self._queues.items()}
            return {
                "total_sent": self._stats["sent"],
                "total_received": self._stats["received"],
                "total_broadcast": self._stats["broadcast"],
                "total_dropped": self._stats["dropped"],
                "registered_agents": len(self._queues),
                "queue_sizes": queue_sizes,
                "history_size": len(self._message_history),
                "pending_requests": len(self._pending_requests),
            }

    def clear(self, agent_id: Optional[str] = None) -> int:
        """清空消息队列

        Args:
            agent_id: 指定智能体ID，为None则清空所有

        Returns:
            清空的消息数量
        """
        with self._lock:
            if agent_id:
                queue = self._queues.get(agent_id)
                if queue:
                    count = len(queue)
                    queue.clear()
                    return count
                return 0
            else:
                total = sum(len(q) for q in self._queues.values())
                self._queues.clear()
                return total

    def register_agent(self, agent_id: str) -> None:
        """注册智能体（创建消息队列）"""
        with self._lock:
            self._ensure_queue(agent_id)

    def deregister_agent(self, agent_id: str) -> int:
        """注销智能体（移除消息队列）"""
        with self._lock:
            queue = self._queues.pop(agent_id, None)
            self._subscribers.pop(agent_id, None)
            return len(queue) if queue else 0
