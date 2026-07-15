"""
监听机制
订阅Agent上下线事件并推送
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Union

from .schema import AgentMetadata, AgentStatus


class WatchEventType(Enum):
    """监听事件类型"""
    AGENT_REGISTERED = "agent_registered"     # Agent注册
    AGENT_DEREGISTERED = "agent_deregistered" # Agent注销
    AGENT_UPDATED = "agent_updated"           # Agent更新
    STATUS_CHANGED = "status_changed"         # 状态变更
    HEARTBEAT_RECEIVED = "heartbeat_received" # 收到心跳
    LEASE_EXPIRED = "lease_expired"           # 租约过期
    CAPABILITY_ADDED = "capability_added"     # 添加能力
    CAPABILITY_REMOVED = "capability_removed" # 移除能力


@dataclass
class WatchEvent:
    """监听事件"""
    event_type: WatchEventType
    agent_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)
    previous_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "previous_data": self.previous_data
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class WatchFilter:
    """监听过滤器"""
    event_types: Optional[Set[WatchEventType]] = None
    agent_ids: Optional[Set[str]] = None
    capabilities: Optional[Set[str]] = None
    status_filter: Optional[AgentStatus] = None

    def matches(self, event: WatchEvent, agent: Optional[AgentMetadata] = None) -> bool:
        """检查事件是否匹配过滤器"""
        if self.event_types and event.event_type not in self.event_types:
            return False
        
        if self.agent_ids and event.agent_id not in self.agent_ids:
            return False
        
        if agent and self.capabilities:
            if not self.capabilities.issubset(agent.capabilities):
                return False
        
        if agent and self.status_filter:
            if agent.status != self.status_filter:
                return False
        
        return True


class WatchSubscription:
    """监听订阅"""

    def __init__(
        self,
        subscription_id: str,
        filter: WatchFilter,
        callback: Optional[Callable[[WatchEvent], None]] = None,
        queue_size: int = 1000
    ):
        self.subscription_id = subscription_id
        self.filter = filter
        self.callback = callback
        self.event_queue: queue.Queue[WatchEvent] = queue.Queue(maxsize=queue_size)
        self.created_at = datetime.utcnow()
        self.event_count = 0
        self.dropped_count = 0
        self._active = True
        self._lock = threading.Lock()

    def is_active(self) -> bool:
        """检查订阅是否活跃"""
        with self._lock:
            return self._active

    def deactivate(self) -> None:
        """停用订阅"""
        with self._lock:
            self._active = False

    def push_event(self, event: WatchEvent, agent: Optional[AgentMetadata] = None) -> bool:
        """推送事件到订阅"""
        if not self.is_active():
            return False
        
        if not self.filter.matches(event, agent):
            return False
        
        with self._lock:
            self.event_count += 1
        
        if self.callback:
            try:
                self.callback(event)
                return True
            except Exception:
                return False
        else:
            try:
                self.event_queue.put_nowait(event)
                return True
            except queue.Full:
                with self._lock:
                    self.dropped_count += 1
                return False

    def get_events(self, timeout: Optional[float] = None) -> Optional[WatchEvent]:
        """获取事件（阻塞）"""
        try:
            return self.event_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_events_batch(self, max_events: int = 100, timeout: float = 0.0) -> List[WatchEvent]:
        """批量获取事件"""
        events = []
        deadline = time.time() + timeout if timeout > 0 else None
        
        while len(events) < max_events:
            remaining = None
            if deadline:
                remaining = max(0, deadline - time.time())
                if remaining == 0:
                    break
            
            event = self.get_events(timeout=remaining)
            if event is None:
                break
            events.append(event)
        
        return events

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "subscription_id": self.subscription_id,
                "active": self._active,
                "created_at": self.created_at.isoformat(),
                "event_count": self.event_count,
                "dropped_count": self.dropped_count,
                "queue_size": self.event_queue.qsize()
            }


class WatchManager:
    """
    监听管理器
    
    管理Agent事件的订阅与推送，支持：
    - 多订阅者模式
    - 事件过滤
    - 回调与队列两种消费模式
    - 批量事件获取
    """

    def __init__(self, max_subscriptions: int = 1000):
        """
        初始化监听管理器
        
        Args:
            max_subscriptions: 最大订阅数
        """
        self._max_subscriptions = max_subscriptions
        self._subscriptions: Dict[str, WatchSubscription] = {}
        self._lock = threading.RLock()
        self._counter = 0
        
        # 统计信息
        self._stats = {
            "events_published": 0,
            "events_delivered": 0,
            "events_dropped": 0
        }

    def subscribe(
        self,
        filter: Optional[WatchFilter] = None,
        callback: Optional[Callable[[WatchEvent], None]] = None,
        subscription_id: Optional[str] = None
    ) -> WatchSubscription:
        """
        订阅事件
        
        Args:
            filter: 事件过滤器
            callback: 回调函数（None则使用队列模式）
            subscription_id: 指定订阅ID（None则自动生成）
            
        Returns:
            订阅对象
        """
        with self._lock:
            if len(self._subscriptions) >= self._max_subscriptions:
                raise RuntimeError("Maximum number of subscriptions reached")
            
            if subscription_id is None:
                self._counter += 1
                subscription_id = f"sub_{self._counter}_{int(time.time())}"
            
            subscription = WatchSubscription(
                subscription_id=subscription_id,
                filter=filter or WatchFilter(),
                callback=callback
            )
            
            self._subscriptions[subscription_id] = subscription
            return subscription

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        取消订阅
        
        Args:
            subscription_id: 订阅ID
            
        Returns:
            是否成功取消
        """
        with self._lock:
            if subscription_id not in self._subscriptions:
                return False
            
            subscription = self._subscriptions[subscription_id]
            subscription.deactivate()
            del self._subscriptions[subscription_id]
            return True

    def publish(
        self,
        event_type: WatchEventType,
        agent_id: str,
        data: Optional[Dict[str, Any]] = None,
        previous_data: Optional[Dict[str, Any]] = None,
        agent: Optional[AgentMetadata] = None
    ) -> int:
        """
        发布事件
        
        Args:
            event_type: 事件类型
            agent_id: Agent ID
            data: 事件数据
            previous_data: 之前的数据（用于比较）
            agent: Agent元数据（用于过滤）
            
        Returns:
            成功投递的订阅数
        """
        event = WatchEvent(
            event_type=event_type,
            agent_id=agent_id,
            data=data or {},
            previous_data=previous_data
        )
        
        delivered = 0
        
        with self._lock:
            self._stats["events_published"] += 1
            subscriptions = list(self._subscriptions.values())
        
        for subscription in subscriptions:
            if subscription.push_event(event, agent):
                delivered += 1
        
        self._stats["events_delivered"] += delivered
        self._stats["events_dropped"] += len(subscriptions) - delivered
        
        return delivered

    def get_subscription(self, subscription_id: str) -> Optional[WatchSubscription]:
        """获取订阅对象"""
        with self._lock:
            return self._subscriptions.get(subscription_id)

    def get_all_subscriptions(self) -> Dict[str, WatchSubscription]:
        """获取所有订阅"""
        with self._lock:
            return dict(self._subscriptions)

    def cleanup_inactive(self, max_inactive_seconds: float = 300) -> int:
        """
        清理不活跃的订阅
        
        Args:
            max_inactive_seconds: 最大不活跃时间
            
        Returns:
            清理的订阅数
        """
        cutoff = datetime.utcnow().timestamp() - max_inactive_seconds
        to_remove = []
        
        with self._lock:
            for sub_id, subscription in self._subscriptions.items():
                if not subscription.is_active():
                    to_remove.append(sub_id)
                    continue
                
                # 检查是否长时间无事件
                last_event_time = subscription.created_at.timestamp()
                if subscription.event_count > 0:
                    # 简化：使用创建时间作为参考
                    pass
                
                if last_event_time < cutoff and subscription.event_queue.empty():
                    to_remove.append(sub_id)
        
        for sub_id in to_remove:
            self.unsubscribe(sub_id)
        
        return len(to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            stats = self._stats.copy()
            stats["active_subscriptions"] = len(self._subscriptions)
            return stats

    def clear(self) -> None:
        """清除所有订阅"""
        with self._lock:
            for subscription in self._subscriptions.values():
                subscription.deactivate()
            self._subscriptions.clear()


class AgentWatcher:
    """
    Agent观察者
    
    提供高级监听功能，如状态变更检测、批量通知等
    """

    def __init__(self, watch_manager: WatchManager):
        self._manager = watch_manager
        self._agent_states: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def on_agent_registered(self, agent: AgentMetadata) -> None:
        """Agent注册事件"""
        with self._lock:
            self._agent_states[agent.agent_id] = self._serialize_state(agent)
        
        self._manager.publish(
            WatchEventType.AGENT_REGISTERED,
            agent.agent_id,
            data=agent.to_dict(),
            agent=agent
        )

    def on_agent_deregistered(self, agent: AgentMetadata) -> None:
        """Agent注销事件"""
        with self._lock:
            if agent.agent_id in self._agent_states:
                del self._agent_states[agent.agent_id]
        
        self._manager.publish(
            WatchEventType.AGENT_DEREGISTERED,
            agent.agent_id,
            data=agent.to_dict(),
            agent=agent
        )

    def on_agent_updated(self, agent: AgentMetadata, previous: AgentMetadata) -> None:
        """Agent更新事件"""
        with self._lock:
            previous_state = self._agent_states.get(agent.agent_id)
            self._agent_states[agent.agent_id] = self._serialize_state(agent)
        
        self._manager.publish(
            WatchEventType.AGENT_UPDATED,
            agent.agent_id,
            data=agent.to_dict(),
            previous_data=previous_state,
            agent=agent
        )

    def on_status_changed(
        self,
        agent: AgentMetadata,
        previous_status: AgentStatus
    ) -> None:
        """状态变更事件"""
        with self._lock:
            self._agent_states[agent.agent_id] = self._serialize_state(agent)
        
        self._manager.publish(
            WatchEventType.STATUS_CHANGED,
            agent.agent_id,
            data={"status": agent.status.value},
            previous_data={"status": previous_status.value},
            agent=agent
        )

    def on_heartbeat(self, agent: AgentMetadata) -> None:
        """心跳事件"""
        self._manager.publish(
            WatchEventType.HEARTBEAT_RECEIVED,
            agent.agent_id,
            data={"last_heartbeat": agent.last_heartbeat.isoformat() if agent.last_heartbeat else None},
            agent=agent
        )

    def on_lease_expired(self, agent_id: str) -> None:
        """租约过期事件"""
        with self._lock:
            if agent_id in self._agent_states:
                del self._agent_states[agent_id]
        
        self._manager.publish(
            WatchEventType.LEASE_EXPIRED,
            agent_id,
            data={"expired_at": datetime.utcnow().isoformat()}
        )

    def on_capability_changed(
        self,
        agent: AgentMetadata,
        added: Optional[Set[str]] = None,
        removed: Optional[Set[str]] = None
    ) -> None:
        """能力变更事件"""
        if added:
            for cap in added:
                self._manager.publish(
                    WatchEventType.CAPABILITY_ADDED,
                    agent.agent_id,
                    data={"capability": cap},
                    agent=agent
                )
        
        if removed:
            for cap in removed:
                self._manager.publish(
                    WatchEventType.CAPABILITY_REMOVED,
                    agent.agent_id,
                    data={"capability": cap},
                    agent=agent
                )

    def _serialize_state(self, agent: AgentMetadata) -> Dict[str, Any]:
        """序列化Agent状态"""
        return {
            "status": agent.status.value,
            "capabilities": list(agent.capabilities),
            "version": agent.version,
            "labels": dict(agent.labels)
        }


class WatchClient:
    """
    监听客户端
    
    简化订阅操作的客户端类
    """

    def __init__(self, watch_manager: WatchManager):
        self._manager = watch_manager

    def watch_agent(self, agent_id: str) -> WatchSubscription:
        """监听特定Agent的所有事件"""
        return self._manager.subscribe(
            filter=WatchFilter(agent_ids={agent_id})
        )

    def watch_event_types(self, event_types: List[WatchEventType]) -> WatchSubscription:
        """监听特定类型的事件"""
        return self._manager.subscribe(
            filter=WatchFilter(event_types=set(event_types))
        )

    def watch_capabilities(self, capabilities: List[str]) -> WatchSubscription:
        """监听具有特定能力的Agent"""
        return self._manager.subscribe(
            filter=WatchFilter(capabilities=set(capabilities))
        )

    def watch_status(self, status: AgentStatus) -> WatchSubscription:
        """监听特定状态的Agent"""
        return self._manager.subscribe(
            filter=WatchFilter(status_filter=status)
        )

    def watch_all(self) -> WatchSubscription:
        """监听所有事件"""
        return self._manager.subscribe()
