"""
租约管理器
TTL过期自动注销失效Agent
"""

from __future__ import annotations

import heapq
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set

from .schema import AgentMetadata, AgentStatus


@dataclass(order=True)
class Lease:
    """租约对象"""
    expiry_time: float
    agent_id: str = field(compare=False)
    ttl_seconds: int = field(default=30, compare=False)
    created_at: datetime = field(default_factory=datetime.utcnow, compare=False)
    renewed_at: Optional[datetime] = field(default=None, compare=False)
    renew_count: int = field(default=0, compare=False)

    def __post_init__(self):
        if self.renewed_at is None:
            self.renewed_at = self.created_at

    def is_expired(self, now: Optional[float] = None) -> bool:
        """检查租约是否已过期"""
        if now is None:
            now = time.time()
        return now >= self.expiry_time

    def renew(self, ttl_seconds: Optional[int] = None) -> None:
        """续租"""
        if ttl_seconds:
            self.ttl_seconds = ttl_seconds
        self.expiry_time = time.time() + self.ttl_seconds
        self.renewed_at = datetime.utcnow()
        self.renew_count += 1


class LeaseEvent:
    """租约事件"""
    EXPIRED = "expired"
    RENEWED = "renewed"
    CREATED = "created"
    REVOKED = "revoked"


class LeaseManager:
    """
    租约管理器
    
    管理Agent的租约生命周期，支持：
    - 租约创建与分配
    - 租约续期（心跳）
    - 租约过期自动回收
    - 租约事件通知
    """

    def __init__(
        self,
        default_ttl: int = 30,
        check_interval: float = 5.0,
        max_ttl: int = 300,
        min_ttl: int = 5
    ):
        """
        初始化租约管理器
        
        Args:
            default_ttl: 默认租约TTL（秒）
            check_interval: 过期检查间隔（秒）
            max_ttl: 最大允许TTL（秒）
            min_ttl: 最小允许TTL（秒）
        """
        self._default_ttl = default_ttl
        self._check_interval = check_interval
        self._max_ttl = max_ttl
        self._min_ttl = min_ttl
        
        # 租约存储: agent_id -> Lease
        self._leases: Dict[str, Lease] = {}
        
        # 优先队列: (expiry_time, agent_id)
        self._expiry_queue: List[tuple] = []
        
        # 事件监听器
        self._listeners: Dict[str, List[Callable[[str, Lease], None]]] = {
            LeaseEvent.EXPIRED: [],
            LeaseEvent.RENEWED: [],
            LeaseEvent.CREATED: [],
            LeaseEvent.REVOKED: []
        }
        
        # 线程安全
        self._lock = threading.RLock()
        self._running = False
        self._check_thread: Optional[threading.Thread] = None
        
        # 统计信息
        self._stats = {
            "leases_created": 0,
            "leases_expired": 0,
            "leases_renewed": 0,
            "leases_revoked": 0
        }

    def start(self) -> None:
        """启动租约管理器"""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._check_thread = threading.Thread(
                target=self._expiry_check_loop,
                daemon=True,
                name="LeaseManager"
            )
            self._check_thread.start()

    def stop(self) -> None:
        """停止租约管理器"""
        with self._lock:
            self._running = False
        if self._check_thread:
            self._check_thread.join(timeout=self._check_interval + 1)

    def create_lease(
        self,
        agent_id: str,
        ttl_seconds: Optional[int] = None
    ) -> Lease:
        """
        创建新租约
        
        Args:
            agent_id: Agent ID
            ttl_seconds: 租约TTL（秒）
            
        Returns:
            创建的租约对象
        """
        ttl = self._clamp_ttl(ttl_seconds or self._default_ttl)
        
        with self._lock:
            # 如果已存在，先撤销旧租约
            if agent_id in self._leases:
                self._revoke_lease(agent_id, notify=False)
            
            expiry_time = time.time() + ttl
            lease = Lease(
                expiry_time=expiry_time,
                agent_id=agent_id,
                ttl_seconds=ttl
            )
            
            self._leases[agent_id] = lease
            heapq.heappush(self._expiry_queue, (expiry_time, agent_id))
            self._stats["leases_created"] += 1
        
        self._notify_event(LeaseEvent.CREATED, lease)
        return lease

    def renew_lease(
        self,
        agent_id: str,
        ttl_seconds: Optional[int] = None
    ) -> Optional[Lease]:
        """
        续租
        
        Args:
            agent_id: Agent ID
            ttl_seconds: 新的TTL（秒），None则使用原TTL
            
        Returns:
            更新后的租约，如果不存在则返回None
        """
        with self._lock:
            if agent_id not in self._leases:
                return None
            
            lease = self._leases[agent_id]
            
            # 检查是否已过期
            if lease.is_expired():
                return None
            
            ttl = self._clamp_ttl(ttl_seconds or lease.ttl_seconds)
            lease.renew(ttl)
            
            # 更新优先队列
            heapq.heappush(self._expiry_queue, (lease.expiry_time, agent_id))
            self._stats["leases_renewed"] += 1
        
        self._notify_event(LeaseEvent.RENEWED, lease)
        return lease

    def revoke_lease(self, agent_id: str) -> bool:
        """
        撤销租约
        
        Args:
            agent_id: Agent ID
            
        Returns:
            是否成功撤销
        """
        return self._revoke_lease(agent_id, notify=True)

    def _revoke_lease(self, agent_id: str, notify: bool = True) -> bool:
        """内部撤销租约"""
        with self._lock:
            if agent_id not in self._leases:
                return False
            
            lease = self._leases[agent_id]
            del self._leases[agent_id]
            self._stats["leases_revoked"] += 1
        
        if notify:
            self._notify_event(LeaseEvent.REVOKED, lease)
        return True

    def get_lease(self, agent_id: str) -> Optional[Lease]:
        """获取租约信息"""
        with self._lock:
            lease = self._leases.get(agent_id)
            if lease and lease.is_expired():
                return None
            return lease

    def is_lease_valid(self, agent_id: str) -> bool:
        """检查租约是否有效"""
        lease = self.get_lease(agent_id)
        return lease is not None and not lease.is_expired()

    def get_remaining_ttl(self, agent_id: str) -> float:
        """
        获取租约剩余TTL
        
        Returns:
            剩余秒数，租约不存在或已过期返回0
        """
        with self._lock:
            lease = self._leases.get(agent_id)
            if not lease:
                return 0.0
            remaining = lease.expiry_time - time.time()
            return max(0.0, remaining)

    def get_all_leases(self) -> Dict[str, Lease]:
        """获取所有有效租约"""
        with self._lock:
            now = time.time()
            return {
                aid: lease for aid, lease in self._leases.items()
                if not lease.is_expired(now)
            }

    def get_expired_leases(self) -> List[Lease]:
        """获取已过期租约"""
        with self._lock:
            now = time.time()
            return [
                lease for lease in self._leases.values()
                if lease.is_expired(now)
            ]

    def add_listener(
        self,
        event_type: str,
        listener: Callable[[str, Lease], None]
    ) -> None:
        """添加事件监听器"""
        if event_type in self._listeners:
            self._listeners[event_type].append(listener)

    def remove_listener(
        self,
        event_type: str,
        listener: Callable[[str, Lease], None]
    ) -> None:
        """移除事件监听器"""
        if event_type in self._listeners and listener in self._listeners[event_type]:
            self._listeners[event_type].remove(listener)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            stats = self._stats.copy()
            stats["active_leases"] = len(self._leases)
            stats["expired_leases"] = sum(
                1 for lease in self._leases.values()
                if lease.is_expired()
            )
            return stats

    def _clamp_ttl(self, ttl: int) -> int:
        """限制TTL在允许范围内"""
        return max(self._min_ttl, min(self._max_ttl, ttl))

    def _expiry_check_loop(self) -> None:
        """过期检查循环"""
        while True:
            time.sleep(self._check_interval)
            
            with self._lock:
                if not self._running:
                    break
            
            self._process_expired_leases()

    def _process_expired_leases(self) -> None:
        """处理过期租约"""
        now = time.time()
        expired = []
        
        with self._lock:
            # 检查优先队列中的过期租约
            while self._expiry_queue:
                expiry_time, agent_id = self._expiry_queue[0]
                if expiry_time > now:
                    break
                
                heapq.heappop(self._expiry_queue)
                
                # 验证租约确实过期且未被续期
                if agent_id in self._leases:
                    lease = self._leases[agent_id]
                    if lease.is_expired(now):
                        expired.append(agent_id)
                        del self._leases[agent_id]
                        self._stats["leases_expired"] += 1
                    else:
                        # 租约已被续期，重新入队
                        heapq.heappush(
                            self._expiry_queue,
                            (lease.expiry_time, agent_id)
                        )
        
        # 通知过期事件（在锁外执行）
        for agent_id in expired:
            lease = Lease(
                expiry_time=now,
                agent_id=agent_id,
                ttl_seconds=0
            )
            self._notify_event(LeaseEvent.EXPIRED, lease)

    def _notify_event(self, event_type: str, lease: Lease) -> None:
        """通知事件监听器"""
        listeners = self._listeners.get(event_type, [])
        for listener in listeners:
            try:
                listener(event_type, lease)
            except Exception:
                pass

    def __enter__(self) -> LeaseManager:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


class LeaseAwareRegistry:
    """
    带租约感知的注册表
    
    将租约管理与Agent注册结合，自动处理过期Agent
    """

    def __init__(
        self,
        lease_manager: Optional[LeaseManager] = None,
        on_lease_expired: Optional[Callable[[str], None]] = None
    ):
        self._lease_manager = lease_manager or LeaseManager()
        self._on_lease_expired = on_lease_expired
        self._agents: Dict[str, AgentMetadata] = {}
        self._lock = threading.RLock()
        
        # 注册租约过期监听器
        self._lease_manager.add_listener(
            LeaseEvent.EXPIRED,
            self._handle_lease_expired
        )

    def start(self) -> None:
        """启动"""
        self._lease_manager.start()

    def stop(self) -> None:
        """停止"""
        self._lease_manager.stop()

    def register(
        self,
        agent: AgentMetadata,
        ttl_seconds: Optional[int] = None
    ) -> Lease:
        """
        注册Agent并创建租约
        
        Args:
            agent: Agent元数据
            ttl_seconds: 租约TTL
            
        Returns:
            创建的租约
        """
        with self._lock:
            self._agents[agent.agent_id] = agent
        
        return self._lease_manager.create_lease(agent.agent_id, ttl_seconds)

    def heartbeat(
        self,
        agent_id: str,
        ttl_seconds: Optional[int] = None
    ) -> Optional[Lease]:
        """
        Agent心跳，续租
        
        Args:
            agent_id: Agent ID
            ttl_seconds: 新的TTL
            
        Returns:
            更新后的租约
        """
        return self._lease_manager.renew_lease(agent_id, ttl_seconds)

    def deregister(self, agent_id: str) -> bool:
        """
        注销Agent并撤销租约
        
        Args:
            agent_id: Agent ID
            
        Returns:
            是否成功
        """
        with self._lock:
            if agent_id in self._agents:
                del self._agents[agent_id]
        
        return self._lease_manager.revoke_lease(agent_id)

    def get_agent(self, agent_id: str) -> Optional[AgentMetadata]:
        """获取Agent（仅返回租约有效的）"""
        with self._lock:
            if not self._lease_manager.is_lease_valid(agent_id):
                return None
            return self._agents.get(agent_id)

    def get_all_agents(self) -> Dict[str, AgentMetadata]:
        """获取所有租约有效的Agent"""
        with self._lock:
            valid_leases = self._lease_manager.get_all_leases()
            return {
                aid: self._agents[aid]
                for aid in valid_leases
                if aid in self._agents
            }

    def _handle_lease_expired(self, event_type: str, lease: Lease) -> None:
        """处理租约过期"""
        agent_id = lease.agent_id
        
        with self._lock:
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                agent.status = AgentStatus.UNHEALTHY
                del self._agents[agent_id]
        
        if self._on_lease_expired:
            try:
                self._on_lease_expired(agent_id)
            except Exception:
                pass

    def __enter__(self) -> LeaseAwareRegistry:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
