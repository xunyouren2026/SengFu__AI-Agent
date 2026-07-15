"""
注册中心服务
基于内存KV存储和健康检查的Agent注册与发现服务
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .schema import (
    AgentMetadata, AgentRegistration, AgentStatus,
    DiscoveryQuery, DiscoveryResult, HealthStatus
)


class RegistryError(Exception):
    """注册中心错误"""
    pass


class AgentNotFoundError(RegistryError):
    """Agent未找到错误"""
    pass


class DuplicateAgentError(RegistryError):
    """重复Agent错误"""
    pass


class RegistryService:
    """
    Agent注册中心服务
    
    提供Agent注册、发现、心跳维护、健康检查等核心功能
    基于内存KV存储实现，支持索引加速查询
    """

    def __init__(
        self,
        default_ttl: int = 30,
        cleanup_interval: float = 10.0,
        max_workers: int = 4
    ):
        self._default_ttl = default_ttl
        self._cleanup_interval = cleanup_interval
        
        # 核心存储: agent_id -> AgentMetadata
        self._agents: Dict[str, AgentMetadata] = {}
        
        # 索引结构加速查询
        self._capability_index: Dict[str, Set[str]] = {}  # capability -> set(agent_id)
        self._role_index: Dict[str, Set[str]] = {}        # role -> set(agent_id)
        self._status_index: Dict[str, Set[str]] = {}      # status -> set(agent_id)
        
        # 健康状态缓存
        self._health_status: Dict[str, HealthStatus] = {}
        
        # 事件监听器
        self._event_listeners: List[Callable[[str, AgentMetadata], None]] = []
        
        # 线程安全锁
        self._lock = threading.RLock()
        
        # 后台清理线程
        self._cleanup_thread: Optional[threading.Thread] = None
        self._running = False
        
        # 线程池用于并行健康检查
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 统计信息
        self._stats = {
            "total_registered": 0,
            "total_deregistered": 0,
            "total_heartbeats": 0,
            "total_discoveries": 0
        }

    def start(self) -> None:
        """启动注册中心服务"""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_expired_agents,
                daemon=True,
                name="RegistryCleanup"
            )
            self._cleanup_thread.start()

    def stop(self) -> None:
        """停止注册中心服务"""
        with self._lock:
            self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)
        self._executor.shutdown(wait=True)

    def register(self, registration: AgentRegistration) -> bool:
        """
        注册Agent
        
        Args:
            registration: Agent注册信息
            
        Returns:
            注册是否成功
            
        Raises:
            DuplicateAgentError: 如果Agent已存在且不允许覆盖
        """
        metadata = registration.metadata
        agent_id = metadata.agent_id
        
        with self._lock:
            if agent_id in self._agents:
                raise DuplicateAgentError(f"Agent {agent_id} already registered")
            
            # 设置默认TTL
            if metadata.ttl_seconds <= 0:
                metadata.ttl_seconds = self._default_ttl
            
            # 存储Agent
            self._agents[agent_id] = metadata
            
            # 更新索引
            self._update_index_add(metadata)
            
            # 初始化健康状态
            self._health_status[agent_id] = HealthStatus(
                agent_id=agent_id,
                status=AgentStatus.STARTING,
                last_check=datetime.utcnow(),
                response_time_ms=0.0
            )
            
            self._stats["total_registered"] += 1
        
        # 触发注册事件
        self._notify_event("registered", metadata)
        return True

    def deregister(self, agent_id: str) -> bool:
        """
        注销Agent
        
        Args:
            agent_id: Agent ID
            
        Returns:
            是否成功注销
        """
        with self._lock:
            if agent_id not in self._agents:
                return False
            
            metadata = self._agents[agent_id]
            
            # 从存储中移除
            del self._agents[agent_id]
            
            # 从索引中移除
            self._update_index_remove(metadata)
            
            # 移除健康状态
            if agent_id in self._health_status:
                del self._health_status[agent_id]
            
            self._stats["total_deregistered"] += 1
        
        # 触发注销事件
        self._notify_event("deregistered", metadata)
        return True

    def heartbeat(self, agent_id: str) -> bool:
        """
        处理Agent心跳
        
        Args:
            agent_id: Agent ID
            
        Returns:
            心跳是否成功处理
        """
        with self._lock:
            if agent_id not in self._agents:
                return False
            
            agent = self._agents[agent_id]
            agent.update_heartbeat()
            
            # 更新状态索引
            self._update_status_index(agent_id, AgentStatus.HEALTHY)
            
            self._stats["total_heartbeats"] += 1
            return True

    def discover(self, query: DiscoveryQuery) -> DiscoveryResult:
        """
        服务发现查询
        
        Args:
            query: 查询条件
            
        Returns:
            查询结果
        """
        start_time = time.time()
        
        with self._lock:
            candidates = self._agents.copy()
            
            # 按能力过滤
            if query.required_capabilities:
                capable_ids = None
                for cap in query.required_capabilities:
                    if cap in self._capability_index:
                        if capable_ids is None:
                            capable_ids = self._capability_index[cap].copy()
                        else:
                            capable_ids &= self._capability_index[cap]
                    else:
                        capable_ids = set()
                        break
                
                if capable_ids is not None:
                    candidates = {
                        aid: meta for aid, meta in candidates.items()
                        if aid in capable_ids
                    }
            
            # 按角色过滤
            if query.role:
                role_key = query.role.value
                if role_key in self._role_index:
                    role_ids = self._role_index[role_key]
                    candidates = {
                        aid: meta for aid, meta in candidates.items()
                        if aid in role_ids
                    }
                else:
                    candidates = {}
            
            # 按状态过滤
            if query.status:
                status_key = query.status.value
                if status_key in self._status_index:
                    status_ids = self._status_index[status_key]
                    candidates = {
                        aid: meta for aid, meta in candidates.items()
                        if aid in status_ids
                    }
                else:
                    candidates = {}
            
            # 按标签过滤
            if query.labels:
                candidates = {
                    aid: meta for aid, meta in candidates.items()
                    if all(
                        meta.labels.get(k) == v
                        for k, v in query.labels.items()
                    )
                }
            
            # 按版本过滤
            if query.min_version or query.max_version:
                filtered = {}
                for aid, meta in candidates.items():
                    version = meta.version
                    if query.min_version and version < query.min_version:
                        continue
                    if query.max_version and version > query.max_version:
                        continue
                    filtered[aid] = meta
                candidates = filtered
            
            # 过滤掉过期的Agent
            now = datetime.utcnow()
            active_agents = [
                meta for meta in candidates.values()
                if not meta.is_expired(now)
            ]
            
            self._stats["total_discoveries"] += 1
        
        query_time_ms = (time.time() - start_time) * 1000
        
        return DiscoveryResult(
            agents=active_agents,
            total_count=len(active_agents),
            query_time_ms=query_time_ms
        )

    def get_agent(self, agent_id: str) -> Optional[AgentMetadata]:
        """获取指定Agent的元数据"""
        with self._lock:
            return self._agents.get(agent_id)

    def get_all_agents(self) -> List[AgentMetadata]:
        """获取所有注册的Agent"""
        with self._lock:
            return list(self._agents.values())

    def get_healthy_agents(self) -> List[AgentMetadata]:
        """获取所有健康的Agent"""
        with self._lock:
            now = datetime.utcnow()
            return [
                meta for meta in self._agents.values()
                if meta.status == AgentStatus.HEALTHY and not meta.is_expired(now)
            ]

    def update_health_status(self, agent_id: str, status: HealthStatus) -> bool:
        """更新Agent健康状态"""
        with self._lock:
            if agent_id not in self._agents:
                return False
            
            self._health_status[agent_id] = status
            
            # 同步更新Agent状态
            agent = self._agents[agent_id]
            agent.status = status.status
            self._update_status_index(agent_id, status.status)
            
            return True

    def get_health_status(self, agent_id: str) -> Optional[HealthStatus]:
        """获取Agent健康状态"""
        with self._lock:
            return self._health_status.get(agent_id)

    def add_event_listener(self, listener: Callable[[str, AgentMetadata], None]) -> None:
        """添加事件监听器"""
        self._event_listeners.append(listener)

    def remove_event_listener(self, listener: Callable[[str, AgentMetadata], None]) -> None:
        """移除事件监听器"""
        if listener in self._event_listeners:
            self._event_listeners.remove(listener)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            stats = self._stats.copy()
            stats["current_agents"] = len(self._agents)
            stats["healthy_agents"] = sum(
                1 for meta in self._agents.values()
                if meta.status == AgentStatus.HEALTHY
            )
            return stats

    def _update_index_add(self, metadata: AgentMetadata) -> None:
        """添加索引"""
        agent_id = metadata.agent_id
        
        # 能力索引
        for cap in metadata.capabilities:
            if cap not in self._capability_index:
                self._capability_index[cap] = set()
            self._capability_index[cap].add(agent_id)
        
        # 角色索引
        role_key = metadata.role.value
        if role_key not in self._role_index:
            self._role_index[role_key] = set()
        self._role_index[role_key].add(agent_id)
        
        # 状态索引
        status_key = metadata.status.value
        if status_key not in self._status_index:
            self._status_index[status_key] = set()
        self._status_index[status_key].add(agent_id)

    def _update_index_remove(self, metadata: AgentMetadata) -> None:
        """移除索引"""
        agent_id = metadata.agent_id
        
        # 能力索引
        for cap in metadata.capabilities:
            if cap in self._capability_index:
                self._capability_index[cap].discard(agent_id)
        
        # 角色索引
        role_key = metadata.role.value
        if role_key in self._role_index:
            self._role_index[role_key].discard(agent_id)
        
        # 状态索引
        status_key = metadata.status.value
        if status_key in self._status_index:
            self._status_index[status_key].discard(agent_id)

    def _update_status_index(self, agent_id: str, new_status: AgentStatus) -> None:
        """更新状态索引"""
        if agent_id not in self._agents:
            return
        
        old_status = self._agents[agent_id].status
        
        # 从旧状态移除
        old_key = old_status.value
        if old_key in self._status_index:
            self._status_index[old_key].discard(agent_id)
        
        # 添加到新状态
        new_key = new_status.value
        if new_key not in self._status_index:
            self._status_index[new_key] = set()
        self._status_index[new_key].add(agent_id)

    def _cleanup_expired_agents(self) -> None:
        """后台清理过期Agent"""
        while True:
            time.sleep(self._cleanup_interval)
            
            with self._lock:
                if not self._running:
                    break
            
            now = datetime.utcnow()
            expired_agents = []
            
            with self._lock:
                for agent_id, metadata in list(self._agents.items()):
                    if metadata.is_expired(now):
                        expired_agents.append(agent_id)
            
            for agent_id in expired_agents:
                self.deregister(agent_id)

    def _notify_event(self, event_type: str, metadata: AgentMetadata) -> None:
        """通知事件监听器"""
        for listener in self._event_listeners:
            try:
                listener(event_type, metadata)
            except Exception:
                pass  # 忽略监听器错误

    def __enter__(self) -> RegistryService:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
