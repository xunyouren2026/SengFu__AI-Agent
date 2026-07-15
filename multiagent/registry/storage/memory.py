"""
内存存储后端
用于开发测试
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Set

from ..schema import AgentMetadata, AgentRegistration


class MemoryStorage:
    """
    内存存储后端
    
    基于内存的键值存储，用于开发和测试环境
    数据在进程重启后会丢失
    """

    def __init__(self):
        # 主存储: agent_id -> AgentMetadata
        self._agents: Dict[str, AgentMetadata] = {}
        
        # 索引结构
        self._capability_index: Dict[str, Set[str]] = {}
        self._role_index: Dict[str, Set[str]] = {}
        self._status_index: Dict[str, Set[str]] = {}
        
        # 元数据存储
        self._metadata: Dict[str, Any] = {}
        
        # 线程锁
        self._lock = threading.RLock()

    def save_agent(self, agent: AgentMetadata) -> bool:
        """
        保存Agent元数据
        
        Args:
            agent: Agent元数据
            
        Returns:
            是否成功保存
        """
        with self._lock:
            # 如果已存在，先移除旧索引
            if agent.agent_id in self._agents:
                self._remove_from_indices(agent.agent_id)
            
            # 保存Agent
            self._agents[agent.agent_id] = agent
            
            # 更新索引
            self._add_to_indices(agent)
            
            return True

    def get_agent(self, agent_id: str) -> Optional[AgentMetadata]:
        """
        获取Agent元数据
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent元数据，不存在则返回None
        """
        with self._lock:
            return self._agents.get(agent_id)

    def get_all_agents(self) -> List[AgentMetadata]:
        """
        获取所有Agent
        
        Returns:
            Agent列表
        """
        with self._lock:
            return list(self._agents.values())

    def delete_agent(self, agent_id: str) -> bool:
        """
        删除Agent
        
        Args:
            agent_id: Agent ID
            
        Returns:
            是否成功删除
        """
        with self._lock:
            if agent_id not in self._agents:
                return False
            
            # 移除索引
            self._remove_from_indices(agent_id)
            
            # 删除Agent
            del self._agents[agent_id]
            
            return True

    def find_by_capability(self, capability: str) -> List[AgentMetadata]:
        """
        按能力查找Agent
        
        Args:
            capability: 能力标签
            
        Returns:
            匹配的Agent列表
        """
        with self._lock:
            agent_ids = self._capability_index.get(capability, set())
            return [self._agents[aid] for aid in agent_ids if aid in self._agents]

    def find_by_role(self, role: str) -> List[AgentMetadata]:
        """
        按角色查找Agent
        
        Args:
            role: 角色
            
        Returns:
            匹配的Agent列表
        """
        with self._lock:
            agent_ids = self._role_index.get(role, set())
            return [self._agents[aid] for aid in agent_ids if aid in self._agents]

    def find_by_status(self, status: str) -> List[AgentMetadata]:
        """
        按状态查找Agent
        
        Args:
            status: 状态
            
        Returns:
            匹配的Agent列表
        """
        with self._lock:
            agent_ids = self._status_index.get(status, set())
            return [self._agents[aid] for aid in agent_ids if aid in self._agents]

    def exists(self, agent_id: str) -> bool:
        """
        检查Agent是否存在
        
        Args:
            agent_id: Agent ID
            
        Returns:
            是否存在
        """
        with self._lock:
            return agent_id in self._agents

    def count(self) -> int:
        """
        获取Agent数量
        
        Returns:
            Agent数量
        """
        with self._lock:
            return len(self._agents)

    def clear(self) -> None:
        """清空所有数据"""
        with self._lock:
            self._agents.clear()
            self._capability_index.clear()
            self._role_index.clear()
            self._status_index.clear()
            self._metadata.clear()

    def set_metadata(self, key: str, value: Any) -> None:
        """设置元数据"""
        with self._lock:
            self._metadata[key] = value

    def get_metadata(self, key: str) -> Optional[Any]:
        """获取元数据"""
        with self._lock:
            return self._metadata.get(key)

    def _add_to_indices(self, agent: AgentMetadata) -> None:
        """添加索引"""
        agent_id = agent.agent_id
        
        # 能力索引
        for cap in agent.capabilities:
            if cap not in self._capability_index:
                self._capability_index[cap] = set()
            self._capability_index[cap].add(agent_id)
        
        # 角色索引
        role_key = agent.role.value
        if role_key not in self._role_index:
            self._role_index[role_key] = set()
        self._role_index[role_key].add(agent_id)
        
        # 状态索引
        status_key = agent.status.value
        if status_key not in self._status_index:
            self._status_index[status_key] = set()
        self._status_index[status_key].add(agent_id)

    def _remove_from_indices(self, agent_id: str) -> None:
        """移除索引"""
        agent = self._agents.get(agent_id)
        if not agent:
            return
        
        # 能力索引
        for cap in agent.capabilities:
            if cap in self._capability_index:
                self._capability_index[cap].discard(agent_id)
        
        # 角色索引
        role_key = agent.role.value
        if role_key in self._role_index:
            self._role_index[role_key].discard(agent_id)
        
        # 状态索引
        status_key = agent.status.value
        if status_key in self._status_index:
            self._status_index[status_key].discard(agent_id)

    def get_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        with self._lock:
            return {
                "total_agents": len(self._agents),
                "total_capabilities": len(self._capability_index),
                "total_roles": len(self._role_index),
                "total_statuses": len(self._status_index),
                "memory_size_estimate": len(str(self._agents))
            }


class MemoryStorageWithSnapshot(MemoryStorage):
    """
    带快照功能的内存存储
    
    支持数据快照和恢复，可用于测试数据持久化逻辑
    """

    def __init__(self):
        super().__init__()
        self._snapshots: Dict[str, Dict[str, Any]] = {}

    def create_snapshot(self, name: str) -> bool:
        """
        创建快照
        
        Args:
            name: 快照名称
            
        Returns:
            是否成功创建
        """
        with self._lock:
            self._snapshots[name] = {
                "agents": dict(self._agents),
                "capability_index": {k: set(v) for k, v in self._capability_index.items()},
                "role_index": {k: set(v) for k, v in self._role_index.items()},
                "status_index": {k: set(v) for k, v in self._status_index.items()},
                "metadata": dict(self._metadata)
            }
            return True

    def restore_snapshot(self, name: str) -> bool:
        """
        恢复快照
        
        Args:
            name: 快照名称
            
        Returns:
            是否成功恢复
        """
        with self._lock:
            if name not in self._snapshots:
                return False
            
            snapshot = self._snapshots[name]
            self._agents = dict(snapshot["agents"])
            self._capability_index = {k: set(v) for k, v in snapshot["capability_index"].items()}
            self._role_index = {k: set(v) for k, v in snapshot["role_index"].items()}
            self._status_index = {k: set(v) for k, v in snapshot["status_index"].items()}
            self._metadata = dict(snapshot["metadata"])
            return True

    def delete_snapshot(self, name: str) -> bool:
        """
        删除快照
        
        Args:
            name: 快照名称
            
        Returns:
            是否成功删除
        """
        with self._lock:
            if name in self._snapshots:
                del self._snapshots[name]
                return True
            return False

    def list_snapshots(self) -> List[str]:
        """
        列出所有快照
        
        Returns:
            快照名称列表
        """
        with self._lock:
            return list(self._snapshots.keys())
