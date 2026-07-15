"""
智能体注册中心模块

提供智能体档案管理、能力描述、状态跟踪、心跳检测、
负载均衡和健康检查等功能。线程安全实现。
"""

import threading
import time
import uuid
import copy
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class AgentStatus(Enum):
    """智能体状态枚举"""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class AgentCapability:
    """智能体能力描述

    描述智能体所具备的某项能力，包括能力名称、熟练程度、
    输入输出schema等信息。

    Attributes:
        name: 能力名称
        level: 能力等级 (1-10)
        description: 能力描述
        input_schema: 输入数据格式描述
        output_schema: 输出数据格式描述
    """

    def __init__(
        self,
        name: str,
        level: int = 5,
        description: str = "",
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
    ):
        if not name or not isinstance(name, str):
            raise ValueError("能力名称必须是非空字符串")
        if not 1 <= level <= 10:
            raise ValueError("能力等级必须在1到10之间")
        self.name = name
        self.level = level
        self.description = description
        self.input_schema = input_schema or {}
        self.output_schema = output_schema or {}

    def matches(self, capability_name: str, min_level: int = 1) -> bool:
        """检查是否匹配指定能力要求

        Args:
            capability_name: 要求的能力名称
            min_level: 最低能力等级

        Returns:
            是否满足要求
        """
        return self.name.lower() == capability_name.lower() and self.level >= min_level

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "name": self.name,
            "level": self.level,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCapability":
        """从字典反序列化"""
        return cls(
            name=data["name"],
            level=data.get("level", 5),
            description=data.get("description", ""),
            input_schema=data.get("input_schema"),
            output_schema=data.get("output_schema"),
        )

    def __repr__(self) -> str:
        return f"AgentCapability(name={self.name!r}, level={self.level})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AgentCapability):
            return NotImplemented
        return self.name == other.name and self.level == other.level

    def __hash__(self) -> int:
        return hash((self.name, self.level))


class AgentProfile:
    """智能体档案

    完整描述一个智能体的元信息，包括身份、类型、能力、
    状态、标签和并发限制等。

    Attributes:
        id: 智能体唯一标识
        name: 智能体名称
        type: 智能体类型
        capabilities: 能力列表
        version: 版本号
        endpoint: 服务端点
        metadata: 自定义元数据
        status: 当前状态
        tags: 标签集合
        max_concurrent_tasks: 最大并发任务数
        current_tasks: 当前任务数
        last_heartbeat: 最后心跳时间
        created_at: 创建时间
        updated_at: 更新时间
    """

    def __init__(
        self,
        id: Optional[str] = None,
        name: str = "",
        type: str = "generic",
        capabilities: Optional[List[AgentCapability]] = None,
        version: str = "1.0.0",
        endpoint: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        status: AgentStatus = AgentStatus.ONLINE,
        tags: Optional[Set[str]] = None,
        max_concurrent_tasks: int = 10,
    ):
        self.id = id or str(uuid.uuid4())
        self.name = name
        self.type = type
        self.capabilities = capabilities or []
        self.version = version
        self.endpoint = endpoint
        self.metadata = metadata or {}
        self.status = status
        self.tags = tags or set()
        self.max_concurrent_tasks = max_concurrent_tasks
        self.current_tasks = 0
        self.last_heartbeat = time.time()
        self.created_at = time.time()
        self.updated_at = time.time()

    def has_capability(self, capability_name: str, min_level: int = 1) -> bool:
        """检查是否具备指定能力"""
        return any(
            cap.matches(capability_name, min_level) for cap in self.capabilities
        )

    def get_capability(self, name: str) -> Optional[AgentCapability]:
        """获取指定能力"""
        for cap in self.capabilities:
            if cap.name.lower() == name.lower():
                return cap
        return None

    def add_capability(self, capability: AgentCapability) -> None:
        """添加能力（同名则更新）"""
        for i, cap in enumerate(self.capabilities):
            if cap.name.lower() == capability.name.lower():
                self.capabilities[i] = capability
                self.updated_at = time.time()
                return
        self.capabilities.append(capability)
        self.updated_at = time.time()

    def remove_capability(self, name: str) -> bool:
        """移除能力"""
        for i, cap in enumerate(self.capabilities):
            if cap.name.lower() == name.lower():
                self.capabilities.pop(i)
                self.updated_at = time.time()
                return True
        return False

    def is_available(self) -> bool:
        """检查智能体是否可用（在线且未满载）"""
        return (
            self.status == AgentStatus.ONLINE
            and self.current_tasks < self.max_concurrent_tasks
        )

    def load_factor(self) -> float:
        """计算负载因子 (0.0 ~ 1.0)"""
        if self.max_concurrent_tasks <= 0:
            return 1.0
        return self.current_tasks / self.max_concurrent_tasks

    def touch(self) -> None:
        """更新心跳和更新时间"""
        self.last_heartbeat = time.time()
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "capabilities": [c.to_dict() for c in self.capabilities],
            "version": self.version,
            "endpoint": self.endpoint,
            "metadata": self.metadata,
            "status": self.status.value,
            "tags": list(self.tags),
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "current_tasks": self.current_tasks,
            "last_heartbeat": self.last_heartbeat,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentProfile":
        """从字典反序列化"""
        capabilities = [
            AgentCapability.from_dict(c) for c in data.get("capabilities", [])
        ]
        profile = cls(
            id=data.get("id"),
            name=data.get("name", ""),
            type=data.get("type", "generic"),
            capabilities=capabilities,
            version=data.get("version", "1.0.0"),
            endpoint=data.get("endpoint", ""),
            metadata=data.get("metadata", {}),
            status=AgentStatus(data.get("status", "online")),
            tags=set(data.get("tags", [])),
            max_concurrent_tasks=data.get("max_concurrent_tasks", 10),
        )
        profile.current_tasks = data.get("current_tasks", 0)
        profile.last_heartbeat = data.get("last_heartbeat", time.time())
        profile.created_at = data.get("created_at", time.time())
        profile.updated_at = data.get("updated_at", time.time())
        return profile

    def __repr__(self) -> str:
        return (
            f"AgentProfile(id={self.id!r}, name={self.name!r}, "
            f"status={self.status.value}, tasks={self.current_tasks})"
        )


class AgentRegistry:
    """智能体注册中心

    线程安全的智能体注册、发现和管理中心。提供注册、注销、
    心跳检测、能力查询、标签过滤、健康检查和负载均衡等功能。

    Attributes:
        heartbeat_ttl: 心跳超时时间（秒），默认60秒
        health_check_interval: 健康检查间隔（秒），默认30秒
    """

    def __init__(self, heartbeat_ttl: float = 60.0, health_check_interval: float = 30.0):
        self._agents: Dict[str, AgentProfile] = {}
        self._lock = threading.RLock()
        self._heartbeat_ttl = heartbeat_ttl
        self._health_check_interval = health_check_interval
        self._event_listeners: List[callable] = []
        self._health_check_running = False
        self._health_check_thread: Optional[threading.Thread] = None

    def register(self, agent: AgentProfile) -> str:
        """注册智能体

        如果智能体ID已存在则更新档案。

        Args:
            agent: 智能体档案

        Returns:
            智能体ID
        """
        if not isinstance(agent, AgentProfile):
            raise TypeError("参数必须是AgentProfile实例")
        with self._lock:
            is_update = agent.id in self._agents
            agent.touch()
            self._agents[agent.id] = copy.deepcopy(agent)
            self._notify("register" if not is_update else "update", agent)
            return agent.id

    def deregister(self, agent_id: str) -> bool:
        """注销智能体

        Args:
            agent_id: 智能体ID

        Returns:
            是否成功注销
        """
        with self._lock:
            agent = self._agents.pop(agent_id, None)
            if agent:
                self._notify("deregister", agent)
                return True
            return False

    def heartbeat(self, agent_id: str) -> bool:
        """更新智能体心跳

        如果智能体存在且之前因超时被标记为OFFLINE，
        心跳会将其恢复为ONLINE状态。

        Args:
            agent_id: 智能体ID

        Returns:
            是否成功更新
        """
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent is None:
                return False
            agent.touch()
            if agent.status == AgentStatus.OFFLINE:
                agent.status = AgentStatus.ONLINE
                self._notify("heartbeat_recover", agent)
            return True

    def get_agent(self, agent_id: str) -> Optional[AgentProfile]:
        """获取智能体档案（深拷贝）

        Args:
            agent_id: 智能体ID

        Returns:
            智能体档案的副本，不存在则返回None
        """
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent:
                return copy.deepcopy(agent)
            return None

    def find_by_capability(
        self,
        capability_name: str,
        min_level: int = 1,
        available_only: bool = True,
    ) -> List[AgentProfile]:
        """按能力查找智能体

        Args:
            capability_name: 能力名称
            min_level: 最低能力等级
            available_only: 是否只返回可用智能体

        Returns:
            匹配的智能体列表
        """
        with self._lock:
            results = []
            for agent in self._agents.values():
                if agent.has_capability(capability_name, min_level):
                    if available_only and not agent.is_available():
                        continue
                    results.append(copy.deepcopy(agent))
            return results

    def find_by_tag(self, tags: Set[str], match_all: bool = False) -> List[AgentProfile]:
        """按标签查找智能体

        Args:
            tags: 要匹配的标签集合
            match_all: 是否要求匹配所有标签（True为AND，False为OR）

        Returns:
            匹配的智能体列表
        """
        if not tags:
            return self.list_agents()
        with self._lock:
            results = []
            for agent in self._agents.values():
                if match_all:
                    if tags.issubset(agent.tags):
                        results.append(copy.deepcopy(agent))
                else:
                    if agent.tags & tags:
                        results.append(copy.deepcopy(agent))
            return results

    def list_agents(
        self,
        status: Optional[AgentStatus] = None,
        agent_type: Optional[str] = None,
    ) -> List[AgentProfile]:
        """列出所有智能体

        Args:
            status: 按状态过滤
            agent_type: 按类型过滤

        Returns:
            智能体列表
        """
        with self._lock:
            results = []
            for agent in self._agents.values():
                if status and agent.status != status:
                    continue
                if agent_type and agent.type != agent_type:
                    continue
                results.append(copy.deepcopy(agent))
            return results

    def health_check(self) -> Dict[str, Any]:
        """执行健康检查

        检查所有智能体的心跳时间，将超时的智能体标记为OFFLINE。
        返回健康检查报告。

        Returns:
            健康检查报告字典
        """
        now = time.time()
        report = {
            "timestamp": now,
            "total_agents": 0,
            "healthy": 0,
            "unhealthy": 0,
            "changed": [],
        }
        with self._lock:
            report["total_agents"] = len(self._agents)
            for agent_id, agent in self._agents.items():
                elapsed = now - agent.last_heartbeat
                if elapsed > self._heartbeat_ttl:
                    if agent.status not in (AgentStatus.OFFLINE, AgentStatus.MAINTENANCE):
                        old_status = agent.status
                        agent.status = AgentStatus.OFFLINE
                        report["changed"].append({
                            "agent_id": agent_id,
                            "old_status": old_status.value,
                            "new_status": AgentStatus.OFFLINE.value,
                            "elapsed_seconds": round(elapsed, 2),
                        })
                        self._notify("health_check_fail", agent)
                    report["unhealthy"] += 1
                else:
                    report["healthy"] += 1
        return report

    def load_balance(
        self,
        capability_name: Optional[str] = None,
        min_level: int = 1,
    ) -> Optional[AgentProfile]:
        """负载均衡选择智能体

        选择当前任务数最少的可用智能体。可指定能力要求进行过滤。

        Args:
            capability_name: 要求的能力名称（可选）
            min_level: 最低能力等级

        Returns:
            负载最低的智能体，无可用智能体时返回None
        """
        with self._lock:
            candidates = []
            for agent in self._agents.values():
                if not agent.is_available():
                    continue
                if capability_name and not agent.has_capability(capability_name, min_level):
                    continue
                candidates.append(agent)

            if not candidates:
                return None

            # 按当前任务数排序，选择最少的
            candidates.sort(key=lambda a: (a.current_tasks, a.load_factor()))
            return copy.deepcopy(candidates[0])

    def update_task_count(self, agent_id: str, delta: int) -> bool:
        """更新智能体当前任务计数

        Args:
            agent_id: 智能体ID
            delta: 变化量（正数增加，负数减少）

        Returns:
            是否成功更新
        """
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent is None:
                return False
            new_count = agent.current_tasks + delta
            if new_count < 0:
                new_count = 0
            if new_count > agent.max_concurrent_tasks:
                new_count = agent.max_concurrent_tasks
            agent.current_tasks = new_count
            agent.updated_at = time.time()
            # 自动更新状态
            if agent.current_tasks >= agent.max_concurrent_tasks:
                if agent.status == AgentStatus.ONLINE:
                    agent.status = AgentStatus.BUSY
            elif agent.status == AgentStatus.BUSY:
                agent.status = AgentStatus.ONLINE
            return True

    def set_status(self, agent_id: str, status: AgentStatus) -> bool:
        """手动设置智能体状态

        Args:
            agent_id: 智能体ID
            status: 目标状态

        Returns:
            是否成功设置
        """
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent is None:
                return False
            agent.status = status
            agent.updated_at = time.time()
            self._notify("status_change", agent)
            return True

    def add_event_listener(self, listener: callable) -> None:
        """添加事件监听器

        监听器签名为: listener(event_type: str, agent: AgentProfile)
        """
        with self._lock:
            self._event_listeners.append(listener)

    def remove_event_listener(self, listener: callable) -> bool:
        """移除事件监听器"""
        with self._lock:
            try:
                self._event_listeners.remove(listener)
                return True
            except ValueError:
                return False

    def _notify(self, event_type: str, agent: AgentProfile) -> None:
        """通知所有事件监听器"""
        for listener in self._event_listeners:
            try:
                listener(event_type, agent)
            except Exception:
                pass

    def start_health_check(self) -> None:
        """启动后台健康检查线程"""
        with self._lock:
            if self._health_check_running:
                return
            self._health_check_running = True
            self._health_check_thread = threading.Thread(
                target=self._health_check_loop,
                daemon=True,
                name="agent-registry-health-check",
            )
            self._health_check_thread.start()

    def stop_health_check(self) -> None:
        """停止后台健康检查线程"""
        with self._lock:
            self._health_check_running = False
        if self._health_check_thread and self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=5.0)

    def _health_check_loop(self) -> None:
        """健康检查循环"""
        while self._health_check_running:
            time.sleep(self._health_check_interval)
            if not self._health_check_running:
                break
            try:
                self.health_check()
            except Exception:
                pass

    def get_statistics(self) -> Dict[str, Any]:
        """获取注册中心统计信息"""
        with self._lock:
            status_counts = {}
            type_counts = {}
            total_tasks = 0
            total_capacity = 0
            for agent in self._agents.values():
                status_val = agent.status.value
                status_counts[status_val] = status_counts.get(status_val, 0) + 1
                type_counts[agent.type] = type_counts.get(agent.type, 0) + 1
                total_tasks += agent.current_tasks
                total_capacity += agent.max_concurrent_tasks

            return {
                "total_agents": len(self._agents),
                "status_distribution": status_counts,
                "type_distribution": type_counts,
                "total_current_tasks": total_tasks,
                "total_capacity": total_capacity,
                "overall_load": round(total_tasks / max(total_capacity, 1), 4),
            }

    def export_agents(self) -> List[Dict[str, Any]]:
        """导出所有智能体档案"""
        with self._lock:
            return [agent.to_dict() for agent in self._agents.values()]

    def import_agents(self, agent_dicts: List[Dict[str, Any]]) -> int:
        """导入智能体档案

        Args:
            agent_dicts: 智能体档案字典列表

        Returns:
            成功导入的数量
        """
        count = 0
        for data in agent_dicts:
            try:
                profile = AgentProfile.from_dict(data)
                self.register(profile)
                count += 1
            except Exception:
                continue
        return count

    def clear(self) -> None:
        """清空注册中心"""
        with self._lock:
            self._agents.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._agents)

    def __contains__(self, agent_id: str) -> bool:
        with self._lock:
            return agent_id in self._agents

    def __repr__(self) -> str:
        with self._lock:
            return f"AgentRegistry(agents={len(self._agents)})"
