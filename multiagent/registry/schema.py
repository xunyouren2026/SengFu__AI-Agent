"""
Agent元数据Schema定义
定义Agent注册所需的元数据结构，包括ID、能力标签、地址、状态等
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set


class AgentStatus(Enum):
    """Agent状态枚举"""
    HEALTHY = "healthy"           # 健康运行中
    UNHEALTHY = "unhealthy"       # 不健康
    STARTING = "starting"         # 启动中
    STOPPING = "stopping"         # 停止中
    UNKNOWN = "unknown"           # 未知状态


class AgentRole(Enum):
    """Agent角色枚举"""
    WORKER = "worker"             # 工作节点
    COORDINATOR = "coordinator"   # 协调节点
    MANAGER = "manager"           # 管理节点
    SPECIALIST = "specialist"     # 专家节点
    GENERALIST = "generalist"     # 通用节点


@dataclass(frozen=True)
class AgentAddress:
    """Agent网络地址"""
    host: str
    port: int
    protocol: str = "http"
    path_prefix: str = ""

    def __str__(self) -> str:
        base = f"{self.protocol}://{self.host}:{self.port}"
        if self.path_prefix:
            return f"{base}/{self.path_prefix.lstrip('/')}"
        return base

    def health_endpoint(self) -> str:
        """获取健康检查端点"""
        return f"{self}/health"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "protocol": self.protocol,
            "path_prefix": self.path_prefix
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentAddress:
        return cls(
            host=data["host"],
            port=data["port"],
            protocol=data.get("protocol", "http"),
            path_prefix=data.get("path_prefix", "")
        )


@dataclass
class AgentMetadata:
    """Agent元数据"""
    agent_id: str
    name: str
    version: str
    address: AgentAddress
    capabilities: Set[str] = field(default_factory=set)
    role: AgentRole = AgentRole.WORKER
    status: AgentStatus = AgentStatus.STARTING
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: Optional[datetime] = None
    ttl_seconds: int = 30  # 默认30秒租约

    def __post_init__(self):
        if self.last_heartbeat is None:
            self.last_heartbeat = self.registered_at

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        """检查Agent是否已过期"""
        if self.last_heartbeat is None:
            return True
        if now is None:
            now = datetime.utcnow()
        elapsed = (now - self.last_heartbeat).total_seconds()
        return elapsed > self.ttl_seconds

    def update_heartbeat(self) -> None:
        """更新心跳时间"""
        self.last_heartbeat = datetime.utcnow()
        self.status = AgentStatus.HEALTHY

    def add_capability(self, capability: str) -> None:
        """添加能力标签"""
        self.capabilities.add(capability)

    def remove_capability(self, capability: str) -> None:
        """移除能力标签"""
        self.capabilities.discard(capability)

    def has_capability(self, capability: str) -> bool:
        """检查是否具有指定能力"""
        return capability in self.capabilities

    def matches_capabilities(self, required: Set[str]) -> bool:
        """检查是否匹配所有必需的能力"""
        return required.issubset(self.capabilities)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "version": self.version,
            "address": self.address.to_dict(),
            "capabilities": list(self.capabilities),
            "role": self.role.value,
            "status": self.status.value,
            "labels": self.labels,
            "metadata": self.metadata,
            "registered_at": self.registered_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "ttl_seconds": self.ttl_seconds
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentMetadata:
        """从字典创建实例"""
        return cls(
            agent_id=data["agent_id"],
            name=data["name"],
            version=data["version"],
            address=AgentAddress.from_dict(data["address"]),
            capabilities=set(data.get("capabilities", [])),
            role=AgentRole(data.get("role", "worker")),
            status=AgentStatus(data.get("status", "starting")),
            labels=data.get("labels", {}),
            metadata=data.get("metadata", {}),
            registered_at=datetime.fromisoformat(data["registered_at"]),
            last_heartbeat=datetime.fromisoformat(data["last_heartbeat"]) if data.get("last_heartbeat") else None,
            ttl_seconds=data.get("ttl_seconds", 30)
        )

    def to_json(self) -> str:
        """序列化为JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> AgentMetadata:
        """从JSON反序列化"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class ServiceEndpoint:
    """服务端点定义"""
    name: str
    path: str
    method: str = "GET"
    version: str = "1.0.0"
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "method": self.method,
            "version": self.version,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ServiceEndpoint:
        return cls(
            name=data["name"],
            path=data["path"],
            method=data.get("method", "GET"),
            version=data.get("version", "1.0.0"),
            input_schema=data.get("input_schema"),
            output_schema=data.get("output_schema"),
            description=data.get("description", "")
        )


@dataclass
class AgentRegistration:
    """Agent注册请求"""
    metadata: AgentMetadata
    endpoints: List[ServiceEndpoint] = field(default_factory=list)
    api_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "endpoints": [ep.to_dict() for ep in self.endpoints],
            "api_version": self.api_version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentRegistration:
        return cls(
            metadata=AgentMetadata.from_dict(data["metadata"]),
            endpoints=[ServiceEndpoint.from_dict(ep) for ep in data.get("endpoints", [])],
            api_version=data.get("api_version", "1.0.0")
        )


@dataclass
class DiscoveryQuery:
    """服务发现查询"""
    required_capabilities: Set[str] = field(default_factory=set)
    role: Optional[AgentRole] = None
    status: Optional[AgentStatus] = None
    labels: Dict[str, str] = field(default_factory=dict)
    min_version: Optional[str] = None
    max_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required_capabilities": list(self.required_capabilities),
            "role": self.role.value if self.role else None,
            "status": self.status.value if self.status else None,
            "labels": self.labels,
            "min_version": self.min_version,
            "max_version": self.max_version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DiscoveryQuery:
        return cls(
            required_capabilities=set(data.get("required_capabilities", [])),
            role=AgentRole(data["role"]) if data.get("role") else None,
            status=AgentStatus(data["status"]) if data.get("status") else None,
            labels=data.get("labels", {}),
            min_version=data.get("min_version"),
            max_version=data.get("max_version")
        )


@dataclass
class DiscoveryResult:
    """服务发现结果"""
    agents: List[AgentMetadata]
    total_count: int
    query_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agents": [agent.to_dict() for agent in self.agents],
            "total_count": self.total_count,
            "query_time_ms": self.query_time_ms
        }


@dataclass
class HealthStatus:
    """健康状态报告"""
    agent_id: str
    status: AgentStatus
    last_check: datetime
    response_time_ms: float
    error_message: Optional[str] = None
    consecutive_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "last_check": self.last_check.isoformat(),
            "response_time_ms": self.response_time_ms,
            "error_message": self.error_message,
            "consecutive_failures": self.consecutive_failures
        }
