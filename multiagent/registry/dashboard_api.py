"""
注册中心仪表盘API
查询当前在线Agent列表
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from .schema import AgentMetadata, AgentStatus, AgentRole


@dataclass
class AgentSummary:
    """Agent摘要信息"""
    agent_id: str
    name: str
    version: str
    role: str
    status: str
    address: str
    capabilities_count: int
    last_heartbeat: Optional[str] = None
    uptime_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "version": self.version,
            "role": self.role,
            "status": self.status,
            "address": self.address,
            "capabilities_count": self.capabilities_count,
            "last_heartbeat": self.last_heartbeat,
            "uptime_seconds": self.uptime_seconds
        }


@dataclass
class DashboardStats:
    """仪表盘统计信息"""
    total_agents: int = 0
    healthy_agents: int = 0
    unhealthy_agents: int = 0
    starting_agents: int = 0
    stopping_agents: int = 0
    unknown_agents: int = 0
    
    agents_by_role: Dict[str, int] = field(default_factory=dict)
    agents_by_capability: Dict[str, int] = field(default_factory=dict)
    
    avg_uptime_seconds: float = 0.0
    total_capabilities: int = 0
    recent_registrations: int = 0
    recent_deregistrations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_agents": self.total_agents,
            "healthy_agents": self.healthy_agents,
            "unhealthy_agents": self.unhealthy_agents,
            "starting_agents": self.starting_agents,
            "stopping_agents": self.stopping_agents,
            "unknown_agents": self.unknown_agents,
            "agents_by_role": self.agents_by_role,
            "agents_by_capability": self.agents_by_capability,
            "avg_uptime_seconds": self.avg_uptime_seconds,
            "total_capabilities": self.total_capabilities,
            "recent_registrations": self.recent_registrations,
            "recent_deregistrations": self.recent_deregistrations
        }


class DashboardAPI:
    """
    注册中心仪表盘API
    
    提供查询当前在线Agent列表和统计信息的功能
    """

    def __init__(self, registry_service=None):
        """
        初始化仪表盘API
        
        Args:
            registry_service: 注册中心服务实例
        """
        self._registry = registry_service
        self._recent_events: List[Dict[str, Any]] = []
        self._max_events = 1000

    def set_registry(self, registry_service) -> None:
        """设置注册中心服务"""
        self._registry = registry_service

    def get_all_agents(
        self,
        status: Optional[str] = None,
        role: Optional[str] = None,
        capability: Optional[str] = None
    ) -> List[AgentSummary]:
        """
        获取所有Agent摘要信息
        
        Args:
            status: 按状态过滤
            role: 按角色过滤
            capability: 按能力过滤
            
        Returns:
            Agent摘要列表
        """
        if not self._registry:
            return []
        
        agents = self._registry.get_all_agents()
        summaries = []
        
        now = datetime.utcnow()
        
        for agent in agents:
            # 应用过滤
            if status and agent.status.value != status:
                continue
            if role and agent.role.value != role:
                continue
            if capability and capability not in agent.capabilities:
                continue
            
            # 计算运行时间
            uptime = None
            if agent.last_heartbeat:
                uptime = (now - agent.registered_at).total_seconds()
            
            summary = AgentSummary(
                agent_id=agent.agent_id,
                name=agent.name,
                version=agent.version,
                role=agent.role.value,
                status=agent.status.value,
                address=str(agent.address),
                capabilities_count=len(agent.capabilities),
                last_heartbeat=agent.last_heartbeat.isoformat() if agent.last_heartbeat else None,
                uptime_seconds=uptime
            )
            summaries.append(summary)
        
        return summaries

    def get_agent_detail(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        获取Agent详细信息
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent详细信息
        """
        if not self._registry:
            return None
        
        agent = self._registry.get_agent(agent_id)
        if not agent:
            return None
        
        return agent.to_dict()

    def get_stats(self) -> DashboardStats:
        """
        获取仪表盘统计信息
        
        Returns:
            统计信息
        """
        if not self._registry:
            return DashboardStats()
        
        agents = self._registry.get_all_agents()
        stats = DashboardStats()
        
        stats.total_agents = len(agents)
        
        all_capabilities: Set[str] = set()
        uptimes = []
        
        now = datetime.utcnow()
        
        for agent in agents:
            # 状态统计
            if agent.status == AgentStatus.HEALTHY:
                stats.healthy_agents += 1
            elif agent.status == AgentStatus.UNHEALTHY:
                stats.unhealthy_agents += 1
            elif agent.status == AgentStatus.STARTING:
                stats.starting_agents += 1
            elif agent.status == AgentStatus.STOPPING:
                stats.stopping_agents += 1
            else:
                stats.unknown_agents += 1
            
            # 角色统计
            role = agent.role.value
            stats.agents_by_role[role] = stats.agents_by_role.get(role, 0) + 1
            
            # 能力统计
            for cap in agent.capabilities:
                all_capabilities.add(cap)
                stats.agents_by_capability[cap] = stats.agents_by_capability.get(cap, 0) + 1
            
            # 运行时间
            if agent.last_heartbeat:
                uptime = (now - agent.registered_at).total_seconds()
                uptimes.append(uptime)
        
        stats.total_capabilities = len(all_capabilities)
        
        if uptimes:
            stats.avg_uptime_seconds = sum(uptimes) / len(uptimes)
        
        # 获取注册中心统计
        registry_stats = self._registry.get_stats()
        stats.recent_registrations = registry_stats.get("total_registered", 0)
        stats.recent_deregistrations = registry_stats.get("total_deregistered", 0)
        
        return stats

    def get_health_overview(self) -> Dict[str, Any]:
        """
        获取健康状态概览
        
        Returns:
            健康状态概览
        """
        stats = self.get_stats()
        
        total = stats.total_agents
        if total == 0:
            health_percentage = 100.0
        else:
            health_percentage = (stats.healthy_agents / total) * 100
        
        return {
            "health_percentage": round(health_percentage, 2),
            "total_agents": total,
            "healthy_agents": stats.healthy_agents,
            "unhealthy_agents": stats.unhealthy_agents,
            "status": "healthy" if health_percentage >= 80 else "warning" if health_percentage >= 50 else "critical"
        }

    def get_top_capabilities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取最常用的能力标签
        
        Args:
            limit: 返回数量限制
            
        Returns:
            能力标签统计列表
        """
        stats = self.get_stats()
        
        capabilities = [
            {"capability": cap, "count": count}
            for cap, count in stats.agents_by_capability.items()
        ]
        
        capabilities.sort(key=lambda x: x["count"], reverse=True)
        return capabilities[:limit]

    def get_recent_events(
        self,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        获取最近的事件
        
        Args:
            event_type: 事件类型过滤
            limit: 返回数量限制
            
        Returns:
            事件列表
        """
        events = self._recent_events
        
        if event_type:
            events = [e for e in events if e.get("type") == event_type]
        
        return events[-limit:]

    def record_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        记录事件
        
        Args:
            event_type: 事件类型
            data: 事件数据
        """
        event = {
            "type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        
        self._recent_events.append(event)
        
        # 限制事件数量
        if len(self._recent_events) > self._max_events:
            self._recent_events = self._recent_events[-self._max_events:]

    def search_agents(
        self,
        query: str,
        limit: int = 20
    ) -> List[AgentSummary]:
        """
        搜索Agent
        
        Args:
            query: 搜索关键词
            limit: 返回数量限制
            
        Returns:
            匹配的Agent列表
        """
        if not self._registry:
            return []
        
        query = query.lower()
        agents = self._registry.get_all_agents()
        matches = []
        
        now = datetime.utcnow()
        
        for agent in agents:
            # 检查是否匹配
            if (query in agent.agent_id.lower() or
                query in agent.name.lower() or
                query in agent.version.lower() or
                any(query in cap.lower() for cap in agent.capabilities)):
                
                uptime = None
                if agent.last_heartbeat:
                    uptime = (now - agent.registered_at).total_seconds()
                
                summary = AgentSummary(
                    agent_id=agent.agent_id,
                    name=agent.name,
                    version=agent.version,
                    role=agent.role.value,
                    status=agent.status.value,
                    address=str(agent.address),
                    capabilities_count=len(agent.capabilities),
                    last_heartbeat=agent.last_heartbeat.isoformat() if agent.last_heartbeat else None,
                    uptime_seconds=uptime
                )
                matches.append(summary)
                
                if len(matches) >= limit:
                    break
        
        return matches

    def export_data(self, format: str = "json") -> str:
        """
        导出数据
        
        Args:
            format: 导出格式 (json)
            
        Returns:
            导出的数据字符串
        """
        if format == "json":
            data = {
                "agents": [a.to_dict() for a in self.get_all_agents()],
                "stats": self.get_stats().to_dict(),
                "health": self.get_health_overview(),
                "exported_at": datetime.utcnow().isoformat()
            }
            return json.dumps(data, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")


class DashboardAPIServer:
    """
    仪表盘API服务器
    
    提供HTTP接口访问仪表盘数据
    """

    def __init__(self, dashboard_api: DashboardAPI, host: str = "0.0.0.0", port: int = 8080):
        """
        初始化API服务器
        
        Args:
            dashboard_api: 仪表盘API实例
            host: 监听地址
            port: 监听端口
        """
        self._api = dashboard_api
        self._host = host
        self._port = port
        self._running = False

    def get_routes(self) -> Dict[str, Callable]:
        """
        获取路由映射
        
        Returns:
            路由字典
        """
        return {
            "/api/dashboard/agents": self._handle_agents,
            "/api/dashboard/agents/<agent_id>": self._handle_agent_detail,
            "/api/dashboard/stats": self._handle_stats,
            "/api/dashboard/health": self._handle_health,
            "/api/dashboard/capabilities": self._handle_capabilities,
            "/api/dashboard/events": self._handle_events,
            "/api/dashboard/search": self._handle_search,
            "/api/dashboard/export": self._handle_export,
        }

    def _handle_agents(self, **kwargs) -> Dict[str, Any]:
        """处理获取Agent列表请求"""
        status = kwargs.get("status")
        role = kwargs.get("role")
        capability = kwargs.get("capability")
        
        agents = self._api.get_all_agents(status, role, capability)
        return {
            "agents": [a.to_dict() for a in agents],
            "total": len(agents)
        }

    def _handle_agent_detail(self, agent_id: str, **kwargs) -> Dict[str, Any]:
        """处理获取Agent详情请求"""
        detail = self._api.get_agent_detail(agent_id)
        if detail:
            return {"agent": detail}
        return {"error": "Agent not found"}, 404

    def _handle_stats(self, **kwargs) -> Dict[str, Any]:
        """处理获取统计信息请求"""
        return self._api.get_stats().to_dict()

    def _handle_health(self, **kwargs) -> Dict[str, Any]:
        """处理获取健康概览请求"""
        return self._api.get_health_overview()

    def _handle_capabilities(self, **kwargs) -> Dict[str, Any]:
        """处理获取能力标签请求"""
        limit = int(kwargs.get("limit", 10))
        return {
            "capabilities": self._api.get_top_capabilities(limit)
        }

    def _handle_events(self, **kwargs) -> Dict[str, Any]:
        """处理获取事件请求"""
        event_type = kwargs.get("type")
        limit = int(kwargs.get("limit", 100))
        return {
            "events": self._api.get_recent_events(event_type, limit)
        }

    def _handle_search(self, **kwargs) -> Dict[str, Any]:
        """处理搜索请求"""
        query = kwargs.get("q", "")
        limit = int(kwargs.get("limit", 20))
        
        if not query:
            return {"error": "Query parameter 'q' is required"}, 400
        
        agents = self._api.search_agents(query, limit)
        return {
            "agents": [a.to_dict() for a in agents],
            "total": len(agents)
        }

    def _handle_export(self, **kwargs) -> tuple:
        """处理导出请求"""
        format = kwargs.get("format", "json")
        data = self._api.export_data(format)
        return data, 200, {"Content-Type": "application/json"}
