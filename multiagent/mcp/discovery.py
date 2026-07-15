"""MCP服务发现模块。

本模块实现MCP服务的自动发现，通过配置文件或环境变量发现可用服务。
支持多种发现机制和动态服务注册。
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional, Callable, Dict, Any, List, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class DiscoverySource(Enum):
    """发现来源枚举。"""
    CONFIG_FILE = "config_file"
    ENVIRONMENT = "environment"
    REGISTRY = "registry"
    MANUAL = "manual"
    AUTO = "auto"


class ServiceStatus(Enum):
    """服务状态枚举。"""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNREACHABLE = "unreachable"


@dataclass
class ServiceEndpoint:
    """服务端点信息。
    
    Attributes:
        name: 服务名称
        url: 服务URL
        transport: 传输类型（stdio, websocket, sse）
        description: 服务描述
        capabilities: 服务能力
        metadata: 元数据
    """
    name: str
    url: str = ""
    transport: str = "stdio"
    description: str = ""
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "name": self.name,
            "url": self.url,
            "transport": self.transport,
            "description": self.description,
            "capabilities": self.capabilities,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ServiceEndpoint:
        """从字典创建。"""
        return cls(
            name=data.get("name", ""),
            url=data.get("url", ""),
            transport=data.get("transport", "stdio"),
            description=data.get("description", ""),
            capabilities=data.get("capabilities", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class ServiceInfo:
    """服务完整信息。
    
    Attributes:
        endpoint: 服务端点
        status: 服务状态
        last_check: 最后检查时间
        response_time: 响应时间（毫秒）
        error_message: 错误消息
    """
    endpoint: ServiceEndpoint
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_check: Optional[float] = None
    response_time: Optional[float] = None
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "endpoint": self.endpoint.to_dict(),
            "status": self.status.value,
            "last_check": self.last_check,
            "response_time": self.response_time,
            "error_message": self.error_message
        }


class ServiceDiscovery:
    """MCP服务发现。
    
    通过多种机制发现和注册MCP服务。
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        env_prefix: str = "MCP_",
        auto_discover: bool = True
    ):
        """初始化服务发现。
        
        Args:
            config_path: 配置文件路径
            env_prefix: 环境变量前缀
            auto_discover: 是否自动发现
        """
        self.config_path = config_path
        self.env_prefix = env_prefix
        self.auto_discover = auto_discover
        
        self._services: Dict[str, ServiceInfo] = {}
        self._discovery_sources: Dict[str, DiscoverySource] = {}
        self._lock = threading.RLock()
        
        # 健康检查
        self._health_check_interval = 60.0
        self._health_check_thread: Optional[threading.Thread] = None
        self._health_check_running = False
        self._health_check_callback: Optional[Callable[[str, ServiceStatus], None]] = None
        
        # 自动发现
        if auto_discover:
            self._auto_discover()
    
    def _auto_discover(self) -> None:
        """执行自动发现。"""
        # 从配置文件发现
        if self.config_path:
            self.discover_from_config(self.config_path)
        else:
            # 尝试默认配置路径
            default_paths = [
                "mcp_config.json",
                "mcp.json",
                ".mcp/config.json",
                os.path.expanduser("~/.mcp/config.json")
            ]
            for path in default_paths:
                if os.path.exists(path):
                    self.discover_from_config(path)
                    break
        
        # 从环境变量发现
        self.discover_from_env()
    
    def discover_from_config(self, config_path: str) -> List[ServiceEndpoint]:
        """从配置文件发现服务。
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            发现的服务列表
        """
        path = Path(config_path)
        
        if not path.exists():
            return []
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            return []
        
        services = []
        
        # 支持多种配置格式
        if "services" in config:
            # 格式: {"services": [...]}
            for service_data in config.get("services", []):
                endpoint = ServiceEndpoint.from_dict(service_data)
                services.append(endpoint)
        
        elif "mcpServers" in config:
            # VS Code MCP配置格式: {"mcpServers": {...}}
            for name, server_config in config.get("mcpServers", {}).items():
                endpoint = ServiceEndpoint(
                    name=name,
                    url=server_config.get("url", ""),
                    transport=server_config.get("transport", "stdio"),
                    description=server_config.get("description", ""),
                    capabilities=server_config.get("capabilities", []),
                    metadata=server_config
                )
                services.append(endpoint)
        
        else:
            # 直接是服务列表
            if isinstance(config, list):
                for service_data in config:
                    endpoint = ServiceEndpoint.from_dict(service_data)
                    services.append(endpoint)
        
        # 注册发现的服务
        for endpoint in services:
            self.register(endpoint, DiscoverySource.CONFIG_FILE)
        
        return services
    
    def discover_from_env(self) -> List[ServiceEndpoint]:
        """从环境变量发现服务。
        
        环境变量格式:
        - MCP_SERVICES: JSON格式的服务列表
        - MCP_<NAME>_URL: 服务URL
        - MCP_<NAME>_TRANSPORT: 传输类型
        
        Returns:
            发现的服务列表
        """
        services = []
        
        # 从MCP_SERVICES环境变量
        env_services = os.environ.get(f"{self.env_prefix}SERVICES")
        if env_services:
            try:
                service_list = json.loads(env_services)
                for service_data in service_list:
                    endpoint = ServiceEndpoint.from_dict(service_data)
                    services.append(endpoint)
            except json.JSONDecodeError:
                pass
        
        # 从单独的环境变量
        # 查找所有 MCP_<NAME>_URL 格式的环境变量
        url_suffix = "_URL"
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix) and key.endswith(url_suffix):
                # 提取服务名称
                name = key[len(self.env_prefix):-len(url_suffix)]
                
                if not value:
                    continue
                
                # 获取传输类型
                transport_key = f"{self.env_prefix}{name}_TRANSPORT"
                transport = os.environ.get(transport_key, "stdio")
                
                # 获取描述
                desc_key = f"{self.env_prefix}{name}_DESCRIPTION"
                description = os.environ.get(desc_key, "")
                
                endpoint = ServiceEndpoint(
                    name=name.lower(),
                    url=value,
                    transport=transport,
                    description=description
                )
                services.append(endpoint)
        
        # 注册发现的服务
        for endpoint in services:
            self.register(endpoint, DiscoverySource.ENVIRONMENT)
        
        return services
    
    def register(
        self,
        endpoint: ServiceEndpoint,
        source: DiscoverySource = DiscoverySource.MANUAL
    ) -> None:
        """注册服务。
        
        Args:
            endpoint: 服务端点
            source: 发现来源
        """
        with self._lock:
            self._services[endpoint.name] = ServiceInfo(endpoint=endpoint)
            self._discovery_sources[endpoint.name] = source
    
    def unregister(self, name: str) -> None:
        """注销服务。
        
        Args:
            name: 服务名称
        """
        with self._lock:
            self._services.pop(name, None)
            self._discovery_sources.pop(name, None)
    
    def get(self, name: str) -> Optional[ServiceInfo]:
        """获取服务信息。
        
        Args:
            name: 服务名称
            
        Returns:
            服务信息
        """
        with self._lock:
            return self._services.get(name)
    
    def get_endpoint(self, name: str) -> Optional[ServiceEndpoint]:
        """获取服务端点。
        
        Args:
            name: 服务名称
            
        Returns:
            服务端点
        """
        with self._lock:
            info = self._services.get(name)
            return info.endpoint if info else None
    
    def list_services(
        self,
        status: Optional[ServiceStatus] = None,
        capability: Optional[str] = None
    ) -> List[ServiceInfo]:
        """列出服务。
        
        Args:
            status: 按状态过滤
            capability: 按能力过滤
            
        Returns:
            服务列表
        """
        with self._lock:
            services = list(self._services.values())
        
        if status:
            services = [s for s in services if s.status == status]
        
        if capability:
            services = [
                s for s in services
                if capability in s.endpoint.capabilities
            ]
        
        return services
    
    def list_endpoints(self) -> List[ServiceEndpoint]:
        """列出所有端点。
        
        Returns:
            端点列表
        """
        with self._lock:
            return [s.endpoint for s in self._services.values()]
    
    def find_by_capability(self, capability: str) -> List[ServiceEndpoint]:
        """按能力查找服务。
        
        Args:
            capability: 能力名称
            
        Returns:
            匹配的端点列表
        """
        with self._lock:
            return [
                s.endpoint for s in self._services.values()
                if capability in s.endpoint.capabilities
            ]
    
    def find_by_transport(self, transport: str) -> List[ServiceEndpoint]:
        """按传输类型查找服务。
        
        Args:
            transport: 传输类型
            
        Returns:
            匹配的端点列表
        """
        with self._lock:
            return [
                s.endpoint for s in self._services.values()
                if s.endpoint.transport == transport
            ]
    
    def start_health_check(
        self,
        interval: float = 60.0,
        callback: Optional[Callable[[str, ServiceStatus], None]] = None
    ) -> None:
        """启动健康检查。
        
        Args:
            interval: 检查间隔（秒）
            callback: 状态变化回调
        """
        if self._health_check_running:
            return
        
        self._health_check_interval = interval
        self._health_check_callback = callback
        self._health_check_running = True
        
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self._health_check_thread.start()
    
    def stop_health_check(self) -> None:
        """停止健康检查。"""
        self._health_check_running = False
        
        if self._health_check_thread and self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=2.0)
    
    def _health_check_loop(self) -> None:
        """健康检查循环。"""
        while self._health_check_running:
            self.check_all_health()
            time.sleep(self._health_check_interval)
    
    def check_health(self, name: str) -> ServiceStatus:
        """检查单个服务健康状态。
        
        Args:
            name: 服务名称
            
        Returns:
            服务状态
        """
        with self._lock:
            info = self._services.get(name)
        
        if not info:
            return ServiceStatus.UNKNOWN
        
        endpoint = info.endpoint
        start_time = time.time()
        
        try:
            if endpoint.transport == "stdio":
                # stdio服务通常本地运行，检查命令是否存在
                status = ServiceStatus.HEALTHY
            
            elif endpoint.transport in ("websocket", "ws", "wss"):
                # WebSocket服务，尝试连接
                status = self._check_websocket_health(endpoint.url)
            
            elif endpoint.transport in ("sse", "http", "https"):
                # HTTP/SSE服务，发送HEAD请求
                status = self._check_http_health(endpoint.url)
            
            else:
                status = ServiceStatus.UNKNOWN
        
        except Exception:
            status = ServiceStatus.UNREACHABLE
        
        # 计算响应时间
        response_time = (time.time() - start_time) * 1000
        
        # 更新服务信息
        with self._lock:
            if name in self._services:
                old_status = self._services[name].status
                self._services[name].status = status
                self._services[name].last_check = time.time()
                self._services[name].response_time = response_time
                
                # 状态变化回调
                if status != old_status and self._health_check_callback:
                    self._health_check_callback(name, status)
        
        return status
    
    def _check_websocket_health(self, url: str) -> ServiceStatus:
        """检查WebSocket服务健康状态。"""
        import socket
        import ssl
        
        try:
            if url.startswith("wss://"):
                host_port = url[6:]
                use_ssl = True
            elif url.startswith("ws://"):
                host_port = url[5:]
                use_ssl = False
            else:
                return ServiceStatus.UNKNOWN
            
            # 分离路径
            if "/" in host_port:
                host_port = host_port.split("/")[0]
            
            # 分离端口
            if ":" in host_port:
                host, port_str = host_port.split(":")
                port = int(port_str)
            else:
                host = host_port
                port = 443 if use_ssl else 80
            
            # 尝试连接
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((host, port))
            
            if use_ssl:
                context = ssl.create_default_context()
                sock = context.wrap_socket(sock, server_hostname=host)
            
            sock.close()
            return ServiceStatus.HEALTHY
        
        except Exception:
            return ServiceStatus.UNREACHABLE
    
    def _check_http_health(self, url: str) -> ServiceStatus:
        """检查HTTP服务健康状态。"""
        import socket
        import ssl
        
        try:
            if url.startswith("https://"):
                host_port = url[8:]
                use_ssl = True
            elif url.startswith("http://"):
                host_port = url[7:]
                use_ssl = False
            else:
                return ServiceStatus.UNKNOWN
            
            # 分离路径
            if "/" in host_port:
                host_port = host_port.split("/")[0]
            
            # 分离端口
            if ":" in host_port:
                host, port_str = host_port.split(":")
                port = int(port_str)
            else:
                host = host_port
                port = 443 if use_ssl else 80
            
            # 尝试连接
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((host, port))
            
            if use_ssl:
                context = ssl.create_default_context()
                sock = context.wrap_socket(sock, server_hostname=host)
            
            sock.close()
            return ServiceStatus.HEALTHY
        
        except Exception:
            return ServiceStatus.UNREACHABLE
    
    def check_all_health(self) -> Dict[str, ServiceStatus]:
        """检查所有服务健康状态。
        
        Returns:
            服务名称到状态的映射
        """
        results = {}
        
        with self._lock:
            names = list(self._services.keys())
        
        for name in names:
            results[name] = self.check_health(name)
        
        return results
    
    def to_config(self) -> Dict[str, Any]:
        """导出为配置格式。
        
        Returns:
            配置字典
        """
        with self._lock:
            services = [
                s.endpoint.to_dict()
                for s in self._services.values()
            ]
        
        return {"services": services}
    
    def save_config(self, path: str) -> None:
        """保存配置到文件。
        
        Args:
            path: 文件路径
        """
        config = self.to_config()
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def __len__(self) -> int:
        """获取服务数量。"""
        with self._lock:
            return len(self._services)
    
    def __contains__(self, name: str) -> bool:
        """检查服务是否存在。"""
        with self._lock:
            return name in self._services


class ServiceRegistry:
    """服务注册中心。
    
    提供服务的动态注册和发现接口。
    """
    
    def __init__(self) -> None:
        """初始化服务注册中心。"""
        self._discovery = ServiceDiscovery(auto_discover=False)
        self._callbacks: List[Callable[[str, ServiceEndpoint], None]] = []
        self._lock = threading.Lock()
    
    def register(
        self,
        name: str,
        url: str,
        transport: str = "stdio",
        description: str = "",
        capabilities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ServiceEndpoint:
        """注册服务。
        
        Args:
            name: 服务名称
            url: 服务URL
            transport: 传输类型
            description: 描述
            capabilities: 能力列表
            metadata: 元数据
            
        Returns:
            服务端点
        """
        endpoint = ServiceEndpoint(
            name=name,
            url=url,
            transport=transport,
            description=description,
            capabilities=capabilities or [],
            metadata=metadata or {}
        )
        
        self._discovery.register(endpoint, DiscoverySource.REGISTRY)
        
        # 通知回调
        with self._lock:
            callbacks = list(self._callbacks)
        
        for callback in callbacks:
            try:
                callback("register", endpoint)
            except Exception:
                pass
        
        return endpoint
    
    def unregister(self, name: str) -> None:
        """注销服务。"""
        endpoint = self._discovery.get_endpoint(name)
        
        self._discovery.unregister(name)
        
        if endpoint:
            with self._lock:
                callbacks = list(self._callbacks)
            
            for callback in callbacks:
                try:
                    callback("unregister", endpoint)
                except Exception:
                    pass
    
    def get(self, name: str) -> Optional[ServiceEndpoint]:
        """获取服务。"""
        return self._discovery.get_endpoint(name)
    
    def list_all(self) -> List[ServiceEndpoint]:
        """列出所有服务。"""
        return self._discovery.list_endpoints()
    
    def on_change(
        self,
        callback: Callable[[str, ServiceEndpoint], None]
    ) -> None:
        """注册变更回调。
        
        Args:
            callback: 回调函数，参数为(操作类型, 端点)
        """
        with self._lock:
            self._callbacks.append(callback)
    
    def off_change(
        self,
        callback: Optional[Callable[[str, ServiceEndpoint], None]] = None
    ) -> None:
        """移除变更回调。"""
        with self._lock:
            if callback is None:
                self._callbacks.clear()
            else:
                try:
                    self._callbacks.remove(callback)
                except ValueError:
                    pass


__all__ = [
    "DiscoverySource",
    "ServiceStatus",
    "ServiceEndpoint",
    "ServiceInfo",
    "ServiceDiscovery",
    "ServiceRegistry",
]
