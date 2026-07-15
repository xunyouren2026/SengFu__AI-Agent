"""MCP客户端模块。

本模块实现了MCP协议客户端，用于调用外部MCP兼容服务。
支持通过stdio或WebSocket传输层与服务器通信。
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Optional, Callable, Dict, Any, List, Union
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, Empty

from .schema import (
    MCPMethod,
    MCPErrorCode,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPTool,
    MCPResource,
    MCPPrompt,
    MCPInitializeParams,
    MCPServerCapabilities,
    MCPInitializeResult,
    MCPToolCallParams,
    MCPToolCallResult,
    MCPProgressNotification,
)


class ClientState(Enum):
    """客户端状态枚举。"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"


@dataclass
class ClientConfig:
    """客户端配置。
    
    Attributes:
        name: 客户端名称
        version: 客户端版本
        protocol_version: 协议版本
        request_timeout: 请求超时时间（秒）
        retry_count: 重试次数
        retry_delay: 重试延迟（秒）
    """
    name: str = "mcp-client"
    version: str = "1.0.0"
    protocol_version: str = "2024-11-05"
    request_timeout: float = 30.0
    retry_count: int = 3
    retry_delay: float = 1.0


class MCPClient:
    """MCP协议客户端。
    
    实现MCP协议的客户端端，用于连接服务器并调用工具、访问资源等。
    
    Attributes:
        config: 客户端配置
        state: 客户端状态
        server_info: 服务器信息
    """
    
    def __init__(
        self,
        config: Optional[ClientConfig] = None,
        credentials: Optional[Any] = None
    ):
        """初始化MCP客户端。
        
        Args:
            config: 客户端配置
            credentials: 认证凭证
        """
        self.config = config or ClientConfig()
        self.credentials = credentials
        self.state = ClientState.DISCONNECTED
        
        # 传输层
        self._transport: Optional[Any] = None
        
        # 服务器信息
        self._server_info: Dict[str, Any] = {}
        self._server_capabilities: Optional[MCPServerCapabilities] = None
        
        # 缓存的工具、资源、提示词
        self._tools: Dict[str, MCPTool] = {}
        self._resources: Dict[str, MCPResource] = {}
        self._prompts: Dict[str, MCPPrompt] = {}
        
        # 通知处理器
        self._notification_handlers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        
        # 进度处理器
        self._progress_handlers: Dict[Union[str, int], Callable[[MCPProgressNotification], None]] = {}
        
        # 请求锁
        self._request_lock = threading.Lock()
    
    def connect(self, transport: Any) -> None:
        """连接到MCP服务器。
        
        Args:
            transport: 传输层实例
            
        Raises:
            ConnectionError: 连接失败
        """
        if self.state != ClientState.DISCONNECTED:
            return
        
        self.state = ClientState.CONNECTING
        self._transport = transport
        
        try:
            # 启动传输层
            if hasattr(transport, "start"):
                transport.start()
            elif hasattr(transport, "connect"):
                transport.connect()
            
            # 设置通知处理器
            if hasattr(transport, "set_notification_handler"):
                transport.set_notification_handler(self._handle_notification)
            elif hasattr(transport, "on_message"):
                transport.on_message(self._handle_message)
            
            # 初始化连接
            self._initialize()
            
            self.state = ClientState.READY
        
        except Exception as e:
            self.state = ClientState.ERROR
            raise ConnectionError(f"Failed to connect: {str(e)}") from e
    
    def _initialize(self) -> None:
        """初始化MCP连接。"""
        self.state = ClientState.INITIALIZING
        
        # 发送初始化请求
        init_params = MCPInitializeParams(
            protocolVersion=self.config.protocol_version,
            capabilities={
                "roots": {"listChanged": True},
                "sampling": {}
            },
            clientInfo={
                "name": self.config.name,
                "version": self.config.version
            }
        )
        
        response = self._request(
            MCPMethod.INITIALIZE.value,
            init_params.to_dict()
        )
        
        # 解析响应
        if "error" in response:
            raise ConnectionError(f"Initialize failed: {response['error']}")
        
        result = MCPInitializeResult.from_dict(response.get("result", {}))
        
        self._server_info = result.serverInfo
        self._server_capabilities = result.capabilities
        
        # 发送初始化完成通知
        self._notify(MCPMethod.NOTIFICATION_INITIALIZED.value)
    
    def disconnect(self) -> None:
        """断开连接。"""
        if self.state == ClientState.DISCONNECTED:
            return
        
        if self._transport:
            if hasattr(self._transport, "close"):
                self._transport.close()
            self._transport = None
        
        self.state = ClientState.DISCONNECTED
        self._tools.clear()
        self._resources.clear()
        self._prompts.clear()
    
    def _request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """发送请求并等待响应。
        
        Args:
            method: 方法名
            params: 参数
            timeout: 超时时间
            
        Returns:
            响应字典
        """
        if not self._transport:
            raise ConnectionError("Not connected")
        
        timeout = timeout or self.config.request_timeout
        
        # 添加认证头
        headers = {}
        if self.credentials and hasattr(self.credentials, "to_headers"):
            headers = self.credentials.to_headers()
        
        # 发送请求
        for attempt in range(self.config.retry_count):
            try:
                with self._request_lock:
                    response = self._transport.request(method, params, timeout)
                
                return response
            
            except TimeoutError:
                if attempt < self.config.retry_count - 1:
                    time.sleep(self.config.retry_delay)
                    continue
                raise
            
            except Exception as e:
                if attempt < self.config.retry_count - 1:
                    time.sleep(self.config.retry_delay)
                    continue
                raise
        
        raise TimeoutError(f"Request {method} failed after {self.config.retry_count} attempts")
    
    def _notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        """发送通知。
        
        Args:
            method: 方法名
            params: 参数
        """
        if not self._transport:
            return
        
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {}
        }
        
        if hasattr(self._transport, "send"):
            self._transport.send(notification)
    
    def _handle_message(self, message: Dict[str, Any]) -> None:
        """处理接收到的消息。
        
        Args:
            message: 消息字典
        """
        # 区分通知和响应
        if "id" not in message and "method" in message:
            self._handle_notification(
                message.get("method", ""),
                message.get("params", {})
            )
    
    def _handle_notification(self, method: str, params: Dict[str, Any]) -> None:
        """处理通知。
        
        Args:
            method: 方法名
            params: 参数
        """
        # 进度通知
        if method == MCPMethod.NOTIFICATION_PROGRESS.value:
            notification = MCPProgressNotification.from_dict(params)
            handler = self._progress_handlers.get(notification.progressToken)
            if handler:
                handler(notification)
        
        # 工具列表变更通知
        elif method == MCPMethod.NOTIFICATION_TOOL_LIST_CHANGED.value:
            self._tools.clear()
        
        # 资源更新通知
        elif method == MCPMethod.NOTIFICATION_RESOURCE_UPDATED.value:
            uri = params.get("uri", "")
            self._resources.pop(uri, None)
        
        # 调用自定义处理器
        handlers = self._notification_handlers.get(method, [])
        for handler in handlers:
            try:
                handler(params)
            except Exception:
                pass
    
    def list_tools(self, use_cache: bool = True) -> List[MCPTool]:
        """获取工具列表。
        
        Args:
            use_cache: 是否使用缓存
            
        Returns:
            工具列表
        """
        if use_cache and self._tools:
            return list(self._tools.values())
        
        response = self._request(MCPMethod.TOOLS_LIST.value)
        
        if "error" in response:
            raise RuntimeError(f"Failed to list tools: {response['error']}")
        
        result = response.get("result", {})
        tools_data = result.get("tools", [])
        
        self._tools.clear()
        for tool_data in tools_data:
            tool = MCPTool.from_dict(tool_data)
            self._tools[tool.name] = tool
        
        return list(self._tools.values())
    
    def get_tool(self, name: str) -> Optional[MCPTool]:
        """获取指定工具。
        
        Args:
            name: 工具名称
            
        Returns:
            工具对象，不存在返回None
        """
        if name not in self._tools:
            self.list_tools()
        
        return self._tools.get(name)
    
    def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        on_progress: Optional[Callable[[float, float], None]] = None
    ) -> MCPToolCallResult:
        """调用工具。
        
        Args:
            name: 工具名称
            arguments: 工具参数
            timeout: 超时时间
            on_progress: 进度回调
            
        Returns:
            工具调用结果
        """
        # 生成进度令牌
        progress_token = None
        if on_progress:
            progress_token = str(uuid.uuid4())
            self._progress_handlers[progress_token] = lambda n: on_progress(n.progress, n.total)
        
        try:
            params = MCPToolCallParams(
                name=name,
                arguments=arguments or {}
            )
            
            # 添加进度令牌
            params_dict = params.to_dict()
            if progress_token:
                params_dict["_meta"] = {"progressToken": progress_token}
            
            response = self._request(
                MCPMethod.TOOLS_CALL.value,
                params_dict,
                timeout
            )
            
            if "error" in response:
                return MCPToolCallResult.text(
                    str(response["error"]),
                    is_error=True
                )
            
            return MCPToolCallResult.from_dict(response.get("result", {}))
        
        finally:
            if progress_token:
                self._progress_handlers.pop(progress_token, None)
    
    def list_resources(self, use_cache: bool = True) -> List[MCPResource]:
        """获取资源列表。
        
        Args:
            use_cache: 是否使用缓存
            
        Returns:
            资源列表
        """
        if use_cache and self._resources:
            return list(self._resources.values())
        
        response = self._request(MCPMethod.RESOURCES_LIST.value)
        
        if "error" in response:
            raise RuntimeError(f"Failed to list resources: {response['error']}")
        
        result = response.get("result", {})
        resources_data = result.get("resources", [])
        
        self._resources.clear()
        for res_data in resources_data:
            resource = MCPResource.from_dict(res_data)
            self._resources[resource.uri] = resource
        
        return list(self._resources.values())
    
    def read_resource(self, uri: str) -> Any:
        """读取资源内容。
        
        Args:
            uri: 资源URI
            
        Returns:
            资源内容
        """
        response = self._request(MCPMethod.RESOURCES_READ.value, {"uri": uri})
        
        if "error" in response:
            raise RuntimeError(f"Failed to read resource: {response['error']}")
        
        result = response.get("result", {})
        contents = result.get("contents", [])
        
        if not contents:
            return None
        
        content = contents[0]
        
        # 解析内容
        if "text" in content:
            return content["text"]
        elif "blob" in content:
            import base64
            return base64.b64decode(content["blob"])
        
        return content
    
    def list_prompts(self, use_cache: bool = True) -> List[MCPPrompt]:
        """获取提示词列表。
        
        Args:
            use_cache: 是否使用缓存
            
        Returns:
            提示词列表
        """
        if use_cache and self._prompts:
            return list(self._prompts.values())
        
        response = self._request(MCPMethod.PROMPTS_LIST.value)
        
        if "error" in response:
            raise RuntimeError(f"Failed to list prompts: {response['error']}")
        
        result = response.get("result", {})
        prompts_data = result.get("prompts", [])
        
        self._prompts.clear()
        for prompt_data in prompts_data:
            prompt = MCPPrompt.from_dict(prompt_data)
            self._prompts[prompt.name] = prompt
        
        return list(self._prompts.values())
    
    def get_prompt(
        self,
        name: str,
        arguments: Optional[Dict[str, str]] = None
    ) -> str:
        """获取渲染后的提示词。
        
        Args:
            name: 提示词名称
            arguments: 参数
            
        Returns:
            渲染后的提示词文本
        """
        response = self._request(
            MCPMethod.PROMPTS_GET.value,
            {"name": name, "arguments": arguments or {}}
        )
        
        if "error" in response:
            raise RuntimeError(f"Failed to get prompt: {response['error']}")
        
        result = response.get("result", {})
        messages = result.get("messages", [])
        
        # 提取文本内容
        texts = []
        for message in messages:
            content = message.get("content", {})
            if content.get("type") == "text":
                texts.append(content.get("text", ""))
        
        return "\n".join(texts)
    
    def on_notification(
        self,
        method: str,
        handler: Callable[[Dict[str, Any]], None]
    ) -> None:
        """注册通知处理器。
        
        Args:
            method: 方法名
            handler: 处理函数
        """
        if method not in self._notification_handlers:
            self._notification_handlers[method] = []
        self._notification_handlers[method].append(handler)
    
    def off_notification(
        self,
        method: str,
        handler: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> None:
        """移除通知处理器。
        
        Args:
            method: 方法名
            handler: 处理函数，为None则移除所有
        """
        if handler is None:
            self._notification_handlers.pop(method, None)
        elif method in self._notification_handlers:
            try:
                self._notification_handlers[method].remove(handler)
            except ValueError:
                pass
    
    @property
    def is_connected(self) -> bool:
        """检查是否已连接。"""
        return self.state == ClientState.READY
    
    @property
    def server_name(self) -> str:
        """获取服务器名称。"""
        return self._server_info.get("name", "")
    
    @property
    def server_version(self) -> str:
        """获取服务器版本。"""
        return self._server_info.get("version", "")
    
    @property
    def capabilities(self) -> Optional[MCPServerCapabilities]:
        """获取服务器能力。"""
        return self._server_capabilities


class MCPClientPool:
    """MCP客户端连接池。
    
    管理多个MCP服务器连接，提供统一的访问接口。
    """
    
    def __init__(self, default_config: Optional[ClientConfig] = None) -> None:
        """初始化客户端池。
        
        Args:
            default_config: 默认客户端配置
        """
        self.default_config = default_config or ClientConfig()
        self._clients: Dict[str, MCPClient] = {}
        self._lock = threading.Lock()
    
    def add_client(
        self,
        name: str,
        transport: Any,
        config: Optional[ClientConfig] = None,
        credentials: Optional[Any] = None
    ) -> MCPClient:
        """添加客户端连接。
        
        Args:
            name: 连接名称
            transport: 传输层
            config: 客户端配置
            credentials: 认证凭证
            
        Returns:
            客户端实例
        """
        client = MCPClient(
            config=config or self.default_config,
            credentials=credentials
        )
        client.connect(transport)
        
        with self._lock:
            self._clients[name] = client
        
        return client
    
    def get_client(self, name: str) -> Optional[MCPClient]:
        """获取客户端。
        
        Args:
            name: 连接名称
            
        Returns:
            客户端实例
        """
        with self._lock:
            return self._clients.get(name)
    
    def remove_client(self, name: str) -> None:
        """移除客户端。
        
        Args:
            name: 连接名称
        """
        with self._lock:
            client = self._clients.pop(name, None)
        
        if client:
            client.disconnect()
    
    def call_tool(
        self,
        client_name: str,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None
    ) -> MCPToolCallResult:
        """在指定客户端上调用工具。
        
        Args:
            client_name: 客户端名称
            tool_name: 工具名称
            arguments: 工具参数
            
        Returns:
            工具调用结果
        """
        client = self.get_client(client_name)
        if not client:
            return MCPToolCallResult.text(
                f"Client not found: {client_name}",
                is_error=True
            )
        
        return client.call_tool(tool_name, arguments)
    
    def broadcast_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, MCPToolCallResult]:
        """在所有客户端上广播调用工具。
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数
            
        Returns:
            客户端名称到结果的映射
        """
        results = {}
        
        with self._lock:
            clients = list(self._clients.items())
        
        for name, client in clients:
            try:
                results[name] = client.call_tool(tool_name, arguments)
            except Exception as e:
                results[name] = MCPToolCallResult.text(str(e), is_error=True)
        
        return results
    
    def list_all_tools(self) -> Dict[str, List[MCPTool]]:
        """列出所有客户端的工具。
        
        Returns:
            客户端名称到工具列表的映射
        """
        result = {}
        
        with self._lock:
            clients = list(self._clients.items())
        
        for name, client in clients:
            try:
                result[name] = client.list_tools()
            except Exception:
                result[name] = []
        
        return result
    
    def close_all(self) -> None:
        """关闭所有连接。"""
        with self._lock:
            clients = list(self._clients.items())
            self._clients.clear()
        
        for name, client in clients:
            try:
                client.disconnect()
            except Exception:
                pass
    
    @property
    def client_names(self) -> List[str]:
        """获取所有客户端名称。"""
        with self._lock:
            return list(self._clients.keys())
    
    @property
    def client_count(self) -> int:
        """获取客户端数量。"""
        with self._lock:
            return len(self._clients)


__all__ = [
    "ClientState",
    "ClientConfig",
    "MCPClient",
    "MCPClientPool",
]
