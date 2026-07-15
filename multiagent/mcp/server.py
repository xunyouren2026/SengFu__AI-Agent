"""MCP协议服务器模块。

本模块实现了MCP协议服务器，用于暴露工具列表、资源和提示词。
支持通过stdio或WebSocket传输层提供服务。
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Optional, Callable, Dict, Any, List, Union
from dataclasses import dataclass, field
from enum import Enum

from .schema import (
    MCPMethod,
    MCPErrorCode,
    JSONRPCRequest,
    JSONRPCResponse,
    JSONRPCError,
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


class ServerState(Enum):
    """服务器状态枚举。"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    SHUTDOWN = "shutdown"


@dataclass
class RequestContext:
    """请求上下文。
    
    Attributes:
        request_id: 请求ID
        method: 方法名
        client_info: 客户端信息
        user_info: 用户信息（认证后）
        progress_token: 进度令牌
    """
    request_id: Optional[Union[str, int]] = None
    method: str = ""
    client_info: Dict[str, Any] = field(default_factory=dict)
    user_info: Optional[Dict[str, Any]] = None
    progress_token: Optional[Union[str, int]] = None


class MCPServer:
    """MCP协议服务器。
    
    实现MCP协议的服务器端，处理客户端请求并返回响应。
    支持工具调用、资源访问、提示词获取等功能。
    
    Attributes:
        name: 服务器名称
        version: 服务器版本
        state: 服务器状态
    """
    
    def __init__(
        self,
        name: str = "mcp-server",
        version: str = "1.0.0",
        protocol_version: str = "2024-11-05"
    ):
        """初始化MCP服务器。
        
        Args:
            name: 服务器名称
            version: 服务器版本
            protocol_version: 协议版本
        """
        self.name = name
        self.version = version
        self.protocol_version = protocol_version
        self.state = ServerState.UNINITIALIZED
        
        # 注册的工具、资源、提示词
        self._tools: Dict[str, MCPTool] = {}
        self._resources: Dict[str, MCPResource] = {}
        self._prompts: Dict[str, MCPPrompt] = {}
        
        # 资源读取处理器
        self._resource_handlers: Dict[str, Callable[[str], Any]] = {}
        
        # 服务器能力
        self._capabilities = MCPServerCapabilities(
            logging=True,
            prompts=True,
            resources=True,
            tools=True
        )
        
        # 客户端信息
        self._client_info: Dict[str, Any] = {}
        
        # 认证中间件
        self._auth_middleware: Optional[Any] = None
        
        # 通知回调
        self._notification_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
        # 进度回调
        self._progress_callbacks: Dict[Union[str, int], Callable[[float, float], None]] = {}
    
    def set_auth_middleware(self, middleware: Any) -> None:
        """设置认证中间件。
        
        Args:
            middleware: 认证中间件
        """
        self._auth_middleware = middleware
    
    def register_tool(
        self,
        name: str,
        description: str,
        handler: Callable[..., Any],
        input_schema: Optional[Dict[str, Any]] = None
    ) -> None:
        """注册工具。
        
        Args:
            name: 工具名称
            description: 工具描述
            handler: 工具处理函数
            input_schema: 输入参数Schema
        """
        from .schema import MCPToolInputSchema
        
        if input_schema:
            schema = MCPToolInputSchema(
                type=input_schema.get("type", "object"),
                properties=input_schema.get("properties", {}),
                required=input_schema.get("required", []),
                additionalProperties=input_schema.get("additionalProperties", False)
            )
        else:
            schema = MCPToolInputSchema()
        
        tool = MCPTool(
            name=name,
            description=description,
            inputSchema=schema,
            handler=handler
        )
        
        self._tools[name] = tool
    
    def unregister_tool(self, name: str) -> None:
        """注销工具。
        
        Args:
            name: 工具名称
        """
        self._tools.pop(name, None)
        
        # 发送工具列表变更通知
        self._notify_tool_list_changed()
    
    def register_resource(
        self,
        uri: str,
        name: str,
        description: str = "",
        mime_type: str = "text/plain",
        handler: Optional[Callable[[str], Any]] = None
    ) -> None:
        """注册资源。
        
        Args:
            uri: 资源URI
            name: 资源名称
            description: 资源描述
            mime_type: MIME类型
            handler: 资源读取处理函数
        """
        resource = MCPResource(
            uri=uri,
            name=name,
            description=description,
            mimeType=mime_type
        )
        
        self._resources[uri] = resource
        
        if handler:
            self._resource_handlers[uri] = handler
    
    def unregister_resource(self, uri: str) -> None:
        """注销资源。
        
        Args:
            uri: 资源URI
        """
        self._resources.pop(uri, None)
        self._resource_handlers.pop(uri, None)
        
        # 发送资源更新通知
        self._notify_resource_updated(uri)
    
    def register_prompt(
        self,
        name: str,
        description: str = "",
        template: str = "",
        arguments: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """注册提示词模板。
        
        Args:
            name: 提示词名称
            description: 描述
            template: 模板字符串
            arguments: 参数定义列表
        """
        from .schema import MCPPromptArgument
        
        args = []
        if arguments:
            for arg in arguments:
                args.append(MCPPromptArgument(
                    name=arg.get("name", ""),
                    description=arg.get("description", ""),
                    required=arg.get("required", False)
                ))
        
        prompt = MCPPrompt(
            name=name,
            description=description,
            arguments=args,
            template=template
        )
        
        self._prompts[name] = prompt
    
    def unregister_prompt(self, name: str) -> None:
        """注销提示词。
        
        Args:
            name: 提示词名称
        """
        self._prompts.pop(name, None)
    
    def handle_request(
        self,
        request: Union[JSONRPCRequest, Dict[str, Any]],
        headers: Optional[Dict[str, str]] = None
    ) -> JSONRPCResponse:
        """处理请求。
        
        Args:
            request: JSON-RPC请求
            headers: HTTP请求头（用于认证）
            
        Returns:
            JSON-RPC响应
        """
        # 转换为JSONRPCRequest
        if isinstance(request, dict):
            req = JSONRPCRequest(
                jsonrpc=request.get("jsonrpc", "2.0"),
                method=request.get("method", ""),
                params=request.get("params", {}),
                id=request.get("id")
            )
        else:
            req = request
        
        # 创建上下文
        ctx = RequestContext(
            request_id=req.id,
            method=req.method
        )
        
        # 认证检查
        if self._auth_middleware and headers:
            try:
                ctx.user_info = self._auth_middleware.process(headers)
            except Exception as e:
                return JSONRPCResponse.error(
                    code=MCPErrorCode.AUTHENTICATION_ERROR.value,
                    message=str(e),
                    request_id=req.id
                )
        
        # 路由请求
        try:
            method = MCPMethod(req.method)
            result = self._dispatch_method(method, req.params, ctx)
            return JSONRPCResponse.success(result, req.id)
        
        except ValueError:
            return JSONRPCResponse.error(
                code=MCPErrorCode.METHOD_NOT_FOUND.value,
                message=f"Method not found: {req.method}",
                request_id=req.id
            )
        
        except Exception as e:
            return JSONRPCResponse.error(
                code=MCPErrorCode.INTERNAL_ERROR.value,
                message=str(e),
                request_id=req.id
            )
    
    def _dispatch_method(
        self,
        method: MCPMethod,
        params: Dict[str, Any],
        ctx: RequestContext
    ) -> Dict[str, Any]:
        """分发方法调用。
        
        Args:
            method: 方法
            params: 参数
            ctx: 请求上下文
            
        Returns:
            结果字典
        """
        if method == MCPMethod.INITIALIZE:
            return self._handle_initialize(params, ctx)
        
        elif method == MCPMethod.TOOLS_LIST:
            return self._handle_tools_list(params, ctx)
        
        elif method == MCPMethod.TOOLS_CALL:
            return self._handle_tools_call(params, ctx)
        
        elif method == MCPMethod.RESOURCES_LIST:
            return self._handle_resources_list(params, ctx)
        
        elif method == MCPMethod.RESOURCES_READ:
            return self._handle_resources_read(params, ctx)
        
        elif method == MCPMethod.PROMPTS_LIST:
            return self._handle_prompts_list(params, ctx)
        
        elif method == MCPMethod.PROMPTS_GET:
            return self._handle_prompts_get(params, ctx)
        
        else:
            raise ValueError(f"Unhandled method: {method}")
    
    def _handle_initialize(
        self,
        params: Dict[str, Any],
        ctx: RequestContext
    ) -> Dict[str, Any]:
        """处理初始化请求。"""
        if self.state != ServerState.UNINITIALIZED:
            raise ValueError("Server already initialized")
        
        self.state = ServerState.INITIALIZING
        
        # 解析客户端参数
        init_params = MCPInitializeParams.from_dict(params)
        self._client_info = init_params.clientInfo
        
        # 创建初始化结果
        result = MCPInitializeResult(
            protocolVersion=self.protocol_version,
            capabilities=self._capabilities,
            serverInfo={
                "name": self.name,
                "version": self.version
            }
        )
        
        self.state = ServerState.READY
        
        return result.to_dict()
    
    def _handle_tools_list(
        self,
        params: Dict[str, Any],
        ctx: RequestContext
    ) -> Dict[str, Any]:
        """处理工具列表请求。"""
        tools = [tool.to_dict() for tool in self._tools.values()]
        return {"tools": tools}
    
    def _handle_tools_call(
        self,
        params: Dict[str, Any],
        ctx: RequestContext
    ) -> Dict[str, Any]:
        """处理工具调用请求。"""
        call_params = MCPToolCallParams.from_dict(params)
        
        tool = self._tools.get(call_params.name)
        if not tool:
            result = MCPToolCallResult.text(
                f"Tool not found: {call_params.name}",
                is_error=True
            )
            return result.to_dict()
        
        try:
            # 执行工具
            output = tool.execute(call_params.arguments)
            
            # 处理结果
            if isinstance(output, str):
                result = MCPToolCallResult.text(output)
            elif isinstance(output, dict):
                result = MCPToolCallResult.text(json.dumps(output, ensure_ascii=False))
            elif isinstance(output, list):
                result = MCPToolCallResult.text(json.dumps(output, ensure_ascii=False))
            else:
                result = MCPToolCallResult.text(str(output))
            
            return result.to_dict()
        
        except ValueError as e:
            result = MCPToolCallResult.text(str(e), is_error=True)
            return result.to_dict()
        
        except Exception as e:
            result = MCPToolCallResult.text(
                f"Tool execution failed: {str(e)}",
                is_error=True
            )
            return result.to_dict()
    
    def _handle_resources_list(
        self,
        params: Dict[str, Any],
        ctx: RequestContext
    ) -> Dict[str, Any]:
        """处理资源列表请求。"""
        resources = [res.to_dict() for res in self._resources.values()]
        return {"resources": resources}
    
    def _handle_resources_read(
        self,
        params: Dict[str, Any],
        ctx: RequestContext
    ) -> Dict[str, Any]:
        """处理资源读取请求。"""
        uri = params.get("uri", "")
        
        resource = self._resources.get(uri)
        if not resource:
            raise ValueError(f"Resource not found: {uri}")
        
        # 调用资源处理器
        handler = self._resource_handlers.get(uri)
        if handler:
            try:
                content = handler(uri)
            except Exception as e:
                raise ValueError(f"Failed to read resource: {str(e)}")
        else:
            content = None
        
        # 构建响应
        from .schema import MCPResourceContent
        
        if isinstance(content, str):
            resource_content = MCPResourceContent(
                uri=uri,
                mimeType=resource.mimeType,
                text=content
            )
        elif isinstance(content, bytes):
            import base64
            resource_content = MCPResourceContent(
                uri=uri,
                mimeType=resource.mimeType,
                blob=base64.b64encode(content).decode("ascii")
            )
        else:
            resource_content = MCPResourceContent(
                uri=uri,
                mimeType="application/json",
                text=json.dumps(content, ensure_ascii=False) if content else ""
            )
        
        return {"contents": [resource_content.to_dict()]}
    
    def _handle_prompts_list(
        self,
        params: Dict[str, Any],
        ctx: RequestContext
    ) -> Dict[str, Any]:
        """处理提示词列表请求。"""
        prompts = [prompt.to_dict() for prompt in self._prompts.values()]
        return {"prompts": prompts}
    
    def _handle_prompts_get(
        self,
        params: Dict[str, Any],
        ctx: RequestContext
    ) -> Dict[str, Any]:
        """处理提示词获取请求。"""
        name = params.get("name", "")
        arguments = params.get("arguments", {})
        
        prompt = self._prompts.get(name)
        if not prompt:
            raise ValueError(f"Prompt not found: {name}")
        
        try:
            rendered = prompt.render(arguments)
        except ValueError as e:
            raise ValueError(str(e))
        
        return {
            "description": prompt.description,
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": rendered
                    }
                }
            ]
        }
    
    def send_progress(
        self,
        progress_token: Union[str, int],
        progress: float,
        total: float = 100.0
    ) -> None:
        """发送进度通知。
        
        Args:
            progress_token: 进度令牌
            progress: 当前进度
            total: 总进度
        """
        notification = MCPProgressNotification(
            progressToken=progress_token,
            progress=progress,
            total=total
        )
        
        self._send_notification(
            MCPMethod.NOTIFICATION_PROGRESS.value,
            notification.to_dict()
        )
        
        # 调用进度回调
        callback = self._progress_callbacks.get(progress_token)
        if callback:
            callback(progress, total)
    
    def _send_notification(
        self,
        method: str,
        params: Dict[str, Any]
    ) -> None:
        """发送通知。"""
        for callback in self._notification_callbacks:
            try:
                callback(method, params)
            except Exception:
                pass
    
    def _notify_tool_list_changed(self) -> None:
        """通知工具列表变更。"""
        self._send_notification(
            MCPMethod.NOTIFICATION_TOOL_LIST_CHANGED.value,
            {}
        )
    
    def _notify_resource_updated(self, uri: str) -> None:
        """通知资源更新。"""
        self._send_notification(
            MCPMethod.NOTIFICATION_RESOURCE_UPDATED.value,
            {"uri": uri}
        )
    
    def on_notification(
        self,
        callback: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """注册通知回调。
        
        Args:
            callback: 回调函数
        """
        self._notification_callbacks.append(callback)
    
    def on_progress(
        self,
        progress_token: Union[str, int],
        callback: Callable[[float, float], None]
    ) -> None:
        """注册进度回调。
        
        Args:
            progress_token: 进度令牌
            callback: 回调函数
        """
        self._progress_callbacks[progress_token] = callback
    
    def shutdown(self) -> None:
        """关闭服务器。"""
        self.state = ServerState.SHUTDOWN
        self._tools.clear()
        self._resources.clear()
        self._prompts.clear()
        self._resource_handlers.clear()
        self._notification_callbacks.clear()
        self._progress_callbacks.clear()
    
    @property
    def is_ready(self) -> bool:
        """检查服务器是否就绪。"""
        return self.state == ServerState.READY
    
    @property
    def tool_count(self) -> int:
        """获取工具数量。"""
        return len(self._tools)
    
    @property
    def resource_count(self) -> int:
        """获取资源数量。"""
        return len(self._resources)
    
    @property
    def prompt_count(self) -> int:
        """获取提示词数量。"""
        return len(self._prompts)


class MCPServerRunner:
    """MCP服务器运行器。
    
    封装传输层和服务器，提供便捷的启动和停止接口。
    """
    
    def __init__(
        self,
        server: MCPServer,
        transport: Any
    ):
        """初始化服务器运行器。
        
        Args:
            server: MCP服务器
            transport: 传输层
        """
        self.server = server
        self.transport = transport
        self._running = False
    
    def start(self) -> None:
        """启动服务器。"""
        if self._running:
            return
        
        self._running = True
        
        # 设置请求处理器
        def handle_request(message: Dict[str, Any]) -> Dict[str, Any]:
            response = self.server.handle_request(message)
            return json.loads(response.to_json())
        
        if hasattr(self.transport, "set_request_handler"):
            self.transport.set_request_handler(handle_request)
        
        # 启动传输层
        if hasattr(self.transport, "start"):
            self.transport.start()
        
        # 设置通知回调
        def send_notification(method: str, params: Dict[str, Any]) -> None:
            notification = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params
            }
            if hasattr(self.transport, "send"):
                self.transport.send(notification)
            elif hasattr(self.transport, "send_raw"):
                self.transport.send_raw(notification)
        
        self.server.on_notification(send_notification)
    
    def stop(self) -> None:
        """停止服务器。"""
        if not self._running:
            return
        
        self._running = False
        
        # 关闭传输层
        if hasattr(self.transport, "close"):
            self.transport.close()
        
        # 关闭服务器
        self.server.shutdown()
    
    @property
    def is_running(self) -> bool:
        """检查服务器是否运行中。"""
        return self._running


class ToolDecorator:
    """工具装饰器辅助类。
    
    用于通过装饰器方式注册工具。
    """
    
    def __init__(self, server: MCPServer) -> None:
        """初始化工具装饰器。
        
        Args:
            server: MCP服务器
        """
        self.server = server
    
    def tool(
        self,
        name: Optional[str] = None,
        description: str = "",
        input_schema: Optional[Dict[str, Any]] = None
    ) -> Callable[[Callable], Callable]:
        """工具装饰器。
        
        Args:
            name: 工具名称，默认使用函数名
            description: 工具描述
            input_schema: 输入参数Schema
            
        Returns:
            装饰器函数
        """
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_desc = description or func.__doc__ or ""
            
            self.server.register_tool(
                name=tool_name,
                description=tool_desc,
                handler=func,
                input_schema=input_schema
            )
            
            return func
        
        return decorator


__all__ = [
    "ServerState",
    "RequestContext",
    "MCPServer",
    "MCPServerRunner",
    "ToolDecorator",
]
