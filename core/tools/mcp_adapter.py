"""
Model Context Protocol (MCP) 适配器模块

提供服务发现、能力协商、工具调用、资源访问和提示处理功能。
仅使用 Python 标准库。
"""

import base64
import hashlib
import json
import re
import time
import uuid
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
)
from urllib.parse import urljoin, urlparse

from .base import Tool, ToolResult, ToolParameter


# ---------------------------------------------------------------------------
# MCP 类型定义
# ---------------------------------------------------------------------------
class MCPErrorCode(Enum):
    """MCP 错误代码"""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    SERVER_NOT_INITIALIZED = -32002
    UNKNOWN_ERROR = -32001


@dataclass
class MCPMessage:
    """MCP 消息

    Attributes:
        jsonrpc: JSON-RPC 版本
        id: 消息 ID
        method: 方法名
        params: 参数
        result: 结果
        error: 错误
    """
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data: Dict[str, Any] = {"jsonrpc": self.jsonrpc}
        if self.id is not None:
            data["id"] = self.id
        if self.method:
            data["method"] = self.method
        if self.params is not None:
            data["params"] = self.params
        if self.result is not None:
            data["result"] = self.result
        if self.error is not None:
            data["error"] = self.error
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPMessage":
        """从字典创建"""
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id"),
            method=data.get("method"),
            params=data.get("params"),
            result=data.get("result"),
            error=data.get("error"),
        )

    def is_request(self) -> bool:
        """是否是请求"""
        return self.method is not None and self.id is not None

    def is_notification(self) -> bool:
        """是否是通知"""
        return self.method is not None and self.id is None

    def is_response(self) -> bool:
        """是否是响应"""
        return self.method is None and self.id is not None


@dataclass
class MCPCapability:
    """MCP 能力

    Attributes:
        name: 能力名称
        version: 版本
        features: 功能列表
    """
    name: str
    version: str = "1.0.0"
    features: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "version": self.version,
            "features": self.features,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPCapability":
        """从字典创建"""
        return cls(
            name=data.get("name", ""),
            version=data.get("version", "1.0.0"),
            features=data.get("features", []),
        )


@dataclass
class MCPToolInfo:
    """MCP 工具信息

    Attributes:
        name: 工具名称
        description: 描述
        input_schema: 输入模式
        output_schema: 输出模式
    """
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "outputSchema": self.output_schema,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPToolInfo":
        """从字典创建"""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            input_schema=data.get("inputSchema", {}),
            output_schema=data.get("outputSchema", {}),
        )


@dataclass
class MCPResource:
    """MCP 资源

    Attributes:
        uri: 资源 URI
        mime_type: MIME 类型
        name: 名称
        description: 描述
        size: 大小
    """
    uri: str
    mime_type: str = "application/octet-stream"
    name: str = ""
    description: str = ""
    size: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result: Dict[str, Any] = {
            "uri": self.uri,
            "mimeType": self.mime_type,
        }
        if self.name:
            result["name"] = self.name
        if self.description:
            result["description"] = self.description
        if self.size is not None:
            result["size"] = self.size
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPResource":
        """从字典创建"""
        return cls(
            uri=data.get("uri", ""),
            mime_type=data.get("mimeType", "application/octet-stream"),
            name=data.get("name", ""),
            description=data.get("description", ""),
            size=data.get("size"),
        )


@dataclass
class MCPPrompt:
    """MCP 提示

    Attributes:
        name: 提示名称
        description: 描述
        arguments: 参数定义
        template: 模板
    """
    name: str
    description: str = ""
    arguments: List[Dict[str, Any]] = field(default_factory=list)
    template: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPPrompt":
        """从字典创建"""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            arguments=data.get("arguments", []),
            template=data.get("template", ""),
        )

    def render(self, arguments: Dict[str, str]) -> str:
        """渲染提示模板"""
        result = self.template
        for key, value in arguments.items():
            result = result.replace(f"{{{key}}}", value)
        return result


# ---------------------------------------------------------------------------
# MCPServerDiscovery - MCP 服务器发现
# ---------------------------------------------------------------------------
class MCPServerDiscovery:
    """MCP 服务器发现

    发现并管理 MCP 服务器连接。
    """

    def __init__(self):
        self._servers: Dict[str, Dict[str, Any]] = {}
        self._discovered_capabilities: Dict[str, List[MCPCapability]] = {}

    def register_server(
        self,
        server_id: str,
        endpoint: str,
        transport: str = "stdio",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """注册 MCP 服务器

        Args:
            server_id: 服务器唯一标识
            endpoint: 端点地址或命令
            transport: 传输方式 (stdio/sse/websocket)
            config: 额外配置
        """
        self._servers[server_id] = {
            "endpoint": endpoint,
            "transport": transport,
            "config": config or {},
            "connected": False,
            "last_ping": 0.0,
        }

    def discover_from_config(self, config_path: str) -> List[str]:
        """从配置文件发现服务器

        Args:
            config_path: 配置文件路径

        Returns:
            发现的服务器 ID 列表
        """
        discovered: List[str] = []
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            mcp_config = config.get("mcp", {})
            servers = mcp_config.get("servers", {})

            for server_id, server_config in servers.items():
                self.register_server(
                    server_id=server_id,
                    endpoint=server_config.get("endpoint", ""),
                    transport=server_config.get("transport", "stdio"),
                    config=server_config.get("config", {}),
                )
                discovered.append(server_id)

        except FileNotFoundError:
            warnings.warn(f"配置文件未找到: {config_path}")
        except json.JSONDecodeError as e:
            warnings.warn(f"配置文件解析错误: {e}")

        return discovered

    def get_server(self, server_id: str) -> Optional[Dict[str, Any]]:
        """获取服务器信息"""
        return self._servers.get(server_id)

    def list_servers(self) -> List[str]:
        """列出所有服务器 ID"""
        return list(self._servers.keys())

    def remove_server(self, server_id: str) -> bool:
        """移除服务器"""
        if server_id in self._servers:
            del self._servers[server_id]
            self._discovered_capabilities.pop(server_id, None)
            return True
        return False

    def update_connection_status(
        self, server_id: str, connected: bool
    ) -> None:
        """更新连接状态"""
        if server_id in self._servers:
            self._servers[server_id]["connected"] = connected
            if connected:
                self._servers[server_id]["last_ping"] = time.time()

    def store_capabilities(
        self, server_id: str, capabilities: List[MCPCapability]
    ) -> None:
        """存储服务器能力"""
        self._discovered_capabilities[server_id] = capabilities

    def get_capabilities(self, server_id: str) -> List[MCPCapability]:
        """获取服务器能力"""
        return self._discovered_capabilities.get(server_id, [])

    def find_servers_by_capability(self, capability_name: str) -> List[str]:
        """按能力查找服务器"""
        matching: List[str] = []
        for server_id, caps in self._discovered_capabilities.items():
            if any(cap.name == capability_name for cap in caps):
                matching.append(server_id)
        return matching


# ---------------------------------------------------------------------------
# MCPCapabilityNegotiator - MCP 能力协商器
# ---------------------------------------------------------------------------
class MCPCapabilityNegotiator:
    """MCP 能力协商器

    处理客户端和服务器之间的能力协商。
    """

    def __init__(self):
        self._client_capabilities: List[MCPCapability] = []
        self._negotiated_capabilities: Dict[str, MCPCapability] = {}
        self._protocol_version: str = "2024-11-05"

    def set_client_capabilities(self, capabilities: List[MCPCapability]) -> None:
        """设置客户端能力"""
        self._client_capabilities = capabilities

    def create_initialize_request(self) -> MCPMessage:
        """创建初始化请求"""
        return MCPMessage(
            id=str(uuid.uuid4()),
            method="initialize",
            params={
                "protocolVersion": self._protocol_version,
                "capabilities": {
                    cap.name: {"version": cap.version, "features": cap.features}
                    for cap in self._client_capabilities
                },
                "clientInfo": {
                    "name": "agi-unified-framework",
                    "version": "1.0.0",
                },
            },
        )

    def process_initialize_response(self, response: MCPMessage) -> bool:
        """处理初始化响应

        Returns:
            协商是否成功
        """
        if response.error:
            warnings.warn(f"初始化失败: {response.error}")
            return False

        if not response.result:
            return False

        server_info = response.result.get("serverInfo", {})
        server_capabilities = response.result.get("capabilities", {})
        protocol_version = response.result.get("protocolVersion", "")

        # 验证协议版本兼容性
        if not self._is_version_compatible(protocol_version):
            warnings.warn(f"不兼容的协议版本: {protocol_version}")
            return False

        # 协商能力
        for cap in self._client_capabilities:
            if cap.name in server_capabilities:
                server_cap = server_capabilities[cap.name]
                negotiated = MCPCapability(
                    name=cap.name,
                    version=self._negotiate_version(
                        cap.version, server_cap.get("version", "1.0.0")
                    ),
                    features=list(
                        set(cap.features)
                        & set(server_cap.get("features", []))
                    ),
                )
                self._negotiated_capabilities[cap.name] = negotiated

        return True

    def get_negotiated_capabilities(self) -> Dict[str, MCPCapability]:
        """获取协商后的能力"""
        return dict(self._negotiated_capabilities)

    def has_capability(self, name: str) -> bool:
        """检查是否协商了指定能力"""
        return name in self._negotiated_capabilities

    def create_initialized_notification(self) -> MCPMessage:
        """创建初始化完成通知"""
        return MCPMessage(
            method="notifications/initialized",
            params={},
        )

    def _is_version_compatible(self, server_version: str) -> bool:
        """检查版本兼容性"""
        # 简化版本检查：主版本号相同即兼容
        client_parts = self._protocol_version.split(".")
        server_parts = server_version.split(".")

        if not client_parts or not server_parts:
            return False

        return client_parts[0] == server_parts[0]

    def _negotiate_version(
        self, client_version: str, server_version: str
    ) -> str:
        """协商版本（取较低版本）"""
        client_parts = [int(x) for x in client_version.split(".")]
        server_parts = [int(x) for x in server_version.split(".")]

        min_len = min(len(client_parts), len(server_parts))
        for i in range(min_len):
            if client_parts[i] < server_parts[i]:
                return client_version
            elif server_parts[i] < client_parts[i]:
                return server_version

        return client_version if len(client_parts) <= len(server_parts) else server_version


# ---------------------------------------------------------------------------
# MCPToolInvoker - MCP 工具调用器
# ---------------------------------------------------------------------------
class MCPToolInvoker:
    """MCP 工具调用器

    处理 MCP 工具的调用和响应处理。
    """

    def __init__(self):
        self._tool_cache: Dict[str, MCPToolInfo] = {}
        self._call_handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self._pending_calls: Dict[str, Dict[str, Any]] = {}

    def register_tool_handler(
        self, tool_name: str, handler: Callable[[Dict[str, Any]], Any]
    ) -> None:
        """注册工具处理器"""
        self._call_handlers[tool_name] = handler

    def create_call_request(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> MCPMessage:
        """创建工具调用请求"""
        call_id = str(uuid.uuid4())
        self._pending_calls[call_id] = {
            "tool": tool_name,
            "arguments": arguments,
            "timestamp": time.time(),
        }

        return MCPMessage(
            id=call_id,
            method="tools/call",
            params={
                "name": tool_name,
                "arguments": arguments,
            },
        )

    def process_call_response(self, response: MCPMessage) -> ToolResult:
        """处理工具调用响应"""
        call_id = response.id

        if call_id in self._pending_calls:
            del self._pending_calls[call_id]

        if response.error:
            return ToolResult.fail(
                error=f"MCP Error {response.error.get('code')}: {response.error.get('message')}",
                tool_name="",
            )

        if not response.result:
            return ToolResult.fail(
                error="空响应",
                tool_name="",
            )

        content = response.result.get("content", [])
        is_error = response.result.get("isError", False)

        if is_error:
            error_text = self._extract_text_from_content(content)
            return ToolResult.fail(error=error_text, tool_name="")

        data = self._parse_content(content)
        return ToolResult.ok(data=data, tool_name="")

    def _extract_text_from_content(self, content: List[Dict[str, Any]]) -> str:
        """从内容提取文本"""
        texts = []
        for item in content:
            if item.get("type") == "text":
                texts.append(item.get("text", ""))
        return "\n".join(texts)

    def _parse_content(self, content: List[Dict[str, Any]]) -> Any:
        """解析内容"""
        if not content:
            return None

        if len(content) == 1:
            item = content[0]
            if item.get("type") == "text":
                text = item.get("text", "")
                # 尝试解析 JSON
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text
            elif item.get("type") == "image":
                return {
                    "type": "image",
                    "data": item.get("data"),
                    "mimeType": item.get("mimeType"),
                }
            elif item.get("type") == "resource":
                return item.get("resource")

        # 多个内容项
        return content

    def handle_local_call(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> ToolResult:
        """处理本地工具调用"""
        if tool_name not in self._call_handlers:
            return ToolResult.fail(error=f"未知工具: {tool_name}", tool_name=tool_name)

        try:
            handler = self._call_handlers[tool_name]
            result = handler(arguments)
            return ToolResult.ok(data=result, tool_name=tool_name)
        except Exception as e:
            return ToolResult.fail(error=str(e), tool_name=tool_name)

    def cache_tool_info(self, tool_info: MCPToolInfo) -> None:
        """缓存工具信息"""
        self._tool_cache[tool_info.name] = tool_info

    def get_cached_tool(self, tool_name: str) -> Optional[MCPToolInfo]:
        """获取缓存的工具信息"""
        return self._tool_cache.get(tool_name)

    def list_cached_tools(self) -> List[str]:
        """列出缓存的工具"""
        return list(self._tool_cache.keys())

    def convert_to_framework_tool(self, mcp_tool: MCPToolInfo) -> Tool:
        """将 MCP 工具转换为框架工具"""
        # 创建参数定义
        parameters: List[ToolParameter] = []
        schema = mcp_tool.input_schema
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        for name, prop in properties.items():
            param = ToolParameter(
                name=name,
                type=prop.get("type", "string"),
                required=name in required,
                description=prop.get("description", ""),
                default=prop.get("default"),
            )
            parameters.append(param)

        # 创建适配器工具类
        class MCPToolAdapter(Tool):
            def __init__(self, tool_info: MCPToolInfo, invoker: "MCPToolInvoker"):
                super().__init__(
                    name=tool_info.name,
                    description=tool_info.description,
                    parameters=parameters,
                )
                self._tool_info = tool_info
                self._invoker = invoker

            def _execute(self, params: Dict[str, Any]) -> Any:
                # 这里应该调用远程 MCP 服务器
                # 简化实现：返回模拟结果
                return {"tool": self._tool_info.name, "params": params}

        return MCPToolAdapter(mcp_tool, self)


# ---------------------------------------------------------------------------
# MCPResourceAccessor - MCP 资源访问器
# ---------------------------------------------------------------------------
class MCPResourceAccessor:
    """MCP 资源访问器

    处理 MCP 资源的读取、订阅和管理。
    """

    def __init__(self):
        self._resource_cache: Dict[str, Dict[str, Any]] = {}
        self._subscriptions: Set[str] = set()
        self._resource_handlers: Dict[str, Callable[[str], Any]] = {}

    def create_read_request(self, uri: str) -> MCPMessage:
        """创建资源读取请求"""
        return MCPMessage(
            id=str(uuid.uuid4()),
            method="resources/read",
            params={"uri": uri},
        )

    def create_list_request(self) -> MCPMessage:
        """创建资源列表请求"""
        return MCPMessage(
            id=str(uuid.uuid4()),
            method="resources/list",
            params={},
        )

    def create_subscribe_request(self, uri: str) -> MCPMessage:
        """创建订阅请求"""
        self._subscriptions.add(uri)
        return MCPMessage(
            id=str(uuid.uuid4()),
            method="resources/subscribe",
            params={"uri": uri},
        )

    def create_unsubscribe_request(self, uri: str) -> MCPMessage:
        """创建取消订阅请求"""
        self._subscriptions.discard(uri)
        return MCPMessage(
            id=str(uuid.uuid4()),
            method="resources/unsubscribe",
            params={"uri": uri},
        )

    def process_read_response(self, response: MCPMessage) -> Dict[str, Any]:
        """处理资源读取响应"""
        if response.error:
            return {
                "success": False,
                "error": response.error,
            }

        if not response.result:
            return {"success": False, "error": "空响应"}

        contents = response.result.get("contents", [])
        if not contents:
            return {"success": False, "error": "无内容"}

        content = contents[0]
        uri = content.get("uri", "")

        # 缓存资源
        self._resource_cache[uri] = {
            "content": content,
            "timestamp": time.time(),
        }

        return {
            "success": True,
            "uri": uri,
            "mimeType": content.get("mimeType"),
            "text": content.get("text"),
            "blob": content.get("blob"),
        }

    def process_list_response(self, response: MCPMessage) -> List[MCPResource]:
        """处理资源列表响应"""
        if response.error or not response.result:
            return []

        resources = response.result.get("resources", [])
        return [MCPResource.from_dict(r) for r in resources]

    def get_cached_resource(self, uri: str) -> Optional[Dict[str, Any]]:
        """获取缓存的资源"""
        return self._resource_cache.get(uri)

    def clear_cache(self) -> None:
        """清除缓存"""
        self._resource_cache.clear()

    def is_subscribed(self, uri: str) -> bool:
        """检查是否已订阅"""
        return uri in self._subscriptions

    def register_resource_handler(
        self, uri_pattern: str, handler: Callable[[str], Any]
    ) -> None:
        """注册资源处理器"""
        self._resource_handlers[uri_pattern] = handler

    def handle_resource_request(self, uri: str) -> Dict[str, Any]:
        """处理本地资源请求"""
        # 匹配 URI 模式
        for pattern, handler in self._resource_handlers.items():
            if re.match(pattern, uri):
                try:
                    result = handler(uri)
                    return {
                        "success": True,
                        "uri": uri,
                        "content": result,
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "uri": uri,
                        "error": str(e),
                    }

        return {
            "success": False,
            "uri": uri,
            "error": "资源未找到",
        }

    def parse_uri(self, uri: str) -> Dict[str, Any]:
        """解析资源 URI"""
        parsed = urlparse(uri)
        return {
            "scheme": parsed.scheme,
            "netloc": parsed.netloc,
            "path": parsed.path,
            "params": parsed.params,
            "query": parsed.query,
            "fragment": parsed.fragment,
        }


# ---------------------------------------------------------------------------
# MCPPromptHandler - MCP 提示处理器
# ---------------------------------------------------------------------------
class MCPPromptHandler:
    """MCP 提示处理器

    处理 MCP 提示的获取、渲染和管理。
    """

    def __init__(self):
        self._prompt_cache: Dict[str, MCPPrompt] = {}
        self._prompt_handlers: Dict[str, Callable[[Dict[str, str]], str]] = {}
        self._rendered_history: List[Dict[str, Any]] = []

    def create_list_request(self) -> MCPMessage:
        """创建提示列表请求"""
        return MCPMessage(
            id=str(uuid.uuid4()),
            method="prompts/list",
            params={},
        )

    def create_get_request(
        self, name: str, arguments: Optional[Dict[str, str]] = None
    ) -> MCPMessage:
        """创建获取提示请求"""
        params: Dict[str, Any] = {"name": name}
        if arguments:
            params["arguments"] = arguments

        return MCPMessage(
            id=str(uuid.uuid4()),
            method="prompts/get",
            params=params,
        )

    def process_list_response(self, response: MCPMessage) -> List[MCPPrompt]:
        """处理提示列表响应"""
        if response.error or not response.result:
            return []

        prompts = response.result.get("prompts", [])
        result: List[MCPPrompt] = []

        for p in prompts:
            prompt = MCPPrompt.from_dict(p)
            self._prompt_cache[prompt.name] = prompt
            result.append(prompt)

        return result

    def process_get_response(
        self, response: MCPMessage, arguments: Dict[str, str]
    ) -> Optional[str]:
        """处理获取提示响应"""
        if response.error or not response.result:
            return None

        description = response.result.get("description", "")
        messages = response.result.get("messages", [])

        # 记录历史
        self._rendered_history.append({
            "description": description,
            "messages": messages,
            "arguments": arguments,
            "timestamp": time.time(),
        })

        # 提取文本内容
        texts = []
        for msg in messages:
            content = msg.get("content", {})
            if content.get("type") == "text":
                texts.append(content.get("text", ""))

        return "\n".join(texts)

    def render_local_prompt(
        self, name: str, arguments: Dict[str, str]
    ) -> Optional[str]:
        """渲染本地提示"""
        if name in self._prompt_handlers:
            try:
                return self._prompt_handlers[name](arguments)
            except Exception as e:
                warnings.warn(f"渲染提示失败 {name}: {e}")
                return None

        if name in self._prompt_cache:
            prompt = self._prompt_cache[name]
            return prompt.render(arguments)

        return None

    def register_prompt_handler(
        self, name: str, handler: Callable[[Dict[str, str]], str]
    ) -> None:
        """注册提示处理器"""
        self._prompt_handlers[name] = handler

    def register_prompt_template(
        self, name: str, template: str, description: str = ""
    ) -> None:
        """注册提示模板"""
        # 提取参数
        args = self._extract_template_args(template)
        prompt = MCPPrompt(
            name=name,
            description=description,
            arguments=args,
            template=template,
        )
        self._prompt_cache[name] = prompt

    def _extract_template_args(self, template: str) -> List[Dict[str, Any]]:
        """从模板提取参数定义"""
        args: List[Dict[str, Any]] = []
        pattern = r"\{(\w+)(?::([^}]+))?\}"

        for match in re.finditer(pattern, template):
            name = match.group(1)
            default = match.group(2)
            arg = {
                "name": name,
                "description": f"参数 {name}",
                "required": default is None,
            }
            if default:
                arg["default"] = default
            args.append(arg)

        return args

    def get_cached_prompt(self, name: str) -> Optional[MCPPrompt]:
        """获取缓存的提示"""
        return self._prompt_cache.get(name)

    def list_prompts(self) -> List[str]:
        """列出所有提示"""
        return list(self._prompt_cache.keys())

    def get_render_history(self) -> List[Dict[str, Any]]:
        """获取渲染历史"""
        return list(self._rendered_history)

    def clear_history(self) -> None:
        """清除历史"""
        self._rendered_history.clear()


# ---------------------------------------------------------------------------
# MCPAdapter - MCP 主适配器类
# ---------------------------------------------------------------------------
class MCPAdapter:
    """MCP 主适配器

    整合所有 MCP 功能：服务器发现、能力协商、工具调用、资源访问、提示处理。
    """

    def __init__(self):
        self.server_discovery = MCPServerDiscovery()
        self.capability_negotiator = MCPCapabilityNegotiator()
        self.tool_invoker = MCPToolInvoker()
        self.resource_accessor = MCPResourceAccessor()
        self.prompt_handler = MCPPromptHandler()
        self._message_handlers: Dict[str, Callable[[MCPMessage], None]] = {}
        self._initialized = False

    def initialize(
        self,
        client_capabilities: Optional[List[MCPCapability]] = None,
    ) -> MCPMessage:
        """初始化 MCP 会话"""
        if client_capabilities:
            self.capability_negotiator.set_client_capabilities(client_capabilities)

        self._initialized = True
        return self.capability_negotiator.create_initialize_request()

    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized

    def create_request(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> MCPMessage:
        """创建通用请求"""
        return MCPMessage(
            id=str(uuid.uuid4()),
            method=method,
            params=params or {},
        )

    def create_notification(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> MCPMessage:
        """创建通知"""
        return MCPMessage(
            method=method,
            params=params or {},
        )

    def parse_message(self, data: Union[str, bytes, Dict[str, Any]]) -> MCPMessage:
        """解析 MCP 消息"""
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        if isinstance(data, str):
            data = json.loads(data)
        return MCPMessage.from_dict(data)

    def serialize_message(self, message: MCPMessage) -> str:
        """序列化 MCP 消息"""
        return json.dumps(message.to_dict(), ensure_ascii=False)

    def handle_incoming_message(self, message: MCPMessage) -> Optional[MCPMessage]:
        """处理传入消息"""
        if message.method:
            # 请求或通知
            handler = self._message_handlers.get(message.method)
            if handler:
                handler(message)

            # 如果是请求，需要返回响应
            if message.id is not None:
                return self._create_response(message.id, {"status": "ok"})

        return None

    def register_message_handler(
        self, method: str, handler: Callable[[MCPMessage], None]
    ) -> None:
        """注册消息处理器"""
        self._message_handlers[method] = handler

    def _create_response(
        self, request_id: Union[str, int], result: Any
    ) -> MCPMessage:
        """创建响应"""
        return MCPMessage(
            id=request_id,
            result=result,
        )

    def _create_error_response(
        self,
        request_id: Union[str, int],
        code: int,
        message: str,
        data: Optional[Any] = None,
    ) -> MCPMessage:
        """创建错误响应"""
        error: Dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error["data"] = data

        return MCPMessage(
            id=request_id,
            error=error,
        )

    def get_all_capabilities(self) -> Dict[str, Any]:
        """获取所有协商的能力"""
        return {
            "negotiated": self.capability_negotiator.get_negotiated_capabilities(),
            "servers": {
                sid: [c.to_dict() for c in caps]
                for sid, caps in self.server_discovery._discovered_capabilities.items()
            },
        }


__all__ = [
    "MCPAdapter",
    "MCPServerDiscovery",
    "MCPCapabilityNegotiator",
    "MCPToolInvoker",
    "MCPResourceAccessor",
    "MCPPromptHandler",
    "MCPMessage",
    "MCPCapability",
    "MCPToolInfo",
    "MCPResource",
    "MCPPrompt",
    "MCPErrorCode",
]
