"""MCP (Model Context Protocol) 消息Schema定义模块。

本模块定义了MCP协议中使用的所有数据类，包括工具、资源、提示词、请求和响应等。
所有类都支持JSON序列化和反序列化。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum, auto


class MCPErrorCode(Enum):
    """MCP标准错误码定义。"""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    SERVER_ERROR = -32000
    AUTHENTICATION_ERROR = -32001
    AUTHORIZATION_ERROR = -32002
    RESOURCE_NOT_FOUND = -32003
    TOOL_EXECUTION_ERROR = -32004
    STREAMING_ERROR = -32005


class MCPMethod(Enum):
    """MCP标准方法名定义。"""
    INITIALIZE = "initialize"
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"
    PROMPTS_LIST = "prompts/list"
    PROMPTS_GET = "prompts/get"
    COMPLETION_COMPLETE = "completion/complete"
    NOTIFICATION_INITIALIZED = "notifications/initialized"
    NOTIFICATION_PROGRESS = "notifications/progress"
    NOTIFICATION_RESOURCE_UPDATED = "notifications/resources/updated"
    NOTIFICATION_TOOL_LIST_CHANGED = "notifications/tools/list_changed"


@dataclass
class JSONRPCRequest:
    """JSON-RPC 2.0请求对象。
    
    Attributes:
        jsonrpc: JSON-RPC版本，固定为"2.0"
        method: 请求方法名
        params: 请求参数
        id: 请求标识符，用于匹配响应
    """
    jsonrpc: str = "2.0"
    method: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    id: Optional[Union[str, int]] = None
    
    def to_json(self) -> str:
        """将请求序列化为JSON字符串。"""
        return json.dumps(asdict(self), ensure_ascii=False)
    
    @classmethod
    def from_json(cls, data: str) -> JSONRPCRequest:
        """从JSON字符串反序列化请求对象。"""
        obj = json.loads(data)
        return cls(
            jsonrpc=obj.get("jsonrpc", "2.0"),
            method=obj.get("method", ""),
            params=obj.get("params", {}),
            id=obj.get("id")
        )


@dataclass
class JSONRPCError:
    """JSON-RPC 2.0错误对象。
    
    Attributes:
        code: 错误码
        message: 错误消息
        data: 附加错误数据
    """
    code: int = 0
    message: str = ""
    data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        result = {"code": self.code, "message": self.message}
        if self.data is not None:
            result["data"] = self.data
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> JSONRPCError:
        """从字典创建错误对象。"""
        return cls(
            code=data.get("code", 0),
            message=data.get("message", ""),
            data=data.get("data")
        )


@dataclass
class JSONRPCResponse:
    """JSON-RPC 2.0响应对象。
    
    Attributes:
        jsonrpc: JSON-RPC版本，固定为"2.0"
        result: 响应结果
        error: 错误对象
        id: 请求标识符
    """
    jsonrpc: str = "2.0"
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None
    
    def to_json(self) -> str:
        """将响应序列化为JSON字符串。"""
        data = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error is not None:
            data["error"] = self.error
        else:
            data["result"] = self.result
        return json.dumps(data, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, data: str) -> JSONRPCResponse:
        """从JSON字符串反序列化响应对象。"""
        obj = json.loads(data)
        return cls(
            jsonrpc=obj.get("jsonrpc", "2.0"),
            result=obj.get("result"),
            error=obj.get("error"),
            id=obj.get("id")
        )
    
    @classmethod
    def success(cls, result: Dict[str, Any], request_id: Optional[Union[str, int]]) -> JSONRPCResponse:
        """创建成功响应。"""
        return cls(jsonrpc="2.0", result=result, error=None, id=request_id)
    
    @classmethod
    def error(cls, code: int, message: str, request_id: Optional[Union[str, int]], 
              data: Optional[Dict[str, Any]] = None) -> JSONRPCResponse:
        """创建错误响应。"""
        error_obj = {"code": code, "message": message}
        if data is not None:
            error_obj["data"] = data
        return cls(jsonrpc="2.0", result=None, error=error_obj, id=request_id)


@dataclass
class MCPToolInputSchema:
    """MCP工具输入参数Schema。
    
    Attributes:
        type: 参数类型，通常为"object"
        properties: 参数属性定义
        required: 必需参数列表
        additionalProperties: 是否允许额外属性
    """
    type: str = "object"
    properties: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    additionalProperties: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "type": self.type,
            "properties": self.properties,
            "required": self.required,
            "additionalProperties": self.additionalProperties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MCPToolInputSchema:
        """从字典创建Schema对象。"""
        return cls(
            type=data.get("type", "object"),
            properties=data.get("properties", {}),
            required=data.get("required", []),
            additionalProperties=data.get("additionalProperties", False)
        )
    
    def validate(self, params: Dict[str, Any]) -> List[str]:
        """验证参数是否符合Schema定义。
        
        Args:
            params: 待验证的参数
            
        Returns:
            验证错误列表，空列表表示验证通过
        """
        errors = []
        
        # 检查必需参数
        for req in self.required:
            if req not in params:
                errors.append(f"Missing required parameter: {req}")
        
        # 检查参数类型
        for key, value in params.items():
            if key in self.properties:
                prop_def = self.properties[key]
                expected_type = prop_def.get("type")
                if expected_type and not self._check_type(value, expected_type):
                    errors.append(f"Parameter '{key}' type mismatch: expected {expected_type}")
        
        # 检查是否允许额外属性
        if not self.additionalProperties:
            for key in params:
                if key not in self.properties:
                    errors.append(f"Unknown parameter: {key}")
        
        return errors
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """检查值是否符合期望类型。"""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }
        expected = type_map.get(expected_type)
        if expected is None:
            return True
        return isinstance(value, expected)


@dataclass
class MCPTool:
    """MCP工具定义。
    
    Attributes:
        name: 工具名称
        description: 工具描述
        inputSchema: 输入参数Schema
        handler: 工具处理函数（服务器端使用，不序列化）
    """
    name: str
    description: str
    inputSchema: MCPToolInputSchema
    handler: Optional[Callable[..., Any]] = field(default=None, compare=False, repr=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（不包含handler）。"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.inputSchema.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MCPTool:
        """从字典创建工具对象。"""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            inputSchema=MCPToolInputSchema.from_dict(data.get("inputSchema", {}))
        )
    
    def execute(self, params: Dict[str, Any]) -> Any:
        """执行工具。
        
        Args:
            params: 工具参数
            
        Returns:
            工具执行结果
            
        Raises:
            ValueError: 参数验证失败
            RuntimeError: 工具执行失败
        """
        # 验证参数
        errors = self.inputSchema.validate(params)
        if errors:
            raise ValueError(f"Parameter validation failed: {'; '.join(errors)}")
        
        if self.handler is None:
            raise RuntimeError(f"Tool '{self.name}' has no handler")
        
        # 执行处理函数
        try:
            return self.handler(**params)
        except Exception as e:
            raise RuntimeError(f"Tool execution failed: {str(e)}") from e


@dataclass
class MCPResource:
    """MCP资源定义。
    
    Attributes:
        uri: 资源URI
        name: 资源名称
        description: 资源描述
        mimeType: MIME类型
        size: 资源大小（字节）
    """
    uri: str
    name: str
    description: str = ""
    mimeType: str = "text/plain"
    size: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        result = {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mimeType
        }
        if self.size is not None:
            result["size"] = self.size
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MCPResource:
        """从字典创建资源对象。"""
        return cls(
            uri=data.get("uri", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            mimeType=data.get("mimeType", "text/plain"),
            size=data.get("size")
        )


@dataclass
class MCPResourceContent:
    """MCP资源内容。
    
    Attributes:
        uri: 资源URI
        mimeType: MIME类型
        text: 文本内容（与blob二选一）
        blob: 二进制内容Base64编码（与text二选一）
    """
    uri: str
    mimeType: str = "text/plain"
    text: Optional[str] = None
    blob: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        result = {"uri": self.uri, "mimeType": self.mimeType}
        if self.text is not None:
            result["text"] = self.text
        if self.blob is not None:
            result["blob"] = self.blob
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MCPResourceContent:
        """从字典创建资源内容对象。"""
        return cls(
            uri=data.get("uri", ""),
            mimeType=data.get("mimeType", "text/plain"),
            text=data.get("text"),
            blob=data.get("blob")
        )


@dataclass
class MCPPromptArgument:
    """MCP提示词参数定义。
    
    Attributes:
        name: 参数名称
        description: 参数描述
        required: 是否必需
    """
    name: str
    description: str = ""
    required: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "name": self.name,
            "description": self.description,
            "required": self.required
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MCPPromptArgument:
        """从字典创建参数对象。"""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            required=data.get("required", False)
        )


@dataclass
class MCPPrompt:
    """MCP提示词模板定义。
    
    Attributes:
        name: 提示词名称
        description: 提示词描述
        arguments: 参数定义列表
        template: 提示词模板字符串
    """
    name: str
    description: str = ""
    arguments: List[MCPPromptArgument] = field(default_factory=list)
    template: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "name": self.name,
            "description": self.description,
            "arguments": [arg.to_dict() for arg in self.arguments]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MCPPrompt:
        """从字典创建提示词对象。"""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            arguments=[MCPPromptArgument.from_dict(a) for a in data.get("arguments", [])],
            template=data.get("template", "")
        )
    
    def render(self, arguments: Dict[str, str]) -> str:
        """渲染提示词模板。
        
        Args:
            arguments: 参数值字典
            
        Returns:
            渲染后的提示词文本
            
        Raises:
            ValueError: 缺少必需参数
        """
        # 检查必需参数
        for arg in self.arguments:
            if arg.required and arg.name not in arguments:
                raise ValueError(f"Missing required argument: {arg.name}")
        
        # 简单模板替换
        result = self.template
        for key, value in arguments.items():
            result = result.replace(f"{{{key}}}", str(value))
        
        return result


@dataclass
class MCPInitializeParams:
    """MCP初始化参数。
    
    Attributes:
        protocolVersion: 协议版本
        capabilities: 客户端能力
        clientInfo: 客户端信息
    """
    protocolVersion: str = "2024-11-05"
    capabilities: Dict[str, Any] = field(default_factory=dict)
    clientInfo: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "protocolVersion": self.protocolVersion,
            "capabilities": self.capabilities,
            "clientInfo": self.clientInfo
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MCPInitializeParams:
        """从字典创建初始化参数。"""
        return cls(
            protocolVersion=data.get("protocolVersion", "2024-11-05"),
            capabilities=data.get("capabilities", {}),
            clientInfo=data.get("clientInfo", {})
        )


@dataclass
class MCPServerCapabilities:
    """MCP服务器能力声明。
    
    Attributes:
        logging: 是否支持日志
        prompts: 是否支持提示词
        resources: 是否支持资源
        tools: 是否支持工具
    """
    logging: bool = False
    prompts: bool = False
    resources: bool = False
    tools: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        result = {}
        if self.logging:
            result["logging"] = {}
        if self.prompts:
            result["prompts"] = {}
        if self.resources:
            result["resources"] = {}
        if self.tools:
            result["tools"] = {}
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MCPServerCapabilities:
        """从字典创建能力声明。"""
        return cls(
            logging="logging" in data,
            prompts="prompts" in data,
            resources="resources" in data,
            tools="tools" in data
        )


@dataclass
class MCPInitializeResult:
    """MCP初始化结果。
    
    Attributes:
        protocolVersion: 协议版本
        capabilities: 服务器能力
        serverInfo: 服务器信息
    """
    protocolVersion: str = "2024-11-05"
    capabilities: MCPServerCapabilities = field(default_factory=MCPServerCapabilities)
    serverInfo: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "protocolVersion": self.protocolVersion,
            "capabilities": self.capabilities.to_dict(),
            "serverInfo": self.serverInfo
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MCPInitializeResult:
        """从字典创建初始化结果。"""
        return cls(
            protocolVersion=data.get("protocolVersion", "2024-11-05"),
            capabilities=MCPServerCapabilities.from_dict(data.get("capabilities", {})),
            serverInfo=data.get("serverInfo", {})
        )


@dataclass
class MCPToolCallParams:
    """MCP工具调用参数。
    
    Attributes:
        name: 工具名称
        arguments: 工具参数
    """
    name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "name": self.name,
            "arguments": self.arguments
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MCPToolCallParams:
        """从字典创建调用参数。"""
        return cls(
            name=data.get("name", ""),
            arguments=data.get("arguments", {})
        )


@dataclass
class MCPToolCallResult:
    """MCP工具调用结果。
    
    Attributes:
        content: 结果内容列表
        isError: 是否错误
    """
    content: List[Dict[str, Any]] = field(default_factory=list)
    isError: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "content": self.content,
            "isError": self.isError
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MCPToolCallResult:
        """从字典创建调用结果。"""
        return cls(
            content=data.get("content", []),
            isError=data.get("isError", False)
        )
    
    @classmethod
    def text(cls, text: str, is_error: bool = False) -> MCPToolCallResult:
        """创建文本结果。"""
        return cls(
            content=[{"type": "text", "text": text}],
            isError=is_error
        )
    
    @classmethod
    def image(cls, data: str, mime_type: str = "image/png") -> MCPToolCallResult:
        """创建图片结果。"""
        return cls(
            content=[{
                "type": "image",
                "data": data,
                "mimeType": mime_type
            }],
            isError=False
        )


@dataclass
class MCPProgressNotification:
    """MCP进度通知。
    
    Attributes:
        progressToken: 进度令牌
        progress: 当前进度
        total: 总进度
    """
    progressToken: Union[str, int] = ""
    progress: float = 0.0
    total: float = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "progressToken": self.progressToken,
            "progress": self.progress,
            "total": self.total
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MCPProgressNotification:
        """从字典创建进度通知。"""
        return cls(
            progressToken=data.get("progressToken", ""),
            progress=data.get("progress", 0.0),
            total=data.get("total", 100.0)
        )
    
    @property
    def percentage(self) -> float:
        """计算进度百分比。"""
        if self.total <= 0:
            return 0.0
        return (self.progress / self.total) * 100.0


# 类型别名
MCPRequest = JSONRPCRequest
MCPResponse = JSONRPCResponse
MCPError = JSONRPCError


__all__ = [
    "MCPErrorCode",
    "MCPMethod",
    "JSONRPCRequest",
    "JSONRPCResponse",
    "JSONRPCError",
    "MCPToolInputSchema",
    "MCPTool",
    "MCPResource",
    "MCPResourceContent",
    "MCPPromptArgument",
    "MCPPrompt",
    "MCPInitializeParams",
    "MCPServerCapabilities",
    "MCPInitializeResult",
    "MCPToolCallParams",
    "MCPToolCallResult",
    "MCPProgressNotification",
    "MCPRequest",
    "MCPResponse",
    "MCPError",
]
