"""工具格式转换模块。

本模块实现将内部Tool定义转换为MCP Tool定义，以及反向转换。
支持多种工具格式的互转。
"""

from __future__ import annotations

import json
import inspect
from typing import Optional, Dict, Any, List, Callable, Union, get_type_hints
from dataclasses import dataclass, field
from enum import Enum

from .schema import MCPTool, MCPToolInputSchema


class ToolFormat(Enum):
    """工具格式枚举。"""
    MCP = "mcp"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LANGCHAIN = "langchain"
    CUSTOM = "custom"


@dataclass
class InternalTool:
    """内部工具定义。
    
    Attributes:
        name: 工具名称
        description: 工具描述
        parameters: 参数定义
        return_type: 返回类型
        handler: 处理函数
        metadata: 元数据
    """
    name: str
    description: str = ""
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    return_type: str = "string"
    handler: Optional[Callable[..., Any]] = field(default=None, compare=False, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> InternalTool:
        """从字典创建。"""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            parameters=data.get("parameters", {}),
            return_type=data.get("return_type", "string"),
            metadata=data.get("metadata", {})
        )


@dataclass
class OpenAITool:
    """OpenAI工具定义格式。
    
    Attributes:
        type: 工具类型，通常为"function"
        function: 函数定义
    """
    type: str = "function"
    function: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "type": self.type,
            "function": self.function
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OpenAITool:
        """从字典创建。"""
        return cls(
            type=data.get("type", "function"),
            function=data.get("function", {})
        )


@dataclass
class AnthropicTool:
    """Anthropic工具定义格式。
    
    Attributes:
        name: 工具名称
        description: 工具描述
        input_schema: 输入Schema
    """
    name: str = ""
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AnthropicTool:
        """从字典创建。"""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            input_schema=data.get("input_schema", {})
        )


class ToolConverter:
    """工具格式转换器。
    
    支持在多种工具格式之间进行转换。
    """
    
    @staticmethod
    def python_type_to_json_type(py_type: type) -> str:
        """将Python类型转换为JSON Schema类型。
        
        Args:
            py_type: Python类型
            
        Returns:
            JSON Schema类型字符串
        """
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            type(None): "null"
        }
        
        # 处理Optional类型
        origin = getattr(py_type, "__origin__", None)
        if origin is Union:
            args = getattr(py_type, "__args__", ())
            non_none_types = [a for a in args if a is not type(None)]
            if len(non_none_types) == 1:
                return ToolConverter.python_type_to_json_type(non_none_types[0])
        
        if py_type in type_map:
            return type_map[py_type]
        
        # 处理泛型类型
        if origin is list:
            return "array"
        if origin is dict:
            return "object"
        
        return "string"
    
    @staticmethod
    def json_type_to_python_type(json_type: str) -> type:
        """将JSON Schema类型转换为Python类型。
        
        Args:
            json_type: JSON Schema类型字符串
            
        Returns:
            Python类型
        """
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }
        return type_map.get(json_type, str)
    
    @staticmethod
    def infer_schema_from_function(func: Callable) -> Dict[str, Any]:
        """从函数推断参数Schema。
        
        Args:
            func: 函数对象
            
        Returns:
            参数Schema字典
        """
        try:
            sig = inspect.signature(func)
            type_hints = get_type_hints(func)
        except Exception:
            return {"type": "object", "properties": {}, "required": []}
        
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name == "self" or param_name == "cls":
                continue
            
            # 获取参数类型
            param_type = type_hints.get(param_name, str)
            json_type = ToolConverter.python_type_to_json_type(param_type)
            
            # 构建属性定义
            prop_def = {"type": json_type}
            
            # 从docstring提取描述
            if func.__doc__:
                # 简单解析docstring中的参数描述
                lines = func.__doc__.split("\n")
                for line in lines:
                    if f":param {param_name}:" in line or f":arg {param_name}:" in line:
                        desc = line.split(":", 2)[-1].strip()
                        if desc:
                            prop_def["description"] = desc
                        break
            
            properties[param_name] = prop_def
            
            # 检查是否必需
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    @staticmethod
    def internal_to_mcp(tool: InternalTool) -> MCPTool:
        """将内部工具转换为MCP工具。
        
        Args:
            tool: 内部工具定义
            
        Returns:
            MCP工具定义
        """
        # 构建输入Schema
        properties = {}
        required = []
        
        for param_name, param_def in tool.parameters.items():
            if isinstance(param_def, dict):
                properties[param_name] = param_def
                if param_def.get("required", False):
                    required.append(param_name)
            else:
                properties[param_name] = {"type": "string", "description": str(param_def)}
        
        input_schema = MCPToolInputSchema(
            type="object",
            properties=properties,
            required=required,
            additionalProperties=False
        )
        
        return MCPTool(
            name=tool.name,
            description=tool.description,
            inputSchema=input_schema,
            handler=tool.handler
        )
    
    @staticmethod
    def mcp_to_internal(tool: MCPTool) -> InternalTool:
        """将MCP工具转换为内部工具。
        
        Args:
            tool: MCP工具定义
            
        Returns:
            内部工具定义
        """
        parameters = {}
        
        for param_name, prop_def in tool.inputSchema.properties.items():
            param_info = dict(prop_def)
            param_info["required"] = param_name in tool.inputSchema.required
            parameters[param_name] = param_info
        
        return InternalTool(
            name=tool.name,
            description=tool.description,
            parameters=parameters,
            handler=tool.handler
        )
    
    @staticmethod
    def internal_to_openai(tool: InternalTool) -> OpenAITool:
        """将内部工具转换为OpenAI格式。
        
        Args:
            tool: 内部工具定义
            
        Returns:
            OpenAI工具定义
        """
        properties = {}
        required = []
        
        for param_name, param_def in tool.parameters.items():
            if isinstance(param_def, dict):
                prop = {k: v for k, v in param_def.items() if k != "required"}
                properties[param_name] = prop
                if param_def.get("required", False):
                    required.append(param_name)
            else:
                properties[param_name] = {"type": "string"}
        
        return OpenAITool(
            type="function",
            function={
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        )
    
    @staticmethod
    def openai_to_internal(tool: OpenAITool) -> InternalTool:
        """将OpenAI工具转换为内部格式。
        
        Args:
            tool: OpenAI工具定义
            
        Returns:
            内部工具定义
        """
        func = tool.function
        params_schema = func.get("parameters", {})
        
        parameters = {}
        properties = params_schema.get("properties", {})
        required = params_schema.get("required", [])
        
        for param_name, prop_def in properties.items():
            param_info = dict(prop_def)
            param_info["required"] = param_name in required
            parameters[param_name] = param_info
        
        return InternalTool(
            name=func.get("name", ""),
            description=func.get("description", ""),
            parameters=parameters
        )
    
    @staticmethod
    def internal_to_anthropic(tool: InternalTool) -> AnthropicTool:
        """将内部工具转换为Anthropic格式。
        
        Args:
            tool: 内部工具定义
            
        Returns:
            Anthropic工具定义
        """
        properties = {}
        required = []
        
        for param_name, param_def in tool.parameters.items():
            if isinstance(param_def, dict):
                prop = {k: v for k, v in param_def.items() if k != "required"}
                properties[param_name] = prop
                if param_def.get("required", False):
                    required.append(param_name)
            else:
                properties[param_name] = {"type": "string"}
        
        return AnthropicTool(
            name=tool.name,
            description=tool.description,
            input_schema={
                "type": "object",
                "properties": properties,
                "required": required
            }
        )
    
    @staticmethod
    def anthropic_to_internal(tool: AnthropicTool) -> InternalTool:
        """将Anthropic工具转换为内部格式。
        
        Args:
            tool: Anthropic工具定义
            
        Returns:
            内部工具定义
        """
        params_schema = tool.input_schema
        
        parameters = {}
        properties = params_schema.get("properties", {})
        required = params_schema.get("required", [])
        
        for param_name, prop_def in properties.items():
            param_info = dict(prop_def)
            param_info["required"] = param_name in required
            parameters[param_name] = param_info
        
        return InternalTool(
            name=tool.name,
            description=tool.description,
            parameters=parameters
        )
    
    @staticmethod
    def mcp_to_openai(tool: MCPTool) -> OpenAITool:
        """将MCP工具转换为OpenAI格式。
        
        Args:
            tool: MCP工具定义
            
        Returns:
            OpenAI工具定义
        """
        return OpenAITool(
            type="function",
            function={
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema.to_dict()
            }
        )
    
    @staticmethod
    def openai_to_mcp(tool: OpenAITool) -> MCPTool:
        """将OpenAI工具转换为MCP格式。
        
        Args:
            tool: OpenAI工具定义
            
        Returns:
            MCP工具定义
        """
        func = tool.function
        params_schema = func.get("parameters", {})
        
        input_schema = MCPToolInputSchema(
            type=params_schema.get("type", "object"),
            properties=params_schema.get("properties", {}),
            required=params_schema.get("required", []),
            additionalProperties=params_schema.get("additionalProperties", True)
        )
        
        return MCPTool(
            name=func.get("name", ""),
            description=func.get("description", ""),
            inputSchema=input_schema
        )
    
    @staticmethod
    def mcp_to_anthropic(tool: MCPTool) -> AnthropicTool:
        """将MCP工具转换为Anthropic格式。
        
        Args:
            tool: MCP工具定义
            
        Returns:
            Anthropic工具定义
        """
        return AnthropicTool(
            name=tool.name,
            description=tool.description,
            input_schema=tool.inputSchema.to_dict()
        )
    
    @staticmethod
    def anthropic_to_mcp(tool: AnthropicTool) -> MCPTool:
        """将Anthropic工具转换为MCP格式。
        
        Args:
            tool: Anthropic工具定义
            
        Returns:
            MCP工具定义
        """
        params_schema = tool.input_schema
        
        input_schema = MCPToolInputSchema(
            type=params_schema.get("type", "object"),
            properties=params_schema.get("properties", {}),
            required=params_schema.get("required", []),
            additionalProperties=params_schema.get("additionalProperties", True)
        )
        
        return MCPTool(
            name=tool.name,
            description=tool.description,
            inputSchema=input_schema
        )


class ToolRegistry:
    """工具注册表。
    
    管理工具的注册、转换和导出。
    """
    
    def __init__(self) -> None:
        """初始化工具注册表。"""
        self._tools: Dict[str, InternalTool] = {}
        self._converter = ToolConverter()
    
    def register(
        self,
        name: str,
        description: str = "",
        parameters: Optional[Dict[str, Dict[str, Any]]] = None,
        handler: Optional[Callable[..., Any]] = None
    ) -> None:
        """注册工具。
        
        Args:
            name: 工具名称
            description: 工具描述
            parameters: 参数定义
            handler: 处理函数
        """
        # 如果没有提供参数定义，从函数推断
        if parameters is None and handler:
            schema = ToolConverter.infer_schema_from_function(handler)
            parameters = {}
            for prop_name, prop_def in schema.get("properties", {}).items():
                param_info = dict(prop_def)
                param_info["required"] = prop_name in schema.get("required", [])
                parameters[prop_name] = param_info
        
        tool = InternalTool(
            name=name,
            description=description,
            parameters=parameters or {},
            handler=handler
        )
        
        self._tools[name] = tool
    
    def register_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> None:
        """通过函数注册工具。
        
        Args:
            func: 函数对象
            name: 工具名称，默认使用函数名
            description: 工具描述，默认使用docstring
        """
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or ""
        
        schema = ToolConverter.infer_schema_from_function(func)
        
        parameters = {}
        for prop_name, prop_def in schema.get("properties", {}).items():
            param_info = dict(prop_def)
            param_info["required"] = prop_name in schema.get("required", [])
            parameters[prop_name] = param_info
        
        tool = InternalTool(
            name=tool_name,
            description=tool_desc.strip(),
            parameters=parameters,
            handler=func
        )
        
        self._tools[tool_name] = tool
    
    def unregister(self, name: str) -> None:
        """注销工具。
        
        Args:
            name: 工具名称
        """
        self._tools.pop(name, None)
    
    def get(self, name: str) -> Optional[InternalTool]:
        """获取工具。
        
        Args:
            name: 工具名称
            
        Returns:
            工具定义
        """
        return self._tools.get(name)
    
    def list_tools(self) -> List[InternalTool]:
        """列出所有工具。
        
        Returns:
            工具列表
        """
        return list(self._tools.values())
    
    def export_mcp(self) -> List[MCPTool]:
        """导出为MCP格式。
        
        Returns:
            MCP工具列表
        """
        return [
            ToolConverter.internal_to_mcp(tool)
            for tool in self._tools.values()
        ]
    
    def export_openai(self) -> List[OpenAITool]:
        """导出为OpenAI格式。
        
        Returns:
            OpenAI工具列表
        """
        return [
            ToolConverter.internal_to_openai(tool)
            for tool in self._tools.values()
        ]
    
    def export_anthropic(self) -> List[AnthropicTool]:
        """导出为Anthropic格式。
        
        Returns:
            Anthropic工具列表
        """
        return [
            ToolConverter.internal_to_anthropic(tool)
            for tool in self._tools.values()
        ]
    
    def import_mcp(self, tools: List[MCPTool]) -> None:
        """导入MCP格式工具。
        
        Args:
            tools: MCP工具列表
        """
        for tool in tools:
            internal = ToolConverter.mcp_to_internal(tool)
            self._tools[internal.name] = internal
    
    def import_openai(self, tools: List[OpenAITool]) -> None:
        """导入OpenAI格式工具。
        
        Args:
            tools: OpenAI工具列表
        """
        for tool in tools:
            internal = ToolConverter.openai_to_internal(tool)
            self._tools[internal.name] = internal
    
    def import_anthropic(self, tools: List[AnthropicTool]) -> None:
        """导入Anthropic格式工具。
        
        Args:
            tools: Anthropic工具列表
        """
        for tool in tools:
            internal = ToolConverter.anthropic_to_internal(tool)
            self._tools[internal.name] = internal
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。
        
        Returns:
            序列化结果
        """
        return {
            name: tool.to_dict()
            for name, tool in self._tools.items()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolRegistry:
        """从字典反序列化。
        
        Args:
            data: 序列化数据
            
        Returns:
            工具注册表
        """
        registry = cls()
        for name, tool_data in data.items():
            tool = InternalTool.from_dict(tool_data)
            registry._tools[name] = tool
        return registry
    
    def __len__(self) -> int:
        """获取工具数量。"""
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        """检查工具是否存在。"""
        return name in self._tools


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Dict[str, Any]]] = None
) -> Callable[[Callable], Callable]:
    """工具装饰器。
    
    Args:
        name: 工具名称
        description: 工具描述
        parameters: 参数定义
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        # 存储元数据
        func._tool_name = name or func.__name__
        func._tool_description = description or func.__doc__ or ""
        func._tool_parameters = parameters
        return func
    
    return decorator


__all__ = [
    "ToolFormat",
    "InternalTool",
    "OpenAITool",
    "AnthropicTool",
    "ToolConverter",
    "ToolRegistry",
    "tool",
]
