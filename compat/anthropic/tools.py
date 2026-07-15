"""
Anthropic Tool Use API 实现模块

提供完整的工具使用功能支持，包括：
- 工具定义和验证
- 工具调用解析
- 工具结果格式化
- 工具管理器

参考: https://docs.anthropic.com/claude/docs/tool-use
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    Union,
    get_type_hints,
)
from inspect import signature, Parameter

from .exceptions import ToolError, ValidationError


# 工具模式类型定义
SchemaType = Literal["string", "integer", "number", "boolean", "array", "object"]


@dataclass
class ToolParameter:
    """
    工具参数定义
    
    用于定义工具函数的参数结构
    """
    type: SchemaType
    description: str
    enum: Optional[List[Any]] = None
    required: bool = True
    properties: Optional[Dict[str, Any]] = None
    items: Optional[Dict[str, Any]] = None
    default: Any = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为JSON Schema格式"""
        result: Dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }
        if self.enum is not None:
            result["enum"] = self.enum
        if self.properties is not None:
            result["properties"] = self.properties
        if self.items is not None:
            result["items"] = self.items
        if self.default is not None:
            result["default"] = self.default
        return result


@dataclass
class Tool:
    """
    工具定义类
    
    封装Anthropic Tool Use API的工具定义格式
    
    示例:
        >>> tool = Tool(
        ...     name="get_weather",
        ...     description="获取指定城市的天气信息",
        ...     parameters={
        ...         "city": ToolParameter(
        ...             type="string",
        ...             description="城市名称",
        ...             required=True
        ...         )
        ...     }
        ... )
    """
    name: str
    description: str
    parameters: Dict[str, ToolParameter]
    required: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """初始化后处理，自动计算required字段"""
        if not self.required:
            self.required = [
                name for name, param in self.parameters.items() 
                if param.required
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为Anthropic API格式
        
        Returns:
            符合Anthropic Tool Schema的字典
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    name: param.to_dict()
                    for name, param in self.parameters.items()
                },
                "required": self.required,
            },
        }
    
    @classmethod
    def from_function(
        cls,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Tool:
        """
        从Python函数自动创建Tool定义
        
        通过函数签名和类型注解自动生成工具定义
        
        Args:
            func: Python函数
            name: 工具名称(默认为函数名)
            description: 工具描述(默认为函数文档字符串)
        
        Returns:
            Tool实例
        
        示例:
            >>> def get_weather(city: str, unit: str = "celsius") -> str:
            ...     \"\"\"获取天气信息\"\"\"
            ...     return f"{city}的天气是..."
            >>>
            >>> tool = Tool.from_function(get_weather)
        """
        func_name = name or func.__name__
        func_desc = description or func.__doc__ or f"Execute {func_name}"
        
        sig = signature(func)
        type_hints = get_type_hints(func)
        
        parameters: Dict[str, ToolParameter] = {}
        
        for param_name, param in sig.parameters.items():
            param_type = type_hints.get(param_name, str)
            param_desc = f"Parameter {param_name}"
            
            # 映射Python类型到JSON Schema类型
            schema_type = cls._python_type_to_schema(param_type)
            
            # 判断是否为必需参数
            is_required = param.default is Parameter.empty
            default_value = None if is_required else param.default
            
            parameters[param_name] = ToolParameter(
                type=schema_type,
                description=param_desc,
                required=is_required,
                default=default_value,
            )
        
        return cls(
            name=func_name,
            description=func_desc,
            parameters=parameters,
        )
    
    @staticmethod
    def _python_type_to_schema(py_type: Type) -> SchemaType:
        """
        将Python类型映射到JSON Schema类型
        
        Args:
            py_type: Python类型
        
        Returns:
            JSON Schema类型字符串
        """
        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }
        
        # 处理Optional类型
        origin = getattr(py_type, "__origin__", None)
        if origin is Union:
            args = getattr(py_type, "__args__", ())
            for arg in args:
                if arg is not type(None):
                    py_type = arg
                    break
        
        return type_mapping.get(py_type, "string")
    
    def validate_input(self, input_data: Dict[str, Any]) -> None:
        """
        验证输入数据是否符合工具定义
        
        Args:
            input_data: 输入参数字典
        
        Raises:
            ValidationError: 验证失败时抛出
        """
        # 检查必需参数
        for req_param in self.required:
            if req_param not in input_data:
                raise ValidationError(
                    f"Missing required parameter: {req_param}"
                )
        
        # 验证参数类型
        for param_name, value in input_data.items():
            if param_name in self.parameters:
                param_def = self.parameters[param_name]
                self._validate_type(param_name, value, param_def.type)
    
    def _validate_type(
        self, 
        name: str, 
        value: Any, 
        expected_type: SchemaType
    ) -> None:
        """
        验证值类型
        
        Args:
            name: 参数名
            value: 参数值
            expected_type: 期望的JSON Schema类型
        
        Raises:
            ValidationError: 类型不匹配时抛出
        """
        type_checks = {
            "string": lambda x: isinstance(x, str),
            "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
            "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
            "boolean": lambda x: isinstance(x, bool),
            "array": lambda x: isinstance(x, list),
            "object": lambda x: isinstance(x, dict),
        }
        
        check = type_checks.get(expected_type)
        if check and not check(value):
            raise ValidationError(
                f"Parameter '{name}' should be of type {expected_type}, "
                f"got {type(value).__name__}"
            )


@dataclass
class ToolUseBlock:
    """
    工具使用块
    
    表示Claude请求调用工具的响应块
    """
    id: str
    name: str
    input: Dict[str, Any]
    type: str = "tool_use"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为API格式"""
        return {
            "type": self.type,
            "id": self.id,
            "name": self.name,
            "input": self.input,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolUseBlock:
        """从API响应创建ToolUseBlock"""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            input=data.get("input", {}),
            type=data.get("type", "tool_use"),
        )


@dataclass
class ToolResult:
    """
    工具执行结果
    
    封装工具执行后的返回结果
    """
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]]]
    is_error: bool = False
    type: str = "tool_result"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为API格式"""
        result: Dict[str, Any] = {
            "type": self.type,
            "tool_use_id": self.tool_use_id,
            "content": self.content,
        }
        if self.is_error:
            result["is_error"] = True
        return result
    
    @classmethod
    def success(
        cls, 
        tool_use_id: str, 
        content: Union[str, Dict[str, Any]]
    ) -> ToolResult:
        """
        创建成功结果
        
        Args:
            tool_use_id: 对应的工具使用ID
            content: 结果内容
        
        Returns:
            ToolResult实例
        """
        if isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False)
        return cls(
            tool_use_id=tool_use_id,
            content=content,
            is_error=False,
        )
    
    @classmethod
    def error(
        cls, 
        tool_use_id: str, 
        error_message: str
    ) -> ToolResult:
        """
        创建错误结果
        
        Args:
            tool_use_id: 对应的工具使用ID
            error_message: 错误信息
        
        Returns:
            ToolResult实例
        """
        return cls(
            tool_use_id=tool_use_id,
            content=error_message,
            is_error=True,
        )


@dataclass
class ToolChoice:
    """
    工具选择配置
    
    控制Claude如何使用工具
    """
    type: Literal["auto", "any", "tool"]
    name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为API格式"""
        result: Dict[str, Any] = {"type": self.type}
        if self.name is not None:
            result["name"] = self.name
        return result
    
    @classmethod
    def auto(cls) -> ToolChoice:
        """自动选择工具"""
        return cls(type="auto")
    
    @classmethod
    def any(cls) -> ToolChoice:
        """必须使用任意工具"""
        return cls(type="any")
    
    @classmethod
    def tool(cls, name: str) -> ToolChoice:
        """强制使用指定工具"""
        return cls(type="tool", name=name)


class ToolManager:
    """
    工具管理器
    
    管理多个工具的注册、查找和执行
    
    示例:
        >>> manager = ToolManager()
        >>> manager.register_tool(get_weather_tool)
        >>> result = manager.execute_tool("get_weather", {"city": "Beijing"})
    """
    
    def __init__(self):
        """初始化工具管理器"""
        self._tools: Dict[str, Tool] = {}
        self._handlers: Dict[str, Callable] = {}
    
    def register_tool(
        self, 
        tool: Tool, 
        handler: Optional[Callable] = None
    ) -> None:
        """
        注册工具
        
        Args:
            tool: 工具定义
            handler: 工具执行函数(可选)
        
        Raises:
            ToolError: 工具名称冲突时抛出
        """
        if tool.name in self._tools:
            raise ToolError(
                f"Tool '{tool.name}' is already registered",
                tool_name=tool.name
            )
        
        self._tools[tool.name] = tool
        
        if handler is not None:
            self._handlers[tool.name] = handler
    
    def register_from_function(
        self, 
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Tool:
        """
        从函数注册工具
        
        Args:
            func: Python函数
            name: 工具名称
            description: 工具描述
        
        Returns:
            创建的Tool实例
        """
        tool = Tool.from_function(func, name, description)
        self.register_tool(tool, func)
        return tool
    
    def unregister_tool(self, name: str) -> None:
        """
        注销工具
        
        Args:
            name: 工具名称
        """
        self._tools.pop(name, None)
        self._handlers.pop(name, None)
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """
        获取工具定义
        
        Args:
            name: 工具名称
        
        Returns:
            Tool实例或None
        """
        return self._tools.get(name)
    
    def get_handler(self, name: str) -> Optional[Callable]:
        """
        获取工具处理器
        
        Args:
            name: 工具名称
        
        Returns:
            处理函数或None
        """
        return self._handlers.get(name)
    
    def list_tools(self) -> List[Tool]:
        """
        获取所有已注册工具
        
        Returns:
            Tool列表
        """
        return list(self._tools.values())
    
    def to_api_format(self) -> List[Dict[str, Any]]:
        """
        转换为API格式
        
        Returns:
            Anthropic API可用的工具定义列表
        """
        return [tool.to_dict() for tool in self._tools.values()]
    
    def execute_tool(
        self, 
        name: str, 
        input_data: Dict[str, Any]
    ) -> ToolResult:
        """
        执行工具
        
        Args:
            name: 工具名称
            input_data: 输入参数
        
        Returns:
            ToolResult实例
        
        Raises:
            ToolError: 工具未找到或执行失败时抛出
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"Tool '{name}' not found", tool_name=name)
        
        handler = self._handlers.get(name)
        if handler is None:
            raise ToolError(
                f"No handler registered for tool '{name}'",
                tool_name=name
            )
        
        # 验证输入
        try:
            tool.validate_input(input_data)
        except ValidationError as e:
            return ToolResult.error(
                tool_use_id="",
                error_message=str(e)
            )
        
        # 执行工具
        try:
            result = handler(**input_data)
            
            # 处理返回值
            if isinstance(result, ToolResult):
                return result
            elif isinstance(result, dict):
                return ToolResult.success("", result)
            else:
                return ToolResult.success("", str(result))
                
        except Exception as e:
            return ToolResult.error(
                tool_use_id="",
                error_message=f"Tool execution failed: {str(e)}"
            )
    
    def execute_tool_use(self, tool_use: ToolUseBlock) -> ToolResult:
        """
        执行ToolUseBlock
        
        Args:
            tool_use: 工具使用块
        
        Returns:
            ToolResult实例
        """
        result = self.execute_tool(tool_use.name, tool_use.input)
        result.tool_use_id = tool_use.id
        return result
    
    def clear(self) -> None:
        """清空所有注册的工具"""
        self._tools.clear()
        self._handlers.clear()
    
    def __contains__(self, name: str) -> bool:
        """检查工具是否已注册"""
        return name in self._tools
    
    def __len__(self) -> int:
        """获取已注册工具数量"""
        return len(self._tools)


# 便捷函数
def create_tool(
    name: str,
    description: str,
    parameters: Dict[str, ToolParameter],
) -> Tool:
    """
    快速创建工具定义
    
    Args:
        name: 工具名称
        description: 工具描述
        parameters: 参数字典
    
    Returns:
        Tool实例
    """
    return Tool(
        name=name,
        description=description,
        parameters=parameters,
    )


def tool_from_function(
    func: Callable,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Tool:
    """
    从函数创建工具定义
    
    Args:
        func: Python函数
        name: 工具名称
        description: 工具描述
    
    Returns:
        Tool实例
    """
    return Tool.from_function(func, name, description)
