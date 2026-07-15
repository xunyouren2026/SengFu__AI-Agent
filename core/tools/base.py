"""
工具基类模块

提供工具抽象基类、工具执行结果、参数定义以及装饰器注册机制。
仅使用 Python 标准库。
"""

import abc
import functools
import hashlib
import inspect
import json
import time
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union


# ---------------------------------------------------------------------------
# 参数类型映射
# ---------------------------------------------------------------------------
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class ToolParameterType(str, Enum):
    """工具参数类型枚举"""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


# ---------------------------------------------------------------------------
# ToolParameter - 参数定义
# ---------------------------------------------------------------------------
@dataclass
class ToolParameter:
    """工具参数定义

    Attributes:
        name: 参数名称
        type: 参数类型（string/integer/number/boolean/array/object）
        required: 是否必填
        default: 默认值
        description: 参数描述
        enum: 可选值列表
        min: 最小值（数值类型）
        max: 最大值（数值类型）
        min_length: 最小长度（字符串/数组）
        max_length: 最大长度（字符串/数组）
        pattern: 正则匹配模式（字符串类型）
        items: 数组元素的类型/定义
        properties: 对象属性定义（对象类型）
    """
    name: str
    type: str = "string"
    required: bool = False
    default: Any = None
    description: str = ""
    enum: Optional[List[Any]] = None
    min: Optional[Union[int, float]] = None
    max: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    items: Optional[Union[Dict, "ToolParameter"]] = None
    properties: Optional[Dict[str, "ToolParameter"]] = None

    def to_json_schema(self) -> dict:
        """将参数定义转换为 JSON Schema 格式"""
        schema: Dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }

        if self.enum is not None:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        if self.min is not None:
            schema["minimum"] = self.min
        if self.max is not None:
            schema["maximum"] = self.max
        if self.min_length is not None:
            schema["minLength"] = self.min_length
        if self.max_length is not None:
            schema["maxLength"] = self.max_length
        if self.pattern is not None:
            schema["pattern"] = self.pattern

        # 数组元素定义
        if self.type == "array" and self.items is not None:
            if isinstance(self.items, ToolParameter):
                schema["items"] = self.items.to_json_schema()
            elif isinstance(self.items, dict):
                schema["items"] = self.items

        # 对象属性定义
        if self.type == "object" and self.properties is not None:
            schema["properties"] = {
                k: v.to_json_schema() for k, v in self.properties.items()
            }
            required_props = [
                k for k, v in self.properties.items() if v.required
            ]
            if required_props:
                schema["required"] = required_props

        return schema


# ---------------------------------------------------------------------------
# ToolResult - 工具执行结果
# ---------------------------------------------------------------------------
@dataclass
class ToolResult:
    """工具执行结果

    Attributes:
        success: 是否执行成功
        data: 返回数据
        error: 错误信息
        duration_ms: 执行耗时（毫秒）
        metadata: 附加元信息
        tool_name: 工具名称
        trace_id: 追踪 ID
    """
    success: bool = True
    data: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_name: str = ""
    trace_id: str = ""

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "tool_name": self.tool_name,
            "trace_id": self.trace_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ToolResult":
        """从字典创建"""
        return cls(
            success=data.get("success", True),
            data=data.get("data"),
            error=data.get("error"),
            duration_ms=data.get("duration_ms", 0.0),
            metadata=data.get("metadata", {}),
            tool_name=data.get("tool_name", ""),
            trace_id=data.get("trace_id", ""),
        )

    @classmethod
    def ok(cls, data: Any = None, tool_name: str = "", **meta) -> "ToolResult":
        """创建成功结果"""
        return cls(
            success=True,
            data=data,
            tool_name=tool_name,
            metadata=meta,
        )

    @classmethod
    def fail(cls, error: str, tool_name: str = "", **meta) -> "ToolResult":
        """创建失败结果"""
        return cls(
            success=False,
            error=error,
            tool_name=tool_name,
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# Tool - 工具抽象基类
# ---------------------------------------------------------------------------
class Tool(abc.ABC):
    """工具抽象基类

    所有自定义工具必须继承此类并实现 execute 方法。
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        version: str = "1.0.0",
        parameters: Optional[Sequence[ToolParameter]] = None,
        tags: Optional[Set[str]] = None,
        category: str = "general",
        timeout: float = 30.0,
    ):
        self.name = name
        self.description = description
        self.version = version
        self._parameters: List[ToolParameter] = list(parameters or [])
        self.tags: Set[str] = set(tags or set())
        self.category = category
        self.timeout = timeout
        self._parameters_schema: Optional[dict] = None

    # ----- 属性 -----

    @property
    def parameters_schema(self) -> dict:
        """获取参数 JSON Schema"""
        if self._parameters_schema is None:
            self._parameters_schema = self._build_parameters_schema()
        return self._parameters_schema

    # ----- 公开方法 -----

    def get_info(self) -> dict:
        """获取工具元信息"""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "parameters": self.parameters_schema,
            "tags": sorted(self.tags),
            "category": self.category,
            "timeout": self.timeout,
            "available": self.is_available(),
        }

    def is_available(self) -> bool:
        """检查工具是否可用（子类可覆盖）"""
        return True

    def validate_params(self, params: dict) -> Tuple[bool, Optional[str]]:
        """校验参数

        Returns:
            (是否合法, 错误信息)
        """
        if not isinstance(params, dict):
            return False, "参数必须是字典类型"

        for param in self._parameters:
            if param.required and param.name not in params:
                if param.default is None:
                    return False, f"缺少必填参数: {param.name}"

            if param.name in params:
                value = params[param.name]
                ok, err = self._validate_single_param(param, value)
                if not ok:
                    return False, err

        return True, None

    def execute(self, params: dict) -> ToolResult:
        """执行工具（带计时、校验和异常捕获）"""
        trace_id = uuid.uuid4().hex[:16]
        start = time.monotonic()

        # 参数校验
        valid, err_msg = self.validate_params(params)
        if not valid:
            return ToolResult.fail(
                error=f"参数校验失败: {err_msg}",
                tool_name=self.name,
                trace_id=trace_id,
            )

        # 可用性检查
        if not self.is_available():
            return ToolResult.fail(
                error="工具当前不可用",
                tool_name=self.name,
                trace_id=trace_id,
            )

        try:
            # 填充默认值
            filled = self._fill_defaults(params)
            data = self._execute(filled)
            duration = (time.monotonic() - start) * 1000
            return ToolResult.ok(
                data=data,
                tool_name=self.name,
                trace_id=trace_id,
                duration_ms=round(duration, 3),
            )
        except Exception as exc:
            duration = (time.monotonic() - start) * 1000
            return ToolResult.fail(
                error=f"{type(exc).__name__}: {exc}",
                tool_name=self.name,
                trace_id=trace_id,
                duration_ms=round(duration, 3),
                metadata={"traceback": traceback.format_exc()},
            )

    @abc.abstractmethod
    def _execute(self, params: dict) -> Any:
        """子类实现的具体执行逻辑"""
        ...

    # ----- 内部方法 -----

    def _fill_defaults(self, params: dict) -> dict:
        """填充默认值"""
        result = dict(params)
        for param in self._parameters:
            if param.name not in result and param.default is not None:
                result[param.name] = param.default
        return result

    def _build_parameters_schema(self) -> dict:
        """构建 JSON Schema"""
        properties = {}
        required = []

        for param in self._parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        schema: Dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema

    def _validate_single_param(
        self, param: ToolParameter, value: Any
    ) -> Tuple[bool, Optional[str]]:
        """校验单个参数"""
        # 类型校验
        ok, err = self._check_type(param.name, value, param.type)
        if not ok:
            return False, err

        # 枚举校验
        if param.enum is not None and value not in param.enum:
            return False, (
                f"参数 '{param.name}' 的值 '{value}' 不在允许列表中: {param.enum}"
            )

        # 数值范围校验
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if param.min is not None and value < param.min:
                return False, (
                    f"参数 '{param.name}' 的值 {value} 小于最小值 {param.min}"
                )
            if param.max is not None and value > param.max:
                return False, (
                    f"参数 '{param.name}' 的值 {value} 大于最大值 {param.max}"
                )

        # 字符串长度校验
        if isinstance(value, str):
            if param.min_length is not None and len(value) < param.min_length:
                return False, (
                    f"参数 '{param.name}' 长度 {len(value)} 小于最小长度 "
                    f"{param.min_length}"
                )
            if param.max_length is not None and len(value) > param.max_length:
                return False, (
                    f"参数 '{param.name}' 长度 {len(value)} 大于最大长度 "
                    f"{param.max_length}"
                )
            if param.pattern is not None:
                import re
                if not re.match(param.pattern, value):
                    return False, (
                        f"参数 '{param.name}' 的值 '{value}' 不匹配模式 "
                        f"'{param.pattern}'"
                    )

        # 数组长度校验
        if isinstance(value, list):
            if param.min_length is not None and len(value) < param.min_length:
                return False, (
                    f"参数 '{param.name}' 元素个数 {len(value)} 小于最小值 "
                    f"{param.min_length}"
                )
            if param.max_length is not None and len(value) > param.max_length:
                return False, (
                    f"参数 '{param.name}' 元素个数 {len(value)} 大于最大值 "
                    f"{param.max_length}"
                )

        return True, None

    @staticmethod
    def _check_type(
        name: str, value: Any, expected_type: str
    ) -> Tuple[bool, Optional[str]]:
        """类型检查"""
        type_checks = {
            "string": lambda v: isinstance(v, str),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
        }
        checker = type_checks.get(expected_type)
        if checker is None:
            return True, None
        if not checker(value):
            actual = type(value).__name__
            return False, (
                f"参数 '{name}' 类型错误: 期望 {expected_type}, 实际 {actual}"
            )
        return True, None


# ---------------------------------------------------------------------------
# FunctionTool - 从函数创建的工具
# ---------------------------------------------------------------------------
class FunctionTool(Tool):
    """将普通函数包装为 Tool 实例"""

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: str = "",
        version: str = "1.0.0",
        parameters: Optional[Sequence[ToolParameter]] = None,
        tags: Optional[Set[str]] = None,
        category: str = "general",
        timeout: float = 30.0,
    ):
        self._func = func
        tool_name = name or func.__name__
        if not description:
            description = (func.__doc__ or "").strip() or f"Tool {tool_name}"
        super().__init__(
            name=tool_name,
            description=description,
            version=version,
            parameters=parameters,
            tags=tags,
            category=category,
            timeout=timeout,
        )

    def _execute(self, params: dict) -> Any:
        return self._func(**params)


# ---------------------------------------------------------------------------
# tool() 装饰器
# ---------------------------------------------------------------------------
def tool(
    name: Optional[str] = None,
    description: str = "",
    version: str = "1.0.0",
    parameters: Optional[Sequence[ToolParameter]] = None,
    tags: Optional[Set[str]] = None,
    category: str = "general",
    timeout: float = 30.0,
) -> Callable:
    """装饰器：将函数注册为工具

    Usage::

        @tool(name="add", description="两数相加")
        def add(a: int, b: int) -> int:
            return a + b

        @tool(parameters=[
            ToolParameter(name="x", type="integer", required=True),
        ])
        def square(x: int) -> int:
            return x * x
    """

    def decorator(func: Callable) -> FunctionTool:
        # 如果没有显式提供 parameters，尝试从函数签名自动推导
        resolved_params = parameters
        if resolved_params is None:
            resolved_params = _infer_parameters(func)

        ft = FunctionTool(
            func=func,
            name=name,
            description=description,
            version=version,
            parameters=resolved_params,
            tags=tags,
            category=category,
            timeout=timeout,
        )
        # 将 Tool 实例附加到函数上，方便外部获取
        func._tool_instance = ft  # type: ignore[attr-defined]
        return ft

    return decorator


# ---------------------------------------------------------------------------
# 辅助：从函数签名自动推导参数定义
# ---------------------------------------------------------------------------
def _infer_parameters(func: Callable) -> List[ToolParameter]:
    """从函数签名和类型注解推导 ToolParameter 列表"""
    sig = inspect.signature(func)
    params: List[ToolParameter] = []

    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue

        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            ptype = "string"
        elif annotation in _TYPE_MAP:
            ptype = _TYPE_MAP[annotation]
        elif hasattr(annotation, "__origin__"):
            origin = annotation.__origin__
            if origin is list:
                ptype = "array"
            elif origin is dict:
                ptype = "object"
            else:
                ptype = "string"
        else:
            ptype = "string"

        has_default = param.default is not inspect.Parameter.empty
        params.append(
            ToolParameter(
                name=pname,
                type=ptype,
                required=not has_default,
                default=param.default if has_default else None,
                description=f"参数 {pname}",
            )
        )

    return params
