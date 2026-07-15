"""
工具执行节点模块

提供高级工具调用功能，包括工具发现、参数验证、
带超时的执行、结果解析、错误处理、重试集成和输出缓存。

Classes:
    ToolNode: 工具执行节点主类
    ToolDiscovery: 工具发现器
    ParameterValidator: 参数验证器
    ToolExecutor: 工具执行器
    ResultParser: 结果解析器
    ToolErrorHandler: 工具错误处理器
    ToolCache: 工具输出缓存
    ToolNodeConfig: 工具节点配置
"""

import copy
import hashlib
import json
import threading
import time
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union


# ============================================================
# 异常类
# ============================================================

class ToolExecutionError(Exception):
    """工具执行异常"""
    def __init__(
        self,
        tool_name: str = "",
        message: str = "",
        cause: Optional[Exception] = None,
    ) -> None:
        self.tool_name = tool_name
        self.cause = cause
        super().__init__(f"工具 '{tool_name}' 执行失败: {message}")


class ToolNotFoundError(ToolExecutionError):
    """工具未找到异常"""
    def __init__(self, tool_name: str = "") -> None:
        super().__init__(tool_name, f"工具 '{tool_name}' 未注册")


class ParameterValidationError(ToolExecutionError):
    """参数验证异常"""
    def __init__(
        self,
        tool_name: str = "",
        errors: Optional[List[str]] = None,
    ) -> None:
        self.validation_errors = errors or []
        message = "; ".join(self.validation_errors)
        super().__init__(tool_name, f"参数验证失败: {message}")


class ToolTimeoutError(ToolExecutionError):
    """工具超时异常"""
    def __init__(
        self,
        tool_name: str = "",
        timeout_seconds: float = 0.0,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        super().__init__(
            tool_name,
            f"执行超时 ({timeout_seconds}s)",
        )


# ============================================================
# 工具定义与注册
# ============================================================

class ParameterType(Enum):
    """参数类型枚举"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    ANY = "any"


@dataclass
class ParameterSpec:
    """参数规格定义"""
    name: str
    param_type: ParameterType = ParameterType.ANY
    required: bool = False
    default: Any = None
    description: str = ""
    choices: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None


@dataclass
class ToolDefinition:
    """工具定义"""
    name: str
    description: str = ""
    function: Optional[Callable[..., Any]] = None
    parameters: List[ParameterSpec] = dataclass_field(default_factory=list)
    timeout: float = 30.0
    retry_count: int = 0
    cache_ttl: float = 0.0
    tags: List[str] = dataclass_field(default_factory=list)
    metadata: Dict[str, Any] = dataclass_field(default_factory=dict)


# ============================================================
# 工具节点配置
# ============================================================

class ToolNodeConfig:
    """
    工具节点配置

    Attributes:
        default_timeout: 默认超时时间
        enable_cache: 是否启用缓存
        cache_max_size: 缓存最大条目数
        cache_default_ttl: 默认缓存过期时间
        enable_validation: 是否启用参数验证
        strict_validation: 是否严格验证（未知参数报错）
        max_retries: 默认最大重试次数
        retry_delay: 重试延迟
    """

    def __init__(
        self,
        default_timeout: float = 30.0,
        enable_cache: bool = True,
        cache_max_size: int = 1000,
        cache_default_ttl: float = 300.0,
        enable_validation: bool = True,
        strict_validation: bool = False,
        max_retries: int = 0,
        retry_delay: float = 1.0,
    ) -> None:
        self.default_timeout = max(0.0, default_timeout)
        self.enable_cache = enable_cache
        self.cache_max_size = max(1, cache_max_size)
        self.cache_default_ttl = max(0.0, cache_default_ttl)
        self.enable_validation = enable_validation
        self.strict_validation = strict_validation
        self.max_retries = max(0, max_retries)
        self.retry_delay = max(0.0, retry_delay)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "default_timeout": self.default_timeout,
            "enable_cache": self.enable_cache,
            "cache_max_size": self.cache_max_size,
            "cache_default_ttl": self.cache_default_ttl,
            "enable_validation": self.enable_validation,
            "strict_validation": self.strict_validation,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
        }


# ============================================================
# 工具发现器
# ============================================================

class ToolDiscovery:
    """
    工具发现器

    管理工具注册、查找和搜索。

    Usage:
        discovery = ToolDiscovery()
        discovery.register(ToolDefinition(
            name="web_search",
            function=search_fn,
            parameters=[ParameterSpec("query", ParameterType.STRING, True)],
        ))
        tool = discovery.find("web_search")
        tools = discovery.search_by_tag("api")
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}
        self._lock = threading.Lock()

    def register(self, definition: ToolDefinition) -> None:
        """注册工具"""
        with self._lock:
            self._tools[definition.name] = definition

    def unregister(self, name: str) -> bool:
        """注销工具"""
        with self._lock:
            return self._tools.pop(name, None) is not None

    def find(self, name: str) -> Optional[ToolDefinition]:
        """查找工具"""
        with self._lock:
            return self._tools.get(name)

    def find_or_raise(self, name: str) -> ToolDefinition:
        """查找工具，未找到则抛出异常"""
        tool = self.find(name)
        if tool is None:
            raise ToolNotFoundError(name)
        return tool

    def list_tools(self) -> List[ToolDefinition]:
        """列出所有已注册工具"""
        with self._lock:
            return list(self._tools.values())

    def search_by_tag(self, tag: str) -> List[ToolDefinition]:
        """按标签搜索工具"""
        with self._lock:
            return [
                t for t in self._tools.values()
                if tag in t.tags
            ]

    def search_by_name(self, pattern: str) -> List[ToolDefinition]:
        """按名称模式搜索工具"""
        with self._lock:
            pattern_lower = pattern.lower()
            return [
                t for t in self._tools.values()
                if pattern_lower in t.name.lower()
            ]

    @property
    def tool_count(self) -> int:
        """已注册工具数量"""
        return len(self._tools)


# ============================================================
# 参数验证器
# ============================================================

class ParameterValidator:
    """
    参数验证器

    验证工具调用参数是否符合规格定义。

    Usage:
        validator = ParameterValidator()
        errors = validator.validate(
            parameters={"query": "hello", "limit": 10},
            specs=[
                ParameterSpec("query", ParameterType.STRING, True),
                ParameterSpec("limit", ParameterType.INTEGER, False, 5),
            ],
        )
    """

    def validate(
        self,
        parameters: Dict[str, Any],
        specs: List[ParameterSpec],
        strict: bool = False,
    ) -> List[str]:
        """
        验证参数

        Args:
            parameters: 输入参数
            specs: 参数规格列表
            strict: 是否严格模式（未知参数报错）

        Returns:
            错误消息列表，空列表表示验证通过
        """
        errors: List[str] = []
        spec_map = {s.name: s for s in specs}

        # 检查必填参数
        for spec in specs:
            if spec.required and spec.name not in parameters:
                if spec.default is None:
                    errors.append(
                        f"缺少必填参数: '{spec.name}'"
                    )

        # 检查未知参数
        if strict:
            for key in parameters:
                if key not in spec_map:
                    errors.append(f"未知参数: '{key}'")

        # 验证每个参数
        for key, value in parameters.items():
            spec = spec_map.get(key)
            if spec is None:
                continue

            param_errors = self._validate_value(value, spec)
            errors.extend(param_errors)

        return errors

    def _validate_value(
        self,
        value: Any,
        spec: ParameterSpec,
    ) -> List[str]:
        """验证单个参数值"""
        errors: List[str] = []

        # 类型检查
        if spec.param_type != ParameterType.ANY:
            type_valid = self._check_type(value, spec.param_type)
            if not type_valid:
                errors.append(
                    f"参数 '{spec.name}' 类型错误: "
                    f"期望 {spec.param_type.value}, "
                    f"实际 {type(value).__name__}"
                )
                return errors  # 类型错误后不再检查其他约束

        # 选择约束
        if spec.choices is not None and value not in spec.choices:
            errors.append(
                f"参数 '{spec.name}' 值 '{value}' "
                f"不在允许范围 {spec.choices} 内"
            )

        # 数值范围
        if isinstance(value, (int, float)):
            if spec.min_value is not None and value < spec.min_value:
                errors.append(
                    f"参数 '{spec.name}' 值 {value} "
                    f"小于最小值 {spec.min_value}"
                )
            if spec.max_value is not None and value > spec.max_value:
                errors.append(
                    f"参数 '{spec.name}' 值 {value} "
                    f"大于最大值 {spec.max_value}"
                )

        # 字符串长度
        if isinstance(value, str):
            if spec.min_length is not None and len(value) < spec.min_length:
                errors.append(
                    f"参数 '{spec.name}' 长度 {len(value)} "
                    f"小于最小长度 {spec.min_length}"
                )
            if spec.max_length is not None and len(value) > spec.max_length:
                errors.append(
                    f"参数 '{spec.name}' 长度 {len(value)} "
                    f"大于最大长度 {spec.max_length}"
                )

            # 正则模式
            if spec.pattern is not None:
                import re
                if not re.match(spec.pattern, value):
                    errors.append(
                        f"参数 '{spec.name}' 值 '{value}' "
                        f"不匹配模式 '{spec.pattern}'"
                    )

        return errors

    def _check_type(
        self,
        value: Any,
        expected: ParameterType,
    ) -> bool:
        """检查值类型"""
        if expected == ParameterType.STRING:
            return isinstance(value, str)
        elif expected == ParameterType.INTEGER:
            return isinstance(value, int) and not isinstance(value, bool)
        elif expected == ParameterType.FLOAT:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        elif expected == ParameterType.BOOLEAN:
            return isinstance(value, bool)
        elif expected == ParameterType.ARRAY:
            return isinstance(value, (list, tuple))
        elif expected == ParameterType.OBJECT:
            return isinstance(value, dict)
        return True

    def apply_defaults(
        self,
        parameters: Dict[str, Any],
        specs: List[ParameterSpec],
    ) -> Dict[str, Any]:
        """应用默认值"""
        result = dict(parameters)
        for spec in specs:
            if spec.name not in result and spec.default is not None:
                result[spec.name] = spec.default
        return result


# ============================================================
# 工具输出缓存
# ============================================================

@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: float
    ttl: float
    hit_count: int = 0
    last_accessed: float = 0.0


class ToolCache:
    """
    工具输出缓存

    缓存工具调用的结果，避免重复执行。

    Usage:
        cache = ToolCache(max_size=100, default_ttl=300.0)
        cache.put("search:hello", {"results": [...]})
        result = cache.get("search:hello")
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float = 300.0,
    ) -> None:
        self.max_size = max(1, max_size)
        self.default_ttl = max(0.0, default_ttl)
        self._entries: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._hits: int = 0
        self._misses: int = 0

    @staticmethod
    def make_key(tool_name: str, params: Dict[str, Any]) -> str:
        """
        生成缓存键

        Args:
            tool_name: 工具名称
            params: 参数字典

        Returns:
            缓存键字符串
        """
        canonical = json.dumps(params, sort_keys=True, default=str)
        raw = f"{tool_name}:{canonical}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None

            # 检查过期
            if entry.ttl > 0:
                if time.time() - entry.created_at > entry.ttl:
                    del self._entries[key]
                    self._misses += 1
                    return None

            entry.hit_count += 1
            entry.last_accessed = time.time()
            self._hits += 1
            return entry.value

    def put(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
    ) -> None:
        """存入缓存"""
        effective_ttl = ttl if ttl is not None else self.default_ttl
        with self._lock:
            # 淘汰策略：如果满了，删除最久未访问的条目
            if len(self._entries) >= self.max_size and key not in self._entries:
                self._evict()

            self._entries[key] = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                ttl=effective_ttl,
                last_accessed=time.time(),
            )

    def invalidate(self, key: str) -> bool:
        """使缓存失效"""
        with self._lock:
            return self._entries.pop(key, None) is not None

    def clear(self) -> int:
        """清除所有缓存"""
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            return count

    def _evict(self) -> None:
        """淘汰最久未访问的缓存条目"""
        if not self._entries:
            return
        oldest_key = min(
            self._entries,
            key=lambda k: self._entries[k].last_accessed,
        )
        del self._entries[oldest_key]

    def cleanup_expired(self) -> int:
        """清理过期条目"""
        now = time.time()
        expired_keys: List[str] = []
        with self._lock:
            for key, entry in self._entries.items():
                if entry.ttl > 0 and now - entry.created_at > entry.ttl:
                    expired_keys.append(key)
            for key in expired_keys:
                del self._entries[key]
        return len(expired_keys)

    @property
    def size(self) -> int:
        """当前缓存大小"""
        return len(self._entries)

    @property
    def hit_rate(self) -> float:
        """缓存命中率"""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            "size": self.size,
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
        }


# ============================================================
# 结果解析器
# ============================================================

class ResultParser:
    """
    工具结果解析器

    解析和标准化工具调用的返回结果。

    Usage:
        parser = ResultParser()
        parsed = parser.parse(raw_result)
    """

    def parse(
        self,
        result: Any,
        extract_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        解析工具结果

        Args:
            result: 原始结果
            extract_keys: 需要提取的键列表

        Returns:
            标准化结果字典
        """
        parsed: Dict[str, Any] = {
            "success": True,
            "data": None,
            "error": None,
            "metadata": {},
        }

        if isinstance(result, dict):
            # 检查是否有错误字段
            if "error" in result:
                parsed["success"] = False
                parsed["error"] = result["error"]
                parsed["data"] = result.get("data")
            elif "success" in result:
                parsed["success"] = bool(result["success"])
                parsed["data"] = result.get("data", result.get("result"))
                parsed["error"] = result.get("error")
            else:
                parsed["data"] = result

            # 提取元数据
            for meta_key in ("metadata", "meta", "_meta"):
                if meta_key in result:
                    parsed["metadata"] = result[meta_key]
                    break

        elif isinstance(result, (list, tuple)):
            parsed["data"] = list(result)

        elif isinstance(result, str):
            # 尝试解析 JSON
            try:
                json_data = json.loads(result)
                parsed["data"] = json_data
            except (json.JSONDecodeError, ValueError):
                parsed["data"] = result

        else:
            parsed["data"] = result

        # 提取指定键
        if extract_keys and isinstance(parsed["data"], dict):
            extracted: Dict[str, Any] = {}
            for key in extract_keys:
                if key in parsed["data"]:
                    extracted[key] = parsed["data"][key]
            parsed["data"] = extracted

        return parsed

    def extract_error(self, result: Any) -> Optional[str]:
        """从结果中提取错误信息"""
        if isinstance(result, dict):
            for error_key in ("error", "err", "message", "errmsg"):
                if error_key in result and result[error_key]:
                    return str(result[error_key])
        if isinstance(result, Exception):
            return str(result)
        return None


# ============================================================
# 工具错误处理器
# ============================================================

class ToolErrorHandler:
    """
    工具错误处理器

    分类和处理工具执行中的各种错误。

    Usage:
        handler = ToolErrorHandler()
        handler.register_handler(TimeoutError, lambda e: {"fallback": True})
        result = handler.handle(error, tool_name="search")
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, Callable[[Exception], Any]] = {}
        self._default_handler: Optional[Callable[[Exception], Any]] = None
        self._error_log: List[Dict[str, Any]] = []

    def register_handler(
        self,
        error_type: Type[Exception],
        handler: Callable[[Exception], Any],
    ) -> None:
        """注册错误处理器"""
        self._handlers[error_type.__name__] = handler

    def set_default_handler(
        self,
        handler: Callable[[Exception], Any],
    ) -> None:
        """设置默认错误处理器"""
        self._default_handler = handler

    def handle(
        self,
        error: Exception,
        tool_name: str = "",
    ) -> Dict[str, Any]:
        """
        处理错误

        Args:
            error: 异常对象
            tool_name: 工具名称

        Returns:
            处理结果字典
        """
        error_info = {
            "tool_name": tool_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": time.time(),
            "handled": False,
            "fallback": None,
        }

        # 查找匹配的处理器
        handler = self._handlers.get(type(error).__name__)
        if handler is not None:
            try:
                fallback = handler(error)
                error_info["handled"] = True
                error_info["fallback"] = fallback
            except Exception as handler_error:
                error_info["handler_error"] = str(handler_error)
        elif self._default_handler is not None:
            try:
                fallback = self._default_handler(error)
                error_info["handled"] = True
                error_info["fallback"] = fallback
            except Exception as handler_error:
                error_info["handler_error"] = str(handler_error)

        self._error_log.append(error_info)
        return error_info

    def classify_error(self, error: Exception) -> str:
        """
        分类错误

        Returns:
            错误类别: retryable, timeout, validation, fatal, unknown
        """
        error_name = type(error).__name__.lower()

        if "timeout" in error_name:
            return "timeout"
        if "validation" in error_name or "value" in error_name:
            return "validation"
        if any(kw in error_name for kw in (
            "connection", "network", "http", "socket"
        )):
            return "retryable"
        if any(kw in error_name for kw in (
            "auth", "permission", "forbidden"
        )):
            return "fatal"
        return "unknown"

    def get_error_log(self, count: int = 10) -> List[Dict[str, Any]]:
        """获取最近的错误日志"""
        return self._error_log[-count:]

    def clear_log(self) -> None:
        """清除错误日志"""
        self._error_log.clear()


# ============================================================
# 工具执行器
# ============================================================

class ToolExecutor:
    """
    工具执行器

    执行工具调用，支持超时控制和重试。

    Usage:
        executor = ToolExecutor(
            discovery=discovery,
            cache=ToolCache(),
            error_handler=ToolErrorHandler(),
        )
        result = executor.execute(
            tool_name="web_search",
            parameters={"query": "hello"},
        )
    """

    def __init__(
        self,
        discovery: Optional[ToolDiscovery] = None,
        cache: Optional[ToolCache] = None,
        error_handler: Optional[ToolErrorHandler] = None,
        validator: Optional[ParameterValidator] = None,
        result_parser: Optional[ResultParser] = None,
        config: Optional[ToolNodeConfig] = None,
    ) -> None:
        self.discovery = discovery or ToolDiscovery()
        self.cache = cache
        self.error_handler = error_handler or ToolErrorHandler()
        self.validator = validator or ParameterValidator()
        self.result_parser = result_parser or ResultParser()
        self.config = config or ToolNodeConfig()

    def execute(
        self,
        tool_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        extract_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        执行工具调用

        Args:
            tool_name: 工具名称
            parameters: 调用参数
            context: 执行上下文
            timeout: 超时时间（覆盖默认值）
            extract_keys: 从结果中提取的键

        Returns:
            解析后的结果字典
        """
        params = parameters or {}
        ctx = context or {}

        # 1. 查找工具
        tool_def = self.discovery.find_or_raise(tool_name)

        # 2. 参数验证
        if self.config.enable_validation and tool_def.parameters:
            errors = self.validator.validate(
                params,
                tool_def.parameters,
                strict=self.config.strict_validation,
            )
            if errors:
                raise ParameterValidationError(tool_name, errors)

            # 应用默认值
            params = self.validator.apply_defaults(params, tool_def.parameters)

        # 3. 缓存检查
        effective_timeout = timeout or tool_def.timeout or self.config.default_timeout
        cache_key: Optional[str] = None
        if self.cache is not None and tool_def.cache_ttl > 0:
            cache_key = ToolCache.make_key(tool_name, params)
            cached = self.cache.get(cache_key)
            if cached is not None:
                parsed = self.result_parser.parse(cached, extract_keys)
                parsed["cached"] = True
                return parsed

        # 4. 执行工具
        result = self._execute_with_timeout(
            tool_def, params, ctx, effective_timeout
        )

        # 5. 缓存结果
        if cache_key is not None and self.cache is not None:
            self.cache.put(cache_key, result, tool_def.cache_ttl)

        # 6. 解析结果
        parsed = self.result_parser.parse(result, extract_keys)
        parsed["tool_name"] = tool_name
        return parsed

    def _execute_with_timeout(
        self,
        tool_def: ToolDefinition,
        params: Dict[str, Any],
        context: Dict[str, Any],
        timeout: float,
    ) -> Any:
        """带超时执行工具"""
        if tool_def.function is None:
            raise ToolExecutionError(
                tool_def.name,
                "工具函数未设置",
            )

        result_holder: List[Any] = [None]
        error_holder: List[Optional[Exception]] = [None]
        done_event = threading.Event()

        def target() -> None:
            try:
                result_holder[0] = tool_def.function(params, context)
            except Exception as e:
                error_holder[0] = e
            finally:
                done_event.set()

        thread = threading.Thread(target=target, daemon=True)
        thread.start()

        completed = done_event.wait(timeout=timeout)

        if not completed:
            raise ToolTimeoutError(tool_def.name, timeout)

        if error_holder[0] is not None:
            error = error_holder[0]
            # 尝试错误处理
            handled = self.error_handler.handle(error, tool_def.name)
            if handled.get("handled") and handled.get("fallback") is not None:
                return handled["fallback"]
            raise ToolExecutionError(
                tool_def.name,
                str(error),
                cause=error,
            )

        return result_holder[0]


# ============================================================
# 工具执行节点
# ============================================================

class ToolNode:
    """
    工具执行节点

    高级工具调用节点，整合发现、验证、执行、缓存和错误处理。

    Usage:
        node = ToolNode(
            node_id="search_node",
            tool_name="web_search",
            discovery=discovery,
            config=ToolNodeConfig(enable_cache=True),
        )
        result = node.execute(
            inputs={"query": "hello world"},
            context={},
        )
    """

    def __init__(
        self,
        node_id: str,
        tool_name: str = "",
        discovery: Optional[ToolDiscovery] = None,
        executor: Optional[ToolExecutor] = None,
        config: Optional[ToolNodeConfig] = None,
        extract_keys: Optional[List[str]] = None,
    ) -> None:
        self.node_id = node_id
        self.tool_name = tool_name
        self.config = config or ToolNodeConfig()
        self.extract_keys = extract_keys

        if executor is not None:
            self.executor = executor
        else:
            cache = ToolCache(
                max_size=self.config.cache_max_size,
                default_ttl=self.config.cache_default_ttl,
            ) if self.config.enable_cache else None
            self.executor = ToolExecutor(
                discovery=discovery,
                cache=cache,
                config=self.config,
            )

        self._execution_count: int = 0
        self._total_duration: float = 0.0
        self._last_result: Optional[Dict[str, Any]] = None

    def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        执行工具节点

        Args:
            inputs: 输入参数
            context: 执行上下文

        Returns:
            执行结果
        """
        start_time = time.time()

        # 从 inputs 中获取工具名（如果未在构造时指定）
        tool = self.tool_name or inputs.get("_tool_name", "")
        params = {k: v for k, v in inputs.items() if not k.startswith("_")}

        result = self.executor.execute(
            tool_name=tool,
            parameters=params,
            context=context or {},
            extract_keys=self.extract_keys,
        )

        duration = time.time() - start_time
        self._execution_count += 1
        self._total_duration += duration
        self._last_result = result

        result["node_id"] = self.node_id
        result["duration"] = round(duration, 4)
        return result

    @property
    def stats(self) -> Dict[str, Any]:
        """获取节点统计"""
        return {
            "node_id": self.node_id,
            "tool_name": self.tool_name,
            "execution_count": self._execution_count,
            "total_duration": round(self._total_duration, 4),
            "avg_duration": (
                round(self._total_duration / self._execution_count, 4)
                if self._execution_count > 0 else 0.0
            ),
        }
