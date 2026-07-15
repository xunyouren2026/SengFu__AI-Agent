"""
AGI Unified Framework - LLM Base Module
LLM抽象基类与核心数据模型定义
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional


class MessageRole(str, Enum):
    """消息角色枚举"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"


class FinishReason(str, Enum):
    """生成结束原因"""
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class Usage:
    """Token使用量统计"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self):
        if self.total_tokens == 0 and (self.prompt_tokens > 0 or self.completion_tokens > 0):
            self.total_tokens = self.prompt_tokens + self.completion_tokens

    def add(self, other: "Usage") -> "Usage":
        """合并两个Usage"""
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "Usage":
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
        )


@dataclass
class ToolCall:
    """工具调用信息"""
    id: str = ""
    type: str = "function"
    function_name: str = ""
    arguments: str = ""

    def to_dict(self) -> Dict[str, str]:
        result: Dict[str, Any] = {"id": self.id, "type": self.type}
        result["function"] = {
            "name": self.function_name,
            "arguments": self.arguments,
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        func = data.get("function", {})
        return cls(
            id=data.get("id", ""),
            type=data.get("type", "function"),
            function_name=func.get("name", ""),
            arguments=func.get("arguments", ""),
        )


@dataclass
class Message:
    """聊天消息"""
    role: str = "user"
    content: str = ""
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_openai_dict(self) -> Dict[str, Any]:
        """转换为OpenAI API格式"""
        msg: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            msg["name"] = self.name
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg

    def to_anthropic_dict(self) -> Dict[str, Any]:
        """转换为Anthropic API格式"""
        msg: Dict[str, Any] = {"role": self.role, "content": self.content}
        return msg

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str, tool_calls: Optional[List[ToolCall]] = None) -> "Message":
        return cls(role="assistant", content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: Optional[str] = None) -> "Message":
        return cls(role="tool", content=content, tool_call_id=tool_call_id, name=name)


@dataclass
class ModelInfo:
    """模型信息"""
    name: str = ""
    max_context: int = 4096
    max_output: int = 2048
    supports_streaming: bool = True
    supports_functions: bool = False
    supports_vision: bool = False
    vendor: str = ""
    version: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "max_context": self.max_context,
            "max_output": self.max_output,
            "supports_streaming": self.supports_streaming,
            "supports_functions": self.supports_functions,
            "supports_vision": self.supports_vision,
            "vendor": self.vendor,
            "version": self.version,
            "description": self.description,
        }


@dataclass
class LLMResponse:
    """LLM完整响应"""
    content: str = ""
    usage: Usage = field(default_factory=Usage)
    finish_reason: FinishReason = FinishReason.STOP
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_calls: Optional[List[ToolCall]] = None
    model: str = ""
    id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "content": self.content,
            "usage": self.usage.to_dict(),
            "finish_reason": self.finish_reason.value if isinstance(self.finish_reason, FinishReason) else self.finish_reason,
            "model": self.model,
            "id": self.id,
        }
        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class LLMChunk:
    """流式响应块"""
    delta_content: str = ""
    finish_reason: Optional[FinishReason] = None
    usage: Optional[Usage] = None
    tool_calls_delta: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_final(self) -> bool:
        return self.finish_reason is not None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"delta_content": self.delta_content}
        if self.finish_reason:
            result["finish_reason"] = self.finish_reason.value if isinstance(self.finish_reason, FinishReason) else self.finish_reason
        if self.usage:
            result["usage"] = self.usage.to_dict()
        if self.tool_calls_delta:
            result["tool_calls_delta"] = self.tool_calls_delta
        return result


@dataclass
class GenerateParams:
    """生成参数"""
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    max_tokens: int = 2048
    n: int = 1
    stop: Optional[List[str]] = None
    stream: bool = False
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    seed: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    response_format: Optional[Dict[str, Any]] = None
    user: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_openai_dict(self) -> Dict[str, Any]:
        """转换为OpenAI API参数格式"""
        params: Dict[str, Any] = {}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.n is not None and self.n > 1:
            params["n"] = self.n
        if self.stop:
            params["stop"] = self.stop
        if self.stream:
            params["stream"] = True
        if self.presence_penalty != 0.0:
            params["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty != 0.0:
            params["frequency_penalty"] = self.frequency_penalty
        if self.seed is not None:
            params["seed"] = self.seed
        if self.tools:
            params["tools"] = self.tools
        if self.tool_choice is not None:
            params["tool_choice"] = self.tool_choice
        if self.response_format:
            params["response_format"] = self.response_format
        if self.user:
            params["user"] = self.user
        return params

    def to_anthropic_dict(self) -> Dict[str, Any]:
        """转换为Anthropic API参数格式"""
        params: Dict[str, Any] = {}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.top_k > 0:
            params["top_k"] = self.top_k
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.stop:
            params["stop_sequences"] = self.stop
        if self.seed is not None:
            params["seed"] = self.seed
        return params

    def to_local_dict(self) -> Dict[str, Any]:
        """转换为本地API参数格式"""
        params: Dict[str, Any] = {}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.top_k > 0:
            params["top_k"] = self.top_k
        if self.max_tokens is not None:
            params["num_predict"] = self.max_tokens
        if self.stop:
            params["stop"] = self.stop
        if self.seed is not None:
            params["seed"] = self.seed
        if self.stream:
            params["stream"] = True
        return params


class LLMBackend(ABC):
    """
    LLM抽象基类

    所有LLM后端适配器必须继承此类并实现其抽象方法。
    提供统一的接口用于文本生成、流式输出、文本嵌入和Token计数。
    """

    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        params: Optional[GenerateParams] = None,
    ) -> LLMResponse:
        """
        生成回复

        Args:
            messages: 消息列表
            params: 生成参数

        Returns:
            LLMResponse: 完整的LLM响应

        Raises:
            LLMError: 当生成过程中发生错误时抛出
        """
        ...

    @abstractmethod
    def stream(
        self,
        messages: List[Message],
        params: Optional[GenerateParams] = None,
    ) -> Iterator[LLMChunk]:
        """
        流式生成回复

        Args:
            messages: 消息列表
            params: 生成参数

        Yields:
            LLMChunk: 流式响应块

        Raises:
            LLMError: 当流式生成过程中发生错误时抛出
        """
        ...

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        文本嵌入

        Args:
            texts: 待嵌入的文本列表

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Token计数

        Args:
            text: 待计数的文本

        Returns:
            int: Token数量
        """
        ...

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """
        获取模型信息

        Returns:
            ModelInfo: 模型详细信息
        """
        ...

    def health_check(self) -> bool:
        """
        健康检查

        Returns:
            bool: 后端是否健康可用
        """
        try:
            info = self.get_model_info()
            return info.name != ""
        except Exception:
            return False

    def validate_messages(self, messages: List[Message]) -> bool:
        """
        验证消息列表格式

        Args:
            messages: 消息列表

        Returns:
            bool: 消息格式是否合法
        """
        if not messages:
            return False
        valid_roles = {"system", "user", "assistant", "tool", "function"}
        for msg in messages:
            if msg.role not in valid_roles:
                return False
            if not isinstance(msg.content, str):
                return False
        return True

    def estimate_cost(self, messages: List[Message], params: Optional[GenerateParams] = None) -> float:
        """
        估算请求费用（基类默认实现返回0，子类可覆盖）

        Args:
            messages: 消息列表
            params: 生成参数

        Returns:
            float: 估算费用（美元）
        """
        return 0.0

    def close(self):
        """清理资源"""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class LLMError(Exception):
    """LLM基础异常"""

    def __init__(self, message: str, error_type: str = "unknown", status_code: int = 0, retryable: bool = False):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.status_code = status_code
        self.retryable = retryable


class RateLimitError(LLMError):
    """速率限制错误"""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: float = 1.0):
        super().__init__(message, error_type="rate_limit", status_code=429, retryable=True)
        self.retry_after = retry_after


class TimeoutError(LLMError):
    """超时错误"""

    def __init__(self, message: str = "Request timed out", timeout: float = 30.0):
        super().__init__(message, error_type="timeout", status_code=0, retryable=True)
        self.timeout = timeout


class AuthenticationError(LLMError):
    """认证错误"""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, error_type="authentication", status_code=401, retryable=False)


class InvalidRequestError(LLMError):
    """无效请求错误"""

    def __init__(self, message: str = "Invalid request"):
        super().__init__(message, error_type="invalid_request", status_code=400, retryable=False)


class ModelNotFoundError(LLMError):
    """模型未找到错误"""

    def __init__(self, model: str = ""):
        message = f"Model not found: {model}" if model else "Model not found"
        super().__init__(message, error_type="model_not_found", status_code=404, retryable=False)
        self.model = model


class ServerError(LLMError):
    """服务器错误"""

    def __init__(self, message: str = "Internal server error", status_code: int = 500):
        super().__init__(message, error_type="server_error", status_code=status_code, retryable=True)
