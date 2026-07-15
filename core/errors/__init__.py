"""
错误处理模块 - AGI统一框架

提供全局错误码定义、异常类层次结构、异常处理器、
重试策略、熔断器和死信队列等功能。
"""

from .definitions import (
    ErrorCode,
    ErrorSeverity,
    AGIError,
    ConfigError,
    NetworkError,
    ModelError,
    ToolError,
    AgentError,
    SecurityError,
    StorageError,
    WorkflowError,
)
from .handlers import ErrorHandler, ErrorContext, ErrorResponse
from .retry import retry, RetryPolicy, BackoffStrategy, RetryState
from .circuit_breaker import CircuitBreaker, CircuitState
from .dlq import DeadLetterQueue, DLQEntry

__all__ = [
    # 错误码和严重级别
    "ErrorCode",
    "ErrorSeverity",
    # 异常类
    "AGIError",
    "ConfigError",
    "NetworkError",
    "ModelError",
    "ToolError",
    "AgentError",
    "SecurityError",
    "StorageError",
    "WorkflowError",
    # 异常处理
    "ErrorHandler",
    "ErrorContext",
    "ErrorResponse",
    # 重试策略
    "retry",
    "RetryPolicy",
    "BackoffStrategy",
    "RetryState",
    # 熔断器
    "CircuitBreaker",
    "CircuitState",
    # 死信队列
    "DeadLetterQueue",
    "DLQEntry",
]
