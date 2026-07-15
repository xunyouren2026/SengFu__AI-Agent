"""
日志与可观测性 - 日志模块

提供结构化日志、敏感数据过滤和审计日志功能。

导出:
    - LoggerFactory: 结构化日志工厂（单例模式）
    - StructuredLogger: 结构化日志器
    - get_logger: 获取日志器的便捷函数
    - LogRecord: 日志记录数据类
    - ProcessorChain: 处理器链
    - TimestampProcessor: 时间戳处理器
    - ProcessIdProcessor: 进程/线程ID处理器
    - ContextProcessor: 上下文处理器
    - StackTraceProcessor: 异常堆栈处理器
    - JsonFormatter: JSON格式化器
    - TextFormatter: 纯文本格式化器
    - SensitiveDataFilter: 敏感数据过滤器
    - SensitivePattern: 敏感模式定义
    - AuditHandler: 审计日志处理器
    - AuditEvent: 审计事件数据类
    - AuditLog: 审计日志存储
    - AuditEventType: 审计事件类型枚举
"""

from .factory import (
    LoggerFactory,
    StructuredLogger,
    StructuredLogHandler,
    get_logger,
    VERBOSE,
)
from .processors import (
    LogRecord,
    ProcessorChain,
    TimestampProcessor,
    ProcessIdProcessor,
    ContextProcessor,
    StackTraceProcessor,
    JsonFormatter,
    TextFormatter,
    LogProcessor,
)
from .sensitive_filter import (
    SensitiveDataFilter,
    SensitivePattern,
)
from .audit_handler import (
    AuditHandler,
    AuditEvent,
    AuditLog,
    AuditEventType,
)

__all__ = [
    # 工厂与日志器
    "LoggerFactory",
    "StructuredLogger",
    "StructuredLogHandler",
    "get_logger",
    "VERBOSE",
    # 处理器与格式化器
    "LogRecord",
    "ProcessorChain",
    "LogProcessor",
    "TimestampProcessor",
    "ProcessIdProcessor",
    "ContextProcessor",
    "StackTraceProcessor",
    "JsonFormatter",
    "TextFormatter",
    # 敏感数据过滤
    "SensitiveDataFilter",
    "SensitivePattern",
    # 审计日志
    "AuditHandler",
    "AuditEvent",
    "AuditLog",
    "AuditEventType",
]
