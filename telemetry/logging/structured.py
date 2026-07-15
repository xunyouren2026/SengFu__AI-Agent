"""
Structured Logging Module

结构化日志实现，提供JSON格式、上下文注入和日志级别管理。
"""

from __future__ import annotations

import sys
import json
import time
import logging
import threading
from contextvars import ContextVar
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union, TextIO

from ..config import LoggingConfig

# Context variables for log context
_LOG_CONTEXT: ContextVar[Dict[str, Any]] = ContextVar("log_context", default={})


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    
    @property
    def numeric_value(self) -> int:
        """获取数值级别"""
        levels = {
            LogLevel.DEBUG: 10,
            LogLevel.INFO: 20,
            LogLevel.WARNING: 30,
            LogLevel.ERROR: 40,
            LogLevel.CRITICAL: 50
        }
        return levels[self]
    
    @classmethod
    def from_string(cls, level: str) -> "LogLevel":
        """从字符串创建"""
        try:
            return cls[level.upper()]
        except KeyError:
            return cls.INFO
    
    @classmethod
    def from_python_level(cls, level: int) -> "LogLevel":
        """从Python日志级别创建"""
        if level <= 10:
            return cls.DEBUG
        elif level <= 20:
            return cls.INFO
        elif level <= 30:
            return cls.WARNING
        elif level <= 40:
            return cls.ERROR
        else:
            return cls.CRITICAL


@dataclass
class LogContext:
    """
    日志上下文
    
    Attributes:
        trace_id: 追踪ID
        span_id: 跨度ID
        request_id: 请求ID
        user_id: 用户ID
        session_id: 会话ID
        service_name: 服务名称
        environment: 环境
        extra: 额外字段
    """
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    service_name: Optional[str] = None
    environment: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "service_name": self.service_name,
            "environment": self.environment
        }
        result.update(self.extra)
        return {k: v for k, v in result.items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogContext":
        """从字典创建"""
        extra = {k: v for k, v in data.items() 
                if k not in ["trace_id", "span_id", "request_id", 
                           "user_id", "session_id", "service_name", "environment"]}
        return cls(
            trace_id=data.get("trace_id"),
            span_id=data.get("span_id"),
            request_id=data.get("request_id"),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            service_name=data.get("service_name"),
            environment=data.get("environment"),
            extra=extra
        )


@dataclass
class LogRecord:
    """
    日志记录
    
    Attributes:
        timestamp: 时间戳
        level: 日志级别
        message: 消息
        logger_name: 记录器名称
        source_file: 源文件
        source_line: 源行号
        function: 函数名
        context: 上下文
        exception: 异常信息
        thread_id: 线程ID
        process_id: 进程ID
    """
    timestamp: float
    level: LogLevel
    message: str
    logger_name: str
    source_file: Optional[str] = None
    source_line: Optional[int] = None
    function: Optional[str] = None
    context: LogContext = field(default_factory=LogContext)
    exception: Optional[str] = None
    thread_id: int = field(default_factory=lambda: threading.current_thread().ident)
    process_id: int = field(default_factory=lambda: __import__('os').getpid())
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp,
            "timestamp_iso": time.strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ", 
                time.gmtime(self.timestamp)
            ),
            "level": self.level.value,
            "message": self.message,
            "logger": self.logger_name,
            "source": {
                "file": self.source_file,
                "line": self.source_line,
                "function": self.function
            },
            "context": self.context.to_dict(),
            "exception": self.exception,
            "thread_id": self.thread_id,
            "process_id": self.process_id
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False)
    
    def to_text(self, format_str: Optional[str] = None) -> str:
        """转换为文本格式"""
        if format_str:
            return format_str.format(**self.to_dict())
        
        context_str = ""
        ctx_dict = self.context.to_dict()
        if ctx_dict:
            context_str = " | " + " ".join(f"{k}={v}" for k, v in ctx_dict.items())
        
        timestamp_str = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(self.timestamp)
        )
        
        base = f"[{timestamp_str}] {self.level.value:8} {self.logger_name}: {self.message}{context_str}"
        
        if self.exception:
            base += f"\nException: {self.exception}"
        
        return base


class JSONFormatter:
    """JSON格式器"""
    
    def __init__(
        self,
        indent: Optional[int] = None,
        sort_keys: bool = False,
        ensure_ascii: bool = False
    ):
        self._indent = indent
        self._sort_keys = sort_keys
        self._ensure_ascii = ensure_ascii
    
    def format(self, record: LogRecord) -> str:
        """格式化记录"""
        return json.dumps(
            record.to_dict(),
            indent=self._indent,
            sort_keys=self._sort_keys,
            ensure_ascii=self._ensure_ascii,
            default=str
        )


class TextFormatter:
    """文本格式器"""
    
    DEFAULT_FORMAT = "[{timestamp}] {level:8} {logger}: {message}"
    
    def __init__(self, format_str: Optional[str] = None):
        self._format = format_str or self.DEFAULT_FORMAT
    
    def format(self, record: LogRecord) -> str:
        """格式化记录"""
        return record.to_text(self._format)


class ContextualFilter:
    """上下文过滤器"""
    
    def __init__(self, context: Optional[LogContext] = None):
        self._context = context or LogContext()
    
    def filter(self, record: LogRecord) -> bool:
        """过滤记录"""
        # Merge context
        record.context.trace_id = record.context.trace_id or self._context.trace_id
        record.context.span_id = record.context.span_id or self._context.span_id
        record.context.request_id = record.context.request_id or self._context.request_id
        record.context.user_id = record.context.user_id or self._context.user_id
        record.context.service_name = record.context.service_name or self._context.service_name
        record.context.environment = record.context.environment or self._context.environment
        return True


class StructuredLogger:
    """
    结构化日志记录器
    
    提供JSON格式、上下文注入和异步处理功能。
    
    Example:
        >>> config = LoggingConfig(level="INFO", format="json")
        >>> logger = StructuredLogger(config)
        >>> 
        >>> # Set context
        >>> logger.set_context(request_id="123", user_id="user456")
        >>> 
        >>> # Log messages
        >>> logger.info("User logged in", extra={"ip": "192.168.1.1"})
        >>> logger.error("Database connection failed", exc_info=True)
    """
    
    def __init__(self, config: Optional[LoggingConfig] = None):
        """
        初始化日志记录器
        
        Args:
            config: 日志配置
        """
        self._config = config or LoggingConfig()
        self._level = LogLevel.from_string(self._config.level)
        self._handlers: List[Callable[[LogRecord], None]] = []
        self._filters: List[Callable[[LogRecord], bool]] = []
        self._lock = threading.Lock()
        
        # Setup formatter
        if self._config.format == "json":
            self._formatter: Union[JSONFormatter, TextFormatter] = JSONFormatter()
        else:
            self._formatter = TextFormatter()
        
        # Setup output
        self._setup_output()
    
    def _setup_output(self) -> None:
        """设置输出"""
        if self._config.enable_console:
            self.add_handler(self._console_handler)
        
        if self._config.enable_file and self._config.output_path:
            self.add_handler(self._file_handler)
    
    def _console_handler(self, record: LogRecord) -> None:
        """控制台处理器"""
        output = sys.stderr if record.level.numeric_value >= LogLevel.ERROR.numeric_value else sys.stdout
        formatted = self._formatter.format(record)
        output.write(formatted + "\n")
        output.flush()
    
    def _file_handler(self, record: LogRecord) -> None:
        """文件处理器"""
        if self._config.output_path:
            formatted = self._formatter.format(record)
            with open(self._config.output_path, "a") as f:
                f.write(formatted + "\n")
    
    def add_handler(self, handler: Callable[[LogRecord], None]) -> None:
        """
        添加处理器
        
        Args:
            handler: 处理函数
        """
        with self._lock:
            self._handlers.append(handler)
    
    def remove_handler(self, handler: Callable[[LogRecord], None]) -> bool:
        """
        移除处理器
        
        Args:
            handler: 处理函数
            
        Returns:
            是否成功移除
        """
        with self._lock:
            if handler in self._handlers:
                self._handlers.remove(handler)
                return True
            return False
    
    def add_filter(self, filter_fn: Callable[[LogRecord], bool]) -> None:
        """
        添加过滤器
        
        Args:
            filter_fn: 过滤函数
        """
        with self._lock:
            self._filters.append(filter_fn)
    
    def set_context(self, **kwargs: Any) -> None:
        """
        设置日志上下文
        
        Args:
            **kwargs: 上下文字段
        """
        current = _LOG_CONTEXT.get()
        current.update(kwargs)
        _LOG_CONTEXT.set(current)
    
    def clear_context(self) -> None:
        """清除日志上下文"""
        _LOG_CONTEXT.set({})
    
    def get_context(self) -> Dict[str, Any]:
        """获取当前上下文"""
        return _LOG_CONTEXT.get().copy()
    
    def _should_log(self, level: LogLevel) -> bool:
        """检查是否应该记录"""
        return level.numeric_value >= self._level.numeric_value
    
    def _create_record(
        self,
        level: LogLevel,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = False
    ) -> LogRecord:
        """创建日志记录"""
        # Get caller info
        import inspect
        frame = inspect.currentframe()
        if frame:
            frame = frame.f_back.f_back  # Skip this method and the public method
        
        source_file = None
        source_line = None
        function = None
        
        if frame:
            source_file = frame.f_code.co_filename
            source_line = frame.f_lineno
            function = frame.f_code.co_name
        
        # Get exception info
        exception_str = None
        if exc_info:
            import traceback
            exception_str = traceback.format_exc()
        
        # Build context
        context_data = self.get_context()
        if extra:
            context_data.update(extra)
        
        context = LogContext.from_dict(context_data)
        
        return LogRecord(
            timestamp=time.time(),
            level=level,
            message=message,
            logger_name="structured_logger",
            source_file=source_file,
            source_line=source_line,
            function=function,
            context=context,
            exception=exception_str
        )
    
    def _process_record(self, record: LogRecord) -> None:
        """处理日志记录"""
        # Apply filters
        for filter_fn in self._filters:
            if not filter_fn(record):
                return
        
        # Apply handlers
        for handler in self._handlers:
            try:
                handler(record)
            except Exception as e:
                # Fallback to stderr
                sys.stderr.write(f"Log handler error: {e}\n")
    
    def log(
        self,
        level: Union[str, LogLevel],
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = False
    ) -> None:
        """
        记录日志
        
        Args:
            level: 日志级别
            message: 消息
            extra: 额外字段
            exc_info: 是否包含异常信息
        """
        if isinstance(level, str):
            level = LogLevel.from_string(level)
        
        if not self._should_log(level):
            return
        
        record = self._create_record(level, message, extra, exc_info)
        self._process_record(record)
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """记录DEBUG级别日志"""
        self.log(LogLevel.DEBUG, message, extra)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """记录INFO级别日志"""
        self.log(LogLevel.INFO, message, extra)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """记录WARNING级别日志"""
        self.log(LogLevel.WARNING, message, extra)
    
    def error(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = False
    ) -> None:
        """记录ERROR级别日志"""
        self.log(LogLevel.ERROR, message, extra, exc_info)
    
    def critical(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = False
    ) -> None:
        """记录CRITICAL级别日志"""
        self.log(LogLevel.CRITICAL, message, extra, exc_info)
    
    def exception(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """记录异常日志"""
        self.log(LogLevel.ERROR, message, extra, exc_info=True)
    
    def set_level(self, level: Union[str, LogLevel]) -> None:
        """
        设置日志级别
        
        Args:
            level: 日志级别
        """
        if isinstance(level, str):
            level = LogLevel.from_string(level)
        self._level = level
    
    def is_enabled_for(self, level: Union[str, LogLevel]) -> bool:
        """
        检查级别是否启用
        
        Args:
            level: 日志级别
            
        Returns:
            是否启用
        """
        if isinstance(level, str):
            level = LogLevel.from_string(level)
        return self._should_log(level)


def get_logger(name: Optional[str] = None) -> StructuredLogger:
    """
    获取日志记录器
    
    Args:
        name: 记录器名称
        
    Returns:
        结构化日志记录器
    """
    return StructuredLogger()


def set_global_context(**kwargs: Any) -> None:
    """设置全局日志上下文"""
    current = _LOG_CONTEXT.get()
    current.update(kwargs)
    _LOG_CONTEXT.set(current)


def clear_global_context() -> None:
    """清除全局日志上下文"""
    _LOG_CONTEXT.set({})
