"""
结构化日志工厂模块

提供日志工厂、结构化日志器和上下文管理功能。
支持JSON和纯文本两种输出格式，可绑定上下文信息。
"""

import json
import logging
import logging.handlers
import os
import sys
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, TextIO, Union

from .processors import (
    ContextProcessor,
    JsonFormatter,
    LogRecord,
    ProcessorChain,
    ProcessIdProcessor,
    StackTraceProcessor,
    TextFormatter,
    TimestampProcessor,
)
from .sensitive_filter import SensitiveDataFilter


# ---------------------------------------------------------------------------
# 自定义日志级别
# ---------------------------------------------------------------------------

# 扩展日志级别
VERBOSE = 5  # 比 DEBUG 更详细

def _register_custom_levels() -> None:
    """注册自定义日志级别。"""
    if not logging.getLevelName("VERBOSE") == "Level VERBOSE":
        logging.addLevelName(VERBOSE, "VERBOSE")


_register_custom_levels()


# ---------------------------------------------------------------------------
# 结构化日志 Handler
# ---------------------------------------------------------------------------

class StructuredLogHandler(logging.Handler):
    """结构化日志处理器。

    将标准 logging.LogRecord 转换为结构化 LogRecord，
    经过处理器链处理后输出。
    """

    def __init__(
        self,
        processor_chain: Optional[ProcessorChain] = None,
        formatter: Optional[Union[JsonFormatter, TextFormatter]] = None,
        stream: Optional[TextIO] = None,
        sensitive_filter: Optional[SensitiveDataFilter] = None,
    ):
        """初始化结构化日志处理器。

        Args:
            processor_chain: 处理器链
            formatter: 格式化器
            stream: 输出流，默认为 sys.stderr
            sensitive_filter: 敏感数据过滤器
        """
        super().__init__()
        self._processor_chain = processor_chain or ProcessorChain()
        self._formatter = formatter or JsonFormatter()
        self._stream = stream or sys.stderr
        self._sensitive_filter = sensitive_filter

    def emit(self, record: logging.LogRecord) -> None:
        """处理并输出日志记录。

        Args:
            record: 标准 logging.LogRecord
        """
        try:
            # 转换为结构化记录
            struct_record = self._convert_record(record)

            # 通过处理器链处理
            struct_record = self._processor_chain.process(struct_record)

            # 敏感数据过滤
            if self._sensitive_filter:
                struct_record.message = self._sensitive_filter.filter(struct_record.message)
                if struct_record.context:
                    struct_record.context = self._sensitive_filter.filter_dict(struct_record.context)

            # 格式化输出
            output = self._formatter.format(struct_record)

            # 写入输出流
            self._stream.write(output + "\n")
            self._stream.flush()

        except Exception:
            self.handleError(record)

    def _convert_record(self, record: logging.LogRecord) -> LogRecord:
        """将标准 LogRecord 转换为结构化 LogRecord。

        Args:
            record: 标准 logging.LogRecord

        Returns:
            结构化 LogRecord
        """
        # 提取额外字段
        extra = {}
        context = {}
        for key, value in record.__dict__.items():
            if key.startswith('_'):
                continue
            if key in (
                'name', 'msg', 'args', 'created', 'relativeCreated',
                'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
                'filename', 'module', 'levelno', 'levelname', 'pathname',
                'thread', 'threadName', 'process', 'message', 'msecs',
                'taskName',
            ):
                continue
            if key in ('request_id', 'agent_id', 'session_id', 'trace_id',
                       'span_id', 'user_id', 'tenant_id'):
                context[key] = value
            else:
                extra[key] = value

        # 处理异常信息
        exception = None
        stack_trace = None
        if record.exc_info and record.exc_info[0] is not None:
            import traceback
            exception = self._format_exception(record.exc_info)
            stack_trace = ''.join(traceback.format_exception(*record.exc_info))
        elif record.exc_text:
            exception = record.exc_text
            stack_trace = record.exc_text

        # 处理 stack_info
        if record.stack_info:
            stack_trace = record.stack_info

        return LogRecord(
            timestamp=datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            level=record.levelname,
            message=record.getMessage(),
            logger_name=record.name,
            module=record.module,
            function=record.funcName,
            line_no=record.lineno,
            process_id=record.process,
            thread_id=record.thread,
            thread_name=record.threadName,
            context=context,
            exception=exception,
            stack_trace=stack_trace,
            extra=extra,
        )

    @staticmethod
    def _format_exception(exc_info: Any) -> str:
        """格式化异常信息。

        Args:
            exc_info: sys.exc_info() 返回的三元组

        Returns:
            格式化后的异常字符串
        """
        import traceback
        return ''.join(traceback.format_exception_only(exc_info[0], exc_info[1])).strip()


# ---------------------------------------------------------------------------
# StructuredLogger: 结构化日志器
# ---------------------------------------------------------------------------

class StructuredLogger:
    """结构化日志器。

    封装标准 logging.Logger，提供结构化日志输出能力。
    支持绑定上下文信息，自动附加到每条日志。

    使用示例::

        logger = StructuredLogger("my_app")
        logger.bind_context(request_id="abc123", agent_id="agent-1")
        logger.info("Processing request", user_id="user42")
    """

    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        output_format: str = "json",
        sensitive_filter: Optional[SensitiveDataFilter] = None,
        context_processor: Optional[ContextProcessor] = None,
    ):
        """初始化结构化日志器。

        Args:
            name: Logger名称
            level: 日志级别
            output_format: 输出格式，"json" 或 "text"
            sensitive_filter: 敏感数据过滤器
            context_processor: 上下文处理器
        """
        self._name = name
        self._logger = logging.getLogger(f"structured.{name}")
        self._logger.setLevel(level)
        self._logger.propagate = False

        # 构建处理器链
        self._processor_chain = ProcessorChain()
        self._processor_chain.add_processor(TimestampProcessor())
        self._processor_chain.add_processor(ProcessIdProcessor())

        # 上下文处理器
        self._context_processor = context_processor or ContextProcessor()
        self._processor_chain.add_processor(self._context_processor)
        self._processor_chain.add_processor(StackTraceProcessor())

        # 格式化器
        if output_format == "text":
            formatter = TextFormatter()
        else:
            formatter = JsonFormatter(indent=None, ensure_ascii=False)

        # 创建并添加Handler
        handler = StructuredLogHandler(
            processor_chain=self._processor_chain,
            formatter=formatter,
            sensitive_filter=sensitive_filter,
        )
        handler.setLevel(level)
        self._logger.addHandler(handler)

        # 线程本地上下文
        self._local = threading.local()

    def bind_context(self, **kwargs: Any) -> None:
        """绑定上下文信息。

        绑定的上下文会自动附加到后续所有日志记录中。

        Args:
            **kwargs: 上下文键值对（如 request_id, agent_id, session_id）
        """
        self._context_processor.set_context(**kwargs)

    def unbind_context(self, *keys: str) -> None:
        """解绑指定的上下文字段。

        Args:
            *keys: 要解绑的上下文键名
        """
        ctx = self._context_processor.get_context()
        for key in keys:
            ctx.pop(key, None)
        # 重新设置上下文
        self._context_processor.clear_context()
        self._context_processor.set_context(**ctx)

    def clear_context(self) -> None:
        """清除所有绑定的上下文信息。"""
        self._context_processor.clear_context()

    def get_context(self) -> Dict[str, Any]:
        """获取当前绑定的上下文。

        Returns:
            上下文字典
        """
        return self._context_processor.get_context()

    def _log(self, level: int, msg: str, **kwargs: Any) -> None:
        """内部日志方法。

        Args:
            level: 日志级别
            msg: 日志消息
            **kwargs: 额外字段（会合并到日志记录中）
        """
        extra = {}
        context_extra = {}
        for key, value in kwargs.items():
            if key in ('request_id', 'agent_id', 'session_id', 'trace_id',
                       'span_id', 'user_id', 'tenant_id'):
                context_extra[key] = value
            else:
                extra[key] = value

        # 合并上下文
        if context_extra:
            self._context_processor.set_context(**context_extra)

        self._logger.log(level, msg, extra=extra)

    def debug(self, msg: str, **kwargs: Any) -> None:
        """记录DEBUG级别日志。"""
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        """记录INFO级别日志。"""
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        """记录WARNING级别日志。"""
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        """记录ERROR级别日志。"""
        self._log(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs: Any) -> None:
        """记录CRITICAL级别日志。"""
        self._log(logging.CRITICAL, msg, **kwargs)

    def verbose(self, msg: str, **kwargs: Any) -> None:
        """记录VERBOSE级别日志。"""
        self._log(VERBOSE, msg, **kwargs)

    def exception(self, msg: str, **kwargs: Any) -> None:
        """记录异常日志（自动附带堆栈信息）。"""
        self._log(logging.ERROR, msg, **kwargs)
        # 使用标准logger的exception方法来记录堆栈
        import sys
        exc_info = sys.exc_info()
        if exc_info and exc_info[0] is not None:
            self._logger.log(logging.ERROR, msg, exc_info=exc_info, extra=kwargs)

    @property
    def name(self) -> str:
        """Logger名称。"""
        return self._name

    def set_level(self, level: Union[int, str]) -> None:
        """设置日志级别。

        Args:
            level: 日志级别（int或字符串如 "DEBUG", "INFO"）
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        self._logger.setLevel(level)
        for handler in self._logger.handlers:
            handler.setLevel(level)

    def add_handler(self, handler: logging.Handler) -> None:
        """添加额外的日志处理器。

        Args:
            handler: 标准 logging.Handler
        """
        self._logger.addHandler(handler)

    def add_file_handler(
        self,
        filepath: str,
        level: int = logging.DEBUG,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
    ) -> None:
        """添加文件日志处理器。

        Args:
            filepath: 日志文件路径
            level: 日志级别
            max_bytes: 单个文件最大字节数
            backup_count: 保留的备份文件数
        """
        # 确保目录存在
        log_dir = os.path.dirname(filepath)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            filepath,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8',
        )
        file_handler.setLevel(level)

        # 使用纯文本格式
        text_formatter = TextFormatter()
        structured_handler = StructuredLogHandler(
            processor_chain=self._processor_chain,
            formatter=text_formatter,
            stream=file_handler.stream,
        )
        structured_handler.setLevel(level)
        # 直接使用RotatingFileHandler
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        self._logger.addHandler(file_handler)


# ---------------------------------------------------------------------------
# LoggerFactory: 结构化日志工厂
# ---------------------------------------------------------------------------

class LoggerFactory:
    """结构化日志工厂。

    管理日志器的创建和配置，支持全局默认配置。
    使用单例模式确保全局一致性。

    使用示例::

        factory = LoggerFactory.get_instance()
        factory.configure(output_format="json", level="INFO")
        logger = factory.get_logger("my_module")
        logger.info("Hello world")
    """

    _instance: Optional["LoggerFactory"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        output_format: str = "json",
        level: Union[int, str] = logging.INFO,
        enable_sensitive_filter: bool = True,
        timezone_str: Optional[str] = None,
    ):
        """初始化日志工厂。

        Args:
            output_format: 默认输出格式，"json" 或 "text"
            level: 默认日志级别
            enable_sensitive_filter: 是否启用敏感数据过滤
            timezone_str: 时区字符串
        """
        self._output_format = output_format
        self._level = level if isinstance(level, int) else getattr(logging, level.upper(), logging.INFO)
        self._timezone_str = timezone_str
        self._loggers: Dict[str, StructuredLogger] = {}
        self._sensitive_filter = SensitiveDataFilter(enabled=enable_sensitive_filter)
        self._factory_lock = threading.Lock()

    @classmethod
    def get_instance(cls, **kwargs: Any) -> "LoggerFactory":
        """获取工厂单例。

        Args:
            **kwargs: 初始化参数（仅在首次调用时生效）

        Returns:
            LoggerFactory 单例
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(**kwargs)
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """重置工厂单例（主要用于测试）。"""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._loggers.clear()
            cls._instance = None

    def configure(
        self,
        output_format: Optional[str] = None,
        level: Optional[Union[int, str]] = None,
        enable_sensitive_filter: Optional[bool] = None,
        timezone_str: Optional[str] = None,
    ) -> None:
        """重新配置工厂。

        Args:
            output_format: 输出格式
            level: 日志级别
            enable_sensitive_filter: 是否启用敏感数据过滤
            timezone_str: 时区字符串
        """
        if output_format is not None:
            self._output_format = output_format
        if level is not None:
            self._level = level if isinstance(level, int) else getattr(logging, level.upper(), logging.INFO)
        if enable_sensitive_filter is not None:
            self._sensitive_filter.enabled = enable_sensitive_filter
        if timezone_str is not None:
            self._timezone_str = timezone_str

    def get_logger(self, name: str, **kwargs: Any) -> StructuredLogger:
        """获取带上下文的结构化日志器。

        如果同名日志器已存在则返回缓存实例。

        Args:
            name: Logger名称
            **kwargs: 覆盖默认配置（level, output_format等）

        Returns:
            StructuredLogger 实例
        """
        with self._factory_lock:
            if name in self._loggers:
                return self._loggers[name]

            level = kwargs.get('level', self._level)
            fmt = kwargs.get('output_format', self._output_format)

            logger = StructuredLogger(
                name=name,
                level=level,
                output_format=fmt,
                sensitive_filter=self._sensitive_filter,
            )
            self._loggers[name] = logger
            return logger

    @property
    def sensitive_filter(self) -> SensitiveDataFilter:
        """获取敏感数据过滤器。"""
        return self._sensitive_filter

    def get_all_loggers(self) -> Dict[str, StructuredLogger]:
        """获取所有已创建的日志器。

        Returns:
            名称到日志器的映射
        """
        with self._factory_lock:
            return dict(self._loggers)

    def set_level(self, level: Union[int, str]) -> None:
        """设置所有日志器的级别。

        Args:
            level: 日志级别
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        self._level = level
        with self._factory_lock:
            for logger in self._loggers.values():
                logger.set_level(level)


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def get_logger(name: str, **kwargs: Any) -> StructuredLogger:
    """获取结构化日志器的便捷函数。

    Args:
        name: Logger名称
        **kwargs: 覆盖默认配置

    Returns:
        StructuredLogger 实例
    """
    return LoggerFactory.get_instance().get_logger(name, **kwargs)
