"""
日志处理器模块

提供日志记录的数据类、处理器链和格式化器。
支持JSON和纯文本两种输出格式。
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable


# ---------------------------------------------------------------------------
# 日志记录数据类
# ---------------------------------------------------------------------------

@dataclass
class LogRecord:
    """结构化日志记录数据类。

    Attributes:
        timestamp: ISO格式时间戳
        level: 日志级别（DEBUG/INFO/WARNING/ERROR/CRITICAL）
        message: 日志消息
        logger_name: Logger名称
        module: 模块名
        function: 函数名
        line_no: 行号
        process_id: 进程ID
        thread_id: 线程ID
        thread_name: 线程名称
        context: 上下文字典（request_id, agent_id, session_id等）
        exception: 异常信息
        stack_trace: 异常堆栈
        extra: 额外字段
    """
    timestamp: str = ""
    level: str = "INFO"
    message: str = ""
    logger_name: str = ""
    module: str = ""
    function: str = ""
    line_no: int = 0
    process_id: int = 0
    thread_id: int = 0
    thread_name: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None
    stack_trace: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，过滤掉空值。"""
        result = {}
        for k, v in asdict(self).items():
            if v is None:
                continue
            if isinstance(v, dict) and not v:
                continue
            result[k] = v
        return result


# ---------------------------------------------------------------------------
# 处理器基类
# ---------------------------------------------------------------------------

class LogProcessor:
    """日志处理器基类。

    处理器按链式顺序对日志记录进行加工。
    子类需要实现 process() 方法。
    """

    def process(self, record: LogRecord) -> LogRecord:
        """处理日志记录，返回处理后的记录。

        Args:
            record: 待处理的日志记录

        Returns:
            处理后的日志记录
        """
        return record


# ---------------------------------------------------------------------------
# 时间戳处理器
# ---------------------------------------------------------------------------

class TimestampProcessor(LogProcessor):
    """添加ISO格式时间戳的处理器。

    如果记录中已有时间戳且非空，则保留原值；
    否则使用当前UTC时间生成ISO格式时间戳。
    """

    def __init__(self, tz: Optional[str] = None):
        """初始化时间戳处理器。

        Args:
            tz: 时区字符串，如 'Asia/Shanghai'。为None时使用UTC。
        """
        self._tz_name = tz
        self._tz = None
        if tz:
            try:
                import zoneinfo
                self._tz = zoneinfo.ZoneInfo(tz)
            except (ImportError, AttributeError):
                # zoneinfo 不可用时回退到 UTC
                self._tz = None

    def process(self, record: LogRecord) -> LogRecord:
        if not record.timestamp:
            now = datetime.now(self._tz) if self._tz else datetime.now(timezone.utc)
            record.timestamp = now.isoformat()
        return record


# ---------------------------------------------------------------------------
# 进程/线程ID处理器
# ---------------------------------------------------------------------------

class ProcessIdProcessor(LogProcessor):
    """添加进程ID和线程ID的处理器。"""

    def process(self, record: LogRecord) -> LogRecord:
        record.process_id = os.getpid()
        record.thread_id = threading.get_ident()
        record.thread_name = threading.current_thread().name
        return record


# ---------------------------------------------------------------------------
# 上下文处理器
# ---------------------------------------------------------------------------

class ContextProcessor(LogProcessor):
    """注入上下文字段的处理器。

    从上下文管理器中获取当前上下文，合并到日志记录中。
    支持多层上下文合并（全局上下文 + 请求级上下文）。
    """

    def __init__(self, context_manager: Optional[Any] = None):
        """初始化上下文处理器。

        Args:
            context_manager: 上下文管理器对象，需提供 get_context() 方法。
                             如果为None，则使用线程本地存储中的上下文。
        """
        self._context_manager = context_manager
        self._local = threading.local()

    def set_context(self, **kwargs: Any) -> None:
        """设置当前线程的上下文字段。

        Args:
            **kwargs: 上下文键值对
        """
        if not hasattr(self._local, 'context'):
            self._local.context = {}
        self._local.context.update(kwargs)

    def clear_context(self) -> None:
        """清除当前线程的上下文。"""
        self._local.context = {}

    def get_context(self) -> Dict[str, Any]:
        """获取当前上下文。

        Returns:
            合并后的上下文字典
        """
        ctx = {}
        # 从上下文管理器获取全局上下文
        if self._context_manager and hasattr(self._context_manager, 'get_context'):
            ctx.update(self._context_manager.get_context())
        # 合并线程本地上下文
        if hasattr(self._local, 'context'):
            ctx.update(self._local.context)
        return ctx

    def process(self, record: LogRecord) -> LogRecord:
        ctx = self.get_context()
        if ctx:
            # 深拷贝避免引用污染
            merged = dict(record.context)
            merged.update(ctx)
            record.context = merged
        return record


# ---------------------------------------------------------------------------
# 异常堆栈处理器
# ---------------------------------------------------------------------------

class StackTraceProcessor(LogProcessor):
    """异常堆栈格式化处理器。

    当日志记录包含异常信息时，自动格式化堆栈跟踪。
    """

    def process(self, record: LogRecord) -> LogRecord:
        if record.exception and not record.stack_trace:
            record.stack_trace = self._format_exception(record.exception)
        return record

    @staticmethod
    def _format_exception(exc_info: str) -> str:
        """格式化异常信息为可读字符串。

        Args:
            exc_info: 异常信息字符串

        Returns:
            格式化后的堆栈字符串
        """
        if isinstance(exc_info, str):
            return exc_info
        try:
            import traceback
            return ''.join(traceback.format_exception(type(exc_info), exc_info, exc_info.__traceback__))
        except Exception:
            return str(exc_info)


# ---------------------------------------------------------------------------
# 格式化器
# ---------------------------------------------------------------------------

class JsonFormatter:
    """JSON格式化器。

    将日志记录序列化为JSON字符串，支持自定义字段排序和缩进。
    """

    def __init__(
        self,
        indent: Optional[int] = None,
        sort_keys: bool = False,
        ensure_ascii: bool = False,
        exclude_fields: Optional[List[str]] = None,
    ):
        """初始化JSON格式化器。

        Args:
            indent: JSON缩进空格数，None表示压缩输出
            sort_keys: 是否按键名排序
            ensure_ascii: 是否转义非ASCII字符
            exclude_fields: 需要排除的字段列表
        """
        self._indent = indent
        self._sort_keys = sort_keys
        self._ensure_ascii = ensure_ascii
        self._exclude_fields = set(exclude_fields or [])

    def format(self, record: LogRecord) -> str:
        """格式化日志记录为JSON字符串。

        Args:
            record: 日志记录

        Returns:
            JSON格式字符串
        """
        data = record.to_dict()
        # 排除指定字段
        for field_name in self._exclude_fields:
            data.pop(field_name, None)
        try:
            return json.dumps(
                data,
                indent=self._indent,
                sort_keys=self._sort_keys,
                ensure_ascii=self._ensure_ascii,
                default=str,
            )
        except (TypeError, ValueError):
            # 序列化失败时回退到简单格式
            fallback = {
                "timestamp": record.timestamp,
                "level": record.level,
                "message": record.message,
            }
            return json.dumps(fallback, ensure_ascii=self._ensure_ascii, default=str)


class TextFormatter:
    """纯文本格式化器。

    将日志记录格式化为人类可读的纯文本格式。
    格式: [TIMESTAMP] [LEVEL] [PID:TID] [MODULE] MESSAGE {context}
    """

    # 默认格式模板
    DEFAULT_FORMAT = (
        "[{timestamp}] [{level:>8s}] [pid={process_id} tid={thread_id}] "
        "[{module}:{function}:{line_no}] {message}"
    )

    def __init__(self, fmt: Optional[str] = None, include_context: bool = True):
        """初始化文本格式化器。

        Args:
            fmt: 自定义格式模板，支持 {field_name} 占位符
            include_context: 是否在末尾追加上下文信息
        """
        self._fmt = fmt or self.DEFAULT_FORMAT
        self._include_context = include_context

    def format(self, record: LogRecord) -> str:
        """格式化日志记录为纯文本字符串。

        Args:
            record: 日志记录

        Returns:
            格式化后的文本字符串
        """
        data = record.to_dict()
        try:
            text = self._fmt.format(**data)
        except KeyError:
            # 字段缺失时使用安全格式化
            text = self._safe_format(self._fmt, data)

        # 追加上下文信息
        if self._include_context and record.context:
            ctx_str = json.dumps(record.context, ensure_ascii=False, default=str)
            text = f"{text} | context={ctx_str}"

        # 追加异常信息
        if record.stack_trace:
            text = f"{text}\n{record.stack_trace}"

        return text

    @staticmethod
    def _safe_format(fmt: str, data: Dict[str, Any]) -> str:
        """安全格式化，缺失字段用 '-' 代替。

        Args:
            fmt: 格式模板
            data: 数据字典

        Returns:
            格式化后的字符串
        """
        import re
        def replacer(match: Any) -> str:
            key = match.group(1)
            val = data.get(key, '-')
            if isinstance(val, str):
                return val
            return str(val)

        return re.sub(r'\{(\w+)\}', replacer, fmt)


# ---------------------------------------------------------------------------
# 处理器链
# ---------------------------------------------------------------------------

class ProcessorChain:
    """处理器链。

    按顺序执行多个处理器，支持动态添加/移除处理器。
    """

    def __init__(self, processors: Optional[List[LogProcessor]] = None):
        """初始化处理器链。

        Args:
            processors: 初始处理器列表
        """
        self._processors: List[LogProcessor] = list(processors or [])
        self._lock = threading.Lock()

    def add_processor(self, processor: LogProcessor) -> None:
        """添加处理器到链尾。

        Args:
            processor: 日志处理器
        """
        with self._lock:
            self._processors.append(processor)

    def remove_processor(self, processor: LogProcessor) -> bool:
        """移除指定处理器。

        Args:
            processor: 要移除的处理器

        Returns:
            是否成功移除
        """
        with self._lock:
            try:
                self._processors.remove(processor)
                return True
            except ValueError:
                return False

    def insert_processor(self, index: int, processor: LogProcessor) -> None:
        """在指定位置插入处理器。

        Args:
            index: 插入位置
            processor: 日志处理器
        """
        with self._lock:
            self._processors.insert(index, processor)

    def process(self, record: LogRecord) -> LogRecord:
        """按顺序执行所有处理器。

        Args:
            record: 待处理的日志记录

        Returns:
            处理后的日志记录
        """
        with self._lock:
            processors = list(self._processors)
        for proc in processors:
            record = proc.process(record)
        return record

    @property
    def processors(self) -> List[LogProcessor]:
        """获取处理器列表的副本。"""
        with self._lock:
            return list(self._processors)

    def clear(self) -> None:
        """清空所有处理器。"""
        with self._lock:
            self._processors.clear()
