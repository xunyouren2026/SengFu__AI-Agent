"""
分布式追踪模块

提供纯Python实现的分布式追踪功能，模拟OpenTelemetry的核心概念。
支持Span嵌套、上下文传播、追踪数据收集和摘要统计。
"""

import random
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 常量与工具函数
# ---------------------------------------------------------------------------

def _generate_trace_id() -> str:
    """生成32字符的十六进制trace ID。"""
    return format(random.getrandbits(128), '032x')


def _generate_span_id() -> str:
    """生成16字符的十六进制span ID。"""
    return format(random.getrandbits(64), '016x')


def _now() -> float:
    """获取当前时间戳（秒）。"""
    return time.time()


def _now_iso() -> str:
    """获取当前ISO格式时间字符串。"""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Span 状态枚举
# ---------------------------------------------------------------------------

class SpanStatus(str, Enum):
    """Span 状态。"""
    UNSET = "UNSET"
    OK = "OK"
    ERROR = "ERROR"


# ---------------------------------------------------------------------------
# SpanContext: 跨度上下文
# ---------------------------------------------------------------------------

@dataclass
class SpanContext:
    """跨度上下文，用于在服务间传播追踪信息。

    Attributes:
        trace_id: 追踪ID
        span_id: 当前跨度ID
        trace_flags: 追踪标志（如采样位）
        trace_state: 追踪状态键值对
    """
    trace_id: str
    span_id: str
    trace_flags: int = 1  # 默认采样
    trace_state: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, str]:
        """转换为字典，用于上下文注入。"""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "trace_flags": str(self.trace_flags),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> Optional["SpanContext"]:
        """从字典创建SpanContext。

        Args:
            data: 包含 trace_id 和 span_id 的字典

        Returns:
            SpanContext 实例，如果缺少必要字段则返回 None
        """
        trace_id = data.get("trace_id", "")
        span_id = data.get("span_id", "")
        if not trace_id or not span_id:
            return None
        flags = int(data.get("trace_flags", "1"))
        return cls(trace_id=trace_id, span_id=span_id, trace_flags=flags)


# ---------------------------------------------------------------------------
# Span: 追踪跨度
# ---------------------------------------------------------------------------

class Span:
    """追踪跨度，表示一个操作的执行过程。

    每个Span包含：
    - 关联的trace_id和span_id
    - 父span引用（用于构建调用树）
    - 操作名称
    - 起止时间
    - 标签和日志事件

    使用示例::

        with tracer.start_span("process_request") as span:
            span.set_tag("http.method", "GET")
            span.log("received request")
            # ... 执行操作
    """

    def __init__(
        self,
        trace_id: str,
        span_id: str,
        operation_name: str,
        parent_id: Optional[str] = None,
        context: Optional[SpanContext] = None,
        start_time: Optional[float] = None,
        on_finish: Optional[Callable[["Span"], None]] = None,
    ):
        """初始化Span。

        Args:
            trace_id: 追踪ID
            span_id: 当前跨度ID
            operation_name: 操作名称
            parent_id: 父跨度ID
            context: 跨度上下文
            start_time: 开始时间（秒），默认为当前时间
            on_finish: Span结束时的回调函数
        """
        self.trace_id = trace_id
        self.span_id = span_id
        self.operation_name = operation_name
        self.parent_id = parent_id
        self.context = context or SpanContext(trace_id=trace_id, span_id=span_id)
        self.start_time = start_time or _now()
        self.end_time: Optional[float] = None
        self.status: SpanStatus = SpanStatus.UNSET
        self.tags: Dict[str, Any] = {}
        self.logs: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._finished = False
        self._on_finish = on_finish

    def set_tag(self, key: str, value: Any) -> "Span":
        """设置标签。

        Args:
            key: 标签键
            value: 标签值

        Returns:
            self（支持链式调用）
        """
        with self._lock:
            self.tags[key] = value
        return self

    def log(self, message: str, **kwargs: Any) -> None:
        """记录日志事件。

        Args:
            message: 日志消息
            **kwargs: 附加字段
        """
        entry = {
            "timestamp": _now_iso(),
            "message": message,
        }
        entry.update(kwargs)
        with self._lock:
            self.logs.append(entry)

    def set_status(self, status: SpanStatus, description: str = "") -> None:
        """设置Span状态。

        Args:
            status: 状态枚举
            description: 状态描述
        """
        with self._lock:
            self.status = status
            if description:
                self.tags["status.description"] = description

    def finish(self, end_time: Optional[float] = None) -> None:
        """结束Span。

        Args:
            end_time: 结束时间（秒），默认为当前时间
        """
        with self._lock:
            if self._finished:
                return
            self.end_time = end_time or _now()
            self._finished = True
        # 回调在锁外执行，避免死锁
        if self._on_finish:
            try:
                self._on_finish(self)
            except Exception:
                pass  # 回调异常不应影响Span结束

    def is_finished(self) -> bool:
        """Span是否已结束。"""
        return self._finished

    @property
    def duration(self) -> Optional[float]:
        """Span持续时间（秒）。如果Span未结束则返回None。"""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        with self._lock:
            return {
                "trace_id": self.trace_id,
                "span_id": self.span_id,
                "parent_id": self.parent_id,
                "operation_name": self.operation_name,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "duration": self.duration,
                "status": self.status.value,
                "tags": dict(self.tags),
                "logs": list(self.logs),
            }

    def __enter__(self) -> "Span":
        """支持上下文管理器。"""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """退出上下文时自动结束Span。"""
        if exc_type is not None:
            self.set_status(SpanStatus.ERROR, str(exc_val))
        elif self.status == SpanStatus.UNSET:
            self.set_status(SpanStatus.OK)
        self.finish()


# ---------------------------------------------------------------------------
# TraceCollector: 追踪数据收集器
# ---------------------------------------------------------------------------

class TraceCollector:
    """追踪数据收集器。

    收集和管理所有已完成的Span，支持按trace_id查询和摘要统计。
    """

    def __init__(self, max_traces: int = 10000):
        """初始化收集器。

        Args:
            max_traces: 最大保存的trace数量，超过后淘汰最旧的
        """
        self._traces: Dict[str, List[Span]] = defaultdict(list)
        self._all_spans: List[Span] = []
        self._lock = threading.Lock()
        self._max_traces = max_traces

    def add_span(self, span: Span) -> None:
        """添加已完成的Span。

        Args:
            span: 已结束的Span
        """
        if not span.is_finished():
            return
        with self._lock:
            self._traces[span.trace_id].append(span)
            self._all_spans.append(span)
            # 淘汰策略：超过上限时移除最旧的trace
            if len(self._traces) > self._max_traces:
                oldest_trace_id = next(iter(self._traces))
                del self._traces[oldest_trace_id]

    def get_trace(self, trace_id: str) -> List[Span]:
        """获取指定trace_id的所有Span。

        Args:
            trace_id: 追踪ID

        Returns:
            Span列表
        """
        with self._lock:
            return list(self._traces.get(trace_id, []))

    def get_all_spans(self) -> List[Span]:
        """获取所有已收集的Span。

        Returns:
            Span列表
        """
        with self._lock:
            return list(self._all_spans)

    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """获取指定trace的摘要统计。

        Args:
            trace_id: 追踪ID

        Returns:
            摘要字典，包含:
            - trace_id: 追踪ID
            - total_duration: 总耗时（秒）
            - span_count: Span数量
            - spans: 各Span详情（名称、耗时、状态）
            - error_count: 错误Span数量
            - error_rate: 错误率
            - root_span: 根Span信息
        """
        spans = self.get_trace(trace_id)
        if not spans:
            return {"trace_id": trace_id, "error": "trace not found"}

        # 找到根Span（没有parent_id的）
        root_spans = [s for s in spans if s.parent_id is None]
        root = root_spans[0] if root_spans else spans[0]

        # 计算总耗时
        start_times = [s.start_time for s in spans]
        end_times = [s.end_time for s in spans if s.end_time is not None]
        total_duration = max(end_times) - min(start_times) if end_times and start_times else 0

        # 统计错误
        error_count = sum(1 for s in spans if s.status == SpanStatus.ERROR)
        error_rate = error_count / len(spans) if spans else 0

        # 各Span详情
        span_details = []
        for s in spans:
            span_details.append({
                "span_id": s.span_id,
                "operation_name": s.operation_name,
                "parent_id": s.parent_id,
                "duration": s.duration,
                "status": s.status.value,
            })

        # 按耗时排序
        span_details.sort(key=lambda x: x.get("duration") or 0, reverse=True)

        return {
            "trace_id": trace_id,
            "total_duration": total_duration,
            "span_count": len(spans),
            "error_count": error_count,
            "error_rate": round(error_rate, 4),
            "root_span": {
                "operation_name": root.operation_name,
                "span_id": root.span_id,
                "duration": root.duration,
            },
            "spans": span_details,
        }

    def clear(self) -> None:
        """清空所有收集的追踪数据。"""
        with self._lock:
            self._traces.clear()
            self._all_spans.clear()

    @property
    def trace_count(self) -> int:
        """当前收集的trace数量。"""
        with self._lock:
            return len(self._traces)

    @property
    def span_count(self) -> int:
        """当前收集的Span总数。"""
        with self._lock:
            return len(self._all_spans)


# ---------------------------------------------------------------------------
# Tracer: 分布式追踪器
# ---------------------------------------------------------------------------

class Tracer:
    """分布式追踪器。

    模拟OpenTelemetry的核心功能，提供：
    - 创建和管理Span
    - Span嵌套（父子关系）
    - 上下文注入和提取（跨服务传播）
    - 追踪数据收集

    使用示例::

        tracer = Tracer(service_name="my_service")

        with tracer.start_span("handle_request") as parent:
            parent.set_tag("http.method", "POST")
            with tracer.start_span("query_db", parent=parent) as child:
                child.set_tag("db.system", "postgresql")
                # ... 数据库操作

        summary = tracer.get_trace_summary(parent.trace_id)
    """

    def __init__(
        self,
        service_name: str = "unknown",
        collector: Optional[TraceCollector] = None,
        enabled: bool = True,
    ):
        """初始化追踪器。

        Args:
            service_name: 服务名称
            collector: 追踪数据收集器，为None时创建默认收集器
            enabled: 是否启用追踪
        """
        self.service_name = service_name
        self._collector = collector or TraceCollector()
        self._enabled = enabled
        self._lock = threading.Lock()
        # 线程本地存储，保存当前活跃的Span
        self._local = threading.local()

    @property
    def collector(self) -> TraceCollector:
        """获取追踪数据收集器。"""
        return self._collector

    @property
    def enabled(self) -> bool:
        """是否启用追踪。"""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """设置是否启用追踪。"""
        self._enabled = value

    def start_span(
        self,
        operation_name: str,
        parent: Optional[Span] = None,
        context: Optional[SpanContext] = None,
        tags: Optional[Dict[str, Any]] = None,
        start_time: Optional[float] = None,
    ) -> Span:
        """开始一个新的追踪跨度。

        Args:
            operation_name: 操作名称
            parent: 父Span，如果为None则自动使用当前活跃Span
            context: 跨度上下文（用于从外部提取上下文）
            tags: 初始标签
            start_time: 自定义开始时间

        Returns:
            新创建的Span
        """
        if not self._enabled:
            # 返回一个空操作的Span
            return Span(
                trace_id="disabled",
                span_id="disabled",
                operation_name=operation_name,
            )

        # 确定trace_id和parent_id
        if context:
            trace_id = context.trace_id
            parent_id = context.span_id
        elif parent:
            trace_id = parent.trace_id
            parent_id = parent.span_id
        else:
            # 尝试从线程本地存储获取当前Span
            current = self._get_active_span()
            if current:
                trace_id = current.trace_id
                parent_id = current.span_id
            else:
                trace_id = _generate_trace_id()
                parent_id = None

        span_id = _generate_span_id()
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            operation_name=operation_name,
            parent_id=parent_id,
            start_time=start_time,
            on_finish=self._on_span_finished,
        )

        # 设置初始标签
        if tags:
            for k, v in tags.items():
                span.set_tag(k, v)
        span.set_tag("service.name", self.service_name)

        # 设置为当前活跃Span
        self._set_active_span(span)

        return span

    def end_span(self, span: Span, end_time: Optional[float] = None) -> None:
        """结束指定的Span。

        Args:
            span: 要结束的Span
            end_time: 自定义结束时间
        """
        span.finish(end_time)
        # 如果当前活跃Span是此Span，则恢复到父Span
        self._restore_parent_span(span)

    def _on_span_finished(self, span: Span) -> None:
        """Span结束时的回调，自动收集到TraceCollector。

        Args:
            span: 已结束的Span
        """
        self._collector.add_span(span)
        self._restore_parent_span(span)

    def inject_context(self, span: Span, carrier: Dict[str, str]) -> Dict[str, str]:
        """注入追踪上下文到字典中。

        用于将追踪信息传播到下游服务。

        Args:
            span: 当前Span
            carrier: 目标字典

        Returns:
            注入了追踪信息的字典
        """
        ctx = span.context.to_dict()
        carrier.update(ctx)
        return carrier

    def extract_context(self, carrier: Dict[str, str]) -> Optional[SpanContext]:
        """从字典中提取追踪上下文。

        用于从上游服务的请求中恢复追踪信息。

        Args:
            carrier: 包含追踪信息的字典

        Returns:
            SpanContext 实例，如果缺少必要字段则返回 None
        """
        return SpanContext.from_dict(carrier)

    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """获取指定trace的摘要统计。

        Args:
            trace_id: 追踪ID

        Returns:
            摘要字典
        """
        return self._collector.get_trace_summary(trace_id)

    def get_active_span(self) -> Optional[Span]:
        """获取当前线程的活跃Span。

        Returns:
            当前活跃的Span，如果没有则返回None
        """
        return self._get_active_span()

    def _get_active_span(self) -> Optional[Span]:
        """获取当前线程的活跃Span（内部方法）。"""
        stack = getattr(self._local, 'span_stack', [])
        return stack[-1] if stack else None

    def _set_active_span(self, span: Span) -> None:
        """设置当前线程的活跃Span。"""
        if not hasattr(self._local, 'span_stack'):
            self._local.span_stack = []
        self._local.span_stack.append(span)

    def _restore_parent_span(self, span: Span) -> None:
        """结束Span后恢复父Span为活跃Span。"""
        stack = getattr(self._local, 'span_stack', [])
        # 从栈中移除该Span
        for i in range(len(stack) - 1, -1, -1):
            if stack[i].span_id == span.span_id:
                stack.pop(i)
                break

    def clear_active_span(self) -> None:
        """清除当前线程的活跃Span。"""
        if hasattr(self._local, 'span_stack'):
            self._local.span_stack.clear()

    def get_all_traces(self) -> List[str]:
        """获取所有已收集的trace ID列表。

        Returns:
            trace ID列表
        """
        with self._lock:
            return list(self._collector._traces.keys())
