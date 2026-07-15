"""
超时包装器模块

提供灵活的超时控制机制，包括截止时间跟踪、分层超时、
超时传播、优雅取消和超时回调。

Classes:
    TimeoutWrapper: 超时包装器主类
    DeadlineTracker: 截止时间跟踪器
    HierarchicalTimeout: 分层超时管理
    TimeoutPropagator: 超时传播器
    GracefulCanceller: 优雅取消器
    TimeoutCallback: 超时回调
    TimeoutConfig: 超时配置
"""

import threading
import time
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


# ============================================================
# 异常类
# ============================================================

class TimeoutExpired(Exception):
    """
    超时过期异常

    Attributes:
        timeout_seconds: 超时时间
        operation: 操作名称
        elapsed: 已用时间
    """

    def __init__(
        self,
        timeout_seconds: float,
        operation: str = "",
        elapsed: float = 0.0,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.operation = operation
        self.elapsed = elapsed
        message = (
            f"操作 '{operation}' 超时 "
            f"(超时={timeout_seconds}s, 已用={elapsed:.2f}s)"
        )
        super().__init__(message)


class CancellationError(Exception):
    """
    取消异常

    Attributes:
        reason: 取消原因
        cancelled_at: 取消时间
    """

    def __init__(self, reason: str = "", cancelled_at: float = 0.0) -> None:
        self.reason = reason
        self.cancelled_at = cancelled_at
        super().__init__(f"操作被取消: {reason}")


# ============================================================
# 超时配置
# ============================================================

class TimeoutConfig:
    """
    超时配置

    Attributes:
        timeout_seconds: 超时时间（秒），None 表示不超时
        operation_name: 操作名称（用于日志和错误信息）
        grace_period: 超时后的优雅取消宽限期（秒）
        propagate: 是否向子操作传播超时
        auto_start: 是否在创建时自动开始计时
    """

    def __init__(
        self,
        timeout_seconds: Optional[float] = None,
        operation_name: str = "",
        grace_period: float = 0.0,
        propagate: bool = True,
        auto_start: bool = False,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.operation_name = operation_name
        self.grace_period = max(0.0, grace_period)
        self.propagate = propagate
        self.auto_start = auto_start

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "timeout_seconds": self.timeout_seconds,
            "operation_name": self.operation_name,
            "grace_period": self.grace_period,
            "propagate": self.propagate,
            "auto_start": self.auto_start,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimeoutConfig":
        """从字典反序列化"""
        return cls(
            timeout_seconds=data.get("timeout_seconds"),
            operation_name=data.get("operation_name", ""),
            grace_period=data.get("grace_period", 0.0),
            propagate=data.get("propagate", True),
            auto_start=data.get("auto_start", False),
        )


# ============================================================
# 截止时间跟踪器
# ============================================================

@dataclass
class DeadlineInfo:
    """截止时间信息"""
    deadline: float
    created_at: float
    operation: str
    timeout_seconds: Optional[float]
    expired: bool = False
    elapsed_at_expiry: float = 0.0


class DeadlineTracker:
    """
    截止时间跟踪器

    跟踪操作的绝对截止时间，支持检查是否已超时、
    计算剩余时间和已用时间。

    Usage:
        tracker = DeadlineTracker(timeout_seconds=30.0, operation="fetch_data")
        tracker.start()
        # ... 执行操作 ...
        if tracker.is_expired():
            raise TimeoutExpired(...)
        remaining = tracker.remaining_time()
    """

    def __init__(
        self,
        timeout_seconds: Optional[float] = None,
        operation: str = "",
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.operation = operation
        self._start_time: float = 0.0
        self._deadline: float = 0.0
        self._started: bool = False
        self._expired: bool = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """开始计时"""
        with self._lock:
            self._start_time = time.monotonic()
            if self.timeout_seconds is not None:
                self._deadline = self._start_time + self.timeout_seconds
            else:
                self._deadline = float("inf")
            self._started = True
            self._expired = False

    def reset(self) -> None:
        """重置计时器"""
        with self._lock:
            self._start_time = 0.0
            self._deadline = 0.0
            self._started = False
            self._expired = False

    def is_expired(self) -> bool:
        """检查是否已超时"""
        with self._lock:
            if not self._started:
                return False
            if self.timeout_seconds is None:
                return False
            now = time.monotonic()
            if now >= self._deadline:
                self._expired = True
            return self._expired

    def remaining_time(self) -> float:
        """
        获取剩余时间

        Returns:
            剩余秒数，如果已超时返回 0.0，如果无超时限制返回 float('inf')
        """
        with self._lock:
            if not self._started:
                return float("inf") if self.timeout_seconds is None else (
                    self.timeout_seconds
                )
            if self.timeout_seconds is None:
                return float("inf")
            now = time.monotonic()
            remaining = self._deadline - now
            if remaining <= 0:
                self._expired = True
                return 0.0
            return remaining

    def elapsed_time(self) -> float:
        """
        获取已用时间

        Returns:
            已用秒数
        """
        with self._lock:
            if not self._started:
                return 0.0
            return time.monotonic() - self._start_time

    def check(self) -> None:
        """检查超时，如果已超时则抛出异常"""
        if self.is_expired():
            elapsed = self.elapsed_time()
            raise TimeoutExpired(
                timeout_seconds=self.timeout_seconds or 0.0,
                operation=self.operation,
                elapsed=elapsed,
            )

    def get_deadline_info(self) -> DeadlineInfo:
        """获取截止时间详情"""
        with self._lock:
            return DeadlineInfo(
                deadline=self._deadline,
                created_at=self._start_time,
                operation=self.operation,
                timeout_seconds=self.timeout_seconds,
                expired=self._expired,
            )

    @property
    def started(self) -> bool:
        """是否已开始计时"""
        return self._started


# ============================================================
# 超时回调
# ============================================================

@dataclass
class TimeoutCallbackEntry:
    """超时回调条目"""
    callback: Callable[[], None]
    once: bool = True
    called: bool = False
    priority: int = 0


class TimeoutCallback:
    """
    超时回调管理器

    管理超时事件触发时的回调函数。

    Usage:
        callbacks = TimeoutCallback()
        callbacks.on_timeout(lambda: print("超时了!"))
        callbacks.on_timeout(cleanup_func, priority=10)
        callbacks.fire_all()
    """

    def __init__(self) -> None:
        self._callbacks: List[TimeoutCallbackEntry] = []
        self._lock = threading.Lock()

    def on_timeout(
        self,
        callback: Callable[[], None],
        once: bool = True,
        priority: int = 0,
    ) -> None:
        """
        注册超时回调

        Args:
            callback: 回调函数
            once: 是否只执行一次
            priority: 优先级（数值越小越先执行）
        """
        with self._lock:
            self._callbacks.append(TimeoutCallbackEntry(
                callback=callback,
                once=once,
                called=False,
                priority=priority,
            ))
            # 按优先级排序
            self._callbacks.sort(key=lambda c: c.priority)

    def fire_all(self) -> List[Tuple[Callable[[], None], Optional[Exception]]]:
        """
        触发所有回调

        Returns:
            (callback, error) 元组列表，error 为 None 表示成功
        """
        results: List[Tuple[Callable[[], None], Optional[Exception]]] = []
        with self._lock:
            callbacks_to_fire = list(self._callbacks)

        for entry in callbacks_to_fire:
            if entry.once and entry.called:
                continue
            try:
                entry.callback()
                entry.called = True
                results.append((entry.callback, None))
            except Exception as e:
                results.append((entry.callback, e))

        return results

    def clear(self) -> None:
        """清除所有回调"""
        with self._lock:
            self._callbacks.clear()

    def remove(self, callback: Callable[[], None]) -> bool:
        """
        移除指定回调

        Returns:
            是否成功移除
        """
        with self._lock:
            for i, entry in enumerate(self._callbacks):
                if entry.callback == callback:
                    self._callbacks.pop(i)
                    return True
            return False

    @property
    def callback_count(self) -> int:
        """已注册的回调数量"""
        return len(self._callbacks)


# ============================================================
# 优雅取消器
# ============================================================

class GracefulCanceller:
    """
    优雅取消器

    在超时后提供宽限期，允许操作完成清理工作。

    Usage:
        canceller = GracefulCanceller(grace_period=5.0)
        canceller.register_cleanup(lambda: save_state())
        canceller.register_cleanup(lambda: close_connections())
        # 超时后
        canceller.cancel(reason="操作超时")
        canceller.wait_for_cleanup()
    """

    def __init__(self, grace_period: float = 5.0) -> None:
        self.grace_period = max(0.0, grace_period)
        self._cleanup_functions: List[Callable[[], None]] = []
        self._cancelled: bool = False
        self._cancel_reason: str = ""
        self._cancel_time: float = 0.0
        self._cleanup_done = threading.Event()
        self._cleanup_results: List[Tuple[str, Optional[Exception]]] = []
        self._lock = threading.Lock()

    def register_cleanup(self, fn: Callable[[], None], name: str = "") -> None:
        """
        注册清理函数

        Args:
            fn: 清理函数
            name: 清理函数名称（用于日志）
        """
        with self._lock:
            self._cleanup_functions.append((name or fn.__name__, fn))

    def cancel(self, reason: str = "") -> None:
        """
        触发取消

        Args:
            reason: 取消原因
        """
        with self._lock:
            self._cancelled = True
            self._cancel_reason = reason
            self._cancel_time = time.monotonic()
            self._cleanup_done.clear()

    @property
    def is_cancelled(self) -> bool:
        """是否已取消"""
        return self._cancelled

    @property
    def cancel_reason(self) -> str:
        """取消原因"""
        return self._cancel_reason

    def execute_cleanup(self) -> List[Tuple[str, Optional[Exception]]]:
        """
        执行所有清理函数

        Returns:
            (name, error) 元组列表
        """
        results: List[Tuple[str, Optional[Exception]]] = []
        with self._lock:
            functions = list(self._cleanup_functions)

        for name, fn in functions:
            try:
                fn()
                results.append((name, None))
            except Exception as e:
                results.append((name, e))

        self._cleanup_results = results
        self._cleanup_done.set()
        return results

    def wait_for_cleanup(self, timeout: Optional[float] = None) -> bool:
        """
        等待清理完成

        Args:
            timeout: 等待超时时间

        Returns:
            清理是否在超时前完成
        """
        effective_timeout = timeout or self.grace_period
        return self._cleanup_done.wait(timeout=effective_timeout)

    def remaining_grace(self) -> float:
        """
        获取剩余宽限期

        Returns:
            剩余秒数
        """
        if not self._cancelled:
            return self.grace_period
        elapsed = time.monotonic() - self._cancel_time
        return max(0.0, self.grace_period - elapsed)

    def get_status(self) -> Dict[str, Any]:
        """获取取消器状态"""
        return {
            "cancelled": self._cancelled,
            "reason": self._cancel_reason,
            "grace_period": self.grace_period,
            "remaining_grace": round(self.remaining_grace(), 4),
            "cleanup_functions": len(self._cleanup_functions),
            "cleanup_done": self._cleanup_done.is_set(),
        }


# ============================================================
# 分层超时
# ============================================================

class HierarchicalTimeout:
    """
    分层超时管理

    管理父子操作之间的超时关系。子操作的超时不能超过
    父操作的剩余时间。

    Usage:
        root = HierarchicalTimeout(timeout_seconds=60.0, name="root")
        root.start()
        child = root.create_child("child_op", timeout_seconds=30.0)
        child.start()
        # child 的实际超时 = min(30.0, root.remaining_time())
    """

    def __init__(
        self,
        timeout_seconds: Optional[float] = None,
        name: str = "root",
        parent: Optional["HierarchicalTimeout"] = None,
    ) -> None:
        self.name = name
        self._requested_timeout = timeout_seconds
        self._parent = parent
        self._tracker = DeadlineTracker(
            timeout_seconds=self._effective_timeout(),
            operation=name,
        )
        self._children: List["HierarchicalTimeout"] = []
        self._lock = threading.Lock()

    def _effective_timeout(self) -> Optional[float]:
        """计算有效超时时间（考虑父级约束）"""
        if self._requested_timeout is None:
            if self._parent is not None:
                return self._parent.remaining_time()
            return None

        if self._parent is not None:
            parent_remaining = self._parent.remaining_time()
            if parent_remaining == float("inf"):
                return self._requested_timeout
            return min(self._requested_timeout, parent_remaining)

        return self._requested_timeout

    def start(self) -> None:
        """开始计时"""
        effective = self._effective_timeout()
        self._tracker = DeadlineTracker(
            timeout_seconds=effective,
            operation=self.name,
        )
        self._tracker.start()

    def remaining_time(self) -> float:
        """获取剩余时间"""
        return self._tracker.remaining_time()

    def elapsed_time(self) -> float:
        """获取已用时间"""
        return self._tracker.elapsed_time()

    def is_expired(self) -> bool:
        """检查是否已超时"""
        return self._tracker.is_expired()

    def check(self) -> None:
        """检查超时，超时则抛出异常"""
        self._tracker.check()

    def create_child(
        self,
        name: str,
        timeout_seconds: Optional[float] = None,
    ) -> "HierarchicalTimeout":
        """
        创建子超时

        Args:
            name: 子操作名称
            timeout_seconds: 子操作请求的超时时间

        Returns:
            HierarchicalTimeout 子实例
        """
        child = HierarchicalTimeout(
            timeout_seconds=timeout_seconds,
            name=name,
            parent=self,
        )
        with self._lock:
            self._children.append(child)
        return child

    def get_tree(self, indent: int = 0) -> str:
        """获取超时树的可视化表示"""
        prefix = "  " * indent
        remaining = self.remaining_time()
        elapsed = self.elapsed_time()
        status = "EXPIRED" if self.is_expired() else "ACTIVE"
        lines = [
            f"{prefix}{self.name}: {status} "
            f"(remaining={remaining:.2f}s, elapsed={elapsed:.2f}s)"
        ]
        for child in self._children:
            lines.append(child.get_tree(indent + 1))
        return "\n".join(lines)

    def get_summary(self) -> Dict[str, Any]:
        """获取分层超时摘要"""
        return {
            "name": self.name,
            "requested_timeout": self._requested_timeout,
            "remaining": round(self.remaining_time(), 4),
            "elapsed": round(self.elapsed_time(), 4),
            "expired": self.is_expired(),
            "children_count": len(self._children),
        }


# ============================================================
# 超时传播器
# ============================================================

class TimeoutPropagator:
    """
    超时传播器

    将超时信息传播到上下文中，使下游操作能够感知上游的超时约束。

    Usage:
        propagator = TimeoutPropagator()
        ctx = propagator.inject_timeout(context, timeout_seconds=30.0)
        # 下游操作可以从 context 中读取超时信息
        remaining = propagator.get_remaining(context)
    """

    CONTEXT_TIMEOUT_KEY = "_timeout_deadline"
    CONTEXT_TIMEOUT_NAME_KEY = "_timeout_operation"
    CONTEXT_TIMEOUT_GRACE_KEY = "_timeout_grace_period"

    def __init__(self) -> None:
        self._active_contexts: Dict[int, float] = {}
        self._lock = threading.Lock()

    def inject_timeout(
        self,
        context: Dict[str, Any],
        timeout_seconds: Optional[float] = None,
        operation_name: str = "",
        grace_period: float = 0.0,
    ) -> Dict[str, Any]:
        """
        将超时信息注入到上下文中

        Args:
            context: 执行上下文
            timeout_seconds: 超时时间
            operation_name: 操作名称
            grace_period: 宽限期

        Returns:
            更新后的上下文
        """
        if timeout_seconds is not None:
            deadline = time.monotonic() + timeout_seconds
            context[self.CONTEXT_TIMEOUT_KEY] = deadline
            context[self.CONTEXT_TIMEOUT_NAME_KEY] = operation_name
            context[self.CONTEXT_TIMEOUT_GRACE_KEY] = grace_period

            ctx_id = id(context)
            with self._lock:
                self._active_contexts[ctx_id] = deadline

        return context

    def get_remaining(self, context: Dict[str, Any]) -> float:
        """
        从上下文中获取剩余时间

        Args:
            context: 执行上下文

        Returns:
            剩余秒数，无超时限制返回 float('inf')
        """
        deadline = context.get(self.CONTEXT_TIMEOUT_KEY)
        if deadline is None:
            return float("inf")
        remaining = deadline - time.monotonic()
        return max(0.0, remaining)

    def is_expired(self, context: Dict[str, Any]) -> bool:
        """
        检查上下文中的超时是否已过期

        Args:
            context: 执行上下文

        Returns:
            是否已超时
        """
        return self.get_remaining(context) <= 0

    def check_context(self, context: Dict[str, Any]) -> None:
        """检查上下文超时，超时则抛出异常"""
        if self.is_expired(context):
            operation = context.get(
                self.CONTEXT_TIMEOUT_NAME_KEY, "unknown"
            )
            raise TimeoutExpired(
                timeout_seconds=0.0,
                operation=operation,
                elapsed=0.0,
            )

    def compute_child_timeout(
        self,
        context: Dict[str, Any],
        requested_timeout: float,
    ) -> float:
        """
        计算子操作的有效超时

        Args:
            context: 父上下文
            requested_timeout: 子操作请求的超时时间

        Returns:
            有效超时时间
        """
        parent_remaining = self.get_remaining(context)
        if parent_remaining == float("inf"):
            return requested_timeout
        return min(requested_timeout, parent_remaining)

    def cleanup(self, context: Dict[str, Any]) -> None:
        """清理上下文中的超时信息"""
        ctx_id = id(context)
        with self._lock:
            self._active_contexts.pop(ctx_id, None)
        context.pop(self.CONTEXT_TIMEOUT_KEY, None)
        context.pop(self.CONTEXT_TIMEOUT_NAME_KEY, None)
        context.pop(self.CONTEXT_TIMEOUT_GRACE_KEY, None)

    def get_active_count(self) -> int:
        """获取活跃的超时上下文数量"""
        with self._lock:
            return len(self._active_contexts)


# ============================================================
# 超时包装器
# ============================================================

class TimeoutWrapper:
    """
    超时包装器

    为任意函数提供超时控制，支持截止时间跟踪、
    分层超时、优雅取消和超时回调。

    Usage:
        wrapper = TimeoutWrapper(
            config=TimeoutConfig(timeout_seconds=10.0, operation_name="api_call"),
        )
        result = wrapper.execute(some_function, arg1, arg2)

        # 使用上下文管理器
        with TimeoutWrapper(config=TimeoutConfig(timeout_seconds=5.0)) as tw:
            tw.execute(long_running_task)
    """

    def __init__(
        self,
        config: Optional[TimeoutConfig] = None,
        timeout_seconds: Optional[float] = None,
        operation_name: str = "",
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = TimeoutConfig(
                timeout_seconds=timeout_seconds,
                operation_name=operation_name,
            )

        self._tracker = DeadlineTracker(
            timeout_seconds=self.config.timeout_seconds,
            operation=self.config.operation_name,
        )
        self._callbacks = TimeoutCallback()
        self._canceller = GracefulCanceller(
            grace_period=self.config.grace_period,
        )
        self._propagator = TimeoutPropagator()
        self._result: Any = None
        self._error: Optional[Exception] = None

    def __enter__(self) -> "TimeoutWrapper":
        self._tracker.start()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> bool:
        if exc_type is not None and self._tracker.is_expired():
            self._callbacks.fire_all()
            self._canceller.cancel(reason="上下文管理器退出时超时")
        return False

    def execute(
        self,
        fn: Callable[..., Any],
        *args: Any,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        执行带超时控制的函数

        Args:
            fn: 要执行的函数
            *args: 函数位置参数
            context: 执行上下文（用于超时传播）
            **kwargs: 函数关键字参数

        Returns:
            函数返回值

        Raises:
            TimeoutExpired: 超时时抛出
            CancellationError: 被取消时抛出
        """
        self._tracker.start()
        ctx = context or {}

        # 注入超时到上下文
        if self.config.propagate:
            self._propagator.inject_timeout(
                ctx,
                timeout_seconds=self.config.timeout_seconds,
                operation_name=self.config.operation_name,
                grace_period=self.config.grace_period,
            )

        # 注册取消回调
        self._callbacks.on_timeout(
            lambda: self._canceller.cancel(reason="超时触发取消"),
            priority=0,
        )

        try:
            # 在线程中执行函数以支持超时中断
            result_holder: List[Any] = [None]
            error_holder: List[Optional[Exception]] = [None]
            done_event = threading.Event()

            def target() -> None:
                try:
                    result_holder[0] = fn(*args, **kwargs)
                except Exception as e:
                    error_holder[0] = e
                finally:
                    done_event.set()

            thread = threading.Thread(target=target, daemon=True)
            thread.start()

            # 等待完成或超时
            timeout = self._tracker.remaining_time()
            if timeout != float("inf"):
                completed = done_event.wait(timeout=timeout)
            else:
                completed = True
                thread.join()

            if not completed:
                # 超时
                self._callbacks.fire_all()
                elapsed = self._tracker.elapsed_time()
                raise TimeoutExpired(
                    timeout_seconds=self.config.timeout_seconds or 0.0,
                    operation=self.config.operation_name,
                    elapsed=elapsed,
                )

            if error_holder[0] is not None:
                raise error_holder[0]

            self._result = result_holder[0]
            return self._result

        except TimeoutExpired:
            raise

        except CancellationError:
            raise

        finally:
            if self.config.propagate:
                self._propagator.cleanup(ctx)

    def on_timeout(
        self,
        callback: Callable[[], None],
        priority: int = 0,
    ) -> None:
        """注册超时回调"""
        self._callbacks.on_timeout(callback, priority=priority)

    def register_cleanup(self, fn: Callable[[], None]) -> None:
        """注册清理函数"""
        self._canceller.register_cleanup(fn)

    def cancel(self, reason: str = "") -> None:
        """手动取消"""
        self._canceller.cancel(reason=reason)

    @property
    def remaining_time(self) -> float:
        """剩余时间"""
        return self._tracker.remaining_time()

    @property
    def elapsed_time(self) -> float:
        """已用时间"""
        return self._tracker.elapsed_time()

    @property
    def is_expired(self) -> bool:
        """是否已超时"""
        return self._tracker.is_expired()

    def create_child_timeout(
        self,
        name: str,
        timeout_seconds: Optional[float] = None,
    ) -> HierarchicalTimeout:
        """创建分层子超时"""
        parent = HierarchicalTimeout(
            timeout_seconds=self.config.timeout_seconds,
            name=self.config.operation_name or "parent",
        )
        parent.start()
        return parent.create_child(name, timeout_seconds)

    def get_status(self) -> Dict[str, Any]:
        """获取超时包装器状态"""
        return {
            "operation": self.config.operation_name,
            "timeout": self.config.timeout_seconds,
            "remaining": round(self.remaining_time, 4),
            "elapsed": round(self.elapsed_time, 4),
            "expired": self.is_expired,
            "callbacks": self._callbacks.callback_count,
            "canceller": self._canceller.get_status(),
        }
