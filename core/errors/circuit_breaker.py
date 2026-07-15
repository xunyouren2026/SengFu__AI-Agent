"""
熔断器模块

实现熔断器模式，防止级联故障。当下游服务出现故障时，
熔断器会快速失败，避免不必要的资源消耗。

状态转换:
    CLOSED -> OPEN: 失败率超过阈值
    OPEN -> HALF_OPEN: 超过恢复超时时间
    HALF_OPEN -> CLOSED: 探测请求成功
    HALF_OPEN -> OPEN: 探测请求仍然失败
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """熔断器状态枚举"""

    CLOSED = "CLOSED"          # 正常状态，允许请求通过
    OPEN = "OPEN"              # 熔断状态，快速失败
    HALF_OPEN = "HALF_OPEN"    # 半开状态，允许探测请求


@dataclass
class _WindowRecord:
    """滑动窗口中的单条记录"""
    timestamp: float
    success: bool


class CircuitBreaker:
    """
    熔断器实现

    使用滑动窗口统计失败率，根据阈值自动切换状态。

    Attributes:
        name: 熔断器名称
        failure_threshold: 触发熔断的失败率阈值 (0.0 ~ 1.0)
        recovery_timeout: 熔断后的恢复等待时间（秒）
        half_open_max_calls: 半开状态下允许的最大探测请求数
        window_size: 滑动窗口大小（记录条数）
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: float = 0.5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
        window_size: int = 100,
    ):
        if not (0 < failure_threshold <= 1.0):
            raise ValueError("failure_threshold 必须在 (0, 1.0] 范围内")
        if recovery_timeout <= 0:
            raise ValueError("recovery_timeout 必须大于0")
        if half_open_max_calls <= 0:
            raise ValueError("half_open_max_calls 必须大于0")
        if window_size <= 0:
            raise ValueError("window_size 必须大于0")

        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.window_size = window_size

        self._state = CircuitState.CLOSED
        self._window: Deque[_WindowRecord] = deque(maxlen=window_size)
        self._last_failure_time: float = 0.0
        self._half_open_calls: int = 0
        self._half_open_successes: int = 0
        self._lock = threading.RLock()
        self._total_calls: int = 0
        self._total_successes: int = 0
        self._total_failures: int = 0
        self._total_circuit_opens: int = 0

    def get_state(self) -> str:
        """
        获取当前熔断器状态

        如果当前是OPEN状态且已超过恢复超时时间，
        自动转换为HALF_OPEN状态。

        Returns:
            当前状态字符串
        """
        with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    self._transition_to(CircuitState.HALF_OPEN)
                    logger.info(
                        "熔断器 [%s] 从 OPEN 转换到 HALF_OPEN "
                        "(已等待 %.1fs)",
                        self.name,
                        elapsed,
                    )
            return self._state.value

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        通过熔断器调用函数

        根据当前状态决定是否允许调用:
        - CLOSED: 正常调用
        - OPEN: 快速失败
        - HALF_OPEN: 允许有限数量的探测调用

        Args:
            func: 要调用的函数
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            函数调用结果

        Raises:
            RuntimeError: 当熔断器处于OPEN状态时
            Exception: 被调用函数抛出的异常
        """
        state = self.get_state()

        if state == CircuitState.OPEN.value:
            elapsed = time.monotonic() - self._last_failure_time
            remaining = self.recovery_timeout - elapsed
            raise RuntimeError(
                f"熔断器 [{self.name}] 处于开启状态，"
                f"剩余恢复时间: {remaining:.1f}s"
            )

        if state == CircuitState.HALF_OPEN.value:
            with self._lock:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise RuntimeError(
                        f"熔断器 [{self.name}] 处于半开状态，"
                        f"探测请求数已达上限 ({self.half_open_max_calls})"
                    )
                self._half_open_calls += 1

        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as exc:
            self.record_failure()
            raise

    def record_success(self) -> None:
        """记录一次成功调用"""
        with self._lock:
            self._window.append(_WindowRecord(
                timestamp=time.monotonic(),
                success=True,
            ))
            self._total_calls += 1
            self._total_successes += 1

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                # 半开状态下，如果探测请求全部成功，关闭熔断器
                if self._half_open_successes >= self.half_open_max_calls:
                    self._transition_to(CircuitState.CLOSED)
                    logger.info(
                        "熔断器 [%s] 从 HALF_OPEN 转换到 CLOSED "
                        "(探测成功 %d/%d)",
                        self.name,
                        self._half_open_successes,
                        self.half_open_max_calls,
                    )

    def record_failure(self) -> None:
        """记录一次失败调用"""
        with self._lock:
            self._window.append(_WindowRecord(
                timestamp=time.monotonic(),
                success=False,
            ))
            self._total_calls += 1
            self._total_failures += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # 半开状态下探测失败，重新打开熔断器
                self._transition_to(CircuitState.OPEN)
                logger.warning(
                    "熔断器 [%s] 从 HALF_OPEN 转换到 OPEN "
                    "(探测失败)",
                    self.name,
                )
            elif self._state == CircuitState.CLOSED:
                # 检查是否需要打开熔断器
                failure_rate = self._calculate_failure_rate()
                if failure_rate >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
                    logger.warning(
                        "熔断器 [%s] 从 CLOSED 转换到 OPEN "
                        "(失败率 %.1f%% >= 阈值 %.1f%%)",
                        self.name,
                        failure_rate * 100,
                        self.failure_threshold * 100,
                    )

    def _calculate_failure_rate(self) -> float:
        """
        计算滑动窗口内的失败率

        Returns:
            失败率 (0.0 ~ 1.0)，如果窗口为空返回0.0
        """
        if not self._window:
            return 0.0

        total = len(self._window)
        failures = sum(1 for record in self._window if not record.success)
        return failures / total

    def _transition_to(self, new_state: CircuitState) -> None:
        """
        执行状态转换

        Args:
            new_state: 目标状态
        """
        old_state = self._state
        self._state = new_state

        if new_state == CircuitState.OPEN:
            self._total_circuit_opens += 1
            self._last_failure_time = time.monotonic()
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._half_open_successes = 0
        elif new_state == CircuitState.CLOSED:
            # 关闭时重置半开计数，但保留窗口历史
            self._half_open_calls = 0
            self._half_open_successes = 0

    def reset(self) -> None:
        """重置熔断器到初始状态"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._window.clear()
            self._last_failure_time = 0.0
            self._half_open_calls = 0
            self._half_open_successes = 0
            logger.info("熔断器 [%s] 已重置", self.name)

    def force_open(self) -> None:
        """强制打开熔断器"""
        with self._lock:
            self._transition_to(CircuitState.OPEN)
            logger.warning("熔断器 [%s] 已被强制打开", self.name)

    def force_close(self) -> None:
        """强制关闭熔断器"""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            logger.info("熔断器 [%s] 已被强制关闭", self.name)

    def get_stats(self) -> Dict[str, Any]:
        """
        获取熔断器统计信息

        Returns:
            包含详细统计信息的字典
        """
        with self._lock:
            failure_rate = self._calculate_failure_rate()
            state = self.get_state()

            return {
                "name": self.name,
                "state": state,
                "failure_rate": round(failure_rate, 4),
                "failure_threshold": self.failure_threshold,
                "window_size": len(self._window),
                "max_window_size": self.window_size,
                "recovery_timeout": self.recovery_timeout,
                "half_open_max_calls": self.half_open_max_calls,
                "half_open_calls": self._half_open_calls,
                "half_open_successes": self._half_open_successes,
                "total_calls": self._total_calls,
                "total_successes": self._total_successes,
                "total_failures": self._total_failures,
                "total_circuit_opens": self._total_circuit_opens,
                "last_failure_time": self._last_failure_time,
                "is_open": state == CircuitState.OPEN.value,
                "is_half_open": state == CircuitState.HALF_OPEN.value,
                "is_closed": state == CircuitState.CLOSED.value,
            }

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(name={self.name!r}, "
            f"state={self.get_state()}, "
            f"failure_rate={self._calculate_failure_rate():.2%})"
        )
