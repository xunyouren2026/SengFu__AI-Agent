"""
重试策略模块

提供灵活的重试机制，支持多种退避策略、可重试异常过滤、
重试回调和熔断器集成。
"""

import functools
import logging
import random
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional, Set, Tuple, Type, TypeVar

from .definitions import AGIError, ErrorCode, ErrorSeverity, NetworkError
from .circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class BackoffStrategy(Enum):
    """退避策略枚举"""

    FIXED = "FIXED"
    EXPONENTIAL = "EXPONENTIAL"
    EXPONENTIAL_WITH_JITTER = "EXPONENTIAL_WITH_JITTER"


@dataclass
class RetryState:
    """
    重试状态

    跟踪当前重试操作的状态信息。

    Attributes:
        attempt: 当前尝试次数（从1开始）
        max_retries: 最大重试次数
        last_error: 最近一次错误
        next_delay: 下次重试延迟（秒）
        total_time: 已消耗总时间（秒）
        start_time: 开始时间戳
        is_exhausted: 重试次数是否已耗尽
    """

    attempt: int = 0
    max_retries: int = 3
    last_error: Optional[Exception] = None
    next_delay: float = 0.0
    total_time: float = 0.0
    start_time: float = 0.0
    is_exhausted: bool = False

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "attempt": self.attempt,
            "max_retries": self.max_retries,
            "last_error": str(self.last_error) if self.last_error else None,
            "next_delay": self.next_delay,
            "total_time": round(self.total_time, 3),
            "is_exhausted": self.is_exhausted,
        }


class RetryPolicy:
    """
    重试策略配置

    封装重试相关的所有配置参数和退避计算逻辑。

    Attributes:
        max_retries: 最大重试次数
        backoff_strategy: 退避策略
        base_delay: 基础延迟（秒）
        max_delay: 最大延迟（秒）
        exponential_base: 指数退避的底数
        jitter_range: 抖动范围（0-1），仅用于EXPONENTIAL_WITH_JITTER
        retryable_exceptions: 可重试的异常类型集合
        circuit_breaker: 关联的熔断器（可选）
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL_WITH_JITTER,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter_range: float = 0.5,
        retryable_exceptions: Optional[Set[Type[Exception]]] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        if max_retries < 0:
            raise ValueError("max_retries 不能为负数")
        if base_delay <= 0:
            raise ValueError("base_delay 必须大于0")
        if max_delay < base_delay:
            raise ValueError("max_delay 不能小于 base_delay")
        if not (0 <= jitter_range <= 1):
            raise ValueError("jitter_range 必须在 [0, 1] 范围内")

        self.max_retries = max_retries
        self.backoff_strategy = backoff_strategy
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter_range = jitter_range
        self.retryable_exceptions = retryable_exceptions or self._default_retryable()
        self.circuit_breaker = circuit_breaker

    @staticmethod
    def _default_retryable() -> Set[Type[Exception]]:
        """默认的可重试异常类型"""
        return {
            NetworkError,
            ConnectionError,
            TimeoutError,
            OSError,
            ConnectionRefusedError,
            ConnectionResetError,
            ConnectionAbortedError,
        }

    def calculate_delay(self, attempt: int) -> float:
        """
        计算下次重试的延迟时间

        根据当前退避策略和尝试次数计算延迟。

        Args:
            attempt: 当前尝试次数（从0开始）

        Returns:
            延迟时间（秒）
        """
        if self.backoff_strategy == BackoffStrategy.FIXED:
            delay = self.base_delay

        elif self.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.base_delay * (self.exponential_base ** attempt)

        elif self.backoff_strategy == BackoffStrategy.EXPONENTIAL_WITH_JITTER:
            exponential_delay = self.base_delay * (self.exponential_base ** attempt)
            jitter = random.uniform(
                -self.jitter_range * exponential_delay,
                self.jitter_range * exponential_delay,
            )
            delay = max(0.1, exponential_delay + jitter)
        else:
            delay = self.base_delay

        return min(delay, self.max_delay)

    def is_retryable(self, exc: Exception) -> bool:
        """
        判断异常是否可重试

        Args:
            exc: 异常对象

        Returns:
            是否可重试
        """
        for exc_type in self.retryable_exceptions:
            if isinstance(exc, exc_type):
                return True

        # AGIError中WARNING级别默认可重试
        if isinstance(exc, AGIError):
            return exc.severity in (
                ErrorSeverity.WARNING,
            )

        return False

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "max_retries": self.max_retries,
            "backoff_strategy": self.backoff_strategy.value,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "exponential_base": self.exponential_base,
            "jitter_range": self.jitter_range,
            "retryable_exceptions": [
                t.__name__ for t in self.retryable_exceptions
            ],
            "has_circuit_breaker": self.circuit_breaker is not None,
        }


class retry:
    """
    重试装饰器

    为函数调用提供自动重试能力，支持多种退避策略和熔断器集成。

    Usage:
        # 基本用法
        @retry(max_retries=3, backoff_strategy=BackoffStrategy.EXPONENTIAL)
        def fetch_data():
            ...

        # 自定义可重试异常
        @retry(max_retries=5, retryable_exceptions={ConnectionError, TimeoutError})
        def call_api():
            ...

        # 带回调
        def on_retry_callback(state):
            print(f"重试 {state.attempt}, 等待 {state.next_delay}s")

        @retry(max_retries=3, on_retry=on_retry_callback)
        def unreliable_operation():
            ...
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL_WITH_JITTER,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter_range: float = 0.5,
        retryable_exceptions: Optional[Set[Type[Exception]]] = None,
        on_retry: Optional[Callable[[RetryState], None]] = None,
        on_success: Optional[Callable[[RetryState], None]] = None,
        on_exhausted: Optional[Callable[[RetryState], None]] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        self.policy = RetryPolicy(
            max_retries=max_retries,
            backoff_strategy=backoff_strategy,
            base_delay=base_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
            jitter_range=jitter_range,
            retryable_exceptions=retryable_exceptions,
            circuit_breaker=circuit_breaker,
        )
        self.on_retry = on_retry
        self.on_success = on_success
        self.on_exhausted = on_exhausted

    def __call__(self, func: F) -> F:
        """装饰器入口"""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._execute_with_retry(func, args, kwargs)

        return wrapper  # type: ignore

    def _execute_with_retry(
        self, func: Callable[..., Any], args: tuple, kwargs: dict
    ) -> Any:
        """
        执行带重试逻辑的函数调用

        Args:
            func: 目标函数
            args: 位置参数
            kwargs: 关键字参数

        Returns:
            函数执行结果

        Raises:
            最后一次异常（当重试耗尽时）
        """
        state = RetryState(
            max_retries=self.policy.max_retries,
            start_time=time.monotonic(),
        )
        last_exception: Optional[Exception] = None

        while state.attempt <= self.policy.max_retries:
            # 检查熔断器状态
            if self.policy.circuit_breaker is not None:
                cb_state = self.policy.circuit_breaker.get_state()
                if cb_state == "OPEN":
                    raise NetworkError(
                        code=ErrorCode.NETWORK_CONNECTION_FAILED,
                        message="熔断器开启，拒绝请求",
                        details={"circuit_state": cb_state},
                    )

            try:
                result = func(*args, **kwargs)

                # 记录成功
                state.total_time = time.monotonic() - state.start_time
                if self.policy.circuit_breaker is not None:
                    self.policy.circuit_breaker.record_success()
                if self.on_success is not None:
                    self.on_success(state)

                return result

            except Exception as exc:
                last_exception = exc
                state.last_error = exc
                state.attempt += 1
                state.total_time = time.monotonic() - state.start_time

                # 记录失败到熔断器
                if self.policy.circuit_breaker is not None:
                    self.policy.circuit_breaker.record_failure()

                # 检查是否可重试
                if not self.policy.is_retryable(exc):
                    logger.debug(
                        "异常不可重试，直接抛出: %s: %s",
                        type(exc).__name__,
                        exc,
                    )
                    raise

                # 检查重试次数
                if state.attempt > self.policy.max_retries:
                    state.is_exhausted = True
                    logger.warning(
                        "重试次数已耗尽 (%d/%d): %s: %s",
                        state.attempt - 1,
                        self.policy.max_retries,
                        type(exc).__name__,
                        exc,
                    )
                    if self.on_exhausted is not None:
                        self.on_exhausted(state)
                    raise

                # 计算延迟并等待
                state.next_delay = self.policy.calculate_delay(state.attempt - 1)
                logger.info(
                    "第 %d 次重试，等待 %.2fs: %s: %s",
                    state.attempt,
                    state.next_delay,
                    type(exc).__name__,
                    exc,
                )

                if self.on_retry is not None:
                    self.on_retry(state)

                time.sleep(state.next_delay)

        # 理论上不会到达这里，但作为安全保障
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("重试逻辑异常终止")


def calculate_delay(
    attempt: int,
    strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter_range: float = 0.5,
) -> float:
    """
    计算重试延迟的便捷函数

    Args:
        attempt: 当前尝试次数（从0开始）
        strategy: 退避策略
        base_delay: 基础延迟
        max_delay: 最大延迟
        exponential_base: 指数底数
        jitter_range: 抖动范围

    Returns:
        延迟时间（秒）
    """
    policy = RetryPolicy(
        backoff_strategy=strategy,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter_range=jitter_range,
    )
    return policy.calculate_delay(attempt)
