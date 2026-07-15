"""
重试策略模块

提供灵活的重试策略，包括固定延迟、指数退避、抖动策略、
断路器集成、重试预算和条件重试。

Classes:
    RetryPolicy: 重试策略基类
    FixedDelayRetry: 固定延迟重试
    ExponentialBackoffRetry: 指数退避重试
    JitterStrategy: 抖动策略
    CircuitBreakerRetry: 断路器重试
    RetryBudget: 重试预算
    ConditionRetry: 条件重试
    RetryResult: 重试结果
    RetryHistory: 重试历史
"""

import random
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union


# ============================================================
# 重试结果与历史
# ============================================================

@dataclass
class RetryAttempt:
    """单次重试尝试记录"""
    attempt: int
    start_time: float
    end_time: float = 0.0
    duration: float = 0.0
    success: bool = False
    error: Optional[str] = None
    error_type: Optional[str] = None
    delay_before: float = 0.0
    will_retry: bool = False


@dataclass
class RetryResult:
    """
    重试执行结果

    Attributes:
        success: 最终是否成功
        result: 成功时的返回值
        error: 最终错误信息
        total_attempts: 总尝试次数
        total_delay: 总延迟时间
        total_duration: 总执行时间
        attempts: 所有尝试记录
    """
    success: bool = False
    result: Any = None
    error: Optional[str] = None
    total_attempts: int = 0
    total_delay: float = 0.0
    total_duration: float = 0.0
    attempts: List[RetryAttempt] = dataclass_field(default_factory=list)

    @property
    def first_error(self) -> Optional[str]:
        """获取第一次错误"""
        for attempt in self.attempts:
            if attempt.error:
                return attempt.error
        return None

    @property
    def last_error(self) -> Optional[str]:
        """获取最后一次错误"""
        for attempt in reversed(self.attempts):
            if attempt.error:
                return attempt.error
        return None

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "success": self.success,
            "total_attempts": self.total_attempts,
            "total_delay": round(self.total_delay, 4),
            "total_duration": round(self.total_duration, 4),
            "error": self.error,
            "first_error": self.first_error,
            "last_error": self.last_error,
        }


class RetryHistory:
    """
    重试历史记录

    跟踪所有重试操作的历史，用于分析和监控。

    Attributes:
        max_records: 最大记录数
    """

    def __init__(self, max_records: int = 1000) -> None:
        self.max_records = max_records
        self._records: List[RetryResult] = []
        self._lock = threading.Lock()

    def add_record(self, result: RetryResult) -> None:
        """添加重试记录"""
        with self._lock:
            self._records.append(result)
            if len(self._records) > self.max_records:
                self._records = self._records[-self.max_records:]

    def get_recent(self, count: int = 10) -> List[RetryResult]:
        """获取最近的记录"""
        with self._lock:
            return list(self._records[-count:])

    def get_failure_rate(self, window: int = 100) -> float:
        """获取最近窗口的失败率"""
        with self._lock:
            recent = self._records[-window:]
            if not recent:
                return 0.0
            failures = sum(1 for r in recent if not r.success)
            return failures / len(recent)

    def clear(self) -> None:
        """清除所有记录"""
        with self._lock:
            self._records.clear()

    @property
    def total_records(self) -> int:
        """总记录数"""
        return len(self._records)

    def summary(self) -> Dict[str, Any]:
        """生成摘要统计"""
        with self._lock:
            if not self._records:
                return {"total": 0}

            total = len(self._records)
            successes = sum(1 for r in self._records if r.success)
            total_attempts = sum(r.total_attempts for r in self._records)
            avg_attempts = total_attempts / total if total > 0 else 0

            return {
                "total": total,
                "successes": successes,
                "failures": total - successes,
                "success_rate": round(successes / total, 4) if total > 0 else 0.0,
                "total_attempts": total_attempts,
                "avg_attempts": round(avg_attempts, 2),
            }


# ============================================================
# 重试策略基类
# ============================================================

class RetryPolicy(ABC):
    """
    重试策略抽象基类

    定义重试策略的统一接口。
    """

    def __init__(
        self,
        max_retries: int = 3,
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    ) -> None:
        self.max_retries = max(1, max_retries)
        self.retryable_exceptions = retryable_exceptions or (Exception,)

    @abstractmethod
    def compute_delay(self, attempt: int) -> float:
        """
        计算第 attempt 次重试的延迟时间

        Args:
            attempt: 当前尝试次数（从0开始）

        Returns:
            延迟秒数
        """
        ...

    def should_retry(
        self,
        attempt: int,
        error: Optional[Exception] = None,
    ) -> bool:
        """
        判断是否应该重试

        Args:
            attempt: 当前尝试次数
            error: 上次错误

        Returns:
            是否继续重试
        """
        if attempt >= self.max_retries:
            return False
        if error is not None:
            return isinstance(error, self.retryable_exceptions)
        return True

    def execute(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> RetryResult:
        """
        执行带重试的函数

        Args:
            fn: 要执行的函数
            *args: 函数位置参数
            **kwargs: 函数关键字参数

        Returns:
            RetryResult 执行结果
        """
        attempts: List[RetryAttempt] = []
        total_delay = 0.0
        start_time = time.time()
        last_error: Optional[str] = None

        for attempt in range(self.max_retries + 1):
            attempt_record = RetryAttempt(
                attempt=attempt,
                start_time=time.time(),
            )

            # 计算并应用延迟（首次不延迟）
            if attempt > 0:
                delay = self.compute_delay(attempt - 1)
                attempt_record.delay_before = delay
                total_delay += delay
                time.sleep(delay)

            try:
                result = fn(*args, **kwargs)
                attempt_record.end_time = time.time()
                attempt_record.duration = (
                    attempt_record.end_time - attempt_record.start_time
                )
                attempt_record.success = True
                attempts.append(attempt_record)

                return RetryResult(
                    success=True,
                    result=result,
                    total_attempts=attempt + 1,
                    total_delay=total_delay,
                    total_duration=time.time() - start_time,
                    attempts=attempts,
                )

            except Exception as e:
                attempt_record.end_time = time.time()
                attempt_record.duration = (
                    attempt_record.end_time - attempt_record.start_time
                )
                attempt_record.success = False
                attempt_record.error = str(e)
                attempt_record.error_type = type(e).__name__
                last_error = str(e)

                will_retry = self.should_retry(attempt, e)
                attempt_record.will_retry = will_retry
                attempts.append(attempt_record)

                if not will_retry:
                    break

        return RetryResult(
            success=False,
            error=last_error,
            total_attempts=len(attempts),
            total_delay=total_delay,
            total_duration=time.time() - start_time,
            attempts=attempts,
        )


# ============================================================
# 固定延迟重试
# ============================================================

class FixedDelayRetry(RetryPolicy):
    """
    固定延迟重试策略

    每次重试之间等待固定的时间间隔。

    Usage:
        policy = FixedDelayRetry(max_retries=5, delay_seconds=2.0)
        result = policy.execute(some_function, arg1, arg2)
    """

    def __init__(
        self,
        max_retries: int = 3,
        delay_seconds: float = 1.0,
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    ) -> None:
        super().__init__(max_retries, retryable_exceptions)
        self.delay_seconds = max(0.0, delay_seconds)

    def compute_delay(self, attempt: int) -> float:
        return self.delay_seconds


# ============================================================
# 指数退避重试
# ============================================================

class ExponentialBackoffRetry(RetryPolicy):
    """
    指数退避重试策略

    延迟时间按指数增长: delay = base * (multiplier ^ attempt)

    Usage:
        policy = ExponentialBackoffRetry(
            max_retries=5,
            base_delay=1.0,
            multiplier=2.0,
            max_delay=60.0,
        )
        result = policy.execute(unstable_function)
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        multiplier: float = 2.0,
        max_delay: float = 60.0,
        jitter: Optional["JitterStrategy"] = None,
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    ) -> None:
        super().__init__(max_retries, retryable_exceptions)
        self.base_delay = max(0.0, base_delay)
        self.multiplier = max(1.0, multiplier)
        self.max_delay = max(0.0, max_delay)
        self.jitter = jitter

    def compute_delay(self, attempt: int) -> float:
        delay = self.base_delay * (self.multiplier ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter is not None:
            delay = self.jitter.apply(delay)

        return delay


# ============================================================
# 抖动策略
# ============================================================

class JitterType(Enum):
    """抖动类型枚举"""
    NONE = "none"
    FULL = "full"          # 完全抖动: random(0, delay)
    HALF = "half"          # 半抖动: delay/2 + random(0, delay/2)
    EQUAL = "equal"        # 等量抖动: random(delay/2, delay*3/2)
    DECORRELATED = "decorrelated"  # 去相关抖动


class JitterStrategy:
    """
    抖动策略

    在重试延迟中加入随机性，避免多个客户端同时重试导致的"惊群效应"。

    Usage:
        jitter = JitterStrategy(jitter_type=JitterType.HALF)
        policy = ExponentialBackoffRetry(max_retries=5, jitter=jitter)
    """

    def __init__(
        self,
        jitter_type: JitterType = JitterType.HALF,
        seed: Optional[int] = None,
    ) -> None:
        self.jitter_type = jitter_type
        self._rng = random.Random(seed)
        self._last_delay: float = 0.0

    def apply(self, delay: float) -> float:
        """
        对延迟应用抖动

        Args:
            delay: 原始延迟时间

        Returns:
            应用抖动后的延迟时间
        """
        if self.jitter_type == JitterType.NONE:
            return delay

        if self.jitter_type == JitterType.FULL:
            return self._rng.uniform(0, delay)

        if self.jitter_type == JitterType.HALF:
            half = delay / 2.0
            return half + self._rng.uniform(0, half)

        if self.jitter_type == JitterType.EQUAL:
            half = delay / 2.0
            return self._rng.uniform(half, delay + half)

        if self.jitter_type == JitterType.DECORRELATED:
            # 去相关抖动: base * (0.5 + random(0, 1) * 0.5) + random(0, 1)
            # 使用上一次延迟来计算
            if self._last_delay == 0:
                self._last_delay = delay
            new_delay = self._last_delay * (
                0.5 + self._rng.uniform(0, 1) * 0.5
            )
            new_delay = max(delay * 0.1, min(new_delay, delay * 3.0))
            self._last_delay = new_delay
            return new_delay

        return delay

    def reset(self) -> None:
        """重置内部状态"""
        self._last_delay = 0.0


# ============================================================
# 断路器重试
# ============================================================

class CircuitState(Enum):
    """断路器状态"""
    CLOSED = "closed"      # 正常状态，允许请求通过
    OPEN = "open"          # 断路状态，拒绝请求
    HALF_OPEN = "half_open"  # 半开状态，允许少量探测请求


class CircuitBreakerRetry(RetryPolicy):
    """
    断路器重试策略

    当失败率达到阈值时打开断路器，阻止后续请求。
    经过冷却期后进入半开状态，允许探测请求。

    Usage:
        breaker = CircuitBreakerRetry(
            max_retries=3,
            failure_threshold=5,
            recovery_timeout=30.0,
            half_open_max_calls=3,
        )
        result = breaker.execute(risky_function)
    """

    def __init__(
        self,
        max_retries: int = 3,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    ) -> None:
        super().__init__(max_retries, retryable_exceptions)
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_timeout = max(1.0, recovery_timeout)
        self.half_open_max_calls = max(1, half_open_max_calls)

        self._state = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._last_failure_time: float = 0.0
        self._half_open_calls: int = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """获取当前断路器状态"""
        with self._lock:
            if self._state == CircuitState.OPEN:
                # 检查是否应该转为半开状态
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
            return self._state

    def _record_success(self) -> None:
        """记录成功"""
        with self._lock:
            self._success_count += 1
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
                if self._half_open_calls >= self.half_open_max_calls:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
            elif self._state == CircuitState.CLOSED:
                # 成功后逐渐减少失败计数
                self._failure_count = max(0, self._failure_count - 1)

    def _record_failure(self) -> None:
        """记录失败"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN

    def compute_delay(self, attempt: int) -> float:
        # 断路器本身不添加额外延迟
        return 0.0

    def should_retry(self, attempt: int, error: Optional[Exception] = None) -> bool:
        """判断是否应该重试，同时检查断路器状态"""
        if attempt >= self.max_retries:
            return False

        current_state = self.state
        if current_state == CircuitState.OPEN:
            return False

        if error is not None:
            return isinstance(error, self.retryable_exceptions)
        return True

    def execute(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> RetryResult:
        """执行带断路器保护的重试"""
        current_state = self.state
        if current_state == CircuitState.OPEN:
            return RetryResult(
                success=False,
                error="断路器处于打开状态，拒绝请求",
                total_attempts=0,
                total_delay=0.0,
                total_duration=0.0,
            )

        result = super().execute(fn, *args, **kwargs)

        if result.success:
            self._record_success()
        else:
            self._record_failure()

        return result

    def reset(self) -> None:
        """重置断路器状态"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = 0.0
            self._half_open_calls = 0

    def trip(self) -> None:
        """手动打开断路器"""
        with self._lock:
            self._state = CircuitState.OPEN
            self._last_failure_time = time.time()

    def get_stats(self) -> Dict[str, Any]:
        """获取断路器统计信息"""
        with self._lock:
            return {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
                "last_failure_time": self._last_failure_time,
            }


# ============================================================
# 重试预算
# ============================================================

class RetryBudget:
    """
    重试预算

    限制在时间窗口内的总重试次数，防止重试风暴。

    Usage:
        budget = RetryBudget(max_retries_per_minute=30)
        if budget.can_retry():
            budget.record_retry()
            # 执行重试
    """

    def __init__(
        self,
        max_retries_per_window: int = 60,
        window_seconds: float = 60.0,
        max_total_retries: Optional[int] = None,
    ) -> None:
        self.max_retries_per_window = max(0, max_retries_per_window)
        self.window_seconds = max(1.0, window_seconds)
        self.max_total_retries = max_total_retries

        self._retry_timestamps: List[float] = []
        self._total_retries: int = 0
        self._lock = threading.Lock()

    def can_retry(self) -> bool:
        """
        检查是否还有重试预算

        Returns:
            是否可以重试
        """
        with self._lock:
            now = time.time()
            cutoff = now - self.window_seconds

            # 清理过期记录
            self._retry_timestamps = [
                ts for ts in self._retry_timestamps if ts > cutoff
            ]

            # 检查窗口内限制
            if len(self._retry_timestamps) >= self.max_retries_per_window:
                return False

            # 检查总限制
            if (self.max_total_retries is not None
                    and self._total_retries >= self.max_total_retries):
                return False

            return True

    def record_retry(self) -> bool:
        """
        记录一次重试

        Returns:
            是否记录成功（预算内）
        """
        with self._lock:
            if not self.can_retry():
                return False

            self._retry_timestamps.append(time.time())
            self._total_retries += 1
            return True

    def remaining(self) -> int:
        """
        获取当前窗口内剩余重试次数

        Returns:
            剩余次数
        """
        with self._lock:
            now = time.time()
            cutoff = now - self.window_seconds
            self._retry_timestamps = [
                ts for ts in self._retry_timestamps if ts > cutoff
            ]
            return max(
                0,
                self.max_retries_per_window - len(self._retry_timestamps),
            )

    def reset(self) -> None:
        """重置预算"""
        with self._lock:
            self._retry_timestamps.clear()
            self._total_retries = 0

    def get_stats(self) -> Dict[str, Any]:
        """获取预算统计"""
        with self._lock:
            now = time.time()
            cutoff = now - self.window_seconds
            current_window = [
                ts for ts in self._retry_timestamps if ts > cutoff
            ]
            return {
                "current_window_usage": len(current_window),
                "max_per_window": self.max_retries_per_window,
                "window_seconds": self.window_seconds,
                "total_retries": self._total_retries,
                "max_total": self.max_total_retries,
                "remaining": max(
                    0,
                    self.max_retries_per_window - len(current_window),
                ),
            }


# ============================================================
# 条件重试
# ============================================================

class ConditionRetry(RetryPolicy):
    """
    条件重试策略

    根据自定义条件决定是否重试，支持基于错误类型、
    错误消息内容和返回值的条件判断。

    Usage:
        policy = ConditionRetry(
            max_retries=5,
            error_predicate=lambda e: "timeout" in str(e).lower(),
            result_predicate=lambda r: r.get("status") != "ok",
            delay_computer=lambda attempt: 0.5 * (2 ** attempt),
        )
        result = policy.execute(flaky_function)
    """

    def __init__(
        self,
        max_retries: int = 3,
        error_predicate: Optional[Callable[[Exception], bool]] = None,
        result_predicate: Optional[Callable[[Any], bool]] = None,
        delay_computer: Optional[Callable[[int], float]] = None,
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    ) -> None:
        super().__init__(max_retries, retryable_exceptions)
        self.error_predicate = error_predicate
        self.result_predicate = result_predicate
        self.delay_computer = delay_computer

    def compute_delay(self, attempt: int) -> float:
        if self.delay_computer is not None:
            return max(0.0, self.delay_computer(attempt))
        return 1.0

    def should_retry(
        self,
        attempt: int,
        error: Optional[Exception] = None,
    ) -> bool:
        if attempt >= self.max_retries:
            return False

        if error is not None:
            if not isinstance(error, self.retryable_exceptions):
                return False
            if self.error_predicate is not None:
                return self.error_predicate(error)
            return True

        return True

    def execute(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> RetryResult:
        """
        执行带条件重试的函数

        支持基于返回值的条件重试：即使函数没有抛出异常，
        如果 result_predicate 返回 True 也会重试。
        """
        attempts: List[RetryAttempt] = []
        total_delay = 0.0
        start_time = time.time()
        last_error: Optional[str] = None

        for attempt in range(self.max_retries + 1):
            attempt_record = RetryAttempt(
                attempt=attempt,
                start_time=time.time(),
            )

            if attempt > 0:
                delay = self.compute_delay(attempt - 1)
                attempt_record.delay_before = delay
                total_delay += delay
                time.sleep(delay)

            try:
                result = fn(*args, **kwargs)
                attempt_record.end_time = time.time()
                attempt_record.duration = (
                    attempt_record.end_time - attempt_record.start_time
                )
                attempt_record.success = True
                attempts.append(attempt_record)

                # 检查结果条件
                if (self.result_predicate is not None
                        and self.result_predicate(result)):
                    if attempt < self.max_retries:
                        continue
                    # 达到最大重试次数，返回最后的结果
                    return RetryResult(
                        success=True,
                        result=result,
                        total_attempts=attempt + 1,
                        total_delay=total_delay,
                        total_duration=time.time() - start_time,
                        attempts=attempts,
                    )

                return RetryResult(
                    success=True,
                    result=result,
                    total_attempts=attempt + 1,
                    total_delay=total_delay,
                    total_duration=time.time() - start_time,
                    attempts=attempts,
                )

            except Exception as e:
                attempt_record.end_time = time.time()
                attempt_record.duration = (
                    attempt_record.end_time - attempt_record.start_time
                )
                attempt_record.success = False
                attempt_record.error = str(e)
                attempt_record.error_type = type(e).__name__
                last_error = str(e)

                will_retry = self.should_retry(attempt, e)
                attempt_record.will_retry = will_retry
                attempts.append(attempt_record)

                if not will_retry:
                    break

        return RetryResult(
            success=False,
            error=last_error,
            total_attempts=len(attempts),
            total_delay=total_delay,
            total_duration=time.time() - start_time,
            attempts=attempts,
        )
