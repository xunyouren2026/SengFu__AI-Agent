"""
工具调用重试模块

提供可配置的工具调用重试机制，支持指数退避、可重试异常判断等。
仅使用 Python 标准库。
"""

import random
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from .base import Tool, ToolResult


# ---------------------------------------------------------------------------
# ToolRetryConfig - 重试配置
# ---------------------------------------------------------------------------
@dataclass
class ToolRetryConfig:
    """工具调用重试配置

    Attributes:
        max_retries: 最大重试次数（不含首次执行）
        base_delay: 基础延迟（秒）
        max_delay: 最大延迟（秒）
        exponential_base: 指数退避基数
        jitter: 是否添加随机抖动
        retryable_exceptions: 可重试的异常类型集合
        retryable_error_patterns: 可重试的错误消息模式
        retry_on_timeout: 是否对超时进行重试
        retry_on_failure: 是否对所有失败结果重试
    """
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        OSError,
    )
    retryable_error_patterns: List[str] = field(default_factory=lambda: [
        "timeout",
        "connection",
        "temporary",
        "rate limit",
        "too many requests",
        "server error",
        "503",
        "502",
        "429",
    ])
    retry_on_timeout: bool = True
    retry_on_failure: bool = False


# ---------------------------------------------------------------------------
# RetryRecord - 单次重试记录
# ---------------------------------------------------------------------------
@dataclass
class RetryRecord:
    """重试记录"""
    attempt: int
    success: bool
    error: Optional[str] = None
    delay: float = 0.0
    duration_ms: float = 0.0
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# ToolRetryHandler - 工具调用重试处理器
# ---------------------------------------------------------------------------
class ToolRetryHandler:
    """工具调用重试处理器

    提供带重试的工具执行能力，支持:
    - 指数退避
    - 随机抖动
    - 可重试异常判断
    - 重试历史记录
    """

    def __init__(self, config: Optional[ToolRetryConfig] = None):
        self._config = config or ToolRetryConfig()
        self._lock = threading.Lock()
        self._history: List[List[RetryRecord]] = []

    @property
    def config(self) -> ToolRetryConfig:
        return self._config

    @config.setter
    def config(self, value: ToolRetryConfig) -> None:
        self._config = value

    # ----- 核心方法 -----

    def execute_with_retry(
        self,
        tool: Tool,
        params: dict,
        config: Optional[ToolRetryConfig] = None,
    ) -> ToolResult:
        """带重试的工具执行

        Args:
            tool: 工具实例
            params: 参数字典
            config: 本次执行的重试配置（覆盖默认配置）

        Returns:
            工具执行结果（最后一次尝试的结果）
        """
        cfg = config or self._config
        records: List[RetryRecord] = []
        last_result: Optional[ToolResult] = None

        for attempt in range(cfg.max_retries + 1):
            # 计算延迟（首次不等待）
            delay = 0.0
            if attempt > 0:
                delay = self._calculate_delay(attempt, cfg)
                time.sleep(delay)

            # 执行
            result = tool.execute(params)
            record = RetryRecord(
                attempt=attempt + 1,
                success=result.success,
                error=result.error,
                delay=delay,
                duration_ms=result.duration_ms,
                timestamp=time.time(),
            )
            records.append(record)

            if result.success:
                self._save_history(records)
                return result

            last_result = result

            # 判断是否应该重试
            if attempt < cfg.max_retries and self._should_retry(result, cfg):
                continue
            else:
                break

        self._save_history(records)

        # 返回最后一次结果，附加重试信息
        if last_result is not None:
            last_result.metadata["retry_attempts"] = len(records)
            last_result.metadata["retry_records"] = [
                {
                    "attempt": r.attempt,
                    "success": r.success,
                    "error": r.error,
                    "delay": r.delay,
                    "duration_ms": r.duration_ms,
                }
                for r in records
            ]
            return last_result

        # 不应到达此处
        return ToolResult.fail(
            error="重试处理器未产生结果",
            tool_name=tool.name,
        )

    def execute_callable_with_retry(
        self,
        func: Callable,
        config: Optional[ToolRetryConfig] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """带重试的普通函数执行

        Args:
            func: 可调用对象
            config: 重试配置
            *args, **kwargs: 传递给 func 的参数

        Returns:
            func 的返回值

        Raises:
            最后一次异常（如果所有重试都失败）
        """
        cfg = config or self._config
        last_exception: Optional[Exception] = None

        for attempt in range(cfg.max_retries + 1):
            if attempt > 0:
                delay = self._calculate_delay(attempt, cfg)
                time.sleep(delay)

            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_exception = exc

                if attempt < cfg.max_retries and self._is_retryable_exception(exc, cfg):
                    continue
                raise

        # 不应到达此处
        if last_exception:
            raise last_exception
        raise RuntimeError("重试处理器未产生结果")

    # ----- 重试判断 -----

    def _should_retry(
        self, result: ToolResult, config: ToolRetryConfig
    ) -> bool:
        """判断是否应该重试"""
        if not result.success:
            error = result.error or ""

            # 检查错误消息模式
            error_lower = error.lower()
            for pattern in config.retryable_error_patterns:
                if pattern.lower() in error_lower:
                    return True

            # 检查超时
            if config.retry_on_timeout and "timeout" in error_lower:
                return True

            # 通用失败重试
            if config.retry_on_failure:
                return True

        return False

    def _is_retryable_exception(
        self, exc: Exception, config: ToolRetryConfig
    ) -> bool:
        """判断异常是否可重试"""
        # 检查异常类型
        for exc_type in config.retryable_exceptions:
            if isinstance(exc, exc_type):
                return True

        # 检查错误消息
        error_msg = str(exc).lower()
        for pattern in config.retryable_error_patterns:
            if pattern.lower() in error_msg:
                return True

        return False

    # ----- 延迟计算 -----

    def _calculate_delay(self, attempt: int, config: ToolRetryConfig) -> float:
        """计算重试延迟（指数退避 + 抖动）"""
        # 指数退避: base_delay * exponential_base ^ (attempt - 1)
        delay = config.base_delay * (config.exponential_base ** (attempt - 1))

        # 限制最大延迟
        delay = min(delay, config.max_delay)

        # 添加随机抖动
        if config.jitter:
            jitter_range = delay * 0.25  # +/- 25%
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0.1, delay)  # 确保最小延迟

        return delay

    # ----- 历史记录 -----

    def _save_history(self, records: List[RetryRecord]) -> None:
        """保存重试历史"""
        with self._lock:
            self._history.append(records)
            # 限制历史大小
            if len(self._history) > 1000:
                self._history = self._history[-500:]

    def get_history(self, limit: int = 10) -> List[List[dict]]:
        """获取最近的重试历史"""
        with self._lock:
            result = []
            for records in reversed(self._history[-limit:]):
                result.append([
                    {
                        "attempt": r.attempt,
                        "success": r.success,
                        "error": r.error,
                        "delay": round(r.delay, 3),
                        "duration_ms": r.duration_ms,
                        "timestamp": r.timestamp,
                    }
                    for r in records
                ])
            return result

    def clear_history(self) -> None:
        """清空历史"""
        with self._lock:
            self._history.clear()
