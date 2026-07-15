"""
AGI Unified Framework - Fallback Strategy
自动降级策略，支持多后端优先级排序和自动故障转移
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import (
    GenerateParams,
    LLMBackend,
    LLMError,
    LLMResponse,
    Message,
    RateLimitError,
    TimeoutError,
)


@dataclass
class FallbackConfig:
    """降级配置"""
    max_retries: int = 3
    timeout_per_backend: float = 30.0
    retry_delay: float = 1.0
    retry_backoff_factor: float = 2.0
    max_retry_delay: float = 30.0
    enable_rate_limit_fallback: bool = True
    enable_timeout_fallback: bool = True
    enable_error_fallback: bool = True
    skip_on_auth_error: bool = True  # 认证错误不降级

    def get_retry_delay(self, attempt: int) -> float:
        """计算第N次重试的延迟时间（指数退避）"""
        delay = self.retry_delay * (self.retry_backoff_factor ** (attempt - 1))
        return min(delay, self.max_retry_delay)


@dataclass
class FallbackResult:
    """降级结果"""
    backend_used: str = ""
    attempts: int = 0
    total_time: float = 0.0
    success: bool = False
    errors: List[str] = field(default_factory=list)
    response: Optional[LLMResponse] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend_used": self.backend_used,
            "attempts": self.attempts,
            "total_time": round(self.total_time, 3),
            "success": self.success,
            "errors": self.errors,
        }


@dataclass
class BackendEntry:
    """后端条目"""
    name: str
    backend: LLMBackend
    priority: int = 0
    weight: float = 1.0
    is_available: bool = True
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    total_requests: int = 0
    total_failures: int = 0

    @property
    def failure_rate(self) -> float:
        return self.total_failures / max(self.total_requests, 1)

    def record_success(self):
        self.total_requests += 1
        self.consecutive_failures = 0
        self.is_available = True

    def record_failure(self):
        self.total_requests += 1
        self.total_failures += 1
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        if self.consecutive_failures >= 3:
            self.is_available = False


class FallbackStrategy:
    """
    自动降级策略

    功能：
    - 按优先级排序多个后端
    - 自动故障转移（超时、错误、速率限制）
    - 指数退避重试
    - 后端可用性管理
    - 熔断保护
    """

    def __init__(self, config: Optional[FallbackConfig] = None):
        self._config = config or FallbackConfig()
        self._backends: List[BackendEntry] = []
        self._lock = __import__("threading").RLock()

    @property
    def config(self) -> FallbackConfig:
        return self._config

    @config.setter
    def config(self, value: FallbackConfig):
        self._config = value

    def add_backend(self, name: str, backend: LLMBackend, priority: int = 0) -> None:
        """
        添加后端

        Args:
            name: 后端名称
            backend: LLM后端实例
            priority: 优先级（数值越小优先级越高）
        """
        with self._lock:
            entry = BackendEntry(name=name, backend=backend, priority=priority)
            self._backends.append(entry)
            self._backends.sort(key=lambda e: e.priority)

    def remove_backend(self, name: str) -> bool:
        """移除后端"""
        with self._lock:
            for i, entry in enumerate(self._backends):
                if entry.name == name:
                    self._backends.pop(i)
                    return True
            return False

    def _should_fallback(self, error: Exception) -> bool:
        """
        判断是否应该降级到下一个后端

        Args:
            error: 捕获的异常

        Returns:
            bool: 是否应该降级
        """
        if isinstance(error, LLMError):
            # 认证错误不降级
            if error.error_type == "authentication" and self._config.skip_on_auth_error:
                return False
            # 速率限制错误降级
            if isinstance(error, RateLimitError) and self._config.enable_rate_limit_fallback:
                return True
            # 超时错误降级
            if isinstance(error, TimeoutError) and self._config.enable_timeout_fallback:
                return True
            # 其他可重试错误降级
            if error.retryable and self._config.enable_error_fallback:
                return True
            # 服务器错误降级
            if error.error_type == "server_error" and self._config.enable_error_fallback:
                return True
        return False

    def _get_available_backends(self) -> List[BackendEntry]:
        """获取可用的后端列表"""
        with self._lock:
            return [e for e in self._backends if e.is_available]

    def _reset_backend_availability(self) -> None:
        """重置所有后端的可用性（用于全部不可用时）"""
        with self._lock:
            all_unavailable = all(not e.is_available for e in self._backends)
            if all_unavailable:
                for entry in self._backends:
                    entry.is_available = True
                    entry.consecutive_failures = 0

    def generate_with_fallback(
        self,
        messages: List[Message],
        params: Optional[GenerateParams] = None,
    ) -> FallbackResult:
        """
        带降级的生成

        按优先级依次尝试后端，遇到可降级错误时自动切换到下一个后端。

        Args:
            messages: 消息列表
            params: 生成参数

        Returns:
            FallbackResult: 降级结果
        """
        start_time = time.time()
        errors = []
        attempts = 0

        self._reset_backend_availability()
        available = self._get_available_backends()

        if not available:
            return FallbackResult(
                attempts=0,
                total_time=time.time() - start_time,
                success=False,
                errors=["No available backends"],
            )

        for entry in available:
            if attempts >= self._config.max_retries:
                break

            attempts += 1

            # 如果不是第一次尝试，等待退避时间
            if attempts > 1:
                delay = self._config.get_retry_delay(attempts - 1)
                time.sleep(delay)

            try:
                response = entry.backend.generate(messages, params)
                entry.record_success()

                return FallbackResult(
                    backend_used=entry.name,
                    attempts=attempts,
                    total_time=time.time() - start_time,
                    success=True,
                    errors=errors,
                    response=response,
                )

            except Exception as e:
                error_msg = f"[{entry.name}] {type(e).__name__}: {str(e)}"
                errors.append(error_msg)
                entry.record_failure()

                if not self._should_fallback(e):
                    return FallbackResult(
                        backend_used=entry.name,
                        attempts=attempts,
                        total_time=time.time() - start_time,
                        success=False,
                        errors=errors,
                    )

                continue

        return FallbackResult(
            attempts=attempts,
            total_time=time.time() - start_time,
            success=False,
            errors=errors,
        )

    def generate_with_retry(
        self,
        backend_name: str,
        messages: List[Message],
        params: Optional[GenerateParams] = None,
    ) -> FallbackResult:
        """
        对单个后端进行重试

        Args:
            backend_name: 后端名称
            messages: 消息列表
            params: 生成参数

        Returns:
            FallbackResult: 重试结果
        """
        start_time = time.time()
        errors = []
        attempts = 0

        with self._lock:
            entry = None
            for e in self._backends:
                if e.name == backend_name:
                    entry = e
                    break

        if entry is None:
            return FallbackResult(
                attempts=0,
                total_time=time.time() - start_time,
                success=False,
                errors=[f"Backend '{backend_name}' not found"],
            )

        for attempt in range(1, self._config.max_retries + 1):
            attempts = attempt

            if attempt > 1:
                delay = self._config.get_retry_delay(attempt - 1)
                time.sleep(delay)

            try:
                response = entry.backend.generate(messages, params)
                entry.record_success()

                return FallbackResult(
                    backend_used=entry.name,
                    attempts=attempts,
                    total_time=time.time() - start_time,
                    success=True,
                    errors=errors,
                    response=response,
                )

            except Exception as e:
                error_msg = f"[attempt {attempt}] {type(e).__name__}: {str(e)}"
                errors.append(error_msg)
                entry.record_failure()

                # 不可重试的错误直接返回
                if isinstance(e, LLMError) and not e.retryable:
                    break

        return FallbackResult(
            backend_used=entry.name,
            attempts=attempts,
            total_time=time.time() - start_time,
            success=False,
            errors=errors,
        )

    def get_backend_status(self) -> List[Dict[str, Any]]:
        """获取所有后端状态"""
        with self._lock:
            return [
                {
                    "name": e.name,
                    "priority": e.priority,
                    "is_available": e.is_available,
                    "consecutive_failures": e.consecutive_failures,
                    "failure_rate": round(e.failure_rate, 4),
                    "total_requests": e.total_requests,
                    "last_failure_time": e.last_failure_time,
                }
                for e in self._backends
            ]

    def reset_backend(self, name: str) -> bool:
        """重置指定后端的可用性"""
        with self._lock:
            for entry in self._backends:
                if entry.name == name:
                    entry.is_available = True
                    entry.consecutive_failures = 0
                    return True
            return False

    def reset_all(self) -> None:
        """重置所有后端"""
        with self._lock:
            for entry in self._backends:
                entry.is_available = True
                entry.consecutive_failures = 0
