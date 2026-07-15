"""
降级处理器 (Fallback Handler)

当主后端不可用或返回错误时，自动切换到备用后端，确保服务可用性。
支持基于错误类型、超时、重试次数等条件的降级策略。

模块路径: compat/unified/fallback_handler.py
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import httpx

logger = logging.getLogger(__name__)


class FallbackTrigger(str, Enum):
    """降级触发条件类型。"""

    TIMEOUT = "timeout"
    HTTP_ERROR = "http_error"
    CONNECTION_ERROR = "connection_error"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    CUSTOM = "custom"


@dataclass
class FallbackRule:
    """单条降级规则。

    Attributes:
        name: 规则名称。
        trigger: 触发条件类型。
        http_status_codes: 触发降级的 HTTP 状态码列表。
        max_retries: 触发降级前的最大重试次数。
        timeout_seconds: 超时阈值（秒）。
        error_types: 触发降级的异常类型列表。
        priority: 规则优先级（数字越小优先级越高）。
        enabled: 是否启用此规则。
    """

    name: str
    trigger: FallbackTrigger = FallbackTrigger.SERVER_ERROR
    http_status_codes: List[int] = field(default_factory=lambda: [500, 502, 503, 504])
    max_retries: int = 1
    timeout_seconds: float = 30.0
    error_types: Tuple[Type[Exception], ...] = (httpx.HTTPError,)
    priority: int = 0
    enabled: bool = True

    def should_trigger(
        self,
        error: Optional[Exception] = None,
        status_code: Optional[int] = None,
        elapsed_time: Optional[float] = None,
        retry_count: int = 0,
    ) -> bool:
        """判断当前条件是否触发降级。

        Args:
            error: 捕获的异常。
            status_code: HTTP 响应状态码。
            elapsed_time: 请求耗时（秒）。
            retry_count: 已重试次数。

        Returns:
            是否应触发降级。
        """
        if not self.enabled:
            return False

        if retry_count < self.max_retries:
            return False

        if self.trigger == FallbackTrigger.TIMEOUT:
            return elapsed_time is not None and elapsed_time >= self.timeout_seconds

        if self.trigger == FallbackTrigger.HTTP_ERROR:
            return status_code is not None and status_code in self.http_status_codes

        if self.trigger == FallbackTrigger.CONNECTION_ERROR:
            return isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout))

        if self.trigger == FallbackTrigger.RATE_LIMIT:
            return status_code == 429

        if self.trigger == FallbackTrigger.SERVER_ERROR:
            return status_code is not None and status_code in self.http_status_codes

        if self.trigger == FallbackTrigger.CUSTOM:
            return error is not None and isinstance(error, self.error_types)

        return False


@dataclass
class FallbackPolicy:
    """降级策略配置。

    Attributes:
        name: 策略名称。
        rules: 降级规则列表（按优先级排序）。
        max_fallbacks: 最大降级次数。
        circuit_breaker_threshold: 熔断器触发阈值（连续失败次数）。
        circuit_breaker_reset_seconds: 熔断器重置时间（秒）。
        cooldown_seconds: 降级后冷却时间（秒）。
    """

    name: str = "default"
    rules: List[FallbackRule] = field(default_factory=list)
    max_fallbacks: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_reset_seconds: float = 60.0
    cooldown_seconds: float = 5.0

    def __post_init__(self) -> None:
        """初始化后按优先级排序规则。"""
        self.rules.sort(key=lambda r: r.priority)

    def add_rule(self, rule: FallbackRule) -> "FallbackPolicy":
        """添加降级规则。

        Args:
            rule: 降级规则。

        Returns:
            self，支持链式调用。
        """
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority)
        return self

    def should_fallback(
        self,
        error: Optional[Exception] = None,
        status_code: Optional[int] = None,
        elapsed_time: Optional[float] = None,
        retry_count: int = 0,
    ) -> bool:
        """判断是否应触发降级。

        按优先级依次检查所有规则，任一规则匹配即触发。

        Args:
            error: 捕获的异常。
            status_code: HTTP 状态码。
            elapsed_time: 请求耗时。
            retry_count: 已重试次数。

        Returns:
            是否应触发降级。
        """
        for rule in self.rules:
            if rule.should_trigger(error, status_code, elapsed_time, retry_count):
                logger.info("降级规则 '%s' 触发", rule.name)
                return True
        return False


@dataclass
class CircuitBreakerState:
    """熔断器状态。

    Attributes:
        is_open: 熔断器是否打开（阻断请求）。
        failure_count: 连续失败计数。
        last_failure_time: 上次失败时间戳。
        last_success_time: 上次成功时间戳。
        total_failures: 总失败次数。
        total_successes: 总成功次数。
    """

    is_open: bool = False
    failure_count: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    total_failures: int = 0
    total_successes: int = 0

    def record_success(self) -> None:
        """记录一次成功调用。"""
        self.is_open = False
        self.failure_count = 0
        self.last_success_time = time.time()
        self.total_successes += 1

    def record_failure(self) -> None:
        """记录一次失败调用。"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.total_failures += 1

    def reset(self) -> None:
        """重置熔断器状态。"""
        self.is_open = False
        self.failure_count = 0


class FallbackHandler:
    """降级处理器。

    管理多个后端端点的降级逻辑，支持熔断器、冷却期、自动恢复等功能。

    Args:
        policy: 降级策略配置。
        backends: 备用后端列表，按优先级排序。
    """

    def __init__(
        self,
        policy: Optional[FallbackPolicy] = None,
        backends: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._policy = policy or FallbackPolicy()
        self._backends: List[Dict[str, Any]] = backends or []
        self._circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self._backend_cooldowns: Dict[str, float] = {}
        self._lock = asyncio.Lock() if asyncio.get_event_loop().is_running() else None  # type: ignore[arg-type]

    @property
    def policy(self) -> FallbackPolicy:
        """获取当前降级策略。"""
        return self._policy

    @property
    def backends(self) -> List[Dict[str, Any]]:
        """获取备用后端列表。"""
        return list(self._backends)

    def add_backend(self, backend: Dict[str, Any], priority: int = 0) -> None:
        """添加备用后端。

        Args:
            backend: 后端配置（包含 name, base_url, api_key 等）。
            priority: 优先级（数字越小越优先）。
        """
        backend["_fallback_priority"] = priority
        self._backends.append(backend)
        self._backends.sort(key=lambda b: b.get("_fallback_priority", 0))
        self._circuit_breakers[backend["name"]] = CircuitBreakerState()

    def remove_backend(self, name: str) -> None:
        """移除备用后端。

        Args:
            name: 后端名称。
        """
        self._backends = [b for b in self._backends if b.get("name") != name]
        self._circuit_breakers.pop(name, None)

    def get_circuit_breaker(self, backend_name: str) -> CircuitBreakerState:
        """获取指定后端的熔断器状态。

        Args:
            backend_name: 后端名称。

        Returns:
            熔断器状态。
        """
        if backend_name not in self._circuit_breakers:
            self._circuit_breakers[backend_name] = CircuitBreakerState()
        return self._circuit_breakers[backend_name]

    def is_backend_available(self, backend_name: str) -> bool:
        """检查后端是否可用（未熔断且不在冷却期）。

        Args:
            backend_name: 后端名称。

        Returns:
            后端是否可用。
        """
        cb = self.get_circuit_breaker(backend_name)

        if cb.is_open:
            elapsed = time.time() - cb.last_failure_time
            if elapsed < self._policy.circuit_breaker_reset_seconds:
                logger.warning("后端 '%s' 熔断器开启中，跳过", backend_name)
                return False
            cb.reset()

        cooldown_end = self._backend_cooldowns.get(backend_name, 0.0)
        if time.time() < cooldown_end:
            logger.warning("后端 '%s' 冷却中，跳过", backend_name)
            return False

        return True

    def record_success(self, backend_name: str) -> None:
        """记录后端调用成功。

        Args:
            backend_name: 后端名称。
        """
        cb = self.get_circuit_breaker(backend_name)
        cb.record_success()
        logger.debug("后端 '%s' 调用成功", backend_name)

    def record_failure(self, backend_name: str) -> None:
        """记录后端调用失败。

        Args:
            backend_name: 后端名称。
        """
        cb = self.get_circuit_breaker(backend_name)
        cb.record_failure()
        self._backend_cooldowns[backend_name] = time.time() + self._policy.cooldown_seconds

        if cb.failure_count >= self._policy.circuit_breaker_threshold:
            cb.is_open = True
            logger.warning(
                "后端 '%s' 熔断器开启（连续失败 %d 次）",
                backend_name,
                cb.failure_count,
            )

    def get_available_backends(self) -> List[Dict[str, Any]]:
        """获取当前可用的后端列表。

        Returns:
            可用后端列表（已过滤熔断和冷却中的后端）。
        """
        return [b for b in self._backends if self.is_backend_available(b.get("name", ""))]

    def should_fallback(
        self,
        error: Optional[Exception] = None,
        status_code: Optional[int] = None,
        elapsed_time: Optional[float] = None,
        retry_count: int = 0,
    ) -> bool:
        """判断是否应触发降级。

        Args:
            error: 捕获的异常。
            status_code: HTTP 状态码。
            elapsed_time: 请求耗时。
            retry_count: 已重试次数。

        Returns:
            是否应触发降级。
        """
        return self._policy.should_fallback(error, status_code, elapsed_time, retry_count)

    def get_next_backend(self, exclude: Optional[Set[str]] = None) -> Optional[Dict[str, Any]]:
        """获取下一个可用的备用后端。

        Args:
            exclude: 需要排除的后端名称集合。

        Returns:
            下一个可用后端配置，或 None（无可用后端）。
        """
        exclude = exclude or set()
        available = self.get_available_backends()
        for backend in available:
            name = backend.get("name", "")
            if name not in exclude:
                return backend
        return None

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有后端的降级统计信息。

        Returns:
            后端名称到统计信息的映射。
        """
        stats: Dict[str, Dict[str, Any]] = {}
        for name, cb in self._circuit_breakers.items():
            stats[name] = {
                "is_open": cb.is_open,
                "failure_count": cb.failure_count,
                "total_failures": cb.total_failures,
                "total_successes": cb.total_successes,
                "last_failure_time": cb.last_failure_time,
                "last_success_time": cb.last_success_time,
            }
        return stats

    def reset_all(self) -> None:
        """重置所有后端的熔断器状态。"""
        for cb in self._circuit_breakers.values():
            cb.reset()
        self._backend_cooldowns.clear()
        logger.info("所有后端熔断器已重置")
