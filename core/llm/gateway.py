"""
AGI Unified Framework - Model Gateway
统一模型网关路由器，支持多后端注册、路由和健康检查
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from .base import (
    GenerateParams,
    LLMBackend,
    LLMChunk,
    LLMError,
    LLMResponse,
    Message,
    ModelNotFoundError,
)


@dataclass
class BackendStats:
    """后端统计信息"""
    name: str = ""
    total_requests: int = 0
    total_errors: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    last_request_time: float = 0.0
    last_error_time: float = 0.0
    is_healthy: bool = True
    last_health_check: float = 0.0

    @property
    def error_rate(self) -> float:
        return self.total_errors / max(self.total_requests, 1)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.total_requests, 1)


@dataclass
class RequestRecord:
    """请求记录"""
    model: str = ""
    backend_name: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""
    tokens_used: int = 0


class ModelGateway:
    """
    统一模型网关路由器

    功能：
    - 注册多个LLM后端
    - 按模型名称前缀自动路由（gpt->OpenAI, claude->Anthropic, local->Local）
    - 默认后端设置
    - 后端健康检查
    - 请求统计和监控
    - 路由规则自定义
    """

    # 默认路由规则：模型名称前缀 -> 后端名称
    DEFAULT_ROUTING_RULES = {
        "gpt": "openai",
        "o1": "openai",
        "o3": "openai",
        "claude": "anthropic",
        "local": "local",
        "llama": "local",
        "mistral": "local",
        "qwen": "local",
        "deepseek": "local",
        "gemini": "google",
    }

    def __init__(
        self,
        default_backend: str = "",
        health_check_interval: float = 60.0,
        enable_stats: bool = True,
    ):
        self._backends: Dict[str, LLMBackend] = {}
        self._default_backend = default_backend
        self._routing_rules: Dict[str, str] = dict(self.DEFAULT_ROUTING_RULES)
        self._custom_routes: Dict[str, str] = {}
        self._health_check_interval = health_check_interval
        self._enable_stats = enable_stats

        self._stats: Dict[str, BackendStats] = {}
        self._request_history: List[RequestRecord] = []
        self._max_history = 1000

        self._lock = threading.RLock()
        self._health_check_thread: Optional[threading.Thread] = None
        self._running = False

    def register_backend(self, name: str, backend: LLMBackend) -> None:
        """
        注册后端

        Args:
            name: 后端名称（如"openai", "anthropic", "local"）
            backend: LLM后端实例
        """
        with self._lock:
            self._backends[name] = backend
            self._stats[name] = BackendStats(name=name)

    def unregister_backend(self, name: str) -> bool:
        """
        注销后端

        Args:
            name: 后端名称

        Returns:
            bool: 是否成功注销
        """
        with self._lock:
            if name in self._backends:
                del self._backends[name]
                self._stats.pop(name, None)
                return True
            return False

    def get_backend(self, name: str) -> Optional[LLMBackend]:
        """获取指定名称的后端"""
        with self._lock:
            return self._backends.get(name)

    def add_routing_rule(self, model_prefix: str, backend_name: str) -> None:
        """
        添加路由规则

        Args:
            model_prefix: 模型名称前缀
            backend_name: 后端名称
        """
        with self._lock:
            self._routing_rules[model_prefix.lower()] = backend_name

    def add_custom_route(self, model_name: str, backend_name: str) -> None:
        """
        添加自定义路由（精确匹配）

        Args:
            model_name: 精确的模型名称
            backend_name: 后端名称
        """
        with self._lock:
            self._custom_routes[model_name.lower()] = backend_name

    def _resolve_backend(self, model: str) -> Tuple[str, LLMBackend]:
        """
        根据模型名称解析对应的后端

        Args:
            model: 模型名称

        Returns:
            (backend_name, backend_instance)

        Raises:
            ModelNotFoundError: 无法找到对应的后端
        """
        with self._lock:
            model_lower = model.lower()

            # 1. 精确匹配自定义路由
            if model_lower in self._custom_routes:
                backend_name = self._custom_routes[model_lower]
                if backend_name in self._backends:
                    return backend_name, self._backends[backend_name]

            # 2. 前缀匹配路由规则
            for prefix, backend_name in self._routing_rules.items():
                if model_lower.startswith(prefix):
                    if backend_name in self._backends:
                        return backend_name, self._backends[backend_name]

            # 3. 使用默认后端
            if self._default_backend and self._default_backend in self._backends:
                return self._default_backend, self._backends[self._default_backend]

            # 4. 尝试第一个可用的后端
            if self._backends:
                first_name = next(iter(self._backends))
                return first_name, self._backends[first_name]

            raise ModelNotFoundError(
                f"No backend registered for model '{model}' and no default backend set"
            )

    def generate(
        self,
        model: str,
        messages: List[Message],
        params: Optional[GenerateParams] = None,
    ) -> LLMResponse:
        """
        路由到对应后端并生成回复

        Args:
            model: 模型名称
            messages: 消息列表
            params: 生成参数

        Returns:
            LLMResponse: 模型响应
        """
        backend_name, backend = self._resolve_backend(model)
        start_time = time.time()

        try:
            response = backend.generate(messages, params)
            latency = (time.time() - start_time) * 1000

            if self._enable_stats:
                self._record_request(
                    model=model,
                    backend_name=backend_name,
                    start_time=start_time,
                    latency_ms=latency,
                    success=True,
                    tokens_used=response.usage.total_tokens,
                )

            response.metadata["gateway_backend"] = backend_name
            return response

        except LLMError as e:
            latency = (time.time() - start_time) * 1000

            if self._enable_stats:
                self._record_request(
                    model=model,
                    backend_name=backend_name,
                    start_time=start_time,
                    latency_ms=latency,
                    success=False,
                    error=str(e),
                )

            e.metadata = getattr(e, "metadata", {})
            e.metadata["gateway_backend"] = backend_name
            raise

    def stream(
        self,
        model: str,
        messages: List[Message],
        params: Optional[GenerateParams] = None,
    ) -> Iterator[LLMChunk]:
        """
        路由到对应后端并流式生成

        Args:
            model: 模型名称
            messages: 消息列表
            params: 生成参数

        Yields:
            LLMChunk: 流式响应块
        """
        backend_name, backend = self._resolve_backend(model)
        start_time = time.time()
        total_tokens = 0

        try:
            for chunk in backend.stream(messages, params):
                if chunk.usage:
                    total_tokens += chunk.usage.completion_tokens
                chunk.metadata["gateway_backend"] = backend_name
                yield chunk

            latency = (time.time() - start_time) * 1000

            if self._enable_stats:
                self._record_request(
                    model=model,
                    backend_name=backend_name,
                    start_time=start_time,
                    latency_ms=latency,
                    success=True,
                    tokens_used=total_tokens,
                )

        except LLMError as e:
            latency = (time.time() - start_time) * 1000

            if self._enable_stats:
                self._record_request(
                    model=model,
                    backend_name=backend_name,
                    start_time=start_time,
                    latency_ms=latency,
                    success=False,
                    error=str(e),
                )
            raise

    def _record_request(
        self,
        model: str,
        backend_name: str,
        start_time: float,
        latency_ms: float,
        success: bool,
        tokens_used: int = 0,
        error: str = "",
    ) -> None:
        """记录请求统计"""
        with self._lock:
            # 更新后端统计
            if backend_name in self._stats:
                stats = self._stats[backend_name]
                stats.total_requests += 1
                stats.total_latency_ms += latency_ms
                stats.last_request_time = time.time()
                stats.total_tokens += tokens_used
                if not success:
                    stats.total_errors += 1
                    stats.last_error_time = time.time()

            # 记录请求历史
            record = RequestRecord(
                model=model,
                backend_name=backend_name,
                start_time=start_time,
                end_time=time.time(),
                latency_ms=latency,
                success=success,
                error=error,
                tokens_used=tokens_used,
            )
            self._request_history.append(record)
            if len(self._request_history) > self._max_history:
                self._request_history = self._request_history[-self._max_history:]

    def check_health(self, backend_name: Optional[str] = None) -> Dict[str, bool]:
        """
        健康检查

        Args:
            backend_name: 指定后端名称，为None时检查所有后端

        Returns:
            Dict[str, bool]: 后端名称 -> 是否健康
        """
        results = {}
        with self._lock:
            targets = (
                {backend_name: self._backends[backend_name]}
                if backend_name and backend_name in self._backends
                else dict(self._backends)
            )

        for name, backend in targets.items():
            try:
                healthy = backend.health_check()
                results[name] = healthy
                if self._enable_stats and name in self._stats:
                    self._stats[name].is_healthy = healthy
                    self._stats[name].last_health_check = time.time()
            except Exception:
                results[name] = False
                if self._enable_stats and name in self._stats:
                    self._stats[name].is_healthy = False
                    self._stats[name].last_health_check = time.time()

        return results

    def start_health_monitor(self) -> None:
        """启动后台健康检查"""
        if self._running:
            return

        self._running = True

        def _health_loop():
            while self._running:
                self.check_health()
                time.sleep(self._health_check_interval)

        self._health_check_thread = threading.Thread(
            target=_health_loop, daemon=True, name="gateway-health-check"
        )
        self._health_check_thread.start()

    def stop_health_monitor(self) -> None:
        """停止后台健康检查"""
        self._running = False
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5)
            self._health_check_thread = None

    def get_stats(self, backend_name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取统计信息

        Args:
            backend_name: 指定后端名称

        Returns:
            统计信息字典
        """
        with self._lock:
            if backend_name:
                stats = self._stats.get(backend_name)
                if stats:
                    return {
                        "name": stats.name,
                        "total_requests": stats.total_requests,
                        "total_errors": stats.total_errors,
                        "error_rate": stats.error_rate,
                        "total_tokens": stats.total_tokens,
                        "avg_latency_ms": stats.avg_latency_ms,
                        "is_healthy": stats.is_healthy,
                        "last_request_time": stats.last_request_time,
                    }
                return {}

            return {
                "backends": {
                    name: {
                        "total_requests": s.total_requests,
                        "total_errors": s.total_errors,
                        "error_rate": s.error_rate,
                        "total_tokens": s.total_tokens,
                        "avg_latency_ms": s.avg_latency_ms,
                        "is_healthy": s.is_healthy,
                    }
                    for name, s in self._stats.items()
                },
                "total_requests": sum(s.total_requests for s in self._stats.values()),
                "total_errors": sum(s.total_errors for s in self._stats.values()),
                "registered_backends": list(self._backends.keys()),
                "routing_rules": dict(self._routing_rules),
                "custom_routes": dict(self._custom_routes),
                "default_backend": self._default_backend,
            }

    def get_recent_requests(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的请求记录"""
        with self._lock:
            recent = self._request_history[-limit:]
            return [
                {
                    "model": r.model,
                    "backend": r.backend_name,
                    "latency_ms": round(r.latency_ms, 2),
                    "success": r.success,
                    "tokens": r.tokens_used,
                    "error": r.error,
                    "time": r.start_time,
                }
                for r in reversed(recent)
            ]

    def list_backends(self) -> List[str]:
        """列出所有注册的后端"""
        with self._lock:
            return list(self._backends.keys())

    def list_models(self) -> Dict[str, List[str]]:
        """列出所有后端支持的模型"""
        result = {}
        with self._lock:
            for name, backend in self._backends.items():
                try:
                    info = backend.get_model_info()
                    result[name] = [info.name]
                except Exception:
                    result[name] = ["unknown"]
        return result

    def close(self):
        """关闭网关，清理资源"""
        self.stop_health_monitor()
        with self._lock:
            for backend in self._backends.values():
                try:
                    backend.close()
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
