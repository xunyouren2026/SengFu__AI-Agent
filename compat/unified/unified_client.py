"""
统一客户端 (Unified Client)

整合模型路由、负载均衡、降级处理，提供统一的多后端推理接口。
用户无需关心底层使用哪个后端，只需调用统一 API 即可。

模块路径: compat/unified/unified_client.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Union

import httpx

from .fallback_handler import FallbackHandler, FallbackPolicy, FallbackRule, FallbackTrigger
from .load_balancer import BackendEndpoint, LoadBalancer, LoadBalancingStrategy
from .model_router import ModelRouter, RoutingRule, RoutingStrategy, TaskType

logger = logging.getLogger(__name__)


@dataclass
class UnifiedClientConfig:
    """统一客户端配置。

    Attributes:
        default_model: 默认模型名称。
        default_timeout: 默认请求超时（秒）。
        max_retries: 默认最大重试次数。
        enable_fallback: 是否启用降级处理。
        enable_load_balancing: 是否启用负载均衡。
        lb_strategy: 负载均衡策略。
        fallback_max: 最大降级次数。
        health_check_interval: 健康检查间隔（秒）。
    """

    default_model: str = "default"
    default_timeout: float = 120.0
    max_retries: int = 3
    enable_fallback: bool = True
    enable_load_balancing: bool = True
    lb_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    fallback_max: int = 3
    health_check_interval: float = 30.0


@dataclass
class UnifiedResponse:
    """统一响应。

    Attributes:
        content: 生成内容。
        model: 使用的模型。
        backend: 使用的后端。
        prompt_tokens: 提示词 token 数。
        completion_tokens: 生成 token 数。
        total_tokens: 总 token 数。
        finish_reason: 结束原因。
        latency_ms: 请求延迟（毫秒）。
        metadata: 额外元数据。
    """

    content: str
    model: str = ""
    backend: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "content": self.content,
            "model": self.model,
            "backend": self.backend,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "finish_reason": self.finish_reason,
            "latency_ms": round(self.latency_ms, 2),
            "metadata": self.metadata,
        }


class UnifiedClient:
    """统一推理客户端。

    整合模型路由、负载均衡、降级处理，提供统一的多后端推理接口。

    Args:
        config: 统一客户端配置。
        router: 可选的模型路由器。
        load_balancer: 可选的负载均衡器。
        fallback_handler: 可选的降级处理器。
    """

    def __init__(
        self,
        config: Optional[UnifiedClientConfig] = None,
        router: Optional[ModelRouter] = None,
        load_balancer: Optional[LoadBalancer] = None,
        fallback_handler: Optional[FallbackHandler] = None,
    ) -> None:
        self._config = config or UnifiedClientConfig()
        self._router = router or ModelRouter(default_backend="default")
        self._load_balancer = load_balancer or LoadBalancer(strategy=self._config.lb_strategy)
        self._fallback = fallback_handler or FallbackHandler(
            policy=FallbackPolicy(
                max_fallbacks=self._config.fallback_max,
                rules=[
                    FallbackRule(
                        name="server_error",
                        trigger=FallbackTrigger.SERVER_ERROR,
                        http_status_codes=[500, 502, 503, 504],
                        max_retries=1,
                    ),
                    FallbackRule(
                        name="connection_error",
                        trigger=FallbackTrigger.CONNECTION_ERROR,
                        max_retries=1,
                    ),
                    FallbackRule(
                        name="timeout",
                        trigger=FallbackTrigger.TIMEOUT,
                        timeout_seconds=self._config.default_timeout,
                        max_retries=1,
                    ),
                ],
            ),
        )
        self._http_client: Optional[httpx.Client] = None
        self._async_http_client: Optional[httpx.AsyncClient] = None

    @property
    def config(self) -> UnifiedClientConfig:
        """获取客户端配置。"""
        return self._config

    @property
    def router(self) -> ModelRouter:
        """获取模型路由器。"""
        return self._router

    @property
    def load_balancer(self) -> LoadBalancer:
        """获取负载均衡器。"""
        return self._load_balancer

    @property
    def fallback_handler(self) -> FallbackHandler:
        """获取降级处理器。"""
        return self._fallback

    def add_backend(
        self,
        name: str,
        base_url: str,
        backend_type: str = "unknown",
        api_key: Optional[str] = None,
        weight: int = 100,
        priority: int = 0,
        supported_models: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """添加推理后端。

        Args:
            name: 后端名称。
            base_url: 服务地址。
            backend_type: 后端类型。
            api_key: API 密钥。
            weight: 负载均衡权重。
            priority: 优先级。
            supported_models: 支持的模型列表。
            **kwargs: 额外参数。
        """
        endpoint = BackendEndpoint(
            name=name,
            base_url=base_url,
            api_key=api_key,
            backend_type=backend_type,
            weight=weight,
            priority=priority,
        )
        self._load_balancer.add_endpoint(endpoint)

        self._fallback.add_backend(
            {"name": name, "base_url": base_url, "api_key": api_key, "backend_type": backend_type},
            priority=priority,
        )

        capabilities: Dict[str, Any] = {
            "base_url": base_url,
            "backend_type": backend_type,
            "supported_models": supported_models or [],
        }
        capabilities.update(kwargs)
        self._router.register_backend(name, capabilities)

        logger.info("添加后端: %s (%s) type=%s", name, base_url, backend_type)

    def _get_http_client(self) -> httpx.Client:
        """获取或创建同步 HTTP 客户端。"""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.Client(
                timeout=httpx.Timeout(self._config.default_timeout),
            )
        return self._http_client

    def _get_async_http_client(self) -> httpx.AsyncClient:
        """获取或创建异步 HTTP 客户端。"""
        if self._async_http_client is None or self._async_http_client.is_closed:
            self._async_http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._config.default_timeout),
            )
        return self._async_http_client

    def _resolve_endpoint(self, model_name: str, task_type: Optional[TaskType] = None) -> Dict[str, Any]:
        """解析请求到具体端点。

        Args:
            model_name: 模型名称。
            task_type: 任务类型。

        Returns:
            包含 name, base_url, api_key, backend_type 的端点信息。
        """
        routing_result = self._router.route(model_name=model_name, task_type=task_type)
        backend_name = routing_result.backend_name

        if self._config.enable_load_balancing:
            endpoint = self._load_balancer.select_endpoint(key=model_name)
            if endpoint:
                return {
                    "name": endpoint.name,
                    "base_url": endpoint.base_url,
                    "api_key": endpoint.api_key,
                    "backend_type": endpoint.backend_type,
                }

        for backend in self._fallback.backends:
            if backend.get("name") == backend_name:
                return backend

        return {"name": backend_name, "base_url": "", "api_key": None, "backend_type": "unknown"}

    def _build_openai_request(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """构建 OpenAI 兼容的请求体。"""
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
        payload.update(kwargs)
        return payload

    def _parse_openai_response(self, data: Dict[str, Any], backend_name: str) -> UnifiedResponse:
        """解析 OpenAI 兼容的响应。"""
        choices = data.get("choices", [])
        content = ""
        finish_reason = "stop"
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            finish_reason = choices[0].get("finish_reason", "stop")

        usage = data.get("usage", {})
        return UnifiedResponse(
            content=content,
            model=data.get("model", ""),
            backend=backend_name,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            finish_reason=finish_reason,
        )

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> UnifiedResponse:
        """同步聊天补全。

        自动路由到合适的后端，支持降级处理。

        Args:
            messages: 消息列表。
            model: 模型名称。
            max_tokens: 最大生成 token 数。
            temperature: 采样温度。
            top_p: nucleus 采样阈值。
            **kwargs: 额外参数。

        Returns:
            UnifiedResponse 统一响应。
        """
        model_name = model or self._config.default_model
        start_time = time.time()
        tried_backends: Set[str] = set()

        for attempt in range(self._config.max_retries + 1):
            endpoint = self._resolve_endpoint(model_name, TaskType.CHAT)
            backend_name = endpoint.get("name", "")

            if backend_name in tried_backends and self._config.enable_fallback:
                next_backend = self._fallback.get_next_backend(exclude=tried_backends)
                if next_backend:
                    endpoint = next_backend
                    backend_name = endpoint.get("name", "")
                else:
                    break

            tried_backends.add(backend_name)
            base_url = endpoint.get("base_url", "")
            if not base_url:
                continue

            self._load_balancer.record_request_start(backend_name)

            try:
                payload = self._build_openai_request(
                    messages=messages,
                    model=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=False,
                    **kwargs,
                )
                headers: Dict[str, str] = {"Content-Type": "application/json"}
                api_key = endpoint.get("api_key")
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"

                client = self._get_http_client()
                response = client.post(
                    f"{base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self._config.default_timeout,
                )
                response.raise_for_status()

                latency_ms = (time.time() - start_time) * 1000
                data = response.json()
                result = self._parse_openai_response(data, backend_name)
                result.latency_ms = latency_ms

                self._load_balancer.record_request_success(backend_name, latency_ms)
                self._fallback.record_success(backend_name)
                return result

            except Exception as exc:
                latency_ms = (time.time() - start_time) * 1000
                self._load_balancer.record_request_failure(backend_name, str(exc))
                self._fallback.record_failure(backend_name)
                logger.warning(
                    "后端 '%s' 请求失败 (尝试 %d): %s",
                    backend_name, attempt + 1, exc,
                )

                if not self._config.enable_fallback:
                    raise

        raise RuntimeError(f"所有后端均不可用，已尝试: {tried_backends}")

    async def async_chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> UnifiedResponse:
        """异步聊天补全。

        Args:
            messages: 消息列表。
            model: 模型名称。
            max_tokens: 最大生成 token 数。
            temperature: 采样温度。
            top_p: nucleus 采样阈值。
            **kwargs: 额外参数。

        Returns:
            UnifiedResponse 统一响应。
        """
        model_name = model or self._config.default_model
        start_time = time.time()
        tried_backends: Set[str] = set()

        for attempt in range(self._config.max_retries + 1):
            endpoint = self._resolve_endpoint(model_name, TaskType.CHAT)
            backend_name = endpoint.get("name", "")

            if backend_name in tried_backends and self._config.enable_fallback:
                next_backend = self._fallback.get_next_backend(exclude=tried_backends)
                if next_backend:
                    endpoint = next_backend
                    backend_name = endpoint.get("name", "")
                else:
                    break

            tried_backends.add(backend_name)
            base_url = endpoint.get("base_url", "")
            if not base_url:
                continue

            self._load_balancer.record_request_start(backend_name)

            try:
                payload = self._build_openai_request(
                    messages=messages,
                    model=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=False,
                    **kwargs,
                )
                headers: Dict[str, str] = {"Content-Type": "application/json"}
                api_key = endpoint.get("api_key")
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"

                client = self._get_async_http_client()
                response = await client.post(
                    f"{base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self._config.default_timeout,
                )
                response.raise_for_status()

                latency_ms = (time.time() - start_time) * 1000
                data = response.json()
                result = self._parse_openai_response(data, backend_name)
                result.latency_ms = latency_ms

                self._load_balancer.record_request_success(backend_name, latency_ms)
                self._fallback.record_success(backend_name)
                return result

            except Exception as exc:
                latency_ms = (time.time() - start_time) * 1000
                self._load_balancer.record_request_failure(backend_name, str(exc))
                self._fallback.record_failure(backend_name)
                logger.warning(
                    "后端 '%s' 异步请求失败 (尝试 %d): %s",
                    backend_name, attempt + 1, exc,
                )

                if not self._config.enable_fallback:
                    raise

        raise RuntimeError(f"所有后端均不可用，已尝试: {tried_backends}")

    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息。

        Returns:
            包含路由、负载均衡、降级统计的字典。
        """
        return {
            "load_balancer": self._load_balancer.get_all_stats(),
            "fallback": self._fallback.get_stats(),
            "routing_table": self._router.get_routing_table(),
            "config": {
                "default_model": self._config.default_model,
                "enable_fallback": self._config.enable_fallback,
                "enable_load_balancing": self._config.enable_load_balancing,
                "lb_strategy": self._config.lb_strategy.value,
            },
        }

    def close(self) -> None:
        """关闭所有 HTTP 客户端。"""
        if self._http_client is not None and not self._http_client.is_closed:
            self._http_client.close()
            self._http_client = None

    async def async_close(self) -> None:
        """异步关闭所有 HTTP 客户端。"""
        if self._async_http_client is not None and not self._async_http_client.is_closed:
            await self._async_http_client.aclose()
            self._async_http_client = None

    def __enter__(self) -> "UnifiedClient":
        self._get_http_client()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> "UnifiedClient":
        self._get_async_http_client()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.async_close()
