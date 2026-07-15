"""
llama.cpp 推理后端客户端

通过 HTTP API 与 llama.cpp server 交互，支持文本生成、嵌入向量、流式输出、
模型加载/卸载等功能。

API 参考: https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md

模块路径: compat/local/llama_cpp.py
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx

logger = logging.getLogger(__name__)


@dataclass
class LlamaCppConfig:
    """llama.cpp 客户端配置。

    Attributes:
        base_url: llama.cpp server 地址。
        api_key: 可选的 API 密钥。
        model: 默认使用的模型名称。
        timeout: HTTP 请求超时时间（秒）。
        max_retries: 失败后最大重试次数。
        retry_delay: 重试之间的基础延迟（秒）。
        n_ctx: 上下文窗口大小。
        n_batch: 批处理大小。
        n_threads: 推理线程数。
        verbose: 是否启用详细日志。
    """

    base_url: str = "http://localhost:8080"
    api_key: Optional[str] = None
    model: Optional[str] = None
    timeout: float = 120.0
    max_retries: int = 3
    retry_delay: float = 1.0
    n_ctx: int = 4096
    n_batch: int = 512
    n_threads: int = -1
    verbose: bool = False


@dataclass
class CompletionResponse:
    """文本生成响应。

    Attributes:
        content: 生成的文本内容。
        model: 使用的模型名称。
        prompt_tokens: 提示词 token 数。
        completion_tokens: 生成 token 数。
        total_tokens: 总 token 数。
        finish_reason: 结束原因（stop/length）。
        usage: token 使用详情。
    """

    content: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = "stop"
    usage: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "content": self.content,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
        }


@dataclass
class EmbeddingResponse:
    """嵌入向量响应。

    Attributes:
        embeddings: 嵌入向量列表。
        model: 使用的模型名称。
        total_tokens: 总 token 数。
    """

    embeddings: List[List[float]]
    model: str = ""
    total_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "embeddings": self.embeddings,
            "model": self.model,
            "total_tokens": self.total_tokens,
        }


class LlamaCppClient:
    """llama.cpp HTTP 客户端。

    封装 llama.cpp server 的 OpenAI 兼容 API，提供文本生成、嵌入向量、
    流式输出、模型管理等功能。

    Args:
        base_url: llama.cpp server 基础 URL。
        api_key: 可选的 API 密钥。
        model: 默认模型名称。
        timeout: 请求超时时间（秒）。
        max_retries: 最大重试次数。
        config: 可选的 LlamaCppConfig 配置对象。
        **kwargs: 传递给 httpx.Client 的额外参数。
    """

    DEFAULT_BASE_URL: str = "http://localhost:8080"

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        config: Optional[LlamaCppConfig] = None,
        **kwargs: Any,
    ) -> None:
        self._config = config or LlamaCppConfig(
            base_url=base_url or self.DEFAULT_BASE_URL,
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None
        self._extra_kwargs = kwargs

    @property
    def base_url(self) -> str:
        """获取服务基础 URL。"""
        return self._config.base_url

    @property
    def model(self) -> Optional[str]:
        """获取默认模型名称。"""
        return self._config.model

    def _get_headers(self) -> Dict[str, str]:
        """构建请求头。"""
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return headers

    def _get_client(self) -> httpx.Client:
        """获取或创建同步 HTTP 客户端。"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                base_url=self._config.base_url,
                timeout=httpx.Timeout(self._config.timeout),
                headers=self._get_headers(),
                **self._extra_kwargs,
            )
        return self._client

    def _get_async_client(self) -> httpx.AsyncClient:
        """获取或创建异步 HTTP 客户端。"""
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                base_url=self._config.base_url,
                timeout=httpx.Timeout(self._config.timeout),
                headers=self._get_headers(),
                **self._extra_kwargs,
            )
        return self._async_client

    def _retry_request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """带重试机制的同步请求。

        Args:
            method: HTTP 方法。
            url: 请求路径。
            **kwargs: 传递给 httpx.request 的参数。

        Returns:
            HTTP 响应对象。

        Raises:
            httpx.HTTPError: 所有重试耗尽后的最后一次错误。
        """
        last_error: Optional[Exception] = None
        client = self._get_client()
        for attempt in range(self._config.max_retries):
            try:
                response = client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning(
                    "请求失败 (尝试 %d/%d): %s %s - %s",
                    attempt + 1,
                    self._config.max_retries,
                    method,
                    url,
                    exc,
                )
                if attempt < self._config.max_retries - 1:
                    delay = self._config.retry_delay * (2 ** attempt)
                    time.sleep(delay)
        raise last_error  # type: ignore[misc]

    async def _async_retry_request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """带重试机制的异步请求。

        Args:
            method: HTTP 方法。
            url: 请求路径。
            **kwargs: 传递给 httpx.request 的参数。

        Returns:
            HTTP 响应对象。

        Raises:
            httpx.HTTPError: 所有重试耗尽后的最后一次错误。
        """
        last_error: Optional[Exception] = None
        client = self._get_async_client()
        for attempt in range(self._config.max_retries):
            try:
                response = await client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning(
                    "异步请求失败 (尝试 %d/%d): %s %s - %s",
                    attempt + 1,
                    self._config.max_retries,
                    method,
                    url,
                    exc,
                )
                if attempt < self._config.max_retries - 1:
                    delay = self._config.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
        raise last_error  # type: ignore[misc]

    def health_check(self) -> Dict[str, Any]:
        """检查 llama.cpp server 健康状态。

        Returns:
            包含健康状态信息的字典。

        Raises:
            httpx.HTTPError: 连接失败时抛出。
        """
        response = self._retry_request("GET", "/health")
        return response.json()

    async def async_health_check(self) -> Dict[str, Any]:
        """异步检查服务健康状态。

        Returns:
            包含健康状态信息的字典。
        """
        response = await self._async_retry_request("GET", "/health")
        return response.json()

    def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 40,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        """同步文本生成。

        Args:
            prompt: 输入提示文本。
            model: 模型名称，默认使用配置中的模型。
            max_tokens: 最大生成 token 数。
            temperature: 采样温度。
            top_p: nucleus 采样概率阈值。
            top_k: top-k 采样参数。
            stop: 停止词列表。
            stream: 是否使用流式输出。
            **kwargs: 额外参数。

        Returns:
            CompletionResponse 生成结果。
        """
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "model": model or self._config.model or "default",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stream": stream,
        }
        if stop:
            payload["stop"] = stop
        payload.update(kwargs)

        response = self._retry_request("POST", "/completion", json=payload)
        data = response.json()
        return CompletionResponse(
            content=data.get("content", ""),
            model=data.get("model", ""),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            finish_reason=data.get("finish_reason", "stop"),
            usage=data.get("usage", {}),
        )

    async def async_complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 40,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """异步文本生成。

        Args:
            prompt: 输入提示文本。
            model: 模型名称。
            max_tokens: 最大生成 token 数。
            temperature: 采样温度。
            top_p: nucleus 采样概率阈值。
            top_k: top-k 采样参数。
            stop: 停止词列表。
            **kwargs: 额外参数。

        Returns:
            CompletionResponse 生成结果。
        """
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "model": model or self._config.model or "default",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop
        payload.update(kwargs)

        response = await self._async_retry_request("POST", "/completion", json=payload)
        data = response.json()
        return CompletionResponse(
            content=data.get("content", ""),
            model=data.get("model", ""),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            finish_reason=data.get("finish_reason", "stop"),
            usage=data.get("usage", {}),
        )

    def stream_complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 40,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """同步流式文本生成。

        Args:
            prompt: 输入提示文本。
            model: 模型名称。
            max_tokens: 最大生成 token 数。
            temperature: 采样温度。
            top_p: nucleus 采样概率阈值。
            top_k: top-k 采样参数。
            stop: 停止词列表。
            **kwargs: 额外参数。

        Yields:
            每个流式 chunk 的字典数据。
        """
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "model": model or self._config.model or "default",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stream": True,
        }
        if stop:
            payload["stop"] = stop
        payload.update(kwargs)

        client = self._get_client()
        with client.stream("POST", "/completion", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    import json
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    async def async_stream_complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 40,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """异步流式文本生成。

        Args:
            prompt: 输入提示文本。
            model: 模型名称。
            max_tokens: 最大生成 token 数。
            temperature: 采样温度。
            top_p: nucleus 采样概率阈值。
            top_k: top-k 采样参数。
            stop: 停止词列表。
            **kwargs: 额外参数。

        Yields:
            每个流式 chunk 的字典数据。
        """
        import json

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "model": model or self._config.model or "default",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stream": True,
        }
        if stop:
            payload["stop"] = stop
        payload.update(kwargs)

        client = self._get_async_client()
        async with client.stream("POST", "/completion", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    def embed(
        self,
        input_text: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """生成文本嵌入向量。

        Args:
            input_text: 输入文本或文本列表。
            model: 模型名称。
            **kwargs: 额外参数。

        Returns:
            EmbeddingResponse 嵌入向量结果。
        """
        payload: Dict[str, Any] = {
            "input": input_text if isinstance(input_text, list) else [input_text],
            "model": model or self._config.model or "default",
        }
        payload.update(kwargs)

        response = self._retry_request("POST", "/embedding", json=payload)
        data = response.json()
        return EmbeddingResponse(
            embeddings=data.get("embeddings", []),
            model=data.get("model", ""),
            total_tokens=data.get("total_tokens", 0),
        )

    async def async_embed(
        self,
        input_text: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """异步生成文本嵌入向量。

        Args:
            input_text: 输入文本或文本列表。
            model: 模型名称。
            **kwargs: 额外参数。

        Returns:
            EmbeddingResponse 嵌入向量结果。
        """
        payload: Dict[str, Any] = {
            "input": input_text if isinstance(input_text, list) else [input_text],
            "model": model or self._config.model or "default",
        }
        payload.update(kwargs)

        response = await self._async_retry_request("POST", "/embedding", json=payload)
        data = response.json()
        return EmbeddingResponse(
            embeddings=data.get("embeddings", []),
            model=data.get("model", ""),
            total_tokens=data.get("total_tokens", 0),
        )

    def list_models(self) -> List[Dict[str, Any]]:
        """列出可用模型。

        Returns:
            模型信息列表。
        """
        response = self._retry_request("GET", "/v1/models")
        data = response.json()
        return data.get("data", [])

    async def async_list_models(self) -> List[Dict[str, Any]]:
        """异步列出可用模型。

        Returns:
            模型信息列表。
        """
        response = await self._async_retry_request("GET", "/v1/models")
        data = response.json()
        return data.get("data", [])

    def close(self) -> None:
        """关闭同步 HTTP 客户端连接。"""
        if self._client is not None and not self._client.is_closed:
            self._client.close()
            self._client = None

    async def async_close(self) -> None:
        """关闭异步 HTTP 客户端连接。"""
        if self._async_client is not None and not self._async_client.is_closed:
            await self._async_client.aclose()
            self._async_client = None

    def __enter__(self) -> "LlamaCppClient":
        """上下文管理器入口。"""
        self._get_client()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """上下文管理器退出。"""
        self.close()

    async def __aenter__(self) -> "LlamaCppClient":
        """异步上下文管理器入口。"""
        self._get_async_client()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """异步上下文管理器退出。"""
        await self.async_close()
