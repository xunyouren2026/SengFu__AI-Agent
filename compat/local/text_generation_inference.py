"""
Text Generation Inference (TGI) 后端客户端

通过 HTTP API 与 HuggingFace TGI 服务交互，支持 OpenAI 兼容的 Chat/Completion/Embedding
接口、流式输出、token 流式、模型信息查询等功能。

API 参考: https://huggingface.co/docs/text-generation-inference/en/

模块路径: compat/local/text_generation_inference.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx

logger = logging.getLogger(__name__)


@dataclass
class TGIConfig:
    """TGI 客户端配置。

    Attributes:
        base_url: TGI 服务地址。
        api_key: 可选的 API 密钥（HF token）。
        model: 默认模型名称。
        timeout: HTTP 请求超时时间（秒）。
        max_retries: 最大重试次数。
        retry_delay: 重试基础延迟（秒）。
        best_of: 最佳采样数。
        details: 是否返回详细信息。
        decoder_input_details: 是否返回 decoder input 详情。
    """

    base_url: str = "http://localhost:8080"
    api_key: Optional[str] = None
    model: Optional[str] = None
    timeout: float = 120.0
    max_retries: int = 3
    retry_delay: float = 1.0
    best_of: int = 1
    details: bool = False
    decoder_input_details: bool = False


@dataclass
class TGICompletionResponse:
    """TGI 文本生成响应。

    Attributes:
        content: 生成的文本。
        model: 使用的模型。
        prompt_tokens: 提示词 token 数。
        completion_tokens: 生成 token 数。
        total_tokens: 总 token 数。
        finish_reason: 结束原因。
        details: 详细生成信息。
    """

    content: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = "stop"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "content": self.content,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "finish_reason": self.finish_reason,
            "details": self.details,
        }


@dataclass
class TGIChatResponse:
    """TGI 聊天补全响应。

    Attributes:
        content: 助手回复内容。
        model: 使用的模型。
        prompt_tokens: 提示词 token 数。
        completion_tokens: 生成 token 数。
        total_tokens: 总 token 数。
        finish_reason: 结束原因。
    """

    content: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = "stop"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "content": self.content,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "finish_reason": self.finish_reason,
        }


@dataclass
class TGIEmbeddingResponse:
    """TGI 嵌入向量响应。

    Attributes:
        embeddings: 嵌入向量列表。
        model: 模型名称。
    """

    embeddings: List[List[float]]
    model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {"embeddings": self.embeddings, "model": self.model}


class TGIClient:
    """HuggingFace Text Generation Inference HTTP 客户端。

    封装 TGI 的 OpenAI 兼容 API，提供文本生成、聊天补全、嵌入向量、
    流式输出、token 流式、模型信息查询等功能。

    Args:
        base_url: TGI 服务基础 URL。
        api_key: 可选的 HF API 密钥。
        model: 默认模型名称。
        timeout: 请求超时时间（秒）。
        max_retries: 最大重试次数。
        config: 可选的 TGIConfig 配置对象。
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
        config: Optional[TGIConfig] = None,
        **kwargs: Any,
    ) -> None:
        self._config = config or TGIConfig(
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

    def _retry_request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """带指数退避重试的同步请求。"""
        last_error: Optional[Exception] = None
        client = self._get_client()
        for attempt in range(self._config.max_retries):
            try:
                response = client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning("TGI 请求失败 (%d/%d): %s", attempt + 1, self._config.max_retries, exc)
                if attempt < self._config.max_retries - 1:
                    time.sleep(self._config.retry_delay * (2 ** attempt))
        raise last_error  # type: ignore[misc]

    async def _async_retry_request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """带指数退避重试的异步请求。"""
        last_error: Optional[Exception] = None
        client = self._get_async_client()
        for attempt in range(self._config.max_retries):
            try:
                response = await client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning("TGI 异步请求失败 (%d/%d): %s", attempt + 1, self._config.max_retries, exc)
                if attempt < self._config.max_retries - 1:
                    await asyncio.sleep(self._config.retry_delay * (2 ** attempt))
        raise last_error  # type: ignore[misc]

    def _extract_usage(self, data: Dict[str, Any]) -> Dict[str, int]:
        """提取 token 使用信息。"""
        usage = data.get("usage", {})
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

    def health_check(self) -> Dict[str, Any]:
        """检查 TGI 服务健康状态。"""
        response = self._retry_request("GET", "/health")
        return response.json()

    async def async_health_check(self) -> Dict[str, Any]:
        """异步检查服务健康状态。"""
        response = await self._async_retry_request("GET", "/health")
        return response.json()

    def get_model_info(self) -> Dict[str, Any]:
        """获取当前加载模型的详细信息。"""
        response = self._retry_request("GET", "/info")
        return response.json()

    async def async_get_model_info(self) -> Dict[str, Any]:
        """异步获取模型详细信息。"""
        response = await self._async_retry_request("GET", "/info")
        return response.json()

    def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> TGICompletionResponse:
        """同步文本生成。

        Args:
            prompt: 输入提示文本。
            model: 模型名称。
            max_new_tokens: 最大生成 token 数。
            temperature: 采样温度。
            top_p: nucleus 采样阈值。
            top_k: top-k 采样参数。
            repetition_penalty: 重复惩罚系数。
            stop: 停止词列表。
            **kwargs: 额外参数。

        Returns:
            TGICompletionResponse 生成结果。
        """
        payload: Dict[str, Any] = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "best_of": self._config.best_of,
                "details": self._config.details,
                "decoder_input_details": self._config.decoder_input_details,
            },
        }
        if top_k is not None:
            payload["parameters"]["top_k"] = top_k
        if stop:
            payload["parameters"]["stop"] = stop
        if model:
            payload["model"] = model
        payload["parameters"].update(kwargs.pop("parameters", {}))
        payload.update(kwargs)

        response = self._retry_request("POST", "/generate", json=payload)
        data = response.json()
        generated = data[0] if isinstance(data, list) and data else data
        content = generated.get("generated_text", "")
        details = generated.get("details", {})
        token_info = details.get("finish_reason", {})
        usage = self._extract_usage(generated)
        return TGICompletionResponse(
            content=content,
            model=generated.get("model", self._config.model or ""),
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            finish_reason=str(token_info.get("finish_reason", "stop")) if isinstance(token_info, dict) else "stop",
            details=details,
        )

    async def async_complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> TGICompletionResponse:
        """异步文本生成。"""
        payload: Dict[str, Any] = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "best_of": self._config.best_of,
                "details": self._config.details,
            },
        }
        if top_k is not None:
            payload["parameters"]["top_k"] = top_k
        if stop:
            payload["parameters"]["stop"] = stop
        if model:
            payload["model"] = model
        payload["parameters"].update(kwargs.pop("parameters", {}))
        payload.update(kwargs)

        response = await self._async_retry_request("POST", "/generate", json=payload)
        data = response.json()
        generated = data[0] if isinstance(data, list) and data else data
        content = generated.get("generated_text", "")
        details = generated.get("details", {})
        usage = self._extract_usage(generated)
        return TGICompletionResponse(
            content=content,
            model=generated.get("model", self._config.model or ""),
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            finish_reason="stop",
            details=details,
        )

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> TGIChatResponse:
        """同步聊天补全（OpenAI 兼容接口）。"""
        payload: Dict[str, Any] = {
            "model": model or self._config.model or "default",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop
        payload.update(kwargs)

        response = self._retry_request("POST", "/v1/chat/completions", json=payload)
        data = response.json()
        choices = data.get("choices", [])
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        usage = self._extract_usage(data)
        return TGIChatResponse(
            content=content,
            model=data.get("model", ""),
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            finish_reason=choices[0].get("finish_reason", "stop") if choices else "stop",
        )

    async def async_chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> TGIChatResponse:
        """异步聊天补全。"""
        payload: Dict[str, Any] = {
            "model": model or self._config.model or "default",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop
        payload.update(kwargs)

        response = await self._async_retry_request("POST", "/v1/chat/completions", json=payload)
        data = response.json()
        choices = data.get("choices", [])
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        usage = self._extract_usage(data)
        return TGIChatResponse(
            content=content,
            model=data.get("model", ""),
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            finish_reason=choices[0].get("finish_reason", "stop") if choices else "stop",
        )

    def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.8,
        **kwargs: Any,
    ) -> Any:
        """同步流式聊天补全。"""
        payload: Dict[str, Any] = {
            "model": model or self._config.model or "default",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        payload.update(kwargs)
        client = self._get_client()
        with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line and line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

    async def async_stream_chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.8,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """异步流式聊天补全。"""
        payload: Dict[str, Any] = {
            "model": model or self._config.model or "default",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        payload.update(kwargs)
        client = self._get_async_client()
        async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line and line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

    def embed(
        self,
        input_text: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> TGIEmbeddingResponse:
        """生成嵌入向量。"""
        payload: Dict[str, Any] = {
            "model": model or self._config.model or "default",
            "inputs": input_text if isinstance(input_text, list) else [input_text],
        }
        payload.update(kwargs)
        response = self._retry_request("POST", "/embed", json=payload)
        data = response.json()
        return TGIEmbeddingResponse(
            embeddings=data.get("embeddings", []),
            model=data.get("model", ""),
        )

    async def async_embed(
        self,
        input_text: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> TGIEmbeddingResponse:
        """异步生成嵌入向量。"""
        payload: Dict[str, Any] = {
            "model": model or self._config.model or "default",
            "inputs": input_text if isinstance(input_text, list) else [input_text],
        }
        payload.update(kwargs)
        response = await self._async_retry_request("POST", "/embed", json=payload)
        data = response.json()
        return TGIEmbeddingResponse(
            embeddings=data.get("embeddings", []),
            model=data.get("model", ""),
        )

    def close(self) -> None:
        """关闭同步 HTTP 客户端。"""
        if self._client is not None and not self._client.is_closed:
            self._client.close()
            self._client = None

    async def async_close(self) -> None:
        """关闭异步 HTTP 客户端。"""
        if self._async_client is not None and not self._async_client.is_closed:
            await self._async_client.aclose()
            self._async_client = None

    def __enter__(self) -> "TGIClient":
        self._get_client()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> "TGIClient":
        self._get_async_client()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.async_close()
