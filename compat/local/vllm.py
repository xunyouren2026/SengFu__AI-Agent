"""
vLLM 推理后端客户端

通过 HTTP API 与 vLLM 服务交互，支持 OpenAI 兼容的 Chat/Completion/Embedding 接口、
流式输出、模型列表、token 编码等功能。

API 参考: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

模块路径: compat/local/vllm.py
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
class VLLMConfig:
    """vLLM 客户端配置。

    Attributes:
        base_url: vLLM 服务地址。
        api_key: 可选的 API 密钥。
        model: 默认模型名称。
        timeout: HTTP 请求超时时间（秒）。
        max_retries: 最大重试次数。
        retry_delay: 重试基础延迟（秒）。
        max_model_len: 模型最大上下文长度。
        gpu_memory_utilization: GPU 显存利用率。
        enable_prefix_caching: 是否启用前缀缓存。
    """

    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    model: Optional[str] = None
    timeout: float = 120.0
    max_retries: int = 3
    retry_delay: float = 1.0
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9
    enable_prefix_caching: bool = False


@dataclass
class VLLMCompletionResponse:
    """vLLM 文本生成响应。

    Attributes:
        content: 生成的文本。
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
class VLLMChatResponse:
    """vLLM 聊天补全响应。

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
class VLLMEmbeddingResponse:
    """vLLM 嵌入向量响应。

    Attributes:
        embeddings: 嵌入向量列表。
        model: 模型名称。
        usage: token 使用统计。
    """

    embeddings: List[List[float]]
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {"embeddings": self.embeddings, "model": self.model, "usage": self.usage}


class VLLMClient:
    """vLLM HTTP 客户端。

    封装 vLLM 的 OpenAI 兼容 API，提供文本生成、聊天补全、嵌入向量、
    流式输出、模型管理等功能。

    Args:
        base_url: vLLM 服务基础 URL。
        api_key: 可选的 API 密钥。
        model: 默认模型名称。
        timeout: 请求超时时间（秒）。
        max_retries: 最大重试次数。
        config: 可选的 VLLMConfig 配置对象。
        **kwargs: 传递给 httpx.Client 的额外参数。
    """

    DEFAULT_BASE_URL: str = "http://localhost:8000"

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        config: Optional[VLLMConfig] = None,
        **kwargs: Any,
    ) -> None:
        self._config = config or VLLMConfig(
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
                logger.warning("vLLM 请求失败 (%d/%d): %s", attempt + 1, self._config.max_retries, exc)
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
                logger.warning("vLLM 异步请求失败 (%d/%d): %s", attempt + 1, self._config.max_retries, exc)
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
        """检查 vLLM 服务健康状态。"""
        response = self._retry_request("GET", "/health")
        return response.json()

    async def async_health_check(self) -> Dict[str, Any]:
        """异步检查服务健康状态。"""
        response = await self._async_retry_request("GET", "/health")
        return response.json()

    def get_model_info(self) -> Dict[str, Any]:
        """获取当前加载模型的详细信息。"""
        response = self._retry_request("GET", "/v1/models")
        return response.json()

    async def async_get_model_info(self) -> Dict[str, Any]:
        """异步获取模型详细信息。"""
        response = await self._async_retry_request("GET", "/v1/models")
        return response.json()

    def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = -1,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        n: int = 1,
        **kwargs: Any,
    ) -> VLLMCompletionResponse:
        """同步文本生成。

        Args:
            prompt: 输入提示文本。
            model: 模型名称。
            max_tokens: 最大生成 token 数。
            temperature: 采样温度。
            top_p: nucleus 采样阈值。
            top_k: top-k 采样参数（-1 表示不限制）。
            frequency_penalty: 频率惩罚。
            presence_penalty: 存在惩罚。
            stop: 停止词列表。
            n: 生成候选数量。
            **kwargs: 额外参数。

        Returns:
            VLLMCompletionResponse 生成结果。
        """
        payload: Dict[str, Any] = {
            "model": model or self._config.model or "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "n": n,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop
        payload.update(kwargs)

        response = self._retry_request("POST", "/v1/completions", json=payload)
        data = response.json()
        choices = data.get("choices", [])
        content = choices[0].get("text", "") if choices else ""
        usage = self._extract_usage(data)
        return VLLMCompletionResponse(
            content=content,
            model=data.get("model", ""),
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            finish_reason=choices[0].get("finish_reason", "stop") if choices else "stop",
        )

    async def async_complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = -1,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> VLLMCompletionResponse:
        """异步文本生成。"""
        payload: Dict[str, Any] = {
            "model": model or self._config.model or "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop
        payload.update(kwargs)

        response = await self._async_retry_request("POST", "/v1/completions", json=payload)
        data = response.json()
        choices = data.get("choices", [])
        content = choices[0].get("text", "") if choices else ""
        usage = self._extract_usage(data)
        return VLLMCompletionResponse(
            content=content,
            model=data.get("model", ""),
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            finish_reason=choices[0].get("finish_reason", "stop") if choices else "stop",
        )

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> VLLMChatResponse:
        """同步聊天补全。

        Args:
            messages: 消息列表。
            model: 模型名称。
            max_tokens: 最大生成 token 数。
            temperature: 采样温度。
            top_p: nucleus 采样阈值。
            frequency_penalty: 频率惩罚。
            presence_penalty: 存在惩罚。
            stop: 停止词列表。
            **kwargs: 额外参数。

        Returns:
            VLLMChatResponse 聊天结果。
        """
        payload: Dict[str, Any] = {
            "model": model or self._config.model or "default",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
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
        return VLLMChatResponse(
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
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> VLLMChatResponse:
        """异步聊天补全。"""
        payload: Dict[str, Any] = {
            "model": model or self._config.model or "default",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
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
        return VLLMChatResponse(
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
        encoding_format: str = "float",
        **kwargs: Any,
    ) -> VLLMEmbeddingResponse:
        """生成嵌入向量。"""
        payload: Dict[str, Any] = {
            "model": model or self._config.model or "default",
            "input": input_text if isinstance(input_text, list) else [input_text],
            "encoding_format": encoding_format,
        }
        payload.update(kwargs)
        response = self._retry_request("POST", "/v1/embeddings", json=payload)
        data = response.json()
        embeddings = [item["embedding"] for item in data.get("data", [])]
        return VLLMEmbeddingResponse(
            embeddings=embeddings,
            model=data.get("model", ""),
            usage=self._extract_usage(data),
        )

    async def async_embed(
        self,
        input_text: Union[str, List[str]],
        model: Optional[str] = None,
        encoding_format: str = "float",
        **kwargs: Any,
    ) -> VLLMEmbeddingResponse:
        """异步生成嵌入向量。"""
        payload: Dict[str, Any] = {
            "model": model or self._config.model or "default",
            "input": input_text if isinstance(input_text, list) else [input_text],
            "encoding_format": encoding_format,
        }
        payload.update(kwargs)
        response = await self._async_retry_request("POST", "/v1/embeddings", json=payload)
        data = response.json()
        embeddings = [item["embedding"] for item in data.get("data", [])]
        return VLLMEmbeddingResponse(
            embeddings=embeddings,
            model=data.get("model", ""),
            usage=self._extract_usage(data),
        )

    def list_models(self) -> List[Dict[str, Any]]:
        """列出可用模型。"""
        response = self._retry_request("GET", "/v1/models")
        return response.json().get("data", [])

    async def async_list_models(self) -> List[Dict[str, Any]]:
        """异步列出可用模型。"""
        response = await self._async_retry_request("GET", "/v1/models")
        return response.json().get("data", [])

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

    def __enter__(self) -> "VLLMClient":
        self._get_client()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> "VLLMClient":
        self._get_async_client()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.async_close()
