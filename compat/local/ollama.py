"""
Ollama 推理后端客户端

通过 HTTP API 与 Ollama 服务交互，支持 Chat、Completion、Embedding、流式输出、
模型拉取/删除/列表等功能。

API 参考: https://github.com/ollama/ollama/blob/main/docs/api.md

模块路径: compat/local/ollama.py
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
class OllamaConfig:
    """Ollama 客户端配置。

    Attributes:
        base_url: Ollama 服务地址。
        model: 默认模型名称。
        timeout: HTTP 请求超时时间（秒）。
        max_retries: 最大重试次数。
        retry_delay: 重试基础延迟（秒）。
        num_ctx: 上下文窗口大小。
        num_batch: 批处理大小。
        num_gpu: GPU 层数（0=全部CPU）。
        temperature: 默认采样温度。
        top_p: 默认 nucleus 采样阈值。
    """

    base_url: str = "http://localhost:11434"
    model: Optional[str] = None
    timeout: float = 300.0
    max_retries: int = 3
    retry_delay: float = 1.0
    num_ctx: int = 4096
    num_batch: int = 512
    num_gpu: int = 0
    temperature: float = 0.8
    top_p: float = 0.9


@dataclass
class OllamaMessage:
    """Ollama 聊天消息。

    Attributes:
        role: 角色（system/user/assistant）。
        content: 消息内容。
    """

    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        """转换为 API 请求格式。"""
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResponse:
    """Ollama 聊天响应。

    Attributes:
        content: 助手回复内容。
        model: 使用的模型。
        done: 是否生成完毕。
        total_duration: 总耗时（纳秒）。
        eval_count: 生成 token 数。
        prompt_eval_count: 提示词 token 数。
    """

    content: str
    model: str = ""
    done: bool = True
    total_duration: int = 0
    eval_count: int = 0
    prompt_eval_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "content": self.content,
            "model": self.model,
            "done": self.done,
            "total_duration": self.total_duration,
            "eval_count": self.eval_count,
            "prompt_eval_count": self.prompt_eval_count,
        }


@dataclass
class GenerateResponse:
    """Ollama 文本生成响应。

    Attributes:
        content: 生成的文本。
        model: 使用的模型。
        done: 是否生成完毕。
        total_duration: 总耗时。
        eval_count: 生成 token 数。
        prompt_eval_count: 提示词 token 数。
    """

    content: str
    model: str = ""
    done: bool = True
    total_duration: int = 0
    eval_count: int = 0
    prompt_eval_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "content": self.content,
            "model": self.model,
            "done": self.done,
            "total_duration": self.total_duration,
            "eval_count": self.eval_count,
            "prompt_eval_count": self.prompt_eval_count,
        }


@dataclass
class EmbeddingResponse:
    """Ollama 嵌入向量响应。

    Attributes:
        embeddings: 嵌入向量列表。
        model: 模型名称。
    """

    embeddings: List[List[float]]
    model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {"embeddings": self.embeddings, "model": self.model}


@dataclass
class ModelInfo:
    """Ollama 模型信息。

    Attributes:
        name: 模型名称。
        size: 模型大小（字节）。
        family: 模型家族。
        parameter_size: 参数规模描述。
        quantization_level: 量化级别。
    """

    name: str
    size: int = 0
    family: str = ""
    parameter_size: str = ""
    quantization_level: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "name": self.name,
            "size": self.size,
            "family": self.family,
            "parameter_size": self.parameter_size,
            "quantization_level": self.quantization_level,
        }


class OllamaClient:
    """Ollama HTTP 客户端。

    封装 Ollama REST API，提供聊天补全、文本生成、嵌入向量、流式输出、
    模型管理（拉取/删除/列表）等功能。

    Args:
        base_url: Ollama 服务基础 URL。
        model: 默认模型名称。
        timeout: 请求超时时间（秒）。
        max_retries: 最大重试次数。
        config: 可选的 OllamaConfig 配置对象。
        **kwargs: 传递给 httpx.Client 的额外参数。
    """

    DEFAULT_BASE_URL: str = "http://localhost:11434"

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 300.0,
        max_retries: int = 3,
        config: Optional[OllamaConfig] = None,
        **kwargs: Any,
    ) -> None:
        self._config = config or OllamaConfig(
            base_url=base_url or self.DEFAULT_BASE_URL,
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
        return {"Content-Type": "application/json"}

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
                logger.warning("Ollama 请求失败 (%d/%d): %s", attempt + 1, self._config.max_retries, exc)
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
                logger.warning("Ollama 异步请求失败 (%d/%d): %s", attempt + 1, self._config.max_retries, exc)
                if attempt < self._config.max_retries - 1:
                    await asyncio.sleep(self._config.retry_delay * (2 ** attempt))
        raise last_error  # type: ignore[misc]

    def health_check(self) -> Dict[str, Any]:
        """检查 Ollama 服务状态。"""
        response = self._retry_request("GET", "/")
        return response.json()

    async def async_health_check(self) -> Dict[str, Any]:
        """异步检查服务状态。"""
        response = await self._async_retry_request("GET", "/")
        return response.json()

    def chat(
        self,
        messages: List[Union[OllamaMessage, Dict[str, str]]],
        model: Optional[str] = None,
        stream: bool = False,
        format: Optional[str] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """同步聊天补全。

        Args:
            messages: 消息列表。
            model: 模型名称。
            stream: 是否流式。
            format: 输出格式（json）。
            **kwargs: 额外参数。

        Returns:
            ChatResponse 聊天结果。
        """
        formatted: List[Dict[str, str]] = []
        for msg in messages:
            if isinstance(msg, OllamaMessage):
                formatted.append(msg.to_dict())
            else:
                formatted.append(msg)

        payload: Dict[str, Any] = {
            "model": model or self._config.model or "llama3",
            "messages": formatted,
            "stream": stream,
        }
        if format:
            payload["format"] = format
        payload["options"] = {
            "num_ctx": self._config.num_ctx,
            "temperature": kwargs.pop("temperature", self._config.temperature),
            "top_p": kwargs.pop("top_p", self._config.top_p),
            **kwargs.pop("options", {}),
        }
        payload.update(kwargs)

        response = self._retry_request("POST", "/api/chat", json=payload)
        data = response.json()
        return ChatResponse(
            content=data.get("message", {}).get("content", ""),
            model=data.get("model", ""),
            done=data.get("done", True),
            total_duration=data.get("total_duration", 0),
            eval_count=data.get("eval_count", 0),
            prompt_eval_count=data.get("prompt_eval_count", 0),
        )

    async def async_chat(
        self,
        messages: List[Union[OllamaMessage, Dict[str, str]]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """异步聊天补全。"""
        formatted: List[Dict[str, str]] = []
        for msg in messages:
            if isinstance(msg, OllamaMessage):
                formatted.append(msg.to_dict())
            else:
                formatted.append(msg)

        payload: Dict[str, Any] = {
            "model": model or self._config.model or "llama3",
            "messages": formatted,
            "stream": False,
        }
        payload["options"] = {
            "num_ctx": self._config.num_ctx,
            "temperature": kwargs.pop("temperature", self._config.temperature),
            "top_p": kwargs.pop("top_p", self._config.top_p),
            **kwargs.pop("options", {}),
        }
        payload.update(kwargs)

        response = await self._async_retry_request("POST", "/api/chat", json=payload)
        data = response.json()
        return ChatResponse(
            content=data.get("message", {}).get("content", ""),
            model=data.get("model", ""),
            done=data.get("done", True),
            total_duration=data.get("total_duration", 0),
            eval_count=data.get("eval_count", 0),
            prompt_eval_count=data.get("prompt_eval_count", 0),
        )

    def stream_chat(
        self,
        messages: List[Union[OllamaMessage, Dict[str, str]]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """同步流式聊天补全。"""
        formatted: List[Dict[str, str]] = []
        for msg in messages:
            if isinstance(msg, OllamaMessage):
                formatted.append(msg.to_dict())
            else:
                formatted.append(msg)

        payload: Dict[str, Any] = {
            "model": model or self._config.model or "llama3",
            "messages": formatted,
            "stream": True,
        }
        payload["options"] = {
            "num_ctx": self._config.num_ctx,
            "temperature": kwargs.pop("temperature", self._config.temperature),
            **kwargs.pop("options", {}),
        }
        payload.update(kwargs)

        client = self._get_client()
        with client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    async def async_stream_chat(
        self,
        messages: List[Union[OllamaMessage, Dict[str, str]]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """异步流式聊天补全。"""
        formatted: List[Dict[str, str]] = []
        for msg in messages:
            if isinstance(msg, OllamaMessage):
                formatted.append(msg.to_dict())
            else:
                formatted.append(msg)

        payload: Dict[str, Any] = {
            "model": model or self._config.model or "llama3",
            "messages": formatted,
            "stream": True,
        }
        payload["options"] = {
            "num_ctx": self._config.num_ctx,
            "temperature": kwargs.pop("temperature", self._config.temperature),
            **kwargs.pop("options", {}),
        }
        payload.update(kwargs)

        client = self._get_async_client()
        async with client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> GenerateResponse:
        """同步文本生成。"""
        payload: Dict[str, Any] = {
            "model": model or self._config.model or "llama3",
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        payload["options"] = {
            "num_ctx": self._config.num_ctx,
            "temperature": kwargs.pop("temperature", self._config.temperature),
            **kwargs.pop("options", {}),
        }
        payload.update(kwargs)

        response = self._retry_request("POST", "/api/generate", json=payload)
        data = response.json()
        return GenerateResponse(
            content=data.get("response", ""),
            model=data.get("model", ""),
            done=data.get("done", True),
            total_duration=data.get("total_duration", 0),
            eval_count=data.get("eval_count", 0),
            prompt_eval_count=data.get("prompt_eval_count", 0),
        )

    async def async_generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> GenerateResponse:
        """异步文本生成。"""
        payload: Dict[str, Any] = {
            "model": model or self._config.model or "llama3",
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        payload["options"] = {
            "num_ctx": self._config.num_ctx,
            "temperature": kwargs.pop("temperature", self._config.temperature),
            **kwargs.pop("options", {}),
        }
        payload.update(kwargs)

        response = await self._async_retry_request("POST", "/api/generate", json=payload)
        data = response.json()
        return GenerateResponse(
            content=data.get("response", ""),
            model=data.get("model", ""),
            done=data.get("done", True),
            total_duration=data.get("total_duration", 0),
            eval_count=data.get("eval_count", 0),
            prompt_eval_count=data.get("prompt_eval_count", 0),
        )

    def embed(
        self,
        input_text: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """生成嵌入向量。"""
        payload: Dict[str, Any] = {
            "model": model or self._config.model or "llama3",
            "input": input_text if isinstance(input_text, list) else [input_text],
        }
        payload.update(kwargs)
        response = self._retry_request("POST", "/api/embed", json=payload)
        data = response.json()
        return EmbeddingResponse(
            embeddings=data.get("embeddings", []),
            model=data.get("model", ""),
        )

    async def async_embed(
        self,
        input_text: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """异步生成嵌入向量。"""
        payload: Dict[str, Any] = {
            "model": model or self._config.model or "llama3",
            "input": input_text if isinstance(input_text, list) else [input_text],
        }
        payload.update(kwargs)
        response = await self._async_retry_request("POST", "/api/embed", json=payload)
        data = response.json()
        return EmbeddingResponse(
            embeddings=data.get("embeddings", []),
            model=data.get("model", ""),
        )

    def list_models(self) -> List[ModelInfo]:
        """列出本地可用模型。"""
        response = self._retry_request("GET", "/api/tags")
        data = response.json()
        models: List[ModelInfo] = []
        for m in data.get("models", []):
            info = m.get("details", {})
            models.append(ModelInfo(
                name=m.get("name", ""),
                size=m.get("size", 0),
                family=info.get("family", ""),
                parameter_size=info.get("parameter_size", ""),
                quantization_level=info.get("quantization_level", ""),
            ))
        return models

    async def async_list_models(self) -> List[ModelInfo]:
        """异步列出本地可用模型。"""
        response = await self._async_retry_request("GET", "/api/tags")
        data = response.json()
        models: List[ModelInfo] = []
        for m in data.get("models", []):
            info = m.get("details", {})
            models.append(ModelInfo(
                name=m.get("name", ""),
                size=m.get("size", 0),
                family=info.get("family", ""),
                parameter_size=info.get("parameter_size", ""),
                quantization_level=info.get("quantization_level", ""),
            ))
        return models

    def pull_model(self, model_name: str, stream: bool = False) -> Dict[str, Any]:
        """拉取远程模型。

        Args:
            model_name: 模型名称（如 'llama3:8b'）。
            stream: 是否流式拉取。

        Returns:
            拉取结果。
        """
        payload: Dict[str, Any] = {"name": model_name, "stream": stream}
        response = self._retry_request("POST", "/api/pull", json=payload)
        return response.json()

    def delete_model(self, model_name: str) -> Dict[str, Any]:
        """删除本地模型。

        Args:
            model_name: 模型名称。

        Returns:
            删除结果。
        """
        payload: Dict[str, str] = {"name": model_name}
        response = self._retry_request("DELETE", "/api/delete", json=payload)
        return response.json()

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

    def __enter__(self) -> "OllamaClient":
        self._get_client()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> "OllamaClient":
        self._get_async_client()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.async_close()
