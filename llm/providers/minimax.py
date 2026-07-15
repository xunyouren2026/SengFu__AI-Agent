"""
MiniMax LLM Provider

MiniMax是国内的AI大模型公司，提供文本生成、聊天和嵌入服务。

支持的模型:
- abab6.5-chat: 标准对话模型
- abab6.5s-chat: 轻量对话模型
- minimax-01: 最新一代模型

API文档: https://api.minimax.chat/

Author: AGI Team
Version: 1.0.0
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import aiohttp

from .base import (
    OpenAICompatibleProvider, LLMConfig, LLMResponse,
    ModelCapability,
)

logger = logging.getLogger(__name__)


class MiniMaxModelType(Enum):
    """MiniMax模型类型"""
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"


@dataclass
class EmbeddingResponse:
    """嵌入响应"""
    embeddings: List[List[float]]
    model: str
    usage: Optional[Dict[str, int]] = None


class MiniMaxProvider(OpenAICompatibleProvider):
    """
    MiniMax LLM Provider

    MiniMax大模型特点:
        - 强大的中文对话能力
        - 支持文本补全和聊天
        - 支持文本嵌入 (Embedding)
        - 支持流式输出
        - 支持Function Calling

    适用场景:
        - 智能对话系统
        - 文本生成和创作
        - 语义搜索和向量检索
        - 知识问答

    Example:
        ```python
        provider = MiniMaxProvider(LLMConfig(
            model_id="abab6.5-chat",
            api_key="your_api_key",
            extra_body={"group_id": "your_group_id"}
        ))

        # 聊天
        response = await provider.chat_completion([
            {"role": "user", "content": "你好"}
        ])

        # 获取嵌入向量
        embeddings = await provider.get_embeddings(["文本1", "文本2"])
        ```
    """

    PROVIDER_NAME = "minimax"
    SUPPORTED_MODELS = {
        "abab6.5-chat",
        "abab6.5s-chat",
        "minimax-01",
        "abab5.5-chat",
        "abab5.5s-chat",
        "embo-01",
    }
    DEFAULT_MODEL = "abab6.5-chat"

    # API配置
    API_BASE = "https://api.minimax.chat/v1"
    CHAT_ENDPOINT = "/text/chatcompletion_v2"
    COMPLETION_ENDPOINT = "/text/completion"
    EMBEDDING_ENDPOINT = "/embeddings"
    RATE_LIMIT_INTERVAL = 0.1

    # 模型特性配置
    MODEL_CONFIGS = {
        "abab6.5-chat": {
            "max_tokens": 8192,
            "context_window": 8192,
            "supports_vision": False,
            "supports_function_calling": True,
            "supports_streaming": True,
            "model_type": MiniMaxModelType.CHAT,
            "description": "标准对话模型，平衡性能和效果",
        },
        "abab6.5s-chat": {
            "max_tokens": 8192,
            "context_window": 8192,
            "supports_vision": False,
            "supports_function_calling": True,
            "supports_streaming": True,
            "model_type": MiniMaxModelType.CHAT,
            "description": "轻量对话模型，响应更快",
        },
        "minimax-01": {
            "max_tokens": 8192,
            "context_window": 8192,
            "supports_vision": False,
            "supports_function_calling": True,
            "supports_streaming": True,
            "model_type": MiniMaxModelType.CHAT,
            "description": "最新一代模型",
        },
        "abab5.5-chat": {
            "max_tokens": 4096,
            "context_window": 4096,
            "supports_vision": False,
            "supports_function_calling": False,
            "supports_streaming": True,
            "model_type": MiniMaxModelType.CHAT,
            "description": "5.5版本对话模型",
        },
        "abab5.5s-chat": {
            "max_tokens": 4096,
            "context_window": 4096,
            "supports_vision": False,
            "supports_function_calling": False,
            "supports_streaming": True,
            "model_type": MiniMaxModelType.CHAT,
            "description": "5.5版本轻量模型",
        },
        "embo-01": {
            "max_tokens": 512,
            "context_window": 512,
            "supports_vision": False,
            "supports_function_calling": False,
            "supports_streaming": False,
            "model_type": MiniMaxModelType.EMBEDDING,
            "description": "文本嵌入模型",
        },
    }

    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        self._group_id: Optional[str] = None

        if config and config.extra_body:
            self._group_id = config.extra_body.get("group_id")

    def get_capabilities(self) -> Set[ModelCapability]:
        """获取当前模型支持的能力"""
        model_config = self._get_model_config()
        capabilities: Set[ModelCapability] = set()

        if model_config.get("supports_streaming"):
            capabilities.add(ModelCapability.STREAMING)

        if model_config.get("supports_function_calling"):
            capabilities.add(ModelCapability.FUNCTION_CALLING)

        if model_config.get("supports_vision"):
            capabilities.add(ModelCapability.VISION)

        capabilities.add(ModelCapability.JSON_MODE)

        return capabilities

    def _build_payload(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> Dict[str, Any]:
        """
        构建聊天请求体。

        MiniMax使用特殊的消息格式，需要区分user和bot角色。
        """
        model_config = self._get_model_config(config.model_id)

        # 转换消息格式
        minimax_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                pass  # MiniMax使用system_settings字段
            elif role == "user":
                minimax_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                minimax_messages.append({"role": "assistant", "content": content})

        payload: Dict[str, Any] = {
            "model": config.model_id,
            "messages": minimax_messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }

        if config.max_tokens:
            payload["max_tokens"] = min(config.max_tokens, model_config["max_tokens"])

        # 添加system prompt
        system_msg = next((m for m in messages if m.get("role") == "system"), None)
        if system_msg:
            payload["system_prompt"] = system_msg.get("content", "")

        # 处理tools
        if config.extra_body and "tools" in config.extra_body:
            payload["tools"] = config.extra_body["tools"]

        if config.extra_body and "tool_choice" in config.extra_body:
            payload["tool_choice"] = config.extra_body["tool_choice"]

        return payload

    def _build_url(self, base_url: str, config: Optional[LLMConfig] = None) -> str:
        """构建请求URL，添加group_id参数"""
        url = f"{base_url}{self.CHAT_ENDPOINT}"
        group_id = self._group_id
        if config and config.extra_body and "group_id" in config.extra_body:
            group_id = config.extra_body["group_id"]
        if group_id:
            url = f"{url}?GroupId={group_id}"
        return url

    async def _async_generate(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> LLMResponse:
        """异步生成响应（MiniMax自定义响应解析）"""
        start_time = time.time()
        self._apply_rate_limit()

        api_key = config.api_key or (self._config.api_key if self._config else None)
        if not api_key:
            return LLMResponse(
                content="",
                model_id=config.model_id,
                error="API key is required",
                latency_ms=(time.time() - start_time) * 1000
            )

        base_url = config.base_url or self._api_base
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = self._build_payload(messages, config)
        url = self._build_url(base_url, config)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=config.timeout)
                ) as response:
                    self._request_count += 1

                    if response.status != 200:
                        error_text = await response.text()
                        try:
                            error_data = json.loads(error_text)
                            error_msg = error_data.get("message", error_text)
                            error_code = error_data.get("code")
                        except Exception:
                            error_msg = error_text
                            error_code = None

                        return LLMResponse(
                            content="",
                            model_id=config.model_id,
                            error=f"HTTP {response.status}: {error_msg}",
                            latency_ms=(time.time() - start_time) * 1000,
                            metadata={
                                "status_code": response.status,
                                "error_code": error_code
                            }
                        )

                    result = await response.json()

                    # MiniMax响应格式解析
                    if result.get("base_resp", {}).get("status_code") != 0:
                        error_msg = result.get("base_resp", {}).get("status_msg", "Unknown error")
                        return LLMResponse(
                            content="",
                            model_id=config.model_id,
                            error=f"MiniMax API error: {error_msg}",
                            latency_ms=(time.time() - start_time) * 1000
                        )

                    choices = result.get("choices", [])
                    if not choices:
                        return LLMResponse(
                            content="",
                            model_id=config.model_id,
                            error="No response from API",
                            latency_ms=(time.time() - start_time) * 1000
                        )

                    choice = choices[0]
                    message = choice.get("message", {})
                    content = message.get("content", "")
                    finish_reason = choice.get("finish_reason")
                    usage = result.get("usage")

                    metadata: Dict[str, Any] = {
                        "provider": self.PROVIDER_NAME,
                        "model": config.model_id,
                    }

                    if "function_call" in message:
                        metadata["function_call"] = message["function_call"]

                    return LLMResponse(
                        content=content,
                        model_id=config.model_id,
                        usage=usage,
                        finish_reason=finish_reason,
                        latency_ms=(time.time() - start_time) * 1000,
                        metadata=metadata
                    )

        except Exception as e:
            logger.error(f"MiniMax error: {e}")
            return LLMResponse(
                content="",
                model_id=config.model_id,
                error=f"Error: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000
            )

    async def _async_stream_generate(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ):
        """异步流式生成（MiniMax自定义delta解析）"""
        self._apply_rate_limit()

        api_key = config.api_key or (self._config.api_key if self._config else None)
        base_url = config.base_url or self._api_base

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = self._build_payload(messages, config)
        payload["stream"] = True
        url = self._build_url(base_url, config)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=config.timeout)
                ) as response:
                    self._request_count += 1

                    if response.status != 200:
                        error_text = await response.text()
                        yield f"Error: HTTP {response.status}: {error_text}"
                        return

                    async for line in response.content:
                        line = line.decode("utf-8").strip()

                        if not line.startswith("data: "):
                            continue

                        data_str = line[6:]

                        if data_str == "[DONE]":
                            break

                        try:
                            chunk_data = json.loads(data_str)
                            choices = chunk_data.get("choices", [])

                            if choices:
                                delta = choices[0].get("delta", "")
                                if delta:
                                    yield delta

                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

        except Exception as e:
            logger.error(f"MiniMax stream error: {e}")
            yield f"Error: {str(e)}"

    async def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        **kwargs
    ) -> EmbeddingResponse:
        """
        获取文本嵌入向量。

        Args:
            texts: 文本列表
            model: 嵌入模型ID，默认使用embo-01
            **kwargs: 其他参数

        Returns:
            EmbeddingResponse 包含嵌入向量
        """
        embedding_model = model or "embo-01"
        start_time = time.time()
        self._apply_rate_limit()

        api_key = self._config.api_key if self._config else None
        if not api_key:
            raise ValueError("API key is required")

        base_url = self._api_base
        group_id = self._group_id

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": embedding_model,
            "input": texts,
        }

        url = f"{base_url}{self.EMBEDDING_ENDPOINT}"
        if group_id:
            url = f"{url}?GroupId={group_id}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    self._request_count += 1

                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")

                    result = await response.json()

                    if result.get("base_resp", {}).get("status_code") != 0:
                        error_msg = result.get("base_resp", {}).get("status_msg", "Unknown error")
                        raise Exception(f"MiniMax API error: {error_msg}")

                    embeddings = result.get("data", [])
                    usage = result.get("usage")

                    return EmbeddingResponse(
                        embeddings=[item.get("embedding", []) for item in embeddings],
                        model=embedding_model,
                        usage=usage
                    )

        except Exception as e:
            logger.error(f"MiniMax embedding error: {e}")
            raise

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> LLMResponse:
        """发送聊天完成请求"""
        extra_body = kwargs.get("extra_body", {})
        if "group_id" in kwargs:
            extra_body["group_id"] = kwargs.pop("group_id")

        config = LLMConfig(
            model_id=model or self.model_id,
            api_key=self._config.api_key if self._config else None,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body if extra_body else None,
            **kwargs
        )
        return await self._async_generate(messages, config)

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ):
        """流式聊天完成"""
        extra_body = kwargs.get("extra_body", {})
        if "group_id" in kwargs:
            extra_body["group_id"] = kwargs.pop("group_id")

        config = LLMConfig(
            model_id=model or self.model_id,
            api_key=self._config.api_key if self._config else None,
            temperature=temperature,
            extra_body=extra_body if extra_body else None,
            **kwargs
        )
        async for chunk in self._async_stream_generate(messages, config):
            yield chunk

    def get_stats(self) -> Dict[str, Any]:
        """获取Provider统计信息"""
        stats = super().get_stats()
        stats["group_id"] = self._group_id
        return stats
