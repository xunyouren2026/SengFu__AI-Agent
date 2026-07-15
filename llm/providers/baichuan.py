"""
百川智能 (Baichuan) Provider

百川智能Baichuan系列大模型的适配器。

支持的模型:
- Baichuan4
- Baichuan3-Turbo
- Baichuan3-Turbo-128k
- Baichuan2-Turbo

API文档: https://platform.baichuan-ai.com/docs/api-reference

Author: AGI Team
Version: 1.0.0
"""

import time
import logging
import json
import asyncio
from dataclasses import dataclass
from typing import (
    Dict, List, Optional, Any, AsyncIterator, Set
)
from .base import (
    BaseLLMProvider, LLMConfig, LLMResponse,
    ModelCapability, LLMError
)

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BaichuanConfig(LLMConfig):
    """
    百川智能配置

    Attributes:
        enable_search: 是否启用搜索增强
        search_top_k: 搜索结果数量
        search_enabled: 搜索增强开关 (别名)
        tools: 函数调用工具定义列表
        tool_choice: 工具选择策略
    """
    enable_search: bool = False
    search_top_k: int = 5
    search_enabled: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[str] = None


class BaichuanProvider(BaseLLMProvider):
    """
    百川智能Baichuan系列Provider

    百川智能开放平台提供的Baichuan系列大语言模型适配器。
    API兼容OpenAI格式, 支持超长上下文和搜索增强。

    Features:
        - 支持Baichuan4/3-Turbo/3-Turbo-128k/2-Turbo全系列
        - 兼容OpenAI API格式
        - 支持超长上下文 (128K tokens)
        - 支持搜索增强 (enable_search)
        - 支持函数调用 (Function Calling)
        - 支持JSON Mode
        - 标准OpenAI错误格式处理

    Example:
        ```python
        provider = BaichuanProvider(BaichuanConfig(
            model_id="Baichuan4",
            api_key="your_api_key",
        ))

        response = await provider.generate([
            {"role": "user", "content": "你好"}
        ])
        print(response.content)
        ```
    """

    PROVIDER_NAME = "baichuan"
    SUPPORTED_MODELS = {
        "Baichuan4",
        "Baichuan3-Turbo",
        "Baichuan3-Turbo-128k",
        "Baichuan2-Turbo",
        "Baichuan2-Turbo-192k",
        "Baichuan3",
        "Baichuan-Text-Embedding",
    }
    DEFAULT_MODEL = "Baichuan4"

    _API_BASE = "https://api.baichuan-ai.com/v1"

    # 模型上下文窗口大小
    _MODEL_CONTEXT_SIZES: Dict[str, int] = {
        "Baichuan4": 32768,
        "Baichuan3-Turbo": 32768,
        "Baichuan3-Turbo-128k": 131072,
        "Baichuan3": 32768,
        "Baichuan2-Turbo": 32768,
        "Baichuan2-Turbo-192k": 196608,
    }

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        初始化百川智能Provider。

        Args:
            config: 百川配置, 推荐使用BaichuanConfig
        """
        super().__init__(config)
        self._client: Optional[httpx.AsyncClient] = None

    def get_capabilities(self) -> Set[ModelCapability]:
        """获取模型能力"""
        capabilities = {
            ModelCapability.STREAMING,
            ModelCapability.JSON_MODE,
            ModelCapability.FUNCTION_CALLING,
        }
        return capabilities

    def _setup_client(self) -> None:
        """设置HTTP客户端"""
        if HTTPX_AVAILABLE:
            timeout = self._config.timeout if self._config else 60.0
            self._client = httpx.AsyncClient(timeout=timeout)

    def _get_context_size(self, model_id: str) -> int:
        """
        获取模型的上下文窗口大小。

        Args:
            model_id: 模型ID

        Returns:
            上下文窗口大小 (token数)
        """
        return self._MODEL_CONTEXT_SIZES.get(model_id, 32768)

    def _convert_messages(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        转换消息格式为百川API要求的格式。

        百川API兼容OpenAI消息格式, 支持system/user/assistant/tool角色。

        Args:
            messages: 标准消息列表

        Returns:
            转换后的消息列表
        """
        converted = []

        for msg in messages:
            role = msg.get("role", "")

            if role in ("system", "user", "assistant"):
                converted_msg: Dict[str, Any] = {
                    "role": role,
                    "content": msg.get("content", ""),
                }

                # 处理tool_calls
                if msg.get("tool_calls"):
                    converted_msg["tool_calls"] = msg["tool_calls"]

                # 处理tool_call_id
                if msg.get("tool_call_id"):
                    converted_msg["tool_call_id"] = msg["tool_call_id"]

                # 处理name字段
                if msg.get("name"):
                    converted_msg["name"] = msg["name"]

                converted.append(converted_msg)

            elif role == "tool":
                converted.append({
                    "role": "tool",
                    "content": msg.get("content", ""),
                    "tool_call_id": msg.get("tool_call_id", ""),
                })
            else:
                logger.warning(f"Unsupported message role: {role}, skipping")

        return converted

    def _build_headers(self, config: LLMConfig) -> Dict[str, str]:
        """
        构建请求头。

        Args:
            config: LLM配置

        Returns:
            请求头字典
        """
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }

        # 合并额外请求头
        if config.extra_headers:
            headers.update(config.extra_headers)

        return headers

    def _parse_error_response(
        self,
        response: httpx.Response,
        start_time: float,
        model_id: str
    ) -> LLMResponse:
        """
        解析错误响应。

        百川API遵循OpenAI错误格式:
        {"error": {"message": "...", "type": "...", "code": "..."}}

        Args:
            response: HTTP响应对象
            start_time: 请求开始时间
            model_id: 模型ID

        Returns:
            错误LLMResponse
        """
        try:
            error_data = response.json()
            error_obj = error_data.get("error", {})
            error_message = error_obj.get("message", response.text)
            error_code = error_obj.get("code")
            error_type = error_obj.get("type")
        except (json.JSONDecodeError, ValueError):
            error_message = response.text
            error_code = None
            error_type = None

        error_detail = f"HTTP {response.status_code}"
        if error_code:
            error_detail += f" [{error_code}]"
        if error_type:
            error_detail += f" ({error_type})"
        error_detail += f": {error_message}"

        return LLMResponse(
            content="",
            model_id=model_id,
            error=error_detail,
            latency_ms=(time.time() - start_time) * 1000,
        )

    async def _async_generate(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> LLMResponse:
        """
        异步生成响应。

        Args:
            messages: 消息列表
            config: 生成配置

        Returns:
            LLM响应
        """
        start_time = time.time()

        if not HTTPX_AVAILABLE:
            return LLMResponse(
                content="",
                model_id=config.model_id,
                error="httpx not available",
                latency_ms=(time.time() - start_time) * 1000,
            )

        headers = self._build_headers(config)
        chat_messages = self._convert_messages(messages)

        payload: Dict[str, Any] = {
            "model": config.model_id,
            "messages": chat_messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }

        if config.max_tokens is not None:
            payload["max_tokens"] = config.max_tokens

        if config.stop:
            payload["stop"] = config.stop

        if config.seed is not None:
            payload["seed"] = config.seed

        if config.response_format:
            payload["response_format"] = config.response_format

        # 百川特色功能: 搜索增强
        if isinstance(config, BaichuanConfig):
            if config.enable_search or config.search_enabled:
                payload["enable_search"] = True
                payload["search_top_k"] = config.search_top_k

            # 函数调用支持
            if config.tools:
                payload["tools"] = config.tools
            if config.tool_choice:
                payload["tool_choice"] = config.tool_choice

        # 合并额外body参数
        if config.extra_body:
            payload.update(config.extra_body)

        # 带重试的请求
        last_error = None
        for attempt in range(config.max_retries):
            try:
                async with httpx.AsyncClient(timeout=config.timeout) as client:
                    response = await client.post(
                        f"{self._API_BASE}/chat/completions",
                        headers=headers,
                        json=payload,
                    )

                    # 限流重试
                    if response.status_code == 429:
                        wait_time = 2 ** attempt
                        logger.warning(
                            f"Rate limited, retrying in {wait_time}s..."
                        )
                        await asyncio.sleep(wait_time)
                        continue

                    # 服务端错误重试
                    if response.status_code >= 500:
                        logger.warning(
                            f"Server error {response.status_code}, "
                            f"retrying (attempt {attempt + 1}/{config.max_retries})"
                        )
                        if attempt < config.max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                        continue

                    if response.status_code != 200:
                        return self._parse_error_response(
                            response, start_time, config.model_id
                        )

                    result = response.json()

                    # 解析响应
                    choices = result.get("choices", [])
                    if not choices:
                        return LLMResponse(
                            content="",
                            model_id=config.model_id,
                            error="No choices in response",
                            latency_ms=(time.time() - start_time) * 1000,
                        )

                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                    finish_reason = choices[0].get("finish_reason")

                    usage = result.get("usage")
                    normalized_usage = None
                    if usage:
                        normalized_usage = {
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get(
                                "completion_tokens", 0
                            ),
                            "total_tokens": usage.get("total_tokens", 0),
                        }

                    metadata: Dict[str, Any] = {
                        "provider": "baichuan",
                        "model": config.model_id,
                        "id": result.get("id", ""),
                        "context_size": self._get_context_size(config.model_id),
                    }

                    # 搜索增强结果
                    search_info = result.get("search_info")
                    if search_info:
                        metadata["search_info"] = search_info

                    # 函数调用结果
                    if message.get("tool_calls"):
                        metadata["tool_calls"] = message["tool_calls"]

                    return LLMResponse(
                        content=content,
                        model_id=config.model_id,
                        usage=normalized_usage,
                        finish_reason=finish_reason,
                        latency_ms=(time.time() - start_time) * 1000,
                        metadata=metadata,
                    )

            except httpx.TimeoutException as e:
                last_error = f"Request timeout: {e}"
                logger.warning(
                    f"Request timeout (attempt {attempt + 1}/{config.max_retries})"
                )
                if attempt < config.max_retries - 1:
                    await asyncio.sleep(1)
                continue
            except Exception as e:
                last_error = str(e)
                break

        return LLMResponse(
            content="",
            model_id=config.model_id,
            error=last_error or "Unknown error",
            latency_ms=(time.time() - start_time) * 1000,
        )

    async def _async_stream_generate(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> AsyncIterator[str]:
        """
        流式生成响应。

        百川API流式输出兼容OpenAI SSE格式。

        Args:
            messages: 消息列表
            config: 生成配置

        Yields:
            响应文本片段
        """
        if not HTTPX_AVAILABLE:
            yield "Error: httpx not available"
            return

        headers = self._build_headers(config)
        chat_messages = self._convert_messages(messages)

        payload: Dict[str, Any] = {
            "model": config.model_id,
            "messages": chat_messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stream": True,
        }

        if config.max_tokens is not None:
            payload["max_tokens"] = config.max_tokens

        # 搜索增强
        if isinstance(config, BaichuanConfig):
            if config.enable_search or config.search_enabled:
                payload["enable_search"] = True
                payload["search_top_k"] = config.search_top_k

            if config.tools:
                payload["tools"] = config.tools
                payload["stream_options"] = {"include_usage": True}

        try:
            async with httpx.AsyncClient(timeout=config.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self._API_BASE}/chat/completions",
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        yield f"Error: HTTP {response.status_code}: {error_text.decode()}"
                        return

                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue

                        if line.startswith("data: "):
                            data_str = line[6:]
                        else:
                            data_str = line

                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            chunk_data = json.loads(data_str)
                            choices = chunk_data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue

        except httpx.TimeoutException as e:
            yield f"Error: Stream timeout: {e}"
        except Exception as e:
            yield f"Error: {e}"

    async def embeddings(
        self,
        texts: List[str],
        model: str = "Baichuan-Text-Embedding",
        **kwargs
    ) -> List[List[float]]:
        """
        获取文本向量嵌入。

        Args:
            texts: 待嵌入的文本列表
            model: 嵌入模型名称
            **kwargs: 额外参数

        Returns:
            向量列表

        Raises:
            LLMError: 嵌入请求失败
        """
        if not HTTPX_AVAILABLE:
            raise LLMError("httpx not available", model_id=model)

        api_key = self._config.api_key if self._config else None
        if not api_key:
            raise LLMError("api_key is required for embeddings", model_id=model)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "input": texts,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self._API_BASE}/embeddings",
                headers=headers,
                json=payload,
            )

            if response.status_code != 200:
                raise LLMError(
                    f"Embedding request failed: HTTP {response.status_code} - {response.text}",
                    model_id=model,
                    status_code=response.status_code,
                )

            result = response.json()
            embeddings = [item["embedding"] for item in result.get("data", [])]
            return embeddings

    async def token_count(
        self,
        text: str,
        model: Optional[str] = None
    ) -> int:
        """
        估算文本的token数量。

        百川API没有专门的token计数接口,
        这里使用简单的字符级估算。

        Args:
            text: 待计数的文本
            model: 模型名称 (未使用)

        Returns:
            估算的token数量
        """
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        estimated_tokens = int(chinese_chars / 1.5 + other_chars / 4)
        return max(1, estimated_tokens)

    def get_stats(self) -> Dict[str, Any]:
        """获取Provider统计信息"""
        return {
            "provider": self.PROVIDER_NAME,
            "model_id": self.model_id,
            "initialized": self._initialized,
            "api_base": self._API_BASE,
            "context_size": self._get_context_size(self.model_id),
            "capabilities": [c.name for c in self.get_capabilities()],
            "supported_models": sorted(self.SUPPORTED_MODELS),
        }
