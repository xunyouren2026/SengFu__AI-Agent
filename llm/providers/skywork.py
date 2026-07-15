"""
昆仑天工 (Skywork) LLM Provider

昆仑天工是昆仑万维推出的AI大模型，提供强大的中文理解和生成能力。

支持的模型:
- skywork-chat-13b: 13B参数对话模型
- skywork-chat-70b: 70B参数对话模型

API文档: https://openapi.tiangong.cn/

Author: AGI Team
Version: 1.0.0
"""

import time
import logging
import json
import asyncio
import base64
from typing import (
    Dict, List, Optional, Any, AsyncIterator, Set, Union
)
from dataclasses import dataclass
from enum import Enum
import aiohttp

from .base import (
    BaseLLMProvider, LLMConfig, LLMResponse,
    ModelCapability, LLMError
)

logger = logging.getLogger(__name__)


class SkyworkModelType(Enum):
    """天工模型类型"""
    CHAT = "chat"
    VISION = "vision"


@dataclass
class ImageContent:
    """图片内容"""
    url: Optional[str] = None
    base64: Optional[str] = None
    mime_type: str = "image/jpeg"


class SkyworkProvider(BaseLLMProvider):
    """
    昆仑天工 (Skywork) LLM Provider

    天工大模型特点:
        - 强大的中文理解和生成能力
        - 支持多模态（文本+图片）
        - 支持长上下文
        - 支持流式输出
        - 支持Function Calling

    适用场景:
        - 智能对话系统
        - 多模态内容理解
        - 文本生成和创作
        - 知识问答
        - 图像描述和分析

    Example:
        ```python
        provider = SkyworkProvider(LLMConfig(
            model_id="skywork-chat-70b",
            api_key="your_api_key"
        ))

        # 普通对话
        response = await provider.chat_completion([
            {"role": "user", "content": "你好"}
        ])

        # 多模态理解
        response = await provider.vision_completion(
            messages=[{"role": "user", "content": "描述这张图片"}],
            image_base64="base64_encoded_image"
        )
        ```
    """

    PROVIDER_NAME = "skywork"
    SUPPORTED_MODELS = {
        "skywork-chat-13b",
        "skywork-chat-70b",
        "skywork-vision",
    }
    DEFAULT_MODEL = "skywork-chat-70b"

    # API配置
    API_BASE = "https://api.skywork.ai/v1"
    CHAT_ENDPOINT = "/chat/completions"

    # 模型特性配置
    MODEL_CONFIGS = {
        "skywork-chat-13b": {
            "max_tokens": 4096,
            "context_window": 8192,
            "supports_vision": False,
            "supports_function_calling": True,
            "supports_streaming": True,
            "model_type": SkyworkModelType.CHAT,
            "description": "13B参数对话模型，性价比高",
        },
        "skywork-chat-70b": {
            "max_tokens": 4096,
            "context_window": 32768,
            "supports_vision": False,
            "supports_function_calling": True,
            "supports_streaming": True,
            "model_type": SkyworkModelType.CHAT,
            "description": "70B参数对话模型，能力更强",
        },
        "skywork-vision": {
            "max_tokens": 4096,
            "context_window": 8192,
            "supports_vision": True,
            "supports_function_calling": True,
            "supports_streaming": True,
            "model_type": SkyworkModelType.VISION,
            "description": "多模态模型，支持图片理解",
        },
    }

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        初始化天工Provider。

        Args:
            config: LLM配置，包含api_key等参数
        """
        super().__init__(config)
        self._api_base = self.API_BASE
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_remaining = 1000
        self._rate_limit_reset = 0
        self._request_count = 0
        self._last_request_time = 0.0

    def _setup_client(self) -> None:
        """设置HTTP客户端会话"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=120),
                headers=self._get_headers()
            )

    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        api_key = self._config.api_key if self._config else None
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _get_model_config(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """获取模型配置"""
        model = model_id or self.model_id
        return self.MODEL_CONFIGS.get(model, self.MODEL_CONFIGS[self.DEFAULT_MODEL])

    def get_capabilities(self) -> Set[ModelCapability]:
        """获取当前模型支持的能力"""
        model_config = self._get_model_config()
        capabilities: Set[ModelCapability] = {
            ModelCapability.STREAMING,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.JSON_MODE,
        }

        if model_config.get("supports_vision"):
            capabilities.add(ModelCapability.VISION)

        return capabilities

    def _apply_rate_limit(self) -> None:
        """应用速率限制"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time

        # 最小请求间隔 0.1秒
        min_interval = 0.1
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)

        self._last_request_time = time.time()

    def _convert_messages(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        转换消息格式。

        支持多模态内容，处理文本和图片混合的消息。
        """
        converted = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role not in ["system", "user", "assistant", "tool"]:
                role = "user"

            converted_msg: Dict[str, Any] = {"role": role}

            # 处理多模态内容
            if isinstance(content, list):
                # 多模态消息
                converted_content = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            converted_content.append({
                                "type": "text",
                                "text": item.get("text", "")
                            })
                        elif item.get("type") == "image_url":
                            image_url = item.get("image_url", {})
                            url = image_url.get("url", "")
                            converted_content.append({
                                "type": "image_url",
                                "image_url": {"url": url}
                            })
                converted_msg["content"] = converted_content
            else:
                converted_msg["content"] = str(content) if content is not None else ""

            # 处理tool_calls
            if "tool_calls" in msg:
                converted_msg["tool_calls"] = msg["tool_calls"]

            if "tool_call_id" in msg:
                converted_msg["tool_call_id"] = msg["tool_call_id"]

            converted.append(converted_msg)

        return converted

    def _build_payload(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> Dict[str, Any]:
        """构建请求体"""
        model_config = self._get_model_config(config.model_id)

        chat_messages = self._convert_messages(messages)

        payload: Dict[str, Any] = {
            "model": config.model_id,
            "messages": chat_messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }

        if config.max_tokens:
            payload["max_tokens"] = min(config.max_tokens, model_config["max_tokens"])

        if config.stop:
            payload["stop"] = config.stop

        if config.seed is not None:
            payload["seed"] = config.seed

        # 处理extra_body
        if config.extra_body:
            if "tools" in config.extra_body:
                payload["tools"] = config.extra_body["tools"]
            if "tool_choice" in config.extra_body:
                payload["tool_choice"] = config.extra_body["tool_choice"]
            if "response_format" in config.extra_body:
                payload["response_format"] = config.extra_body["response_format"]

        return payload

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
            LLMResponse
        """
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

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}{self.CHAT_ENDPOINT}",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=config.timeout)
                ) as response:
                    self._request_count += 1

                    # 更新速率限制信息
                    self._rate_limit_remaining = int(
                        response.headers.get("X-RateLimit-Remaining", 1000)
                    )

                    if response.status != 200:
                        error_text = await response.text()
                        try:
                            error_data = json.loads(error_text)
                            error_msg = error_data.get("error", {}).get("message", error_text)
                            error_code = error_data.get("error", {}).get("code")
                        except:
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

                    # 解析响应
                    choice = result["choices"][0]
                    message = choice.get("message", {})
                    content = message.get("content", "")
                    finish_reason = choice.get("finish_reason")
                    usage = result.get("usage")

                    # 处理tool_calls
                    metadata = {
                        "provider": self.PROVIDER_NAME,
                        "model": config.model_id,
                        "rate_limit_remaining": self._rate_limit_remaining,
                    }

                    tool_calls = message.get("tool_calls")
                    if tool_calls:
                        metadata["tool_calls"] = tool_calls

                    return LLMResponse(
                        content=content,
                        model_id=config.model_id,
                        usage=usage,
                        finish_reason=finish_reason,
                        latency_ms=(time.time() - start_time) * 1000,
                        metadata=metadata
                    )

        except aiohttp.ClientError as e:
            logger.error(f"Skywork HTTP error: {e}")
            return LLMResponse(
                content="",
                model_id=config.model_id,
                error=f"HTTP error: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            logger.error(f"Skywork error: {e}")
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
    ) -> AsyncIterator[str]:
        """异步流式生成"""
        self._apply_rate_limit()

        api_key = config.api_key or (self._config.api_key if self._config else None)
        base_url = config.base_url or self._api_base

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = self._build_payload(messages, config)
        payload["stream"] = True

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}{self.CHAT_ENDPOINT}",
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
                            delta = chunk_data["choices"][0].get("delta", {})

                            if delta.get("content"):
                                yield delta["content"]

                            if delta.get("tool_calls"):
                                yield json.dumps({"tool_calls": delta["tool_calls"]})

                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

        except Exception as e:
            logger.error(f"Skywork stream error: {e}")
            yield f"Error: {str(e)}"

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        发送聊天完成请求。

        Args:
            messages: 消息列表
            model: 模型ID
            temperature: 温度参数
            max_tokens: 最大token数
            tools: 工具/函数定义列表
            **kwargs: 其他参数

        Returns:
            LLM响应对象
        """
        extra_body = kwargs.get("extra_body", {})
        if tools:
            extra_body["tools"] = tools

        config = LLMConfig(
            model_id=model or self.model_id,
            api_key=self._config.api_key if self._config else None,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body if extra_body else None,
            **{k: v for k, v in kwargs.items() if k != "extra_body"}
        )
        return await self._async_generate(messages, config)

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        """流式聊天完成"""
        config = LLMConfig(
            model_id=model or self.model_id,
            api_key=self._config.api_key if self._config else None,
            temperature=temperature,
            **kwargs
        )
        async for chunk in self._async_stream_generate(messages, config):
            yield chunk

    async def vision_completion(
        self,
        messages: List[Dict[str, Any]],
        image_base64: str,
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        视觉理解完成。

        Args:
            messages: 消息列表
            image_base64: 图片base64编码
            model: 模型ID，默认使用skywork-vision
            prompt: 图片相关的提示词
            **kwargs: 其他参数

        Returns:
            LLM响应对象
        """
        vision_model = model or "skywork-vision"
        model_config = self._get_model_config(vision_model)

        if not model_config.get("supports_vision"):
            raise ValueError(f"Model {vision_model} does not support vision")

        # 构建视觉消息
        vision_messages = messages.copy()

        # 构建多模态内容
        image_content = [
            {"type": "text", "text": prompt or "描述这张图片"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            }
        ]

        vision_messages.append({
            "role": "user",
            "content": image_content
        })

        config = LLMConfig(
            model_id=vision_model,
            api_key=self._config.api_key if self._config else None,
            **kwargs
        )

        return await self._async_generate(vision_messages, config)

    async def analyze_image(
        self,
        image_base64: str,
        prompt: str = "详细描述这张图片的内容",
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        分析图片内容。

        Args:
            image_base64: 图片base64编码
            prompt: 分析提示词
            model: 模型ID
            **kwargs: 其他参数

        Returns:
            LLM响应对象
        """
        messages = [{"role": "user", "content": prompt}]
        return await self.vision_completion(
            messages=messages,
            image_base64=image_base64,
            model=model,
            prompt=prompt,
            **kwargs
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取Provider统计信息"""
        return {
            "provider": self.PROVIDER_NAME,
            "model_id": self.model_id,
            "initialized": self._initialized,
            "api_base": self._api_base,
            "request_count": self._request_count,
            "rate_limit_remaining": self._rate_limit_remaining,
            "supported_models": list(self.SUPPORTED_MODELS),
        }

    async def close(self) -> None:
        """关闭Provider"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None


# ==================== CLI测试代码 ====================

async def test_skywork_chat():
    """测试天工聊天"""
    import os

    print("=" * 60)
    print("天工 (Skywork) Provider 测试 - 聊天")
    print("=" * 60)

    api_key = os.environ.get("SKYWORK_API_KEY", "your_api_key_here")

    config = LLMConfig(
        model_id="skywork-chat-70b",
        api_key=api_key,
        temperature=0.7,
        max_tokens=2048
    )
    provider = SkyworkProvider(config)
    provider.initialize()

    print(f"\nProvider: {provider.provider_name}")
    print(f"Model: {provider.model_id}")
    print(f"Capabilities: {provider.get_capabilities()}")
    print(f"Stats: {provider.get_stats()}")

    # 测试1: 简单对话
    print("\n" + "-" * 40)
    print("测试1: 简单对话")
    print("-" * 40)

    messages = [
        {"role": "system", "content": "你是一个 helpful AI assistant。"},
        {"role": "user", "content": "你好！请用中文介绍一下自己。"}
    ]

    try:
        response = await provider.chat_completion(messages)
        print(f"Response: {response.content}")
        print(f"Latency: {response.latency_ms:.2f}ms")
        if response.usage:
            print(f"Tokens: {response.usage}")
    except Exception as e:
        print(f"Error: {e}")

    # 测试2: 流式输出
    print("\n" + "-" * 40)
    print("测试2: 流式输出")
    print("-" * 40)

    messages = [
        {"role": "user", "content": "请用3句话描述成都。"}
    ]

    try:
        print("Response: ", end="", flush=True)
        async for chunk in provider.stream_chat(messages):
            print(chunk, end="", flush=True)
        print()
    except Exception as e:
        print(f"Error: {e}")

    await provider.close()


async def test_skywork_vision():
    """测试天工视觉理解"""
    import os

    print("\n" + "=" * 60)
    print("天工 (Skywork) Provider 测试 - 视觉理解")
    print("=" * 60)

    api_key = os.environ.get("SKYWORK_API_KEY", "your_api_key_here")

    config = LLMConfig(
        model_id="skywork-vision",
        api_key=api_key,
    )
    provider = SkyworkProvider(config)
    provider.initialize()

    print(f"\nModel: {provider.model_id}")
    print(f"Supports vision: {provider._get_model_config().get('supports_vision')}")

    # 使用示例图片
    sample_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    print("\n" + "-" * 40)
    print("测试: 视觉理解 (示例)")
    print("-" * 40)

    try:
        response = await provider.vision_completion(
            messages=[],
            image_base64=sample_image_b64,
            prompt="描述这张图片"
        )
        print(f"Response: {response.content}")
    except Exception as e:
        print(f"Error (expected with sample image): {e}")

    # 测试analyze_image方法
    print("\n" + "-" * 40)
    print("测试: analyze_image方法")
    print("-" * 40)

    try:
        response = await provider.analyze_image(
            image_base64=sample_image_b64,
            prompt="这张图片里有什么？"
        )
        print(f"Response: {response.content}")
    except Exception as e:
        print(f"Error (expected with sample image): {e}")

    await provider.close()


async def test_skywork_models():
    """测试不同模型"""
    import os

    print("\n" + "=" * 60)
    print("天工 (Skywork) Provider 测试 - 不同模型")
    print("=" * 60)

    api_key = os.environ.get("SKYWORK_API_KEY", "your_api_key_here")

    models = ["skywork-chat-13b", "skywork-chat-70b"]

    for model_id in models:
        print(f"\n--- 测试模型: {model_id} ---")

        config = LLMConfig(
            model_id=model_id,
            api_key=api_key,
            temperature=0.7,
        )
        provider = SkyworkProvider(config)
        provider.initialize()

        model_config = provider._get_model_config()
        print(f"Description: {model_config.get('description')}")
        print(f"Context window: {model_config.get('context_window')}")
        print(f"Max tokens: {model_config.get('max_tokens')}")

        messages = [
            {"role": "user", "content": "你好"}
        ]

        try:
            response = await provider.chat_completion(messages)
            print(f"Response: {response.content[:50]}...")
            print(f"Latency: {response.latency_ms:.2f}ms")
        except Exception as e:
            print(f"Error: {e}")

        await provider.close()


async def test_skywork_multimodal():
    """测试天工多模态功能"""
    import os

    print("\n" + "=" * 60)
    print("天工 (Skywork) Provider 测试 - 多模态消息")
    print("=" * 60)

    api_key = os.environ.get("SKYWORK_API_KEY", "your_api_key_here")

    config = LLMConfig(
        model_id="skywork-vision",
        api_key=api_key,
    )
    provider = SkyworkProvider(config)
    provider.initialize()

    print(f"\nModel: {provider.model_id}")

    # 测试多模态消息格式
    sample_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    print("\n" + "-" * 40)
    print("测试: 多模态消息格式")
    print("-" * 40)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这张图片展示了什么？"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{sample_image_b64}"
                    }
                }
            ]
        }
    ]

    try:
        response = await provider.chat_completion(messages)
        print(f"Response: {response.content}")
    except Exception as e:
        print(f"Error (expected with sample image): {e}")

    await provider.close()


if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_skywork_chat())
    asyncio.run(test_skywork_vision())
    asyncio.run(test_skywork_models())
    asyncio.run(test_skywork_multimodal())
