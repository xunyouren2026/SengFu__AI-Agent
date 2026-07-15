"""
字节跳动豆包 (Doubao) LLM Provider

豆包是字节跳动推出的AI大模型，提供强大的中文理解和生成能力。

支持的模型:
- doubao-pro-32k: 专业版，32K上下文
- doubao-lite-32k: 轻量版，32K上下文
- doubao-vision: 视觉理解版

API文档: https://www.volcengine.com/docs/82379

Author: AGI Team
Version: 1.0.0
"""

import time
import logging
import json
import asyncio
from typing import (
    Dict, List, Optional, Any, AsyncIterator, Set, Callable
)
from dataclasses import dataclass
import aiohttp

from .base import (
    BaseLLMProvider, LLMConfig, LLMResponse,
    ModelCapability, LLMError
)

logger = logging.getLogger(__name__)


@dataclass
class FunctionDefinition:
    """函数定义"""
    name: str
    description: str
    parameters: Dict[str, Any]


@dataclass
class FunctionCall:
    """函数调用结果"""
    name: str
    arguments: Dict[str, Any]


class DoubaoProvider(BaseLLMProvider):
    """
    字节跳动豆包 LLM Provider

    豆包大模型特点:
        - 强大的中文理解和生成能力
        - 支持Function Calling (函数调用)
        - 支持长上下文 (32K)
        - 支持流式输出
        - 支持视觉理解 (vision版)

    适用场景:
        - 智能客服和对话系统
        - 内容创作和文案生成
        - 代码辅助和编程
        - 知识问答
        - 多模态理解

    Example:
        ```python
        provider = DoubaoProvider(LLMConfig(
            model_id="doubao-pro-32k",
            api_key="your_api_key"
        ))

        # 普通对话
        response = await provider.chat_completion([
            {"role": "user", "content": "你好"}
        ])

        # 函数调用
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取天气",
                "parameters": {...}
            }
        }]
        response = await provider.chat_completion(messages, tools=tools)
        ```
    """

    PROVIDER_NAME = "doubao"
    SUPPORTED_MODELS = {
        "doubao-pro-32k",
        "doubao-lite-32k",
        "doubao-vision",
        "doubao-pro-4k",
        "doubao-lite-4k",
    }
    DEFAULT_MODEL = "doubao-pro-32k"

    # API配置
    API_BASE = "https://ark.cn-beijing.volces.com/api/v3"
    CHAT_ENDPOINT = "/chat/completions"

    # 模型特性配置
    MODEL_CONFIGS = {
        "doubao-pro-32k": {
            "max_tokens": 4096,
            "context_window": 32768,
            "supports_vision": False,
            "supports_function_calling": True,
            "supports_streaming": True,
            "description": "专业版，适合复杂任务",
        },
        "doubao-lite-32k": {
            "max_tokens": 4096,
            "context_window": 32768,
            "supports_vision": False,
            "supports_function_calling": True,
            "supports_streaming": True,
            "description": "轻量版，速度快成本低",
        },
        "doubao-vision": {
            "max_tokens": 4096,
            "context_window": 4096,
            "supports_vision": True,
            "supports_function_calling": True,
            "supports_streaming": True,
            "description": "视觉版，支持图片理解",
        },
        "doubao-pro-4k": {
            "max_tokens": 4096,
            "context_window": 4096,
            "supports_vision": False,
            "supports_function_calling": True,
            "supports_streaming": True,
            "description": "专业版4K上下文",
        },
        "doubao-lite-4k": {
            "max_tokens": 4096,
            "context_window": 4096,
            "supports_vision": False,
            "supports_function_calling": True,
            "supports_streaming": True,
            "description": "轻量版4K上下文",
        },
    }

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        初始化豆包Provider。

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
        self._function_registry: Dict[str, Callable] = {}

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

    def get_capabilities(self) -> Set[ModelCapability]:
        """获取当前模型支持的能力"""
        model_config = self._get_model_config()
        capabilities = {
            ModelCapability.STREAMING,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.JSON_MODE,
        }

        if model_config.get("supports_vision"):
            capabilities.add(ModelCapability.VISION)

        return capabilities

    def _get_model_config(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """获取模型配置"""
        model = model_id or self.model_id
        return self.MODEL_CONFIGS.get(model, self.MODEL_CONFIGS[self.DEFAULT_MODEL])

    def _apply_rate_limit(self) -> None:
        """应用速率限制"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time

        # 最小请求间隔 0.1秒 (10 requests/second max)
        min_interval = 0.1
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)

        self._last_request_time = time.time()

    def register_function(self, name: str, func: Callable) -> None:
        """
        注册可调用的函数。

        Args:
            name: 函数名称
            func: 函数实现
        """
        self._function_registry[name] = func
        logger.info(f"Registered function: {name}")

    async def execute_function(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        执行已注册的函数。

        Args:
            name: 函数名称
            arguments: 函数参数

        Returns:
            函数执行结果
        """
        if name not in self._function_registry:
            raise ValueError(f"Function '{name}' not registered")

        func = self._function_registry[name]

        # 支持异步函数
        if asyncio.iscoroutinefunction(func):
            return await func(**arguments)
        else:
            return func(**arguments)

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
        model_config = self._get_model_config(config.model_id)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # 构建请求体
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

        # 处理extra_body中的tools
        if config.extra_body:
            if "tools" in config.extra_body:
                payload["tools"] = config.extra_body["tools"]
            if "tool_choice" in config.extra_body:
                payload["tool_choice"] = config.extra_body["tool_choice"]

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
                    tool_calls = message.get("tool_calls")
                    metadata = {
                        "provider": self.PROVIDER_NAME,
                        "model": config.model_id,
                    }

                    if tool_calls:
                        metadata["tool_calls"] = tool_calls
                        # 如果有函数调用，执行它们
                        for tool_call in tool_calls:
                            if tool_call.get("type") == "function":
                                func = tool_call.get("function", {})
                                func_name = func.get("name")
                                func_args = json.loads(func.get("arguments", "{}"))

                                try:
                                    func_result = await self.execute_function(func_name, func_args)
                                    metadata[f"result_{func_name}"] = func_result
                                except Exception as e:
                                    metadata[f"error_{func_name}"] = str(e)

                    return LLMResponse(
                        content=content,
                        model_id=config.model_id,
                        usage=usage,
                        finish_reason=finish_reason,
                        latency_ms=(time.time() - start_time) * 1000,
                        metadata=metadata
                    )

        except aiohttp.ClientError as e:
            logger.error(f"Doubao HTTP error: {e}")
            return LLMResponse(
                content="",
                model_id=config.model_id,
                error=f"HTTP error: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            logger.error(f"Doubao error: {e}")
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

        chat_messages = self._convert_messages(messages)

        payload = {
            "model": config.model_id,
            "messages": chat_messages,
            "temperature": config.temperature,
            "stream": True,
        }

        if config.max_tokens:
            payload["max_tokens"] = config.max_tokens

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

                            # 处理tool_calls
                            if delta.get("tool_calls"):
                                yield json.dumps({"tool_calls": delta["tool_calls"]})

                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

        except Exception as e:
            logger.error(f"Doubao stream error: {e}")
            yield f"Error: {str(e)}"

    def _convert_messages(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """转换消息格式"""
        converted = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role not in ["system", "user", "assistant", "tool"]:
                role = "user"

            converted_msg: Dict[str, Any] = {"role": role}

            # 处理多模态内容
            if isinstance(content, list):
                converted_msg["content"] = content
            else:
                converted_msg["content"] = str(content) if content is not None else ""

            # 处理tool_calls
            if "tool_calls" in msg:
                converted_msg["tool_calls"] = msg["tool_calls"]

            if "tool_call_id" in msg:
                converted_msg["tool_call_id"] = msg["tool_call_id"]

            converted.append(converted_msg)

        return converted

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
            model: 模型ID，默认使用doubao-vision
            prompt: 图片相关的提示词
            **kwargs: 其他参数

        Returns:
            LLM响应对象
        """
        vision_model = model or "doubao-vision"
        model_config = self._get_model_config(vision_model)

        if not model_config.get("supports_vision"):
            raise ValueError(f"Model {vision_model} does not support vision")

        # 构建视觉消息
        vision_messages = messages.copy()

        # 添加图片消息
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
            "registered_functions": list(self._function_registry.keys()),
        }

    async def close(self) -> None:
        """关闭Provider"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None


# ==================== CLI测试代码 ====================

async def test_doubao_chat():
    """测试豆包普通对话"""
    import os

    print("=" * 60)
    print("豆包 (Doubao) Provider 测试 - 普通对话")
    print("=" * 60)

    api_key = os.environ.get("DOUBAO_API_KEY", "your_api_key_here")

    config = LLMConfig(
        model_id="doubao-pro-32k",
        api_key=api_key,
        temperature=0.7,
        max_tokens=2048
    )
    provider = DoubaoProvider(config)
    provider.initialize()

    print(f"\nProvider: {provider.provider_name}")
    print(f"Model: {provider.model_id}")
    print(f"Capabilities: {provider.get_capabilities()}")

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
        {"role": "user", "content": "请用3句话描述北京的美食。"}
    ]

    try:
        print("Response: ", end="", flush=True)
        async for chunk in provider.stream_chat(messages):
            print(chunk, end="", flush=True)
        print()
    except Exception as e:
        print(f"Error: {e}")

    await provider.close()


async def test_doubao_function_calling():
    """测试豆包函数调用"""
    import os

    print("\n" + "=" * 60)
    print("豆包 (Doubao) Provider 测试 - 函数调用")
    print("=" * 60)

    api_key = os.environ.get("DOUBAO_API_KEY", "your_api_key_here")

    config = LLMConfig(
        model_id="doubao-pro-32k",
        api_key=api_key,
        temperature=0.7,
    )
    provider = DoubaoProvider(config)
    provider.initialize()

    # 注册示例函数
    def get_current_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
        """获取当前天气"""
        return {
            "location": location,
            "temperature": 25 if unit == "celsius" else 77,
            "unit": unit,
            "condition": "sunny"
        }

    provider.register_function("get_current_weather", get_current_weather)

    print(f"\nRegistered functions: {list(provider._function_registry.keys())}")

    # 定义工具
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "获取指定城市的当前天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "城市名称，如北京、上海"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "温度单位"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    print("\n" + "-" * 40)
    print("测试: 函数调用")
    print("-" * 40)

    messages = [
        {"role": "user", "content": "北京今天天气怎么样？"}
    ]

    try:
        response = await provider.chat_completion(messages, tools=tools)
        print(f"Response: {response.content}")
        print(f"Metadata: {response.metadata}")
    except Exception as e:
        print(f"Error: {e}")

    await provider.close()


async def test_doubao_vision():
    """测试豆包视觉理解"""
    import os

    print("\n" + "=" * 60)
    print("豆包 (Doubao) Provider 测试 - 视觉理解")
    print("=" * 60)

    api_key = os.environ.get("DOUBAO_API_KEY", "your_api_key_here")

    config = LLMConfig(
        model_id="doubao-vision",
        api_key=api_key,
    )
    provider = DoubaoProvider(config)
    provider.initialize()

    print(f"\nModel: {provider.model_id}")
    print(f"Supports vision: {provider._get_model_config().get('supports_vision')}")

    # 注意：这里使用一个示例base64图片
    # 实际使用时需要提供真实的图片base64编码
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

    await provider.close()


async def test_doubao_models():
    """测试不同模型"""
    import os

    print("\n" + "=" * 60)
    print("豆包 (Doubao) Provider 测试 - 不同模型")
    print("=" * 60)

    api_key = os.environ.get("DOUBAO_API_KEY", "your_api_key_here")

    models = ["doubao-pro-32k", "doubao-lite-32k"]

    for model_id in models:
        print(f"\n--- 测试模型: {model_id} ---")

        config = LLMConfig(
            model_id=model_id,
            api_key=api_key,
            temperature=0.7,
        )
        provider = DoubaoProvider(config)
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


if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_doubao_chat())
    asyncio.run(test_doubao_function_calling())
    asyncio.run(test_doubao_vision())
    asyncio.run(test_doubao_models())
