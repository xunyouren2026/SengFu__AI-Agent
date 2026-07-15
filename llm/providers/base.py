"""
LLM Provider基类

定义所有LLM Provider的公共接口和功能。

Author: AGI Team
Version: 1.0.0
"""

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict, List, Optional, Any, Set, Tuple,
    Callable, Union, AsyncIterator, TypeVar
)
from datetime import datetime
import asyncio
import json

# 配置日志
logger = logging.getLogger(__name__)

T = TypeVar('T')


class ModelCapability(Enum):
    """模型能力"""
    STREAMING = auto()
    FUNCTION_CALLING = auto()
    VISION = auto()
    JSON_MODE = auto()
    SEED = auto()
    RESPONSE_FORMAT = auto()


@dataclass
class LLMConfig:
    """
    LLM配置
    
    Attributes:
        model_id: 模型ID
        api_key: API密钥
        base_url: API基础URL
        timeout: 超时时间(秒)
        max_retries: 最大重试次数
        temperature: 温度参数
        max_tokens: 最大Token数
        top_p: Top-p采样
        stop: 停止序列
        seed: 随机种子
        response_format: 响应格式
    """
    model_id: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    stop: Optional[List[str]] = None
    seed: Optional[int] = None
    response_format: Optional[Dict[str, Any]] = None
    extra_headers: Optional[Dict[str, str]] = None
    extra_body: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """
    LLM响应
    
    Attributes:
        content: 响应内容
        model_id: 使用的模型ID
        usage: Token使用量
        finish_reason: 完成原因
        created_at: 创建时间
        latency_ms: 延迟(毫秒)
        error: 错误信息
        metadata: 其他元数据
    """
    content: str
    model_id: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    latency_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_success(self) -> bool:
        """是否成功"""
        return self.error is None and self.content is not None
    
    @property
    def input_tokens(self) -> int:
        """输入Token数"""
        return self.usage.get("prompt_tokens", 0) if self.usage else 0
    
    @property
    def output_tokens(self) -> int:
        """输出Token数"""
        return self.usage.get("completion_tokens", 0) if self.usage else 0
    
    @property
    def total_tokens(self) -> int:
        """总Token数"""
        return self.input_tokens + self.output_tokens
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "content": self.content,
            "model_id": self.model_id,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "created_at": self.created_at.isoformat(),
            "latency_ms": self.latency_ms,
            "error": self.error,
            "metadata": self.metadata,
        }


class LLMError(Exception):
    """LLM异常"""
    
    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        model_id: Optional[str] = None,
        status_code: Optional[int] = None,
        response: Optional[Dict] = None
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.model_id = model_id
        self.status_code = status_code
        self.response = response


class BaseLLMProvider(ABC):
    """
    LLM Provider基类
    
    所有具体Provider都应继承此类并实现抽象方法。
    
    Example:
        ```python
        class MyProvider(BaseLLMProvider):
            async def _async_generate(self, messages, config):
                # 实现具体的API调用
                pass
        ```
    """
    
    # Provider元信息
    PROVIDER_NAME: str = "base"
    SUPPORTED_MODELS: Set[str] = set()
    DEFAULT_MODEL: str = ""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        初始化Provider。
        
        Args:
            config: LLM配置
        """
        self._config = config
        self._initialized = False
    
    @property
    def config(self) -> LLMConfig:
        """获取配置"""
        return self._config
    
    @property
    def model_id(self) -> str:
        """获取当前模型ID"""
        return self._config.model_id if self._config else self.DEFAULT_MODEL
    
    @property
    def provider_name(self) -> str:
        """获取Provider名称"""
        return self.PROVIDER_NAME
    
    def initialize(self) -> None:
        """初始化Provider"""
        if self._initialized:
            return
        
        self._validate_config()
        self._setup_client()
        self._initialized = True
    
    def _validate_config(self) -> None:
        """验证配置"""
        if not self._config:
            self._config = LLMConfig(model_id=self.DEFAULT_MODEL)
        
        if not self._config.model_id:
            raise ValueError("model_id is required")
    
    def _setup_client(self) -> None:
        """设置客户端（可被子类重写）"""
        pass
    
    @abstractmethod
    async def _async_generate(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> LLMResponse:
        """
        异步生成响应（子类必须实现）
        
        Args:
            messages: 消息列表
            config: 生成配置
            
        Returns:
            LLM响应
        """
        pass
    
    async def generate(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> LLMResponse:
        """
        生成响应。
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数，会合并到配置中
            
        Returns:
            LLM响应
        """
        if not self._initialized:
            self.initialize()
        
        # 合并配置
        config = self._merge_config(**kwargs)
        
        # 执行生成
        start_time = time.time()
        
        try:
            response = await self._async_generate(messages, config)
            response.latency_ms = (time.time() - start_time) * 1000
            return response
            
        except Exception as e:
            logger.error(f"LLM generate error: {e}")
            return LLMResponse(
                content="",
                model_id=config.model_id,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )
    
    def generate_sync(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> LLMResponse:
        """
        同步生成响应。
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Returns:
            LLM响应
        """
        return asyncio.run(self.generate(messages, **kwargs))
    
    async def stream_generate(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> AsyncIterator[str]:
        """
        流式生成响应。
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Yields:
            响应片段
        """
        if not self._initialized:
            self.initialize()
        
        config = self._merge_config(**kwargs)
        
        try:
            async for chunk in self._async_stream_generate(messages, config):
                yield chunk
        except Exception as e:
            logger.error(f"LLM stream generate error: {e}")
            yield f"Error: {e}"
    
    async def _async_stream_generate(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> AsyncIterator[str]:
        """
        异步流式生成（子类可选实现）
        
        Args:
            messages: 消息列表
            config: 生成配置
            
        Yields:
            响应片段
        """
        # 默认实现：一次性生成后流式返回
        response = await self._async_generate(messages, config)
        
        if response.error:
            yield f"Error: {response.error}"
            return
        
        # 模拟流式输出
        for i in range(0, len(response.content), 10):
            yield response.content[i:i+10]
            await asyncio.sleep(0.01)
    
    def _merge_config(self, **kwargs) -> LLMConfig:
        """合并配置"""
        if not self._config:
            return LLMConfig(
                model_id=kwargs.get("model_id", self.DEFAULT_MODEL),
                **{k: v for k, v in kwargs.items() if k != "model_id"}
            )
        
        config_dict = self._config.__dict__.copy()
        config_dict.update({k: v for k, v in kwargs.items() if v is not None})
        
        return LLMConfig(**config_dict)
    
    @staticmethod
    def format_messages(
        prompt: Optional[str] = None,
        system: Optional[str] = None,
        messages: Optional[List[Dict]] = None
    ) -> List[Dict[str, Any]]:
        """
        格式化消息。
        
        Args:
            prompt: 用户提示
            system: 系统消息
            messages: 已有消息列表
            
        Returns:
            格式化的消息列表
        """
        result = []
        
        if system:
            result.append({"role": "system", "content": system})
        
        if messages:
            result.extend(messages)
        
        if prompt:
            result.append({"role": "user", "content": prompt})
        
        return result
    
    def get_capabilities(self) -> Set[ModelCapability]:
        """获取模型能力"""
        return set()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取Provider统计"""
        return {
            "provider": self.PROVIDER_NAME,
            "model_id": self.model_id,
            "initialized": self._initialized,
        }


def create_provider(
    provider_name: str,
    config: Optional[LLMConfig] = None
) -> BaseLLMProvider:
    """
    创建Provider实例。
    
    Args:
        provider_name: Provider名称
        config: 配置
        
    Returns:
        Provider实例
    """
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "zhipuai": ZhipuAIProvider,
        "dashscope": DashScopeProvider,
        "moonshot": MoonshotProvider,
        "deepseek": DeepSeekProvider,
        "local": LocalModelProvider,
    }
    
    provider_class = providers.get(provider_name.lower())
    if not provider_class:
        raise ValueError(f"Unknown provider: {provider_name}")
    
    return provider_class(config)


class OpenAICompatibleProvider(BaseLLMProvider):
    """
    OpenAI兼容API的通用Provider基类。

    大多数国内LLM服务商都提供OpenAI兼容的API接口（/chat/completions），
    此基类封装了通用的HTTP请求、消息转换、流式解析等逻辑，
    子类只需定义配置差异即可。

    子类需要定义的类属性:
        - PROVIDER_NAME: Provider名称
        - SUPPORTED_MODELS: 支持的模型集合
        - DEFAULT_MODEL: 默认模型
        - API_BASE: API基础URL
        - CHAT_ENDPOINT: 聊天接口路径 (默认 "/chat/completions")
        - MODEL_CONFIGS: 模型配置字典
        - RATE_LIMIT_INTERVAL: 速率限制间隔秒数 (默认 0.1)

    Example:
        ```python
        class MyProvider(OpenAICompatibleProvider):
            PROVIDER_NAME = "my_provider"
            SUPPORTED_MODELS = {"my-model-v1"}
            DEFAULT_MODEL = "my-model-v1"
            API_BASE = "https://api.example.com/v1"
            MODEL_CONFIGS = {
                "my-model-v1": {
                    "max_tokens": 4096,
                    "context_window": 8192,
                    "supports_vision": False,
                    "supports_function_calling": True,
                    "supports_streaming": True,
                },
            }
        ```
    """

    # 子类应重写以下属性
    CHAT_ENDPOINT: str = "/chat/completions"
    MODEL_CONFIGS: Dict[str, Any] = {}
    RATE_LIMIT_INTERVAL: float = 0.1

    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        self._api_base = self.API_BASE
        self._session: Optional['aiohttp.ClientSession'] = None
        self._rate_limit_remaining = 1000
        self._request_count = 0
        self._last_request_time = 0.0

    def _setup_client(self) -> None:
        """设置HTTP客户端会话"""
        import aiohttp
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
        if time_since_last < self.RATE_LIMIT_INTERVAL:
            time.sleep(self.RATE_LIMIT_INTERVAL - time_since_last)
        self._last_request_time = time.time()

    def _convert_messages(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """转换消息格式为标准OpenAI格式"""
        converted = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role not in ["system", "user", "assistant", "tool"]:
                role = "user"

            converted_msg: Dict[str, Any] = {"role": role}

            if isinstance(content, list):
                converted_msg["content"] = content
            else:
                converted_msg["content"] = str(content) if content is not None else ""

            if "tool_calls" in msg:
                converted_msg["tool_calls"] = msg["tool_calls"]
            if "tool_call_id" in msg:
                converted_msg["tool_call_id"] = msg["tool_call_id"]
            if "function_call" in msg:
                converted_msg["function_call"] = msg["function_call"]

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

        if config.extra_body:
            for key in ("tools", "task_type", "response_format", "device_id"):
                if key in config.extra_body:
                    payload[key] = config.extra_body[key]

        return payload

    async def _async_generate(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> LLMResponse:
        """异步生成响应"""
        import aiohttp

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

                    if response.status != 200:
                        error_text = await response.text()
                        try:
                            error_data = json.loads(error_text)
                            error_msg = error_data.get("error", {}).get("message", error_text)
                            error_code = error_data.get("error", {}).get("code")
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

                    # 解析OpenAI兼容格式响应
                    choice = result["choices"][0]
                    message = choice.get("message", {})
                    content = message.get("content", "")
                    finish_reason = choice.get("finish_reason")
                    usage = result.get("usage")

                    metadata: Dict[str, Any] = {
                        "provider": self.PROVIDER_NAME,
                        "model": config.model_id,
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

        except Exception as e:
            logger.error(f"{self.PROVIDER_NAME} error: {e}")
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
        import aiohttp

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

                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

        except Exception as e:
            logger.error(f"{self.PROVIDER_NAME} stream error: {e}")
            yield f"Error: {str(e)}"

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """发送聊天完成请求"""
        config = LLMConfig(
            model_id=model or self.model_id,
            api_key=self._config.api_key if self._config else None,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
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
        **kwargs
    ) -> LLMResponse:
        """视觉理解完成（默认不支持，子类可重写）"""
        raise NotImplementedError(f"{self.PROVIDER_NAME}暂不支持视觉功能")

    def get_stats(self) -> Dict[str, Any]:
        """获取Provider统计信息"""
        return {
            "provider": self.PROVIDER_NAME,
            "model_id": self.model_id,
            "initialized": self._initialized,
            "api_base": self._api_base,
            "request_count": self._request_count,
            "supported_models": list(self.SUPPORTED_MODELS),
        }

    async def close(self) -> None:
        """关闭Provider"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
