"""
OpenAI Provider

OpenAI GPT系列模型的适配器。

支持的模型:
- gpt-4
- gpt-4-turbo
- gpt-4-32k
- gpt-3.5-turbo
- gpt-3.5-turbo-16k

Author: AGI Team
Version: 1.0.0
"""

import time
import logging
import json
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

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI GPT系列Provider
    
    Features:
        - 支持GPT-4和GPT-3.5系列
        - 支持流式输出
        - 支持函数调用
        - 支持视觉理解
    
    Example:
        ```python
        provider = OpenAIProvider(LLMConfig(
            model_id="gpt-4",
            api_key="sk-..."
        ))
        
        response = await provider.generate([
            {"role": "user", "content": "Hello!"}
        ])
        print(response.content)
        ```
    """
    
    PROVIDER_NAME = "openai"
    SUPPORTED_MODELS = {
        "gpt-4", "gpt-4-turbo", "gpt-4-32k", "gpt-4-0613",
        "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0613"
    }
    DEFAULT_MODEL = "gpt-3.5-turbo"
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        初始化OpenAI Provider。
        
        Args:
            config: OpenAI配置
        """
        super().__init__(config)
        self._client = None
        self._async_client = None
    
    def _setup_client(self) -> None:
        """设置OpenAI客户端"""
        if not OPENAI_AVAILABLE:
            logger.warning("openai package not available, using httpx")
            return
        
        api_key = self._config.api_key
        base_url = self._config.base_url
        
        if base_url:
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self._config.timeout
            )
        else:
            self._client = openai.OpenAI(
                api_key=api_key,
                timeout=self._config.timeout
            )
    
    def get_capabilities(self) -> Set[ModelCapability]:
        """获取模型能力"""
        capabilities = {
            ModelCapability.STREAMING,
            ModelCapability.JSON_MODE,
        }
        
        # GPT-4和部分GPT-3.5支持函数调用
        if self.model_id and ("gpt-4" in self.model_id or 
                             "gpt-3.5-turbo-0613" in self.model_id or
                             "gpt-3.5-turbo-16k-0613" in self.model_id):
            capabilities.add(ModelCapability.FUNCTION_CALLING)
        
        # GPT-4V支持视觉
        if "gpt-4-vision" in self.model_id:
            capabilities.add(ModelCapability.VISION)
        
        return capabilities
    
    async def _async_generate(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> LLMResponse:
        """异步生成响应"""
        start_time = time.time()
        
        if OPENAI_AVAILABLE and self._client:
            return await self._generate_with_openai(messages, config, start_time)
        elif HTTPX_AVAILABLE:
            return await self._generate_with_httpx(messages, config, start_time)
        else:
            return LLMResponse(
                content="",
                model_id=config.model_id,
                error="No HTTP client available",
                latency_ms=(time.time() - start_time) * 1000
            )
    
    async def _generate_with_openai(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig,
        start_time: float
    ) -> LLMResponse:
        """使用OpenAI SDK生成"""
        try:
            # 构建请求参数
            params = {
                "model": config.model_id,
                "messages": self._prepare_messages(messages),
                "temperature": config.temperature,
                "top_p": config.top_p,
            }
            
            if config.max_tokens:
                params["max_tokens"] = config.max_tokens
            
            if config.stop:
                params["stop"] = config.stop
            
            if config.seed is not None:
                params["seed"] = config.seed
            
            if config.response_format:
                params["response_format"] = config.response_format
            
            # 添加函数调用支持
            if hasattr(self, '_functions') and self._functions:
                params["functions"] = self._functions
                if hasattr(self, '_function_call') and self._function_call:
                    params["function_call"] = self._function_call
            
            # 发送请求
            response = await self._client.chat.completions.create(**params)
            
            # 解析响应
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason
            
            # 解析usage
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            
            # 处理函数调用响应
            if hasattr(response.choices[0].message, 'function_call'):
                fc = response.choices[0].message.function_call
                content = json.dumps({
                    "name": fc.name,
                    "arguments": fc.arguments
                })
            
            return LLMResponse(
                content=content,
                model_id=config.model_id,
                usage=usage,
                finish_reason=finish_reason,
                latency_ms=(time.time() - start_time) * 1000,
                metadata={
                    "provider": "openai",
                    "model": config.model_id,
                }
            )
            
        except openai.APIError as e:
            return LLMResponse(
                content="",
                model_id=config.model_id,
                error=f"OpenAI API Error: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return LLMResponse(
                content="",
                model_id=config.model_id,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )
    
    async def _generate_with_httpx(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig,
        start_time: float
    ) -> LLMResponse:
        """使用httpx直接调用"""
        api_key = config.api_key or self._config.api_key
        base_url = config.base_url or self._config.base_url or "https://api.openai.com/v1"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        if config.extra_headers:
            headers.update(config.extra_headers)
        
        data = {
            "model": config.model_id,
            "messages": self._prepare_messages(messages),
            "temperature": config.temperature,
            "top_p": config.top_p,
        }
        
        if config.max_tokens:
            data["max_tokens"] = config.max_tokens
        
        if config.stop:
            data["stop"] = config.stop
        
        if config.seed is not None:
            data["seed"] = config.seed
        
        if config.extra_body:
            data.update(config.extra_body)
        
        try:
            async with httpx.AsyncClient(timeout=config.timeout) as client:
                response = await client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=data
                )
                
                if response.status_code != 200:
                    error_body = response.text
                    return LLMResponse(
                        content="",
                        model_id=config.model_id,
                        error=f"HTTP {response.status_code}: {error_body}",
                        latency_ms=(time.time() - start_time) * 1000
                    )
                
                result = response.json()
                
                content = result["choices"][0]["message"].get("content", "")
                finish_reason = result["choices"][0].get("finish_reason")
                
                usage = result.get("usage")
                
                return LLMResponse(
                    content=content,
                    model_id=config.model_id,
                    usage=usage,
                    finish_reason=finish_reason,
                    latency_ms=(time.time() - start_time) * 1000,
                    metadata={"provider": "openai"}
                )
                
        except Exception as e:
            return LLMResponse(
                content="",
                model_id=config.model_id,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )
    
    async def _async_stream_generate(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> AsyncIterator[str]:
        """流式生成"""
        if OPENAI_AVAILABLE and self._client:
            async for chunk in self._stream_with_openai(messages, config):
                yield chunk
        elif HTTPX_AVAILABLE:
            async for chunk in self._stream_with_httpx(messages, config):
                yield chunk
        else:
            yield "Error: No HTTP client available"
    
    async def _stream_with_openai(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> AsyncIterator[str]:
        """使用OpenAI SDK流式生成"""
        try:
            params = {
                "model": config.model_id,
                "messages": self._prepare_messages(messages),
                "temperature": config.temperature,
                "stream": True,
            }
            
            if config.max_tokens:
                params["max_tokens"] = config.max_tokens
            
            stream = await self._client.chat.completions.create(**params)
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"Error: {str(e)}"
    
    async def _stream_with_httpx(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> AsyncIterator[str]:
        """使用httpx流式生成"""
        api_key = config.api_key or self._config.api_key
        base_url = config.base_url or self._config.base_url or "https://api.openai.com/v1"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": config.model_id,
            "messages": self._prepare_messages(messages),
            "temperature": config.temperature,
            "stream": True,
        }
        
        if config.max_tokens:
            data["max_tokens"] = config.max_tokens
        
        try:
            async with httpx.AsyncClient(timeout=config.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=data
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk_data = json.loads(data_str)
                                if chunk_data["choices"][0]["delta"].get("content"):
                                    yield chunk_data["choices"][0]["delta"]["content"]
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def _prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """准备消息格式"""
        prepared = []
        
        for msg in messages:
            if isinstance(msg, dict):
                prepared.append(msg)
            elif hasattr(msg, 'role') and hasattr(msg, 'content'):
                prepared.append({"role": msg.role, "content": msg.content})
        
        return prepared
    
    def set_functions(self, functions: List[Dict[str, Any]]) -> None:
        """设置函数定义"""
        self._functions = functions
    
    def clear_functions(self) -> None:
        """清除函数定义"""
        self._functions = []
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "provider": self.PROVIDER_NAME,
            "model_id": self.model_id,
            "initialized": self._initialized,
            "client_type": "openai" if OPENAI_AVAILABLE else "httpx",
        }
