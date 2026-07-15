"""
Anthropic Provider

Anthropic Claude系列模型的适配器。

支持的模型:
- claude-3-opus
- claude-3-sonnet
- claude-3-haiku
- claude-2.1
- claude-2.0

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
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude系列Provider
    
    Features:
        - 支持Claude 3系列
        - 支持长上下文
        - 支持视觉理解
        - 支持流式输出
    
    Example:
        ```python
        provider = AnthropicProvider(LLMConfig(
            model_id="claude-3-sonnet",
            api_key="sk-ant-..."
        ))
        
        response = await provider.generate([
            {"role": "user", "content": "Hello!"}
        ])
        print(response.content)
        ```
    """
    
    PROVIDER_NAME = "anthropic"
    SUPPORTED_MODELS = {
        "claude-3-opus", "claude-3-sonnet", "claude-3-haiku",
        "claude-2.1", "claude-2.0", "claude-instant-1"
    }
    DEFAULT_MODEL = "claude-3-sonnet"
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        初始化Anthropic Provider。
        
        Args:
            config: Anthropic配置
        """
        super().__init__(config)
        self._client = None
    
    def _setup_client(self) -> None:
        """设置Anthropic客户端"""
        if not ANTHROPIC_AVAILABLE:
            logger.warning("anthropic package not available, using httpx")
            return
        
        api_key = self._config.api_key
        base_url = self._config.base_url
        
        if base_url:
            self._client = anthropic.Anthropic(
                api_key=api_key,
                base_url=base_url,
                timeout=self._config.timeout
            )
        else:
            self._client = anthropic.Anthropic(
                api_key=api_key,
                timeout=self._config.timeout
            )
    
    def get_capabilities(self) -> Set[ModelCapability]:
        """获取模型能力"""
        capabilities = {
            ModelCapability.STREAMING,
        }
        
        # Claude 3系列支持视觉
        if "claude-3" in self.model_id:
            capabilities.add(ModelCapability.VISION)
        
        # Claude 3 Sonnet和Opus支持函数调用
        if "claude-3-opus" in self.model_id or "claude-3-sonnet" in self.model_id:
            capabilities.add(ModelCapability.FUNCTION_CALLING)
        
        return capabilities
    
    async def _async_generate(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> LLMResponse:
        """异步生成响应"""
        start_time = time.time()
        
        # 转换消息格式
        system, anthropic_messages = self._convert_messages(messages)
        
        if ANTHROPIC_AVAILABLE and self._client:
            return await self._generate_with_anthropic(
                system, anthropic_messages, config, start_time
            )
        elif HTTPX_AVAILABLE:
            return await self._generate_with_httpx(
                system, anthropic_messages, config, start_time
            )
        else:
            return LLMResponse(
                content="",
                model_id=config.model_id,
                error="No HTTP client available",
                latency_ms=(time.time() - start_time) * 1000
            )
    
    def _convert_messages(
        self,
        messages: List[Dict[str, Any]]
    ) -> tuple:
        """
        转换消息格式
        
        Args:
            messages: 标准消息格式
            
        Returns:
            (system消息, Anthropic消息)
        """
        system = ""
        anthropic_messages = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            
            if role == "system":
                system = content
            elif role == "user":
                # 检查是否包含图片
                if isinstance(content, list):
                    anthropic_content = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "image_url":
                                # 转换图片格式
                                image_data = item["image_url"].get("url", "")
                                anthropic_content.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": image_data.split(",")[-1] if "," in image_data else image_data
                                    }
                                })
                            else:
                                anthropic_content.append(item)
                        else:
                            anthropic_content.append({"type": "text", "text": str(item)})
                    anthropic_messages.append({
                        "role": "user",
                        "content": anthropic_content
                    })
                else:
                    anthropic_messages.append({
                        "role": "user",
                        "content": content
                    })
            elif role == "assistant":
                anthropic_messages.append({
                    "role": "assistant",
                    "content": content
                })
        
        return system, anthropic_messages
    
    async def _generate_with_anthropic(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        config: LLMConfig,
        start_time: float
    ) -> LLMResponse:
        """使用Anthropic SDK生成"""
        try:
            params = {
                "model": config.model_id,
                "messages": messages,
                "temperature": config.temperature,
                "top_p": config.top_p,
            }
            
            if system:
                params["system"] = system
            
            if config.max_tokens:
                params["max_tokens"] = config.max_tokens
            else:
                params["max_tokens"] = 4096  # Anthropic需要指定
            
            if config.stop:
                params["stop_sequences"] = config.stop
            
            if config.seed is not None:
                params["random_seed"] = config.seed
            
            # 发送请求
            response = await self._client.messages.create(**params)
            
            # 解析响应
            content = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, 'text'):
                        content += block.text
            
            # 解析usage
            usage = None
            if hasattr(response, 'usage'):
                usage = {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                }
            
            return LLMResponse(
                content=content,
                model_id=config.model_id,
                usage=usage,
                finish_reason=response.stop_reason if hasattr(response, 'stop_reason') else None,
                latency_ms=(time.time() - start_time) * 1000,
                metadata={
                    "provider": "anthropic",
                    "model": config.model_id,
                }
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
        system: str,
        messages: List[Dict[str, Any]],
        config: LLMConfig,
        start_time: float
    ) -> LLMResponse:
        """使用httpx直接调用"""
        api_key = config.api_key or self._config.api_key
        base_url = config.base_url or "https://api.anthropic.com"
        
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": config.model_id,
            "messages": messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }
        
        if system:
            data["system"] = system
        
        if config.max_tokens:
            data["max_tokens"] = config.max_tokens
        else:
            data["max_tokens"] = 4096
        
        if config.stop:
            data["stop_sequences"] = config.stop
        
        if config.seed is not None:
            data["random_seed"] = config.seed
        
        try:
            async with httpx.AsyncClient(timeout=config.timeout) as client:
                response = await client.post(
                    f"{base_url}/v1/messages",
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
                
                # 解析content
                content = ""
                if result.get("content"):
                    for block in result["content"]:
                        if block.get("type") == "text":
                            content += block.get("text", "")
                
                usage = result.get("usage")
                
                return LLMResponse(
                    content=content,
                    model_id=config.model_id,
                    usage=usage,
                    finish_reason=result.get("stop_reason"),
                    latency_ms=(time.time() - start_time) * 1000,
                    metadata={"provider": "anthropic"}
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
        system, anthropic_messages = self._convert_messages(messages)
        
        if ANTHROPIC_AVAILABLE and self._client:
            async for chunk in self._stream_with_anthropic(
                system, anthropic_messages, config
            ):
                yield chunk
        elif HTTPX_AVAILABLE:
            async for chunk in self._stream_with_httpx(
                system, anthropic_messages, config
            ):
                yield chunk
        else:
            yield "Error: No HTTP client available"
    
    async def _stream_with_anthropic(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> AsyncIterator[str]:
        """使用Anthropic SDK流式生成"""
        try:
            params = {
                "model": config.model_id,
                "messages": messages,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens or 4096,
                "stream": True,
            }
            
            if system:
                params["system"] = system
            
            stream = await self._client.messages.create(**params)
            
            async for chunk in stream:
                if chunk.type == "content_block_delta":
                    if hasattr(chunk.delta, 'text'):
                        yield chunk.delta.text
                        
        except Exception as e:
            yield f"Error: {str(e)}"
    
    async def _stream_with_httpx(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> AsyncIterator[str]:
        """使用httpx流式生成"""
        api_key = config.api_key or self._config.api_key
        base_url = config.base_url or "https://api.anthropic.com"
        
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": config.model_id,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens or 4096,
            "stream": True,
        }
        
        if system:
            data["system"] = system
        
        try:
            async with httpx.AsyncClient(timeout=config.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{base_url}/v1/messages",
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
                                if chunk_data.get("type") == "content_block_delta":
                                    if chunk_data.get("delta", {}).get("text"):
                                        yield chunk_data["delta"]["text"]
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "provider": self.PROVIDER_NAME,
            "model_id": self.model_id,
            "initialized": self._initialized,
            "client_type": "anthropic" if ANTHROPIC_AVAILABLE else "httpx",
        }
