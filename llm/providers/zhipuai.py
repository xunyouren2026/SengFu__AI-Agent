"""
智谱AI (ZhipuAI) Provider

智谱GLM系列模型的适配器。

支持的模型:
- glm-4
- glm-4v
- glm-3-turbo

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

logger = logging.getLogger(__name__)


class ZhipuAIProvider(BaseLLMProvider):
    """
    智谱AI GLM系列Provider
    
    Features:
        - 支持GLM-4系列
        - 支持视觉理解 (GLM-4V)
        - 支持中英双语
        - 高性价比
    
    Example:
        ```python
        provider = ZhipuAIProvider(LLMConfig(
            model_id="glm-4",
            api_key="..."
        ))
        
        response = await provider.generate([
            {"role": "user", "content": "Hello!"}
        ])
        print(response.content)
        ```
    """
    
    PROVIDER_NAME = "zhipuai"
    SUPPORTED_MODELS = {
        "glm-4", "glm-4v", "glm-4-plus",
        "glm-3-turbo", "glm-3plus-turbo"
    }
    DEFAULT_MODEL = "glm-4"
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        初始化智谱AI Provider。
        
        Args:
            config: 智谱AI配置
        """
        super().__init__(config)
        self._api_base = "https://open.bigmodel.cn/api/paas/v4"
    
    def get_capabilities(self) -> Set[ModelCapability]:
        """获取模型能力"""
        capabilities = {
            ModelCapability.STREAMING,
            ModelCapability.JSON_MODE,
        }
        
        # GLM-4V支持视觉
        if "glm-4v" in self.model_id:
            capabilities.add(ModelCapability.VISION)
        
        return capabilities
    
    async def _async_generate(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> LLMResponse:
        """异步生成响应"""
        start_time = time.time()
        
        if not HTTPX_AVAILABLE:
            return LLMResponse(
                content="",
                model_id=config.model_id,
                error="httpx not available",
                latency_ms=(time.time() - start_time) * 1000
            )
        
        api_key = config.api_key or self._config.api_key
        base_url = config.base_url or self._api_base
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        # 转换消息格式
        chat_messages = self._convert_messages(messages)
        
        data = {
            "model": config.model_id,
            "messages": chat_messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }
        
        if config.max_tokens:
            data["max_tokens"] = config.max_tokens
        
        if config.stop:
            data["stop"] = config.stop
        
        if config.seed is not None:
            data["seed"] = config.seed
        
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
                    metadata={
                        "provider": "zhipuai",
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
    
    def _convert_messages(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """转换消息格式"""
        converted = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            
            if role == "system":
                converted.append({
                    "role": "system",
                    "content": content
                })
            elif role == "user":
                converted.append({
                    "role": "user",
                    "content": content
                })
            elif role == "assistant":
                converted.append({
                    "role": "assistant",
                    "content": content
                })
        
        return converted
    
    async def _async_stream_generate(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> AsyncIterator[str]:
        """流式生成"""
        api_key = config.api_key or self._config.api_key
        base_url = config.base_url or self._api_base
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        chat_messages = self._convert_messages(messages)
        
        data = {
            "model": config.model_id,
            "messages": chat_messages,
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
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "provider": self.PROVIDER_NAME,
            "model_id": self.model_id,
            "initialized": self._initialized,
            "api_base": self._api_base,
        }
