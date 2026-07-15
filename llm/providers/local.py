"""
本地模型 Provider

支持Ollama、vLLM等本地部署模型的适配器。

支持的模型:
- Ollama: llama2, mistral, vicuna等
- vLLM: 任意vLLM支持的模型
- LocalAI: 任意LocalAI支持的模型

Author: AGI Team
Version: 1.0.0
"""

import time
import logging
import json
from enum import Enum
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


class LocalProviderType(Enum):
    """本地Provider类型"""
    OLLAMA = "ollama"
    VLLM = "vllm"
    LOCALAI = "localai"
    LMSTUDIO = "lmstudio"


class LocalModelProvider(BaseLLMProvider):
    """
    本地模型Provider
    
    Features:
        - 支持Ollama
        - 支持vLLM
        - 支持LocalAI
        - 支持LM Studio
        - 完全本地化部署
    
    Example:
        ```python
        # Ollama
        provider = LocalModelProvider(LLMConfig(
            model_id="llama2",
            base_url="http://localhost:11434"
        ), provider_type=LocalProviderType.OLLAMA)
        
        # vLLM
        provider = LocalModelProvider(LLMConfig(
            model_id="meta-llama/Llama-2-7b-chat-hf",
            base_url="http://localhost:8000/v1"
        ), provider_type=LocalProviderType.VLLM)
        
        response = await provider.generate([
            {"role": "user", "content": "Hello!"}
        ])
        ```
    """
    
    PROVIDER_NAME = "local"
    SUPPORTED_MODELS = set()  # 动态确定
    DEFAULT_MODEL = "llama2"
    
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        provider_type: str = "ollama"
    ):
        """
        初始化本地模型Provider。
        
        Args:
            config: LLM配置
            provider_type: Provider类型 (ollama, vllm, localai, lmstudio)
        """
        super().__init__(config)
        self._provider_type = provider_type
        self._set_api_endpoints()
    
    def _set_api_endpoints(self) -> None:
        """设置API端点"""
        if self._provider_type == "ollama":
            self._api_base = self._config.base_url or "http://localhost:11434"
            self._chat_endpoint = "/api/chat"
            self._generate_endpoint = "/api/generate"
            self._embed_endpoint = "/api/embeddings"
        elif self._provider_type == "vllm":
            self._api_base = self._config.base_url or "http://localhost:8000/v1"
            self._chat_endpoint = "/chat/completions"
            self._generate_endpoint = "/completions"
            self._embed_endpoint = "/embeddings"
        elif self._provider_type == "localai":
            self._api_base = self._config.base_url or "http://localhost:8080/v1"
            self._chat_endpoint = "/chat/completions"
            self._generate_endpoint = "/completions"
            self._embed_endpoint = "/embeddings"
        elif self._provider_type == "lmstudio":
            self._api_base = self._config.base_url or "http://localhost:1234/v1"
            self._chat_endpoint = "/chat/completions"
            self._generate_endpoint = "/completions"
            self._embed_endpoint = "/embeddings"
        else:
            self._api_base = self._config.base_url or "http://localhost:11434"
            self._chat_endpoint = "/api/chat"
            self._generate_endpoint = "/api/generate"
            self._embed_endpoint = "/api/embeddings"
    
    def get_capabilities(self) -> Set[ModelCapability]:
        """获取模型能力"""
        capabilities = {ModelCapability.STREAMING}
        
        if self._provider_type in ["vllm", "localai"]:
            capabilities.add(ModelCapability.JSON_MODE)
        
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
        
        if self._provider_type == "ollama":
            return await self._generate_ollama(messages, config, start_time)
        else:
            return await self._generate_openai_compatible(messages, config, start_time)
    
    async def _generate_ollama(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig,
        start_time: float
    ) -> LLMResponse:
        """Ollama生成"""
        # 合并消息为单个提示
        prompt = self._merge_messages_to_prompt(messages)
        
        data = {
            "model": config.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config.temperature,
            }
        }
        
        if config.max_tokens:
            data["options"]["num_predict"] = config.max_tokens
        
        if config.stop:
            data["options"]["stop"] = config.stop
        
        try:
            async with httpx.AsyncClient(timeout=config.timeout) as client:
                response = await client.post(
                    f"{self._api_base}{self._generate_endpoint}",
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
                
                content = result.get("response", "")
                total_duration = result.get("total_duration", 0)
                
                # 计算token使用
                prompt_eval_count = result.get("prompt_eval_count", 0)
                eval_count = result.get("eval_count", 0)
                
                return LLMResponse(
                    content=content,
                    model_id=config.model_id,
                    usage={
                        "prompt_tokens": prompt_eval_count,
                        "completion_tokens": eval_count,
                        "total_tokens": prompt_eval_count + eval_count,
                    },
                    latency_ms=(time.time() - start_time) * 1000,
                    metadata={
                        "provider": "ollama",
                        "model": config.model_id,
                        "total_duration_ns": total_duration,
                    }
                )
                
        except Exception as e:
            return LLMResponse(
                content="",
                model_id=config.model_id,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )
    
    async def _generate_openai_compatible(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig,
        start_time: float
    ) -> LLMResponse:
        """OpenAI兼容API生成"""
        headers = {"Content-Type": "application/json"}
        
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        
        data = {
            "model": config.model_id,
            "messages": messages,
            "temperature": config.temperature,
        }
        
        if config.max_tokens:
            data["max_tokens"] = config.max_tokens
        
        if config.top_p:
            data["top_p"] = config.top_p
        
        if config.stop:
            data["stop"] = config.stop
        
        if config.seed is not None:
            data["seed"] = config.seed
        
        try:
            async with httpx.AsyncClient(timeout=config.timeout) as client:
                response = await client.post(
                    f"{self._api_base}{self._chat_endpoint}",
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
                        "provider": self._provider_type,
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
    
    def _merge_messages_to_prompt(
        self,
        messages: List[Dict[str, Any]]
    ) -> str:
        """将消息合并为单个提示"""
        parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        
        parts.append("Assistant:")
        return "\n\n".join(parts)
    
    async def _async_stream_generate(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> AsyncIterator[str]:
        """流式生成"""
        if not HTTPX_AVAILABLE:
            yield "Error: httpx not available"
            return
        
        if self._provider_type == "ollama":
            async for chunk in self._stream_ollama(messages, config):
                yield chunk
        else:
            async for chunk in self._stream_openai_compatible(messages, config):
                yield chunk
    
    async def _stream_ollama(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> AsyncIterator[str]:
        """Ollama流式生成"""
        prompt = self._merge_messages_to_prompt(messages)
        
        data = {
            "model": config.model_id,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": config.temperature,
            }
        }
        
        if config.max_tokens:
            data["options"]["num_predict"] = config.max_tokens
        
        try:
            async with httpx.AsyncClient(timeout=config.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self._api_base}{self._generate_endpoint}",
                    json=data
                ) as response:
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                chunk = json.loads(line)
                                if chunk.get("response"):
                                    yield chunk["response"]
                                if chunk.get("done"):
                                    break
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            yield f"Error: {str(e)}"
    
    async def _stream_openai_compatible(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> AsyncIterator[str]:
        """OpenAI兼容流式生成"""
        headers = {"Content-Type": "application/json"}
        
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        
        data = {
            "model": config.model_id,
            "messages": messages,
            "temperature": config.temperature,
            "stream": True,
        }
        
        if config.max_tokens:
            data["max_tokens"] = config.max_tokens
        
        try:
            async with httpx.AsyncClient(timeout=config.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self._api_base}{self._chat_endpoint}",
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
    
    async def list_models(self) -> List[str]:
        """
        列出可用模型。
        
        Returns:
            模型列表
        """
        if not HTTPX_AVAILABLE:
            return []
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                if self._provider_type == "ollama":
                    response = await client.get(f"{self._api_base}/api/tags")
                    if response.status_code == 200:
                        data = response.json()
                        return [m["name"] for m in data.get("models", [])]
                else:
                    # OpenAI兼容API
                    response = await client.get(f"{self._api_base}/models")
                    if response.status_code == 200:
                        data = response.json()
                        return [m["id"] for m in data.get("data", [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        
        return []
    
    async def check_health(self) -> bool:
        """
        检查Provider健康状态。
        
        Returns:
            是否健康
        """
        if not HTTPX_AVAILABLE:
            return False
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                if self._provider_type == "ollama":
                    response = await client.get(f"{self._api_base}/api/tags")
                    return response.status_code == 200
                else:
                    response = await client.get(f"{self._api_base}/models")
                    return response.status_code == 200
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "provider": self.PROVIDER_NAME,
            "provider_type": self._provider_type,
            "model_id": self.model_id,
            "initialized": self._initialized,
            "api_base": self._api_base,
        }


from enum import Enum
