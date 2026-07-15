"""
智谱AI GLM-4-Flash 系列 Provider

GLM-4-Flash 和 GLM-4-FlashX 是智谱AI提供的完全免费大模型。
完全免费、高速、支持长文本。

支持的模型:
- glm-4-flash: 基础版，完全免费
- glm-4-flashx: 增强版，完全免费

API文档: https://open.bigmodel.cn/dev/api

Author: AGI Team
Version: 1.0.0
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any, Set

from .base import (
    OpenAICompatibleProvider, LLMConfig, LLMResponse,
    ModelCapability,
)

logger = logging.getLogger(__name__)


class GLMFlashProvider(OpenAICompatibleProvider):
    """
    智谱AI GLM-4-Flash 系列 Provider

    GLM-4-Flash系列是完全免费的大语言模型，适合:
        - 日常对话和问答
        - 文本生成和创作
        - 代码辅助
        - 学习辅助
        - 轻量级应用集成

    特性:
        - 完全免费，无调用限制
        - 响应速度快
        - 支持长上下文
        - 支持流式输出
        - 支持Function Calling

    Example:
        ```python
        provider = GLMFlashProvider(LLMConfig(
            model_id="glm-4-flash",
            api_key="your_api_key"
        ))

        # 普通对话
        response = await provider.chat_completion([
            {"role": "user", "content": "你好，请介绍一下自己"}
        ])
        print(response.content)

        # 流式输出
        async for chunk in provider.stream_chat([
            {"role": "user", "content": "讲一个故事"}
        ]):
            print(chunk, end="", flush=True)
        ```
    """

    PROVIDER_NAME = "glm_flash"
    SUPPORTED_MODELS = {
        "glm-4-flash",
        "glm-4-flashx",
    }
    DEFAULT_MODEL = "glm-4-flash"

    # API配置
    API_BASE = "https://open.bigmodel.cn/api/paas/v4"
    CHAT_ENDPOINT = "/chat/completions"
    RATE_LIMIT_INTERVAL = 0.05

    # 模型特性配置
    MODEL_CONFIGS = {
        "glm-4-flash": {
            "max_tokens": 4096,
            "context_window": 128000,
            "supports_vision": False,
            "supports_function_calling": True,
            "supports_streaming": True,
        },
        "glm-4-flashx": {
            "max_tokens": 4096,
            "context_window": 128000,
            "supports_vision": False,
            "supports_function_calling": True,
            "supports_streaming": True,
        },
    }

    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        self._rate_limit_reset = 0

    def get_capabilities(self) -> Set[ModelCapability]:
        """获取当前模型支持的能力"""
        return {
            ModelCapability.STREAMING,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.JSON_MODE,
        }

    def _build_payload(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> Dict[str, Any]:
        """构建请求体（GLM-Flash支持seed和response_format）"""
        payload = super()._build_payload(messages, config)

        if config.seed is not None:
            payload["seed"] = config.seed

        if config.response_format:
            payload["response_format"] = config.response_format

        return payload

    async def _async_generate(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> LLMResponse:
        """异步生成响应（GLM-Flash自定义速率限制头处理）"""
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

                    # 更新速率限制信息
                    self._rate_limit_remaining = int(
                        response.headers.get("X-RateLimit-Remaining", 1000)
                    )

                    if response.status != 200:
                        error_text = await response.text()
                        error_data = json.loads(error_text) if error_text else {}
                        error_msg = error_data.get("error", {}).get("message", error_text)

                        return LLMResponse(
                            content="",
                            model_id=config.model_id,
                            error=f"HTTP {response.status}: {error_msg}",
                            latency_ms=(time.time() - start_time) * 1000,
                            metadata={
                                "status_code": response.status,
                                "error_code": error_data.get("error", {}).get("code")
                            }
                        )

                    result = await response.json()

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
                            "provider": self.PROVIDER_NAME,
                            "model": config.model_id,
                            "rate_limit_remaining": self._rate_limit_remaining,
                        }
                    )

        except json.JSONDecodeError as e:
            logger.error(f"GLM-Flash JSON decode error: {e}")
            return LLMResponse(
                content="",
                model_id=config.model_id,
                error=f"JSON decode error: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            logger.error(f"GLM-Flash error: {e}")
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
        """异步流式生成（GLM-Flash支持function_call delta）"""
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

                            # 检查是否有function call
                            if delta.get("function_call"):
                                yield json.dumps(delta["function_call"])

                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

        except Exception as e:
            logger.error(f"GLM-Flash stream error: {e}")
            yield f"Error: {str(e)}"

    def get_stats(self) -> Dict[str, Any]:
        """获取Provider统计信息"""
        stats = super().get_stats()
        stats["rate_limit_remaining"] = self._rate_limit_remaining
        return stats
