"""
小米 Mimo LLM Provider

Mimo是小米推出的AI大模型，针对IoT和边缘设备进行了优化。

支持的模型:
- mimo-chat: 通用对话模型
- mimo-edge: 边缘设备优化模型

API文档: https://api.xiaomi.ai/v1

Author: AGI Team
Version: 1.0.0
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import aiohttp

from .base import (
    OpenAICompatibleProvider, LLMConfig, LLMResponse,
)

logger = logging.getLogger(__name__)


class MimoDeviceType(Enum):
    """Mimo支持的设备类型"""
    MOBILE = "mobile"
    IOT = "iot"
    EDGE = "edge"
    CLOUD = "cloud"


@dataclass
class EdgeOptimizationConfig:
    """边缘设备优化配置"""
    quantization: str = "int8"  # int8, int4, fp16
    pruning: bool = True
    batch_size: int = 1
    max_latency_ms: int = 500


class MimoProvider(OpenAICompatibleProvider):
    """
    小米 Mimo LLM Provider

    Mimo大模型特点:
        - 轻量级设计，适合端侧部署
        - 针对IoT设备优化
        - 低延迟响应
        - 支持量化推理
        - 边缘-云协同

    适用场景:
        - 智能家居语音助手
        - 移动设备AI功能
        - 边缘计算场景
        - IoT设备交互
        - 低功耗设备

    Example:
        ```python
        provider = MimoProvider(LLMConfig(
            model_id="mimo-edge",
            api_key="your_api_key"
        ))

        # 普通对话
        response = await provider.chat_completion([
            {"role": "user", "content": "打开客厅的灯"}
        ])

        # 边缘优化推理
        response = await provider.edge_inference(
            messages=[{"role": "user", "content": "查询天气"}],
            device_type="iot"
        )
        ```
    """

    PROVIDER_NAME = "mimo"
    SUPPORTED_MODELS = {
        "mimo-chat",
        "mimo-edge",
    }
    DEFAULT_MODEL = "mimo-chat"

    # API配置
    API_BASE = "https://api.xiaomi.ai/v1"
    CHAT_ENDPOINT = "/chat/completions"
    EDGE_ENDPOINT = "/edge/inference"
    RATE_LIMIT_INTERVAL = 0.05

    # 模型特性配置
    MODEL_CONFIGS = {
        "mimo-chat": {
            "max_tokens": 2048,
            "context_window": 4096,
            "supports_vision": False,
            "supports_function_calling": True,
            "supports_streaming": True,
            "description": "通用对话模型",
            "optimized_for": MimoDeviceType.CLOUD,
        },
        "mimo-edge": {
            "max_tokens": 1024,
            "context_window": 2048,
            "supports_vision": False,
            "supports_function_calling": True,
            "supports_streaming": True,
            "description": "边缘设备优化模型",
            "optimized_for": MimoDeviceType.EDGE,
        },
    }

    # 设备类型配置
    DEVICE_CONFIGS = {
        MimoDeviceType.MOBILE: {
            "max_tokens": 1024,
            "quantization": "int8",
            "timeout": 5.0,
        },
        MimoDeviceType.IOT: {
            "max_tokens": 512,
            "quantization": "int4",
            "timeout": 3.0,
        },
        MimoDeviceType.EDGE: {
            "max_tokens": 1024,
            "quantization": "int8",
            "timeout": 5.0,
        },
        MimoDeviceType.CLOUD: {
            "max_tokens": 2048,
            "quantization": "fp16",
            "timeout": 10.0,
        },
    }

    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        self._edge_config = EdgeOptimizationConfig()

    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = super()._get_headers()
        headers["X-Client-Type"] = "python-sdk"
        return headers

    def _build_payload(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig,
        device_type: Optional[MimoDeviceType] = None
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

        # 根据设备类型调整参数
        if device_type:
            device_config = self.DEVICE_CONFIGS.get(device_type, self.DEVICE_CONFIGS[MimoDeviceType.CLOUD])
            max_tokens = min(
                config.max_tokens or model_config["max_tokens"],
                device_config["max_tokens"]
            )
            payload["max_tokens"] = max_tokens
            payload["quantization"] = device_config["quantization"]
        else:
            if config.max_tokens:
                payload["max_tokens"] = min(config.max_tokens, model_config["max_tokens"])

        if config.stop:
            payload["stop"] = config.stop

        if config.extra_body:
            if "tools" in config.extra_body:
                payload["tools"] = config.extra_body["tools"]
            if "device_id" in config.extra_body:
                payload["device_id"] = config.extra_body["device_id"]

        return payload

    async def edge_inference(
        self,
        messages: List[Dict[str, Any]],
        device_type: str = "edge",
        **kwargs
    ) -> LLMResponse:
        """
        边缘设备推理。

        Args:
            messages: 消息列表
            device_type: 设备类型 (mobile, iot, edge, cloud)
            **kwargs: 其他参数

        Returns:
            LLM响应对象
        """
        device = MimoDeviceType(device_type)
        device_config = self.DEVICE_CONFIGS.get(device, self.DEVICE_CONFIGS[MimoDeviceType.EDGE])

        start_time = time.time()
        self._apply_rate_limit()

        api_key = self._config.api_key if self._config else None
        if not api_key:
            return LLMResponse(
                content="",
                model_id=self.model_id,
                error="API key is required",
                latency_ms=(time.time() - start_time) * 1000
            )

        base_url = self._api_base

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Device-Type": device_type,
        }

        payload = self._build_payload(
            messages,
            LLMConfig(model_id="mimo-edge", api_key=api_key),
            device_type=device
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}{self.EDGE_ENDPOINT}",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=device_config["timeout"])
                ) as response:
                    self._request_count += 1

                    if response.status != 200:
                        error_text = await response.text()
                        return LLMResponse(
                            content="",
                            model_id="mimo-edge",
                            error=f"HTTP {response.status}: {error_text}",
                            latency_ms=(time.time() - start_time) * 1000
                        )

                    result = await response.json()

                    choice = result["choices"][0]
                    message = choice.get("message", {})
                    content = message.get("content", "")
                    usage = result.get("usage")

                    return LLMResponse(
                        content=content,
                        model_id="mimo-edge",
                        usage=usage,
                        latency_ms=(time.time() - start_time) * 1000,
                        metadata={
                            "provider": self.PROVIDER_NAME,
                            "device_type": device_type,
                            "quantization": device_config["quantization"],
                        }
                    )

        except Exception as e:
            logger.error(f"Mimo edge inference error: {e}")
            return LLMResponse(
                content="",
                model_id="mimo-edge",
                error=f"Error: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000
            )

    def get_device_recommendation(self, device_specs: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据设备规格获取模型推荐。

        Args:
            device_specs: 设备规格，包含memory, cpu, battery等

        Returns:
            推荐配置
        """
        memory_mb = device_specs.get("memory_mb", 1024)
        is_battery_powered = device_specs.get("battery_powered", False)

        if memory_mb < 512:
            return {
                "recommended_model": "mimo-edge",
                "quantization": "int4",
                "max_tokens": 256,
                "streaming": False,
                "reason": "Limited memory, using int4 quantization",
            }
        elif memory_mb < 2048:
            return {
                "recommended_model": "mimo-edge",
                "quantization": "int8",
                "max_tokens": 512,
                "streaming": True,
                "reason": "Moderate memory, using int8 quantization",
            }
        else:
            if is_battery_powered:
                return {
                    "recommended_model": "mimo-edge",
                    "quantization": "int8",
                    "max_tokens": 1024,
                    "streaming": True,
                    "reason": "Battery powered, optimizing for efficiency",
                }
            else:
                return {
                    "recommended_model": "mimo-chat",
                    "quantization": "fp16",
                    "max_tokens": 2048,
                    "streaming": True,
                    "reason": "Sufficient resources, using full model",
                }

    def get_stats(self) -> Dict[str, Any]:
        """获取Provider统计信息"""
        stats = super().get_stats()
        stats["edge_config"] = {
            "quantization": self._edge_config.quantization,
            "pruning": self._edge_config.pruning,
            "batch_size": self._edge_config.batch_size,
        }
        return stats
