"""
LLM Providers模块 - 多模型适配器

该模块提供统一的大语言模型接口，封装各种模型提供方的API。

包含的Provider:
- OpenAI: GPT-4/3.5系列
- Anthropic: Claude系列
- 智谱AI: GLM系列
- 阿里云: 通义千问系列
- 月之暗面: Kimi系列
- DeepSeek: DeepSeek系列
- 本地模型: Ollama/vLLM等
- 百度文心一言: ERNIE系列
- 讯飞星火: Spark系列
- MiniMax: abab系列
- 零一万物: Yi系列
- 百川智能: Baichuan系列
- 智谱Flash: GLM-4-Flash系列 (免费)
- 字节豆包: Doubao系列
- 昆仑天工: Skywork系列
- WPS灵犀: Lingxi系列
- 小米Mimo: Mimo系列
- 悟道: WuDao系列
- Universal: 通用模型适配器 (通过配置接入任意模型)

Author: AGI Team
Version: 2.0.0
"""

from .base import (
    BaseLLMProvider,
    LLMConfig,
    LLMResponse,
    LLMError,
    ModelCapability,
)
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .zhipuai import ZhipuAIProvider
from .dashscope import DashScopeProvider
from .moonshot import MoonshotProvider
from .deepseek import DeepSeekProvider
from .local import LocalModelProvider
from .wenxin import WenxinProvider
from .spark import SparkProvider
from .minimax import MiniMaxProvider
from .yi import YiProvider
from .baichuan import BaichuanProvider

# 新增Provider
from .glm_flash import GLMFlashProvider
from .doubao import DoubaoProvider
from .skywork import SkyworkProvider
from .wps_lingxi import WPSLingxiProvider
from .mimo import MimoProvider
from .wudao import WuDaoProvider

from .universal import UniversalLLMProvider, UniversalModelConfig
from .model_registry import ModelRegistry
from .auth_handler import AuthHandler, AuthConfig


def create_provider(
    provider_name: str,
    config: LLMConfig = None,
    model_config: UniversalModelConfig = None,
) -> BaseLLMProvider:
    """
    创建Provider实例。

    Args:
        provider_name: Provider名称 (openai, anthropic, zhipuai, dashscope,
            moonshot, deepseek, local, wenxin, spark, minimax, yi, baichuan,
            glm_flash, doubao, skywork, wps_lingxi, mimo, wudao, universal)
        config: LLM基础配置 (用于传统Provider)
        model_config: 通用模型配置 (用于Universal Provider)

    Returns:
        Provider实例

    Example:
        ```python
        # 传统方式
        provider = create_provider("openai", LLMConfig(model_id="gpt-4"))

        # 通用适配器方式
        provider = create_provider("universal", model_config=UniversalModelConfig(
            model_id="zhipu/glm-4",
            api_base="https://open.bigmodel.cn/api/paas/v4",
            api_key="your-key",
        ))
        ```
    """
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "zhipuai": ZhipuAIProvider,
        "dashscope": DashScopeProvider,
        "moonshot": MoonshotProvider,
        "deepseek": DeepSeekProvider,
        "local": LocalModelProvider,
        "wenxin": WenxinProvider,
        "spark": SparkProvider,
        "minimax": MiniMaxProvider,
        "yi": YiProvider,
        "baichuan": BaichuanProvider,
        # 新增Provider
        "glm_flash": GLMFlashProvider,
        "doubao": DoubaoProvider,
        "skywork": SkyworkProvider,
        "wps_lingxi": WPSLingxiProvider,
        "mimo": MimoProvider,
        "wudao": WuDaoProvider,
    }

    provider_name_lower = provider_name.lower()

    # Universal Provider 特殊处理
    if provider_name_lower == "universal":
        if model_config:
            return UniversalLLMProvider(model_config=model_config)
        elif config:
            return UniversalLLMProvider(config=config)
        else:
            raise ValueError(
                "Universal provider requires either 'model_config' "
                "(UniversalModelConfig) or 'config' (LLMConfig) parameter."
            )

    provider_class = providers.get(provider_name_lower)

    if not provider_class:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Available providers: {list(providers.keys()) + ['universal']}"
        )

    return provider_class(config)


__all__ = [
    # 基础类
    "BaseLLMProvider",
    "LLMConfig",
    "LLMResponse",
    "LLMError",
    "ModelCapability",
    # Providers
    "OpenAIProvider",
    "AnthropicProvider",
    "ZhipuAIProvider",
    "DashScopeProvider",
    "MoonshotProvider",
    "DeepSeekProvider",
    "LocalModelProvider",
    "WenxinProvider",
    "SparkProvider",
    "MiniMaxProvider",
    "YiProvider",
    "BaichuanProvider",
    # 新增Providers
    "GLMFlashProvider",
    "DoubaoProvider",
    "SkyworkProvider",
    "WPSLingxiProvider",
    "MimoProvider",
    "WuDaoProvider",
    # 通用适配器
    "UniversalLLMProvider",
    "UniversalModelConfig",
    # 模型注册中心
    "ModelRegistry",
    # 认证处理器
    "AuthHandler",
    "AuthConfig",
    # 工厂函数
    "create_provider",
]
