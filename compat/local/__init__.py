"""
本地推理引擎兼容层

提供 llama.cpp、LocalAI、Ollama、TGI、vLLM 等本地推理后端的统一客户端封装。
每个后端客户端遵循相同的接口协议，支持同步/异步调用、流式输出、嵌入向量等。

模块路径: compat/local/__init__.py
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

from .llama_cpp import LlamaCppClient
from .localai import LocalAIClient
from .ollama import OllamaClient
from .text_generation_inference import TGIClient
from .vllm import VLLMClient

__all__ = [
    "BackendType",
    "LlamaCppClient",
    "LocalAIClient",
    "OllamaClient",
    "TGIClient",
    "VLLMClient",
    "create_local_client",
    "get_client_class",
]

logger = logging.getLogger(__name__)


class BackendType(str, Enum):
    """支持的本地推理后端类型枚举"""

    LLAMA_CPP = "llama_cpp"
    LOCALAI = "localai"
    OLLAMA = "ollama"
    TGI = "tgi"
    VLLM = "vllm"


_BACKEND_REGISTRY: Dict[BackendType, Type[Any]] = {
    BackendType.LLAMA_CPP: LlamaCppClient,
    BackendType.LOCALAI: LocalAIClient,
    BackendType.OLLAMA: OllamaClient,
    BackendType.TGI: TGIClient,
    BackendType.VLLM: VLLMClient,
}


def get_client_class(backend: Union[str, BackendType]) -> Type[Any]:
    """根据后端类型获取对应的客户端类。

    Args:
        backend: 后端类型名称或枚举值。

    Returns:
        对应的客户端类。

    Raises:
        ValueError: 不支持的后端类型。
    """
    if isinstance(backend, str):
        try:
            backend = BackendType(backend)
        except ValueError:
            raise ValueError(
                f"不支持的后端类型: {backend}。"
                f"可选值: {[b.value for b in BackendType]}"
            )
    client_cls = _BACKEND_REGISTRY.get(backend)
    if client_cls is None:
        raise ValueError(f"未注册的后端类型: {backend}")
    return client_cls


def create_local_client(
    backend: Union[str, BackendType],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 120.0,
    max_retries: int = 3,
    **kwargs: Any,
) -> Any:
    """工厂方法：创建本地推理后端客户端实例。

    Args:
        backend: 后端类型名称或枚举值。
        base_url: 服务基础 URL，各后端有不同默认值。
        api_key: API 密钥（部分后端不需要）。
        model: 默认模型名称。
        timeout: 请求超时时间（秒）。
        max_retries: 最大重试次数。
        **kwargs: 传递给客户端构造函数的额外参数。

    Returns:
        初始化完成的客户端实例。

    Raises:
        ValueError: 不支持的后端类型。
    """
    client_cls = get_client_class(backend)
    default_urls: Dict[BackendType, str] = {
        BackendType.LLAMA_CPP: "http://localhost:8080",
        BackendType.LOCALAI: "http://localhost:8080",
        BackendType.OLLAMA: "http://localhost:11434",
        BackendType.TGI: "http://localhost:8080",
        BackendType.VLLM: "http://localhost:8000",
    }
    resolved_url = base_url or default_urls.get(
        BackendType(backend) if isinstance(backend, str) else backend, "http://localhost:8080"
    )
    logger.info("创建本地推理客户端: backend=%s, url=%s", backend, resolved_url)
    client = client_cls(
        base_url=resolved_url,
        api_key=api_key,
        model=model,
        timeout=timeout,
        max_retries=max_retries,
        **kwargs,
    )
    return client


def list_available_backends() -> List[str]:
    """列出所有已注册的可用后端。

    Returns:
        后端类型名称列表。
    """
    return [b.value for b in BackendType]
