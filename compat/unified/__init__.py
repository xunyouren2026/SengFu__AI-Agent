"""
统一AI接口层

提供多后端统一客户端、模型路由、负载均衡、降级处理等功能，
支持在 llama.cpp、LocalAI、Ollama、TGI、vLLM 等后端之间无缝切换。

模块路径: compat/unified/__init__.py
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from .fallback_handler import FallbackHandler, FallbackPolicy, FallbackRule
from .load_balancer import LoadBalancer, LoadBalancingStrategy, BackendEndpoint
from .model_router import ModelRouter, RoutingRule, RoutingStrategy
from .unified_client import UnifiedClient, UnifiedClientConfig

__all__ = [
    "FallbackHandler",
    "FallbackPolicy",
    "FallbackRule",
    "LoadBalancer",
    "LoadBalancingStrategy",
    "BackendEndpoint",
    "ModelRouter",
    "RoutingRule",
    "RoutingStrategy",
    "UnifiedClient",
    "UnifiedClientConfig",
]

logger = logging.getLogger(__name__)
