"""
AGI Unified Framework - API Gateway Layer

API网关层模块，提供统一的REST API接口用于管理AGI系统的各个组件。

模块结构:
    - main: FastAPI主应用
    - routes: API路由定义
    - middleware: 中间件组件
    - validators: 请求/响应验证
    - dependencies: 依赖注入

主要功能:
    - 人格管理API
    - 渠道管理API
    - 消息处理API
    - 插件管理API
    - 路由规则API
    - 指标查询API
    - 健康检查API

使用示例:
    >>> from agi_unified_framework.api import create_app
    >>> app = create_app()
    >>> # 启动服务
    >>> import uvicorn
    >>> uvicorn.run(app, host="0.0.0.0", port=8000)

Author: AGI Team
Version: 1.0.0
License: Apache 2.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, Any, Callable
import logging

# 配置日志
logger = logging.getLogger(__name__)

# 版本信息
__version__ = "1.0.0"
__api_version__ = "v1"

# 导入主要组件
from .main import create_app, get_application, AGIAPIApplication
from .validators.schemas import (
    # 基础模型
    BaseResponse,
    ErrorResponse,
    PaginationParams,
    PaginatedResponse,
    # 人格模型
    PersonalityCreateRequest,
    PersonalityUpdateRequest,
    PersonalityResponse,
    PersonalityListResponse,
    PersonalityApplyRequest,
    # 渠道模型
    ChannelCreateRequest,
    ChannelUpdateRequest,
    ChannelResponse,
    ChannelListResponse,
    ChannelTestRequest,
    ChannelTestResponse,
    # 消息模型
    MessageCreateRequest,
    MessageResponse,
    MessageListResponse,
    MessageQueryParams,
    # 插件模型
    PluginInstallRequest,
    PluginResponse,
    PluginListResponse,
    PluginMarketplaceItem,
    # 路由规则模型
    RoutingRuleCreateRequest,
    RoutingRuleUpdateRequest,
    RoutingRuleResponse,
    RoutingRuleListResponse,
    RoutingTestRequest,
    RoutingTestResponse,
    # 指标模型
    MetricsOverviewResponse,
    MetricsLLMResponse,
    MetricsChannelResponse,
    MetricsCostResponse,
    # 健康检查模型
    HealthResponse,
    HealthReadyResponse,
    HealthLiveResponse,
)

# 导入依赖注入
from .dependencies.injection import (
    get_db_session,
    get_current_user,
    get_current_active_user,
    require_permissions,
    get_metrics_collector,
    get_channel_manager,
    get_personality_engine,
    get_plugin_manager,
    get_routing_engine,
    get_message_service,
)

# 导入中间件
from .middleware.auth import AuthMiddleware, JWTAuthBackend, APIKeyAuthBackend
from .middleware.rate_limit import RateLimitMiddleware
from .middleware.logging import LoggingMiddleware
from .middleware.cors import CORSMiddleware

if TYPE_CHECKING:
    from fastapi import FastAPI


# 全局应用实例
_app_instance: Optional["FastAPI"] = None


def get_app() -> "FastAPI":
    """
    获取全局FastAPI应用实例（单例模式）
    
    Returns:
        FastAPI: 应用实例
        
    Example:
        >>> app = get_app()
        >>> # 使用app进行测试或挂载
    """
    global _app_instance
    if _app_instance is None:
        _app_instance = create_app()
    return _app_instance


def reset_app() -> None:
    """
    重置全局应用实例
    
    主要用于测试场景，清除单例状态。
    """
    global _app_instance
    _app_instance = None
    logger.debug("API application instance reset")


# 快捷函数
def init_api(
    title: str = "AGI Unified Framework API",
    version: str = "1.0.0",
    debug: bool = False,
    **kwargs: Any
) -> "FastAPI":
    """
    初始化API网关
    
    Args:
        title: API标题
        version: API版本
        debug: 是否启用调试模式
        **kwargs: 其他配置参数
        
    Returns:
        FastAPI: 配置好的应用实例
        
    Example:
        >>> app = init_api(debug=True)
        >>> uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    global _app_instance
    _app_instance = create_app(
        title=title,
        version=version,
        debug=debug,
        **kwargs
    )
    return _app_instance


def get_api_info() -> Dict[str, Any]:
    """
    获取API信息
    
    Returns:
        Dict: 包含API元信息的字典
    """
    return {
        "name": "AGI Unified Framework API",
        "version": __version__,
        "api_version": __api_version__,
        "description": "Unified API Gateway for AGI System",
        "endpoints": {
            "personality": "/api/v1/personality",
            "channels": "/api/v1/channels",
            "messages": "/api/v1/messages",
            "plugins": "/api/v1/plugins",
            "routing": "/api/v1/routing",
            "metrics": "/api/v1/metrics",
            "health": "/api/v1/health",
        },
        "documentation": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json",
    }


# 导出列表
__all__ = [
    # 版本信息
    "__version__",
    "__api_version__",
    
    # 主应用
    "create_app",
    "get_application",
    "AGIAPIApplication",
    "get_app",
    "reset_app",
    "init_api",
    "get_api_info",
    
    # 基础模型
    "BaseResponse",
    "ErrorResponse",
    "PaginationParams",
    "PaginatedResponse",
    
    # 人格模型
    "PersonalityCreateRequest",
    "PersonalityUpdateRequest",
    "PersonalityResponse",
    "PersonalityListResponse",
    "PersonalityApplyRequest",
    
    # 渠道模型
    "ChannelCreateRequest",
    "ChannelUpdateRequest",
    "ChannelResponse",
    "ChannelListResponse",
    "ChannelTestRequest",
    "ChannelTestResponse",
    
    # 消息模型
    "MessageCreateRequest",
    "MessageResponse",
    "MessageListResponse",
    "MessageQueryParams",
    
    # 插件模型
    "PluginInstallRequest",
    "PluginResponse",
    "PluginListResponse",
    "PluginMarketplaceItem",
    
    # 路由规则模型
    "RoutingRuleCreateRequest",
    "RoutingRuleUpdateRequest",
    "RoutingRuleResponse",
    "RoutingRuleListResponse",
    "RoutingTestRequest",
    "RoutingTestResponse",
    
    # 指标模型
    "MetricsOverviewResponse",
    "MetricsLLMResponse",
    "MetricsChannelResponse",
    "MetricsCostResponse",
    
    # 健康检查模型
    "HealthResponse",
    "HealthReadyResponse",
    "HealthLiveResponse",
    
    # 依赖注入
    "get_db_session",
    "get_current_user",
    "get_current_active_user",
    "require_permissions",
    "get_metrics_collector",
    "get_channel_manager",
    "get_personality_engine",
    "get_plugin_manager",
    "get_routing_engine",
    "get_message_service",
    
    # 中间件
    "AuthMiddleware",
    "JWTAuthBackend",
    "APIKeyAuthBackend",
    "RateLimitMiddleware",
    "LoggingMiddleware",
    "CORSMiddleware",
]
