"""
API路由模块

提供所有API端点的路由定义。

子模块:
    - personality: 人格管理API
    - channel: 渠道管理API
    - message: 消息处理API
    - plugin: 插件管理API
    - routing: 路由规则API
    - metrics: 指标查询API
    - health: 健康检查API
    - dashboard: 仪表盘API
    - chat: 智能对话API
    - system: 系统管理API

使用示例:
    >>> from agi_unified_framework.api.routes import get_all_routers
    >>> routers = get_all_routers()
    >>> for router in routers:
    ...     app.include_router(router, prefix="/api/v1")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from fastapi import APIRouter

# 导入所有路由模块
from . import personality
from . import channel
from . import message
from . import plugin
from . import routing
from . import metrics
from . import health
from . import dashboard
from . import chat
from . import chat_integrated
from . import models
from . import orchestration
from . import workflows
from . import agents
from . import cognitive
from . import training
from . import system
from . import advanced
from . import algorithms  # 胜复学算法集成
from . import multiagent_algorithms  # 多智能体算法集成
from . import training_algorithms  # 训练算法集成
from . import swing_layer_algorithms  # Swing层算法集成
from . import reasoning_algorithms  # 推理算法集成

# 导入真实化改造后的新路由
from . import dashboard_real
from . import generation_real
from . import chat_multimodal
from . import computer_use_api
from . import telemetry
from . import hardware

# 获取路由器（用于测试或手动注册）
def get_personality_router() -> "APIRouter":
    """获取人格管理路由器"""
    return personality.router


def get_channel_router() -> "APIRouter":
    """获取渠道管理路由器"""
    return channel.router


def get_message_router() -> "APIRouter":
    """获取消息处理路由器"""
    return message.router


def get_plugin_router() -> "APIRouter":
    """获取插件管理路由器"""
    return plugin.router


def get_routing_router() -> "APIRouter":
    """获取路由规则路由器"""
    return routing.router


def get_metrics_router() -> "APIRouter":
    """获取指标查询路由器"""
    return metrics.router


def get_health_router() -> "APIRouter":
    """获取健康检查路由器"""
    return health.router


def get_dashboard_router() -> "APIRouter":
    """获取仪表盘路由器"""
    return dashboard.router


def get_chat_router() -> "APIRouter":
    """获取智能对话路由器"""
    return chat.router


def get_chat_integrated_router() -> "APIRouter":
    """获取集成算法对话路由器"""
    return chat_integrated.router


def get_models_router() -> "APIRouter":
    """获取模型管理路由器"""
    return models.router


def get_orchestration_router() -> "APIRouter":
    """获取模型编排路由器"""
    return orchestration.router


def get_workflows_router() -> "APIRouter":
    """获取工作流路由器"""
    return workflows.router


def get_agents_router() -> "APIRouter":
    """获取多智能体路由器"""
    return agents.router


def get_cognitive_router() -> "APIRouter":
    """获取认知系统路由器"""
    return cognitive.router


def get_training_router() -> "APIRouter":
    """获取训练中心路由器"""
    return training.router


def get_system_router() -> "APIRouter":
    """获取系统管理路由器"""
    return system.router


def get_all_routers() -> List["APIRouter"]:
    """
    获取所有API路由器
    
    Returns:
        List[APIRouter]: 所有路由器的列表
    """
    return [
        personality.router,
        channel.router,
        message.router,
        plugin.router,
        routing.router,
        metrics.router,
        health.router,
        dashboard.router,
        dashboard_real.router,  # 真实仪表盘API
        chat.router,
        models.router,
        orchestration.router,
        workflows.router,
        agents.router,
        cognitive.router,
        training.router,
        system.router,
        generation_real.router,  # 真实生成API
        chat_multimodal.router,  # 多模态聊天API
        computer_use_api.router,  # 计算机操作API
        algorithms.router,  # 胜复学算法API
        multiagent_algorithms.router,  # 多智能体算法API
        training_algorithms.router,  # 训练算法API
        swing_layer_algorithms.router,  # Swing层算法API
        reasoning_algorithms.router,  # 推理算法API
    ]


# 路由配置信息
ROUTES_INFO = {
    "personality": {
        "path": "/personality",
        "tags": ["Personality"],
        "description": "人格管理API - 创建、更新、删除和应用人格配置",
    },
    "channels": {
        "path": "/channels",
        "tags": ["Channels"],
        "description": "渠道管理API - 管理IM渠道连接",
    },
    "messages": {
        "path": "/messages",
        "tags": ["Messages"],
        "description": "消息处理API - 发送和查询消息",
    },
    "plugins": {
        "path": "/plugins",
        "tags": ["Plugins"],
        "description": "插件管理API - 安装、启用和管理插件",
    },
    "routing": {
        "path": "/routing",
        "tags": ["Routing"],
        "description": "路由规则API - 配置消息路由规则",
    },
    "metrics": {
        "path": "/metrics",
        "tags": ["Metrics"],
        "description": "指标查询API - 获取系统和业务指标",
    },
    "health": {
        "path": "/health",
        "tags": ["Health"],
        "description": "健康检查API - 系统和组件健康状态",
    },
    "dashboard": {
        "path": "/dashboard",
        "tags": ["Dashboard"],
        "description": "仪表盘API - 系统统计、实时指标和活动监控",
    },
    "chat": {
        "path": "/chat",
        "tags": ["Chat"],
        "description": "智能对话API - 对话管理和消息交互",
    },
    "models": {
        "path": "/models",
        "tags": ["Models"],
        "description": "模型管理API - AI模型的注册、配置和监控",
    },
    "orchestration": {
        "path": "/orchestration",
        "tags": ["Orchestration"],
        "description": "模型编排API - 策略管理、路由和负载均衡",
    },
    "workflows": {
        "path": "/workflows",
        "tags": ["Workflows"],
        "description": "工作流API - 工作流设计、执行和监控",
    },
    "agents": {
        "path": "/agents",
        "tags": ["Agents"],
        "description": "多智能体API - 智能体管理、联盟和辩论系统",
    },
    "cognitive": {
        "path": "/cognitive",
        "tags": ["Cognitive"],
        "description": "认知系统API - 认知状态、反思、记忆和目标管理",
    },
    "training": {
        "path": "/training",
        "tags": ["Training"],
        "description": "训练中心API - 训练任务、检查点和数据集管理",
    },
    "system": {
        "path": "/system",
        "tags": ["System"],
        "description": "系统管理API - 监控遥测、硬件管理、系统设置、帮助文档和系统维护",
    },
}


def get_all_routes_info() -> dict:
    """
    获取所有路由信息
    
    Returns:
        dict: 路由配置信息字典
    """
    return ROUTES_INFO.copy()


__all__ = [
    # 模块
    "personality",
    "channel",
    "message",
    "plugin",
    "routing",
    "metrics",
    "health",
    "dashboard",
    "chat",
    "models",
    "orchestration",
    "workflows",
    "agents",
    "cognitive",
    "training",
    "system",
    # 函数
    "get_personality_router",
    "get_channel_router",
    "get_message_router",
    "get_plugin_router",
    "get_routing_router",
    "get_metrics_router",
    "get_health_router",
    "get_dashboard_router",
    "get_chat_router",
    "get_models_router",
    "get_orchestration_router",
    "get_workflows_router",
    "get_agents_router",
    "get_cognitive_router",
    "get_training_router",
    "get_system_router",
    "get_all_routers",
    "get_all_routes_info",
    # 配置
    "ROUTES_INFO",
]
