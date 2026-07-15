"""
依赖注入模块

提供FastAPI依赖注入函数。

主要组件:
    - get_db_session: 数据库会话依赖
    - get_current_user: 当前用户依赖
    - get_current_active_user: 当前活跃用户依赖
    - require_permissions: 权限检查依赖
    - get_*_service: 各种服务依赖

使用示例:
    >>> from agi_unified_framework.api.dependencies import get_current_user
    >>> @app.get("/profile")
    ... async def profile(user: dict = Depends(get_current_user)):
    ...     return user
"""

from __future__ import annotations

from .injection import (
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
    get_cache_client,
    get_config_manager,
    get_event_bus,
    get_task_queue,
    get_notification_service,
    get_audit_logger,
)

__all__ = [
    # 基础依赖
    "get_db_session",
    "get_current_user",
    "get_current_active_user",
    "require_permissions",
    # 服务依赖
    "get_metrics_collector",
    "get_channel_manager",
    "get_personality_engine",
    "get_plugin_manager",
    "get_routing_engine",
    "get_message_service",
    "get_cache_client",
    "get_config_manager",
    "get_event_bus",
    "get_task_queue",
    "get_notification_service",
    "get_audit_logger",
]
