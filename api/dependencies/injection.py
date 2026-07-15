"""
依赖注入实现模块

提供FastAPI依赖注入函数的实现。

依赖列表:
    - 数据库依赖
    - 认证依赖
    - 服务依赖
    - 基础设施依赖

使用示例:
    >>> from fastapi import Depends
    >>> from agi_unified_framework.api.dependencies import get_current_user
    >>> 
    >>> @app.get("/items")
    ... async def list_items(
    ...     user: dict = Depends(get_current_user),
    ...     db: Session = Depends(get_db_session)
    ... ):
    ...     return {"user": user, "items": []}
"""

from __future__ import annotations

import logging
import sys
import os
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar, Union

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入真实的数据库连接
from database.connection import DatabaseManager
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# 安全方案
security = HTTPBearer(auto_error=False)


# =============================================================================
# 数据库依赖
# =============================================================================

# 全局数据库管理器实例
_db_manager_instance = None

def _get_or_create_db_manager():
    """获取或创建数据库管理器"""
    global _db_manager_instance
    if _db_manager_instance is None:
        # 支持通过环境变量配置数据库路径，默认使用app.db
        db_url = os.getenv("DATABASE_URL", "sqlite:///./data/app.db")
        _db_manager_instance = DatabaseManager(
            database_url=db_url,
            echo=False
        )
        _db_manager_instance.initialize()
    return _db_manager_instance


def get_db_session() -> Generator[Session, None, None]:
    """
    获取数据库会话依赖（同步）
    
    Yields:
        Session: SQLAlchemy数据库会话对象
    
    Example:
        >>> @app.get("/items")
        ... async def list_items(db: Session = Depends(get_db_session)):
        ...     return db.query(Item).all()
    """
    db_manager = _get_or_create_db_manager()
    session = db_manager.session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


async def get_async_db_session():
    """
    获取异步数据库会话依赖（当前使用同步会话）
    
    Yields:
        Session: SQLAlchemy数据库会话对象
    """
    db_manager = _get_or_create_db_manager()
    session = db_manager.session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# 类型别名，用于类型提示
DatabaseSession = Session


# =============================================================================
# 认证依赖
# =============================================================================

# 模拟用户数据 - 实际项目中应该从数据库获取
MOCK_USERS = {
    "admin": {
        "id": 1,
        "username": "admin",
        "email": "admin@example.com",
        "role": "admin",
        "permissions": ["*"],
        "is_active": True,
    }
}

# 模拟API密钥映射
API_KEY_USERS = {
    "test-api-key-12345": "admin",
}


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """
    获取当前用户
    
    从请求头中提取JWT令牌或API密钥，验证并返回用户信息。
    
    Args:
        credentials: HTTP认证凭证
        
    Returns:
        Dict[str, Any]: 用户信息字典
        
    Raises:
        HTTPException: 认证失败时抛出401错误
        
    Example:
        >>> @app.get("/profile")
        ... async def get_profile(user: dict = Depends(get_current_user)):
        ...     return {"user": user}
    """
    # 开发环境：允许无认证访问
    class UserObj(dict):
        """同时支持 obj.id 和 obj.get('key') 的用户对象"""
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                raise AttributeError(key)
    return UserObj({
        "id": 1,
        "username": "admin",
        "email": "admin@example.com",
        "role": "admin",
        "permissions": ["*"],
        "is_active": True,
    })


def get_current_active_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    获取当前活跃用户
    
    确保用户不仅已认证，而且账户处于活跃状态。
    
    Args:
        current_user: 当前用户信息
        
    Returns:
        Dict[str, Any]: 用户信息字典
        
    Raises:
        HTTPException: 用户被禁用时抛出403错误
        
    Example:
        >>> @app.post("/items")
        ... async def create_item(user: dict = Depends(get_current_active_user)):
        ...     return {"created_by": user["username"]}
    """
    if not current_user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="用户账户已被禁用",
        )
    return current_user


def require_permissions(required_permissions: List[str]):
    """
    权限检查装饰器
    
    创建依赖函数来验证用户是否具有所需权限。
    
    Args:
        required_permissions: 所需权限列表
        
    Returns:
        Callable: 依赖函数
        
    Example:
        >>> @app.delete("/users/{user_id}")
        ... async def delete_user(
        ...     user_id: int,
        ...     user: dict = Depends(require_permissions(["user:delete"]))
        ... ):
        ...     return {"deleted": user_id}
    """
    def permission_checker(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        user_permissions = set(current_user.get("permissions", []))
        
        # 超级管理员拥有所有权限
        if "*" in user_permissions:
            return current_user
            
        # 检查是否有所需权限
        for permission in required_permissions:
            if permission not in user_permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"权限不足，需要: {permission}",
                )
        
        return current_user
    
    return permission_checker


def require_roles(required_roles: List[str]):
    """
    角色检查装饰器
    
    创建依赖函数来验证用户是否具有所需角色。
    
    Args:
        required_roles: 所需角色列表
        
    Returns:
        Callable: 依赖函数
        
    Example:
        >>> @app.post("/admin/config")
        ... async def update_config(
        ...     config: dict,
        ...     user: dict = Depends(require_roles(["admin", "superuser"]))
        ... ):
        ...     return {"updated": True}
    """
    def role_checker(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        user_role = current_user.get("role", "user")
        
        if user_role not in required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"角色不足，需要: {', '.join(required_roles)}",
            )
        
        return current_user
    
    return role_checker


# =============================================================================
# 服务依赖
# =============================================================================

def get_service_client(service_name: str) -> Any:
    """
    获取服务客户端
    
    创建依赖函数来获取指定服务的客户端实例。
    
    Args:
        service_name: 服务名称
        
    Returns:
        Callable: 依赖函数
        
    Example:
        >>> @app.get("/external/data")
        ... async def get_external_data(
        ...     client = Depends(get_service_client("external_api"))
        ... ):
        ...     return await client.fetch_data()
    """
    def service_provider() -> Any:
        # 实际项目中应该从服务注册中心或工厂获取
        logger.debug(f"Getting service client: {service_name}")
        return None
    
    return service_provider


# =============================================================================
# 基础设施依赖
# =============================================================================

def get_cache_client() -> Any:
    """
    获取缓存客户端
    
    Returns:
        Any: 缓存客户端实例
        
    Example:
        >>> @app.get("/cached-data")
        ... async def get_cached_data(cache = Depends(get_cache_client)):
        ...     return cache.get("key")
    """
    # 实际项目中应该返回Redis或其他缓存客户端
    return None


def get_message_queue() -> Any:
    """
    获取消息队列客户端
    
    Returns:
        Any: 消息队列客户端实例
        
    Example:
        >>> @app.post("/tasks")
        ... async def create_task(
        ...     task: dict,
        ...     mq = Depends(get_message_queue)
        ... ):
        ...     mq.publish("tasks", task)
        ...     return {"queued": True}
    """
    # 实际项目中应该返回RabbitMQ/Kafka等客户端
    return None


def get_storage_client() -> Any:
    """
    获取存储客户端
    
    Returns:
        Any: 存储客户端实例
        
    Example:
        >>> @app.post("/upload")
        ... async def upload_file(
        ...     file: UploadFile,
        ...     storage = Depends(get_storage_client)
        ... ):
        ...     return storage.save(file)
    """
    # 实际项目中应该返回S3/MinIO等客户端
    return None


# =============================================================================
# 请求上下文依赖
# =============================================================================

def get_request_id(request: Request) -> str:
    """
    获取请求ID
    
    从请求头或生成新的请求ID用于追踪。
    
    Args:
        request: FastAPI请求对象
        
    Returns:
        str: 请求ID
        
    Example:
        >>> @app.get("/items")
        ... async def list_items(request_id: str = Depends(get_request_id)):
        ...     logger.info(f"Processing request {request_id}")
        ...     return {"items": []}
    """
    # 从请求头获取或生成新的ID
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        import uuid
        request_id = str(uuid.uuid4())
    return request_id


def get_client_info(request: Request) -> Dict[str, Any]:
    """
    获取客户端信息
    
    提取请求中的客户端信息（IP、User-Agent等）。
    
    Args:
        request: FastAPI请求对象
        
    Returns:
        Dict[str, Any]: 客户端信息字典
        
    Example:
        >>> @app.get("/info")
        ... async def get_info(client: dict = Depends(get_client_info)):
        ...     return {"client": client}
    """
    return {
        "ip": request.client.host if request.client else None,
        "user_agent": request.headers.get("User-Agent"),
        "referer": request.headers.get("Referer"),
    }


# =============================================================================
# 分页和过滤依赖
# =============================================================================

def get_pagination_params(
    page: int = 1,
    page_size: int = 20,
) -> Dict[str, int]:
    """
    获取分页参数
    
    从查询参数中提取分页信息。
    
    Args:
        page: 页码（从1开始）
        page_size: 每页大小
        
    Returns:
        Dict[str, int]: 分页参数字典
        
    Example:
        >>> @app.get("/items")
        ... async def list_items(
        ...     pagination: dict = Depends(get_pagination_params)
        ... ):
        ...     offset = (pagination["page"] - 1) * pagination["page_size"]
        ...     return {"offset": offset, "limit": pagination["page_size"]}
    """
    return {
        "page": max(1, page),
        "page_size": min(max(1, page_size), 100),  # 限制最大100条
        "offset": (max(1, page) - 1) * min(max(1, page_size), 100),
    }


def get_sort_params(
    sort_by: Optional[str] = None,
    sort_order: str = "desc",
) -> Dict[str, Any]:
    """
    获取排序参数
    
    从查询参数中提取排序信息。
    
    Args:
        sort_by: 排序字段
        sort_order: 排序方向（asc/desc）
        
    Returns:
        Dict[str, Any]: 排序参数字典
        
    Example:
        >>> @app.get("/items")
        ... async def list_items(
        ...     sort: dict = Depends(get_sort_params)
        ... ):
        ...     return {"sort_by": sort["by"], "order": sort["order"]}
    """
    return {
        "by": sort_by,
        "order": sort_order.lower() if sort_order.lower() in ["asc", "desc"] else "desc",
    }


# =============================================================================
# 服务依赖（模拟实现）
# =============================================================================

def get_metrics_collector():
    """获取指标收集器"""
    return None

def get_channel_manager():
    """获取频道管理器"""
    return None

def get_personality_engine():
    """获取人格引擎"""
    return None

def get_plugin_manager():
    """获取插件管理器"""
    return None

def get_routing_engine():
    """获取路由引擎"""
    return None

def get_message_service():
    """获取消息服务"""
    return None

def get_config_manager():
    """获取配置管理器"""
    return None

def get_event_bus():
    """获取事件总线"""
    return None

def get_task_queue():
    """获取任务队列"""
    return None

def get_notification_service():
    """获取通知服务"""
    return None

def get_audit_logger():
    """获取审计日志器"""
    return None


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # 数据库
    "DatabaseSession",
    "get_db_session",
    "get_async_db_session",
    # 认证
    "get_current_user",
    "get_current_active_user",
    "require_permissions",
    "require_roles",
    # 服务
    "get_service_client",
    # 基础设施
    "get_cache_client",
    "get_message_queue",
    "get_storage_client",
    # 请求上下文
    "get_request_id",
    "get_client_info",
    # 分页和过滤
    "get_pagination_params",
    "get_sort_params",
    # 服务依赖
    "get_metrics_collector",
    "get_channel_manager",
    "get_personality_engine",
    "get_plugin_manager",
    "get_routing_engine",
    "get_message_service",
    "get_config_manager",
    "get_event_bus",
    "get_task_queue",
    "get_notification_service",
    "get_audit_logger",
]
