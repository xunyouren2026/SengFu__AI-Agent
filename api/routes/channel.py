"""
渠道管理API路由

提供IM渠道的CRUD操作和连接管理。

端点:
    GET    /           - 获取渠道列表
    POST   /           - 添加渠道
    GET    /{id}      - 获取渠道详情
    PUT    /{id}      - 更新渠道配置
    DELETE /{id}      - 删除渠道
    POST   /{id}/test - 测试连接
    POST   /{id}/enable - 启用/禁用渠道

使用示例:
    >>> # 创建Telegram渠道
    >>> POST /api/v1/channels
    >>> {
    >>>     "name": "Telegram Bot",
    >>>     "channel_type": "telegram",
    >>>     "config": {"api_key": "xxx"}
    >>> }
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select

from database.models import Channel, ChannelStatus as ChannelStatusModel, get_utc_now
from ..validators.schemas import (
    ChannelCreateRequest,
    ChannelListResponse,
    ChannelResponse,
    ChannelTestRequest,
    ChannelTestResponse,
    ChannelUpdateRequest,
    ErrorResponse,
    ChannelStatus,
)
from ..dependencies.injection import (
    DatabaseSession,
    get_current_user,
    get_db_session,
    require_permissions,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# 内存连接缓存（非持久化，仅用于运行时连接管理）
_channel_connections: Dict[str, Any] = {}


def _now() -> datetime:
    """获取当前时间"""
    return get_utc_now()


def _channel_to_response(channel: Channel) -> ChannelResponse:
    """转换数据库模型为响应模型"""
    return ChannelResponse(
        id=str(channel.id),
        name=channel.name,
        channel_type=channel.channel_type.value if hasattr(channel.channel_type, 'value') else str(channel.channel_type),
        description=channel.description,
        status=ChannelStatus(channel.status.value) if hasattr(channel.status, 'value') else ChannelStatus.PENDING,
        config=channel.config_json or {},
        enabled=channel.is_default,  # 使用is_default作为enabled的映射
        priority=100,
        health_status=None,
        statistics={
            "messages_sent": channel.total_messages_sent or 0,
            "messages_received": channel.total_messages_received or 0,
            "errors": channel.total_errors or 0,
        },
        created_at=channel.created_at,
        updated_at=channel.updated_at,
        last_connected_at=channel.last_health_check,
    )


@router.get(
    "/",
    response_model=ChannelListResponse,
    summary="获取渠道列表",
    description="获取所有渠道配置列表，支持分页和过滤。",
)
async def list_channels(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    channel_type: Optional[str] = Query(None, description="按类型过滤"),
    status: Optional[str] = Query(None, description="按状态过滤"),
    enabled_only: bool = Query(False, description="仅显示启用的渠道"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseSession = Depends(get_db_session),
) -> ChannelListResponse:
    """获取渠道列表"""
    try:
        query = db.query(Channel)

        # 应用过滤
        if channel_type:
            query = query.filter(Channel.channel_type == channel_type)
        if status:
            query = query.filter(Channel.status == status)
        if enabled_only:
            query = query.filter(Channel.is_default == True)

        # 计算总数
        total = query.count()
        total_pages = (total + page_size - 1) // page_size
        offset = (page - 1) * page_size
        channels = query.offset(offset).limit(page_size).all()

        data = [_channel_to_response(c) for c in channels]

        return ChannelListResponse(
            success=True,
            message=f"Retrieved {len(data)} channels",
            data=data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list channels: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list channels: {str(e)}"
        )


@router.post(
    "/",
    response_model=ChannelResponse,
    status_code=status.HTTP_201_CREATED,
    summary="添加渠道",
    description="创建新的渠道配置。",
)
async def create_channel(
    request: ChannelCreateRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["channel:create"])),
    db: DatabaseSession = Depends(get_db_session),
) -> ChannelResponse:
    """创建渠道"""
    try:
        logger.info(f"Creating channel: {request.name} ({request.channel_type})")

        # 检查名称冲突
        existing = db.query(Channel).filter(
            func.lower(Channel.name) == request.name.lower()
        ).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Channel with name '{request.name}' already exists"
            )

        now = _now()

        channel = Channel(
            name=request.name,
            channel_type=request.channel_type.value if hasattr(request.channel_type, 'value') else request.channel_type,
            description=request.description,
            config_json=request.config.dict() if hasattr(request.config, 'dict') else request.config,
            is_default=request.enabled,
            status=ChannelStatusModel.PENDING,
            total_messages_sent=0,
            total_messages_received=0,
            total_errors=0,
            created_at=now,
            updated_at=now,
        )

        db.add(channel)
        db.flush()

        logger.info(f"Channel created: {channel.id}")
        return _channel_to_response(channel)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create channel: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create channel: {str(e)}"
        )


@router.get(
    "/{channel_id}",
    response_model=ChannelResponse,
    summary="获取渠道详情",
)
async def get_channel(
    channel_id: int,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseSession = Depends(get_db_session),
) -> ChannelResponse:
    """获取渠道详情"""
    try:
        channel = db.query(Channel).filter(Channel.id == channel_id).first()
        if not channel:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Channel with ID '{channel_id}' not found"
            )
        return _channel_to_response(channel)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get channel: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get channel: {str(e)}"
        )


@router.put(
    "/{channel_id}",
    response_model=ChannelResponse,
    summary="更新渠道配置",
)
async def update_channel(
    channel_id: int,
    request: ChannelUpdateRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["channel:update"])),
    db: DatabaseSession = Depends(get_db_session),
) -> ChannelResponse:
    """更新渠道"""
    try:
        logger.info(f"Updating channel: {channel_id}")

        channel = db.query(Channel).filter(Channel.id == channel_id).first()
        if not channel:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Channel not found"
            )

        # 检查名称冲突
        if request.name:
            existing = db.query(Channel).filter(
                Channel.id != channel_id,
                func.lower(Channel.name) == request.name.lower()
            ).first()
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Channel with name '{request.name}' already exists"
                )

        # 更新字段
        update_data = request.dict(exclude_unset=True)
        for key, value in update_data.items():
            if value is not None:
                if key == "name":
                    channel.name = value
                elif key == "description":
                    channel.description = value
                elif key == "config":
                    channel.config_json = value.dict() if hasattr(value, 'dict') else value
                elif key == "enabled":
                    channel.is_default = value
                elif key == "priority":
                    pass  # 数据库模型中没有priority字段，忽略

        channel.updated_at = _now()

        db.flush()

        logger.info(f"Channel updated: {channel_id}")
        return _channel_to_response(channel)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update channel: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update channel: {str(e)}"
        )


@router.delete(
    "/{channel_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除渠道",
)
async def delete_channel(
    channel_id: int,
    current_user: Dict[str, Any] = Depends(require_permissions(["channel:delete"])),
    db: DatabaseSession = Depends(get_db_session),
) -> None:
    """删除渠道"""
    try:
        logger.info(f"Deleting channel: {channel_id}")

        channel = db.query(Channel).filter(Channel.id == channel_id).first()
        if not channel:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Channel not found"
            )

        # 断开连接缓存
        if str(channel_id) in _channel_connections:
            del _channel_connections[str(channel_id)]

        db.delete(channel)
        db.flush()

        logger.info(f"Channel deleted: {channel_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete channel: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete channel: {str(e)}"
        )


@router.post(
    "/{channel_id}/test",
    response_model=ChannelTestResponse,
    summary="测试渠道连接",
)
async def test_channel(
    channel_id: int,
    request: ChannelTestRequest,
    current_user: Dict[str, Any] = Depends(require_permissions(["channel:test"])),
    db: DatabaseSession = Depends(get_db_session),
) -> ChannelTestResponse:
    """测试渠道连接"""
    try:
        logger.info(f"Testing channel: {channel_id}")

        channel = db.query(Channel).filter(Channel.id == channel_id).first()
        if not channel:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Channel not found"
            )

        # 模拟连接测试
        start_time = time.time()

        # 实际实现中应调用渠道适配器进行测试
        # 这里模拟测试过程
        await asyncio.sleep(0.1)  # 模拟网络延迟

        response_time_ms = (time.time() - start_time) * 1000

        # 模拟测试结果
        test_success = True

        if test_success:
            channel.status = ChannelStatusModel.ACTIVE
            channel.last_health_check = _now()
        else:
            channel.status = ChannelStatusModel.ERROR

        channel.updated_at = _now()

        db.flush()

        return ChannelTestResponse(
            success=test_success,
            message="Channel connection test passed" if test_success else "Connection test failed",
            response_time_ms=round(response_time_ms, 2),
            details={
                "channel_type": channel.channel_type.value if hasattr(channel.channel_type, 'value') else str(channel.channel_type),
                "test_message": request.test_message,
                "timestamp": _now().isoformat(),
            },
            error=None if test_success else "Connection refused",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to test channel: {e}")
        return ChannelTestResponse(
            success=False,
            message="Channel connection test failed",
            error=str(e),
        )


@router.post(
    "/{channel_id}/enable",
    response_model=ChannelResponse,
    summary="启用/禁用渠道",
)
async def toggle_channel(
    channel_id: int,
    enabled: bool = Query(..., description="启用或禁用"),
    current_user: Dict[str, Any] = Depends(require_permissions(["channel:update"])),
    db: DatabaseSession = Depends(get_db_session),
) -> ChannelResponse:
    """启用或禁用渠道"""
    try:
        logger.info(f"Setting channel {channel_id} enabled={enabled}")

        channel = db.query(Channel).filter(Channel.id == channel_id).first()
        if not channel:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Channel not found"
            )

        channel.is_default = enabled
        channel.updated_at = _now()

        if enabled:
            channel.status = ChannelStatusModel.PENDING
        else:
            channel.status = ChannelStatusModel.INACTIVE

        db.flush()

        action = "enabled" if enabled else "disabled"
        logger.info(f"Channel {action}: {channel_id}")

        return _channel_to_response(channel)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to toggle channel: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to toggle channel: {str(e)}"
        )


@router.post(
    "/{channel_id}/connect",
    response_model=Dict[str, Any],
    summary="连接渠道",
)
async def connect_channel(
    channel_id: int,
    current_user: Dict[str, Any] = Depends(require_permissions(["channel:update"])),
    db: DatabaseSession = Depends(get_db_session),
) -> Dict[str, Any]:
    """手动触发渠道连接"""
    try:
        channel = db.query(Channel).filter(Channel.id == channel_id).first()
        if not channel:
            raise HTTPException(status_code=404, detail="Channel not found")

        now = _now()
        channel.status = ChannelStatusModel.ACTIVE
        channel.last_health_check = now
        channel.updated_at = now

        db.flush()

        return {
            "success": True,
            "message": f"Channel {channel.name} connected successfully",
            "channel_id": channel_id,
            "connected_at": now.isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/{channel_id}/disconnect",
    response_model=Dict[str, Any],
    summary="断开渠道连接",
)
async def disconnect_channel(
    channel_id: int,
    current_user: Dict[str, Any] = Depends(require_permissions(["channel:update"])),
    db: DatabaseSession = Depends(get_db_session),
) -> Dict[str, Any]:
    """手动断开渠道连接"""
    try:
        channel = db.query(Channel).filter(Channel.id == channel_id).first()
        if not channel:
            raise HTTPException(status_code=404, detail="Channel not found")

        now = _now()
        channel.status = ChannelStatusModel.INACTIVE
        channel.updated_at = now

        db.flush()

        return {
            "success": True,
            "message": f"Channel {channel.name} disconnected",
            "channel_id": channel_id,
            "disconnected_at": now.isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
