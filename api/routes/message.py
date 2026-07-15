"""
消息处理API路由

提供消息的发送、查询和管理功能。

端点:
    GET    /           - 消息查询
    POST   /           - 发送消息
    GET    /{id}      - 消息详情
    DELETE /{id}      - 删除消息

使用示例:
    >>> # 发送消息
    >>> POST /api/v1/messages
    >>> {
    >>>     "content": "Hello, World!",
    >>>     "channel_id": "channel-123"
    >>> }
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func

from database.models import Message, get_utc_now
from ..validators.schemas import (
    ErrorResponse,
    MessageCreateRequest,
    MessageListResponse,
    MessageQueryParams,
    MessageResponse,
    MessageStatus,
    MessageType,
)
from ..dependencies.injection import get_current_user, require_permissions, get_db_session, DatabaseSession

logger = logging.getLogger(__name__)
router = APIRouter()


def _message_to_response(message: Message) -> MessageResponse:
    """转换数据库Message对象为响应模型"""
    metadata = message.metadata_json or {}
    return MessageResponse(
        id=str(message.id),
        content=message.content,
        message_type=metadata.get("message_type", MessageType.TEXT),
        status=metadata.get("status", MessageStatus.PENDING),
        channel_id=metadata.get("channel_id"),
        user_id=metadata.get("user_id"),
        session_id=metadata.get("session_id"),
        reply_to=metadata.get("reply_to"),
        attachments=message.attachments or [],
        metadata=metadata,
        priority=metadata.get("priority", 1),
        created_at=message.created_at,
        sent_at=metadata.get("sent_at"),
        delivered_at=metadata.get("delivered_at"),
        read_at=metadata.get("read_at"),
        error_info=metadata.get("error_info"),
    )


@router.get(
    "/",
    response_model=MessageListResponse,
    summary="消息查询",
    description="查询消息历史，支持多种过滤条件。",
)
async def list_messages(
    channel_id: Optional[str] = Query(None, description="渠道ID"),
    user_id: Optional[str] = Query(None, description="用户ID"),
    session_id: Optional[str] = Query(None, description="会话ID"),
    message_type: Optional[str] = Query(None, description="消息类型"),
    status: Optional[str] = Query(None, description="消息状态"),
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间"),
    keyword: Optional[str] = Query(None, description="关键词搜索"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> MessageListResponse:
    """查询消息列表"""
    try:
        query = db.query(Message).filter(Message.is_deleted == False)

        # 应用过滤条件 - 通过metadata_json过滤
        if channel_id:
            query = query.filter(Message.metadata_json["channel_id"].astext == channel_id)
        if user_id:
            query = query.filter(Message.metadata_json["user_id"].astext == user_id)
        if session_id:
            query = query.filter(Message.metadata_json["session_id"].astext == session_id)
        if message_type:
            query = query.filter(Message.metadata_json["message_type"].astext == message_type)
        if status:
            query = query.filter(Message.metadata_json["status"].astext == status)

        # 时间范围过滤
        if start_time:
            query = query.filter(Message.created_at >= start_time)
        if end_time:
            query = query.filter(Message.created_at <= end_time)

        # 关键词搜索
        if keyword:
            query = query.filter(Message.content.ilike(f"%{keyword}%"))

        # 按时间倒序排序
        query = query.order_by(Message.created_at.desc())

        # 计算总数
        total = query.count()
        total_pages = (total + page_size - 1) // page_size
        offset = (page - 1) * page_size
        messages = query.offset(offset).limit(page_size).all()

        data = [_message_to_response(m) for m in messages]

        return MessageListResponse(
            success=True,
            message=f"Retrieved {len(data)} messages",
            data=data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )
    except Exception as e:
        logger.error(f"Failed to list messages: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list messages: {str(e)}"
        )


@router.post(
    "/",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="发送消息",
    description="发送新消息到指定渠道或用户。",
)
async def create_message(
    request: MessageCreateRequest,
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(require_permissions(["message:create"])),
) -> MessageResponse:
    """发送消息"""
    try:
        logger.info(f"Creating message: channel={request.channel_id}, user={request.user_id}")

        now = get_utc_now()

        # 构建metadata，存储API特有的字段
        metadata = dict(request.metadata) if request.metadata else {}
        metadata["message_type"] = request.message_type.value if hasattr(request.message_type, 'value') else request.message_type
        metadata["status"] = MessageStatus.SENT
        metadata["channel_id"] = request.channel_id
        metadata["user_id"] = request.user_id
        metadata["session_id"] = request.session_id
        metadata["reply_to"] = request.reply_to
        metadata["priority"] = request.priority
        metadata["sent_at"] = now.isoformat(),
        metadata["created_by"] = current_user.get("id"),

        # 构建附件数据
        attachments_data = [a.dict() for a in request.attachments] if request.attachments else []

        # 创建数据库记录
        message = Message(
            conversation_id=1,  # 默认对话ID，可根据实际需求调整
            role="user",
            content=request.content,
            attachments=attachments_data,
            metadata_json=metadata,
            created_at=now,
            updated_at=now,
        )

        db.add(message)
        db.commit()
        db.refresh(message)

        logger.info(f"Message created: {message.id}")
        return _message_to_response(message)

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create message: {str(e)}"
        )


@router.get(
    "/{message_id}",
    response_model=MessageResponse,
    summary="消息详情",
)
async def get_message(
    message_id: str,
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> MessageResponse:
    """获取消息详情"""
    try:
        message = db.query(Message).filter(
            Message.id == int(message_id),
            Message.is_deleted == False,
        ).first()
        if not message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Message with ID '{message_id}' not found"
            )
        return _message_to_response(message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get message: {str(e)}"
        )


@router.delete(
    "/{message_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除消息",
)
async def delete_message(
    message_id: str,
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(require_permissions(["message:delete"])),
) -> None:
    """删除消息"""
    try:
        logger.info(f"Deleting message: {message_id}")

        message = db.query(Message).filter(
            Message.id == int(message_id),
            Message.is_deleted == False,
        ).first()

        if not message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Message not found"
            )

        # 软删除
        message.is_deleted = True
        db.commit()

        logger.info(f"Message deleted: {message_id}")

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete message: {str(e)}"
        )


@router.post(
    "/{message_id}/retry",
    response_model=MessageResponse,
    summary="重发消息",
)
async def retry_message(
    message_id: str,
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(require_permissions(["message:update"])),
) -> MessageResponse:
    """重发失败的消息"""
    try:
        message = db.query(Message).filter(
            Message.id == int(message_id),
            Message.is_deleted == False,
        ).first()
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")

        # 更新metadata中的状态
        metadata = message.metadata_json or {}
        metadata["status"] = MessageStatus.PENDING
        metadata["error_info"] = None

        # 模拟发送成功
        now = get_utc_now()
        metadata["status"] = MessageStatus.SENT
        metadata["sent_at"] = now.isoformat()

        message.metadata_json = metadata
        message.updated_at = now
        db.commit()
        db.refresh(message)

        logger.info(f"Message retried: {message_id}")
        return _message_to_response(message)

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/stats/overview",
    response_model=Dict[str, Any],
    summary="消息统计概览",
)
async def get_message_stats(
    db: DatabaseSession = Depends(get_db_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """获取消息统计信息"""
    try:
        # 统计各状态消息数（通过metadata_json中的status字段）
        total_messages = db.query(func.count(Message.id)).filter(
            Message.is_deleted == False
        ).scalar() or 0

        # 获取所有未删除消息的metadata
        messages = db.query(Message.metadata_json).filter(
            Message.is_deleted == False
        ).all()

        status_counts = {}
        for (metadata,) in messages:
            if metadata:
                s = metadata.get("status", "unknown")
                status_counts[s] = status_counts.get(s, 0) + 1

        return {
            "success": True,
            "total_messages": total_messages,
            "status_distribution": status_counts,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
