"""
智能对话API路由

提供智能对话相关的API端点，包括对话管理和消息交互。

端点:
    GET /conversations - 获取对话列表
    POST /conversations - 创建新对话
    GET /conversations/{id} - 获取对话详情
    PUT /conversations/{id} - 更新对话
    DELETE /conversations/{id} - 删除对话
    GET /conversations/{id}/messages - 获取消息列表
    POST /conversations/{id}/messages - 发送消息
    POST /conversations/{id}/stream - 流式回复(SSE)
    DELETE /messages/{id} - 删除消息
    POST /conversations/{id}/clear - 清空对话
    POST /conversations/{id}/export - 导出对话
    POST /conversations/{id}/fork - 分叉对话
    POST /conversations/{id}/share - 分享对话
    POST /conversations/{id}/regenerate - 重新生成回复
    PUT /messages/{id} - 编辑消息
    POST /messages/{id}/rate - 评分消息
    GET /conversations/{id}/stats - 获取对话统计
    WebSocket /ws/chat/{conversation_id} - WebSocket实时对话

使用示例:
    >>> # 创建新对话
    >>> POST /api/v1/chat/conversations
    >>> {"title": "新对话", "model_name": "gpt-4"}
    >>>
    >>> # 发送消息
    >>> POST /api/v1/chat/conversations/123/messages
    >>> {"content": "你好", "role": "user"}
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

# LLM配置 - 从.env文件读取
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Dict, List, Optional, Set

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Path,
    Query,
    UploadFile,
    File,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy import func, asc, desc
from sqlalchemy.orm import Session

from database.models import (
    Conversation,
    Message,
    Model,
    User,
    get_utc_now,
)
from api.dependencies.injection import (
    DatabaseSession,
    get_current_user,
    get_db_session,
)
from api.validators.schemas import (
    BaseResponse,
    ErrorResponse,
    PaginatedResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Chat"])


# =============================================================================
# Pydantic模型定义
# =============================================================================

class ConversationCreate(BaseModel):
    """创建对话请求"""
    title: Optional[str] = Field(None, description="对话标题")
    model_name: Optional[str] = Field(None, description="模型名称")
    system_prompt: Optional[str] = Field(None, description="系统提示词")
    description: Optional[str] = Field(None, description="对话描述")
    tags: Optional[List[str]] = Field(default_factory=list, description="标签列表")
    category: Optional[str] = Field(None, description="分类")
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "新对话",
                "model_name": "gpt-4",
                "system_prompt": "你是一个有帮助的助手。",
                "tags": ["工作", "重要"]
            }
        }


class ConversationUpdate(BaseModel):
    """更新对话请求"""
    title: Optional[str] = Field(None, description="对话标题")
    description: Optional[str] = Field(None, description="对话描述")
    tags: Optional[List[str]] = Field(None, description="标签列表")
    category: Optional[str] = Field(None, description="分类")
    is_archived: Optional[bool] = Field(None, description="是否归档")
    is_pinned: Optional[bool] = Field(None, description="是否置顶")
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "更新后的标题",
                "tags": ["工作"]
            }
        }


class MessageCreate(BaseModel):
    """创建消息请求"""
    content: str = Field(..., min_length=1, description="消息内容")
    role: str = Field(default="user", description="角色(user/assistant/system)")
    parent_id: Optional[int] = Field(None, description="父消息ID")
    attachments: Optional[List[Dict]] = Field(default_factory=list, description="附件列表")
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "你好，请帮我分析这段代码",
                "role": "user"
            }
        }


class MessageUpdate(BaseModel):
    """更新消息请求"""
    content: Optional[str] = Field(None, min_length=1, description="消息内容")
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "更新后的消息内容"
            }
        }


class MessageRate(BaseModel):
    """消息评分请求"""
    rating: int = Field(..., ge=1, le=5, description="评分1-5")
    feedback: Optional[str] = Field(None, description="反馈文本")
    
    class Config:
        json_schema_extra = {
            "example": {
                "rating": 5,
                "feedback": "非常有用的回答"
            }
        }


class ConversationResponse(BaseModel):
    """对话响应"""
    id: int
    user_id: int
    title: str
    model_name: Optional[str]
    description: Optional[str]
    is_archived: bool
    is_pinned: bool
    is_bookmarked: bool
    tags: List[str]
    category: Optional[str]
    summary: Optional[str]
    message_count: int
    total_tokens: int
    total_cost: float
    created_at: datetime
    updated_at: datetime
    last_message_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class MessageResponse(BaseModel):
    """消息响应"""
    id: int
    conversation_id: int
    role: str
    content: str
    model_name: Optional[str]
    model_provider: Optional[str]
    parent_id: Optional[int]
    is_edited: bool
    is_deleted: bool
    is_pinned: bool
    is_flagged: bool
    attachments: List[Dict]
    images: List[str]
    tool_calls: Optional[List[Dict]]
    tool_results: Optional[List[Dict]]
    rating: Optional[int]
    feedback: Optional[str]
    cost: float
    latency_ms: Optional[int]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ConversationListResponse(PaginatedResponse):
    """对话列表响应"""
    data: List[ConversationResponse] = Field(default_factory=list, description="对话列表")


class MessageListResponse(PaginatedResponse):
    """消息列表响应"""
    data: List[MessageResponse] = Field(default_factory=list, description="消息列表")


class ConversationStats(BaseModel):
    """对话统计"""
    message_count: int
    user_message_count: int
    assistant_message_count: int
    total_tokens: int
    total_cost: float
    average_latency_ms: float
    first_message_at: Optional[datetime]
    last_message_at: Optional[datetime]
    duration_minutes: float


class StreamChunk(BaseModel):
    """流式响应块"""
    chunk: str = Field(..., description="内容块")
    index: int = Field(..., description="块索引")
    total: int = Field(..., description="总块数")
    done: bool = Field(..., description="是否完成")


# =============================================================================
# 辅助函数
# =============================================================================

def _conversation_to_response(conversation: Conversation) -> ConversationResponse:
    """将Conversation ORM对象转换为响应模型"""
    return ConversationResponse(
        id=conversation.id,
        user_id=conversation.user_id,
        title=conversation.title or "新对话",
        model_name=conversation.model_name,
        description=conversation.description,
        is_archived=conversation.is_archived or False,
        is_pinned=conversation.is_pinned or False,
        is_bookmarked=conversation.is_bookmarked or False,
        tags=conversation.tags or [],
        category=conversation.category,
        summary=conversation.summary,
        message_count=conversation.message_count or 0,
        total_tokens=conversation.total_tokens or 0,
        total_cost=float(conversation.total_cost or 0),
        created_at=conversation.created_at or get_utc_now(),
        updated_at=conversation.updated_at or get_utc_now(),
        last_message_at=conversation.last_message_at,
    )


def _message_to_response(message: Message) -> MessageResponse:
    """将Message ORM对象转换为响应模型"""
    return MessageResponse(
        id=message.id,
        conversation_id=message.conversation_id,
        role=message.role,
        content=message.content,
        model_name=message.model_name,
        model_provider=message.model_provider,
        parent_id=message.parent_id,
        is_edited=message.is_edited or False,
        is_deleted=message.is_deleted or False,
        is_pinned=message.is_pinned or False,
        is_flagged=message.is_flagged or False,
        attachments=message.attachments or [],
        images=message.images or [],
        tool_calls=message.tool_calls,
        tool_results=message.tool_results,
        rating=message.rating,
        feedback=message.feedback,
        cost=float(message.cost or 0),
        latency_ms=message.latency_ms,
        prompt_tokens=message.prompt_tokens or 0,
        completion_tokens=message.completion_tokens or 0,
        total_tokens=message.total_tokens or 0,
        created_at=message.created_at or get_utc_now(),
        updated_at=message.updated_at or get_utc_now(),
    )


def _estimate_tokens(text: str) -> int:
    """估算文本的token数量（简化版）"""
    if not text:
        return 0
    # 中文字符按1.5个token计算，英文按0.25计算
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    return int(chinese_chars * 1.5 + other_chars * 0.25)


async def _generate_ai_response(content: str, model_name: Optional[str], history: List[Dict]) -> str:
    """生成AI回复"""
    # 如果有OpenAI API密钥，调用真实API
    if OPENAI_API_KEY:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                messages = []
                for msg in history[-10:]:  # 只取最近10条
                    messages.append({"role": msg["role"], "content": msg["content"]})
                messages.append({"role": "user", "content": content})
                
                async with session.post(
                    f"{OPENAI_BASE_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    json={
                        "model": model_name or DEFAULT_MODEL,
                        "messages": messages,
                        "temperature": 0.7,
                    }
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"调用OpenAI API失败: {e}")
    
    # 降级到模拟回复
    return f"我理解您的问题：'{content[:50]}...'。作为一个AI助手，我会尽力帮助您。当前模型: {model_name or 'default'}"


def _calculate_conversation_stats(db: Session, conversation_id: int) -> ConversationStats:
    """计算对话统计信息"""
    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id,
        Message.is_deleted == False
    ).all()
    
    user_messages = [m for m in messages if m.role == "user"]
    assistant_messages = [m for m in messages if m.role == "assistant"]
    
    total_tokens = sum(m.total_tokens or 0 for m in messages)
    total_cost = sum(float(m.cost or 0) for m in messages)
    latencies = [m.latency_ms for m in messages if m.latency_ms]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    
    first_msg = min((m.created_at for m in messages if m.created_at), default=None)
    last_msg = max((m.created_at for m in messages if m.created_at), default=None)
    
    duration = 0
    if first_msg and last_msg:
        duration = (last_msg - first_msg).total_seconds() / 60
    
    return ConversationStats(
        message_count=len(messages),
        user_message_count=len(user_messages),
        assistant_message_count=len(assistant_messages),
        total_tokens=total_tokens,
        total_cost=total_cost,
        average_latency_ms=avg_latency,
        first_message_at=first_msg,
        last_message_at=last_msg,
        duration_minutes=duration,
    )


def _export_to_html(conversation: Conversation, messages: List[Message]) -> str:
    """导出对话为HTML格式"""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{conversation.title or '对话导出'}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        .message {{ margin: 10px 0; padding: 10px; border-radius: 8px; }}
        .user {{ background: #e3f2fd; text-align: right; }}
        .assistant {{ background: #f5f5f5; }}
        .meta {{ font-size: 12px; color: #666; margin-top: 5px; }}
    </style>
</head>
<body>
    <h1>{conversation.title or '对话导出'}</h1>
    <p>导出时间: {get_utc_now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <hr>
"""
    for msg in messages:
        role_class = "user" if msg.role == "user" else "assistant"
        html += f"""
    <div class="message {role_class}">
        <div>{msg.content}</div>
        <div class="meta">{msg.role} | {msg.created_at.strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>
"""
    html += """
</body>
</html>
"""
    return html


# =============================================================================
# API端点
# =============================================================================

@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    status: Optional[str] = Query(None, description="状态过滤(active/archived/all)"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    获取对话列表
    
    支持分页、状态过滤和搜索。
    """
    query = db.query(Conversation).filter(Conversation.user_id == current_user.id)
    
    if status == "archived":
        query = query.filter(Conversation.is_archived == True)
    elif status == "active":
        query = query.filter(Conversation.is_archived == False)
    
    if search:
        query = query.filter(Conversation.title.ilike(f"%{search}%"))
    
    total = query.count()
    conversations = query.order_by(desc(Conversation.is_pinned), desc(Conversation.updated_at)).offset((page - 1) * page_size).limit(page_size).all()
    
    return ConversationListResponse(
        data=[_conversation_to_response(c) for c in conversations],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("/conversations", response_model=BaseResponse[ConversationResponse])
async def create_conversation(
    request: ConversationCreate,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    创建新对话
    """
    now = get_utc_now()
    conversation = Conversation(
        user_id=current_user.id,
        title=request.title or "新对话",
        model_name=request.model_name or DEFAULT_MODEL,
        system_prompt=request.system_prompt,
        description=request.description,
        tags=request.tags or [],
        category=request.category,
        is_archived=False,
        is_pinned=False,
        is_bookmarked=False,
        message_count=0,
        total_tokens=0,
        total_cost=0.0,
        created_at=now,
        updated_at=now,
        last_message_at=None,
    )
    
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    
    return BaseResponse(
        success=True,
        message="对话创建成功",
        data=_conversation_to_response(conversation),
    )


@router.get("/conversations/{conversation_id}", response_model=BaseResponse[ConversationResponse])
async def get_conversation(
    conversation_id: int = Path(..., description="对话ID"),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    获取对话详情
    """
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    return BaseResponse(
        success=True,
        data=_conversation_to_response(conversation),
    )


@router.put("/conversations/{conversation_id}", response_model=BaseResponse[ConversationResponse])
async def update_conversation(
    request: ConversationUpdate,
    conversation_id: int = Path(..., description="对话ID"),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    更新对话
    """
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    if request.title is not None:
        conversation.title = request.title
    if request.description is not None:
        conversation.description = request.description
    if request.tags is not None:
        conversation.tags = request.tags
    if request.category is not None:
        conversation.category = request.category
    if request.is_archived is not None:
        conversation.is_archived = request.is_archived
    if request.is_pinned is not None:
        conversation.is_pinned = request.is_pinned
    
    conversation.updated_at = get_utc_now()
    db.commit()
    db.refresh(conversation)
    
    return BaseResponse(
        success=True,
        message="对话更新成功",
        data=_conversation_to_response(conversation),
    )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int = Path(..., description="对话ID"),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    删除对话
    """
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    # 级联删除消息
    db.query(Message).filter(Message.conversation_id == conversation_id).delete()
    db.delete(conversation)
    db.commit()
    
    return BaseResponse(success=True, message="对话已删除")


@router.get("/conversations/{conversation_id}/messages", response_model=MessageListResponse)
async def list_messages(
    conversation_id: int = Path(..., description="对话ID"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    获取对话消息列表
    """
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    query = db.query(Message).filter(
        Message.conversation_id == conversation_id,
        Message.is_deleted == False
    )
    
    total = query.count()
    messages = query.order_by(asc(Message.created_at)).offset((page - 1) * page_size).limit(page_size).all()
    
    return MessageListResponse(
        data=[_message_to_response(m) for m in messages],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("/conversations/{conversation_id}/messages", response_model=BaseResponse[MessageResponse])
async def send_message(
    request: MessageCreate,
    conversation_id: int = Path(..., description="对话ID"),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    发送消息
    
    创建用户消息并生成AI回复。
    """
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    now = get_utc_now()
    
    # 保存用户消息
    user_message = Message(
        conversation_id=conversation_id,
        role="user",
        content=request.content,
        parent_id=request.parent_id,
        attachments=request.attachments or [],
        prompt_tokens=_estimate_tokens(request.content),
        total_tokens=_estimate_tokens(request.content),
        created_at=now,
        updated_at=now,
    )
    db.add(user_message)
    db.flush()
    
    # 获取历史消息
    history = []
    history_messages = db.query(Message).filter(
        Message.conversation_id == conversation_id,
        Message.is_deleted == False
    ).order_by(asc(Message.created_at)).limit(20).all()
    
    for msg in history_messages:
        history.append({"role": msg.role, "content": msg.content})
    
    # 生成AI回复
    start_time = datetime.now()
    response_content = await _generate_ai_response(request.content, conversation.model_name, history)
    latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
    
    completion_tokens = _estimate_tokens(response_content)
    total_tokens = user_message.total_tokens + completion_tokens
    cost = _calculate_cost(user_message.total_tokens, completion_tokens, conversation.model_name or DEFAULT_MODEL)
    
    # 保存AI消息
    ai_message = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=response_content,
        model_name=conversation.model_name or DEFAULT_MODEL,
        model_provider="openai",
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost=cost,
        latency_ms=latency_ms,
        created_at=get_utc_now(),
        updated_at=get_utc_now(),
    )
    db.add(ai_message)
    db.flush()
    
    # 更新对话统计
    conversation.message_count = db.query(func.count(Message.id)).filter(
        Message.conversation_id == conversation_id,
        Message.is_deleted == False
    ).scalar() or 0
    conversation.total_tokens = db.query(func.coalesce(func.sum(Message.total_tokens), 0)).filter(
        Message.conversation_id == conversation_id,
        Message.is_deleted == False
    ).scalar() or 0
    conversation.last_message_at = get_utc_now()
    conversation.updated_at = get_utc_now()
    
    db.commit()
    db.refresh(ai_message)
    
    return BaseResponse(
        success=True,
        message="消息发送成功",
        data=_message_to_response(ai_message),
    )


@router.post("/conversations/{conversation_id}/stream")
async def stream_message(
    request: MessageCreate,
    conversation_id: int = Path(..., description="对话ID"),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    流式发送消息（SSE）
    
    创建用户消息并以流式方式返回AI回复。
    """
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    async def generate_stream():
        now = get_utc_now()
        
        # 保存用户消息
        user_message = Message(
            conversation_id=conversation_id,
            role="user",
            content=request.content,
            prompt_tokens=_estimate_tokens(request.content),
            total_tokens=_estimate_tokens(request.content),
            created_at=now,
            updated_at=now,
        )
        db.add(user_message)
        db.flush()
        
        yield f"data: {json.dumps({'type': 'user_message', 'data': _message_to_response(user_message).dict()})}\n\n"
        
        # 获取历史
        history = []
        history_messages = db.query(Message).filter(
            Message.conversation_id == conversation_id,
            Message.is_deleted == False
        ).order_by(asc(Message.created_at)).limit(20).all()
        
        for msg in history_messages:
            history.append({"role": msg.role, "content": msg.content})
        
        # 生成回复
        start_time = datetime.now()
        response_content = await _generate_ai_response(request.content, conversation.model_name, history)
        latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # 流式发送
        words = response_content.split()
        for i, word in enumerate(words):
            yield f"data: {json.dumps({'type': 'stream', 'chunk': word + ' ', 'index': i, 'total': len(words), 'done': False})}\n\n"
            await asyncio.sleep(0.03)
        
        # 保存AI消息
        completion_tokens = _estimate_tokens(response_content)
        total_tokens = user_message.total_tokens + completion_tokens
        cost = _calculate_cost(user_message.total_tokens, completion_tokens, conversation.model_name or DEFAULT_MODEL)
        
        ai_message = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=response_content,
            model_name=conversation.model_name or DEFAULT_MODEL,
            model_provider="openai",
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
            latency_ms=latency_ms,
            created_at=get_utc_now(),
            updated_at=get_utc_now(),
        )
        db.add(ai_message)
        db.flush()
        
        # 更新对话
        conversation.message_count = db.query(func.count(Message.id)).filter(
            Message.conversation_id == conversation_id,
            Message.is_deleted == False
        ).scalar() or 0
        conversation.last_message_at = get_utc_now()
        conversation.updated_at = get_utc_now()
        db.commit()
        
        yield f"data: {json.dumps({'type': 'stream', 'chunk': '', 'done': True, 'message': _message_to_response(ai_message).dict()})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.put("/messages/{message_id}", response_model=BaseResponse[MessageResponse])
async def update_message(
    request: MessageUpdate,
    message_id: int = Path(..., description="消息ID"),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    编辑消息
    """
    message = db.query(Message).join(Conversation).filter(
        Message.id == message_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not message:
        raise HTTPException(status_code=404, detail="消息不存在")
    
    if request.content:
        message.content = request.content
        message.is_edited = True
        message.updated_at = get_utc_now()
        db.commit()
        db.refresh(message)
    
    return BaseResponse(
        success=True,
        message="消息已更新",
        data=_message_to_response(message),
    )


@router.delete("/messages/{message_id}")
async def delete_message(
    message_id: int = Path(..., description="消息ID"),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    删除消息（软删除）
    """
    message = db.query(Message).join(Conversation).filter(
        Message.id == message_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not message:
        raise HTTPException(status_code=404, detail="消息不存在")
    
    message.is_deleted = True
    message.updated_at = get_utc_now()
    db.commit()
    
    return BaseResponse(success=True, message="消息已删除")


@router.post("/messages/{message_id}/rate")
async def rate_message(
    request: MessageRate,
    message_id: int = Path(..., description="消息ID"),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    为消息评分
    """
    message = db.query(Message).join(Conversation).filter(
        Message.id == message_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not message:
        raise HTTPException(status_code=404, detail="消息不存在")
    
    message.rating = request.rating
    message.feedback = request.feedback
    message.updated_at = get_utc_now()
    db.commit()
    
    return BaseResponse(success=True, message="评分已保存")


@router.post("/conversations/{conversation_id}/clear")
async def clear_conversation(
    conversation_id: int = Path(..., description="对话ID"),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    清空对话消息
    """
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    # 软删除所有消息
    db.query(Message).filter(Message.conversation_id == conversation_id).update({
        Message.is_deleted: True,
        Message.updated_at: get_utc_now(),
    })
    
    conversation.message_count = 0
    conversation.total_tokens = 0
    conversation.updated_at = get_utc_now()
    db.commit()
    
    return BaseResponse(success=True, message="对话已清空")


@router.post("/conversations/{conversation_id}/export")
async def export_conversation(
    conversation_id: int = Path(..., description="对话ID"),
    format: str = Query("html", description="导出格式(html/json)"),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    导出对话
    """
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id,
        Message.is_deleted == False
    ).order_by(asc(Message.created_at)).all()
    
    if format == "json":
        data = {
            "conversation": _conversation_to_response(conversation).dict(),
            "messages": [_message_to_response(m).dict() for m in messages],
        }
        return {
            "success": True,
            "data": data,
            "filename": f"conversation_{conversation_id}.json",
        }
    else:
        html = _export_to_html(conversation, messages)
        return {
            "success": True,
            "data": {"html": html},
            "filename": f"conversation_{conversation_id}.html",
        }


@router.post("/conversations/{conversation_id}/fork", response_model=BaseResponse[ConversationResponse])
async def fork_conversation(
    conversation_id: int = Path(..., description="对话ID"),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    分叉对话（创建副本）
    """
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    # 获取原对话的消息
    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id,
        Message.is_deleted == False
    ).all()
    
    now = get_utc_now()
    
    # 创建新对话
    new_conversation = Conversation(
        user_id=current_user.id,
        title=f"{conversation.title or '对话'} (副本)",
        model_name=conversation.model_name,
        system_prompt=conversation.system_prompt,
        description=conversation.description,
        tags=conversation.tags,
        category=conversation.category,
        created_at=now,
        updated_at=now,
    )
    db.add(new_conversation)
    db.flush()
    
    # 复制消息
    for msg in messages:
        new_msg = Message(
            conversation_id=new_conversation.id,
            role=msg.role,
            content=msg.content,
            model_name=msg.model_name,
            model_provider=msg.model_provider,
            attachments=msg.attachments,
            prompt_tokens=msg.prompt_tokens,
            completion_tokens=msg.completion_tokens,
            total_tokens=msg.total_tokens,
            cost=msg.cost,
            created_at=msg.created_at,
            updated_at=msg.updated_at,
        )
        db.add(new_msg)
    
    new_conversation.message_count = len(messages)
    db.commit()
    db.refresh(new_conversation)
    
    return BaseResponse(
        success=True,
        message="对话已分叉",
        data=_conversation_to_response(new_conversation),
    )


@router.get("/conversations/{conversation_id}/stats", response_model=BaseResponse[ConversationStats])
async def get_conversation_stats(
    conversation_id: int = Path(..., description="对话ID"),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    获取对话统计
    """
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    stats = _calculate_conversation_stats(db, conversation_id)
    
    return BaseResponse(success=True, data=stats)


@router.post("/conversations/{conversation_id}/regenerate", response_model=BaseResponse[MessageResponse])
async def regenerate_message(
    conversation_id: int = Path(..., description="对话ID"),
    message_id: Optional[int] = Query(None, description="要重新生成的消息ID"),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    重新生成AI回复
    """
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    # 获取最后一条用户消息
    last_user_message = db.query(Message).filter(
        Message.conversation_id == conversation_id,
        Message.role == "user",
        Message.is_deleted == False
    ).order_by(desc(Message.created_at)).first()
    
    if not last_user_message:
        raise HTTPException(status_code=400, detail="没有可重新生成的用户消息")
    
    # 获取历史
    history = []
    history_messages = db.query(Message).filter(
        Message.conversation_id == conversation_id,
        Message.is_deleted == False,
        Message.created_at <= last_user_message.created_at
    ).order_by(asc(Message.created_at)).all()
    
    for msg in history_messages[:-1]:  # 排除最后一条用户消息
        history.append({"role": msg.role, "content": msg.content})
    
    # 生成新回复
    start_time = datetime.now()
    response_content = await _generate_ai_response(last_user_message.content, conversation.model_name, history)
    latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
    
    completion_tokens = _estimate_tokens(response_content)
    total_tokens = last_user_message.total_tokens + completion_tokens
    cost = _calculate_cost(last_user_message.total_tokens, completion_tokens, conversation.model_name or DEFAULT_MODEL)
    
    # 保存新消息
    ai_message = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=response_content,
        model_name=conversation.model_name or DEFAULT_MODEL,
        model_provider="openai",
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost=cost,
        latency_ms=latency_ms,
        created_at=get_utc_now(),
        updated_at=get_utc_now(),
    )
    db.add(ai_message)
    db.flush()
    
    # 更新对话统计
    conversation.message_count = db.query(func.count(Message.id)).filter(
        Message.conversation_id == conversation_id,
        Message.is_deleted == False
    ).scalar() or 0
    conversation.updated_at = get_utc_now()
    db.commit()
    db.refresh(ai_message)
    
    return BaseResponse(
        success=True,
        message="消息已重新生成",
        data=_message_to_response(ai_message),
    )


@router.post("/conversations/batch-delete")
async def batch_delete_conversations(
    ids: List[int],
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    批量删除对话
    """
    deleted = 0
    for cid in ids:
        conversation = db.query(Conversation).filter(
            Conversation.id == cid,
            Conversation.user_id == current_user.id
        ).first()
        
        if conversation:
            db.query(Message).filter(Message.conversation_id == cid).delete()
            db.delete(conversation)
            deleted += 1
    
    db.commit()
    return BaseResponse(success=True, message=f"已删除 {deleted} 个对话")


@router.get("/conversations/search")
async def search_conversations(
    q: str = Query(..., description="搜索关键词"),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    """
    搜索对话
    """
    # 搜索对话标题
    conversations = db.query(Conversation).filter(
        Conversation.user_id == current_user.id,
        Conversation.title.ilike(f"%{q}%")
    ).all()
    
    # 搜索消息内容
    messages = db.query(Message).join(Conversation).filter(
        Conversation.user_id == current_user.id,
        Message.content.ilike(f"%{q}%"),
        Message.is_deleted == False
    ).all()
    
    # 获取相关对话
    conversation_ids = {m.conversation_id for m in messages}
    related_conversations = db.query(Conversation).filter(
        Conversation.id.in_(conversation_ids)
    ).all()
    
    all_conversations = list({c.id: c for c in conversations + related_conversations}.values())
    
    return BaseResponse(
        success=True,
        data=[_conversation_to_response(c) for c in all_conversations],
    )


# =============================================================================
# WebSocket端点
# =============================================================================

@router.websocket("/ws/chat/{conversation_id}")
async def chat_websocket(websocket: WebSocket, conversation_id: int):
    """
    对话WebSocket连接
    
    提供实时对话功能，支持发送消息和接收AI回复。
    """
    await websocket.accept()
    logger.info(f"Chat WebSocket connected for conversation {conversation_id}")
    
    # 注意：WebSocket端点不通过依赖注入获取db session，需要手动创建
    from database.connection import SessionLocal
    db = SessionLocal()
    
    try:
        while True:
            # 接收消息
            data = await websocket.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "message":
                content = data.get("content", "")
                role = data.get("role", "user")
                
                if not content.strip():
                    await websocket.send_json({
                        "type": "error",
                        "message": "消息内容不能为空",
                    })
                    continue
                
                # 检查对话是否存在
                conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
                if not conversation:
                    await websocket.send_json({
                        "type": "error",
                        "message": "对话不存在",
                    })
                    continue
                
                # 保存用户消息
                now = get_utc_now()
                user_message = Message(
                    conversation_id=conversation_id,
                    role=role,
                    content=content,
                    prompt_tokens=_estimate_tokens(content),
                    total_tokens=_estimate_tokens(content),
                    created_at=now,
                    updated_at=now,
                )
                db.add(user_message)
                db.flush()
                
                # 更新对话
                conversation.last_message_at = now
                conversation.updated_at = now
                db.commit()
                db.refresh(user_message)
                
                await websocket.send_json({
                    "type": "message",
                    "data": _message_to_response(user_message).dict(),
                })
                
                # 模拟AI流式回复
                if role == "user":
                    model_name = conversation.model_name
                    
                    # 构建对话历史
                    ws_history = []
                    history_messages = db.query(Message).filter(
                        Message.conversation_id == conversation_id,
                        Message.is_deleted == False
                    ).order_by(asc(Message.created_at)).limit(20).all()
                    
                    for msg in history_messages:
                        ws_history.append({"role": msg.role, "content": msg.content})
                    
                    response = await _generate_ai_response(content, model_name, ws_history)
                    words = response.split()
                    
                    # 发送流式数据
                    for i, word in enumerate(words):
                        await websocket.send_json({
                            "type": "stream",
                            "chunk": word + " ",
                            "index": i,
                            "total": len(words),
                            "done": False,
                        })
                        await asyncio.sleep(0.03)
                    
                    # 保存AI消息
                    ai_message = Message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=response,
                        model_name=model_name or "gpt-4",
                        model_provider="openai",
                        completion_tokens=_estimate_tokens(response),
                        total_tokens=_estimate_tokens(response),
                        latency_ms=len(words) * 30,
                        created_at=get_utc_now(),
                        updated_at=get_utc_now(),
                    )
                    db.add(ai_message)
                    db.flush()
                    
                    # 更新对话统计
                    conversation.message_count = db.query(func.count(Message.id)).filter(
                        Message.conversation_id == conversation_id,
                        Message.is_deleted == False
                    ).scalar() or 0
                    conversation.total_tokens = db.query(func.coalesce(func.sum(Message.total_tokens), 0)).filter(
                        Message.conversation_id == conversation_id,
                        Message.is_deleted == False
                    ).scalar() or 0
                    db.commit()
                    db.refresh(ai_message)
                    
                    await websocket.send_json({
                        "type": "stream",
                        "chunk": "",
                        "done": True,
                        "message": _message_to_response(ai_message).dict(),
                    })
            
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong", "timestamp": get_utc_now().isoformat()})
            
            elif msg_type == "typing":
                # 用户正在输入，可以广播给其他客户端
                await websocket.send_json({"type": "typing_ack"})
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"未知的消息类型: {msg_type}",
                })
                
    except WebSocketDisconnect:
        logger.info(f"Chat WebSocket disconnected for conversation {conversation_id}")
    except Exception as e:
        logger.error(f"Chat WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass
    finally:
        db.close()


# =============================================================================
# ChatStorage - 对话存储工具类（供 chat_integrated 使用）
# =============================================================================

class ChatStorage:
    """对话存储工具类，提供对话和消息的CRUD操作"""

    def _get_session(self):
        from database.connection import get_db_manager
        return get_db_manager().get_session()

    def create_conversation(self, data: dict) -> dict:
        """创建对话"""
        from database.models import Conversation
        db = self._get_session()
        try:
            conv = Conversation(
                user_id=data.get("user_id"),
                title=data.get("title", "新对话"),
                model_name=data.get("model_name"),
                system_prompt=data.get("system_prompt"),
                description=data.get("description"),
                tags=data.get("tags", []),
                category=data.get("category"),
                config=data.get("config"),
            )
            db.add(conv)
            db.commit()
            db.refresh(conv)
            return {
                "id": conv.id,
                "title": conv.title,
                "model_name": conv.model_name,
                "system_prompt": conv.system_prompt,
                "created_at": str(conv.created_at) if conv.created_at else None,
            }
        except Exception as e:
            db.rollback()
            logger.error(f"创建对话失败: {e}")
            raise
        finally:
            db.close()

    def get_conversation(self, conversation_id) -> dict:
        """获取对话"""
        from database.models import Conversation
        db = self._get_session()
        try:
            conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
            if not conv:
                return None
            return {
                "id": conv.id,
                "title": conv.title,
                "model_name": conv.model_name,
                "system_prompt": conv.system_prompt,
                "description": conv.description,
                "tags": conv.tags or [],
                "category": conv.category,
                "created_at": str(conv.created_at) if conv.created_at else None,
            }
        finally:
            db.close()

    def create_message(self, conversation_id, data: dict) -> dict:
        """创建消息"""
        from database.models import Message
        db = self._get_session()
        try:
            msg = Message(
                conversation_id=conversation_id,
                role=data.get("role", "user"),
                content=data.get("content", ""),
                model_name=data.get("model_name"),
                parent_id=data.get("parent_id"),
            )
            db.add(msg)
            db.commit()
            db.refresh(msg)
            return {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "model_name": msg.model_name,
                "parent_id": msg.parent_id,
                "created_at": str(msg.created_at) if msg.created_at else None,
            }
        except Exception as e:
            db.rollback()
            logger.error(f"创建消息失败: {e}")
            raise
        finally:
            db.close()


# 全局实例（供 chat_integrated 导入使用）
chat_storage = ChatStorage()
