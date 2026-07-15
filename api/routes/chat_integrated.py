"""
智能对话API路由 - 集成高级算法版本
=====================================

本版本集成了UFO框架的高级算法模块：
- ChatMemorySystem: 多轮对话记忆管理
- HierarchicalContextMemory: 分层记忆系统
- AdaptiveContextCompressor: 自适应上下文压缩
- ModelGateway: 统一模型路由网关
- RAGPipeline: 检索增强生成

相比原版chat.py，本版本：
1. 使用高级记忆系统替代简单SQL查询
2. 使用ModelGateway替代直接HTTP调用
3. 集成RAG支持文档问答
4. 自动上下文压缩节省Token
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

import asyncio
import hashlib
import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Union

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

# =============================================================================
# 导入UFO高级算法模块（核心集成点）
# =============================================================================

try:
    from core.long_context import (
        ChatMemorySystem,
        HierarchicalContextMemory,
        AdaptiveContextCompressor,
        ContextChunk,
    )
    from core.llm import ModelGateway
    from rag import RAGPipeline, RAGResult
    ADVANCED_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"高级算法模块导入失败: {e}")
    ADVANCED_MODULES_AVAILABLE = False

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
router = APIRouter(tags=["Chat Integrated"])


# =============================================================================
# 集成配置
# =============================================================================

class IntegratedChatConfig:
    """集成版聊天配置"""
    
    # 记忆系统配置
    MAX_HISTORY_TURNS = 50
    SHORT_TERM_CAPACITY = 10
    MEDIUM_TERM_CAPACITY = 100
    LONG_TERM_CAPACITY = 1000
    COMPRESSION_RATIO = 0.5
    
    # 压缩配置
    ENABLE_COMPRESSION = True
    COMPRESSION_THRESHOLD = 4000  # Token阈值，超过则压缩
    TARGET_COMPRESSED_TOKENS = 2000
    
    # RAG配置
    ENABLE_RAG = True
    RAG_TOP_K = 5
    RAG_SIMILARITY_THRESHOLD = 0.7
    
    # 模型网关配置
    DEFAULT_BACKEND = "openai"
    ENABLE_FALLBACK = True
    
    # LLM配置 - 从.env文件读取
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")


# =============================================================================
# Pydantic模型定义
# =============================================================================

class ConversationCreateRequest(BaseModel):
    """创建对话请求"""
    title: Optional[str] = Field(default="新对话", max_length=255, description="对话标题")
    model_name: Optional[str] = Field(default=None, description="使用的模型名称")
    system_prompt: Optional[str] = Field(default=None, description="系统提示词")
    description: Optional[str] = Field(default=None, description="对话描述")
    tags: Optional[List[str]] = Field(default_factory=list, description="标签")
    category: Optional[str] = Field(default=None, description="分类")
    config: Optional[Dict[str, Any]] = Field(default=None, description="对话配置")
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2, description="温度参数")
    max_tokens: Optional[int] = Field(default=4096, ge=1, le=8192, description="最大token数")
    enable_rag: Optional[bool] = Field(default=True, description="是否启用RAG")
    enable_compression: Optional[bool] = Field(default=True, description="是否启用上下文压缩")


class MessageCreateRequest(BaseModel):
    """创建消息请求"""
    content: str = Field(..., min_length=1, description="消息内容")
    role: str = Field(default="user", description="消息角色")
    attachments: Optional[List[Dict[str, Any]]] = Field(default=None, description="附件")
    images: Optional[List[str]] = Field(default=None, description="图片列表")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="元数据")
    parent_id: Optional[int] = Field(default=None, description="父消息ID")
    use_rag: Optional[bool] = Field(default=True, description="是否使用RAG检索")


class RAGQueryRequest(BaseModel):
    """RAG查询请求"""
    query: str = Field(..., min_length=1, description="查询内容")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="返回结果数量")
    conversation_id: Optional[int] = Field(default=None, description="关联对话ID")


class RAGQueryResponse(BaseResponse):
    """RAG查询响应"""
    answer: str = Field(default="", description="生成的回答")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="来源文档")
    context: str = Field(default="", description="检索上下文")
    confidence: float = Field(default=0.0, description="置信度")


# =============================================================================
# 集成版聊天管理器
# =============================================================================

class IntegratedChatManager:
    """
    集成版聊天管理器
    
    整合UFO高级算法模块：
    - ChatMemorySystem: 对话记忆管理
    - HierarchicalContextMemory: 分层记忆
    - AdaptiveContextCompressor: 上下文压缩
    - ModelGateway: 模型路由
    - RAGPipeline: 检索增强
    """
    
    def __init__(self):
        self.config = IntegratedChatConfig()
        self._init_advanced_modules()
        
    def _init_advanced_modules(self):
        """初始化高级算法模块"""
        if not ADVANCED_MODULES_AVAILABLE:
            logger.warning("高级算法模块不可用，回退到基础模式")
            self.memory_system = None
            self.hierarchical_memory = None
            self.compressor = None
            self.model_gateway = None
            self.rag_pipeline = None
            return
            
        try:
            # 初始化对话记忆系统
            self.memory_system = ChatMemorySystem(
                max_history_turns=self.config.MAX_HISTORY_TURNS
            )
            logger.info("ChatMemorySystem初始化成功")
            
            # 初始化分层记忆系统
            self.hierarchical_memory = HierarchicalContextMemory(
                short_term_capacity=self.config.SHORT_TERM_CAPACITY,
                medium_term_capacity=self.config.MEDIUM_TERM_CAPACITY,
                long_term_capacity=self.config.LONG_TERM_CAPACITY,
                compression_ratio=self.config.COMPRESSION_RATIO
            )
            logger.info("HierarchicalContextMemory初始化成功")
            
            # 初始化自适应压缩器
            self.compressor = AdaptiveContextCompressor(
                target_size=self.config.TARGET_COMPRESSED_TOKENS
            )
            logger.info("AdaptiveContextCompressor初始化成功")
            
            # 初始化模型网关
            self.model_gateway = ModelGateway(
                default_backend=self.config.DEFAULT_BACKEND
            )
            # 注册默认后端
            self._register_default_backends()
            logger.info("ModelGateway初始化成功")
            
            # 初始化RAG管道
            if self.config.ENABLE_RAG:
                try:
                    from rag.embedder import HashEmbedder
                    from rag.vector_store import VectorStore
                    embedder = HashEmbedder(dimension=384)
                    vector_store = VectorStore()
                    self.rag_pipeline = RAGPipeline(embedder=embedder, vector_store=vector_store)
                    logger.info("RAGPipeline初始化成功")
                except Exception as rag_err:
                    logger.warning(f"RAGPipeline初始化失败(非关键): {rag_err}")
                    self.rag_pipeline = None
            else:
                self.rag_pipeline = None
                
        except Exception as e:
            logger.error(f"高级模块初始化失败: {e}")
            self.memory_system = None
            self.hierarchical_memory = None
            self.compressor = None
            self.model_gateway = None
            self.rag_pipeline = None
    
    def _register_default_backends(self):
        """注册默认LLM后端"""
        if self.model_gateway is None:
            return
            
        try:
            from core.llm import OpenAIBackend
            
            # 注册OpenAI后端
            if self.config.OPENAI_API_KEY:
                openai_backend = OpenAIBackend(
                    api_key=self.config.OPENAI_API_KEY,
                    base_url=self.config.OPENAI_BASE_URL
                )
                self.model_gateway.register_backend("openai", openai_backend)
                logger.info("OpenAI后端注册成功")
        except Exception as e:
            logger.warning(f"注册默认后端失败: {e}")
    
    # =====================================================================
    # 记忆系统操作
    # =====================================================================
    
    def add_to_memory(self, conversation_id: int, role: str, content: str, 
                     metadata: Optional[Dict] = None) -> str:
        """
        添加消息到记忆系统
        
        Args:
            conversation_id: 对话ID
            role: 角色（user/assistant/system）
            content: 消息内容
            metadata: 元数据
            
        Returns:
            记忆ID
        """
        if self.memory_system is None:
            return ""
            
        try:
            memory_id = self.memory_system.add_message(
                conversation_id=str(conversation_id),
                role=role,
                content=content,
                metadata=metadata or {}
            )
            
            # 同时添加到分层记忆
            if self.hierarchical_memory is not None:
                import random
                # 生成简单的嵌入向量（实际应使用真实嵌入）
                embedding = [random.gauss(0, 0.1) for _ in range(768)]
                self.hierarchical_memory.add(
                    content=content,
                    embedding=embedding,
                    importance=0.5,
                    metadata={
                        "conversation_id": conversation_id,
                        "role": role,
                        "memory_id": memory_id
                    }
                )
            
            return memory_id
        except Exception as e:
            logger.warning(f"添加到记忆系统失败: {e}")
            return ""
    
    def get_context_with_compression(self, conversation_id: int, 
                                    max_tokens: int = 4000) -> List[Dict[str, str]]:
        """
        获取压缩后的对话上下文
        
        Args:
            conversation_id: 对话ID
            max_tokens: 最大token数
            
        Returns:
            压缩后的消息列表
        """
        if self.memory_system is None:
            return []
            
        try:
            # 获取原始上下文
            context = self.memory_system.get_context(
                conversation_id=str(conversation_id),
                max_turns=self.config.MAX_HISTORY_TURNS
            )
            
            # 如果启用压缩且超出阈值，进行压缩
            if (self.config.ENABLE_COMPRESSION and 
                self.compressor is not None and
                len(str(context)) > self.config.COMPRESSION_THRESHOLD):
                
                context = self.compressor.compress_context(
                    context,
                    target_tokens=self.config.TARGET_COMPRESSED_TOKENS
                )
                logger.info(f"上下文已压缩到 {len(context)} 条消息")
            
            return context
        except Exception as e:
            logger.warning(f"获取压缩上下文失败: {e}")
            return []
    
    # =====================================================================
    # RAG操作
    # =====================================================================
    
    async def query_with_rag(self, query: str, conversation_id: Optional[int] = None,
                            top_k: int = 5) -> Optional[RAGResult]:
        """
        使用RAG查询
        
        Args:
            query: 查询内容
            conversation_id: 关联对话ID
            top_k: 返回结果数量
            
        Returns:
            RAG结果
        """
        if self.rag_pipeline is None:
            return None
            
        try:
            result = self.rag_pipeline.query(
                query=query,
                top_k=top_k
            )
            return result
        except Exception as e:
            logger.warning(f"RAG查询失败: {e}")
            return None
    
    # =====================================================================
    # 模型网关操作
    # =====================================================================
    
    async def generate_with_gateway(self, messages: List[Dict[str, str]], 
                                    model: Optional[str] = None,
                                    stream: bool = False,
                                    **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        """
        使用ModelGateway生成回复
        
        Args:
            messages: 消息列表
            model: 模型名称
            stream: 是否流式输出
            **kwargs: 其他参数
            
        Returns:
            生成的回复或流式生成器
        """
        if self.model_gateway is None:
            # 回退到直接HTTP调用
            return await self._fallback_generate(messages, model, stream, **kwargs)
        
        try:
            model = model or self.config.DEFAULT_MODEL
            
            if stream:
                return self.model_gateway.generate_stream(
                    messages=messages,
                    model=model,
                    **kwargs
                )
            else:
                response = self.model_gateway.generate(
                    messages=messages,
                    model=model,
                    **kwargs
                )
                return response.content
        except Exception as e:
            logger.error(f"ModelGateway生成失败: {e}")
            if self.config.ENABLE_FALLBACK:
                return await self._fallback_generate(messages, model, stream, **kwargs)
            raise
    
    async def _fallback_generate(self, messages: List[Dict[str, str]], 
                                 model: Optional[str] = None,
                                 stream: bool = False,
                                 **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        """回退到直接HTTP调用"""
        import httpx
        
        model = model or self.config.DEFAULT_MODEL
        
        headers = {
            "Authorization": f"Bearer {self.config.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        
        async with httpx.AsyncClient() as client:
            if stream:
                async def stream_generator():
                    async with client.stream(
                        "POST",
                        f"{self.config.OPENAI_BASE_URL}/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=60.0
                    ) as response:
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                chunk = line[6:]
                                if chunk == "[DONE]":
                                    break
                                try:
                                    data = json.loads(chunk)
                                    if delta := data.get("choices", [{}])[0].get("delta", {}).get("content"):
                                        yield delta
                                except:
                                    pass
                return stream_generator()
            else:
                response = await client.post(
                    f"{self.config.OPENAI_BASE_URL}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60.0
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]


# =============================================================================
# 全局管理器实例
# =============================================================================

chat_manager = IntegratedChatManager()


# =============================================================================
# API端点
# =============================================================================

@router.get("/conversations/integrated", response_model=BaseResponse)
async def list_conversations_integrated(
    page: int = 1,
    limit: int = 50,
    current_user: User = Depends(get_current_user)
):
    """获取对话列表（集成版）"""
    try:
        from database.connection import SessionLocal
        from database.models import Conversation
        db = SessionLocal()
        try:
            query = db.query(Conversation).filter(
                Conversation.user_id == current_user.id,
                Conversation.is_deleted == False
            ).order_by(Conversation.updated_at.desc())
            
            total = query.count()
            conversations = query.offset((page - 1) * limit).limit(limit).all()
            
            return BaseResponse(
                success=True,
                data={
                    "items": [
                        {
                            "id": c.id,
                            "title": c.title,
                            "model_name": c.model_name,
                            "created_at": str(c.created_at) if c.created_at else None,
                            "updated_at": str(c.updated_at) if c.updated_at else None,
                        }
                        for c in conversations
                    ],
                    "total": total,
                    "page": page,
                    "limit": limit,
                }
            )
        finally:
            db.close()
    except Exception as e:
        logger.error(f"获取对话列表失败: {e}")
        return BaseResponse(success=False, message=f"获取对话列表失败: {str(e)}")


@router.post("/conversations/integrated", response_model=BaseResponse)
async def create_conversation_integrated(
    request: ConversationCreateRequest,
    current_user: User = Depends(get_current_user)
):
    """
    创建对话（集成版）
    
    使用HierarchicalContextMemory初始化对话记忆
    """
    try:
        # 创建对话记录（使用原有ChatStorage）
        from api.routes.chat import chat_storage
        
        conv_data = {
            "user_id": current_user.id,
            "title": request.title,
            "model_name": request.model_name,
            "system_prompt": request.system_prompt,
            "description": request.description,
            "tags": request.tags,
            "category": request.category,
            "config": {
                **(request.config or {}),
                "enable_rag": request.enable_rag,
                "enable_compression": request.enable_compression
            }
        }
        
        conversation = chat_storage.create_conversation(conv_data)
        
        # 初始化记忆系统
        if chat_manager.memory_system is not None:
            chat_manager.memory_system.create_conversation_context(
                conversation_id=str(conversation["id"]),
                system_prompt=request.system_prompt
            )
        
        return BaseResponse(
            success=True,
            message="对话创建成功（集成高级记忆系统）",
            data=conversation
        )
    except Exception as e:
        logger.error(f"创建对话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversations/{conversation_id}/messages/integrated", response_model=BaseResponse)
async def send_message_integrated(
    conversation_id: int,
    request: MessageCreateRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    发送消息（集成版）
    
    集成特性：
    1. 使用ChatMemorySystem管理对话历史
    2. 使用AdaptiveContextCompressor自动压缩上下文
    3. 使用ModelGateway路由到最优后端
    4. 可选RAG检索增强
    """
    try:
        from api.routes.chat import chat_storage
        
        # 1. 保存用户消息到数据库
        user_msg_data = {
            "role": "user",
            "content": request.content,
            "parent_id": request.parent_id
        }
        user_message = chat_storage.create_message(conversation_id, user_msg_data)
        
        # 2. 添加到记忆系统
        chat_manager.add_to_memory(
            conversation_id=conversation_id,
            role="user",
            content=request.content
        )
        
        # 3. 获取压缩后的上下文
        context = chat_manager.get_context_with_compression(conversation_id)
        
        # 4. 如果使用RAG，检索相关文档
        rag_context = ""
        if request.use_rag and chat_manager.rag_pipeline is not None:
            rag_result = await chat_manager.query_with_rag(
                query=request.content,
                conversation_id=conversation_id
            )
            if rag_result:
                rag_context = rag_result.context
        
        # 5. 构建消息列表
        messages = []
        
        # 添加系统提示
        conversation = chat_storage.get_conversation(conversation_id)
        if conversation and conversation.get("system_prompt"):
            messages.append({
                "role": "system",
                "content": conversation["system_prompt"] + (f"\n\n相关文档：{rag_context}" if rag_context else "")
            })
        
        # 添加上下文
        messages.extend(context)
        
        # 添加当前消息
        messages.append({"role": "user", "content": request.content})
        
        # 6. 使用ModelGateway生成回复
        model_name = conversation.get("model_name") if conversation else None
        
        response_content = await chat_manager.generate_with_gateway(
            messages=messages,
            model=model_name,
            stream=False,
            temperature=conversation.get("temperature", 0.7) if conversation else 0.7,
            max_tokens=conversation.get("max_tokens", 4096) if conversation else 4096
        )
        
        # 7. 保存AI回复
        assistant_msg_data = {
            "role": "assistant",
            "content": response_content,
            "model_name": model_name or chat_manager.config.DEFAULT_MODEL,
            "model_provider": "openai",
            "parent_id": user_message["id"]
        }
        assistant_message = chat_storage.create_message(conversation_id, assistant_msg_data)
        
        # 8. 添加到记忆系统
        chat_manager.add_to_memory(
            conversation_id=conversation_id,
            role="assistant",
            content=response_content
        )
        
        return BaseResponse(
            success=True,
            message="消息发送成功（使用高级算法）",
            data={
                "user_message": user_message,
                "assistant_message": assistant_message,
                "context_tokens": len(str(context)),
                "rag_used": rag_context != ""
            }
        )
        
    except Exception as e:
        logger.error(f"发送消息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversations/{conversation_id}/stream/integrated")
async def stream_message_integrated(
    conversation_id: int,
    request: MessageCreateRequest,
    current_user: User = Depends(get_current_user)
):
    """
    流式发送消息（集成版）
    
    使用ModelGateway的流式生成功能
    """
    async def event_generator():
        try:
            from api.routes.chat import chat_storage
            
            # 保存用户消息
            user_msg_data = {
                "role": "user",
                "content": request.content
            }
            user_message = chat_storage.create_message(conversation_id, user_msg_data)
            
            # 获取上下文
            context = chat_manager.get_context_with_compression(conversation_id)
            
            # 构建消息
            conversation = chat_storage.get_conversation(conversation_id)
            messages = []
            if conversation and conversation.get("system_prompt"):
                messages.append({"role": "system", "content": conversation["system_prompt"]})
            messages.extend(context)
            messages.append({"role": "user", "content": request.content})
            
            # 流式生成
            model_name = conversation.get("model_name") if conversation else None
            
            full_content = ""
            async for chunk in await chat_manager.generate_with_gateway(
                messages=messages,
                model=model_name,
                stream=True
            ):
                full_content += chunk
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            
            # 保存完整回复
            assistant_msg_data = {
                "role": "assistant",
                "content": full_content,
                "model_name": model_name or chat_manager.config.DEFAULT_MODEL,
                "parent_id": user_message["id"]
            }
            chat_storage.create_message(conversation_id, assistant_msg_data)
            
            # 添加到记忆系统
            chat_manager.add_to_memory(
                conversation_id=conversation_id,
                role="assistant",
                content=full_content
            )
            
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@router.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(
    request: RAGQueryRequest,
    current_user: User = Depends(get_current_user)
):
    """
    RAG查询端点
    
    使用RAGPipeline进行文档检索和回答生成
    """
    try:
        if chat_manager.rag_pipeline is None:
            raise HTTPException(status_code=503, detail="RAG服务未初始化")
        
        result = await chat_manager.query_with_rag(
            query=request.query,
            conversation_id=request.conversation_id,
            top_k=request.top_k
        )
        
        if result is None:
            raise HTTPException(status_code=500, detail="RAG查询失败")
        
        return RAGQueryResponse(
            success=True,
            message="RAG查询成功",
            answer=result.answer,
            sources=[s.to_dict() for s in result.sources],
            context=result.context,
            confidence=result.confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{conversation_id}/memory/stats", response_model=BaseResponse)
async def get_memory_stats(
    conversation_id: int,
    current_user: User = Depends(get_current_user)
):
    """
    获取对话记忆统计
    
    返回分层记忆的统计信息
    """
    try:
        if chat_manager.memory_system is None:
            return BaseResponse(
                success=False,
                message="记忆系统未初始化",
                data={}
            )
        
        stats = chat_manager.memory_system.get_conversation_stats(
            conversation_id=str(conversation_id)
        )
        
        return BaseResponse(
            success=True,
            message="获取记忆统计成功",
            data=stats
        )
        
    except Exception as e:
        logger.error(f"获取记忆统计失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversations/{conversation_id}/memory/compress", response_model=BaseResponse)
async def compress_conversation_memory(
    conversation_id: int,
    current_user: User = Depends(get_current_user)
):
    """
    手动压缩对话记忆
    
    触发AdaptiveContextCompressor进行上下文压缩
    """
    try:
        if chat_manager.compressor is None:
            return BaseResponse(
                success=False,
                message="压缩器未初始化",
                data={}
            )
        
        # 获取当前上下文
        context = chat_manager.get_context_with_compression(conversation_id)
        original_count = len(context)
        
        # 强制压缩
        compressed = chat_manager.compressor.compress_context(
            context,
            target_tokens=chat_manager.config.TARGET_COMPRESSED_TOKENS // 2
        )
        compressed_count = len(compressed)
        
        return BaseResponse(
            success=True,
            message=f"记忆压缩成功：{original_count} -> {compressed_count} 条",
            data={
                "original_count": original_count,
                "compressed_count": compressed_count,
                "compression_ratio": compressed_count / original_count if original_count > 0 else 0
            }
        )
        
    except Exception as e:
        logger.error(f"压缩记忆失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 健康检查端点
# =============================================================================

@router.get("/health/integrated", response_model=BaseResponse)
async def health_check_integrated():
    """
    集成系统健康检查
    
    检查所有高级算法模块的状态
    """
    health = {
        "advanced_modules_available": ADVANCED_MODULES_AVAILABLE,
        "memory_system": chat_manager.memory_system is not None,
        "hierarchical_memory": chat_manager.hierarchical_memory is not None,
        "compressor": chat_manager.compressor is not None,
        "model_gateway": chat_manager.model_gateway is not None,
        "rag_pipeline": chat_manager.rag_pipeline is not None,
    }
    
    all_healthy = all(health.values())
    
    return BaseResponse(
        success=all_healthy,
        message="所有高级模块正常运行" if all_healthy else "部分模块未初始化",
        data=health
    )
