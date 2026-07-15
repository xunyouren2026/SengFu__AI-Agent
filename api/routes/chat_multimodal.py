"""
Multimodal Chat API Routes
多模态聊天API路由

提供多模态对话、图像理解、语音识别等功能
"""

import asyncio
import base64
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, AsyncGenerator

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# 导入核心模块
from core.multimodal import (
    MultimodalChatEngine,
    ChatMessage,
    ChatSession,
    MessageType,
    Role,
    get_chat_engine,
    init_chat_engine,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Multimodal Chat"])


# ============== 请求模型 ==============

class CreateSessionRequest(BaseModel):
    """创建会话请求"""
    title: str = ""
    model: str = "gpt-4o"
    system_prompt: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096


class SendMessageRequest(BaseModel):
    """发送消息请求"""
    content: str
    message_type: str = "text"
    attachments: Optional[List[Dict[str, Any]]] = None
    stream: bool = False


class GenerateImageRequest(BaseModel):
    """生成图像请求"""
    prompt: str
    width: int = 512
    height: int = 512
    num_inference_steps: int = 30
    guidance_scale: float = 7.5


class GenerateSpeechRequest(BaseModel):
    """生成语音请求"""
    text: str
    voice_id: str = "default"
    language: str = "zh-CN"


# ============== 会话管理 API ==============

@router.post("/sessions", summary="Create chat session")
async def create_session(request: CreateSessionRequest):
    """
    创建新的聊天会话
    
    支持多种模型：
    - gpt-4o: OpenAI GPT-4o（多模态）
    - gpt-4-turbo: OpenAI GPT-4 Turbo
    - claude-3-opus: Anthropic Claude 3 Opus
    - claude-3-sonnet: Anthropic Claude 3 Sonnet
    """
    try:
        engine = get_chat_engine()
        
        session = await engine.create_session(
            title=request.title,
            model=request.model,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        
        return {
            "success": True,
            "session": session.to_dict(),
        }
        
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions", summary="List chat sessions")
async def list_sessions(limit: int = 100):
    """列出所有聊天会话"""
    try:
        engine = get_chat_engine()
        sessions = await engine.list_sessions(limit)
        
        return {
            "sessions": [s.to_dict() for s in sessions],
            "total": len(sessions),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}", summary="Get session")
async def get_session(session_id: str):
    """获取会话详情"""
    try:
        engine = get_chat_engine()
        session = await engine.get_session(session_id)
        
        if session:
            return session.to_dict()
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}", summary="Delete session")
async def delete_session(session_id: str):
    """删除会话"""
    try:
        engine = get_chat_engine()
        success = await engine.delete_session(session_id)
        
        return {
            "success": success,
            "session_id": session_id,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== 消息 API ==============

@router.post("/sessions/{session_id}/messages", summary="Send message")
async def send_message(session_id: str, request: SendMessageRequest):
    """
    发送消息
    
    支持多种消息类型：
    - text: 纯文本
    - image: 图像（需要attachments）
    - audio: 音频（需要attachments）
    - file: 文件（需要attachments）
    """
    try:
        engine = get_chat_engine()
        
        message_type = MessageType(request.message_type)
        
        result = await engine.send_message(
            session_id=session_id,
            content=request.content,
            message_type=message_type,
            attachments=request.attachments,
            stream=request.stream,
        )
        
        if request.stream:
            # 流式响应
            async def generate():
                async for chunk in result:
                    yield f"data: {chunk.to_dict()}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
            )
        else:
            return {
                "success": True,
                "message": result.to_dict(),
            }
            
    except Exception as e:
        logger.error(f"Failed to send message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/messages", summary="Get messages")
async def get_messages(session_id: str, limit: int = 50):
    """获取会话消息"""
    try:
        engine = get_chat_engine()
        session = await engine.get_session(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        messages = session.messages[-limit:] if len(session.messages) > limit else session.messages
        
        return {
            "messages": [m.to_dict() for m in messages],
            "total": len(session.messages),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== 多模态输入 API ==============

@router.post("/sessions/{session_id}/images", summary="Upload and analyze image")
async def upload_image(
    session_id: str,
    file: UploadFile = File(...),
    prompt: str = "请描述这张图片",
):
    """
    上传并分析图像
    
    支持的功能：
    - 图像描述
    - OCR文字识别
    - 物体检测
    - 场景理解
    """
    try:
        engine = get_chat_engine()
        
        # 读取图像
        image_data = await file.read()
        image_base64 = base64.b64encode(image_data).decode()
        
        # 发送消息
        result = await engine.send_message(
            session_id=session_id,
            content=prompt,
            message_type=MessageType.IMAGE,
            attachments=[{
                "type": "image",
                "base64": f"data:image/jpeg;base64,{image_base64}",
                "filename": file.filename,
            }],
        )
        
        return {
            "success": True,
            "message": result.to_dict(),
        }
        
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/audio", summary="Upload and transcribe audio")
async def upload_audio(
    session_id: str,
    file: UploadFile = File(...),
    prompt: str = "",
):
    """
    上传并转录音频
    
    支持的功能：
    - 语音转文字
    - 多语言识别
    - 长音频处理
    """
    try:
        engine = get_chat_engine()
        
        # 读取音频
        audio_data = await file.read()
        audio_base64 = base64.b64encode(audio_data).decode()
        
        # 发送消息
        result = await engine.send_message(
            session_id=session_id,
            content=prompt or "请转录这段音频",
            message_type=MessageType.AUDIO,
            attachments=[{
                "type": "audio",
                "base64": audio_base64,
                "filename": file.filename,
            }],
        )
        
        return {
            "success": True,
            "message": result.to_dict(),
        }
        
    except Exception as e:
        logger.error(f"Audio transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/files", summary="Upload and parse document")
async def upload_file(
    session_id: str,
    file: UploadFile = File(...),
    prompt: str = "",
):
    """
    上传并解析文档
    
    支持的格式：
    - PDF
    - Word (docx)
    - Excel (xlsx)
    - PowerPoint (pptx)
    - 文本文件 (txt, md)
    """
    try:
        engine = get_chat_engine()
        
        # 保存上传的文件
        file_path = f"/tmp/upload_{uuid.uuid4().hex}_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 获取文件类型
        file_ext = file.filename.split(".")[-1].lower() if file.filename else "txt"
        
        # 发送消息
        result = await engine.send_message(
            session_id=session_id,
            content=prompt or "请总结这个文档的内容",
            message_type=MessageType.FILE,
            attachments=[{
                "type": "file",
                "path": file_path,
                "filename": file.filename,
                "file_type": file_ext,
            }],
        )
        
        return {
            "success": True,
            "message": result.to_dict(),
        }
        
    except Exception as e:
        logger.error(f"Document parsing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== 生成 API ==============

@router.post("/sessions/{session_id}/generate-image", summary="Generate image in chat")
async def generate_image_in_chat(session_id: str, request: GenerateImageRequest):
    """在对话中生成图像"""
    try:
        engine = get_chat_engine()
        
        result = await engine.generate_image(
            session_id=session_id,
            prompt=request.prompt,
            config={
                "width": request.width,
                "height": request.height,
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
            },
        )
        
        return {
            "success": True,
            "result": result,
        }
        
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/generate-speech", summary="Generate speech in chat")
async def generate_speech_in_chat(session_id: str, request: GenerateSpeechRequest):
    """在对话中生成语音"""
    try:
        engine = get_chat_engine()
        
        result = await engine.generate_speech(
            session_id=session_id,
            text=request.text,
            config={
                "voice_id": request.voice_id,
                "language": request.language,
            },
        )
        
        return {
            "success": True,
            "result": result,
        }
        
    except Exception as e:
        logger.error(f"Speech generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== 工具调用 API ==============

@router.post("/sessions/{session_id}/tools", summary="Execute tool")
async def execute_tool(
    session_id: str,
    tool_name: str,
    parameters: Dict[str, Any],
):
    """
    执行工具
    
    支持的工具：
    - web_search: 网页搜索
    - code_execute: 代码执行
    - file_operation: 文件操作
    - api_call: API调用
    """
    try:
        engine = get_chat_engine()
        
        # TODO: 实现工具执行
        
        return {
            "success": True,
            "tool_name": tool_name,
            "result": "Tool execution not implemented yet",
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 导出路由
__all__ = ["router"]
