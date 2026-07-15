"""
Multimodal Chat System
多模态聊天系统

支持文本、图像、音频、视频、文件等多种输入
集成视觉理解、语音识别、文档解析等能力
"""

import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, AsyncGenerator

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """消息类型"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    SYSTEM = "system"


class Role(Enum):
    """角色"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ChatMessage:
    """聊天消息"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: Role = Role.USER
    content: str = ""
    message_type: MessageType = MessageType.TEXT
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "message_type": self.message_type.value,
            "attachments": self.attachments,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }
    
    def to_openai_format(self) -> Dict[str, Any]:
        """转换为OpenAI格式"""
        if self.message_type == MessageType.TEXT:
            return {
                "role": self.role.value,
                "content": self.content,
            }
        elif self.message_type == MessageType.IMAGE:
            # 多模态格式
            content = [{"type": "text", "text": self.content}]
            for attachment in self.attachments:
                if attachment.get("type") == "image":
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": attachment.get("url", attachment.get("base64", ""))
                        }
                    })
            return {
                "role": self.role.value,
                "content": content,
            }
        else:
            return {
                "role": self.role.value,
                "content": self.content,
            }


@dataclass
class ChatSession:
    """聊天会话"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    messages: List[ChatMessage] = field(default_factory=list)
    model: str = "gpt-4o"
    system_prompt: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "messages": [m.to_dict() for m in self.messages],
            "model": self.model,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    def get_context_messages(self, max_messages: int = 50) -> List[Dict[str, Any]]:
        """获取上下文消息"""
        messages = []
        
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt,
            })
        
        recent_messages = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        
        for msg in recent_messages:
            messages.append(msg.to_openai_format())
        
        return messages


class MultimodalChatEngine:
    """
    多模态聊天引擎
    
    功能：
    - 支持多种输入类型
    - 视觉理解（图像描述、OCR、分析）
    - 语音识别（语音转文字）
    - 文档解析（PDF、Word、Excel等）
    - 代码执行
    - 工具调用
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._sessions: Dict[str, ChatSession] = {}
        self._llm_client = None
        self._vision_model = None
        self._speech_recognizer = None
        self._initialized = False
        
    async def _ensure_initialized(self):
        """确保引擎已初始化"""
        if self._initialized:
            return
        
        # 初始化LLM客户端
        await self._init_llm_client()
        
        # 初始化视觉模型
        await self._init_vision_model()
        
        # 初始化语音识别
        await self._init_speech_recognizer()
        
        self._initialized = True
        logger.info("Multimodal chat engine initialized")
    
    async def _init_llm_client(self):
        """初始化LLM客户端"""
        try:
            from openai import AsyncOpenAI
            
            api_key = self.config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
            base_url = self.config.get("openai_base_url") or os.environ.get("OPENAI_BASE_URL")
            
            if api_key:
                self._llm_client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                )
                logger.info("OpenAI client initialized")
        except ImportError:
            logger.warning("openai not installed, using mock client")
    
    async def _init_vision_model(self):
        """初始化视觉模型"""
        # 可以使用本地视觉模型或API
        vision_backend = self.config.get("vision_backend", "openai")
        
        if vision_backend == "openai":
            # 使用GPT-4V
            self._vision_model = "gpt-4o"
        elif vision_backend == "local":
            # 使用本地模型
            try:
                from transformers import AutoModelForCausalLM, AutoProcessor
                # TODO: 加载本地视觉模型
            except ImportError:
                logger.warning("transformers not installed")
        
        logger.info(f"Vision model initialized: {self._vision_model}")
    
    async def _init_speech_recognizer(self):
        """初始化语音识别"""
        speech_backend = self.config.get("speech_backend", "whisper")
        
        if speech_backend == "whisper":
            try:
                import whisper
                model_size = self.config.get("whisper_model", "base")
                self._speech_recognizer = whisper.load_model(model_size)
                logger.info(f"Whisper model loaded: {model_size}")
            except ImportError:
                logger.warning("whisper not installed")
        elif speech_backend == "openai":
            # 使用OpenAI Whisper API
            pass
    
    async def create_session(
        self,
        title: str = "",
        model: str = "gpt-4o",
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ChatSession:
        """创建新会话"""
        session = ChatSession(
            title=title or f"Chat {len(self._sessions) + 1}",
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        self._sessions[session.id] = session
        
        logger.info(f"Created session: {session.id}")
        
        return session
    
    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """获取会话"""
        return self._sessions.get(session_id)
    
    async def list_sessions(self, limit: int = 100) -> List[ChatSession]:
        """列出会话"""
        sessions = list(self._sessions.values())
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions[:limit]
    
    async def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
    
    async def send_message(
        self,
        session_id: str,
        content: str,
        message_type: MessageType = MessageType.TEXT,
        attachments: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
    ) -> Union[ChatMessage, AsyncGenerator[ChatMessage, None]]:
        """
        发送消息
        
        Args:
            session_id: 会话ID
            content: 消息内容
            message_type: 消息类型
            attachments: 附件列表
            stream: 是否流式输出
        
        Returns:
            回复消息或流式生成器
        """
        await self._ensure_initialized()
        
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        # 创建用户消息
        user_message = ChatMessage(
            role=Role.USER,
            content=content,
            message_type=message_type,
            attachments=attachments or [],
        )
        
        session.messages.append(user_message)
        session.updated_at = time.time()
        
        # 处理多模态输入
        processed_content = await self._process_multimodal_input(content, message_type, attachments)
        
        # 调用LLM
        if stream:
            return self._stream_response(session, processed_content)
        else:
            return await self._generate_response(session, processed_content)
    
    async def _process_multimodal_input(
        self,
        content: str,
        message_type: MessageType,
        attachments: Optional[List[Dict[str, Any]]],
    ) -> str:
        """处理多模态输入"""
        if message_type == MessageType.TEXT:
            return content
        
        elif message_type == MessageType.IMAGE:
            # 图像理解
            image_descriptions = []
            
            for attachment in (attachments or []):
                if attachment.get("type") == "image":
                    description = await self._analyze_image(attachment)
                    image_descriptions.append(description)
            
            if image_descriptions:
                return f"{content}\n\n[Image Analysis]\n" + "\n".join(image_descriptions)
            return content
        
        elif message_type == MessageType.AUDIO:
            # 语音识别
            for attachment in (attachments or []):
                if attachment.get("type") == "audio":
                    transcript = await self._transcribe_audio(attachment)
                    return f"{content}\n\n[Transcript]\n{transcript}"
            return content
        
        elif message_type == MessageType.FILE:
            # 文档解析
            for attachment in (attachments or []):
                if attachment.get("type") == "file":
                    text = await self._parse_document(attachment)
                    return f"{content}\n\n[Document Content]\n{text}"
            return content
        
        return content
    
    async def _analyze_image(self, attachment: Dict[str, Any]) -> str:
        """分析图像"""
        try:
            if self._llm_client:
                # 使用GPT-4V
                image_url = attachment.get("url")
                if not image_url and attachment.get("base64"):
                    image_url = f"data:image/jpeg;base64,{attachment['base64']}"
                
                response = await self._llm_client.chat.completions.create(
                    model=self._vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "请详细描述这张图片的内容。"},
                                {"type": "image_url", "image_url": {"url": image_url}},
                            ],
                        }
                    ],
                    max_tokens=1000,
                )
                
                return response.choices[0].message.content
            else:
                return "[Image analysis not available - LLM client not initialized]"
                
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return f"[Image analysis error: {str(e)}]"
    
    async def _transcribe_audio(self, attachment: Dict[str, Any]) -> str:
        """转录音频"""
        try:
            if self._speech_recognizer:
                # 使用本地Whisper
                audio_path = attachment.get("path")
                if audio_path:
                    result = self._speech_recognizer.transcribe(audio_path)
                    return result["text"]
                elif attachment.get("base64"):
                    # 保存临时文件
                    audio_data = base64.b64decode(attachment["base64"])
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        f.write(audio_data)
                        temp_path = f.name
                    
                    result = self._speech_recognizer.transcribe(temp_path)
                    os.unlink(temp_path)
                    return result["text"]
            elif self._llm_client:
                # 使用OpenAI Whisper API
                if attachment.get("base64"):
                    audio_data = base64.b64decode(attachment["base64"])
                    audio_file = io.BytesIO(audio_data)
                    audio_file.name = "audio.wav"
                    
                    transcript = await self._llm_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                    )
                    return transcript.text
            
            return "[Audio transcription not available]"
            
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return f"[Transcription error: {str(e)}]"
    
    async def _parse_document(self, attachment: Dict[str, Any]) -> str:
        """解析文档"""
        try:
            file_path = attachment.get("path")
            file_type = attachment.get("file_type", "").lower()
            
            if file_type == "pdf":
                # 解析PDF
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(file_path)
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    return text[:10000]  # 限制长度
                except ImportError:
                    return "[PDF parsing requires PyMuPDF]"
            
            elif file_type in ["docx", "doc"]:
                # 解析Word
                try:
                    from docx import Document
                    doc = Document(file_path)
                    text = "\n".join([p.text for p in doc.paragraphs])
                    return text[:10000]
                except ImportError:
                    return "[Word parsing requires python-docx]"
            
            elif file_type in ["xlsx", "xls"]:
                # 解析Excel
                try:
                    import pandas as pd
                    df = pd.read_excel(file_path)
                    return df.to_string()[:10000]
                except ImportError:
                    return "[Excel parsing requires pandas and openpyxl]"
            
            elif file_type == "txt":
                # 纯文本
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()[:10000]
            
            return f"[Unsupported file type: {file_type}]"
            
        except Exception as e:
            logger.error(f"Document parsing failed: {e}")
            return f"[Parsing error: {str(e)}]"
    
    async def _generate_response(
        self,
        session: ChatSession,
        processed_content: str,
    ) -> ChatMessage:
        """生成回复"""
        try:
            if self._llm_client:
                messages = session.get_context_messages()
                
                response = await self._llm_client.chat.completions.create(
                    model=session.model,
                    messages=messages,
                    temperature=session.temperature,
                    max_tokens=session.max_tokens,
                )
                
                assistant_content = response.choices[0].message.content
                
            else:
                # 模拟响应
                assistant_content = f"[Mock response] You said: {processed_content[:100]}..."
            
            # 创建助手消息
            assistant_message = ChatMessage(
                role=Role.ASSISTANT,
                content=assistant_content,
                message_type=MessageType.TEXT,
            )
            
            session.messages.append(assistant_message)
            session.updated_at = time.time()
            
            return assistant_message
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            
            error_message = ChatMessage(
                role=Role.ASSISTANT,
                content=f"Error: {str(e)}",
                message_type=MessageType.TEXT,
            )
            
            return error_message
    
    async def _stream_response(
        self,
        session: ChatSession,
        processed_content: str,
    ) -> AsyncGenerator[ChatMessage, None]:
        """流式生成回复"""
        try:
            if self._llm_client:
                messages = session.get_context_messages()
                
                stream = await self._llm_client.chat.completions.create(
                    model=session.model,
                    messages=messages,
                    temperature=session.temperature,
                    max_tokens=session.max_tokens,
                    stream=True,
                )
                
                full_content = ""
                
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_content += content
                        
                        yield ChatMessage(
                            role=Role.ASSISTANT,
                            content=content,
                            message_type=MessageType.TEXT,
                            metadata={"streaming": True},
                        )
                
                # 保存完整消息
                assistant_message = ChatMessage(
                    role=Role.ASSISTANT,
                    content=full_content,
                    message_type=MessageType.TEXT,
                )
                
                session.messages.append(assistant_message)
                session.updated_at = time.time()
                
            else:
                # 模拟流式响应
                mock_response = f"[Mock streaming response] Processing: {processed_content[:50]}..."
                
                for char in mock_response:
                    yield ChatMessage(
                        role=Role.ASSISTANT,
                        content=char,
                        message_type=MessageType.TEXT,
                        metadata={"streaming": True},
                    )
                    await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield ChatMessage(
                role=Role.ASSISTANT,
                content=f"Error: {str(e)}",
                message_type=MessageType.TEXT,
            )
    
    async def generate_image(
        self,
        session_id: str,
        prompt: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """在对话中生成图像"""
        from .generation import create_image_engine, ImageConfig
        
        engine = create_image_engine(
            config.get("engine", "diffusers") if config else "diffusers",
            self.config.get("image", {})
        )
        
        image_config = ImageConfig(
            prompt=prompt,
            width=config.get("width", 512) if config else 512,
            height=config.get("height", 512) if config else 512,
            num_inference_steps=config.get("num_inference_steps", 30) if config else 30,
            guidance_scale=config.get("guidance_scale", 7.5) if config else 7.5,
        )
        
        result = await engine.generate(image_config)
        
        # 添加到会话
        session = self._sessions.get(session_id)
        if session:
            message = ChatMessage(
                role=Role.ASSISTANT,
                content=f"Generated image: {prompt}",
                message_type=MessageType.IMAGE,
                attachments=[{
                    "type": "image",
                    "path": result.image_paths[0] if result.image_paths else None,
                }],
            )
            session.messages.append(message)
        
        return result.to_dict()
    
    async def generate_speech(
        self,
        session_id: str,
        text: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """在对话中生成语音"""
        from .generation import create_tts_engine, VoiceConfig
        
        engine = create_tts_engine(
            config.get("engine", "edge") if config else "edge",
            self.config.get("tts", {})
        )
        
        voice_config = VoiceConfig(
            voice_id=config.get("voice_id", "default") if config else "default",
            language=config.get("language", "zh-CN") if config else "zh-CN",
        )
        
        result = await engine.synthesize(text, voice_config)
        
        # 添加到会话
        session = self._sessions.get(session_id)
        if session:
            message = ChatMessage(
                role=Role.ASSISTANT,
                content=text,
                message_type=MessageType.AUDIO,
                attachments=[{
                    "type": "audio",
                    "path": result.audio_path,
                }],
            )
            session.messages.append(message)
        
        return result.to_dict()


# 全局实例
_chat_engine: Optional[MultimodalChatEngine] = None


def get_chat_engine() -> MultimodalChatEngine:
    """获取全局聊天引擎"""
    global _chat_engine
    if _chat_engine is None:
        _chat_engine = MultimodalChatEngine()
    return _chat_engine


async def init_chat_engine(config: Optional[Dict[str, Any]] = None):
    """初始化全局聊天引擎"""
    global _chat_engine
    _chat_engine = MultimodalChatEngine(config)
    await _chat_engine._ensure_initialized()
    return _chat_engine
