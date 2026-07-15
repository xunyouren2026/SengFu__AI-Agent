"""
PaLM API客户端 - Google AI兼容接口

提供对Google PaLM API的完整支持，包括：
- 文本生成
- 聊天对话
- 嵌入生成
- 流式响应
- 错误处理和重试机制

注意：PaLM API已被Gemini API取代，此客户端提供向后兼容性。

模块路径: compat/google/palm.py
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field
from typing import (
    Any, AsyncIterator, Dict, List, Optional, Union, Callable
)
from enum import Enum
from pathlib import Path

import httpx
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log
)

# 导入基础异常
from . import (
    GoogleAIError, AuthenticationError, RateLimitError,
    InvalidRequestError, ModelNotFoundError, ServerError
)

logger = logging.getLogger(__name__)

# PaLM API常量
PALM_API_BASE = "https://generativelanguage.googleapis.com"
PALM_API_VERSION = "v1beta2"


class PaLMModel(str, Enum):
    """PaLM模型枚举"""
    TEXT_BISON_001 = "text-bison-001"
    CHAT_BISON_001 = "chat-bison-001"
    EMBEDDING_GECKO_001 = "embedding-gecko-001"


@dataclass
class PalmConfig:
    """PaLM客户端配置"""
    api_key: Optional[str] = None
    api_base: str = PALM_API_BASE
    api_version: str = PALM_API_VERSION
    timeout: float = 60.0
    max_retries: int = 3
    default_text_model: str = "text-bison-001"
    default_chat_model: str = "chat-bison-001"
    default_embedding_model: str = "embedding-gecko-001"


@dataclass
class TextGenerationConfig:
    """文本生成配置"""
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    candidate_count: Optional[int] = None
    max_output_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        if self.temperature is not None:
            config["temperature"] = self.temperature
        if self.top_p is not None:
            config["topP"] = self.top_p
        if self.top_k is not None:
            config["topK"] = self.top_k
        if self.candidate_count is not None:
            config["candidateCount"] = self.candidate_count
        if self.max_output_tokens is not None:
            config["maxOutputTokens"] = self.max_output_tokens
        if self.stop_sequences:
            config["stopSequences"] = self.stop_sequences
        return config


@dataclass
class Message:
    """聊天消息"""
    author: str  # '0' for user, '1' for bot
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "author": self.author,
            "content": self.content
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "Message":
        return cls(author=data.get("author", "0"), content=data.get("content", ""))


@dataclass
class TextCompletion:
    """文本生成结果"""
    output: str
    safety_ratings: Optional[List[Dict[str, Any]]] = None
    citation_metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextCompletion":
        return cls(
            output=data.get("output", ""),
            safety_ratings=data.get("safetyRatings"),
            citation_metadata=data.get("citationMetadata")
        )


@dataclass
class TextGenerationResponse:
    """文本生成响应"""
    candidates: List[TextCompletion]
    filters: Optional[List[Dict[str, Any]]] = None
    safety_feedback: Optional[List[Dict[str, Any]]] = None
    raw_response: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextGenerationResponse":
        candidates = [
            TextCompletion.from_dict(c) 
            for c in data.get("candidates", [])
        ]
        return cls(
            candidates=candidates,
            filters=data.get("filters"),
            safety_feedback=data.get("safetyFeedback"),
            raw_response=data
        )
    
    @property
    def text(self) -> str:
        """获取第一个候选的输出"""
        if self.candidates:
            return self.candidates[0].output
        return ""


@dataclass
class ChatResponse:
    """聊天响应"""
    candidates: List[Message]
    messages: List[Message]  # 完整对话历史
    filters: Optional[List[Dict[str, Any]]] = None
    raw_response: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], messages: List[Message]) -> "ChatResponse":
        candidates = [
            Message.from_dict(c) 
            for c in data.get("candidates", [])
        ]
        return cls(
            candidates=candidates,
            messages=messages,
            filters=data.get("filters"),
            raw_response=data
        )
    
    @property
    def reply(self) -> str:
        """获取回复内容"""
        if self.candidates:
            return self.candidates[0].content
        return ""


@dataclass
class Embedding:
    """嵌入向量"""
    value: List[float]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Embedding":
        return cls(value=data.get("value", []))


@dataclass
class EmbeddingResponse:
    """嵌入响应"""
    embedding: Embedding
    raw_response: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingResponse":
        embedding_data = data.get("embedding", {})
        return cls(
            embedding=Embedding.from_dict(embedding_data),
            raw_response=data
        )


@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    version: str
    display_name: str
    description: str
    input_token_limit: int
    output_token_limit: int
    supported_generation_methods: List[str]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInfo":
        return cls(
            name=data.get("name", ""),
            version=data.get("version", ""),
            display_name=data.get("displayName", ""),
            description=data.get("description", ""),
            input_token_limit=data.get("inputTokenLimit", 0),
            output_token_limit=data.get("outputTokenLimit", 0),
            supported_generation_methods=data.get("supportedGenerationMethods", []),
            temperature=data.get("temperature"),
            top_p=data.get("topP"),
            top_k=data.get("topK")
        )


def _handle_api_error(response: httpx.Response) -> None:
    """处理API错误响应"""
    try:
        error_data = response.json()
        error = error_data.get("error", {})
        message = error.get("message", "Unknown error")
        code = error.get("code", response.status_code)
    except Exception:
        message = response.text or f"HTTP {response.status_code}"
        code = response.status_code
    
    if response.status_code == 401:
        raise AuthenticationError(message, status_code=code)
    elif response.status_code == 429:
        raise RateLimitError(message, status_code=code)
    elif response.status_code == 400:
        raise InvalidRequestError(message, status_code=code)
    elif response.status_code == 404:
        raise ModelNotFoundError(message, status_code=code)
    elif response.status_code >= 500:
        raise ServerError(message, status_code=code)
    else:
        raise GoogleAIError(message, status_code=code)


class PalmClient:
    """
    PaLM API客户端
    
    提供对Google PaLM API的完整访问，支持文本生成、聊天和嵌入。
    
    注意：PaLM API已被Gemini API取代，建议使用GeminiClient获得更好性能。
    
    Example:
        >>> client = PalmClient(api_key="your-api-key")
        >>> response = client.generate_text("What is machine learning?")
        >>> print(response.text)
    """
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[PalmConfig] = None):
        """
        初始化PaLM客户端
        
        Args:
            api_key: Google API密钥
            config: 客户端配置
        """
        self.config = config or PalmConfig()
        self.api_key = api_key or self.config.api_key or os.environ.get("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "API key is required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self._client: Optional[httpx.AsyncClient] = None
        self._sync_client: Optional[httpx.Client] = None
        
        logger.info("PalmClient initialized (deprecated, consider using GeminiClient)")
    
    def _get_client(self) -> httpx.AsyncClient:
        """获取或创建异步HTTP客户端"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.config.api_base,
                timeout=httpx.Timeout(self.config.timeout),
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key
                }
            )
        return self._client
    
    def _get_sync_client(self) -> httpx.Client:
        """获取或创建同步HTTP客户端"""
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = httpx.Client(
                base_url=self.config.api_base,
                timeout=httpx.Timeout(self.config.timeout),
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key
                }
            )
        return self._sync_client
    
    def _build_url(self, model: str, method: str) -> str:
        """构建API URL"""
        return f"/{self.config.api_version}/{model}:{method}"
    
    @retry(
        retry=retry_if_exception_type((RateLimitError, ServerError, httpx.NetworkError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def _make_request(
        self,
        method: str,
        url: str,
        json_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """发送HTTP请求（带重试）"""
        client = self._get_client()
        
        try:
            response = await client.request(
                method=method,
                url=url,
                json=json_data,
                params={"key": self.api_key}
            )
            
            if response.status_code != 200:
                _handle_api_error(response)
            
            return response.json()
        
        except httpx.HTTPStatusError as e:
            _handle_api_error(e.response)
            raise
    
    @retry(
        retry=retry_if_exception_type((RateLimitError, ServerError, httpx.NetworkError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _make_sync_request(
        self,
        method: str,
        url: str,
        json_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """发送同步HTTP请求（带重试）"""
        client = self._get_sync_client()
        
        try:
            response = client.request(
                method=method,
                url=url,
                json=json_data,
                params={"key": self.api_key}
            )
            
            if response.status_code != 200:
                _handle_api_error(response)
            
            return response.json()
        
        except httpx.HTTPStatusError as e:
            _handle_api_error(e.response)
            raise
    
    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        candidate_count: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> TextGenerationResponse:
        """
        异步生成文本
        
        Args:
            prompt: 提示文本
            model: 模型名称
            temperature: 温度参数 (0.0-1.0)
            top_p: top-p采样参数
            top_k: top-k采样参数
            candidate_count: 候选数量
            max_output_tokens: 最大输出token数
            stop_sequences: 停止序列
            
        Returns:
            TextGenerationResponse对象
        """
        model_name = model or self.config.default_text_model
        url = self._build_url(f"models/{model_name}", "generateText")
        
        config = TextGenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            candidate_count=candidate_count,
            max_output_tokens=max_output_tokens,
            stop_sequences=stop_sequences
        )
        
        body: Dict[str, Any] = {"prompt": {"text": prompt}}
        if config.to_dict():
            body["temperature"] = config.temperature if config.temperature is not None else 0.7
            body["candidateCount"] = config.candidate_count if config.candidate_count is not None else 1
            if config.max_output_tokens:
                body["maxOutputTokens"] = config.max_output_tokens
            if config.top_p:
                body["topP"] = config.top_p
            if config.top_k:
                body["topK"] = config.top_k
            if config.stop_sequences:
                body["stopSequences"] = config.stop_sequences
        
        logger.debug(f"Generating text with model: {model_name}")
        response_data = await self._make_request("POST", url, body)
        
        return TextGenerationResponse.from_dict(response_data)
    
    def generate_text_sync(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        candidate_count: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> TextGenerationResponse:
        """
        同步生成文本
        
        Args:
            prompt: 提示文本
            model: 模型名称
            temperature: 温度参数
            top_p: top-p采样参数
            top_k: top-k采样参数
            candidate_count: 候选数量
            max_output_tokens: 最大输出token数
            stop_sequences: 停止序列
            
        Returns:
            TextGenerationResponse对象
        """
        model_name = model or self.config.default_text_model
        url = self._build_url(f"models/{model_name}", "generateText")
        
        config = TextGenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            candidate_count=candidate_count,
            max_output_tokens=max_output_tokens,
            stop_sequences=stop_sequences
        )
        
        body: Dict[str, Any] = {"prompt": {"text": prompt}}
        if config.to_dict():
            body["temperature"] = config.temperature if config.temperature is not None else 0.7
            body["candidateCount"] = config.candidate_count if config.candidate_count is not None else 1
            if config.max_output_tokens:
                body["maxOutputTokens"] = config.max_output_tokens
            if config.top_p:
                body["topP"] = config.top_p
            if config.top_k:
                body["topK"] = config.top_k
            if config.stop_sequences:
                body["stopSequences"] = config.stop_sequences
        
        logger.debug(f"Generating text with model: {model_name}")
        response_data = self._make_sync_request("POST", url, body)
        
        return TextGenerationResponse.from_dict(response_data)
    
    async def generate_message(
        self,
        messages: List[Message],
        context: Optional[str] = None,
        examples: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        candidate_count: Optional[int] = None
    ) -> ChatResponse:
        """
        异步生成聊天消息
        
        Args:
            messages: 消息历史
            context: 上下文提示
            examples: 示例对话
            model: 模型名称
            temperature: 温度参数
            top_p: top-p采样参数
            top_k: top-k采样参数
            candidate_count: 候选数量
            
        Returns:
            ChatResponse对象
        """
        model_name = model or self.config.default_chat_model
        url = self._build_url(f"models/{model_name}", "generateMessage")
        
        body: Dict[str, Any] = {
            "prompt": {
                "messages": [m.to_dict() for m in messages]
            }
        }
        
        if context:
            body["prompt"]["context"] = context
        if examples:
            body["prompt"]["examples"] = examples
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["topP"] = top_p
        if top_k is not None:
            body["topK"] = top_k
        if candidate_count is not None:
            body["candidateCount"] = candidate_count
        
        logger.debug(f"Generating message with model: {model_name}")
        response_data = await self._make_request("POST", url, body)
        
        # 更新消息历史
        updated_messages = messages.copy()
        for candidate in response_data.get("candidates", []):
            updated_messages.append(Message.from_dict(candidate))
        
        return ChatResponse.from_dict(response_data, updated_messages)
    
    def generate_message_sync(
        self,
        messages: List[Message],
        context: Optional[str] = None,
        examples: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        candidate_count: Optional[int] = None
    ) -> ChatResponse:
        """
        同步生成聊天消息
        
        Args:
            messages: 消息历史
            context: 上下文提示
            examples: 示例对话
            model: 模型名称
            temperature: 温度参数
            top_p: top-p采样参数
            top_k: top-k采样参数
            candidate_count: 候选数量
            
        Returns:
            ChatResponse对象
        """
        model_name = model or self.config.default_chat_model
        url = self._build_url(f"models/{model_name}", "generateMessage")
        
        body: Dict[str, Any] = {
            "prompt": {
                "messages": [m.to_dict() for m in messages]
            }
        }
        
        if context:
            body["prompt"]["context"] = context
        if examples:
            body["prompt"]["examples"] = examples
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["topP"] = top_p
        if top_k is not None:
            body["topK"] = top_k
        if candidate_count is not None:
            body["candidateCount"] = candidate_count
        
        logger.debug(f"Generating message with model: {model_name}")
        response_data = self._make_sync_request("POST", url, body)
        
        # 更新消息历史
        updated_messages = messages.copy()
        for candidate in response_data.get("candidates", []):
            updated_messages.append(Message.from_dict(candidate))
        
        return ChatResponse.from_dict(response_data, updated_messages)
    
    async def embed_text(
        self,
        text: str,
        model: Optional[str] = None
    ) -> EmbeddingResponse:
        """
        异步生成文本嵌入
        
        Args:
            text: 输入文本
            model: 嵌入模型
            
        Returns:
            EmbeddingResponse对象
        """
        model_name = model or self.config.default_embedding_model
        url = self._build_url(f"models/{model_name}", "embedText")
        
        body = {"text": text}
        
        logger.debug(f"Embedding text with model: {model_name}")
        response_data = await self._make_request("POST", url, body)
        
        return EmbeddingResponse.from_dict(response_data)
    
    def embed_text_sync(
        self,
        text: str,
        model: Optional[str] = None
    ) -> EmbeddingResponse:
        """
        同步生成文本嵌入
        
        Args:
            text: 输入文本
            model: 嵌入模型
            
        Returns:
            EmbeddingResponse对象
        """
        model_name = model or self.config.default_embedding_model
        url = self._build_url(f"models/{model_name}", "embedText")
        
        body = {"text": text}
        
        logger.debug(f"Embedding text with model: {model_name}")
        response_data = self._make_sync_request("POST", url, body)
        
        return EmbeddingResponse.from_dict(response_data)
    
    async def batch_embed_texts(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[Embedding]:
        """
        异步批量生成文本嵌入
        
        Args:
            texts: 文本列表
            model: 嵌入模型
            
        Returns:
            Embedding列表
        """
        model_name = model or self.config.default_embedding_model
        url = self._build_url(f"models/{model_name}", "batchEmbedTexts")
        
        body = {"texts": texts}
        
        logger.debug(f"Batch embedding {len(texts)} texts with model: {model_name}")
        response_data = await self._make_request("POST", url, body)
        
        embeddings = response_data.get("embeddings", [])
        return [Embedding.from_dict(e) for e in embeddings]
    
    async def count_message_tokens(
        self,
        messages: List[Message],
        context: Optional[str] = None,
        examples: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None
    ) -> int:
        """
        异步统计消息token数量
        
        Args:
            messages: 消息列表
            context: 上下文提示
            examples: 示例对话
            model: 模型名称
            
        Returns:
            Token数量
        """
        model_name = model or self.config.default_chat_model
        url = self._build_url(f"models/{model_name}", "countMessageTokens")
        
        body: Dict[str, Any] = {
            "prompt": {
                "messages": [m.to_dict() for m in messages]
            }
        }
        
        if context:
            body["prompt"]["context"] = context
        if examples:
            body["prompt"]["examples"] = examples
        
        response_data = await self._make_request("POST", url, body)
        return response_data.get("tokenCount", 0)
    
    async def list_models(self) -> List[ModelInfo]:
        """
        异步列出可用模型
        
        Returns:
            ModelInfo列表
        """
        url = f"/{self.config.api_version}/models"
        response_data = await self._make_request("GET", url)
        
        models = response_data.get("models", [])
        return [ModelInfo.from_dict(m) for m in models]
    
    async def get_model(self, model_name: str) -> ModelInfo:
        """
        异步获取模型信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            ModelInfo对象
        """
        url = f"/{self.config.api_version}/{model_name}"
        response_data = await self._make_request("GET", url)
        return ModelInfo.from_dict(response_data)
    
    async def close(self) -> None:
        """关闭客户端连接"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.info("PalmClient async connection closed")
        
        if self._sync_client and not self._sync_client.is_closed:
            self._sync_client.close()
            logger.info("PalmClient sync connection closed")
    
    async def __aenter__(self) -> "PalmClient":
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """异步上下文管理器出口"""
        await self.close()


class PalmChatSession:
    """
    PaLM聊天会话
    
    管理多轮对话状态，提供便捷的聊天接口。
    
    Example:
        >>> client = PalmClient(api_key="your-api-key")
        >>> session = PalmChatSession(client, context="You are a helpful assistant.")
        >>> response = await session.send_message("Hello!")
        >>> print(response.reply)
    """
    
    def __init__(
        self,
        client: PalmClient,
        context: Optional[str] = None,
        examples: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None
    ):
        """
        初始化聊天会话
        
        Args:
            client: PalmClient实例
            context: 上下文提示
            examples: 示例对话
            model: 模型名称
            temperature: 温度参数
        """
        self.client = client
        self.context = context
        self.examples = examples
        self.model = model
        self.temperature = temperature
        self.messages: List[Message] = []
    
    async def send_message(self, message: str) -> ChatResponse:
        """
        发送消息并获取回复
        
        Args:
            message: 用户消息
            
        Returns:
            ChatResponse对象
        """
        # 添加用户消息
        self.messages.append(Message(author="0", content=message))
        
        # 生成回复
        response = await self.client.generate_message(
            messages=self.messages,
            context=self.context,
            examples=self.examples,
            model=self.model,
            temperature=self.temperature
        )
        
        # 更新消息历史
        self.messages = response.messages
        
        return response
    
    def clear_history(self) -> None:
        """清除对话历史"""
        self.messages = []
    
    @property
    def history(self) -> List[Message]:
        """获取对话历史"""
        return self.messages.copy()
