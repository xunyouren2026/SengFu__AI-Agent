"""
Gemini API客户端 - Google AI兼容接口

提供对Google Gemini API的完整支持，包括：
- 文本生成
- 多模态处理（文本+图像）
- 嵌入生成
- 流式响应
- 错误处理和重试机制

模块路径: compat/google/gemini.py
"""

from __future__ import annotations

import os
import json
import base64
import asyncio
import logging
from dataclasses import dataclass, field
from typing import (
    Any, AsyncIterator, Dict, List, Optional, Union, 
    Callable, Iterator, BinaryIO, Tuple
)
from pathlib import Path
from enum import Enum

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

# Gemini API常量
GEMINI_API_BASE = "https://generativelanguage.googleapis.com"
GEMINI_API_VERSION = "v1beta"


class HarmCategory(str, Enum):
    """有害内容类别"""
    HARASSMENT = "HARM_CATEGORY_HARASSMENT"
    HATE_SPEECH = "HARM_CATEGORY_HATE_SPEECH"
    SEXUALLY_EXPLICIT = "HARM_CATEGORY_SEXUALLY_EXPLICIT"
    DANGEROUS_CONTENT = "HARM_CATEGORY_DANGEROUS_CONTENT"


class HarmBlockThreshold(str, Enum):
    """有害内容拦截阈值"""
    UNSPECIFIED = "HARM_BLOCK_THRESHOLD_UNSPECIFIED"
    BLOCK_LOW_AND_ABOVE = "BLOCK_LOW_AND_ABOVE"
    BLOCK_MEDIUM_AND_ABOVE = "BLOCK_MEDIUM_AND_ABOVE"
    BLOCK_ONLY_HIGH = "BLOCK_ONLY_HIGH"
    BLOCK_NONE = "BLOCK_NONE"


class FinishReason(str, Enum):
    """生成结束原因"""
    FINISH_REASON_UNSPECIFIED = "FINISH_REASON_UNSPECIFIED"
    STOP = "STOP"
    MAX_TOKENS = "MAX_TOKENS"
    SAFETY = "SAFETY"
    RECITATION = "RECITATION"
    OTHER = "OTHER"


@dataclass
class SafetySetting:
    """安全设置配置"""
    category: HarmCategory
    threshold: HarmBlockThreshold
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "category": self.category.value,
            "threshold": self.threshold.value
        }


@dataclass
class GenerationConfig:
    """生成配置"""
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_output_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    candidate_count: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        if self.temperature is not None:
            config["temperature"] = self.temperature
        if self.top_p is not None:
            config["topP"] = self.top_p
        if self.top_k is not None:
            config["topK"] = self.top_k
        if self.max_output_tokens is not None:
            config["maxOutputTokens"] = self.max_output_tokens
        if self.stop_sequences:
            config["stopSequences"] = self.stop_sequences
        if self.candidate_count is not None:
            config["candidateCount"] = self.candidate_count
        return config


@dataclass
class GeminiConfig:
    """Gemini客户端配置"""
    api_key: Optional[str] = None
    api_base: str = GEMINI_API_BASE
    api_version: str = GEMINI_API_VERSION
    timeout: float = 60.0
    max_retries: int = 3
    default_model: str = "gemini-1.5-flash"
    
    # 默认安全设置
    safety_settings: List[SafetySetting] = field(default_factory=lambda: [
        SafetySetting(HarmCategory.HARASSMENT, HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
        SafetySetting(HarmCategory.HATE_SPEECH, HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
        SafetySetting(HarmCategory.SEXUALLY_EXPLICIT, HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
        SafetySetting(HarmCategory.DANGEROUS_CONTENT, HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
    ])
    
    # 默认生成配置
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)


@dataclass
class ContentPart:
    """内容部分（文本或图像）"""
    text: Optional[str] = None
    mime_type: Optional[str] = None
    data: Optional[bytes] = None
    
    @classmethod
    def from_text(cls, text: str) -> "ContentPart":
        """从文本创建内容部分"""
        return cls(text=text)
    
    @classmethod
    def from_image_bytes(cls, data: bytes, mime_type: str = "image/jpeg") -> "ContentPart":
        """从图像字节创建内容部分"""
        return cls(mime_type=mime_type, data=data)
    
    @classmethod
    def from_image_file(cls, file_path: Union[str, Path], mime_type: Optional[str] = None) -> "ContentPart":
        """从图像文件创建内容部分"""
        path = Path(file_path)
        if mime_type is None:
            mime_type = _guess_mime_type(path.suffix)
        with open(path, "rb") as f:
            data = f.read()
        return cls.from_image_bytes(data, mime_type)
    
    def to_dict(self) -> Dict[str, Any]:
        if self.text is not None:
            return {"text": self.text}
        elif self.data is not None:
            return {
                "inlineData": {
                    "mimeType": self.mime_type,
                    "data": base64.b64encode(self.data).decode("utf-8")
                }
            }
        return {}


@dataclass
class Content:
    """消息内容"""
    role: str
    parts: List[ContentPart]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "parts": [part.to_dict() for part in self.parts]
        }
    
    @classmethod
    def from_text(cls, text: str, role: str = "user") -> "Content":
        """从文本创建内容"""
        return cls(role=role, parts=[ContentPart.from_text(text)])
    
    @classmethod
    def from_multimodal(
        cls, 
        text: str, 
        images: List[Union[str, Path, bytes]],
        role: str = "user"
    ) -> "Content":
        """从文本和图像创建多模态内容"""
        parts: List[ContentPart] = [ContentPart.from_text(text)]
        
        for img in images:
            if isinstance(img, (str, Path)):
                parts.append(ContentPart.from_image_file(img))
            elif isinstance(img, bytes):
                parts.append(ContentPart.from_image_bytes(img))
        
        return cls(role=role, parts=parts)


@dataclass
class Candidate:
    """生成候选结果"""
    content: Content
    finish_reason: Optional[FinishReason] = None
    index: int = 0
    safety_ratings: Optional[List[Dict[str, Any]]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Candidate":
        content_data = data.get("content", {})
        content = Content(
            role=content_data.get("role", "model"),
            parts=[ContentPart(text=p.get("text", "")) for p in content_data.get("parts", [])]
        )
        
        finish_reason_str = data.get("finishReason", "")
        finish_reason = None
        if finish_reason_str:
            try:
                finish_reason = FinishReason(finish_reason_str)
            except ValueError:
                finish_reason = FinishReason.FINISH_REASON_UNSPECIFIED
        
        return cls(
            content=content,
            finish_reason=finish_reason,
            index=data.get("index", 0),
            safety_ratings=data.get("safetyRatings")
        )
    
    @property
    def text(self) -> str:
        """获取文本内容"""
        texts = [p.text for p in self.content.parts if p.text]
        return "".join(texts)


@dataclass
class GeminiResponse:
    """Gemini API响应"""
    candidates: List[Candidate]
    prompt_feedback: Optional[Dict[str, Any]] = None
    usage_metadata: Optional[Dict[str, Any]] = None
    raw_response: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeminiResponse":
        candidates = [Candidate.from_dict(c) for c in data.get("candidates", [])]
        return cls(
            candidates=candidates,
            prompt_feedback=data.get("promptFeedback"),
            usage_metadata=data.get("usageMetadata"),
            raw_response=data
        )
    
    @property
    def text(self) -> str:
        """获取第一个候选的文本"""
        if self.candidates:
            return self.candidates[0].text
        return ""
    
    @property
    def finish_reason(self) -> Optional[FinishReason]:
        """获取第一个候选的结束原因"""
        if self.candidates:
            return self.candidates[0].finish_reason
        return None


@dataclass
class Embedding:
    """嵌入向量"""
    values: List[float]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Embedding":
        return cls(values=data.get("values", []))


@dataclass
class EmbeddingResponse:
    """嵌入响应"""
    embeddings: List[Embedding]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingResponse":
        embeddings = [Embedding.from_dict(e) for e in data.get("embeddings", [])]
        return cls(embeddings=embeddings)


def _guess_mime_type(suffix: str) -> str:
    """根据文件后缀猜测MIME类型"""
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }
    return mime_types.get(suffix.lower(), "image/jpeg")


def _handle_api_error(response: httpx.Response) -> None:
    """处理API错误响应"""
    try:
        error_data = response.json()
        error = error_data.get("error", {})
        message = error.get("message", "Unknown error")
        code = error.get("code", response.status_code)
        status = error.get("status", "")
    except Exception:
        message = response.text or f"HTTP {response.status_code}"
        code = response.status_code
        status = ""
    
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


class GeminiClient:
    """
    Gemini API客户端
    
    提供对Google Gemini API的完整访问，支持文本生成、多模态处理和嵌入。
    
    Example:
        >>> client = GeminiClient(api_key="your-api-key")
        >>> response = client.generate_text("What is AI?")
        >>> print(response.text)
    """
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[GeminiConfig] = None):
        """
        初始化Gemini客户端
        
        Args:
            api_key: Google API密钥
            config: 客户端配置
        """
        self.config = config or GeminiConfig()
        self.api_key = api_key or self.config.api_key or os.environ.get("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        self._client: Optional[httpx.AsyncClient] = None
        self._sync_client: Optional[httpx.Client] = None
        
        logger.info(f"GeminiClient initialized with model: {self.config.default_model}")
    
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
    
    def _build_url(self, model: str, endpoint: str, stream: bool = False) -> str:
        """构建API URL"""
        stream_suffix = "?alt=sse" if stream else ""
        return f"/{self.config.api_version}/models/{model}:{endpoint}{stream_suffix}"
    
    def _build_request_body(
        self,
        contents: List[Content],
        generation_config: Optional[GenerationConfig] = None,
        safety_settings: Optional[List[SafetySetting]] = None
    ) -> Dict[str, Any]:
        """构建请求体"""
        body: Dict[str, Any] = {
            "contents": [c.to_dict() for c in contents]
        }
        
        config = generation_config or self.config.generation_config
        if config.to_dict():
            body["generationConfig"] = config.to_dict()
        
        safety = safety_settings or self.config.safety_settings
        if safety:
            body["safetySettings"] = [s.to_dict() for s in safety]
        
        return body
    
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
        json_data: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Union[Dict[str, Any], httpx.Response]:
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
            
            if stream:
                return response
            
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
        json_data: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Union[Dict[str, Any], httpx.Response]:
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
            
            if stream:
                return response
            
            return response.json()
        
        except httpx.HTTPStatusError as e:
            _handle_api_error(e.response)
            raise
    
    async def generate_content(
        self,
        contents: List[Content],
        model: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
        safety_settings: Optional[List[SafetySetting]] = None
    ) -> GeminiResponse:
        """
        异步生成内容
        
        Args:
            contents: 消息内容列表
            model: 模型名称
            generation_config: 生成配置
            safety_settings: 安全设置
            
        Returns:
            GeminiResponse对象
        """
        model_name = model or self.config.default_model
        url = self._build_url(model_name, "generateContent")
        body = self._build_request_body(contents, generation_config, safety_settings)
        
        logger.debug(f"Generating content with model: {model_name}")
        response_data = await self._make_request("POST", url, body)
        
        return GeminiResponse.from_dict(response_data)
    
    def generate_content_sync(
        self,
        contents: List[Content],
        model: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
        safety_settings: Optional[List[SafetySetting]] = None
    ) -> GeminiResponse:
        """
        同步生成内容
        
        Args:
            contents: 消息内容列表
            model: 模型名称
            generation_config: 生成配置
            safety_settings: 安全设置
            
        Returns:
            GeminiResponse对象
        """
        model_name = model or self.config.default_model
        url = self._build_url(model_name, "generateContent")
        body = self._build_request_body(contents, generation_config, safety_settings)
        
        logger.debug(f"Generating content with model: {model_name}")
        response_data = self._make_sync_request("POST", url, body)
        
        return GeminiResponse.from_dict(response_data)
    
    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> GeminiResponse:
        """
        异步生成文本（便捷方法）
        
        Args:
            prompt: 提示文本
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大输出token数
            top_p: top-p采样参数
            top_k: top-k采样参数
            stop_sequences: 停止序列
            
        Returns:
            GeminiResponse对象
        """
        contents = [Content.from_text(prompt)]
        config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_tokens,
            stop_sequences=stop_sequences
        )
        
        return await self.generate_content(contents, model, config)
    
    def generate_text_sync(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> GeminiResponse:
        """
        同步生成文本（便捷方法）
        
        Args:
            prompt: 提示文本
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大输出token数
            top_p: top-p采样参数
            top_k: top-k采样参数
            stop_sequences: 停止序列
            
        Returns:
            GeminiResponse对象
        """
        contents = [Content.from_text(prompt)]
        config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_tokens,
            stop_sequences=stop_sequences
        )
        
        return self.generate_content_sync(contents, model, config)
    
    async def generate_multimodal(
        self,
        prompt: str,
        images: List[Union[str, Path, bytes]],
        model: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None
    ) -> GeminiResponse:
        """
        异步生成多模态内容
        
        Args:
            prompt: 提示文本
            images: 图像列表（文件路径或字节）
            model: 模型名称
            generation_config: 生成配置
            
        Returns:
            GeminiResponse对象
        """
        content = Content.from_multimodal(prompt, images)
        return await self.generate_content([content], model, generation_config)
    
    async def stream_generate_content(
        self,
        contents: List[Content],
        model: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
        safety_settings: Optional[List[SafetySetting]] = None
    ) -> AsyncIterator[Candidate]:
        """
        异步流式生成内容
        
        Args:
            contents: 消息内容列表
            model: 模型名称
            generation_config: 生成配置
            safety_settings: 安全设置
            
        Yields:
            Candidate对象
        """
        model_name = model or self.config.default_model
        url = self._build_url(model_name, "streamGenerateContent", stream=True)
        body = self._build_request_body(contents, generation_config, safety_settings)
        
        logger.debug(f"Streaming content with model: {model_name}")
        
        client = self._get_client()
        async with client.stream(
            "POST",
            url,
            json=body,
            params={"key": self.api_key}
        ) as response:
            if response.status_code != 200:
                _handle_api_error(await response.aread())
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        for candidate_data in data.get("candidates", []):
                            yield Candidate.from_dict(candidate_data)
                    except json.JSONDecodeError:
                        continue
    
    async def stream_generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        异步流式生成文本（便捷方法）
        
        Args:
            prompt: 提示文本
            model: 模型名称
            **kwargs: 其他生成参数
            
        Yields:
            文本片段
        """
        contents = [Content.from_text(prompt)]
        config = GenerationConfig(**kwargs) if kwargs else None
        
        async for candidate in self.stream_generate_content(contents, model, config):
            for part in candidate.content.parts:
                if part.text:
                    yield part.text
    
    async def embed_content(
        self,
        content: Union[str, Content],
        model: str = "models/embedding-001",
        task_type: Optional[str] = None,
        title: Optional[str] = None
    ) -> EmbeddingResponse:
        """
        异步生成嵌入向量
        
        Args:
            content: 文本内容或Content对象
            model: 嵌入模型
            task_type: 任务类型
            title: 标题（用于RETRIEVAL_DOCUMENT任务）
            
        Returns:
            EmbeddingResponse对象
        """
        url = f"/{self.config.api_version}/{model}:embedContent"
        
        if isinstance(content, str):
            content = Content.from_text(content)
        
        body: Dict[str, Any] = {"content": content.to_dict()}
        
        if task_type:
            body["taskType"] = task_type
        if title:
            body["title"] = title
        
        logger.debug(f"Embedding content with model: {model}")
        response_data = await self._make_request("POST", url, body)
        
        return EmbeddingResponse.from_dict(response_data.get("embedding", {}))
    
    async def batch_embed_contents(
        self,
        contents: List[Union[str, Content]],
        model: str = "models/embedding-001"
    ) -> List[Embedding]:
        """
        异步批量生成嵌入向量
        
        Args:
            contents: 内容列表
            model: 嵌入模型
            
        Returns:
            Embedding列表
        """
        url = f"/{self.config.api_version}/{model}:batchEmbedContents"
        
        requests = []
        for content in contents:
            if isinstance(content, str):
                content = Content.from_text(content)
            requests.append({"content": content.to_dict()})
        
        body = {"requests": requests}
        
        logger.debug(f"Batch embedding {len(contents)} contents with model: {model}")
        response_data = await self._make_request("POST", url, body)
        
        embeddings = response_data.get("embeddings", [])
        return [Embedding.from_dict(e) for e in embeddings]
    
    async def count_tokens(
        self,
        contents: List[Content],
        model: Optional[str] = None
    ) -> int:
        """
        异步统计token数量
        
        Args:
            contents: 内容列表
            model: 模型名称
            
        Returns:
            Token数量
        """
        model_name = model or self.config.default_model
        url = self._build_url(model_name, "countTokens")
        body = {"contents": [c.to_dict() for c in contents]}
        
        response_data = await self._make_request("POST", url, body)
        return response_data.get("totalTokens", 0)
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        异步列出可用模型
        
        Returns:
            模型信息列表
        """
        url = f"/{self.config.api_version}/models"
        response_data = await self._make_request("GET", url)
        return response_data.get("models", [])
    
    async def get_model(self, model_name: str) -> Dict[str, Any]:
        """
        异步获取模型信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型信息
        """
        url = f"/{self.config.api_version}/models/{model_name}"
        return await self._make_request("GET", url)
    
    async def close(self) -> None:
        """关闭客户端连接"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.info("GeminiClient async connection closed")
        
        if self._sync_client and not self._sync_client.is_closed:
            self._sync_client.close()
            logger.info("GeminiClient sync connection closed")
    
    async def __aenter__(self) -> "GeminiClient":
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """异步上下文管理器出口"""
        await self.close()
