"""
Vertex AI客户端 - Google AI兼容接口

提供对Google Cloud Vertex AI的完整支持，包括：
- Gemini模型（通过Vertex AI）
- 文本生成和聊天
- 嵌入生成
- 多模态处理
- 流式响应
- 自定义模型部署
- 错误处理和重试机制

模块路径: compat/google/vertex_ai.py
"""

from __future__ import annotations

import os
import json
import base64
import logging
from dataclasses import dataclass, field
from typing import (
    Any, AsyncIterator, Dict, List, Optional, Union, Callable, BinaryIO
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
    InvalidRequestError, ModelNotFoundError, ServerError, ConfigurationError
)

logger = logging.getLogger(__name__)

# Vertex AI常量
VERTEX_AI_API_BASE_TEMPLATE = "https://{location}-aiplatform.googleapis.com"
VERTEX_AI_API_VERSION = "v1"


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
class VertexAIConfig:
    """Vertex AI客户端配置"""
    project_id: Optional[str] = None
    location: str = "us-central1"
    credentials_path: Optional[str] = None
    api_version: str = VERTEX_AI_API_VERSION
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
    
    @property
    def api_base(self) -> str:
        """获取API基础URL"""
        return VERTEX_AI_API_BASE_TEMPLATE.format(location=self.location)


@dataclass
class ContentPart:
    """内容部分（文本或图像）"""
    text: Optional[str] = None
    mime_type: Optional[str] = None
    data: Optional[bytes] = None
    file_uri: Optional[str] = None
    
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
    
    @classmethod
    def from_file_uri(cls, uri: str, mime_type: str = "application/pdf") -> "ContentPart":
        """从GCS文件URI创建内容部分"""
        return cls(file_uri=uri, mime_type=mime_type)
    
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
        elif self.file_uri is not None:
            return {
                "fileData": {
                    "mimeType": self.mime_type,
                    "fileUri": self.file_uri
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
    citation_metadata: Optional[Dict[str, Any]] = None
    
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
            safety_ratings=data.get("safetyRatings"),
            citation_metadata=data.get("citationMetadata")
        )
    
    @property
    def text(self) -> str:
        """获取文本内容"""
        texts = [p.text for p in self.content.parts if p.text]
        return "".join(texts)


@dataclass
class GenerateContentResponse:
    """生成内容响应"""
    candidates: List[Candidate]
    prompt_feedback: Optional[Dict[str, Any]] = None
    usage_metadata: Optional[Dict[str, Any]] = None
    raw_response: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerateContentResponse":
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
    statistics: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Embedding":
        return cls(
            values=data.get("values", []),
            statistics=data.get("statistics")
        )


@dataclass
class Prediction:
    """预测结果"""
    embeddings: Optional[Embedding] = None
    content: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Prediction":
        embeddings_data = data.get("embeddings", {})
        embeddings = Embedding.from_dict(embeddings_data) if embeddings_data else None
        return cls(
            embeddings=embeddings,
            content=data.get("content")
        )


@dataclass
class EmbedTextResponse:
    """文本嵌入响应"""
    predictions: List[Prediction]
    deployed_model_id: Optional[str] = None
    model_version_id: Optional[str] = None
    model: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbedTextResponse":
        predictions = [Prediction.from_dict(p) for p in data.get("predictions", [])]
        return cls(
            predictions=predictions,
            deployed_model_id=data.get("deployedModelId"),
            model_version_id=data.get("modelVersionId"),
            model=data.get("model")
        )
    
    @property
    def embeddings(self) -> List[Embedding]:
        """获取所有嵌入向量"""
        return [p.embeddings for p in self.predictions if p.embeddings]


@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    display_name: str
    description: str
    version_id: str
    version_aliases: List[str]
    version_create_time: str
    version_update_time: str
    model_type: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInfo":
        return cls(
            name=data.get("name", ""),
            display_name=data.get("displayName", ""),
            description=data.get("description", ""),
            version_id=data.get("versionId", ""),
            version_aliases=data.get("versionAliases", []),
            version_create_time=data.get("versionCreateTime", ""),
            version_update_time=data.get("versionUpdateTime", ""),
            model_type=data.get("modelType", "")
        )


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
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".html": "text/html",
        ".json": "application/json",
    }
    return mime_types.get(suffix.lower(), "application/octet-stream")


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
    elif response.status_code == 403:
        raise AuthenticationError(f"Permission denied: {message}", status_code=code)
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


class VertexAIClient:
    """
    Vertex AI客户端
    
    提供对Google Cloud Vertex AI的完整访问，支持Gemini模型、嵌入和自定义模型。
    
    Example:
        >>> client = VertexAIClient(project_id="my-project", location="us-central1")
        >>> response = await client.generate_content([Content.from_text("Hello!")])
        >>> print(response.text)
    """
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        credentials_path: Optional[str] = None,
        config: Optional[VertexAIConfig] = None
    ):
        """
        初始化Vertex AI客户端
        
        Args:
            project_id: Google Cloud项目ID
            location: Google Cloud区域
            credentials_path: 服务账号凭证文件路径
            config: 客户端配置
        """
        self.config = config or VertexAIConfig()
        self.project_id = project_id or self.config.project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = location or self.config.location
        self.credentials_path = credentials_path or self.config.credentials_path
        
        if not self.project_id:
            raise ConfigurationError(
                "Project ID is required. Set GOOGLE_CLOUD_PROJECT environment variable "
                "or pass project_id parameter."
            )
        
        self._client: Optional[httpx.AsyncClient] = None
        self._sync_client: Optional[httpx.Client] = None
        self._access_token: Optional[str] = None
        
        logger.info(f"VertexAIClient initialized for project: {self.project_id}, location: {self.location}")
    
    def _get_access_token(self) -> str:
        """获取GCP访问令牌"""
        import subprocess
        
        if self._access_token:
            return self._access_token
        
        try:
            # 尝试使用gcloud获取访问令牌
            result = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                capture_output=True,
                text=True,
                check=True
            )
            self._access_token = result.stdout.strip()
            return self._access_token
        except (subprocess.CalledProcessError, FileNotFoundError):
            # 如果gcloud不可用，尝试从环境变量获取
            token = os.environ.get("GOOGLE_ACCESS_TOKEN")
            if token:
                return token
            raise AuthenticationError(
                "Failed to get access token. Please ensure gcloud is installed and authenticated, "
                "or set GOOGLE_ACCESS_TOKEN environment variable."
            )
    
    def _get_client(self) -> httpx.AsyncClient:
        """获取或创建异步HTTP客户端"""
        if self._client is None or self._client.is_closed:
            token = self._get_access_token()
            self._client = httpx.AsyncClient(
                base_url=self.config.api_base,
                timeout=httpx.Timeout(self.config.timeout),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}"
                }
            )
        return self._client
    
    def _get_sync_client(self) -> httpx.Client:
        """获取或创建同步HTTP客户端"""
        if self._sync_client is None or self._sync_client.is_closed:
            token = self._get_access_token()
            self._sync_client = httpx.Client(
                base_url=self.config.api_base,
                timeout=httpx.Timeout(self.config.timeout),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}"
                }
            )
        return self._sync_client
    
    def _build_model_path(self, model: str) -> str:
        """构建模型路径"""
        return f"projects/{self.project_id}/locations/{self.location}/publishers/google/models/{model}"
    
    def _build_endpoint_path(self, endpoint_id: str) -> str:
        """构建端点路径"""
        return f"projects/{self.project_id}/locations/{self.location}/endpoints/{endpoint_id}"
    
    def _build_url(self, path: str, method: str, stream: bool = False) -> str:
        """构建API URL"""
        stream_suffix = "?alt=sse" if stream else ""
        return f"/{self.config.api_version}/{path}:{method}{stream_suffix}"
    
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
                json=json_data
            )
            
            if response.status_code == 401:
                # Token可能已过期，尝试刷新
                self._access_token = None
                client = self._get_client()
                response = await client.request(
                    method=method,
                    url=url,
                    json=json_data
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
                json=json_data
            )
            
            if response.status_code == 401:
                # Token可能已过期，尝试刷新
                self._access_token = None
                client = self._get_sync_client()
                response = client.request(
                    method=method,
                    url=url,
                    json=json_data
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
    ) -> GenerateContentResponse:
        """
        异步生成内容
        
        Args:
            contents: 消息内容列表
            model: 模型名称
            generation_config: 生成配置
            safety_settings: 安全设置
            
        Returns:
            GenerateContentResponse对象
        """
        model_name = model or self.config.default_model
        model_path = self._build_model_path(model_name)
        url = self._build_url(model_path, "generateContent")
        body = self._build_request_body(contents, generation_config, safety_settings)
        
        logger.debug(f"Generating content with model: {model_name}")
        response_data = await self._make_request("POST", url, body)
        
        return GenerateContentResponse.from_dict(response_data)
    
    def generate_content_sync(
        self,
        contents: List[Content],
        model: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
        safety_settings: Optional[List[SafetySetting]] = None
    ) -> GenerateContentResponse:
        """
        同步生成内容
        
        Args:
            contents: 消息内容列表
            model: 模型名称
            generation_config: 生成配置
            safety_settings: 安全设置
            
        Returns:
            GenerateContentResponse对象
        """
        model_name = model or self.config.default_model
        model_path = self._build_model_path(model_name)
        url = self._build_url(model_path, "generateContent")
        body = self._build_request_body(contents, generation_config, safety_settings)
        
        logger.debug(f"Generating content with model: {model_name}")
        response_data = self._make_sync_request("POST", url, body)
        
        return GenerateContentResponse.from_dict(response_data)
    
    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> GenerateContentResponse:
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
            GenerateContentResponse对象
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
    ) -> GenerateContentResponse:
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
            GenerateContentResponse对象
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
        model_path = self._build_model_path(model_name)
        url = self._build_url(model_path, "streamGenerateContent", stream=True)
        body = self._build_request_body(contents, generation_config, safety_settings)
        
        logger.debug(f"Streaming content with model: {model_name}")
        
        client = self._get_client()
        async with client.stream("POST", url, json=body) as response:
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
    
    async def embed_text(
        self,
        texts: Union[str, List[str]],
        model: str = "textembedding-gecko@003"
    ) -> EmbedTextResponse:
        """
        异步生成文本嵌入
        
        Args:
            texts: 文本或文本列表
            model: 嵌入模型
            
        Returns:
            EmbedTextResponse对象
        """
        model_path = self._build_model_path(model)
        url = self._build_url(model_path, "predict")
        
        if isinstance(texts, str):
            texts = [texts]
        
        instances = [{"content": text} for text in texts]
        body = {"instances": instances}
        
        logger.debug(f"Embedding {len(texts)} texts with model: {model}")
        response_data = await self._make_request("POST", url, body)
        
        return EmbedTextResponse.from_dict(response_data)
    
    def embed_text_sync(
        self,
        texts: Union[str, List[str]],
        model: str = "textembedding-gecko@003"
    ) -> EmbedTextResponse:
        """
        同步生成文本嵌入
        
        Args:
            texts: 文本或文本列表
            model: 嵌入模型
            
        Returns:
            EmbedTextResponse对象
        """
        model_path = self._build_model_path(model)
        url = self._build_url(model_path, "predict")
        
        if isinstance(texts, str):
            texts = [texts]
        
        instances = [{"content": text} for text in texts]
        body = {"instances": instances}
        
        logger.debug(f"Embedding {len(texts)} texts with model: {model}")
        response_data = self._make_sync_request("POST", url, body)
        
        return EmbedTextResponse.from_dict(response_data)
    
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
        model_path = self._build_model_path(model_name)
        url = self._build_url(model_path, "countTokens")
        body = {"contents": [c.to_dict() for c in contents]}
        
        response_data = await self._make_request("POST", url, body)
        return response_data.get("totalTokens", 0)
    
    async def predict_custom_model(
        self,
        endpoint_id: str,
        instances: List[Dict[str, Any]],
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        异步调用自定义模型端点
        
        Args:
            endpoint_id: 端点ID
            instances: 输入实例列表
            parameters: 预测参数
            
        Returns:
            预测结果
        """
        endpoint_path = self._build_endpoint_path(endpoint_id)
        url = self._build_url(endpoint_path, "predict")
        
        body: Dict[str, Any] = {"instances": instances}
        if parameters:
            body["parameters"] = parameters
        
        logger.debug(f"Predicting with custom endpoint: {endpoint_id}")
        return await self._make_request("POST", url, body)
    
    async def list_models(
        self,
        filter_str: Optional[str] = None,
        page_size: int = 100
    ) -> List[ModelInfo]:
        """
        异步列出可用模型
        
        Args:
            filter_str: 过滤条件
            page_size: 每页大小
            
        Returns:
            ModelInfo列表
        """
        url = f"/{self.config.api_version}/projects/{self.project_id}/locations/{self.location}/models"
        
        params: Dict[str, Any] = {"pageSize": page_size}
        if filter_str:
            params["filter"] = filter_str
        
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
        url = f"/{self.config.api_version}/projects/{self.project_id}/locations/{self.location}/models/{model_name}"
        response_data = await self._make_request("GET", url)
        return ModelInfo.from_dict(response_data)
    
    async def close(self) -> None:
        """关闭客户端连接"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.info("VertexAIClient async connection closed")
        
        if self._sync_client and not self._sync_client.is_closed:
            self._sync_client.close()
            logger.info("VertexAIClient sync connection closed")
    
    async def __aenter__(self) -> "VertexAIClient":
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """异步上下文管理器出口"""
        await self.close()


class VertexAIChatSession:
    """
    Vertex AI聊天会话
    
    管理多轮对话状态，提供便捷的聊天接口。
    
    Example:
        >>> client = VertexAIClient(project_id="my-project")
        >>> session = VertexAIChatSession(client)
        >>> response = await session.send_message("Hello!")
        >>> print(response.text)
    """
    
    def __init__(
        self,
        client: VertexAIClient,
        model: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
        safety_settings: Optional[List[SafetySetting]] = None
    ):
        """
        初始化聊天会话
        
        Args:
            client: VertexAIClient实例
            model: 模型名称
            generation_config: 生成配置
            safety_settings: 安全设置
        """
        self.client = client
        self.model = model
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self.history: List[Content] = []
    
    async def send_message(self, message: str) -> GenerateContentResponse:
        """
        发送消息并获取回复
        
        Args:
            message: 用户消息
            
        Returns:
            GenerateContentResponse对象
        """
        # 添加用户消息
        self.history.append(Content.from_text(message, role="user"))
        
        # 生成回复
        response = await self.client.generate_content(
            contents=self.history,
            model=self.model,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        )
        
        # 更新历史
        if response.candidates:
            self.history.append(response.candidates[0].content)
        
        return response
    
    async def send_message_stream(self, message: str) -> AsyncIterator[str]:
        """
        流式发送消息并获取回复
        
        Args:
            message: 用户消息
            
        Yields:
            文本片段
        """
        # 添加用户消息
        self.history.append(Content.from_text(message, role="user"))
        
        # 流式生成回复
        full_response = ""
        async for candidate in self.client.stream_generate_content(
            contents=self.history,
            model=self.model,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        ):
            for part in candidate.content.parts:
                if part.text:
                    full_response += part.text
                    yield part.text
        
        # 更新历史
        self.history.append(Content.from_text(full_response, role="model"))
    
    def clear_history(self) -> None:
        """清除对话历史"""
        self.history = []
    
    def get_history(self) -> List[Content]:
        """获取对话历史"""
        return self.history.copy()
