"""
AGI Unified Framework - Ollama Adapter Module
Ollama本地模型适配器：模型管理、拉取/推送操作、嵌入生成、聊天完成、流式支持

提供与Ollama本地LLM服务的完整集成，支持模型生命周期管理和多种推理模式。
"""

from __future__ import annotations

import json
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from .base import (
    FinishReason,
    GenerateParams,
    LLMBackend,
    LLMChunk,
    LLMError,
    LLMResponse,
    Message,
    ModelInfo,
    ToolCall,
    Usage,
)


class OllamaError(LLMError):
    """Ollama特定错误"""
    pass


class ModelStatus(str, Enum):
    """模型状态"""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    VERIFYING = "verifying"
    READY = "ready"
    ERROR = "error"
    UNLOADING = "unloading"


@dataclass
class ModelManifest:
    """Ollama模型清单"""
    name: str
    tag: str = "latest"
    architecture: str = ""
    parameters: str = ""
    quantization: str = ""
    format: str = "gguf"
    size_bytes: int = 0
    digest: str = ""
    layers: List[Dict[str, Any]] = field(default_factory=list)
    template: str = ""
    system_prompt: str = ""
    license: str = ""
    
    @property
    def full_name(self) -> str:
        return f"{self.name}:{self.tag}"
    
    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)


@dataclass
class PullProgress:
    """拉取进度"""
    status: str
    completed: int = 0
    total: int = 0
    percent: float = 0.0
    layer_id: str = ""
    layer_type: str = ""
    
    @property
    def is_complete(self) -> bool:
        return self.completed >= self.total and self.total > 0


@dataclass
class EmbeddingResult:
    """嵌入结果"""
    text: str
    embedding: List[float]
    model: str
    tokens: int = 0
    duration_ms: float = 0.0


class OllamaModelManager:
    """
    Ollama模型管理器
    
    管理本地Ollama模型的生命周期，包括列出、拉取、推送、删除和复制模型。
    """
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self._lock = threading.RLock()
        self._local_models: Dict[str, ModelManifest] = {}
        self._active_pulls: Dict[str, threading.Thread] = {}
        self._pull_progress: Dict[str, PullProgress] = {}
    
    def _api_request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        发送API请求
        
        Args:
            endpoint: API端点
            method: HTTP方法
            data: 请求数据
            stream: 是否流式响应
            
        Returns:
            响应数据或流式迭代器
        """
        url = f"{self.base_url}/api/{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        request = urllib.request.Request(
            url,
            data=json.dumps(data).encode() if data else None,
            headers=headers,
            method=method,
        )
        
        try:
            response = urllib.request.urlopen(request, timeout=300)
            
            if stream:
                return self._parse_stream(response)
            else:
                return json.loads(response.read().decode())
                
        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            raise OllamaError(f"HTTP {e.code}: {error_body}", status_code=e.code)
        except urllib.error.URLError as e:
            raise OllamaError(f"Connection error: {e.reason}", retryable=True)
    
    def _parse_stream(self, response) -> Iterator[Dict[str, Any]]:
        """解析流式响应"""
        for line in response:
            line = line.decode().strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    
    def list_models(self) -> List[ModelManifest]:
        """
        列出本地所有模型
        
        Returns:
            模型清单列表
        """
        response = self._api_request("tags")
        models = []
        
        for model_data in response.get("models", []):
            name_tag = model_data.get("name", "")
            name, tag = self._parse_name_tag(name_tag)
            
            manifest = ModelManifest(
                name=name,
                tag=tag,
                size_bytes=model_data.get("size", 0),
                digest=model_data.get("digest", ""),
                format=model_data.get("details", {}).get("format", "gguf"),
                architecture=model_data.get("details", {}).get("family", ""),
                parameters=model_data.get("details", {}).get("parameter_size", ""),
                quantization=model_data.get("details", {}).get("quantization_level", ""),
            )
            models.append(manifest)
        
        with self._lock:
            self._local_models = {m.full_name: m for m in models}
        
        return models
    
    def _parse_name_tag(self, name_tag: str) -> Tuple[str, str]:
        """解析名称和标签"""
        if ":" in name_tag:
            parts = name_tag.rsplit(":", 1)
            return parts[0], parts[1]
        return name_tag, "latest"
    
    def pull_model(
        self,
        model_name: str,
        insecure: bool = False,
        stream: bool = True,
    ) -> Iterator[PullProgress]:
        """
        拉取模型
        
        Args:
            model_name: 模型名称
            insecure: 允许不安全的连接
            stream: 是否流式返回进度
            
        Yields:
            拉取进度
        """
        data = {
            "name": model_name,
            "insecure": insecure,
            "stream": stream,
        }
        
        if stream:
            for chunk in self._api_request("pull", method="POST", data=data, stream=True):
                progress = self._parse_pull_progress(chunk)
                with self._lock:
                    self._pull_progress[model_name] = progress
                yield progress
        else:
            response = self._api_request("pull", method="POST", data=data)
            yield self._parse_pull_progress(response)
        
        # 更新本地模型列表
        self.list_models()
    
    def _parse_pull_progress(self, data: Dict[str, Any]) -> PullProgress:
        """解析拉取进度"""
        status = data.get("status", "")
        completed = data.get("completed", 0)
        total = data.get("total", 0)
        
        percent = (completed / total * 100) if total > 0 else 0
        
        return PullProgress(
            status=status,
            completed=completed,
            total=total,
            percent=percent,
            layer_id=data.get("digest", ""),
        )
    
    def push_model(
        self,
        model_name: str,
        insecure: bool = False,
    ) -> Iterator[PullProgress]:
        """
        推送模型到注册表
        
        Args:
            model_name: 模型名称
            insecure: 允许不安全的连接
            
        Yields:
            推送进度
        """
        data = {
            "name": model_name,
            "insecure": insecure,
            "stream": True,
        }
        
        for chunk in self._api_request("push", method="POST", data=data, stream=True):
            yield self._parse_pull_progress(chunk)
    
    def delete_model(self, model_name: str) -> bool:
        """
        删除本地模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            是否成功删除
        """
        data = {"name": model_name}
        
        try:
            self._api_request("delete", method="DELETE", data=data)
            
            with self._lock:
                if model_name in self._local_models:
                    del self._local_models[model_name]
            
            return True
        except OllamaError:
            return False
    
    def copy_model(self, source: str, destination: str) -> bool:
        """
        复制模型
        
        Args:
            source: 源模型名称
            destination: 目标模型名称
            
        Returns:
            是否成功复制
        """
        data = {
            "source": source,
            "destination": destination,
        }
        
        try:
            self._api_request("copy", method="POST", data=data)
            return True
        except OllamaError:
            return False
    
    def show_model(self, model_name: str) -> ModelManifest:
        """
        显示模型详细信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型清单
        """
        data = {"name": model_name}
        response = self._api_request("show", method="POST", data=data)
        
        name, tag = self._parse_name_tag(model_name)
        
        return ModelManifest(
            name=name,
            tag=tag,
            architecture=response.get("details", {}).get("family", ""),
            parameters=response.get("details", {}).get("parameter_size", ""),
            quantization=response.get("details", {}).get("quantization_level", ""),
            format=response.get("details", {}).get("format", "gguf"),
            template=response.get("template", ""),
            system_prompt=response.get("system", ""),
            license=response.get("license", ""),
            layers=response.get("layers", []),
        )
    
    def get_model_status(self, model_name: str) -> ModelStatus:
        """
        获取模型状态
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型状态
        """
        with self._lock:
            if model_name in self._local_models:
                return ModelStatus.READY
            
            if model_name in self._pull_progress:
                progress = self._pull_progress[model_name]
                if progress.status == "downloading":
                    return ModelStatus.DOWNLOADING
                elif progress.status == "verifying":
                    return ModelStatus.VERIFYING
                elif progress.status == "error":
                    return ModelStatus.ERROR
        
        return ModelStatus.PENDING
    
    def is_model_available(self, model_name: str) -> bool:
        """检查模型是否可用"""
        return self.get_model_status(model_name) == ModelStatus.READY


class OllamaEmbedder:
    """
    Ollama嵌入生成器
    
    使用Ollama模型生成文本嵌入向量。
    """
    
    def __init__(self, model_manager: OllamaModelManager):
        self.model_manager = model_manager
        self._cache: Dict[str, EmbeddingResult] = {}
        self._cache_lock = threading.RLock()
        self._max_cache_size = 1000
    
    def embed(
        self,
        texts: Union[str, List[str]],
        model: str = "nomic-embed-text",
        truncate: bool = True,
        options: Optional[Dict[str, Any]] = None,
    ) -> Union[EmbeddingResult, List[EmbeddingResult]]:
        """
        生成文本嵌入
        
        Args:
            texts: 文本或文本列表
            model: 嵌入模型名称
            truncate: 是否截断长文本
            options: 额外选项
            
        Returns:
            嵌入结果或结果列表
        """
        single_input = isinstance(texts, str)
        text_list = [texts] if single_input else texts
        
        results = []
        for text in text_list:
            result = self._embed_single(text, model, truncate, options)
            results.append(result)
        
        return results[0] if single_input else results
    
    def _embed_single(
        self,
        text: str,
        model: str,
        truncate: bool,
        options: Optional[Dict[str, Any]],
    ) -> EmbeddingResult:
        """嵌入单个文本"""
        # 检查缓存
        cache_key = f"{model}:{hash(text)}"
        with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        start_time = time.time()
        
        data = {
            "model": model,
            "input": text,
            "truncate": truncate,
        }
        if options:
            data["options"] = options
        
        response = self.model_manager._api_request("embed", method="POST", data=data)
        
        embedding = response.get("embeddings", [[]])[0]
        tokens = response.get("prompt_eval_count", 0)
        duration = (time.time() - start_time) * 1000
        
        result = EmbeddingResult(
            text=text,
            embedding=embedding,
            model=model,
            tokens=tokens,
            duration_ms=duration,
        )
        
        # 更新缓存
        with self._cache_lock:
            if len(self._cache) >= self._max_cache_size:
                # LRU淘汰
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[cache_key] = result
        
        return result
    
    def batch_embed(
        self,
        texts: List[str],
        model: str = "nomic-embed-text",
        batch_size: int = 32,
        **kwargs,
    ) -> List[EmbeddingResult]:
        """
        批量嵌入
        
        Args:
            texts: 文本列表
            model: 模型名称
            batch_size: 批大小
            
        Returns:
            嵌入结果列表
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self.embed(batch, model=model, **kwargs)
            if isinstance(batch_results, list):
                results.extend(batch_results)
            else:
                results.append(batch_results)
        return results
    
    def clear_cache(self) -> int:
        """清除缓存，返回清除的条目数"""
        with self._cache_lock:
            count = len(self._cache)
            self._cache.clear()
            return count


class OllamaChatClient:
    """
    Ollama聊天客户端
    
    提供与Ollama模型的聊天完成功能。
    """
    
    def __init__(self, model_manager: OllamaModelManager):
        self.model_manager = model_manager
    
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        聊天完成
        
        Args:
            model: 模型名称
            messages: 消息列表
            stream: 是否流式输出
            format: 输出格式 (json)
            options: 模型选项
            keep_alive: 模型保持加载时间
            tools: 工具定义
            
        Returns:
            响应或流式迭代器
        """
        data: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        
        if format:
            data["format"] = format
        if options:
            data["options"] = options
        if keep_alive:
            data["keep_alive"] = keep_alive
        if tools:
            data["tools"] = tools
        
        return self.model_manager._api_request("chat", method="POST", data=data, stream=stream)
    
    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        template: Optional[str] = None,
        context: Optional[List[int]] = None,
        stream: bool = False,
        raw: bool = False,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        文本生成
        
        Args:
            model: 模型名称
            prompt: 提示文本
            system: 系统提示
            template: 自定义模板
            context: 上下文
            stream: 是否流式
            raw: 是否使用原始模式
            format: 输出格式
            options: 模型选项
            keep_alive: 保持时间
            
        Returns:
            响应或流式迭代器
        """
        data: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "raw": raw,
        }
        
        if system:
            data["system"] = system
        if template:
            data["template"] = template
        if context:
            data["context"] = context
        if format:
            data["format"] = format
        if options:
            data["options"] = options
        if keep_alive:
            data["keep_alive"] = keep_alive
        
        return self.model_manager._api_request("generate", method="POST", data=data, stream=stream)


class OllamaStreamHandler:
    """
    Ollama流式处理器
    
    处理Ollama的流式响应，提供缓冲、解析和回调功能。
    """
    
    def __init__(self, buffer_size: int = 1024):
        self.buffer_size = buffer_size
        self._buffer = ""
        self._lock = threading.Lock()
        self._callbacks: List[callable] = []
        self._stop_event = threading.Event()
    
    def add_callback(self, callback: callable) -> None:
        """添加流式回调"""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: callable) -> None:
        """移除回调"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def process_stream(self, stream: Iterator[Dict[str, Any]]) -> Iterator[str]:
        """
        处理流式响应
        
        Args:
            stream: 流式响应迭代器
            
        Yields:
            文本块
        """
        for chunk in stream:
            if self._stop_event.is_set():
                break
            
            # 解析内容
            content = self._extract_content(chunk)
            if content:
                with self._lock:
                    self._buffer += content
                
                # 触发回调
                for callback in self._callbacks:
                    try:
                        callback(content, chunk)
                    except Exception:
                        pass
                
                yield content
            
            # 检查是否完成
            if chunk.get("done", False):
                break
    
    def _extract_content(self, chunk: Dict[str, Any]) -> str:
        """从chunk中提取内容"""
        # 聊天模式
        if "message" in chunk:
            return chunk["message"].get("content", "")
        
        # 生成模式
        return chunk.get("response", "")
    
    def get_buffer(self) -> str:
        """获取当前缓冲区内容"""
        with self._lock:
            return self._buffer
    
    def clear_buffer(self) -> None:
        """清空缓冲区"""
        with self._lock:
            self._buffer = ""
    
    def stop(self) -> None:
        """停止流式处理"""
        self._stop_event.set()
    
    def reset(self) -> None:
        """重置状态"""
        self._stop_event.clear()
        self.clear_buffer()


class OllamaAdapter(LLMBackend):
    """
    Ollama适配器
    
    提供与Ollama本地LLM服务的完整集成，实现LLMBackend接口。
    """
    
    def __init__(
        self,
        model_name: str = "llama2",
        base_url: str = "http://localhost:11434",
        system_prompt: Optional[str] = None,
        keep_alive: str = "5m",
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.keep_alive = keep_alive
        
        # 初始化组件
        self.model_manager = OllamaModelManager(base_url)
        self.embedder = OllamaEmbedder(self.model_manager)
        self.chat_client = OllamaChatClient(self.model_manager)
        self.stream_handler = OllamaStreamHandler()
        
        # 模型信息
        self._model_info: Optional[ModelInfo] = None
        self._options: Dict[str, Any] = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
        }
    
    def _messages_to_ollama_format(self, messages: List[Message]) -> List[Dict[str, str]]:
        """转换消息格式"""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                **({"name": msg.name} if msg.name else {}),
            }
            for msg in messages
        ]
    
    def generate(
        self,
        messages: List[Message],
        params: Optional[GenerateParams] = None,
    ) -> LLMResponse:
        """
        生成回复
        
        Args:
            messages: 消息列表
            params: 生成参数
            
        Returns:
            LLM响应
        """
        if params is None:
            params = GenerateParams()
        
        # 构建选项
        options = self._build_options(params)
        
        # 转换消息
        ollama_messages = self._messages_to_ollama_format(messages)
        
        # 发送请求
        response = self.chat_client.chat(
            model=self.model_name,
            messages=ollama_messages,
            stream=False,
            options=options,
            keep_alive=self.keep_alive,
        )
        
        # 解析响应
        message = response.get("message", {})
        content = message.get("content", "")
        
        # 构建使用统计
        prompt_tokens = response.get("prompt_eval_count", 0)
        completion_tokens = response.get("eval_count", 0)
        
        return LLMResponse(
            content=content,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
            finish_reason=FinishReason.STOP,
            model=self.model_name,
        )
    
    def stream(
        self,
        messages: List[Message],
        params: Optional[GenerateParams] = None,
    ) -> Iterator[LLMChunk]:
        """
        流式生成
        
        Args:
            messages: 消息列表
            params: 生成参数
            
        Yields:
            流式响应块
        """
        if params is None:
            params = GenerateParams()
        
        options = self._build_options(params)
        ollama_messages = self._messages_to_ollama_format(messages)
        
        # 发送流式请求
        stream = self.chat_client.chat(
            model=self.model_name,
            messages=ollama_messages,
            stream=True,
            options=options,
            keep_alive=self.keep_alive,
        )
        
        # 处理流式响应
        for chunk in self.stream_handler.process_stream(stream):
            yield LLMChunk(delta_content=chunk)
        
        # 发送完成标记
        yield LLMChunk(delta_content="", finish_reason=FinishReason.STOP)
    
    def _build_options(self, params: GenerateParams) -> Dict[str, Any]:
        """构建Ollama选项"""
        options = self._options.copy()
        
        if params.temperature is not None:
            options["temperature"] = params.temperature
        if params.top_p is not None:
            options["top_p"] = params.top_p
        if params.top_k > 0:
            options["top_k"] = params.top_k
        if params.seed is not None:
            options["seed"] = params.seed
        if params.max_tokens is not None:
            options["num_predict"] = params.max_tokens
        if params.stop:
            options["stop"] = params.stop
        if params.frequency_penalty != 0.0:
            options["repeat_penalty"] = 1.0 + params.frequency_penalty
        
        return options
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        文本嵌入
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        results = self.embedder.batch_embed(texts)
        return [r.embedding for r in results]
    
    def count_tokens(self, text: str) -> int:
        """
        统计token数
        
        Args:
            text: 文本
            
        Returns:
            Token数量
        """
        # 使用嵌入API获取token数
        try:
            result = self.embedder.embed(text, model=self.model_name)
            return result.tokens
        except Exception:
            # 估算：平均每个token约4个字符
            return len(text) // 4
    
    def get_model_info(self) -> ModelInfo:
        """
        获取模型信息
        
        Returns:
            模型信息
        """
        if self._model_info is None:
            try:
                manifest = self.model_manager.show_model(self.model_name)
                
                # 解析参数规模
                max_context = 4096
                param_str = manifest.parameters.lower()
                if "7b" in param_str or "8b" in param_str:
                    max_context = 4096
                elif "13b" in param_str or "14b" in param_str:
                    max_context = 8192
                elif "70b" in param_str or "72b" in param_str:
                    max_context = 16384
                
                self._model_info = ModelInfo(
                    name=self.model_name,
                    max_context=max_context,
                    max_output=2048,
                    supports_streaming=True,
                    supports_functions=False,
                    vendor="ollama",
                    description=f"{manifest.architecture} {manifest.parameters}",
                )
            except Exception:
                self._model_info = ModelInfo(
                    name=self.model_name,
                    vendor="ollama",
                )
        
        return self._model_info
    
    def pull_model(self, model_name: Optional[str] = None) -> Iterator[PullProgress]:
        """
        拉取模型
        
        Args:
            model_name: 模型名称，默认为当前模型
            
        Yields:
            拉取进度
        """
        name = model_name or self.model_name
        yield from self.model_manager.pull_model(name)
    
    def is_model_ready(self) -> bool:
        """检查模型是否就绪"""
        return self.model_manager.is_model_available(self.model_name)
    
    def set_options(self, **options) -> None:
        """设置模型选项"""
        self._options.update(options)
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            self.model_manager.list_models()
            return True
        except Exception:
            return False
