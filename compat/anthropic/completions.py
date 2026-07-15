"""
Anthropic Completions API 实现模块

提供文本补全功能支持，包括：
- 同步和异步补全请求
- 流式响应处理
- 自定义生成参数
- 错误处理和重试

注意: Anthropic推荐使用Messages API代替Completions API
此模块提供向后兼容性支持

参考: https://docs.anthropic.com/claude/reference/complete_post
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional, Union

from .exceptions import (
    APIError,
    AuthenticationError,
    InvalidRequestError,
    RateLimitError,
    ServerError,
    StreamError,
)


# 模型常量
class Model:
    """Anthropic模型常量"""
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    CLAUDE_2_1 = "claude-2.1"
    CLAUDE_2_0 = "claude-2.0"
    CLAUDE_INSTANT_1_2 = "claude-instant-1.2"


@dataclass
class CompletionRequest:
    """
    补全请求参数
    
    封装所有补全API请求参数
    
    示例:
        >>> request = CompletionRequest(
        ...     prompt="Human: Hello!\\n\\nAssistant:",
        ...     model=Model.CLAUDE_3_SONNET,
        ...     max_tokens_to_sample=1024,
        ...     temperature=0.7,
        ... )
    """
    prompt: str
    model: str = Model.CLAUDE_3_SONNET
    max_tokens_to_sample: int = 256
    stop_sequences: Optional[List[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    stream: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为API请求体格式
        
        Returns:
            API请求参数字典
        """
        body: Dict[str, Any] = {
            "prompt": self.prompt,
            "model": self.model,
            "max_tokens_to_sample": self.max_tokens_to_sample,
        }
        
        if self.stop_sequences is not None:
            body["stop_sequences"] = self.stop_sequences
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.top_p is not None:
            body["top_p"] = self.top_p
        if self.top_k is not None:
            body["top_k"] = self.top_k
        if self.metadata is not None:
            body["metadata"] = self.metadata
        if self.stream:
            body["stream"] = True
            
        return body
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CompletionRequest:
        """
        从字典创建请求对象
        
        Args:
            data: 参数字典
        
        Returns:
            CompletionRequest实例
        """
        return cls(
            prompt=data.get("prompt", ""),
            model=data.get("model", Model.CLAUDE_3_SONNET),
            max_tokens_to_sample=data.get("max_tokens_to_sample", 256),
            stop_sequences=data.get("stop_sequences"),
            temperature=data.get("temperature"),
            top_p=data.get("top_p"),
            top_k=data.get("top_k"),
            metadata=data.get("metadata"),
            stream=data.get("stream", False),
        )


@dataclass
class CompletionResponse:
    """
    补全响应对象
    
    包含API返回的完整补全结果
    
    示例:
        >>> response = CompletionResponse(
        ...     completion="Hello! How can I help you today?",
        ...     stop_reason="stop_sequence",
        ...     model=Model.CLAUDE_3_SONNET,
        ... )
    """
    completion: str
    stop_reason: str
    model: str
    stop: Optional[str] = None
    log_id: Optional[str] = None
    
    @property
    def text(self) -> str:
        """获取补全文本内容"""
        return self.completion
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            响应字典
        """
        result: Dict[str, Any] = {
            "completion": self.completion,
            "stop_reason": self.stop_reason,
            "model": self.model,
        }
        if self.stop is not None:
            result["stop"] = self.stop
        if self.log_id is not None:
            result["log_id"] = self.log_id
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CompletionResponse:
        """
        从API响应创建对象
        
        Args:
            data: API响应字典
        
        Returns:
            CompletionResponse实例
        """
        return cls(
            completion=data.get("completion", ""),
            stop_reason=data.get("stop_reason", ""),
            model=data.get("model", ""),
            stop=data.get("stop"),
            log_id=data.get("log_id"),
        )


@dataclass
class CompletionStreamEvent:
    """
    补全流式事件
    
    用于处理流式响应中的单个事件
    """
    completion: str
    stop_reason: Optional[str] = None
    model: Optional[str] = None
    stop: Optional[str] = None
    log_id: Optional[str] = None
    
    @property
    def text(self) -> str:
        """获取事件文本内容"""
        return self.completion
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CompletionStreamEvent:
        """
        从API响应创建事件对象
        
        Args:
            data: API响应数据
        
        Returns:
            CompletionStreamEvent实例
        """
        return cls(
            completion=data.get("completion", ""),
            stop_reason=data.get("stop_reason"),
            model=data.get("model"),
            stop=data.get("stop"),
            log_id=data.get("log_id"),
        )


class Completions:
    """
    同步Completions API客户端
    
    提供完整的同步文本补全功能
    
    示例:
        >>> completions = Completions(client)
        >>> response = completions.create(
        ...     prompt="Human: Hello!\\n\\nAssistant:",
        ...     model="claude-3-sonnet-20240229",
        ...     max_tokens_to_sample=1024,
        ... )
    """
    
    def __init__(self, client: Any) -> None:
        """
        初始化Completions客户端
        
        Args:
            client: 父级HTTP客户端实例
        """
        self._client = client
    
    def create(
        self,
        prompt: str,
        model: str = Model.CLAUDE_3_SONNET,
        max_tokens_to_sample: int = 256,
        stop_sequences: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Union[CompletionResponse, Iterator[CompletionStreamEvent]]:
        """
        创建补全请求
        
        Args:
            prompt: 提示文本，必须包含Human/Assistant格式
            model: 模型ID
            max_tokens_to_sample: 最大生成token数
            stop_sequences: 停止序列列表
            temperature: 采样温度 (0-1)
            top_p: nucleus采样参数
            top_k: top-k采样参数
            metadata: 请求元数据
            stream: 是否使用流式响应
        
        Returns:
            CompletionResponse对象或流式事件迭代器
        """
        request = CompletionRequest(
            prompt=prompt,
            model=model,
            max_tokens_to_sample=max_tokens_to_sample,
            stop_sequences=stop_sequences,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            metadata=metadata,
            stream=stream,
        )
        
        if stream:
            return self._stream_create(request)
        
        response = self._client.post(
            "/v1/complete",
            json=request.to_dict(),
        )
        return CompletionResponse.from_dict(response)
    
    def _stream_create(
        self,
        request: CompletionRequest,
    ) -> Iterator[CompletionStreamEvent]:
        """
        处理流式补全响应
        
        Args:
            request: 补全请求对象
        
        Yields:
            CompletionStreamEvent对象
        """
        response = self._client.post(
            "/v1/complete",
            json=request.to_dict(),
            stream=True,
        )
        
        for line in response.iter_lines():
            if not line:
                continue
            
            line_str = line.decode("utf-8") if isinstance(line, bytes) else line
            
            if line_str.startswith("data: "):
                data_str = line_str[6:]
                
                if data_str == "[DONE]":
                    break
                
                try:
                    data = json.loads(data_str)
                    yield CompletionStreamEvent.from_dict(data)
                except json.JSONDecodeError as e:
                    raise StreamError(
                        f"Failed to parse stream chunk: {e}",
                        chunk=data_str,
                    )
    
    def create_simple(
        self,
        prompt: str,
        max_tokens: int = 256,
        **kwargs,
    ) -> str:
        """
        简化版补全请求，直接返回文本
        
        Args:
            prompt: 提示文本
            max_tokens: 最大生成token数
            **kwargs: 其他参数
        
        Returns:
            补全文本
        """
        response = self.create(
            prompt=prompt,
            max_tokens_to_sample=max_tokens,
            **kwargs,
        )
        
        if isinstance(response, CompletionResponse):
            return response.completion
        
        # 流式响应，合并所有片段
        chunks = []
        for event in response:
            chunks.append(event.completion)
        return "".join(chunks)


class AsyncCompletions:
    """
    异步Completions API客户端
    
    提供完整的异步文本补全功能
    
    示例:
        >>> completions = AsyncCompletions(client)
        >>> response = await completions.create(
        ...     prompt="Human: Hello!\\n\\nAssistant:",
        ...     model="claude-3-sonnet-20240229",
        ...     max_tokens_to_sample=1024,
        ... )
    """
    
    def __init__(self, client: Any) -> None:
        """
        初始化异步Completions客户端
        
        Args:
            client: 父级异步HTTP客户端实例
        """
        self._client = client
    
    async def create(
        self,
        prompt: str,
        model: str = Model.CLAUDE_3_SONNET,
        max_tokens_to_sample: int = 256,
        stop_sequences: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Union[CompletionResponse, AsyncIterator[CompletionStreamEvent]]:
        """
        异步创建补全请求
        
        Args:
            prompt: 提示文本
            model: 模型ID
            max_tokens_to_sample: 最大生成token数
            stop_sequences: 停止序列列表
            temperature: 采样温度
            top_p: nucleus采样参数
            top_k: top-k采样参数
            metadata: 请求元数据
            stream: 是否使用流式响应
        
        Returns:
            CompletionResponse对象或异步流式事件迭代器
        """
        request = CompletionRequest(
            prompt=prompt,
            model=model,
            max_tokens_to_sample=max_tokens_to_sample,
            stop_sequences=stop_sequences,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            metadata=metadata,
            stream=stream,
        )
        
        if stream:
            return self._stream_create(request)
        
        response = await self._client.post(
            "/v1/complete",
            json=request.to_dict(),
        )
        return CompletionResponse.from_dict(response)
    
    async def _stream_create(
        self,
        request: CompletionRequest,
    ) -> AsyncIterator[CompletionStreamEvent]:
        """
        处理异步流式补全响应
        
        Args:
            request: 补全请求对象
        
        Yields:
            CompletionStreamEvent对象
        """
        response = await self._client.post(
            "/v1/complete",
            json=request.to_dict(),
            stream=True,
        )
        
        async for line in response.aiter_lines():
            if not line:
                continue
            
            line_str = line.decode("utf-8") if isinstance(line, bytes) else line
            
            if line_str.startswith("data: "):
                data_str = line_str[6:]
                
                if data_str == "[DONE]":
                    break
                
                try:
                    data = json.loads(data_str)
                    yield CompletionStreamEvent.from_dict(data)
                except json.JSONDecodeError as e:
                    raise StreamError(
                        f"Failed to parse stream chunk: {e}",
                        chunk=data_str,
                    )
    
    async def create_simple(
        self,
        prompt: str,
        max_tokens: int = 256,
        **kwargs,
    ) -> str:
        """
        简化版异步补全请求，直接返回文本
        
        Args:
            prompt: 提示文本
            max_tokens: 最大生成token数
            **kwargs: 其他参数
        
        Returns:
            补全文本
        """
        response = await self.create(
            prompt=prompt,
            max_tokens_to_sample=max_tokens,
            **kwargs,
        )
        
        if isinstance(response, CompletionResponse):
            return response.completion
        
        # 流式响应，合并所有片段
        chunks = []
        async for event in response:
            chunks.append(event.completion)
        return "".join(chunks)


# 便捷函数
def format_prompt(
    user_message: str,
    assistant_prefix: str = "",
    system_message: Optional[str] = None,
) -> str:
    """
    格式化提示文本为Anthropic格式
    
    Anthropic Completions API需要特定的Human/Assistant格式
    
    Args:
        user_message: 用户消息
        assistant_prefix: 助手回复前缀
        system_message: 可选的系统消息
    
    Returns:
        格式化后的提示文本
    
    示例:
        >>> prompt = format_prompt("Hello!", "Hi there")
        >>> print(prompt)
        \\n\\nHuman: Hello!\\n\\nAssistant: Hi there
    """
    prompt_parts = []
    
    if system_message:
        prompt_parts.append(system_message)
    
    prompt_parts.extend([
        "",
        f"Human: {user_message}",
        "",
        f"Assistant: {assistant_prefix}",
    ])
    
    return "\n".join(prompt_parts)


def create_completion_prompt(
    messages: List[Dict[str, str]],
) -> str:
    """
    从消息列表创建补全提示
    
    将消息列表转换为Anthropic Completions API格式
    
    Args:
        messages: 消息列表，每个消息包含role和content
    
    Returns:
        格式化后的提示文本
    
    示例:
        >>> messages = [
        ...     {"role": "user", "content": "Hello!"},
        ...     {"role": "assistant", "content": "Hi!"},
        ... ]
        >>> prompt = create_completion_prompt(messages)
    """
    prompt_parts = []
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "user" or role == "human":
            prompt_parts.extend(["", f"Human: {content}"])
        elif role == "assistant":
            prompt_parts.extend(["", f"Assistant: {content}"])
        elif role == "system":
            prompt_parts.append(content)
    
    # 添加最终的Assistant前缀
    prompt_parts.extend(["", "Assistant:"])
    
    return "\n".join(prompt_parts)


def count_tokens_approximate(text: str) -> int:
    """
    估算文本的token数量
    
    这是一个简化的估算，实际token数可能有所不同
    
    Args:
        text: 输入文本
    
    Returns:
        估算的token数量
    """
    # 简化的估算：英文约4字符/token，中文约1.5字符/token
    # 实际应使用tiktoken或Anthropic的token计数API
    import re
    
    # 中文字符
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    # 其他字符
    other_chars = len(text) - chinese_chars
    
    # 估算token数
    tokens = chinese_chars * 1.5 + other_chars / 4
    
    return int(tokens)


def truncate_prompt(
    prompt: str,
    max_tokens: int,
    reserve_tokens: int = 256,
) -> str:
    """
    截断提示文本以适应token限制
    
    Args:
        prompt: 原始提示文本
        max_tokens: 最大token限制
        reserve_tokens: 为响应保留的token数
    
    Returns:
        截断后的提示文本
    """
    available_tokens = max_tokens - reserve_tokens
    current_tokens = count_tokens_approximate(prompt)
    
    if current_tokens <= available_tokens:
        return prompt
    
    # 简单截断：保留开头和结尾，中间用...代替
    # 更复杂的实现可以考虑语义截断
    chars_to_keep = int(available_tokens * 3)  # 粗略估算
    
    if len(prompt) <= chars_to_keep:
        return prompt
    
    half_keep = chars_to_keep // 2
    return prompt[:half_keep] + "\n... [内容截断] ...\n" + prompt[-half_keep:]
