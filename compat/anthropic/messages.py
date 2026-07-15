"""
Anthropic Messages API 实现模块

提供完整的Messages API支持，包括：
- 同步和异步消息发送
- 流式响应处理
- 工具使用集成
- 多模态内容(文本+图像)

参考: https://docs.anthropic.com/claude/reference/messages_post
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Union,
)

from .exceptions import (
    APIError,
    AuthenticationError,
    InvalidRequestError,
    RateLimitError,
    ServerError,
    StreamError,
)


# 类型定义
Role = Literal["user", "assistant"]
ContentType = Literal["text", "image", "tool_use", "tool_result"]


@dataclass
class ContentBlock:
    """
    消息内容块
    
    支持多种内容类型：文本、图像、工具使用等
    """
    type: ContentType
    text: Optional[str] = None
    source: Optional[Dict[str, Any]] = None
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    tool_use_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为API格式字典"""
        result: Dict[str, Any] = {"type": self.type}
        
        if self.text is not None:
            result["text"] = self.text
        if self.source is not None:
            result["source"] = self.source
        if self.id is not None:
            result["id"] = self.id
        if self.name is not None:
            result["name"] = self.name
        if self.input is not None:
            result["input"] = self.input
        if self.content is not None:
            result["content"] = self.content
        if self.tool_use_id is not None:
            result["tool_use_id"] = self.tool_use_id
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ContentBlock:
        """从API响应字典创建ContentBlock"""
        return cls(
            type=data.get("type", "text"),
            text=data.get("text"),
            source=data.get("source"),
            id=data.get("id"),
            name=data.get("name"),
            input=data.get("input"),
            content=data.get("content"),
            tool_use_id=data.get("tool_use_id"),
        )


@dataclass
class MessageParam:
    """
    消息参数
    
    用于构建对话历史的消息格式
    """
    role: Role
    content: Union[str, List[ContentBlock], List[Dict[str, Any]]]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为API格式字典"""
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        elif isinstance(self.content, list) and len(self.content) > 0:
            if isinstance(self.content[0], ContentBlock):
                return {
                    "role": self.role,
                    "content": [block.to_dict() for block in self.content]
                }
            else:
                return {"role": self.role, "content": self.content}
        return {"role": self.role, "content": self.content}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MessageParam:
        """从字典创建MessageParam"""
        content = data.get("content", "")
        if isinstance(content, list):
            content = [ContentBlock.from_dict(c) if isinstance(c, dict) else c 
                      for c in content]
        return cls(role=data.get("role", "user"), content=content)


@dataclass
class Usage:
    """
    Token使用量统计
    """
    input_tokens: int = 0
    output_tokens: int = 0
    
    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> Usage:
        """从API响应创建Usage对象"""
        return cls(
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
        )


@dataclass
class Message:
    """
    API响应消息对象
    
    包含完整的响应内容、元数据和使用统计
    """
    id: str
    type: str
    role: Role
    content: List[ContentBlock]
    model: str
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: Usage = field(default_factory=Usage)
    
    @property
    def text(self) -> str:
        """获取响应中的文本内容"""
        texts = [
            block.text for block in self.content 
            if block.type == "text" and block.text
        ]
        return "".join(texts)
    
    @property
    def tool_calls(self) -> List[ContentBlock]:
        """获取响应中的工具调用块"""
        return [block for block in self.content if block.type == "tool_use"]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "type": self.type,
            "role": self.role,
            "content": [block.to_dict() for block in self.content],
            "model": self.model,
            "stop_reason": self.stop_reason,
            "stop_sequence": self.stop_sequence,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Message:
        """从API响应字典创建Message对象"""
        content_blocks = [
            ContentBlock.from_dict(block) if isinstance(block, dict) else block
            for block in data.get("content", [])
        ]
        
        return cls(
            id=data.get("id", ""),
            type=data.get("type", "message"),
            role=data.get("role", "assistant"),
            content=content_blocks,
            model=data.get("model", ""),
            stop_reason=data.get("stop_reason"),
            stop_sequence=data.get("stop_sequence"),
            usage=Usage.from_dict(data.get("usage", {})),
        )


class Messages:
    """
    同步Messages API客户端
    
    提供完整的同步消息发送功能
    """
    
    def __init__(self, client: Any) -> None:
        """
        初始化Messages客户端
        
        Args:
            client: 父级HTTP客户端实例
        """
        self._client = client
    
    def create(
        self,
        model: str,
        max_tokens: int,
        messages: List[Union[MessageParam, Dict[str, Any]]],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[Message, Iterator[MessageStreamEvent]]:
        """
        发送消息请求
        
        Args:
            model: 模型ID (如 "claude-3-sonnet-20240229")
            max_tokens: 最大生成token数
            messages: 消息历史列表
            system: 系统提示词
            temperature: 采样温度 (0-1)
            top_p: nucleus采样参数
            top_k: top-k采样参数
            stop_sequences: 停止序列列表
            stream: 是否使用流式响应
            tools: 可用工具定义列表
            tool_choice: 工具选择配置
            metadata: 请求元数据
        
        Returns:
            Message对象或流式事件迭代器
        """
        # 构建请求体
        body = self._build_request_body(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            system=system,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
            metadata=metadata,
        )
        
        if stream:
            return self._stream_create(body)
        
        response = self._client.post("/v1/messages", json=body)
        return Message.from_dict(response)
    
    def _build_request_body(
        self,
        model: str,
        max_tokens: int,
        messages: List[Union[MessageParam, Dict[str, Any]]],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """构建请求体"""
        body: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                msg.to_dict() if isinstance(msg, MessageParam) else msg
                for msg in messages
            ],
        }
        
        if system is not None:
            body["system"] = system
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if top_k is not None:
            body["top_k"] = top_k
        if stop_sequences is not None:
            body["stop_sequences"] = stop_sequences
        if stream:
            body["stream"] = True
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if metadata is not None:
            body["metadata"] = metadata
            
        return body
    
    def _stream_create(
        self, 
        body: Dict[str, Any]
    ) -> Iterator[MessageStreamEvent]:
        """
        处理流式响应
        
        Args:
            body: 请求体
        
        Yields:
            MessageStreamEvent对象
        """
        response = self._client.post(
            "/v1/messages", 
            json=body, 
            stream=True
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
                    yield MessageStreamEvent.from_dict(data)
                except json.JSONDecodeError as e:
                    raise StreamError(
                        f"Failed to parse stream chunk: {e}",
                        chunk=data_str
                    )


class AsyncMessages:
    """
    异步Messages API客户端
    
    提供完整的异步消息发送功能
    """
    
    def __init__(self, client: Any) -> None:
        """
        初始化异步Messages客户端
        
        Args:
            client: 父级异步HTTP客户端实例
        """
        self._client = client
    
    async def create(
        self,
        model: str,
        max_tokens: int,
        messages: List[Union[MessageParam, Dict[str, Any]]],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[Message, AsyncIterator[AsyncMessageStreamEvent]]:
        """
        异步发送消息请求
        
        Args:
            model: 模型ID
            max_tokens: 最大生成token数
            messages: 消息历史列表
            system: 系统提示词
            temperature: 采样温度
            top_p: nucleus采样参数
            top_k: top-k采样参数
            stop_sequences: 停止序列列表
            stream: 是否使用流式响应
            tools: 可用工具定义列表
            tool_choice: 工具选择配置
            metadata: 请求元数据
        
        Returns:
            Message对象或异步流式事件迭代器
        """
        body = self._build_request_body(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            system=system,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
            metadata=metadata,
        )
        
        if stream:
            return self._stream_create(body)
        
        response = await self._client.post("/v1/messages", json=body)
        return Message.from_dict(response)
    
    def _build_request_body(
        self,
        model: str,
        max_tokens: int,
        messages: List[Union[MessageParam, Dict[str, Any]]],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """构建请求体"""
        body: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                msg.to_dict() if isinstance(msg, MessageParam) else msg
                for msg in messages
            ],
        }
        
        if system is not None:
            body["system"] = system
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if top_k is not None:
            body["top_k"] = top_k
        if stop_sequences is not None:
            body["stop_sequences"] = stop_sequences
        if stream:
            body["stream"] = True
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if metadata is not None:
            body["metadata"] = metadata
            
        return body
    
    async def _stream_create(
        self, 
        body: Dict[str, Any]
    ) -> AsyncIterator[AsyncMessageStreamEvent]:
        """
        处理异步流式响应
        
        Args:
            body: 请求体
        
        Yields:
            AsyncMessageStreamEvent对象
        """
        response = await self._client.post(
            "/v1/messages", 
            json=body, 
            stream=True
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
                    yield AsyncMessageStreamEvent.from_dict(data)
                except json.JSONDecodeError as e:
                    raise StreamError(
                        f"Failed to parse stream chunk: {e}",
                        chunk=data_str
                    )


@dataclass
class MessageStreamEvent:
    """
    流式响应事件
    
    用于处理SSE流式响应中的单个事件
    """
    type: str
    index: Optional[int] = None
    delta: Optional[Dict[str, Any]] = None
    message: Optional[Message] = None
    usage: Optional[Usage] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MessageStreamEvent:
        """从API响应创建事件对象"""
        message = None
        if "message" in data:
            message = Message.from_dict(data["message"])
        
        usage = None
        if "usage" in data:
            usage = Usage.from_dict(data["usage"])
        
        return cls(
            type=data.get("type", ""),
            index=data.get("index"),
            delta=data.get("delta"),
            message=message,
            usage=usage,
        )


@dataclass
class AsyncMessageStreamEvent(MessageStreamEvent):
    """
    异步流式响应事件
    
    继承自MessageStreamEvent，用于异步上下文
    """
    pass
