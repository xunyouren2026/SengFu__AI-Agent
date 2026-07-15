"""流式响应支持模块。

本模块实现服务器发送事件(SSE)流式响应，支持实时数据推送。
用于MCP协议中的流式工具调用结果和进度通知。
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Optional, Callable, Dict, Any, List, Union, Iterator, Generator
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, Empty
from abc import ABC, abstractmethod


class SSEEventType(Enum):
    """SSE事件类型枚举。"""
    MESSAGE = "message"
    ERROR = "error"
    DONE = "done"
    PROGRESS = "progress"
    DATA = "data"
    CUSTOM = "custom"


@dataclass
class SSEEvent:
    """服务器发送事件。
    
    Attributes:
        event: 事件类型
        data: 事件数据
        id: 事件ID
        retry: 重试时间（毫秒）
    """
    event: str = "message"
    data: str = ""
    id: Optional[str] = None
    retry: Optional[int] = None
    
    def encode(self) -> str:
        """编码为SSE格式字符串。
        
        Returns:
            SSE格式字符串
        """
        lines = []
        
        if self.id is not None:
            lines.append(f"id: {self.id}")
        
        lines.append(f"event: {self.event}")
        
        # 数据可能包含多行
        for line in self.data.split("\n"):
            lines.append(f"data: {line}")
        
        if self.retry is not None:
            lines.append(f"retry: {self.retry}")
        
        lines.append("")
        lines.append("")
        
        return "\n".join(lines)
    
    @classmethod
    def message(cls, data: Any, event_id: Optional[str] = None) -> SSEEvent:
        """创建消息事件。
        
        Args:
            data: 数据内容
            event_id: 事件ID
            
        Returns:
            SSE事件
        """
        if not isinstance(data, str):
            data = json.dumps(data, ensure_ascii=False)
        
        return cls(event="message", data=data, id=event_id)
    
    @classmethod
    def error(cls, message: str, event_id: Optional[str] = None) -> SSEEvent:
        """创建错误事件。
        
        Args:
            message: 错误消息
            event_id: 事件ID
            
        Returns:
            SSE事件
        """
        return cls(
            event="error",
            data=json.dumps({"error": message}),
            id=event_id
        )
    
    @classmethod
    def done(cls, event_id: Optional[str] = None) -> SSEEvent:
        """创建完成事件。
        
        Args:
            event_id: 事件ID
            
        Returns:
            SSE事件
        """
        return cls(event="done", data="{}", id=event_id)
    
    @classmethod
    def progress(
        cls,
        current: float,
        total: float,
        message: Optional[str] = None,
        event_id: Optional[str] = None
    ) -> SSEEvent:
        """创建进度事件。
        
        Args:
            current: 当前进度
            total: 总进度
            message: 进度消息
            event_id: 事件ID
            
        Returns:
            SSE事件
        """
        data = {
            "current": current,
            "total": total,
            "percentage": (current / total * 100) if total > 0 else 0
        }
        if message:
            data["message"] = message
        
        return cls(
            event="progress",
            data=json.dumps(data),
            id=event_id
        )


class StreamProducer:
    """流生产者。
    
    生成SSE事件流，支持推送数据、进度和错误。
    """
    
    def __init__(
        self,
        buffer_size: int = 100,
        keep_alive_interval: float = 15.0
    ):
        """初始化流生产者。
        
        Args:
            buffer_size: 事件缓冲区大小
            keep_alive_interval: 保活间隔（秒）
        """
        self._queue: Queue[Optional[SSEEvent]] = Queue(maxsize=buffer_size)
        self._keep_alive_interval = keep_alive_interval
        self._closed = False
        self._event_count = 0
        self._lock = threading.Lock()
        
        # 保活线程
        self._keep_alive_thread: Optional[threading.Thread] = None
        self._last_event_time = time.time()
    
    def start(self) -> None:
        """启动流生产者。"""
        self._closed = False
        self._keep_alive_thread = threading.Thread(
            target=self._keep_alive_loop,
            daemon=True
        )
        self._keep_alive_thread.start()
    
    def _keep_alive_loop(self) -> None:
        """保活循环。"""
        while not self._closed:
            time.sleep(self._keep_alive_interval)
            
            if self._closed:
                break
            
            # 如果超过间隔时间没有发送事件，发送注释保活
            if time.time() - self._last_event_time >= self._keep_alive_interval:
                # 保活注释不会触发客户端事件
                pass
    
    def push(self, data: Any, event_type: str = "message") -> None:
        """推送数据。
        
        Args:
            data: 数据内容
            event_type: 事件类型
        """
        if self._closed:
            return
        
        with self._lock:
            self._event_count += 1
            event_id = str(self._event_count)
            self._last_event_time = time.time()
        
        event = SSEEvent(
            event=event_type,
            data=json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else data,
            id=event_id
        )
        
        try:
            self._queue.put(event, timeout=1.0)
        except Exception:
            pass
    
    def push_message(self, data: Any) -> None:
        """推送消息事件。
        
        Args:
            data: 数据内容
        """
        self.push(data, "message")
    
    def push_error(self, message: str) -> None:
        """推送错误事件。
        
        Args:
            message: 错误消息
        """
        if self._closed:
            return
        
        with self._lock:
            self._event_count += 1
            event_id = str(self._event_count)
        
        event = SSEEvent.error(message, event_id)
        
        try:
            self._queue.put(event, timeout=1.0)
        except Exception:
            pass
    
    def push_progress(
        self,
        current: float,
        total: float,
        message: Optional[str] = None
    ) -> None:
        """推送进度事件。
        
        Args:
            current: 当前进度
            total: 总进度
            message: 进度消息
        """
        if self._closed:
            return
        
        with self._lock:
            self._event_count += 1
            event_id = str(self._event_count)
            self._last_event_time = time.time()
        
        event = SSEEvent.progress(current, total, message, event_id)
        
        try:
            self._queue.put(event, timeout=1.0)
        except Exception:
            pass
    
    def push_done(self) -> None:
        """推送完成事件。"""
        if self._closed:
            return
        
        with self._lock:
            self._event_count += 1
            event_id = str(self._event_count)
        
        event = SSEEvent.done(event_id)
        
        try:
            self._queue.put(event, timeout=1.0)
        except Exception:
            pass
    
    def close(self) -> None:
        """关闭流。"""
        if self._closed:
            return
        
        self._closed = True
        
        # 发送结束标记
        try:
            self._queue.put(None, timeout=1.0)
        except Exception:
            pass
    
    def events(self) -> Generator[SSEEvent, None, None]:
        """生成事件迭代器。
        
        Yields:
            SSE事件
        """
        while not self._closed:
            try:
                event = self._queue.get(timeout=1.0)
                
                if event is None:
                    break
                
                yield event
            
            except Empty:
                continue
    
    def __iter__(self) -> Iterator[str]:
        """迭代SSE格式字符串。"""
        for event in self.events():
            yield event.encode()
    
    @property
    def is_closed(self) -> bool:
        """检查是否已关闭。"""
        return self._closed
    
    @property
    def event_count(self) -> int:
        """获取已发送事件数量。"""
        with self._lock:
            return self._event_count


class StreamConsumer:
    """流消费者。
    
    解析SSE事件流，提供事件处理接口。
    """
    
    def __init__(
        self,
        on_message: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[float, float, Optional[str]], None]] = None,
        on_done: Optional[Callable[[], None]] = None
    ):
        """初始化流消费者。
        
        Args:
            on_message: 消息处理函数
            on_error: 错误处理函数
            on_progress: 进度处理函数
            on_done: 完成处理函数
        """
        self._on_message = on_message
        self._on_error = on_error
        self._on_progress = on_progress
        self._on_done = on_done
        
        self._last_event_id: Optional[str] = None
        self._event_handlers: Dict[str, Callable[[Any], None]] = {}
    
    def on_event(self, event_type: str, handler: Callable[[Any], None]) -> None:
        """注册事件处理器。
        
        Args:
            event_type: 事件类型
            handler: 处理函数
        """
        self._event_handlers[event_type] = handler
    
    def parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """解析单行SSE数据。
        
        Args:
            line: SSE行
            
        Returns:
            解析结果
        """
        if not line:
            return None
        
        if line.startswith("id:"):
            return {"type": "id", "value": line[3:].strip()}
        
        if line.startswith("event:"):
            return {"type": "event", "value": line[6:].strip()}
        
        if line.startswith("data:"):
            return {"type": "data", "value": line[5:].strip()}
        
        if line.startswith("retry:"):
            return {"type": "retry", "value": int(line[6:].strip())}
        
        return None
    
    def process(self, chunk: str) -> None:
        """处理数据块。
        
        Args:
            chunk: SSE数据块
        """
        lines = chunk.split("\n")
        
        event_type = "message"
        data_lines = []
        event_id = None
        
        for line in lines:
            parsed = self.parse_line(line)
            
            if parsed is None:
                continue
            
            if parsed["type"] == "id":
                event_id = parsed["value"]
                self._last_event_id = event_id
            
            elif parsed["type"] == "event":
                event_type = parsed["value"]
            
            elif parsed["type"] == "data":
                data_lines.append(parsed["value"])
        
        # 处理完整事件
        if data_lines:
            data_str = "\n".join(data_lines)
            self._handle_event(event_type, data_str, event_id)
    
    def _handle_event(
        self,
        event_type: str,
        data: str,
        event_id: Optional[str]
    ) -> None:
        """处理事件。
        
        Args:
            event_type: 事件类型
            data: 事件数据
            event_id: 事件ID
        """
        # 尝试解析JSON
        try:
            data_obj = json.loads(data)
        except json.JSONDecodeError:
            data_obj = data
        
        # 调用特定处理器
        if event_type == "message" and self._on_message:
            self._on_message(data_obj)
        
        elif event_type == "error" and self._on_error:
            if isinstance(data_obj, dict):
                self._on_error(data_obj.get("error", str(data_obj)))
            else:
                self._on_error(str(data_obj))
        
        elif event_type == "progress" and self._on_progress:
            if isinstance(data_obj, dict):
                self._on_progress(
                    data_obj.get("current", 0),
                    data_obj.get("total", 100),
                    data_obj.get("message")
                )
        
        elif event_type == "done" and self._on_done:
            self._on_done()
        
        # 调用自定义处理器
        handler = self._event_handlers.get(event_type)
        if handler:
            handler(data_obj)
    
    @property
    def last_event_id(self) -> Optional[str]:
        """获取最后一个事件ID。"""
        return self._last_event_id


class StreamingResponse:
    """流式响应。
    
    封装流生产者，提供HTTP响应接口。
    """
    
    def __init__(
        self,
        producer: Optional[StreamProducer] = None,
        content_type: str = "text/event-stream"
    ):
        """初始化流式响应。
        
        Args:
            producer: 流生产者
            content_type: 内容类型
        """
        self.producer = producer or StreamProducer()
        self.content_type = content_type
        self.headers = {
            "Content-Type": content_type,
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    
    def start(self) -> None:
        """启动流。"""
        self.producer.start()
    
    def push(self, data: Any, event_type: str = "message") -> None:
        """推送数据。"""
        self.producer.push(data, event_type)
    
    def push_error(self, message: str) -> None:
        """推送错误。"""
        self.producer.push_error(message)
    
    def push_progress(
        self,
        current: float,
        total: float,
        message: Optional[str] = None
    ) -> None:
        """推送进度。"""
        self.producer.push_progress(current, total, message)
    
    def push_done(self) -> None:
        """推送完成。"""
        self.producer.push_done()
    
    def close(self) -> None:
        """关闭流。"""
        self.producer.close()
    
    def iter_content(self) -> Iterator[bytes]:
        """迭代响应内容。
        
        Yields:
            字节内容
        """
        for sse_str in self.producer:
            yield sse_str.encode("utf-8")
    
    def __iter__(self) -> Iterator[bytes]:
        """迭代响应内容。"""
        return self.iter_content()


class StreamingToolExecutor:
    """流式工具执行器。
    
    支持工具执行的流式输出。
    """
    
    def __init__(self) -> None:
        """初始化流式工具执行器。"""
        self._active_streams: Dict[str, StreamProducer] = {}
        self._lock = threading.Lock()
    
    def execute_streaming(
        self,
        tool_name: str,
        handler: Callable[[StreamProducer, Dict[str, Any]], None],
        arguments: Dict[str, Any]
    ) -> StreamingResponse:
        """执行流式工具。
        
        Args:
            tool_name: 工具名称
            handler: 处理函数，接收生产者和参数
            arguments: 工具参数
            
        Returns:
            流式响应
        """
        producer = StreamProducer()
        response = StreamingResponse(producer)
        
        stream_id = str(uuid.uuid4())
        
        with self._lock:
            self._active_streams[stream_id] = producer
        
        def run_handler() -> None:
            try:
                producer.start()
                handler(producer, arguments)
            except Exception as e:
                producer.push_error(str(e))
            finally:
                producer.push_done()
                producer.close()
                
                with self._lock:
                    self._active_streams.pop(stream_id, None)
        
        thread = threading.Thread(target=run_handler, daemon=True)
        thread.start()
        
        return response
    
    def cancel(self, stream_id: str) -> None:
        """取消流式执行。
        
        Args:
            stream_id: 流ID
        """
        with self._lock:
            producer = self._active_streams.pop(stream_id, None)
        
        if producer:
            producer.push_error("Cancelled")
            producer.close()
    
    @property
    def active_count(self) -> int:
        """获取活跃流数量。"""
        with self._lock:
            return len(self._active_streams)


class ChunkedStream:
    """分块流。
    
    将大数据分块发送。
    """
    
    def __init__(
        self,
        chunk_size: int = 4096,
        producer: Optional[StreamProducer] = None
    ):
        """初始化分块流。
        
        Args:
            chunk_size: 块大小
            producer: 流生产者
        """
        self.chunk_size = chunk_size
        self.producer = producer or StreamProducer()
    
    def send_text(self, text: str) -> None:
        """发送文本。
        
        Args:
            text: 文本内容
        """
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            self.producer.push_message({"chunk": chunk, "index": i // self.chunk_size})
        
        self.producer.push_done()
    
    def send_bytes(self, data: bytes) -> None:
        """发送字节。
        
        Args:
            data: 字节数据
        """
        import base64
        
        for i in range(0, len(data), self.chunk_size):
            chunk = data[i:i + self.chunk_size]
            self.producer.push_message({
                "chunk": base64.b64encode(chunk).decode("ascii"),
                "index": i // self.chunk_size,
                "encoding": "base64"
            })
        
        self.producer.push_done()
    
    def send_lines(self, lines: List[str]) -> None:
        """发送行列表。
        
        Args:
            lines: 行列表
        """
        total = len(lines)
        
        for i, line in enumerate(lines):
            self.producer.push_message({"line": line, "index": i})
            self.producer.push_progress(i + 1, total)
        
        self.producer.push_done()
    
    def send_iterator(
        self,
        items: Iterator[Any],
        total: Optional[int] = None
    ) -> None:
        """发送迭代器。
        
        Args:
            items: 迭代器
            total: 总数量（可选）
        """
        count = 0
        
        for item in items:
            count += 1
            self.producer.push_message({"item": item, "index": count - 1})
            
            if total:
                self.producer.push_progress(count, total)
        
        self.producer.push_done()


def stream_generator(
    func: Callable[[StreamProducer, Any], None]
) -> Callable[[Any], StreamingResponse]:
    """流式生成器装饰器。
    
    Args:
        func: 处理函数
        
    Returns:
        装饰后的函数
    """
    executor = StreamingToolExecutor()
    
    def wrapper(arguments: Any) -> StreamingResponse:
        return executor.execute_streaming(
            tool_name=func.__name__,
            handler=func,
            arguments=arguments or {}
        )
    
    return wrapper


__all__ = [
    "SSEEventType",
    "SSEEvent",
    "StreamProducer",
    "StreamConsumer",
    "StreamingResponse",
    "StreamingToolExecutor",
    "ChunkedStream",
    "stream_generator",
]
