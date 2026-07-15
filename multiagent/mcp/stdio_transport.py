"""MCP标准IO传输层模块。

本模块实现了基于标准输入输出的传输层，用于通过子进程stdin/stdout与MCP服务通信。
适用于本地进程间通信场景。
"""

from __future__ import annotations

import sys
import io
import json
import threading
import time
import uuid
from typing import Optional, Callable, Dict, Any, List, Protocol
from queue import Queue, Empty
from dataclasses import dataclass

# 尝试导入select/poll，但仅使用标准库
try:
    import select
    HAS_SELECT = True
except ImportError:
    HAS_SELECT = False


class Transport(Protocol):
    """传输层接口协议。"""
    
    def start(self) -> None:
        """启动传输层。"""
        ...
    
    def send(self, message: Dict[str, Any]) -> None:
        """发送消息。"""
        ...
    
    def close(self) -> None:
        """关闭传输层。"""
        ...
    
    def on_message(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """注册消息回调。"""
        ...
    
    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """注册错误回调。"""
        ...


@dataclass
class TransportMessage:
    """传输消息包装器。
    
    Attributes:
        message_id: 消息唯一标识
        message: 消息内容
        timestamp: 时间戳
    """
    message_id: str
    message: Dict[str, Any]
    timestamp: float


class StdioTransport:
    """标准IO传输层实现。
    
    通过stdin读取JSON-RPC消息，通过stdout发送JSON-RPC消息。
    支持基于消息ID的请求-响应匹配。
    
    Attributes:
        input_stream: 输入流，默认sys.stdin.buffer
        output_stream: 输出流，默认sys.stdout.buffer
        message_queue: 消息队列
        running: 是否正在运行
    """
    
    def __init__(
        self,
        input_stream: Optional[io.RawIOBase] = None,
        output_stream: Optional[io.RawIOBase] = None,
        buffer_size: int = 8192
    ):
        """初始化标准IO传输层。
        
        Args:
            input_stream: 输入流，默认为sys.stdin.buffer
            output_stream: 输出流，默认为sys.stdout.buffer
            buffer_size: 读取缓冲区大小
        """
        self.input_stream = input_stream or sys.stdin.buffer
        self.output_stream = output_stream or sys.stdout.buffer
        self.buffer_size = buffer_size
        
        self.message_queue: Queue[TransportMessage] = Queue()
        self.running = False
        self._read_thread: Optional[threading.Thread] = None
        self._write_thread: Optional[threading.Thread] = None
        self._pending_requests: Dict[str, Queue[Dict[str, Any]]] = {}
        self._pending_lock = threading.Lock()
        
        self._message_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self._error_callback: Optional[Callable[[Exception], None]] = None
        
        self._read_buffer = b""
        self._message_delimiter = b"\n"
    
    def start(self) -> None:
        """启动传输层。"""
        if self.running:
            return
        
        self.running = True
        
        # 启动读取线程
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()
        
        # 启动写入线程
        self._write_thread = threading.Thread(target=self._write_loop, daemon=True)
        self._write_thread.start()
    
    def _read_loop(self) -> None:
        """读取循环，从stdin读取消息并放入队列。"""
        try:
            while self.running:
                try:
                    # 使用select或非阻塞读取
                    if HAS_SELECT:
                        ready, _, _ = select.select([self.input_stream], [], [], 0.1)
                        if not ready:
                            continue
                    
                    # 读取数据
                    chunk = self.input_stream.read(self.buffer_size)
                    if not chunk:
                        # EOF
                        time.sleep(0.01)
                        continue
                    
                    self._read_buffer += chunk
                    
                    # 处理完整消息
                    while self._message_delimiter in self._read_buffer:
                        line_end = self._read_buffer.index(self._message_delimiter)
                        line = self._read_buffer[:line_end]
                        self._read_buffer = self._read_buffer[line_end + len(self._message_delimiter):]
                        
                        if line.strip():
                            try:
                                message = json.loads(line.decode("utf-8"))
                                msg_id = message.get("id", str(uuid.uuid4()))
                                
                                transport_msg = TransportMessage(
                                    message_id=msg_id,
                                    message=message,
                                    timestamp=time.time()
                                )
                                
                                self.message_queue.put(transport_msg)
                                
                                # 如果有对应的pending请求，唤醒等待者
                                with self._pending_lock:
                                    if msg_id in self._pending_requests:
                                        self._pending_requests[msg_id].put(message)
                                
                                # 调用消息回调
                                if self._message_callback:
                                    try:
                                        self._message_callback(message)
                                    except Exception as e:
                                        if self._error_callback:
                                            self._error_callback(e)
                            
                            except json.JSONDecodeError as e:
                                if self._error_callback:
                                    self._error_callback(e)
                
                except Exception as e:
                    if self._error_callback:
                        self._error_callback(e)
                    time.sleep(0.1)
        
        except Exception as e:
            if self._error_callback:
                self._error_callback(e)
    
    def _write_loop(self) -> None:
        """写入循环，从队列取出消息并写入stdout。"""
        while self.running:
            try:
                # 非阻塞方式从队列获取消息
                msg = self.message_queue.get(timeout=0.1)
                
                if isinstance(msg, TransportMessage):
                    message = msg.message
                else:
                    message = msg
                
                # 序列化消息
                json_str = json.dumps(message, ensure_ascii=False)
                line = json_str.encode("utf-8") + self._message_delimiter
                
                # 写入stdout
                self.output_stream.write(line)
                self.output_stream.flush()
            
            except Empty:
                continue
            except Exception as e:
                if self._error_callback:
                    self._error_callback(e)
                time.sleep(0.1)
    
    def send(self, message: Dict[str, Any]) -> None:
        """发送消息。
        
        Args:
            message: JSON-RPC消息字典
        """
        transport_msg = TransportMessage(
            message_id=message.get("id", str(uuid.uuid4())),
            message=message,
            timestamp=time.time()
        )
        self.message_queue.put(transport_msg)
    
    def send_raw(self, message: Dict[str, Any]) -> None:
        """直接发送消息（不经过队列）。
        
        Args:
            message: JSON-RPC消息字典
        """
        json_str = json.dumps(message, ensure_ascii=False)
        line = json_str.encode("utf-8") + self._message_delimiter
        self.output_stream.write(line)
        self.output_stream.flush()
    
    def request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """发送请求并等待响应。
        
        Args:
            method: 方法名
            params: 参数
            timeout: 超时时间（秒）
            
        Returns:
            响应消息
            
        Raises:
            TimeoutError: 请求超时
            Exception: 传输错误
        """
        request_id = str(uuid.uuid4())
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {}
        }
        
        # 创建响应队列
        response_queue: Queue[Dict[str, Any]] = Queue()
        
        with self._pending_lock:
            self._pending_requests[request_id] = response_queue
        
        try:
            # 发送请求
            self.send_raw(request)
            
            # 等待响应
            try:
                response = response_queue.get(timeout=timeout)
                return response
            except Empty:
                raise TimeoutError(f"Request {method} timed out after {timeout}s")
        
        finally:
            with self._pending_lock:
                self._pending_requests.pop(request_id, None)
    
    def close(self) -> None:
        """关闭传输层。"""
        self.running = False
        
        # 等待线程结束
        if self._read_thread and self._read_thread.is_alive():
            self._read_thread.join(timeout=1.0)
        
        if self._write_thread and self._write_thread.is_alive():
            self._write_thread.join(timeout=1.0)
        
        # 清空队列
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
            except Empty:
                break
    
    def on_message(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """注册消息回调。
        
        Args:
            callback: 消息回调函数
        """
        self._message_callback = callback
    
    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """注册错误回调。
        
        Args:
            callback: 错误回调函数
        """
        self._error_callback = callback
    
    @property
    def is_running(self) -> bool:
        """检查传输层是否正在运行。"""
        return self.running
    
    def get_queue_size(self) -> int:
        """获取消息队列当前大小。"""
        return self.message_queue.qsize()


class StdioServerTransport(StdioTransport):
    """标准IO服务器传输层。
    
    专门用于MCP服务器端，通过stdin接收请求，通过stdout发送响应。
    """
    
    def __init__(
        self,
        input_stream: Optional[io.RawIOBase] = None,
        output_stream: Optional[io.RawIOBase] = None,
        buffer_size: int = 8192
    ):
        """初始化服务器传输层。"""
        super().__init__(input_stream, output_stream, buffer_size)
        self._request_handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    
    def set_request_handler(
        self,
        handler: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> None:
        """设置请求处理器。
        
        Args:
            handler: 请求处理函数，输入请求，返回响应
        """
        self._request_handler = handler
        
        # 设置默认消息回调
        def handle_message(message: Dict[str, Any]) -> None:
            if "id" in message:
                try:
                    response = self._request_handler(message)
                    if response:
                        self.send_raw(response)
                except Exception as e:
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": message.get("id"),
                        "error": {
                            "code": -32603,
                            "message": f"Internal error: {str(e)}"
                        }
                    }
                    self.send_raw(error_response)
        
        self.on_message(handle_message)
    
    def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        """发送通知（不需要响应）。
        
        Args:
            method: 方法名
            params: 参数
        """
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {}
        }
        self.send_raw(notification)


class StdioClientTransport(StdioTransport):
    """标准IO客户端传输层。
    
    专门用于MCP客户端，通过stdin接收响应，通过stdout发送请求。
    """
    
    def __init__(
        self,
        input_stream: Optional[io.RawIOBase] = None,
        output_stream: Optional[io.RawIOBase] = None,
        buffer_size: int = 8192
    ):
        """初始化客户端传输层。"""
        super().__init__(input_stream, output_stream, buffer_size)
        self._notification_handler: Optional[Callable[[str, Dict[str, Any]], None]] = None
    
    def set_notification_handler(
        self,
        handler: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """设置通知处理器。
        
        Args:
            handler: 通知处理函数，输入方法名和参数
        """
        self._notification_handler = handler
        
        # 设置消息回调来区分通知和响应
        original_callback = self._message_callback
        
        def handle_message(message: Dict[str, Any]) -> None:
            # 如果没有id字段，则是通知
            if "id" not in message and "method" in message:
                if self._notification_handler:
                    self._notification_handler(
                        message.get("method", ""),
                        message.get("params", {})
                    )
            elif original_callback:
                original_callback(message)
        
        self.on_message(handle_message)
    
    def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """调用工具。
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数
            timeout: 超时时间
            
        Returns:
            工具调用结果
        """
        return self.request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        }, timeout)
    
    def list_tools(self, timeout: float = 30.0) -> Dict[str, Any]:
        """列出所有可用工具。
        
        Args:
            timeout: 超时时间
            
        Returns:
            工具列表响应
        """
        return self.request("tools/list", timeout=timeout)
    
    def list_resources(self, timeout: float = 30.0) -> Dict[str, Any]:
        """列出所有可用资源。
        
        Args:
            timeout: 超时时间
            
        Returns:
            资源列表响应
        """
        return self.request("resources/list", timeout=timeout)
    
    def read_resource(
        self,
        uri: str,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """读取资源内容。
        
        Args:
            uri: 资源URI
            timeout: 超时时间
            
        Returns:
            资源内容响应
        """
        return self.request("resources/read", {"uri": uri}, timeout)
    
    def list_prompts(self, timeout: float = 30.0) -> Dict[str, Any]:
        """列出所有可用提示词。
        
        Args:
            timeout: 超时时间
            
        Returns:
            提示词列表响应
        """
        return self.request("prompts/list", timeout=timeout)
    
    def get_prompt(
        self,
        name: str,
        arguments: Dict[str, Any],
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """获取渲染后的提示词。
        
        Args:
            name: 提示词名称
            arguments: 提示词参数
            timeout: 超时时间
            
        Returns:
            渲染后的提示词响应
        """
        return self.request("prompts/get", {
            "name": name,
            "arguments": arguments
        }, timeout)


__all__ = [
    "Transport",
    "TransportMessage",
    "StdioTransport",
    "StdioServerTransport",
    "StdioClientTransport",
]
