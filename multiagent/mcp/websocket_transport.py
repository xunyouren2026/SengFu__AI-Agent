"""MCP WebSocket传输层模块。

本模块实现了基于WebSocket的传输层，用于与远程MCP服务通信。
支持双向实时通信，适用于远程服务调用场景。
"""

from __future__ import annotations

import json
import threading
import time
import uuid
import hashlib
import base64
import struct
from typing import Optional, Callable, Dict, Any, List, Union
from queue import Queue, Empty
from dataclasses import dataclass, field
from enum import Enum, auto
import socket
import ssl


class WebSocketOpcode(Enum):
    """WebSocket帧操作码。"""
    CONTINUATION = 0x0
    TEXT = 0x1
    BINARY = 0x2
    CLOSE = 0x8
    PING = 0x9
    PONG = 0xA


class WebSocketState(Enum):
    """WebSocket连接状态。"""
    CONNECTING = auto()
    OPEN = auto()
    CLOSING = auto()
    CLOSED = auto()


@dataclass
class WebSocketFrame:
    """WebSocket帧。
    
    Attributes:
        fin: 是否为最后一帧
        opcode: 操作码
        masked: 是否掩码
        payload: 负载数据
    """
    fin: bool = True
    opcode: WebSocketOpcode = WebSocketOpcode.TEXT
    masked: bool = False
    payload: bytes = b""
    
    def to_bytes(self) -> bytes:
        """将帧编码为字节。"""
        result = bytearray()
        
        # 第一个字节: FIN + RSV1-3 + Opcode
        first_byte = (0x80 if self.fin else 0x00) | self.opcode.value
        result.append(first_byte)
        
        # 第二个字节: MASK + Payload length
        payload_len = len(self.payload)
        if payload_len <= 125:
            second_byte = (0x80 if self.masked else 0x00) | payload_len
            result.append(second_byte)
        elif payload_len <= 65535:
            second_byte = (0x80 if self.masked else 0x00) | 126
            result.append(second_byte)
            result.extend(struct.pack(">H", payload_len))
        else:
            second_byte = (0x80 if self.masked else 0x00) | 127
            result.append(second_byte)
            result.extend(struct.pack(">Q", payload_len))
        
        # 掩码键
        mask_key = b""
        if self.masked:
            mask_key = bytes([int.from_bytes(uuid.uuid4().bytes[i:i+1], 'big') for i in range(4)])
            result.extend(mask_key)
        
        # 负载数据
        if self.masked:
            masked_payload = bytearray()
            for i, byte in enumerate(self.payload):
                masked_payload.append(byte ^ mask_key[i % 4])
            result.extend(masked_payload)
        else:
            result.extend(self.payload)
        
        return bytes(result)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> tuple[WebSocketFrame, int]:
        """从字节解码帧。
        
        Returns:
            (帧对象, 消耗的字节数)
        """
        if len(data) < 2:
            raise ValueError("Insufficient data for WebSocket frame")
        
        # 解析第一个字节
        first_byte = data[0]
        fin = bool(first_byte & 0x80)
        opcode = WebSocketOpcode(first_byte & 0x0F)
        
        # 解析第二个字节
        second_byte = data[1]
        masked = bool(second_byte & 0x80)
        payload_len = second_byte & 0x7F
        
        offset = 2
        
        # 解析扩展长度
        if payload_len == 126:
            if len(data) < offset + 2:
                raise ValueError("Insufficient data for extended length")
            payload_len = struct.unpack(">H", data[offset:offset+2])[0]
            offset += 2
        elif payload_len == 127:
            if len(data) < offset + 8:
                raise ValueError("Insufficient data for extended length")
            payload_len = struct.unpack(">Q", data[offset:offset+8])[0]
            offset += 8
        
        # 解析掩码键
        mask_key = b""
        if masked:
            if len(data) < offset + 4:
                raise ValueError("Insufficient data for mask key")
            mask_key = data[offset:offset+4]
            offset += 4
        
        # 解析负载
        if len(data) < offset + payload_len:
            raise ValueError("Insufficient data for payload")
        
        payload = data[offset:offset+payload_len]
        offset += payload_len
        
        # 解除掩码
        if masked:
            unmasked_payload = bytearray()
            for i, byte in enumerate(payload):
                unmasked_payload.append(byte ^ mask_key[i % 4])
            payload = bytes(unmasked_payload)
        
        return cls(fin=fin, opcode=opcode, masked=False, payload=payload), offset


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


class WebSocketTransport:
    """WebSocket传输层实现。
    
    通过WebSocket协议与远程MCP服务通信。
    支持连接管理、心跳保活、自动重连等功能。
    
    Attributes:
        url: WebSocket URL
        state: 连接状态
        message_queue: 消息队列
    """
    
    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        ping_interval: float = 30.0,
        ping_timeout: float = 10.0,
        reconnect_interval: float = 5.0,
        max_reconnect_attempts: int = 5,
        buffer_size: int = 8192
    ):
        """初始化WebSocket传输层。
        
        Args:
            url: WebSocket URL，格式为 ws://host:port/path 或 wss://host:port/path
            headers: 自定义HTTP头
            ping_interval: 心跳间隔（秒）
            ping_timeout: 心跳超时（秒）
            reconnect_interval: 重连间隔（秒）
            max_reconnect_attempts: 最大重连次数
            buffer_size: 接收缓冲区大小
        """
        self.url = url
        self.headers = headers or {}
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self.buffer_size = buffer_size
        
        self.state = WebSocketState.CLOSED
        self.message_queue: Queue[TransportMessage] = Queue()
        self._socket: Optional[socket.socket] = None
        self._read_thread: Optional[threading.Thread] = None
        self._ping_thread: Optional[threading.Thread] = None
        self._pending_requests: Dict[str, Queue[Dict[str, Any]]] = {}
        self._pending_lock = threading.Lock()
        
        self._message_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self._error_callback: Optional[Callable[[Exception], None]] = None
        self._close_callback: Optional[Callable[[], None]] = None
        
        self._last_ping_time = 0.0
        self._last_pong_time = 0.0
        self._reconnect_count = 0
        self._running = False
        
        # 解析URL
        self._parse_url()
    
    def _parse_url(self) -> None:
        """解析WebSocket URL。"""
        if self.url.startswith("wss://"):
            self._use_ssl = True
            host_port = self.url[6:]
        elif self.url.startswith("ws://"):
            self._use_ssl = False
            host_port = self.url[5:]
        else:
            raise ValueError(f"Invalid WebSocket URL: {self.url}")
        
        # 分离路径
        if "/" in host_port:
            host_port, self._path = host_port.split("/", 1)
            self._path = "/" + self._path
        else:
            self._path = "/"
        
        # 分离端口
        if ":" in host_port:
            self._host, port_str = host_port.split(":")
            self._port = int(port_str)
        else:
            self._host = host_port
            self._port = 443 if self._use_ssl else 80
    
    def connect(self) -> None:
        """建立WebSocket连接。"""
        if self.state != WebSocketState.CLOSED:
            return
        
        self.state = WebSocketState.CONNECTING
        
        try:
            # 创建TCP连接
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(10.0)
            self._socket.connect((self._host, self._port))
            
            # SSL包装
            if self._use_ssl:
                context = ssl.create_default_context()
                self._socket = context.wrap_socket(
                    self._socket,
                    server_hostname=self._host
                )
            
            # 发送HTTP升级请求
            self._send_handshake()
            
            # 接收握手响应
            self._receive_handshake()
            
            self.state = WebSocketState.OPEN
            self._reconnect_count = 0
            self._running = True
            
            # 启动读取线程
            self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self._read_thread.start()
            
            # 启动心跳线程
            self._ping_thread = threading.Thread(target=self._ping_loop, daemon=True)
            self._ping_thread.start()
            
        except Exception as e:
            self.state = WebSocketState.CLOSED
            if self._socket:
                try:
                    self._socket.close()
                except Exception:
                    pass
                self._socket = None
            raise ConnectionError(f"WebSocket connection failed: {e}") from e
    
    def _send_handshake(self) -> None:
        """发送WebSocket握手请求。"""
        # 生成Sec-WebSocket-Key
        key = base64.b64encode(uuid.uuid4().bytes).decode("ascii")
        self._ws_key = key
        
        # 构建HTTP请求
        request_lines = [
            f"GET {self._path} HTTP/1.1",
            f"Host: {self._host}:{self._port}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}",
            "Sec-WebSocket-Version: 13",
        ]
        
        # 添加自定义头
        for name, value in self.headers.items():
            request_lines.append(f"{name}: {value}")
        
        request_lines.append("")
        request_lines.append("")
        
        request = "\r\n".join(request_lines)
        self._socket.sendall(request.encode("utf-8"))
    
    def _receive_handshake(self) -> None:
        """接收WebSocket握手响应。"""
        response = b""
        while b"\r\n\r\n" not in response:
            chunk = self._socket.recv(self.buffer_size)
            if not chunk:
                raise ConnectionError("Connection closed during handshake")
            response += chunk
        
        # 解析响应
        response_text = response.decode("utf-8")
        lines = response_text.split("\r\n")
        
        # 检查状态行
        if not lines[0].startswith("HTTP/1.1 101"):
            raise ConnectionError(f"Handshake failed: {lines[0]}")
        
        # 验证Sec-WebSocket-Accept
        expected_accept = base64.b64encode(
            hashlib.sha1((self._ws_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
        ).decode("ascii")
        
        for line in lines[1:]:
            if line.lower().startswith("sec-websocket-accept:"):
                accept = line.split(":", 1)[1].strip()
                if accept != expected_accept:
                    raise ConnectionError("Invalid Sec-WebSocket-Accept")
                break
        else:
            raise ConnectionError("Missing Sec-WebSocket-Accept header")
    
    def _read_loop(self) -> None:
        """读取循环。"""
        buffer = b""
        
        try:
            while self._running and self.state == WebSocketState.OPEN:
                try:
                    chunk = self._socket.recv(self.buffer_size)
                    if not chunk:
                        break
                    
                    buffer += chunk
                    
                    # 处理完整帧
                    while buffer:
                        try:
                            frame, consumed = WebSocketFrame.from_bytes(buffer)
                            buffer = buffer[consumed:]
                            
                            self._handle_frame(frame)
                        
                        except ValueError:
                            # 数据不完整，等待更多数据
                            break
                
                except socket.timeout:
                    continue
                except Exception as e:
                    if self._error_callback:
                        self._error_callback(e)
                    break
        
        finally:
            if self.state == WebSocketState.OPEN:
                self._handle_disconnect()
    
    def _handle_frame(self, frame: WebSocketFrame) -> None:
        """处理接收到的帧。"""
        if frame.opcode == WebSocketOpcode.TEXT:
            try:
                message = json.loads(frame.payload.decode("utf-8"))
                msg_id = message.get("id", str(uuid.uuid4()))
                
                transport_msg = TransportMessage(
                    message_id=msg_id,
                    message=message,
                    timestamp=time.time()
                )
                
                self.message_queue.put(transport_msg)
                
                # 唤醒等待的请求
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
        
        elif frame.opcode == WebSocketOpcode.BINARY:
            # 处理二进制消息
            pass
        
        elif frame.opcode == WebSocketOpcode.PONG:
            self._last_pong_time = time.time()
        
        elif frame.opcode == WebSocketOpcode.PING:
            # 响应PONG
            pong_frame = WebSocketFrame(
                fin=True,
                opcode=WebSocketOpcode.PONG,
                masked=True,
                payload=frame.payload
            )
            self._send_frame(pong_frame)
        
        elif frame.opcode == WebSocketOpcode.CLOSE:
            self.close()
    
    def _ping_loop(self) -> None:
        """心跳循环。"""
        while self._running and self.state == WebSocketState.OPEN:
            time.sleep(self.ping_interval)
            
            if self.state != WebSocketState.OPEN:
                break
            
            # 发送PING
            ping_frame = WebSocketFrame(
                fin=True,
                opcode=WebSocketOpcode.PING,
                masked=True,
                payload=str(time.time()).encode("utf-8")
            )
            self._send_frame(ping_frame)
            self._last_ping_time = time.time()
    
    def _send_frame(self, frame: WebSocketFrame) -> None:
        """发送WebSocket帧。"""
        if self._socket and self.state == WebSocketState.OPEN:
            try:
                self._socket.sendall(frame.to_bytes())
            except Exception as e:
                if self._error_callback:
                    self._error_callback(e)
    
    def _handle_disconnect(self) -> None:
        """处理断开连接。"""
        self.state = WebSocketState.CLOSED
        
        if self._close_callback:
            self._close_callback()
        
        # 尝试重连
        if self._reconnect_count < self.max_reconnect_attempts:
            self._reconnect_count += 1
            time.sleep(self.reconnect_interval)
            try:
                self.connect()
            except Exception:
                pass
    
    def send(self, message: Dict[str, Any]) -> None:
        """发送消息。
        
        Args:
            message: JSON-RPC消息字典
        """
        if self.state != WebSocketState.OPEN:
            raise ConnectionError("WebSocket is not connected")
        
        json_str = json.dumps(message, ensure_ascii=False)
        frame = WebSocketFrame(
            fin=True,
            opcode=WebSocketOpcode.TEXT,
            masked=True,
            payload=json_str.encode("utf-8")
        )
        self._send_frame(frame)
    
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
        """
        request_id = str(uuid.uuid4())
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {}
        }
        
        response_queue: Queue[Dict[str, Any]] = Queue()
        
        with self._pending_lock:
            self._pending_requests[request_id] = response_queue
        
        try:
            self.send(request)
            
            try:
                response = response_queue.get(timeout=timeout)
                return response
            except Empty:
                raise TimeoutError(f"Request {method} timed out after {timeout}s")
        
        finally:
            with self._pending_lock:
                self._pending_requests.pop(request_id, None)
    
    def close(self) -> None:
        """关闭WebSocket连接。"""
        if self.state == WebSocketState.CLOSED:
            return
        
        self.state = WebSocketState.CLOSING
        self._running = False
        
        # 发送CLOSE帧
        if self._socket:
            try:
                close_frame = WebSocketFrame(
                    fin=True,
                    opcode=WebSocketOpcode.CLOSE,
                    masked=True,
                    payload=struct.pack(">H", 1000)  # Normal closure
                )
                self._socket.sendall(close_frame.to_bytes())
            except Exception:
                pass
        
        # 等待线程结束
        if self._read_thread and self._read_thread.is_alive():
            self._read_thread.join(timeout=1.0)
        
        if self._ping_thread and self._ping_thread.is_alive():
            self._ping_thread.join(timeout=1.0)
        
        # 关闭socket
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
        
        self.state = WebSocketState.CLOSED
    
    def on_message(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """注册消息回调。"""
        self._message_callback = callback
    
    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """注册错误回调。"""
        self._error_callback = callback
    
    def on_close(self, callback: Callable[[], None]) -> None:
        """注册关闭回调。"""
        self._close_callback = callback
    
    @property
    def is_connected(self) -> bool:
        """检查是否已连接。"""
        return self.state == WebSocketState.OPEN


class WebSocketServerTransport:
    """WebSocket服务器传输层。
    
    用于MCP服务器端，接受客户端WebSocket连接。
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        path: str = "/mcp",
        ssl_context: Optional[ssl.SSLContext] = None,
        max_connections: int = 100
    ):
        """初始化WebSocket服务器。
        
        Args:
            host: 监听地址
            port: 监听端口
            path: WebSocket路径
            ssl_context: SSL上下文（用于wss）
            max_connections: 最大连接数
        """
        self.host = host
        self.port = port
        self.path = path
        self.ssl_context = ssl_context
        self.max_connections = max_connections
        
        self._server_socket: Optional[socket.socket] = None
        self._connections: Dict[str, WebSocketTransport] = {}
        self._connection_lock = threading.Lock()
        self._running = False
        self._accept_thread: Optional[threading.Thread] = None
        
        self._message_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self._connect_callback: Optional[Callable[[str], None]] = None
        self._disconnect_callback: Optional[Callable[[str], None]] = None
    
    def start(self) -> None:
        """启动服务器。"""
        if self._running:
            return
        
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(self.max_connections)
        
        if self.ssl_context:
            self._server_socket = self.ssl_context.wrap_socket(
                self._server_socket,
                server_side=True
            )
        
        self._running = True
        
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()
    
    def _accept_loop(self) -> None:
        """接受连接循环。"""
        while self._running:
            try:
                client_socket, addr = self._server_socket.accept()
                conn_id = str(uuid.uuid4())
                
                # 处理新连接
                threading.Thread(
                    target=self._handle_connection,
                    args=(conn_id, client_socket, addr),
                    daemon=True
                ).start()
            
            except Exception:
                continue
    
    def _handle_connection(
        self,
        conn_id: str,
        client_socket: socket.socket,
        addr: tuple
    ) -> None:
        """处理客户端连接。"""
        try:
            # 接收HTTP请求
            request = b""
            while b"\r\n\r\n" not in request:
                chunk = client_socket.recv(4096)
                if not chunk:
                    return
                request += chunk
            
            request_text = request.decode("utf-8")
            lines = request_text.split("\r\n")
            
            # 验证WebSocket升级请求
            if not lines[0].startswith("GET"):
                self._send_http_error(client_socket, 400, "Bad Request")
                return
            
            # 检查路径
            path = lines[0].split()[1]
            if not path.startswith(self.path):
                self._send_http_error(client_socket, 404, "Not Found")
                return
            
            # 提取WebSocket Key
            ws_key = None
            for line in lines[1:]:
                if line.lower().startswith("sec-websocket-key:"):
                    ws_key = line.split(":", 1)[1].strip()
                    break
            
            if not ws_key:
                self._send_http_error(client_socket, 400, "Missing WebSocket Key")
                return
            
            # 发送握手响应
            accept = base64.b64encode(
                hashlib.sha1((ws_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
            ).decode("ascii")
            
            response = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n"
                "\r\n"
            )
            client_socket.sendall(response.encode("utf-8"))
            
            # 记录连接
            with self._connection_lock:
                self._connections[conn_id] = client_socket
            
            if self._connect_callback:
                self._connect_callback(conn_id)
            
            # 处理消息
            self._handle_messages(conn_id, client_socket)
        
        except Exception:
            pass
        
        finally:
            # 清理连接
            with self._connection_lock:
                self._connections.pop(conn_id, None)
            
            try:
                client_socket.close()
            except Exception:
                pass
            
            if self._disconnect_callback:
                self._disconnect_callback(conn_id)
    
    def _handle_messages(self, conn_id: str, client_socket: socket.socket) -> None:
        """处理客户端消息。"""
        buffer = b""
        
        while self._running:
            try:
                chunk = client_socket.recv(8192)
                if not chunk:
                    break
                
                buffer += chunk
                
                while buffer:
                    try:
                        frame, consumed = WebSocketFrame.from_bytes(buffer)
                        buffer = buffer[consumed:]
                        
                        if frame.opcode == WebSocketOpcode.TEXT:
                            try:
                                message = json.loads(frame.payload.decode("utf-8"))
                                if self._message_callback:
                                    self._message_callback(conn_id, message)
                            except json.JSONDecodeError:
                                pass
                        
                        elif frame.opcode == WebSocketOpcode.CLOSE:
                            return
                    
                    except ValueError:
                        break
            
            except Exception:
                break
    
    def _send_http_error(
        self,
        socket: socket.socket,
        code: int,
        message: str
    ) -> None:
        """发送HTTP错误响应。"""
        response = f"HTTP/1.1 {code} {message}\r\n\r\n"
        socket.sendall(response.encode("utf-8"))
        socket.close()
    
    def send(self, conn_id: str, message: Dict[str, Any]) -> None:
        """向指定连接发送消息。"""
        with self._connection_lock:
            client_socket = self._connections.get(conn_id)
        
        if client_socket:
            json_str = json.dumps(message, ensure_ascii=False)
            frame = WebSocketFrame(
                fin=True,
                opcode=WebSocketOpcode.TEXT,
                masked=False,
                payload=json_str.encode("utf-8")
            )
            client_socket.sendall(frame.to_bytes())
    
    def broadcast(self, message: Dict[str, Any]) -> None:
        """广播消息到所有连接。"""
        with self._connection_lock:
            conn_ids = list(self._connections.keys())
        
        for conn_id in conn_ids:
            try:
                self.send(conn_id, message)
            except Exception:
                pass
    
    def close(self) -> None:
        """关闭服务器。"""
        self._running = False
        
        # 关闭所有客户端连接
        with self._connection_lock:
            for conn_id, client_socket in list(self._connections.items()):
                try:
                    client_socket.close()
                except Exception:
                    pass
            self._connections.clear()
        
        # 关闭服务器socket
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass
            self._server_socket = None
    
    def on_message(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """注册消息回调。"""
        self._message_callback = callback
    
    def on_connect(self, callback: Callable[[str], None]) -> None:
        """注册连接回调。"""
        self._connect_callback = callback
    
    def on_disconnect(self, callback: Callable[[str], None]) -> None:
        """注册断开回调。"""
        self._disconnect_callback = callback
    
    @property
    def connection_count(self) -> int:
        """获取当前连接数。"""
        with self._connection_lock:
            return len(self._connections)


__all__ = [
    "WebSocketOpcode",
    "WebSocketState",
    "WebSocketFrame",
    "TransportMessage",
    "WebSocketTransport",
    "WebSocketServerTransport",
]
