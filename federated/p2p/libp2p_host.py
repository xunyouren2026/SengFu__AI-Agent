"""
libp2p Host - P2P主机封装
基于libp2p协议栈的主机抽象层实现

提供传输协议管理、对等节点连接、消息流处理和协议多路复用等功能。
支持TCP、QUIC和WebSocket三种传输协议，实现协议协商和消息路由。

Author: AGI Unified Framework
"""

import hashlib
import json
import time
import random
import struct
import socket
import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any, Callable
from enum import Enum
from collections import deque


# ============== 传输协议 ==============

class TransportProtocol(Enum):
    """
    传输协议枚举

    支持的传输协议类型：
    - TCP: 可靠的面向连接传输，适合大多数场景
    - QUIC: 基于UDP的低延迟传输，支持多路复用
    - WebSocket: 基于HTTP的传输，适合穿透防火墙
    """
    TCP = "tcp"
    QUIC = "quic"
    WEBSOCKET = "ws"

    @classmethod
    def from_string(cls, s: str) -> 'TransportProtocol':
        """从字符串解析协议类型"""
        s_lower = s.lower().strip()
        for proto in cls:
            if proto.value == s_lower:
                return proto
        raise ValueError(f"Unknown transport protocol: {s}")

    @property
    def default_port(self) -> int:
        """获取协议的默认端口"""
        defaults = {
            TransportProtocol.TCP: 4001,
            TransportProtocol.QUIC: 4002,
            TransportProtocol.WEBSOCKET: 4003
        }
        return defaults[self]

    @property
    def is_reliable(self) -> bool:
        """是否为可靠传输"""
        return self in (TransportProtocol.TCP, TransportProtocol.WEBSOCKET)

    @property
    def is_connection_oriented(self) -> bool:
        """是否为面向连接的协议"""
        return True


# ============== 对等节点信息 ==============

@dataclass
class PeerInfo:
    """
    对等节点信息

    包含节点的唯一标识、网络地址和支持的协议信息。
    每个对等节点通过其peer_id唯一标识。

    Attributes:
        peer_id: 节点的唯一标识符（SHA-256哈希）
        addresses: 节点的多地址列表 (protocol, host, port)
        protocols: 节点支持的协议列表
        public_key: 节点的公钥（可选）
        latency: 到该节点的估计延迟（毫秒）
        is_connected: 是否已连接
        connected_at: 连接建立时间
        metadata: 节点的元数据
    """
    peer_id: str
    addresses: List[Tuple[TransportProtocol, str, int]] = field(default_factory=list)
    protocols: List[str] = field(default_factory=list)
    public_key: Optional[str] = None
    latency: float = 0.0
    is_connected: bool = False
    connected_at: float = 0.0
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def primary_address(self) -> Optional[Tuple[TransportProtocol, str, int]]:
        """获取主地址（第一个地址）"""
        return self.addresses[0] if self.addresses else None

    def get_address_string(self) -> str:
        """获取地址字符串表示"""
        if not self.addresses:
            return f"{self.peer_id[:12]} (no address)"
        proto, host, port = self.addresses[0]
        return f"/{proto.value}/{host}/{port}"

    def supports_protocol(self, protocol: str) -> bool:
        """检查是否支持指定协议"""
        return protocol in self.protocols

    def update_latency(self, latency_ms: float) -> None:
        """更新延迟估计（使用指数移动平均）"""
        if self.latency == 0.0:
            self.latency = latency_ms
        else:
            alpha = 0.3
            self.latency = alpha * latency_ms + (1 - alpha) * self.latency

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            'peer_id': self.peer_id,
            'addresses': [(p.value, h, port) for p, h, port in self.addresses],
            'protocols': self.protocols,
            'latency': self.latency,
            'is_connected': self.is_connected,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PeerInfo':
        """从字典反序列化"""
        addresses = [
            (TransportProtocol.from_string(p), h, port)
            for p, h, port in data.get('addresses', [])
        ]
        return cls(
            peer_id=data['peer_id'],
            addresses=addresses,
            protocols=data.get('protocols', []),
            public_key=data.get('public_key'),
            latency=data.get('latency', 0.0),
            is_connected=data.get('is_connected', False),
            metadata=data.get('metadata', {})
        )

    @classmethod
    def generate(cls, host: str, port: int,
                 protocol: TransportProtocol = TransportProtocol.TCP) -> 'PeerInfo':
        """生成随机peer_id的PeerInfo"""
        raw = f"{host}:{port}:{time.time()}:{random.getrandbits(256)}"
        peer_id = hashlib.sha256(raw.encode()).hexdigest()
        return cls(
            peer_id=peer_id,
            addresses=[(protocol, host, port)],
            protocols=["/federated/1.0.0"]
        )


# ============== 主机配置 ==============

@dataclass
class HostConfig:
    """
    主机配置

    配置P2P主机的各项参数，包括传输协议、监听地址、连接限制等。

    Attributes:
        listen_host: 监听地址
        listen_port: 监听端口
        transport: 传输协议
        max_connections: 最大连接数
        connection_timeout: 连接超时（秒）
        read_timeout: 读取超时（秒）
        write_timeout: 写入超时（秒）
        enable_nat_traversal: 是否启用NAT穿透
        enable_relay: 是否启用中继
        user_agent: 用户代理标识
    """
    listen_host: str = "0.0.0.0"
    listen_port: int = 4001
    transport: TransportProtocol = TransportProtocol.TCP
    max_connections: int = 128
    connection_timeout: float = 30.0
    read_timeout: float = 60.0
    write_timeout: float = 30.0
    enable_nat_traversal: bool = True
    enable_relay: bool = False
    user_agent: str = "agi-federated/1.0"

    @property
    def listen_address(self) -> Tuple[str, int]:
        """获取监听地址"""
        return (self.listen_host, self.listen_port)


# ============== 消息流抽象 ==============

class Stream:
    """
    消息流抽象

    表示两个对等节点之间的双向通信通道。
    支持消息的发送、接收和流的生命周期管理。

    每个Stream关联一个协议标识符，用于协议多路复用。
    """

    def __init__(self, stream_id: str, protocol: str,
                 local_peer: str, remote_peer: str):
        self._stream_id = stream_id
        self._protocol = protocol
        self._local_peer = local_peer
        self._remote_peer = remote_peer
        self._is_open = True
        self._lock = threading.Lock()
        self._send_buffer: deque = deque(maxlen=1024)
        self._recv_buffer: deque = deque(maxlen=1024)
        self._stats = {
            'bytes_sent': 0,
            'bytes_received': 0,
            'messages_sent': 0,
            'messages_received': 0
        }
        self._created_at = time.time()
        self._close_callbacks: List[Callable] = []

    @property
    def stream_id(self) -> str:
        return self._stream_id

    @property
    def protocol(self) -> str:
        return self._protocol

    @property
    def local_peer(self) -> str:
        return self._local_peer

    @property
    def remote_peer(self) -> str:
        return self._remote_peer

    @property
    def is_open(self) -> bool:
        return self._is_open

    def send(self, data: bytes) -> int:
        """
        发送数据到流

        Args:
            data: 要发送的字节数据

        Returns:
            发送的字节数

        Raises:
            ConnectionError: 流已关闭
        """
        if not self._is_open:
            raise ConnectionError("Stream is closed")

        with self._lock:
            self._send_buffer.append(data)
            self._stats['bytes_sent'] += len(data)
            self._stats['messages_sent'] += 1
            return len(data)

    def send_message(self, message: Dict[str, Any]) -> int:
        """
        发送结构化消息

        将消息序列化为JSON后发送。

        Args:
            message: 消息字典

        Returns:
            发送的字节数
        """
        data = json.dumps(message, default=str).encode('utf-8')
        # 添加长度前缀
        length_prefix = struct.pack('!I', len(data))
        return self.send(length_prefix + data)

    def receive(self, timeout: float = 30.0) -> Optional[bytes]:
        """
        从流接收数据

        Args:
            timeout: 接收超时（秒）

        Returns:
            接收到的字节数据，超时返回None
        """
        if not self._is_open:
            return None

        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                if self._recv_buffer:
                    data = self._recv_buffer.popleft()
                    self._stats['bytes_received'] += len(data)
                    self._stats['messages_received'] += 1
                    return data
            time.sleep(0.01)

        return None

    def receive_message(self, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """
        接收结构化消息

        从流中读取长度前缀的消息并反序列化。

        Args:
            timeout: 接收超时

        Returns:
            消息字典或None
        """
        data = self.receive(timeout)
        if data is None:
            return None

        # 解析长度前缀
        if len(data) < 4:
            return None

        length = struct.unpack('!I', data[:4])[0]
        message_data = data[4:4 + length]

        try:
            return json.loads(message_data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def push_data(self, data: bytes) -> None:
        """向接收缓冲区推送数据（模拟网络接收）"""
        with self._lock:
            self._recv_buffer.append(data)

    def pop_send_data(self) -> Optional[bytes]:
        """从发送缓冲区取出数据（模拟网络发送）"""
        with self._lock:
            if self._send_buffer:
                return self._send_buffer.popleft()
        return None

    def close(self) -> None:
        """关闭流"""
        self._is_open = False
        for callback in self._close_callbacks:
            try:
                callback(self)
            except Exception:
                pass
        self._close_callbacks.clear()

    def on_close(self, callback: Callable[['Stream'], None]) -> None:
        """注册关闭回调"""
        self._close_callbacks.append(callback)

    def get_stats(self) -> Dict[str, Any]:
        """获取流统计信息"""
        return {
            'stream_id': self._stream_id,
            'protocol': self._protocol,
            'remote_peer': self._remote_peer,
            'is_open': self._is_open,
            'age_seconds': time.time() - self._created_at,
            **self._stats
        }


# ============== 协议处理器 ==============

class ProtocolHandler:
    """
    协议处理器基类

    处理特定协议的消息。子类需要实现handle_message方法。
    每个协议处理器关联一个协议标识符（如"/federated/1.0.0"）。
    """

    def __init__(self, protocol_id: str):
        self._protocol_id = protocol_id
        self._host: Optional['LibP2PHost'] = None

    @property
    def protocol_id(self) -> str:
        """获取协议标识符"""
        return self._protocol_id

    @property
    def host(self) -> Optional['LibP2PHost']:
        """获取关联的主机"""
        return self._host

    def set_host(self, host: 'LibP2PHost') -> None:
        """设置关联的主机"""
        self._host = host

    def handle_message(self, stream: Stream, message: Dict[str, Any]) -> None:
        """
        处理接收到的消息

        默认实现：记录消息并通过消息类型分发到对应的处理方法。
        子类可覆盖此方法或注册特定的消息处理器。

        Args:
            stream: 消息来源的流
            message: 消息内容
        """
        import logging
        _logger = logging.getLogger(__name__)

        msg_type = message.get("type", "unknown") if isinstance(message, dict) else "unknown"
        _logger.debug(
            "ProtocolHandler %s received message (type=%s) on stream %s",
            self._protocol_id, msg_type, stream.stream_id if hasattr(stream, 'stream_id') else id(stream),
        )

        # 尝试分发到类型特定的处理方法
        handler_name = f"_handle_{msg_type}"
        handler = getattr(self, handler_name, None)
        if handler is not None and callable(handler):
            try:
                handler(stream, message)
            except Exception as exc:
                _logger.error(
                    "Error in message handler %s for protocol %s: %s",
                    handler_name, self._protocol_id, exc,
                )
        else:
            _logger.debug(
                "No specific handler for message type '%s' on protocol %s; "
                "message payload: %s",
                msg_type, self._protocol_id, message,
            )

    def on_stream_opened(self, stream: Stream) -> None:
        """当新流打开时调用"""
        pass

    def on_stream_closed(self, stream: Stream) -> None:
        """当流关闭时调用"""
        pass

    def on_peer_connected(self, peer_info: PeerInfo) -> None:
        """当新对等节点连接时调用"""
        pass

    def on_peer_disconnected(self, peer_id: str) -> None:
        """当对等节点断开连接时调用"""
        pass


# ============== libp2p主机 ==============

class LibP2PHost:
    """
    libp2p主机类

    P2P网络的核心组件，负责：
    - 管理传输层连接
    - 协议多路复用
    - 对等节点管理
    - 消息路由和分发
    - 节点发现

    实现了libp2p核心接口的Python模拟版本。

    Author: AGI Unified Framework
    """

    def __init__(self, config: Optional[HostConfig] = None):
        self._config = config or HostConfig()
        self._peer_id = self._generate_peer_id()
        self._running = False

        # 连接管理
        self._connections: Dict[str, PeerInfo] = {}  # peer_id -> PeerInfo
        self._streams: Dict[str, Stream] = {}  # stream_id -> Stream
        self._peer_streams: Dict[str, List[str]] = {}  # peer_id -> [stream_ids]

        # 协议管理
        self._handlers: Dict[str, ProtocolHandler] = {}  # protocol_id -> handler
        self._supported_protocols: List[str] = []

        # 线程安全
        self._lock = threading.RLock()
        self._threads: List[threading.Thread] = []

        # 消息队列
        self._message_queue: deque = deque(maxlen=10000)
        self._event_handlers: Dict[str, List[Callable]] = {}

        # 统计
        self._stats = {
            'total_connections': 0,
            'total_streams_opened': 0,
            'total_messages_sent': 0,
            'total_messages_received': 0,
            'total_bytes_sent': 0,
            'total_bytes_received': 0
        }

    @property
    def peer_id(self) -> str:
        """获取本节点的peer_id"""
        return self._peer_id

    @property
    def config(self) -> HostConfig:
        """获取主机配置"""
        return self._config

    @property
    def is_running(self) -> bool:
        """主机是否正在运行"""
        return self._running

    @property
    def connected_peers(self) -> List[PeerInfo]:
        """获取所有已连接的对等节点"""
        with self._lock:
            return [p for p in self._connections.values() if p.is_connected]

    @property
    def peer_count(self) -> int:
        """获取已连接的对等节点数量"""
        with self._lock:
            return sum(1 for p in self._connections.values() if p.is_connected)

    def _generate_peer_id(self) -> str:
        """生成唯一的peer_id"""
        raw = f"libp2p-host:{self._config.listen_host}:{self._config.listen_port}"
        raw += f":{time.time()}:{random.getrandbits(512)}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get_peer_info(self) -> PeerInfo:
        """获取本节点的PeerInfo"""
        return PeerInfo(
            peer_id=self._peer_id,
            addresses=[(
                self._config.transport,
                self._config.listen_host,
                self._config.listen_port
            )],
            protocols=list(self._handlers.keys()),
            metadata={'user_agent': self._config.user_agent}
        )

    def start(self) -> None:
        """
        启动主机

        开始监听指定地址和端口，启动后台处理线程。
        """
        if self._running:
            return

        self._running = True

        # 启动消息处理线程
        msg_thread = threading.Thread(
            target=self._message_processing_loop,
            daemon=True,
            name="host-message-processor"
        )
        msg_thread.start()
        self._threads.append(msg_thread)

        # 启动连接维护线程
        maintain_thread = threading.Thread(
            target=self._connection_maintenance_loop,
            daemon=True,
            name="host-connection-maintainer"
        )
        maintain_thread.start()
        self._threads.append(maintain_thread)

        self._emit_event("host_started", {'peer_id': self._peer_id})

    def stop(self) -> None:
        """
        停止主机

        关闭所有连接和流，停止后台线程。
        """
        self._running = False

        # 关闭所有流
        with self._lock:
            for stream_id, stream in list(self._streams.items()):
                stream.close()
            self._streams.clear()
            self._peer_streams.clear()

        # 断开所有连接
        with self._lock:
            for peer_id, peer_info in self._connections.items():
                peer_info.is_connected = False
            self._connections.clear()

        # 等待线程结束
        for thread in self._threads:
            thread.join(timeout=5.0)
        self._threads.clear()

        self._emit_event("host_stopped", {'peer_id': self._peer_id})

    def connect(self, peer_info: PeerInfo) -> bool:
        """
        连接远程节点

        建立到指定对等节点的传输层连接。

        Args:
            peer_info: 目标节点信息

        Returns:
            是否连接成功
        """
        if not self._running:
            return False

        if peer_info.peer_id == self._peer_id:
            return False

        with self._lock:
            # 检查是否已连接
            if peer_info.peer_id in self._connections:
                existing = self._connections[peer_info.peer_id]
                if existing.is_connected:
                    return True

            # 检查连接数限制
            connected_count = sum(
                1 for p in self._connections.values() if p.is_connected
            )
            if connected_count >= self._config.max_connections:
                return False

            # 模拟连接建立
            try:
                # 检查地址可达性
                address = peer_info.primary_address
                if address is None:
                    return False

                proto, host, port = address
                # 模拟TCP连接
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self._config.connection_timeout)
                try:
                    # 尝试连接（可能失败，因为是模拟环境）
                    result = sock.connect_ex((host, port))
                    if result != 0 and result != 111:
                        # 111 = Connection refused (expected in test)
                        pass
                except Exception:
                    pass
                finally:
                    sock.close()

                # 更新节点信息
                peer_info.is_connected = True
                peer_info.connected_at = time.time()
                self._connections[peer_info.peer_id] = peer_info
                self._peer_streams[peer_info.peer_id] = []

                self._stats['total_connections'] += 1

                # 通知协议处理器
                for handler in self._handlers.values():
                    handler.on_peer_connected(peer_info)

                self._emit_event("peer_connected", {
                    'peer_id': peer_info.peer_id
                })

                return True

            except Exception:
                return False

    def connect_to(self, host: str, port: int,
                   protocol: TransportProtocol = TransportProtocol.TCP) -> bool:
        """
        通过地址连接远程节点

        Args:
            host: 远程主机地址
            port: 远程端口
            protocol: 传输协议

        Returns:
            是否连接成功
        """
        peer_info = PeerInfo.generate(host, port, protocol)
        return self.connect(peer_info)

    def disconnect(self, peer_id: str) -> bool:
        """
        断开与指定对等节点的连接

        Args:
            peer_id: 要断开的对等节点ID

        Returns:
            是否断开成功
        """
        with self._lock:
            if peer_id not in self._connections:
                return False

            peer_info = self._connections[peer_id]
            peer_info.is_connected = False

            # 关闭该节点的所有流
            stream_ids = self._peer_streams.get(peer_id, [])
            for sid in stream_ids:
                if sid in self._streams:
                    self._streams[sid].close()
                    del self._streams[sid]

            self._peer_streams.pop(peer_id, None)
            del self._connections[peer_id]

            # 通知协议处理器
            for handler in self._handlers.values():
                handler.on_peer_disconnected(peer_id)

            self._emit_event("peer_disconnected", {'peer_id': peer_id})

            return True

    def send_message(self, peer_id: str, protocol: str,
                     message: Dict[str, Any]) -> bool:
        """
        发送消息到指定对等节点

        通过指定协议向目标节点发送消息。
        如果没有到该节点的活跃流，会自动创建一个。

        Args:
            peer_id: 目标节点ID
            protocol: 使用的协议
            message: 消息内容

        Returns:
            是否发送成功
        """
        if not self._running:
            return False

        with self._lock:
            if peer_id not in self._connections:
                return False

            peer_info = self._connections[peer_id]
            if not peer_info.is_connected:
                return False

            # 查找或创建流
            stream = self._get_or_create_stream(peer_id, protocol)
            if stream is None:
                return False

            try:
                bytes_sent = stream.send_message(message)
                self._stats['total_messages_sent'] += 1
                self._stats['total_bytes_sent'] += bytes_sent
                return True
            except Exception:
                return False

    def broadcast_message(self, protocol: str,
                          message: Dict[str, Any]) -> int:
        """
        广播消息到所有已连接的对等节点

        Args:
            protocol: 使用的协议
            message: 消息内容

        Returns:
            成功发送的节点数量
        """
        sent_count = 0
        with self._lock:
            peer_ids = [
                pid for pid, p in self._connections.items()
                if p.is_connected
            ]

        for pid in peer_ids:
            if self.send_message(pid, protocol, message):
                sent_count += 1

        return sent_count

    def register_handler(self, handler: ProtocolHandler) -> None:
        """
        注册协议处理器

        Args:
            handler: 协议处理器实例
        """
        with self._lock:
            handler.set_host(self)
            self._handlers[handler.protocol_id] = handler
            if handler.protocol_id not in self._supported_protocols:
                self._supported_protocols.append(handler.protocol_id)

    def unregister_handler(self, protocol_id: str) -> bool:
        """
        注销协议处理器

        Args:
            protocol_id: 协议标识符

        Returns:
            是否注销成功
        """
        with self._lock:
            if protocol_id in self._handlers:
                del self._handlers[protocol_id]
                if protocol_id in self._supported_protocols:
                    self._supported_protocols.remove(protocol_id)
                return True
        return False

    def get_handler(self, protocol_id: str) -> Optional[ProtocolHandler]:
        """获取指定协议的处理器"""
        return self._handlers.get(protocol_id)

    def new_stream(self, peer_id: str, protocol: str) -> Optional[Stream]:
        """
        创建到指定对等节点的新流

        Args:
            peer_id: 目标节点ID
            protocol: 协议标识符

        Returns:
            新创建的流或None
        """
        with self._lock:
            if peer_id not in self._connections:
                return None

            peer_info = self._connections[peer_id]
            if not peer_info.is_connected:
                return None

            stream_id = str(uuid.uuid4())
            stream = Stream(stream_id, protocol, self._peer_id, peer_id)
            stream.on_close(self._on_stream_closed)

            self._streams[stream_id] = stream
            if peer_id not in self._peer_streams:
                self._peer_streams[peer_id] = []
            self._peer_streams[peer_id].append(stream_id)

            self._stats['total_streams_opened'] += 1

            # 通知处理器
            handler = self._handlers.get(protocol)
            if handler:
                handler.on_stream_opened(stream)

            return stream

    def discover_peers(self) -> List[PeerInfo]:
        """
        发现对等节点

        通过已连接的节点发现网络中的其他节点。
        实现随机漫步式的节点发现。

        Returns:
            新发现的节点列表
        """
        discovered: List[PeerInfo] = []

        with self._lock:
            connected = [
                p for p in self._connections.values()
                if p.is_connected
            ]

        for peer in connected:
            # 向已连接的节点请求其已知的对等节点
            # 模拟：每个节点返回2-5个已知节点
            num_new = random.randint(2, 5)
            for _ in range(num_new):
                new_host = f"10.0.{random.randint(1, 255)}.{random.randint(1, 255)}"
                new_port = random.randint(4001, 4100)
                new_peer = PeerInfo.generate(new_host, new_port)

                # 检查是否已知
                if new_peer.peer_id not in self._connections:
                    discovered.append(new_peer)

        return discovered

    def _get_or_create_stream(self, peer_id: str,
                               protocol: str) -> Optional[Stream]:
        """获取或创建到指定节点的流"""
        # 查找现有流
        stream_ids = self._peer_streams.get(peer_id, [])
        for sid in stream_ids:
            if sid in self._streams:
                stream = self._streams[sid]
                if stream.is_open and stream.protocol == protocol:
                    return stream

        # 创建新流
        return self.new_stream(peer_id, protocol)

    def _on_stream_closed(self, stream: Stream) -> None:
        """流关闭回调"""
        with self._lock:
            if stream.stream_id in self._streams:
                del self._streams[stream.stream_id]

            # 从peer_streams中移除
            peer_id = stream.remote_peer
            if peer_id in self._peer_streams:
                if stream.stream_id in self._peer_streams[peer_id]:
                    self._peer_streams[peer_id].remove(stream.stream_id)

        # 通知处理器
        handler = self._handlers.get(stream.protocol)
        if handler:
            handler.on_stream_closed(stream)

    def _message_processing_loop(self) -> None:
        """消息处理循环"""
        while self._running:
            time.sleep(0.1)

            # 模拟从流中接收消息并分发
            with self._lock:
                for stream_id, stream in list(self._streams.items()):
                    if not stream.is_open:
                        continue

                    # 检查是否有待发送的数据（模拟网络传输）
                    data = stream.pop_send_data()
                    if data:
                        # 模拟网络延迟后投递到接收缓冲区
                        pass

    def _connection_maintenance_loop(self) -> None:
        """连接维护循环"""
        while self._running:
            time.sleep(30.0)

            # 检查过期连接
            now = time.time()
            with self._lock:
                stale_peers = [
                    pid for pid, p in self._connections.items()
                    if p.is_connected and
                    (now - p.connected_at) > 3600 and
                    random.random() < 0.1  # 10%概率断开
                ]

            for pid in stale_peers:
                self.disconnect(pid)

            # 更新延迟估计
            with self._lock:
                for peer_info in self._connections.values():
                    if peer_info.is_connected:
                        # 模拟延迟波动
                        base_latency = random.uniform(10, 200)
                        peer_info.update_latency(base_latency)

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """触发事件"""
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(data)
            except Exception:
                pass

    def on_event(self, event_type: str,
                 handler: Callable[[Dict[str, Any]], None]) -> None:
        """注册事件处理器"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def get_peer(self, peer_id: str) -> Optional[PeerInfo]:
        """获取指定对等节点的信息"""
        return self._connections.get(peer_id)

    def get_status(self) -> Dict[str, Any]:
        """获取主机状态"""
        with self._lock:
            return {
                'peer_id': self._peer_id,
                'is_running': self._running,
                'listen_address': f"{self._config.listen_host}:{self._config.listen_port}",
                'transport': self._config.transport.value,
                'connected_peers': self.peer_count,
                'active_streams': len(self._streams),
                'supported_protocols': self._supported_protocols,
                'stats': dict(self._stats)
            }


# ============== 主程序入口 ==============

if __name__ == "__main__":
    print("=== libp2p Host Demo ===\n")

    # 创建主机
    config = HostConfig(listen_host="127.0.0.1", listen_port=5001)
    host = LibP2PHost(config)

    # 注册事件处理器
    def on_peer_connected(data: Dict[str, Any]) -> None:
        print(f"Peer connected: {data['peer_id'][:12]}...")

    host.on_event("peer_connected", on_peer_connected)

    # 启动主机
    host.start()
    print(f"Host started: {host.peer_id[:16]}...")
    print(f"Listening on {config.listen_host}:{config.listen_port}")

    # 模拟连接
    peer = PeerInfo.generate("127.0.0.1", 5002)
    success = host.connect(peer)
    print(f"Connect to peer: {success}")

    # 发现节点
    discovered = host.discover_peers()
    print(f"Discovered peers: {len(discovered)}")

    # 状态
    status = host.get_status()
    print(f"\nStatus: {json.dumps(status, indent=2)}")

    # 停止
    host.stop()
    print("\nHost stopped.")

    print("\n=== Demo Complete ===")
