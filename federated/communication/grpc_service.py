"""
gRPC通信服务
"""
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum
import json
import struct


class MessageType(Enum):
    """消息类型"""
    REGISTER = "register"
    UNREGISTER = "unregister"
    HEARTBEAT = "heartbeat"
    MODEL_UPDATE = "model_update"
    MODEL_REQUEST = "model_request"
    AGGREGATION_TRIGGER = "aggregation_trigger"
    CLIENT_SELECTION = "client_selection"
    ERROR = "error"


class Message:
    """消息"""
    
    def __init__(
        self,
        msg_type: MessageType,
        sender_id: str,
        receiver_id: str,
        payload: Optional[Dict[str, Any]] = None,
        msg_id: Optional[str] = None
    ):
        self.msg_type = msg_type
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.payload = payload or {}
        self.msg_id = msg_id or self._generate_id()
        self.timestamp = datetime.now().timestamp()
        self.ttl: int = 100  # 生存时间
    
    def _generate_id(self) -> str:
        """生成消息ID"""
        import random
        return f"{self.sender_id}_{int(self.timestamp * 1000)}_{random.randint(0, 9999)}"
    
    def serialize(self) -> bytes:
        """序列化消息"""
        data = {
            'msg_type': self.msg_type.value,
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'payload': self.payload,
            'msg_id': self.msg_id,
            'timestamp': self.timestamp,
            'ttl': self.ttl
        }
        return json.dumps(data).encode('utf-8')
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'Message':
        """反序列化消息"""
        obj = json.loads(data.decode('utf-8'))
        msg = cls(
            msg_type=MessageType(obj['msg_type']),
            sender_id=obj['sender_id'],
            receiver_id=obj['receiver_id'],
            payload=obj.get('payload', {}),
            msg_id=obj.get('msg_id')
        )
        msg.timestamp = obj.get('timestamp', msg.timestamp)
        msg.ttl = obj.get('ttl', 100)
        return msg
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'msg_type': self.msg_type.value,
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'payload': self.payload,
            'msg_id': self.msg_id,
            'timestamp': self.timestamp
        }


class Connection:
    """连接"""
    
    def __init__(
        self,
        conn_id: str,
        remote_address: str
    ):
        self.conn_id = conn_id
        self.remote_address = remote_address
        self.created_at = datetime.now().timestamp()
        self.last_active = self.created_at
        self.is_active = True
        self.bytes_sent = 0
        self.bytes_received = 0
    
    def update_activity(self) -> None:
        """更新活动时间"""
        self.last_active = datetime.now().timestamp()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            'conn_id': self.conn_id,
            'remote_address': self.remote_address,
            'is_active': self.is_active,
            'bytes_sent': self.bytes_sent,
            'bytes_received': self.bytes_received,
            'uptime': datetime.now().timestamp() - self.created_at
        }


class ServiceConfig:
    """服务配置"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 50051,
        max_connections: int = 100,
        timeout: float = 30.0,
        max_message_size: int = 10 * 1024 * 1024  # 10MB
    ):
        self.host = host
        self.port = port
        self.max_connections = max_connections
        self.timeout = timeout
        self.max_message_size = max_message_size


class GRPCService:
    """
    gRPC通信服务
    
    提供联邦学习的通信基础设施
    """
    
    def __init__(self, config: Optional[ServiceConfig] = None):
        self.config = config or ServiceConfig()
        
        self._connections: Dict[str, Connection] = {}
        self._handlers: Dict[MessageType, List[Callable]] = {}
        self._message_queue: List[Message] = []
        
        self._is_running = False
        self._total_messages = 0
        self._total_bytes = 0
    
    def start(self) -> None:
        """启动服务"""
        self._is_running = True
    
    def stop(self) -> None:
        """停止服务"""
        self._is_running = False
        self._connections.clear()
    
    def register_handler(
        self,
        msg_type: MessageType,
        handler: Callable[[Message], Optional[Message]]
    ) -> None:
        """注册消息处理器"""
        if msg_type not in self._handlers:
            self._handlers[msg_type] = []
        self._handlers[msg_type].append(handler)
    
    def unregister_handler(
        self,
        msg_type: MessageType,
        handler: Callable
    ) -> None:
        """注销消息处理器"""
        if msg_type in self._handlers:
            try:
                self._handlers[msg_type].remove(handler)
            except ValueError:
                pass
    
    def accept_connection(
        self,
        conn_id: str,
        remote_address: str
    ) -> Optional[Connection]:
        """接受连接"""
        if len(self._connections) >= self.config.max_connections:
            return None
        
        conn = Connection(conn_id, remote_address)
        self._connections[conn_id] = conn
        return conn
    
    def close_connection(self, conn_id: str) -> None:
        """关闭连接"""
        if conn_id in self._connections:
            self._connections[conn_id].is_active = False
            del self._connections[conn_id]
    
    def send(
        self,
        conn_id: str,
        message: Message
    ) -> bool:
        """
        发送消息
        
        Args:
            conn_id: 连接ID
            message: 消息
        
        Returns:
            是否成功
        """
        if conn_id not in self._connections:
            return False
        
        conn = self._connections[conn_id]
        if not conn.is_active:
            return False
        
        # 检查消息大小
        serialized = message.serialize()
        if len(serialized) > self.config.max_message_size:
            return False
        
        conn.bytes_sent += len(serialized)
        conn.update_activity()
        
        self._total_messages += 1
        self._total_bytes += len(serialized)
        
        return True
    
    def receive(
        self,
        conn_id: str,
        data: bytes
    ) -> Optional[Message]:
        """
        接收消息
        
        Args:
            conn_id: 连接ID
            data: 原始数据
        
        Returns:
            解析后的消息
        """
        if conn_id not in self._connections:
            return None
        
        conn = self._connections[conn_id]
        conn.bytes_received += len(data)
        conn.update_activity()
        
        try:
            message = Message.deserialize(data)
            self._process_message(message)
            return message
        except Exception:
            return None
    
    def _process_message(self, message: Message) -> None:
        """处理消息"""
        handlers = self._handlers.get(message.msg_type, [])
        
        for handler in handlers:
            try:
                response = handler(message)
                if response:
                    self._message_queue.append(response)
            except Exception:
                pass
    
    def broadcast(
        self,
        message: Message,
        exclude: Optional[set] = None
    ) -> int:
        """
        广播消息
        
        Args:
            message: 消息
            exclude: 排除的连接ID集合
        
        Returns:
            发送成功的数量
        """
        exclude = exclude or set()
        count = 0
        
        for conn_id, conn in self._connections.items():
            if conn_id not in exclude and conn.is_active:
                if self.send(conn_id, message):
                    count += 1
        
        return count
    
    def get_pending_messages(self) -> List[Message]:
        """获取待处理消息"""
        messages = self._message_queue.copy()
        self._message_queue.clear()
        return messages
    
    def get_connection(self, conn_id: str) -> Optional[Connection]:
        """获取连接"""
        return self._connections.get(conn_id)
    
    def get_active_connections(self) -> List[str]:
        """获取活跃连接"""
        return [
            conn_id for conn_id, conn in self._connections.items()
            if conn.is_active
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        active = sum(1 for c in self._connections.values() if c.is_active)
        
        return {
            'is_running': self._is_running,
            'host': self.config.host,
            'port': self.config.port,
            'total_connections': len(self._connections),
            'active_connections': active,
            'max_connections': self.config.max_connections,
            'total_messages': self._total_messages,
            'total_bytes': self._total_bytes
        }


class FederatedService(GRPCService):
    """
    联邦学习服务
    
    扩展基础gRPC服务，添加联邦学习特定功能
    """
    
    def __init__(self, config: Optional[ServiceConfig] = None):
        super().__init__(config)
        
        self._clients: Dict[str, Dict[str, Any]] = {}
        self._global_model: Optional[Dict[str, Any]] = None
        self._current_round: int = 0
        
        # 注册默认处理器
        self._register_default_handlers()
    
    def _register_default_handlers(self) -> None:
        """注册默认处理器"""
        self.register_handler(
            MessageType.REGISTER,
            self._handle_register
        )
        self.register_handler(
            MessageType.UNREGISTER,
            self._handle_unregister
        )
        self.register_handler(
            MessageType.HEARTBEAT,
            self._handle_heartbeat
        )
        self.register_handler(
            MessageType.MODEL_UPDATE,
            self._handle_model_update
        )
        self.register_handler(
            MessageType.MODEL_REQUEST,
            self._handle_model_request
        )
    
    def _handle_register(self, message: Message) -> Optional[Message]:
        """处理注册"""
        client_id = message.sender_id
        info = message.payload
        
        self._clients[client_id] = {
            'info': info,
            'registered_at': datetime.now().timestamp(),
            'last_heartbeat': datetime.now().timestamp()
        }
        
        return Message(
            msg_type=MessageType.REGISTER,
            sender_id="server",
            receiver_id=client_id,
            payload={'status': 'success', 'client_id': client_id}
        )
    
    def _handle_unregister(self, message: Message) -> Optional[Message]:
        """处理注销"""
        client_id = message.sender_id
        self._clients.pop(client_id, None)
        return None
    
    def _handle_heartbeat(self, message: Message) -> Optional[Message]:
        """处理心跳"""
        client_id = message.sender_id
        if client_id in self._clients:
            self._clients[client_id]['last_heartbeat'] = datetime.now().timestamp()
        return None
    
    def _handle_model_update(self, message: Message) -> Optional[Message]:
        """处理模型更新"""
        # 将更新放入队列等待聚合
        return None
    
    def _handle_model_request(self, message: Message) -> Optional[Message]:
        """处理模型请求"""
        return Message(
            msg_type=MessageType.MODEL_REQUEST,
            sender_id="server",
            receiver_id=message.sender_id,
            payload={
                'model': self._global_model,
                'round': self._current_round
            }
        )
    
    def set_global_model(
        self,
        model: Dict[str, Any],
        round_num: int
    ) -> None:
        """设置全局模型"""
        self._global_model = model
        self._current_round = round_num
    
    def get_registered_clients(self) -> List[str]:
        """获取已注册客户端"""
        return list(self._clients.keys())
    
    def get_client_info(self, client_id: str) -> Optional[Dict[str, Any]]:
        """获取客户端信息"""
        return self._clients.get(client_id)
