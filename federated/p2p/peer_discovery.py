"""
P2P Peer Discovery and Communication Module
模拟古代分封制的"驿站"系统 - 分布式节点发现与通信

This module implements a decentralized P2P network architecture inspired by
the ancient feudal messenger station system. It provides node discovery,
DHT-based storage, gossip protocol for message propagation, and feudal
hierarchy management.

Author: AGI Unified Framework
"""

import threading
import time
import random
import hashlib
import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Callable, Any, Set
from collections import deque
from enum import Enum
import socket
import struct


# ============== 数据结构定义 ==============

class TrustLevel(Enum):
    """信任等级枚举"""
    UNTRUSTED = 0
    NEWCOMER = 1
    TRUSTED = 2
    ELITE = 3
    ROYAL = 4  # 最高信任等级


class Region(Enum):
    """区域枚举 - 类似古代的诸侯封地"""
    CAPITAL = "capital"           # 王都
    EASTERN_PROVINCE = "east"      # 东方领地
    WESTERN_PROVINCE = "west"      # 西方领地
    NORTHERN_FRONTIER = "north"    # 北境边疆
    SOUTHERN_REACH = "south"       # 南疆
    CENTRAL_HEARTLAND = "center"   # 中原腹地


@dataclass
class PeerNode:
    """
    P2P网络中的对等节点
    类似古代分封制中的"驿站"或"城镇"
    """
    peer_id: str
    address: str
    port: int
    public_key: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)  # 能力列表
    trust_level: TrustLevel = TrustLevel.NEWCOMER
    last_seen: float = field(default_factory=time.time)
    region: Region = Region.CENTRAL_HEARTLAND
    is_online: bool = True
    bandwidth: int = 1000  # 带宽(Mbps)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        d = asdict(self)
        d['trust_level'] = self.trust_level.value
        d['region'] = self.region.value
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PeerNode':
        """从字典创建"""
        data['trust_level'] = TrustLevel(data['trust_level'])
        data['region'] = Region(data['region'])
        return cls(**data)
    
    def is_stale(self, timeout: float = 300.0) -> bool:
        """检查节点是否过期（太久没响应）"""
        return (time.time() - self.last_seen) > timeout
    
    def update_last_seen(self):
        """更新最后可见时间"""
        self.last_seen = time.time()
        self.is_online = True


@dataclass
class DHTEntry:
    """
    分布式哈希表条目
    存储在网络中的键值对
    """
    key: str
    value: Any
    owner_id: str  # 所有者节点ID
    ttl: int = 3600  # 生存时间（秒）
    timestamp: float = field(default_factory=time.time)
    version: int = 1
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        return (time.time() - self.timestamp) > self.ttl
    
    def to_dict(self) -> dict:
        """转换为字典用于网络传输"""
        return {
            'key': self.key,
            'value': self.value,
            'owner_id': self.owner_id,
            'ttl': self.ttl,
            'timestamp': self.timestamp,
            'version': self.version
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DHTEntry':
        """从字典创建"""
        return cls(**data)


# ============== DHT实现 ==============

class DHTNode:
    """
    分布式哈希表节点
    使用Kademlia风格的算法进行节点发现和存储
    
    Kademlia使用XOR距离来衡量节点之间的"距离"，
    这类似于地理距离，但在逻辑空间中
    """
    
    K_BUCKET_SIZE = 20  # 每个K桶的节点数
    REPLICATION_FACTOR = 3  # 复制因子
    REFRESH_INTERVAL = 3600  # 刷新间隔（秒）
    
    def __init__(self, node_id: str, address: str, port: int):
        self.node_id = node_id
        self.address = address
        self.port = port
        
        # K桶路由表：按节点ID的二进制前缀分桶
        self._routing_table: Dict[int, List[PeerNode]] = {}
        self._local_storage: Dict[str, DHTEntry] = {}  # 本地存储
        self._pending_requests: Dict[str, Any] = {}  # 待处理的请求
        
        self._lock = threading.RLock()
        self._node_id_int = int(hashlib.sha256(node_id.encode()).hexdigest(), 16)
        
    def _node_distance(self, a_id: str, b_id: str) -> int:
        """
        计算两个节点ID之间的XOR距离
        XOR距离越小，两个节点越"接近"
        """
        a_int = int(hashlib.sha256(a_id.encode()).hexdigest(), 16)
        b_int = int(hashlib.sha256(b_id.encode()).hexdigest(), 16)
        return a_int ^ b_int
    
    def _key_distance(self, key: str) -> int:
        """计算节点ID与键的XOR距离"""
        key_int = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        return self._node_id_int ^ key_int
    
    def _get_bucket_index(self, node_id: str) -> int:
        """获取节点所属的K桶索引"""
        distance = self._node_distance(self.node_id, node_id)
        if distance == 0:
            return 0
        return distance.bit_length() - 1
    
    def _closest_nodes(self, key: str, k: int = None) -> List[PeerNode]:
        """
        找到距离键最近的K个节点
        这用于确定在哪里存储数据以及向谁查询
        """
        if k is None:
            k = self.REPLICATION_FACTOR
            
        all_nodes = []
        for bucket in self._routing_table.values():
            all_nodes.extend(bucket)
        
        # 按距离排序
        distances = [(self._node_distance(n.peer_id, key), n) for n in all_nodes]
        distances.sort(key=lambda x: x[0])
        
        return [node for _, node in distances[:k]]
    
    def put(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """
        存储键值对到最近的K个节点
        
        类似古代的"通缉令"会被分发到各个驿站
        """
        entry = DHTEntry(
            key=key,
            value=value,
            owner_id=self.node_id,
            ttl=ttl,
            timestamp=time.time(),
            version=1
        )
        
        with self._lock:
            # 本地存储
            self._local_storage[key] = entry
            
            # 找到最近的节点进行复制
            closest = self._closest_nodes(key)
            
            # 模拟网络传输延迟
            time.sleep(random.uniform(0.01, 0.05))
            
        # 返回是否成功存储
        return True
    
    def get(self, key: str) -> Optional[Any]:
        """
        从网络获取值
        首先检查本地，然后查询最近的节点
        """
        with self._lock:
            # 检查本地存储
            if key in self._local_storage:
                entry = self._local_storage[key]
                if not entry.is_expired():
                    return entry.value
                else:
                    del self._local_storage[key]
        
        # 向最近的节点查询
        closest = self._closest_nodes(key)
        for node in closest:
            # 模拟网络请求
            time.sleep(random.uniform(0.01, 0.1))
            
        return None
    
    def find_node(self, peer_id: str) -> List[PeerNode]:
        """
        Kademlia风格的节点发现
        返回距离目标ID最近的K个节点
        """
        with self._lock:
            bucket_idx = self._get_bucket_index(peer_id)
            
            # 首先检查对应桶
            if bucket_idx in self._routing_table:
                bucket = self._routing_table[bucket_idx]
                return sorted(bucket, 
                           key=lambda n: self._node_distance(n.peer_id, peer_id))[:self.K_BUCKET_SIZE]
            
            # 否则返回所有桶中最接近的节点
            return self._closest_nodes(peer_id)
    
    def add_peer(self, peer: PeerNode) -> bool:
        """
        添加对等节点到路由表
        
        类似于在驿站系统中登记新的驿站
        """
        with self._lock:
            bucket_idx = self._get_bucket_index(peer.peer_id)
            
            if bucket_idx not in self._routing_table:
                self._routing_table[bucket_idx] = []
            
            bucket = self._routing_table[bucket_idx]
            
            # 检查是否已存在
            for i, existing in enumerate(bucket):
                if existing.peer_id == peer.peer_id:
                    # 更新现有节点
                    bucket[i] = peer
                    return False
            
            # 添加到桶中
            if len(bucket) < self.K_BUCKET_SIZE:
                bucket.append(peer)
            else:
                # 移除最旧的节点
                bucket.sort(key=lambda n: n.last_seen)
                bucket.pop(0)
                bucket.append(peer)
            
            return True
    
    def remove_peer(self, peer_id: str) -> bool:
        """从路由表中移除节点"""
        with self._lock:
            for bucket_idx, bucket in self._routing_table.items():
                for i, peer in enumerate(bucket):
                    if peer.peer_id == peer_id:
                        bucket.pop(i)
                        return True
        return False
    
    def _replicate(self, key: str, value: Any) -> None:
        """
        将数据复制到多个节点以提高容错性
        
        类似古代重要文书会在多处存档
        """
        closest = self._closest_nodes(key, k=self.REPLICATION_FACTOR)
        for node in closest:
            # 模拟异步复制
            time.sleep(random.uniform(0.01, 0.05))
    
    def _refresh_stale(self) -> int:
        """
        刷新过期的条目
        移除已过期的DHT条目
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._local_storage.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self._local_storage[key]
            return len(expired_keys)
    
    def get_stored_entries(self) -> List[DHTEntry]:
        """获取所有本地存储的条目"""
        with self._lock:
            return [
                entry for entry in self._local_storage.values()
                if not entry.is_expired()
            ]
    
    def get_bucket_count(self) -> int:
        """获取路由桶数量"""
        return len(self._routing_table)
    
    def get_total_peers(self) -> int:
        """获取路由表中的总节点数"""
        with self._lock:
            return sum(len(bucket) for bucket in self._routing_table.values())


# ============== Gossip传播协议 ==============

class GossipProtocol:
    """
    Gossip传播协议
    类似"流言蜚语"的传播方式，消息通过随机选择邻居进行传播
    
    在分布式系统中，Gossip协议用于：
    - 信息传播（如节点状态、模型更新）
    - 状态同步
    - 领导者选举
    """
    
    MAX_HISTORY = 1000  # 历史记录最大长度
    DEFAULT_TTL = 300   # 默认生存时间（秒）
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self._peers: Dict[str, PeerNode] = {}
        self._history: deque = deque(maxlen=self.MAX_HISTORY)
        self._subscriptions: Dict[str, List[Callable]] = {}
        self._pending_messages: Dict[str, Any] = {}
        
        self._lock = threading.RLock()
        self._gossip_counter: Dict[str, int] = {}  # 消息计数器
        
    def add_peer(self, peer: PeerNode) -> None:
        """添加对等节点"""
        with self._lock:
            self._peers[peer.peer_id] = peer
    
    def remove_peer(self, peer_id: str) -> None:
        """移除对等节点"""
        with self._lock:
            if peer_id in self._peers:
                del self._peers[peer_id]
    
    def get_peers(self) -> List[PeerNode]:
        """获取所有对等节点"""
        with self._lock:
            return list(self._peers.values())
    
    def get_peers_by_region(self, region: Region) -> List[PeerNode]:
        """按区域获取节点"""
        with self._lock:
            return [
                peer for peer in self._peers.values()
                if peer.region == region and peer.is_online
            ]
    
    def get_online_peers(self) -> List[PeerNode]:
        """获取所有在线节点"""
        with self._lock:
            return [
                peer for peer in self._peers.values()
                if peer.is_online
            ]
    
    def _generate_message_id(self, topic: str, content: Any) -> str:
        """生成唯一的消息ID"""
        data = f"{self.node_id}:{topic}:{time.time()}:{json.dumps(content, sort_keys=True)}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def publish(self, topic: str, message: Any, ttl: int = None) -> str:
        """
        发布消息到主题
        Gossip协议会将消息传播给所有订阅者
        
        类似于在领地上张贴告示
        """
        if ttl is None:
            ttl = self.DEFAULT_TTL
            
        message_id = self._generate_message_id(topic, message)
        
        gossip_msg = {
            'id': message_id,
            'topic': topic,
            'content': message,
            'origin': self.node_id,
            'ttl': ttl,
            'timestamp': time.time(),
            'depth': 0
        }
        
        with self._lock:
            self._pending_messages[message_id] = gossip_msg
            self._history.append(message_id)
            self._gossip_counter[message_id] = 0
        
        # 触发本地订阅者
        self._deliver_to_subscribers(topic, message)
        
        # 开始Gossip传播
        self._spread(gossip_msg, depth=0, max_depth=3)
        
        return message_id
    
    def subscribe(self, topic: str, callback: Callable) -> None:
        """
        订阅主题
        当收到该主题的消息时调用回调函数
        """
        with self._lock:
            if topic not in self._subscriptions:
                self._subscriptions[topic] = []
            self._subscriptions[topic].append(callback)
    
    def unsubscribe(self, topic: str, callback: Callable) -> None:
        """取消订阅"""
        with self._lock:
            if topic in self._subscriptions:
                try:
                    self._subscriptions[topic].remove(callback)
                except ValueError:
                    pass
    
    def _deliver_to_subscribers(self, topic: str, message: Any) -> None:
        """将消息投递给订阅者"""
        with self._lock:
            callbacks = self._subscriptions.get(topic, [])
        
        for callback in callbacks:
            try:
                callback(message)
            except Exception as e:
                print(f"Error in subscriber callback: {e}")
    
    def _spread(self, message: dict, depth: int = 0, max_depth: int = 3) -> None:
        """
        递归传播消息
        
        每次传播随机选择部分邻居节点，深度递减
        这模拟了Gossip协议的随机传播特性
        """
        if depth >= max_depth:
            return
            
        # 模拟网络延迟
        time.sleep(random.uniform(0.01, 0.1))
        
        with self._lock:
            online_peers = [
                peer for peer in self._peers.values()
                if peer.is_online and peer.peer_id != message['origin']
            ]
            
            # 随机选择一部分邻居进行传播（类似Gossip的随机选择）
            fanout = max(1, min(3, len(online_peers) // 2))
            selected = random.sample(online_peers, min(fanout, len(online_peers)))
        
        for peer in selected:
            # 更新计数器
            with self._lock:
                self._gossip_counter[message['id']] = self._gossip_counter.get(message['id'], 0) + 1
            
            # 模拟向邻居发送消息
            propagated_msg = message.copy()
            propagated_msg['depth'] = depth + 1
            
            # 递归传播
            self._spread(propagated_msg, depth + 1, max_depth)
    
    def _elect_leader(self, candidates: List[str]) -> str:
        """
        简单领导者选举
        
        在去中心化系统中，需要选举一个领导者来协调操作
        这里使用基于ID的简单选举
        """
        if not candidates:
            return self.node_id
            
        # 按信任等级和ID排序
        with self._lock:
            def get_priority(peer_id: str) -> Tuple[int, str]:
                if peer_id in self._peers:
                    peer = self._peers[peer_id]
                    return (peer.trust_level.value, peer_id)
                return (0, peer_id)
            
            candidates.sort(key=get_priority, reverse=True)
        
        return candidates[0]
    
    def get_statistics(self) -> dict:
        """获取Gossip协议统计信息"""
        with self._lock:
            return {
                'total_peers': len(self._peers),
                'online_peers': len([p for p in self._peers.values() if p.is_online]),
                'history_size': len(self._history),
                'total_messages': len(self._pending_messages),
                'subscriptions': len(self._subscriptions)
            }


# ============== P2P服务器 ==============

class P2PServer:
    """
    P2P服务器 - 分封制的"领主城堡"
    
    负责：
    - 节点间的通信
    - 数据同步
    - 消息路由
    - 心跳检测
    - NAT穿透（模拟）
    """
    
    HEARTBEAT_INTERVAL = 30  # 心跳间隔（秒）
    SYNC_INTERVAL = 60       # 同步间隔（秒）
    NAT_TIMEOUT = 10         # NAT穿透超时（秒）
    
    def __init__(self, node_id: str, host: str = "0.0.0.0", port: int = 8000):
        self._node_id = node_id
        self._host = host
        self._port = port
        
        # 初始化DHT和Gossip组件
        self._node = DHTNode(node_id, host, port)
        self._gossip = GossipProtocol(node_id)
        
        # 邻居节点
        self._neighbors: Dict[str, PeerNode] = {}
        
        # 本地数据存储
        self._data_store: Dict[str, Any] = {}
        
        # 同步状态
        self._sync_state: Dict[str, Any] = {
            'last_sync': 0,
            'pending_syncs': [],
            'sync_version': 0
        }
        
        # 模型更新缓存
        self._model_updates: Dict[str, Any] = {}
        
        # 运行状态
        self._running = False
        self._threads: List[threading.Thread] = []
        self._lock = threading.RLock()
        
    def bootstrap(self, bootstrap_nodes: List[Tuple[str, int]]) -> bool:
        """
        加入P2P网络
        
        通过连接引导节点来发现网络中的其他节点
        类似新领主前往王都觐见以加入分封体系
        """
        connected = False
        
        for host, port in bootstrap_nodes:
            try:
                # 模拟连接到引导节点
                peer = PeerNode(
                    peer_id=f"bootstrap_{host}:{port}",
                    address=host,
                    port=port,
                    public_key=None,
                    capabilities=['bootstrap', 'relay'],
                    trust_level=TrustLevel.TRUSTED,
                    region=Region.CAPITAL
                )
                
                # 添加到路由表
                self._node.add_peer(peer)
                self._neighbors[peer.peer_id] = peer
                
                # 发现更多节点
                self._discover_nodes(peer)
                
                connected = True
                
            except Exception as e:
                print(f"Failed to connect to bootstrap node {host}:{port}: {e}")
        
        return connected
    
    def _discover_nodes(self, peer: PeerNode) -> List[PeerNode]:
        """
        从给定节点发现更多节点
        
        类似于从已知的驿站打听其他驿站的位置
        """
        discovered = []
        
        # 模拟节点发现
        # 在真实实现中，这里会发送FIND_NODE请求
        time.sleep(random.uniform(0.1, 0.3))
        
        # 假设发现了一些节点
        for i in range(random.randint(1, 5)):
            new_peer = PeerNode(
                peer_id=f"discovered_{peer.peer_id}_{i}",
                address=f"192.168.{random.randint(1,255)}.{random.randint(1,255)}",
                port=random.randint(8000, 9000),
                trust_level=TrustLevel.NEWCOMER,
                region=random.choice(list(Region))
            )
            
            self._node.add_peer(new_peer)
            discovered.append(new_peer)
        
        return discovered
    
    def announce_capabilities(self) -> None:
        """
        向邻居宣告自身能力
        
        类似领主宣布自己的军事力量和资源
        """
        announcement = {
            'type': 'capability_announce',
            'node_id': self._node_id,
            'capabilities': [
                'model_training',
                'data_sharing',
                'relay',
                'storage'
            ],
            'timestamp': time.time(),
            'port': self._port
        }
        
        # 通过Gossip传播
        self._gossip.publish('capabilities', announcement)
        
        # 直接通知邻居
        with self._lock:
            for neighbor_id, neighbor in self._neighbors.items():
                if neighbor.is_online:
                    time.sleep(random.uniform(0.01, 0.05))
    
    def request_sync(self, peer_id: str, data_keys: List[str]) -> Dict[str, Any]:
        """
        请求从对等节点同步数据
        
        类似臣民向领主请求分配资源
        """
        sync_request = {
            'type': 'sync_request',
            'requestor': self._node_id,
            'keys': data_keys,
            'timestamp': time.time()
        }
        
        with self._lock:
            if peer_id not in self._neighbors:
                return {'error': 'Peer not found'}
        
        # 模拟网络延迟
        time.sleep(random.uniform(0.1, 0.3))
        
        # 返回同步数据
        with self._lock:
            return {
                'peer_id': peer_id,
                'data': {k: self._data_store.get(k) for k in data_keys if k in self._data_store},
                'version': self._sync_state['sync_version']
            }
    
    def respond_sync(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        响应同步请求
        
        类似领主向臣民分发资源
        """
        requestor = request.get('requestor')
        keys = request.get('keys', [])
        
        # 检查权限
        with self._lock:
            if requestor not in self._neighbors:
                return {'error': 'Unauthorized'}
            
            response_data = {}
            for key in keys:
                if key in self._data_store:
                    response_data[key] = self._data_store[key]
        
        return {
            'type': 'sync_response',
            'data': response_data,
            'version': self._sync_state['sync_version']
        }
    
    def store_data(self, key: str, value: Any) -> bool:
        """
        存储数据
        
        类似在城堡的仓库中储存物资
        """
        with self._lock:
            self._data_store[key] = value
            self._sync_state['sync_version'] += 1
        
        # 同时存入DHT以提高可用性
        self._node.put(key, value)
        
        return True
    
    def retrieve_data(self, key: str) -> Optional[Any]:
        """检索数据"""
        with self._lock:
            if key in self._data_store:
                return self._data_store[key]
        
        # 尝试从DHT获取
        return self._node.get(key)
    
    def broadcast_model_update(self, model_id: str, update: Any) -> str:
        """
        广播模型更新
        
        类似领主向所有臣民发布新法令
        """
        broadcast_msg = {
            'type': 'model_update',
            'model_id': model_id,
            'update': update,
            'origin': self._node_id,
            'timestamp': time.time(),
            'version': len(self._model_updates.get(model_id, []))
        }
        
        with self._lock:
            if model_id not in self._model_updates:
                self._model_updates[model_id] = []
            self._model_updates[model_id].append(broadcast_msg)
        
        # 通过Gossip传播
        message_id = self._gossip.publish('model_updates', broadcast_msg)
        
        return message_id
    
    def get_model_updates(self, model_id: str) -> List[Any]:
        """获取特定模型的更新历史"""
        with self._lock:
            return self._model_updates.get(model_id, []).copy()
    
    def _route_message(self, msg: dict, dest_id: str) -> bool:
        """
        路由消息到目标节点
        
        使用类似源路由的方式找到路径
        """
        if dest_id == self._node_id:
            return True
        
        with self._lock:
            neighbors = list(self._neighbors.values())
        
        # 简单的随机路由
        if neighbors:
            next_hop = random.choice(neighbors)
            time.sleep(random.uniform(0.01, 0.05))
            return True
        
        return False
    
    def _heartbeat_loop(self) -> None:
        """
        心跳循环
        
        定期向邻居发送心跳以维持连接
        类似古代的烽火台传递信号
        """
        while self._running:
            time.sleep(self.HEARTBEAT_INTERVAL)
            
            with self._lock:
                for neighbor_id, neighbor in list(self._neighbors.items()):
                    # 检查是否超时
                    if neighbor.is_stale(timeout=self.HEARTBEAT_INTERVAL * 3):
                        neighbor.is_online = False
                        print(f"Peer {neighbor_id} marked as offline")
                    
                    # 发送心跳
                    time.sleep(random.uniform(0.01, 0.05))
    
    def _resolve_nat(self) -> bool:
        """
        NAT穿透尝试
        
        模拟UDP打洞技术
        在真实的P2P网络中，需要NAT穿透才能直接通信
        """
        # 模拟NAT穿透过程
        time.sleep(random.uniform(0.5, 2.0))
        
        # 随机决定是否成功
        success = random.random() > 0.3
        
        if success:
            print(f"NAT traversal successful for {self._node_id}")
        else:
            print(f"NAT traversal failed for {self._node_id}, using relay")
        
        return success
    
    def _sync_loop(self) -> None:
        """
        同步循环
        
        定期与邻居同步数据
        """
        while self._running:
            time.sleep(self.SYNC_INTERVAL)
            
            with self._lock:
                neighbors = [n for n in self._neighbors.values() if n.is_online]
            
            for neighbor in neighbors:
                try:
                    # 请求同步
                    keys = list(self._data_store.keys())[:10]
                    response = self.request_sync(neighbor.peer_id, keys)
                    
                    if 'data' in response:
                        with self._lock:
                            for key, value in response['data'].items():
                                if key not in self._data_store:
                                    self._data_store[key] = value
                                    
                except Exception as e:
                    print(f"Sync error with {neighbor.peer_id}: {e}")
    
    def start(self) -> None:
        """启动P2P服务器"""
        self._running = True
        
        # 启动心跳线程
        heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        heartbeat_thread.start()
        self._threads.append(heartbeat_thread)
        
        # 启动同步线程
        sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        sync_thread.start()
        self._threads.append(sync_thread)
        
        print(f"P2P Server started on {self._host}:{self._port}")
    
    def stop(self) -> None:
        """停止P2P服务器"""
        self._running = False
        
        for thread in self._threads:
            thread.join(timeout=1.0)
        
        self._threads.clear()
        print(f"P2P Server stopped")
    
    def get_status(self) -> dict:
        """获取服务器状态"""
        with self._lock:
            return {
                'node_id': self._node_id,
                'address': f"{self._host}:{self._port}",
                'neighbors': len(self._neighbors),
                'online_neighbors': len([n for n in self._neighbors.values() if n.is_online]),
                'data_stored': len(self._data_store),
                'routing_table_size': self._node.get_total_peers(),
                'gossip_stats': self._gossip.get_statistics()
            }


# ============== 分封制节点 ==============

class FeudalNode:
    """
    分封制节点 - 模拟"封建领主"
    
    在联邦学习中，模拟分封制结构：
    - 大领主（参数服务器/协调者）
    - 中领主（区域协调者）
    - 小领主/臣民（数据持有者）
    
    节点可以：
    - 宣誓效忠/接受臣服
    - 收集/进贡资源
    - 裁决纠纷
    - 宣战/议和
    """
    
    WAR_DECLARATION_COOLDOWN = 3600  # 宣战冷却时间（秒）
    
    def __init__(self, node_id: str, fief_size: int = 100):
        self._node_id = node_id
        self._fief_size = fief_size
        
        # 分封关系
        self._vassals: List[str] = []  # 臣属节点
        self._liege: Optional[str] = None  # 主君
        
        # 封地数据
        self._fief: Dict[str, Any] = {
            'resources': fief_size,
            'troops': fief_size * 10,  # 每单位封地10单位兵力
            'treasury': 0,
            'land_quality': random.uniform(0.5, 1.0)
        }
        
        # 朝贡记录
        self._tribute: Dict[str, List[Dict]] = {
            'received': [],  # 收到的贡品
            'paid': []       # 支付的贡品
        }
        
        # 关系
        self._alliances: Set[str] = set()  # 盟友
        self._enemies: Set[str] = set()    # 敌人
        self._wars: Dict[str, float] = {}  # 战争（敌人ID -> 开始时间）
        
        # 纠纷记录
        self._disputes: List[Dict] = []
        
        self._lock = threading.RLock()
        self._last_war_time: Dict[str, float] = {}
        
    def pledge_to_lord(self, lord_id: str) -> bool:
        """
        宣誓效忠
        
        臣民节点向领主节点宣誓效忠
        """
        with self._lock:
            # 检查是否已有主君
            if self._liege is not None:
                return False
            
            self._liege = lord_id
            
            # 记录效忠时间
            self._tribute['paid'].append({
                'lord_id': lord_id,
                'type': 'pledge',
                'time': time.time()
            })
        
        return True
    
    def accept_vassal(self, vassal_id: str) -> bool:
        """
        接受臣服
        
        领主节点接受新臣民的效忠
        """
        with self._lock:
            if vassal_id in self._vassals:
                return False
            
            self._vassals.append(vassal_id)
            
            # 记录臣民信息
            self._tribute['received'].append({
                'vassal_id': vassal_id,
                'type': 'pledge',
                'time': time.time()
            })
        
        return True
    
    def _collect_tribute(self) -> Dict[str, Any]:
        """
        收集臣民的朝贡
        
        领主向所有臣民收集资源
        """
        collected = {
            'resources': 0,
            'troops': 0,
            'count': 0
        }
        
        with self._lock:
            for vassal_id in self._vassals:
                # 模拟从臣民处收集贡品
                tribute_amount = self._fief_size * 0.1 * random.uniform(0.8, 1.2)
                collected['resources'] += tribute_amount
                collected['troops'] += int(tribute_amount * 10)
                collected['count'] += 1
                
                self._tribute['received'].append({
                    'vassal_id': vassal_id,
                    'type': 'tribute',
                    'amount': tribute_amount,
                    'time': time.time()
                })
        
        # 更新封地资源
        with self._lock:
            self._fief['resources'] += collected['resources']
            self._fief['treasury'] += collected['resources'] * 0.5
            self._fief['troops'] += collected['troops']
        
        return collected
    
    def _pay_tribute(self, lord_id: str, amount: float) -> bool:
        """
        向领主进贡
        
        臣民向主君支付资源
        """
        with self._lock:
            if self._fief['resources'] < amount:
                return False
            
            self._fief['resources'] -= amount
            self._fief['treasury'] -= amount * 0.8
            
            self._tribute['paid'].append({
                'lord_id': lord_id,
                'type': 'tribute',
                'amount': amount,
                'time': time.time()
            })
        
        return True
    
    def _resolve_disputes(self, other_node: 'FeudalNode') -> Dict[str, Any]:
        """
        裁决纠纷
        
        当两个臣民之间发生纠纷时，由领主裁决
        """
        with self._lock:
            # 只有领主才能裁决纠纷
            if self._node_id != self._liege and self._node_id not in self._vassals:
                return {'success': False, 'reason': 'Not authorized to arbitrate'}
            
            dispute = {
                'arbitrator': self._node_id,
                'parties': [self._node_id, other_node._node_id],
                'time': time.time(),
                'settlement': None
            }
            
            # 模拟裁决过程（简单加权随机）
            party_a_score = self._fief_size
            party_b_score = other_node._fief_size
            total = party_a_score + party_b_score
            
            winner = self._node_id if random.random() < party_a_score / total else other_node._node_id
            
            dispute['settlement'] = {
                'winner': winner,
                'compensation': random.uniform(0.1, 0.3) * party_b_score
            }
            
            self._disputes.append(dispute)
        
        return dispute
    
    def get_troops(self) -> int:
        """
        获取可用兵力
        
        包括自身封地的兵力和臣民贡献的兵力
        """
        with self._lock:
            base_troops = self._fief['troops']
            
            # 加上臣民的贡献
            vassal_contribution = len(self._vassals) * 50
            
            # 减去战争消耗
            war_penalty = len(self._wars) * 20
            
            return max(0, base_troops + vassal_contribution - war_penalty)
    
    def declare_war(self, enemy_id: str) -> bool:
        """
        宣战
        
        拒绝与敌人合作
        """
        with self._lock:
            # 检查冷却时间
            if enemy_id in self._last_war_time:
                time_since_last = time.time() - self._last_war_time[enemy_id]
                if time_since_last < self.WAR_DECLARATION_COOLDOWN:
                    return False
            
            self._enemies.add(enemy_id)
            self._wars[enemy_id] = time.time()
            self._last_war_time[enemy_id] = time.time()
            
            # 移除可能的同盟
            if enemy_id in self._alliances:
                self._alliances.remove(enemy_id)
        
        return True
    
    def make_peace(self, enemy_id: str) -> bool:
        """
        议和
        
        结束与敌人的战争状态
        """
        with self._lock:
            if enemy_id not in self._enemies:
                return False
            
            self._enemies.remove(enemy_id)
            
            if enemy_id in self._wars:
                del self._wars[enemy_id]
        
        return True
    
    def form_alliance(self, ally_id: str) -> bool:
        """
        结盟
        
        与另一个节点建立同盟关系
        """
        with self._lock:
            if ally_id in self._enemies:
                return False
            
            self._alliances.add(ally_id)
        
        return True
    
    def break_alliance(self, ally_id: str) -> bool:
        """
        解除同盟
        """
        with self._lock:
            if ally_id in self._alliances:
                self._alliances.remove(ally_id)
                return True
        return False
    
    def get_power_level(self) -> float:
        """
        计算节点的实力等级
        
        综合考虑封地大小、臣民数量、兵力等
        """
        with self._lock:
            return (
                self._fief['resources'] * 1.0 +
                len(self._vassals) * 50 * 0.5 +
                self._fief['troops'] * 0.1 +
                len(self._alliances) * 30 * 0.3 -
                len(self._enemies) * 20
            )
    
    def get_status(self) -> dict:
        """获取封建节点状态"""
        with self._lock:
            return {
                'node_id': self._node_id,
                'liege': self._liege,
                'vassals': self._vassals.copy(),
                'vassal_count': len(self._vassals),
                'fief': self._fief.copy(),
                'allies': list(self._alliances),
                'enemies': list(self._enemies),
                'active_wars': len(self._wars),
                'power_level': self.get_power_level(),
                'troops': self.get_troops()
            }
    
    def get_hierarchy_info(self) -> dict:
        """获取封建等级信息"""
        with self._lock:
            return {
                'node_id': self._node_id,
                'rank': self._get_rank(),
                'liege': self._liege,
                'vassals': self._vassals,
                'total_subjects': self._count_all_subjects()
            }
    
    def _get_rank(self) -> str:
        """根据臣民数量确定等级"""
        with self._lock:
            vassal_count = len(self._vassals)
            
            if self._liege is None:
                return "Emperor/King" if vassal_count > 5 else "Grand Duke"
            elif vassal_count > 10:
                return "Duke"
            elif vassal_count > 5:
                return "Count"
            elif vassal_count > 0:
                return "Viscount"
            else:
                return "Baron/Knight"
    
    def _count_all_subjects(self) -> int:
        """计算所有臣属（递归）"""
        with self._lock:
            count = len(self._vassals)
            return count  # 简化版本，实际应该递归计算


# ============== 工厂函数 ==============

def create_feudal_network(
    num_lords: int = 3,
    vassals_per_lord: int = 5
) -> Dict[str, FeudalNode]:
    """
    创建分封制网络
    
    工厂函数，用于创建完整的封建网络拓扑
    """
    nodes = {}
    
    # 创建大领主
    for i in range(num_lords):
        lord_id = f"lord_{i}"
        nodes[lord_id] = FeudalNode(lord_id, fief_size=1000)
    
    # 为每个大领主创建臣属
    for lord_id, lord in list(nodes.items()):
        for j in range(vassals_per_lord):
            vassal_id = f"vassal_{lord_id}_{j}"
            vassal = FeudalNode(vassal_id, fief_size=random.randint(50, 200))
            
            # 建立效忠关系
            vassal.pledge_to_lord(lord_id)
            lord.accept_vassal(vassal_id)
            
            nodes[vassal_id] = vassal
    
    return nodes


# ============== 主程序入口 ==============

if __name__ == "__main__":
    # 演示P2P和分封制系统
    
    print("=== 初始化P2P网络 ===")
    
    # 创建引导节点
    bootstrap = P2PServer("bootstrap_1", "localhost", 8000)
    bootstrap.start()
    
    # 创建普通节点
    node1 = P2PServer("node_1", "localhost", 8001)
    node2 = P2PServer("node_2", "localhost", 8002)
    
    # 引导节点加入网络
    node1.bootstrap([("localhost", 8000)])
    node2.bootstrap([("localhost", 8000)])
    
    # 测试数据存储和广播
    node1.store_data("model_v1", {"weights": [1.0, 2.0, 3.0]})
    update_id = node1.broadcast_model_update("model_1", {"delta": [0.1, 0.2, 0.3]})
    
    print(f"广播消息ID: {update_id}")
    print(f"节点1状态: {node1.get_status()}")
    
    print("\n=== 初始化分封制网络 ===")
    
    # 创建封建网络
    feudal_network = create_feudal_network(num_lords=2, vassals_per_lord=3)
    
    # 查看网络结构
    for node_id, node in feudal_network.items():
        status = node.get_status()
        print(f"{status['rank']} {node_id}: {status['vassal_count']} 臣民, 兵力 {status['troops']}")
    
    # 测试宣战和议和
    lord_0 = feudal_network["lord_0"]
    lord_1 = feudal_network["lord_1"]
    
    lord_0.declare_war("lord_1")
    print(f"\nLord_0 宣战 Lord_1: {lord_0.get_status()['enemies']}")
    
    lord_0.make_peace("lord_1")
    print(f"Lord_0 与 Lord_1 议和: {lord_0.get_status()['enemies']}")
    
    # 测试朝贡
    lord_0._collect_tribute()
    print(f"\nLord_0 收集朝贡后: {lord_0.get_status()['fief']['resources']} 资源")
    
    # 停止服务
    bootstrap.stop()
    
    print("\n=== 演示完成 ===")
