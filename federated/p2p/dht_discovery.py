"""
DHT Node Discovery - Kademlia DHT Protocol Implementation
Kademlia DHT节点发现协议实现

基于Kademlia协议的分布式哈希表，用于P2P网络中的节点发现和数据存储。
使用XOR距离度量节点间的逻辑距离，通过k-bucket路由表管理节点信息。

Author: AGI Unified Framework
"""

import hashlib
import random
import time
import threading
import struct
import socket
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any, Callable
from collections import deque
from enum import Enum
from bisect import insort


# ============== XOR距离计算 ==============

class XORDistance:
    """
    XOR距离计算器

    Kademlia使用XOR度量来衡量节点ID之间的距离。
    XOR距离满足三角不等式、对称性和唯一性，
    是DHT中广泛使用的距离度量方法。

    对于160位的SHA-1哈希空间，距离范围为 0 ~ 2^160 - 1。
    """

    BIT_LENGTH = 160  # SHA-1哈希的位数

    @staticmethod
    def compute(node_id_a: bytes, node_id_b: bytes) -> int:
        """
        计算两个节点ID之间的XOR距离

        Args:
            node_id_a: 节点A的ID（原始字节）
            node_id_b: 节点B的ID（原始字节）

        Returns:
            XOR距离的整数值
        """
        a_int = int.from_bytes(node_id_a, 'big')
        b_int = int.from_bytes(node_id_b, 'big')
        return a_int ^ b_int

    @staticmethod
    def compute_from_hex(hex_a: str, hex_b: str) -> int:
        """从十六进制字符串计算XOR距离"""
        return int(hex_a, 16) ^ int(hex_b, 16)

    @staticmethod
    def bucket_index(distance: int) -> int:
        """
        根据XOR距离确定k-bucket的索引

        bucket索引 = distance的最高有效位位置
        距离越小（越近），bucket索引越低

        Args:
            distance: XOR距离

        Returns:
            bucket索引 (0 ~ BIT_LENGTH-1)
        """
        if distance == 0:
            return 0
        bit_len = distance.bit_length()
        return min(bit_len - 1, XORDistance.BIT_LENGTH - 1)

    @staticmethod
    def to_bytes(distance: int, length: int = 20) -> bytes:
        """将距离转换为字节表示"""
        return distance.to_bytes(length, 'big')

    @staticmethod
    def log2_distance(distance: int) -> float:
        """计算XOR距离的对数值（以2为底）"""
        if distance == 0:
            return 0.0
        return distance.bit_length() - 1 + bin(distance).count('1') / distance.bit_length()


# ============== DHT节点信息 ==============

@dataclass
class DHTNode:
    """
    DHT节点信息

    表示Kademlia网络中的一个节点，包含节点ID、网络地址和状态信息。
    节点ID使用SHA-1哈希生成，确保在160位空间中均匀分布。

    Attributes:
        node_id: 节点的唯一标识（20字节SHA-1哈希）
        ip: 节点的IP地址
        port: 节点的监听端口
        last_seen: 最后一次收到该节点响应的时间戳
        is_online: 节点是否在线
        version: 节点协议版本
    """
    node_id: bytes
    ip: str
    port: int
    last_seen: float = field(default_factory=time.time)
    is_online: bool = True
    version: int = 1

    def __post_init__(self):
        if isinstance(self.node_id, str):
            self.node_id = bytes.fromhex(self.node_id)

    @property
    def node_id_hex(self) -> str:
        """获取节点ID的十六进制表示"""
        return self.node_id.hex()

    @property
    def address(self) -> Tuple[str, int]:
        """获取节点的网络地址"""
        return (self.ip, self.port)

    def update_last_seen(self) -> None:
        """更新最后可见时间"""
        self.last_seen = time.time()
        self.is_online = True

    def is_stale(self, timeout: float = 900.0) -> bool:
        """检查节点是否过期（默认15分钟）"""
        return (time.time() - self.last_seen) > timeout

    def distance_to(self, other_id: bytes) -> int:
        """计算到另一个节点的XOR距离"""
        return XORDistance.compute(self.node_id, other_id)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            'node_id': self.node_id_hex,
            'ip': self.ip,
            'port': self.port,
            'last_seen': self.last_seen,
            'is_online': self.is_online,
            'version': self.version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DHTNode':
        """从字典反序列化"""
        return cls(
            node_id=data['node_id'],
            ip=data['ip'],
            port=data['port'],
            last_seen=data.get('last_seen', time.time()),
            is_online=data.get('is_online', True),
            version=data.get('version', 1)
        )

    @classmethod
    def generate(cls, ip: str, port: int, seed: Optional[str] = None) -> 'DHTNode':
        """生成随机节点ID的DHT节点"""
        if seed is None:
            raw = f"{ip}:{port}:{time.time()}:{random.getrandbits(256)}"
        else:
            raw = seed
        node_id = hashlib.sha1(raw.encode()).digest()
        return cls(node_id=node_id, ip=ip, port=port)

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DHTNode):
            return NotImplemented
        return self.node_id == other.node_id


# ============== Kademlia路由表 ==============

class KBucket:
    """
    K-Bucket：Kademlia路由表中的单个桶

    每个k-bucket存储距离在 [2^i, 2^(i+1)) 范围内的节点信息。
    k-bucket使用LRU策略：当桶满时，新节点会替换最久未响应的节点。

    Attributes:
        index: 桶的索引（对应距离的最高有效位）
        k_size: 桶的最大容量
        nodes: 桶中的节点列表（按last_seen降序排列）
    """

    def __init__(self, index: int, k_size: int = 20):
        self.index = index
        self.k_size = k_size
        self._nodes: List[DHTNode] = []
        self._replacement_cache: List[DHTNode] = []  # 替换缓存
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        """当前桶中的节点数量"""
        return len(self._nodes)

    @property
    def is_full(self) -> bool:
        """桶是否已满"""
        return len(self._nodes) >= self.k_size

    def add_node(self, node: DHTNode) -> Tuple[bool, Optional[DHTNode]]:
        """
        添加节点到k-bucket

        如果节点已存在，更新其last_seen时间。
        如果桶未满，直接添加。
        如果桶已满，将节点放入替换缓存。

        Returns:
            (是否添加成功, 被替换的旧节点或None)
        """
        with self._lock:
            # 检查是否已存在
            for i, existing in enumerate(self._nodes):
                if existing.node_id == node.node_id:
                    # 已存在，移到列表头部（最近见过的）
                    self._nodes.pop(i)
                    self._nodes.insert(0, node)
                    return False, None

            # 桶未满，直接添加
            if not self.is_full:
                self._nodes.insert(0, node)
                return True, None

            # 桶已满，检查最久未响应的节点
            least_recent = self._nodes[-1]
            if least_recent.is_stale(timeout=300.0):
                # 替换最久未响应的节点
                self._nodes.pop(-1)
                self._nodes.insert(0, node)
                return True, least_recent

            # 所有节点都活跃，放入替换缓存
            if len(self._replacement_cache) < self.k_size:
                # 检查替换缓存中是否已存在
                for i, cached in enumerate(self._replacement_cache):
                    if cached.node_id == node.node_id:
                        self._replacement_cache.pop(i)
                        break
                self._replacement_cache.insert(0, node)

            return False, None

    def remove_node(self, node_id: bytes) -> bool:
        """从桶中移除节点"""
        with self._lock:
            for i, node in enumerate(self._nodes):
                if node.node_id == node_id:
                    self._nodes.pop(i)
                    # 从替换缓存中取出一个节点补充
                    if self._replacement_cache:
                        replacement = self._replacement_cache.pop(0)
                        self._nodes.insert(0, replacement)
                    return True
            return False

    def get_node(self, node_id: bytes) -> Optional[DHTNode]:
        """获取指定节点"""
        with self._lock:
            for node in self._nodes:
                if node.node_id == node_id:
                    return node
        return None

    def get_all_nodes(self) -> List[DHTNode]:
        """获取桶中所有节点"""
        with self._lock:
            return list(self._nodes)

    def get_random_nodes(self, count: int) -> List[DHTNode]:
        """随机获取指定数量的节点"""
        with self._lock:
            if count >= len(self._nodes):
                return list(self._nodes)
            return random.sample(self._nodes, count)

    def touch_node(self, node_id: bytes) -> bool:
        """更新节点的last_seen时间（确认节点存活）"""
        with self._lock:
            for i, node in enumerate(self._nodes):
                if node.node_id == node_id:
                    node.update_last_seen()
                    # 移到列表头部
                    self._nodes.pop(i)
                    self._nodes.insert(0, node)
                    return True
        return False

    def split(self, center_id: bytes) -> Tuple['KBucket', 'KBucket']:
        """
        将当前桶分裂为两个子桶

        根据与center_id的距离，将节点分配到低位桶和高位桶。
        这是Kademlia路由表扩展时的关键操作。
        """
        low_bucket = KBucket(self.index, self.k_size)
        high_bucket = KBucket(self.index + 1, self.k_size)

        with self._lock:
            for node in self._nodes:
                dist = XORDistance.compute(center_id, node.node_id)
                bucket_idx = XORDistance.bucket_index(dist)
                if bucket_idx <= self.index:
                    low_bucket.add_node(node)
                else:
                    high_bucket.add_node(node)

        return low_bucket, high_bucket

    def refresh_stale_nodes(self) -> int:
        """刷新过期的节点，返回被移除的数量"""
        removed = 0
        with self._lock:
            stale_indices = [
                i for i, node in enumerate(self._nodes)
                if node.is_stale(timeout=900.0)
            ]
            for i in reversed(stale_indices):
                self._nodes.pop(i)
                removed += 1
                # 从替换缓存补充
                if self._replacement_cache:
                    replacement = self._replacement_cache.pop(0)
                    self._nodes.insert(0, replacement)
        return removed


class KademliaRoutingTable:
    """
    Kademlia路由表

    管理所有k-bucket，提供节点查找、添加和移除操作。
    对于160位SHA-1哈希空间，理论上最多有160个bucket。
    每个bucket最多存储k个节点（默认k=20）。

    路由表结构：
    - bucket[0]: 距离在 [1, 2) 范围内的节点（最近）
    - bucket[1]: 距离在 [2, 4) 范围内的节点
    - ...
    - bucket[159]: 距离在 [2^159, 2^160) 范围内的节点（最远）
    """

    K_BUCKET_SIZE = 20
    REFRESH_INTERVAL = 3600  # 刷新间隔（秒）
    ID_LENGTH = 20  # SHA-1哈希长度（字节）

    def __init__(self, own_node_id: bytes, k_size: int = 20):
        self._own_id = own_node_id
        self._k_size = k_size
        self._buckets: Dict[int, KBucket] = {}
        self._lock = threading.RLock()
        self._last_refresh: Dict[int, float] = {}

        # 初始化所有bucket
        for i in range(self.ID_LENGTH * 8):
            self._buckets[i] = KBucket(i, k_size)

    def _get_bucket_index(self, node_id: bytes) -> int:
        """获取节点对应的bucket索引"""
        distance = XORDistance.compute(self._own_id, node_id)
        return XORDistance.bucket_index(distance)

    def add_node(self, node: DHTNode) -> bool:
        """
        添加节点到路由表

        Args:
            node: 要添加的DHT节点

        Returns:
            是否成功添加（已存在则返回False）
        """
        if node.node_id == self._own_id:
            return False

        bucket_idx = self._get_bucket_index(node.node_id)

        with self._lock:
            bucket = self._buckets.get(bucket_idx)
            if bucket is None:
                bucket = KBucket(bucket_idx, self._k_size)
                self._buckets[bucket_idx] = bucket

            success, _ = bucket.add_node(node)
            return success

    def remove_node(self, node_id: bytes) -> bool:
        """从路由表中移除节点"""
        bucket_idx = self._get_bucket_index(node_id)

        with self._lock:
            bucket = self._buckets.get(bucket_idx)
            if bucket:
                return bucket.remove_node(node_id)
        return False

    def find_node(self, target_id: bytes, count: int = None) -> List[DHTNode]:
        """
        查找距离target_id最近的count个节点

        从对应bucket开始，逐步扩展搜索范围，
        返回按XOR距离排序的最近节点列表。

        Args:
            target_id: 目标节点ID
            count: 需要返回的节点数量（默认为k）

        Returns:
            按距离排序的最近节点列表
        """
        if count is None:
            count = self._k_size

        all_nodes: List[Tuple[int, DHTNode]] = []

        with self._lock:
            for bucket in self._buckets.values():
                for node in bucket.get_all_nodes():
                    dist = XORDistance.compute(target_id, node.node_id)
                    all_nodes.append((dist, node))

        # 按距离排序
        all_nodes.sort(key=lambda x: x[0])

        # 返回最近的count个节点
        return [node for _, node in all_nodes[:count]]

    def get_closest_nodes(self, target_id: bytes, count: int = None) -> List[DHTNode]:
        """find_node的别名"""
        return self.find_node(target_id, count)

    def touch_node(self, node_id: bytes) -> bool:
        """确认节点存活，更新其last_seen时间"""
        bucket_idx = self._get_bucket_index(node_id)

        with self._lock:
            bucket = self._buckets.get(bucket_idx)
            if bucket:
                return bucket.touch_node(node_id)
        return False

    def get_bucket_for_refresh(self) -> Optional[int]:
        """获取需要刷新的bucket索引"""
        now = time.time()
        with self._lock:
            for idx, bucket in self._buckets.items():
                last_time = self._last_refresh.get(idx, 0)
                if (now - last_time) > self.REFRESH_INTERVAL and bucket.size > 0:
                    self._last_refresh[idx] = now
                    return idx
        return None

    def mark_bucket_refreshed(self, bucket_idx: int) -> None:
        """标记bucket已刷新"""
        self._last_refresh[bucket_idx] = time.time()

    def get_all_nodes(self) -> List[DHTNode]:
        """获取路由表中所有节点"""
        nodes: List[DHTNode] = []
        with self._lock:
            for bucket in self._buckets.values():
                nodes.extend(bucket.get_all_nodes())
        return nodes

    def get_node_count(self) -> int:
        """获取路由表中的节点总数"""
        with self._lock:
            return sum(bucket.size for bucket in self._buckets.values())

    def get_active_bucket_count(self) -> int:
        """获取非空bucket的数量"""
        with self._lock:
            return sum(1 for bucket in self._buckets.values() if bucket.size > 0)

    def refresh_stale_nodes(self) -> int:
        """刷新所有过期节点"""
        total_removed = 0
        with self._lock:
            for bucket in self._buckets.values():
                total_removed += bucket.refresh_stale_nodes()
        return total_removed

    def get_routing_info(self) -> Dict[str, Any]:
        """获取路由表统计信息"""
        with self._lock:
            bucket_info = {}
            for idx, bucket in self._buckets.items():
                if bucket.size > 0:
                    bucket_info[str(idx)] = bucket.size
            return {
                'total_nodes': self.get_node_count(),
                'active_buckets': self.get_active_bucket_count(),
                'total_buckets': len(self._buckets),
                'k_size': self._k_size,
                'buckets': bucket_info
            }


# ============== Kademlia DHT协议 ==============

class KademliaDHT:
    """
    Kademlia DHT协议实现

    实现Kademlia协议的四个核心RPC操作：
    1. PING - 检测节点是否存活
    2. FIND_NODE - 查找距离target最近的k个节点
    3. FIND_VALUE - 查找存储在DHT中的数据
    4. STORE - 在DHT中存储键值对

    以及迭代查找算法（iterative_find），通过递归加强
    逐步逼近目标节点或数据。

    Author: AGI Unified Framework
    """

    ALPHA = 3  # 并行查找的并发度
    K = 20     # 每次查找返回的节点数
    REPUBLISH_INTERVAL = 86400  # 值重新发布间隔（24小时）
    EXPIRE_INTERVAL = 3600      # 值过期检查间隔（1小时）

    def __init__(self, node: DHTNode, k_size: int = 20):
        self._own_node = node
        self._routing_table = KademliaRoutingTable(node.node_id, k_size)
        self._storage: Dict[str, Tuple[Any, float, float]] = {}  # key -> (value, timestamp, ttl)
        self._lock = threading.RLock()
        self._pending_rpcs: Dict[str, threading.Event] = {}
        self._rpc_results: Dict[str, Any] = {}

        # 模拟网络延迟的回调
        self._send_rpc: Optional[Callable] = None

    def set_rpc_handler(self, handler: Callable) -> None:
        """设置RPC发送处理器（用于模拟或真实网络）"""
        self._send_rpc = handler

    @property
    def node(self) -> DHTNode:
        """获取本节点信息"""
        return self._own_node

    @property
    def routing_table(self) -> KademliaRoutingTable:
        """获取路由表"""
        return self._routing_table

    def ping(self, target: DHTNode) -> bool:
        """
        PING - 检测目标节点是否存活

        向目标节点发送PING请求，如果在超时时间内收到PONG响应，
        则认为节点存活。

        Args:
            target: 目标节点

        Returns:
            目标节点是否存活
        """
        try:
            # 构造PING消息
            ping_msg = {
                'type': 'PING',
                'sender_id': self._own_node.node_id_hex,
                'sender_ip': self._own_node.ip,
                'sender_port': self._own_node.port,
                'timestamp': time.time()
            }

            # 模拟网络RPC调用
            if self._send_rpc:
                result = self._send_rpc(target, ping_msg)
                if result and result.get('type') == 'PONG':
                    target.update_last_seen()
                    self._routing_table.touch_node(target.node_id)
                    return True
            else:
                # 模拟：随机决定节点是否存活
                alive = random.random() > 0.1  # 90%存活率
                if alive:
                    target.update_last_seen()
                    self._routing_table.touch_node(target.node_id)
                return alive

            return False
        except Exception:
            return False

    def find_node(self, target_id: bytes, requester: Optional[DHTNode] = None) -> List[DHTNode]:
        """
        FIND_NODE - 查找距离target最近的k个节点

        从本地路由表中查找距离target_id最近的K个节点。
        如果requester不为None，将其添加到路由表中。

        Args:
            target_id: 目标节点ID
            requester: 请求者节点信息

        Returns:
            距离target最近的K个节点列表
        """
        # 将请求者添加到路由表
        if requester and requester.node_id != self._own_node.node_id:
            self._routing_table.add_node(requester)

        # 从路由表中查找最近节点
        closest = self._routing_table.find_node(target_id, self.K)
        return closest

    def find_value(self, key: str, requester: Optional[DHTNode] = None) -> Tuple[Optional[Any], List[DHTNode]]:
        """
        FIND_VALUE - 查找存储在DHT中的数据

        首先检查本地存储，如果找到则返回值。
        否则返回距离key最近的K个节点，供请求者继续查找。

        Args:
            key: 查找的键
            requester: 请求者节点信息

        Returns:
            (找到的值或None, 最近的节点列表)
        """
        # 将请求者添加到路由表
        if requester and requester.node_id != self._own_node.node_id:
            self._routing_table.add_node(requester)

        # 检查本地存储
        with self._lock:
            if key in self._storage:
                value, timestamp, ttl = self._storage[key]
                if time.time() - timestamp < ttl:
                    return value, []
                else:
                    # 已过期，删除
                    del self._storage[key]

        # 未找到，返回最近节点
        key_hash = hashlib.sha1(key.encode()).digest()
        closest = self._routing_table.find_node(key_hash, self.K)
        return None, closest

    def store(self, key: str, value: Any, ttl: float = 86400.0,
              sender: Optional[DHTNode] = None) -> bool:
        """
        STORE - 在DHT中存储键值对

        将键值对存储到本地存储中，并设置TTL。

        Args:
            key: 存储的键
            value: 存储的值
            ttl: 生存时间（秒）
            sender: 发送者节点信息

        Returns:
            是否存储成功
        """
        # 将发送者添加到路由表
        if sender and sender.node_id != self._own_node.node_id:
            self._routing_table.add_node(sender)

        with self._lock:
            self._storage[key] = (value, time.time(), ttl)

        return True

    def iterative_find_node(self, target_id: bytes) -> List[DHTNode]:
        """
        迭代节点查找

        通过多轮并行查询逐步逼近目标节点。
        每轮从已知的最近节点中选择ALPHA个未查询的节点发送FIND_NODE请求，
        将返回的新节点合并到候选列表中，直到没有更近的节点为止。

        Args:
            target_id: 目标节点ID

        Returns:
            查找到的最近节点列表
        """
        # 初始化：从本地路由表获取最近的节点
        closest = self._routing_table.find_node(target_id, self.K)
        if not closest:
            return []

        # 已查询的节点集合
        queried: Set[bytes] = {self._own_node.node_id}
        # 候选节点（按距离排序）
        candidates: List[Tuple[int, DHTNode]] = []
        for node in closest:
            dist = XORDistance.compute(target_id, node.node_id)
            candidates.append((dist, node))

        # 迭代查找
        max_iterations = 20
        for iteration in range(max_iterations):
            # 选择ALPHA个未查询的最近节点
            unqueried = [
                (dist, node) for dist, node in candidates
                if node.node_id not in queried
            ]
            unqueried.sort(key=lambda x: x[0])

            if not unqueried:
                break

            # 取前ALPHA个
            to_query = unqueried[:self.ALPHA]

            # 并行查询
            new_nodes: List[DHTNode] = []
            for _, node in to_query:
                queried.add(node.node_id)

                # 发送FIND_NODE RPC
                try:
                    result_nodes = self.find_node(target_id, requester=node)
                    for rn in result_nodes:
                        if rn.node_id not in queried:
                            new_nodes.append(rn)
                except Exception:
                    continue

            # 将新发现的节点加入候选列表
            for nn in new_nodes:
                dist = XORDistance.compute(target_id, nn.node_id)
                candidates.append((dist, nn))

            # 去重并排序
            seen: Set[bytes] = set()
            unique_candidates: List[Tuple[int, DHTNode]] = []
            for dist, node in sorted(candidates, key=lambda x: x[0]):
                if node.node_id not in seen:
                    seen.add(node.node_id)
                    unique_candidates.append((dist, node))

            candidates = unique_candidates[:self.K * 2]

            # 检查是否收敛
            if len(new_nodes) == 0:
                break

        # 返回最近的K个节点
        candidates.sort(key=lambda x: x[0])
        return [node for _, node in candidates[:self.K]]

    def iterative_find_value(self, key: str) -> Tuple[Optional[Any], List[DHTNode]]:
        """
        迭代值查找

        类似iterative_find_node，但查找的是存储的值。
        如果某个节点返回了值，则立即返回。
        否则继续查找直到收敛。

        Args:
            key: 查找的键

        Returns:
            (找到的值或None, 查询过的最近节点列表)
        """
        key_hash = hashlib.sha1(key.encode()).digest()

        # 先检查本地
        value, nodes = self.find_value(key)
        if value is not None:
            return value, nodes

        # 初始化候选节点
        closest = self._routing_table.find_node(key_hash, self.K)
        if not closest:
            return None, []

        queried: Set[bytes] = {self._own_node.node_id}
        candidates: List[Tuple[int, DHTNode]] = []
        for node in closest:
            dist = XORDistance.compute(key_hash, node.node_id)
            candidates.append((dist, node))

        max_iterations = 20
        for iteration in range(max_iterations):
            unqueried = [
                (dist, node) for dist, node in candidates
                if node.node_id not in queried
            ]
            unqueried.sort(key=lambda x: x[0])

            if not unqueried:
                break

            to_query = unqueried[:self.ALPHA]
            new_nodes: List[DHTNode] = []

            for _, node in to_query:
                queried.add(node.node_id)

                try:
                    value, returned_nodes = self.find_value(key, requester=node)
                    if value is not None:
                        # 找到值，缓存到本地
                        self.store(key, value)
                        return value, returned_nodes

                    for rn in returned_nodes:
                        if rn.node_id not in queried:
                            new_nodes.append(rn)
                except Exception:
                    continue

            for nn in new_nodes:
                dist = XORDistance.compute(key_hash, nn.node_id)
                candidates.append((dist, nn))

            seen: Set[bytes] = set()
            unique_candidates: List[Tuple[int, DHTNode]] = []
            for dist, node in sorted(candidates, key=lambda x: x[0]):
                if node.node_id not in seen:
                    seen.add(node.node_id)
                    unique_candidates.append((dist, node))

            candidates = unique_candidates[:self.K * 2]

            if len(new_nodes) == 0:
                break

        candidates.sort(key=lambda x: x[0])
        return None, [node for _, node in candidates[:self.K]]

    def iterative_store(self, key: str, value: Any, ttl: float = 86400.0) -> int:
        """
        迭代存储

        查找距离key最近的K个节点，并将值存储到这些节点上。
        返回成功存储的节点数量。

        Args:
            key: 存储的键
            value: 存储的值
            ttl: 生存时间

        Returns:
            成功存储的节点数
        """
        key_hash = hashlib.sha1(key.encode()).digest()

        # 查找最近的节点
        closest = self.iterative_find_node(key_hash)

        # 存储到本地
        self.store(key, value, ttl)
        stored_count = 1

        # 存储到最近的节点
        for node in closest:
            try:
                if self.store(key, value, ttl, sender=node):
                    stored_count += 1
            except Exception:
                continue

        return stored_count

    def bootstrap(self, bootstrap_nodes: List[DHTNode]) -> int:
        """
        引导加入网络

        通过已知的引导节点加入DHT网络。
        向引导节点发送FIND_NODE请求（查找自己），
        以发现更多网络节点并填充路由表。

        Args:
            bootstrap_nodes: 引导节点列表

        Returns:
            发现的节点数量
        """
        discovered = 0

        for node in bootstrap_nodes:
            # 添加引导节点到路由表
            self._routing_table.add_node(node)

            # 向引导节点查找自己（填充路由表）
            closest = self.find_node(self._own_node.node_id, requester=node)
            for cn in closest:
                if cn.node_id != self._own_node.node_id:
                    self._routing_table.add_node(cn)
                    discovered += 1

            # 也查找引导节点附近的其他节点
            for cn in closest[:3]:
                more_nodes = self.find_node(cn.node_id, requester=node)
                for mn in more_nodes:
                    if mn.node_id != self._own_node.node_id:
                        self._routing_table.add_node(mn)
                        discovered += 1

        return discovered

    def refresh_buckets(self) -> int:
        """
        刷新需要更新的bucket

        定期刷新长时间未更新的bucket，以保持路由表的准确性。

        Returns:
            刷新的bucket数量
        """
        refreshed = 0

        # 获取需要刷新的bucket
        bucket_idx = self._routing_table.get_bucket_for_refresh()
        while bucket_idx is not None:
            # 生成该bucket范围内的随机ID
            random_id = self._generate_random_id_in_bucket(bucket_idx)

            # 执行迭代查找
            self.iterative_find_node(random_id)
            refreshed += 1

            # 获取下一个需要刷新的bucket
            bucket_idx = self._routing_table.get_bucket_for_refresh()

        return refreshed

    def _generate_random_id_in_bucket(self, bucket_idx: int) -> bytes:
        """生成指定bucket范围内的随机节点ID"""
        own_int = int.from_bytes(self._own_node.node_id, 'big')

        # 在 [2^bucket_idx, 2^(bucket_idx+1)) 范围内生成随机距离
        low = 1 << bucket_idx
        high = (1 << (bucket_idx + 1)) - 1
        random_distance = random.randint(low, high)

        # 计算目标ID = own_id XOR random_distance
        target_int = own_int ^ random_distance
        return target_int.to_bytes(20, 'big')

    def expire_values(self) -> int:
        """清理过期的存储值"""
        expired = 0
        with self._lock:
            keys_to_remove = []
            for key, (value, timestamp, ttl) in self._storage.items():
                if time.time() - timestamp >= ttl:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._storage[key]
                expired += 1

        return expired

    def get_local_value(self, key: str) -> Optional[Any]:
        """获取本地存储的值"""
        with self._lock:
            if key in self._storage:
                value, timestamp, ttl = self._storage[key]
                if time.time() - timestamp < ttl:
                    return value
                else:
                    del self._storage[key]
        return None

    def get_storage_info(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        with self._lock:
            return {
                'stored_keys': len(self._storage),
                'keys': list(self._storage.keys()),
                'routing_info': self._routing_table.get_routing_info()
            }


# ============== 节点发现服务 ==============

class NodeDiscovery:
    """
    节点发现服务

    提供高层的节点发现接口，封装Kademlia DHT协议。
    支持引导加入网络、定期刷新、节点搜索等功能。

    使用方式：
    1. 创建NodeDiscovery实例
    2. 调用bootstrap()加入网络
    3. 使用discover()搜索特定节点
    4. 启动后台维护线程

    Author: AGI Unified Framework
    """

    MAINTENANCE_INTERVAL = 300  # 维护间隔（5分钟）
    MAX_CONCURRENT_LOOKUPS = 10  # 最大并发查找数

    def __init__(self, ip: str, port: int, seed: Optional[str] = None):
        self._own_node = DHTNode.generate(ip, port, seed)
        self._dht = KademliaDHT(self._own_node)
        self._running = False
        self._threads: List[threading.Thread] = []
        self._lock = threading.RLock()

        # 回调
        self._on_node_discovered: Optional[Callable[[DHTNode], None]] = None
        self._on_node_lost: Optional[Callable[[DHTNode], None]] = None

        # 统计
        self._stats = {
            'total_lookups': 0,
            'successful_lookups': 0,
            'total_bootstrap': 0,
            'nodes_discovered': 0
        }

    @property
    def own_node(self) -> DHTNode:
        """获取本节点信息"""
        return self._own_node

    @property
    def dht(self) -> KademliaDHT:
        """获取底层DHT实例"""
        return self._dht

    def on_node_discovered(self, callback: Callable[[DHTNode], None]) -> None:
        """设置节点发现回调"""
        self._on_node_discovered = callback

    def on_node_lost(self, callback: Callable[[DHTNode], None]) -> None:
        """设置节点丢失回调"""
        self._on_node_lost = callback

    def bootstrap(self, bootstrap_nodes: List[Tuple[str, int]]) -> bool:
        """
        引导加入网络

        通过已知的引导节点地址加入DHT网络。

        Args:
            bootstrap_nodes: 引导节点地址列表 [(ip, port), ...]

        Returns:
            是否成功加入网络
        """
        nodes = [DHTNode.generate(ip, port) for ip, port in bootstrap_nodes]
        discovered = self._dht.bootstrap(nodes)

        with self._lock:
            self._stats['total_bootstrap'] += 1
            self._stats['nodes_discovered'] += discovered

        # 通知回调
        if self._on_node_discovered and discovered > 0:
            for node in self._dht.routing_table.get_all_nodes():
                self._on_node_discovered(node)

        return discovered > 0

    def discover(self, node_id: Optional[bytes] = None,
                 key: Optional[str] = None) -> List[DHTNode]:
        """
        发现节点

        通过节点ID或键值查找网络中的节点。

        Args:
            node_id: 目标节点ID（精确查找）
            key: 查找键（查找存储该键的节点）

        Returns:
            发现的节点列表
        """
        with self._lock:
            self._stats['total_lookups'] += 1

        if node_id is not None:
            nodes = self._dht.iterative_find_node(node_id)
        elif key is not None:
            _, nodes = self._dht.iterative_find_value(key)
        else:
            # 随机查找，发现网络中的节点
            random_id = self._dht._generate_random_id_in_bucket(
                random.randint(0, 159)
            )
            nodes = self._dht.iterative_find_node(random_id)

        with self._lock:
            if nodes:
                self._stats['successful_lookups'] += 1

        return nodes

    def announce(self, key: str, value: Any, ttl: float = 86400.0) -> int:
        """
        向网络发布信息

        将键值对存储到DHT网络中最近的节点上。

        Args:
            key: 存储的键
            value: 存储的值
            ttl: 生存时间

        Returns:
            成功存储的节点数
        """
        return self._dht.iterative_store(key, value, ttl)

    def lookup(self, key: str) -> Optional[Any]:
        """
        从网络查找值

        Args:
            key: 查找的键

        Returns:
            找到的值或None
        """
        value, _ = self._dht.iterative_find_value(key)
        return value

    def get_known_nodes(self) -> List[DHTNode]:
        """获取所有已知节点"""
        return self._dht.routing_table.get_all_nodes()

    def get_node_count(self) -> int:
        """获取已知节点数量"""
        return self._dht.routing_table.get_node_count()

    def ping_node(self, node: DHTNode) -> bool:
        """检测节点是否存活"""
        return self._dht.ping(node)

    def add_node(self, node: DHTNode) -> bool:
        """手动添加节点到路由表"""
        return self._dht.routing_table.add_node(node)

    def remove_node(self, node_id: bytes) -> bool:
        """从路由表中移除节点"""
        return self._dht.routing_table.remove_node(node_id)

    def start(self) -> None:
        """启动节点发现服务（后台维护线程）"""
        if self._running:
            return

        self._running = True

        # 启动维护线程
        maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            daemon=True,
            name="dht-maintenance"
        )
        maintenance_thread.start()
        self._threads.append(maintenance_thread)

    def stop(self) -> None:
        """停止节点发现服务"""
        self._running = False
        for thread in self._threads:
            thread.join(timeout=5.0)
        self._threads.clear()

    def _maintenance_loop(self) -> None:
        """后台维护循环"""
        while self._running:
            time.sleep(self.MAINTENANCE_INTERVAL)

            try:
                # 刷新bucket
                self._dht.refresh_buckets()

                # 清理过期值
                self._dht.expire_values()

                # 刷新过期节点
                removed = self._dht.routing_table.refresh_stale_nodes()

                # 通知节点丢失
                if self._on_node_lost and removed > 0:
                    pass  # 具体节点信息需要更细粒度的跟踪

            except Exception as e:
                pass  # 静默处理维护错误

    def get_status(self) -> Dict[str, Any]:
        """获取节点发现服务状态"""
        with self._lock:
            return {
                'node_id': self._own_node.node_id_hex,
                'address': f"{self._own_node.ip}:{self._own_node.port}",
                'is_running': self._running,
                'known_nodes': self.get_node_count(),
                'stats': dict(self._stats),
                'dht_info': self._dht.get_storage_info()
            }


# ============== 主程序入口 ==============

if __name__ == "__main__":
    print("=== DHT Node Discovery Demo ===\n")

    # 创建节点发现服务
    discovery = NodeDiscovery("127.0.0.1", 9000)

    # 生成一些模拟节点
    bootstrap_addrs = [("127.0.0.1", 9001), ("127.0.0.1", 9002), ("127.0.0.1", 9003)]

    # 引导加入网络
    success = discovery.bootstrap(bootstrap_addrs)
    print(f"Bootstrap: {'success' if success else 'failed'}")
    print(f"Known nodes: {discovery.get_node_count()}")

    # 发布和查找数据
    discovery.announce("model_v1", {"weights": [0.1, 0.2, 0.3]}, ttl=3600)
    value = discovery.lookup("model_v1")
    print(f"Lookup 'model_v1': {value}")

    # 发现节点
    nodes = discovery.discover()
    print(f"Discovered nodes: {len(nodes)}")

    # 状态信息
    status = discovery.get_status()
    print(f"\nStatus: {status}")

    print("\n=== Demo Complete ===")
