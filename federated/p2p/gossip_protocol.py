"""
Gossip Protocol - Gossip传播协议实现
基于流行病算法的分布式消息传播系统

实现多种Gossip变体：
- Rumor Mongering（谣言传播）：Push/Pull/Push-Pull模式
- Anti-Entropy（反熵同步）：定期全量状态交换
- Plumtree（广播树优化）：基于Epidemic Broadcast Trees的高效广播

Author: AGI Unified Framework
"""

import hashlib
import random
import time
import threading
import uuid
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any, Callable
from collections import deque
from enum import Enum


# ============== Gossip配置 ==============

@dataclass
class GossipConfig:
    """
    Gossip协议配置

    Attributes:
        fanout: 每轮传播的邻居数量（扇出因子）
        gossip_period: Gossip传播周期（秒）
        max_ttl: 消息最大生存时间（跳数）
        max_hops: 消息最大跳数
        history_size: 消息历史记录大小（防重复）
        anti_entropy_period: 反熵同步周期（秒）
        suspicion_timeout: 节点疑似失效超时（秒）
        dead_timeout: 节点确认失效超时（秒）
        plumtree_lazy_period: Plumtree懒推送周期（秒）
        plumtree_optimize_interval: Plumtree优化间隔（秒）
    """
    fanout: int = 3
    gossip_period: float = 0.5
    max_ttl: int = 64
    max_hops: int = 32
    history_size: int = 10000
    anti_entropy_period: float = 10.0
    suspicion_timeout: float = 30.0
    dead_timeout: float = 60.0
    plumtree_lazy_period: float = 1.0
    plumtree_optimize_interval: float = 60.0


# ============== Gossip消息 ==============

@dataclass
class GossipMessage:
    """
    Gossip消息

    在网络中传播的基本消息单元。
    每条消息有唯一的message_id用于去重。

    Attributes:
        message_id: 消息唯一标识
        payload: 消息载荷
        source: 消息来源节点
        ttl: 剩余生存时间
        hops: 已经过的跳数
        timestamp: 创建时间
        topic: 消息主题
    """
    message_id: str
    payload: Any
    source: str
    ttl: int = 64
    hops: int = 0
    timestamp: float = field(default_factory=time.time)
    topic: str = "default"

    def __post_init__(self):
        if not self.message_id:
            self.message_id = self._generate_id()

    def _generate_id(self) -> str:
        """生成消息ID"""
        raw = f"{self.source}:{self.topic}:{self.timestamp}:{id(self)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    @property
    def is_expired(self) -> bool:
        """消息是否已过期"""
        return self.ttl <= 0 or self.hops >= 64

    def decrement_ttl(self) -> None:
        """递减TTL并增加跳数"""
        self.ttl -= 1
        self.hops += 1

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            'message_id': self.message_id,
            'payload': self.payload,
            'source': self.source,
            'ttl': self.ttl,
            'hops': self.hops,
            'timestamp': self.timestamp,
            'topic': self.topic
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GossipMessage':
        """从字典反序列化"""
        return cls(
            message_id=data['message_id'],
            payload=data['payload'],
            source=data['source'],
            ttl=data.get('ttl', 64),
            hops=data.get('hops', 0),
            timestamp=data.get('timestamp', time.time()),
            topic=data.get('topic', 'default')
        )

    @classmethod
    def create(cls, source: str, payload: Any,
               topic: str = "default", ttl: int = 64) -> 'GossipMessage':
        """创建新消息"""
        return cls(
            message_id="",
            payload=payload,
            source=source,
            ttl=ttl,
            topic=topic
        )


# ============== 成员管理 ==============

class MemberState(Enum):
    """成员状态"""
    ALIVE = "alive"
    SUSPECT = "suspect"
    DEAD = "dead"
    LEFT = "left"


@dataclass
class Member:
    """
    网络成员信息

    Attributes:
        node_id: 节点ID
        address: 节点地址
        state: 成员状态
        incarnation: 世代号（用于冲突解决）
        last_seen: 最后一次收到心跳的时间
        metadata: 成员元数据
    """
    node_id: str
    address: str
    state: MemberState = MemberState.ALIVE
    incarnation: int = 0
    last_seen: float = field(default_factory=time.time)
    metadata: Dict[str, str] = field(default_factory=dict)

    def update_seen(self) -> None:
        """更新最后可见时间"""
        self.last_seen = time.time()
        if self.state == MemberState.SUSPECT:
            self.state = MemberState.ALIVE

    def is_alive(self) -> bool:
        """是否存活"""
        return self.state == MemberState.ALIVE


class GossipMembership:
    """
    成员管理

    管理Gossip网络中的活跃节点列表，实现失效检测机制。
    基于SWIM（Scalable Weakly-consistent Infection-style
    Membership）协议的简化版本。

    成员状态转换：
    ALIVE -> SUSPECT -> DEAD -> (从列表中移除)

    Attributes:
        _members: 成员列表
        _config: 配置参数
    """

    def __init__(self, node_id: str, config: GossipConfig):
        self._own_id = node_id
        self._config = config
        self._members: Dict[str, Member] = {}
        self._lock = threading.RLock()
        self._incarnation = 0

        # 回调
        self._on_join: Optional[Callable[[Member], None]] = None
        self._on_leave: Optional[Callable[[Member], None]] = None
        self._on_suspect: Optional[Callable[[Member], None]] = None
        self._on_dead: Optional[Callable[[Member], None]] = None

    def set_callbacks(self, on_join: Optional[Callable] = None,
                      on_leave: Optional[Callable] = None,
                      on_suspect: Optional[Callable] = None,
                      on_dead: Optional[Callable] = None) -> None:
        """设置成员状态变更回调"""
        self._on_join = on_join
        self._on_leave = on_leave
        self._on_suspect = on_suspect
        self._on_dead = on_dead

    def join(self, node_id: str, address: str,
             metadata: Optional[Dict[str, str]] = None) -> None:
        """
        节点加入网络

        Args:
            node_id: 节点ID
            address: 节点地址
            metadata: 元数据
        """
        if node_id == self._own_id:
            return

        with self._lock:
            if node_id in self._members:
                member = self._members[node_id]
                if member.state == MemberState.DEAD or member.state == MemberState.LEFT:
                    member.state = MemberState.ALIVE
                    member.incarnation += 1
                    member.update_seen()
                return

            member = Member(
                node_id=node_id,
                address=address,
                metadata=metadata or {}
            )
            self._members[node_id] = member

        if self._on_join:
            self._on_join(member)

    def leave(self, node_id: str) -> None:
        """节点离开网络"""
        with self._lock:
            if node_id in self._members:
                member = self._members[node_id]
                member.state = MemberState.LEFT
                del self._members[node_id]

        if self._on_leave:
            self._on_leave(Member(node_id=node_id, address=""))

    def update_heartbeat(self, node_id: str, incarnation: int = 0) -> bool:
        """
        更新节点心跳

        Args:
            node_id: 节点ID
            incarnation: 世代号

        Returns:
            是否更新成功
        """
        with self._lock:
            if node_id not in self._members:
                return False

            member = self._members[node_id]
            if incarnation > member.incarnation:
                member.incarnation = incarnation
            member.update_seen()
            return True

    def suspect(self, node_id: str) -> bool:
        """
        标记节点为疑似失效

        Args:
            node_id: 节点ID

        Returns:
            是否标记成功
        """
        with self._lock:
            if node_id not in self._members:
                return False

            member = self._members[node_id]
            if member.state == MemberState.ALIVE:
                member.state = MemberState.SUSPECT

                if self._on_suspect:
                    self._on_suspect(member)
                return True
        return False

    def check_failures(self) -> List[str]:
        """
        检查失效节点

        将超时的疑似节点标记为死亡。

        Returns:
            被标记为死亡的节点ID列表
        """
        now = time.time()
        dead_nodes: List[str] = []

        with self._lock:
            for node_id, member in list(self._members.items()):
                if member.state == MemberState.SUSPECT:
                    if (now - member.last_seen) > self._config.dead_timeout:
                        member.state = MemberState.DEAD
                        dead_nodes.append(node_id)

                        if self._on_dead:
                            self._on_dead(member)

                elif member.state == MemberState.ALIVE:
                    if (now - member.last_seen) > self._config.suspicion_timeout:
                        self.suspect(node_id)

        return dead_nodes

    def get_alive_members(self) -> List[Member]:
        """获取所有存活成员"""
        with self._lock:
            return [
                m for m in self._members.values()
                if m.is_alive()
            ]

    def get_random_members(self, count: int) -> List[Member]:
        """随机获取指定数量的存活成员"""
        alive = self.get_alive_members()
        if count >= len(alive):
            return alive
        return random.sample(alive, count)

    def get_member(self, node_id: str) -> Optional[Member]:
        """获取指定成员"""
        return self._members.get(node_id)

    def get_member_count(self) -> int:
        """获取成员总数"""
        with self._lock:
            return len(self._members)

    def get_alive_count(self) -> int:
        """获取存活成员数"""
        return len(self.get_alive_members())

    def remove_dead_members(self) -> int:
        """移除死亡成员"""
        with self._lock:
            dead_ids = [
                nid for nid, m in self._members.items()
                if m.state == MemberState.DEAD
            ]
            for nid in dead_ids:
                del self._members[nid]
            return len(dead_ids)

    def get_member_list(self) -> List[Dict[str, Any]]:
        """获取成员列表（序列化）"""
        with self._lock:
            return [
                {
                    'node_id': m.node_id,
                    'address': m.address,
                    'state': m.state.value,
                    'incarnation': m.incarnation,
                    'last_seen': m.last_seen
                }
                for m in self._members.values()
            ]


# ============== 谣言传播策略 ==============

class RumorMongering:
    """
    谣言传播策略

    实现三种Gossip传播模式：
    - Push: 源节点主动推送消息给随机邻居
    - Pull: 节点主动从随机邻居拉取消息
    - Push-Pull: 双向交换，最有效的传播模式

    传播过程：
    1. 节点收到新消息（"谣言"）
    2. 每轮随机选择fanout个邻居
    3. 通过Push/Pull/Push-Pull传播消息
    4. 消息到达所有节点后传播停止
    """

    def __init__(self, node_id: str, config: GossipConfig,
                 membership: GossipMembership):
        self._node_id = node_id
        self._config = config
        self._membership = membership

        # 已知消息集合（去重）
        self._known_messages: Set[str] = set()
        # 待传播的消息队列
        self._rumor_buffer: Dict[str, GossipMessage] = {}
        # 传播计数
        self._spread_count: Dict[str, int] = {}
        self._lock = threading.Lock()

    def receive_rumor(self, message: GossipMessage) -> bool:
        """
        接收谣言

        Args:
            message: 收到的消息

        Returns:
            是否为新消息
        """
        with self._lock:
            if message.message_id in self._known_messages:
                return False

            self._known_messages.add(message.message_id)
            self._rumor_buffer[message.message_id] = message
            self._spread_count[message.message_id] = 0

            # 限制历史大小
            if len(self._known_messages) > self._config.history_size:
                oldest = next(iter(self._known_messages))
                self._known_messages.discard(oldest)
                self._rumor_buffer.pop(oldest, None)
                self._spread_count.pop(oldest, None)

            return True

    def push(self, message: GossipMessage) -> int:
        """
        Push传播：将消息推送给随机邻居

        源节点主动选择fanout个邻居，将消息发送给它们。

        Args:
            message: 要传播的消息

        Returns:
            成功推送的节点数
        """
        targets = self._membership.get_random_members(self._config.fanout)
        pushed = 0

        for target in targets:
            if target.node_id == message.source:
                continue

            # 模拟发送消息
            message.decrement_ttl()
            if not message.is_expired:
                pushed += 1

        with self._lock:
            self._spread_count[message.message_id] = \
                self._spread_count.get(message.message_id, 0) + pushed

        return pushed

    def pull(self) -> List[GossipMessage]:
        """
        Pull传播：从随机邻居拉取消息

        节点向随机邻居发送自己的已知消息摘要，
        邻居返回自己有但对方没有的消息。

        Returns:
            拉取到的消息列表（模拟）
        """
        # 模拟从邻居拉取
        pulled: List[GossipMessage] = []

        with self._lock:
            # 返回当前缓冲区中的消息（模拟拉取结果）
            for msg_id, msg in self._rumor_buffer.items():
                if msg.ttl > 0:
                    pulled.append(msg)

        return pulled

    def push_pull(self, message: GossipMessage) -> Tuple[int, List[GossipMessage]]:
        """
        Push-Pull传播：双向交换

        同时执行Push和Pull操作，是最有效的传播模式。
        先推送自己的消息，再拉取对方的消息。

        Args:
            message: 要传播的消息

        Returns:
            (推送的节点数, 拉取到的消息列表)
        """
        pushed = self.push(message)
        pulled = self.pull()
        return pushed, pulled

    def gossip_round(self) -> int:
        """
        执行一轮Gossip传播

        对缓冲区中的每条消息执行Push-Pull传播。
        如果消息已被充分传播（超过阈值），则从缓冲区移除。

        Returns:
            本轮传播的消息数
        """
        messages_to_spread: List[GossipMessage] = []

        with self._lock:
            for msg_id, msg in list(self._rumor_buffer.items()):
                if msg.is_expired:
                    self._rumor_buffer.pop(msg_id, None)
                    continue
                messages_to_spread.append(msg)

        spread_count = 0
        for msg in messages_to_spread:
            self.push_pull(msg)
            spread_count += 1

            # 检查是否已充分传播
            with self._lock:
                count = self._spread_count.get(msg.message_id, 0)
                if count >= self._config.fanout * 3:
                    self._rumor_buffer.pop(msg.message_id, None)

        return spread_count

    def get_rumor_count(self) -> int:
        """获取当前缓冲区中的谣言数量"""
        with self._lock:
            return len(self._rumor_buffer)

    def get_known_count(self) -> int:
        """获取已知消息总数"""
        with self._lock:
            return len(self._known_messages)


# ============== 反熵同步 ==============

class AntiEntropy:
    """
    反熵同步

    定期执行全量状态交换，确保所有节点的状态最终一致。
    使用Merkle树或简单摘要来高效比较状态差异。

    反熵同步是Gossip协议的补充机制：
    - Rumor Mongering速度快但可能丢失消息
    - Anti-Entropy速度慢但保证最终一致性

    同步策略：
    1. 节点定期选择一个随机邻居
    2. 交换双方的状态摘要
    3. 比较差异并同步缺失的数据
    """

    def __init__(self, node_id: str, config: GossipConfig,
                 membership: GossipMembership):
        self._node_id = node_id
        self._config = config
        self._membership = membership

        # 本地状态版本
        self._state_version: int = 0
        self._state_digest: str = ""
        self._last_sync: float = 0.0
        self._sync_count: int = 0
        self._lock = threading.Lock()

        # 状态存储
        self._state: Dict[str, Any] = {}

    def update_state(self, key: str, value: Any) -> None:
        """更新本地状态"""
        with self._lock:
            self._state[key] = {
                'value': value,
                'version': self._state_version,
                'timestamp': time.time()
            }
            self._state_version += 1
            self._update_digest()

    def get_state(self, key: str) -> Optional[Any]:
        """获取本地状态"""
        with self._lock:
            entry = self._state.get(key)
            if entry:
                return entry['value']
        return None

    def get_full_state(self) -> Dict[str, Any]:
        """获取完整状态"""
        with self._lock:
            return {k: v['value'] for k, v in self._state.items()}

    def _update_digest(self) -> None:
        """更新状态摘要"""
        content = json.dumps(
            {k: v['version'] for k, v in self._state.items()},
            sort_keys=True
        )
        self._state_digest = hashlib.sha256(content.encode()).hexdigest()[:16]

    def get_digest(self) -> Dict[str, Any]:
        """
        获取状态摘要

        Returns:
            包含版本和摘要的字典
        """
        with self._lock:
            return {
                'node_id': self._node_id,
                'version': self._state_version,
                'digest': self._state_digest,
                'key_count': len(self._state),
                'timestamp': time.time()
            }

    def sync_with(self, remote_digest: Dict[str, Any]) -> Dict[str, Any]:
        """
        与远程节点同步

        比较本地和远程的状态摘要，返回需要交换的数据。

        Args:
            remote_digest: 远程节点的状态摘要

        Returns:
            同步结果（差异信息）
        """
        with self._lock:
            local_digest = self.get_digest()
            self._last_sync = time.time()
            self._sync_count += 1

            # 如果版本相同，无需同步
            if local_digest['version'] == remote_digest.get('version'):
                return {'status': 'in_sync', 'diff_keys': []}

            # 计算差异
            diff_keys: List[str] = []
            remote_version = remote_digest.get('version', 0)

            for key, entry in self._state.items():
                if entry['version'] > remote_version:
                    diff_keys.append(key)

            return {
                'status': 'needs_sync',
                'diff_keys': diff_keys,
                'local_version': local_digest['version'],
                'remote_version': remote_version
            }

    def merge_state(self, remote_state: Dict[str, Any],
                    remote_version: int) -> int:
        """
        合并远程状态

        Args:
            remote_state: 远程状态
            remote_version: 远程版本号

        Returns:
            合并的键数量
        """
        merged = 0
        with self._lock:
            for key, value in remote_state.items():
                if key not in self._state:
                    self._state[key] = {
                        'value': value,
                        'version': self._state_version,
                        'timestamp': time.time()
                    }
                    self._state_version += 1
                    merged += 1

            self._update_digest()

        return merged

    def should_sync(self) -> bool:
        """是否需要执行反熵同步"""
        return (time.time() - self._last_sync) >= self._config.anti_entropy_period

    def get_sync_stats(self) -> Dict[str, Any]:
        """获取同步统计"""
        with self._lock:
            return {
                'state_version': self._state_version,
                'state_digest': self._state_digest,
                'sync_count': self._sync_count,
                'last_sync': self._last_sync,
                'state_keys': len(self._state)
            }


# ============== Plumtree广播树 ==============

class Plumtree:
    """
    Plumtree - Epidemic Broadcast Tree

    基于流行病广播树的高效广播协议。
    在Gossip的基础上构建一棵覆盖网络中所有节点的广播树，
    实现O(N)的消息复杂度（而非Gossip的O(N*logN)）。

    核心思想：
    1. 初始阶段使用Gossip传播消息（乐观传播）
    2. 通过消息到达的路径信息构建广播树
    3. 后续消息沿广播树传播（懒推送）
    4. 定期使用Gossip修复广播树（容错）

    节点角色：
    - Eager Peer: 树上的父节点，主动推送消息
    - Lazy Peer: 非树上的邻居，仅在请求时发送消息

    Author: AGI Unified Framework
    """

    def __init__(self, node_id: str, config: GossipConfig,
                 membership: GossipMembership):
        self._node_id = node_id
        self._config = config
        self._membership = membership

        # 广播树结构
        self._eager_peers: Set[str] = set()  # 树上父节点
        self._lazy_peers: Set[str] = set()   # 懒推送节点
        self._all_peers: Set[str] = set()    # 所有已知节点

        # 消息状态
        self._seen_messages: Set[str] = set()
        self._pending_messages: Dict[str, GossipMessage] = {}  # 待确认消息
        self._lazy_queue: Dict[str, Set[str]] = {}  # msg_id -> 已懒推送的节点

        # 统计
        self._messages_broadcast: int = 0
        self._tree_optimizations: int = 0
        self._lock = threading.Lock()

    def add_peer(self, peer_id: str) -> None:
        """添加对等节点"""
        with self._lock:
            self._all_peers.add(peer_id)
            # 初始时所有节点都是懒推送节点
            self._lazy_peers.add(peer_id)

    def remove_peer(self, peer_id: str) -> None:
        """移除对等节点"""
        with self._lock:
            self._all_peers.discard(peer_id)
            self._eager_peers.discard(peer_id)
            self._lazy_peers.discard(peer_id)

    def receive_message(self, message: GossipMessage,
                        from_peer: str) -> Tuple[bool, List[str]]:
        """
        接收消息

        处理接收到的消息，更新广播树结构。

        Args:
            message: 接收到的消息
            from_peer: 发送者节点ID

        Returns:
            (是否为新消息, 需要响应的节点列表)
        """
        with self._lock:
            is_new = message.message_id not in self._seen_messages
            self._seen_messages.add(message.message_id)

            if not is_new:
                return False, []

            # 更新广播树：发送者成为eager peer
            self._eager_peers.add(from_peer)
            self._lazy_peers.discard(from_peer)

            # 限制eager peers数量
            max_eager = max(1, self._config.fanout)
            while len(self._eager_peers) > max_eager:
                # 将多余的eager peer降级为lazy peer
                demoted = self._eager_peers.pop()
                self._lazy_peers.add(demoted)

            # 保存待确认消息
            self._pending_messages[message.message_id] = message
            self._lazy_queue[message.message_id] = set()

            self._messages_broadcast += 1

            # 返回需要eager推送的节点
            eager_targets = list(self._eager_peers - {from_peer})
            return True, eager_targets

    def get_lazy_push_targets(self, message_id: str) -> List[str]:
        """
        获取懒推送目标

        对尚未收到该消息的lazy peer进行懒推送。

        Args:
            message_id: 消息ID

        Returns:
            需要懒推送的节点列表
        """
        with self._lock:
            if message_id not in self._lazy_queue:
                return []

            pushed = self._lazy_queue[message_id]
            targets = list(self._lazy_peers - pushed)

            # 标记为已推送
            self._lazy_queue[message_id].update(targets)
            return targets

    def handle_graft(self, peer_id: str, message_id: str) -> Optional[GossipMessage]:
        """
        处理Graft请求

        当lazy peer收到消息通知后，发送Graft请求
        将自己升级为eager peer。

        Args:
            peer_id: 请求者节点ID
            message_id: 消息ID

        Returns:
            请求的消息或None
        """
        with self._lock:
            # 将请求者升级为eager peer
            self._lazy_peers.discard(peer_id)
            self._eager_peers.add(peer_id)

            # 限制eager peers数量
            max_eager = max(1, self._config.fanout)
            while len(self._eager_peers) > max_eager:
                demoted = self._eager_peers.pop()
                self._lazy_peers.add(demoted)

            # 返回请求的消息
            message = self._pending_messages.get(message_id)
            return message

    def handle_prune(self, peer_id: str, message_id: str) -> None:
        """
        处理Prune消息

        当节点不想接收某条消息时发送Prune，
        将发送者降级为lazy peer。

        Args:
            peer_id: 发送者节点ID
            message_id: 消息ID
        """
        with self._lock:
            self._eager_peers.discard(peer_id)
            self._lazy_peers.add(peer_id)

    def handle_ihave(self, peer_id: str,
                     message_ids: List[str]) -> List[str]:
        """
        处理IHave消息

        当lazy peer收到消息通知时，检查自己是否已有该消息。
        对没有的消息发送Graft请求。

        Args:
            peer_id: 通知者节点ID
            message_ids: 消息ID列表

        Returns:
            需要Graft的消息ID列表
        """
        with self._lock:
            needed = [
                mid for mid in message_ids
                if mid not in self._seen_messages
            ]
            return needed

    def optimize_tree(self) -> int:
        """
        优化广播树

        定期检查并优化广播树结构：
        - 移除失效的eager peer
        - 将活跃的lazy peer升级
        - 随机化树结构以提高鲁棒性

        Returns:
            优化操作的次数
        """
        optimizations = 0
        with self._lock:
            alive_members = {
                m.node_id for m in self._membership.get_alive_members()
            }

            # 移除失效的eager peer
            dead_eager = self._eager_peers - alive_members
            for peer_id in dead_eager:
                self._eager_peers.discard(peer_id)
                optimizations += 1

            # 从lazy peers中补充
            alive_lazy = self._lazy_peers & alive_members
            while len(self._eager_peers) < self._config.fanout and alive_lazy:
                promoted = alive_lazy.pop()
                self._lazy_peers.discard(promoted)
                self._eager_peers.add(promoted)
                optimizations += 1

            # 随机化：偶尔交换eager和lazy peer
            if random.random() < 0.1 and self._lazy_peers:
                demoted = random.choice(list(self._eager_peers)) if self._eager_peers else None
                promoted = random.choice(list(self._lazy_peers))
                if demoted:
                    self._eager_peers.discard(demoted)
                    self._lazy_peers.add(demoted)
                self._lazy_peers.discard(promoted)
                self._eager_peers.add(promoted)
                optimizations += 1

            # 清理旧消息
            now = time.time()
            old_messages = [
                mid for mid, msg in self._pending_messages.items()
                if (now - msg.timestamp) > 300  # 5分钟过期
            ]
            for mid in old_messages:
                del self._pending_messages[mid]
                self._lazy_queue.pop(mid, None)

        self._tree_optimizations += optimizations
        return optimizations

    def get_tree_info(self) -> Dict[str, Any]:
        """获取广播树信息"""
        with self._lock:
            return {
                'eager_peers': len(self._eager_peers),
                'lazy_peers': len(self._lazy_peers),
                'total_peers': len(self._all_peers),
                'pending_messages': len(self._pending_messages),
                'seen_messages': len(self._seen_messages),
                'messages_broadcast': self._messages_broadcast,
                'tree_optimizations': self._tree_optimizations
            }


# ============== Gossip协议主类 ==============

class GossipProtocol:
    """
    Gossip协议主类

    整合谣言传播、成员管理和反熵同步，
    提供统一的Gossip通信接口。

    使用方式：
    1. 创建GossipProtocol实例
    2. 添加初始成员
    3. 调用broadcast()广播消息
    4. 调用start()启动后台Gossip线程
    5. 注册消息处理器接收消息

    Author: AGI Unified Framework
    """

    def __init__(self, node_id: str, config: Optional[GossipConfig] = None):
        self._node_id = node_id
        self._config = config or GossipConfig()

        # 子组件
        self._membership = GossipMembership(node_id, self._config)
        self._rumor = RumorMongering(node_id, self._config, self._membership)
        self._anti_entropy = AntiEntropy(node_id, self._config, self._membership)
        self._plumtree = Plumtree(node_id, self._config, self._membership)

        # 消息处理器
        self._subscribers: Dict[str, List[Callable[[GossipMessage], None]]] = {}

        # 运行状态
        self._running = False
        self._threads: List[threading.Thread] = []
        self._lock = threading.RLock()

        # 统计
        self._stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'messages_broadcast': 0,
            'duplicates_dropped': 0
        }

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def membership(self) -> GossipMembership:
        return self._membership

    @property
    def config(self) -> GossipConfig:
        return self._config

    def subscribe(self, topic: str,
                  handler: Callable[[GossipMessage], None]) -> None:
        """
        订阅主题

        Args:
            topic: 消息主题
            handler: 消息处理函数
        """
        with self._lock:
            if topic not in self._subscribers:
                self._subscribers[topic] = []
            self._subscribers[topic].append(handler)

    def unsubscribe(self, topic: str,
                    handler: Callable[[GossipMessage], None]) -> None:
        """取消订阅"""
        with self._lock:
            if topic in self._subscribers:
                try:
                    self._subscribers[topic].remove(handler)
                except ValueError:
                    pass

    def broadcast(self, payload: Any, topic: str = "default",
                  ttl: int = None) -> str:
        """
        广播消息

        创建新消息并通过Gossip协议传播到网络中的所有节点。

        Args:
            payload: 消息载荷
            topic: 消息主题
            ttl: 生存时间

        Returns:
            消息ID
        """
        if ttl is None:
            ttl = self._config.max_ttl

        message = GossipMessage.create(
            source=self._node_id,
            payload=payload,
            topic=topic,
            ttl=ttl
        )

        # 接收自己的消息
        self._receive_message(message, self._node_id)

        with self._lock:
            self._stats['messages_broadcast'] += 1

        return message.message_id

    def _receive_message(self, message: GossipMessage,
                         from_peer: str) -> bool:
        """
        接收并处理消息

        Args:
            message: 收到的消息
            from_peer: 发送者节点ID

        Returns:
            是否为新消息
        """
        # 检查是否为新消息
        is_new = self._rumor.receive_rumor(message)

        if not is_new:
            with self._lock:
                self._stats['duplicates_dropped'] += 1
            return False

        with self._lock:
            self._stats['messages_received'] += 1

        # Plumtree处理
        self._plumtree.receive_message(message, from_peer)

        # 投递给订阅者
        self._deliver_to_subscribers(message)

        # 更新反熵状态
        self._anti_entropy.update_state(
            f"msg:{message.message_id}",
            message.payload
        )

        return True

    def _deliver_to_subscribers(self, message: GossipMessage) -> None:
        """将消息投递给订阅者"""
        with self._lock:
            handlers = self._subscribers.get(message.topic, [])
            all_handlers = self._subscribers.get("*", [])
            all_handlers.extend(handlers)

        for handler in all_handlers:
            try:
                handler(message)
            except Exception:
                pass

    def add_member(self, node_id: str, address: str) -> None:
        """添加网络成员"""
        self._membership.join(node_id, address)
        self._plumtree.add_peer(node_id)

    def remove_member(self, node_id: str) -> None:
        """移除网络成员"""
        self._membership.leave(node_id)
        self._plumtree.remove_peer(node_id)

    def start(self) -> None:
        """启动Gossip协议后台线程"""
        if self._running:
            return

        self._running = True

        # Gossip传播线程
        gossip_thread = threading.Thread(
            target=self._gossip_loop,
            daemon=True,
            name="gossip-spread"
        )
        gossip_thread.start()
        self._threads.append(gossip_thread)

        # 反熵同步线程
        entropy_thread = threading.Thread(
            target=self._anti_entropy_loop,
            daemon=True,
            name="anti-entropy"
        )
        entropy_thread.start()
        self._threads.append(entropy_thread)

        # 失效检测线程
        failure_thread = threading.Thread(
            target=self._failure_detection_loop,
            daemon=True,
            name="failure-detection"
        )
        failure_thread.start()
        self._threads.append(failure_thread)

        # Plumtree优化线程
        plumtree_thread = threading.Thread(
            target=self._plumtree_optimize_loop,
            daemon=True,
            name="plumtree-optimize"
        )
        plumtree_thread.start()
        self._threads.append(plumtree_thread)

    def stop(self) -> None:
        """停止Gossip协议"""
        self._running = False
        for thread in self._threads:
            thread.join(timeout=5.0)
        self._threads.clear()

    def _gossip_loop(self) -> None:
        """Gossip传播循环"""
        while self._running:
            time.sleep(self._config.gossip_period)
            try:
                self._rumor.gossip_round()
            except Exception:
                pass

    def _anti_entropy_loop(self) -> None:
        """反熵同步循环"""
        while self._running:
            time.sleep(self._config.anti_entropy_period)
            try:
                if self._anti_entropy.should_sync():
                    members = self._membership.get_random_members(1)
                    if members:
                        remote_digest = self._anti_entropy.get_digest()
                        self._anti_entropy.sync_with(remote_digest)
            except Exception:
                pass

    def _failure_detection_loop(self) -> None:
        """失效检测循环"""
        while self._running:
            time.sleep(10.0)
            try:
                self._membership.check_failures()
                self._membership.remove_dead_members()
            except Exception:
                pass

    def _plumtree_optimize_loop(self) -> None:
        """Plumtree优化循环"""
        while self._running:
            time.sleep(self._config.plumtree_optimize_interval)
            try:
                self._plumtree.optimize_tree()
            except Exception:
                pass

    def get_status(self) -> Dict[str, Any]:
        """获取协议状态"""
        with self._lock:
            return {
                'node_id': self._node_id,
                'is_running': self._running,
                'members': self._membership.get_alive_count(),
                'rumors': self._rumor.get_rumor_count(),
                'known_messages': self._rumor.get_known_count(),
                'plumtree': self._plumtree.get_tree_info(),
                'anti_entropy': self._anti_entropy.get_sync_stats(),
                'stats': dict(self._stats)
            }


# ============== 主程序入口 ==============

if __name__ == "__main__":
    print("=== Gossip Protocol Demo ===\n")

    config = GossipConfig(fanout=3, gossip_period=0.1)
    gossip = GossipProtocol("node_0", config)

    # 添加成员
    for i in range(1, 10):
        gossip.add_member(f"node_{i}", f"10.0.0.{i}")

    # 订阅消息
    received_messages: List[GossipMessage] = []

    def on_message(msg: GossipMessage) -> None:
        received_messages.append(msg)

    gossip.subscribe("test", on_message)

    # 广播消息
    msg_id = gossip.broadcast({"round": 1, "data": "hello"}, topic="test")
    print(f"Broadcast message: {msg_id[:16]}...")

    # 启动Gossip
    gossip.start()
    time.sleep(1)

    # 状态
    status = gossip.get_status()
    print(f"Members: {status['members']}")
    print(f"Known messages: {status['known_messages']}")

    gossip.stop()
    print("\n=== Demo Complete ===")
