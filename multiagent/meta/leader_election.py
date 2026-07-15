"""
分布式选主系统 - Leader Election

实现了两种经典的分布式选主算法:
1. Bully算法 - 基于节点ID的选举
2. Raft算法 - 基于任期和投票的选举
"""

from __future__ import annotations

import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class NodeState(Enum):
    """节点状态"""
    FOLLOWER = auto()
    CANDIDATE = auto()
    LEADER = auto()
    DOWN = auto()  # 节点宕机


class ElectionState(Enum):
    """选举状态"""
    IDLE = auto()
    ELECTING = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class NodeInfo:
    """节点信息"""
    node_id: str
    address: str
    priority: int = 0  # 优先级，越高越可能成为leader
    last_heartbeat: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_alive(self, timeout: float = 5.0) -> bool:
        """检查节点是否存活"""
        return time.time() - self.last_heartbeat < timeout


@dataclass
class VoteRecord:
    """投票记录"""
    term: int
    voted_for: Optional[str]
    timestamp: float = field(default_factory=time.time)


class ElectionMessage:
    """选举消息基类"""
    
    def __init__(self, msg_type: str, sender_id: str, term: int = 0,
                 data: Optional[Dict[str, Any]] = None):
        self.msg_type = msg_type
        self.sender_id = sender_id
        self.term = term
        self.data = data or {}
        self.timestamp = time.time()


class ElectionAlgorithm(ABC):
    """选举算法基类"""
    
    def __init__(self, node_id: str, nodes: Dict[str, NodeInfo]):
        self.node_id = node_id
        self.nodes = nodes
        self.state = NodeState.FOLLOWER
        self.leader_id: Optional[str] = None
        self.election_callbacks: List[Callable[[str, str], None]] = []
        self._lock = threading.RLock()
        self._running = False
        
    @abstractmethod
    def start(self) -> None:
        """启动选举算法"""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """停止选举算法"""
        pass
    
    @abstractmethod
    def handle_message(self, message: ElectionMessage) -> Optional[ElectionMessage]:
        """处理选举消息"""
        pass
    
    @abstractmethod
    def get_leader(self) -> Optional[str]:
        """获取当前leader"""
        pass
    
    def register_callback(self, callback: Callable[[str, str], None]) -> None:
        """注册leader变更回调"""
        with self._lock:
            self.election_callbacks.append(callback)
    
    def _notify_leader_change(self, old_leader: Optional[str], new_leader: str) -> None:
        """通知leader变更"""
        for callback in self.election_callbacks:
            try:
                callback(old_leader or "none", new_leader)
            except Exception:
                pass
    
    def add_node(self, node_info: NodeInfo) -> None:
        """添加节点"""
        with self._lock:
            self.nodes[node_info.node_id] = node_info
    
    def remove_node(self, node_id: str) -> None:
        """移除节点"""
        with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
            if self.leader_id == node_id:
                self.leader_id = None


class BullyElection(ElectionAlgorithm):
    """
    Bully选举算法实现
    
    算法原理:
    1. 当节点发现leader失效时，发起选举
    2. 节点向所有ID比自己大的节点发送选举消息
    3. 如果有更高ID的节点响应，则该节点退出选举
    4. 如果没有更高ID的节点响应，该节点成为leader并广播
    """
    
    def __init__(self, node_id: str, nodes: Dict[str, NodeInfo],
                 election_timeout: float = 3.0):
        super().__init__(node_id, nodes)
        self.election_timeout = election_timeout
        self.election_state = ElectionState.IDLE
        self._election_timer: Optional[threading.Timer] = None
        self._heartbeat_timer: Optional[threading.Timer] = None
        self._received_ok = False
        self._message_handlers: Dict[str, Callable[[ElectionMessage], Optional[ElectionMessage]]] = {
            "ELECTION": self._handle_election_msg,
            "OK": self._handle_ok_msg,
            "COORDINATOR": self._handle_coordinator_msg,
            "HEARTBEAT": self._handle_heartbeat_msg
        }
        
    def start(self) -> None:
        """启动Bully选举"""
        with self._lock:
            self._running = True
            self.state = NodeState.FOLLOWER
            # 启动心跳检测
            self._start_heartbeat_check()
    
    def stop(self) -> None:
        """停止Bully选举"""
        with self._lock:
            self._running = False
            self.state = NodeState.DOWN
            if self._election_timer:
                self._election_timer.cancel()
            if self._heartbeat_timer:
                self._heartbeat_timer.cancel()
    
    def handle_message(self, message: ElectionMessage) -> Optional[ElectionMessage]:
        """处理选举消息"""
        with self._lock:
            handler = self._message_handlers.get(message.msg_type)
            if handler:
                return handler(message)
            return None
    
    def _handle_election_msg(self, message: ElectionMessage) -> Optional[ElectionMessage]:
        """处理选举消息"""
        sender_id = message.sender_id
        
        # 如果发送者ID小于自己，发送OK响应并发起自己的选举
        if self._compare_node_id(sender_id, self.node_id) < 0:
            # 发送OK响应
            response = ElectionMessage("OK", self.node_id, data={"to": sender_id})
            
            # 如果自己不是leader且不在选举中，发起选举
            if self.state != NodeState.LEADER and self.election_state != ElectionState.ELECTING:
                self._start_election()
            
            return response
        
        return None
    
    def _handle_ok_msg(self, message: ElectionMessage) -> None:
        """处理OK响应"""
        if self.election_state == ElectionState.ELECTING:
            self._received_ok = True
    
    def _handle_coordinator_msg(self, message: ElectionMessage) -> None:
        """处理协调者消息（新leader广播）"""
        new_leader = message.sender_id
        old_leader = self.leader_id
        
        self.leader_id = new_leader
        self.state = NodeState.FOLLOWER
        self.election_state = ElectionState.COMPLETED
        
        # 更新leader节点信息
        if new_leader in self.nodes:
            self.nodes[new_leader].last_heartbeat = time.time()
        
        if old_leader != new_leader:
            self._notify_leader_change(old_leader, new_leader)
    
    def _handle_heartbeat_msg(self, message: ElectionMessage) -> None:
        """处理心跳消息"""
        sender_id = message.sender_id
        if sender_id in self.nodes:
            self.nodes[sender_id].last_heartbeat = time.time()
        
        # 如果收到leader的心跳，确认leader状态
        if sender_id == self.leader_id:
            pass  # Leader正常
        elif self._compare_node_id(sender_id, self.leader_id or "") > 0:
            # 收到更高ID节点的心跳，可能是新leader
            old_leader = self.leader_id
            self.leader_id = sender_id
            self.state = NodeState.FOLLOWER
            if old_leader != sender_id:
                self._notify_leader_change(old_leader, sender_id)
    
    def _start_election(self) -> None:
        """发起选举"""
        with self._lock:
            self.election_state = ElectionState.ELECTING
            self._received_ok = False
            self.state = NodeState.CANDIDATE
            
            # 向所有ID比自己大的节点发送选举消息
            higher_nodes = self._get_higher_id_nodes()
            
            if not higher_nodes:
                # 没有更高ID的节点，自己成为leader
                self._become_leader()
                return
            
            # 发送选举消息（模拟）
            for node_id in higher_nodes:
                self._send_election_message(node_id)
            
            # 设置选举超时
            self._election_timer = threading.Timer(self.election_timeout, self._on_election_timeout)
            self._election_timer.start()
    
    def _on_election_timeout(self) -> None:
        """选举超时处理"""
        with self._lock:
            if not self._received_ok and self.election_state == ElectionState.ELECTING:
                # 没有收到OK响应，成为leader
                self._become_leader()
    
    def _become_leader(self) -> None:
        """成为leader"""
        with self._lock:
            old_leader = self.leader_id
            self.leader_id = self.node_id
            self.state = NodeState.LEADER
            self.election_state = ElectionState.COMPLETED
            
            # 广播coordination消息
            self._broadcast_coordinator()
            
            # 启动leader心跳
            self._start_leader_heartbeat()
            
            if old_leader != self.node_id:
                self._notify_leader_change(old_leader, self.node_id)
    
    def _start_leader_heartbeat(self) -> None:
        """启动leader心跳"""
        def send_heartbeat():
            if self.state == NodeState.LEADER and self._running:
                self._broadcast_heartbeat()
                self._heartbeat_timer = threading.Timer(1.0, send_heartbeat)
                self._heartbeat_timer.start()
        
        send_heartbeat()
    
    def _start_heartbeat_check(self) -> None:
        """启动心跳检测"""
        def check_heartbeat():
            if not self._running:
                return
            
            with self._lock:
                # 检查leader是否存活
                if self.leader_id and self.leader_id != self.node_id:
                    leader_alive = False
                    if self.leader_id in self.nodes:
                        leader_alive = self.nodes[self.leader_id].is_alive(timeout=5.0)
                    
                    if not leader_alive:
                        # Leader失效，发起选举
                        self._start_election()
            
            # 继续检测
            threading.Timer(2.0, check_heartbeat).start()
        
        threading.Timer(2.0, check_heartbeat).start()
    
    def _get_higher_id_nodes(self) -> List[str]:
        """获取ID比自己大的节点"""
        return [
            node_id for node_id in self.nodes.keys()
            if self._compare_node_id(node_id, self.node_id) > 0 and node_id != self.node_id
        ]
    
    def _compare_node_id(self, id1: str, id2: str) -> int:
        """比较节点ID，返回1表示id1>id2，-1表示id1<id2，0表示相等"""
        # 首先比较优先级
        p1 = self.nodes.get(id1, NodeInfo(id1, "")).priority if id1 in self.nodes else 0
        p2 = self.nodes.get(id2, NodeInfo(id2, "")).priority if id2 in self.nodes else 0
        
        if p1 != p2:
            return 1 if p1 > p2 else -1
        
        # 优先级相同，比较ID字符串
        if id1 > id2:
            return 1
        elif id1 < id2:
            return -1
        return 0
    
    def _send_election_message(self, target_id: str) -> None:
        """发送选举消息（模拟，实际应通过网络发送）"""
        # 这里只是模拟，实际实现需要网络通信
        pass
    
    def _broadcast_coordinator(self) -> None:
        """广播coordination消息"""
        # 模拟广播
        pass
    
    def _broadcast_heartbeat(self) -> None:
        """广播心跳消息"""
        # 模拟广播
        pass
    
    def get_leader(self) -> Optional[str]:
        """获取当前leader"""
        with self._lock:
            return self.leader_id


class RaftElection(ElectionAlgorithm):
    """
    Raft选举算法实现
    
    算法原理:
    1. 每个节点有一个任期号(term)，单调递增
    2. 节点初始为Follower状态
    3. 如果Follower在超时时间内没有收到leader心跳，变为Candidate
    4. Candidate增加term，向其他节点请求投票
    5. 获得多数票的Candidate成为Leader
    6. Leader定期发送心跳维持权威
    """
    
    def __init__(self, node_id: str, nodes: Dict[str, NodeInfo],
                 min_election_timeout: float = 1.5,
                 max_election_timeout: float = 3.0,
                 heartbeat_interval: float = 0.5):
        super().__init__(node_id, nodes)
        self.min_election_timeout = min_election_timeout
        self.max_election_timeout = max_election_timeout
        self.heartbeat_interval = heartbeat_interval
        
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.vote_count = 0
        self.votes_received: Set[str] = set()
        
        self.commit_index = 0
        self.last_applied = 0
        
        self._election_timer: Optional[threading.Timer] = None
        self._heartbeat_timer: Optional[threading.Timer] = None
        self._random_timeout = self._generate_random_timeout()
        
        self._message_handlers: Dict[str, Callable[[ElectionMessage], Optional[ElectionMessage]]] = {
            "REQUEST_VOTE": self._handle_request_vote,
            "REQUEST_VOTE_RESPONSE": self._handle_vote_response,
            "APPEND_ENTRIES": self._handle_append_entries,
            "APPEND_ENTRIES_RESPONSE": self._handle_append_entries_response
        }
        
        self._vote_records: Dict[int, VoteRecord] = {}
    
    def _generate_random_timeout(self) -> float:
        """生成随机超时时间"""
        return random.uniform(self.min_election_timeout, self.max_election_timeout)
    
    def start(self) -> None:
        """启动Raft选举"""
        with self._lock:
            self._running = True
            self.state = NodeState.FOLLOWER
            self._reset_election_timer()
    
    def stop(self) -> None:
        """停止Raft选举"""
        with self._lock:
            self._running = False
            self.state = NodeState.DOWN
            self._cancel_timers()
    
    def _cancel_timers(self) -> None:
        """取消所有定时器"""
        if self._election_timer:
            self._election_timer.cancel()
        if self._heartbeat_timer:
            self._heartbeat_timer.cancel()
    
    def _reset_election_timer(self) -> None:
        """重置选举定时器"""
        self._cancel_timers()
        self._random_timeout = self._generate_random_timeout()
        self._election_timer = threading.Timer(self._random_timeout, self._on_election_timeout)
        self._election_timer.start()
    
    def _on_election_timeout(self) -> None:
        """选举超时，转换为Candidate"""
        with self._lock:
            if not self._running:
                return
            
            if self.state != NodeState.LEADER:
                self._start_election()
    
    def _start_election(self) -> None:
        """开始选举"""
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.vote_count = 1
        self.votes_received = {self.node_id}
        
        # 重置选举定时器
        self._reset_election_timer()
        
        # 向所有其他节点请求投票
        self._request_votes()
    
    def _request_votes(self) -> None:
        """向其他节点请求投票"""
        for node_id in self.nodes:
            if node_id != self.node_id:
                msg = ElectionMessage(
                    "REQUEST_VOTE",
                    self.node_id,
                    self.current_term,
                    {
                        "candidate_id": self.node_id,
                        "last_log_index": self.commit_index,
                        "last_log_term": self.current_term
                    }
                )
                self._send_message(node_id, msg)
    
    def _handle_request_vote(self, message: ElectionMessage) -> Optional[ElectionMessage]:
        """处理投票请求"""
        with self._lock:
            term = message.term
            candidate_id = message.data.get("candidate_id")
            
            # 如果term小于当前term，拒绝投票
            if term < self.current_term:
                return ElectionMessage(
                    "REQUEST_VOTE_RESPONSE",
                    self.node_id,
                    self.current_term,
                    {"vote_granted": False, "reason": "stale_term"}
                )
            
            # 如果term大于当前term，更新term并转为follower
            if term > self.current_term:
                self.current_term = term
                self.state = NodeState.FOLLOWER
                self.voted_for = None
            
            # 检查是否可以为该candidate投票
            vote_granted = False
            if (self.voted_for is None or self.voted_for == candidate_id):
                # 这里简化处理，实际应检查日志完整性
                vote_granted = True
                self.voted_for = candidate_id
                self._vote_records[term] = VoteRecord(term, candidate_id)
            
            # 重置选举定时器
            self._reset_election_timer()
            
            return ElectionMessage(
                "REQUEST_VOTE_RESPONSE",
                self.node_id,
                self.current_term,
                {"vote_granted": vote_granted}
            )
    
    def _handle_vote_response(self, message: ElectionMessage) -> None:
        """处理投票响应"""
        with self._lock:
            if self.state != NodeState.CANDIDATE:
                return
            
            term = message.term
            
            # 如果收到更高term，转为follower
            if term > self.current_term:
                self.current_term = term
                self.state = NodeState.FOLLOWER
                self.voted_for = None
                self._reset_election_timer()
                return
            
            if message.data.get("vote_granted"):
                self.votes_received.add(message.sender_id)
                
                # 检查是否获得多数票
                majority = (len(self.nodes) // 2) + 1
                if len(self.votes_received) >= majority:
                    self._become_leader()
    
    def _become_leader(self) -> None:
        """成为leader"""
        old_leader = self.leader_id
        self.state = NodeState.LEADER
        self.leader_id = self.node_id
        
        # 取消选举定时器
        if self._election_timer:
            self._election_timer.cancel()
        
        # 启动心跳
        self._start_heartbeat()
        
        if old_leader != self.node_id:
            self._notify_leader_change(old_leader, self.node_id)
    
    def _start_heartbeat(self) -> None:
        """启动leader心跳"""
        def send_heartbeats():
            if self.state == NodeState.LEADER and self._running:
                self._send_append_entries()
                self._heartbeat_timer = threading.Timer(self.heartbeat_interval, send_heartbeats)
                self._heartbeat_timer.start()
        
        send_heartbeats()
    
    def _send_append_entries(self) -> None:
        """发送AppendEntries（心跳或日志复制）"""
        for node_id in self.nodes:
            if node_id != self.node_id:
                msg = ElectionMessage(
                    "APPEND_ENTRIES",
                    self.node_id,
                    self.current_term,
                    {
                        "leader_id": self.node_id,
                        "prev_log_index": self.commit_index,
                        "prev_log_term": self.current_term,
                        "entries": [],
                        "leader_commit": self.commit_index
                    }
                )
                self._send_message(node_id, msg)
    
    def _handle_append_entries(self, message: ElectionMessage) -> Optional[ElectionMessage]:
        """处理AppendEntries"""
        with self._lock:
            term = message.term
            leader_id = message.data.get("leader_id")
            
            # 如果term小于当前term，拒绝
            if term < self.current_term:
                return ElectionMessage(
                    "APPEND_ENTRIES_RESPONSE",
                    self.node_id,
                    self.current_term,
                    {"success": False, "conflict_index": self.commit_index}
                )
            
            # 更新leader信息
            old_leader = self.leader_id
            self.leader_id = leader_id
            
            # 如果term大于当前term，更新term
            if term > self.current_term:
                self.current_term = term
                self.voted_for = None
            
            # 转为follower
            self.state = NodeState.FOLLOWER
            self._reset_election_timer()
            
            # 更新leader节点的心跳时间
            if leader_id in self.nodes:
                self.nodes[leader_id].last_heartbeat = time.time()
            
            if old_leader != leader_id:
                self._notify_leader_change(old_leader, leader_id)
            
            return ElectionMessage(
                "APPEND_ENTRIES_RESPONSE",
                self.node_id,
                self.current_term,
                {"success": True, "match_index": self.commit_index}
            )
    
    def _handle_append_entries_response(self, message: ElectionMessage) -> None:
        """处理AppendEntries响应"""
        with self._lock:
            term = message.term
            
            # 如果收到更高term，转为follower
            if term > self.current_term:
                self.current_term = term
                self.state = NodeState.FOLLOWER
                self.voted_for = None
                self.leader_id = None
                self._reset_election_timer()
    
    def handle_message(self, message: ElectionMessage) -> Optional[ElectionMessage]:
        """处理选举消息"""
        with self._lock:
            handler = self._message_handlers.get(message.msg_type)
            if handler:
                return handler(message)
            return None
    
    def _send_message(self, target_id: str, message: ElectionMessage) -> None:
        """发送消息（模拟，实际应通过网络发送）"""
        # 这里只是模拟，实际实现需要网络通信
        pass
    
    def get_leader(self) -> Optional[str]:
        """获取当前leader"""
        with self._lock:
            return self.leader_id
    
    def get_term(self) -> int:
        """获取当前任期"""
        with self._lock:
            return self.current_term
    
    def is_leader(self) -> bool:
        """检查自己是否是leader"""
        with self._lock:
            return self.state == NodeState.LEADER


class LeaderElectionManager:
    """Leader选举管理器 - 统一管理多种选举算法"""
    
    def __init__(self):
        self.algorithms: Dict[str, ElectionAlgorithm] = {}
        self._lock = threading.RLock()
    
    def create_bully_election(self, election_id: str, node_id: str,
                               nodes: Dict[str, NodeInfo],
                               election_timeout: float = 3.0) -> BullyElection:
        """创建Bully选举实例"""
        with self._lock:
            election = BullyElection(node_id, nodes, election_timeout)
            self.algorithms[election_id] = election
            return election
    
    def create_raft_election(self, election_id: str, node_id: str,
                              nodes: Dict[str, NodeInfo],
                              min_timeout: float = 1.5,
                              max_timeout: float = 3.0) -> RaftElection:
        """创建Raft选举实例"""
        with self._lock:
            election = RaftElection(node_id, nodes, min_timeout, max_timeout)
            self.algorithms[election_id] = election
            return election
    
    def get_election(self, election_id: str) -> Optional[ElectionAlgorithm]:
        """获取选举实例"""
        with self._lock:
            return self.algorithms.get(election_id)
    
    def remove_election(self, election_id: str) -> None:
        """移除选举实例"""
        with self._lock:
            if election_id in self.algorithms:
                self.algorithms[election_id].stop()
                del self.algorithms[election_id]
    
    def get_all_leaders(self) -> Dict[str, Optional[str]]:
        """获取所有选举的leader"""
        with self._lock:
            return {
                eid: alg.get_leader()
                for eid, alg in self.algorithms.items()
            }
    
    def shutdown_all(self) -> None:
        """关闭所有选举"""
        with self._lock:
            for alg in self.algorithms.values():
                alg.stop()
            self.algorithms.clear()


# 辅助函数
def create_node_cluster(node_ids: List[str], base_address: str = "localhost") -> Dict[str, NodeInfo]:
    """创建节点集群"""
    nodes = {}
    for i, node_id in enumerate(node_ids):
        nodes[node_id] = NodeInfo(
            node_id=node_id,
            address=f"{base_address}:{8000 + i}",
            priority=len(node_ids) - i  # ID越小优先级越高
        )
    return nodes
