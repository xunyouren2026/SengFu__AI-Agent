"""
Raft Node - Raft共识节点
基于Raft协议的分布式共识实现

实现完整的Raft共识算法，包括：
- 领导者选举（Leader Election）
- 日志复制（Log Replication）
- 快照（Snapshotting）
- 成员变更（Membership Change）

Raft将共识问题分解为三个子问题：
1. Leader Election：确保任何时候最多有一个Leader
2. Log Replication：Leader将日志复制到所有Follower
3. Safety：如果某个日志条目被提交，那么它一定在所有可用节点的日志中

Author: AGI Unified Framework
"""

import random
import time
import threading
import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from collections import deque


# ============== 角色枚举 ==============

class RaftRole(Enum):
    """
    Raft节点角色

    - FOLLOWER: 跟随者，被动接收Leader的日志
    - CANDIDATE: 候选者，发起选举
    - LEADER: 领导者，处理所有客户端请求
    """
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


# ============== 日志条目 ==============

@dataclass
class LogEntry:
    """
    Raft日志条目

    每个日志条目包含：
    - term: 条目被创建时的任期
    - index: 条目在日志中的索引
    - command: 要执行的命令（状态机指令）
    - timestamp: 创建时间

    日志条目按index严格递增排列。
    """
    term: int
    index: int
    command: Any
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            'term': self.term,
            'index': self.index,
            'command': self.command,
            'timestamp': self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogEntry':
        """从字典反序列化"""
        return cls(
            term=data['term'],
            index=data['index'],
            command=data['command'],
            timestamp=data.get('timestamp', time.time())
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LogEntry):
            return NotImplemented
        return self.term == other.term and self.index == other.index


# ============== Raft配置 ==============

@dataclass
class RaftConfig:
    """
    Raft配置参数

    Attributes:
        election_timeout_min: 选举超时下限（毫秒）
        election_timeout_max: 选举超时上限（毫秒）
        heartbeat_interval: 心跳间隔（毫秒）
        max_entries_per_append: 每次追加的最大条目数
        snapshot_threshold: 触发快照的日志条目数阈值
        rpc_timeout: RPC超时（毫秒）
    """
    election_timeout_min: float = 1500.0   # 1.5秒
    election_timeout_max: float = 3000.0   # 3.0秒
    heartbeat_interval: float = 500.0      # 0.5秒
    max_entries_per_append: int = 100
    snapshot_threshold: int = 1000
    rpc_timeout: float = 500.0


# ============== 选举定时器 ==============

class ElectionTimer:
    """
    选举定时器

    实现Raft的随机化选举超时机制。
    每个Follower在election_timeout内未收到Leader的心跳，
    就会转变为Candidate并发起选举。

    随机化超时的目的是防止多个节点同时发起选举（split vote）。
    """

    def __init__(self, config: RaftConfig):
        self._config = config
        self._deadline: float = 0.0
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """重置定时器（随机化超时时间）"""
        with self._lock:
            timeout = random.uniform(
                self._config.election_timeout_min / 1000.0,
                self._config.election_timeout_max / 1000.0
            )
            self._deadline = time.time() + timeout

    def is_expired(self) -> bool:
        """检查定时器是否已过期"""
        with self._lock:
            return time.time() >= self._deadline

    @property
    def remaining_ms(self) -> float:
        """剩余时间（毫秒）"""
        with self._lock:
            remaining = self._deadline - time.time()
            return max(0.0, remaining * 1000.0)


# ============== 心跳管理器 ==============

class HeartbeatManager:
    """
    心跳管理器

    管理Leader向Follower发送心跳的机制。
    心跳是空的AppendEntries RPC，用于维持Leader的权威。

    Leader在每个heartbeat_interval向所有Follower发送心跳。
    Follower收到心跳后重置选举定时器。
    """

    def __init__(self, config: RaftConfig):
        self._config = config
        self._last_heartbeat: float = 0.0
        self._heartbeat_count: int = 0
        self._lock = threading.Lock()

    def should_send_heartbeat(self) -> bool:
        """是否应该发送心跳"""
        with self._lock:
            elapsed = (time.time() - self._last_heartbeat) * 1000.0
            return elapsed >= self._config.heartbeat_interval

    def record_heartbeat(self) -> None:
        """记录心跳发送"""
        with self._lock:
            self._last_heartbeat = time.time()
            self._heartbeat_count += 1

    def record_received(self) -> None:
        """记录收到心跳"""
        with self._lock:
            self._last_heartbeat = time.time()

    @property
    def heartbeat_count(self) -> int:
        """心跳计数"""
        return self._heartbeat_count


# ============== 日志复制器 ==============

class LogReplicator:
    """
    日志复制器

    管理Leader的日志复制过程。
    Leader为每个Follower维护nextIndex和matchIndex，
    确保所有Follower最终拥有与Leader一致的日志。

    复制过程：
    1. Leader将命令追加到本地日志
    2. 向每个Follower发送AppendEntries RPC
    3. Follower确认后更新matchIndex
    4. 当多数Follower确认后，Leader提交日志
    """

    def __init__(self):
        self._next_index: Dict[str, int] = {}   # follower_id -> next log index
        self._match_index: Dict[str, int] = {}  # follower_id -> highest matched index
        self._lock = threading.Lock()

    def initialize(self, follower_ids: List[str], last_log_index: int) -> None:
        """
        初始化复制状态

        当节点成为Leader时，为每个Follower初始化索引。

        Args:
            follower_ids: Follower ID列表
            last_log_index: Leader的最后日志索引
        """
        with self._lock:
            for fid in follower_ids:
                self._next_index[fid] = last_log_index + 1
                self._match_index[fid] = 0

    def get_next_index(self, follower_id: str) -> int:
        """获取Follower的nextIndex"""
        with self._lock:
            return self._next_index.get(follower_id, 1)

    def set_next_index(self, follower_id: str, index: int) -> None:
        """设置Follower的nextIndex"""
        with self._lock:
            self._next_index[follower_id] = index

    def get_match_index(self, follower_id: str) -> int:
        """获取Follower的matchIndex"""
        with self._lock:
            return self._match_index.get(follower_id, 0)

    def set_match_index(self, follower_id: str, index: int) -> None:
        """设置Follower的matchIndex"""
        with self._lock:
            self._match_index[follower_id] = index

    def advance_next_index(self, follower_id: str) -> None:
        """推进Follower的nextIndex"""
        with self._lock:
            match = self._match_index.get(follower_id, 0)
            self._next_index[follower_id] = match + 1

    def decrement_next_index(self, follower_id: str) -> None:
        """回退Follower的nextIndex（冲突时）"""
        with self._lock:
            current = self._next_index.get(follower_id, 1)
            if current > 1:
                self._next_index[follower_id] = current - 1

    def remove_follower(self, follower_id: str) -> None:
        """移除Follower"""
        with self._lock:
            self._next_index.pop(follower_id, None)
            self._match_index.pop(follower_id, None)

    def get_replication_status(self) -> Dict[str, Dict[str, int]]:
        """获取复制状态"""
        with self._lock:
            return {
                fid: {
                    'next_index': self._next_index.get(fid, 0),
                    'match_index': self._match_index.get(fid, 0)
                }
                for fid in self._next_index
            }


# ============== Raft节点 ==============

class RaftNode:
    """
    Raft共识节点

    实现完整的Raft协议，包括领导者选举、日志复制和快照。

    状态机：
    - 所有节点上的状态机按相同顺序执行相同的命令
    - Leader接收客户端请求，将其作为日志条目追加
    - 日志条目被复制到多数节点后提交
    - 提交的日志条目被应用到状态机

    安全性保证：
    - Election Safety: 每个任期最多一个Leader
    - Leader Append-Only: Leader从不覆盖或删除日志条目
    - Log Matching: 如果两个日志包含相同index和term的条目，则之前的条目都相同
    - Leader Completeness: 如果一个日志条目在某个任期被提交，它一定在所有未来Leader的日志中
    - State Machine Safety: 如果一个节点已将index处的日志条目应用到状态机，
      那么其他任何节点在相同index处不会应用不同的条目

    Author: AGI Unified Framework
    """

    def __init__(self, node_id: str, config: Optional[RaftConfig] = None):
        self._node_id = node_id
        self._config = config or RaftConfig()

        # 持久状态（所有服务器）
        self._current_term: int = 0
        self._voted_for: Optional[str] = None
        self._log: List[LogEntry] = []

        # 易失状态（所有服务器）
        self._commit_index: int = 0
        self._last_applied: int = 0
        self._role: RaftRole = RaftRole.FOLLOWER
        self._leader_id: Optional[str] = None

        # 易失状态（Leader专用）
        self._log_replicator = LogReplicator()

        # 组件
        self._election_timer = ElectionTimer(self._config)
        self._heartbeat_mgr = HeartbeatManager(self._config)

        # 集群成员
        self._peers: Dict[str, 'RaftNode'] = {}

        # 状态机
        self._state_machine: Dict[str, Any] = {}

        # 快照
        self._snapshot: Optional[Dict[str, Any]] = None
        self._snapshot_index: int = 0
        self._snapshot_term: int = 0

        # 投票记录
        self._votes_received: Set[str] = set()

        # 线程和锁
        self._lock = threading.RLock()
        self._running = False
        self._threads: List[threading.Thread] = []

        # 回调
        self._apply_callback: Optional[Callable[[Any], None]] = None
        self._leader_change_callback: Optional[Callable[[Optional[str]], None]] = None

        # 统计
        self._stats = {
            'elections_started': 0,
            'elections_won': 0,
            'terms_served': 0,
            'entries_committed': 0,
            'entries_applied': 0,
            'heartbeats_sent': 0,
            'snapshots_taken': 0
        }

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def role(self) -> RaftRole:
        return self._role

    @property
    def current_term(self) -> int:
        return self._current_term

    @property
    def leader_id(self) -> Optional[str]:
        return self._leader_id

    @property
    def commit_index(self) -> int:
        return self._commit_index

    @property
    def last_log_index(self) -> int:
        """获取最后日志索引"""
        if self._log:
            return self._log[-1].index
        return self._snapshot_index

    @property
    def last_log_term(self) -> int:
        """获取最后日志条目的任期"""
        if self._log:
            return self._log[-1].term
        return self._snapshot_term

    def set_apply_callback(self, callback: Callable[[Any], None]) -> None:
        """设置状态机应用回调"""
        self._apply_callback = callback

    def set_leader_change_callback(self, callback: Callable[[Optional[str]], None]) -> None:
        """设置Leader变更回调"""
        self._leader_change_callback = callback

    def add_peer(self, peer: 'RaftNode') -> None:
        """添加集群成员"""
        with self._lock:
            self._peers[peer.node_id] = peer

    def remove_peer(self, peer_id: str) -> None:
        """移除集群成员"""
        with self._lock:
            self._peers.pop(peer_id, None)
            self._log_replicator.remove_follower(peer_id)

    def start(self) -> None:
        """启动Raft节点"""
        if self._running:
            return

        self._running = True

        # 启动选举超时检测线程
        election_thread = threading.Thread(
            target=self._election_loop,
            daemon=True,
            name=f"raft-election-{self._node_id}"
        )
        election_thread.start()
        self._threads.append(election_thread)

        # 启动心跳线程
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name=f"raft-heartbeat-{self._node_id}"
        )
        heartbeat_thread.start()
        self._threads.append(heartbeat_thread)

        # 启动提交检查线程
        commit_thread = threading.Thread(
            target=self._commit_loop,
            daemon=True,
            name=f"raft-commit-{self._node_id}"
        )
        commit_thread.start()
        self._threads.append(commit_thread)

    def stop(self) -> None:
        """停止Raft节点"""
        self._running = False
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads.clear()

    # ============== 选举 ==============

    def start_election(self) -> Dict[str, Any]:
        """
        发起选举

        当Follower的选举定时器过期时，转变为Candidate并发起选举。
        流程：
        1. 自增current_term
        2. 转变为Candidate
        3. 投票给自己
        4. 向所有Peer发送RequestVote RPC
        5. 如果获得多数票，成为Leader

        Returns:
            选举结果信息
        """
        with self._lock:
            self._current_term += 1
            self._role = RaftRole.CANDIDATE
            self._voted_for = self._node_id
            self._votes_received = {self._node_id}
            self._leader_id = None

            self._stats['elections_started'] += 1
            self._stats['terms_served'] += 1

            self._election_timer.reset()

            term = self._current_term
            last_log_index = self.last_log_index
            last_log_term = self.last_log_term

        # 向所有Peer请求投票
        votes_needed = (len(self._peers) + 1) // 2 + 1

        for peer_id, peer in self._peers.items():
            try:
                granted = peer.vote(term, self._node_id,
                                   last_log_index, last_log_term)
                if granted:
                    with self._lock:
                        self._votes_received.add(peer_id)

                        if len(self._votes_received) >= votes_needed:
                            self._become_leader()
                            return {
                                'won': True,
                                'term': self._current_term,
                                'votes': len(self._votes_received)
                            }
            except Exception:
                continue

        return {
            'won': False,
            'term': term,
            'votes': len(self._votes_received)
        }

    def vote(self, term: int, candidate_id: str,
             last_log_index: int, last_log_term: int) -> bool:
        """
        处理投票请求

        如果候选者的日志至少与自己一样新，则授予投票。

        Args:
            term: 候选者的任期
            candidate_id: 候选者ID
            last_log_index: 候选者的最后日志索引
            last_log_term: 候选者的最后日志任期

        Returns:
            是否授予投票
        """
        with self._lock:
            # 如果对方的term更大，更新自己的term并转为Follower
            if term > self._current_term:
                self._current_term = term
                self._voted_for = None
                self._role = RaftRole.FOLLOWER
                self._leader_id = None

            # 如果term小于当前term，拒绝
            if term < self._current_term:
                return False

            # 如果已经投票给其他候选人，拒绝
            if self._voted_for is not None and self._voted_for != candidate_id:
                return False

            # 检查候选者的日志是否至少和自己一样新
            if last_log_term < self.last_log_term:
                return False
            if (last_log_term == self.last_log_term and
                    last_log_index < self.last_log_index):
                return False

            # 授予投票
            self._voted_for = candidate_id
            self._election_timer.reset()

            return True

    def _become_leader(self) -> None:
        """成为Leader"""
        with self._lock:
            if self._role != RaftRole.CANDIDATE:
                return

            self._role = RaftRole.LEADER
            self._leader_id = self._node_id

            self._stats['elections_won'] += 1

            # 初始化复制状态
            follower_ids = list(self._peers.keys())
            self._log_replicator.initialize(follower_ids, self.last_log_index)

            # 立即发送心跳
            self._heartbeat_mgr.record_heartbeat()

        if self._leader_change_callback:
            self._leader_change_callback(self._node_id)

    def _step_down(self, new_term: int) -> None:
        """降级为Follower"""
        with self._lock:
            if new_term > self._current_term:
                self._current_term = new_term
                self._role = RaftRole.FOLLOWER
                self._voted_for = None
                self._leader_id = None
                self._votes_received.clear()

    # ============== 日志复制 ==============

    def append_entries(self, term: int, leader_id: str,
                       prev_log_index: int, prev_log_term: int,
                       entries: List[LogEntry],
                       leader_commit: int) -> Dict[str, Any]:
        """
        处理AppendEntries RPC

        Leader发送日志条目给Follower。
        Follower验证并追加日志条目。

        Args:
            term: Leader的任期
            leader_id: Leader ID
            prev_log_index: 前一条日志的索引
            prev_log_term: 前一条日志的任期
            entries: 要追加的日志条目
            leader_commit: Leader的commit_index

        Returns:
            (success, current_term, match_index)
        """
        with self._lock:
            # 任期检查
            if term < self._current_term:
                return {
                    'success': False,
                    'term': self._current_term,
                    'match_index': 0
                }

            # 更新term和角色
            if term >= self._current_term:
                self._current_term = term
                self._role = RaftRole.FOLLOWER
                self._voted_for = None
                self._leader_id = leader_id

            # 重置选举定时器
            self._election_timer.reset()

            # 检查前一条日志是否匹配
            if prev_log_index > 0:
                log_entry = self._get_log_entry(prev_log_index)
                if log_entry is None or log_entry.term != prev_log_term:
                    return {
                        'success': False,
                        'term': self._current_term,
                        'match_index': self.last_log_index
                    }

            # 追加新日志条目（可能需要覆盖冲突条目）
            for entry in entries:
                existing = self._get_log_entry(entry.index)
                if existing is not None:
                    if existing.term != entry.term:
                        # 冲突：删除从entry.index开始的所有条目
                        self._log = [
                            e for e in self._log if e.index < entry.index
                        ]
                        self._log.append(entry)
                    # 如果term相同，跳过（已存在）
                else:
                    self._log.append(entry)

            # 更新commit_index
            if leader_commit > self._commit_index:
                self._commit_index = min(
                    leader_commit,
                    self.last_log_index
                )
                self._apply_committed()

            return {
                'success': True,
                'term': self._current_term,
                'match_index': self.last_log_index
            }

    def _get_log_entry(self, index: int) -> Optional[LogEntry]:
        """获取指定索引的日志条目"""
        for entry in self._log:
            if entry.index == index:
                return entry
        return None

    def propose(self, command: Any) -> bool:
        """
        提交命令（客户端接口）

        只有Leader可以接受客户端请求。

        Args:
            command: 要执行的命令

        Returns:
            是否成功提交
        """
        with self._lock:
            if self._role != RaftRole.LEADER:
                return False

            # 追加到本地日志
            entry = LogEntry(
                term=self._current_term,
                index=self.last_log_index + 1,
                command=command
            )
            self._log.append(entry)

        # 复制到Follower
        self._replicate_log()

        return True

    def _replicate_log(self) -> None:
        """复制日志到所有Follower"""
        with self._lock:
            if self._role != RaftRole.LEADER:
                return

            for peer_id, peer in self._peers.items():
                next_idx = self._log_replicator.get_next_index(peer_id)
                prev_idx = next_idx - 1

                # 获取前一条日志的term
                prev_entry = self._get_log_entry(prev_idx)
                prev_term = prev_entry.term if prev_entry else 0

                # 获取要发送的日志条目
                entries = [
                    e for e in self._log if e.index >= next_idx
                ][:self._config.max_entries_per_append]

                try:
                    result = peer.append_entries(
                        term=self._current_term,
                        leader_id=self._node_id,
                        prev_log_index=prev_idx,
                        prev_log_term=prev_term,
                        entries=entries,
                        leader_commit=self._commit_index
                    )

                    if result['success']:
                        match_idx = result.get('match_index', 0)
                        self._log_replicator.set_match_index(
                            peer_id, match_idx
                        )
                        self._log_replicator.advance_next_index(peer_id)
                    else:
                        self._log_replicator.decrement_next_index(peer_id)

                except Exception:
                    continue

    def _advance_commit_index(self) -> None:
        """推进commit_index"""
        with self._lock:
            if self._role != RaftRole.LEADER:
                return

            for n in range(self.last_log_index, self._commit_index, -1):
                entry = self._get_log_entry(n)
                if entry is None or entry.term != self._current_term:
                    continue

                # 统计有多少Follower已复制此条目
                replicated = 1  # Leader自己
                for peer_id in self._peers:
                    if self._log_replicator.get_match_index(peer_id) >= n:
                        replicated += 1

                # 多数确认
                if replicated > (len(self._peers) + 1) // 2:
                    self._commit_index = n
                    self._apply_committed()
                    break

    def _apply_committed(self) -> None:
        """应用已提交的日志到状态机"""
        while self._last_applied < self._commit_index:
            self._last_applied += 1
            entry = self._get_log_entry(self._last_applied)

            if entry is None:
                continue

            # 应用到状态机
            if isinstance(entry.command, dict) and 'key' in entry.command:
                key = entry.command['key']
                value = entry.command.get('value')
                if value is not None:
                    self._state_machine[key] = value
                elif key in self._state_machine:
                    del self._state_machine[key]

            self._stats['entries_applied'] += 1

            if self._apply_callback:
                try:
                    self._apply_callback(entry.command)
                except Exception:
                    pass

    # ============== 快照 ==============

    def install_snapshot(self, term: int, leader_id: str,
                         snapshot_index: int, snapshot_term: int,
                         snapshot_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        安装快照

        当Follower落后太多时，Leader发送快照而不是逐条复制。

        Args:
            term: Leader的任期
            leader_id: Leader ID
            snapshot_index: 快照的最后日志索引
            snapshot_term: 快照的最后日志任期
            snapshot_data: 快照数据

        Returns:
            安装结果
        """
        with self._lock:
            if term < self._current_term:
                return {'term': self._current_term, 'success': False}

            self._current_term = term
            self._role = RaftRole.FOLLOWER
            self._leader_id = leader_id

            # 丢弃快照覆盖的所有日志条目
            self._log = [
                e for e in self._log if e.index > snapshot_index
            ]

            # 安装快照
            self._snapshot = snapshot_data
            self._snapshot_index = snapshot_index
            self._snapshot_term = snapshot_term
            self._state_machine = dict(snapshot_data)

            if snapshot_index > self._commit_index:
                self._commit_index = snapshot_index
            if snapshot_index > self._last_applied:
                self._last_applied = snapshot_index

            self._stats['snapshots_taken'] += 1

            return {'term': self._current_term, 'success': True}

    def take_snapshot(self) -> Optional[Dict[str, Any]]:
        """
        创建快照

        当日志条目数超过阈值时，创建快照以减少日志大小。

        Returns:
            快照数据
        """
        with self._lock:
            if self._commit_index <= self._snapshot_index:
                return None

            # 只快照已提交的条目
            snapshot_data = dict(self._state_machine)
            entry = self._get_log_entry(self._commit_index)

            self._snapshot = snapshot_data
            self._snapshot_index = self._commit_index
            self._snapshot_term = entry.term if entry else 0

            # 丢弃已快照的日志条目
            self._log = [
                e for e in self._log if e.index > self._commit_index
            ]

            return snapshot_data

    def commit(self) -> int:
        """
        手动触发提交检查

        Returns:
            当前commit_index
        """
        self._advance_commit_index()
        return self._commit_index

    # ============== 后台循环 ==============

    def _election_loop(self) -> None:
        """选举超时检测循环"""
        while self._running:
            time.sleep(0.05)  # 50ms检查间隔

            with self._lock:
                if self._role == RaftRole.LEADER:
                    continue

                if self._election_timer.is_expired():
                    self.start_election()

    def _heartbeat_loop(self) -> None:
        """心跳发送循环"""
        while self._running:
            time.sleep(0.05)

            with self._lock:
                if self._role != RaftRole.LEADER:
                    continue

                if self._heartbeat_mgr.should_send_heartbeat():
                    self._heartbeat_mgr.record_heartbeat()
                    self._stats['heartbeats_sent'] += 1

            # 发送心跳（空AppendEntries）
            self._replicate_log()

    def _commit_loop(self) -> None:
        """提交检查循环"""
        while self._running:
            time.sleep(0.1)
            try:
                self._advance_commit_index()

                # 检查是否需要快照
                with self._lock:
                    log_size = len(self._log)
                if log_size >= self._config.snapshot_threshold:
                    self.take_snapshot()

            except Exception:
                continue

    # ============== 状态查询 ==============

    def get_status(self) -> Dict[str, Any]:
        """获取节点状态"""
        with self._lock:
            return {
                'node_id': self._node_id,
                'role': self._role.value,
                'term': self._current_term,
                'leader': self._leader_id,
                'voted_for': self._voted_for,
                'log_length': len(self._log),
                'commit_index': self._commit_index,
                'last_applied': self._last_applied,
                'last_log_index': self.last_log_index,
                'last_log_term': self.last_log_term,
                'peers': len(self._peers),
                'state_machine_size': len(self._state_machine),
                'snapshot_index': self._snapshot_index,
                'stats': dict(self._stats)
            }

    def get_log(self, start_index: int = 0,
                limit: int = 100) -> List[LogEntry]:
        """获取日志条目"""
        with self._lock:
            entries = [e for e in self._log if e.index >= start_index]
            return entries[:limit]

    def get_state(self, key: str, default: Any = None) -> Any:
        """查询状态机"""
        with self._lock:
            return self._state_machine.get(key, default)


# ============== Raft集群 ==============

class RaftCluster:
    """
    Raft集群管理

    管理多个Raft节点组成的集群，提供便捷的集群操作接口。

    功能：
    - 创建集群
    - 添加/移除节点
    - 提交命令
    - 查询集群状态
    - 模拟网络分区

    Author: AGI Unified Framework
    """

    def __init__(self, node_count: int = 5,
                 config: Optional[RaftConfig] = None):
        self._config = config or RaftConfig()
        self._nodes: Dict[str, RaftNode] = {}
        self._lock = threading.Lock()

        # 创建节点
        for i in range(node_count):
            node_id = f"node_{i}"
            node = RaftNode(node_id, self._config)
            self._nodes[node_id] = node

        # 建立对等连接
        for node_id, node in self._nodes.items():
            for other_id, other in self._nodes.items():
                if node_id != other_id:
                    node.add_peer(other)

    @property
    def nodes(self) -> Dict[str, RaftNode]:
        return self._nodes

    def start_all(self) -> None:
        """启动所有节点"""
        for node in self._nodes.values():
            node.start()

    def stop_all(self) -> None:
        """停止所有节点"""
        for node in self._nodes.values():
            node.stop()

    def get_leader(self) -> Optional[RaftNode]:
        """获取当前Leader"""
        for node in self._nodes.values():
            if node.role == RaftRole.LEADER:
                return node
        return None

    def submit(self, key: str, value: Any) -> bool:
        """
        向集群提交命令

        Args:
            key: 状态键
            value: 状态值

        Returns:
            是否成功
        """
        leader = self.get_leader()
        if leader is None:
            return False
        return leader.propose({'key': key, 'value': value})

    def query(self, key: str, default: Any = None) -> Any:
        """查询状态（从Leader读取）"""
        leader = self.get_leader()
        if leader is None:
            return default
        return leader.get_state(key, default)

    def add_node(self, node_id: str) -> RaftNode:
        """添加新节点到集群"""
        node = RaftNode(node_id, self._config)
        with self._lock:
            self._nodes[node_id] = node
            for other_id, other in self._nodes.items():
                if other_id != node_id:
                    node.add_peer(other)
                    other.add_peer(node)
        return node

    def remove_node(self, node_id: str) -> None:
        """从集群移除节点"""
        with self._lock:
            node = self._nodes.pop(node_id, None)
            if node:
                node.stop()
                for other in self._nodes.values():
                    other.remove_peer(node_id)

    def simulate_partition(self, group_a: List[str],
                           group_b: List[str]) -> None:
        """
        模拟网络分区

        Args:
            group_a: 分区A的节点列表
            group_b: 分区B的节点列表
        """
        with self._lock:
            for node_id_a in group_a:
                for node_id_b in group_b:
                    node_a = self._nodes.get(node_id_a)
                    node_b = self._nodes.get(node_id_b)
                    if node_a:
                        node_a.remove_peer(node_id_b)
                    if node_b:
                        node_b.remove_peer(node_id_a)

    def heal_partition(self) -> None:
        """修复网络分区"""
        with self._lock:
            for node_id, node in self._nodes.items():
                for other_id, other in self._nodes.items():
                    if node_id != other_id:
                        node.add_peer(other)

    def get_cluster_status(self) -> Dict[str, Any]:
        """获取集群状态"""
        roles: Dict[str, int] = {}
        for node in self._nodes.values():
            role = node.role.value
            roles[role] = roles.get(role, 0) + 1

        leader = self.get_leader()

        return {
            'node_count': len(self._nodes),
            'roles': roles,
            'leader': leader.node_id if leader else None,
            'term': leader.current_term if leader else 0,
            'nodes': {
                nid: node.get_status()
                for nid, node in self._nodes.items()
            }
        }


# ============== 主程序入口 ==============

if __name__ == "__main__":
    print("=== Raft Consensus Demo ===\n")

    # 创建集群
    cluster = RaftCluster(node_count=5)
    cluster.start_all()

    # 等待选举完成
    print("Waiting for election...")
    time.sleep(3)

    # 查看集群状态
    status = cluster.get_cluster_status()
    print(f"Leader: {status['leader']}")
    print(f"Roles: {status['roles']}")

    # 提交命令
    if status['leader']:
        for i in range(10):
            success = cluster.submit(f"key_{i}", f"value_{i}")
            print(f"Submit key_{i}: {success}")
            time.sleep(0.1)

        # 等待复制
        time.sleep(1)

        # 查询
        for i in range(10):
            value = cluster.query(f"key_{i}")
            print(f"Query key_{i}: {value}")

    # 最终状态
    final_status = cluster.get_cluster_status()
    print(f"\nFinal state:")
    for nid, node_status in final_status['nodes'].items():
        print(f"  {nid}: role={node_status['role']}, "
              f"log={node_status['log_length']}, "
              f"commit={node_status['commit_index']}")

    cluster.stop_all()
    print("\n=== Demo Complete ===")
