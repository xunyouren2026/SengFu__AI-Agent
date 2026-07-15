"""
Decentralized Consensus Mechanisms
去中心化共识机制 - 实现分布式系统的一致性

This module implements various consensus mechanisms inspired by historical
governance systems and modern distributed algorithms:

1. PBFT (Practical Byzantine Fault Tolerance) - 实用拜占庭容错
2. Raft Consensus - Raft共识算法
3. Proof of Stake (PoS) - 权益证明
4. Feudal Consensus - 分封制共识（原创设计）

Author: AGI Unified Framework
"""

import random
import hashlib
import time
import threading
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict, deque
from enum import Enum


# ============== 枚举和配置 ==============

class ConsensusType(Enum):
    """共识协议类型"""
    PBFT = "pbft"
    RAFT = "raft"
    POS = "pos"
    FEUDAL = "feudal"


class NodeRole(Enum):
    """节点角色"""
    LEADER = "leader"
    CANDIDATE = "candidate"
    FOLLOWER = "follower"
    VALIDATOR = "validator"


@dataclass
class Vote:
    """
    投票数据结构
    
    用于各种共识协议中的投票
    """
    voter_id: str
    candidate: str
    round: int
    timestamp: float = field(default_factory=time.time)
    signature: str = ""  # 投票签名
    
    def is_valid(self, current_round: int, timeout: float = 10.0) -> bool:
        """检查投票是否有效"""
        return (
            self.round == current_round and
            (time.time() - self.timestamp) < timeout
        )


@dataclass
class ConsensusConfig:
    """
    共识配置
    """
    consensus_type: ConsensusType = ConsensusType.PBFT
    timeout: float = 5.0          # 共识超时时间
    max_retry: int = 3             # 最大重试次数
    min_validators: int = 4        # 最小验证者数量
    byzantine_tolerance: float = 0.33  # 拜占庭容错比例


@dataclass
class Block:
    """
    区块链中的区块
    
    用于PoS等基于区块链的共识机制
    """
    index: int
    timestamp: float
    data: Any
    prev_hash: str
    hash: str = ""
    validator_id: str = ""
    signatures: List[str] = field(default_factory=list)
    
    def calculate_hash(self) -> str:
        """计算区块哈希"""
        content = f"{self.index}{self.timestamp}{json.dumps(self.data, sort_keys=True)}{self.prev_hash}"
        return hashlib.sha256(content.encode()).hexdigest()


# ============== PBFT协议 ==============

class PBFTProtocol:
    """
    Practical Byzantine Fault Tolerance (PBFT) 协议
    
    PBFT是一种拜占庭容错共识协议，能够在最多f个节点出现
    拜占庭故障（恶意或任意错误）时保持系统正确运行。
    
    协议流程（三阶段协议）：
    1. Pre-prepare: 主节点广播预准备消息
    2. Prepare: 所有节点广播准备消息（需2f+1个确认）
    3. Commit: 所有节点广播提交消息（需2f+1个确认）
    4. 执行: 消息被确认后执行
    
    历史类比：类似贵族议事会，需要多数贵族的同意才能通过法令
    """
    
    QUORUM_RATIO = 2 / 3  # 需要2/3节点的确认
    
    def __init__(self, node_id: str, all_nodes: List[str]):
        self.node_id = node_id
        self.all_nodes = set(all_nodes)
        self._num_nodes = len(all_nodes)
        self._f = (self._num_nodes - 1) // 3  # 可容忍的拜占庭节点数
        
        # 当前视图（view）
        self._view = 0
        self._sequence = 0  # 序列号
        
        # 主节点选择
        self._primary = all_nodes[0] if all_nodes else node_id
        self._is_primary = (node_id == self._primary)
        
        # 消息日志
        self._preprepare_log: Dict[int, dict] = {}  # sequence -> pre-prepare message
        self._prepare_log: Dict[int, List[Vote]] = {}  # sequence -> prepare votes
        self._commit_log: Dict[int, List[Vote]] = {}  # sequence -> commit votes
        self._replies: Dict[str, Any] = {}  # client_id -> reply
        
        # 已确认的消息
        self._executed: Set[int] = set()
        
        # 锁
        self._lock = threading.Lock()
        
        # 回调函数
        self._execute_callback: Optional[callable] = None
        
    def set_execute_callback(self, callback: callable) -> None:
        """设置执行回调"""
        self._execute_callback = callback
    
    def _get_primary(self, view: int) -> str:
        """根据视图选择主节点"""
        if not self.all_nodes:
            return self.node_id
        return list(self.all_nodes)[view % len(self.all_nodes)]
    
    def _quorum(self, count: int) -> bool:
        """检查是否达到法定人数"""
        return count >= 2 * self._f + 1
    
    def _quorum_count(self) -> int:
        """返回法定人数数量"""
        return 2 * self._f + 1
    
    def pre_prepare(self, msg: Any, view: int = None) -> Tuple[dict, List[str]]:
        """
        预准备阶段
        
        主节点收到客户端请求后，分配序列号并广播预准备消息
        """
        if view is None:
            view = self._view
        
        # 只有主节点可以发起预准备
        if not self._is_primary:
            return {}, []
        
        with self._lock:
            self._sequence += 1
            seq = self._sequence
            
            # 创建预准备消息
            msg_digest = self._digest(msg)
            preprepare_msg = {
                'view': view,
                'sequence': seq,
                'digest': msg_digest,
                'message': msg,
                'timestamp': time.time()
            }
            
            self._preprepare_log[seq] = preprepare_msg
            
            # 初始化日志
            if seq not in self._prepare_log:
                self._prepare_log[seq] = []
            if seq not in self._commit_log:
                self._commit_log[seq] = []
            
            return preprepare_msg, []
    
    def prepare(self, node_id: str, msg_digest: str, view: int, 
                sequence: int) -> Optional[dict]:
        """
        准备阶段
        
        节点收到预准备消息后，验证并广播准备消息
        """
        with self._lock:
            # 验证预准备消息
            if view != self._view:
                return None
            
            if sequence not in self._preprepare_log:
                # 需要请求预准备消息
                return None
            
            pp_msg = self._preprepare_log[sequence]
            if pp_msg['digest'] != msg_digest:
                return None
            
            # 创建准备消息
            vote = Vote(
                voter_id=node_id,
                candidate=msg_digest,
                round=sequence,
                timestamp=time.time()
            )
            
            self._prepare_log[sequence].append(vote)
            
            # 检查是否收到足够的准备消息
            prepare_count = len(self._prepare_log[sequence])
            
            prepare_msg = {
                'type': 'prepare',
                'view': view,
                'sequence': sequence,
                'digest': msg_digest,
                'voter': node_id,
                'timestamp': time.time()
            }
            
            return prepare_msg if self._quorum(prepare_count + 1) else None
    
    def commit(self, node_id: str, msg_digest: str, view: int,
               sequence: int) -> Optional[dict]:
        """
        提交阶段
        
        节点收到足够的准备消息后，广播提交消息
        """
        with self._lock:
            # 验证准备阶段完成
            if sequence not in self._prepare_log:
                return None
            
            # 检查是否有足够的准备消息
            if len(self._prepare_log[sequence]) < self._quorum_count():
                return None
            
            # 创建提交消息
            vote = Vote(
                voter_id=node_id,
                candidate=msg_digest,
                round=sequence,
                timestamp=time.time()
            )
            
            self._commit_log[sequence].append(vote)
            
            # 检查是否收到足够的提交消息
            commit_count = len(self._commit_log[sequence])
            
            commit_msg = {
                'type': 'commit',
                'view': view,
                'sequence': sequence,
                'digest': msg_digest,
                'voter': node_id,
                'timestamp': time.time()
            }
            
            return commit_msg if self._quorum(commit_count + 1) else None
    
    def receive_preprepare(self, msg: dict) -> bool:
        """接收预准备消息"""
        with self._lock:
            view = msg['view']
            seq = msg['sequence']
            
            if view != self._view:
                return False
            
            self._preprepare_log[seq] = msg
            
            if seq not in self._prepare_log:
                self._prepare_log[seq] = []
            if seq not in self._commit_log:
                self._commit_log[seq] = []
            
            return True
    
    def receive_prepare(self, vote: Vote) -> bool:
        """接收准备消息"""
        with self._lock:
            seq = vote.round
            
            if seq not in self._prepare_log:
                self._prepare_log[seq] = []
            
            # 检查是否已存在
            for v in self._prepare_log[seq]:
                if v.voter_id == vote.voter_id:
                    return False
            
            self._prepare_log[seq].append(vote)
            
            # 检查是否达到提交条件
            return len(self._prepare_log[seq]) >= self._quorum_count()
    
    def receive_commit(self, vote: Vote) -> bool:
        """接收提交消息"""
        with self._lock:
            seq = vote.round
            
            if seq not in self._commit_log:
                self._commit_log[seq] = []
            
            for v in self._commit_log[seq]:
                if v.voter_id == vote.voter_id:
                    return False
            
            self._commit_log[seq].append(vote)
            
            # 检查是否可以执行
            if len(self._commit_log[seq]) >= self._quorum_count() and seq not in self._executed:
                self._executed.add(seq)
                
                # 执行消息
                if seq in self._preprepare_log:
                    msg = self._preprepare_log[seq]['message']
                    self._execute(msg)
                
                return True
            
            return False
    
    def _execute(self, msg: Any) -> Any:
        """执行已共识的消息"""
        if self._execute_callback:
            return self._execute_callback(msg)
        return msg
    
    def _digest(self, msg: Any) -> str:
        """计算消息摘要"""
        content = json.dumps(msg, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def execute(self, msg: Any) -> Any:
        """
        执行消息（完整PBFT流程）
        
        这是对外的主要接口
        """
        # 如果是主节点，发起预准备
        if self._is_primary:
            preprepare, _ = self.pre_prepare(msg)
            # 返回准备好的消息供广播
            return preprepare
        
        return None
    
    def view_change(self, new_view: int) -> dict:
        """
        视图变更
        
        当主节点故障时，触发视图变更
        """
        with self._lock:
            self._view = new_view
            self._primary = self._get_primary(new_view)
            self._is_primary = (self.node_id == self._primary)
            
            # 重置日志
            self._prepare_log.clear()
            self._commit_log.clear()
            
            return {
                'type': 'view_change',
                'old_view': new_view - 1,
                'new_view': new_view,
                'node_id': self.node_id,
                'timestamp': time.time()
            }
    
    def _select_primary(self, view: int) -> str:
        """选择新主节点"""
        return self._get_primary(view)
    
    def get_status(self) -> dict:
        """获取PBFT状态"""
        with self._lock:
            return {
                'node_id': self.node_id,
                'view': self._view,
                'is_primary': self._is_primary,
                'primary': self._primary,
                'sequence': self._sequence,
                'executed_count': len(self._executed),
                'prepare_log_size': len(self._prepare_log),
                'commit_log_size': len(self._commit_log)
            }


# ============== Raft共识 ==============

class RaftConsensus:
    """
    Raft共识算法
    
    Raft是一种用于管理复制状态机的共识算法，
    设计的目标是易于理解。
    
    主要组件：
    - Leader Election: 领导者选举
    - Log Replication: 日志复制
    - Safety: 安全性保证
    
    角色：Leader（领导者）、Candidate（候选者）、Follower（跟随者）
    
    历史类比：类似王位继承，需要获得多数领主的支持才能成为国王
    """
    
    ELECTION_TIMEOUT_MIN = 1.5  # 选举超时最小值（秒）
    ELECTION_TIMEOUT_MAX = 3.0  # 选举超时最大值（秒）
    HEARTBEAT_INTERVAL = 0.5    # 心跳间隔（秒）
    
    def __init__(self, node_id: str, all_nodes: List[str]):
        self.node_id = node_id
        self.all_nodes = set(all_nodes)
        self._num_nodes = len(all_nodes)
        
        # 角色和任期
        self._role: NodeRole = NodeRole.FOLLOWER
        self._term = 0
        self._voted_for: Optional[str] = None
        
        # 日志
        self._log: List[Tuple[int, Any]] = []  # (term, command)
        self._commit_index = -1
        self._last_applied = -1
        
        # 领导者信息
        self._leader_id: Optional[str] = None
        
        # 选举相关
        self._last_heartbeat = time.time()
        self._election_timeout = random.uniform(
            self.ELECTION_TIMEOUT_MIN,
            self.ELECTION_TIMEOUT_MAX
        )
        
        # 复制状态
        self._next_index: Dict[str, int] = {}  # 每个跟随者的下一个日志索引
        self._match_index: Dict[str, int] = {}  # 每个跟随者已复制的最高日志索引
        
        # 锁
        self._lock = threading.Lock()
        
        # 回调
        self._apply_callback: Optional[callable] = None
    
    def set_apply_callback(self, callback: callable) -> None:
        """设置应用回调"""
        self._apply_callback = callback
    
    def reset_election_timer(self) -> None:
        """重置选举计时器"""
        self._last_heartbeat = time.time()
    
    def _check_election_timeout(self) -> bool:
        """检查是否选举超时"""
        elapsed = time.time() - self._last_heartbeat
        return elapsed > self._election_timeout
    
    def request_vote(self, candidate_id: str, last_log_index: int, 
                    last_log_term: int) -> Tuple[bool, int]:
        """
        处理投票请求
        
        Args:
            candidate_id: 候选者ID
            last_log_index: 候选者最后日志索引
            last_log_term: 候选者最后日志任期
            
        Returns:
            (granted, current_term)
        """
        with self._lock:
            current_term = self._term
            granted = False
            
            # 任期小于当前，不投票
            if candidate_id == self.node_id:
                return True, current_term
            
            # 已经投票给其他人
            if self._voted_for and self._voted_for != candidate_id:
                return False, current_term
            
            # 检查候选者日志是否至少与本地一样新
            last_local_term = 0
            if self._log:
                last_local_term = self._log[-1][0]
            
            if last_log_term < last_local_term:
                return False, current_term
            
            if last_log_term == last_local_term and last_log_index < len(self._log) - 1:
                return False, current_term
            
            # 授予投票
            granted = True
            self._voted_for = candidate_id
            self.reset_election_timer()
            
            return granted, current_term
    
    def append_entries(self, entries: List[Tuple[int, Any]], 
                      prev_log_index: int, prev_log_term: int,
                      leader_commit: int) -> Tuple[bool, int]:
        """
        处理日志追加请求
        
        Args:
            entries: 要追加的日志条目
            prev_log_index: 前一日志索引
            prev_log_term: 前一日志任期
            leader_commit: 领导者已提交的索引
            
        Returns:
            (success, current_term)
        """
        with self._lock:
            current_term = self._term
            
            # 任期太小，拒绝
            if self._term > current_term:
                return False, self._term
            
            # 如果是跟随者，重置选举计时器
            if self._role != NodeRole.LEADER:
                self._role = NodeRole.FOLLOWER
                self._leader_id = None
            self.reset_election_timer()
            
            # 检查前一条日志
            if prev_log_index >= 0:
                if prev_log_index >= len(self._log):
                    return False, self._term
                if self._log[prev_log_index][0] != prev_log_term:
                    return False, self._term
            
            # 追加新日志
            for i, (term, command) in enumerate(entries):
                log_index = prev_log_index + 1 + i
                
                if log_index < len(self._log):
                    # 冲突的日志替换
                    if self._log[log_index][0] != term:
                        self._log = self._log[:log_index]
                        self._log.append((term, command))
                else:
                    self._log.append((term, command))
            
            # 更新提交索引
            if leader_commit > self._commit_index:
                self._commit_index = min(leader_commit, len(self._log) - 1)
                self._apply_committed()
            
            return True, self._term
    
    def _apply_committed(self) -> None:
        """应用已提交的日志"""
        while self._last_applied < self._commit_index:
            self._last_applied += 1
            term, command = self._log[self._last_applied]
            
            if self._apply_callback:
                self._apply_callback(command)
    
    def start_election(self) -> Tuple[bool, int]:
        """
        开始选举
        
        跟随者超时后成为候选者，请求其他节点投票
        """
        with self._lock:
            self._role = NodeRole.CANDIDATE
            self._term += 1
            self._voted_for = self.node_id
            
            last_log_index = len(self._log) - 1
            last_log_term = self._log[-1][0] if self._log else 0
            
            return True, self._term
    
    def become_leader(self) -> bool:
        """
        成为领导者
        
        获得多数票后成为领导者
        """
        with self._lock:
            if self._role != NodeRole.CANDIDATE:
                return False
            
            # 检查是否获得多数票
            votes = 1  # 自己的票
            # 这里简化处理，实际需要统计其他节点的投票
            
            if votes >= self._num_nodes // 2 + 1:
                self._role = NodeRole.LEADER
                self._leader_id = self.node_id
                
                # 初始化复制状态
                for node in self.all_nodes:
                    if node != self.node_id:
                        self._next_index[node] = len(self._log)
                        self._match_index[node] = 0
                
                return True
            
            return False
    
    def _send_heartbeat(self) -> None:
        """发送心跳"""
        if self._role != NodeRole.LEADER:
            return
        
        with self._lock:
            for node in self.all_nodes:
                if node != self.node_id:
                    # 发送心跳（空日志条目）
                    prev_log_index = len(self._log) - 1
                    prev_log_term = self._log[-1][0] if self._log else 0
                    
                    # 构造心跳消息，包含当前任期和领导者提交索引
                    heartbeat_msg = {
                        'term': self._term,
                        'leader_id': self.node_id,
                        'prev_log_index': prev_log_index,
                        'prev_log_term': prev_log_term,
                        'leader_commit': self._commit_index,
                        'entries': [],  # 心跳为空日志条目
                        'timestamp': time.time()
                    }
                    
                    # 模拟发送心跳到跟随者
                    # 实际实现中通过网络将 heartbeat_msg 发送到 node
                    # 收到心跳的跟随者会重置选举计时器
                    pass
    
    def _replicate_log(self, follower_id: str) -> bool:
        """
        复制日志到跟随者
        
        返回是否复制成功
        """
        with self._lock:
            if self._role != NodeRole.LEADER:
                return False
            
            next_idx = self._next_index.get(follower_id, 0)
            
            if next_idx > len(self._log):
                return False
            
            # 获取要复制的日志
            entries = self._log[next_idx:]
            prev_log_index = next_idx - 1
            prev_log_term = self._log[prev_log_index][0] if prev_log_index >= 0 and self._log else 0
            
            # 模拟复制
            return True
    
    def submit_command(self, command: Any) -> bool:
        """
        提交命令
        
        只有领导者可以提交新命令
        """
        if self._role != NodeRole.LEADER:
            return False
        
        with self._lock:
            # 添加到本地日志
            self._log.append((self._term, command))
            
            # 复制到所有跟随者
            success_count = 1  # 领导者自己的日志
            for node in self.all_nodes:
                if node != self.node_id:
                    if self._replicate_log(node):
                        success_count += 1
            
            # 如果多数节点已复制，可以提交
            if success_count >= self._num_nodes // 2 + 1:
                self._commit_index = len(self._log) - 1
                self._apply_committed()
                return True
            
            return False
    
    def get_status(self) -> dict:
        """获取Raft状态"""
        with self._lock:
            return {
                'node_id': self.node_id,
                'role': self._role.value,
                'term': self._term,
                'voted_for': self._voted_for,
                'leader': self._leader_id,
                'log_size': len(self._log),
                'commit_index': self._commit_index,
                'last_applied': self._last_applied
            }


# ============== 权益证明(PoS) ==============

class ProofOfStake:
    """
    Proof of Stake (PoS) 共识
    
    权益证明通过质押代币来选择验证者。
    质押越多，被选中的概率越高。
    
    特点：
    - 能源效率高（不需要工作量证明）
    - 验证者有经济激励诚实行事
    - 可以惩罚恶意验证者
    
    历史类比：类似封地大小决定话语权，封地越大（质押越多），
             在领主会议中发言权越大
    """
    
    MIN_STAKE = 100  # 最小质押量
    REWARD_RATE = 0.05  # 年化收益率
    SLASHING_RATE = 0.1  # 惩罚比例
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        
        # 质押信息
        self._stakes: Dict[str, float] = {}  # node_id -> 质押量
        self._total_stake = 0.0
        
        # 验证者
        self._validators: Dict[str, float] = {}  # 活跃验证者
        self._inactive: Set[str] = set()  # 不活跃验证者
        
        # 区块链
        self._chain: List[Block] = []
        self._pending_transactions: List[dict] = []
        
        # 锁
        self._lock = threading.Lock()
        
        # 创世区块
        self._create_genesis_block()
    
    def _create_genesis_block(self) -> None:
        """创建创世区块"""
        genesis = Block(
            index=0,
            timestamp=time.time(),
            data="Genesis Block",
            prev_hash="0" * 64,
            hash=self._calculate_block_hash(0, time.time(), "Genesis Block", "0" * 64),
            validator_id="genesis"
        )
        self._chain.append(genesis)
    
    def _calculate_block_hash(self, index: int, timestamp: float, 
                              data: Any, prev_hash: str) -> str:
        """计算区块哈希"""
        content = f"{index}{timestamp}{json.dumps(data, sort_keys=True)}{prev_hash}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def stake(self, node_id: str, amount: float) -> bool:
        """
        质押代币
        
        Args:
            node_id: 节点ID
            amount: 质押数量
            
        Returns:
            是否成功
        """
        if amount < self.MIN_STAKE:
            return False
        
        with self._lock:
            if node_id not in self._stakes:
                self._stakes[node_id] = 0.0
            
            self._stakes[node_id] += amount
            self._total_stake += amount
            
            # 激活为验证者
            if node_id not in self._validators:
                self._validators[node_id] = self._stakes[node_id]
            
            return True
    
    def unstake(self, node_id: str, amount: float) -> bool:
        """
        解除质押
        
        Args:
            node_id: 节点ID
            amount: 解除数量
            
        Returns:
            是否成功
        """
        with self._lock:
            if node_id not in self._stakes:
                return False
            
            if self._stakes[node_id] < amount:
                return False
            
            self._stakes[node_id] -= amount
            self._total_stake -= amount
            
            # 如果质押量低于最小值，移出验证者
            if self._stakes[node_id] < self.MIN_STAKE:
                self._validators.pop(node_id, None)
                self._inactive.add(node_id)
            
            return True
    
    def _select_validator(self, seed: float = None) -> Optional[str]:
        """
        按权益加权随机选择验证者
        
        使用随机预言机选择下一轮的验证者
        质押越多的验证者被选中的概率越高
        """
        if not self._validators or self._total_stake == 0:
            return None
        
        if seed is None:
            seed = random.random()
        
        with self._lock:
            # 计算累积权重
            cumulative = 0.0
            for node_id, stake in self._validators.items():
                cumulative += stake / self._total_stake
                if seed <= cumulative:
                    return node_id
            
            # 浮点误差，返回最后一个
            return list(self._validators.keys())[-1]
    
    def _slash_validator(self, node_id: str, penalty: float = None) -> bool:
        """
        惩罚恶意验证者
        
        Args:
            node_id: 验证者ID
            penalty: 惩罚数量（默认为质押量的一定比例）
        """
        with self._lock:
            if node_id not in self._stakes:
                return False
            
            if penalty is None:
                penalty = self._stakes[node_id] * self.SLASHING_RATE
            
            self._stakes[node_id] = max(0, self._stakes[node_id] - penalty)
            self._total_stake = max(0, self._total_stake - penalty)
            
            # 更新验证者状态
            if node_id in self._validators:
                self._validators[node_id] = self._stakes[node_id]
                
                if self._stakes[node_id] < self.MIN_STAKE:
                    self._validators.pop(node_id, None)
                    self._inactive.add(node_id)
            
            return True
    
    def _reward_validator(self, node_id: str, reward: float) -> None:
        """
        奖励诚实验证者
        
        Args:
            node_id: 验证者ID
            reward: 奖励数量
        """
        with self._lock:
            if node_id not in self._stakes:
                self._stakes[node_id] = 0.0
            
            self._stakes[node_id] += reward
            self._total_stake += reward
            
            if node_id in self._validators:
                self._validators[node_id] = self._stakes[node_id]
    
    def propose_block(self, data: Any) -> Optional[Block]:
        """
        提议新区块
        
        由选中的验证者提议新区块
        """
        if not self._validators:
            return None
        
        # 随机选择验证者
        validator_id = self._select_validator()
        if not validator_id:
            return None
        
        with self._lock:
            last_block = self._chain[-1]
            
            block = Block(
                index=last_block.index + 1,
                timestamp=time.time(),
                data=data,
                prev_hash=last_block.hash,
                validator_id=validator_id
            )
            block.hash = self._calculate_block_hash(
                block.index, block.timestamp, block.data, block.prev_hash
            )
            
            return block
    
    def validate_block(self, block: Block) -> bool:
        """
        验证区块
        
        检查区块的有效性
        """
        with self._lock:
            last_block = self._chain[-1]
            
            # 检查索引连续性
            if block.index != last_block.index + 1:
                return False
            
            # 检查前一个哈希
            if block.prev_hash != last_block.hash:
                return False
            
            # 检查区块哈希
            expected_hash = self._calculate_block_hash(
                block.index, block.timestamp, block.data, block.prev_hash
            )
            if block.hash != expected_hash:
                return False
            
            # 检查验证者
            if block.validator_id not in self._validators:
                return False
            
            return True
    
    def add_block(self, block: Block) -> bool:
        """
        添加区块到链
        
        验证并添加新区块
        """
        if not self.validate_block(block):
            return False
        
        with self._lock:
            self._chain.append(block)
            
            # 奖励验证者
            reward = self._calculate_block_reward()
            self._reward_validator(block.validator_id, reward)
            
            return True
    
    def _calculate_block_reward(self) -> float:
        """计算区块奖励"""
        # 基于质押总量计算奖励
        base_reward = 10.0
        stake_factor = self._total_stake / 10000.0
        return base_reward * (1.0 + stake_factor)
    
    def get_chain_length(self) -> int:
        """获取链长度"""
        return len(self._chain)
    
    def get_validator_info(self, node_id: str) -> dict:
        """获取验证者信息"""
        with self._lock:
            return {
                'node_id': node_id,
                'stake': self._stakes.get(node_id, 0.0),
                'is_validator': node_id in self._validators,
                'is_active': node_id not in self._inactive,
                'voting_power': self._validators.get(node_id, 0.0) / self._total_stake if self._total_stake > 0 else 0.0
            }
    
    def get_status(self) -> dict:
        """获取PoS状态"""
        with self._lock:
            return {
                'total_stake': self._total_stake,
                'validator_count': len(self._validators),
                'chain_length': len(self._chain),
                'last_block': len(self._chain) - 1,
                'pending_transactions': len(self._pending_transactions)
            }


# ============== 分封制共识 ==============

class FeudalConsensus:
    """
    分封制共识
    
    原创设计的共识机制，灵感来自中国古代分封制：
    
    核心思想：
    1. 节点分为不同等级：大领主（验证者）、小领主、普通节点
    2. 封地大小决定话语权（类似PoS）
    3. 决策需要多级投票：贵族院投票 + 平民投票
    4. 领主可裁决臣民间的纠纷
    5. 朝贡制度：资源按等级分配
    
    特点：
    - 分层治理：不同级别节点有不同权限
    - 等级继承：臣民可晋升为领主
    - 集体决策：重大决策需多级同意
    - 纠纷裁决：领主主持公道
    """
    
    COUNCIL_THRESHOLD = 0.6      # 贵族院通过阈值
    COMMON_THRESHOLD = 0.5       # 平民投票通过阈值
    MAJOR_FIEF_SIZE = 500        # 大领主封地门槛
    ROYAL_FIEF_SIZE = 2000       # 王族封地门槛
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        
        # 领主和封地
        self._lords: Dict[str, float] = {}  # lord_id -> 封地大小
        self._vassals: Dict[str, str] = {}   # vassal_id -> lord_id
        self._fief_data: Dict[str, Any] = {} # 封地数据
        
        # 议会
        self._council: List[str] = []  # 议会成员（大领主）
        self._royal_council: List[str] = []  # 王室会议（最大领主）
        
        # 法令日志
        self._decree_log: List[dict] = []
        self._pending_decrees: Dict[str, dict] = {}
        
        # 朝贡
        self._tribute_pool: Dict[str, float] = {}  # lord_id -> 待分配贡品
        self._tribute_history: List[dict] = []
        
        # 纠纷
        self._disputes: List[dict] = []
        self._alliances: Dict[str, Set[str]] = {}  # 联盟关系
        
        # 锁
        self._lock = threading.RLock()
    
    def register_lord(self, lord_id: str, fief_size: float) -> str:
        """
        注册领主
        
        Args:
            lord_id: 领主ID
            fief_size: 封地大小
            
        Returns:
            领主等级
        """
        with self._lock:
            self._lords[lord_id] = fief_size
            self._fief_data[lord_id] = {
                'troops': fief_size * 10,
                'treasury': fief_size * 5,
                'population': fief_size * 100
            }
            
            # 更新议会
            self._update_council()
            
            return self._get_lord_rank(lord_id)
    
    def register_vassal(self, vassal_id: str, lord_id: str) -> bool:
        """
        注册臣民
        
        Args:
            vassal_id: 臣民ID
            lord_id: 领主ID
        """
        with self._lock:
            if lord_id not in self._lords:
                return False
            
            self._vassals[vassal_id] = lord_id
            return True
    
    def _get_lord_rank(self, lord_id: str) -> str:
        """获取领主等级"""
        fief_size = self._lords.get(lord_id, 0)
        
        if fief_size >= self.ROYAL_FIEF_SIZE:
            return "King/Emperor"
        elif fief_size >= self.MAJOR_FIEF_SIZE:
            return "Duke/Prince"
        elif fief_size >= 200:
            return "Count"
        elif fief_size >= 100:
            return "Viscount"
        else:
            return "Baron"
    
    def _update_council(self) -> None:
        """更新议会成员"""
        # 大领主（封地>=MAJOR_FIEF_SIZE）
        self._council = [
            lord_id for lord_id, size in self._lords.items()
            if size >= self.MAJOR_FIEF_SIZE
        ]
        
        # 王室会议（最大领主）
        if self._lords:
            max_lord = max(self._lords.items(), key=lambda x: x[1])
            self._royal_council = [max_lord[0]]
    
    def propose_decree(self, proposer_id: str, content: Any,
                      decree_type: str = "normal") -> str:
        """
        提议法令
        
        Args:
            proposer_id: 提议者ID
            content: 法令内容
            decree_type: 法令类型 (normal/critical/fundamental)
            
        Returns:
            法令ID
        """
        with self._lock:
            decree_id = f"decree_{len(self._decree_log)}_{int(time.time())}"
            
            decree = {
                'id': decree_id,
                'proposer': proposer_id,
                'content': content,
                'type': decree_type,
                'timestamp': time.time(),
                'noble_votes': {},
                'common_votes': {},
                'status': 'proposed',
                'required_thresholds': self._get_required_thresholds(decree_type)
            }
            
            self._pending_decrees[decree_id] = decree
            
            return decree_id
    
    def _get_required_thresholds(self, decree_type: str) -> dict:
        """获取不同类型法令的阈值"""
        if decree_type == "fundamental":
            return {'noble': 0.8, 'common': 0.6}
        elif decree_type == "critical":
            return {'noble': 0.7, 'common': 0.5}
        else:
            return {'noble': self.COUNCIL_THRESHOLD, 'common': self.COUNCIL_THRESHOLD}
    
    def vote_decree(self, voter_id: str, decree_id: str, 
                   vote: bool, is_noble: bool = None) -> dict:
        """
        投票法令
        
        Args:
            voter_id: 投票者ID
            decree_id: 法令ID
            vote: 赞成(True)或反对(False)
            is_noble: 是否为贵族投票
        """
        with self._lock:
            if decree_id not in self._pending_decrees:
                return {'success': False, 'reason': 'Decree not found'}
            
            decree = self._pending_decrees[decree_id]
            
            # 确定是否为贵族
            if is_noble is None:
                is_noble = voter_id in self._council or voter_id in self._lords
            
            if is_noble:
                decree['noble_votes'][voter_id] = vote
            else:
                decree['common_votes'][voter_id] = vote
            
            # 检查是否通过
            return self._check_decree_passed(decree)
    
    def _check_decree_passed(self, decree: dict) -> dict:
        """检查法令是否通过"""
        thresholds = decree['required_thresholds']
        
        # 检查贵族投票
        noble_yes = sum(1 for v in decree['noble_votes'].values() if v)
        noble_total = len(decree['noble_votes'])
        noble_ratio = noble_yes / noble_total if noble_total > 0 else 0
        
        # 检查平民投票
        common_yes = sum(1 for v in decree['common_votes'].values() if v)
        common_total = len(decree['common_votes'])
        common_ratio = common_yes / common_total if common_total > 0 else 0
        
        noble_passed = noble_ratio >= thresholds['noble']
        common_passed = common_ratio >= thresholds['common']
        
        if noble_passed and common_passed:
            decree['status'] = 'passed'
            decree['passed_at'] = time.time()
            self._enforce_decree(decree)
        
        return {
            'noble_ratio': noble_ratio,
            'common_ratio': common_ratio,
            'noble_passed': noble_passed,
            'common_passed': common_passed,
            'status': decree['status']
        }
    
    def _noble_council_vote(self, decree: dict, threshold: float = None) -> bool:
        """
        贵族院投票
        
        只有大领主参与
        """
        if threshold is None:
            threshold = self.COUNCIL_THRESHOLD
        
        if not self._council:
            return False
        
        votes = decree.get('noble_votes', {})
        
        # 计算加权投票（按封地大小）
        total_weight = 0.0
        voted_weight = 0.0
        
        for lord_id in self._council:
            weight = self._lords.get(lord_id, 100)
            total_weight += weight
            
            if votes.get(lord_id) == True:
                voted_weight += weight
        
        return (voted_weight / total_weight) >= threshold if total_weight > 0 else False
    
    def _common_land_vote(self, decree: dict, threshold: float = None) -> bool:
        """
        平民投票
        
        所有臣民参与（一人一票）
        """
        if threshold is None:
            threshold = self.COMMON_THRESHOLD
        
        if not self._vassals:
            return True  # 没有平民时默认通过
        
        votes = decree.get('common_votes', {})
        
        yes_votes = sum(1 for v in votes.values() if v)
        total_votes = len(self._vassals) + len(votes)
        
        return (yes_votes / total_votes) >= threshold
    
    def _enforce_decree(self, decree: dict) -> None:
        """
        强制执行法令
        
        将通过的法令添加到日志并执行
        """
        decree['enforced_at'] = time.time()
        self._decree_log.append(decree)
        
        if decree['id'] in self._pending_decrees:
            del self._pending_decrees[decree['id']]
    
    def _distribute_tribute(self) -> dict:
        """
        分配贡品
        
        领主按比例分配收集的贡品
        """
        with self._lock:
            distribution = {}
            
            # 按封地大小分配
            total_fief = sum(self._lords.values())
            if total_fief == 0:
                return distribution
            
            for lord_id, fief_size in self._lords.items():
                share = fief_size / total_fief
                distribution[lord_id] = share
                
                # 更新封地数据
                if lord_id in self._fief_data:
                    self._fief_data[lord_id]['treasury'] += share * 100
            
            self._tribute_history.append({
                'timestamp': time.time(),
                'distribution': distribution
            })
            
            return distribution
    
    def _resolve_feud(self, lord_a: str, lord_b: str) -> dict:
        """
        裁决领主间的纷争
        
        由王室会议或贵族院裁决
        """
        with self._lock:
            # 确定裁决者
            if self._royal_council:
                arbitrator = self._royal_council[0]
            elif self._council:
                arbitrator = self._council[0]
            else:
                arbitrator = self.node_id
            
            # 模拟裁决（基于封地大小加权随机）
            power_a = self._lords.get(lord_a, 100)
            power_b = self._lords.get(lord_b, 100)
            total_power = power_a + power_b
            
            if total_power == 0:
                winner = arbitrator
            else:
                # 随机选择，考虑实力差距
                rand = random.random()
                if rand < power_a / total_power * 0.7 + 0.15:
                    winner = lord_a
                elif rand < (power_a + power_b) / total_power * 0.7 + 0.15:
                    winner = lord_b
                else:
                    winner = arbitrator
            
            # 计算赔偿
            loser = lord_b if winner == lord_a else lord_a
            loser_fief = self._lords.get(loser, 100)
            compensation = loser_fief * 0.1
            
            dispute = {
                'arbitrator': arbitrator,
                'parties': [lord_a, lord_b],
                'winner': winner,
                'loser': loser,
                'compensation': compensation,
                'timestamp': time.time()
            }
            
            self._disputes.append(dispute)
            
            # 执行赔偿
            if loser in self._lords:
                self._lords[loser] -= compensation
                self._lords[winner] += compensation
            
            return dispute
    
    def get_consensus_weight(self, node_id: str) -> float:
        """
        计算节点的共识权重
        
        综合考虑封地大小、臣民数量、议会地位等
        """
        with self._lock:
            weight = 0.0
            
            # 封地权重
            fief_size = self._lords.get(node_id, 0)
            weight += fief_size * 1.0
            
            # 臣民权重
            vassals = [v for v, l in self._vassals.items() if l == node_id]
            weight += len(vassals) * 10
            
            # 议会权重
            if node_id in self._royal_council:
                weight += 1000
            elif node_id in self._council:
                weight += 500
            
            return weight
    
    def get_fief_summary(self) -> dict:
        """获取封地总览"""
        with self._lock:
            total_fief = sum(self._lords.values())
            total_vassals = len(self._vassals)
            
            return {
                'total_lords': len(self._lords),
                'total_vassals': total_vassals,
                'total_fief_size': total_fief,
                'council_size': len(self._council),
                'royal_council_size': len(self._royal_council),
                'total_decrees': len(self._decree_log),
                'pending_decrees': len(self._pending_decrees),
                'total_disputes': len(self._disputes)
            }
    
    def get_hierarchy(self) -> dict:
        """获取封建等级结构"""
        with self._lock:
            hierarchy = {
                'royal_council': [
                    {
                        'id': lord_id,
                        'fief': self._lords.get(lord_id, 0),
                        'rank': self._get_lord_rank(lord_id)
                    }
                    for lord_id in self._royal_council
                ],
                'council': [
                    {
                        'id': lord_id,
                        'fief': self._lords.get(lord_id, 0),
                        'rank': self._get_lord_rank(lord_id)
                    }
                    for lord_id in self._council if lord_id not in self._royal_council
                ],
                'lords': [
                    {
                        'id': lord_id,
                        'fief': self._lords.get(lord_id, 0),
                        'rank': self._get_lord_rank(lord_id),
                        'vassals': [v for v, l in self._vassals.items() if l == lord_id]
                    }
                    for lord_id, _ in self._lords.items()
                    if lord_id not in self._council and lord_id not in self._royal_council
                ]
            }
            return hierarchy


# ============== 主程序入口 ==============

if __name__ == "__main__":
    print("=== 去中心化共识机制演示 ===\n")
    
    # 1. PBFT演示
    print("1. PBFT (Practical Byzantine Fault Tolerance):")
    nodes = ["node_0", "node_1", "node_2", "node_3"]
    
    pbft_nodes = {}
    for node_id in nodes:
        pbft_nodes[node_id] = PBFTProtocol(node_id, nodes)
    
    # 主节点发起请求
    primary = pbft_nodes["node_0"]
    test_msg = {"command": "update_model", "round": 1}
    
    preprepare = primary.execute(test_msg)
    print(f"主节点发起预准备: 序列号={preprepare.get('sequence', 'N/A')}")
    
    status = primary.get_status()
    print(f"PBFT状态: view={status['view']}, primary={status['primary']}")
    
    # 2. Raft演示
    print("\n2. Raft Consensus:")
    raft_nodes = {}
    for node_id in nodes:
        raft_nodes[node_id] = RaftConsensus(node_id, nodes)
    
    # 模拟领导者选举
    leader = raft_nodes["node_0"]
    success, term = leader.start_election()
    print(f"开始选举: success={success}, term={term}")
    
    became_leader = leader.become_leader()
    print(f"成为领导者: {became_leader}")
    
    # 提交命令
    if became_leader:
        success = leader.submit_command({"type": "train", "model_id": "v1"})
        print(f"提交命令: {success}")
    
    status = leader.get_status()
    print(f"Raft状态: role={status['role']}, term={status['term']}, log_size={status['log_size']}")
    
    # 3. Proof of Stake演示
    print("\n3. Proof of Stake (PoS):")
    pos = ProofOfStake("validator_0")
    
    # 质押
    validators = ["validator_0", "validator_1", "validator_2", "validator_3"]
    for v in validators:
        amount = random.randint(100, 1000)
        pos.stake(v, amount)
        print(f"验证者 {v} 质押: {amount}")
    
    print(f"总质押量: {pos.get_status()['total_stake']}")
    
    # 选择验证者
    selected = []
    for _ in range(10):
        v = pos._select_validator()
        if v:
            selected.append(v)
    
    from collections import Counter
    counts = Counter(selected)
    print(f"验证者选择统计: {dict(counts)}")
    
    # 提议和添加区块
    block = pos.propose_block({"transactions": ["tx1", "tx2"]})
    if block:
        print(f"提议区块: index={block.index}, validator={block.validator_id}")
        success = pos.add_block(block)
        print(f"添加区块: {success}")
    
    # 4. 分封制共识演示
    print("\n4. Feudal Consensus (分封制共识):")
    feudal = FeudalConsensus("emperor")
    
    # 注册领主
    feudal.register_lord("emperor", 3000)
    feudal.register_lord("king_east", 1500)
    feudal.register_lord("king_west", 1200)
    feudal.register_lord("duke_north", 600)
    feudal.register_lord("count_south", 300)
    feudal.register_lord("baron_forest", 150)
    
    # 注册臣民
    feudal.register_vassal("vassal_1", "king_east")
    feudal.register_vassal("vassal_2", "king_east")
    feudal.register_vassal("vassal_3", "duke_north")
    feudal.register_vassal("vassal_4", "count_south")
    
    # 查看封建结构
    hierarchy = feudal.get_hierarchy()
    print(f"封建结构: {len(hierarchy['royal_council'])} 王室, {len(hierarchy['council'])} 贵族")
    
    # 提议法令
    decree_id = feudal.propose_decree("king_east", 
        {"type": "tax", "rate": 0.1}, 
        decree_type="normal")
    print(f"提议法令: {decree_id}")
    
    # 投票
    feudal.vote_decree("emperor", decree_id, True, is_noble=True)
    feudal.vote_decree("king_west", decree_id, True, is_noble=True)
    feudal.vote_decree("duke_north", decree_id, True, is_noble=True)
    feudal.vote_decree("vassal_1", decree_id, True, is_noble=False)
    feudal.vote_decree("vassal_2", decree_id, True, is_noble=False)
    
    # 查看法令状态
    decree = feudal._pending_decrees.get(decree_id, {})
    if decree:
        print(f"法令状态: {decree.get('status', 'unknown')}")
    
    # 裁决纷争
    dispute = feudal._resolve_feud("king_east", "king_west")
    print(f"裁决纷争: 胜者={dispute['winner']}, 赔偿={dispute['compensation']:.2f}")
    
    # 计算共识权重
    for node_id in ["emperor", "king_east", "baron_forest"]:
        weight = feudal.get_consensus_weight(node_id)
        print(f"共识权重 {node_id}: {weight:.2f}")
    
    # 封地总览
    summary = feudal.get_fief_summary()
    print(f"封地总览: {summary['total_lords']} 领主, {summary['total_vassals']} 臣民")
    
    print("\n=== 演示完成 ===")
