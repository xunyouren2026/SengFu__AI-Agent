"""
共识协议系统 - Consensus Protocol System

实现多种共识算法，包括Raft、PBFT拜占庭容错，以及多种投票机制。
仅使用Python标准库。
"""

import uuid
import time
import math
import enum
import random
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set
from collections import defaultdict


# ============================================================
# 共识结果
# ============================================================

@dataclass
class ConsensusResult:
    """共识结果"""
    agreed_value: Optional[Any] = None
    participants: List[str] = field(default_factory=list)
    votes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    rounds_needed: int = 0
    is_final: bool = False


# ============================================================
# 共识协议抽象基类
# ============================================================

class ConsensusProtocol(ABC):
    """共识协议抽象基类"""

    @abstractmethod
    def propose(self, proposer_id: str, value: Any) -> str:
        """提交提案，返回提案ID"""
        ...

    @abstractmethod
    def vote(self, voter_id: str, proposal_id: str, vote: bool) -> bool:
        """对提案进行投票"""
        ...

    @abstractmethod
    def get_result(self, proposal_id: str) -> Optional[ConsensusResult]:
        """获取共识结果"""
        ...


# ============================================================
# 投票机制
# ============================================================

class VotingMechanism:
    """多种投票机制实现"""

    @staticmethod
    def simple_majority(votes: Dict[str, bool]) -> Tuple[bool, float]:
        """
        简单多数投票。

        超过半数赞成则通过。
        返回 (是否通过, 赞成比例)。
        """
        if not votes:
            return False, 0.0

        total = len(votes)
        in_favor = sum(1 for v in votes.values() if v)
        ratio = in_favor / total

        return ratio > 0.5, round(ratio, 4)

    @staticmethod
    def weighted_vote(
        votes: Dict[str, bool],
        weights: Dict[str, float],
    ) -> Tuple[bool, float]:
        """
        加权投票。

        每个投票者的票数根据权重计算。
        总权重超过半数赞成则通过。

        Args:
            votes: 投票者 -> 投票结果
            weights: 投票者 -> 权重

        Returns:
            (是否通过, 加权赞成比例)
        """
        if not votes:
            return False, 0.0

        total_weight = 0.0
        in_favor_weight = 0.0

        for voter, vote in votes.items():
            w = weights.get(voter, 1.0)
            total_weight += w
            if vote:
                in_favor_weight += w

        if total_weight == 0:
            return False, 0.0

        ratio = in_favor_weight / total_weight
        return ratio > 0.5, round(ratio, 4)

    @staticmethod
    def supermajority(votes: Dict[str, bool], threshold: float = 2.0 / 3.0) -> Tuple[bool, float]:
        """
        超级多数投票。

        默认需要2/3以上赞成。

        Args:
            votes: 投票者 -> 投票结果
            threshold: 通过阈值

        Returns:
            (是否通过, 赞成比例)
        """
        if not votes:
            return False, 0.0

        total = len(votes)
        in_favor = sum(1 for v in votes.values() if v)
        ratio = in_favor / total

        return ratio >= threshold, round(ratio, 4)

    @staticmethod
    def quadratic_voting(
        votes: Dict[str, int],
        max_credits: float = 100.0,
    ) -> Tuple[bool, float]:
        """
        二次方投票。

        每个投票者可以分配积分来表达偏好强度。
        成本 = vote_credits^2，总成本不超过max_credits。
        净票数 > 0 则通过。

        Args:
            votes: 投票者 -> 分配的票数（正数赞成，负数反对）
            max_credits: 每个投票者的最大积分

        Returns:
            (是否通过, 净赞成强度)
        """
        if not votes:
            return False, 0.0

        # 计算每个投票者的成本并验证
        total_net_votes = 0
        valid_voters = 0

        for voter, vote_credits in votes.items():
            # 二次方成本
            cost = vote_credits ** 2
            if cost <= max_credits:
                total_net_votes += vote_credits
                valid_voters += 1

        if valid_voters == 0:
            return False, 0.0

        # 归一化净票数到 [-1, 1]
        max_possible = valid_voters * math.sqrt(max_credits)
        if max_possible > 0:
            normalized = total_net_votes / max_possible
        else:
            normalized = 0.0

        return total_net_votes > 0, round(normalized, 4)


# ============================================================
# Raft 共识算法
# ============================================================

class Role(enum.Enum):
    """Raft角色"""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


@dataclass
class LogEntry:
    """Raft日志条目"""
    term: int
    index: int
    command: Any
    committed: bool = False


@dataclass
class Term:
    """任期"""
    number: int
    start_time: float = field(default_factory=time.time)
    leader_id: Optional[str] = None
    votes_received: Set[str] = field(default_factory=set)


class RaftConsensus(ConsensusProtocol):
    """
    Raft共识算法实现。

    角色转换: Follower -> Candidate -> Leader
    使用timer模拟选举超时。
    """

    def __init__(self, node_id: str, peers: Optional[List[str]] = None):
        self.node_id = node_id
        self.peers = set(peers or [])

        # 状态
        self.current_term = 0
        self.role = Role.FOLLOWER
        self.voted_for: Optional[str] = None
        self.leader_id: Optional[str] = None

        # 日志
        self.log: List[LogEntry] = []
        self.commit_index = 0
        self.last_applied = 0

        # Leader状态
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}

        # 选举
        self._election_timer = None
        self._election_timeout_range = (1.5, 3.0)  # 秒
        self._last_heartbeat = time.time()
        self._reset_election_timer()

        # 提案跟踪
        self._proposals: Dict[str, Dict[str, Any]] = {}

        # 初始化
        for peer in self.peers:
            self.next_index[peer] = 1
            self.match_index[peer] = 0

    def _reset_election_timer(self):
        """重置选举计时器"""
        low, high = self._election_timeout_range
        self._election_deadline = time.time() + random.uniform(low, high)

    def _start_election(self):
        """发起选举"""
        self.current_term += 1
        self.role = Role.CANDIDATE
        self.voted_for = self.node_id
        self._reset_election_timer()

        # 投票给自己
        votes = {self.node_id}
        # 模拟其他节点的投票响应
        for peer in self.peers:
            if random.random() < 0.6:  # 60%概率获得其他节点投票
                votes.add(peer)

        majority = (len(self.peers) + 1) // 2 + 1
        if len(votes) >= majority:
            self.role = Role.LEADER
            self.leader_id = self.node_id
            # 初始化Leader状态
            for peer in self.peers:
                self.next_index[peer] = len(self.log) + 1
                self.match_index[peer] = 0

    def heartbeat(self) -> Dict[str, Any]:
        """
        发送心跳。

        Leader定期发送心跳维持权威。
        Follower收到心跳后重置选举计时器。
        """
        now = time.time()

        if self.role == Role.LEADER:
            # Leader发送心跳
            self._last_heartbeat = now
            return {
                "type": "heartbeat",
                "term": self.current_term,
                "leader_id": self.node_id,
                "leader_commit": self.commit_index,
            }
        else:
            # 检查选举超时
            if now >= self._election_deadline:
                self._start_election()
                return {
                    "type": "election_started",
                    "term": self.current_term,
                    "role": self.role.value,
                }

            return {
                "type": "heartbeat_ack",
                "term": self.current_term,
                "role": self.role.value,
            }

    def request_vote(self, candidate_id: str, term: int) -> Dict[str, Any]:
        """
        处理投票请求。

        授票条件：
        1. 候选人任期 >= 当前任期
        2. 候选人日志至少和自己一样新
        """
        if term < self.current_term:
            return {"vote_granted": False, "term": self.current_term}

        if term > self.current_term:
            self.current_term = term
            self.role = Role.FOLLOWER
            self.voted_for = None

        if self.voted_for is not None and self.voted_for != candidate_id:
            return {"vote_granted": False, "term": self.current_term}

        # 检查日志是否至少一样新
        candidate_log_ok = True
        if self.log:
            last_log_term = self.log[-1].term
            last_log_index = len(self.log)
            # 候选人的最后日志条目（简化处理）
            candidate_last_index = len(self.log)  # 模拟
            candidate_last_term = self.current_term

            if candidate_last_term < last_log_term:
                candidate_log_ok = False
            elif (candidate_last_term == last_log_term
                  and candidate_last_index < last_log_index):
                candidate_log_ok = False

        if candidate_log_ok:
            self.voted_for = candidate_id
            self._reset_election_timer()
            return {"vote_granted": True, "term": self.current_term}

        return {"vote_granted": False, "term": self.current_term}

    def append_entries(
        self,
        leader_id: str,
        term: int,
        entries: Optional[List[Any]] = None,
        prev_log_index: int = 0,
        prev_log_term: int = 0,
    ) -> Dict[str, Any]:
        """
        追加日志条目（Leader调用）。

        验证：
        1. Leader任期合法
        2. 前一条日志匹配
        """
        if term < self.current_term:
            return {"success": False, "term": self.current_term}

        if term >= self.current_term:
            self.current_term = term
            self.role = Role.FOLLOWER
            self.leader_id = leader_id
            self._reset_election_timer()
            self._last_heartbeat = time.time()

        # 验证前一条日志
        if prev_log_index > 0:
            if prev_log_index > len(self.log):
                return {"success": False, "term": self.current_term, "reason": "log_short"}
            if self.log[prev_log_index - 1].term != prev_log_term:
                return {"success": False, "term": self.current_term, "reason": "term_mismatch"}

        # 追加新条目
        entries = entries or []
        for i, entry_data in enumerate(entries):
            log_index = prev_log_index + i + 1
            if log_index <= len(self.log):
                # 如果已存在且任期不同，删除冲突条目
                if self.log[log_index - 1].term != term:
                    self.log = self.log[:log_index - 1]
                    new_entry = LogEntry(
                        term=term,
                        index=log_index,
                        command=entry_data,
                    )
                    self.log.append(new_entry)
            else:
                new_entry = LogEntry(
                    term=term,
                    index=log_index,
                    command=entry_data,
                )
                self.log.append(new_entry)

        # 更新commit_index
        if entries:
            new_commit_index = prev_log_index + len(entries)
            if new_commit_index > self.commit_index:
                # 确保当前任期的条目才提交
                if self.log[new_commit_index - 1].term == self.current_term:
                    self.commit_index = new_commit_index

        return {"success": True, "term": self.current_term}

    def election_timeout(self) -> bool:
        """检查是否选举超时"""
        return time.time() >= self._election_deadline

    # ConsensusProtocol 接口实现

    def propose(self, proposer_id: str, value: Any) -> str:
        """提交提案（仅Leader可以提案）"""
        proposal_id = str(uuid.uuid4())

        if self.role != Role.LEADER:
            # 如果不是Leader，尝试发起选举
            self._start_election()
            if self.role != Role.LEADER:
                raise ValueError("当前节点不是Leader，无法提案")

        # 创建日志条目
        entry = LogEntry(
            term=self.current_term,
            index=len(self.log) + 1,
            command=value,
        )
        self.log.append(entry)

        # 模拟复制到大多数节点
        replicated = 1  # 自己
        for peer in self.peers:
            if random.random() < 0.8:  # 80%复制成功率
                replicated += 1
                self.match_index[peer] = entry.index
                self.next_index[peer] = entry.index + 1

        majority = (len(self.peers) + 1) // 2 + 1
        if replicated >= majority:
            self.commit_index = entry.index
            entry.committed = True

        self._proposals[proposal_id] = {
            "value": value,
            "proposer": proposer_id,
            "term": self.current_term,
            "index": entry.index,
            "committed": entry.committed,
        }

        return proposal_id

    def vote(self, voter_id: str, proposal_id: str, vote: bool) -> bool:
        """对提案投票（Raft中投票在选举阶段完成）"""
        if proposal_id not in self._proposals:
            return False

        proposal = self._proposals[proposal_id]
        if proposal["committed"]:
            return True  # 已提交

        # 在Raft中，日志复制成功即视为"投票通过"
        return proposal.get("committed", False)

    def get_result(self, proposal_id: str) -> Optional[ConsensusResult]:
        """获取共识结果"""
        if proposal_id not in self._proposals:
            return None

        proposal = self._proposals[proposal_id]
        return ConsensusResult(
            agreed_value=proposal["value"] if proposal["committed"] else None,
            participants=[self.node_id] + list(self.peers),
            votes={self.node_id: proposal["committed"]},
            confidence=1.0 if proposal["committed"] else 0.0,
            rounds_needed=1,
            is_final=proposal["committed"],
        )


# ============================================================
# PBFT 拜占庭容错共识
# ============================================================

class PBFTPhase(enum.Enum):
    """PBFT阶段"""
    IDLE = "idle"
    PRE_PREPARE = "pre_prepare"
    PREPARE = "prepare"
    COMMIT = "commit"
    COMMITTED = "committed"


class PBFTConsensus(ConsensusProtocol):
    """
    PBFT拜占庭容错共识算法。

    容忍f个拜占庭节点，需要至少3f+1个节点。
    三阶段协议：预准备 -> 准备 -> 提交
    """

    def __init__(self, node_id: str, nodes: Optional[List[str]] = None):
        self.node_id = node_id
        self.nodes = list(nodes or [node_id])
        self.n = len(self.nodes)

        # 计算容错能力
        self.f = (self.n - 1) // 3
        self._validate_fault_tolerance()

        # 视图
        self.view_number = 0
        self.sequence_number = 0

        # 阶段消息记录
        self._pre_prepare_msgs: Dict[str, Dict[str, Any]] = {}
        self._prepare_msgs: Dict[str, Dict[str, Set[str]]] = {}
        self._commit_msgs: Dict[str, Dict[str, Set[str]]] = {}

        # 提案状态
        self._proposals: Dict[str, Dict[str, Any]] = {}

        # 视图切换
        self._view_change_msgs: Dict[int, Set[str]] = {}

    def _validate_fault_tolerance(self):
        """验证节点数量是否满足拜占庭容错要求"""
        if self.n < 3 * self.f + 1:
            raise ValueError(
                f"PBFT需要至少 3f+1={3 * self.f + 1} 个节点，"
                f"当前只有 {self.n} 个"
            )

    def _quorum(self) -> int:
        """计算法定人数：2f+1"""
        return 2 * self.f + 1

    def _make_key(self, view: int, seq: int) -> str:
        """生成消息键"""
        return f"{view}:{seq}"

    def pre_prepare(
        self,
        proposal_id: str,
        value: Any,
        proposer_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        预准备阶段（由主节点发起）。

        主节点广播预准备消息给所有备份节点。
        """
        proposer = proposer_id or self.node_id
        self.sequence_number += 1
        seq = self.sequence_number
        key = self._make_key(self.view_number, seq)

        # 记录预准备消息
        self._pre_prepare_msgs[key] = {
            "proposal_id": proposal_id,
            "value": value,
            "proposer": proposer,
            "view": self.view_number,
            "seq": seq,
            "digest": hash(str(value)) if value is not None else 0,
        }

        # 初始化准备和提交消息集合
        self._prepare_msgs[key] = {"votes": {self.node_id}, "value": value}
        self._commit_msgs[key] = {"votes": set(), "value": value}

        # 记录提案
        self._proposals[proposal_id] = {
            "value": value,
            "proposer": proposer,
            "view": self.view_number,
            "seq": seq,
            "phase": PBFTPhase.PRE_PREPARE,
            "key": key,
        }

        # 模拟备份节点响应预准备
        for node in self.nodes:
            if node != self.node_id:
                if random.random() < 0.85:  # 85%响应率
                    self._prepare_msgs[key]["votes"].add(node)

        return {
            "proposal_id": proposal_id,
            "view": self.view_number,
            "seq": seq,
            "phase": PBFTPhase.PRE_PREPARE.value,
        }

    def prepare(self, proposal_id: str, voter_id: Optional[str] = None) -> Dict[str, Any]:
        """
        准备阶段。

        备份节点收到预准备消息后，广播准备消息。
        当收到2f+1个准备消息（包括自己）时，进入提交阶段。
        """
        voter = voter_id or self.node_id

        if proposal_id not in self._proposals:
            return {"success": False, "reason": "unknown_proposal"}

        proposal = self._proposals[proposal_id]
        key = proposal["key"]

        if key not in self._prepare_msgs:
            self._prepare_msgs[key] = {"votes": set(), "value": proposal["value"]}

        self._prepare_msgs[key]["votes"].add(voter)

        # 检查是否达到法定人数
        prepare_count = len(self._prepare_msgs[key]["votes"])
        quorum = self._quorum()

        result = {
            "proposal_id": proposal_id,
            "phase": PBFTPhase.PREPARE.value,
            "prepare_count": prepare_count,
            "quorum": quorum,
            "quorum_reached": prepare_count >= quorum,
        }

        if prepare_count >= quorum:
            # 进入提交阶段
            proposal["phase"] = PBFTPhase.COMMIT
            self._commit_msgs[key]["votes"].add(self.node_id)

            # 模拟其他节点提交
            for node in self.nodes:
                if node != self.node_id:
                    if random.random() < 0.85:
                        self._commit_msgs[key]["votes"].add(node)

            # 检查提交是否完成
            commit_count = len(self._commit_msgs[key]["votes"])
            if commit_count >= quorum:
                proposal["phase"] = PBFTPhase.COMMITTED
                result["committed"] = True
                result["commit_count"] = commit_count

        return result

    def commit(self, proposal_id: str, voter_id: Optional[str] = None) -> Dict[str, Any]:
        """
        提交阶段。

        节点在收到2f+1个准备消息后广播提交消息。
        收到2f+1个提交消息后，执行并提交。
        """
        voter = voter_id or self.node_id

        if proposal_id not in self._proposals:
            return {"success": False, "reason": "unknown_proposal"}

        proposal = self._proposals[proposal_id]
        key = proposal["key"]

        if key not in self._commit_msgs:
            self._commit_msgs[key] = {"votes": set(), "value": proposal["value"]}

        self._commit_msgs[key]["votes"].add(voter)

        commit_count = len(self._commit_msgs[key]["votes"])
        quorum = self._quorum()

        committed = commit_count >= quorum
        if committed:
            proposal["phase"] = PBFTPhase.COMMITTED

        return {
            "proposal_id": proposal_id,
            "phase": PBFTPhase.COMMIT.value,
            "commit_count": commit_count,
            "quorum": quorum,
            "committed": committed,
        }

    def view_change(self, new_view: Optional[int] = None) -> Dict[str, Any]:
        """
        视图切换。

        当节点检测到主节点故障时发起视图切换。
        新视图 = 当前视图 + 1（或指定值）。
        """
        old_view = self.view_number
        self.view_number = new_view if new_view is not None else self.view_number + 1

        # 记录视图切换消息
        if self.view_number not in self._view_change_msgs:
            self._view_change_msgs[self.view_number] = set()
        self._view_change_msgs[self.view_number].add(self.node_id)

        # 模拟其他节点响应视图切换
        for node in self.nodes:
            if node != self.node_id:
                if random.random() < 0.8:
                    self._view_change_msgs[self.view_number].add(node)

        vc_count = len(self._view_change_msgs[self.view_number])
        quorum = self._quorum()

        return {
            "old_view": old_view,
            "new_view": self.view_number,
            "view_change_count": vc_count,
            "quorum": quorum,
            "view_change_complete": vc_count >= quorum,
        }

    # ConsensusProtocol 接口实现

    def propose(self, proposer_id: str, value: Any) -> str:
        """提交提案"""
        proposal_id = str(uuid.uuid4())
        self.pre_prepare(proposal_id, value, proposer_id)
        self.prepare(proposal_id)
        self.commit(proposal_id)
        return proposal_id

    def vote(self, voter_id: str, proposal_id: str, vote: bool) -> bool:
        """对提案投票"""
        if proposal_id not in self._proposals:
            return False
        if vote:
            self.prepare(proposal_id, voter_id)
            self.commit(proposal_id, voter_id)
        return self._proposals[proposal_id]["phase"] == PBFTPhase.COMMITTED

    def get_result(self, proposal_id: str) -> Optional[ConsensusResult]:
        """获取共识结果"""
        if proposal_id not in self._proposals:
            return None

        proposal = self._proposals[proposal_id]
        key = proposal["key"]
        is_committed = proposal["phase"] == PBFTPhase.COMMITTED

        # 计算置信度
        if is_committed:
            prepare_count = len(self._prepare_msgs.get(key, {}).get("votes", set()))
            commit_count = len(self._commit_msgs.get(key, {}).get("votes", set()))
            confidence = (prepare_count + commit_count) / (2 * self.n)
        else:
            confidence = 0.0

        return ConsensusResult(
            agreed_value=proposal["value"] if is_committed else None,
            participants=list(self.nodes),
            votes={
                node: node in self._commit_msgs.get(key, {}).get("votes", set())
                for node in self.nodes
            },
            confidence=round(confidence, 4),
            rounds_needed=1,
            is_final=is_committed,
        )
