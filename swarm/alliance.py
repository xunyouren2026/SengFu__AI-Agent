"""
联盟管理模块

提供智能体联盟的组建、管理和解散功能，以及基于FIPA标准的
合同网协议（Contract Net Protocol）实现，用于任务招标和提案评估。
"""

import threading
import time
import uuid
import copy
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from collections import deque


class AllianceStatus(Enum):
    """联盟状态枚举"""
    FORMING = "forming"
    ACTIVE = "active"
    DISSOLVING = "dissolving"
    DISSOLVED = "dissolved"


class Proposal:
    """合同网协议提案

    描述智能体对任务的投标提案，包含预估成本、时间、
    置信度和相关能力。

    Attributes:
        agent_id: 提案智能体ID
        task_id: 任务ID
        estimated_cost: 预估成本
        estimated_time: 预估完成时间（秒）
        confidence: 置信度 (0.0 ~ 1.0)
        capabilities: 相关能力列表
        proposal_id: 提案唯一标识
        timestamp: 提案时间戳
        metadata: 附加元数据
    """

    def __init__(
        self,
        agent_id: str,
        task_id: str,
        estimated_cost: float = 0.0,
        estimated_time: float = 0.0,
        confidence: float = 0.5,
        capabilities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.proposal_id = str(uuid.uuid4())
        self.agent_id = agent_id
        self.task_id = task_id
        self.estimated_cost = max(0.0, estimated_cost)
        self.estimated_time = max(0.0, estimated_time)
        self.confidence = max(0.0, min(1.0, confidence))
        self.capabilities = capabilities or []
        self.metadata = metadata or {}
        self.timestamp = time.time()
        self.accepted = False
        self.rejected = False

    def score(self, cost_weight: float = 0.3, time_weight: float = 0.3,
              confidence_weight: float = 0.4) -> float:
        """计算提案综合评分

        评分越高越好。成本和时间越低越好，置信度越高越好。

        Args:
            cost_weight: 成本权重
            time_weight: 时间权重
            confidence_weight: 置信度权重

        Returns:
            综合评分
        """
        # 归一化成本和时间（使用sigmoid-like变换避免除零）
        cost_score = 1.0 / (1.0 + self.estimated_cost)
        time_score = 1.0 / (1.0 + self.estimated_time)
        confidence_score = self.confidence

        total_weight = cost_weight + time_weight + confidence_weight
        if total_weight <= 0:
            return 0.0

        raw = (
            cost_weight * cost_score
            + time_weight * time_score
            + confidence_weight * confidence_score
        )
        return raw / total_weight

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "proposal_id": self.proposal_id,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "estimated_cost": self.estimated_cost,
            "estimated_time": self.estimated_time,
            "confidence": self.confidence,
            "capabilities": self.capabilities,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "accepted": self.accepted,
            "rejected": self.rejected,
        }

    def __repr__(self) -> str:
        return (
            f"Proposal(agent={self.agent_id!r}, task={self.task_id!r}, "
            f"cost={self.estimated_cost}, confidence={self.confidence})"
        )


class Alliance:
    """智能体联盟

    一组智能体为完成共同任务而组成的协作联盟。

    Attributes:
        id: 联盟唯一标识
        name: 联盟名称
        members: 成员智能体ID列表
        leader: 领导者智能体ID
        task_queue: 待处理任务队列
        status: 联盟状态
        created_at: 创建时间
        description: 联盟描述
    """

    def __init__(
        self,
        name: str,
        member_ids: Optional[List[str]] = None,
        leader: Optional[str] = None,
        description: str = "",
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.members = list(member_ids) if member_ids else []
        self.leader = leader or (self.members[0] if self.members else "")
        self.task_queue: deque = deque()
        self.status = AllianceStatus.FORMING
        self.created_at = time.time()
        self.description = description
        self._lock = threading.Lock()

    def add_member(self, agent_id: str) -> bool:
        """添加成员"""
        with self._lock:
            if agent_id in self.members:
                return False
            self.members.append(agent_id)
            return True

    def remove_member(self, agent_id: str) -> bool:
        """移除成员"""
        with self._lock:
            if agent_id not in self.members:
                return False
            self.members.remove(agent_id)
            if self.leader == agent_id:
                self.leader = self.members[0] if self.members else ""
            return True

    def enqueue_task(self, task: Any) -> None:
        """将任务加入队列"""
        with self._lock:
            self.task_queue.append(task)

    def dequeue_task(self) -> Optional[Any]:
        """从队列取出任务"""
        with self._lock:
            if self.task_queue:
                return self.task_queue.popleft()
            return None

    def task_count(self) -> int:
        """获取待处理任务数"""
        with self._lock:
            return len(self.task_queue)

    def member_count(self) -> int:
        """获取成员数量"""
        with self._lock:
            return len(self.members)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        with self._lock:
            return {
                "id": self.id,
                "name": self.name,
                "members": list(self.members),
                "leader": self.leader,
                "task_count": len(self.task_queue),
                "status": self.status.value,
                "created_at": self.created_at,
                "description": self.description,
            }

    def __repr__(self) -> str:
        return (
            f"Alliance(id={self.id!r}, name={self.name!r}, "
            f"members={len(self.members)}, status={self.status.value})"
        )


class AllianceManager:
    """联盟管理器

    管理智能体联盟的完整生命周期：组建、运行、任务分配和解散。
    线程安全实现。

    Attributes:
        alliances: 联盟字典
    """

    def __init__(self):
        self._alliances: Dict[str, Alliance] = {}
        self._lock = threading.RLock()
        self._event_listeners: List[callable] = []

    def form_alliance(
        self,
        name: str,
        member_ids: List[str],
        task: Optional[Any] = None,
        leader: Optional[str] = None,
        description: str = "",
    ) -> Alliance:
        """组建联盟

        Args:
            name: 联盟名称
            member_ids: 成员智能体ID列表
            task: 初始任务（可选）
            leader: 领导者ID（默认为第一个成员）
            description: 联盟描述

        Returns:
            创建的联盟对象

        Raises:
            ValueError: 成员列表为空
        """
        if not member_ids:
            raise ValueError("联盟至少需要一个成员")

        alliance = Alliance(
            name=name,
            member_ids=member_ids,
            leader=leader,
            description=description,
        )

        if task is not None:
            alliance.enqueue_task(task)

        with self._lock:
            self._alliances[alliance.id] = alliance
            alliance.status = AllianceStatus.ACTIVE
            self._notify("formed", alliance)

        return alliance

    def dissolve_alliance(self, alliance_id: str) -> bool:
        """解散联盟

        Args:
            alliance_id: 联盟ID

        Returns:
            是否成功解散
        """
        with self._lock:
            alliance = self._alliances.get(alliance_id)
            if alliance is None:
                return False
            if alliance.status == AllianceStatus.DISSOLVED:
                return False
            alliance.status = AllianceStatus.DISSOLVING
            self._notify("dissolving", alliance)

        # 模拟解散过程
        with self._lock:
            alliance.status = AllianceStatus.DISSOLVED
            self._notify("dissolved", alliance)
            return True

    def add_member(self, alliance_id: str, agent_id: str) -> bool:
        """向联盟添加成员

        Args:
            alliance_id: 联盟ID
            agent_id: 智能体ID

        Returns:
            是否成功添加
        """
        with self._lock:
            alliance = self._alliances.get(alliance_id)
            if alliance is None or alliance.status != AllianceStatus.ACTIVE:
                return False
            result = alliance.add_member(agent_id)
            if result:
                self._notify("member_added", alliance)
            return result

    def remove_member(self, alliance_id: str, agent_id: str) -> bool:
        """从联盟移除成员

        Args:
            alliance_id: 联盟ID
            agent_id: 智能体ID

        Returns:
            是否成功移除
        """
        with self._lock:
            alliance = self._alliances.get(alliance_id)
            if alliance is None or alliance.status != AllianceStatus.ACTIVE:
                return False
            if len(alliance.members) <= 1:
                return False  # 不允许移除最后一个成员
            result = alliance.remove_member(agent_id)
            if result:
                self._notify("member_removed", alliance)
            return result

    def assign_task(self, alliance_id: str, task: Any) -> bool:
        """向联盟分配任务

        Args:
            alliance_id: 联盟ID
            task: 任务对象

        Returns:
            是否成功分配
        """
        with self._lock:
            alliance = self._alliances.get(alliance_id)
            if alliance is None or alliance.status != AllianceStatus.ACTIVE:
                return False
            alliance.enqueue_task(task)
            self._notify("task_assigned", alliance)
            return True

    def get_alliance(self, alliance_id: str) -> Optional[Alliance]:
        """获取联盟"""
        with self._lock:
            alliance = self._alliances.get(alliance_id)
            if alliance:
                return copy.deepcopy(alliance)
            return None

    def get_alliance_status(self, alliance_id: str) -> Optional[Dict[str, Any]]:
        """获取联盟状态信息"""
        with self._lock:
            alliance = self._alliances.get(alliance_id)
            if alliance is None:
                return None
            return {
                "id": alliance.id,
                "name": alliance.name,
                "status": alliance.status.value,
                "member_count": alliance.member_count(),
                "task_count": alliance.task_count(),
                "leader": alliance.leader,
            }

    def list_alliances(
        self,
        status: Optional[AllianceStatus] = None,
    ) -> List[Dict[str, Any]]:
        """列出所有联盟

        Args:
            status: 按状态过滤

        Returns:
            联盟信息列表
        """
        with self._lock:
            results = []
            for alliance in self._alliances.values():
                if status and alliance.status != status:
                    continue
                results.append(alliance.to_dict())
            return results

    def get_agent_alliances(self, agent_id: str) -> List[Alliance]:
        """获取智能体参与的所有联盟"""
        with self._lock:
            results = []
            for alliance in self._alliances.values():
                if agent_id in alliance.members:
                    results.append(copy.deepcopy(alliance))
            return results

    def add_event_listener(self, listener: callable) -> None:
        """添加事件监听器"""
        with self._lock:
            self._event_listeners.append(listener)

    def remove_event_listener(self, listener: callable) -> bool:
        """移除事件监听器"""
        with self._lock:
            try:
                self._event_listeners.remove(listener)
                return True
            except ValueError:
                return False

    def _notify(self, event_type: str, alliance: Alliance) -> None:
        """通知事件监听器"""
        for listener in self._event_listeners:
            try:
                listener(event_type, alliance)
            except Exception:
                pass

    def __len__(self) -> int:
        with self._lock:
            return len(self._alliances)


class ContractNetProtocol:
    """合同网协议（FIPA标准）

    实现FIPA合同网协议，用于任务招标、提案提交、评估和决策。
    管理完整的招标流程：发起招标 -> 接收提案 -> 评估 -> 接受/拒绝。

    Attributes:
        proposals: 当前所有提案
        task_proposals: 按任务分组的提案
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._proposals: Dict[str, Proposal] = {}  # proposal_id -> Proposal
        self._task_proposals: Dict[str, List[str]] = {}  # task_id -> [proposal_ids]
        self._task_status: Dict[str, str] = {}  # task_id -> status
        self._accepted_proposals: Dict[str, str] = {}  # task_id -> proposal_id
        self._deadline_tasks: Dict[str, float] = {}  # task_id -> deadline

    def call_for_proposal(
        self,
        task_id: str,
        candidates: List[str],
        deadline: Optional[float] = None,
        task_description: str = "",
    ) -> Dict[str, Any]:
        """发起招标（Call for Proposals, CFP）

        向候选智能体发起任务招标。

        Args:
            task_id: 任务ID
            candidates: 候选智能体ID列表
            deadline: 提案截止时间（Unix时间戳），默认30秒后
            task_description: 任务描述

        Returns:
            招标信息字典
        """
        if not candidates:
            raise ValueError("候选智能体列表不能为空")

        if deadline is None:
            deadline = time.time() + 30.0

        with self._lock:
            self._task_proposals[task_id] = []
            self._task_status[task_id] = "cfp"
            self._deadline_tasks[task_id] = deadline

        return {
            "task_id": task_id,
            "candidates": candidates,
            "deadline": deadline,
            "task_description": task_description,
            "status": "cfp_open",
        }

    def submit_proposal(
        self,
        agent_id: str,
        task_id: str,
        estimated_cost: float,
        estimated_time: float,
        confidence: float,
        capabilities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Proposal]:
        """提交提案

        智能体对招标任务提交提案。

        Args:
            agent_id: 提案智能体ID
            task_id: 任务ID
            estimated_cost: 预估成本
            estimated_time: 预估时间
            confidence: 置信度
            capabilities: 能力列表
            metadata: 附加元数据

        Returns:
            创建的提案对象，失败返回None
        """
        with self._lock:
            # 检查任务是否在招标中
            if task_id not in self._task_status:
                return None
            if self._task_status[task_id] != "cfp":
                return None

            # 检查是否超过截止时间
            deadline = self._deadline_tasks.get(task_id, float("inf"))
            if time.time() > deadline:
                return None

            # 检查是否已提交过提案
            existing = self._task_proposals.get(task_id, [])
            for pid in existing:
                prop = self._proposals.get(pid)
                if prop and prop.agent_id == agent_id:
                    return None  # 每个智能体只能提交一次

            proposal = Proposal(
                agent_id=agent_id,
                task_id=task_id,
                estimated_cost=estimated_cost,
                estimated_time=estimated_time,
                confidence=confidence,
                capabilities=capabilities,
                metadata=metadata,
            )

            self._proposals[proposal.proposal_id] = proposal
            if task_id not in self._task_proposals:
                self._task_proposals[task_id] = []
            self._task_proposals[task_id].append(proposal.proposal_id)

            return proposal

    def evaluate_proposals(
        self,
        task_id: str,
        cost_weight: float = 0.3,
        time_weight: float = 0.3,
        confidence_weight: float = 0.4,
    ) -> List[Tuple[Proposal, float]]:
        """评估提案

        对指定任务的所有提案进行评分和排序。

        Args:
            task_id: 任务ID
            cost_weight: 成本权重
            time_weight: 时间权重
            confidence_weight: 置信度权重

        Returns:
            按评分降序排列的(提案, 评分)元组列表
        """
        with self._lock:
            proposal_ids = self._task_proposals.get(task_id, [])
            scored = []
            for pid in proposal_ids:
                proposal = self._proposals.get(pid)
                if proposal and not proposal.rejected:
                    score = proposal.score(cost_weight, time_weight, confidence_weight)
                    scored.append((proposal, score))

            # 按评分降序排列
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored

    def accept_proposal(self, task_id: str, proposal_id: str) -> bool:
        """接受提案

        接受指定提案，同时拒绝同一任务的其他提案。

        Args:
            task_id: 任务ID
            proposal_id: 要接受的提案ID

        Returns:
            是否成功接受
        """
        with self._lock:
            proposal = self._proposals.get(proposal_id)
            if proposal is None or proposal.task_id != task_id:
                return False
            if proposal.accepted or proposal.rejected:
                return False

            # 接受该提案
            proposal.accepted = True
            self._accepted_proposals[task_id] = proposal_id
            self._task_status[task_id] = "accepted"

            # 拒绝同一任务的其他提案
            for pid in self._task_proposals.get(task_id, []):
                if pid != proposal_id:
                    other = self._proposals.get(pid)
                    if other:
                        other.rejected = True

            return True

    def reject_proposal(self, task_id: str, proposal_id: str) -> bool:
        """拒绝指定提案

        Args:
            task_id: 任务ID
            proposal_id: 要拒绝的提案ID

        Returns:
            是否成功拒绝
        """
        with self._lock:
            proposal = self._proposals.get(proposal_id)
            if proposal is None or proposal.task_id != task_id:
                return False
            if proposal.accepted or proposal.rejected:
                return False

            proposal.rejected = True
            return True

    def get_proposals(self, task_id: str) -> List[Proposal]:
        """获取任务的所有提案"""
        with self._lock:
            proposal_ids = self._task_proposals.get(task_id, [])
            return [
                copy.deepcopy(self._proposals[pid])
                for pid in proposal_ids
                if pid in self._proposals
            ]

    def get_accepted_proposal(self, task_id: str) -> Optional[Proposal]:
        """获取任务已接受的提案"""
        with self._lock:
            pid = self._accepted_proposals.get(task_id)
            if pid and pid in self._proposals:
                return copy.deepcopy(self._proposals[pid])
            return None

    def get_task_status(self, task_id: str) -> Optional[str]:
        """获取任务招标状态"""
        with self._lock:
            return self._task_status.get(task_id)

    def cancel_cfp(self, task_id: str) -> bool:
        """取消招标"""
        with self._lock:
            if task_id not in self._task_status:
                return False
            self._task_status[task_id] = "cancelled"
            # 拒绝所有未处理的提案
            for pid in self._task_proposals.get(task_id, []):
                prop = self._proposals.get(pid)
                if prop and not prop.accepted and not prop.rejected:
                    prop.rejected = True
            return True

    def get_statistics(self) -> Dict[str, Any]:
        """获取合同网协议统计信息"""
        with self._lock:
            total_proposals = len(self._proposals)
            accepted = sum(1 for p in self._proposals.values() if p.accepted)
            rejected = sum(1 for p in self._proposals.values() if p.rejected)
            pending = total_proposals - accepted - rejected
            active_tasks = sum(
                1 for s in self._task_status.values() if s == "cfp"
            )
            return {
                "total_proposals": total_proposals,
                "accepted": accepted,
                "rejected": rejected,
                "pending": pending,
                "active_cfp_tasks": active_tasks,
                "total_tasks": len(self._task_status),
            }
