"""
合同网协议 (Contract Net Protocol)

实现经典的合同网协议，支持任务招标→投标→授予→执行的完整流程。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum, auto
from abc import ABC, abstractmethod


class CNPMessageType(Enum):
    """合同网协议消息类型"""
    CALL_FOR_PROPOSAL = auto()  # 招标
    PROPOSAL = auto()           # 投标
    REJECT_PROPOSAL = auto()    # 拒绝投标
    ACCEPT_PROPOSAL = auto()    # 接受投标
    REFUSE = auto()             # 拒绝参与
    INFORM = auto()             # 执行结果通知
    FAILURE = auto()            # 执行失败


class TaskStatus(Enum):
    """任务状态"""
    PENDING = auto()        # 待分配
    ANNOUNCED = auto()      # 已发布招标
    BIDDING = auto()        # 投标中
    ASSIGNED = auto()       # 已分配
    EXECUTING = auto()      # 执行中
    COMPLETED = auto()      # 已完成
    FAILED = auto()         # 失败
    CANCELLED = auto()      # 已取消


@dataclass
class TaskSpecification:
    """任务规格说明"""
    task_id: str
    description: str = ""
    required_capabilities: Set[str] = field(default_factory=set)
    constraints: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[float] = None
    priority: int = 1
    
    def __hash__(self) -> int:
        return hash(self.task_id)


@dataclass
class Proposal:
    """投标书"""
    proposal_id: str
    task_id: str
    contractor_id: str
    bid_price: float
    estimated_duration: float
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        return hash(self.proposal_id)
    
    def __lt__(self, other: Proposal) -> bool:
        """按价格排序（越低越好）"""
        return self.bid_price < other.bid_price


@dataclass
class Contract:
    """合同"""
    contract_id: str
    task_id: str
    manager_id: str
    contractor_id: str
    agreed_price: float
    deadline: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    status: TaskStatus = TaskStatus.ASSIGNED
    result: Optional[Any] = None


@dataclass
class CNPMessage:
    """合同网协议消息"""
    message_id: str
    message_type: CNPMessageType
    sender_id: str
    receiver_id: str
    task_id: Optional[str] = None
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class ContractNetParticipant(ABC):
    """合同网参与者基类"""
    
    def __init__(self, participant_id: str, capabilities: Set[str] = None):
        self.participant_id = participant_id
        self.capabilities = capabilities or set()
        self.reputation: float = 1.0
        self.active_contracts: Dict[str, Contract] = {}
        self.completed_contracts: List[Contract] = []
    
    @abstractmethod
    def evaluate_task(self, task_spec: TaskSpecification) -> Tuple[bool, float]:
        """
        评估任务
        返回: (能否执行, 估价)
        """
        pass
    
    @abstractmethod
    def execute_task(self, contract: Contract) -> Any:
        """执行任务"""
        pass
    
    def can_perform(self, task_spec: TaskSpecification) -> bool:
        """检查是否能执行任务"""
        return task_spec.required_capabilities.issubset(self.capabilities)


class ContractNetManager:
    """合同网管理器（任务管理者）"""
    
    def __init__(self, manager_id: str):
        self.manager_id = manager_id
        self.tasks: Dict[str, TaskSpecification] = {}
        self.task_status: Dict[str, TaskStatus] = {}
        self.proposals: Dict[str, List[Proposal]] = {}  # task_id -> proposals
        self.contracts: Dict[str, Contract] = {}
        self.participants: Dict[str, ContractNetParticipant] = {}
        self.message_history: List[CNPMessage] = []
        
        # 回调函数
        self.on_task_completed: Optional[Callable[[Contract], None]] = None
        self.on_task_failed: Optional[Callable[[Contract, str], None]] = None
    
    def register_participant(self, participant: ContractNetParticipant) -> None:
        """注册参与者"""
        self.participants[participant.participant_id] = participant
    
    def announce_task(
        self,
        task_spec: TaskSpecification,
        eligible_participants: Optional[List[str]] = None
    ) -> str:
        """
        发布任务招标
        
        Args:
            task_spec: 任务规格
            eligible_participants: 指定可参与的参与者，None表示所有
            
        Returns:
            task_id
        """
        self.tasks[task_spec.task_id] = task_spec
        self.task_status[task_spec.task_id] = TaskStatus.ANNOUNCED
        self.proposals[task_spec.task_id] = []
        
        # 确定招标对象
        targets = eligible_participants or list(self.participants.keys())
        
        # 发送招标通知
        for participant_id in targets:
            if participant_id in self.participants:
                message = CNPMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=CNPMessageType.CALL_FOR_PROPOSAL,
                    sender_id=self.manager_id,
                    receiver_id=participant_id,
                    task_id=task_spec.task_id,
                    content={
                        "task_spec": task_spec,
                        "deadline": task_spec.deadline
                    }
                )
                self.message_history.append(message)
                
                # 模拟接收投标
                self._receive_proposal(message)
        
        self.task_status[task_spec.task_id] = TaskStatus.BIDDING
        return task_spec.task_id
    
    def _receive_proposal(self, cfp_message: CNPMessage) -> None:
        """接收投标（模拟）"""
        participant_id = cfp_message.receiver_id
        task_id = cfp_message.task_id
        
        if participant_id not in self.participants or task_id is None:
            return
        
        participant = self.participants[participant_id]
        task_spec = self.tasks.get(task_id)
        
        if not task_spec:
            return
        
        # 参与者评估任务
        can_perform, bid_price = participant.evaluate_task(task_spec)
        
        if can_perform and bid_price > 0:
            proposal = Proposal(
                proposal_id=str(uuid.uuid4()),
                task_id=task_id,
                contractor_id=participant_id,
                bid_price=bid_price,
                estimated_duration=task_spec.constraints.get("estimated_duration", 1.0),
                confidence=participant.reputation
            )
            self.proposals[task_id].append(proposal)
            
            # 发送投标消息
            message = CNPMessage(
                message_id=str(uuid.uuid4()),
                message_type=CNPMessageType.PROPOSAL,
                sender_id=participant_id,
                receiver_id=self.manager_id,
                task_id=task_id,
                content={"proposal": proposal}
            )
            self.message_history.append(message)
        else:
            # 拒绝参与
            message = CNPMessage(
                message_id=str(uuid.uuid4()),
                message_type=CNPMessageType.REFUSE,
                sender_id=participant_id,
                receiver_id=self.manager_id,
                task_id=task_id,
                content={"reason": "Cannot perform task"}
            )
            self.message_history.append(message)
    
    def award_contract(
        self,
        task_id: str,
        selection_strategy: str = "lowest_price"
    ) -> Optional[Contract]:
        """
        授予合同
        
        Args:
            task_id: 任务ID
            selection_strategy: 选择策略 (lowest_price, highest_confidence, best_value)
        """
        if task_id not in self.proposals or not self.proposals[task_id]:
            self.task_status[task_id] = TaskStatus.FAILED
            return None
        
        proposals = self.proposals[task_id]
        
        # 选择最佳投标
        if selection_strategy == "lowest_price":
            best_proposal = min(proposals, key=lambda p: p.bid_price)
        elif selection_strategy == "highest_confidence":
            best_proposal = max(proposals, key=lambda p: p.confidence)
        elif selection_strategy == "best_value":
            # 性价比 = 置信度 / 价格
            best_proposal = max(proposals, key=lambda p: p.confidence / max(p.bid_price, 0.001))
        else:
            best_proposal = min(proposals, key=lambda p: p.bid_price)
        
        # 创建合同
        contract = Contract(
            contract_id=str(uuid.uuid4()),
            task_id=task_id,
            manager_id=self.manager_id,
            contractor_id=best_proposal.contractor_id,
            agreed_price=best_proposal.bid_price,
            deadline=self.tasks[task_id].deadline
        )
        
        self.contracts[contract.contract_id] = contract
        self.task_status[task_id] = TaskStatus.ASSIGNED
        
        # 发送接受通知
        accept_message = CNPMessage(
            message_id=str(uuid.uuid4()),
            message_type=CNPMessageType.ACCEPT_PROPOSAL,
            sender_id=self.manager_id,
            receiver_id=best_proposal.contractor_id,
            task_id=task_id,
            content={"contract": contract}
        )
        self.message_history.append(accept_message)
        
        # 发送拒绝通知给其他投标者
        for proposal in proposals:
            if proposal.proposal_id != best_proposal.proposal_id:
                reject_message = CNPMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=CNPMessageType.REJECT_PROPOSAL,
                    sender_id=self.manager_id,
                    receiver_id=proposal.contractor_id,
                    task_id=task_id,
                    content={"reason": "Better proposal selected"}
                )
                self.message_history.append(reject_message)
        
        return contract
    
    def execute_contract(self, contract_id: str) -> bool:
        """执行合同"""
        if contract_id not in self.contracts:
            return False
        
        contract = self.contracts[contract_id]
        task_id = contract.task_id
        
        self.task_status[task_id] = TaskStatus.EXECUTING
        contract.status = TaskStatus.EXECUTING
        
        # 获取承包商
        contractor = self.participants.get(contract.contractor_id)
        if not contractor:
            contract.status = TaskStatus.FAILED
            self.task_status[task_id] = TaskStatus.FAILED
            return False
        
        try:
            # 执行任务
            result = contractor.execute_task(contract)
            
            # 更新合同状态
            contract.status = TaskStatus.COMPLETED
            contract.completed_at = time.time()
            contract.result = result
            
            self.task_status[task_id] = TaskStatus.COMPLETED
            contractor.completed_contracts.append(contract)
            
            # 发送完成通知
            inform_message = CNPMessage(
                message_id=str(uuid.uuid4()),
                message_type=CNPMessageType.INFORM,
                sender_id=contract.contractor_id,
                receiver_id=self.manager_id,
                task_id=task_id,
                content={"result": result, "contract_id": contract_id}
            )
            self.message_history.append(inform_message)
            
            if self.on_task_completed:
                self.on_task_completed(contract)
            
            return True
            
        except Exception as e:
            contract.status = TaskStatus.FAILED
            self.task_status[task_id] = TaskStatus.FAILED
            
            # 发送失败通知
            failure_message = CNPMessage(
                message_id=str(uuid.uuid4()),
                message_type=CNPMessageType.FAILURE,
                sender_id=contract.contractor_id,
                receiver_id=self.manager_id,
                task_id=task_id,
                content={"error": str(e), "contract_id": contract_id}
            )
            self.message_history.append(failure_message)
            
            if self.on_task_failed:
                self.on_task_failed(contract, str(e))
            
            return False
    
    def get_task_statistics(self, task_id: str) -> Dict[str, Any]:
        """获取任务统计"""
        if task_id not in self.tasks:
            return {}
        
        proposals = self.proposals.get(task_id, [])
        
        stats = {
            "task_id": task_id,
            "status": self.task_status.get(task_id, TaskStatus.PENDING),
            "num_proposals": len(proposals),
            "proposals": [
                {
                    "contractor_id": p.contractor_id,
                    "bid_price": p.bid_price,
                    "confidence": p.confidence
                }
                for p in proposals
            ]
        }
        
        # 找到对应的合同
        for contract in self.contracts.values():
            if contract.task_id == task_id:
                stats["contract"] = {
                    "contract_id": contract.contract_id,
                    "contractor_id": contract.contractor_id,
                    "agreed_price": contract.agreed_price,
                    "status": contract.status
                }
                break
        
        return stats


class SimpleContractor(ContractNetParticipant):
    """简单承包商实现"""
    
    def __init__(
        self,
        participant_id: str,
        capabilities: Set[str],
        cost_factor: float = 1.0,
        success_rate: float = 0.95
    ):
        super().__init__(participant_id, capabilities)
        self.cost_factor = cost_factor
        self.success_rate = success_rate
    
    def evaluate_task(self, task_spec: TaskSpecification) -> Tuple[bool, float]:
        """评估任务"""
        if not self.can_perform(task_spec):
            return False, 0.0
        
        # 基于任务复杂度估价
        complexity = len(task_spec.required_capabilities)
        base_cost = complexity * 10.0 * self.cost_factor
        
        # 加入随机因素
        import random
        noise = random.uniform(-0.1, 0.1) * base_cost
        estimated_cost = max(1.0, base_cost + noise)
        
        return True, estimated_cost
    
    def execute_task(self, contract: Contract) -> Any:
        """执行任务"""
        import random
        
        # 模拟执行时间
        time.sleep(0.001)
        
        # 模拟成功率
        if random.random() > self.success_rate:
            raise Exception("Task execution failed")
        
        # 返回模拟结果
        return {
            "contract_id": contract.contract_id,
            "executed_by": self.participant_id,
            "result": f"Task {contract.task_id} completed successfully"
        }


class ContractNetSystem:
    """合同网系统"""
    
    def __init__(self):
        self.managers: Dict[str, ContractNetManager] = {}
        self.participants: Dict[str, ContractNetParticipant] = {}
    
    def create_manager(self, manager_id: str) -> ContractNetManager:
        """创建管理器"""
        manager = ContractNetManager(manager_id)
        self.managers[manager_id] = manager
        return manager
    
    def register_participant(self, participant: ContractNetParticipant) -> None:
        """注册参与者"""
        self.participants[participant.participant_id] = participant
        # 同时注册到所有管理器
        for manager in self.managers.values():
            manager.register_participant(participant)
    
    def distribute_task(
        self,
        task_spec: TaskSpecification,
        manager_id: Optional[str] = None
    ) -> Optional[Contract]:
        """
        分发任务
        
        完整的合同网协议流程：
        1. 发布招标
        2. 收集投标
        3. 授予合同
        4. 执行合同
        """
        # 选择管理器
        if manager_id and manager_id in self.managers:
            manager = self.managers[manager_id]
        elif self.managers:
            manager = list(self.managers.values())[0]
        else:
            manager = self.create_manager("default_manager")
        
        # 注册所有参与者
        for participant in self.participants.values():
            if participant.participant_id not in manager.participants:
                manager.register_participant(participant)
        
        # 1. 发布招标
        task_id = manager.announce_task(task_spec)
        
        # 2. 收集投标（已在announce_task中完成）
        
        # 3. 授予合同
        contract = manager.award_contract(task_id)
        if not contract:
            return None
        
        # 4. 执行合同
        success = manager.execute_contract(contract.contract_id)
        
        if success:
            return contract
        return None
