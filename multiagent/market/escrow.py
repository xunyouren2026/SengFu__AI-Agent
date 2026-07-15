"""
托管支付系统 - 任务完成前资金锁定

实现安全的资金托管机制，支持多阶段释放、条件触发释放等功能。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Callable, Any, Tuple
from collections import defaultdict
import threading


class EscrowStatus(Enum):
    """托管状态"""
    PENDING = auto()           # 待支付
    FUNDED = auto()            # 已充值
    LOCKED = auto()            # 已锁定
    RELEASED = auto()          # 已释放
    DISPUTED = auto()          # 争议中
    REFUNDED = auto()          # 已退款
    CANCELLED = auto()         # 已取消


class ReleaseCondition(Enum):
    """释放条件类型"""
    TASK_COMPLETION = auto()   # 任务完成
    MILESTONE = auto()         # 里程碑达成
    TIME_BASED = auto()        # 时间条件
    MANUAL = auto()            # 手动释放
    ORACLE = auto()            # 预言机确认
    MULTI_SIG = auto()         # 多签确认


@dataclass
class ReleaseStage:
    """释放阶段"""
    stage_id: str
    description: str
    amount: float
    condition: ReleaseCondition
    condition_params: Dict[str, Any] = field(default_factory=dict)
    is_released: bool = False
    released_at: Optional[float] = None
    released_by: Optional[str] = None


@dataclass
class EscrowAccount:
    """托管账户"""
    escrow_id: str
    task_id: str
    payer_id: str
    payee_id: str
    total_amount: float
    status: EscrowStatus
    created_at: float
    expires_at: float
    stages: List[ReleaseStage] = field(default_factory=list)
    released_amount: float = 0.0
    locked_amount: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def remaining_amount(self) -> float:
        """剩余金额"""
        return self.total_amount - self.released_amount
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "escrow_id": self.escrow_id,
            "task_id": self.task_id,
            "payer_id": self.payer_id,
            "payee_id": self.payee_id,
            "total_amount": self.total_amount,
            "status": self.status.name,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "stages": [
                {
                    "stage_id": s.stage_id,
                    "description": s.description,
                    "amount": s.amount,
                    "condition": s.condition.name,
                    "is_released": s.is_released,
                    "released_at": s.released_at
                } for s in self.stages
            ],
            "released_amount": self.released_amount,
            "locked_amount": self.locked_amount,
            "remaining_amount": self.remaining_amount
        }


@dataclass
class PaymentReceipt:
    """支付凭证"""
    receipt_id: str
    escrow_id: str
    amount: float
    timestamp: float
    transaction_hash: Optional[str] = None
    confirmed: bool = False
    confirmed_at: Optional[float] = None


class EscrowManager:
    """托管管理器 - 核心托管逻辑"""
    
    def __init__(self):
        self._escrows: Dict[str, EscrowAccount] = {}
        self._task_escrow: Dict[str, str] = {}  # task_id -> escrow_id
        self._user_escrows: Dict[str, Set[str]] = defaultdict(set)  # user_id -> escrow_ids
        self._receipts: Dict[str, PaymentReceipt] = {}
        self._lock = threading.RLock()
        self._release_callbacks: List[Callable[[EscrowAccount, ReleaseStage], None]] = []
        self._dispute_callbacks: List[Callable[[EscrowAccount], None]] = []
        
        # 模拟资金池
        self._user_balances: Dict[str, float] = defaultdict(float)
    
    def create_escrow(
        self,
        task_id: str,
        payer_id: str,
        payee_id: str,
        total_amount: float,
        stages: Optional[List[Tuple[str, float, ReleaseCondition, Dict[str, Any]]]] = None,
        ttl_hours: float = 168,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EscrowAccount:
        """
        创建托管账户
        
        Args:
            task_id: 关联任务ID
            payer_id: 付款方ID
            payee_id: 收款方ID
            total_amount: 总金额
            stages: 释放阶段列表 [(描述, 金额, 条件, 参数), ...]
            ttl_hours: 托管有效期（小时）
            metadata: 元数据
            
        Returns:
            创建的托管账户
        """
        now = time.time()
        escrow = EscrowAccount(
            escrow_id=str(uuid.uuid4()),
            task_id=task_id,
            payer_id=payer_id,
            payee_id=payee_id,
            total_amount=total_amount,
            status=EscrowStatus.PENDING,
            created_at=now,
            expires_at=now + ttl_hours * 3600,
            metadata=metadata or {}
        )
        
        # 创建默认阶段（如果不提供）
        if not stages:
            stages = [("Task Completion", total_amount, ReleaseCondition.TASK_COMPLETION, {})]
        
        stage_sum = 0.0
        for desc, amount, condition, params in stages:
            stage = ReleaseStage(
                stage_id=str(uuid.uuid4()),
                description=desc,
                amount=amount,
                condition=condition,
                condition_params=params
            )
            escrow.stages.append(stage)
            stage_sum += amount
        
        # 验证阶段金额总和
        if abs(stage_sum - total_amount) > 0.01:
            raise ValueError(f"Stage amounts sum ({stage_sum}) must equal total amount ({total_amount})")
        
        with self._lock:
            self._escrows[escrow.escrow_id] = escrow
            self._task_escrow[task_id] = escrow.escrow_id
            self._user_escrows[payer_id].add(escrow.escrow_id)
            self._user_escrows[payee_id].add(escrow.escrow_id)
        
        return escrow
    
    def fund_escrow(
        self,
        escrow_id: str,
        amount: Optional[float] = None,
        transaction_hash: Optional[str] = None
    ) -> PaymentReceipt:
        """
        为托管账户充值
        
        Args:
            escrow_id: 托管账户ID
            amount: 充值金额（默认全额）
            transaction_hash: 交易哈希
            
        Returns:
            支付凭证
        """
        with self._lock:
            escrow = self._escrows.get(escrow_id)
            if not escrow:
                raise ValueError(f"Escrow {escrow_id} not found")
            
            if escrow.status != EscrowStatus.PENDING:
                raise ValueError(f"Escrow is not in PENDING status, current: {escrow.status.name}")
            
            fund_amount = amount or escrow.total_amount
            
            # 检查付款方余额
            if self._user_balances[escrow.payer_id] < fund_amount:
                raise ValueError(f"Insufficient balance for payer {escrow.payer_id}")
            
            # 扣除余额
            self._user_balances[escrow.payer_id] -= fund_amount
            
            # 创建支付凭证
            receipt = PaymentReceipt(
                receipt_id=str(uuid.uuid4()),
                escrow_id=escrow_id,
                amount=fund_amount,
                timestamp=time.time(),
                transaction_hash=transaction_hash,
                confirmed=True,
                confirmed_at=time.time()
            )
            
            self._receipts[receipt.receipt_id] = receipt
            
            # 更新托管状态
            escrow.status = EscrowStatus.FUNDED
            escrow.locked_amount = fund_amount
            
            if fund_amount >= escrow.total_amount:
                escrow.status = EscrowStatus.LOCKED
            
            return receipt
    
    def release_stage(
        self,
        escrow_id: str,
        stage_id: str,
        released_by: str,
        proof: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        释放指定阶段的资金
        
        Args:
            escrow_id: 托管账户ID
            stage_id: 阶段ID
            released_by: 释放操作者
            proof: 释放证明
            
        Returns:
            是否成功释放
        """
        with self._lock:
            escrow = self._escrows.get(escrow_id)
            if not escrow:
                return False
            
            if escrow.status not in [EscrowStatus.LOCKED, EscrowStatus.FUNDED]:
                return False
            
            stage = next((s for s in escrow.stages if s.stage_id == stage_id), None)
            if not stage or stage.is_released:
                return False
            
            # 验证释放条件
            if not self._validate_release_condition(stage, proof):
                return False
            
            # 执行释放
            stage.is_released = True
            stage.released_at = time.time()
            stage.released_by = released_by
            
            escrow.released_amount += stage.amount
            escrow.locked_amount -= stage.amount
            
            # 转账给收款方
            self._user_balances[escrow.payee_id] += stage.amount
            
            # 检查是否全部释放
            if escrow.released_amount >= escrow.total_amount:
                escrow.status = EscrowStatus.RELEASED
            
            # 触发回调
            for callback in self._release_callbacks:
                try:
                    callback(escrow, stage)
                except Exception:
                    pass
            
            return True
    
    def _validate_release_condition(
        self,
        stage: ReleaseStage,
        proof: Optional[Dict[str, Any]]
    ) -> bool:
        """验证释放条件"""
        if stage.condition == ReleaseCondition.MANUAL:
            return True
        
        elif stage.condition == ReleaseCondition.TASK_COMPLETION:
            if not proof:
                return False
            return proof.get("task_completed", False)
        
        elif stage.condition == ReleaseCondition.MILESTONE:
            if not proof:
                return False
            milestone_id = stage.condition_params.get("milestone_id")
            return proof.get("milestone_id") == milestone_id and proof.get("achieved", False)
        
        elif stage.condition == ReleaseCondition.TIME_BASED:
            release_time = stage.condition_params.get("release_timestamp")
            if release_time:
                return time.time() >= release_time
            return False
        
        elif stage.condition == ReleaseCondition.ORACLE:
            if not proof:
                return False
            return proof.get("oracle_confirmed", False)
        
        elif stage.condition == ReleaseCondition.MULTI_SIG:
            required_sigs = stage.condition_params.get("required_signatures", 2)
            signatures = proof.get("signatures", []) if proof else []
            return len(signatures) >= required_sigs
        
        return False
    
    def release_all(
        self,
        escrow_id: str,
        released_by: str,
        proof: Optional[Dict[str, Any]] = None
    ) -> bool:
        """释放所有未释放的阶段"""
        with self._lock:
            escrow = self._escrows.get(escrow_id)
            if not escrow:
                return False
            
            success = True
            for stage in escrow.stages:
                if not stage.is_released:
                    if not self.release_stage(escrow_id, stage.stage_id, released_by, proof):
                        success = False
            
            return success
    
    def initiate_dispute(
        self,
        escrow_id: str,
        initiator_id: str,
        reason: str
    ) -> bool:
        """
        发起争议
        
        Args:
            escrow_id: 托管账户ID
            initiator_id: 发起者ID
            reason: 争议原因
            
        Returns:
            是否成功发起
        """
        with self._lock:
            escrow = self._escrows.get(escrow_id)
            if not escrow:
                return False
            
            if escrow.status not in [EscrowStatus.LOCKED, EscrowStatus.FUNDED, EscrowStatus.PENDING]:
                return False
            
            if initiator_id not in [escrow.payer_id, escrow.payee_id]:
                return False
            
            escrow.status = EscrowStatus.DISPUTED
            escrow.metadata["dispute_initiated_by"] = initiator_id
            escrow.metadata["dispute_reason"] = reason
            escrow.metadata["dispute_initiated_at"] = time.time()
            
            # 触发回调
            for callback in self._dispute_callbacks:
                try:
                    callback(escrow)
                except Exception:
                    pass
            
            return True
    
    def resolve_dispute(
        self,
        escrow_id: str,
        resolution: str,
        release_stages: Optional[List[str]] = None,
        refund_amount: Optional[float] = None
    ) -> bool:
        """
        解决争议
        
        Args:
            escrow_id: 托管账户ID
            resolution: 解决方案描述
            release_stages: 要释放的阶段ID列表
            refund_amount: 退款金额
            
        Returns:
            是否成功解决
        """
        with self._lock:
            escrow = self._escrows.get(escrow_id)
            if not escrow or escrow.status != EscrowStatus.DISPUTED:
                return False
            
            escrow.metadata["dispute_resolution"] = resolution
            escrow.metadata["dispute_resolved_at"] = time.time()
            
            # 释放指定阶段
            if release_stages:
                for stage_id in release_stages:
                    stage = next((s for s in escrow.stages if s.stage_id == stage_id), None)
                    if stage and not stage.is_released:
                        stage.is_released = True
                        stage.released_at = time.time()
                        escrow.released_amount += stage.amount
                        escrow.locked_amount -= stage.amount
                        self._user_balances[escrow.payee_id] += stage.amount
            
            # 处理退款
            if refund_amount and refund_amount > 0:
                actual_refund = min(refund_amount, escrow.remaining_amount)
                self._user_balances[escrow.payer_id] += actual_refund
                escrow.metadata["refund_amount"] = actual_refund
            
            # 更新状态
            if escrow.released_amount >= escrow.total_amount:
                escrow.status = EscrowStatus.RELEASED
            elif escrow.remaining_amount > 0 and escrow.released_amount == 0:
                escrow.status = EscrowStatus.REFUNDED
            else:
                escrow.status = EscrowStatus.RELEASED
            
            return True
    
    def cancel_escrow(self, escrow_id: str, cancelled_by: str) -> bool:
        """取消托管"""
        with self._lock:
            escrow = self._escrows.get(escrow_id)
            if not escrow:
                return False
            
            if escrow.status not in [EscrowStatus.PENDING, EscrowStatus.FUNDED]:
                return False
            
            # 退款
            if escrow.status == EscrowStatus.FUNDED:
                self._user_balances[escrow.payer_id] += escrow.locked_amount
            
            escrow.status = EscrowStatus.CANCELLED
            escrow.metadata["cancelled_by"] = cancelled_by
            escrow.metadata["cancelled_at"] = time.time()
            
            return True
    
    def get_escrow(self, escrow_id: str) -> Optional[EscrowAccount]:
        """获取托管账户"""
        return self._escrows.get(escrow_id)
    
    def get_escrow_by_task(self, task_id: str) -> Optional[EscrowAccount]:
        """通过任务ID获取托管账户"""
        escrow_id = self._task_escrow.get(task_id)
        return self._escrows.get(escrow_id) if escrow_id else None
    
    def get_user_escrows(
        self,
        user_id: str,
        status_filter: Optional[Set[EscrowStatus]] = None
    ) -> List[EscrowAccount]:
        """获取用户的托管账户"""
        escrow_ids = self._user_escrows.get(user_id, set())
        escrows = [self._escrows[eid] for eid in escrow_ids if eid in self._escrows]
        
        if status_filter:
            escrows = [e for e in escrows if e.status in status_filter]
        
        return escrows
    
    def deposit(self, user_id: str, amount: float) -> float:
        """用户充值"""
        with self._lock:
            self._user_balances[user_id] += amount
            return self._user_balances[user_id]
    
    def get_balance(self, user_id: str) -> float:
        """获取用户余额"""
        return self._user_balances.get(user_id, 0.0)
    
    def withdraw(self, user_id: str, amount: float) -> bool:
        """用户提现"""
        with self._lock:
            if self._user_balances[user_id] < amount:
                return False
            self._user_balances[user_id] -= amount
            return True
    
    def add_release_callback(self, callback: Callable[[EscrowAccount, ReleaseStage], None]) -> None:
        """添加释放回调"""
        self._release_callbacks.append(callback)
    
    def add_dispute_callback(self, callback: Callable[[EscrowAccount], None]) -> None:
        """添加争议回调"""
        self._dispute_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_escrows = len(self._escrows)
        status_counts = defaultdict(int)
        total_locked = 0.0
        total_released = 0.0
        
        for escrow in self._escrows.values():
            status_counts[escrow.status.name] += 1
            if escrow.status in [EscrowStatus.LOCKED, EscrowStatus.FUNDED, EscrowStatus.DISPUTED]:
                total_locked += escrow.locked_amount
            total_released += escrow.released_amount
        
        return {
            "total_escrows": total_escrows,
            "status_distribution": dict(status_counts),
            "total_locked": total_locked,
            "total_released": total_released,
            "total_users": len(self._user_balances),
            "total_balance": sum(self._user_balances.values())
        }
    
    def check_expired_escrows(self) -> List[str]:
        """检查并返回过期的托管账户"""
        now = time.time()
        expired = []
        
        for escrow in self._escrows.values():
            if escrow.status in [EscrowStatus.PENDING, EscrowStatus.FUNDED] and now > escrow.expires_at:
                expired.append(escrow.escrow_id)
        
        return expired
