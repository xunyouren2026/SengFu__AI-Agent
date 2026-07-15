"""
质押机制模块

Agent需质押代币才能接高风险任务
质押可获得收益，同时作为作恶惩罚的抵押
"""

from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import math


class StakingStatus(Enum):
    """质押状态"""
    ACTIVE = "active"           # 质押中
    UNSTAKING = "unstaking"     # 解除质押中
    WITHDRAWN = "withdrawn"     # 已提取
    SLASHED = "slashed"         # 已被罚没


class TaskRiskLevel(Enum):
    """任务风险等级"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Stake:
    """质押记录"""
    stake_id: str
    agent_id: str
    amount: float
    start_time: float
    lock_period_days: int
    apy: float
    status: StakingStatus
    accumulated_rewards: float = 0.0
    unstake_time: Optional[float] = None
    withdraw_time: Optional[float] = None


@dataclass
class StakingRequirement:
    """质押要求"""
    risk_level: TaskRiskLevel
    min_stake_amount: float
    min_staking_days: int
    reputation_threshold: float


class StakingManager:
    """
    质押管理器

    实现质押机制:
    1. Agent质押代币获得收益
    2. 不同风险等级任务需要不同质押门槛
    3. 质押有锁定期，提前解除有惩罚
    4. 质押作为作恶惩罚的抵押
    """

    # 默认质押要求配置
    DEFAULT_REQUIREMENTS: Dict[TaskRiskLevel, StakingRequirement] = {
        TaskRiskLevel.LOW: StakingRequirement(
            risk_level=TaskRiskLevel.LOW,
            min_stake_amount=100.0,
            min_staking_days=7,
            reputation_threshold=20.0
        ),
        TaskRiskLevel.MEDIUM: StakingRequirement(
            risk_level=TaskRiskLevel.MEDIUM,
            min_stake_amount=500.0,
            min_staking_days=14,
            reputation_threshold=40.0
        ),
        TaskRiskLevel.HIGH: StakingRequirement(
            risk_level=TaskRiskLevel.HIGH,
            min_stake_amount=2000.0,
            min_staking_days=30,
            reputation_threshold=60.0
        ),
        TaskRiskLevel.CRITICAL: StakingRequirement(
            risk_level=TaskRiskLevel.CRITICAL,
            min_stake_amount=10000.0,
            min_staking_days=90,
            reputation_threshold=80.0
        ),
    }

    # 年化收益率配置 (根据质押时长)
    APY_TIERS: Dict[int, float] = {
        7: 0.05,     # 7天: 5%
        30: 0.08,    # 30天: 8%
        90: 0.12,    # 90天: 12%
        180: 0.15,   # 180天: 15%
        365: 0.20,   # 365天: 20%
    }

    # 提前解除惩罚比例
    EARLY_UNSTAKE_PENALTY: float = 0.1

    def __init__(
        self,
        requirements: Optional[Dict[TaskRiskLevel, StakingRequirement]] = None
    ):
        self.requirements = requirements or self.DEFAULT_REQUIREMENTS.copy()
        self._stakes: Dict[str, Stake] = {}
        self._agent_stakes: Dict[str, List[str]] = {}
        self._agent_reputations: Dict[str, float] = {}
        self._staking_callbacks: List[Callable[[Stake], None]] = []
        self._unstaking_callbacks: List[Callable[[Stake], None]] = []

    def register_staking_callback(self, callback: Callable[[Stake], None]) -> None:
        """注册质押回调"""
        self._staking_callbacks.append(callback)

    def register_unstaking_callback(self, callback: Callable[[Stake], None]) -> None:
        """注册解除质押回调"""
        self._unstaking_callbacks.append(callback)

    def _get_apy(self, lock_period_days: int) -> float:
        """根据锁定期获取年化收益率"""
        # 找到最适合的APY档位
        applicable_apy = 0.05  # 默认最低5%

        for days, apy in sorted(self.APY_TIERS.items()):
            if lock_period_days >= days:
                applicable_apy = apy

        return applicable_apy

    def can_stake(
        self,
        agent_id: str,
        amount: float,
        agent_balance: float
    ) -> Tuple[bool, str]:
        """
        检查是否可以质押

        Returns:
            (是否可以, 原因)
        """
        if amount <= 0:
            return False, "质押金额必须大于0"

        if agent_balance < amount:
            return False, "余额不足"

        return True, "可以质押"

    def stake(
        self,
        stake_id: str,
        agent_id: str,
        amount: float,
        lock_period_days: int
    ) -> Optional[Stake]:
        """
        质押代币

        Args:
            stake_id: 质押ID
            agent_id: Agent ID
            amount: 质押金额
            lock_period_days: 锁定期天数

        Returns:
            Stake对象或None
        """
        if lock_period_days < 1:
            return None

        apy = self._get_apy(lock_period_days)

        stake = Stake(
            stake_id=stake_id,
            agent_id=agent_id,
            amount=amount,
            start_time=time.time(),
            lock_period_days=lock_period_days,
            apy=apy,
            status=StakingStatus.ACTIVE
        )

        self._stakes[stake_id] = stake

        if agent_id not in self._agent_stakes:
            self._agent_stakes[agent_id] = []
        self._agent_stakes[agent_id].append(stake_id)

        # 触发回调
        for callback in self._staking_callbacks:
            callback(stake)

        return stake

    def calculate_rewards(self, stake_id: str) -> float:
        """
        计算质押收益

        公式: 收益 = 质押金额 * 年化率 * (质押天数/365)
        """
        if stake_id not in self._stakes:
            return 0.0

        stake = self._stakes[stake_id]

        if stake.status != StakingStatus.ACTIVE:
            return stake.accumulated_rewards

        current_time = time.time()
        staking_duration_days = (current_time - stake.start_time) / 86400

        # 计算收益
        rewards = stake.amount * stake.apy * (staking_duration_days / 365)

        return rewards

    def start_unstaking(self, stake_id: str) -> Tuple[bool, float, str]:
        """
        开始解除质押

        Returns:
            (是否成功, 实际可提取金额, 消息)
        """
        if stake_id not in self._stakes:
            return False, 0.0, "质押不存在"

        stake = self._stakes[stake_id]

        if stake.status != StakingStatus.ACTIVE:
            return False, 0.0, "质押状态不允许解除"

        current_time = time.time()
        staking_duration_days = (current_time - stake.start_time) / 86400

        # 计算最终收益
        final_rewards = self.calculate_rewards(stake_id)
        stake.accumulated_rewards = final_rewards

        # 检查是否提前解除
        penalty = 0.0
        if staking_duration_days < stake.lock_period_days:
            # 提前解除惩罚
            penalty = (stake.amount + final_rewards) * self.EARLY_UNSTAKE_PENALTY

        withdraw_amount = stake.amount + final_rewards - penalty

        stake.status = StakingStatus.UNSTAKING
        stake.unstake_time = current_time

        # 触发回调
        for callback in self._unstaking_callbacks:
            callback(stake)

        message = f"解除质押成功"
        if penalty > 0:
            message += f"，提前解除惩罚: {penalty:.4f}"

        return True, withdraw_amount, message

    def complete_withdrawal(self, stake_id: str) -> Optional[float]:
        """完成提取"""
        if stake_id not in self._stakes:
            return None

        stake = self._stakes[stake_id]

        if stake.status != StakingStatus.UNSTAKING:
            return None

        stake.status = StakingStatus.WITHDRAWN
        stake.withdraw_time = time.time()

        # 计算最终可提取金额
        final_rewards = stake.accumulated_rewards
        withdraw_amount = stake.amount + final_rewards

        # 检查是否提前解除的惩罚已在start_unstaking中计算
        current_time = time.time()
        staking_duration_days = (current_time - stake.start_time) / 86400

        if staking_duration_days < stake.lock_period_days:
            penalty = withdraw_amount * self.EARLY_UNSTAKE_PENALTY
            withdraw_amount -= penalty

        return withdraw_amount

    def slash_stake(
        self,
        stake_id: str,
        slash_percentage: float
    ) -> Tuple[bool, float]:
        """
        罚没质押

        Args:
            stake_id: 质押ID
            slash_percentage: 罚没比例 (0-1)

        Returns:
            (是否成功, 罚没金额)
        """
        if stake_id not in self._stakes:
            return False, 0.0

        stake = self._stakes[stake_id]

        if stake.status not in [StakingStatus.ACTIVE, StakingStatus.UNSTAKING]:
            return False, 0.0

        slash_percentage = max(0.0, min(1.0, slash_percentage))

        # 计算罚没金额 (包含收益)
        total_value = stake.amount + stake.accumulated_rewards
        slash_amount = total_value * slash_percentage

        stake.amount = total_value - slash_amount
        stake.accumulated_rewards = 0.0
        stake.status = StakingStatus.SLASHED

        return True, slash_amount

    def check_task_eligibility(
        self,
        agent_id: str,
        risk_level: TaskRiskLevel
    ) -> Tuple[bool, str]:
        """
        检查Agent是否有资格接取某风险等级的任务

        Returns:
            (是否有资格, 原因)
        """
        requirement = self.requirements.get(risk_level)
        if not requirement:
            return False, "未知的任务风险等级"

        # 获取Agent总质押金额
        total_staked = self.get_total_staked_amount(agent_id)

        if total_staked < requirement.min_stake_amount:
            return False, (
                f"质押金额不足，需要至少 {requirement.min_stake_amount}，"
                f"当前 {total_staked}"
            )

        # 检查是否有足够时长的质押
        has_valid_stake = self._has_valid_stake_duration(
            agent_id, requirement.min_staking_days
        )

        if not has_valid_stake:
            return False, f"需要至少 {requirement.min_staking_days} 天的质押记录"

        # 检查信誉
        reputation = self._agent_reputations.get(agent_id, 0)
        if reputation < requirement.reputation_threshold:
            return False, (
                f"信誉分不足，需要至少 {requirement.reputation_threshold}，"
                f"当前 {reputation}"
            )

        return True, "有资格接取任务"

    def _has_valid_stake_duration(
        self,
        agent_id: str,
        min_days: int
    ) -> bool:
        """检查Agent是否有满足最小时长的质押"""
        stake_ids = self._agent_stakes.get(agent_id, [])

        for stake_id in stake_ids:
            stake = self._stakes.get(stake_id)
            if stake and stake.status == StakingStatus.ACTIVE:
                if stake.lock_period_days >= min_days:
                    return True

        return False

    def get_total_staked_amount(self, agent_id: str) -> float:
        """获取Agent总质押金额"""
        stake_ids = self._agent_stakes.get(agent_id, [])

        total = 0.0
        for stake_id in stake_ids:
            stake = self._stakes.get(stake_id)
            if stake and stake.status == StakingStatus.ACTIVE:
                total += stake.amount

        return total

    def get_stake_info(self, stake_id: str) -> Optional[Dict]:
        """获取质押详情"""
        if stake_id not in self._stakes:
            return None

        stake = self._stakes[stake_id]
        current_rewards = self.calculate_rewards(stake_id)

        current_time = time.time()
        staking_duration_days = (current_time - stake.start_time) / 86400
        remaining_lock_days = max(0, stake.lock_period_days - staking_duration_days)

        return {
            "stake_id": stake.stake_id,
            "agent_id": stake.agent_id,
            "amount": stake.amount,
            "status": stake.status.value,
            "apy": stake.apy,
            "lock_period_days": stake.lock_period_days,
            "staking_duration_days": round(staking_duration_days, 2),
            "remaining_lock_days": round(remaining_lock_days, 2),
            "accumulated_rewards": round(stake.accumulated_rewards, 4),
            "current_rewards": round(current_rewards, 4),
            "total_value": round(stake.amount + current_rewards, 4)
        }

    def get_agent_staking_summary(self, agent_id: str) -> Dict:
        """获取Agent质押摘要"""
        stake_ids = self._agent_stakes.get(agent_id, [])

        active_stakes = []
        total_staked = 0.0
        total_rewards = 0.0

        for stake_id in stake_ids:
            stake = self._stakes.get(stake_id)
            if stake:
                info = self.get_stake_info(stake_id)
                if info:
                    active_stakes.append(info)
                    if stake.status == StakingStatus.ACTIVE:
                        total_staked += stake.amount
                        total_rewards += info["current_rewards"]

        return {
            "agent_id": agent_id,
            "total_stakes": len(stake_ids),
            "active_stakes": len([s for s in active_stakes if s["status"] == "active"]),
            "total_staked": round(total_staked, 4),
            "total_rewards": round(total_rewards, 4),
            "total_value": round(total_staked + total_rewards, 4),
            "stakes": active_stakes
        }

    def update_agent_reputation(self, agent_id: str, reputation: float) -> None:
        """更新Agent信誉"""
        self._agent_reputations[agent_id] = reputation

    def get_eligible_risk_levels(self, agent_id: str) -> List[TaskRiskLevel]:
        """获取Agent有资格接取的所有风险等级"""
        eligible = []

        for risk_level in TaskRiskLevel:
            can_accept, _ = self.check_task_eligibility(agent_id, risk_level)
            if can_accept:
                eligible.append(risk_level)

        return eligible

    def compound_rewards(self, stake_id: str) -> Optional[Stake]:
        """
        复利 - 将收益加入本金继续质押

        Returns:
            新的质押记录
        """
        if stake_id not in self._stakes:
            return None

        old_stake = self._stakes[stake_id]

        if old_stake.status != StakingStatus.ACTIVE:
            return None

        rewards = self.calculate_rewards(stake_id)
        new_amount = old_stake.amount + rewards

        # 创建新质押
        new_stake_id = f"{stake_id}_compound"
        new_stake = self.stake(
            stake_id=new_stake_id,
            agent_id=old_stake.agent_id,
            amount=new_amount,
            lock_period_days=old_stake.lock_period_days
        )

        if new_stake:
            # 标记旧质押为已提取
            old_stake.status = StakingStatus.WITHDRAWN
            old_stake.accumulated_rewards = 0.0

        return new_stake
