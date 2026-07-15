"""
代币经济学模块

定义任务奖励、惩罚、销毁机制
实现通缩/通胀模型，维护代币经济平衡
"""

from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import math


class TokenAction(Enum):
    """代币操作类型"""
    REWARD = "reward"           # 任务奖励
    PENALTY = "penalty"         # 惩罚扣除
    BURN = "burn"               # 销毁
    MINT = "mint"               # 增发
    TRANSFER = "transfer"       # 转账
    STAKE = "stake"             # 质押
    UNSTAKE = "unstake"         # 解除质押


class TaskDifficulty(Enum):
    """任务难度等级"""
    EASY = 1
    NORMAL = 2
    HARD = 3
    EXPERT = 4
    LEGENDARY = 5


@dataclass
class TokenTransaction:
    """代币交易记录"""
    tx_id: str
    action: TokenAction
    from_agent: Optional[str]
    to_agent: Optional[str]
    amount: float
    timestamp: float
    task_id: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class EconomicPolicy:
    """经济政策配置"""
    # 基础奖励
    base_reward: float = 100.0
    # 难度系数
    difficulty_multiplier: Dict[TaskDifficulty, float] = field(default_factory=lambda: {
        TaskDifficulty.EASY: 0.5,
        TaskDifficulty.NORMAL: 1.0,
        TaskDifficulty.HARD: 2.0,
        TaskDifficulty.EXPERT: 4.0,
        TaskDifficulty.LEGENDARY: 10.0,
    })
    # 惩罚比例
    penalty_rate: float = 0.2
    # 销毁比例 (每笔奖励的销毁比例)
    burn_rate: float = 0.05
    # 质押奖励年化率
    staking_apy: float = 0.1
    # 通胀率上限
    max_inflation_rate: float = 0.05
    # 通缩触发阈值 (流通量/总供应量)
    deflation_threshold: float = 0.8


class TokenEconomy:
    """
    代币经济学管理器

    实现完整的代币经济模型:
    1. 任务奖励计算 (基于难度、质量、时效)
    2. 惩罚机制 (失败、作恶)
    3. 销毁机制 (通缩模型)
    4. 通胀控制
    5. 经济平衡调节
    """

    def __init__(self, policy: Optional[EconomicPolicy] = None):
        self.policy = policy or EconomicPolicy()
        self._balances: Dict[str, float] = {}
        self._transactions: List[TokenTransaction] = []
        self._total_supply: float = 0.0
        self._circulating_supply: float = 0.0
        self._burned_tokens: float = 0.0
        self._minted_tokens: float = 0.0
        self._transaction_callbacks: List[Callable[[TokenTransaction], None]] = []

        # 经济统计数据
        self._daily_stats: Dict[str, Dict] = {}

    def register_callback(self, callback: Callable[[TokenTransaction], None]) -> None:
        """注册交易回调"""
        self._transaction_callbacks.append(callback)

    def get_balance(self, agent_id: str) -> float:
        """获取Agent余额"""
        return self._balances.get(agent_id, 0.0)

    def set_balance(self, agent_id: str, amount: float) -> None:
        """设置Agent余额 (初始化用)"""
        old_balance = self._balances.get(agent_id, 0.0)
        self._balances[agent_id] = max(0.0, amount)

        # 更新供应量
        delta = self._balances[agent_id] - old_balance
        self._circulating_supply += delta

    def calculate_task_reward(
        self,
        difficulty: TaskDifficulty,
        quality_score: float,  # 0-100
        timeliness_score: float,  # 0-100
        base_amount: Optional[float] = None
    ) -> Dict[str, float]:
        """
        计算任务奖励

        公式: 奖励 = 基础奖励 * 难度系数 * 质量系数 * 时效系数

        Returns:
            {
                "gross_reward": 毛奖励,
                "burn_amount": 销毁量,
                "net_reward": 净奖励
            }
        """
        base = base_amount or self.policy.base_reward

        # 难度系数
        difficulty_mult = self.policy.difficulty_multiplier.get(difficulty, 1.0)

        # 质量系数 (0.5 - 1.5)
        quality_mult = 0.5 + (quality_score / 100)

        # 时效系数 (0.8 - 1.2)
        timeliness_mult = 0.8 + (timeliness_score / 100) * 0.4

        # 计算毛奖励
        gross_reward = base * difficulty_mult * quality_mult * timeliness_mult

        # 计算销毁量
        burn_amount = gross_reward * self.policy.burn_rate

        # 净奖励
        net_reward = gross_reward - burn_amount

        return {
            "gross_reward": round(gross_reward, 4),
            "burn_amount": round(burn_amount, 4),
            "net_reward": round(net_reward, 4),
            "difficulty_mult": difficulty_mult,
            "quality_mult": round(quality_mult, 4),
            "timeliness_mult": round(timeliness_mult, 4)
        }

    def reward_agent(
        self,
        tx_id: str,
        agent_id: str,
        difficulty: TaskDifficulty,
        quality_score: float,
        timeliness_score: float,
        task_id: Optional[str] = None
    ) -> TokenTransaction:
        """
        奖励Agent

        Returns:
            交易记录
        """
        reward_calc = self.calculate_task_reward(
            difficulty, quality_score, timeliness_score
        )

        net_reward = reward_calc["net_reward"]
        burn_amount = reward_calc["burn_amount"]

        # 增发净奖励
        self._mint(net_reward)

        # 增加余额
        self._balances[agent_id] = self._balances.get(agent_id, 0.0) + net_reward
        self._circulating_supply += net_reward

        # 销毁部分
        self._burn(burn_amount)

        # 记录交易
        tx = TokenTransaction(
            tx_id=tx_id,
            action=TokenAction.REWARD,
            from_agent=None,
            to_agent=agent_id,
            amount=net_reward,
            timestamp=time.time(),
            task_id=task_id,
            reason=f"Task reward: gross={reward_calc['gross_reward']}, burn={burn_amount}"
        )

        self._transactions.append(tx)
        self._notify_callbacks(tx)
        self._update_daily_stats("reward", net_reward)

        return tx

    def penalize_agent(
        self,
        tx_id: str,
        agent_id: str,
        reason: str,
        penalty_amount: Optional[float] = None,
        task_id: Optional[str] = None
    ) -> Optional[TokenTransaction]:
        """
        惩罚Agent

        从Agent余额中扣除代币，扣除部分销毁
        """
        current_balance = self._balances.get(agent_id, 0.0)

        if penalty_amount is None:
            # 默认惩罚为余额的一定比例
            penalty_amount = current_balance * self.policy.penalty_rate

        # 确保不超过余额
        penalty_amount = min(penalty_amount, current_balance)

        if penalty_amount <= 0:
            return None

        # 扣除余额
        self._balances[agent_id] = current_balance - penalty_amount
        self._circulating_supply -= penalty_amount

        # 销毁惩罚的代币
        self._burned_tokens += penalty_amount

        tx = TokenTransaction(
            tx_id=tx_id,
            action=TokenAction.PENALTY,
            from_agent=agent_id,
            to_agent=None,
            amount=penalty_amount,
            timestamp=time.time(),
            task_id=task_id,
            reason=reason
        )

        self._transactions.append(tx)
        self._notify_callbacks(tx)
        self._update_daily_stats("penalty", penalty_amount)

        return tx

    def _mint(self, amount: float) -> None:
        """增发代币"""
        self._total_supply += amount
        self._minted_tokens += amount

    def _burn(self, amount: float) -> None:
        """销毁代币"""
        self._burned_tokens += amount
        self._total_supply -= amount

    def transfer(
        self,
        tx_id: str,
        from_agent: str,
        to_agent: str,
        amount: float
    ) -> Optional[TokenTransaction]:
        """转账"""
        if self._balances.get(from_agent, 0.0) < amount:
            return None

        self._balances[from_agent] -= amount
        self._balances[to_agent] = self._balances.get(to_agent, 0.0) + amount

        tx = TokenTransaction(
            tx_id=tx_id,
            action=TokenAction.TRANSFER,
            from_agent=from_agent,
            to_agent=to_agent,
            amount=amount,
            timestamp=time.time()
        )

        self._transactions.append(tx)
        self._notify_callbacks(tx)

        return tx

    def get_economic_stats(self) -> Dict:
        """获取经济统计数据"""
        circulation_ratio = (
            self._circulating_supply / self._total_supply
            if self._total_supply > 0 else 0
        )

        # 计算通胀率 (基于最近30天)
        recent_mint = sum(
            tx.amount for tx in self._transactions[-1000:]
            if tx.action == TokenAction.REWARD
        )
        inflation_rate = (
            recent_mint / self._total_supply if self._total_supply > 0 else 0
        )

        return {
            "total_supply": round(self._total_supply, 4),
            "circulating_supply": round(self._circulating_supply, 4),
            "burned_tokens": round(self._burned_tokens, 4),
            "minted_tokens": round(self._minted_tokens, 4),
            "circulation_ratio": round(circulation_ratio, 4),
            "inflation_rate": round(inflation_rate, 6),
            "is_deflationary": self._burned_tokens > self._minted_tokens,
            "total_transactions": len(self._transactions)
        }

    def get_agent_stats(self, agent_id: str) -> Dict:
        """获取Agent经济统计"""
        agent_txs = [
            tx for tx in self._transactions
            if tx.from_agent == agent_id or tx.to_agent == agent_id
        ]

        total_earned = sum(
            tx.amount for tx in agent_txs
            if tx.to_agent == agent_id and tx.action in [TokenAction.REWARD, TokenAction.TRANSFER]
        )

        total_paid = sum(
            tx.amount for tx in agent_txs
            if tx.from_agent == agent_id and tx.action in [TokenAction.PENALTY, TokenAction.TRANSFER]
        )

        total_burned = sum(
            tx.amount for tx in agent_txs
            if tx.from_agent == agent_id and tx.action == TokenAction.PENALTY
        )

        return {
            "agent_id": agent_id,
            "balance": round(self.get_balance(agent_id), 4),
            "total_earned": round(total_earned, 4),
            "total_paid": round(total_paid, 4),
            "total_burned": round(total_burned, 4),
            "net_flow": round(total_earned - total_paid, 4),
            "transaction_count": len(agent_txs)
        }

    def adjust_policy(self, new_policy: EconomicPolicy) -> None:
        """调整经济政策"""
        self.policy = new_policy

    def calculate_staking_reward(
        self,
        agent_id: str,
        staked_amount: float,
        staking_days: int
    ) -> float:
        """
        计算质押奖励

        公式: 奖励 = 质押金额 * 年化率 * (天数/365)
        """
        daily_rate = self.policy.staking_apy / 365
        reward = staked_amount * daily_rate * staking_days
        return round(reward, 4)

    def _notify_callbacks(self, tx: TokenTransaction) -> None:
        """通知回调"""
        for callback in self._transaction_callbacks:
            callback(tx)

    def _update_daily_stats(self, action_type: str, amount: float) -> None:
        """更新每日统计"""
        today = time.strftime("%Y-%m-%d")

        if today not in self._daily_stats:
            self._daily_stats[today] = {
                "rewards": 0.0,
                "penalties": 0.0,
                "burns": 0.0,
                "mints": 0.0
            }

        if action_type == "reward":
            self._daily_stats[today]["rewards"] += amount
            self._daily_stats[today]["mints"] += amount
        elif action_type == "penalty":
            self._daily_stats[today]["penalties"] += amount
            self._daily_stats[today]["burns"] += amount

    def get_daily_stats(self, days: int = 30) -> Dict[str, Dict]:
        """获取最近N天统计"""
        sorted_days = sorted(self._daily_stats.keys(), reverse=True)
        recent_days = sorted_days[:days]

        return {day: self._daily_stats[day] for day in recent_days}

    def is_healthy(self) -> Tuple[bool, str]:
        """
        检查经济健康状况

        Returns:
            (是否健康, 诊断信息)
        """
        stats = self.get_economic_stats()

        # 检查通胀率
        if stats["inflation_rate"] > self.policy.max_inflation_rate:
            return False, f"通胀率过高: {stats['inflation_rate']:.2%}"

        # 检查流通比例
        if stats["circulation_ratio"] < 0.3:
            return False, f"流通比例过低: {stats['circulation_ratio']:.2%}"

        # 检查是否过度通缩
        if stats["burned_tokens"] > stats["minted_tokens"] * 2:
            return False, "通缩过度，可能影响经济活力"

        return True, "经济系统健康"


class DynamicTokenEconomy(TokenEconomy):
    """
    动态代币经济学

    根据市场情况自动调整经济参数
    """

    def __init__(self, policy: Optional[EconomicPolicy] = None):
        super().__init__(policy)
        self._price_history: List[float] = []
        self._adjustment_history: List[Dict] = []

    def record_price(self, price: float) -> None:
        """记录代币价格 (用于动态调整)"""
        self._price_history.append(price)

        # 只保留最近30个价格点
        if len(self._price_history) > 30:
            self._price_history = self._price_history[-30:]

    def auto_adjust(self) -> Optional[Dict]:
        """
        自动调整经济参数

        基于价格趋势和供需关系
        """
        if len(self._price_history) < 7:
            return None

        recent_prices = self._price_history[-7:]
        price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]

        adjustments = {}

        # 价格下跌 -> 减少销毁，增加激励
        if price_trend < -0.1:
            old_burn = self.policy.burn_rate
            self.policy.burn_rate = max(0.01, self.policy.burn_rate * 0.8)
            adjustments["burn_rate"] = (old_burn, self.policy.burn_rate)

        # 价格上涨 -> 增加销毁，控制通胀
        elif price_trend > 0.1:
            old_burn = self.policy.burn_rate
            self.policy.burn_rate = min(0.2, self.policy.burn_rate * 1.1)
            adjustments["burn_rate"] = (old_burn, self.policy.burn_rate)

        if adjustments:
            record = {
                "timestamp": time.time(),
                "price_trend": price_trend,
                "adjustments": adjustments
            }
            self._adjustment_history.append(record)

        return adjustments if adjustments else None

    def get_adjustment_history(self) -> List[Dict]:
        """获取调整历史"""
        return self._adjustment_history
