"""
信誉衰减模块

长期不活动的Agent信誉逐渐下降，保持活跃度才能维持高信誉
"""

from typing import Dict, Optional, Callable
from dataclasses import dataclass
import math
import time


@dataclass
class DecayConfig:
    """衰减配置"""
    # 半衰期 (秒) - 信誉减半所需时间
    halflife_seconds: float = 30 * 24 * 3600  # 默认30天
    # 最小信誉值 (防止信誉归零)
    min_reputation: float = 10.0
    # 最大衰减比例
    max_decay_ratio: float = 0.95
    # 活跃阈值 (秒) - 超过此时间未活动开始衰减
    inactivity_threshold: float = 7 * 24 * 3600  # 默认7天
    # 恢复系数 - 活动后恢复速度
    recovery_factor: float = 0.1


@dataclass
class AgentActivity:
    """Agent活动记录"""
    agent_id: str
    last_active_time: float
    current_reputation: float
    base_reputation: float  # 衰减前的基准信誉
    decay_start_time: Optional[float] = None


class ReputationDecay:
    """
    信誉衰减管理器

    使用指数衰减模型，模拟长期不活动对信誉的影响。
    公式: R(t) = R0 * exp(-λ * t)
    其中 λ = ln(2) / halflife
    """

    def __init__(self, config: Optional[DecayConfig] = None):
        self.config = config or DecayConfig()
        self._agent_activities: Dict[str, AgentActivity] = {}
        self._decay_rate = math.log(2) / self.config.halflife_seconds
        self._activity_callbacks: Dict[str, Callable[[str, float], None]] = {}

    def register_agent(self, agent_id: str, initial_reputation: float) -> None:
        """注册Agent到衰减系统"""
        current_time = time.time()
        self._agent_activities[agent_id] = AgentActivity(
            agent_id=agent_id,
            last_active_time=current_time,
            current_reputation=initial_reputation,
            base_reputation=initial_reputation,
            decay_start_time=None
        )

    def record_activity(self, agent_id: str, reputation_change: float = 0.0) -> float:
        """
        记录Agent活动，更新最后活动时间

        Args:
            agent_id: Agent ID
            reputation_change: 信誉变化值 (可为正负)

        Returns:
            更新后的信誉值
        """
        current_time = time.time()

        if agent_id not in self._agent_activities:
            # 新Agent，使用当前信誉作为初始值
            new_reputation = max(0, 50 + reputation_change)
            self.register_agent(agent_id, new_reputation)
            return new_reputation

        activity = self._agent_activities[agent_id]

        # 先计算衰减后的当前信誉
        decayed_reputation = self._calculate_decayed_reputation(agent_id, current_time)

        # 应用新的信誉变化
        new_reputation = decayed_reputation + reputation_change
        new_reputation = max(self.config.min_reputation, new_reputation)

        # 更新活动记录
        activity.last_active_time = current_time
        activity.current_reputation = new_reputation
        activity.base_reputation = new_reputation
        activity.decay_start_time = None

        # 触发回调
        if agent_id in self._activity_callbacks:
            self._activity_callbacks[agent_id](agent_id, new_reputation)

        return new_reputation

    def get_current_reputation(self, agent_id: str) -> float:
        """获取Agent当前信誉 (考虑衰减)"""
        if agent_id not in self._agent_activities:
            return self.config.min_reputation

        return self._calculate_decayed_reputation(agent_id, time.time())

    def _calculate_decayed_reputation(
        self,
        agent_id: str,
        current_time: float
    ) -> float:
        """计算衰减后的信誉值"""
        activity = self._agent_activities[agent_id]

        # 计算不活动时间
        inactive_duration = current_time - activity.last_active_time

        # 未超过阈值，不衰减
        if inactive_duration <= self.config.inactivity_threshold:
            return activity.base_reputation

        # 计算衰减时间 (从阈值后开始计算)
        decay_duration = inactive_duration - self.config.inactivity_threshold

        # 记录衰减开始时间
        if activity.decay_start_time is None:
            activity.decay_start_time = activity.last_active_time + self.config.inactivity_threshold

        # 指数衰减公式
        decay_factor = math.exp(-self._decay_rate * decay_duration)

        # 限制最大衰减
        decay_factor = max(1 - self.config.max_decay_ratio, decay_factor)

        decayed = activity.base_reputation * decay_factor

        # 不低于最小值
        return max(self.config.min_reputation, decayed)

    def get_inactivity_duration(self, agent_id: str) -> float:
        """获取Agent不活动时长 (秒)"""
        if agent_id not in self._agent_activities:
            return float('inf')

        activity = self._agent_activities[agent_id]
        return time.time() - activity.last_active_time

    def get_decay_status(self, agent_id: str) -> Dict:
        """获取衰减状态详情"""
        if agent_id not in self._agent_activities:
            return {
                "agent_id": agent_id,
                "registered": False,
                "current_reputation": self.config.min_reputation,
                "inactivity_days": float('inf'),
                "is_decaying": False,
                "decay_factor": 1.0
            }

        activity = self._agent_activities[agent_id]
        current_time = time.time()
        inactive_duration = current_time - activity.last_active_time
        current_reputation = self.get_current_reputation(agent_id)

        is_decaying = inactive_duration > self.config.inactivity_threshold

        decay_factor = 1.0
        if is_decaying:
            decay_duration = inactive_duration - self.config.inactivity_threshold
            decay_factor = math.exp(-self._decay_rate * decay_duration)
            decay_factor = max(1 - self.config.max_decay_ratio, decay_factor)

        return {
            "agent_id": agent_id,
            "registered": True,
            "base_reputation": round(activity.base_reputation, 2),
            "current_reputation": round(current_reputation, 2),
            "inactivity_days": round(inactive_duration / 86400, 2),
            "is_decaying": is_decaying,
            "decay_factor": round(decay_factor, 4),
            "decay_percentage": round((1 - decay_factor) * 100, 2)
        }

    def batch_update(self, agent_ids: Optional[list] = None) -> Dict[str, float]:
        """
        批量更新信誉值

        Returns:
            各Agent当前信誉值的字典
        """
        if agent_ids is None:
            agent_ids = list(self._agent_activities.keys())

        results = {}
        for agent_id in agent_ids:
            if agent_id in self._agent_activities:
                results[agent_id] = self.get_current_reputation(agent_id)

        return results

    def set_activity_callback(
        self,
        agent_id: str,
        callback: Callable[[str, float], None]
    ) -> None:
        """设置Agent活动回调函数"""
        self._activity_callbacks[agent_id] = callback

    def remove_agent(self, agent_id: str) -> bool:
        """移除Agent"""
        if agent_id in self._agent_activities:
            del self._agent_activities[agent_id]
            self._activity_callbacks.pop(agent_id, None)
            return True
        return False

    def get_all_agent_status(self) -> Dict[str, Dict]:
        """获取所有Agent的衰减状态"""
        return {
            agent_id: self.get_decay_status(agent_id)
            for agent_id in self._agent_activities.keys()
        }

    def predict_reputation(
        self,
        agent_id: str,
        days_in_future: int
    ) -> float:
        """预测未来某天的信誉值 (假设不活动)"""
        if agent_id not in self._agent_activities:
            return self.config.min_reputation

        activity = self._agent_activities[agent_id]
        base_rep = activity.base_reputation

        # 计算未来时间
        future_time = time.time() + days_in_future * 86400
        inactive_duration = future_time - activity.last_active_time

        if inactive_duration <= self.config.inactivity_threshold:
            return base_rep

        decay_duration = inactive_duration - self.config.inactivity_threshold
        decay_factor = math.exp(-self._decay_rate * decay_duration)
        decay_factor = max(1 - self.config.max_decay_ratio, decay_factor)

        return max(self.config.min_reputation, base_rep * decay_factor)


class AdaptiveDecay(ReputationDecay):
    """
    自适应信誉衰减

    根据Agent历史表现动态调整衰减速度:
    - 高信誉Agent衰减更慢
    - 低信誉Agent衰减更快
    """

    def __init__(self, config: Optional[DecayConfig] = None):
        super().__init__(config)
        self._agent_tiers: Dict[str, int] = {}  # Agent等级

    def set_agent_tier(self, agent_id: str, tier: int) -> None:
        """
        设置Agent等级 (1-5)

        等级越高，衰减越慢
        """
        self._agent_tiers[agent_id] = max(1, min(5, tier))

    def _get_adjusted_halflife(self, agent_id: str) -> float:
        """获取调整后的半衰期"""
        base_halflife = self.config.halflife_seconds
        tier = self._agent_tiers.get(agent_id, 3)

        # 等级系数: 等级1 = 0.5倍, 等级5 = 2倍
        tier_multiplier = 0.5 + (tier - 1) * 0.375

        return base_halflife * tier_multiplier

    def _calculate_decayed_reputation(
        self,
        agent_id: str,
        current_time: float
    ) -> float:
        """计算自适应衰减后的信誉值"""
        if agent_id not in self._agent_activities:
            return self.config.min_reputation

        activity = self._agent_activities[agent_id]
        inactive_duration = current_time - activity.last_active_time

        if inactive_duration <= self.config.inactivity_threshold:
            return activity.base_reputation

        decay_duration = inactive_duration - self.config.inactivity_threshold
        adjusted_halflife = self._get_adjusted_halflife(agent_id)
        adjusted_rate = math.log(2) / adjusted_halflife

        decay_factor = math.exp(-adjusted_rate * decay_duration)
        decay_factor = max(1 - self.config.max_decay_ratio, decay_factor)

        decayed = activity.base_reputation * decay_factor
        return max(self.config.min_reputation, decayed)
