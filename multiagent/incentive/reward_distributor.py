"""
奖励分配器模块

根据贡献度使用Shapley值分配奖励
公平计算每个参与者在协作任务中的边际贡献
"""

from typing import Dict, List, Set, Tuple, Optional, Callable
from dataclasses import dataclass
from itertools import combinations
import math


@dataclass
class Contribution:
    """贡献记录"""
    agent_id: str
    task_id: str
    contribution_score: float  # 0-100
    effort_hours: float
    skill_level: float  # 1-5
    collaboration_score: float  # 0-100


@dataclass
class ShapleyValue:
    """Shapley值结果"""
    agent_id: str
    value: float
    marginal_contributions: Dict[Tuple[str, ...], float]
    coalition_count: int


class ShapleyCalculator:
    """
    Shapley值计算器

    Shapley值是合作博弈论中的公平分配方案，满足:
    1. 效率性: 所有价值分配给参与者
    2. 对称性: 相同贡献者获得相同分配
    3. 虚拟性: 无贡献者获得零分配
    4. 可加性: 独立博弈的分配可相加

    公式: φ(i) = Σ [|S|!(n-|S|-1)! / n!] * [v(S∪{i}) - v(S)]
    其中S是不包含i的所有子集
    """

    def __init__(self):
        self._characteristic_function: Dict[Tuple[str, ...], float] = {}

    def set_coalition_value(
        self,
        coalition: Tuple[str, ...],
        value: float
    ) -> None:
        """设置联盟的特征函数值"""
        sorted_coalition = tuple(sorted(coalition))
        self._characteristic_function[sorted_coalition] = value

    def get_coalition_value(self, coalition: Tuple[str, ...]) -> float:
        """获取联盟的特征函数值"""
        sorted_coalition = tuple(sorted(coalition))
        return self._characteristic_function.get(sorted_coalition, 0.0)

    def calculate_shapley_value(
        self,
        agent_id: str,
        all_agents: List[str]
    ) -> ShapleyValue:
        """
        计算单个Agent的Shapley值

        Args:
            agent_id: 目标Agent
            all_agents: 所有参与者列表

        Returns:
            ShapleyValue对象
        """
        n = len(all_agents)
        other_agents = [a for a in all_agents if a != agent_id]

        shapley_value = 0.0
        marginal_contributions: Dict[Tuple[str, ...], float] = {}
        coalition_count = 0

        # 遍历所有不包含agent_id的子集
        for r in range(len(other_agents) + 1):
            for subset in combinations(other_agents, r):
                subset = tuple(sorted(subset))

                # 计算权重: |S|!(n-|S|-1)! / n!
                weight = (
                    math.factorial(len(subset)) *
                    math.factorial(n - len(subset) - 1)
                ) / math.factorial(n)

                # 计算边际贡献: v(S∪{i}) - v(S)
                coalition_with_agent = tuple(sorted(subset + (agent_id,)))
                value_with = self.get_coalition_value(coalition_with_agent)
                value_without = self.get_coalition_value(subset)
                marginal_contribution = value_with - value_without

                shapley_value += weight * marginal_contribution
                marginal_contributions[subset] = marginal_contribution
                coalition_count += 1

        return ShapleyValue(
            agent_id=agent_id,
            value=round(shapley_value, 6),
            marginal_contributions=marginal_contributions,
            coalition_count=coalition_count
        )

    def calculate_all_shapley_values(
        self,
        agents: List[str]
    ) -> Dict[str, ShapleyValue]:
        """计算所有Agent的Shapley值"""
        return {
            agent: self.calculate_shapley_value(agent, agents)
            for agent in agents
        }


class RewardDistributor:
    """
    奖励分配器

    基于Shapley值公平分配协作任务奖励
    支持多种贡献度计算方式
    """

    def __init__(self):
        self._contributions: Dict[str, List[Contribution]] = {}
        self._shapley_calculator = ShapleyCalculator()
        self._distribution_callbacks: List[Callable[[str, float, str], None]] = []

    def register_callback(self, callback: Callable[[str, float, str], None]) -> None:
        """注册分配回调"""
        self._distribution_callbacks.append(callback)

    def record_contribution(self, contribution: Contribution) -> None:
        """记录贡献"""
        agent_id = contribution.agent_id

        if agent_id not in self._contributions:
            self._contributions[agent_id] = []

        self._contributions[agent_id].append(contribution)

    def calculate_agent_contribution_score(
        self,
        agent_id: str,
        task_id: Optional[str] = None
    ) -> float:
        """
        计算Agent的综合贡献分

        基于多个维度:
        - 任务完成质量
        - 投入时间
        - 技能水平
        - 协作能力
        """
        if agent_id not in self._contributions:
            return 0.0

        contributions = self._contributions[agent_id]

        if task_id:
            contributions = [c for c in contributions if c.task_id == task_id]

        if not contributions:
            return 0.0

        total_score = 0.0
        total_weight = 0.0

        for c in contributions:
            # 综合贡献分 = 质量*0.4 + 时间*0.2 + 技能*0.2 + 协作*0.2
            score = (
                c.contribution_score * 0.4 +
                min(c.effort_hours / 10, 10) * 10 * 0.2 +  # 归一化到0-100
                c.skill_level * 20 * 0.2 +  # 技能1-5转换为0-100
                c.collaboration_score * 0.2
            )

            weight = c.effort_hours + 1  # 时间越多权重越高
            total_score += score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def build_characteristic_function(
        self,
        agents: List[str],
        total_reward: float
    ) -> None:
        """
        构建特征函数

        基于各Agent的贡献度计算联盟价值
        """
        self._shapley_calculator = ShapleyCalculator()

        # 计算所有子集的价值
        for r in range(len(agents) + 1):
            for subset in combinations(agents, r):
                if not subset:
                    self._shapley_calculator.set_coalition_value((), 0.0)
                    continue

                # 联盟价值 = 成员贡献分之和 / 总贡献分 * 总奖励
                total_contribution = sum(
                    self.calculate_agent_contribution_score(agent)
                    for agent in agents
                )

                coalition_contribution = sum(
                    self.calculate_agent_contribution_score(agent)
                    for agent in subset
                )

                if total_contribution > 0:
                    value = (coalition_contribution / total_contribution) * total_reward
                else:
                    value = total_reward / len(agents) if agents else 0

                self._shapley_calculator.set_coalition_value(
                    tuple(sorted(subset)),
                    value
                )

    def distribute_reward(
        self,
        task_id: str,
        agents: List[str],
        total_reward: float
    ) -> Dict[str, float]:
        """
        分配奖励

        Args:
            task_id: 任务ID
            agents: 参与Agent列表
            total_reward: 总奖励金额

        Returns:
            {agent_id: reward_amount}
        """
        if not agents or total_reward <= 0:
            return {}

        # 构建特征函数
        self.build_characteristic_function(agents, total_reward)

        # 计算Shapley值
        shapley_values = self._shapley_calculator.calculate_all_shapley_values(agents)

        # 转换为奖励分配
        distribution = {}
        total_shapley = sum(sv.value for sv in shapley_values.values())

        for agent_id, sv in shapley_values.items():
            if total_shapley > 0:
                reward = (sv.value / total_shapley) * total_reward
            else:
                reward = total_reward / len(agents)

            distribution[agent_id] = round(reward, 6)

            # 触发回调
            for callback in self._distribution_callbacks:
                callback(agent_id, reward, task_id)

        return distribution

    def calculate_marginal_contribution(
        self,
        agent_id: str,
        coalition: List[str]
    ) -> float:
        """
        计算Agent对特定联盟的边际贡献

        边际贡献 = v(S∪{i}) - v(S)
        """
        coalition_tuple = tuple(sorted(coalition))
        coalition_with_agent = tuple(sorted(coalition + [agent_id]))

        value_with = self._shapley_calculator.get_coalition_value(coalition_with_agent)
        value_without = self._shapley_calculator.get_coalition_value(coalition_tuple)

        return value_with - value_without

    def get_contribution_breakdown(self, agent_id: str) -> Dict:
        """获取Agent贡献详细分解"""
        if agent_id not in self._contributions:
            return {
                "agent_id": agent_id,
                "total_contributions": 0,
                "average_score": 0.0,
                "total_hours": 0.0,
                "tasks": []
            }

        contributions = self._contributions[agent_id]

        tasks = {}
        for c in contributions:
            if c.task_id not in tasks:
                tasks[c.task_id] = {
                    "count": 0,
                    "total_score": 0.0,
                    "total_hours": 0.0
                }
            tasks[c.task_id]["count"] += 1
            tasks[c.task_id]["total_score"] += c.contribution_score
            tasks[c.task_id]["total_hours"] += c.effort_hours

        return {
            "agent_id": agent_id,
            "total_contributions": len(contributions),
            "average_score": round(
                sum(c.contribution_score for c in contributions) / len(contributions), 2
            ),
            "total_hours": round(sum(c.effort_hours for c in contributions), 2),
            "tasks": tasks
        }

    def compare_agents(self, agent_ids: List[str]) -> List[Tuple[str, float]]:
        """比较多个Agent的贡献度"""
        scores = [
            (agent_id, self.calculate_agent_contribution_score(agent_id))
            for agent_id in agent_ids
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)


class ApproximateShapleyCalculator(ShapleyCalculator):
    """
    近似Shapley值计算器

    当参与者较多时，使用蒙特卡洛方法近似计算
    降低计算复杂度从O(n!)到O(n^2 * samples)
    """

    def __init__(self, num_samples: int = 1000):
        super().__init__()
        self.num_samples = num_samples

    def calculate_shapley_value(
        self,
        agent_id: str,
        all_agents: List[str]
    ) -> ShapleyValue:
        """使用蒙特卡洛方法近似计算Shapley值"""
        import random

        n = len(all_agents)
        other_agents = [a for a in all_agents if a != agent_id]

        marginal_contributions_sum = 0.0
        sample_count = 0

        for _ in range(self.num_samples):
            # 随机排列其他Agent
            random.shuffle(other_agents)

            # 找到agent_id加入的位置
            for i in range(len(other_agents) + 1):
                subset = tuple(sorted(other_agents[:i]))

                coalition_with_agent = tuple(sorted(subset + (agent_id,)))
                value_with = self.get_coalition_value(coalition_with_agent)
                value_without = self.get_coalition_value(subset)
                marginal_contribution = value_with - value_without

                marginal_contributions_sum += marginal_contribution
                sample_count += 1

        average_marginal = (
            marginal_contributions_sum / sample_count if sample_count > 0 else 0.0
        )

        return ShapleyValue(
            agent_id=agent_id,
            value=round(average_marginal, 6),
            marginal_contributions={},
            coalition_count=sample_count
        )
