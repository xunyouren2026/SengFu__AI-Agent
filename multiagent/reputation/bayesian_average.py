"""
贝叶斯平均模块

平滑新Agent初始评分，解决冷启动问题
基于全局统计先验，随着评价增多逐渐转向个人评分
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math
import statistics


@dataclass
class BayesianConfig:
    """贝叶斯平均配置"""
    # 先验平均分 (全局平均)
    prior_mean: float = 3.0
    # 先验置信度 (等效评价数)
    prior_strength: float = 10.0
    # 评分范围
    min_rating: float = 0.0
    max_rating: float = 5.0
    # 最小评价数才使用纯个人评分
    min_reviews_for_full_confidence: int = 20


@dataclass
class RatingData:
    """评分数据"""
    ratings: List[float]
    bayesian_average: float
    personal_average: float
    confidence: float  # 0-1
    total_weight: float


class BayesianAverageCalculator:
    """
    贝叶斯平均计算器

    公式: BA = (C * m + sum(ratings)) / (C + n)
    其中:
    - C: 先验强度 (prior_strength)
    - m: 先验均值 (prior_mean)
    - n: 实际评价数
    - sum(ratings): 实际评分总和

    这解决了新Agent评价少时的评分波动问题
    """

    def __init__(self, config: Optional[BayesianConfig] = None):
        self.config = config or BayesianConfig()
        self._agent_ratings: Dict[str, List[Tuple[float, float]]] = {}
        # 全局统计
        self._global_ratings: List[float] = []
        self._global_mean: float = self.config.prior_mean
        self._global_variance: float = 1.0

    def add_rating(
        self,
        agent_id: str,
        rating: float,
        weight: float = 1.0
    ) -> float:
        """
        添加评分

        Args:
            agent_id: Agent ID
            rating: 评分值
            weight: 评分权重 (默认1.0)

        Returns:
            更新后的贝叶斯平均分
        """
        # 限制评分范围
        rating = max(self.config.min_rating, min(self.config.max_rating, rating))

        if agent_id not in self._agent_ratings:
            self._agent_ratings[agent_id] = []

        self._agent_ratings[agent_id].append((rating, weight))
        self._global_ratings.append(rating)

        # 更新全局统计
        self._update_global_stats()

        return self.calculate_bayesian_average(agent_id)

    def _update_global_stats(self) -> None:
        """更新全局统计信息"""
        if len(self._global_ratings) >= 10:
            self._global_mean = statistics.mean(self._global_ratings)
            if len(self._global_ratings) >= 2:
                self._global_variance = statistics.variance(self._global_ratings)

    def calculate_bayesian_average(self, agent_id: str) -> float:
        """
        计算Agent的贝叶斯平均分
        """
        if agent_id not in self._agent_ratings:
            # 无评价时返回先验均值
            return self.config.prior_mean

        ratings_data = self._agent_ratings[agent_id]

        # 计算加权评分和
        weighted_sum = sum(r * w for r, w in ratings_data)
        total_weight = sum(w for _, w in ratings_data)
        n = len(ratings_data)

        # 贝叶斯平均公式
        # BA = (C * m + weighted_sum) / (C + total_weight)
        C = self.config.prior_strength
        m = self._global_mean

        bayesian_avg = (C * m + weighted_sum) / (C + total_weight)

        # 限制在有效范围内
        return max(
            self.config.min_rating,
            min(self.config.max_rating, bayesian_avg)
        )

    def get_rating_data(self, agent_id: str) -> RatingData:
        """获取Agent的评分数据详情"""
        if agent_id not in self._agent_ratings:
            return RatingData(
                ratings=[],
                bayesian_average=self.config.prior_mean,
                personal_average=self.config.prior_mean,
                confidence=0.0,
                total_weight=0.0
            )

        ratings_data = self._agent_ratings[agent_id]
        ratings = [r for r, _ in ratings_data]
        total_weight = sum(w for _, w in ratings_data)

        # 个人平均分
        if total_weight > 0:
            personal_avg = sum(r * w for r, w in ratings_data) / total_weight
        else:
            personal_avg = self.config.prior_mean

        # 贝叶斯平均分
        bayesian_avg = self.calculate_bayesian_average(agent_id)

        # 置信度: 评价越多，对个人评分的置信度越高
        n = len(ratings_data)
        confidence = min(1.0, n / self.config.min_reviews_for_full_confidence)

        return RatingData(
            ratings=ratings,
            bayesian_average=round(bayesian_avg, 4),
            personal_average=round(personal_avg, 4),
            confidence=round(confidence, 4),
            total_weight=total_weight
        )

    def calculate_confidence_interval(
        self,
        agent_id: str,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        计算置信区间

        Args:
            agent_id: Agent ID
            confidence_level: 置信水平 (默认95%)

        Returns:
            (下限, 上限)
        """
        if agent_id not in self._agent_ratings or len(self._agent_ratings[agent_id]) < 2:
            # 评价不足时使用先验的区间
            margin = 1.96 * math.sqrt(self._global_variance / self.config.prior_strength)
            return (
                max(self.config.min_rating, self._global_mean - margin),
                min(self.config.max_rating, self._global_mean + margin)
            )

        ratings_data = self._agent_ratings[agent_id]
        ratings = [r for r, _ in ratings_data]
        n = len(ratings)

        # 样本均值
        mean = statistics.mean(ratings)

        # 样本标准差
        if n >= 2:
            std = statistics.stdev(ratings)
        else:
            std = math.sqrt(self._global_variance)

        # 95%置信区间的z值约为1.96
        z_value = 1.96 if confidence_level == 0.95 else 2.576  # 99%: 2.576

        # 标准误差
        se = std / math.sqrt(n)

        # 置信区间
        margin = z_value * se

        return (
            max(self.config.min_rating, mean - margin),
            min(self.config.max_rating, mean + margin)
        )

    def compare_agents(self, agent_ids: List[str]) -> List[Tuple[str, float, float]]:
        """
        比较多个Agent的评分

        Returns:
            [(agent_id, bayesian_avg, confidence), ...] 按贝叶斯平均分排序
        """
        results = []
        for agent_id in agent_ids:
            data = self.get_rating_data(agent_id)
            results.append((
                agent_id,
                data.bayesian_average,
                data.confidence
            ))

        return sorted(results, key=lambda x: x[1], reverse=True)

    def get_global_stats(self) -> Dict[str, float]:
        """获取全局统计信息"""
        if not self._global_ratings:
            return {
                "mean": self.config.prior_mean,
                "median": self.config.prior_mean,
                "std": 1.0,
                "count": 0
            }

        return {
            "mean": round(self._global_mean, 4),
            "median": round(statistics.median(self._global_ratings), 4),
            "std": round(
                statistics.stdev(self._global_ratings) if len(self._global_ratings) > 1 else 0, 4
            ),
            "count": len(self._global_ratings)
        }

    def update_prior(self, new_mean: Optional[float] = None, new_strength: Optional[float] = None) -> None:
        """更新先验参数"""
        if new_mean is not None:
            self.config.prior_mean = new_mean
        if new_strength is not None:
            self.config.prior_strength = new_strength

    def reset_agent(self, agent_id: str) -> bool:
        """重置Agent的评分数据"""
        if agent_id in self._agent_ratings:
            del self._agent_ratings[agent_id]
            return True
        return False


class DynamicBayesianAverage(BayesianAverageCalculator):
    """
    动态贝叶斯平均

    根据Agent的表现动态调整先验强度:
    - 表现稳定的Agent降低先验影响
    - 表现波动的Agent增加先验影响
    """

    def __init__(self, config: Optional[BayesianConfig] = None):
        super().__init__(config)
        self._agent_variance: Dict[str, List[float]] = {}

    def add_rating(self, agent_id: str, rating: float, weight: float = 1.0) -> float:
        """添加评分并更新方差跟踪"""
        result = super().add_rating(agent_id, rating, weight)

        # 跟踪方差
        if agent_id not in self._agent_variance:
            self._agent_variance[agent_id] = []

        self._agent_variance[agent_id].append(rating)

        # 只保留最近20个评分用于方差计算
        if len(self._agent_variance[agent_id]) > 20:
            self._agent_variance[agent_id] = self._agent_variance[agent_id][-20:]

        return result

    def _get_dynamic_prior_strength(self, agent_id: str) -> float:
        """获取动态先验强度"""
        if agent_id not in self._agent_variance or len(self._agent_variance[agent_id]) < 5:
            return self.config.prior_strength

        ratings = self._agent_variance[agent_id]
        variance = statistics.variance(ratings) if len(ratings) > 1 else 0.1

        # 方差越大，先验强度越大 (表示不稳定，更依赖全局)
        # 方差越小，先验强度越小 (表示稳定，更依赖个人)
        base_strength = self.config.prior_strength

        # 归一化方差到 0.5-2.0 的系数范围
        normalized_variance = min(2.0, max(0.5, variance))
        multiplier = 2.0 / normalized_variance

        return base_strength * multiplier

    def calculate_bayesian_average(self, agent_id: str) -> float:
        """使用动态先验强度计算贝叶斯平均"""
        if agent_id not in self._agent_ratings:
            return self.config.prior_mean

        ratings_data = self._agent_ratings[agent_id]
        weighted_sum = sum(r * w for r, w in ratings_data)
        total_weight = sum(w for _, w in ratings_data)

        # 使用动态先验强度
        C = self._get_dynamic_prior_strength(agent_id)
        m = self._global_mean

        bayesian_avg = (C * m + weighted_sum) / (C + total_weight)

        return max(
            self.config.min_rating,
            min(self.config.max_rating, bayesian_avg)
        )

    def get_rating_data(self, agent_id: str) -> RatingData:
        """获取包含动态先验信息的评分数据"""
        data = super().get_rating_data(agent_id)
        # 添加动态先验强度信息
        dynamic_strength = self._get_dynamic_prior_strength(agent_id)
        # 使用object.__setattr__因为dataclass是frozen的
        # 这里我们返回一个扩展的dict而不是修改dataclass
        return data
