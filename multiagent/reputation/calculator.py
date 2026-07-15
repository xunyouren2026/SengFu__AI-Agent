"""
信誉分计算器模块

融合成功率、响应速度、评价反馈计算综合信誉分
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math
import time


class ReputationLevel(Enum):
    """信誉等级"""
    UNTRUSTED = 0      # 不可信 (< 20)
    NOVICE = 1         # 新手 (20-40)
    RELIABLE = 2       # 可靠 (40-60)
    TRUSTED = 3        # 可信 (60-80)
    EXPERT = 4         # 专家 (80-95)
    LEGENDARY = 5      # 传奇 (>= 95)


@dataclass
class TaskMetrics:
    """任务执行指标"""
    task_id: str
    success: bool
    response_time_ms: float
    quality_score: float  # 0-100
    timestamp: float


@dataclass
class ReputationScore:
    """信誉分数据结构"""
    agent_id: str
    overall_score: float  # 0-100
    success_rate_score: float
    response_speed_score: float
    feedback_score: float
    level: ReputationLevel
    last_updated: float


class ReputationCalculator:
    """
    信誉分计算器

    使用加权融合算法，综合考虑:
    - 任务成功率 (40%)
    - 响应速度 (25%)
    - 评价反馈 (35%)
    """

    # 权重配置
    WEIGHT_SUCCESS_RATE = 0.40
    WEIGHT_RESPONSE_SPEED = 0.25
    WEIGHT_FEEDBACK = 0.35

    # 响应速度基准 (毫秒)
    BASELINE_FAST = 500      # < 500ms 为优秀
    BASELINE_NORMAL = 2000   # < 2000ms 为良好
    BASELINE_SLOW = 5000     # < 5000ms 为及格

    def __init__(self):
        self._agent_metrics: Dict[str, List[TaskMetrics]] = {}
        self._agent_feedbacks: Dict[str, List[float]] = {}

    def record_task_completion(
        self,
        agent_id: str,
        task_id: str,
        success: bool,
        response_time_ms: float,
        quality_score: float
    ) -> None:
        """记录任务完成指标"""
        if agent_id not in self._agent_metrics:
            self._agent_metrics[agent_id] = []

        metric = TaskMetrics(
            task_id=task_id,
            success=success,
            response_time_ms=response_time_ms,
            quality_score=quality_score,
            timestamp=time.time()
        )
        self._agent_metrics[agent_id].append(metric)

        # 只保留最近100条记录
        if len(self._agent_metrics[agent_id]) > 100:
            self._agent_metrics[agent_id] = self._agent_metrics[agent_id][-100:]

    def record_feedback(self, agent_id: str, rating: float) -> None:
        """记录评价反馈 (0-5分)"""
        if agent_id not in self._agent_feedbacks:
            self._agent_feedbacks[agent_id] = []

        # 归一化到 0-100
        normalized_rating = (rating / 5.0) * 100
        self._agent_feedbacks[agent_id].append(normalized_rating)

        # 只保留最近50条评价
        if len(self._agent_feedbacks[agent_id]) > 50:
            self._agent_feedbacks[agent_id] = self._agent_feedbacks[agent_id][-50:]

    def calculate_success_rate_score(self, agent_id: str) -> float:
        """计算成功率得分 (0-100)"""
        if agent_id not in self._agent_metrics or not self._agent_metrics[agent_id]:
            return 50.0  # 默认中等分数

        metrics = self._agent_metrics[agent_id]
        total_tasks = len(metrics)
        successful_tasks = sum(1 for m in metrics if m.success)

        # 基础成功率
        base_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.5

        # 考虑任务质量加权
        weighted_sum = 0.0
        total_weight = 0.0

        for i, metric in enumerate(metrics):
            # 时间衰减权重，越近权重越高
            weight = math.exp(0.01 * (i - len(metrics)))
            task_score = 100.0 if metric.success else 0.0
            # 成功时考虑质量分
            if metric.success:
                task_score = 0.7 * 100 + 0.3 * metric.quality_score

            weighted_sum += task_score * weight
            total_weight += weight

        weighted_score = weighted_sum / total_weight if total_weight > 0 else 50.0

        # 结合基础成功率和加权分数
        final_score = 0.6 * base_rate * 100 + 0.4 * weighted_score

        return min(100.0, max(0.0, final_score))

    def calculate_response_speed_score(self, agent_id: str) -> float:
        """计算响应速度得分 (0-100)"""
        if agent_id not in self._agent_metrics or not self._agent_metrics[agent_id]:
            return 50.0

        metrics = self._agent_metrics[agent_id]

        # 计算平均响应时间 (带时间衰减)
        weighted_time = 0.0
        total_weight = 0.0

        for i, metric in enumerate(metrics):
            weight = math.exp(0.02 * (i - len(metrics)))
            weighted_time += metric.response_time_ms * weight
            total_weight += weight

        avg_time = weighted_time / total_weight if total_weight > 0 else self.BASELINE_NORMAL

        # 根据平均响应时间计算分数
        if avg_time <= self.BASELINE_FAST:
            score = 100 - (avg_time / self.BASELINE_FAST) * 10
        elif avg_time <= self.BASELINE_NORMAL:
            score = 90 - ((avg_time - self.BASELINE_FAST) /
                         (self.BASELINE_NORMAL - self.BASELINE_FAST)) * 20
        elif avg_time <= self.BASELINE_SLOW:
            score = 70 - ((avg_time - self.BASELINE_NORMAL) /
                         (self.BASELINE_SLOW - self.BASELINE_NORMAL)) * 40
        else:
            score = max(0, 30 - (avg_time - self.BASELINE_SLOW) / 100)

        return min(100.0, max(0.0, score))

    def calculate_feedback_score(self, agent_id: str) -> float:
        """计算评价反馈得分 (0-100)"""
        if agent_id not in self._agent_feedbacks or not self._agent_feedbacks[agent_id]:
            return 50.0  # 默认中等分数

        feedbacks = self._agent_feedbacks[agent_id]

        # 使用指数加权移动平均
        alpha = 0.3
        ema = feedbacks[0]

        for feedback in feedbacks[1:]:
            ema = alpha * feedback + (1 - alpha) * ema

        # 考虑评价数量进行置信度调整
        confidence = min(1.0, len(feedbacks) / 20)  # 20条评价达到满置信度
        adjusted_score = 50 + (ema - 50) * confidence

        return min(100.0, max(0.0, adjusted_score))

    def calculate_reputation(self, agent_id: str) -> ReputationScore:
        """计算综合信誉分"""
        success_rate = self.calculate_success_rate_score(agent_id)
        response_speed = self.calculate_response_speed_score(agent_id)
        feedback = self.calculate_feedback_score(agent_id)

        # 加权融合
        overall = (
            self.WEIGHT_SUCCESS_RATE * success_rate +
            self.WEIGHT_RESPONSE_SPEED * response_speed +
            self.WEIGHT_FEEDBACK * feedback
        )

        # 确定等级
        level = self._get_level(overall)

        return ReputationScore(
            agent_id=agent_id,
            overall_score=round(overall, 2),
            success_rate_score=round(success_rate, 2),
            response_speed_score=round(response_speed, 2),
            feedback_score=round(feedback, 2),
            level=level,
            last_updated=time.time()
        )

    def _get_level(self, score: float) -> ReputationLevel:
        """根据分数确定等级"""
        if score >= 95:
            return ReputationLevel.LEGENDARY
        elif score >= 80:
            return ReputationLevel.EXPERT
        elif score >= 60:
            return ReputationLevel.TRUSTED
        elif score >= 40:
            return ReputationLevel.RELIABLE
        elif score >= 20:
            return ReputationLevel.NOVICE
        else:
            return ReputationLevel.UNTRUSTED

    def get_agent_stats(self, agent_id: str) -> Dict[str, any]:
        """获取Agent统计信息"""
        metrics = self._agent_metrics.get(agent_id, [])
        feedbacks = self._agent_feedbacks.get(agent_id, [])

        if not metrics:
            return {
                "agent_id": agent_id,
                "total_tasks": 0,
                "success_rate": 0.0,
                "avg_response_time": 0.0,
                "total_feedbacks": 0,
                "avg_feedback": 0.0
            }

        total_tasks = len(metrics)
        successful = sum(1 for m in metrics if m.success)
        avg_time = sum(m.response_time_ms for m in metrics) / total_tasks

        return {
            "agent_id": agent_id,
            "total_tasks": total_tasks,
            "success_rate": successful / total_tasks,
            "avg_response_time": round(avg_time, 2),
            "total_feedbacks": len(feedbacks),
            "avg_feedback": round(sum(feedbacks) / len(feedbacks), 2) if feedbacks else 0.0
        }

    def compare_agents(self, agent_ids: List[str]) -> List[ReputationScore]:
        """比较多个Agent的信誉分"""
        scores = [self.calculate_reputation(aid) for aid in agent_ids]
        return sorted(scores, key=lambda x: x.overall_score, reverse=True)
