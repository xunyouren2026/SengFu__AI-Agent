"""
评价收集模块

任务完成后收集仲裁者和其他Agent的评分
支持多维度评价、权重调整和争议处理
"""

from typing import Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import statistics


class ReviewerRole(Enum):
    """评价者角色"""
    ARBITER = "arbiter"           # 仲裁者
    PEER = "peer"                 # 同行Agent
    CLIENT = "client"             # 任务委托方
    OBSERVER = "observer"         # 观察者
    SYSTEM = "system"             # 系统自动评价


class ReviewDimension(Enum):
    """评价维度"""
    QUALITY = "quality"           # 任务质量
    TIMELINESS = "timeliness"     # 及时性
    COOPERATION = "cooperation"   # 合作度
    COMMUNICATION = "communication"  # 沟通能力
    TECHNICAL = "technical"       # 技术能力
    OVERALL = "overall"           # 总体评价


@dataclass
class Review:
    """评价记录"""
    review_id: str
    task_id: str
    reviewer_id: str
    reviewee_id: str
    role: ReviewerRole
    dimension: ReviewDimension
    score: float  # 0-5
    weight: float  # 评价权重
    comment: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    verified: bool = False  # 是否已验证


@dataclass
class ReviewSummary:
    """评价汇总"""
    reviewee_id: str
    task_id: str
    overall_score: float
    dimension_scores: Dict[ReviewDimension, float]
    review_count: int
    weighted_average: float
    confidence: float  # 置信度


class ReviewCollector:
    """
    评价收集器

    收集任务完成后的多维度评价，支持:
    - 多角色评价 (仲裁者、同行、委托方)
    - 多维度评分 (质量、及时性、合作度等)
    - 权重调整
    - 评价验证
    """

    # 默认角色权重
    DEFAULT_ROLE_WEIGHTS: Dict[ReviewerRole, float] = {
        ReviewerRole.ARBITER: 1.5,      # 仲裁者权重最高
        ReviewerRole.CLIENT: 1.2,       # 委托方次之
        ReviewerRole.PEER: 1.0,         # 同行标准权重
        ReviewerRole.OBSERVER: 0.7,     # 观察者较低
        ReviewerRole.SYSTEM: 0.8,       # 系统评价
    }

    # 维度权重
    DEFAULT_DIMENSION_WEIGHTS: Dict[ReviewDimension, float] = {
        ReviewDimension.OVERALL: 0.3,
        ReviewDimension.QUALITY: 0.25,
        ReviewDimension.TIMELINESS: 0.15,
        ReviewDimension.TECHNICAL: 0.15,
        ReviewDimension.COOPERATION: 0.1,
        ReviewDimension.COMMUNICATION: 0.05,
    }

    def __init__(
        self,
        role_weights: Optional[Dict[ReviewerRole, float]] = None,
        dimension_weights: Optional[Dict[ReviewDimension, float]] = None
    ):
        self.role_weights = role_weights or self.DEFAULT_ROLE_WEIGHTS.copy()
        self.dimension_weights = dimension_weights or self.DEFAULT_DIMENSION_WEIGHTS.copy()
        self._reviews: Dict[str, List[Review]] = {}  # task_id -> reviews
        self._agent_reviews: Dict[str, List[Review]] = {}  # agent_id -> reviews
        self._review_callbacks: List[Callable[[Review], None]] = []
        self._validators: List[Callable[[Review], bool]] = []

    def add_validator(self, validator: Callable[[Review], bool]) -> None:
        """添加评价验证器"""
        self._validators.append(validator)

    def add_callback(self, callback: Callable[[Review], None]) -> None:
        """添加评价提交回调"""
        self._review_callbacks.append(callback)

    def submit_review(
        self,
        review_id: str,
        task_id: str,
        reviewer_id: str,
        reviewee_id: str,
        role: ReviewerRole,
        dimension: ReviewDimension,
        score: float,
        comment: Optional[str] = None,
        custom_weight: Optional[float] = None
    ) -> Review:
        """
        提交评价

        Args:
            review_id: 评价唯一ID
            task_id: 任务ID
            reviewer_id: 评价者ID
            reviewee_id: 被评价者ID
            role: 评价者角色
            dimension: 评价维度
            score: 评分 (0-5)
            comment: 评价备注
            custom_weight: 自定义权重 (覆盖默认角色权重)

        Returns:
            Review对象
        """
        # 限制评分范围
        score = max(0.0, min(5.0, score))

        # 计算权重
        weight = custom_weight if custom_weight is not None else self.role_weights.get(role, 1.0)

        review = Review(
            review_id=review_id,
            task_id=task_id,
            reviewer_id=reviewer_id,
            reviewee_id=reviewee_id,
            role=role,
            dimension=dimension,
            score=score,
            weight=weight,
            comment=comment,
            timestamp=time.time(),
            verified=False
        )

        # 验证评价
        review.verified = self._validate_review(review)

        # 存储评价
        if task_id not in self._reviews:
            self._reviews[task_id] = []
        self._reviews[task_id].append(review)

        if reviewee_id not in self._agent_reviews:
            self._agent_reviews[reviewee_id] = []
        self._agent_reviews[reviewee_id].append(review)

        # 触发回调
        for callback in self._review_callbacks:
            callback(review)

        return review

    def _validate_review(self, review: Review) -> bool:
        """验证评价有效性"""
        # 基础验证
        if review.score < 0 or review.score > 5:
            return False

        # 运行自定义验证器
        for validator in self._validators:
            if not validator(review):
                return False

        return True

    def get_task_reviews(self, task_id: str) -> List[Review]:
        """获取任务的所有评价"""
        return self._reviews.get(task_id, [])

    def get_agent_reviews(
        self,
        agent_id: str,
        dimension: Optional[ReviewDimension] = None,
        role: Optional[ReviewerRole] = None
    ) -> List[Review]:
        """获取Agent的评价列表"""
        reviews = self._agent_reviews.get(agent_id, [])

        if dimension:
            reviews = [r for r in reviews if r.dimension == dimension]
        if role:
            reviews = [r for r in reviews if r.role == role]

        return reviews

    def calculate_task_summary(self, task_id: str) -> Optional[ReviewSummary]:
        """计算任务评价汇总"""
        reviews = self._reviews.get(task_id, [])
        if not reviews:
            return None

        # 按被评价者分组
        reviewee_ids = set(r.reviewee_id for r in reviews)

        summaries = []
        for reviewee_id in reviewee_ids:
            agent_reviews = [r for r in reviews if r.reviewee_id == reviewee_id]
            summary = self._calculate_summary(reviewee_id, task_id, agent_reviews)
            summaries.append(summary)

        # 返回第一个 (通常一个任务主要评价一个执行者)
        return summaries[0] if summaries else None

    def _calculate_summary(
        self,
        reviewee_id: str,
        task_id: str,
        reviews: List[Review]
    ) -> ReviewSummary:
        """计算单个Agent的评价汇总"""
        # 按维度分组计算
        dimension_scores: Dict[ReviewDimension, float] = {}

        for dimension in ReviewDimension:
            dim_reviews = [r for r in reviews if r.dimension == dimension]
            if dim_reviews:
                weighted_sum = sum(r.score * r.weight for r in dim_reviews)
                total_weight = sum(r.weight for r in dim_reviews)
                dimension_scores[dimension] = weighted_sum / total_weight if total_weight > 0 else 0

        # 计算总体加权分
        overall_score = 0.0
        total_dim_weight = 0.0

        for dim, score in dimension_scores.items():
            weight = self.dimension_weights.get(dim, 0.1)
            overall_score += score * weight
            total_dim_weight += weight

        if total_dim_weight > 0:
            overall_score /= total_dim_weight

        # 计算置信度 (基于评价数量)
        review_count = len(reviews)
        confidence = min(1.0, review_count / 10)  # 10条评价达到满置信度

        # 加权平均分 (所有评价的加权平均)
        all_weighted_sum = sum(r.score * r.weight for r in reviews)
        all_total_weight = sum(r.weight for r in reviews)
        weighted_average = all_weighted_sum / all_total_weight if all_total_weight > 0 else 0

        return ReviewSummary(
            reviewee_id=reviewee_id,
            task_id=task_id,
            overall_score=round(overall_score, 2),
            dimension_scores={k: round(v, 2) for k, v in dimension_scores.items()},
            review_count=review_count,
            weighted_average=round(weighted_average, 2),
            confidence=round(confidence, 2)
        )

    def get_agent_summary(self, agent_id: str) -> Dict:
        """获取Agent的综合评价摘要"""
        reviews = self._agent_reviews.get(agent_id, [])

        if not reviews:
            return {
                "agent_id": agent_id,
                "total_reviews": 0,
                "average_score": 0.0,
                "dimension_averages": {},
                "role_breakdown": {}
            }

        # 总体平均分
        all_scores = [r.score for r in reviews]
        average_score = statistics.mean(all_scores) if all_scores else 0

        # 各维度平均分
        dimension_averages = {}
        for dimension in ReviewDimension:
            dim_scores = [r.score for r in reviews if r.dimension == dimension]
            if dim_scores:
                dimension_averages[dimension.value] = round(statistics.mean(dim_scores), 2)

        # 角色分布
        role_breakdown = {}
        for role in ReviewerRole:
            role_reviews = [r for r in reviews if r.role == role]
            if role_reviews:
                role_scores = [r.score for r in role_reviews]
                role_breakdown[role.value] = {
                    "count": len(role_reviews),
                    "average": round(statistics.mean(role_scores), 2)
                }

        return {
            "agent_id": agent_id,
            "total_reviews": len(reviews),
            "average_score": round(average_score, 2),
            "dimension_averages": dimension_averages,
            "role_breakdown": role_breakdown
        }

    def detect_anomalies(self, task_id: str, threshold: float = 1.5) -> List[Review]:
        """
        检测异常评价

        识别偏离平均值过大的评价
        """
        reviews = self._reviews.get(task_id, [])
        if len(reviews) < 3:
            return []

        # 计算平均分
        scores = [r.score for r in reviews]
        mean = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 1.0

        if std == 0:
            return []

        # 找出偏离超过阈值的评价
        anomalies = []
        for review in reviews:
            z_score = abs(review.score - mean) / std
            if z_score > threshold:
                anomalies.append(review)

        return anomalies

    def request_peer_reviews(
        self,
        task_id: str,
        reviewee_id: str,
        peer_ids: List[str],
        dimensions: Optional[List[ReviewDimension]] = None
    ) -> List[str]:
        """
        请求同行评价

        Returns:
            请求ID列表
        """
        if dimensions is None:
            dimensions = [ReviewDimension.OVERALL, ReviewDimension.QUALITY]

        request_ids = []
        for peer_id in peer_ids:
            for dim in dimensions:
                req_id = f"req_{task_id}_{peer_id}_{dim.value}_{int(time.time())}"
                request_ids.append(req_id)
                # 这里可以触发实际的评价请求逻辑

        return request_ids

    def export_reviews(
        self,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None
    ) -> List[Dict]:
        """导出评价数据"""
        reviews = []

        if task_id:
            reviews = self._reviews.get(task_id, [])
        elif agent_id:
            reviews = self._agent_reviews.get(agent_id, [])
        else:
            for task_reviews in self._reviews.values():
                reviews.extend(task_reviews)

        return [
            {
                "review_id": r.review_id,
                "task_id": r.task_id,
                "reviewer_id": r.reviewer_id,
                "reviewee_id": r.reviewee_id,
                "role": r.role.value,
                "dimension": r.dimension.value,
                "score": r.score,
                "weight": r.weight,
                "comment": r.comment,
                "timestamp": r.timestamp,
                "verified": r.verified
            }
            for r in reviews
        ]
