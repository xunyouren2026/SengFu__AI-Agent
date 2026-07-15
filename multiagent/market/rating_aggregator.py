"""
市场评分汇总 - 跨任务聚合Agent信誉

实现多维度评分系统，包括时间加权、任务类型加权、贝叶斯平均等算法。
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Callable, Any, Tuple
from collections import defaultdict, deque
from statistics import mean, stdev


class RatingCategory(Enum):
    """评分类别"""
    OVERALL = auto()            # 总体评分
    QUALITY = auto()           # 质量评分
    COMMUNICATION = auto()     # 沟通评分
    TIMELINESS = auto()        # 时效评分
    VALUE = auto()             # 性价比评分
    PROFESSIONALISM = auto()   # 专业性评分


@dataclass
class Rating:
    """单条评分"""
    rating_id: str
    task_id: str
    agent_id: str
    reviewer_id: str
    task_type: str
    timestamp: float
    categories: Dict[RatingCategory, float]
    overall: float
    comment: str = ""
    verified: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentReputation:
    """Agent信誉"""
    agent_id: str
    ratings: List[Rating]
    average_ratings: Dict[RatingCategory, float]
    overall_average: float
    wilson_lower_bound: float
    total_reviews: int
    verified_reviews: int
    response_rate: float
    last_rated_at: Optional[float] = None
    trend: float = 0.0
    calculated_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "overall_average": self.overall_average,
            "wilson_lower_bound": self.wilson_lower_bound,
            "total_reviews": self.total_reviews,
            "verified_reviews": self.verified_reviews,
            "average_ratings": {k.name: v for k, v in self.average_ratings.items()},
            "response_rate": self.response_rate,
            "last_rated_at": self.last_rated_at,
            "trend": self.trend
        }


class RatingAggregator:
    """评分聚合器 - 核心评分处理逻辑"""
    
    def __init__(self):
        self._ratings: Dict[str, Rating] = {}
        self._agent_ratings: Dict[str, List[str]] = defaultdict(list)
        self._task_ratings: Dict[str, str] = {}
        self._user_ratings: Dict[str, List[str]] = defaultdict(list)
        self._reputations: Dict[str, AgentReputation] = {}
        self._rating_callbacks: List[Callable[[Rating, AgentReputation], None]] = []
        
        self._decay_factor = 0.95
        self._time_window_days = 90
        self._min_reviews_for_trust = 5
    
    def submit_rating(
        self,
        task_id: str,
        agent_id: str,
        reviewer_id: str,
        task_type: str,
        categories: Dict[RatingCategory, float],
        overall: float,
        comment: str = "",
        verified: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Rating, AgentReputation]:
        """提交评分"""
        rating_id = f"rating_{task_id}_{reviewer_id}"
        
        if rating_id in self._ratings:
            raise ValueError(f"Rating for task {task_id} by {reviewer_id} already exists")
        
        for cat, score in categories.items():
            if not 0 <= score <= 5:
                raise ValueError(f"Rating score must be between 0 and 5, got {score}")
        
        if not 0 <= overall <= 5:
            raise ValueError(f"Overall rating must be between 0 and 5, got {overall}")
        
        rating = Rating(
            rating_id=rating_id,
            task_id=task_id,
            agent_id=agent_id,
            reviewer_id=reviewer_id,
            task_type=task_type,
            timestamp=time.time(),
            categories=categories,
            overall=overall,
            comment=comment,
            verified=verified,
            metadata=metadata or {}
        )
        
        self._ratings[rating_id] = rating
        self._agent_ratings[agent_id].append(rating_id)
        self._task_ratings[task_id] = rating_id
        self._user_ratings[reviewer_id].append(rating_id)
        
        reputation = self._calculate_reputation(agent_id)
        
        for callback in self._rating_callbacks:
            try:
                callback(rating, reputation)
            except Exception:
                pass
        
        return rating, reputation
    
    def _calculate_reputation(self, agent_id: str) -> AgentReputation:
        """计算Agent信誉"""
        rating_ids = self._agent_ratings.get(agent_id, [])
        
        if not rating_ids:
            return AgentReputation(
                agent_id=agent_id,
                ratings=[],
                average_ratings={},
                overall_average=0.0,
                wilson_lower_bound=0.0,
                total_reviews=0,
                verified_reviews=0,
                response_rate=0.0
            )
        
        ratings = [self._ratings[rid] for rid in rating_ids if rid in self._ratings]
        
        time_window = self._time_window_days * 24 * 3600
        cutoff = time.time() - time_window
        recent_ratings = [r for r in ratings if r.timestamp > cutoff]
        
        category_averages: Dict[RatingCategory, float] = {}
        for cat in RatingCategory:
            scores = [r.categories.get(cat, r.overall) for r in recent_ratings]
            if scores:
                category_averages[cat] = self._time_weighted_average(
                    [(r.timestamp, r.categories.get(cat, r.overall)) for r in recent_ratings]
                )
            else:
                category_averages[cat] = 0.0
        
        overall_scores = [(r.timestamp, r.overall) for r in recent_ratings]
        overall_average = self._time_weighted_average(overall_scores) if overall_scores else 0.0
        
        wilson = self._calculate_wilson_score(len(ratings), overall_average / 5.0)
        
        trend = self._calculate_trend(ratings)
        
        verified_count = sum(1 for r in ratings if r.verified)
        
        response_rate = 0.0
        if len(ratings) > 0:
            recent_count = len(recent_ratings)
            response_rate = recent_count / max(len(ratings), 1)
        
        last_rated = max([r.timestamp for r in ratings]) if ratings else None
        
        reputation = AgentReputation(
            agent_id=agent_id,
            ratings=ratings,
            average_ratings=category_averages,
            overall_average=overall_average,
            wilson_lower_bound=wilson,
            total_reviews=len(ratings),
            verified_reviews=verified_count,
            response_rate=response_rate,
            last_rated_at=last_rated,
            trend=trend,
            calculated_at=time.time()
        )
        
        self._reputations[agent_id] = reputation
        return reputation
    
    def _time_weighted_average(self, scores: List[Tuple[float, float]]) -> float:
        """计算时间加权平均"""
        if not scores:
            return 0.0
        
        now = time.time()
        total_weight = 0.0
        weighted_sum = 0.0
        
        for timestamp, score in scores:
            age_seconds = now - timestamp
            weight = math.pow(self._decay_factor, age_seconds / (24 * 3600))
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _calculate_wilson_score(self, n: int, p: float, z: float = 1.96) -> float:
        """计算威尔逊区间下界"""
        if n == 0:
            return 0.0
        
        denominator = 1 + z * z / n
        center = p + z * z / (2 * n)
        spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        
        lower_bound = (center - spread) / denominator
        
        return max(0, min(1, lower_bound)) * 5.0
    
    def _calculate_trend(self, ratings: List[Rating]) -> float:
        """计算评分趋势"""
        if len(ratings) < 3:
            return 0.0
        
        sorted_ratings = sorted(ratings, key=lambda r: r.timestamp)
        
        recent = sorted_ratings[-5:]
        previous = sorted_ratings[-10:-5] if len(sorted_ratings) >= 10 else sorted_ratings[:-5]
        
        if not previous:
            return 0.0
        
        recent_avg = mean([r.overall for r in recent])
        previous_avg = mean([r.overall for r in previous])
        
        trend = recent_avg - previous_avg
        
        return max(-2, min(2, trend))
    
    def get_reputation(self, agent_id: str) -> Optional[AgentReputation]:
        """获取Agent信誉"""
        reputation = self._reputations.get(agent_id)
        
        if not reputation:
            rating_ids = self._agent_ratings.get(agent_id, [])
            if rating_ids:
                reputation = self._calculate_reputation(agent_id)
        
        return reputation
    
    def get_agent_ratings(
        self,
        agent_id: str,
        task_type: Optional[str] = None,
        min_rating: Optional[float] = None,
        max_results: int = 100
    ) -> List[Rating]:
        """获取Agent的评分列表"""
        rating_ids = self._agent_ratings.get(agent_id, [])
        ratings = [self._ratings[rid] for rid in rating_ids if rid in self._ratings]
        
        if task_type:
            ratings = [r for r in ratings if r.task_type == task_type]
        
        if min_rating is not None:
            ratings = [r for r in ratings if r.overall >= min_rating]
        
        ratings.sort(key=lambda r: r.timestamp, reverse=True)
        return ratings[:max_results]
    
    def compare_agents(self, agent_ids: List[str]) -> Dict[str, Any]:
        """比较多个Agent"""
        reputations = []
        
        for agent_id in agent_ids:
            rep = self.get_reputation(agent_id)
            if rep:
                reputations.append(rep)
        
        if not reputations:
            return {}
        
        return {
            "agents": [rep.to_dict() for rep in reputations],
            "rankings": {
                "by_overall": sorted(reputations, key=lambda r: r.overall_average, reverse=True),
                "by_wilson": sorted(reputations, key=lambda r: r.wilson_lower_bound, reverse=True),
                "by_trend": sorted(reputations, key=lambda r: r.trend, reverse=True),
                "by_volume": sorted(reputations, key=lambda r: r.total_reviews, reverse=True)
            },
            "recommendation": self._generate_comparison_recommendation(reputations)
        }
    
    def _generate_comparison_recommendation(self, reputations: List[AgentReputation]) -> str:
        """生成比较建议"""
        if not reputations:
            return "No agents to compare"
        
        best_overall = max(reputations, key=lambda r: r.overall_average)
        best_wilson = max(reputations, key=lambda r: r.wilson_lower_bound)
        best_trend = max(reputations, key=lambda r: r.trend)
        most_reviewed = max(reputations, key=lambda r: r.total_reviews)
        
        parts = []
        if best_overall.agent_id == best_wilson.agent_id:
            parts.append(f"{best_overall.agent_id} has the highest overall rating and trust score.")
        else:
            parts.append(f"{best_overall.agent_id} has the highest rating ({best_overall.overall_average:.2f}).")
            parts.append(f"{best_wilson.agent_id} has the highest trust score ({best_wilson.wilson_lower_bound:.2f}).")
        
        if best_trend.trend > 0.5:
            parts.append(f"{best_trend.agent_id} shows the most improvement trend (+{best_trend.trend:.2f}).")
        
        if most_reviewed.total_reviews > 10:
            parts.append(f"{most_reviewed.agent_id} has the most reviews ({most_reviewed.total_reviews}), suggesting consistent performance.")
        
        return " ".join(parts)
    
    def get_trusted_agents(self, min_reviews: int = 10, min_rating: float = 4.0, limit: int = 20) -> List[AgentReputation]:
        """获取可信Agent列表"""
        trusted = []
        
        for agent_id in self._agent_ratings.keys():
            rep = self.get_reputation(agent_id)
            if rep and rep.total_reviews >= min_reviews and rep.overall_average >= min_rating:
                trusted.append(rep)
        
        trusted.sort(key=lambda r: (r.wilson_lower_bound, r.total_reviews), reverse=True)
        return trusted[:limit]
    
    def add_rating_callback(self, callback: Callable[[Rating, AgentReputation], None]) -> None:
        """添加评分回调"""
        self._rating_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_ratings = len(self._ratings)
        total_agents = len(self._agent_ratings)
        total_reviewers = len(self._user_ratings)
        
        if not self._ratings:
            return {
                "total_ratings": 0,
                "total_agents": 0,
                "total_reviewers": 0,
                "average_rating": 0.0,
                "verified_percentage": 0.0
            }
        
        all_ratings = list(self._ratings.values())
        avg_rating = mean([r.overall for r in all_ratings])
        verified_pct = sum(1 for r in all_ratings if r.verified) / len(all_ratings) * 100
        
        category_avgs = {}
        for cat in RatingCategory:
            scores = [r.categories.get(cat, 0) for r in all_ratings if cat in r.categories]
            if scores:
                category_avgs[cat.name] = mean(scores)
        
        return {
            "total_ratings": total_ratings,
            "total_agents": total_agents,
            "total_reviewers": total_reviewers,
            "average_rating": avg_rating,
            "verified_percentage": verified_pct,
            "category_averages": category_avgs,
            "rating_distribution": self._get_rating_distribution()
        }
    
    def _get_rating_distribution(self) -> Dict[int, int]:
        """获取评分分布"""
        distribution = defaultdict(int)
        for rating in self._ratings.values():
            rounded = int(round(rating.overall))
            distribution[rounded] += 1
        return dict(distribution)
