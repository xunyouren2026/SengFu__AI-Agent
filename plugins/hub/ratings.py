"""
评分系统模块

提供用户评分、评论管理、信誉算法和排序算法功能。
"""

import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from collections import defaultdict


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass
class UserRating:
    """用户评分"""
    user_id: str
    plugin_id: str
    rating: int  # 1-5
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if not 1 <= self.rating <= 5:
            raise ValueError("Rating must be between 1 and 5")
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserRating":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Review:
    """用户评论"""
    review_id: str
    user_id: str
    plugin_id: str
    rating: int
    title: str = ""
    content: str = ""
    timestamp: float = field(default_factory=time.time)
    edited_at: Optional[float] = None
    helpful_count: int = 0
    unhelpful_count: int = 0
    verified_purchase: bool = False
    version_used: str = ""
    reply_to: Optional[str] = None
    replies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Review":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @property
    def helpfulness_score(self) -> float:
        """计算有用性分数"""
        total = self.helpful_count + self.unhelpful_count
        if total == 0:
            return 0.5
        return self.helpful_count / total


@dataclass
class RatingSummary:
    """评分摘要"""
    plugin_id: str
    average_rating: float = 0.0
    total_ratings: int = 0
    distribution: Dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(1, 6)})
    weighted_average: float = 0.0
    bayesian_average: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RatingSummary":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class UserReputation:
    """用户信誉"""
    user_id: str
    score: float = 1.0
    review_count: int = 0
    helpful_votes_received: int = 0
    total_votes_received: int = 0
    registration_date: float = field(default_factory=time.time)
    verified_purchases: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserReputation":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @property
    def helpfulness_ratio(self) -> float:
        """有用性比例"""
        if self.total_votes_received == 0:
            return 0.5
        return self.helpful_votes_received / self.total_votes_received


# ---------------------------------------------------------------------------
# 信誉算法
# ---------------------------------------------------------------------------

class ReputationAlgorithm:
    """信誉算法
    
    计算用户信誉分数，基于评论历史、有用性投票等因素。
    """
    
    def __init__(self):
        self._global_average_helpfulness = 0.5
        self._min_reviews_for_reputation = 3
    
    def calculate_user_reputation(self, user: UserReputation) -> float:
        """计算用户信誉分数
        
        使用威尔逊区间下限计算有用性置信度，结合其他因素。
        
        Args:
            user: 用户信誉数据
            
        Returns:
            信誉分数 (0-1)
        """
        if user.total_votes_received < self._min_reviews_for_reputation:
            # 新用户，给予中等信誉
            return 0.5 + 0.1 * min(user.review_count / 5, 1)
        
        # 威尔逊区间下限
        p = user.helpfulness_ratio
        n = user.total_votes_received
        z = 1.96  # 95%置信区间
        
        wilson_lower = (p + z*z/(2*n) - z*math.sqrt((p*(1-p)+z*z/(4*n))/n)) / (1+z*z/n)
        
        # 考虑账户年龄
        account_age_days = (time.time() - user.registration_date) / 86400
        age_factor = min(account_age_days / 30, 1.0)  # 30天后达到最大
        
        # 考虑验证购买
        verified_factor = min(user.verified_purchases / 5, 1.0)
        
        # 综合计算
        reputation = wilson_lower * 0.6 + age_factor * 0.2 + verified_factor * 0.2
        
        return max(0.0, min(1.0, reputation))
    
    def calculate_review_weight(self, review: Review, user_rep: UserReputation) -> float:
        """计算评论权重
        
        Args:
            review: 评论
            user_rep: 用户信誉
            
        Returns:
            权重值
        """
        base_weight = 1.0
        
        # 用户信誉影响
        base_weight *= (0.5 + 0.5 * user_rep.score)
        
        # 验证购买增加权重
        if review.verified_purchase:
            base_weight *= 1.5
        
        # 评论长度影响（较长评论通常更有价值）
        content_length = len(review.content)
        if content_length > 200:
            base_weight *= 1.2
        elif content_length < 50:
            base_weight *= 0.8
        
        # 时效性影响（较新评论权重更高）
        age_days = (time.time() - review.timestamp) / 86400
        if age_days < 30:
            base_weight *= 1.0
        elif age_days < 90:
            base_weight *= 0.9
        elif age_days < 365:
            base_weight *= 0.8
        else:
            base_weight *= 0.6
        
        return base_weight
    
    def detect_suspicious_activity(self, user: UserReputation,
                                    recent_reviews: List[Review]) -> List[str]:
        """检测可疑活动
        
        Args:
            user: 用户信誉
            recent_reviews: 最近评论
            
        Returns:
            可疑活动列表
        """
        flags = []
        
        # 短时间内大量评论
        if len(recent_reviews) > 10:
            time_span = recent_reviews[-1].timestamp - recent_reviews[0].timestamp
            if time_span < 3600:  # 1小时内
                flags.append("Rapid review posting")
        
        # 所有评分相同
        if recent_reviews:
            ratings = [r.rating for r in recent_reviews]
            if len(set(ratings)) == 1:
                flags.append("Uniform ratings")
        
        # 评论内容相似度高
        if len(recent_reviews) >= 3:
            contents = [r.content.lower() for r in recent_reviews]
            similarities = []
            for i in range(len(contents)):
                for j in range(i + 1, len(contents)):
                    sim = self._calculate_similarity(contents[i], contents[j])
                    similarities.append(sim)
            if similarities and sum(similarities) / len(similarities) > 0.8:
                flags.append("Similar review content")
        
        return flags
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（简化版Jaccard）"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# 排序算法
# ---------------------------------------------------------------------------

class RankingAlgorithm:
    """排序算法
    
    提供多种插件排序算法。
    """
    
    def __init__(self):
        self._global_average_rating = 3.0
        self._global_rating_count = 100
    
    def bayesian_average(self, avg_rating: float, rating_count: int,
                         global_avg: Optional[float] = None) -> float:
        """计算贝叶斯平均
        
        对小样本评分进行平滑处理。
        
        Args:
            avg_rating: 平均评分
            rating_count: 评分数量
            global_avg: 全局平均评分
            
        Returns:
            贝叶斯平均评分
        """
        global_avg = global_avg or self._global_average_rating
        m = self._global_rating_count
        
        return (m * global_avg + rating_count * avg_rating) / (m + rating_count)
    
    def wilson_score(self, positive: int, total: int, confidence: float = 0.95) -> float:
        """计算威尔逊分数
        
        用于排序，考虑置信区间。
        
        Args:
            positive: 正面评价数
            total: 总评价数
            confidence: 置信水平
            
        Returns:
            威尔逊分数
        """
        if total == 0:
            return 0.0
        
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
        phat = positive / total
        
        denominator = 1 + z*z/total
        centre = phat + z*z/(2*total)
        width = z * math.sqrt((phat*(1-phat) + z*z/(4*total)) / total)
        
        return (centre - width) / denominator
    
    def hot_score(self, rating: float, rating_count: int,
                  timestamp: float) -> float:
        """计算热度分数（类似Reddit Hot算法）
        
        Args:
            rating: 平均评分
            rating_count: 评分数量
            timestamp: 时间戳
            
        Returns:
            热度分数
        """
        # 评分标准化到-1到1
        score = (rating - 3) / 2
        
        # 订单数量影响
        order = math.log10(max(abs(rating_count), 1))
        
        # 时间衰减
        seconds = timestamp - 1609459200  # 2021-01-01
        
        return order + score * rating_count / seconds * 45000
    
    def trending_score(self, recent_ratings: List[int],
                       time_window_hours: int = 24) -> float:
        """计算趋势分数
        
        Args:
            recent_ratings: 近期评分列表
            time_window_hours: 时间窗口（小时）
            
        Returns:
            趋势分数
        """
        if not recent_ratings:
            return 0.0
        
        avg = sum(recent_ratings) / len(recent_ratings)
        velocity = len(recent_ratings) / time_window_hours
        
        return avg * math.log1p(velocity)
    
    def personalized_score(self, plugin_ratings: Dict[str, Any],
                           user_preferences: Dict[str, Any]) -> float:
        """计算个性化推荐分数
        
        Args:
            plugin_ratings: 插件评分数据
            user_preferences: 用户偏好
            
        Returns:
            个性化分数
        """
        base_score = plugin_ratings.get('average_rating', 3.0)
        
        # 类别匹配
        category_match = 0
        if 'preferred_categories' in user_preferences:
            if plugin_ratings.get('category') in user_preferences['preferred_categories']:
                category_match = 1.0
        
        # 标签匹配
        tag_match = 0
        if 'preferred_tags' in user_preferences and 'tags' in plugin_ratings:
            common_tags = set(plugin_ratings['tags']) & set(user_preferences['preferred_tags'])
            tag_match = len(common_tags) / max(len(user_preferences['preferred_tags']), 1)
        
        # 作者信誉
        author_boost = 0
        if plugin_ratings.get('author_verified'):
            author_boost = 0.5
        
        return base_score * (1 + 0.3 * category_match + 0.2 * tag_match + 0.1 * author_boost)


# ---------------------------------------------------------------------------
# 评分系统
# ---------------------------------------------------------------------------

class RatingSystem:
    """评分系统
    
    管理用户评分、评论和信誉。
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Args:
            storage_path: 存储路径
        """
        self._storage_path = storage_path or os.path.join(
            os.path.expanduser("~"), ".clawhub", "ratings"
        )
        
        self._ratings: Dict[str, Dict[str, UserRating]] = defaultdict(dict)
        self._reviews: Dict[str, Review] = {}
        self._plugin_summaries: Dict[str, RatingSummary] = {}
        self._user_reputations: Dict[str, UserReputation] = {}
        
        self._reputation_algo = ReputationAlgorithm()
        self._ranking_algo = RankingAlgorithm()
        
        self._lock = threading.RLock()
        self._listeners: List[Callable[[str, Any], None]] = []
        
        os.makedirs(self._storage_path, exist_ok=True)
        self._load_from_disk()
    
    def add_listener(self, callback: Callable[[str, Any], None]) -> None:
        """添加事件监听器"""
        self._listeners.append(callback)
    
    def remove_listener(self, callback: Callable[[str, Any], None]) -> None:
        """移除事件监听器"""
        if callback in self._listeners:
            self._listeners.remove(callback)
    
    def _notify(self, event: str, data: Any) -> None:
        """通知监听器"""
        for listener in self._listeners:
            try:
                listener(event, data)
            except Exception:
                pass
    
    def submit_rating(self, user_id: str, plugin_id: str,
                      rating: int) -> bool:
        """提交评分
        
        Args:
            user_id: 用户ID
            plugin_id: 插件ID
            rating: 评分 (1-5)
            
        Returns:
            是否成功
        """
        if not 1 <= rating <= 5:
            return False
        
        with self._lock:
            user_rating = UserRating(
                user_id=user_id,
                plugin_id=plugin_id,
                rating=rating,
            )
            
            # 存储评分
            self._ratings[plugin_id][user_id] = user_rating
            
            # 更新摘要
            self._update_summary(plugin_id)
            
            # 保存
            self._save_to_disk()
            
            # 通知
            self._notify('rating_submitted', user_rating)
            
            return True
    
    def submit_review(self, review: Review) -> bool:
        """提交评论
        
        Args:
            review: 评论对象
            
        Returns:
            是否成功
        """
        with self._lock:
            # 存储评论
            self._reviews[review.review_id] = review
            
            # 同时更新评分
            if review.user_id not in self._ratings[review.plugin_id]:
                self.submit_rating(review.user_id, review.plugin_id, review.rating)
            
            # 更新用户评论计数
            if review.user_id in self._user_reputations:
                self._user_reputations[review.user_id].review_count += 1
            else:
                self._user_reputations[review.user_id] = UserReputation(
                    user_id=review.user_id,
                    review_count=1,
                )
            
            # 保存
            self._save_to_disk()
            
            # 通知
            self._notify('review_submitted', review)
            
            return True
    
    def vote_review(self, review_id: str, user_id: str,
                    helpful: bool) -> bool:
        """投票评论有用性
        
        Args:
            review_id: 评论ID
            user_id: 投票用户ID
            helpful: 是否有用
            
        Returns:
            是否成功
        """
        with self._lock:
            if review_id not in self._reviews:
                return False
            
            review = self._reviews[review_id]
            
            if helpful:
                review.helpful_count += 1
            else:
                review.unhelpful_count += 1
            
            # 更新评论作者的信誉
            author_id = review.user_id
            if author_id in self._user_reputations:
                if helpful:
                    self._user_reputations[author_id].helpful_votes_received += 1
                self._user_reputations[author_id].total_votes_received += 1
            
            self._save_to_disk()
            
            return True
    
    def get_plugin_rating(self, plugin_id: str) -> RatingSummary:
        """获取插件评分摘要"""
        with self._lock:
            if plugin_id not in self._plugin_summaries:
                self._update_summary(plugin_id)
            
            return self._plugin_summaries.get(plugin_id, RatingSummary(plugin_id=plugin_id))
    
    def get_plugin_reviews(self, plugin_id: str,
                           sort_by: str = "helpful",
                           page: int = 1,
                           page_size: int = 10) -> Tuple[List[Review], int]:
        """获取插件评论
        
        Args:
            plugin_id: 插件ID
            sort_by: 排序方式 (helpful, newest, rating)
            page: 页码
            page_size: 每页数量
            
        Returns:
            (评论列表, 总数)
        """
        with self._lock:
            reviews = [
                r for r in self._reviews.values()
                if r.plugin_id == plugin_id
            ]
            
            # 排序
            if sort_by == "helpful":
                reviews.sort(key=lambda r: r.helpfulness_score, reverse=True)
            elif sort_by == "newest":
                reviews.sort(key=lambda r: r.timestamp, reverse=True)
            elif sort_by == "rating":
                reviews.sort(key=lambda r: r.rating, reverse=True)
            
            total = len(reviews)
            start = (page - 1) * page_size
            end = start + page_size
            
            return reviews[start:end], total
    
    def get_user_reviews(self, user_id: str) -> List[Review]:
        """获取用户评论"""
        with self._lock:
            return [
                r for r in self._reviews.values()
                if r.user_id == user_id
            ]
    
    def get_user_reputation(self, user_id: str) -> UserReputation:
        """获取用户信誉"""
        with self._lock:
            if user_id not in self._user_reputations:
                self._user_reputations[user_id] = UserReputation(user_id=user_id)
            
            user_rep = self._user_reputations[user_id]
            user_rep.score = self._reputation_algo.calculate_user_reputation(user_rep)
            
            return user_rep
    
    def _update_summary(self, plugin_id: str) -> None:
        """更新插件评分摘要"""
        ratings = list(self._ratings[plugin_id].values())
        
        if not ratings:
            self._plugin_summaries[plugin_id] = RatingSummary(plugin_id=plugin_id)
            return
        
        total = len(ratings)
        avg = sum(r.rating for r in ratings) / total
        
        distribution = {i: 0 for i in range(1, 6)}
        for r in ratings:
            distribution[r.rating] += 1
        
        # 贝叶斯平均
        bayesian = self._ranking_algo.bayesian_average(avg, total)
        
        self._plugin_summaries[plugin_id] = RatingSummary(
            plugin_id=plugin_id,
            average_rating=round(avg, 2),
            total_ratings=total,
            distribution=distribution,
            weighted_average=round(avg, 2),  # 简化
            bayesian_average=round(bayesian, 2),
        )
    
    def get_top_rated(self, limit: int = 10,
                      min_ratings: int = 5) -> List[Tuple[str, RatingSummary]]:
        """获取最高评分插件
        
        Args:
            limit: 返回数量
            min_ratings: 最少评分数量
            
        Returns:
            (插件ID, 评分摘要)列表
        """
        with self._lock:
            results = []
            for plugin_id, summary in self._plugin_summaries.items():
                if summary.total_ratings >= min_ratings:
                    results.append((plugin_id, summary))
            
            # 使用贝叶斯平均排序
            results.sort(key=lambda x: x[1].bayesian_average, reverse=True)
            
            return results[:limit]
    
    def get_trending(self, limit: int = 10,
                     hours: int = 24) -> List[Tuple[str, float]]:
        """获取趋势插件
        
        Args:
            limit: 返回数量
            hours: 时间窗口
            
        Returns:
            (插件ID, 趋势分数)列表
        """
        with self._lock:
            cutoff = time.time() - hours * 3600
            
            # 收集近期评分
            recent_ratings: Dict[str, List[int]] = defaultdict(list)
            for plugin_id, ratings in self._ratings.items():
                for rating in ratings.values():
                    if rating.timestamp > cutoff:
                        recent_ratings[plugin_id].append(rating.rating)
            
            # 计算趋势分数
            results = []
            for plugin_id, ratings in recent_ratings.items():
                score = self._ranking_algo.trending_score(ratings, hours)
                results.append((plugin_id, score))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:limit]
    
    def delete_review(self, review_id: str, moderator_id: Optional[str] = None) -> bool:
        """删除评论
        
        Args:
            review_id: 评论ID
            moderator_id: 审核员ID（可选）
            
        Returns:
            是否成功
        """
        with self._lock:
            if review_id not in self._reviews:
                return False
            
            review = self._reviews[review_id]
            del self._reviews[review_id]
            
            # 更新摘要
            self._update_summary(review.plugin_id)
            
            self._save_to_disk()
            
            self._notify('review_deleted', {
                'review_id': review_id,
                'moderator_id': moderator_id,
            })
            
            return True
    
    def report_review(self, review_id: str, reporter_id: str,
                      reason: str) -> bool:
        """举报评论
        
        Args:
            review_id: 评论ID
            reporter_id: 举报者ID
            reason: 举报原因
            
        Returns:
            是否成功
        """
        with self._lock:
            if review_id not in self._reviews:
                return False
            
            self._notify('review_reported', {
                'review_id': review_id,
                'reporter_id': reporter_id,
                'reason': reason,
            })
            
            return True
    
    def _save_to_disk(self) -> None:
        """保存到磁盘"""
        try:
            data = {
                'ratings': {
                    pid: {uid: r.to_dict() for uid, r in ratings.items()}
                    for pid, ratings in self._ratings.items()
                },
                'reviews': {rid: r.to_dict() for rid, r in self._reviews.items()},
                'summaries': {pid: s.to_dict() for pid, s in self._plugin_summaries.items()},
                'reputations': {uid: r.to_dict() for uid, r in self._user_reputations.items()},
            }
            
            file_path = os.path.join(self._storage_path, 'ratings.json')
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
    
    def _load_from_disk(self) -> None:
        """从磁盘加载"""
        file_path = os.path.join(self._storage_path, 'ratings.json')
        if not os.path.exists(file_path):
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 加载评分
            for pid, ratings in data.get('ratings', {}).items():
                for uid, r in ratings.items():
                    self._ratings[pid][uid] = UserRating.from_dict(r)
            
            # 加载评论
            for rid, r in data.get('reviews', {}).items():
                self._reviews[rid] = Review.from_dict(r)
            
            # 加载摘要
            for pid, s in data.get('summaries', {}).items():
                self._plugin_summaries[pid] = RatingSummary.from_dict(s)
            
            # 加载信誉
            for uid, r in data.get('reputations', {}).items():
                self._user_reputations[uid] = UserReputation.from_dict(r)
        except Exception:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            total_ratings = sum(len(r) for r in self._ratings.values())
            total_reviews = len(self._reviews)
            
            return {
                'total_ratings': total_ratings,
                'total_reviews': total_reviews,
                'rated_plugins': len(self._ratings),
                'active_reviewers': len(self._user_reputations),
            }
