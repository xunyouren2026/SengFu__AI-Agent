"""
Feedback Learner - 反馈学习器
从用户反馈中学习，持续优化系统表现
"""

import re
import time
import logging
import threading
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """反馈类型"""
    POSITIVE = "positive"       # 正面反馈
    NEGATIVE = "negative"       # 负面反馈
    NEUTRAL = "neutral"         # 中性反馈
    CORRECTION = "correction"   # 纠正反馈
    PREFERENCE = "preference"   # 偏好反馈
    RATING = "rating"           # 评分反馈


class LearningMode(Enum):
    """学习模式"""
    ONLINE = "online"           # 在线学习
    BATCH = "batch"             # 批量学习
    REINFORCEMENT = "reinforcement"  # 强化学习
    HYBRID = "hybrid"           # 混合模式


@dataclass
class Feedback:
    """反馈数据"""
    id: str
    feedback_type: FeedbackType
    content: str
    score: float  # -1.0 到 1.0
    
    # 关联信息
    conversation_id: str = ""
    message_id: str = ""
    user_id: str = ""
    
    # 详细信息
    aspects: Dict[str, float] = field(default_factory=dict)  # 各方面评分
    tags: Set[str] = field(default_factory=set)
    correction: Optional[str] = None  # 纠正内容
    
    # 元数据
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "feedback_type": self.feedback_type.value,
            "score": self.score,
            "aspects": self.aspects,
            "tags": list(self.tags),
            "timestamp": self.timestamp
        }


@dataclass
class LearningRule:
    """学习规则"""
    id: str
    condition: Dict[str, Any]  # 触发条件
    action: Dict[str, Any]     # 执行动作
    confidence: float = 0.5
    success_count: int = 0
    failure_count: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5
    
    def update(self, success: bool):
        """更新规则统计"""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.updated_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "condition": self.condition,
            "action": self.action,
            "confidence": self.confidence,
            "success_rate": self.success_rate,
            "success_count": self.success_count,
            "failure_count": self.failure_count
        }


@dataclass
class FeedbackConfig:
    """反馈学习配置"""
    # 存储配置
    max_feedbacks: int = 10000
    max_rules: int = 1000
    
    # 学习参数
    learning_rate: float = 0.1
    min_samples_for_rule: int = 5
    rule_confidence_threshold: float = 0.7
    
    # 模式配置
    learning_mode: LearningMode = LearningMode.HYBRID
    
    # 权重配置
    recent_feedback_weight: float = 0.3
    user_preference_weight: float = 0.2
    
    # 遗忘配置
    enable_forgetting: bool = True
    feedback_decay_hours: float = 720.0  # 30天


class FeedbackCollector:
    """反馈收集器"""
    
    def __init__(self):
        self._feedback_patterns = self._init_patterns()
    
    def _init_patterns(self) -> Dict[FeedbackType, List[str]]:
        """初始化反馈模式"""
        return {
            FeedbackType.POSITIVE: [
                "很好", "不错", "太棒了", "完美", "优秀", "谢谢",
                "good", "great", "excellent", "perfect", "thanks", "helpful"
            ],
            FeedbackType.NEGATIVE: [
                "不好", "错误", "不对", "很差", "糟糕", "不满意",
                "bad", "wrong", "incorrect", "poor", "terrible", "unhelpful"
            ],
            FeedbackType.CORRECTION: [
                "应该是", "实际上是", "正确的是", "我要纠正",
                "actually", "should be", "the correct answer is", "let me correct"
            ],
        }
    
    def collect(
        self,
        content: str,
        score: Optional[float] = None,
        aspects: Optional[Dict[str, float]] = None
    ) -> Feedback:
        """收集反馈"""
        # 自动检测反馈类型
        feedback_type = self._detect_type(content)
        
        # 如果没有提供分数，从内容推断
        if score is None:
            score = self._infer_score(content, feedback_type)
        
        return Feedback(
            id=self._generate_id(),
            feedback_type=feedback_type,
            content=content,
            score=score,
            aspects=aspects or {}
        )
    
    def _detect_type(self, content: str) -> FeedbackType:
        """检测反馈类型"""
        content_lower = content.lower()
        
        for feedback_type, patterns in self._feedback_patterns.items():
            for pattern in patterns:
                if pattern.lower() in content_lower:
                    return feedback_type
        
        return FeedbackType.NEUTRAL
    
    def _infer_score(self, content: str, feedback_type: FeedbackType) -> float:
        """推断分数"""
        if feedback_type == FeedbackType.POSITIVE:
            return 0.8
        elif feedback_type == FeedbackType.NEGATIVE:
            return -0.5
        elif feedback_type == FeedbackType.CORRECTION:
            return 0.0
        else:
            return 0.0
    
    def _generate_id(self) -> str:
        """生成反馈ID"""
        return f"fb_{int(time.time() * 1000)}"


class PatternMiner:
    """模式挖掘器"""
    
    def __init__(self, min_samples: int = 5):
        self.min_samples = min_samples
    
    def mine_patterns(
        self,
        feedbacks: List[Feedback]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        挖掘反馈模式
        
        Returns:
            [(pattern, avg_score), ...]
        """
        if len(feedbacks) < self.min_samples:
            return []
        
        patterns = []
        
        # 按标签分组
        tag_groups = defaultdict(list)
        for fb in feedbacks:
            for tag in fb.tags:
                tag_groups[tag].append(fb)
        
        for tag, group in tag_groups.items():
            if len(group) >= self.min_samples:
                avg_score = sum(fb.score for fb in group) / len(group)
                patterns.append(({"tag": tag}, avg_score))
        
        # 按反馈类型分组
        type_groups = defaultdict(list)
        for fb in feedbacks:
            type_groups[fb.feedback_type].append(fb)
        
        for fb_type, group in type_groups.items():
            if len(group) >= self.min_samples:
                avg_score = sum(fb.score for fb in group) / len(group)
                patterns.append(({"feedback_type": fb_type.value}, avg_score))
        
        # 按方面评分分组
        aspect_patterns = self._mine_aspect_patterns(feedbacks)
        patterns.extend(aspect_patterns)
        
        return patterns
    
    def _mine_aspect_patterns(
        self,
        feedbacks: List[Feedback]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """挖掘方面模式"""
        patterns = []
        
        # 收集所有方面
        all_aspects = set()
        for fb in feedbacks:
            all_aspects.update(fb.aspects.keys())
        
        for aspect in all_aspects:
            # 按方面评分高低分组
            high_group = [fb for fb in feedbacks 
                         if aspect in fb.aspects and fb.aspects[aspect] >= 0.7]
            low_group = [fb for fb in feedbacks 
                        if aspect in fb.aspects and fb.aspects[aspect] <= 0.3]
            
            if len(high_group) >= self.min_samples:
                avg_score = sum(fb.score for fb in high_group) / len(high_group)
                patterns.append(({"aspect": aspect, "level": "high"}, avg_score))
            
            if len(low_group) >= self.min_samples:
                avg_score = sum(fb.score for fb in low_group) / len(low_group)
                patterns.append(({"aspect": aspect, "level": "low"}, avg_score))
        
        return patterns


class RuleGenerator:
    """规则生成器"""
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
    
    def generate_rules(
        self,
        patterns: List[Tuple[Dict[str, Any], float]]
    ) -> List[LearningRule]:
        """从模式生成规则"""
        rules = []
        
        for pattern, avg_score in patterns:
            # 只为显著的模式生成规则
            if abs(avg_score) < 0.3:
                continue
            
            # 确定动作
            if avg_score > 0.5:
                action = {"type": "reinforce", "strength": avg_score}
            elif avg_score < -0.3:
                action = {"type": "avoid", "strength": -avg_score}
            else:
                action = {"type": "neutral", "strength": 0}
            
            rule = LearningRule(
                id=self._generate_rule_id(),
                condition=pattern,
                action=action,
                confidence=abs(avg_score)
            )
            
            if rule.confidence >= self.confidence_threshold:
                rules.append(rule)
        
        return rules
    
    def _generate_rule_id(self) -> str:
        """生成规则ID"""
        return f"rule_{int(time.time() * 1000)}"


class PreferenceTracker:
    """偏好追踪器"""
    
    def __init__(self):
        self._user_preferences: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._global_preferences: Dict[str, float] = defaultdict(float)
    
    def update(
        self,
        user_id: str,
        preference_key: str,
        score: float,
        weight: float = 1.0
    ):
        """更新偏好"""
        # 更新用户偏好
        current = self._user_preferences[user_id][preference_key]
        self._user_preferences[user_id][preference_key] = current * 0.9 + score * weight * 0.1
        
        # 更新全局偏好
        global_current = self._global_preferences[preference_key]
        self._global_preferences[preference_key] = global_current * 0.99 + score * weight * 0.01
    
    def get_user_preference(
        self,
        user_id: str,
        preference_key: str,
        default: float = 0.5
    ) -> float:
        """获取用户偏好"""
        if user_id in self._user_preferences:
            return self._user_preferences[user_id].get(preference_key, default)
        return default
    
    def get_global_preference(self, preference_key: str, default: float = 0.5) -> float:
        """获取全局偏好"""
        return self._global_preferences.get(preference_key, default)
    
    def get_all_user_preferences(self, user_id: str) -> Dict[str, float]:
        """获取用户所有偏好"""
        return dict(self._user_preferences.get(user_id, {}))


class FeedbackLearner:
    """反馈学习器主类"""
    
    def __init__(
        self,
        config: Optional[FeedbackConfig] = None,
        llm_client: Optional[Any] = None
    ):
        self.config = config or FeedbackConfig()
        self.llm_client = llm_client
        
        # 组件
        self.collector = FeedbackCollector()
        self.pattern_miner = PatternMiner(self.config.min_samples_for_rule)
        self.rule_generator = RuleGenerator(self.config.rule_confidence_threshold)
        self.preference_tracker = PreferenceTracker()
        
        # 存储
        self._feedbacks: List[Feedback] = []
        self._rules: Dict[str, LearningRule] = {}
        self._feedback_id_counter = 0
        
        self._lock = threading.Lock()
    
    def submit_feedback(
        self,
        content: str,
        score: Optional[float] = None,
        aspects: Optional[Dict[str, float]] = None,
        tags: Optional[Set[str]] = None,
        user_id: str = "",
        conversation_id: str = "",
        message_id: str = ""
    ) -> Feedback:
        """提交反馈"""
        feedback = self.collector.collect(content, score, aspects)
        feedback.user_id = user_id
        feedback.conversation_id = conversation_id
        feedback.message_id = message_id
        
        if tags:
            feedback.tags = tags
        
        with self._lock:
            self._feedbacks.append(feedback)
            
            # 限制存储量
            if len(self._feedbacks) > self.config.max_feedbacks:
                self._feedbacks = self._feedbacks[-self.config.max_feedbacks:]
        
        # 更新偏好
        self._update_preferences(feedback)
        
        # 在线学习模式：立即学习
        if self.config.learning_mode in [LearningMode.ONLINE, LearningMode.HYBRID]:
            self._learn_from_feedback(feedback)
        
        return feedback
    
    def submit_rating(
        self,
        rating: int,
        max_rating: int = 5,
        user_id: str = "",
        conversation_id: str = "",
        message_id: str = ""
    ) -> Feedback:
        """提交评分"""
        # 转换为-1到1的分数
        score = (rating / max_rating) * 2 - 1
        
        feedback = Feedback(
            id=f"fb_{int(time.time() * 1000)}",
            feedback_type=FeedbackType.RATING,
            content=f"Rating: {rating}/{max_rating}",
            score=score,
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=message_id
        )
        
        with self._lock:
            self._feedbacks.append(feedback)
        
        return feedback
    
    def submit_correction(
        self,
        original: str,
        correction: str,
        user_id: str = "",
        conversation_id: str = "",
        message_id: str = ""
    ) -> Feedback:
        """提交纠正"""
        feedback = Feedback(
            id=f"fb_{int(time.time() * 1000)}",
            feedback_type=FeedbackType.CORRECTION,
            content=f"Original: {original}\nCorrection: {correction}",
            score=0.0,
            correction=correction,
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=message_id
        )
        
        with self._lock:
            self._feedbacks.append(feedback)
        
        return feedback
    
    def _update_preferences(self, feedback: Feedback):
        """更新偏好"""
        if not feedback.user_id:
            return
        
        # 从标签更新偏好
        for tag in feedback.tags:
            self.preference_tracker.update(
                feedback.user_id,
                f"tag_{tag}",
                feedback.score
            )
        
        # 从方面更新偏好
        for aspect, score in feedback.aspects.items():
            self.preference_tracker.update(
                feedback.user_id,
                f"aspect_{aspect}",
                score
            )
    
    def _learn_from_feedback(self, feedback: Feedback):
        """从单个反馈学习"""
        # 检查是否有匹配的规则
        for rule in self._rules.values():
            if self._matches_condition(feedback, rule.condition):
                # 更新规则
                success = (feedback.score > 0 and rule.action["type"] == "reinforce") or \
                         (feedback.score < 0 and rule.action["type"] == "avoid")
                rule.update(success)
    
    def _matches_condition(
        self,
        feedback: Feedback,
        condition: Dict[str, Any]
    ) -> bool:
        """检查反馈是否匹配条件"""
        for key, value in condition.items():
            if key == "tag" and value not in feedback.tags:
                return False
            elif key == "feedback_type" and feedback.feedback_type.value != value:
                return False
            elif key == "aspect" and value not in feedback.aspects:
                return False
        return True
    
    def batch_learn(self) -> Dict[str, Any]:
        """批量学习"""
        with self._lock:
            feedbacks = list(self._feedbacks)
        
        # 挖掘模式
        patterns = self.pattern_miner.mine_patterns(feedbacks)
        
        # 生成规则
        new_rules = self.rule_generator.generate_rules(patterns)
        
        # 添加规则
        with self._lock:
            for rule in new_rules:
                self._rules[rule.id] = rule
                
                # 限制规则数量
                if len(self._rules) > self.config.max_rules:
                    # 移除成功率最低的规则
                    sorted_rules = sorted(
                        self._rules.items(),
                        key=lambda x: x[1].success_rate
                    )
                    self._rules = dict(sorted_rules[-self.config.max_rules:])
        
        return {
            "patterns_found": len(patterns),
            "rules_generated": len(new_rules),
            "total_rules": len(self._rules)
        }
    
    def get_recommendations(
        self,
        context: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取推荐"""
        recommendations = []
        
        # 基于规则的推荐
        for rule in self._rules.values():
            if rule.confidence >= self.config.rule_confidence_threshold:
                recommendations.append({
                    "type": "rule_based",
                    "action": rule.action,
                    "confidence": rule.confidence,
                    "rule_id": rule.id
                })
        
        # 基于偏好的推荐
        if user_id:
            user_prefs = self.preference_tracker.get_all_user_preferences(user_id)
            for pref_key, pref_value in user_prefs.items():
                if abs(pref_value) > 0.5:
                    recommendations.append({
                        "type": "preference_based",
                        "preference": pref_key,
                        "value": pref_value
                    })
        
        # 按置信度排序
        recommendations.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return recommendations[:10]
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """获取反馈统计"""
        with self._lock:
            feedbacks = list(self._feedbacks)
        
        if not feedbacks:
            return {"total_feedbacks": 0}
        
        # 类型分布
        type_counts = defaultdict(int)
        for fb in feedbacks:
            type_counts[fb.feedback_type.value] += 1
        
        # 平均分数
        avg_score = sum(fb.score for fb in feedbacks) / len(feedbacks)
        
        # 评分趋势
        recent = feedbacks[-100:] if len(feedbacks) >= 100 else feedbacks
        recent_avg = sum(fb.score for fb in recent) / len(recent)
        
        return {
            "total_feedbacks": len(feedbacks),
            "type_distribution": dict(type_counts),
            "average_score": avg_score,
            "recent_average_score": recent_avg,
            "total_rules": len(self._rules),
            "unique_users": len(set(fb.user_id for fb in feedbacks if fb.user_id))
        }
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """获取用户统计"""
        with self._lock:
            user_feedbacks = [fb for fb in self._feedbacks if fb.user_id == user_id]
        
        if not user_feedbacks:
            return {"user_id": user_id, "feedback_count": 0}
        
        avg_score = sum(fb.score for fb in user_feedbacks) / len(user_feedbacks)
        preferences = self.preference_tracker.get_all_user_preferences(user_id)
        
        return {
            "user_id": user_id,
            "feedback_count": len(user_feedbacks),
            "average_score": avg_score,
            "preferences": preferences
        }
    
    def export_feedbacks(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """导出反馈数据"""
        with self._lock:
            feedbacks = list(self._feedbacks[-limit:])
        return [fb.to_dict() for fb in feedbacks]
    
    def export_rules(self) -> List[Dict[str, Any]]:
        """导出规则"""
        with self._lock:
            return [rule.to_dict() for rule in self._rules.values()]
    
    def clear_old_feedbacks(self, max_age_hours: float = 720):
        """清理旧反馈"""
        cutoff = time.time() - max_age_hours * 3600
        
        with self._lock:
            self._feedbacks = [fb for fb in self._feedbacks if fb.timestamp >= cutoff]


# 工厂函数
def create_feedback_learner(
    config: Optional[FeedbackConfig] = None,
    llm_client: Optional[Any] = None
) -> FeedbackLearner:
    """创建反馈学习器"""
    return FeedbackLearner(config, llm_client)
