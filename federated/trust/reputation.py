"""
联邦信誉系统
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import math


class ReputationMetric(Enum):
    """信誉度量"""
    CONTRIBUTION = "contribution"  # 贡献度
    ACCURACY = "accuracy"  # 准确率
    CONSISTENCY = "consistency"  # 一致性
    RELIABILITY = "reliability"  # 可靠性


class ReputationScore:
    """信誉分数"""
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.score: float = 1.0  # 初始信誉
        self.contribution_score: float = 0.0
        self.accuracy_score: float = 0.0
        self.consistency_score: float = 1.0
        self.reliability_score: float = 1.0
        
        self.history: List[Tuple[float, datetime]] = []
        self.total_contributions: int = 0
        self.successful_contributions: int = 0
        self.last_update: datetime = datetime.now()
    
    def update(self, new_score: float) -> None:
        """更新分数"""
        self.score = new_score
        self.history.append((new_score, datetime.now()))
        self.last_update = datetime.now()
        
        # 保留最近100条历史
        if len(self.history) > 100:
            self.history = self.history[-100:]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'client_id': self.client_id,
            'score': self.score,
            'contribution_score': self.contribution_score,
            'accuracy_score': self.accuracy_score,
            'consistency_score': self.consistency_score,
            'reliability_score': self.reliability_score,
            'total_contributions': self.total_contributions,
            'successful_contributions': self.successful_contributions
        }


class ReputationSystem:
    """
    联邦信誉系统
    
    管理客户端信誉分数
    """
    
    def __init__(
        self,
        initial_score: float = 1.0,
        min_score: float = 0.0,
        max_score: float = 2.0,
        decay_rate: float = 0.01,
        update_rate: float = 0.1
    ):
        self.initial_score = initial_score
        self.min_score = min_score
        self.max_score = max_score
        self.decay_rate = decay_rate
        self.update_rate = update_rate
        
        self._scores: Dict[str, ReputationScore] = {}
        self._global_round: int = 0
    
    def register_client(self, client_id: str) -> None:
        """注册客户端"""
        if client_id not in self._scores:
            self._scores[client_id] = ReputationScore(client_id)
            self._scores[client_id].score = self.initial_score
    
    def unregister_client(self, client_id: str) -> None:
        """注销客户端"""
        self._scores.pop(client_id, None)
    
    def get_score(self, client_id: str) -> float:
        """获取信誉分数"""
        if client_id in self._scores:
            return self._scores[client_id].score
        return self.initial_score
    
    def update_contribution(
        self,
        client_id: str,
        contribution_value: float,
        success: bool = True
    ) -> float:
        """
        更新贡献
        
        Args:
            client_id: 客户端ID
            contribution_value: 贡献值
            success: 是否成功
        
        Returns:
            新的信誉分数
        """
        self.register_client(client_id)
        score = self._scores[client_id]
        
        score.total_contributions += 1
        if success:
            score.successful_contributions += 1
        
        # 更新贡献分数
        old_contrib = score.contribution_score
        score.contribution_score = (
            (1 - self.update_rate) * old_contrib +
            self.update_rate * contribution_value
        )
        
        # 更新可靠性分数
        reliability = score.successful_contributions / score.total_contributions
        score.reliability_score = reliability
        
        # 综合分数
        new_score = self._compute_composite_score(score)
        score.update(new_score)
        
        return new_score
    
    def update_accuracy(
        self,
        client_id: str,
        accuracy: float
    ) -> float:
        """更新准确率分数"""
        self.register_client(client_id)
        score = self._scores[client_id]
        
        old_acc = score.accuracy_score
        score.accuracy_score = (
            (1 - self.update_rate) * old_acc +
            self.update_rate * accuracy
        )
        
        new_score = self._compute_composite_score(score)
        score.update(new_score)
        
        return new_score
    
    def update_consistency(
        self,
        client_id: str,
        consistency: float
    ) -> float:
        """更新一致性分数"""
        self.register_client(client_id)
        score = self._scores[client_id]
        
        old_cons = score.consistency_score
        score.consistency_score = (
            (1 - self.update_rate) * old_cons +
            self.update_rate * consistency
        )
        
        new_score = self._compute_composite_score(score)
        score.update(new_score)
        
        return new_score
    
    def _compute_composite_score(self, score: ReputationScore) -> float:
        """计算综合分数"""
        # 加权平均
        weights = {
            'contribution': 0.3,
            'accuracy': 0.3,
            'consistency': 0.2,
            'reliability': 0.2
        }
        
        composite = (
            weights['contribution'] * min(score.contribution_score, 1.0) +
            weights['accuracy'] * score.accuracy_score +
            weights['consistency'] * score.consistency_score +
            weights['reliability'] * score.reliability_score
        )
        
        # 裁剪到范围
        return max(self.min_score, min(self.max_score, composite))
    
    def apply_decay(self) -> None:
        """应用衰减"""
        for score in self._scores.values():
            decayed = score.score * (1 - self.decay_rate)
            score.update(max(self.min_score, decayed))
    
    def advance_round(self) -> None:
        """推进轮次"""
        self._global_round += 1
        self.apply_decay()
    
    def get_top_clients(self, k: int = 10) -> List[Tuple[str, float]]:
        """获取信誉最高的客户端"""
        sorted_clients = sorted(
            self._scores.items(),
            key=lambda x: x[1].score,
            reverse=True
        )
        return [(cid, s.score) for cid, s in sorted_clients[:k]]
    
    def get_low_reputation_clients(
        self,
        threshold: float = 0.3
    ) -> List[str]:
        """获取低信誉客户端"""
        return [
            cid for cid, score in self._scores.items()
            if score.score < threshold
        ]
    
    def normalize_scores(self) -> Dict[str, float]:
        """归一化分数"""
        if not self._scores:
            return {}
        
        total = sum(s.score for s in self._scores.values())
        
        if total == 0:
            return {cid: 1.0 / len(self._scores) for cid in self._scores}
        
        return {
            cid: s.score / total
            for cid, s in self._scores.items()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self._scores:
            return {
                'total_clients': 0,
                'avg_score': 0.0,
                'min_score': 0.0,
                'max_score': 0.0
            }
        
        scores = [s.score for s in self._scores.values()]
        
        return {
            'total_clients': len(self._scores),
            'global_round': self._global_round,
            'avg_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'initial_score': self.initial_score,
            'decay_rate': self.decay_rate
        }


class ByzantineFilter:
    """
    拜占庭过滤器
    
    过滤恶意客户端
    """
    
    def __init__(
        self,
        reputation_threshold: float = 0.3,
        outlier_threshold: float = 2.0
    ):
        self.reputation_threshold = reputation_threshold
        self.outlier_threshold = outlier_threshold
    
    def filter_by_reputation(
        self,
        clients: List[str],
        reputation_system: ReputationSystem
    ) -> List[str]:
        """按信誉过滤"""
        return [
            c for c in clients
            if reputation_system.get_score(c) >= self.reputation_threshold
        ]
    
    def filter_by_outlier(
        self,
        updates: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """
        按异常值过滤
        
        使用中位数绝对偏差(MAD)检测异常
        """
        if len(updates) < 3:
            return list(updates.keys())
        
        # 计算每个更新的范数
        norms = {}
        for client_id, update in updates.items():
            norm = 0.0
            for key, value in update.items():
                if isinstance(value, (int, float)):
                    norm += value ** 2
                elif isinstance(value, list):
                    norm += sum(v ** 2 for v in value if isinstance(v, (int, float)))
            norms[client_id] = math.sqrt(norm)
        
        # 计算中位数
        sorted_norms = sorted(norms.values())
        n = len(sorted_norms)
        median = sorted_norms[n // 2] if n % 2 == 1 else (sorted_norms[n // 2 - 1] + sorted_norms[n // 2]) / 2
        
        # 计算MAD
        deviations = [abs(v - median) for v in norms.values()]
        sorted_devs = sorted(deviations)
        mad = sorted_devs[n // 2] if n % 2 == 1 else (sorted_devs[n // 2 - 1] + sorted_devs[n // 2]) / 2
        
        # 过滤异常
        threshold = median + self.outlier_threshold * mad * 1.4826  # 1.4826是正态分布的缩放因子
        
        return [
            client_id for client_id, norm in norms.items()
            if norm <= threshold
        ]
