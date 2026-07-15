"""
客户端选择器 - 重要性采样
"""
from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime
from enum import Enum
import random
import math


class SamplingStrategy(Enum):
    """采样策略"""
    UNIFORM = "uniform"  # 均匀采样
    IMPORTANCE = "importance"  # 重要性采样
    LOSS_BASED = "loss_based"  # 基于损失
    GRADIENT_BASED = "gradient_based"  # 基于梯度范数
    AGE_BASED = "age_based"  # 基于年龄
    HYBRID = "hybrid"  # 混合策略


class ClientStatistics:
    """客户端统计信息"""
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.num_samples: int = 0
        self.last_loss: float = 1.0
        self.avg_loss: float = 1.0
        self.loss_history: List[float] = []
        self.gradient_norm: float = 1.0
        self.last_participation_round: int = 0
        self.total_participations: int = 0
        self.compute_time: float = 1.0  # 相对计算时间
        self.data_quality: float = 1.0  # 数据质量分数
    
    def update_loss(self, loss: float) -> None:
        """更新损失"""
        self.last_loss = loss
        self.loss_history.append(loss)
        
        # 指数移动平均
        alpha = 0.3
        self.avg_loss = alpha * loss + (1 - alpha) * self.avg_loss
    
    def update_gradient_norm(self, norm: float) -> None:
        """更新梯度范数"""
        self.gradient_norm = norm
    
    def get_age(self, current_round: int) -> int:
        """获取未参与轮次（年龄）"""
        return current_round - self.last_participation_round
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'client_id': self.client_id,
            'num_samples': self.num_samples,
            'last_loss': self.last_loss,
            'avg_loss': self.avg_loss,
            'gradient_norm': self.gradient_norm,
            'last_participation_round': self.last_participation_round,
            'total_participations': self.total_participations
        }


class ImportanceWeighter:
    """重要性权重计算器"""
    
    def __init__(
        self,
        loss_weight: float = 0.4,
        gradient_weight: float = 0.3,
        age_weight: float = 0.2,
        data_weight: float = 0.1
    ):
        self.loss_weight = loss_weight
        self.gradient_weight = gradient_weight
        self.age_weight = age_weight
        self.data_weight = data_weight
    
    def compute_weight(
        self,
        stats: ClientStatistics,
        current_round: int,
        normalize: bool = True
    ) -> float:
        """
        计算重要性权重
        
        高损失、高梯度、长时间未参与的客户端更重要
        """
        # 损失分量（归一化）
        loss_score = min(stats.avg_loss, 10.0) / 10.0
        
        # 梯度分量（归一化）
        grad_score = min(stats.gradient_norm, 10.0) / 10.0
        
        # 年龄分量
        age = stats.get_age(current_round)
        age_score = min(age / 10.0, 1.0)
        
        # 数据分量
        data_score = stats.data_quality
        
        # 加权组合
        weight = (
            self.loss_weight * loss_score +
            self.gradient_weight * grad_score +
            self.age_weight * age_score +
            self.data_weight * data_score
        )
        
        return weight


class ClientSelector:
    """
    客户端选择器
    
    实现多种客户端选择策略:
    - 均匀采样
    - 重要性采样
    - 基于损失的选择
    - 基于梯度的选择
    - 混合策略
    """
    
    def __init__(
        self,
        strategy: SamplingStrategy = SamplingStrategy.IMPORTANCE,
        participation_rate: float = 0.1,
        min_clients: int = 1,
        max_clients: Optional[int] = None,
        fairness_factor: float = 0.1
    ):
        self.strategy = strategy
        self.participation_rate = participation_rate
        self.min_clients = min_clients
        self.max_clients = max_clients
        self.fairness_factor = fairness_factor  # 公平性因子
        
        self._client_stats: Dict[str, ClientStatistics] = {}
        self._weighter = ImportanceWeighter()
        self._current_round: int = 0
    
    def register_client(
        self,
        client_id: str,
        num_samples: int = 0,
        data_quality: float = 1.0
    ) -> None:
        """注册客户端"""
        if client_id not in self._client_stats:
            stats = ClientStatistics(client_id)
            stats.num_samples = num_samples
            stats.data_quality = data_quality
            self._client_stats[client_id] = stats
        else:
            self._client_stats[client_id].num_samples = num_samples
            self._client_stats[client_id].data_quality = data_quality
    
    def unregister_client(self, client_id: str) -> None:
        """注销客户端"""
        self._client_stats.pop(client_id, None)
    
    def update_client_stats(
        self,
        client_id: str,
        loss: Optional[float] = None,
        gradient_norm: Optional[float] = None
    ) -> None:
        """更新客户端统计"""
        if client_id not in self._client_stats:
            self.register_client(client_id)
        
        stats = self._client_stats[client_id]
        
        if loss is not None:
            stats.update_loss(loss)
        
        if gradient_norm is not None:
            stats.update_gradient_norm(gradient_norm)
        
        stats.last_participation_round = self._current_round
        stats.total_participations += 1
    
    def select(
        self,
        available_clients: Optional[Set[str]] = None,
        current_round: Optional[int] = None
    ) -> List[str]:
        """
        选择客户端
        
        Args:
            available_clients: 可用客户端集合
            current_round: 当前轮次
        
        Returns:
            选中的客户端ID列表
        """
        if current_round is not None:
            self._current_round = current_round
        
        # 确定候选客户端
        candidates = available_clients or set(self._client_stats.keys())
        candidates = {c for c in candidates if c in self._client_stats}
        
        if not candidates:
            return []
        
        # 计算选择数量
        target_count = self._compute_target_count(len(candidates))
        
        # 根据策略选择
        if self.strategy == SamplingStrategy.UNIFORM:
            return self._select_uniform(candidates, target_count)
        elif self.strategy == SamplingStrategy.IMPORTANCE:
            return self._select_importance(candidates, target_count)
        elif self.strategy == SamplingStrategy.LOSS_BASED:
            return self._select_loss_based(candidates, target_count)
        elif self.strategy == SamplingStrategy.GRADIENT_BASED:
            return self._select_gradient_based(candidates, target_count)
        elif self.strategy == SamplingStrategy.AGE_BASED:
            return self._select_age_based(candidates, target_count)
        elif self.strategy == SamplingStrategy.HYBRID:
            return self._select_hybrid(candidates, target_count)
        else:
            return self._select_uniform(candidates, target_count)
    
    def _compute_target_count(self, num_candidates: int) -> int:
        """计算目标选择数量"""
        count = max(
            self.min_clients,
            int(num_candidates * self.participation_rate)
        )
        
        if self.max_clients:
            count = min(count, self.max_clients)
        
        return min(count, num_candidates)
    
    def _select_uniform(
        self,
        candidates: Set[str],
        count: int
    ) -> List[str]:
        """均匀随机采样"""
        return random.sample(list(candidates), count)
    
    def _select_importance(
        self,
        candidates: Set[str],
        count: int
    ) -> List[str]:
        """重要性采样"""
        # 计算权重
        weights = {}
        for client_id in candidates:
            stats = self._client_stats[client_id]
            weight = self._weighter.compute_weight(stats, self._current_round)
            
            # 添加公平性调整
            if stats.total_participations > 0:
                fairness_penalty = self.fairness_factor / stats.total_participations
                weight = weight * (1 + fairness_penalty)
            
            weights[client_id] = weight
        
        # 加权采样
        return self._weighted_sample(weights, count)
    
    def _select_loss_based(
        self,
        candidates: Set[str],
        count: int
    ) -> List[str]:
        """基于损失的选择"""
        # 按损失排序（高损失优先）
        sorted_clients = sorted(
            candidates,
            key=lambda c: self._client_stats[c].avg_loss,
            reverse=True
        )
        
        # 贪婪选择 + 随机性
        selected = []
        for client_id in sorted_clients:
            if len(selected) >= count:
                break
            
            # 添加随机性避免总是选择相同的客户端
            if random.random() < 0.8:  # 80%概率按顺序选择
                selected.append(client_id)
            else:
                # 随机选择一个未选中的
                remaining = [c for c in candidates if c not in selected]
                if remaining:
                    selected.append(random.choice(remaining))
        
        return selected[:count]
    
    def _select_gradient_based(
        self,
        candidates: Set[str],
        count: int
    ) -> List[str]:
        """基于梯度范数的选择"""
        # 按梯度范数排序
        sorted_clients = sorted(
            candidates,
            key=lambda c: self._client_stats[c].gradient_norm,
            reverse=True
        )
        
        return sorted_clients[:count]
    
    def _select_age_based(
        self,
        candidates: Set[str],
        count: int
    ) -> List[str]:
        """基于年龄的选择"""
        # 按年龄排序（长时间未参与优先）
        sorted_clients = sorted(
            candidates,
            key=lambda c: self._client_stats[c].get_age(self._current_round),
            reverse=True
        )
        
        return sorted_clients[:count]
    
    def _select_hybrid(
        self,
        candidates: Set[str],
        count: int
    ) -> List[str]:
        """混合策略"""
        # 组合多种策略
        importance_selected = set(self._select_importance(candidates, count))
        age_selected = set(self._select_age_based(candidates, count // 2))
        
        # 合并
        combined = importance_selected | age_selected
        
        # 如果不够，随机补充
        if len(combined) < count:
            remaining = [c for c in candidates if c not in combined]
            needed = count - len(combined)
            combined.update(random.sample(remaining, min(needed, len(remaining))))
        
        return list(combined)[:count]
    
    def _weighted_sample(
        self,
        weights: Dict[str, float],
        count: int
    ) -> List[str]:
        """加权采样"""
        selected = []
        remaining = dict(weights)
        
        for _ in range(min(count, len(remaining))):
            total = sum(remaining.values())
            if total == 0:
                break
            
            r = random.uniform(0, total)
            cumulative = 0.0
            
            for client_id, weight in remaining.items():
                cumulative += weight
                if r <= cumulative:
                    selected.append(client_id)
                    del remaining[client_id]
                    break
        
        return selected
    
    def advance_round(self) -> int:
        """推进轮次"""
        self._current_round += 1
        return self._current_round
    
    def get_client_stats(self, client_id: str) -> Optional[ClientStatistics]:
        """获取客户端统计"""
        return self._client_stats.get(client_id)
    
    def get_all_stats(self) -> Dict[str, ClientStatistics]:
        """获取所有统计"""
        return dict(self._client_stats)
    
    def get_selection_history(self, client_id: str) -> int:
        """获取客户端参与次数"""
        stats = self._client_stats.get(client_id)
        return stats.total_participations if stats else 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取选择器统计"""
        total_participations = sum(
            s.total_participations for s in self._client_stats.values()
        )
        
        avg_loss = (
            sum(s.avg_loss for s in self._client_stats.values()) / len(self._client_stats)
            if self._client_stats else 0
        )
        
        return {
            'strategy': self.strategy.value,
            'current_round': self._current_round,
            'total_clients': len(self._client_stats),
            'total_participations': total_participations,
            'avg_loss': avg_loss,
            'participation_rate': self.participation_rate
        }
