"""
领域适配器
"""
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime
from enum import Enum
import math


class AdaptationMethod(Enum):
    """适配方法"""
    FINETUNING = "finetuning"
    FEATURE_TRANSFER = "feature_transfer"
    DOMAIN_ADVERSARIAL = "domain_adversarial"
    MOMENT_MATCHING = "moment_matching"
    CORRELATION_ALIGNMENT = "correlation_alignment"


class DomainInfo:
    """领域信息"""
    
    def __init__(
        self,
        domain_id: str,
        name: str,
        feature_stats: Optional[Dict[str, Any]] = None
    ):
        self.domain_id = domain_id
        self.name = name
        self.feature_stats = feature_stats or {}
        self.num_samples: int = 0
        self.label_distribution: Dict[int, float] = {}


class DomainAdapter:
    """
    领域适配器
    
    适配不同领域间的模型
    """
    
    def __init__(
        self,
        method: AdaptationMethod = AdaptationMethod.FINETUNING,
        adaptation_rate: float = 0.01
    ):
        self.method = method
        self.adaptation_rate = adaptation_rate
        
        self._source_domain: Optional[DomainInfo] = None
        self._target_domain: Optional[DomainInfo] = None
        self._adaptation_history: List[Dict[str, Any]] = []
    
    def set_source_domain(self, domain: DomainInfo) -> None:
        """设置源领域"""
        self._source_domain = domain
    
    def set_target_domain(self, domain: DomainInfo) -> None:
        """设置目标领域"""
        self._target_domain = domain
    
    def compute_domain_distance(
        self,
        source_features: List[List[float]],
        target_features: List[List[float]]
    ) -> float:
        """
        计算领域距离
        
        使用最大均值差异(MMD)的简化版本
        """
        if not source_features or not target_features:
            return 0.0
        
        # 计算均值
        def mean(features: List[List[float]]) -> List[float]:
            n = len(features)
            if n == 0:
                return []
            dim = len(features[0])
            result = [0.0] * dim
            for f in features:
                for i, v in enumerate(f):
                    if i < dim:
                        result[i] += v
            return [v / n for v in result]
        
        source_mean = mean(source_features)
        target_mean = mean(target_features)
        
        # L2距离
        if len(source_mean) != len(target_mean):
            return float('inf')
        
        distance = math.sqrt(
            sum((s - t) ** 2 for s, t in zip(source_mean, target_mean))
        )
        
        return distance
    
    def adapt(
        self,
        model_params: Dict[str, Any],
        target_data: List[Tuple[Any, Any]],
        num_epochs: int = 5
    ) -> Dict[str, Any]:
        """
        适配模型
        
        Args:
            model_params: 源模型参数
            target_data: 目标领域数据
            num_epochs: 适配轮数
        
        Returns:
            适配后的模型参数
        """
        adapted_params = model_params.copy()
        
        if self.method == AdaptationMethod.FINETUNING:
            adapted_params = self._finetune(adapted_params, target_data, num_epochs)
        elif self.method == AdaptationMethod.FEATURE_TRANSFER:
            adapted_params = self._feature_transfer(adapted_params, target_data)
        elif self.method == AdaptationMethod.MOMENT_MATCHING:
            adapted_params = self._moment_matching(adapted_params, target_data)
        
        # 记录历史
        self._adaptation_history.append({
            'method': self.method.value,
            'num_epochs': num_epochs,
            'timestamp': datetime.now().timestamp()
        })
        
        return adapted_params
    
    def _finetune(
        self,
        params: Dict[str, Any],
        data: List[Tuple[Any, Any]],
        num_epochs: int
    ) -> Dict[str, Any]:
        """微调适配"""
        # 简化实现：添加小扰动
        import random
        adapted = {}
        
        for key, value in params.items():
            if isinstance(value, (int, float)):
                adapted[key] = value + random.gauss(0, self.adaptation_rate)
            elif isinstance(value, list):
                adapted[key] = [
                    v + random.gauss(0, self.adaptation_rate)
                    for v in value
                ]
            else:
                adapted[key] = value
        
        return adapted
    
    def _feature_transfer(
        self,
        params: Dict[str, Any],
        data: List[Tuple[Any, Any]]
    ) -> Dict[str, Any]:
        """特征迁移"""
        # 简化实现
        return params
    
    def _moment_matching(
        self,
        params: Dict[str, Any],
        data: List[Tuple[Any, Any]]
    ) -> Dict[str, Any]:
        """矩匹配"""
        # 简化实现
        return params
    
    def compute_adaptation_score(
        self,
        source_performance: float,
        target_performance: float
    ) -> float:
        """
        计算适配分数
        
        衡量迁移学习的效果
        """
        # 传输比
        if source_performance == 0:
            return 0.0
        
        return target_performance / source_performance
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'method': self.method.value,
            'adaptation_rate': self.adaptation_rate,
            'total_adaptations': len(self._adaptation_history),
            'source_domain': self._source_domain.name if self._source_domain else None,
            'target_domain': self._target_domain.name if self._target_domain else None
        }


class MultiDomainAdapter:
    """
    多领域适配器
    
    管理多个领域间的适配
    """
    
    def __init__(self):
        self._domains: Dict[str, DomainInfo] = {}
        self._adapters: Dict[Tuple[str, str], DomainAdapter] = {}
        self._transfer_matrix: Dict[Tuple[str, str], float] = {}
    
    def register_domain(self, domain: DomainInfo) -> None:
        """注册领域"""
        self._domains[domain.domain_id] = domain
    
    def unregister_domain(self, domain_id: str) -> None:
        """注销领域"""
        self._domains.pop(domain_id, None)
    
    def get_adapter(
        self,
        source_id: str,
        target_id: str
    ) -> Optional[DomainAdapter]:
        """获取适配器"""
        return self._adapters.get((source_id, target_id))
    
    def create_adapter(
        self,
        source_id: str,
        target_id: str,
        method: AdaptationMethod = AdaptationMethod.FINETUNING
    ) -> DomainAdapter:
        """创建适配器"""
        adapter = DomainAdapter(method=method)
        
        if source_id in self._domains:
            adapter.set_source_domain(self._domains[source_id])
        
        if target_id in self._domains:
            adapter.set_target_domain(self._domains[target_id])
        
        self._adapters[(source_id, target_id)] = adapter
        return adapter
    
    def record_transfer_score(
        self,
        source_id: str,
        target_id: str,
        score: float
    ) -> None:
        """记录迁移分数"""
        self._transfer_matrix[(source_id, target_id)] = score
    
    def get_best_source(
        self,
        target_id: str
    ) -> Optional[Tuple[str, float]]:
        """
        获取最佳源领域
        
        Args:
            target_id: 目标领域ID
        
        Returns:
            (最佳源领域ID, 分数)
        """
        candidates = [
            (src, score)
            for (src, tgt), score in self._transfer_matrix.items()
            if tgt == target_id
        ]
        
        if not candidates:
            return None
        
        return max(candidates, key=lambda x: x[1])
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'num_domains': len(self._domains),
            'num_adapters': len(self._adapters),
            'transfer_scores': {
                f"{src}->{tgt}": score
                for (src, tgt), score in self._transfer_matrix.items()
            }
        }
