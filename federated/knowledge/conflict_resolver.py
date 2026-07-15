"""
知识冲突解决
"""
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime
from enum import Enum
import math


class ConflictType(Enum):
    """冲突类型"""
    PREDICTION = "prediction"  # 预测冲突
    PARAMETER = "parameter"  # 参数冲突
    GRADIENT = "gradient"  # 梯度冲突
    LABEL = "label"  # 标签冲突（噪声标签）


class Conflict:
    """知识冲突"""
    
    def __init__(
        self,
        conflict_id: str,
        conflict_type: ConflictType,
        sources: List[str],
        values: List[Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.conflict_id = conflict_id
        self.conflict_type = conflict_type
        self.sources = sources  # 冲突来源（客户端ID）
        self.values = values  # 冲突值
        self.metadata = metadata or {}
        self.detected_at = datetime.now().timestamp()
        self.resolved = False
        self.resolution: Optional[Any] = None


class ConflictDetector:
    """冲突检测器"""
    
    def __init__(
        self,
        prediction_threshold: float = 0.3,
        parameter_threshold: float = 1.0
    ):
        self.prediction_threshold = prediction_threshold
        self.parameter_threshold = parameter_threshold
    
    def detect_prediction_conflict(
        self,
        predictions: Dict[str, List[float]]
    ) -> List[Conflict]:
        """
        检测预测冲突
        
        Args:
            predictions: 客户端ID -> 预测值列表
        """
        conflicts = []
        
        if len(predictions) < 2:
            return conflicts
        
        client_ids = list(predictions.keys())
        pred_lists = list(predictions.values())
        
        # 检查每个预测位置
        n_preds = len(pred_lists[0])
        
        for i in range(n_preds):
            values_at_i = [preds[i] for preds in pred_lists if i < len(preds)]
            
            if len(values_at_i) < 2:
                continue
            
            # 计算方差
            mean = sum(values_at_i) / len(values_at_i)
            variance = sum((v - mean) ** 2 for v in values_at_i) / len(values_at_i)
            
            if variance > self.prediction_threshold ** 2:
                conflict = Conflict(
                    conflict_id=f"pred_{i}_{int(datetime.now().timestamp())}",
                    conflict_type=ConflictType.PREDICTION,
                    sources=client_ids,
                    values=values_at_i,
                    metadata={'position': i, 'variance': variance}
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def detect_parameter_conflict(
        self,
        params: Dict[str, Dict[str, Any]]
    ) -> List[Conflict]:
        """
        检测参数冲突
        
        Args:
            params: 客户端ID -> 参数字典
        """
        conflicts = []
        
        if len(params) < 2:
            return conflicts
        
        client_ids = list(params.keys())
        param_dicts = list(params.values())
        
        # 收集所有参数键
        all_keys: Set[str] = set()
        for p in param_dicts:
            all_keys.update(p.keys())
        
        for key in all_keys:
            values = []
            sources = []
            
            for cid, p in zip(client_ids, param_dicts):
                if key in p:
                    val = p[key]
                    if isinstance(val, (int, float)):
                        values.append(val)
                        sources.append(cid)
            
            if len(values) < 2:
                continue
            
            # 计算差异
            mean = sum(values) / len(values)
            max_diff = max(abs(v - mean) for v in values)
            
            if max_diff > self.parameter_threshold:
                conflict = Conflict(
                    conflict_id=f"param_{key}_{int(datetime.now().timestamp())}",
                    conflict_type=ConflictType.PARAMETER,
                    sources=sources,
                    values=values,
                    metadata={'key': key, 'max_diff': max_diff}
                )
                conflicts.append(conflict)
        
        return conflicts


class ConflictResolver:
    """
    知识冲突解决器
    
    解决联邦学习中的知识冲突
    """
    
    def __init__(
        self,
        strategy: str = "weighted_average",
        trust_scores: Optional[Dict[str, float]] = None
    ):
        self.strategy = strategy
        self._trust_scores = trust_scores or {}
        self._resolved_conflicts: List[Conflict] = []
    
    def set_trust_score(self, client_id: str, score: float) -> None:
        """设置信任分数"""
        self._trust_scores[client_id] = score
    
    def resolve(self, conflict: Conflict) -> Any:
        """
        解决冲突
        
        Args:
            conflict: 冲突对象
        
        Returns:
            解决后的值
        """
        if self.strategy == "weighted_average":
            resolution = self._resolve_weighted_average(conflict)
        elif self.strategy == "median":
            resolution = self._resolve_median(conflict)
        elif self.strategy == "trust_based":
            resolution = self._resolve_trust_based(conflict)
        elif self.strategy == "voting":
            resolution = self._resolve_voting(conflict)
        else:
            resolution = self._resolve_weighted_average(conflict)
        
        conflict.resolved = True
        conflict.resolution = resolution
        self._resolved_conflicts.append(conflict)
        
        return resolution
    
    def _resolve_weighted_average(self, conflict: Conflict) -> Any:
        """加权平均解决"""
        values = conflict.values
        sources = conflict.sources
        
        if not values:
            return None
        
        # 计算权重
        weights = []
        for source in sources:
            weight = self._trust_scores.get(source, 1.0)
            weights.append(weight)
        
        total_weight = sum(weights)
        if total_weight == 0:
            return sum(values) / len(values)
        
        weighted_sum = sum(w * v for w, v in zip(weights, values))
        return weighted_sum / total_weight
    
    def _resolve_median(self, conflict: Conflict) -> Any:
        """中位数解决"""
        values = sorted(conflict.values)
        n = len(values)
        
        if n == 0:
            return None
        
        if n % 2 == 0:
            return (values[n // 2 - 1] + values[n // 2]) / 2
        else:
            return values[n // 2]
    
    def _resolve_trust_based(self, conflict: Conflict) -> Any:
        """基于信任的解决"""
        # 选择信任度最高的来源
        sources = conflict.sources
        values = conflict.values
        
        if not sources or not values:
            return None
        
        best_idx = 0
        best_trust = 0.0
        
        for i, source in enumerate(sources):
            trust = self._trust_scores.get(source, 0.5)
            if trust > best_trust:
                best_trust = trust
                best_idx = i
        
        return values[best_idx]
    
    def _resolve_voting(self, conflict: Conflict) -> Any:
        """投票解决"""
        values = conflict.values
        
        if not values:
            return None
        
        # 统计票数
        vote_count: Dict[Any, int] = {}
        for val in values:
            # 离散化连续值
            key = round(val, 2) if isinstance(val, float) else val
            vote_count[key] = vote_count.get(key, 0) + 1
        
        # 选择票数最多的
        best = max(vote_count.items(), key=lambda x: x[1])
        return best[0]
    
    def resolve_all(
        self,
        conflicts: List[Conflict]
    ) -> Dict[str, Any]:
        """
        批量解决冲突
        
        Returns:
            冲突ID -> 解决值
        """
        resolutions = {}
        
        for conflict in conflicts:
            resolution = self.resolve(conflict)
            resolutions[conflict.conflict_id] = resolution
        
        return resolutions
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        by_type: Dict[ConflictType, int] = {}
        
        for conflict in self._resolved_conflicts:
            t = conflict.conflict_type
            by_type[t] = by_type.get(t, 0) + 1
        
        return {
            'strategy': self.strategy,
            'total_resolved': len(self._resolved_conflicts),
            'by_type': {t.value: c for t, c in by_type.items()},
            'num_trusted_clients': len(self._trust_scores)
        }


class LabelConflictResolver(ConflictResolver):
    """
    标签冲突解决器
    
    专门处理标签噪声问题
    """
    
    def __init__(self, noise_threshold: float = 0.3):
        super().__init__(strategy="trust_based")
        self.noise_threshold = noise_threshold
        self._label_history: Dict[str, List[Tuple[str, int]]] = {}  # sample_id -> [(client_id, label)]
    
    def record_label(
        self,
        sample_id: str,
        client_id: str,
        label: int
    ) -> None:
        """记录标签"""
        if sample_id not in self._label_history:
            self._label_history[sample_id] = []
        
        self._label_history[sample_id].append((client_id, label))
    
    def detect_label_noise(
        self,
        sample_id: str
    ) -> Optional[Conflict]:
        """检测标签噪声"""
        history = self._label_history.get(sample_id)
        
        if not history or len(history) < 2:
            return None
        
        labels = [label for _, label in history]
        sources = [client_id for client_id, _ in history]
        
        # 检查标签是否一致
        unique_labels = set(labels)
        
        if len(unique_labels) > 1:
            return Conflict(
                conflict_id=f"label_{sample_id}",
                conflict_type=ConflictType.LABEL,
                sources=sources,
                values=labels,
                metadata={'sample_id': sample_id}
            )
        
        return None
    
    def resolve_label(
        self,
        sample_id: str
    ) -> Optional[int]:
        """解决标签冲突"""
        conflict = self.detect_label_noise(sample_id)
        
        if conflict is None:
            history = self._label_history.get(sample_id)
            if history:
                return history[0][1]
            return None
        
        resolution = self.resolve(conflict)
        return int(round(resolution)) if resolution is not None else None
