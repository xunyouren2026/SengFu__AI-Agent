"""
特征对齐
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import math


class AlignmentMethod(Enum):
    """对齐方法"""
    CORAL = "coral"  # CORAL
    MMD = "mmd"  # 最大均值差异
    ADVERSARIAL = "adversarial"  # 对抗对齐
    STATISTICAL = "statistical"  # 统计对齐


class FeatureStatistics:
    """特征统计"""
    
    def __init__(self, features: List[List[float]]):
        self.features = features
        self.mean: List[float] = []
        self.std: List[float] = []
        self.covariance: List[List[float]] = []
        
        self._compute_statistics()
    
    def _compute_statistics(self) -> None:
        """计算统计量"""
        if not self.features:
            return
        
        n = len(self.features)
        dim = len(self.features[0]) if self.features else 0
        
        # 均值
        self.mean = [0.0] * dim
        for f in self.features:
            for i, v in enumerate(f):
                if i < dim:
                    self.mean[i] += v
        self.mean = [m / n for m in self.mean]
        
        # 标准差
        self.std = [0.0] * dim
        for f in self.features:
            for i, v in enumerate(f):
                if i < dim:
                    self.std[i] += (v - self.mean[i]) ** 2
        self.std = [math.sqrt(s / n) for s in self.std]
        
        # 协方差（简化：只计算对角）
        self.covariance = [[0.0] * dim for _ in range(dim)]
        for f in self.features:
            for i in range(dim):
                for j in range(dim):
                    if i < len(f) and j < len(f):
                        self.covariance[i][j] += (f[i] - self.mean[i]) * (f[j] - self.mean[j])
        
        for i in range(dim):
            for j in range(dim):
                self.covariance[i][j] /= n


class FeatureAligner:
    """
    特征对齐器
    
    对齐不同领域的特征分布
    """
    
    def __init__(
        self,
        method: AlignmentMethod = AlignmentMethod.CORAL,
        regularization: float = 1e-5
    ):
        self.method = method
        self.regularization = regularization
        
        self._source_stats: Optional[FeatureStatistics] = None
        self._target_stats: Optional[FeatureStatistics] = None
        self._transform_matrix: Optional[List[List[float]]] = None
    
    def fit(
        self,
        source_features: List[List[float]],
        target_features: List[List[float]]
    ) -> None:
        """
        拟合对齐变换
        
        Args:
            source_features: 源领域特征
            target_features: 目标领域特征
        """
        self._source_stats = FeatureStatistics(source_features)
        self._target_stats = FeatureStatistics(target_features)
        
        if self.method == AlignmentMethod.CORAL:
            self._fit_coral()
        elif self.method == AlignmentMethod.STATISTICAL:
            self._fit_statistical()
        else:
            self._fit_statistical()
    
    def _fit_coral(self) -> None:
        """CORAL对齐"""
        if self._source_stats is None or self._target_stats is None:
            return
        
        dim = len(self._source_stats.mean)
        if dim == 0:
            return
        
        # 简化的CORAL：白化源特征，再着色为目标分布
        # 这里使用简化的对角近似
        
        self._transform_matrix = []
        for i in range(dim):
            row = [0.0] * dim
            if i < len(self._source_stats.std) and i < len(self._target_stats.std):
                src_std = self._source_stats.std[i] + self.regularization
                tgt_std = self._target_stats.std[i]
                row[i] = tgt_std / src_std
            else:
                row[i] = 1.0
            self._transform_matrix.append(row)
    
    def _fit_statistical(self) -> None:
        """统计对齐"""
        if self._source_stats is None or self._target_stats is None:
            return
        
        dim = len(self._source_stats.mean)
        if dim == 0:
            return
        
        # 标准化变换
        self._transform_matrix = []
        for i in range(dim):
            row = [0.0] * dim
            if i < len(self._source_stats.std) and self._source_stats.std[i] > 0:
                row[i] = self._target_stats.std[i] / self._source_stats.std[i]
            else:
                row[i] = 1.0
            self._transform_matrix.append(row)
    
    def transform(
        self,
        features: List[List[float]]
    ) -> List[List[float]]:
        """
        变换特征
        
        Args:
            features: 原始特征
        
        Returns:
            对齐后的特征
        """
        if self._transform_matrix is None:
            return features
        
        if self._source_stats is None or self._target_stats is None:
            return features
        
        transformed = []
        
        for f in features:
            # 中心化
            centered = [
                v - m for v, m in zip(f, self._source_stats.mean)
            ]
            
            # 线性变换
            scaled = []
            for i in range(len(centered)):
                val = 0.0
                for j in range(len(centered)):
                    if i < len(self._transform_matrix) and j < len(self._transform_matrix[i]):
                        val += self._transform_matrix[i][j] * centered[j]
                scaled.append(val)
            
            # 加上目标均值
            aligned = [
                v + m for v, m in zip(scaled, self._target_stats.mean)
            ]
            
            transformed.append(aligned)
        
        return transformed
    
    def compute_alignment_score(
        self,
        source_features: List[List[float]],
        target_features: List[List[float]]
    ) -> float:
        """
        计算对齐分数
        
        使用特征距离
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
        
        src_mean = mean(source_features)
        tgt_mean = mean(target_features)
        
        # 计算距离
        if len(src_mean) != len(tgt_mean):
            return 0.0
        
        distance = math.sqrt(
            sum((s - t) ** 2 for s, t in zip(src_mean, tgt_mean))
        )
        
        # 转换为分数（距离越小分数越高）
        return 1.0 / (1.0 + distance)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'method': self.method.value,
            'regularization': self.regularization,
            'has_transform': self._transform_matrix is not None,
            'transform_dim': len(self._transform_matrix) if self._transform_matrix else 0
        }


class MultiSourceAligner:
    """
    多源对齐器
    
    对齐多个源领域到目标领域
    """
    
    def __init__(self, method: AlignmentMethod = AlignmentMethod.CORAL):
        self.method = method
        self._aligners: Dict[str, FeatureAligner] = {}
        self._alignment_scores: Dict[str, float] = {}
    
    def add_source(
        self,
        source_id: str,
        source_features: List[List[float]],
        target_features: List[List[float]]
    ) -> None:
        """添加源领域"""
        aligner = FeatureAligner(method=self.method)
        aligner.fit(source_features, target_features)
        
        self._aligners[source_id] = aligner
        
        # 计算对齐分数
        score = aligner.compute_alignment_score(source_features, target_features)
        self._alignment_scores[source_id] = score
    
    def remove_source(self, source_id: str) -> None:
        """移除源领域"""
        self._aligners.pop(source_id, None)
        self._alignment_scores.pop(source_id, None)
    
    def transform(
        self,
        source_id: str,
        features: List[List[float]]
    ) -> List[List[float]]:
        """变换特定源的特征"""
        if source_id not in self._aligners:
            return features
        
        return self._aligners[source_id].transform(features)
    
    def get_best_source(self) -> Optional[str]:
        """获取最佳源领域"""
        if not self._alignment_scores:
            return None
        
        return max(self._alignment_scores.items(), key=lambda x: x[1])[0]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'method': self.method.value,
            'num_sources': len(self._aligners),
            'alignment_scores': dict(self._alignment_scores)
        }
