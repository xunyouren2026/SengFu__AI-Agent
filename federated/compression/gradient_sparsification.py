"""
梯度稀疏化 - TopK/随机稀疏
"""
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import random
import math


class SparsificationMethod(Enum):
    """稀疏化方法"""
    TOP_K = "top_k"  # TopK稀疏化
    RANDOM_K = "random_k"  # 随机稀疏化
    THRESHOLD = "threshold"  # 阈值稀疏化
    GRADIENT_DROP = "gradient_drop"  # 梯度丢弃


class GradientVector:
    """梯度向量"""
    
    def __init__(self, gradients: Optional[Dict[str, Any]] = None):
        self.gradients = gradients or {}
        self._flat_values: Optional[List[float]] = None
        self._flat_keys: Optional[List[Tuple[str, int]]] = None
    
    def set(self, key: str, value: Union[float, List[float]]) -> None:
        """设置梯度"""
        self.gradients[key] = value
        self._flat_values = None
        self._flat_keys = None
    
    def get(self, key: str) -> Optional[Union[float, List[float]]]:
        """获取梯度"""
        return self.gradients.get(key)
    
    def _flatten(self) -> Tuple[List[float], List[Tuple[str, int]]]:
        """展平梯度"""
        if self._flat_values is not None:
            return self._flat_values, self._flat_keys
        
        values = []
        keys = []
        
        for key, grad in self.gradients.items():
            if isinstance(grad, (int, float)):
                values.append(float(grad))
                keys.append((key, -1))  # -1 表示标量
            elif isinstance(grad, list):
                for i, g in enumerate(grad):
                    if isinstance(g, (int, float)):
                        values.append(float(g))
                        keys.append((key, i))
        
        self._flat_values = values
        self._flat_keys = keys
        
        return values, keys
    
    def _unflatten(
        self,
        values: List[float],
        keys: List[Tuple[str, int]]
    ) -> Dict[str, Any]:
        """从展平形式恢复"""
        result: Dict[str, Any] = {}
        
        for (key, idx), val in zip(keys, values):
            if idx == -1:
                result[key] = val
            else:
                if key not in result:
                    # 找到原始长度
                    orig_grad = self.gradients.get(key, [])
                    if isinstance(orig_grad, list):
                        result[key] = [0.0] * len(orig_grad)
                    else:
                        result[key] = []
                if isinstance(result[key], list) and idx < len(result[key]):
                    result[key][idx] = val
        
        return result
    
    def compute_norm(self) -> float:
        """计算L2范数"""
        values, _ = self._flatten()
        return math.sqrt(sum(v ** 2 for v in values))
    
    def get_num_elements(self) -> int:
        """获取元素数量"""
        values, _ = self._flatten()
        return len(values)


class TopKSparsifier:
    """
    TopK稀疏化器
    
    保留绝对值最大的k个元素
    """
    
    def __init__(self, sparsity: float = 0.9):
        """
        Args:
            sparsity: 稀疏度，保留 (1-sparsity) 的元素
        """
        self.sparsity = sparsity
    
    def sparsify(self, gradient: GradientVector) -> GradientVector:
        """执行TopK稀疏化"""
        values, keys = gradient._flatten()
        
        if not values:
            return GradientVector()
        
        # 计算保留数量
        k = max(1, int(len(values) * (1 - self.sparsity)))
        
        # 找到TopK的索引
        indexed_values = [(abs(v), i) for i, v in enumerate(values)]
        indexed_values.sort(reverse=True)
        
        top_k_indices = {idx for _, idx in indexed_values[:k]}
        
        # 创建稀疏梯度
        sparse_values = [
            values[i] if i in top_k_indices else 0.0
            for i in range(len(values))
        ]
        
        sparse_grads = gradient._unflatten(sparse_values, keys)
        return GradientVector(sparse_grads)
    
    def get_compression_ratio(self, gradient: GradientVector) -> float:
        """获取压缩比"""
        return self.sparsity


class RandomKSparsifier:
    """
    随机稀疏化器
    
    随机保留k个元素
    """
    
    def __init__(self, sparsity: float = 0.9):
        self.sparsity = sparsity
    
    def sparsify(self, gradient: GradientVector) -> GradientVector:
        """执行随机稀疏化"""
        values, keys = gradient._flatten()
        
        if not values:
            return GradientVector()
        
        # 计算保留数量
        k = max(1, int(len(values) * (1 - self.sparsity)))
        
        # 随机选择k个索引
        indices = list(range(len(values)))
        selected = set(random.sample(indices, k))
        
        # 创建稀疏梯度
        # 为了保持无偏性，需要缩放
        scale = len(values) / k
        
        sparse_values = [
            values[i] * scale if i in selected else 0.0
            for i in range(len(values))
        ]
        
        sparse_grads = gradient._unflatten(sparse_values, keys)
        return GradientVector(sparse_grads)


class ThresholdSparsifier:
    """
    阈值稀疏化器
    
    丢弃绝对值小于阈值的元素
    """
    
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
    
    def sparsify(self, gradient: GradientVector) -> GradientVector:
        """执行阈值稀疏化"""
        values, keys = gradient._flatten()
        
        if not values:
            return GradientVector()
        
        # 应用阈值
        sparse_values = [
            v if abs(v) >= self.threshold else 0.0
            for v in values
        ]
        
        sparse_grads = gradient._unflatten(sparse_values, keys)
        return GradientVector(sparse_grads)
    
    def get_sparsity(self, gradient: GradientVector) -> float:
        """获取实际稀疏度"""
        values, _ = gradient._flatten()
        if not values:
            return 0.0
        
        zero_count = sum(1 for v in values if abs(v) < self.threshold)
        return zero_count / len(values)


class GradientDropSparsifier:
    """
    梯度丢弃稀疏化器
    
    基于梯度历史动态调整丢弃阈值
    """
    
    def __init__(
        self,
        initial_threshold: float = 0.01,
        adapt_rate: float = 0.1
    ):
        self.threshold = initial_threshold
        self.adapt_rate = adapt_rate
        self._gradient_history: List[float] = []
    
    def sparsify(self, gradient: GradientVector) -> GradientVector:
        """执行梯度丢弃"""
        values, keys = gradient._flatten()
        
        if not values:
            return GradientVector()
        
        # 计算当前梯度统计
        current_norm = math.sqrt(sum(v ** 2 for v in values))
        self._gradient_history.append(current_norm)
        
        # 自适应调整阈值
        if len(self._gradient_history) > 1:
            avg_norm = sum(self._gradient_history) / len(self._gradient_history)
            self.threshold = self.adapt_rate * avg_norm
        
        # 应用阈值
        sparse_values = [
            v if abs(v) >= self.threshold else 0.0
            for v in values
        ]
        
        sparse_grads = gradient._unflatten(sparse_values, keys)
        return GradientVector(sparse_grads)


class GradientSparsifier:
    """
    梯度稀疏化器
    
    统一的稀疏化接口
    """
    
    def __init__(
        self,
        method: SparsificationMethod = SparsificationMethod.TOP_K,
        sparsity: float = 0.9,
        threshold: float = 0.01
    ):
        self.method = method
        self.sparsity = sparsity
        self.threshold = threshold
        
        # 创建具体的稀疏化器
        self._sparsifier = self._create_sparsifier()
        
        # 统计信息
        self._total_sparsified = 0
        self._total_original_size = 0
        self._total_sparse_size = 0
    
    def _create_sparsifier(self) -> Any:
        """创建稀疏化器实例"""
        if self.method == SparsificationMethod.TOP_K:
            return TopKSparsifier(self.sparsity)
        elif self.method == SparsificationMethod.RANDOM_K:
            return RandomKSparsifier(self.sparsity)
        elif self.method == SparsificationMethod.THRESHOLD:
            return ThresholdSparsifier(self.threshold)
        elif self.method == SparsificationMethod.GRADIENT_DROP:
            return GradientDropSparsifier(self.threshold)
        else:
            return TopKSparsifier(self.sparsity)
    
    def sparsify(
        self,
        gradients: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        稀疏化梯度
        
        Args:
            gradients: 梯度字典
        
        Returns:
            稀疏化后的梯度
        """
        gradient = GradientVector(gradients)
        sparse_gradient = self._sparsifier.sparsify(gradient)
        
        # 更新统计
        self._total_sparsified += 1
        self._total_original_size += gradient.get_num_elements()
        self._total_sparse_size += sum(
            1 for v in sparse_gradient._flatten()[0] if v != 0
        )
        
        return sparse_gradient.gradients
    
    def sparsify_update(
        self,
        update: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        稀疏化更新并返回索引
        
        用于高效传输
        """
        sparse_update = self.sparsify(update)
        
        # 提取非零索引
        indices: Dict[str, List[int]] = {}
        for key, value in sparse_update.items():
            if isinstance(value, list):
                non_zero_idx = [i for i, v in enumerate(value) if v != 0]
                if non_zero_idx:
                    indices[key] = non_zero_idx
            elif value != 0:
                indices[key] = [-1]  # 标量标记
        
        return sparse_update, indices
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """获取压缩统计"""
        if self._total_original_size == 0:
            actual_sparsity = 0.0
        else:
            actual_sparsity = 1.0 - self._total_sparse_size / self._total_original_size
        
        return {
            'method': self.method.value,
            'target_sparsity': self.sparsity,
            'actual_sparsity': actual_sparsity,
            'total_sparsified': self._total_sparsified,
            'total_original_size': self._total_original_size,
            'total_sparse_size': self._total_sparse_size,
            'compression_ratio': 1.0 / (1.0 - actual_sparsity) if actual_sparsity < 1.0 else float('inf')
        }
    
    def reset_stats(self) -> None:
        """重置统计"""
        self._total_sparsified = 0
        self._total_original_size = 0
        self._total_sparse_size = 0


class SparseAggregator:
    """
    稀疏梯度聚合器
    
    处理来自不同客户端的稀疏梯度
    """
    
    def __init__(self):
        self._accumulated: Dict[str, Any] = {}
        self._counts: Dict[str, int] = {}
    
    def accumulate(
        self,
        sparse_gradients: Dict[str, Any],
        weight: float = 1.0
    ) -> None:
        """累积稀疏梯度"""
        for key, value in sparse_gradients.items():
            if key not in self._accumulated:
                if isinstance(value, list):
                    self._accumulated[key] = [0.0] * len(value)
                    self._counts[key] = 0
                else:
                    self._accumulated[key] = 0.0
                    self._counts[key] = 0
            
            if isinstance(value, list):
                for i, v in enumerate(value):
                    if v != 0:
                        self._accumulated[key][i] += weight * v
                self._counts[key] += 1
            elif value != 0:
                self._accumulated[key] += weight * value
                self._counts[key] += 1
    
    def aggregate(self) -> Dict[str, Any]:
        """计算聚合结果"""
        result: Dict[str, Any] = {}
        
        for key, value in self._accumulated.items():
            count = self._counts[key]
            if count > 0:
                if isinstance(value, list):
                    result[key] = [v / count for v in value]
                else:
                    result[key] = value / count
        
        return result
    
    def reset(self) -> None:
        """重置累积器"""
        self._accumulated = {}
        self._counts = {}
