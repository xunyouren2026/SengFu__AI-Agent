"""
统一注意力机制模块

提供通用的注意力计算功能，支持AGI和视频生成系统的注意力需求。
包括滑动窗口注意力、动态路由和稀疏注意力等。

核心组件：
- UnifiedAttention: 注意力抽象基类
- UnifiedSlidingWindowAttention: 滑动窗口注意力
- UnifiedDynamicRouting: 动态Token路由
- UnifiedSparseAttention: 稀疏注意力
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, List, Dict, Any, Callable, Tuple, Set
from enum import Enum, auto
import math

from .unified_config import (
    UnifiedAlgorithmConfig,
    AttentionPattern,
    T, S
)


# ============================================================================
# 注意力相关的数据结构
# ============================================================================

@dataclass
class AttentionContext:
    """
    注意力上下文
    
    存储注意力计算所需的上下文信息。
    
    Attributes:
        query_length: 查询序列长度
        key_length: 键序列长度
        position: 当前位置
        cache_valid: 缓存是否有效
    """
    query_length: int = 0
    key_length: int = 0
    position: int = 0
    cache_valid: bool = False


@dataclass
class CacheZone:
    """
    缓存区域
    
    用于滑动窗口注意力的分区缓存。
    
    Attributes:
        name: 区域名称
        start: 起始位置
        end: 结束位置
        priority: 优先级
        data: 缓存数据
    """
    name: str
    start: int
    end: int
    priority: int = 0
    data: Optional[Any] = None


@dataclass
class RoutingDecision:
    """
    路由决策
    
    动态路由的输出结果。
    
    Attributes:
        token_indices: 被路由的token索引
        route_weights: 路由权重
        expert_assignments: 专家分配
    """
    token_indices: List[int] = field(default_factory=list)
    route_weights: List[float] = field(default_factory=list)
    expert_assignments: Dict[int, int] = field(default_factory=dict)


# ============================================================================
# 抽象基类
# ============================================================================

class UnifiedAttention(ABC, Generic[T]):
    """
    统一注意力抽象基类
    
    所有注意力机制的基类，定义通用接口。
    支持任意类型的序列数据。
    
    Attributes:
        config: 算法配置
        context: 注意力上下文
    """
    
    def __init__(self, config: Optional[UnifiedAlgorithmConfig] = None):
        """
        初始化注意力机制
        
        Args:
            config: 算法配置
        """
        self.config = config or UnifiedAlgorithmConfig.default_config()
        self.context = AttentionContext()
    
    @abstractmethod
    def compute(self, query: List[T], key: List[T], value: List[T]) -> List[T]:
        """
        计算注意力
        
        Args:
            query: 查询序列
            key: 键序列
            value: 值序列
            
        Returns:
            注意力输出
        """
        pass
    
    @abstractmethod
    def get_attention_weights(self) -> List[List[float]]:
        """
        获取注意力权重
        
        Returns:
            注意力权重矩阵
        """
        pass
    
    def reset_cache(self) -> None:
        """重置缓存"""
        self.context.cache_valid = False
    
    def _softmax(self, values: List[float]) -> List[float]:
        """
        计算softmax
        
        Args:
            values: 输入值
            
        Returns:
            softmax结果
        """
        if not values:
            return []
        
        # 数值稳定性处理
        max_val = max(values)
        exp_vals = [math.exp(v - max_val) for v in values]
        sum_exp = sum(exp_vals)
        
        if sum_exp == 0:
            return [1.0 / len(values)] * len(values)
        
        return [v / sum_exp for v in exp_vals]
    
    def _dot_product(self, a: List[float], b: List[float]) -> float:
        """
        计算点积
        
        Args:
            a: 第一个向量
            b: 第二个向量
            
        Returns:
            点积结果
        """
        return sum(x * y for x, y in zip(a, b))
    
    def _scale(self, values: List[float], scale_factor: float) -> List[float]:
        """
        缩放值
        
        Args:
            values: 输入值
            scale_factor: 缩放因子
            
        Returns:
            缩放后的值
        """
        return [v * scale_factor for v in values]


# ============================================================================
# 滑动窗口注意力
# ============================================================================

class UnifiedSlidingWindowAttention(UnifiedAttention[T]):
    """
    滑动窗口注意力
    
    使用滑动窗口机制限制注意力范围，降低计算复杂度。
    支持三区缓存：当前窗口、邻近窗口、远距离摘要。
    
    Attributes:
        window_size: 窗口大小
        num_zones: 缓存区域数量
        zones: 缓存区域列表
        attention_weights: 注意力权重缓存
    """
    
    def __init__(self, 
                 window_size: Optional[int] = None,
                 num_zones: Optional[int] = None,
                 config: Optional[UnifiedAlgorithmConfig] = None):
        """
        初始化滑动窗口注意力
        
        Args:
            window_size: 窗口大小（默认从配置获取）
            num_zones: 缓存区域数量（默认从配置获取）
            config: 算法配置
        """
        super().__init__(config)
        self.window_size = window_size or self.config.attention_window_size
        self.num_zones = num_zones or self.config.attention_cache_zones
        
        # 初始化缓存区域
        self.zones: List[CacheZone] = []
        self._init_zones()
        
        # 注意力权重缓存
        self._attention_weights: List[List[float]] = []
    
    def _init_zones(self) -> None:
        """初始化缓存区域"""
        zone_configs = [
            ("current", 0, self.window_size, 3),      # 当前窗口 - 最高优先级
            ("near", self.window_size, self.window_size * 2, 2),  # 邻近窗口
            ("distant", self.window_size * 2, self.window_size * 4, 1),  # 远距离
        ]
        
        for name, start, end, priority in zone_configs[:self.num_zones]:
            self.zones.append(CacheZone(name, start, end, priority))
    
    def compute(self, query: List[T], key: List[T], value: List[T]) -> List[T]:
        """
        计算滑动窗口注意力
        
        只在窗口范围内计算注意力，降低计算复杂度。
        
        Args:
            query: 查询序列
            key: 键序列
            value: 值序列
            
        Returns:
            注意力输出
        """
        if not query or not key or not value:
            return []
        
        output = []
        self._attention_weights = []
        
        for i, q in enumerate(query):
            # 确定窗口范围
            window_start = max(0, i - self.window_size // 2)
            window_end = min(len(key), i + self.window_size // 2 + 1)
            
            # 在窗口内计算注意力
            window_keys = key[window_start:window_end]
            window_values = value[window_start:window_end]
            
            # 计算注意力分数
            scores = self._compute_attention_scores(q, window_keys)
            
            # 应用softmax
            weights = self._softmax(scores)
            
            # 加权求和
            output.append(self._weighted_sum(weights, window_values))
            
            # 记录权重（用于可视化/调试）
            full_weights = [0.0] * len(key)
            for j, w in enumerate(weights):
                full_weights[window_start + j] = w
            self._attention_weights.append(full_weights)
        
        self.context.cache_valid = True
        return output
    
    def _compute_attention_scores(self, query: T, keys: List[T]) -> List[float]:
        """
        计算注意力分数
        
        Args:
            query: 查询
            keys: 键列表
            
        Returns:
            注意力分数
        """
        scores = []
        for k in keys:
            score = self._compute_score(query, k)
            scores.append(score)
        return scores
    
    def _compute_score(self, query: T, key: T) -> float:
        """
        计算单个查询-键对的分数
        
        Args:
            query: 查询
            key: 键
            
        Returns:
            分数
        """
        # 默认实现：尝试转换为向量计算点积
        try:
            if isinstance(query, (list, tuple)) and isinstance(key, (list, tuple)):
                q_vec = [float(x) for x in query]
                k_vec = [float(x) for x in key]
                if len(q_vec) == len(k_vec):
                    scale = 1.0 / math.sqrt(len(q_vec))
                    return self._dot_product(q_vec, k_vec) * scale
        except (ValueError, TypeError):
            pass
        
        # 回退到简单比较
        return 1.0 if query == key else 0.0
    
    def _weighted_sum(self, weights: List[float], values: List[T]) -> T:
        """
        加权求和
        
        Args:
            weights: 权重
            values: 值
            
        Returns:
            加权结果
        """
        if not weights or not values:
            return values[0] if values else None  # type: ignore
        
        # 尝试数值加权
        try:
            if isinstance(values[0], (int, float)):
                return sum(w * float(v) for w, v in zip(weights, values))  # type: ignore
            elif isinstance(values[0], (list, tuple)):
                result = []
                for i in range(len(values[0])):
                    val = sum(w * float(v[i]) for w, v in zip(weights, values))
                    result.append(val)
                return type(values[0])(result)  # type: ignore
        except (ValueError, TypeError, IndexError):
            pass
        
        # 回退：返回权重最高的值
        max_idx = max(range(len(weights)), key=lambda i: weights[i])
        return values[max_idx]
    
    def get_attention_weights(self) -> List[List[float]]:
        """获取注意力权重"""
        return self._attention_weights
    
    def update_zone(self, zone_name: str, data: Any) -> None:
        """
        更新缓存区域数据
        
        Args:
            zone_name: 区域名称
            data: 新数据
        """
        for zone in self.zones:
            if zone.name == zone_name:
                zone.data = data
                break
    
    def get_zone_data(self, zone_name: str) -> Optional[Any]:
        """
        获取缓存区域数据
        
        Args:
            zone_name: 区域名称
            
        Returns:
            区域数据
        """
        for zone in self.zones:
            if zone.name == zone_name:
                return zone.data
        return None


# ============================================================================
# 动态Token路由
# ============================================================================

class UnifiedDynamicRouting(Generic[T]):
    """
    动态Token路由
    
    根据token特征动态决定哪些token需要完整计算，哪些可以跳过或简化。
    用于优化长序列的计算效率。
    
    Attributes:
        threshold: 路由阈值
        capacity_factor: 容量因子
        routing_history: 路由历史
    """
    
    def __init__(self, 
                 threshold: float = 0.5,
                 capacity_factor: float = 1.0,
                 config: Optional[UnifiedAlgorithmConfig] = None):
        """
        初始化动态路由
        
        Args:
            threshold: 路由阈值
            capacity_factor: 容量因子
            config: 算法配置
        """
        self.config = config or UnifiedAlgorithmConfig.default_config()
        self.threshold = threshold
        self.capacity_factor = capacity_factor
        self.routing_history: List[RoutingDecision] = []
    
    def route(self, tokens: List[T], 
              importance_scores: Optional[List[float]] = None) -> RoutingDecision:
        """
        路由tokens
        
        根据重要性分数决定哪些tokens需要完整处理。
        
        Args:
            tokens: 输入tokens
            importance_scores: 重要性分数（可选）
            
        Returns:
            路由决策
        """
        if not tokens:
            return RoutingDecision()
        
        # 如果没有提供重要性分数，使用均匀分布
        if importance_scores is None:
            importance_scores = [0.5] * len(tokens)
        
        decision = RoutingDecision()
        
        # 计算容量
        capacity = int(len(tokens) * self.capacity_factor)
        
        # 根据重要性选择tokens
        indexed_scores = [(i, score) for i, score in enumerate(importance_scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected = indexed_scores[:capacity]
        selected_indices = [i for i, _ in selected]
        selected_indices.sort()
        
        decision.token_indices = selected_indices
        decision.route_weights = [importance_scores[i] for i in selected_indices]
        
        # 简单的专家分配（基于位置）
        for idx in selected_indices:
            decision.expert_assignments[idx] = idx % self.config.num_experts
        
        self.routing_history.append(decision)
        
        # 限制历史大小
        if len(self.routing_history) > 100:
            self.routing_history.pop(0)
        
        return decision
    
    def apply_routing(self, tokens: List[T], 
                     decision: RoutingDecision) -> List[T]:
        """
        应用路由决策
        
        Args:
            tokens: 所有tokens
            decision: 路由决策
            
        Returns:
            被路由的tokens
        """
        return [tokens[i] for i in decision.token_indices]
    
    def merge_routed_outputs(self, 
                            routed_outputs: List[T],
                            decision: RoutingDecision,
                            original_length: int) -> List[T]:
        """
        合并路由输出到原始序列
        
        Args:
            routed_outputs: 路由后的输出
            decision: 路由决策
            original_length: 原始序列长度
            
        Returns:
            合并后的完整序列
        """
        result: List[Any] = [None] * original_length
        
        for i, idx in enumerate(decision.token_indices):
            if i < len(routed_outputs):
                result[idx] = routed_outputs[i]
        
        # 填充未路由的位置（使用最近的路由结果）
        last_valid = None
        for i in range(original_length):
            if result[i] is not None:
                last_valid = result[i]
            elif last_valid is not None:
                result[i] = last_valid
        
        # 前向填充
        first_valid = None
        for i in range(original_length - 1, -1, -1):
            if result[i] is not None:
                first_valid = result[i]
            elif first_valid is not None:
                result[i] = first_valid
        
        return result  # type: ignore
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """
        获取路由统计信息
        
        Returns:
            统计信息
        """
        if not self.routing_history:
            return {
                'total_routings': 0,
                'avg_selected_ratio': 0.0,
            }
        
        total_tokens = sum(len(r.token_indices) for r in self.routing_history)
        
        return {
            'total_routings': len(self.routing_history),
            'avg_selected_ratio': total_tokens / (len(self.routing_history) * 100) if self.routing_history else 0.0,
            'threshold': self.threshold,
            'capacity_factor': self.capacity_factor,
        }


# ============================================================================
# 稀疏注意力
# ============================================================================

class UnifiedSparseAttention(UnifiedAttention[T]):
    """
    稀疏注意力
    
    通过稀疏模式减少注意力计算量。
    支持多种稀疏模式：随机稀疏、块稀疏、步进稀疏。
    
    Attributes:
        pattern: 稀疏模式
        sparsity_ratio: 稀疏率
        block_size: 块大小（用于块稀疏）
        stride: 步长（用于步进稀疏）
    """
    
    def __init__(self,
                 pattern: Optional[AttentionPattern] = None,
                 sparsity_ratio: float = 0.9,
                 block_size: int = 64,
                 stride: int = 4,
                 config: Optional[UnifiedAlgorithmConfig] = None):
        """
        初始化稀疏注意力
        
        Args:
            pattern: 稀疏模式
            sparsity_ratio: 稀疏率（0-1，越高越稀疏）
            block_size: 块大小
            stride: 步长
            config: 算法配置
        """
        super().__init__(config)
        self.pattern = pattern or self.config.attention_pattern
        self.sparsity_ratio = sparsity_ratio
        self.block_size = block_size
        self.stride = stride
        
        self._attention_weights: List[List[float]] = []
        self._sparse_mask: Optional[List[List[bool]]] = None
    
    def compute(self, query: List[T], key: List[T], value: List[T]) -> List[T]:
        """
        计算稀疏注意力
        
        只计算稀疏模式指定的位置，降低计算量。
        
        Args:
            query: 查询序列
            key: 键序列
            value: 值序列
            
        Returns:
            注意力输出
        """
        if not query or not key or not value:
            return []
        
        seq_len = len(query)
        
        # 生成稀疏掩码
        self._sparse_mask = self._generate_sparse_mask(seq_len)
        
        output = []
        self._attention_weights = []
        
        for i, q in enumerate(query):
            # 获取当前查询的稀疏掩码
            mask = self._sparse_mask[i] if i < len(self._sparse_mask) else [True] * len(key)
            
            # 根据掩码选择keys和values
            selected_keys = [k for k, m in zip(key, mask) if m]
            selected_values = [v for v, m in zip(value, mask) if m]
            
            if not selected_keys:
                # 如果没有选中任何key，使用所有key
                selected_keys = key
                selected_values = value
                mask = [True] * len(key)
            
            # 计算注意力分数
            scores = self._compute_attention_scores(q, selected_keys)
            weights = self._softmax(scores)
            
            # 加权求和
            output.append(self._weighted_sum(weights, selected_values))
            
            # 记录完整权重
            full_weights = [0.0] * len(key)
            w_idx = 0
            for j, m in enumerate(mask):
                if m and w_idx < len(weights):
                    full_weights[j] = weights[w_idx]
                    w_idx += 1
            self._attention_weights.append(full_weights)
        
        return output
    
    def _generate_sparse_mask(self, seq_len: int) -> List[List[bool]]:
        """
        生成稀疏掩码
        
        Args:
            seq_len: 序列长度
            
        Returns:
            稀疏掩码矩阵
        """
        mask = [[False] * seq_len for _ in range(seq_len)]
        
        if self.pattern == AttentionPattern.DENSE:
            # 密集模式（不稀疏）
            for i in range(seq_len):
                for j in range(seq_len):
                    mask[i][j] = True
        
        elif self.pattern == AttentionPattern.SPARSE_RANDOM:
            # 随机稀疏
            import random
            num_to_select = max(1, int(seq_len * (1 - self.sparsity_ratio)))
            for i in range(seq_len):
                indices = random.sample(range(seq_len), min(num_to_select, seq_len))
                for j in indices:
                    mask[i][j] = True
        
        elif self.pattern == AttentionPattern.SPARSE_BLOCK:
            # 块稀疏
            for i in range(seq_len):
                block_start = (i // self.block_size) * self.block_size
                block_end = min(block_start + self.block_size, seq_len)
                for j in range(block_start, block_end):
                    mask[i][j] = True
                
                # 添加一些全局注意力
                for j in range(0, seq_len, self.block_size):
                    mask[i][j] = True
        
        elif self.pattern == AttentionPattern.SPARSE_STRIDED:
            # 步进稀疏
            for i in range(seq_len):
                # 局部注意力
                local_range = max(1, int(seq_len * (1 - self.sparsity_ratio)))
                for j in range(max(0, i - local_range // 2), min(seq_len, i + local_range // 2)):
                    mask[i][j] = True
                
                # 步进注意力
                for j in range(0, seq_len, self.stride):
                    mask[i][j] = True
        
        return mask
    
    def _compute_attention_scores(self, query: T, keys: List[T]) -> List[float]:
        """计算注意力分数"""
        scores = []
        for k in keys:
            score = self._compute_score(query, k)
            scores.append(score)
        return scores
    
    def _compute_score(self, query: T, key: T) -> float:
        """计算单个查询-键对的分数"""
        try:
            if isinstance(query, (list, tuple)) and isinstance(key, (list, tuple)):
                q_vec = [float(x) for x in query]
                k_vec = [float(x) for x in key]
                if len(q_vec) == len(k_vec):
                    scale = 1.0 / math.sqrt(len(q_vec))
                    return self._dot_product(q_vec, k_vec) * scale
        except (ValueError, TypeError):
            pass
        
        return 1.0 if query == key else 0.0
    
    def _weighted_sum(self, weights: List[float], values: List[T]) -> T:
        """加权求和"""
        if not weights or not values:
            return values[0] if values else None  # type: ignore
        
        try:
            if isinstance(values[0], (int, float)):
                return sum(w * float(v) for w, v in zip(weights, values))  # type: ignore
            elif isinstance(values[0], (list, tuple)):
                result = []
                for i in range(len(values[0])):
                    val = sum(w * float(v[i]) for w, v in zip(weights, values))
                    result.append(val)
                return type(values[0])(result)  # type: ignore
        except (ValueError, TypeError, IndexError):
            pass
        
        max_idx = max(range(len(weights)), key=lambda i: weights[i])
        return values[max_idx]
    
    def get_attention_weights(self) -> List[List[float]]:
        """获取注意力权重"""
        return self._attention_weights
    
    def get_sparsity_stats(self) -> Dict[str, Any]:
        """
        获取稀疏度统计
        
        Returns:
            统计信息
        """
        if self._sparse_mask is None:
            return {
                'pattern': self.pattern.name,
                'configured_sparsity': self.sparsity_ratio,
                'actual_sparsity': 0.0,
            }
        
        total = len(self._sparse_mask) * len(self._sparse_mask[0]) if self._sparse_mask else 0
        active = sum(sum(1 for m in row if m) for row in self._sparse_mask) if self._sparse_mask else 0
        
        actual_sparsity = 1.0 - (active / total) if total > 0 else 0.0
        
        return {
            'pattern': self.pattern.name,
            'configured_sparsity': self.sparsity_ratio,
            'actual_sparsity': actual_sparsity,
            'total_positions': total,
            'active_positions': active,
            'computation_reduction': actual_sparsity,
        }
    
    def set_pattern(self, pattern: AttentionPattern) -> None:
        """
        设置稀疏模式
        
        Args:
            pattern: 新的稀疏模式
        """
        self.pattern = pattern
        self._sparse_mask = None  # 重置掩码
