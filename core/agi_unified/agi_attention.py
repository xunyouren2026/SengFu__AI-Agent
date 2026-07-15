"""
AGI注意力系统
==============

基于UnifiedSlidingWindowAttention的AGI注意力系统实现。

特点：
- 三区缓存机制（当前、局部、全局）
- 动态token路由
- 稀疏注意力模式
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import math
import random
import time

from ..unified_algorithms.unified_attention import (
    UnifiedSlidingWindowAttention,
    UnifiedDynamicRouting,
    UnifiedSparseAttention,
    AttentionContext,
    RoutingDecision,
)
from ..unified_algorithms.unified_config import (
    UnifiedAlgorithmConfig,
    AttentionPattern,
)


@dataclass
class AttentionState:
    """注意力状态"""
    query: List[float] = field(default_factory=list)
    keys: List[List[float]] = field(default_factory=list)
    values: List[List[float]] = field(default_factory=list)
    attention_weights: List[float] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class AGIAttentionSystem:
    """
    AGI注意力系统

    基于统一核心的注意力机制：
    1. 滑动窗口注意力：处理长序列
    2. 动态路由：根据重要性选择token
    3. 稀疏注意力：降低计算复杂度

    Attributes:
        dim: 特征维度
        window_size: 滑动窗口大小
        num_heads: 注意力头数
    """

    def __init__(
        self,
        dim: int = 768,
        window_size: int = 512,
        global_summary_size: int = 64,
        num_heads: int = 8,
        dropout: float = 0.0
    ):
        self.dim = dim
        self.window_size = window_size
        self.global_summary_size = global_summary_size
        self.num_heads = num_heads
        self.d_k = dim // num_heads
        self.dropout = dropout

        # 创建统一配置
        self.config = UnifiedAlgorithmConfig.default_config()

        # 统一滑动窗口注意力
        self._sliding_attention = UnifiedSlidingWindowAttention[
            List[float]
        ](
            window_size=window_size,
            num_zones=3,
            config=self.config
        )

        # 统一动态路由
        self._dynamic_routing = UnifiedDynamicRouting[List[float]](
            threshold=0.5,
            capacity_factor=0.8,
            config=self.config
        )

        # 统一稀疏注意力
        self._sparse_attention = UnifiedSparseAttention[List[float]](
            pattern=AttentionPattern.SLIDING_WINDOW,
            sparsity=0.9,
            config=self.config
        )

        # 投影矩阵
        self.w_q = self._init_weight(dim, dim)
        self.w_k = self._init_weight(dim, dim)
        self.w_v = self._init_weight(dim, dim)
        self.w_o = self._init_weight(dim, dim)

        # 缓存
        self.cache: Dict[str, AttentionState] = {}
        self.cache_size = 100

        # 统计
        self.stats = {
            'forward_calls': 0,
            'cache_hits': 0,
            'tokens_processed': 0,
            'avg_attention_span': 0
        }

    def _init_weight(self, in_dim: int, out_dim: int) -> List[List[float]]:
        """初始化权重"""
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        return [[random.gauss(0, scale) for _ in range(out_dim)] for _ in range(in_dim)]

    def forward(
        self,
        x: List[List[float]],
        use_cache: bool = True,
        attention_type: str = "sliding"
    ) -> List[List[float]]:
        """
        前向传播

        Args:
            x: 输入序列 [seq_len, dim]
            use_cache: 是否使用缓存
            attention_type: 注意力类型 ("sliding", "sparse", "dynamic")

        Returns:
            输出序列 [seq_len, dim]
        """
        self.stats['forward_calls'] += 1
        self.stats['tokens_processed'] += len(x)

        # 检查缓存
        cache_key = self._compute_cache_key(x)
        if use_cache and cache_key in self.cache:
            self.stats['cache_hits'] += 1
            return self._apply_cached_attention(x, cache_key)

        # 根据类型选择注意力机制
        if attention_type == "sliding":
            output = self._sliding_attention.compute(x, x, x)
        elif attention_type == "sparse":
            output = self._sparse_attention.compute(x, x, x)
        elif attention_type == "dynamic":
            output = self._dynamic_attention(x)
        else:
            output = self._standard_attention(x)

        # 更新缓存
        if use_cache:
            self._update_cache(cache_key, x, output)

        return output

    def _dynamic_attention(self, x: List[List[float]]) -> List[List[float]]:
        """动态注意力"""
        seq_len = len(x)

        # 计算token重要性
        importances = [self._compute_token_importance(token) for token in x]

        # 使用统一动态路由
        decision = self._dynamic_routing.route(x, importances)

        # 根据路由决策处理token
        output = []
        for i, token in enumerate(x):
            if i in decision.token_indices:
                # 保留的token进行注意力计算
                output.append(token)
            else:
                # 跳过的token使用简化处理
                output.append([t * 0.5 for t in token])

        return output

    def _standard_attention(self, x: List[List[float]]) -> List[List[float]]:
        """标准注意力计算"""
        # 线性投影
        q = self._linear(x, self.w_q)
        k = self._linear(x, self.w_k)
        v = self._linear(x, self.w_v)

        # 计算注意力
        seq_len = len(x)
        output = []

        for i in range(seq_len):
            # 计算注意力分数
            scores = []
            for j in range(seq_len):
                score = sum(q[i][d] * k[j][d] for d in range(self.dim))
                score = score / math.sqrt(self.dim)
                scores.append(score)

            # Softmax
            weights = self._softmax(scores)

            # 加权求和
            out_i = [0.0] * self.dim
            for j in range(seq_len):
                for d in range(self.dim):
                    out_i[d] += weights[j] * v[j][d]

            output.append(out_i)

        # 输出投影
        output = self._linear(output, self.w_o)

        return output

    def _compute_token_importance(self, token: List[float]) -> float:
        """计算token重要性"""
        # 基于L2范数
        norm = math.sqrt(sum(t * t for t in token))
        return min(norm, 1.0)

    def _linear(
        self,
        x: List[List[float]],
        weight: List[List[float]]
    ) -> List[List[float]]:
        """线性变换"""
        return [
            [sum(x[i][k] * weight[k][j] for k in range(len(x[i]))) for j in range(len(weight[0]))]
            for i in range(len(x))
        ]

    def _softmax(self, x: List[float]) -> List[float]:
        """Softmax"""
        max_x = max(x)
        exp_x = [math.exp(xi - max_x) for xi in x]
        sum_exp = sum(exp_x)
        return [e / sum_exp for e in exp_x]

    def _compute_cache_key(self, x: List[List[float]]) -> str:
        """计算缓存键"""
        # 简化的哈希
        sample = str(x[0][:5]) if x else ""
        return f"attn_{hash(sample) % 10000}"

    def _apply_cached_attention(
        self,
        x: List[List[float]],
        cache_key: str
    ) -> List[List[float]]:
        """应用缓存的注意力"""
        state = self.cache[cache_key]
        # 这里可以添加更复杂的缓存复用逻辑
        return x

    def _update_cache(
        self,
        cache_key: str,
        x: List[List[float]],
        output: List[List[float]]
    ):
        """更新缓存"""
        if len(self.cache) >= self.cache_size:
            # LRU淘汰
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
            del self.cache[oldest_key]

        self.cache[cache_key] = AttentionState(
            timestamp=time.time()
        )

    def get_attention_pattern(self, layer: int = 0) -> Dict[str, Any]:
        """
        获取注意力模式

        Args:
            layer: 层索引

        Returns:
            注意力模式统计
        """
        return {
            'window_size': self.window_size,
            'global_summary_size': self.global_summary_size,
            'num_heads': self.num_heads,
            'cache_size': len(self.cache),
            'stats': self.stats
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'cache_utilization': len(self.cache) / self.cache_size,
            'avg_tokens_per_call': self.stats['tokens_processed'] / max(self.stats['forward_calls'], 1),
            'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['forward_calls'], 1)
        }

    def clear_cache(self):
        """清除缓存"""
        self.cache.clear()
