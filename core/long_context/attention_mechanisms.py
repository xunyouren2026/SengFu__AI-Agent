"""
长上下文注意力机制 - 借鉴视频生成技术
========================================

本模块实现了多种长上下文注意力机制，灵感来源于视频生成领域。

重构说明：
- 内部使用core/unified_algorithms/统一核心
- 通过unified_adapter.py适配器保持API兼容
- 原有API完全保持不变

核心组件：
1. SlidingWindowContextAttention: 滑动窗口注意力（基于UnifiedSlidingWindowAttention）
2. DynamicContextRouting: 动态路由注意力（基于UnifiedDynamicRouting）
3. HierarchicalContextAttention: 分层注意力
4. CrossModalContextAttention: 跨模态注意力

纯Python实现，仅使用标准库。
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable, Set
from enum import Enum, auto

# 导入统一核心适配器
from .unified_adapter import (
    ContextAttentionAdapter,
    DynamicRoutingAdapter,
    TokenInfo,
    cosine_similarity,
    normalize_vector,
)

# 导入统一核心
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


# ============================================================================
# 工具函数
# ============================================================================

def softmax(x: List[float], temperature: float = 1.0) -> List[float]:
    """Softmax函数"""
    if not x:
        return []
    max_x = max(x)
    exp_x = [math.exp((xi - max_x) / temperature) for xi in x]
    sum_exp = sum(exp_x)
    return [e / sum_exp for e in exp_x]


def stable_softmax(x: List[float]) -> List[float]:
    """数值稳定的softmax"""
    if not x:
        return []
    max_x = max(x)
    exp_x = [math.exp(xi - max_x) for xi in x]
    sum_exp = sum(exp_x)
    return [e / sum_exp for e in exp_x]


def matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """矩阵乘法"""
    m, n = len(a), len(b[0])
    k = len(b)
    result = [[sum(a[i][l] * b[l][j] for l in range(k)) for j in range(n)] for i in range(m)]
    return result


def transpose(x: List[List[float]]) -> List[List[float]]:
    """矩阵转置"""
    return [[x[i][j] for i in range(len(x))] for j in range(len(x[0]))]


def random_vector(dim: int, scale: float = 1.0) -> List[float]:
    """生成随机向量"""
    return [random.gauss(0, scale) for _ in range(dim)]


# cosine_similarity和normalize_vector现在从unified_adapter导入


# ============================================================================
# 数据类定义
# ============================================================================

# TokenInfo现在从unified_adapter导入

class CacheZone(Enum):
    """缓存区域类型"""
    CURRENT = auto()      # 当前处理区
    LOCAL = auto()        # 局部上下文区
    GLOBAL = auto()       # 全局摘要区


@dataclass
class SparseMaskPattern:
    """稀疏掩码模式"""
    pattern_type: str  # 'local', 'strided', 'dilated', 'random'
    mask: List[List[float]]  # 注意力掩码
    active_positions: Set[Tuple[int, int]]  # 活跃位置集合


# ============================================================================
# 1. SlidingWindowContextAttention - 滑动窗口注意力+三区缓存
# ============================================================================

class SlidingWindowContextAttention:
    """
    滑动窗口上下文注意力 + 三区缓存机制

    借鉴视频处理中的滑动窗口技术：
    - 当前区：当前处理的token（高分辨率）
    - 局部上下文区：滑动窗口内的token（中分辨率）
    - 全局摘要区：全局压缩表示（低分辨率）

    优势：
    - O(n * w) 复杂度，w为窗口大小
    - 保留局部细节的同时获得全局视野
    - 适合超长序列（百万级token）

    Attributes:
        dim: 特征维度
        window_size: 局部窗口大小
        global_summary_size: 全局摘要token数
        num_heads: 注意力头数

    重构说明：
    - 内部使用ContextAttentionAdapter包装UnifiedSlidingWindowAttention
    - 保持原有API完全不变
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

        # 内部使用适配器
        self._adapter = ContextAttentionAdapter(
            dim=dim,
            window_size=window_size,
            global_summary_size=global_summary_size,
            num_heads=num_heads,
            dropout=dropout
        )

        # 投影矩阵（委托给适配器）
        self.w_q = self._adapter.w_q
        self.w_k = self._adapter.w_k
        self.w_v = self._adapter.w_v
        self.w_o = self._adapter.w_o

        # 三区缓存（委托给适配器）
        self.current_tokens: List[TokenInfo] = self._adapter.current_tokens
        self.local_cache: List[TokenInfo] = self._adapter.local_cache
        self.global_summary: List[TokenInfo] = self._adapter.global_summary

        # 缓存统计（委托给适配器）
        self.cache_stats = self._adapter.cache_stats
    
    def _init_weight(self, in_dim: int, out_dim: int) -> List[List[float]]:
        """初始化权重"""
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        return [[random.gauss(0, scale) for _ in range(out_dim)] for _ in range(in_dim)]
    
    def forward(
        self,
        x: List[List[float]],
        use_cache: bool = True
    ) -> List[List[float]]:
        """
        前向传播

        Args:
            x: 输入序列 [seq_len, dim]
            use_cache: 是否使用缓存

        Returns:
            输出序列 [seq_len, dim]
        """
        # 委托给适配器
        return self._adapter.forward(x, use_cache)
    
    def _compute_three_zone_attention(
        self,
        position: int,
        q_heads: List[List[List[float]]],
        k_heads: List[List[List[float]]],
        v_heads: List[List[List[float]]],
        seq_len: int
    ) -> List[List[float]]:
        """
        计算三区注意力
        
        对单个位置，分别计算：
        1. 当前区注意力（自己）
        2. 局部上下文区注意力（滑动窗口）
        3. 全局摘要区注意力（全局压缩表示）
        """
        head_outputs = []
        
        for h in range(self.num_heads):
            q_i = q_heads[h][position]
            
            # 1. 当前区（自己）
            current_k = [k_heads[h][position]]
            current_v = [v_heads[h][position]]
            current_score = self._compute_attention_score(q_i, current_k)[0]
            
            # 2. 局部上下文区（滑动窗口）
            half_window = self.window_size // 2
            local_start = max(0, position - half_window)
            local_end = min(seq_len, position + half_window + 1)
            
            local_k = k_heads[h][local_start:local_end]
            local_v = v_heads[h][local_start:local_end]
            local_scores = self._compute_attention_score(q_i, local_k)
            
            self.cache_stats['local_hits'] += len(local_k)
            
            # 3. 全局摘要区
            global_k = self._get_global_summary_k(h, k_heads[h], seq_len)
            global_v = self._get_global_summary_v(h, v_heads[h], seq_len)
            global_scores = self._compute_attention_score(q_i, global_k)
            
            self.cache_stats['global_hits'] += len(global_k)
            
            # 合并三区
            all_k = current_k + local_k + global_k
            all_v = current_v + local_v + global_v
            all_scores = [current_score * 1.5] + local_scores + global_scores
            
            # Softmax
            weights = stable_softmax(all_scores)
            
            # 加权求和
            output_i = [0.0] * self.d_k
            for weight, v_vec in zip(weights, all_v):
                for j in range(self.d_k):
                    output_i[j] += weight * v_vec[j]
            
            head_outputs.append(output_i)
            
            self.cache_stats['total_queries'] += 1
        
        return head_outputs
    
    def _compute_attention_score(
        self,
        query: List[float],
        keys: List[List[float]]
    ) -> List[float]:
        """计算注意力分数"""
        scores = []
        for key in keys:
            score = sum(q * k for q, k in zip(query, key)) / math.sqrt(self.d_k)
            scores.append(score)
        return scores
    
    def _get_global_summary_k(
        self,
        head_idx: int,
        k_head: List[List[float]],
        seq_len: int
    ) -> List[List[float]]:
        """获取全局摘要的Key"""
        if not self.global_summary:
            # 动态生成全局摘要
            return self._generate_global_summary(k_head, seq_len)
        
        # 从缓存获取
        return [token.embedding[:self.d_k] for token in self.global_summary]
    
    def _get_global_summary_v(
        self,
        head_idx: int,
        v_head: List[List[float]],
        seq_len: int
    ) -> List[List[float]]:
        """获取全局摘要的Value"""
        if not self.global_summary:
            return self._generate_global_summary(v_head, seq_len)
        
        return [token.embedding[:self.d_k] for token in self.global_summary]
    
    def _generate_global_summary(
        self,
        vectors: List[List[float]],
        seq_len: int
    ) -> List[List[float]]:
        """生成全局摘要（均匀采样）"""
        if seq_len <= self.global_summary_size:
            return vectors
        
        step = seq_len // self.global_summary_size
        summary = [vectors[i * step] for i in range(self.global_summary_size)]
        return summary
    
    def _update_cache(self, x: List[List[float]]):
        """更新三区缓存"""
        seq_len = len(x)
        
        # 更新当前区（最后几个token）
        current_size = min(16, seq_len)
        self.current_tokens = [
            TokenInfo(
                token_id=seq_len - current_size + i,
                embedding=x[seq_len - current_size + i],
                position=seq_len - current_size + i,
                zone=CacheZone.CURRENT
            )
            for i in range(current_size)
        ]
        
        # 更新局部缓存（滑动窗口）
        if seq_len > self.window_size:
            local_start = seq_len - self.window_size
            self.local_cache = [
                TokenInfo(
                    token_id=local_start + i,
                    embedding=x[local_start + i],
                    position=local_start + i,
                    zone=CacheZone.LOCAL
                )
                for i in range(self.window_size)
            ]
        else:
            self.local_cache = [
                TokenInfo(
                    token_id=i,
                    embedding=x[i],
                    position=i,
                    zone=CacheZone.LOCAL
                )
                for i in range(seq_len)
            ]
        
        # 更新全局摘要
        self.global_summary = [
            TokenInfo(
                token_id=i * (seq_len // self.global_summary_size),
                embedding=x[i * (seq_len // self.global_summary_size)],
                position=i * (seq_len // self.global_summary_size),
                zone=CacheZone.GLOBAL
            )
            for i in range(min(self.global_summary_size, seq_len))
        ]
    
    def _linear(
        self,
        x: List[List[float]],
        weight: List[List[float]]
    ) -> List[List[float]]:
        """线性变换"""
        return matmul(x, weight)
    
    def _split_heads(self, x: List[List[float]]) -> List[List[List[float]]]:
        """分割为多头"""
        seq_len = len(x)
        result = []
        for h in range(self.num_heads):
            head = []
            for i in range(seq_len):
                start = h * self.d_k
                end = start + self.d_k
                head.append(x[i][start:end])
            result.append(head)
        return result
    
    def _concat_heads(self, heads: List[List[List[float]]]) -> List[List[float]]:
        """合并多头"""
        seq_len = len(heads)
        result = []
        for i in range(seq_len):
            concat = []
            for h in range(self.num_heads):
                concat.extend(heads[i][h])
            result.append(concat)
        return result
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total = self.cache_stats['total_queries']
        return {
            **self.cache_stats,
            'current_size': len(self.current_tokens),
            'local_size': len(self.local_cache),
            'global_size': len(self.global_summary),
            'local_hit_rate': self.cache_stats['local_hits'] / max(total, 1),
            'global_hit_rate': self.cache_stats['global_hits'] / max(total, 1)
        }


# ============================================================================
# 2. DynamicContextRouting - 动态Token路由
# ============================================================================

class DynamicContextRouting:
    """
    动态Token路由

    根据token重要性动态选择参与计算的token：
    - 重要token：完整参与注意力计算
    - 次要token：压缩参与或跳过
    - 借鉴视频中的关键帧选择思想

    优势：
    - 动态调整计算资源分配
    - 重要内容获得更多注意力
    - 显著降低长序列计算成本

    重构说明：
    - 内部使用DynamicRoutingAdapter包装UnifiedDynamicRouting
    - 保持原有API完全不变

    Attributes:
        dim: 特征维度
        num_important: 重要token数量
        num_compressed: 压缩token数量
        compression_ratio: 压缩比例
    """

    def __init__(
        self,
        dim: int = 768,
        num_important: int = 128,
        num_compressed: int = 256,
        compression_ratio: float = 0.5,
        routing_temperature: float = 0.5
    ):
        self.dim = dim
        self.num_important = num_important
        self.num_compressed = num_compressed
        self.compression_ratio = compression_ratio
        self.routing_temperature = routing_temperature

        # 内部使用适配器
        self._adapter = DynamicRoutingAdapter(
            dim=dim,
            num_important=num_important,
            num_compressed=num_compressed,
            compression_ratio=compression_ratio,
            routing_temperature=routing_temperature
        )

        # 重要性评估器（委托给适配器）
        self.importance_proj = self._adapter.importance_proj

        # 压缩投影（委托给适配器）
        self.compress_proj = self._adapter.compress_proj
        self.decompress_proj = self._adapter.decompress_proj

        # 路由历史（委托给适配器）
        self.routing_history: List[Dict] = self._adapter.routing_history
    
    def _init_weight(self, in_dim: int, out_dim: int) -> List[List[float]]:
        """初始化权重"""
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        return [[random.gauss(0, scale) for _ in range(out_dim)] for _ in range(in_dim)]
    
    def route(
        self,
        tokens: List[List[float]],
        query: Optional[List[float]] = None
    ) -> Tuple[List[List[float]], List[int], List[int], List[int]]:
        """
        路由token

        Args:
            tokens: 输入token [seq_len, dim]
            query: 可选的查询向量，用于相关性路由

        Returns:
            (路由后的tokens, 重要token索引, 压缩token索引, 跳过token索引)
        """
        # 委托给适配器
        return self._adapter.route(tokens, query)
    
    def _compute_importance(
        self,
        tokens: List[List[float]],
        query: Optional[List[float]]
    ) -> List[float]:
        """计算token重要性"""
        importances = []
        
        for token in tokens:
            # 基础重要性：投影分数
            base_importance = sum(
                token[i] * self.importance_proj[i][0] 
                for i in range(self.dim)
            )
            
            # 如果有查询，加入相关性
            if query is not None:
                relevance = cosine_similarity(token, query)
                base_importance += relevance * 0.5
            
            # L2范数作为重要性的补充
            norm = math.sqrt(sum(x * x for x in token))
            base_importance += norm * 0.1
            
            importances.append(max(0, base_importance))
        
        return importances
    
    def _compress_token(self, token: List[float]) -> List[float]:
        """压缩token"""
        # 降维
        compressed_dim = int(self.dim * self.compression_ratio)
        compressed = [
            sum(token[i] * self.compress_proj[i][j] for i in range(self.dim))
            for j in range(compressed_dim)
        ]
        # 升维回来（近似）
        reconstructed = [
            sum(compressed[j] * self.decompress_proj[j][i] for j in range(compressed_dim))
            for i in range(self.dim)
        ]
        return reconstructed
    
    def route_with_attention(
        self,
        query: List[float],
        keys: List[List[float]],
        values: List[List[float]]
    ) -> Tuple[List[float], Dict[str, Any]]:
        """
        带路由的注意力计算
        
        Args:
            query: 查询向量
            keys: 键向量
            values: 值向量
            
        Returns:
            (输出向量, 路由信息)
        """
        # 路由key
        routed_keys, important_idx, compressed_idx, skipped_idx = self.route(keys, query)
        
        # 根据索引获取对应的value
        routed_values = []
        for idx in important_idx + compressed_idx:
            routed_values.append(values[idx])
        
        # 计算注意力
        scores = [cosine_similarity(query, k) for k in routed_keys]
        weights = stable_softmax(scores)
        
        # 加权求和
        output = [0.0] * self.dim
        for weight, value in zip(weights, routed_values):
            for i in range(self.dim):
                output[i] += weight * value[i]
        
        routing_info = {
            'important_tokens': len(important_idx),
            'compressed_tokens': len(compressed_idx),
            'skipped_tokens': len(skipped_idx),
            'effective_tokens': len(routed_keys),
            'original_tokens': len(keys),
            'saving_ratio': 1 - len(routed_keys) / max(len(keys), 1)
        }
        
        return output, routing_info
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """获取路由统计"""
        return self._adapter.get_routing_stats()


# ============================================================================
# 3. CalibContextAttention - 预计算稀疏掩码注意力
# ============================================================================

class CalibContextAttention:
    """
    预计算稀疏掩码注意力 (Calibration-based Sparse Attention)
    
    借鉴视频处理中的预计算技术：
    - 预计算稀疏注意力模式
    - 根据内容校准掩码
    - 显著减少长序列计算
    
    支持的稀疏模式：
    - local: 局部窗口
    - strided: 步长采样
    - dilated: 空洞模式
    - random: 随机采样
    - block_local: 分块局部
    
    Attributes:
        dim: 特征维度
        pattern_type: 稀疏模式类型
        sparsity: 稀疏度（0-1，越大越稀疏）
        block_size: 块大小
    """
    
    def __init__(
        self,
        dim: int = 768,
        pattern_type: str = 'local',
        sparsity: float = 0.9,
        block_size: int = 64,
        num_heads: int = 8,
        calib_steps: int = 100
    ):
        self.dim = dim
        self.pattern_type = pattern_type
        self.sparsity = sparsity
        self.block_size = block_size
        self.num_heads = num_heads
        self.d_k = dim // num_heads
        self.calib_steps = calib_steps
        
        # 投影矩阵
        self.w_q = self._init_weight(dim, dim)
        self.w_k = self._init_weight(dim, dim)
        self.w_v = self._init_weight(dim, dim)
        self.w_o = self._init_weight(dim, dim)
        
        # 预计算的掩码模式
        self.precomputed_masks: Dict[int, SparseMaskPattern] = {}
        
        # 校准统计
        self.calib_stats = {
            'patterns_generated': 0,
            'avg_active_ratio': 0.0
        }
    
    def _init_weight(self, in_dim: int, out_dim: int) -> List[List[float]]:
        """初始化权重"""
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        return [[random.gauss(0, scale) for _ in range(out_dim)] for _ in range(in_dim)]
    
    def calibrate(
        self,
        sample_sequences: List[List[List[float]]],
        target_sparsity: Optional[float] = None
    ):
        """
        校准稀疏模式
        
        根据样本序列学习最优的稀疏模式
        
        Args:
            sample_sequences: 样本序列列表
            target_sparsity: 目标稀疏度
        """
        sparsity = target_sparsity or self.sparsity
        
        for seq in sample_sequences:
            seq_len = len(seq)
            if seq_len not in self.precomputed_masks:
                pattern = self._generate_pattern(seq_len, sparsity)
                self.precomputed_masks[seq_len] = pattern
                self.calib_stats['patterns_generated'] += 1
        
        # 更新平均活跃比例
        if self.precomputed_masks:
            total_active = sum(
                len(p.active_positions) for p in self.precomputed_masks.values()
            )
            total_possible = sum(
                seq_len * seq_len for seq_len in self.precomputed_masks.keys()
            )
            self.calib_stats['avg_active_ratio'] = total_active / max(total_possible, 1)
    
    def _generate_pattern(
        self,
        seq_len: int,
        sparsity: float
    ) -> SparseMaskPattern:
        """生成稀疏掩码模式"""
        active_ratio = 1 - sparsity
        num_active = int(seq_len * seq_len * active_ratio)
        
        mask = [[-float('inf')] * seq_len for _ in range(seq_len)]
        active_positions = set()
        
        if self.pattern_type == 'local':
            # 局部窗口模式
            window_size = max(1, int(seq_len * active_ratio / 2))
            for i in range(seq_len):
                for j in range(max(0, i - window_size), min(seq_len, i + window_size + 1)):
                    mask[i][j] = 0.0
                    active_positions.add((i, j))
        
        elif self.pattern_type == 'strided':
            # 步长模式
            stride = max(1, int(1 / active_ratio))
            for i in range(seq_len):
                for j in range(0, seq_len, stride):
                    mask[i][j] = 0.0
                    active_positions.add((i, j))
                # 添加局部窗口
                for j in range(max(0, i - 2), min(seq_len, i + 3)):
                    mask[i][j] = 0.0
                    active_positions.add((i, j))
        
        elif self.pattern_type == 'dilated':
            # 空洞模式
            dilation = max(1, int(1 / active_ratio))
            for i in range(seq_len):
                for j in range(i % dilation, seq_len, dilation):
                    mask[i][j] = 0.0
                    active_positions.add((i, j))
        
        elif self.pattern_type == 'random':
            # 随机模式
            positions = [(i, j) for i in range(seq_len) for j in range(seq_len)]
            selected = random.sample(positions, min(num_active, len(positions)))
            for i, j in selected:
                mask[i][j] = 0.0
                active_positions.add((i, j))
        
        elif self.pattern_type == 'block_local':
            # 分块局部模式
            num_blocks = max(1, seq_len // self.block_size)
            active_per_block = max(1, int(self.block_size * self.block_size * active_ratio))
            
            for block_i in range(num_blocks):
                for block_j in range(num_blocks):
                    start_i = block_i * self.block_size
                    start_j = block_j * self.block_size
                    
                    # 对角线块全连接
                    if block_i == block_j:
                        for i in range(start_i, min(start_i + self.block_size, seq_len)):
                            for j in range(start_j, min(start_j + self.block_size, seq_len)):
                                mask[i][j] = 0.0
                                active_positions.add((i, j))
                    else:
                        # 非对角线块稀疏连接
                        for _ in range(active_per_block):
                            i = random.randint(start_i, min(start_i + self.block_size - 1, seq_len - 1))
                            j = random.randint(start_j, min(start_j + self.block_size - 1, seq_len - 1))
                            mask[i][j] = 0.0
                            active_positions.add((i, j))
        
        return SparseMaskPattern(
            pattern_type=self.pattern_type,
            mask=mask,
            active_positions=active_positions
        )
    
    def forward(
        self,
        x: List[List[float]],
        custom_mask: Optional[List[List[float]]] = None
    ) -> List[List[float]]:
        """
        前向传播
        
        Args:
            x: 输入序列 [seq_len, dim]
            custom_mask: 可选的自定义掩码
            
        Returns:
            输出序列 [seq_len, dim]
        """
        seq_len = len(x)
        
        # 获取或生成掩码
        if seq_len in self.precomputed_masks:
            sparse_pattern = self.precomputed_masks[seq_len]
            mask = sparse_pattern.mask
        else:
            # 动态生成
            sparse_pattern = self._generate_pattern(seq_len, self.sparsity)
            mask = sparse_pattern.mask
        
        # 合并自定义掩码
        if custom_mask is not None:
            mask = self._merge_masks(mask, custom_mask)
        
        # 线性投影
        q = self._linear(x, self.w_q)
        k = self._linear(x, self.w_k)
        v = self._linear(x, self.w_v)
        
        # 分割为多头
        q_heads = self._split_heads(q)
        k_heads = self._split_heads(k)
        v_heads = self._split_heads(v)
        
        # 稀疏注意力计算
        outputs = []
        for h in range(self.num_heads):
            head_output = self._sparse_attention(
                q_heads[h], k_heads[h], v_heads[h], mask
            )
            outputs.append(head_output)
        
        # 合并多头
        concat = self._concat_heads(outputs)
        
        # 输出投影
        output = self._linear(concat, self.w_o)
        
        return output
    
    def _sparse_attention(
        self,
        q: List[List[float]],
        k: List[List[float]],
        v: List[List[float]],
        mask: List[List[float]]
    ) -> List[List[float]]:
        """稀疏注意力计算"""
        seq_len = len(q)
        output = []
        
        for i in range(seq_len):
            # 只计算掩码允许的位置
            scores = []
            valid_indices = []
            
            for j in range(seq_len):
                if mask[i][j] > -float('inf') / 2:  # 不是-inf
                    score = sum(q[i][d] * k[j][d] for d in range(self.d_k))
                    score = score / math.sqrt(self.d_k) + mask[i][j]
                    scores.append(score)
                    valid_indices.append(j)
            
            if not scores:
                output.append([0.0] * self.d_k)
                continue
            
            # Softmax
            weights = stable_softmax(scores)
            
            # 加权求和
            out_i = [0.0] * self.d_k
            for weight, idx in zip(weights, valid_indices):
                for d in range(self.d_k):
                    out_i[d] += weight * v[idx][d]
            
            output.append(out_i)
        
        return output
    
    def _merge_masks(
        self,
        mask1: List[List[float]],
        mask2: List[List[float]]
    ) -> List[List[float]]:
        """合并两个掩码"""
        seq_len = len(mask1)
        merged = []
        for i in range(seq_len):
            row = []
            for j in range(seq_len):
                # 取更严格的掩码
                row.append(min(mask1[i][j], mask2[i][j]))
            merged.append(row)
        return merged
    
    def _linear(
        self,
        x: List[List[float]],
        weight: List[List[float]]
    ) -> List[List[float]]:
        """线性变换"""
        return matmul(x, weight)
    
    def _split_heads(self, x: List[List[float]]) -> List[List[List[float]]]:
        """分割为多头"""
        seq_len = len(x)
        result = []
        for h in range(self.num_heads):
            head = []
            for i in range(seq_len):
                start = h * self.d_k
                end = start + self.d_k
                head.append(x[i][start:end])
            result.append(head)
        return result
    
    def _concat_heads(self, heads: List[List[List[float]]]) -> List[List[float]]:
        """合并多头"""
        seq_len = len(heads[0])
        result = []
        for i in range(seq_len):
            concat = []
            for h in range(self.num_heads):
                concat.extend(heads[h][i])
            result.append(concat)
        return result
    
    def get_pattern_stats(self) -> Dict[str, Any]:
        """获取模式统计"""
        return {
            **self.calib_stats,
            'pattern_type': self.pattern_type,
            'sparsity': self.sparsity,
            'precomputed_lengths': list(self.precomputed_masks.keys()),
            'block_size': self.block_size
        }


# ============================================================================
# 4. LongContextAttentionFactory - 工厂类
# ============================================================================

class LongContextAttentionFactory:
    """长上下文注意力工厂"""
    
    @staticmethod
    def create(
        attention_type: str,
        **kwargs
    ):
        """
        创建注意力机制
        
        Args:
            attention_type: 类型 ('sliding_window', 'dynamic_routing', 'calib_sparse')
            **kwargs: 额外参数
            
        Returns:
            注意力实例
        """
        if attention_type == 'sliding_window':
            return SlidingWindowContextAttention(**kwargs)
        elif attention_type == 'dynamic_routing':
            return DynamicContextRouting(**kwargs)
        elif attention_type == 'calib_sparse':
            return CalibContextAttention(**kwargs)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
    
    @staticmethod
    def get_available_types() -> List[str]:
        """获取可用类型"""
        return ['sliding_window', 'dynamic_routing', 'calib_sparse']
