"""
SnapKV - 基于重要性评分的KV缓存淘汰

保留注意力分数最高的token

作者: UFO Framework Team
"""

import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import OrderedDict
import numpy as np


@dataclass
class KVEntry:
    """KV缓存条目"""
    key: np.ndarray
    value: np.ndarray
    importance_score: float = 0.0
    access_count: int = 0
    position: int = 0


class ImportanceScorer:
    """重要性评分器"""
    
    def __init__(
        self,
        window_size: int = 32,
        decay_factor: float = 0.9
    ):
        self.window_size = window_size
        self.decay_factor = decay_factor
        
        # 注意力历史
        self.attention_history: List[np.ndarray] = []
    
    def update(self, attention_weights: np.ndarray) -> None:
        """更新注意力历史"""
        self.attention_history.append(attention_weights)
        if len(self.attention_history) > self.window_size:
            self.attention_history.pop(0)
    
    def compute_importance(self, position: int) -> float:
        """
        计算位置的重要性分数
        
        Args:
            position: token位置
            
        Returns:
            重要性分数
        """
        if not self.attention_history:
            return 1.0
        
        # 计算累积注意力权重
        total_importance = 0.0
        weight = 1.0
        
        for attn in reversed(self.attention_history):
            if position < attn.shape[-1]:
                total_importance += attn[..., position].mean() * weight
            weight *= self.decay_factor
        
        return float(total_importance)


class SnapKVCache:
    """
    SnapKV缓存
    
    基于重要性评分淘汰低价值KV对
    """
    
    def __init__(
        self,
        max_size: int = 4096,
        eviction_ratio: float = 0.3,
        num_heads: int = 12,
        head_dim: int = 64
    ):
        self.max_size = max_size
        self.eviction_ratio = eviction_ratio
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # KV存储
        self.cache: OrderedDict[int, KVEntry] = OrderedDict()
        
        # 重要性评分器
        self.importance_scorer = ImportanceScorer()
        
        # 统计
        self.stats = {
            'total_stored': 0,
            'total_evicted': 0,
            'total_accessed': 0,
            'avg_importance': 0.0,
            'hit_rate': 0.0
        }
    
    def store(
        self,
        position: int,
        key: np.ndarray,
        value: np.ndarray,
        attention_weights: Optional[np.ndarray] = None
    ) -> None:
        """
        存储KV对
        
        Args:
            position: 位置
            key: 键 [num_heads, head_dim]
            value: 值 [num_heads, head_dim]
            attention_weights: 注意力权重
        """
        # 计算重要性
        if attention_weights is not None:
            self.importance_scorer.update(attention_weights)
        
        importance = self.importance_scorer.compute_importance(position)
        
        # 创建条目
        entry = KVEntry(
            key=key,
            value=value,
            importance_score=importance,
            position=position
        )
        
        # 存储
        self.cache[position] = entry
        self.stats['total_stored'] += 1
        
        # 检查容量
        if len(self.cache) > self.max_size:
            self._evict()
    
    def retrieve(
        self,
        positions: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        检索KV对
        
        Args:
            positions: 位置列表
            
        Returns:
            (keys, values, found_positions)
        """
        keys = []
        values = []
        found = []
        
        for pos in positions:
            if pos in self.cache:
                entry = self.cache[pos]
                entry.access_count += 1
                keys.append(entry.key)
                values.append(entry.value)
                found.append(pos)
                
                self.stats['total_accessed'] += 1
        
        if keys:
            return np.stack(keys), np.stack(values), found
        
        return np.array([]), np.array([]), []
    
    def _evict(self) -> int:
        """
        淘汰低重要性条目
        
        Returns:
            淘汰数量
        """
        num_to_evict = int(len(self.cache) * self.eviction_ratio)
        
        if num_to_evict == 0:
            return 0
        
        # 按重要性排序
        items = list(self.cache.items())
        items.sort(key=lambda x: x[1].importance_score)
        
        # 淘汰最低的
        for pos, _ in items[:num_to_evict]:
            del self.cache[pos]
        
        self.stats['total_evicted'] += num_to_evict
        
        return num_to_evict
    
    def get_all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取所有KV对
        
        Returns:
            (keys, values, positions)
        """
        if not self.cache:
            return np.array([]), np.array([]), np.array([])
        
        keys = []
        values = []
        positions = []
        
        for pos, entry in self.cache.items():
            keys.append(entry.key)
            values.append(entry.value)
            positions.append(pos)
        
        return np.stack(keys), np.stack(values), np.array(positions)
    
    def update_importance(
        self,
        position: int,
        delta: float
    ) -> None:
        """更新重要性"""
        if position in self.cache:
            self.cache[position].importance_score += delta
    
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
    
    def get_stats(self) -> Dict:
        """获取统计"""
        if self.cache:
            avg_importance = np.mean([e.importance_score for e in self.cache.values()])
        else:
            avg_importance = 0.0
        
        return {
            **self.stats,
            'current_size': len(self.cache),
            'max_size': self.max_size,
            'utilization': len(self.cache) / self.max_size,
            'avg_importance': avg_importance
        }


class SnapKVAttention:
    """
    SnapKV注意力
    
    集成重要性感知的KV缓存
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        head_dim: int = 64,
        cache_size: int = 4096
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # KV缓存
        self.kv_cache = SnapKVCache(
            max_size=cache_size,
            num_heads=num_heads,
            head_dim=head_dim
        )
        
        # 投影权重
        scale = 1.0 / math.sqrt(hidden_size)
        self.W_q = np.random.randn(hidden_size, hidden_size).astype(np.float32) * scale
        self.W_k = np.random.randn(hidden_size, hidden_size).astype(np.float32) * scale
        self.W_v = np.random.randn(hidden_size, hidden_size).astype(np.float32) * scale
        self.W_o = np.random.randn(hidden_size, hidden_size).astype(np.float32) * scale
    
    def forward(
        self,
        hidden_states: np.ndarray,
        use_cache: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        前向传播
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            use_cache: 是否使用缓存
            
        Returns:
            (输出, 统计信息)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 投影
        Q = np.dot(hidden_states, self.W_q)
        K = np.dot(hidden_states, self.W_k)
        V = np.dot(hidden_states, self.W_v)
        
        # 多头分割
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)
        
        if use_cache:
            # 存储新的KV
            for i in range(seq_len):
                self.kv_cache.store(
                    position=i,
                    key=K[0, :, i, :],  # 简化：只处理第一个batch
                    value=V[0, :, i, :]
                )
            
            # 获取缓存的KV
            cached_K, cached_V, positions = self.kv_cache.get_all()
            
            if len(positions) > 0:
                # 扩展为batch
                cached_K = np.tile(cached_K[np.newaxis, :, :, :], (batch_size, 1, 1, 1))
                cached_V = np.tile(cached_V[np.newaxis, :, :, :], (batch_size, 1, 1, 1))
                
                # 计算注意力
                attn_scores = np.matmul(Q, cached_K.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
                attn_probs = self._softmax(attn_scores)
                attn_output = np.matmul(attn_probs, cached_V)
            else:
                attn_output = np.zeros_like(Q)
        else:
            # 标准注意力
            attn_scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
            attn_probs = self._softmax(attn_scores)
            attn_output = np.matmul(attn_probs, V)
        
        # 合并多头
        attn_output = self._merge_heads(attn_output)
        
        # 输出投影
        output = np.dot(attn_output, self.W_o)
        
        return output, {
            'cache_stats': self.kv_cache.get_stats()
        }
    
    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """分割多头"""
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)
    
    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """合并多头"""
        x = x.transpose(0, 2, 1, 3)
        batch_size, seq_len, _, _ = x.shape
        return x.reshape(batch_size, seq_len, -1)
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax"""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# 便捷函数
def create_snapkv_cache(max_size: int = 4096) -> SnapKVCache:
    """创建SnapKV缓存"""
    return SnapKVCache(max_size=max_size)


if __name__ == "__main__":
    # 测试
    cache = SnapKVCache(max_size=100)
    
    print("=" * 60)
    print("SnapKV测试")
    print("=" * 60)
    
    # 模拟存储
    for i in range(150):
        key = np.random.randn(12, 64).astype(np.float32)
        value = np.random.randn(12, 64).astype(np.float32)
        cache.store(i, key, value)
    
    stats = cache.get_stats()
    print(f"\n缓存大小: {stats['current_size']}/{stats['max_size']}")
    print(f"淘汰数量: {stats['total_evicted']}")
    print(f"利用率: {stats['utilization']:.1%}")
