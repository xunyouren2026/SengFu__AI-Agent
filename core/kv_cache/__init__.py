"""
AGI Unified Framework - KV缓存压缩模块 (H2O: Heavy-Hitter Oracle)
========================================================================

本模块实现了KV缓存压缩算法，用于减少Transformer模型推理时的内存占用。

H2O算法原理：
- 识别并保留"Heavy Hitters"（频繁出现的关键token）
- 使用轻量级哈希过滤器追踪token重要性
- 在内存受限场景下显著减少KV缓存大小

主要功能：
1. H2O压缩算法实现
2. KV缓存管理
3. 动态压缩策略
4. 压缩效果评估

使用示例：
    from core.kv_cache import H2OKVCache, KVCacheCompressor
    
    compressor = H2OKVCache(max_cache_size=1024)
    compressed_cache = compressor.compress(kv_cache, keep_ratio=0.5)
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import OrderedDict
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 数据结构定义
# =============================================================================

class CacheEntry:
    """缓存条目"""
    
    def __init__(self, key: np.ndarray, value: np.ndarray, 
                 position: int, importance: float = 0.0):
        """
        Args:
            key: Key向量 (seq_len, head_dim)
            value: Value向量 (seq_len, head_dim)
            position: 在序列中的位置
            importance: 重要性分数
        """
        self.key = key
        self.value = value
        self.position = position
        self.importance = importance
        self.access_count = 0
        self.last_access = time.time()
    
    def update_access(self):
        """更新访问信息"""
        self.access_count += 1
        self.last_access = time.time()
    
    def update_importance(self, score: float):
        """更新重要性分数"""
        self.importance = score
    
    @property
    def seq_len(self) -> int:
        return self.key.shape[0]
    
    @property
    def memory_size(self) -> int:
        """估算内存大小（字节）"""
        return self.key.nbytes + self.value.nbytes


@dataclass
class KVCache:
    """KV缓存数据结构"""
    num_heads: int
    head_dim: int
    max_seq_len: int
    dtype: str = "float32"
    
    keys: Optional[np.ndarray] = None
    values: Optional[np.ndarray] = None
    positions: Optional[List[int]] = None
    importance_scores: Optional[np.ndarray] = None
    
    def __post_init__(self):
        # 延迟初始化
        if self.keys is None:
            self.keys = np.zeros((self.num_heads, self.max_seq_len, self.head_dim), 
                                dtype=getattr(np, self.dtype))
            self.values = np.zeros((self.num_heads, self.max_seq_len, self.head_dim),
                                  dtype=getattr(np, self.dtype))
            self.positions = []
            self.importance_scores = np.zeros((self.num_heads, self.max_seq_len),
                                               dtype=np.float32)
    
    @property
    def current_seq_len(self) -> int:
        return len(self.positions)
    
    def is_full(self) -> bool:
        return self.current_seq_len >= self.max_seq_len
    
    def memory_usage(self) -> int:
        """计算当前内存使用（字节）"""
        seq_len = self.current_seq_len
        if seq_len == 0:
            return 0
        return 2 * self.num_heads * seq_len * self.head_dim * np.dtype(self.dtype).itemsize


# =============================================================================
# H2O算法实现
# =============================================================================

class H2OKVCache:
    """
    H2O (Heavy-Hitter Oracle) KV缓存压缩器
    
    H2O是一种轻量级的KV缓存压缩算法，通过识别和保留最重要的token来减少内存使用。
    
    核心思想：
    1. 使用多个哈希过滤器追踪每个token的"累计影响"
    2. 当缓存满时，删除影响最小的token
    3. 保留的token称为"Heavy Hitters"
    
    参考论文: H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models
    """
    
    def __init__(self, max_cache_size: int = 2048,
                 num_hash_functions: int = 4,
                 num_buckets: int = 1024,
                 dtype: str = "float32"):
        """
        Args:
            max_cache_size: 最大缓存token数量
            num_hash_functions: 哈希函数数量
            num_buckets: 哈希桶数量
            dtype: 数据类型
        """
        self.max_cache_size = max_cache_size
        self.num_hash_functions = num_hash_functions
        self.num_buckets = num_buckets
        self.dtype = dtype
        
        # 哈希过滤器状态
        self.hash_table = np.zeros((num_hash_functions, num_buckets), dtype=np.float32)
        self.token_to_hash = {}  # token_position -> [hash_values]
        
        # 缓存内容
        self.entries: OrderedDict[int, CacheEntry] = OrderedDict()
        
        # 统计信息
        self.stats = {
            'total_evictions': 0,
            'heavy_hitters_kept': 0,
            'compression_ratio': 1.0,
        }
    
    def add(self, position: int, key: np.ndarray, value: np.ndarray,
           attention_weights: Optional[np.ndarray] = None) -> None:
        """
        添加新的KV条目
        
        Args:
            position: 位置
            key: Key向量
            value: Value向量
            attention_weights: 注意力权重（用于计算重要性）
        """
        # 计算重要性分数
        if attention_weights is not None:
            importance = float(np.mean(np.abs(attention_weights)))
        else:
            # 使用key的范数作为重要性代理
            importance = float(np.linalg.norm(key))
        
        # 创建缓存条目
        entry = CacheEntry(key, value, position, importance)
        
        # 添加到缓存
        self.entries[position] = entry
        
        # 更新哈希过滤器
        self._update_hash_filters(position, importance)
        
        # 如果超出最大大小，进行压缩
        if len(self.entries) > self.max_cache_size:
            self._evict_lru()
    
    def _update_hash_filters(self, position: int, importance: float) -> None:
        """更新哈希过滤器"""
        # 为这个位置生成哈希值
        hash_values = self._hash_position(position)
        self.token_to_hash[position] = hash_values
        
        # 增加对应的桶计数
        for i, h in enumerate(hash_values):
            self.hash_table[i, h] += importance
    
    def _hash_position(self, position: int) -> List[int]:
        """使用多个哈希函数哈希位置"""
        results = []
        position_bytes = str(position).encode()
        
        for i in range(self.num_hash_functions):
            # 不同的种子产生不同的哈希
            seed = hashlib.sha256(position_bytes + bytes([i])).digest()
            hash_val = int.from_bytes(seed[:4], 'big') % self.num_buckets
            results.append(hash_val)
        
        return results
    
    def _get_heavy_hitter_score(self, position: int) -> float:
        """获取一个位置的Heavy Hitter分数"""
        if position not in self.token_to_hash:
            return 0.0
        
        hash_values = self.token_to_hash[position]
        
        # 分数是所有对应桶的最小值（类似Count-Min Sketch）
        scores = [self.hash_table[i, h] for i, h in enumerate(hash_values)]
        return min(scores)
    
    def _evict_lru(self) -> None:
        """驱逐最不重要的条目"""
        # 计算所有条目的HH分数
        positions = list(self.entries.keys())
        scores = [self._get_heavy_hitter_score(p) for p in positions]
        
        # 找到分数最低的
        min_idx = np.argmin(scores)
        evicted_position = positions[min_idx]
        
        # 驱逐
        del self.entries[evicted_position]
        del self.token_to_hash[evicted_position]
        
        self.stats['total_evictions'] += 1
    
    def compress(self, keep_ratio: float = 0.5) -> 'CompressedCache':
        """
        执行压缩，返回压缩后的缓存
        
        Args:
            keep_ratio: 保留比例 (0.0 - 1.0)
            
        Returns:
            CompressedCache: 压缩后的缓存
        """
        if len(self.entries) == 0:
            return CompressedCache({}, self.dtype)
        
        # 计算保留数量
        num_keep = max(1, int(len(self.entries) * keep_ratio))
        
        # 获取所有条目的HH分数
        positions = list(self.entries.keys())
        scores = [self._get_heavy_hitter_score(p) for p in positions]
        
        # 按分数排序，保留最高的
        sorted_indices = np.argsort(scores)[::-1][:num_keep]
        keep_positions = [positions[i] for i in sorted_indices]
        
        # 构建压缩缓存
        compressed_data = {
            pos: self.entries[pos] 
            for pos in keep_positions
            if pos in self.entries
        }
        
        # 更新统计
        self.stats['heavy_hitters_kept'] = len(compressed_data)
        self.stats['compression_ratio'] = len(compressed_data) / max(1, len(self.entries))
        
        return CompressedCache(compressed_data, self.dtype, keep_ratio)
    
    def get(self, position: int) -> Optional[CacheEntry]:
        """获取缓存条目"""
        if position in self.entries:
            entry = self.entries[position]
            entry.update_access()
            # 移到最后（LRU策略）
            self.entries.move_to_end(position)
            return entry
        return None
    
    def get_recent(self, n: int) -> List[CacheEntry]:
        """获取最近的n个条目"""
        items = list(self.entries.items())[-n:]
        return [entry for _, entry in items]
    
    def clear(self) -> None:
        """清空缓存"""
        self.entries.clear()
        self.token_to_hash.clear()
        self.hash_table.fill(0)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'current_size': len(self.entries),
            'max_size': self.max_cache_size,
            'memory_usage': sum(e.memory_size for e in self.entries.values()),
            'avg_importance': np.mean([e.importance for e in self.entries.values()]) if self.entries else 0,
        }


class CompressedCache:
    """压缩后的KV缓存"""
    
    def __init__(self, data: Dict[int, CacheEntry], 
                 dtype: str = "float32",
                 keep_ratio: float = 1.0):
        self.data = data
        self.dtype = dtype
        self.keep_ratio = keep_ratio
    
    @property
    def num_entries(self) -> int:
        return len(self.data)
    
    def get_keys(self) -> List[np.ndarray]:
        """获取所有keys"""
        return [entry.key for entry in self.data.values()]
    
    def get_values(self) -> List[np.ndarray]:
        """获取所有values"""
        return [entry.value for entry in self.data.values()]
    
    def to_list(self) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        """转换为列表格式"""
        return [(pos, entry.key, entry.value) 
                for pos, entry in sorted(self.data.items())]


# =============================================================================
# 通用KV缓存管理器
# =============================================================================

class KVCacheManager:
    """
    KV缓存管理器
    
    管理多个attention head的缓存，支持：
    - 多层缓存
    - 动态压缩
    - 缓存预取
    """
    
    def __init__(self, num_layers: int, num_heads: int, head_dim: int,
                 max_seq_len: int = 4096, dtype: str = "float32",
                 compression_strategy: str = "h2o"):
        """
        Args:
            num_layers: 层数
            num_heads: 注意力头数
            head_dim: 每个头的维度
            max_seq_len: 最大序列长度
            dtype: 数据类型
            compression_strategy: 压缩策略 ("h2o", "snap", "infadapter")
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.compression_strategy = compression_strategy
        
        # 创建压缩器
        self._create_compressor()
        
        # 创建缓存
        self.layer_caches: List[H2OKVCache] = []
        for _ in range(num_layers):
            cache = H2OKVCache(
                max_cache_size=max_seq_len,
                num_hash_functions=4,
                num_buckets=1024,
                dtype=dtype
            )
            self.layer_caches.append(cache)
        
        # 统计
        self.stats = {
            'total_tokens': 0,
            'compressed_tokens': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }
    
    def _create_compressor(self):
        """根据策略创建压缩器"""
        if self.compression_strategy == "h2o":
            self.compressor_class = H2OKVCache
        else:
            logger.warning(f"Unknown strategy {self.compression_strategy}, using H2O")
            self.compressor_class = H2OKVCache
    
    def add_layer(self, layer_idx: int, position: int,
                 keys: np.ndarray, values: np.ndarray,
                 attention_weights: Optional[np.ndarray] = None) -> None:
        """
        添加某一层的KV
        
        Args:
            layer_idx: 层索引
            position: 位置
            keys: Key向量 (num_heads, seq_len, head_dim) 或 (seq_len, head_dim)
            values: Value向量
            attention_weights: 注意力权重
        """
        if layer_idx >= self.num_layers:
            raise ValueError(f"Layer index {layer_idx} out of range [0, {self.num_layers})")
        
        cache = self.layer_caches[layer_idx]
        
        # 处理不同的输入格式
        if keys.ndim == 2:
            # 单个head
            keys = keys[np.newaxis, :, :]
            values = values[np.newaxis, :, :]
        
        # 为每个head添加
        for head_idx in range(min(keys.shape[0], self.num_heads)):
            key = keys[head_idx]
            value = values[head_idx]
            
            attn_w = attention_weights[head_idx] if attention_weights is not None else None
            cache.add(position, key, value, attn_w)
    
    def get_layer(self, layer_idx: int, position: int) -> Optional[CacheEntry]:
        """获取某一层的缓存"""
        if layer_idx >= self.num_layers:
            return None
        return self.layer_caches[layer_idx].get(position)
    
    def compress_all(self, keep_ratio: float = 0.5) -> List[CompressedCache]:
        """
        压缩所有层的缓存
        
        Args:
            keep_ratio: 保留比例
            
        Returns:
            压缩后的缓存列表
        """
        compressed = []
        total_entries = 0
        total_compressed = 0
        
        for cache in self.layer_caches:
            comp = cache.compress(keep_ratio)
            compressed.append(comp)
            total_entries += len(cache.entries) + cache.stats['total_evictions']
            total_compressed += comp.num_entries
        
        self.stats['compressed_tokens'] = total_compressed
        if total_entries > 0:
            self.stats['compression_ratio'] = total_compressed / total_entries
        
        return compressed
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        layer_stats = [cache.get_stats() for cache in self.layer_caches]
        
        return {
            'total_tokens': self.stats['total_tokens'],
            'compressed_tokens': self.stats['compressed_tokens'],
            'overall_compression_ratio': self.stats.get('compression_ratio', 1.0),
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'layers': layer_stats,
            'strategy': self.compression_strategy,
        }


# =============================================================================
# StreamingLLM集成
# =============================================================================

class StreamingLLMCache:
    """
    StreamingLLM缓存
    
    结合Sink tokens和Sliding Window的缓存策略。
    用于无限长度的流式生成。
    
    特点：
    1. 保留前几个tokens作为"锚点"（Sink）
    2. 使用滑动窗口保留最近的tokens
    3. 丢弃中间tokens
    """
    
    def __init__(self, sink_size: int = 4, window_size: int = 128,
                 dtype: str = "float32"):
        """
        Args:
            sink_size: Sink tokens数量
            window_size: 滑动窗口大小
            dtype: 数据类型
        """
        self.sink_size = sink_size
        self.window_size = window_size
        self.dtype = dtype
        
        # Sink tokens（始终保留）
        self.sink_keys: List[np.ndarray] = []
        self.sink_values: List[np.ndarray] = []
        
        # 滑动窗口
        self.window_keys: List[np.ndarray] = []
        self.window_values: List[np.ndarray] = []
        
        # 位置映射
        self.position_to_sink_idx: Dict[int, int] = {}
        self.position_to_window_idx: Dict[int, int] = {}
        
        self.total_tokens = 0
    
    def add(self, position: int, key: np.ndarray, value: np.ndarray) -> None:
        """添加token"""
        self.total_tokens += 1
        
        if position < self.sink_size:
            # Sink区域
            idx = len(self.sink_keys)
            self.sink_keys.append(key)
            self.sink_values.append(value)
            self.position_to_sink_idx[position] = idx
        else:
            # 滑动窗口
            if len(self.window_keys) >= self.window_size:
                # 移除最旧的
                self.window_keys.pop(0)
                self.window_values.pop(0)
                
                # 重新索引
                old_positions = list(self.position_to_window_idx.keys())
                for p in old_positions:
                    self.position_to_window_idx[p] -= 1
            
            self.window_keys.append(key)
            self.window_values.append(value)
            self.position_to_window_idx[position] = len(self.window_keys) - 1
    
    def get_context(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """获取完整上下文"""
        return self.sink_keys + self.window_keys, self.sink_values + self.window_values
    
    def memory_usage(self) -> int:
        """计算内存使用"""
        sink_mem = sum(k.nbytes for k in self.sink_keys)
        window_mem = sum(k.nbytes for k in self.window_keys)
        return sink_mem * 2  # keys + values
    
    def clear(self) -> None:
        """清空缓存"""
        self.sink_keys.clear()
        self.sink_values.clear()
        self.window_keys.clear()
        self.window_values.clear()
        self.position_to_sink_idx.clear()
        self.position_to_window_idx.clear()
        self.total_tokens = 0


# =============================================================================
# 压缩效果评估
# =============================================================================

class CompressionEvaluator:
    """评估压缩效果"""
    
    @staticmethod
    def evaluate(original_cache: KVCache, 
               compressed_cache: CompressedCache) -> Dict[str, float]:
        """
        评估压缩效果
        
        计算压缩比、信息保留度等指标。
        """
        original_size = original_cache.memory_usage()
        
        if compressed_cache.num_entries == 0:
            return {
                'compression_ratio': 0.0,
                'original_size': original_size,
                'compressed_size': 0,
                'tokens_retained': 0,
                'tokens_removed': original_cache.current_seq_len,
            }
        
        compressed_size = sum(
            entry.memory_size for entry in compressed_cache.data.values()
        )
        
        original_tokens = original_cache.current_seq_len
        retained_tokens = compressed_cache.num_entries
        
        return {
            'compression_ratio': compressed_size / max(1, original_size),
            'size_reduction': 1 - compressed_size / max(1, original_size),
            'original_size': original_size,
            'compressed_size': compressed_size,
            'tokens_retained': retained_tokens,
            'tokens_removed': original_tokens - retained_tokens,
            'retention_rate': retained_tokens / max(1, original_tokens),
        }
    
    @staticmethod
    def benchmark_compression(cache_size: int = 1024,
                             num_heads: int = 32,
                             head_dim: int = 128,
                             keep_ratios: List[float] = None) -> Dict[str, Any]:
        """
        基准测试不同压缩比例下的效果
        """
        if keep_ratios is None:
            keep_ratios = [0.1, 0.25, 0.5, 0.75, 1.0]
        
        results = []
        
        # 创建模拟缓存
        compressor = H2OKVCache(max_cache_size=cache_size)
        
        for i in range(cache_size):
            # 模拟添加（使用随机数据）
            key = np.random.randn(num_heads, head_dim).astype(np.float32)
            value = np.random.randn(num_heads, head_dim).astype(np.float32)
            attn = np.random.rand(num_heads).astype(np.float32)
            
            compressor.add(i, key[0], value[0], attn)
        
        # 测试不同压缩比例
        for ratio in keep_ratios:
            compressed = compressor.compress(keep_ratio=ratio)
            
            results.append({
                'keep_ratio': ratio,
                'num_entries': compressed.num_entries,
                'compression_ratio': compressor.stats.get('compression_ratio', 1.0),
            })
        
        return {
            'cache_size': cache_size,
            'num_heads': num_heads,
            'head_dim': head_dim,
            'results': results,
        }


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    'CacheEntry',
    'KVCache',
    'H2OKVCache',
    'CompressedCache',
    'KVCacheManager',
    'StreamingLLMCache',
    'CompressionEvaluator',
]
